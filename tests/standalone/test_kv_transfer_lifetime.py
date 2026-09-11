# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Execute scheduler transfer lifetime methods without loading device backends."""

import ast
from collections import defaultdict
import enum
from pathlib import Path
from threading import Lock
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_methods(path, names, namespace):
    source = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [
        n
        for n in ast.walk(source)
        if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    for node in nodes:
        node.decorator_list = []
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[future, *nodes], type_ignores=[])
            ),
            str(path),
            "exec",
        ),
        namespace,
    )


def fixture(*, sending=True, received=False):
    source = ast.parse((ROOT / "vllm/v1/request.py").read_text(encoding="utf-8"))
    status = next(
        n
        for n in source.body
        if isinstance(n, ast.ClassDef) and n.name == "RequestStatus"
    )
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    ns = {"enum": enum}
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[future, status], type_ignores=[])
            ),
            "status",
            "exec",
        ),
        ns,
    )
    states = ns["RequestStatus"]
    ns.update(
        logger=NS(debug=lambda *a: None),
        remove_all=lambda items, removed: [r for r in items if r not in removed],
        _mtp_dw_diag_enabled=lambda: False,
        _mtp_dw_deep_diag_enabled=lambda: False,
        _mtp_dw_cleanup_request=lambda *a: None,
    )
    methods = {
        "finish_requests",
        "_free_request",
        "_free_blocks",
        "_update_from_kv_xfer_finished",
    }
    load_methods(ROOT / "vllm/v1/core/sched/scheduler.py", methods, ns)
    scheduler_type = type(
        "SchedulerUnderTest", (), {name: ns[name] for name in methods}
    )

    class Request:
        request_id = "r"
        client_index = 0
        status = states.WAITING_FOR_REMOTE_KVS

        def is_finished(self):
            return states.is_finished(self.status)

    class Queue(list):
        def remove_requests(self, requests):
            self[:] = [r for r in self if r not in requests]

    request = Request()
    freed = []
    scheduler = scheduler_type()
    scheduler.requests = {"r": request}
    scheduler.running, scheduler.waiting, scheduler.skipped_waiting = (
        [],
        Queue(),
        Queue([request]),
    )
    scheduler.finished_recving_kv_req_ids = {"r"} if received else set()
    scheduler.failed_recving_kv_req_ids = set()
    scheduler.finished_req_ids = set()
    scheduler.finished_req_ids_dict = defaultdict(set)
    scheduler.connector = NS(update_connector_output=lambda output: None)
    scheduler._connector_finished = lambda r: (sending, None)
    scheduler.encoder_cache_manager = NS(free=lambda r: None)
    scheduler.kv_cache_manager = NS(free=lambda r: freed.append(r.request_id))
    return scheduler, request, states, freed


def completion(*, received=(), sent=()):
    return NS(
        finished_recving=set(received),
        finished_sending=set(sent),
        completed_decode_window_saves={},
    )


def test_failed_cold_receive_then_ascend_store_completion_does_not_crash():
    scheduler, request, states, freed = fixture()
    scheduler.finish_requests({"r"}, states.FINISHED_ERROR)
    scheduler._update_from_kv_xfer_finished(completion(received={"r"}))

    # Ascend's actual store completion routine acknowledges a finished request
    # even when it has no outstanding store job. This is a distinct send fence.
    ns = {}
    load_methods(
        ROOT.parent / "LMCache-Ascend/lmcache_ascend/v1/cache_engine.py",
        {"get_finished_stores"},
        ns,
    )
    engine = NS(
        is_store_async=True,
        _store_lock=Lock(),
        _reported_finished_store_ids=set(),
        _deferred_finished_req_ids=set(),
        _pending_store_reqs=set(),
    )
    sent = ns["get_finished_stores"](engine, {"r"})
    assert sent == {"r"}
    scheduler._update_from_kv_xfer_finished(completion(sent=sent))
    assert freed == ["r"] and request.request_id not in scheduler.requests


@pytest.mark.parametrize("status_name", ["FINISHED_ERROR", "FINISHED_ABORTED"])
@pytest.mark.parametrize("order", ["receive_first", "send_first", "together"])
def test_bidirectional_retirement_waits_for_both_fences(status_name, order):
    scheduler, request, states, freed = fixture()
    scheduler.finish_requests({"r"}, getattr(states, status_name))
    assert request.is_finished() and not freed
    if order == "receive_first":
        scheduler._update_from_kv_xfer_finished(completion(received={"r"}))
        assert not freed and "r" in scheduler.requests
        scheduler._update_from_kv_xfer_finished(completion(sent={"r"}))
    elif order == "send_first":
        scheduler._update_from_kv_xfer_finished(completion(sent={"r"}))
        assert not freed and "r" in scheduler.requests
        scheduler._update_from_kv_xfer_finished(completion(received={"r"}))
    else:
        scheduler._update_from_kv_xfer_finished(completion(received={"r"}, sent={"r"}))
    assert freed == ["r"] and "r" not in scheduler.requests


def test_receive_only_abort_keeps_existing_completion_behavior():
    scheduler, _, states, freed = fixture(sending=False)
    scheduler.finish_requests({"r"}, states.FINISHED_ABORTED)
    assert not freed
    scheduler._update_from_kv_xfer_finished(completion(received={"r"}))
    assert freed == ["r"]


def test_already_received_request_waits_only_for_its_send():
    scheduler, _, states, freed = fixture(received=True)
    scheduler.finish_requests({"r"}, states.FINISHED_ERROR)
    assert not freed and not scheduler.finished_recving_kv_req_ids
    scheduler._update_from_kv_xfer_finished(completion(sent={"r"}))
    assert freed == ["r"]


def test_successful_receive_does_not_release_an_active_request():
    scheduler, _, _, freed = fixture()
    scheduler._update_from_kv_xfer_finished(completion(received={"r"}))
    assert not freed and scheduler.finished_recving_kv_req_ids == {"r"}


def test_unexpected_send_completion_still_exposes_protocol_errors():
    scheduler, _, _, _ = fixture()
    with pytest.raises(AssertionError):
        scheduler._update_from_kv_xfer_finished(completion(sent={"unknown"}))


@pytest.mark.parametrize("sending", [False, True])
def test_normal_finished_decode_does_not_wait_for_an_incoming_transfer(sending):
    scheduler, request, states, freed = fixture(sending=sending)
    request.status = states.RUNNING
    scheduler.skipped_waiting.clear()
    scheduler.running.append(request)
    scheduler.finish_requests({"r"}, states.FINISHED_STOPPED)
    if sending:
        assert not freed
        scheduler._update_from_kv_xfer_finished(completion(sent={"r"}))
    assert freed == ["r"] and not scheduler.requests


def test_unexpected_receive_completion_still_exposes_protocol_errors():
    scheduler, _, _, _ = fixture()
    with pytest.raises(AssertionError):
        scheduler._update_from_kv_xfer_finished(completion(received={"unknown"}))


def test_send_completion_cannot_free_an_active_receiver():
    scheduler, _, _, freed = fixture()
    with pytest.raises(AssertionError):
        scheduler._update_from_kv_xfer_finished(completion(sent={"r"}))
    assert not freed


def test_failed_transfer_retirement_preserves_other_running_requests():
    scheduler, request, states, freed = fixture()
    peer = type(request)()
    peer.request_id, peer.status = "peer", states.RUNNING
    scheduler.requests[peer.request_id] = peer
    scheduler.running.append(peer)
    scheduler.finish_requests({"r"}, states.FINISHED_ERROR)
    scheduler._update_from_kv_xfer_finished(completion(received={"r"}, sent={"r"}))
    assert freed == ["r"]
    assert scheduler.requests == {"peer": peer}
    assert scheduler.running == [peer] and peer.status == states.RUNNING
