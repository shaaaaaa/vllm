# SPDX-License-Identifier: Apache-2.0
"""Execute scheduler lifecycle methods on CPU, without importing device code."""

import ast
import json
import logging
import os
import socket
import time
from enum import IntEnum
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

SOURCE = Path(__file__).resolve().parents[2] / "vllm/v1/core/sched/scheduler.py"


class Status(IntEnum):
    WAITING = 1
    RUNNING = 2
    PREEMPTED = 3
    WAITING_FOR_REMOTE_KVS = 4


@pytest.fixture
def subject(monkeypatch, caplog):
    monkeypatch.setenv("PD_SERVING_PERF", "0")
    log = logging.getLogger("test.lifecycle")
    caplog.set_level(logging.INFO, logger=log.name)
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Scheduler"
    )
    methods = {
        node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)
    }
    ns = dict(
        json=json,
        os=os,
        socket=socket,
        time=time,
        logger=log,
        RequestStatus=Status,
        SERVING_PERF_ENABLED=False,
    )
    module = ast.parse("from __future__ import annotations")
    names = (
        "_log_request_transition",
        "_preempt_request",
        "_update_waiting_for_remote_kv",
    )
    module.body.extend(methods[name] for name in names)
    exec(compile(module, str(SOURCE), "exec"), ns)
    obj = type("SchedulerHostMethods", (), {name: ns[name] for name in names})()
    obj.vllm_config = NS(parallel_config=NS(data_parallel_rank=2))
    obj.log_stats = False
    obj.waiting = NS(prepend_request=lambda value: None)
    obj.kv_cache_manager = NS(free=lambda value: None, cache_blocks=lambda *args: None)
    obj.encoder_cache_manager = NS(free=lambda value: None)
    obj.connector = object()
    obj.failed_recving_kv_req_ids = set()
    obj.finished_recving_kv_req_ids = {"r"}
    req = NS(
        request_id="r",
        status=Status.RUNNING,
        num_preemptions=0,
        num_computed_tokens=123,
        num_tokens=130,
        num_prompt_tokens=100,
        num_output_tokens=30,
        num_output_placeholders=2,
        spec_token_ids=[-1],
        num_external_computed_tokens=100,
        dsa_compact_allocated=False,
        num_cached_tokens=-1,
        bootstrap_sample_pending=False,
    )
    records = lambda: [
        json.loads(record.message.split("] ", 1)[1]) for record in caplog.records
    ]
    return obj, req, records, methods


def test_preemption_records_previous_frontier_and_generation_at_perf_zero(subject):
    obj, req, records, _ = subject
    obj._preempt_request(req, 0.0)
    event = records()[0]
    assert event["event"] == "request_preempted" and event["status"] == "PREEMPTED"
    assert event["preemption_count"] == 1 and event["computed_tokens"] == 0
    assert event["previous_computed_tokens"] == 123 and event["history_tokens"] == 130
    assert event["dp_rank"] == 2 and event["wall_time_ns"] > 0
    assert req.spec_token_ids == []


def test_invalid_preemption_has_no_success_log(subject):
    obj, req, records, _ = subject
    req.status = Status.WAITING
    with pytest.raises(AssertionError):
        obj._preempt_request(req, 0.0)
    assert records() == []


@pytest.mark.parametrize("failed", [False, True])
def test_async_load_completion_is_not_reported_as_running_admission(subject, failed):
    obj, req, records, _ = subject
    req.num_preemptions = 1
    req.status = Status.WAITING_FOR_REMOTE_KVS
    if failed:
        obj.failed_recving_kv_req_ids.add(req.request_id)
        req.num_computed_tokens = 0
    obj._update_waiting_for_remote_kv(req)
    assert records()[0]["event"] == "resume_load_completed"
    assert records()[0]["success"] is not failed
    assert records()[0]["status"] == "WAITING_FOR_REMOTE_KVS"
    assert not obj.finished_recving_kv_req_ids


@pytest.mark.parametrize("count", [0, 1, 3])
def test_actual_admission_and_async_load_log_sites(subject, count):
    obj, req, records, methods = subject
    req.num_preemptions = count
    # Execute the actual condition/call sites inside schedule(), with the same
    # CPU operands. Allocation/promotion are covered by scheduler suites.
    sites = [
        node
        for node in ast.walk(methods["schedule"])
        if isinstance(node, ast.If)
        and any(
            isinstance(child, ast.Expr)
            and isinstance(child.value, ast.Call)
            and child.value.args
            and isinstance(child.value.args[0], ast.Constant)
            and child.value.args[0].value in ("resume_load_started", "request_resumed")
            for child in node.body
        )
    ]
    assert len(sites) == 2
    scope = dict(
        self=obj,
        request=req,
        request_id="r",
        num_new_tokens=2,
        num_external_computed_tokens=100,
        scheduled_spec_decode_tokens={"r": [-1]},
        scheduled_resumed_reqs=[req] if count else [],
    )
    for site in sites:
        exec(
            compile(ast.Module(body=[site], type_ignores=[]), str(SOURCE), "exec"),
            scope,
        )
    if not count:
        assert records() == []
    else:
        assert {event["event"] for event in records()} == {
            "resume_load_started",
            "request_resumed",
        }
        resumed = next(
            event for event in records() if event["event"] == "request_resumed"
        )
        assert (
            resumed["scheduled_tokens"] == 2 and resumed["scheduled_draft_tokens"] == 1
        )
        assert resumed["preemption_count"] == count


def test_streaming_continuation_does_not_reannounce_an_old_preemption(subject):
    obj, req, records, methods = subject
    req.num_preemptions = 2
    # _update_request_as_session preserves this counter but requeues WAITING;
    # schedule() puts it in scheduled_new_reqs, not scheduled_resumed_reqs.
    candidates = [
        node
        for node in ast.walk(methods["schedule"])
        if isinstance(node, ast.If)
        and any(
            isinstance(child, ast.Expr)
            and isinstance(child.value, ast.Call)
            and child.value.args
            and isinstance(child.value.args[0], ast.Constant)
            and child.value.args[0].value == "request_resumed"
            for child in node.body
        )
    ]
    assert len(candidates) == 1
    scope = dict(
        self=obj,
        request=req,
        request_id="r",
        num_new_tokens=2,
        num_external_computed_tokens=0,
        scheduled_spec_decode_tokens={},
        scheduled_resumed_reqs=[],
    )
    exec(
        compile(ast.Module(body=candidates, type_ignores=[]), str(SOURCE), "exec"),
        scope,
    )
    assert records() == []
