# SPDX-License-Identifier: Apache-2.0
"""Execute changed executor/EngineCore methods with CPU message/worker stand-ins."""

import ast
import logging
import queue
import threading
import time
import traceback
import weakref
from collections import deque
from enum import IntEnum
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, Mock

import pytest

ROOT = Path(__file__).resolve().parents[3]
EXECUTOR = ROOT / "vllm/v1/executor/multiproc_executor.py"
CORE = ROOT / "vllm/v1/engine/core.py"


def method(path, owner, name, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == owner
    )
    node = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    node.decorator_list = []
    module = ast.parse("from __future__ import annotations")
    module.body.append(node)
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


def executor_namespace():
    tree = ast.parse(EXECUTOR.read_text(encoding="utf-8"))
    constants = next(
        n
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(getattr(t, "id", "") == "_GUARD_READ_ONLY_RPCS" for t in n.targets)
    )
    ns = dict(
        time=time,
        logger=logging.getLogger("guard-integration"),
        traceback=traceback,
        envs=NS(VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30, VLLM_ENGINE_READY_TIMEOUT_S=60),
        WorkerProc=NS(ResponseStatus=NS(SUCCESS=0)),
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[constants], type_ignores=[])),
            str(EXECUTOR),
            "exec",
        ),
        ns,
    )
    return ns


@pytest.mark.parametrize("name,fatal", [("list_loras", False), ("execute_model", True)])
def test_worker_preserves_control_errors_but_fails_model_epoch(name, fatal):
    ns = executor_namespace()
    loop = method(EXECUTOR, "WorkerProc", "worker_busy_loop", ns)
    guard = NS(fail=Mock(side_effect=SystemExit(1)))
    worker = NS(dp_guardian=guard)
    setattr(worker, name, Mock(side_effect=ValueError("injected")))
    owner = NS(
        worker=worker,
        rank=0,
        handle_output=Mock(),
        rpc_broadcast_mq=NS(
            dequeue=Mock(side_effect=[(name, (), {}, None), SystemExit(0)])
        ),
    )
    with pytest.raises(SystemExit) as result:
        loop(owner)
    assert result.value.code == int(fatal)
    assert guard.fail.call_count == int(fatal)
    assert owner.handle_output.call_count == int(not fatal)


@pytest.mark.parametrize("name,fatal", [("list_loras", False), ("execute_model", True)])
def test_parent_preserves_terminal_control_error_and_tracks_existing_deadline(
    name, fatal
):
    ns = executor_namespace()
    call = method(EXECUTOR, "MultiprocExecutor", "collective_rpc", ns)
    guard = NS(
        begin=Mock(return_value=7), end=Mock(), fail=Mock(), phase=NS(value="active")
    )
    owner = NS(
        dp_guardian=guard,
        is_failed=False,
        rpc_broadcast_mq=NS(enqueue=Mock()),
        response_mqs=[NS(dequeue=Mock(return_value=(1, "injected")))],
        futures_queue=deque(),
    )
    with pytest.raises(RuntimeError, match="injected"):
        call(owner, name, timeout=30, unique_reply_rank=0)
    assert guard.fail.call_count == int(fatal)
    assert guard.end.call_count == int(not fatal)
    assert 29 < guard.begin.call_args.args[0] - time.monotonic() <= 30


def test_disabled_rpc_has_no_supervision_or_new_timeout():
    ns = executor_namespace()
    call = method(EXECUTOR, "MultiprocExecutor", "collective_rpc", ns)
    response = Mock(return_value=(0, "ok"))
    owner = NS(
        dp_guardian=None,
        is_failed=False,
        rpc_broadcast_mq=NS(enqueue=Mock()),
        response_mqs=[NS(dequeue=response)],
        futures_queue=deque(),
    )
    assert call(owner, "list_loras", unique_reply_rank=0) == "ok"
    assert response.call_args.kwargs["timeout"] is None


def test_duplicate_failure_notifications_shutdown_once():
    ns = executor_namespace()
    fail = method(EXECUTOR, "MultiprocExecutor", "_handle_worker_failure", ns)
    callback = Mock()
    owner = NS(
        _failure_lock=threading.Lock(),
        is_failed=False,
        failure_callback=callback,
        shutdown=Mock(),
        dp_guardian=None,
    )
    threads = [threading.Thread(target=fail, args=(owner,)) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(3)
    owner.shutdown.assert_called_once()
    callback.assert_called_once()


def test_worker_monitor_reports_observed_pid_and_exitcode_before_shutdown():
    ns = executor_namespace()
    ns.update(
        weakref=weakref,
        multiprocessing=NS(connection=NS(wait=Mock(return_value=[7]))),
    )
    monitor = method(EXECUTOR, "MultiprocExecutor", "start_worker_monitor", ns)
    logger = ns["logger"] = Mock()
    proc = NS(sentinel=7, name="worker", pid=123, exitcode=-11)
    owner = Mock(workers=[NS(proc=proc)], shutting_down=False, _guardian_drained=False)
    owner._handle_worker_failure.side_effect = lambda: logger.error.assert_called_once()
    monitor(owner, inline=True)
    assert logger.error.call_args.args[1:] == ("worker", 123, -11)
    owner._handle_worker_failure.assert_called_once()


@pytest.mark.parametrize(
    "failure", [RuntimeError("queue"), SystemExit(0), KeyboardInterrupt()]
)
def test_guarded_worker_loop_exit_notifies_before_any_cleanup(monkeypatch, failure):
    import sys

    ns = executor_namespace()
    guard = NS(fail=Mock(side_effect=SystemExit(91)))
    owner = NS(
        worker=NS(dp_guardian=guard, worker=NS()),
        worker_response_mq=Mock(),
        rpc_broadcast_mq=Mock(),
        peer_response_handles=[],
        monitor_death_pipe=Mock(),
        worker_busy_loop=Mock(side_effect=failure),
        # The stand-in raises SystemExit instead of os._exit; this finally runs
        # only in the test and verifies notification preceded any cleanup.
        shutdown=Mock(side_effect=lambda: guard.fail.assert_called_once_with(failure)),
    )
    constructor = Mock(return_value=owner, READY_STR="READY")
    ns.update(
        threading=threading,
        signal=Mock(),
        os=Mock(),
        sys=sys,
        maybe_init_worker_tracer=Mock(),
        WorkerProc=constructor,
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.executor.dp_guardian_client",
        NS(WorkerGuard=Mock(return_value=guard)),
    )
    main = method(EXECUTOR, "WorkerProc", "worker_main", ns)
    with pytest.raises(SystemExit) as result:
        main(
            vllm_config=NS(_dp_guardian_worker={"epoch": "test"}),
            ready_pipe=Mock(),
            rank=0,
        )
    assert result.value.code == 91
    guard.fail.assert_called_once_with(failure)


@pytest.mark.parametrize("case", ["constructor", "fallback", "disabled"])
def test_worker_main_supervision_covers_constructor_and_preserves_legacy(
    monkeypatch, case
):
    import sys

    ns = executor_namespace()
    failure = (
        RuntimeError("initialization failed")
        if case == "constructor"
        else SystemExit(0)
    )
    guard = NS(fail=Mock(side_effect=SystemExit(91)))
    owner = NS(
        worker=NS(worker=NS(), dp_guardian=guard),
        worker_response_mq=Mock(),
        rpc_broadcast_mq=Mock(),
        peer_response_handles=[],
        monitor_death_pipe=Mock(),
        shutdown=Mock(),
    )

    def loop():
        owner.worker.dp_guardian = None  # Successful common fallback or disabled.
        raise failure

    owner.worker_busy_loop = loop
    constructor = Mock(return_value=owner, READY_STR="READY")
    if case == "constructor":
        constructor.side_effect = failure
    ns.update(
        threading=threading,
        signal=Mock(),
        os=Mock(),
        maybe_init_worker_tracer=Mock(),
        WorkerProc=constructor,
    )
    factory = Mock(return_value=guard)
    monkeypatch.setitem(
        sys.modules, "vllm.v1.executor.dp_guardian_client", NS(WorkerGuard=factory)
    )
    main = method(EXECUTOR, "WorkerProc", "worker_main", ns)
    with pytest.raises(SystemExit) as result:
        main(
            vllm_config=NS(
                _dp_guardian_worker=None if case == "disabled" else {"epoch": "test"}
            ),
            ready_pipe=Mock(),
            rank=0,
        )
    assert result.value.code == (91 if case == "constructor" else 0)
    assert guard.fail.call_count == int(case == "constructor")
    assert factory.call_count == int(case != "disabled")
    assert owner.shutdown.call_count == int(case != "constructor")


class Shutdown(IntEnum):
    RUNNING = 0
    REQUESTED = 1
    SHUTTING_DOWN = 2


def core_types():
    tree = ast.parse(CORE.read_text(encoding="utf-8"))
    ns = dict(
        EngineShutdownState=Shutdown,
        logger=Mock(),
        RequestStatus=NS(FINISHED_ABORTED="aborted"),
    )
    module = ast.parse("from __future__ import annotations")
    for source, target, bases, methods in (
        ("EngineCoreProc", "Base", [], ("_handle_shutdown", "_start_shutdown")),
        (
            "DPEngineCoreProc",
            "Core",
            [ast.Name(id="Base", ctx=ast.Load())],
            ("_handle_shutdown",),
        ),
    ):
        cls = next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == source
        )
        body = [
            n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in methods
        ]
        module.body.append(
            ast.ClassDef(
                name=target, bases=bases, keywords=[], body=body, decorator_list=[]
            )
        )
    exec(compile(ast.fix_missing_locations(module), str(CORE), "exec"), ns)
    return ns["Core"]


def test_core_waits_for_group_quiescence_before_resource_drain():
    owner = core_types()()
    guard = NS(
        phase=NS(value="active"),
        fatal="",
        reason="",
        command=Mock(),
        condition=MagicMock(),
        config=dict(interval=0.1),
        quiescent_wave=None,
    )
    owner.model_executor = NS(dp_guardian=guard, drain_independent_dp=Mock())
    owner.shutdown_state = Shutdown.REQUESTED
    owner._guardian_stop_requested = owner._guardian_drained = False
    owner.vllm_config = NS(shutdown_timeout=0)
    owner.scheduler = NS(
        get_num_unfinished_requests=lambda: 0, finish_requests=Mock(return_value=[])
    )
    owner._send_abort_outputs = Mock()
    owner.has_work = Mock(return_value=False)
    owner.current_wave = 7
    owner.input_queue = queue.Queue()
    owner._process_input_queue = Mock()
    assert owner._handle_shutdown()
    guard.command.assert_called_once_with("STOP_REQUEST")
    owner.model_executor.drain_independent_dp.assert_not_called()
    guard.phase.value = "quiescing"
    assert owner._handle_shutdown() and guard.quiescent_wave == 7
    owner.model_executor.drain_independent_dp.assert_not_called()
    guard.phase.value = "draining"
    assert owner._handle_shutdown()
    assert owner._handle_shutdown()
    owner.model_executor.drain_independent_dp.assert_called_once_with(7)
    guard.phase.value = "stopped"
    assert not owner._handle_shutdown()
    owner.scheduler.finish_requests.assert_called_once()
