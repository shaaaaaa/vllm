# SPDX-License-Identifier: Apache-2.0
"""Exercise actual CPU guardian failure handling without loading NPU modules."""

import importlib.util
import sys
import time
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

SOURCE = Path(__file__).resolve().parents[3] / "vllm/v1/executor"
sys.path.insert(0, str(SOURCE))
SPEC = importlib.util.spec_from_file_location(
    "dp_guardian_tested", SOURCE / "dp_guardian.py"
)
guardian_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guardian_module)
Phase = guardian_module.Phase


@pytest.fixture
def guardian(tmp_path, monkeypatch):
    monkeypatch.setattr(
        guardian_module, "ProcessHandle", lambda *args: Mock(alive=lambda: True)
    )
    config = dict(
        epoch="epoch",
        session="dp0",
        members=["dp0", "dp1"],
        rank=0,
        key="ab" * 32,
        local_key="cd" * 32,
        parent_pid=1,
        interval=0.1,
        lease=5,
        startup_timeout=30,
        drain_timeout=60,
        tp_size=2,
        host="127.0.0.1",
        tombstone=str(tmp_path / "failed.json"),
    )
    guard = guardian_module.Guardian(config)
    guard.workers[0] = Mock(pid=2, alive=lambda: True)
    guard.member.phase = guard.cohort.phase = Phase.ACTIVE
    yield guard
    guard.listener.close()
    guard.selector.close()


def test_failed_tombstone_write_does_not_prevent_worker_termination(
    guardian, monkeypatch
):
    monkeypatch.setattr(Path, "write_text", Mock(side_effect=OSError("disk full")))
    guardian.abort("injected source failure")
    guardian.tick(time.monotonic())
    guardian.workers[0].kill.assert_called_once()


def test_backpressured_peer_cannot_prevent_local_termination(guardian):
    sock = Mock()
    peer = Mock(sock=sock, role="peer")
    peer.send.side_effect = BufferError("full control queue")
    guardian.peers["dp1"] = peer
    guardian.links[sock] = peer
    guardian.selector = Mock()
    guardian.abort("injected source failure")
    guardian.tick(time.monotonic())
    guardian.workers[0].kill.assert_called_once()


def test_unauthenticated_connection_loss_does_not_abort_epoch(guardian):
    sock = Mock()
    link = guardian_module.Link(sock, guardian.key)
    guardian.links[sock] = link
    guardian.selector = Mock()
    with pytest.raises(ValueError, match="Unauthenticated"):
        guardian.receive(link, b"unknown connection", time.monotonic())
    guardian.drop(link)
    assert guardian.member.phase == guardian.cohort.phase == Phase.ACTIVE
    guardian.workers[0].kill.assert_not_called()


def test_completed_rpc_is_not_failed_by_previous_heartbeat_deadline(
    guardian, monkeypatch
):
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    from dp_guardian_probe import client

    parent = client.ExecutorGuard.__new__(client.ExecutorGuard)
    parent.operation, parent.pending = 0, {}
    now = time.monotonic()
    operation = parent.begin(now + 0.02)
    guardian.local_message(
        NS(role="parent"), dict(kind="HEARTBEAT", remaining=0.02), now
    )
    parent.end(operation)  # Response consumed before its deadline, after the snapshot.
    guardian.tick(now + 0.03)  # Before the next fixed-rate parent heartbeat.
    assert not parent.pending
    assert guardian.member.phase == Phase.ACTIVE
    guardian.workers[0].kill.assert_not_called()


def test_partial_frames_remain_bounded():
    import socket

    a, b = socket.socketpair()
    link = guardian_module.Link(a, b"secret")
    try:
        b.sendall(b"x" * guardian_module.MAX_FRAME_BYTES)
        assert link.read() == []
        b.sendall(b"x")
        with pytest.raises(ValueError):
            link.read()
    finally:
        a.close()
        b.close()


@pytest.mark.parametrize("sent", [True, False])
def test_worker_reports_traceback_only_after_nonblocking_failure_notification(
    monkeypatch, sent
):
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    from dp_guardian_probe import client

    guard = client.WorkerGuard.__new__(client.WorkerGuard)
    guard.key, guard.epoch, guard.session = b"key", "epoch", "worker-1"
    guard.sock = Mock()
    guard.sock.send.side_effect = (
        (lambda data: len(data)) if sent else BlockingIOError()
    )
    exit_process = Mock(side_effect=SystemExit(1))
    report = Mock()
    monkeypatch.setattr(client.os, "_exit", exit_process)
    monkeypatch.setattr(client.traceback, "print_exception", report)
    with pytest.raises(SystemExit):
        guard.fail(ValueError("invalid source layout"))
    assert report.call_count == int(sent)
    exit_process.assert_called_once_with(1)


def test_failed_process_launch_does_not_poison_unallocated_epoch(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    from dp_guardian_probe import client

    marker = tmp_path / "epoch.json"
    config = dict(local_key="cd" * 32, epoch="e", tombstone=str(marker), lease=3)
    monkeypatch.setattr(
        client.subprocess, "Popen", Mock(side_effect=OSError("cannot launch"))
    )
    with pytest.raises(OSError, match="cannot launch"):
        client.ExecutorGuard(config)
    assert not marker.exists()


def test_terminal_guard_rejects_registration_before_opening_process_handle(
    guardian, monkeypatch
):
    guardian.abort("failed epoch")
    process_handle = Mock()
    monkeypatch.setattr(guardian_module, "ProcessHandle", process_handle)
    link = NS(role="", key=guardian.key)
    data = guardian_module.encode_frame(
        guardian.local_key,
        dict(
            epoch=guardian.epoch,
            session="worker-1",
            seq=0,
            kind="HELLO",
            role="worker",
            rank=1,
            pid=22,
        ),
    )
    with pytest.raises(ValueError, match="registration"):
        guardian.receive(link, data, time.monotonic())
    process_handle.assert_not_called()


def test_fallback_restores_previous_parent_death_policy(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    from dp_guardian_probe import client

    calls = []

    def prctl(operation, value, *unused):
        if operation == 2:
            value._obj.value = 15
        else:
            calls.append(value)
        return 0

    monkeypatch.setattr(client.sys, "platform", "linux")
    monkeypatch.setattr(client.signal, "SIGKILL", 9, raising=False)
    monkeypatch.setattr(client.ctypes, "CDLL", lambda *args, **kwargs: NS(prctl=prctl))
    monkeypatch.setattr(client.os, "getppid", lambda: 11)
    sock = Mock()
    data = iter(
        client.encode_frame(
            bytes.fromhex("ab" * 32), dict(epoch="e", kind="REGISTERED")
        )
    )
    sock.recv.side_effect = lambda count: bytes([next(data)])
    monkeypatch.setattr(
        client.socket, "create_connection", lambda *args, **kwargs: sock
    )
    guard = client.WorkerGuard(
        dict(parent_pid=11, key="ab" * 32, epoch="e", endpoint=("host", 1), lease=3), 0
    )
    guard.close()
    guard.close()
    assert calls == [client.signal.SIGKILL, 15]


def test_termination_precedes_diagnostic_disk_io(guardian, monkeypatch):
    def write(*args, **kwargs):
        guardian.workers[0].kill.assert_called_once()

    monkeypatch.setattr(Path, "write_text", write)
    guardian.abort("injected failure")
    guardian.tick(time.monotonic())


@pytest.mark.parametrize(
    "case",
    ["eligible", "disabled", "mixed", "unsupported_os", "lease", "single", "marker"],
)
def test_bootstrap_common_decision_and_serializable_descriptor(
    tmp_path, monkeypatch, case
):
    import pickle

    import torch
    import torch.distributed as dist

    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    from dp_guardian_probe import client

    monkeypatch.setattr(client.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(
        client.sys, "platform", "win32" if case == "unsupported_os" else "linux"
    )
    monkeypatch.setattr(client.os, "pidfd_open", lambda pid: 100, raising=False)
    monkeypatch.setattr(
        client.signal, "pidfd_send_signal", lambda *args: None, raising=False
    )
    monkeypatch.setattr(client.os, "close", Mock())
    parallel = NS(
        data_parallel_rank=0,
        data_parallel_size=2,
        tensor_parallel_size=4,
        data_parallel_master_ip="127.0.0.1",
        data_parallel_master_port=29600,
    )
    config = NS(
        parallel_config=parallel,
        shutdown_timeout=10,
        _sfa_independent_options=dict(
            requested=case != "disabled",
            eligible=True,
            single=case == "single",
            lease=30,
            valid_lease=True,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm",
        NS(
            envs=NS(
                VLLM_ENGINE_READY_TIMEOUT_S=60, VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30
            )
        ),
    )
    monkeypatch.setitem(sys.modules, "vllm.logger", NS(init_logger=lambda name: Mock()))
    monkeypatch.setitem(
        sys.modules, "vllm.utils.network_utils", NS(get_ip=lambda: "127.0.0.1")
    )
    values = {}
    rendezvous = Mock(
        set=lambda name, value: values.update({name: value}),
        get=lambda name: values[name],
    )
    factory = Mock(return_value=rendezvous)
    monkeypatch.setattr(dist, "TCPStore", factory)

    def all_reduce(rows, **kwargs):
        assert rows.device == torch.device("cpu")
        rows[:, 1] = rows[:, 0]
        if case == "mixed":
            rows[0, 1] = 0
        elif case == "lease":
            rows[4, 1] += 1
        elif case == "marker":
            rows[3, 1] = 1

    monkeypatch.setattr(dist, "all_reduce", all_reduce)
    if case in ("mixed", "lease", "single", "marker"):
        with pytest.raises((ValueError, RuntimeError)):
            client.prepare_dp_guardian(config, "group", NS(host="host", port=1))
    else:
        result = client.prepare_dp_guardian(config, "group", NS(host="host", port=1))
        if case == "eligible":
            assert pickle.loads(pickle.dumps(result)) == result
            assert result["members"] == ["dp0", "dp1"] and result["tp_size"] == 4
            assert result["drain_timeout"] == 30
        else:
            assert result is None
    assert factory.call_count == int(case == "eligible")
