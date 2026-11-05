# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Executor/worker attachment to the CPU-only DP guardian.

Only the executor owns the fixed-rate control thread. Workers register once and
send only on failure. Outstanding RPC deadlines are observed, never consumed or
cancelled, by the control thread.
"""

import ctypes
import hashlib
import json
import os
import queue
import select
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import Any

from .dp_guardian_protocol import MAX_FRAME_BYTES, Phase, decode_frame, encode_frame


def prepare_dp_guardian(config: Any, group: Any, store: Any) -> dict[str, Any] | None:
    """Agree on platform opt-in before worker creation, using the existing CPU group."""
    options = getattr(config, "_sfa_independent_options", None)
    if options is None:
        return None
    import torch
    import torch.distributed as dist

    from vllm import envs
    from vllm.logger import init_logger
    from vllm.utils.network_utils import get_ip

    parallel = config.parallel_config
    rank, size = parallel.data_parallel_rank, parallel.data_parallel_size
    identity = json.dumps(
        (parallel.data_parallel_master_ip, parallel.data_parallel_master_port, rank),
        separators=(",", ":"),
    )
    marker = (
        Path(tempfile.gettempdir())
        / "vllm-dp-guardian"
        / (hashlib.sha256(identity.encode()).hexdigest() + ".json")
    )
    supported_os = (
        sys.platform == "linux"
        and hasattr(os, "pidfd_open")
        and hasattr(signal, "pidfd_send_signal")
    )
    if supported_os and options["requested"]:
        try:
            handle = os.pidfd_open(os.getpid())
            try:
                signal.pidfd_send_signal(handle, 0)
            finally:
                os.close(handle)
        except OSError:
            supported_os = (
                False  # Includes old kernels and seccomp-denied pidfd syscalls.
            )
    valid_lease = options["valid_lease"]
    rows = torch.zeros((6, size), dtype=torch.int64)
    rows[:, rank] = torch.tensor(
        [
            bool(options["requested"]),
            bool(options["eligible"] and supported_os),
            bool(options["single"]),
            marker.exists(),
            round(options["lease"] * 1000) if valid_lease else 0,
            valid_lease,
        ],
        dtype=torch.int64,
    )
    dist.all_reduce(rows, group=group)
    if bool(rows[3].any()):
        raise RuntimeError(
            "A DP guardian failed-epoch marker remains; prove resource drain/"
            "device fencing before removing it"
        )
    if not bool(rows[0].any()):
        return None
    if not bool(rows[0].all()) or bool(rows[2].any()):
        raise ValueError(
            "Independent DP must be requested on every peer and cannot use "
            "SINGLE_AGREEMENT"
        )
    if not bool(rows[5].all()) or int(rows[4].min()) != int(rows[4].max()):
        raise ValueError("DP guardian leases must match and be in [2, 86400] seconds")
    if not bool(rows[1].all()):
        init_logger(__name__).warning(
            "Independent SFA DP is ineligible; retaining existing execution (%s)",
            options.get("reason") or "peer topology/OS",
        )
        return None
    timeout = float(envs.VLLM_ENGINE_READY_TIMEOUT_S)
    rendezvous = dist.TCPStore(
        store.host, store.port, None, False, timeout=timedelta(seconds=timeout)
    )
    if rank == 0:
        rendezvous.set(
            "sfa_guardian/epoch",
            json.dumps(dict(epoch=uuid.uuid4().hex, key=os.urandom(32).hex())),
        )
    shared = json.loads(rendezvous.get("sfa_guardian/epoch"))
    return dict(
        **shared,
        rank=rank,
        members=[f"dp{i}" for i in range(size)],
        session=f"dp{rank}",
        store=(store.host, store.port),
        host=get_ip(),
        tp_size=parallel.tensor_parallel_size,
        startup_timeout=timeout,
        lease=float(options["lease"]),
        interval=0.5,
        drain_timeout=float(
            max(envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS, config.shutdown_timeout)
        ),
        tombstone=str(marker),
    )


def start_dp_guardian(bootstrap: dict[str, Any]) -> "ExecutorGuard":
    """Launch before memory profiling; exchange addresses only during startup."""
    import torch.distributed as dist

    config = dict(bootstrap, parent_pid=os.getpid(), local_key=os.urandom(32).hex())
    guard = ExecutorGuard(config)
    try:
        store = dist.TCPStore(
            *bootstrap["store"],
            None,
            False,
            timeout=timedelta(seconds=bootstrap["startup_timeout"]),
        )
        name = f"sfa_guardian/{bootstrap['epoch']}/leader"
        if bootstrap["rank"] == 0:
            store.set(name, json.dumps(guard.endpoint))
        guard.connect(tuple(json.loads(store.get(name))))
        return guard
    except BaseException:
        guard.cancel_startup()  # This helper has not exposed any worker descriptor yet.
        raise


def receive_line(sock: socket.socket) -> bytes:
    """Read a bounded startup reply; only used before live execution."""
    data = bytearray()
    while not data.endswith(b"\n"):
        byte = sock.recv(1)
        if not byte:
            raise EOFError("Guardian closed during registration")
        data.extend(byte)
        if len(data) > MAX_FRAME_BYTES:
            raise ValueError("Oversized guardian registration reply")
    return bytes(data)


class WorkerGuard:
    """Registration and error-only notification, without a worker-side thread."""

    def __init__(self, info: dict[str, Any], rank: int):
        parent = info["parent_pid"]
        if os.getppid() != parent:
            raise RuntimeError("Independent DP worker has the wrong parent")
        self._previous_parent_signal = None
        if sys.platform == "linux":
            # PR_SET_PDEATHSIG must precede device initialization, with the race
            # check after installation. SIGKILL does not depend on the Python GIL.
            self._libc = ctypes.CDLL(None, use_errno=True)
            previous = ctypes.c_int()
            if self._libc.prctl(2, ctypes.byref(previous), 0, 0, 0):
                raise OSError(ctypes.get_errno(), "Cannot read parent-death signal")
            if self._libc.prctl(1, signal.SIGKILL, 0, 0, 0):
                raise OSError(ctypes.get_errno(), "Cannot arm parent-death signal")
            self._previous_parent_signal = previous.value
            if os.getppid() != parent:
                raise RuntimeError("Executor exited during worker registration")
        self.key, self.epoch = bytes.fromhex(info["key"]), info["epoch"]
        self.session = f"worker-{rank}"
        self.sock = None
        try:
            self.sock = socket.create_connection(
                tuple(info["endpoint"]), timeout=info["lease"]
            )
            self.sock.sendall(
                encode_frame(
                    self.key,
                    dict(
                        epoch=self.epoch,
                        session=self.session,
                        seq=0,
                        kind="HELLO",
                        role="worker",
                        rank=rank,
                        pid=os.getpid(),
                    ),
                )
            )
            reply = decode_frame(self.key, receive_line(self.sock))
            if reply.get("epoch") != self.epoch or reply.get("kind") != "REGISTERED":
                raise RuntimeError("Guardian did not register this worker")
            self.sock.setblocking(False)
        except BaseException:
            self.close()
            raise

    def fail(self, error: BaseException) -> None:
        """Exit on fatal work even if notification cannot be sent immediately."""
        notified = False
        try:
            reason = type(error).__name__
            try:
                # Preserve useful evidence before the supervisor can kill us.
                # Read only builtin exception/traceback metadata: custom __str__
                # or repr of an argument may block. Never serialize live frames.
                origin = BaseException.__traceback__.__get__(error)
                while origin is not None and origin.tb_next is not None:
                    origin = origin.tb_next
                location = (
                    f" at {Path(origin.tb_frame.f_code.co_filename).name[:80]}"
                    f":{origin.tb_lineno}"
                    if origin is not None
                    else ""
                )
                args = BaseException.args.__get__(error)
                detail = args[0][:256] if args and type(args[0]) is str else ""
                reason = f"{self.session} {reason[:64]}{location}: {detail}"[:256]
                reason = reason.replace("\n", " ").replace("\r", " ")
            except BaseException:
                pass  # Metadata reporting must not prevent failure notification.
            message = dict(
                epoch=self.epoch,
                session=self.session,
                seq=1,
                kind="ABORT",
                reason=reason,
            )
            data = encode_frame(self.key, message)
            notified = self.sock.send(data) == len(data)
        except BaseException:
            pass
        try:
            if notified:
                # Notify first: if stderr blocks, the independent supervisor can
                # still terminate us. Keep the original traceback out of IPC.
                traceback.print_exception(
                    type(error), error, error.__traceback__, file=sys.stderr
                )
        finally:
            # No finally/finalizer may release a source with unknown completion.
            # The independent guardian notices this exit even if send was partial.
            os._exit(1)

    def close(self) -> None:
        previous = self._previous_parent_signal
        if previous is not None:
            if self._libc.prctl(1, previous, 0, 0, 0):
                raise OSError(ctypes.get_errno(), "Cannot restore parent-death signal")
            self._previous_parent_signal = None
        if self.sock is not None:
            self.sock.close()


class ExecutorGuard:
    """Parent-owned supervisor with fixed-rate, bounded control traffic."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.key, self.epoch = bytes.fromhex(config["local_key"]), config["epoch"]
        self.phase, self.reason = Phase.BOOT, ""
        self.connected = False
        self.commands: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=8)
        self.pending: dict[int, float] = {}
        self.sequence = 0
        self.operation = 0
        self.condition = threading.Condition()
        self.fatal = ""
        self.closed = False
        self.failure_callback: Callable[[], None] | None = None
        self.quiescent_wave: int | None = None
        self.drain_callback: Callable[[], None] | None = None
        self._workers_exposed = False
        self.process = self.sock = self.thread = self._startup_reader = None
        marker = Path(config["tombstone"])
        marker.parent.mkdir(parents=True, exist_ok=True)
        # Leave a durable barrier even if a later failure prevents writing details.
        reservation = marker.open("x", encoding="utf-8")
        try:
            with reservation:
                json.dump(dict(epoch=self.epoch, completion="unreleased"), reservation)
            self._launch(config)
        except BaseException:
            self.cancel_startup()
            raise

    def _launch(self, config: dict[str, Any]) -> None:
        env = os.environ.copy()
        env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
        self.process = subprocess.Popen(
            [sys.executable, "-u", str(Path(__file__).with_name("dp_guardian.py"))],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            close_fds=True,
            env=env,
        )
        assert self.process.stdin is not None and self.process.stdout is not None
        self.process.stdin.write(json.dumps(config).encode() + b"\n")
        self.process.stdin.close()
        replies: queue.Queue[bytes] = queue.Queue(maxsize=1)
        self._startup_reader = threading.Thread(
            target=lambda: replies.put(self.process.stdout.readline()), daemon=True
        )
        self._startup_reader.start()
        reply = replies.get(timeout=config["lease"])
        self._startup_reader.join(timeout=config["lease"])
        self.process.stdout.close()
        port = json.loads(reply)["port"]
        self.endpoint = (config["host"], port)
        self.sock = socket.create_connection(self.endpoint, timeout=config["lease"])
        self.sock.sendall(encode_frame(self.key, self.message("HELLO", role="parent")))
        self.sock.setblocking(False)
        self.thread = threading.Thread(
            target=self.run, daemon=True, name="DPGuardianControl"
        )
        self.thread.start()

    def message(self, kind: str, **values: Any) -> dict[str, Any]:
        self.sequence += 1
        return dict(
            epoch=self.epoch, session="parent", seq=self.sequence, kind=kind, **values
        )

    def worker_info(self) -> dict[str, Any]:
        self._workers_exposed = True
        return dict(
            epoch=self.epoch,
            key=self.config["local_key"],
            endpoint=self.endpoint,
            parent_pid=self.config["parent_pid"],
            lease=self.config["lease"],
        )

    def cancel_startup(self) -> None:
        """Undo a launch only before any worker descriptor/ownership is exported."""
        if self._workers_exposed:
            raise RuntimeError("Cannot disarm a guardian with exported workers")
        self.closed = True
        if self.process is not None:
            if self.process.poll() is None:
                self.process.kill()
            self.process.wait(timeout=self.config["lease"])
        if self.sock is not None:
            self.sock.close()
        for thread in (self.thread, self._startup_reader):
            if thread is not None and thread.is_alive():
                thread.join(timeout=self.config["lease"])
        if self.process is not None:
            for stream in (self.process.stdin, self.process.stdout):
                if stream is not None:
                    stream.close()
        Path(self.config["tombstone"]).unlink(missing_ok=True)

    def command(self, kind: str, **values: Any) -> None:
        """Queue lifecycle control; warm RPCs never call this method."""
        try:
            self.commands.put_nowait(dict(kind=kind, **values))
        except queue.Full:
            self.fail("Guardian lifecycle queue is full")

    def connect(self, leader: tuple[str, int]) -> None:
        self.command("CONNECT", leader=leader)
        deadline = time.monotonic() + self.config["startup_timeout"]
        with self.condition:
            while not self.connected and not self.fatal:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.fail(
                        "DP supervisors did not all register before worker startup"
                    )
                    break
                self.condition.wait(min(remaining, self.config["lease"]))
        if self.fatal:
            raise RuntimeError(self.fatal)

    def activate(
        self, digest: str | None, setup: Callable[[bool], None] | None = None
    ) -> bool:
        self.command("READY", digest=digest)
        deadline = time.monotonic() + self.config["startup_timeout"]

        def wait_for(phases):
            with self.condition:
                while self.phase not in (*phases, Phase.ABORTED):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        self.fail("Independent DP activation timed out")
                        break
                    self.condition.wait(min(remaining, self.config["lease"]))
            if self.fatal or self.phase == Phase.ABORTED:
                raise RuntimeError(
                    self.fatal or self.reason or "Independent DP activation failed"
                )

        wait_for((Phase.COMMITTING, Phase.DISARMING))
        enabled = self.phase == Phase.COMMITTING
        try:
            if setup is not None:
                setup(enabled)
        except BaseException:
            self.fail("Worker activation failed")
            raise
        # No parent may advertise serving readiness until every worker cohort
        # has installed the selected path (including pre-activation fallback).
        self.command("COMMIT_ACK" if enabled else "FALLBACK_ACK")
        wait_for((Phase.ACTIVE, Phase.DISARMED))
        if self.phase == Phase.DISARMED:
            return False
        if self.phase != Phase.ACTIVE or self.fatal:
            raise RuntimeError(
                self.fatal or self.reason or "Independent DP activation failed"
            )
        return True

    def begin(self, deadline: float) -> int:
        """Track the existing RPC deadline without sending a control message."""
        self.operation += 1
        self.pending[self.operation] = deadline
        return self.operation

    def end(self, operation: int) -> None:
        self.pending.pop(operation, None)

    def fail(self, reason: str) -> None:
        if not self.fatal:
            self.fatal = reason[:256]

    def run(self) -> None:
        buffer, output = bytearray(), bytearray()
        last_sequence, seen = -1, time.monotonic()
        next_heartbeat = seen
        terminal_ack = False
        try:
            while not self.closed:
                now = time.monotonic()
                if now >= next_heartbeat:
                    # CPython copies these scalar entries while holding its GIL.
                    # This observer neither mutates nor drains executor futures.
                    deadlines = self.pending.copy().values()
                    remaining = min(deadlines) - now if deadlines else None
                    if remaining is not None and remaining <= 0:
                        self.fail("Executor operation deadline expired")
                    if self.fatal:
                        data = self.message("ABORT", reason=self.fatal)
                    else:
                        data = self.message("HEARTBEAT")
                    output.extend(encode_frame(self.key, data))
                    if self.phase == Phase.QUIESCING:
                        output.extend(
                            encode_frame(
                                self.key,
                                self.message("QUIESCENT", wave=self.quiescent_wave),
                            )
                        )
                    next_heartbeat = now + self.config["interval"]
                while not self.commands.empty():
                    command = self.commands.get_nowait()
                    output.extend(encode_frame(self.key, self.message(**command)))
                if len(output) > MAX_FRAME_BYTES * 2:
                    raise BufferError("Guardian is not consuming parent control")
                reads, writes, _ = select.select(
                    [self.sock],
                    [self.sock] if output else [],
                    [],
                    self.config["interval"],
                )
                if writes:
                    sent = self.sock.send(output)
                    del output[:sent]
                if reads:
                    data = self.sock.recv(MAX_FRAME_BYTES)
                    if not data:
                        if self.phase in (Phase.STOPPED, Phase.DISARMED):
                            break
                        raise EOFError("DP guardian exited")
                    buffer.extend(data)
                    while b"\n" in buffer:
                        raw, _, rest = buffer.partition(b"\n")
                        buffer[:] = rest
                        message = decode_frame(self.key, raw)
                        seq = message.get("seq")
                        if (
                            message.get("epoch") != self.epoch
                            or type(seq) is not int
                            or seq <= last_sequence
                        ):
                            continue
                        last_sequence, seen = seq, time.monotonic()
                        with self.condition:
                            previous = self.phase
                            self.phase = Phase(message["phase"])
                            self.connected = message.get("connected") is True
                            self.reason = str(message.get("reason", ""))
                            self.condition.notify_all()
                        if (
                            self.phase in (Phase.QUIESCING, Phase.DRAINING)
                            and previous != self.phase
                            and self.drain_callback
                        ):
                            self.drain_callback()
                        if (
                            self.phase in (Phase.STOPPED, Phase.DISARMED)
                            and not terminal_ack
                        ):
                            output.extend(
                                encode_frame(self.key, self.message("CLOSED"))
                            )
                            terminal_ack = True
                        if self.phase == Phase.ABORTED:
                            self.fail(self.reason or "DP peer failed")
                    if len(buffer) > MAX_FRAME_BYTES:
                        raise ValueError("Oversized guardian state")
                if time.monotonic() - seen >= self.config["lease"]:
                    raise TimeoutError("DP guardian stopped responding")
                if self.fatal and self.failure_callback:
                    callback, self.failure_callback = self.failure_callback, None
                    callback()
                if terminal_ack and not output:
                    break
        except BaseException as exc:
            self.fail(str(exc))
            with self.condition:
                self.phase = Phase.ABORTED
                self.condition.notify_all()
            if self.failure_callback:
                callback, self.failure_callback = self.failure_callback, None
                callback()
        finally:
            self.sock.close()

    def close(self) -> None:
        """Disarm only after group stop/fallback; unknown epochs retain marker."""
        if self.closed:
            return
        clean = self.phase in (Phase.STOPPED, Phase.DISARMED) and not self.fatal
        if not clean:
            self.fail("Executor closed before a completed group drain")
            return
        self.thread.join(timeout=self.config["lease"])
        if self.thread.is_alive():
            raise TimeoutError("Guardian terminal acknowledgement did not drain")
        self.closed = True
        self.failure_callback = self.drain_callback = None
        self.process.wait(timeout=self.config["lease"])
        Path(self.config["tombstone"]).unlink()
