# SPDX-License-Identifier: Apache-2.0
"""Real CPU processes and TCP transport; no NPU or actual DMA is simulated."""

import json
import os
import queue
import secrets
import select
import socket
import subprocess
import sys
import sysconfig
import threading
import time
from pathlib import Path

import psutil
import pytest

PROBE = Path(__file__).with_name("dp_guardian_probe.py")


def read(process):
    replies = queue.Queue(maxsize=1)
    thread = threading.Thread(
        target=lambda: replies.put(process.stdout.readline()), daemon=True
    )
    thread.start()
    data = replies.get(timeout=15)
    assert data, f"executor exited: {process.poll()}"
    return json.loads(data)


def send(process, data):
    process.stdin.write(json.dumps(data).encode() + b"\n")
    process.stdin.flush()


def relay(listener, destination, blocked, stopped):
    """A real one-way blackhole: member health still reaches the leader."""
    connections = []
    try:
        listener.settimeout(10)
        member, _ = listener.accept()
        connections.append(member)
        leader = socket.create_connection(destination, timeout=3)
        connections.append(leader)
        for connection in connections:
            connection.settimeout(1)
        while not stopped.is_set():
            readable, _, _ = select.select(connections, [], [], 0.1)
            for source in readable:
                data = source.recv(16384)
                if not data:
                    return
                if not (source is leader and blocked.is_set()):
                    (member if source is leader else leader).sendall(data)
    except OSError:
        pass  # Termination closes either peer; the test checks cohort outcomes.
    finally:
        for connection in connections:
            connection.close()


@pytest.mark.parametrize(
    "fault",
    [
        "failure",
        "passive_worker",
        "guardian",
        "deadline",
        "drain",
        "fallback",
        "activation",
        "contract",
        "parent",
        "silent_guardian",
        "one_way",
    ],
)
def test_actual_guardian_cohort(tmp_path, fault):
    parents, children, records = [], [], []
    suspended = None
    proxy = thread = None
    blocked, stopped = threading.Event(), threading.Event()
    epoch, key = secrets.token_hex(16), secrets.token_hex(32)
    try:
        for rank in range(4):
            # Windows venv launchers insert another parent process. Launch the
            # underlying interpreter directly so the real parent/pidfd contract
            # is exercised; expose this test environment's CPU dependencies.
            executable = (
                sys._base_executable if sys.platform == "win32" else sys.executable
            )
            env = os.environ.copy()
            env["PYTHONPATH"] = os.pathsep.join(
                filter(
                    None,
                    (
                        sysconfig.get_path("purelib"),
                        env.get("PYTHONPATH"),
                    ),
                )
            )
            process = subprocess.Popen(
                [executable, "-u", str(PROBE), "executor"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                env=env,
            )
            parents.append(process)
            send(
                process,
                dict(
                    epoch=epoch,
                    session=f"dp{rank}",
                    members=[f"dp{i}" for i in range(4)],
                    rank=rank,
                    key=key,
                    local_key=secrets.token_hex(32),
                    interval=0.1,
                    lease=3.0,
                    startup_timeout=30.0,
                    drain_timeout=10.0,
                    tp_size=2,
                    host="127.0.0.1",
                    tombstone=str(tmp_path / f"dp{rank}.json"),
                ),
            )
            record = read(process)
            records.append(record)
            children.append(psutil.Process(record["guardian"]))
        if fault == "one_way":
            proxy = socket.socket()
            proxy.bind(("127.0.0.1", 0))
            proxy.listen(1)
            thread = threading.Thread(
                target=relay,
                args=(proxy, tuple(records[0]["endpoint"]), blocked, stopped),
                daemon=True,
            )
            thread.start()
        for rank, process in enumerate(parents):
            send(
                process,
                dict(
                    leader=proxy.getsockname()
                    if proxy is not None and rank == 1
                    else records[0]["endpoint"],
                    digest=(
                        None
                        if fault == "fallback" and rank == 2
                        else "b" * 64
                        if fault == "contract" and rank == 2
                        else "a" * 64
                    ),
                    fail_install=fault == "activation" and rank == 2,
                ),
            )
        for record, process in zip(records, parents):
            record.update(read(process))
            children.extend(psutil.Process(pid) for pid in record["workers"])
        startup_failure = fault in ("activation", "contract")
        if not startup_failure:
            for process in parents:
                assert read(process)["active"] == (fault != "fallback")
        started = time.monotonic()
        if fault == "fallback" or startup_failure:
            pass
        elif fault == "drain":
            for process in parents:
                send(process, dict(kind="drain"))
        else:
            for rank, process in enumerate(parents):
                kind = (
                    "fail"
                    if fault == "failure"
                    else "timeout"
                    if fault == "deadline"
                    else "wait"
                )
                send(process, dict(kind=kind if rank == 0 else "wait"))
            if fault in ("passive_worker", "guardian"):
                pid = (
                    records[0]["workers"][1]
                    if fault == "passive_worker"
                    else records[0]["guardian"]
                )
                next(p for p in children if p.pid == pid).kill()
            elif fault == "parent":
                parents[0].kill()
            elif fault == "silent_guardian":
                suspended = next(p for p in children if p.pid == records[0]["guardian"])
                suspended.suspend()  # Open sockets, no health frames or EOF.
            elif fault == "one_way":
                blocked.set()
        for process in parents:
            code = process.wait(timeout=12)
            assert (code == 0) == (fault in ("drain", "fallback"))
        assert time.monotonic() - started < 12
        if suspended is not None:
            suspended.resume()
            suspended = None
        if fault in ("drain", "fallback"):
            assert not list(tmp_path.glob("*.json"))
        else:
            assert len(list(tmp_path.glob("*.json"))) == 4
        for child in children:
            child.wait(timeout=12)
    finally:
        stopped.set()
        if proxy is not None:
            proxy.close()
            thread.join(timeout=3)
        if suspended is not None:
            suspended.resume()
        # Only handles recorded from this test's children are eligible for cleanup.
        for process in parents:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
            process.stdin.close()
            process.stdout.close()
        for child in children:
            try:
                if child.is_running():
                    child.kill()
                child.wait(timeout=5)
            except psutil.NoSuchProcess:
                pass
        if thread is not None:
            assert not thread.is_alive()
