# SPDX-License-Identifier: Apache-2.0
"""CPU subprocess fixture: actual guardian/client, blocked mock model workers."""

import gc
import importlib.util
import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[3] / "vllm/v1/executor"
package = types.ModuleType("guardian_probe_package")
package.__path__ = [str(SOURCE)]
sys.modules[package.__name__] = package
spec = importlib.util.spec_from_file_location(
    "guardian_probe_package.dp_guardian_client", SOURCE / "dp_guardian_client.py"
)
client = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = client
spec.loader.exec_module(client)


def main():
    gc.disable()
    options = json.loads(sys.stdin.readline())
    if sys.argv[1] == "worker":
        guard = client.WorkerGuard(options["info"], options["rank"])
        print("registered", flush=True)
        time.sleep(60)  # A blocked execution thread cannot service Python control.
        guard.close()
        return

    options["parent_pid"] = os.getpid()
    guard = client.ExecutorGuard(options)
    workers = []

    def fail():
        # Reproduce the existing executor's independent worker monitor/cleanup.
        print(
            f"mock executor failure: {guard.fatal}; phase={guard.phase}",
            file=sys.stderr,
            flush=True,
        )
        for worker in workers:
            if worker.poll() is None:
                worker.kill()
        os._exit(1)

    guard.failure_callback = fail
    print(
        json.dumps(dict(endpoint=guard.endpoint, guardian=guard.process.pid)),
        flush=True,
    )
    request = json.loads(sys.stdin.readline())
    guard.connect(tuple(request["leader"]))
    for rank in range(options["tp_size"]):
        worker = subprocess.Popen(
            [sys.executable, "-u", __file__, "worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        worker.stdin.write(
            json.dumps(dict(info=guard.worker_info(), rank=rank)).encode() + b"\n"
        )
        worker.stdin.close()
        workers.append(worker)
        assert worker.stdout.readline().strip() == b"registered"
        worker.stdout.close()
    print(
        json.dumps(
            dict(
                endpoint=guard.endpoint,
                guardian=guard.process.pid,
                workers=[p.pid for p in workers],
            )
        ),
        flush=True,
    )

    def install(enabled):
        assert guard.phase == (
            client.Phase.COMMITTING if enabled else client.Phase.DISARMING
        )
        if request.get("fail_install"):
            raise RuntimeError("injected partial activation failure")

    enabled = guard.activate(request["digest"], install)
    print(json.dumps(dict(active=enabled)), flush=True)
    if not enabled:
        guard.close()
        for worker in workers:
            worker.kill()
            worker.wait(5)
        return
    request = json.loads(sys.stdin.readline())
    if request["kind"] == "fail":
        guard.fail("injected preparation failure")
    elif request["kind"] == "timeout":
        guard.begin(time.monotonic() + 0.2)
    elif request["kind"] == "drain":
        guard.command("STOP_REQUEST")
        deadline = time.monotonic() + 10
        while guard.phase != client.Phase.QUIESCING:
            assert time.monotonic() < deadline
            time.sleep(0.02)
        guard.quiescent_wave = 1
        while guard.phase != client.Phase.DRAINING:
            assert time.monotonic() < deadline
            time.sleep(0.02)
        guard.command("DRAINED", wave=1, complete=True)
        while guard.phase != client.Phase.STOPPED:
            assert time.monotonic() < deadline
            time.sleep(0.02)
        guard.close()
        for worker in workers:
            worker.wait(5)
        return
    else:
        assert request["kind"] == "wait"
    time.sleep(30)
    raise AssertionError("Fatal epoch was not stopped")


if __name__ == "__main__":
    main()
