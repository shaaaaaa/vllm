# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independent DP process supervision; executable without importing vLLM/torch.

The parent and workers authenticate locally with a different key from the peer
guardians. No network message supplies a process to signal: only registered local
children of the executor are held by stable OS handles. Termination never releases
cache leases or asserts completion of a native transfer.
"""

import json
import os
import select
import selectors
import signal
import socket
import sys
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

if __package__:
    from .dp_guardian_protocol import (
        MAX_FRAME_BYTES,
        TERMINAL,
        CohortState,
        MemberState,
        Phase,
        decode_frame,
        encode_frame,
    )
else:
    from dp_guardian_protocol import (  # type: ignore[no-redef]
        MAX_FRAME_BYTES,
        TERMINAL,
        CohortState,
        MemberState,
        Phase,
        decode_frame,
        encode_frame,
    )


class ProcessHandle:
    """Stable process identity; Windows support allows real CPU process tests."""

    def __init__(self, pid: int, parent: int | None = None):
        if pid <= 0 or pid == os.getpid():
            raise ValueError("Invalid supervised process")
        self.pid = pid
        if sys.platform == "win32":
            import ctypes

            import psutil

            self.api = ctypes.WinDLL("kernel32", use_last_error=True)
            self.api.OpenProcess.restype = ctypes.c_void_p
            self.handle = self.api.OpenProcess(0x100001, False, pid)
            if not self.handle:
                raise ProcessLookupError(pid)
            self.handle = ctypes.c_void_p(self.handle)
            try:
                actual_parent = psutil.Process(pid).ppid()
            except BaseException:
                self.close()
                raise
        else:
            self.handle = os.pidfd_open(pid)
            # pidfd pins identity before checking parentage. An already exited
            # handle is rejected even if this numeric PID has since been reused.
            try:
                stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
                actual_parent = int(stat[1])
            except BaseException:
                self.close()
                raise
        if (parent is not None and actual_parent != parent) or not self.alive():
            self.close()
            raise ProcessLookupError("Supervised process identity changed")

    def alive(self) -> bool:
        if sys.platform == "win32":
            result = self.api.WaitForSingleObject(self.handle, 0)
            if result not in (0, 0x102):
                raise OSError("Cannot determine supervised process completion")
            return result == 0x102
        return not select.select([self.handle], [], [], 0)[0]

    def kill(self) -> None:
        if not self.alive():
            return
        if sys.platform == "win32":
            if not self.api.TerminateProcess(self.handle, 1):
                raise OSError("Cannot terminate supervised worker")
        else:
            with suppress(ProcessLookupError):
                signal.pidfd_send_signal(self.handle, signal.SIGKILL)

    def close(self) -> None:
        if sys.platform == "win32":
            self.api.CloseHandle(self.handle)
        else:
            os.close(self.handle)


class Link:
    """One bounded, ordered JSON stream; all I/O belongs to the guardian loop."""

    def __init__(
        self, sock: socket.socket, key: bytes, role: str = "", session: str = ""
    ):
        self.sock, self.key, self.role, self.session = sock, key, role, session
        self.input = bytearray()
        self.output = bytearray()
        self.last_sequence = -1
        self.created = time.monotonic()
        self.sock.setblocking(False)

    def send(self, message: dict[str, Any]) -> None:
        data = encode_frame(self.key, message)
        if len(self.output) + len(data) > MAX_FRAME_BYTES * 2:
            raise BufferError("Guardian control peer is not consuming messages")
        self.output.extend(data)

    def flush(self) -> None:
        if self.output:
            try:
                sent = self.sock.send(self.output)
                del self.output[:sent]
            except BlockingIOError:
                pass

    def read(self) -> list[bytes]:
        try:
            data = self.sock.recv(MAX_FRAME_BYTES)
        except BlockingIOError:
            return []
        if not data:
            raise EOFError
        self.input.extend(data)
        frames = []
        while b"\n" in self.input:
            frame, _, rest = self.input.partition(b"\n")
            self.input[:] = rest
            frames.append(frame)
        if len(self.input) > MAX_FRAME_BYTES:
            raise ValueError("Oversized guardian control frame")
        return frames


class Guardian:
    """Wall-clock supervision, independent of worker GIL/native execution."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.epoch, self.session = config["epoch"], config["session"]
        self.key, self.local_key = (
            bytes.fromhex(config["key"]),
            bytes.fromhex(config["local_key"]),
        )
        self.interval, self.lease = config["interval"], config["lease"]
        now = time.monotonic()
        self.member = MemberState(
            self.epoch, now + config["startup_timeout"], self.lease, now
        )
        self.cohort = (
            CohortState(
                self.epoch,
                tuple(config["members"]),
                self.member.startup_deadline,
                self.lease,
                config["drain_timeout"],
            )
            if config["rank"] == 0
            else None
        )
        self.parent_handle = ProcessHandle(config["parent_pid"])
        self.workers: dict[int, ProcessHandle] = {}
        self.links: dict[socket.socket, Link] = {}
        self.parent: Link | None = None
        self.leader: Link | None = None
        self.peers: dict[str, Link] = {}
        self.seq, self.parent_seen = 0, now
        self.terminal_started: float | None = None
        self.abort_recorded = False
        self.parent_acknowledged = False
        self.connected = False
        self.selector = selectors.DefaultSelector()
        family, kind, protocol, _, address = socket.getaddrinfo(
            config["host"], 0, type=socket.SOCK_STREAM
        )[0]
        self.listener = socket.socket(family, kind, protocol)
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind(address)
        self.listener.listen(len(config["members"]) + config["tp_size"] + 2)
        self.listener.setblocking(False)
        self.selector.register(self.listener, selectors.EVENT_READ)

    def message(self, kind: str, **values: Any) -> dict[str, Any]:
        self.seq += 1
        return dict(
            epoch=self.epoch, session=self.session, seq=self.seq, kind=kind, **values
        )

    def add_link(self, link: Link) -> None:
        self.links[link.sock] = link
        self.selector.register(link.sock, selectors.EVENT_READ, link)

    def abort(self, reason: str) -> None:
        self.member.abort(reason)
        if self.cohort:
            self.cohort.abort(reason)

    def send(self, link: Link, message: dict[str, Any]) -> None:
        if link.sock not in self.links:
            return
        try:
            link.send(message)
        except (OSError, BufferError, ValueError):
            # Closing an unresponsive peer propagates failure without putting
            # notification ahead of local worker termination.
            self.drop(link)

    def to_leader(self, kind: str, **values: Any) -> None:
        message = self.message(kind, **values)
        if self.cohort:
            self.cohort.receive(self.session, message, time.monotonic())
        elif self.leader:
            self.send(self.leader, message)

    def leader_state(self) -> dict[str, Any]:
        assert self.cohort is not None
        self.connected = (
            len(self.peers) + 1 == len(self.cohort.members)
            and all(peer.sock in self.links for peer in self.peers.values())
            and self.parent is not None
            and self.parent.sock in self.links
        )
        return self.message(
            "STATE",
            phase=self.cohort.phase.value,
            reason=self.cohort.reason,
            wave=next(iter(self.cohort.quiescent.values()), None),
            connected=self.connected,
        )

    def local_message(self, link: Link, message: dict[str, Any], now: float) -> None:
        kind = message.get("kind")
        if link.role == "worker":
            if kind == "ABORT":
                self.abort("Worker failed: " + str(message.get("reason", "unknown")))
            else:
                raise ValueError("Worker sent non-terminal runtime control")
            return
        self.parent_seen = now
        if kind == "CLOSED" and self.member.phase in (Phase.STOPPED, Phase.DISARMED):
            self.parent_acknowledged = True
        elif kind == "CONNECT":
            if self.cohort is None and self.leader is None:
                sock = socket.create_connection(
                    tuple(message["leader"]), timeout=self.lease
                )
                self.leader = Link(sock, self.key, "leader")
                self.add_link(self.leader)
                self.send(self.leader, self.message("HELLO", role="peer"))
        elif kind == "HEARTBEAT":
            # Only the executor sees RPC retirement between health snapshots.
            # It reports an expired pending RPC as ABORT; its own loss is
            # independently bounded by the health lease below.
            pass
        elif kind == "READY":
            if len(self.workers) != self.config["tp_size"]:
                raise ValueError("Not all TP workers registered")
            self.to_leader("READY", digest=message.get("digest"))
        elif kind in ("COMMIT_ACK", "FALLBACK_ACK"):
            expected = Phase.COMMITTING if kind == "COMMIT_ACK" else Phase.DISARMING
            if self.member.phase != expected:
                raise ValueError("Executor acknowledged an unexpected startup decision")
            self.to_leader(kind)
        elif kind == "QUIESCENT":
            self.to_leader("QUIESCENT", wave=message.get("wave"))
        elif kind == "DRAINED":
            if (
                self.member.phase != Phase.DRAINING
                or message.get("complete") is not True
            ):
                raise ValueError("Unproven local drain")
            wave = message["wave"]
            if type(wave) is not int or wave < 0:
                raise ValueError("Invalid drain wave")
            self.member.drained_wave = wave
            self.to_leader("DRAINED", wave=wave)
        elif kind in ("STOP_REQUEST", "ABORT"):
            if kind == "ABORT":
                self.abort(str(message.get("reason", "Executor failed")))
            else:
                self.to_leader(kind)
        else:
            raise ValueError("Unexpected executor control")

    def receive(self, link: Link, raw: bytes, now: float) -> None:
        if not link.role:
            try:
                message = decode_frame(self.local_key, raw)
                link.key = self.local_key
            except ValueError:
                message = decode_frame(self.key, raw)
            if message.get("epoch") != self.epoch or message.get("kind") != "HELLO":
                raise ValueError("Invalid guardian session hello")
            role = message.get("role")
            session = message.get("session")
            if role == "peer" and link.key == self.key and self.cohort:
                if (
                    session not in self.cohort.members
                    or session == self.session
                    or session in self.peers
                ):
                    raise ValueError("Unknown or duplicate peer session")
                self.peers[session] = link
            elif (
                role == "parent" and link.key == self.local_key and self.parent is None
            ):
                self.parent = link
            elif role == "worker" and link.key == self.local_key:
                if self.member.phase != Phase.BOOT:
                    raise ValueError("Worker registration is closed for this epoch")
                rank = message.get("rank")
                if (
                    type(rank) is not int
                    or not 0 <= rank < self.config["tp_size"]
                    or rank in self.workers
                ):
                    raise ValueError("Unknown or duplicate TP worker")
                self.workers[rank] = ProcessHandle(
                    int(message["pid"]), self.config["parent_pid"]
                )
                self.send(link, self.message("REGISTERED"))
            else:
                raise ValueError("Invalid guardian connection role")
            link.role, link.session = role, session
            return
        message = decode_frame(link.key, raw)
        if message.get("epoch") != self.epoch:
            return
        seq = message.get("seq")
        if type(seq) is not int or seq <= link.last_sequence:
            return
        link.last_sequence = seq
        if link.role in ("parent", "worker"):
            self.local_message(link, message, now)
        elif link.role == "leader":
            self.member.receive(message, now)
            self.connected = message.get("connected") is True
        elif self.cohort:
            self.cohort.receive(link.session, message, now)

    def drop(self, link: Link) -> None:
        self.selector.unregister(link.sock)
        self.links.pop(link.sock, None)
        link.sock.close()
        expected_detach = link.role == "worker" and self.member.phase == Phase.DISARMING
        if (
            link.role
            and not expected_detach
            and self.member.phase not in (Phase.STOPPING, Phase.STOPPED, Phase.DISARMED)
        ):
            self.abort(f"{link.role} control connection lost")

    def tick(self, now: float) -> None:
        if self.member.phase not in TERMINAL:
            if not self.parent_handle.alive() or now - self.parent_seen >= self.lease:
                self.abort("Executor lost")
            elif self.member.phase != Phase.STOPPING and any(
                not p.alive() for p in self.workers.values()
            ):
                self.abort("TP worker lost")
        if self.cohort:
            self.cohort.tick(now)
            message = self.leader_state()
            self.member.receive(message, now)
            for peer in self.peers.values():
                if peer.sock in self.links:
                    self.send(peer, message)
        else:
            self.member.tick(now)
        if self.member.phase == Phase.ABORTED:
            self.to_leader("ABORT", reason=self.member.reason)
        else:
            self.to_leader("HEARTBEAT")
        if self.parent and self.parent.sock in self.links:
            self.send(
                self.parent,
                self.message(
                    "STATE",
                    phase=self.member.phase.value,
                    reason=self.member.reason,
                    connected=self.connected,
                ),
            )
        if self.member.phase in (Phase.ABORTED, Phase.STOPPING):
            if self.terminal_started is None:
                self.terminal_started = now
            for process in self.workers.values():
                try:
                    process.kill()
                except OSError:
                    self.abort("Cannot terminate TP worker; host fencing required")
            if self.member.phase == Phase.STOPPING and all(
                not p.alive() for p in self.workers.values()
            ):
                self.to_leader("TERMINATED")
            if (
                self.member.phase == Phase.ABORTED
                and now - self.terminal_started >= self.lease
            ):
                self.parent_handle.kill()
        if self.member.phase == Phase.ABORTED and not self.abort_recorded:
            # Stop/notify before best-effort diagnostics. The original reservation
            # already blocks restart even if this filesystem operation fails.
            for link in tuple(self.links.values()):
                try:
                    link.flush()
                except OSError:
                    self.drop(link)
            try:
                Path(self.config["tombstone"]).write_text(
                    json.dumps(
                        dict(
                            epoch=self.epoch,
                            session=self.session,
                            workers=[p.pid for p in self.workers.values()],
                            reason=self.member.reason,
                            completion="unknown",
                        )
                    ),
                    encoding="utf-8",
                )
            except OSError:
                print(
                    "DP guardian: cannot update failed-epoch marker",
                    file=sys.stderr,
                    flush=True,
                )
            self.abort_recorded = True

    def run(self) -> None:
        next_tick = time.monotonic() + self.interval
        try:
            while True:
                now = time.monotonic()
                for key, _ in self.selector.select(
                    max(0.0, min(self.interval, next_tick - now))
                ):
                    if key.fileobj is self.listener:
                        sock, _ = self.listener.accept()
                        if (
                            len(self.links)
                            >= len(self.config["members"]) + self.config["tp_size"] + 4
                        ):
                            sock.close()
                        else:
                            self.add_link(Link(sock, self.key))
                        continue
                    link = key.data
                    try:
                        for raw in link.read():
                            self.receive(link, raw, time.monotonic())
                    except (OSError, EOFError, ValueError, KeyError, TypeError):
                        self.drop(link)
                now = time.monotonic()
                if (
                    self.member.phase in (Phase.STOPPED, Phase.DISARMED)
                    and self.terminal_started is None
                ):
                    # Publish received terminal state before closing.
                    next_tick = now
                if now >= next_tick:
                    try:
                        self.tick(now)
                    except (OSError, ValueError, BufferError):
                        self.abort("Guardian control or supervision failure")
                    next_tick = now + self.interval
                for link in tuple(self.links.values()):
                    try:
                        if not link.role and now - link.created >= self.lease:
                            self.drop(link)
                            continue
                        link.flush()
                    except OSError:
                        self.drop(link)
                if self.member.phase in (Phase.STOPPED, Phase.DISARMED):
                    self.terminal_started = self.terminal_started or now
                    if (
                        self.parent_acknowledged
                        or now - self.terminal_started >= self.lease
                    ):
                        break
                if (
                    self.member.phase == Phase.ABORTED
                    and self.terminal_started is not None
                    and now - self.terminal_started >= self.lease * 2
                ):
                    break
        finally:
            for link in self.links.values():
                link.sock.close()
            self.listener.close()
            self.selector.close()
            for process in self.workers.values():
                process.close()
            self.parent_handle.close()


def main() -> None:
    config = json.loads(sys.stdin.buffer.readline(MAX_FRAME_BYTES))
    guardian = Guardian(config)
    print(json.dumps(dict(port=guardian.listener.getsockname()[1])), flush=True)
    guardian.run()


if __name__ == "__main__":
    main()
