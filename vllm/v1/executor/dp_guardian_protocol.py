# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded, CPU-only control protocol for independently scheduled DP workers.

Only startup, wall-clock health and terminal control cross this boundary. Model
steps and tensor ownership never do. ABORT is not evidence of DMA completion.
This module deliberately has no torch/vLLM imports: guardians use a fresh Python
interpreter without loading a device backend.
"""

import hashlib
import hmac
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

MAX_FRAME_BYTES = 16384


class Phase(str, Enum):
    BOOT = "boot"
    COMMITTING = "committing"
    ACTIVE = "active"
    QUIESCING = "quiescing"
    DRAINING = "draining"
    STOPPING = "stopping"
    STOPPED = "stopped"
    DISARMING = "disarming"
    DISARMED = "disarmed"
    ABORTED = "aborted"


TERMINAL = frozenset((Phase.STOPPED, Phase.DISARMED, Phase.ABORTED))
MEMBER_TRANSITIONS = frozenset(
    {
        (Phase.COMMITTING, Phase.ACTIVE),
        (Phase.ACTIVE, Phase.QUIESCING),
        (Phase.QUIESCING, Phase.DRAINING),
        (Phase.STOPPING, Phase.STOPPED),
        (Phase.DISARMING, Phase.DISARMED),
    }
)


def encode_frame(secret: bytes, message: dict[str, Any]) -> bytes:
    """Encode authenticated JSON; never deserialize Python objects."""
    body = json.dumps(message, separators=(",", ":"), sort_keys=True).encode()
    mac = hmac.new(secret, body, hashlib.sha256).hexdigest().encode()
    frame = mac + b" " + body + b"\n"
    if len(frame) > MAX_FRAME_BYTES:
        raise ValueError("DP guardian control frame is too large")
    return frame


def decode_frame(secret: bytes, frame: bytes) -> dict[str, Any]:
    """Reject malformed, unauthenticated and oversized frames before use."""
    if len(frame) > MAX_FRAME_BYTES:
        raise ValueError("DP guardian control frame is too large")
    mac, separator, body = frame.rstrip(b"\n").partition(b" ")
    expected = hmac.new(secret, body, hashlib.sha256).hexdigest().encode()
    if not separator or not hmac.compare_digest(mac, expected):
        raise ValueError("Unauthenticated DP guardian frame")
    value = json.loads(body)
    if not isinstance(value, dict):
        raise ValueError("DP guardian frame must be an object")
    return value


@dataclass
class CohortState:
    """Leader state; membership is immutable and decisions are epoch-scoped.

    The transport provides one ordered, authenticated connection per session.
    Time is always the receiver's monotonic clock. Repeated messages cannot renew
    a lease, and health traffic cannot extend activation/drain/stop deadlines.
    """

    epoch: str
    members: tuple[str, ...]
    startup_deadline: float
    lease: float
    drain_timeout: float
    phase: Phase = Phase.BOOT
    reason: str = ""
    ready: dict[str, str | None] = field(default_factory=dict)
    acknowledged: set[str] = field(default_factory=set)
    drained: dict[str, int] = field(default_factory=dict)
    quiescent: dict[str, int | None] = field(default_factory=dict)
    terminated: set[str] = field(default_factory=set)
    last_seen: dict[str, float] = field(default_factory=dict)
    sequences: dict[str, int] = field(default_factory=dict)
    deadline: float | None = None

    def abort(self, reason: str) -> bool:
        """Latch terminal failure without claiming process or memory drain."""
        if self.phase in TERMINAL:
            return False
        self.phase, self.reason = Phase.ABORTED, reason[:256]
        return True

    def receive(self, member: str, message: dict[str, Any], now: float) -> bool:
        """Apply a bound member's frame; return whether the phase changed."""
        if (
            self.phase in TERMINAL
            or member not in self.members
            or message.get("epoch") != self.epoch
            or message.get("session") != member
        ):
            return False
        seq, kind = message.get("seq"), message.get("kind")
        if type(seq) is not int or seq <= self.sequences.get(member, -1):
            return False
        if kind not in {
            "HEARTBEAT",
            "READY",
            "COMMIT_ACK",
            "FALLBACK_ACK",
            "ABORT",
            "STOP_REQUEST",
            "QUIESCENT",
            "DRAINED",
            "TERMINATED",
        }:
            return self.abort("Unknown member control message")
        old = self.phase
        self.sequences[member], self.last_seen[member] = seq, now
        if kind == "ABORT":
            self.abort(str(message.get("reason", "Member failed")))
        elif kind == "READY" and self.phase == Phase.BOOT:
            digest = message.get("digest")
            if digest is not None and (
                not isinstance(digest, str) or len(digest) != 64
            ):
                return self.abort("Invalid startup contract digest")
            if member in self.ready and self.ready[member] != digest:
                return self.abort("Member changed its startup contract")
            self.ready[member] = digest
            if len(self.ready) == len(self.members):
                if any(value is None for value in self.ready.values()):
                    self.phase = Phase.DISARMING
                    self.reason = "Independent execution is ineligible on a member"
                elif len(set(self.ready.values())) != 1:
                    return self.abort("DP execution contracts differ")
                else:
                    self.phase = Phase.COMMITTING
        elif kind in ("COMMIT_ACK", "FALLBACK_ACK"):
            if (
                kind == "COMMIT_ACK"
                and self.phase == Phase.ACTIVE
                and member in self.acknowledged
            ):
                return False
            expected = Phase.COMMITTING if kind == "COMMIT_ACK" else Phase.DISARMING
            if self.phase != expected:
                return self.abort("Unexpected startup acknowledgement")
            self.acknowledged.add(member)
            if len(self.acknowledged) == len(self.members):
                self.phase = Phase.ACTIVE if kind == "COMMIT_ACK" else Phase.DISARMED
        elif kind == "STOP_REQUEST" and self.phase == Phase.ACTIVE:
            self.phase = Phase.QUIESCING
            self.deadline = now + self.drain_timeout
        elif kind == "QUIESCENT" and self.phase == Phase.QUIESCING:
            wave = message.get("wave")
            if wave is not None and (type(wave) is not int or wave < 0):
                return self.abort("Invalid model quiescence wave")
            self.quiescent[member] = wave
            values = set(self.quiescent.values())
            if (
                len(self.quiescent) == len(self.members)
                and len(values) == 1
                and None not in values
            ):
                self.phase = Phase.DRAINING
        elif kind == "DRAINED":
            wave = message.get("wave")
            if self.phase == Phase.STOPPING and self.drained.get(member) == wave:
                return False
            if (
                self.phase != Phase.DRAINING
                or type(wave) is not int
                or wave < 0
                or self.quiescent.get(member) != wave
            ):
                return self.abort("Invalid drain acknowledgement")
            self.drained[member] = wave
            if len(set(self.drained.values())) != 1:
                return self.abort("DP drain waves differ")
            if len(self.drained) == len(self.members):
                self.phase, self.deadline = Phase.STOPPING, now + self.lease
        elif kind == "TERMINATED":
            if self.phase != Phase.STOPPING:
                return self.abort("Unexpected worker termination")
            self.terminated.add(member)
            if len(self.terminated) == len(self.members):
                self.phase = Phase.STOPPED
        return old != self.phase

    def tick(self, now: float) -> bool:
        """Enforce deadlines even when peers continue sending heartbeats."""
        if self.phase in TERMINAL:
            return False
        if self.phase in (Phase.BOOT, Phase.COMMITTING, Phase.DISARMING):
            if now >= self.startup_deadline:
                return self.abort("DP activation deadline expired")
        elif self.deadline is not None and now >= self.deadline:
            return self.abort("DP drain/termination deadline expired")
        if any(now - seen >= self.lease for seen in self.last_seen.values()):
            return self.abort("DP guardian lease expired")
        return False


@dataclass
class MemberState:
    """Local state: leader health never proves local completion or releases data."""

    epoch: str
    startup_deadline: float
    lease: float
    last_seen: float
    phase: Phase = Phase.BOOT
    sequence: int = -1
    reason: str = ""
    drained_wave: int | None = None

    def abort(self, reason: str) -> None:
        if self.phase not in TERMINAL:
            self.phase, self.reason = Phase.ABORTED, reason[:256]

    def receive(self, message: dict[str, Any], now: float) -> str | None:
        """Apply leader state, returning a required startup acknowledgement."""
        if self.phase in TERMINAL or message.get("epoch") != self.epoch:
            return None
        seq = message.get("seq")
        if type(seq) is not int or seq <= self.sequence:
            return None
        self.sequence, self.last_seen = seq, now
        try:
            phase = Phase(message["phase"])
        except (KeyError, ValueError):
            self.abort("Invalid leader phase")
            return None
        if phase == Phase.ABORTED:
            self.abort(str(message.get("reason", "Peer failed")))
        elif phase == Phase.COMMITTING and self.phase in (Phase.BOOT, Phase.COMMITTING):
            self.phase = phase
            return "COMMIT_ACK"
        elif phase == Phase.DISARMING and self.phase in (Phase.BOOT, Phase.DISARMING):
            self.phase = phase
            return "FALLBACK_ACK"
        elif (self.phase, phase) in MEMBER_TRANSITIONS:
            self.phase = phase
        elif phase == Phase.STOPPING and self.phase in (Phase.DRAINING, Phase.STOPPING):
            if self.drained_wave is None or message.get("wave") != self.drained_wave:
                self.abort("Stop without a matching completed local drain")
            else:
                self.phase = phase
        elif phase != self.phase:
            self.abort("Invalid leader transition")
        return None

    def tick(self, now: float) -> None:
        if self.phase in TERMINAL:
            return
        if (
            self.phase in (Phase.BOOT, Phase.COMMITTING, Phase.DISARMING)
            and now >= self.startup_deadline
        ):
            self.abort("DP activation deadline expired")
        elif now - self.last_seen >= self.lease:
            self.abort("DP leader lease expired")
