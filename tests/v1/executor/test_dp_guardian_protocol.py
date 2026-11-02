# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the actual guardian protocol, without importing device backends."""

import importlib.util
import itertools
import sys
from pathlib import Path

import pytest

SOURCE = (
    Path(__file__).resolve().parents[3] / "vllm/v1/executor/dp_guardian_protocol.py"
)
SPEC = importlib.util.spec_from_file_location("dp_guardian_protocol_tested", SOURCE)
protocol = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = protocol
SPEC.loader.exec_module(protocol)
CohortState, MemberState, Phase = (
    protocol.CohortState,
    protocol.MemberState,
    protocol.Phase,
)
MEMBERS = tuple(f"session-{i}" for i in range(4))
DIGEST = "a" * 64


def frame(member, seq, kind, **data):
    return dict(epoch="epoch", session=member, seq=seq, kind=kind, **data)


def active():
    state = CohortState("epoch", MEMBERS, 30, 5, 60)
    for member in MEMBERS:
        state.receive(member, frame(member, 0, "READY", digest=DIGEST), 0)
    for member in MEMBERS:
        state.receive(member, frame(member, 1, "COMMIT_ACK"), 1)
    assert state.phase == Phase.ACTIVE
    return state


def test_every_startup_arrival_order_and_duplicate_ack():
    for order in itertools.permutations(MEMBERS):
        state = CohortState("epoch", MEMBERS, 30, 5, 60)
        for member in order:
            message = frame(member, 0, "READY", digest=DIGEST)
            state.receive(member, message, 0)
            state.receive(member, message, 0)
        assert state.phase == Phase.COMMITTING
        for member in reversed(order):
            state.receive(member, frame(member, 1, "COMMIT_ACK"), 1)
        for member in order:
            state.receive(member, frame(member, 2, "COMMIT_ACK"), 1.1)
        assert state.phase == Phase.ACTIVE


def test_common_ineligible_fallback_is_pre_activation_only():
    state = CohortState("epoch", MEMBERS, 30, 5, 60)
    for i, member in enumerate(MEMBERS):
        state.receive(
            member, frame(member, 0, "READY", digest=DIGEST if i else None), 0
        )
    assert state.phase == Phase.DISARMING
    for member in MEMBERS:
        state.receive(member, frame(member, 1, "FALLBACK_ACK"), 1)
    assert state.phase == Phase.DISARMED
    state.receive(MEMBERS[0], frame(MEMBERS[0], 2, "READY", digest=DIGEST), 2)
    assert state.phase == Phase.DISARMED
    state = active()
    state.receive(MEMBERS[0], frame(MEMBERS[0], 2, "FALLBACK_ACK"), 2)
    assert state.phase == Phase.ABORTED


def test_contract_mismatch_is_not_silently_fallback():
    state = CohortState("epoch", MEMBERS, 30, 5, 60)
    for i, member in enumerate(MEMBERS):
        state.receive(
            member, frame(member, 0, "READY", digest=("a" if i else "b") * 64), 0
        )
    assert state.phase == Phase.ABORTED


def test_heartbeats_do_not_hide_lost_activation_ack():
    state = CohortState("epoch", MEMBERS, 3, 5, 60)
    for member in MEMBERS:
        state.receive(member, frame(member, 0, "READY", digest=DIGEST), 0)
        state.receive(member, frame(member, 1, "HEARTBEAT"), 2.9)
    state.tick(3)
    assert state.phase == Phase.ABORTED


@pytest.mark.parametrize("failure", ["epoch", "session", "seq"])
def test_stale_messages_cannot_extend_member_lease(failure):
    state = active()
    message = frame(MEMBERS[0], 2, "HEARTBEAT")
    message[failure] = 1 if failure == "seq" else "wrong"
    state.receive(MEMBERS[0], message, 4)
    assert state.last_seen[MEMBERS[0]] == 1
    state.tick(6)
    assert state.phase == Phase.ABORTED


def test_drain_and_termination_are_separate_and_idempotent():
    state = active()
    state.receive(MEMBERS[0], frame(MEMBERS[0], 2, "STOP_REQUEST"), 2)
    for member in MEMBERS:
        state.receive(member, frame(member, 3, "QUIESCENT", wave=4), 2.5)
    assert state.phase == Phase.DRAINING
    for member in MEMBERS:
        state.receive(member, frame(member, 4, "DRAINED", wave=4), 3)
    assert state.phase == Phase.STOPPING
    for member in MEMBERS:
        state.receive(member, frame(member, 5, "DRAINED", wave=4), 3.1)
    assert state.phase == Phase.STOPPING
    for member in MEMBERS:
        state.receive(member, frame(member, 6, "TERMINATED"), 4)
    assert state.phase == Phase.STOPPED


@pytest.mark.parametrize("phase", ["quiesce", "drain", "stop"])
def test_live_heartbeats_do_not_extend_shutdown_deadline(phase):
    state = active()
    state.receive(MEMBERS[0], frame(MEMBERS[0], 2, "STOP_REQUEST"), 2)
    if phase in ("drain", "stop"):
        for member in MEMBERS:
            state.receive(member, frame(member, 3, "QUIESCENT", wave=4), 2.5)
    if phase == "stop":
        for member in MEMBERS:
            state.receive(member, frame(member, 4, "DRAINED", wave=4), 3)
    deadline = state.deadline
    for member in MEMBERS:
        state.receive(member, frame(member, 5, "HEARTBEAT"), deadline - 0.1)
    state.tick(deadline)
    assert state.phase == Phase.ABORTED


def test_member_cannot_stop_without_matching_local_drain():
    member = MemberState("epoch", 30, 5, 0, phase=Phase.DRAINING)
    member.receive(dict(epoch="epoch", seq=1, phase="stopping", wave=2), 1)
    assert member.phase == Phase.ABORTED
    member.receive(dict(epoch="epoch", seq=2, phase="active"), 2)
    assert member.phase == Phase.ABORTED


def test_member_activation_and_replayed_leader_heartbeat():
    member = MemberState("epoch", 30, 5, 0)
    assert (
        member.receive(dict(epoch="epoch", seq=1, phase="committing"), 1)
        == "COMMIT_ACK"
    )
    member.receive(dict(epoch="epoch", seq=2, phase="active"), 2)
    assert member.phase == Phase.ACTIVE
    member.receive(dict(epoch="epoch", seq=2, phase="active"), 6)
    member.tick(7)
    assert member.phase == Phase.ABORTED


def test_authenticated_bounded_json_and_tampering():
    message = frame(MEMBERS[0], 2, "ABORT")
    encoded = protocol.encode_frame(b"secret", message)
    assert protocol.decode_frame(b"secret", encoded) == message
    for invalid in (b"not json", encoded.replace(b"ABORT", b"READY"), encoded * 1000):
        with pytest.raises(ValueError):
            protocol.decode_frame(b"secret", invalid)
    with pytest.raises(ValueError):
        protocol.decode_frame(b"wrong", encoded)
    with pytest.raises(ValueError):
        protocol.encode_frame(b"secret", dict(payload="x" * protocol.MAX_FRAME_BYTES))
