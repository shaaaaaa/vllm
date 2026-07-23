# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import base64
import hashlib

from vllm.sampling_params import SamplingParams
from vllm.v1.request import (
    Request,
    RequestStatus,
    compute_prompt_token_fingerprint,
)


def test_request_status_fmt_str():
    """Test that the string representation of RequestStatus is correct."""
    assert f"{RequestStatus.WAITING}" == "WAITING"
    assert f"{RequestStatus.WAITING_FOR_FSM}" == "WAITING_FOR_FSM"
    assert f"{RequestStatus.WAITING_FOR_REMOTE_KVS}" == "WAITING_FOR_REMOTE_KVS"
    assert f"{RequestStatus.WAITING_FOR_STREAMING_REQ}" == "WAITING_FOR_STREAMING_REQ"
    assert f"{RequestStatus.RUNNING}" == "RUNNING"
    assert f"{RequestStatus.PREEMPTED}" == "PREEMPTED"
    assert f"{RequestStatus.FINISHED_STOPPED}" == "FINISHED_STOPPED"
    assert f"{RequestStatus.FINISHED_LENGTH_CAPPED}" == "FINISHED_LENGTH_CAPPED"
    assert f"{RequestStatus.FINISHED_ABORTED}" == "FINISHED_ABORTED"
    assert f"{RequestStatus.FINISHED_IGNORED}" == "FINISHED_IGNORED"


def test_normal_request_does_not_allocate_final_hidden_state() -> None:
    request = Request(
        request_id="normal",
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=2),
        pooling_params=None,
    )

    lazy_fields = {
        "capture_final_hidden",
        "bootstrap_final_hidden",
        "bootstrap_sample_pending",
        "dsa_compact_allocated",
        "captured_final_hidden",
        "final_hidden_prompt_fingerprint",
    }
    assert lazy_fields.isdisjoint(request.__dict__)
    assert not request.capture_final_hidden
    assert request.bootstrap_final_hidden is None


def test_final_hidden_artifact_must_match_prompt() -> None:
    prompt_token_ids = [1, 2, 3, 4]
    raw_hidden = b"\0" * 4
    artifact = {
        "version": 1,
        "dtype": "bfloat16",
        "shape": [2],
        "encoding": "base64",
        "data": base64.b64encode(raw_hidden).decode("ascii"),
        "data_sha256": hashlib.sha256(raw_hidden).hexdigest(),
        "prompt_length": len(prompt_token_ids),
        "prompt_sha256": compute_prompt_token_fingerprint(prompt_token_ids),
    }
    sampling_params = SamplingParams(
        max_tokens=2,
        extra_args={
            "kv_transfer_params": {"bootstrap_final_hidden": artifact}
        },
    )

    request = Request(
        request_id="matched",
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
        pooling_params=None,
    )

    assert request.bootstrap_sample_pending
    assert request.bootstrap_final_hidden == artifact

    mismatch = Request(
        request_id="mismatch",
        prompt_token_ids=prompt_token_ids + [5],
        sampling_params=sampling_params,
        pooling_params=None,
    )

    assert not mismatch.bootstrap_sample_pending
    assert mismatch.bootstrap_final_hidden is None
    assert "final_hidden_prompt_fingerprint" not in mismatch.__dict__
    assert "bootstrap_final_hidden" not in mismatch.__dict__


def test_invalid_final_hidden_artifact_falls_back_to_prefill() -> None:
    prompt_token_ids = [1, 2, 3, 4]
    sampling_params = SamplingParams(
        max_tokens=2,
        extra_args={
            "kv_transfer_params": {
                "bootstrap_final_hidden": {
                    "version": 1,
                    "dtype": "bfloat16",
                    "shape": [2],
                    "encoding": "base64",
                    "data": "AAAAAA==",
                    "data_sha256": "0" * 64,
                    "prompt_length": len(prompt_token_ids),
                    "prompt_sha256": compute_prompt_token_fingerprint(
                        prompt_token_ids
                    ),
                }
            }
        },
    )

    request = Request(
        request_id="corrupt",
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
        pooling_params=None,
    )

    assert not request.bootstrap_sample_pending
    assert request.bootstrap_final_hidden is None


def test_final_hidden_handoff_rejects_prompt_logprobs() -> None:
    prompt_token_ids = [1, 2, 3, 4]
    raw_hidden = b"\0" * 4
    artifact = {
        "version": 1,
        "dtype": "bfloat16",
        "shape": [2],
        "encoding": "base64",
        "data": base64.b64encode(raw_hidden).decode("ascii"),
        "data_sha256": hashlib.sha256(raw_hidden).hexdigest(),
        "prompt_length": len(prompt_token_ids),
        "prompt_sha256": compute_prompt_token_fingerprint(prompt_token_ids),
    }
    sampling_params = SamplingParams(
        max_tokens=2,
        prompt_logprobs=1,
        extra_args={
            "kv_transfer_params": {"bootstrap_final_hidden": artifact}
        },
    )

    request = Request(
        request_id="prompt-logprobs",
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
        pooling_params=None,
    )

    assert not request.bootstrap_sample_pending
    assert request.bootstrap_final_hidden is None
