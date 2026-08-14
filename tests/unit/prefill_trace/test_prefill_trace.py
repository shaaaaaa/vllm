# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm import prefill_trace
from vllm.v1.engine import core as core_module


def test_trace_env_is_strict_boolean(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(prefill_trace.TRACE_ENV, raising=False)
    prefill_trace.enabled.cache_clear()
    assert prefill_trace.enabled() is False
    monkeypatch.setenv(prefill_trace.TRACE_ENV, "true")
    prefill_trace.enabled.cache_clear()
    assert prefill_trace.enabled() is True
    monkeypatch.setenv(prefill_trace.TRACE_ENV, "false")
    prefill_trace.enabled.cache_clear()
    assert prefill_trace.enabled() is False
    monkeypatch.setenv(prefill_trace.TRACE_ENV, "1")
    prefill_trace.enabled.cache_clear()
    with pytest.raises(ValueError, match=prefill_trace.TRACE_ENV):
        prefill_trace.enabled()
    prefill_trace.enabled.cache_clear()


def test_point_records_one_json_timestamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(prefill_trace.TRACE_ENV, "true")
    prefill_trace.enabled.cache_clear()


def test_points_at_batches_log_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(prefill_trace.TRACE_ENV, "true")
    prefill_trace.enabled.cache_clear()
    with patch.object(prefill_trace.logger, "info") as info:
        count = prefill_trace.points_at(
            [
                ("worker_execute_start", "cmpl-prefill-test", 10, {"rank": 0}),
                ("worker_execute_return", "cmpl-prefill-test", 20, {"rank": 0}),
            ]
        )

    assert count == 2
    payload = json.loads(info.call_args.args[2])
    assert [item["event"] for item in payload] == [
        "worker_execute_start",
        "worker_execute_return",
    ]
    prefill_trace.enabled.cache_clear()
    with (
        patch.object(prefill_trace.time, "time_ns", return_value=123),
        patch.object(prefill_trace.logger, "info") as info,
    ):
        assert prefill_trace.point("core_request_received", "cmpl-prefill-test") == 123

    payload = json.loads(info.call_args.args[2])
    assert payload["event"] == "core_request_received"
    assert payload["unix_ns"] == 123
    assert payload["request"] == "cmpl-prefill-test"
    prefill_trace.enabled.cache_clear()


def test_extract_prefill_steps_across_chunks() -> None:
    new_request = SimpleNamespace(
        req_id="cmpl-prefill-test-0-deadbeef",
        prompt_token_ids=list(range(8)),
        prefill_token_ids=None,
        num_computed_tokens=0,
    )
    first = SimpleNamespace(
        scheduled_new_reqs=[new_request],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[],
            num_computed_tokens=[],
        ),
        num_scheduled_tokens={new_request.req_id: 4},
    )
    prompt_lens: dict[str, int] = {}
    assert prefill_trace.prefill_steps(first, prompt_lens) == (
        prefill_trace.PrefillStep(new_request.req_id, 0, 4, 8),
    )

    second = SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[new_request.req_id],
            num_computed_tokens=[4],
        ),
        num_scheduled_tokens={new_request.req_id: 4},
    )
    assert prefill_trace.prefill_steps(second, prompt_lens) == (
        prefill_trace.PrefillStep(new_request.req_id, 4, 4, 8),
    )


def test_engine_core_retains_chunk_points_until_result() -> None:
    request_id = "cmpl-prefill-test-0-deadbeef"
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(
                req_id=request_id,
                prompt_token_ids=list(range(8)),
                prefill_token_ids=None,
                num_computed_tokens=0,
            )
        ],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[],
            num_computed_tokens=[],
        ),
        num_scheduled_tokens={request_id: 8},
    )
    core = object.__new__(core_module.EngineCore)
    core.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4)
    )

    with (
        patch.object(core_module, "prefill_trace_enabled", return_value=True),
        patch.object(core_module, "prefill_trace_points_at") as trace_points_at,
    ):
        core._record_prefill_schedule(
            scheduler_output,
            schedule_start_ns=90,
            schedule_done_ns=91,
        )
        core._capture_prefill_batch(
            "core_model_dispatched",
            scheduler_output,
            unix_ns=92,
        )
        core._capture_prefill_batch(
            "core_scheduler_update_done",
            scheduler_output,
            unix_ns=93,
        )
        core._flush_prefill_batch(
            scheduler_output,
            finish=True,
        )

    points = trace_points_at.call_args.args[0]
    assert [point[0] for point in points] == [
        "core_schedule_start",
        "core_chunk_scheduled",
        "core_model_dispatched",
        "core_scheduler_update_done",
    ]
    assert [point[2] for point in points] == [
        90,
        91,
        92,
        93,
    ]
    assert core._prefill_trace_batches == {}
    assert core._prefill_trace_prompt_lens == {}
