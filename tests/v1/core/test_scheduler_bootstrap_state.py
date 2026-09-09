# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dependency-light regression tests for scheduler bootstrap initialization.

Run with --confcutdir=tests/v1/core to avoid model/device test bootstrap.
The production constructors and schedule preamble execute unchanged; heavyweight
dependencies are mocked and scheduling stops before request/KV processing.
"""

import ast
import time
from collections import defaultdict, deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.cpu_test
_SCHED_DIR = Path(__file__).resolve().parents[3] / "vllm/v1/core/sched"


class _ReachedKVProcessing(Exception):
    pass


@pytest.fixture(params=[False, True], ids=["sync", "async"])
def scheduler(request):
    namespace = {
        "SchedulerInterface": object,
        "MULTIMODAL_REGISTRY": Mock(
            supports_multimodal_inputs=Mock(return_value=False)
        ),
        "_model_fingerprint": Mock(return_value="test-model"),
        "EventPublisherFactory": Mock(),
        "SchedulingPolicy": str,
        "create_request_queue": lambda policy: deque(),
        "EncoderCacheManager": Mock(),
        "KVCacheManager": Mock(),
        "envs": SimpleNamespace(VLLM_USE_V2_MODEL_RUNNER=False),
        "PauseState": SimpleNamespace(UNPAUSED=0, PAUSED_ALL=1),
        "defaultdict": defaultdict,
        "time": time,
        "logger": Mock(),
    }
    for filename, class_name in (
        ("scheduler.py", "Scheduler"),
        ("async_scheduler.py", "AsyncScheduler"),
    ):
        source_path = _SCHED_DIR / filename
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        cls = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        cls.body = [
            node
            for node in cls.body
            if isinstance(node, ast.FunctionDef)
            and node.name in ("__init__", "schedule")
        ]
        module = ast.parse("from __future__ import annotations")
        module.body.append(cls)
        exec(compile(module, str(source_path), "exec"), namespace)

    config = Mock()
    config.configure_mock(
        **{
            "scheduler_config.async_scheduling": request.param,
            "scheduler_config.max_num_seqs": 1,
            "scheduler_config.max_num_scheduled_tokens": 16,
            "scheduler_config.policy": "fcfs",
            "cache_config.num_gpu_blocks": 16,
            "model_config.is_encoder_decoder": False,
            "model_config.enable_return_routed_experts": False,
            "parallel_config.pipeline_parallel_size": 1,
            "parallel_config.prefill_context_parallel_size": 1,
            "parallel_config.decode_context_parallel_size": 1,
            "observability_config.kv_cache_metrics": False,
            "speculative_config": None,
            "kv_transfer_config": None,
            "ec_transfer_config": None,
        }
    )
    scheduler_cls = namespace["AsyncScheduler" if request.param else "Scheduler"]
    instance = scheduler_cls(
        vllm_config=config,
        kv_cache_config=SimpleNamespace(
            has_mamba_layers=False, needs_kv_cache_zeroing=False
        ),
        structured_output_manager=Mock(),
        block_size=16,
    )
    instance.kv_cache_manager.new_step_starts.side_effect = _ReachedKVProcessing
    return instance


def test_first_schedule_has_initialized_bootstrap_state(scheduler):
    # Do not seed the flag in the fixture: the production constructor must do it.
    for _ in range(2):
        with pytest.raises(_ReachedKVProcessing):
            scheduler.schedule()
        assert scheduler._bootstrap_sample_ready is False
    scheduler.kv_cache_manager.new_step_starts.assert_called()
    scheduler.schedule.__globals__["logger"].info.assert_not_called()


def test_schedule_consumes_pending_bootstrap_once(scheduler):
    assert scheduler._bootstrap_sample_ready is False
    scheduler._bootstrap_sample_ready = True
    for _ in range(2):
        with pytest.raises(_ReachedKVProcessing):
            scheduler.schedule()
        assert scheduler._bootstrap_sample_ready is False
    scheduler.schedule.__globals__["logger"].info.assert_called_once()
