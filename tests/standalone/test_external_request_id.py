# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise request identity transport without importing accelerator backends."""

import ast
import enum
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_node(path, name, namespace, parent=None):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    body = tree.body
    if parent is not None:
        body = next(node for node in body if getattr(node, "name", None) == parent).body
    node = next(node for node in body if getattr(node, "name", None) == name)
    module = ast.parse("from __future__ import annotations")
    module.body.append(node)
    exec(compile(module, str(ROOT / path), "exec"), namespace)
    return namespace[name]


@pytest.fixture
def code(monkeypatch):
    module = ModuleType("_external_request_id_test")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    namespace = vars(module)
    namespace.update(
        enum=enum,
        time=time,
        dataclass=dataclass,
        SERVING_PERF_ENABLED=False,
        StructuredOutputRequest=SimpleNamespace(from_sampling_params=lambda _: None),
        ConstantList=tuple,
        length_from_prompt_token_ids_or_embeds=lambda tokens, embeds: len(tokens),
    )
    load_node("vllm/v1/request.py", "RequestStatus", namespace)
    load_node("vllm/v1/request.py", "Request", namespace)
    load_node("vllm/v1/core/sched/output.py", "NewRequestData", namespace)
    namespace["DSABlockAllocationMode"] = SimpleNamespace(PREFILL_CHILD="child")
    load_node(
        "vllm/v1/core/sched/scheduler.py",
        "_make_new_request_data",
        namespace,
        parent="Scheduler",
    )
    return module


def core_request(internal_id, external_id):
    return SimpleNamespace(
        request_id=internal_id,
        external_req_id=external_id,
        client_index=0,
        prompt_token_ids=[10, 20, 30],
        prompt_embeds=None,
        mm_features=[],
        sampling_params=SimpleNamespace(
            max_tokens=16, extra_args=None, skip_reading_prefix_cache=None
        ),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        priority=0,
        trace_headers=None,
        resumable=False,
        reasoning_ended=None,
    )


@pytest.mark.parametrize("external_id", [None, "chatcmpl-pd-request/unchanged"])
@pytest.mark.parametrize("prefill_banks", [False, True])
def test_pd_identity_survives_scheduler_transport(code, external_id, prefill_banks):
    # P and D retain their distinct engine IDs, even for the same API request.
    internal_ids = ("request-p-12345678", "request-d-abcdef01")
    blocks = ([1, 2], [3, 4])
    bank_ids = (blocks, ([5, 6], [7, 8])) if prefill_banks else None
    scheduler = SimpleNamespace(
        kv_cache_manager=SimpleNamespace(
            coordinator=SimpleNamespace(layerwise_prefill_p_node=prefill_banks)
        )
    )
    for internal_id in internal_ids:
        request = code.Request.from_engine_core_request(
            core_request(internal_id, external_id), block_hasher=None
        )
        request.append_output_token_ids(40)
        request.num_computed_tokens = 3
        scheduled = code._make_new_request_data(
            scheduler,
            request,
            SimpleNamespace(
                get_block_ids=lambda: blocks,
                get_block_ids_by_bank=lambda: bank_ids,
            ),
            request.all_token_ids,
        )
        delivered = pickle.loads(pickle.dumps(scheduled))

        assert request.external_req_id == delivered.external_req_id == external_id
        assert request.request_id == delivered.req_id == internal_id
        assert delivered.block_ids == blocks
        assert delivered.block_ids_by_bank == bank_ids
        assert delivered.num_computed_tokens == 3
        assert delivered.prefill_token_ids == request.all_token_ids
        assert delivered.sampling_params.max_tokens == 16


def test_direct_request_has_unknown_external_id(code):
    request = code.Request(
        request_id="internal-only",
        prompt_token_ids=[1],
        sampling_params=core_request("", None).sampling_params,
        pooling_params=None,
    )
    scheduled = code.NewRequestData.from_request(request, ([1],))
    assert request.external_req_id is None
    assert scheduled.external_req_id is None
    assert scheduled.req_id == "internal-only"


def test_legacy_request_like_object_does_not_invent_external_id(code):
    request = core_request("internal-only", None)
    del request.external_req_id
    request.num_computed_tokens = 0
    scheduled = code.NewRequestData.from_request(request, ([1],))
    assert scheduled.external_req_id is None


def test_existing_positional_new_request_construction_remains_valid(code):
    scheduled = code.NewRequestData(
        "internal-only", [1], [], None, None, ([1],), 0, None
    )
    assert scheduled.external_req_id is None
    assert scheduled.req_id == "internal-only"
