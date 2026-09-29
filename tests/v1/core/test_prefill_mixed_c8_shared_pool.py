# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F811 -- imported fixture
"""Actual allocator/sizing paths for rotating BF16 and mixed-C8 prefill."""

import ast
from functools import partial
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from tests.v1.core.test_indexer_c8_shared_pool import ROOT, specs
from tests.v1.core.test_indexer_c8_shared_pool import api as api


def utilities(api, prefill):
    path = ROOT / "vllm/v1/core/kv_cache_utils.py"
    names = {
        "dsa_required_bundles",
        "get_kv_cache_config_from_groups",
        "get_max_concurrency_for_kv_cache_config",
        "_max_memory_usage_bytes_from_groups",
        "get_kv_cache_configs",
        "may_override_num_blocks",
    }
    body = [
        n
        for n in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    ns = dict(
        dsa_shared_block_layout=api.dsa_shared_block_layout,
        dsa_two_groups_enabled=lambda: True,
        dsa_shared_pool_enabled=lambda: True,
        layerwise_prefill_p_node_enabled=lambda: prefill,
        dsa_kv_residency_mode=lambda: "prefill_child" if prefill else "legacy",
        LAYERWISE_PREFILL_BANK_COUNT=2,
        cdiv=lambda a, b: (a + b - 1) // b,
        KVCacheTensor=NS,
        KVCacheConfig=NS,
        logger=Mock(),
        partial=partial,
        UniformTypeKVCacheSpecs=type("Unused", (), {}),
        _report_kv_cache_config=Mock(),
        _check_enough_kv_cache_memory=Mock(),
        _estimate_max_model_len_from_groups=Mock(),
    )
    module = ast.parse("from __future__ import annotations")
    module.body.extend(body)
    exec(compile(module, str(path), "exec"), ns)
    return ns


def setup_case(api, c8, prefill=True):
    latent, index = specs(c8)
    names = [f"model.layers.{i}.self_attn.attn" for i in range(4)]
    owners = [n.rsplit(".", 1)[0] + ".indexer.k_cache" for n in names[:2]]
    if c8:
        index.indexer_c8_layer_names = (owners[0],)
        index.indexer_scale_layer_count = 1
        index.shared_indexer_key_dtype = index.dtype
        index.indexer_paired_banks = True
        index.page_size_bytes = 33024
        index.shared_pool_alignment_bytes = 512
    groups = [
        NS(kv_cache_spec=latent, layer_names=names),
        NS(kv_cache_spec=index, layer_names=owners),
    ]
    config = NS(
        model_config=NS(
            max_model_len=128,
            original_max_model_len=128,
            hf_text_config=NS(index_topk=2048),
        ),
        num_speculative_tokens=1,
        cache_config=NS(num_gpu_blocks_override=None),
        scheduler_config=NS(max_num_seqs=1),
    )
    layout = api.dsa_shared_block_layout(latent, index, 3, prefill_child=prefill)
    return config, groups, layout


@pytest.mark.parametrize("c8", [False, True])
def test_prefill_geometry_budget_and_cross_worker_reconciliation(api, c8):
    ns = utilities(api, True)
    config, groups, layout = setup_case(api, c8)
    assert layout.bundle_page_size_bytes == 589824
    assert layout.latent_blocks_per_bundle == 4
    assert layout.indexer_blocks_per_bundle == 18
    assert layout.scale_bytes_per_bundle == (9216 if c8 else 0)
    assert not layout.paired_indexer_banks
    payload = layout.bundle_page_size_bytes + layout.scale_bytes_per_bundle
    overhead = layout.backing_alignment_bytes
    budgets = [(4 * c + 1) * payload + overhead for c in (3, 1)]
    ns["get_kv_cache_groups"] = lambda *_: groups
    ns["_project_kv_cache_groups_to_worker"] = lambda *_: groups
    worker = {n: g.kv_cache_spec for g in groups for n in g.layer_names}
    results = ns["get_kv_cache_configs"](config, [worker, worker], budgets)
    for result in results:
        assert result.num_blocks == 1 and result.dsa_paired_bank_slots == 0
        assert len(result.kv_cache_tensors) == 1
        assert result.kv_cache_tensors[0].size == 5 * payload
        assert result.kv_cache_tensors[0].shared_by == [
            n for g in groups for n in g.layer_names
        ]
        assert ns["get_max_concurrency_for_kv_cache_config"](config, result) == 1
    assert (
        ns["_max_memory_usage_bytes_from_groups"](config, groups)
        == 5 * payload + overhead
    )
    config.model_config.max_model_len = 513
    assert (
        ns["_max_memory_usage_bytes_from_groups"](config, groups)
        == 9 * payload + overhead
    )
    config.model_config.max_model_len = 128
    config.cache_config.num_gpu_blocks_override = 1
    with pytest.raises(ValueError, match="exceeds the HBM budget"):
        ns["get_kv_cache_config_from_groups"](config, groups, budgets[1] - 1)
    profile = ns["get_kv_cache_config_from_groups"](
        config, groups, 0, for_profiling=True
    )
    assert profile.kv_cache_tensors[0].size == 5 * payload


def test_decode_paired_capacity_reconciles_without_prefill_geometry(api):
    ns = utilities(api, False)
    config, groups, layout = setup_case(api, True, False)
    payload = layout.allocation_bytes_per_bundle(4, 2)
    overhead = layout.allocation_overhead_bytes(4, 2)
    ns["get_kv_cache_groups"] = lambda *_: groups
    ns["_project_kv_cache_groups_to_worker"] = lambda *_: groups
    worker = {n: g.kv_cache_spec for g in groups for n in g.layer_names}
    results = ns["get_kv_cache_configs"](
        config, [worker, worker], [4 * payload + overhead, 2 * payload + overhead]
    )
    for result in results:
        assert result.num_blocks == 1 and result.dsa_paired_bank_slots == 2
        assert sum(t.size for t in result.kv_cache_tensors) == 2 * payload


def test_real_child_pool_two_bank_address_proof_and_rollback(api):
    _, _, layout = setup_case(api, True)
    parent = api.DSASharedBundleAllocator(layout)
    pool = api.PrefillLayerBundlePool(parent, 4)
    assert pool.layout.scale_bytes_per_bundle == layout.scale_bytes_per_bundle
    owner = api.DSASharedBlockOwner.INDEXER
    banks = pool.allocate_banks(owner, 2)
    # Parent-packed allocation deliberately yields discontinuous IDs.
    assert banks == ((1, 4), (7, 10))
    raw = bytearray(pool.layout.slot_count * layout.bundle_page_size_bytes)
    scales = bytearray(pool.layout.slot_count * layout.scale_bytes_per_bundle)
    addresses = []
    for value, bank in enumerate(banks, 1):
        blocks = [
            b for child in bank for b in pool.layout.block_ids_for_bundle(owner, child)
        ]
        addresses.append(set(blocks))
        for block in blocks:
            physical = 2 * block
            start = physical * 16384
            raw[start : start + 16384] = bytes([value]) * 16384
            scales[physical * 256 : (physical + 1) * 256] = bytes([value]) * 256
            assert raw[block * 32768 : block * 32768 + 16384] == bytes([value]) * 16384
            assert raw[block * 32768 + 16384 : (block + 1) * 32768] == bytes(16384)
            assert scales[(physical + 1) * 256 : (physical + 2) * 256] == bytes(256)
    assert addresses[0].isdisjoint(addresses[1])
    assert raw[:32768] == bytes(32768) and scales[:512] == bytes(512)
    held = pool.owner_bundle_counts()
    with pytest.raises(ValueError):
        pool.allocate_banks(owner, pool.free_bundle_count + 1)
    assert pool.owner_bundle_counts() == held
    for bank in banks:
        pool.free(owner, bank)
    assert parent.free_bundle_count == layout.capacity_bundles


def test_layout_preserves_legacy_positional_dimensions(api):
    layout = api.DSASharedBlockLayout(147456, 32768, 3, 512, 64, 128)
    assert layout.bundle_multiplier == 1 and layout.latent_blocks_per_bundle == 2
    prefill = api.DSASharedBlockLayout(
        147456, 32768, 3, 512, 64, 128, bundle_multiplier=2
    )
    assert prefill.latent_blocks_per_bundle == 4
