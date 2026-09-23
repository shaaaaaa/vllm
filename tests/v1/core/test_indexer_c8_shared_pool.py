# SPDX-License-Identifier: Apache-2.0
"""Shared allocator byte geometry and real sizing path, without device imports."""

import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def api(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "c8_shared_layout", ROOT / "vllm/v1/core/dsa_shared_pool.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def specs(c8=True):
    latent = NS(
        page_size_bytes=147456,
        sparse_head_dim=(512, 64),
        block_size=128,
        dtype=torch.bfloat16,
    )
    indexer = NS(
        page_size_bytes=16640 if c8 else 32768,
        sparse_head_dim=(128,),
        block_size=128,
        dtype=torch.bfloat16,
        c8_k_cache_dtype=torch.int8,
        indexer_scale_page_size_bytes=256 if c8 else 0,
    )
    return latent, indexer


@pytest.mark.parametrize("c8", [False, True])
def test_shared_key_ownership_covers_exact_latent_bytes(api, c8):
    layout = api.dsa_shared_block_layout(*specs(c8), capacity_bundles=4)
    assert layout.latent_dim == 576
    assert layout.latent_blocks_per_bundle == (1 if c8 else 2)
    assert layout.indexer_blocks_per_bundle == 9
    assert layout.scale_bytes_per_bundle == (2304 if c8 else 0)
    nope_slab = layout.slot_count * layout.latent_blocks_per_bundle * 128 * 512 * 2
    occupied = set()
    for bundle in range(1, layout.slot_count):
        ids = layout.block_ids_for_bundle(api.DSASharedBlockOwner.INDEXER, bundle)
        assert all(
            layout.bundle_id_for_block(api.DSASharedBlockOwner.INDEXER, i) == bundle
            for i in ids
        )
        lo = bundle * layout.latent_blocks_per_bundle
        hi = lo + layout.latent_blocks_per_bundle
        expected = set(range(lo * 128 * 512 * 2, hi * 128 * 512 * 2))
        expected.update(
            range(nope_slab + lo * 128 * 64 * 2, nope_slab + hi * 128 * 64 * 2)
        )
        actual = {
            i * layout.indexer_page_size_bytes + b
            for i in ids
            for b in range(layout.indexer_page_size_bytes)
        }
        assert expected == actual and not (occupied & actual)
        occupied.update(actual)


@pytest.mark.parametrize("c8", [False, True])
@pytest.mark.parametrize("layers", [(4, 4), (5, 2)])
def test_actual_sizing_charges_null_bundle_and_all_scales(api, c8, layers):
    path = ROOT / "vllm/v1/core/kv_cache_utils.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name == "get_kv_cache_config_from_groups"
    )
    namespace = dict(
        dsa_shared_block_layout=api.dsa_shared_block_layout,
        dsa_two_groups_enabled=lambda: True,
        dsa_shared_pool_enabled=lambda: True,
        cdiv=lambda a, b: (a + b - 1) // b,
        may_override_num_blocks=lambda config, n: n,
        KVCacheTensor=NS,
        KVCacheConfig=NS,
        logger=Mock(),
    )
    module = ast.parse("from __future__ import annotations")
    module.body.append(node)
    exec(compile(module, str(path), "exec"), namespace)
    latent, indexer = specs(c8)
    names = [f"model.layers.{i}.self_attn.attn" for i in range(layers[0])]
    groups = [
        NS(kv_cache_spec=latent, layer_names=names),
        NS(
            kv_cache_spec=indexer,
            layer_names=[
                name.rsplit(".", 1)[0] + ".indexer.k_cache"
                for name in names[: layers[1]]
            ],
        ),
    ]
    config = NS(
        model_config=NS(max_model_len=4096, hf_text_config=NS(index_topk=2048)),
        num_speculative_tokens=1,
        cache_config=NS(),
        scheduler_config=NS(max_num_seqs=16),
    )
    layout = api.dsa_shared_block_layout(latent, indexer, 4)
    charge = layout.allocation_bytes_per_bundle(*layers)
    budget = 5 * charge + 17
    result = namespace[node.name](config, groups, budget)
    assert result.num_blocks == 4
    assert sum(t.size for t in result.kv_cache_tensors) == 5 * charge <= budget
    for tensor in result.kv_cache_tensors:
        extra = layout.scale_bytes_per_bundle if len(tensor.shared_by) == 2 else 0
        assert tensor.size == 5 * (layout.bundle_page_size_bytes + extra)


def test_quantized_bundle_owner_reuse_keeps_one_allocator(api):
    layout = api.dsa_shared_block_layout(*specs(), capacity_bundles=2)
    allocator = api.DSASharedBundleAllocator(layout)
    latent = api.DSASharedBlockOwner.LATENT
    indexer = api.DSASharedBlockOwner.INDEXER
    first = allocator.allocate(latent, 1)
    second = allocator.allocate(indexer, 1)
    assert first != second and allocator.free_bundle_count == 0
    allocator.free(latent, first)
    assert allocator.allocate(indexer, 1) == first
