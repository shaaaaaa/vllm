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


@pytest.mark.parametrize("c8", [False, True, "mixed"])
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
    if c8 == "mixed":
        indexer.indexer_c8_layer_names = tuple(groups[1].layer_names[::2])
        indexer.indexer_scale_layer_count = len(indexer.indexer_c8_layer_names)
        indexer.shared_indexer_key_dtype = torch.bfloat16
        indexer.indexer_scale_page_size_bytes = 512
        indexer.page_size_bytes = 32768 + 512
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
        has_scales = len(tensor.shared_by) == 2 and (
            c8 != "mixed" or tensor.shared_by[1] in indexer.indexer_c8_layer_names
        )
        extra = layout.scale_bytes_per_bundle if has_scales else 0
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


@pytest.fixture
def profiling_case(api, monkeypatch):
    """Execute the real profiler caller and shared sizing helper together."""
    path = ROOT / "vllm/v1/core/kv_cache_utils.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name in ("get_kv_cache_config_from_groups", "may_override_num_blocks")
    ]
    module = ast.parse("from __future__ import annotations")
    module.body.extend(functions)
    ns = dict(
        dsa_shared_block_layout=api.dsa_shared_block_layout,
        dsa_two_groups_enabled=lambda: True,
        dsa_shared_pool_enabled=lambda: True,
        cdiv=lambda a, b: (a + b - 1) // b,
        KVCacheTensor=NS,
        KVCacheConfig=NS,
        logger=Mock(),
    )
    exec(compile(module, str(path), "exec"), ns)
    sizing = ns["get_kv_cache_config_from_groups"]
    path = ROOT / "vllm/v1/worker/gpu_model_runner.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    runner_cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "GPUModelRunner"
    )
    method = next(
        n
        for n in runner_cls.body
        if isinstance(n, ast.FunctionDef)
        and n.name == "_init_minimal_kv_cache_for_profiling"
    )
    caller_ns = dict(logger=Mock())
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
        caller_ns,
    )
    caller = caller_ns[method.name]

    def make(c8, capture_size=32, saved_override=None):
        latent, indexer = specs(c8)
        names = [f"model.layers.{i}.self_attn.attn" for i in range(79)]
        groups = [
            NS(kv_cache_spec=latent, layer_names=names),
            NS(
                kv_cache_spec=indexer,
                layer_names=[
                    n.rsplit(".", 1)[0] + ".indexer.k_cache" for n in names[:22]
                ],
            ),
        ]
        if c8 == "mixed":
            indexer.indexer_c8_layer_names = tuple(groups[1].layer_names[:16])
            indexer.indexer_scale_layer_count = 16
            indexer.shared_indexer_key_dtype = torch.bfloat16
            indexer.indexer_scale_page_size_bytes = 512
            indexer.page_size_bytes = 32768 + 512
        config = NS(
            model_config=NS(max_model_len=178000, hf_text_config=NS(index_topk=2048)),
            num_speculative_tokens=1,
            cache_config=NS(num_gpu_blocks_override=saved_override),
            scheduler_config=NS(max_num_seqs=16),
        )
        exports = NS(
            get_kv_cache_config_from_groups=sizing,
            get_kv_cache_groups=lambda *_: groups,
        )
        monkeypatch.setitem(sys.modules, "vllm.v1.core.kv_cache_utils", exports)
        runner = NS(
            vllm_config=config,
            cache_config=config.cache_config,
            compilation_config=NS(max_cudagraph_capture_size=capture_size),
            get_kv_cache_spec=lambda: {},
            initialize_kv_cache=Mock(),
        )
        charge = api.dsa_shared_block_layout(
            latent, indexer
        ).allocation_bytes_per_bundle(79, 22)
        return runner, groups, charge, exports

    return make, caller, sizing


@pytest.mark.parametrize("c8", [False, True, "mixed"])
@pytest.mark.parametrize("capture_size", [None, 32])
def test_graph_profiling_zero_sentinel_allocates_exact_pool(
    profiling_case, c8, capture_size
):
    make, caller, _ = profiling_case
    runner, _, charge, _ = make(c8, capture_size, saved_override=7)
    caller(runner)
    minimal = runner.initialize_kv_cache.call_args.args[0]
    blocks = capture_size or 1
    assert minimal.num_blocks == blocks
    assert sum(t.size for t in minimal.kv_cache_tensors) == (blocks + 1) * charge
    assert runner.cache_config.num_gpu_blocks_override == 7
    assert runner.cache_config.num_gpu_blocks == blocks


@pytest.mark.parametrize("stage", ["sizing", "initialization"])
@pytest.mark.parametrize("saved_override", [None, 19])
def test_graph_profiling_restores_override_on_failure(
    profiling_case, stage, saved_override
):
    make, caller, _ = profiling_case
    runner, _, _, exports = make(True, saved_override=saved_override)
    fail = Mock(side_effect=RuntimeError("injected failure"))
    if stage == "sizing":
        exports.get_kv_cache_config_from_groups = fail
    else:
        runner.initialize_kv_cache = fail
    with pytest.raises(RuntimeError, match="injected failure"):
        caller(runner)
    assert runner.cache_config.num_gpu_blocks_override is saved_override
    if stage == "sizing":
        runner.initialize_kv_cache.assert_not_called()


@pytest.mark.parametrize("c8", [True, "mixed"])
def test_real_c8_budget_still_rejects_oversized_override(profiling_case, c8):
    make, _, sizing = profiling_case
    runner, groups, charge, _ = make(c8, saved_override=32)
    for budget in (0, 33 * charge - 1):
        with pytest.raises(ValueError, match="exceeds the HBM budget"):
            sizing(runner.vllm_config, groups, available_memory=budget)
    allocated = sizing(runner.vllm_config, groups, available_memory=33 * charge)
    assert sum(t.size for t in allocated.kv_cache_tensors) == 33 * charge


@pytest.mark.parametrize("budget,override", [(1, 32), (0, None), (0, 0), (0, -1)])
def test_profiling_bypass_requires_explicit_minimal_allocation(
    profiling_case, budget, override
):
    make, _, sizing = profiling_case
    runner, groups, _, _ = make(True, saved_override=override)
    with pytest.raises(ValueError, match="zero budget sentinel"):
        sizing(runner.vllm_config, groups, available_memory=budget, for_profiling=True)
