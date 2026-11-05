# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F811 -- imported pytest fixture
"""Explicit cold-compact startup sizing without relaxing allocation budgets."""

import ast
from types import SimpleNamespace as NS

import pytest

from tests.v1.core.test_indexer_c8_shared_pool import ROOT, specs
from tests.v1.core.test_indexer_c8_shared_pool import api as api
from tests.v1.core.test_prefill_mixed_c8_shared_pool import utilities


def methods(api, prefill=False):
    namespace = utilities(api, prefill)
    path = ROOT / "vllm/v1/core/kv_cache_utils.py"
    names = {
        "_dsa_compact_startup_bundles",
        "dsa_dense_request_token_capacity",
        "_estimate_max_model_len_from_groups",
        "_check_enough_kv_cache_memory",
        "_report_kv_cache_config",
    }
    module = ast.parse("from __future__ import annotations")
    module.body.extend(
        n
        for n in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(n, ast.FunctionDef) and n.name in names
    )
    namespace["format_gib"] = lambda size: str(size / 2**30)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def case(api, *, length=1000000, precision="mixed", marked=True, threshold=0):
    latent, indexer = specs(precision != "bf16")
    names = [f"model.layers.{i}.self_attn.attn" for i in range(79)]
    owners = [name.rsplit(".", 1)[0] + ".indexer.k_cache" for name in names[:22]]
    if precision == "mixed":
        indexer.shared_indexer_key_dtype = indexer.dtype
        indexer.indexer_scale_layer_count = 16
        indexer.indexer_c8_layer_names = tuple(owners[:16])
        indexer.indexer_paired_banks = True
        indexer.indexer_scale_page_size_bytes = 256
        indexer.page_size_bytes = 33024
    indexer.shared_pool_alignment_bytes = 2 * 1024**2
    latent.dsa_compact_startup_scratch_tokens = 4096 if marked else 0
    latent.dsa_compact_startup_dense_tokens = threshold
    latent.max_memory_usage_bytes = lambda config: (
        (config.model_config.max_model_len + 127) // 128 * latent.page_size_bytes
    )
    groups = [
        NS(kv_cache_spec=latent, layer_names=names),
        NS(kv_cache_spec=indexer, layer_names=owners),
    ]
    config = NS(
        model_config=NS(
            max_model_len=length,
            original_max_model_len=length,
            hf_text_config=NS(index_topk=2048),
        ),
        num_speculative_tokens=1,
        cache_config=NS(num_gpu_blocks_override=None),
        scheduler_config=NS(max_num_seqs=1),
    )
    return config, groups, api.dsa_shared_block_layout(latent, indexer)


def actual_manager(bundle_width):
    path = ROOT / "vllm/v1/core/single_type_kv_cache_manager.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "DSALatentManager"
    )
    names = {
        "_blocks_per_bundle",
        "_round_up_to_bundle",
        "_round_down_to_bundle",
        "_compact_required_indices",
    }
    module = ast.parse("from __future__ import annotations\nclass Manager: pass")
    module.body[1].body = [
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    namespace = {"cdiv": lambda a, b: (a + b - 1) // b}
    exec(compile(module, str(path), "exec"), namespace)
    manager = namespace["Manager"]()
    manager.block_pool = NS(blocks_per_bundle=bundle_width)
    manager.block_size = 128
    manager.scratch_blocks = 32
    return manager


@pytest.mark.parametrize("precision", ["bf16", "uniform", "mixed"])
def test_envelope_matches_actual_initial_planner_for_every_shorter_prompt(
    api, precision
):
    ns = methods(api)
    _, _, layout = case(api, precision=precision)
    manager = actual_manager(layout.latent_blocks_per_bundle)
    # Verify boundary spikes, short dense policy and monotonicity independently
    # against actual manager rows, not a copy of the new compact formula.
    threshold = 6000
    required = 0
    for tokens in range(1, 20001):
        if tokens <= threshold:
            latent_blocks = (tokens + 127) // 128
        else:
            latent_blocks = len(
                manager._compact_required_indices(
                    tokens - 1, tokens, tokens - 1, force_compact=True
                )
            )
        indexer_blocks = (tokens + 127) // 128
        cost = (
            latent_blocks + layout.latent_blocks_per_bundle - 1
        ) // layout.latent_blocks_per_bundle + (
            indexer_blocks + layout.indexer_blocks_per_bundle - 1
        ) // layout.indexer_blocks_per_bundle
        required = max(required, cost)
        assert (
            ns["_dsa_compact_startup_bundles"](
                tokens,
                layout,
                latent_block_size=128,
                indexer_block_size=128,
                scratch_tokens=4096,
                dense_tokens=threshold,
            )
            == required
        )


@pytest.mark.parametrize("length", [1000000, 1048576])
def test_one_million_budget_is_exact_and_allocation_override_still_guarded(api, length):
    ns = methods(api)
    config, groups, layout = case(api, length=length)
    needed = ns["_max_memory_usage_bytes_from_groups"](config, groups)
    charge = layout.allocation_bytes_per_bundle(79, 22)
    overhead = layout.allocation_overhead_bytes(79, 22)
    result = ns["get_kv_cache_config_from_groups"](config, groups, needed)
    assert sum(t.size for t in result.kv_cache_tensors) + overhead == needed
    required = ns["_dsa_compact_startup_bundles"](
        length,
        layout,
        latent_block_size=128,
        indexer_block_size=128,
        scratch_tokens=4096,
        dense_tokens=0,
    )
    assert result.num_blocks == required
    assert needed == (required + 1) * charge + overhead
    ns["_check_enough_kv_cache_memory"](needed, lambda: needed, length, lambda _: 0)
    with pytest.raises(ValueError, match="larger than the available"):
        ns["_check_enough_kv_cache_memory"](
            needed - 1, lambda: needed, length, lambda _: 0
        )
    config.cache_config.num_gpu_blocks_override = result.num_blocks + 1
    with pytest.raises(ValueError, match="exceeds the HBM budget"):
        ns["get_kv_cache_config_from_groups"](config, groups, needed)


def test_dense_default_and_prefill_behavior_remain_explicit(api):
    config, groups, _ = case(api, marked=False)
    ns = methods(api)
    assert ns["_max_memory_usage_bytes_from_groups"](config, groups) == 109473652736
    cache = NS(kv_cache_groups=groups, num_blocks=453)
    assert ns["dsa_dense_request_token_capacity"](cache) is None
    assert (
        methods(api, True)["_max_memory_usage_bytes_from_groups"](config, groups)
        == 2889469952
    )
    groups[0].kv_cache_spec.dsa_compact_startup_scratch_tokens = 4096
    with pytest.raises(ValueError, match="shared D pool"):
        methods(api, True)["_max_memory_usage_bytes_from_groups"](config, groups)


@pytest.mark.parametrize("threshold", [0, 140000, 2000000])
def test_dense_policy_floor_and_reconciled_capacity(api, threshold):
    ns = methods(api)
    config, groups, layout = case(api, threshold=threshold)
    floor = min(config.model_config.max_model_len, max(4096, threshold))
    needed = ns["_max_memory_usage_bytes_from_groups"](config, groups)
    dense_bundles = (floor + 255) // 256 + (floor + 2303) // 2304
    charge = layout.allocation_bytes_per_bundle(79, 22)
    overhead = layout.allocation_overhead_bytes(79, 22)
    assert needed >= (dense_bundles + 1) * charge + overhead
    ns["get_kv_cache_groups"] = lambda *_: groups
    ns["_project_kv_cache_groups_to_worker"] = lambda *_: groups
    ns["_check_enough_kv_cache_memory"] = lambda *args: None
    worker = {
        name: group.kv_cache_spec for group in groups for name in group.layer_names
    }
    configs = ns["get_kv_cache_configs"](
        config, [worker, worker], [needed + 10 * charge, needed]
    )
    for cache in configs:
        limit = ns["dsa_dense_request_token_capacity"](cache)
        assert (limit + 255) // 256 + (limit + 2303) // 2304 <= cache.num_blocks
        assert (limit + 1 + 255) // 256 + (limit + 1 + 2303) // 2304 > cache.num_blocks
        assert cache.num_blocks == configs[1].num_blocks
    if threshold >= config.model_config.max_model_len:
        assert needed == 109473652736


@pytest.mark.parametrize("scratch,dense", [(True, 0), (-1, 0), (4095, 0), (4096, -1)])
def test_invalid_marker_residency_rejected(api, scratch, dense):
    ns = methods(api)
    config, groups, _ = case(api)
    groups[0].kv_cache_spec.dsa_compact_startup_scratch_tokens = scratch
    groups[0].kv_cache_spec.dsa_compact_startup_dense_tokens = dense
    with pytest.raises(ValueError, match="Invalid DSA compact"):
        ns["_max_memory_usage_bytes_from_groups"](config, groups)


def test_compact_report_distinguishes_dense_capacity_and_initial_budget(api):
    ns = methods(api)
    config, groups, _ = case(api)
    needed = ns["_max_memory_usage_bytes_from_groups"](config, groups)
    cache = ns["get_kv_cache_config_from_groups"](config, groups, needed)
    ns["logger"].reset_mock()
    ns["_report_kv_cache_config"](config, cache)
    args = ns["logger"].info_once.call_args.args
    assert "dense full-resident request capacity" in args[0]
    assert "initial admission budget" in args[0]
    assert "generated-tail growth" in args[0]
    assert args[3] == "1,000,000"
    # Preserve the legacy metric's behavior; only the explicit startup report
    # changes when this deployment has opted into the compact contract.
    groups[0].kv_cache_spec.dsa_compact_startup_scratch_tokens = 0
    assert ns["get_max_concurrency_for_kv_cache_config"](config, cache) == (
        cache.num_blocks / (3907 + 435)
    )


def test_compact_auto_fit_keeps_budget_boundary_monotonic(api):
    ns = methods(api)
    config, groups, _ = case(api)
    needed = ns["_max_memory_usage_bytes_from_groups"](config, groups)
    fitted = ns["_estimate_max_model_len_from_groups"](config, groups, needed - 1)
    assert 0 < fitted < 1000000
    assert config.model_config.max_model_len == 1000000
    config.model_config.max_model_len = fitted
    assert ns["_max_memory_usage_bytes_from_groups"](config, groups) < needed
    config.model_config.max_model_len += 1
    assert ns["_max_memory_usage_bytes_from_groups"](config, groups) == needed
