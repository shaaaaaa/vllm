# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F811 -- imported pytest fixtures
"""Prove paired-bank byte ownership with the production layout/allocator."""

from types import SimpleNamespace as NS

import pytest

from tests.v1.core.test_indexer_c8_shared_pool import api as api


def layout(api, capacity=7):
    latent = NS(page_size_bytes=147456, sparse_head_dim=(512, 64), dtype=NS(itemsize=2))
    index = NS(
        page_size_bytes=33024,
        sparse_head_dim=(128,),
        dtype=NS(itemsize=2),
        shared_indexer_key_dtype=NS(itemsize=2),
        c8_k_cache_dtype=NS(itemsize=1),
        indexer_scale_page_size_bytes=256,
        indexer_scale_layer_count=16,
        indexer_paired_banks=True,
    )
    return api.dsa_shared_block_layout(latent, index, capacity)


def merge(ranges):
    out = []
    for a, b in sorted(ranges):
        if out and out[-1][1] == a:
            out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    return out


@pytest.mark.parametrize("capacity", [1, 2, 7, 16])
def test_actual_layout_has_bijective_mapping_and_no_cross_owner_overlap(api, capacity):
    plan = layout(api, capacity)
    t = plan.slot_count
    assert plan.latent_blocks_per_bundle == 2 and plan.indexer_blocks_per_bundle == 18
    mapping = api.paired_bank_indexer_block_map(t)
    assert sorted(mapping) == list(range(18 * t)) and mapping[0] == 0
    g0, g1 = api.DSASharedBlockOwner.LATENT, api.DSASharedBlockOwner.INDEXER
    all_ids = []
    for u in range(1, t):
        blocks = plan.block_ids_for_bundle(g1, u)
        all_ids.extend(blocks)
        assert len(blocks) == 18 and all(
            plan.bundle_id_for_block(g1, b) == u for b in blocks
        )
        for quantized in (True, False):
            size = 16384 if quantized else 32768
            ir = merge(
                [
                    (b * size, (b + 1) * size)
                    for b in (mapping[b] if quantized else b for b in blocks)
                ]
            )
            for v in range(1, t):
                pe = 2 * t * 131072 * (1 if quantized else 2)
                lr = [
                    (2 * v * 131072, 2 * (v + 1) * 131072),
                    (pe + 2 * v * 16384, pe + 2 * (v + 1) * 16384),
                ]
                overlaps = any(max(a, c) < min(b, d) for a, b in ir for c, d in lr)
                assert overlaps == (u == v)
    assert len(set(all_ids)) == 18 * capacity
    assert plan.allocation_bytes_per_bundle(79, 22) == 25141248
    allocator = api.DSASharedBundleAllocator(plan)
    held = allocator.allocate(g0, capacity)
    with pytest.raises(ValueError):
        allocator.allocate(g1, 1)
    allocator.free(g0, held)
    assert allocator.allocate(g1, capacity) == held


@pytest.mark.parametrize("slots", [0, -1, True, 1.5])
def test_invalid_mapping_capacity_rejected(api, slots):
    with pytest.raises(ValueError):
        api.paired_bank_indexer_block_map(slots)
