# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm.v1.core.dsa_shared_pool import (
    DSASharedBlockLayout,
    DSASharedBlockOwner,
    DSASharedBundleAllocator,
    dsa_block_pool_index,
    dsa_scratch_blocks_for_topk,
)


def _layout(capacity_bundles: int = 8) -> DSASharedBlockLayout:
    return DSASharedBlockLayout(
        latent_page_size_bytes=128 * 576 * 2,
        indexer_page_size_bytes=128 * 128 * 2,
        capacity_bundles=capacity_bundles,
    )


def test_layout_uses_128_dim_base_pages_for_512_64_128_dims():
    layout = _layout(capacity_bundles=4)

    assert layout.bundle_page_size_bytes == 128 * 1152 * 2
    assert layout.latent_blocks_per_bundle == 2
    assert layout.indexer_blocks_per_bundle == 9
    assert layout.nope_pages_per_bundle == 8
    assert layout.pe_pages_per_bundle == 1


def test_layout_maps_latent_and_indexer_blocks_to_same_bundle():
    layout = _layout(capacity_bundles=4)

    assert layout.block_ids_for_bundle(DSASharedBlockOwner.LATENT, 1) == (2, 3)
    assert layout.block_ids_for_bundle(DSASharedBlockOwner.LATENT, 2) == (4, 5)

    assert layout.block_ids_for_bundle(DSASharedBlockOwner.INDEXER, 1) == (
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        41,
    )
    assert layout.block_ids_for_bundle(DSASharedBlockOwner.INDEXER, 2) == (
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        42,
    )


def test_bundle_id_reverse_mapping_keeps_nope_and_pe_together():
    layout = _layout(capacity_bundles=4)

    for block_id in (8, 9, 10, 11, 12, 13, 14, 15, 41):
        assert layout.bundle_id_for_block(
            DSASharedBlockOwner.INDEXER, block_id
        ) == 1

    for block_id in (16, 17, 18, 19, 20, 21, 22, 23, 42):
        assert layout.bundle_id_for_block(
            DSASharedBlockOwner.INDEXER, block_id
        ) == 2


def test_allocator_reuses_latent_freed_bundles_for_indexer():
    allocator = DSASharedBundleAllocator(_layout(capacity_bundles=6))

    latent_bundles = allocator.allocate(DSASharedBlockOwner.LATENT, 2)
    indexer_bundles = allocator.allocate(DSASharedBlockOwner.INDEXER, 1)

    assert latent_bundles == (1, 2)
    assert indexer_bundles == (3,)
    assert allocator.free_bundle_count == 3

    allocator.free(DSASharedBlockOwner.LATENT, latent_bundles)

    reused = allocator.allocate(DSASharedBlockOwner.INDEXER, 2)
    assert reused == (1, 2)
    assert allocator.free_bundle_count == 3


def test_allocator_prefers_smallest_contiguous_range():
    allocator = DSASharedBundleAllocator(_layout(capacity_bundles=8))

    first = allocator.allocate(DSASharedBlockOwner.LATENT, 2)
    middle = allocator.allocate(DSASharedBlockOwner.INDEXER, 2)
    tail = allocator.allocate(DSASharedBlockOwner.LATENT, 2)

    assert first == (1, 2)
    assert middle == (3, 4)
    assert tail == (5, 6)

    allocator.free(DSASharedBlockOwner.LATENT, first)
    allocator.free(DSASharedBlockOwner.LATENT, tail)

    assert allocator.allocate(DSASharedBlockOwner.INDEXER, 2) == (1, 2)
    assert allocator.allocate(DSASharedBlockOwner.INDEXER, 2) == (5, 6)


def test_allocator_rejects_wrong_owner_and_duplicate_free():
    allocator = DSASharedBundleAllocator(_layout(capacity_bundles=2))
    bundles = allocator.allocate(DSASharedBlockOwner.LATENT, 1)

    with pytest.raises(ValueError, match="owned by"):
        allocator.free(DSASharedBlockOwner.INDEXER, bundles)

    with pytest.raises(ValueError, match="duplicate"):
        allocator.free(DSASharedBlockOwner.LATENT, (bundles[0], bundles[0]))


def test_bundle_count_for_blocks_rounds_up_to_whole_bundle():
    allocator = DSASharedBundleAllocator(_layout(capacity_bundles=8))

    assert allocator.bundle_count_for_blocks(DSASharedBlockOwner.LATENT, 1) == 1
    assert allocator.bundle_count_for_blocks(DSASharedBlockOwner.LATENT, 2) == 1
    assert allocator.bundle_count_for_blocks(DSASharedBlockOwner.LATENT, 3) == 2
    assert allocator.bundle_count_for_blocks(DSASharedBlockOwner.INDEXER, 8) == 1
    assert allocator.bundle_count_for_blocks(DSASharedBlockOwner.INDEXER, 9) == 1
    assert allocator.bundle_count_for_blocks(DSASharedBlockOwner.INDEXER, 10) == 2


def test_shared_pool_uses_each_group_own_logical_pool():
    assert (
        dsa_block_pool_index(
            1,
            use_per_group_block_pools=False,
            use_dsa_shared_block_pool=False,
        )
        == 0
    )
    assert (
        dsa_block_pool_index(
            1,
            use_per_group_block_pools=True,
            use_dsa_shared_block_pool=False,
        )
        == 1
    )
    assert (
        dsa_block_pool_index(
            1,
            use_per_group_block_pools=False,
            use_dsa_shared_block_pool=True,
        )
        == 1
    )


def test_scratch_blocks_are_derived_from_index_topk():
    assert dsa_scratch_blocks_for_topk(2048, 128) == 16
    assert dsa_scratch_blocks_for_topk(1024, 128) == 8

    with pytest.raises(ValueError, match="integer multiple.*index_topk=2049"):
        dsa_scratch_blocks_for_topk(2049, 128)

    with pytest.raises(ValueError, match="index_topk"):
        dsa_scratch_blocks_for_topk(0, 128)
