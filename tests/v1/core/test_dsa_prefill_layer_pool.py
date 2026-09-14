# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.core.block_pool import DSASharedLogicalBlockPool
from vllm.v1.core.dsa_shared_pool import (
    DSABlockAllocationMode,
    DSASharedBlockLayout,
    DSASharedBlockOwner,
    DSASharedBundleAllocator,
    PrefillLayerBundlePool,
)
from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.kv_cache_interface import layerwise_prefill_p_node_enabled


def make_pool(
    *, parent_capacity: int = 4, physical_slots: int = 5
) -> tuple[DSASharedBundleAllocator, PrefillLayerBundlePool]:
    layout = DSASharedBlockLayout(
        latent_page_size_bytes=576,
        indexer_page_size_bytes=128,
        capacity_bundles=parent_capacity,
    )
    parent = DSASharedBundleAllocator(layout)
    return parent, PrefillLayerBundlePool(parent, physical_slots)


def test_child_id_mapping_is_dense_and_reversible() -> None:
    _, pool = make_pool(parent_capacity=7, physical_slots=4)

    child_ids = []
    for slot in range(pool.num_physical_slots):
        for parent_id in range(1, pool.parent_capacity + 1):
            child_id = pool.child_bundle_id(slot, parent_id)
            child_ids.append(child_id)
            assert pool.physical_slot(child_id) == slot
            assert pool.parent_bundle_id(child_id) == parent_id

    assert child_ids == list(range(1, pool.layout.capacity_bundles + 1))
    with pytest.raises(ValueError):
        pool.parent_bundle_id(0)
    with pytest.raises(ValueError):
        pool.child_bundle_id(pool.num_physical_slots, 1)


def test_child_allocator_packs_and_releases_reserved_parents() -> None:
    parent, pool = make_pool(parent_capacity=4, physical_slots=5)

    first = pool.allocate(DSASharedBlockOwner.LATENT, 3)
    assert first == (1, 5, 9)
    assert pool.reserved_parent_count == 1
    assert parent.free_bundle_count == 3

    # Fill the two remaining children in parent 1 before using parent 2.
    second = pool.allocate(DSASharedBlockOwner.INDEXER, 3)
    assert second == (13, 17, 2)
    assert pool.reserved_parent_count == 2
    assert parent.free_bundle_count == 2

    pool.free(DSASharedBlockOwner.LATENT, first)
    assert pool.reserved_parent_count == 2
    pool.free(DSASharedBlockOwner.INDEXER, second)
    assert pool.reserved_parent_count == 0
    assert parent.free_bundle_count == parent.layout.capacity_bundles


def test_child_allocator_rejects_invalid_free_atomically() -> None:
    parent, pool = make_pool(parent_capacity=2, physical_slots=3)
    latent_ids = pool.allocate(DSASharedBlockOwner.LATENT, 2)
    indexer_ids = pool.allocate(DSASharedBlockOwner.INDEXER, 1)
    owners_before = pool.owner_bundle_counts()
    free_before = pool.free_bundle_count

    with pytest.raises(ValueError, match="duplicate"):
        pool.free(
            DSASharedBlockOwner.LATENT,
            (latent_ids[0], latent_ids[0]),
        )
    with pytest.raises(ValueError, match="not DSASharedBlockOwner.LATENT"):
        pool.free(DSASharedBlockOwner.LATENT, indexer_ids)

    assert pool.owner_bundle_counts() == owners_before
    assert pool.free_bundle_count == free_before
    assert parent.free_bundle_count == 1

    pool.free(DSASharedBlockOwner.LATENT, latent_ids)
    pool.free(DSASharedBlockOwner.INDEXER, indexer_ids)
    assert parent.free_bundle_count == parent.layout.capacity_bundles


def test_two_bank_logical_blocks_pin_and_free_all_physical_banks() -> None:
    parent, child = make_pool(parent_capacity=4, physical_slots=5)
    latent = DSASharedLogicalBlockPool(child, DSASharedBlockOwner.LATENT)

    blocks = latent.get_new_blocks(1)
    # Bundle granularity rounds one latent block up to two logical blocks.
    assert len(blocks) == 2
    assert child.reserved_parent_count == 1
    assert all(block.bank_block_ids is not None for block in blocks)
    assert all(
        block.allocation_mode == DSABlockAllocationMode.PREFILL_CHILD
        for block in blocks
    )
    physical_ids = {
        block_id
        for block in blocks
        for block_id in block.bank_block_ids or ()
    }
    assert len(physical_ids) == 4
    assert all(latent.blocks[block_id].ref_cnt == 1 for block_id in physical_ids)

    latent.pin_blocks(blocks)
    assert all(latent.blocks[block_id].ref_cnt == 2 for block_id in physical_ids)
    latent.free_blocks(blocks)
    assert all(latent.blocks[block_id].ref_cnt == 1 for block_id in physical_ids)
    assert child.reserved_parent_count == 1
    latent.free_blocks(blocks)
    assert all(latent.blocks[block_id].ref_cnt == 0 for block_id in physical_ids)
    assert child.reserved_parent_count == 0
    assert parent.free_bundle_count == parent.layout.capacity_bundles


def test_logical_pool_rejects_duplicate_free_without_changing_refcounts() -> None:
    parent, child = make_pool(parent_capacity=2, physical_slots=3)
    latent = DSASharedLogicalBlockPool(child, DSASharedBlockOwner.LATENT)
    blocks = latent.get_new_blocks(1)
    physical_ids = blocks[0].bank_block_ids
    assert physical_ids is not None
    refcounts_before = [latent.blocks[block_id].ref_cnt for block_id in physical_ids]

    with pytest.raises(ValueError, match="duplicate.*logical blocks"):
        latent.free_blocks([blocks[0], blocks[0]])

    assert [
        latent.blocks[block_id].ref_cnt for block_id in physical_ids
    ] == refcounts_before
    assert child.reserved_parent_count == 1
    latent.free_blocks(blocks)
    assert parent.free_bundle_count == parent.layout.capacity_bundles


def test_latent_and_indexer_share_child_capacity_without_overlap() -> None:
    _, child = make_pool(parent_capacity=3, physical_slots=4)
    latent = DSASharedLogicalBlockPool(child, DSASharedBlockOwner.LATENT)
    indexer = DSASharedLogicalBlockPool(child, DSASharedBlockOwner.INDEXER)

    latent_blocks = latent.get_new_blocks(2)
    indexer_blocks = indexer.get_new_blocks(9)
    latent_physical = {
        block_id
        for block in latent_blocks
        for block_id in block.bank_block_ids or ()
    }
    indexer_bundles = {
        child.layout.bundle_id_for_block(
            DSASharedBlockOwner.INDEXER, block_id
        )
        for block in indexer_blocks
        for block_id in block.bank_block_ids or ()
    }
    latent_bundles = {
        child.layout.bundle_id_for_block(
            DSASharedBlockOwner.LATENT, block_id
        )
        for block_id in latent_physical
    }
    assert latent_bundles.isdisjoint(indexer_bundles)

    latent.free_blocks(latent_blocks)
    indexer.free_blocks(indexer_blocks)
    assert child.free_bundle_count == child.layout.capacity_bundles


def test_two_bank_capacity_is_charged_before_allocation() -> None:
    _, child = make_pool(parent_capacity=2, physical_slots=3)
    latent = DSASharedLogicalBlockPool(child, DSASharedBlockOwner.LATENT)

    # Six child bundles hold three two-bank logical bundles, i.e. six latent
    # blocks after bundle rounding.
    assert latent.get_num_free_blocks() == 6
    blocks = latent.get_new_blocks(6)
    assert len(blocks) == 6
    assert latent.get_num_free_blocks() == 0
    with pytest.raises(ValueError):
        latent.get_new_blocks(1)
    latent.free_blocks(blocks)


def test_scheduler_exports_bank_major_physical_ids() -> None:
    _, child = make_pool(parent_capacity=3, physical_slots=4)
    latent = DSASharedLogicalBlockPool(child, DSASharedBlockOwner.LATENT)
    indexer = DSASharedLogicalBlockPool(child, DSASharedBlockOwner.INDEXER)
    latent_blocks = latent.get_new_blocks(2)
    indexer_blocks = indexer.get_new_blocks(9)

    blocks = KVCacheBlocks((latent_blocks, indexer_blocks))
    assert (
        blocks.get_allocation_mode()
        == DSABlockAllocationMode.PREFILL_CHILD
    )
    bank_ids = blocks.get_block_ids_by_bank()
    assert bank_ids is not None
    assert len(bank_ids) == 2
    assert bank_ids[0][0] == [block.block_id for block in latent_blocks]
    assert bank_ids[0][1] == [block.block_id for block in indexer_blocks]
    for bank in range(2):
        assert bank_ids[bank][0] == [
            block.bank_block_ids[bank] for block in latent_blocks
        ]
        assert bank_ids[bank][1] == [
            block.bank_block_ids[bank] for block in indexer_blocks
        ]


@pytest.mark.parametrize(
    ("blocks", "error"),
    [
        (
            (
                KVCacheBlock(
                    1,
                    bank_block_ids=(1, 2),
                    allocation_mode=DSABlockAllocationMode.PREFILL_CHILD,
                ),
                KVCacheBlock(2),
            ),
            "typed and untyped",
        ),
        (
            (
                KVCacheBlock(
                    1,
                    bank_block_ids=(1, 2),
                    allocation_mode=DSABlockAllocationMode.PREFILL_CHILD,
                ),
                KVCacheBlock(
                    2,
                    allocation_mode=DSABlockAllocationMode.FULL_PARENT,
                ),
            ),
            "mixed allocation modes",
        ),
    ],
)
def test_kv_cache_blocks_rejects_mixed_allocation_identity(
    blocks: tuple[KVCacheBlock, ...], error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        KVCacheBlocks((blocks,)).get_allocation_mode()


@pytest.mark.parametrize(
    ("blocks", "error"),
    [
        (
            (
                KVCacheBlock(
                    1,
                    bank_block_ids=(1, 2),
                    allocation_mode=DSABlockAllocationMode.PREFILL_CHILD,
                ),
                KVCacheBlock(
                    2,
                    allocation_mode=DSABlockAllocationMode.PREFILL_CHILD,
                ),
            ),
            "banked and ordinary",
        ),
        (
            (
                KVCacheBlock(
                    1,
                    bank_block_ids=(1, 2),
                    allocation_mode=DSABlockAllocationMode.PREFILL_CHILD,
                ),
                KVCacheBlock(
                    2,
                    bank_block_ids=(4, 5, 6),
                    allocation_mode=DSABlockAllocationMode.PREFILL_CHILD,
                ),
            ),
            "inconsistent bank counts",
        ),
    ],
)
def test_kv_cache_blocks_rejects_malformed_bank_metadata(
    blocks: tuple[KVCacheBlock, ...], error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        KVCacheBlocks((blocks,)).get_block_ids_by_bank()


def test_p_node_environment_is_strict_and_defaults_false(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE", raising=False)
    assert not layerwise_prefill_p_node_enabled()
    monkeypatch.setenv("VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE", "TrUe")
    assert layerwise_prefill_p_node_enabled()
    monkeypatch.setenv("VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE", "0")
    with pytest.raises(ValueError, match="must be 'true' or 'false'"):
        layerwise_prefill_p_node_enabled()


def test_two_group_allocation_rolls_back_atomically() -> None:
    old_block = KVCacheBlock(1)
    new_block = KVCacheBlock(2)
    freed = []

    class FakePool:
        @staticmethod
        def free_blocks(blocks) -> None:
            freed.extend(blocks)

    class FirstManager:
        def __init__(self) -> None:
            self.req_to_blocks = {"req": [old_block]}
            self.block_pool = FakePool()

        def allocate_new_blocks(self, request_id, _tokens, _main_tokens):
            self.req_to_blocks[request_id].append(new_block)
            return [new_block]

    class FailingManager:
        req_to_blocks = {"req": []}
        block_pool = FakePool()

        @staticmethod
        def allocate_new_blocks(_request_id, _tokens, _main_tokens):
            raise RuntimeError("injected allocation failure")

    class ConcreteCoordinator(KVCacheCoordinator):
        def find_longest_cache_hit(self, *args, **kwargs):
            raise NotImplementedError

    first = FirstManager()
    coordinator = ConcreteCoordinator.__new__(ConcreteCoordinator)
    coordinator.single_type_managers = [first, FailingManager()]
    coordinator.layerwise_prefill_p_node = True

    with pytest.raises(RuntimeError, match="injected allocation failure"):
        coordinator.allocate_new_blocks("req", 16, 16)

    assert first.req_to_blocks["req"] == [old_block]
    assert freed == [new_block]
