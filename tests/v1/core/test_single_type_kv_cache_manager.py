# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random

import pytest
import torch

from vllm.v1.core.block_pool import BlockPool, DSASharedLogicalBlockPool
from vllm.v1.core.dsa_shared_pool import (
    DSASharedBlockLayout,
    DSASharedBlockOwner,
    DSASharedBundleAllocator,
    dsa_scratch_blocks_for_topk,
)
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    KVCacheBlock,
    make_block_hash_with_group_id,
)
from vllm.v1.core.single_type_kv_cache_manager import (
    ChunkedLocalAttentionManager,
    DSALatentManager,
    SlidingWindowManager,
)
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
)

pytestmark = pytest.mark.cpu_test


def get_sliding_window_manager(sliding_window_spec, block_pool, enable_caching=True):
    return SlidingWindowManager(
        sliding_window_spec,
        block_pool=block_pool,
        enable_caching=enable_caching,
        kv_cache_group_id=0,
    )


def get_chunked_local_attention_manager(
    chunked_local_attention_spec, block_pool, enable_caching=True
):
    return ChunkedLocalAttentionManager(
        chunked_local_attention_spec,
        block_pool=block_pool,
        enable_caching=enable_caching,
        kv_cache_group_id=0,
    )


def get_dsa_latent_manager(attention_spec, block_pool, enable_caching=True):
    return DSALatentManager(
        attention_spec,
        block_pool=block_pool,
        enable_caching=enable_caching,
        kv_cache_group_id=0,
    )


def test_chunked_local_attention_possible_cached_prefix():
    block_size = 2
    chunked_local_attention_spec = ChunkedLocalAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        attention_chunk_size=4,
    )

    block_pool = BlockPool(
        num_gpu_blocks=100, enable_caching=True, hash_block_size=block_size
    )
    manager = get_chunked_local_attention_manager(
        chunked_local_attention_spec, block_pool
    )

    def run_one_case(block_is_cached, tail_token, expect_length):
        block_hash_list = [
            BlockHash(str(i).encode()) for i in range(len(block_is_cached))
        ]

        block_pool.cached_block_hash_to_block._cache.clear()

        # Mock the block pool with the cached blocks
        for i, (block_hash, is_cached) in enumerate(
            zip(block_hash_list, block_is_cached)
        ):
            if is_cached:
                block_pool.cached_block_hash_to_block.insert(
                    make_block_hash_with_group_id(block_hash, 0),
                    block_pool.blocks[i + 10],
                )

        computed_blocks = manager.find_longest_cache_hit(
            block_hashes=block_hash_list,
            max_length=len(block_hash_list) * block_size + tail_token,
            kv_cache_group_ids=[0],
            block_pool=block_pool,
            kv_cache_spec=chunked_local_attention_spec,
            use_eagle=False,
            alignment_tokens=block_size,
        )[0]
        assert len(computed_blocks) == expect_length

        assert all(
            block == block_pool.null_block
            for block in computed_blocks[: (expect_length - 1) // 2]
        )

    run_one_case([True], 0, 1)
    run_one_case([True], 1, 1)
    run_one_case([True, False], 0, 2)
    run_one_case([True, False], 1, 2)
    run_one_case([True, True], 0, 2)
    run_one_case([True, True], 1, 2)
    run_one_case([True, True, False], 0, 2)
    run_one_case([True, True, False], 1, 2)
    run_one_case([True, True, True], 0, 3)
    run_one_case([True, True, True], 1, 3)
    run_one_case([True, True, True, False], 0, 4)
    run_one_case([True, True, True, False], 1, 4)
    run_one_case([random.choice([True, False])] * 8 + [True], 1, 9)
    run_one_case([random.choice([True, False])] * 8 + [False], 1, 8)
    run_one_case([random.choice([True, False])] * 8 + [True, True], 1, 10)
    run_one_case([random.choice([True, False])] * 8 + [True, False], 0, 10)
    run_one_case([random.choice([True, False])] * 8 + [True, False], 1, 10)
    run_one_case([random.choice([True, False])] * 8 + [False, True], 0, 10)
    run_one_case([random.choice([True, False])] * 8 + [False, True], 1, 10)
    run_one_case([random.choice([True, False])] * 8 + [False, False], 0, 10)
    run_one_case([random.choice([True, False])] * 8 + [False, False], 1, 10)


def test_sliding_window_possible_cached_prefix():
    block_size = 2
    sliding_window_spec = SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=4,
    )

    block_pool = BlockPool(
        num_gpu_blocks=100, enable_caching=True, hash_block_size=block_size
    )
    manager = get_sliding_window_manager(sliding_window_spec, block_pool)

    def run_one_case(block_is_cached, expect_length):
        block_hash_list = [
            BlockHash(str(i).encode()) for i in range(len(block_is_cached))
        ]

        block_pool.cached_block_hash_to_block._cache.clear()

        # Mock the block pool with the cached blocks
        for i, (block_hash, is_cached) in enumerate(
            zip(block_hash_list, block_is_cached)
        ):
            if is_cached:
                block_pool.cached_block_hash_to_block.insert(
                    make_block_hash_with_group_id(block_hash, 0),
                    block_pool.blocks[i + 10],
                )

        computed_blocks = manager.find_longest_cache_hit(
            block_hashes=block_hash_list,
            max_length=len(block_hash_list) * block_size,
            kv_cache_group_ids=[0],
            block_pool=block_pool,
            kv_cache_spec=sliding_window_spec,
            use_eagle=False,
            alignment_tokens=block_size,
        )[0]
        assert len(computed_blocks) == expect_length

        assert all(
            block == block_pool.null_block
            for block in computed_blocks[: expect_length - 2]
        )
        for i in range(2):
            if i < expect_length:
                block_index = expect_length - i - 1
                assert computed_blocks[block_index].block_id == block_index + 10

    run_one_case([False] * 10, 0)
    run_one_case([True], 1)
    run_one_case([True, False], 1)
    run_one_case([True, True], 2)
    run_one_case([True, True, False], 2)
    run_one_case([True, True, True], 3)
    run_one_case([True, True, True, False], 3)
    run_one_case(
        [True, True, False, True, False, False, True, True, False, True, True, True], 12
    )
    run_one_case(
        [True, True, False, True, False, False, True, True, False, False, False], 8
    )
    run_one_case(
        [True, True, False, True, False, False, True, True, False, False, False, True],
        8,
    )


def test_chunked_local_attention_remove_skipped_blocks():
    attention_spec = ChunkedLocalAttentionSpec(
        block_size=2,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        attention_chunk_size=4,
    )

    block_pool = BlockPool(num_gpu_blocks=2000, enable_caching=True, hash_block_size=2)

    manager = get_chunked_local_attention_manager(attention_spec, block_pool)

    null_block_id = block_pool.null_block.block_id

    def id_to_block_table(ids) -> list[KVCacheBlock]:
        return [
            KVCacheBlock(id_) if id_ != null_block_id else block_pool.null_block
            for id_ in ids
        ]

    def assert_block_id(block_table: list[KVCacheBlock], ids: list[int]):
        for block, id_ in zip(block_table, ids):
            if id_ == null_block_id:
                assert block == block_pool.null_block
            else:
                assert block.block_id == id_

    original_block_ids = [
        1000,
        1001,
        1002,
        1003,
        1004,
        1005,
        1006,
        1007,
        1008,
        1009,
        1010,
    ]
    block_table = id_to_block_table(original_block_ids)
    manager.req_to_blocks["test"] = block_table

    manager.remove_skipped_blocks("test", 0)
    assert_block_id(block_table, original_block_ids)

    # For 4th token (0-indexed), token 0-3 is out of the local attention window.
    manager.remove_skipped_blocks("test", 4)
    assert_block_id(block_table, [null_block_id] * 2)

    # For 6th token (0-indexed), token 4 - 6 are in local attention window,
    # token 0 - 3 are out, 2 blocks can be removed.
    manager.remove_skipped_blocks("test", 6)
    assert_block_id(block_table, [null_block_id] * 2 + original_block_ids[2:])
    # For 12th token (0-indexed),
    # token 0-11 are out, 6 block can be removed.
    manager.remove_skipped_blocks("test", 12)
    assert_block_id(block_table, [null_block_id] * 6)


def test_sliding_window_remove_skipped_blocks():
    sliding_window_spec = SlidingWindowSpec(
        block_size=2,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=4,
    )

    block_pool = BlockPool(num_gpu_blocks=2000, enable_caching=True, hash_block_size=2)

    manager = get_sliding_window_manager(sliding_window_spec, block_pool)

    null_block_id = block_pool.null_block.block_id

    def id_to_block_table(ids) -> list[KVCacheBlock]:
        return [
            KVCacheBlock(id_) if id_ != null_block_id else block_pool.null_block
            for id_ in ids
        ]

    def assert_block_id(block_table: list[KVCacheBlock], ids: list[int]):
        for block, id_ in zip(block_table, ids):
            if id_ == null_block_id:
                assert block == block_pool.null_block
            else:
                assert block.block_id == id_

    original_block_ids = [
        1000,
        1001,
        1002,
        1003,
        1004,
        1005,
        1006,
        1007,
        1008,
        1009,
        1010,
    ]
    block_table = id_to_block_table(original_block_ids)
    manager.req_to_blocks["test"] = block_table

    manager.remove_skipped_blocks("test", 0)
    assert_block_id(block_table, original_block_ids)

    # 4 tokens are computed. Only token 0 is out of the sliding window. As
    # block 1000 also contains token 1 that is in the sliding window, block 1000
    # cannot be removed.
    manager.remove_skipped_blocks("test", 4)
    assert_block_id(block_table, original_block_ids)

    # 5 tokens are computed. Token 0 & 1 are out of the sliding window.
    # Block 1000 can be removed.
    manager.remove_skipped_blocks("test", 5)
    assert_block_id(block_table, [null_block_id] + original_block_ids[1:])

    # 6 tokens are computed. Token 0-2 are out of the sliding window.
    # Cannot remove new block as the block 1001 is still used by token 3.
    manager.remove_skipped_blocks("test", 6)
    assert_block_id(block_table, [null_block_id] + original_block_ids[1:])

    # 7 tokens are computed. Token 0-3 are out of the sliding window.
    # Block 1001 can be removed and block 1000 is already removed.
    manager.remove_skipped_blocks("test", 7)
    assert_block_id(block_table, [null_block_id] * 2 + original_block_ids[2:])

    # 11 tokens are computed. Token 0-7 are out of the sliding window.
    # Block 1002 & 1003 can be removed now. Block 1003 represents a longer
    # sequence, and is expected to be evicted earlier than 1002, so the order
    # of removed blocks should be [1003, 1002].
    manager.remove_skipped_blocks("test", 11)
    assert_block_id(block_table, [null_block_id] * 4 + original_block_ids[4:])


def test_dsa_latent_decode_window_release_waits_for_completed_save(monkeypatch):
    attention_spec = MLAAttentionSpec(
        block_size=2,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.float32,
    )
    block_pool = BlockPool(num_gpu_blocks=2000, enable_caching=True, hash_block_size=2)
    manager = get_dsa_latent_manager(attention_spec, block_pool)
    manager.scratch_blocks = 2

    null_block_id = block_pool.null_block.block_id

    def id_to_block_table(ids) -> list[KVCacheBlock]:
        return [
            KVCacheBlock(id_) if id_ != null_block_id else block_pool.null_block
            for id_ in ids
        ]

    def assert_block_id(block_table: list[KVCacheBlock], ids: list[int]):
        for block, id_ in zip(block_table, ids):
            if id_ == null_block_id:
                assert block == block_pool.null_block
            else:
                assert block.block_id == id_

    original_block_ids = [1000, 1001, 1002, 1003, 1004, 1005]
    block_table = id_to_block_table(original_block_ids)
    manager.req_to_blocks["test"] = block_table

    monkeypatch.setenv("LMCACHE_DECODE_WINDOW_SAVE_WINDOW_SIZE", "4")
    manager.remove_skipped_blocks("test", total_computed_tokens=6, num_prompt_tokens=4)
    assert_block_id(block_table, original_block_ids)

    assert manager.remove_saved_decode_window_blocks("test", committed_end=4) == 0
    assert_block_id(block_table, original_block_ids)

    assert manager.remove_saved_decode_window_blocks("test", committed_end=8) == 2
    assert_block_id(
        block_table,
        original_block_ids[:2]
        + [null_block_id, null_block_id]
        + original_block_ids[4:],
    )


def test_dsa_scratch_capacity_uses_max_mtp_rows_and_requires_topk_alignment():
    assert dsa_scratch_blocks_for_topk(2048, 256, num_rows=2) == 16
    with pytest.raises(ValueError, match="integer multiple.*index_topk=2049"):
        dsa_scratch_blocks_for_topk(2049, 256, num_rows=2)


def test_dsa_release_keeps_fixed_mtp_union_scratch_prefix():
    attention_spec = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.float32,
    )
    block_pool = BlockPool(
        num_gpu_blocks=64, enable_caching=False, hash_block_size=256
    )
    manager = get_dsa_latent_manager(attention_spec, block_pool)
    manager.scratch_blocks = 16  # (1 + one speculative token) * 2048 / 256
    blocks = block_pool.get_new_blocks(40)
    manager.req_to_blocks["mtp"] = blocks

    assert manager.remove_saved_decode_window_blocks("mtp", 512) == 0
    assert manager.remove_saved_decode_window_blocks("mtp", 3072) == 0
    assert manager.remove_saved_decode_window_blocks("mtp", 8192) == 16
    assert all(block != block_pool.null_block for block in blocks[:16])
    assert all(block == block_pool.null_block for block in blocks[16:32])
    assert all(block != block_pool.null_block for block in blocks[32:])


@pytest.mark.parametrize(
    ("request_id", "prompt_len", "external_tokens"),
    [("short", 3000, 2999), ("long", 8192, 8191)],
)
def test_dsa_external_prefix_hit_allocates_every_latent_block(
    request_id, prompt_len, external_tokens
):
    attention_spec = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.float32,
    )
    block_pool = BlockPool(
        num_gpu_blocks=64, enable_caching=False, hash_block_size=256
    )
    manager = get_dsa_latent_manager(attention_spec, block_pool)
    manager.scratch_blocks = 16
    manager.allocate_new_computed_blocks(request_id, [], 0, external_tokens)

    blocks = manager.req_to_blocks[request_id]
    assert len(blocks) == (prompt_len + 255) // 256
    assert all(block != block_pool.null_block for block in blocks)
    assert not manager._compact_external_prefix_tokens


def test_dsa_compact_external_admission_reserves_scratch_and_live_tail():
    attention_spec = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.float32,
    )
    block_pool = BlockPool(
        num_gpu_blocks=128, enable_caching=False, hash_block_size=256
    )
    manager = get_dsa_latent_manager(attention_spec, block_pool)
    manager.scratch_blocks = 16

    manager.allocate_new_computed_blocks_compact_external(
        "cold", [], 0, 8191
    )
    blocks = manager.req_to_blocks["cold"]
    assert len(blocks) == 32
    assert sum(not block.is_null for block in blocks) == 16
    assert all(not block.is_null for block in blocks[:16])
    assert all(block.is_null for block in blocks[16:])
    assert (
        manager.get_num_blocks_to_allocate_compact_external(
            "cold", 8193, [], 8191, 8192
        )
        == 2
    )
    tail = manager.allocate_new_blocks_compact_external("cold", 8193, 8192)
    assert len(tail) == 2
    assert len(blocks) == 33
    assert all(block.is_null for block in blocks[16:31])
    assert all(not block.is_null for block in blocks[31:])


def test_dsa_compact_external_grows_tail_without_materializing_prompt():
    attention_spec = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.float32,
    )
    block_pool = BlockPool(
        num_gpu_blocks=128, enable_caching=False, hash_block_size=256
    )
    manager = get_dsa_latent_manager(attention_spec, block_pool)
    manager.scratch_blocks = 16
    manager.allocate_new_computed_blocks_compact_external("cold", [], 0, 8191)

    new_blocks = manager.allocate_new_blocks("cold", 8194, 8192)
    blocks = manager.req_to_blocks["cold"]
    assert len(new_blocks) == 2
    assert not blocks[31].is_null
    assert not blocks[32].is_null
    assert all(block.is_null for block in blocks[16:31])


def test_dsa_compact_external_release_keeps_worker_updates_append_only():
    attention_spec = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.float32,
    )
    layout = DSASharedBlockLayout(
        latent_page_size_bytes=576,
        indexer_page_size_bytes=128,
        capacity_bundles=64,
    )
    allocator = DSASharedBundleAllocator(layout)
    block_pool = DSASharedLogicalBlockPool(
        allocator, DSASharedBlockOwner.LATENT
    )
    manager = get_dsa_latent_manager(
        attention_spec, block_pool, enable_caching=False
    )
    manager.scratch_blocks = 16

    manager.allocate_new_computed_blocks_compact_external("cold", [], 0, 8191)
    manager.allocate_new_blocks_compact_external("cold", 8193, 8192)
    blocks = manager.req_to_blocks["cold"]
    worker_block_count = len(blocks)
    assert worker_block_count == 34
    assert all(not block.is_null for block in blocks[30:])

    # Once LMCache has persisted blocks 0..31, positions 30 and 31 become
    # historical holes. A later scheduler update must allocate only the new
    # suffix: CachedRequestData tells a running worker to append these IDs.
    assert manager.remove_saved_decode_window_blocks("cold", 8192) == 2
    assert all(block.is_null for block in blocks[16:32])
    assert manager.get_num_blocks_to_allocate("cold", 8705, [], 8192, 8704) == 2

    new_blocks = manager.allocate_new_blocks("cold", 8705, 8704)
    assert len(new_blocks) == 2
    assert blocks[worker_block_count:] == new_blocks
    assert len(blocks) == worker_block_count + len(new_blocks)


def test_dsa_compact_external_load_failure_expands_dense_prefill():
    attention_spec = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.float32,
    )
    block_pool = BlockPool(
        num_gpu_blocks=128, enable_caching=False, hash_block_size=256
    )
    manager = get_dsa_latent_manager(attention_spec, block_pool)
    manager.scratch_blocks = 16
    manager.allocate_new_computed_blocks_compact_external("cold", [], 0, 8191)
    manager.free("cold")
    assert not manager._compact_external_released_blocks

    new_blocks = manager.allocate_new_blocks("cold", 5000, 4999)
    blocks = manager.req_to_blocks["cold"]
    assert len(new_blocks) == 20
    assert all(not block.is_null for block in blocks)


@pytest.mark.parametrize(
    "owner", [DSASharedBlockOwner.LATENT, DSASharedBlockOwner.INDEXER]
)
def test_dsa_shared_pool_reclaims_bundle_after_partial_frees(owner):
    layout = DSASharedBlockLayout(
        latent_page_size_bytes=576,
        indexer_page_size_bytes=128,
        capacity_bundles=4,
    )
    allocator = DSASharedBundleAllocator(layout)
    pool = DSASharedLogicalBlockPool(allocator, owner)
    blocks = pool.get_new_blocks(pool.blocks_per_bundle)
    assert allocator.free_bundle_count == 3

    pool.free_blocks(blocks[:1])
    assert allocator.free_bundle_count == 3
    assert blocks[0].ref_cnt == 0

    pool.free_blocks(blocks[1:])
    assert allocator.free_bundle_count == 4
    assert all(block.ref_cnt == 0 for block in blocks)


def test_get_num_blocks_to_allocate():
    block_size = 2
    sliding_window_spec = SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=4,  # Placeholder value, not related to test result
    )

    block_pool = BlockPool(
        num_gpu_blocks=100, enable_caching=True, hash_block_size=block_size
    )
    manager = get_sliding_window_manager(sliding_window_spec, block_pool)
    cached_blocks_1 = [KVCacheBlock(i + 1) for i in range(10)]
    cached_blocks_2 = [block_pool.null_block for _ in range(5)] + [
        KVCacheBlock(i + 1) for i in range(5)
    ]

    assert (
        manager.get_num_blocks_to_allocate(
            "1", 20 * block_size, cached_blocks_1, 0, 20 * block_size
        )
        == 20
    )
    assert (
        manager.get_num_blocks_to_allocate(
            "2", 20 * block_size, cached_blocks_2, 0, 20 * block_size
        )
        == 15
    )


def test_evictable_cached_blocks_not_double_allocated():
    block_size = 2
    sliding_window_length = 2 * block_size
    sliding_window_spec = SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=sliding_window_length,
    )

    block_pool = BlockPool(
        num_gpu_blocks=100, enable_caching=True, hash_block_size=block_size
    )
    manager = get_sliding_window_manager(sliding_window_spec, block_pool)

    request_id = "req"
    evictable_block = block_pool.blocks[1]  # ref_cnt == 0, eviction candidate

    num_blocks_to_allocate = manager.get_num_blocks_to_allocate(
        request_id=request_id,
        num_tokens=2 * block_size,
        new_computed_blocks=[evictable_block],
        total_computed_tokens=block_size,
        num_tokens_main_model=2 * block_size,
    )
    # Free capacity check should count evictable cached blocks, but allocation
    # should only allocate the truly new block.
    assert num_blocks_to_allocate == 2

    manager.allocate_new_computed_blocks(
        request_id,
        [evictable_block],
        num_local_computed_tokens=block_size,
        num_external_computed_tokens=0,
    )
    new_blocks = manager.allocate_new_blocks(
        request_id, num_tokens=4, num_tokens_main_model=4
    )
    assert len(new_blocks) == 1
    assert len(manager.req_to_blocks[request_id]) == 2


def test_chunked_local_attention_get_num_blocks_to_allocate():
    block_size = 2
    attention_spec = ChunkedLocalAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        attention_chunk_size=4,  # Placeholder value, not related to test result
    )

    block_pool = BlockPool(
        num_gpu_blocks=100, enable_caching=True, hash_block_size=block_size
    )
    manager = get_chunked_local_attention_manager(attention_spec, block_pool)
    cached_blocks_1 = [KVCacheBlock(i + 1) for i in range(10)]
    cached_blocks_2 = [block_pool.null_block for _ in range(5)] + [
        KVCacheBlock(i + 1) for i in range(5)
    ]

    assert (
        manager.get_num_blocks_to_allocate(
            "1", 20 * block_size, cached_blocks_1, 0, 20 * block_size
        )
        == 20
    )
    assert (
        manager.get_num_blocks_to_allocate(
            "2", 20 * block_size, cached_blocks_2, 0, 20 * block_size
        )
        == 15
    )
