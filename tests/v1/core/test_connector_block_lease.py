# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBlockLease,
)
from vllm.v1.core.block_pool import BlockPool, DSASharedLogicalBlockPool
from vllm.v1.core.dsa_shared_pool import (
    DSASharedBlockLayout,
    DSASharedBlockOwner,
    DSASharedBundleAllocator,
)
from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator

pytestmark = pytest.mark.cpu_test


def test_connector_lease_keeps_only_job_blocks_alive() -> None:
    pool = BlockPool(
        num_gpu_blocks=8,
        enable_caching=False,
        hash_block_size=16,
    )
    blocks = pool.get_new_blocks(3)
    manager = SimpleNamespace(
        block_pool=pool,
        req_to_blocks={"request": blocks},
    )
    coordinator = SimpleNamespace(single_type_managers=[manager])
    lease = KVConnectorBlockLease(
        source="lmcache",
        request_id="request",
        generation=1,
        job_id=2,
        block_ids=((blocks[1].block_id, blocks[2].block_id),),
    )

    KVCacheCoordinator.acquire_connector_block_lease(coordinator, lease)
    assert [block.ref_cnt for block in blocks] == [1, 2, 2]

    # Drop request ownership. Only the exact save range remains pinned.
    pool.free_blocks(reversed(blocks))
    assert [block.ref_cnt for block in blocks] == [0, 1, 1]

    assert KVCacheCoordinator.release_connector_block_lease(
        coordinator, lease.lease_key
    )
    assert [block.ref_cnt for block in blocks] == [0, 0, 0]


def test_connector_lease_rejects_blocks_from_another_request() -> None:
    pool = BlockPool(
        num_gpu_blocks=8,
        enable_caching=False,
        hash_block_size=16,
    )
    owned = pool.get_new_blocks(1)
    other = pool.get_new_blocks(1)
    manager = SimpleNamespace(
        block_pool=pool,
        req_to_blocks={"request": owned},
    )
    coordinator = SimpleNamespace(single_type_managers=[manager])
    lease = KVConnectorBlockLease(
        source="lmcache",
        request_id="request",
        generation=1,
        job_id=1,
        block_ids=((other[0].block_id,),),
    )

    with pytest.raises(ValueError, match="not owned"):
        KVCacheCoordinator.acquire_connector_block_lease(coordinator, lease)


def test_connector_lease_pins_dsa_shared_bundle() -> None:
    layout = DSASharedBlockLayout(
        latent_page_size_bytes=576,
        indexer_page_size_bytes=128,
        capacity_bundles=8,
    )
    allocator = DSASharedBundleAllocator(layout)
    pool = DSASharedLogicalBlockPool(allocator, DSASharedBlockOwner.LATENT)
    blocks = pool.get_new_blocks(3)
    manager = SimpleNamespace(
        block_pool=pool,
        req_to_blocks={"request": blocks},
    )
    coordinator = SimpleNamespace(single_type_managers=[manager])
    lease = KVConnectorBlockLease(
        source="lmcache",
        request_id="request",
        generation=1,
        job_id=1,
        block_ids=((blocks[0].block_id, blocks[1].block_id),),
    )

    KVCacheCoordinator.acquire_connector_block_lease(coordinator, lease)
    assert [block.ref_cnt for block in blocks] == [2, 2, 1, 1]

    pool.free_blocks(reversed(blocks))
    assert [block.ref_cnt for block in blocks] == [1, 1, 0, 0]
    assert allocator.free_bundle_count == 7

    assert KVCacheCoordinator.release_connector_block_lease(
        coordinator, lease.lease_key
    )
    assert [block.ref_cnt for block in blocks] == [0, 0, 0, 0]
    assert allocator.free_bundle_count == 8
