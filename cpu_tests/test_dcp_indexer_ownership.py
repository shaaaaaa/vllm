# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace as NS

import pytest
import torch

from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, MLAAttentionSpec


def manager(monkeypatch, parents, world=4, paired=False):
    for name, value in (
        ("TWO_GROUPS", "1"),
        ("SHARED_POOL", "1"),
        ("SHRINK_LATENT", "2"),
    ):
        monkeypatch.setenv("VLLM_ASCEND_DSA_" + name, value)
    groups = []
    for dim, size in ((576, world), (128, 1)):
        spec = MLAAttentionSpec(
            block_size=128, num_kv_heads=1, head_size=dim, dtype=torch.bfloat16
        )
        object.__setattr__(spec, "dsa_dcp_size", size)
        object.__setattr__(spec, "dcp_request_sharding", dim == 128)
        object.__setattr__(spec, "indexer_paired_banks", paired and dim == 128)
        if paired and dim == 128:
            object.__setattr__(spec, "page_size_padded", spec.page_size_bytes + 256)
            object.__setattr__(spec, "indexer_scale_page_size_bytes", 256)
            object.__setattr__(spec, "c8_k_cache_dtype", torch.int8)
            object.__setattr__(spec, "shared_indexer_key_dtype", torch.bfloat16)
        groups.append(KVCacheGroupSpec([str(dim)], spec))
    config = KVCacheConfig(
        parents, [], groups, dsa_index_topk=2048, dsa_num_speculative_tokens=1
    )
    return KVCacheManager(
        config, 100_000, 128, enable_caching=False, dcp_world_size=world
    )


def req(name, length):
    return NS(
        request_id=name,
        num_computed_tokens=0,
        num_prompt_tokens=length,
        num_tokens=length,
    )


@pytest.mark.parametrize("paired", [False, True])
def test_physical_storage_and_wire_deltas(monkeypatch, paired):
    from vllm.v1.core.dcp_indexer_pool import LATENT

    cache = manager(monkeypatch, 256, paired=paired)
    pool = cache.coordinator.dcp_request_pool
    requests = [req(str(i), 16384) for i in range(8)]
    physical = [set() for _ in pool.ranks]
    logical = set()
    for i, r in enumerate(requests):
        result = cache.allocate_slots(
            r, 2, num_external_computed_tokens=16382, dsa_compact_external_load=True
        )
        assert result is not None
        latent, indexer = cache.get_blocks(r.request_id).get_block_ids()
        assert len(indexer) == 128
        p = r.dcp_indexer_placement
        assert p.owner == i % 4
        assert not logical.intersection(indexer)
        logical.update(indexer)
        pages = p.indexer_table(indexer)
        assert len(set(pages)) == 128 and not physical[p.owner].intersection(pages)
        physical[p.owner].update(pages)
        for rank, table in enumerate(p.latent_tables(latent)):
            for page in set(table) - {0}:
                parent = pool.layout.bundle_id_for_block(LATENT, page)
                assert pool.ranks[rank]._owners[parent] == LATENT
        item = NS(req_id=r.request_id, block_ids=(latent, indexer))
        output = NS(
            scheduled_new_reqs=[item], scheduled_cached_reqs=NS(new_block_ids=[])
        )
        pool.prepare_output(output)
        assert item.block_ids[1] == pages
        assert output.dcp_indexer_placements[r.request_id] == (
            p.owner,
            p.epoch,
            p.latent_tables(latent),
        )
    assert sum(map(len, physical)) == len(requests) * 128
    for rank in range(4):
        assert len(physical[rank]) == 256
    for r in requests:
        cache.free(r)
    assert all(r.free_bundle_count == 256 for r in pool.ranks)
    assert not pool.groups and not pool.placements


def test_independent_parent_numbers_and_shared_subpages(monkeypatch):
    cache = manager(monkeypatch, 2, world=2)
    from vllm.v1.core.dcp_indexer_pool import INDEXER, LATENT

    p = cache.coordinator.dcp_request_pool
    p.bind("a", 0, 10)
    a = p.allocate("a", INDEXER, 1)
    p.bind("b", 0, 10)
    b = p.allocate("b", INDEXER, 1)
    assert p.ranks[0].free_bundle_count == 1
    p.bind("c", 1, 10)
    first = p.allocate("c", INDEXER, 9)
    c = p.allocate("c", INDEXER, 1)
    p.release(INDEXER, first)
    assert p.choose("d", 2, 0) is not None
    d = p.bind("d", 1, 10)
    rows = p.allocate("d", LATENT, 1)
    assert d.latent[rows[0].block_id] == (4, 2)
    assert p.choose("a", 0, 7) == 0
    p.release(INDEXER, a)
    assert p.ranks[0].free_bundle_count == 0
    p.release(INDEXER, b)
    p.release(INDEXER, c)
    p.release(LATENT, rows)
    assert p.free_bundle_count == 2


def test_partial_allocation_rollback_and_new_epoch(monkeypatch):
    cache = manager(monkeypatch, 100)
    pool = cache.coordinator.dcp_request_pool
    r = req("x", 4096)
    group = cache.coordinator.single_type_managers[1]
    with monkeypatch.context() as mp:

        def fail(*args, **kwargs):
            raise RuntimeError("injected after latent allocation")

        mp.setattr(group, "allocate_new_blocks", fail)
        with pytest.raises(RuntimeError, match="injected"):
            cache.allocate_slots(r, 4096)
    assert pool.free_bundle_count == 100 and not pool.groups and not pool.placements
    assert cache.allocate_slots(r, 4096) is not None
    old = r.dcp_indexer_placement
    old_ids = cache.get_blocks("x").get_block_ids()[1]
    cache.free(r)
    assert cache.allocate_slots(r, 4096) is not None
    assert old.epoch != r.dcp_indexer_placement.epoch
    assert set(old_ids).isdisjoint(cache.get_blocks("x").get_block_ids()[1])


def test_rank_completion_requires_the_same_epoch_and_generation():
    from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
    from vllm.v1.outputs import KVConnectorOutput

    agg = KVOutputAggregator(2)

    def tick(*attempts):
        outputs = [
            NS(
                kv_connector_output=KVConnectorOutput(
                    finished_recving={"r"} if a else None,
                    dcp_received={a} if a else None,
                )
            )
            for a in attempts
        ]
        return agg.aggregate(outputs).kv_connector_output

    old, new = ("r", 5, 1), ("r", 9, 2)
    assert not tick(old, new).finished_recving
    assert not tick(old, None).finished_recving
    assert tick(None, old).dcp_received == {old}
    assert tick(new, None).dcp_received == {new}


def test_failed_rank_reservation_returns_other_ranks_parents(monkeypatch):
    cache = manager(monkeypatch, 120)
    pool = cache.coordinator.dcp_request_pool

    def fail(*args):
        raise ValueError("rank reservation failed")

    monkeypatch.setattr(pool.ranks[2], "allocate", fail)
    with pytest.raises(ValueError, match="rank reservation failed"):
        cache.allocate_slots(req("r", 32768), 128)
    assert pool.free_bundle_count == 120 and not pool.placements and not pool.groups


def test_owner_pressure_excludes_release_on_unrelated_rank(monkeypatch):
    from vllm.v1.core.dcp_indexer_pool import INDEXER

    cache = manager(monkeypatch, 2, world=2)
    pool = cache.coordinator.dcp_request_pool
    pool.bind("growing", 0, 100)
    pool.allocate("growing", INDEXER, 17)
    pool.bind("sibling", 0, 30)
    sibling = pool.allocate("sibling", INDEXER, 1)
    pool.bind("elsewhere", 1, 20)
    pool.allocate("elsewhere", INDEXER, 1)
    assert pool.choose("growing", 0, 1) is None
    a, b = req("sibling", 128), req("elsewhere", 128)
    assert pool.victims("growing", [b]) == []
    assert pool.victims("growing", [a, b]) == [a]
    pool.release(INDEXER, sibling)
    assert pool.choose("growing", 0, 1) == 0


def test_two_pending_requests_allocate_against_their_own_reservation(monkeypatch):
    from vllm.v1.core.dcp_indexer_pool import INDEXER

    pool = manager(monkeypatch, 2, world=2).coordinator.dcp_request_pool
    a = pool.bind("a", 0, 9)
    b = pool.bind("b", 1, 9)
    a_pages = pool.allocate("a", INDEXER, 2)
    b_pages = pool.allocate("b", INDEXER, 2)
    assert a.indexer_table([v.block_id for v in a_pages]) == b.indexer_table(
        [v.block_id for v in b_pages]
    )
    assert set(a.indexer).isdisjoint(b.indexer)
    assert [r.free_bundle_count for r in pool.ranks] == [1, 1]


def test_growth_failure_preserves_live_placement(monkeypatch):
    cache = manager(monkeypatch, 24)
    r = req("live", 128)
    assert cache.allocate_slots(r, 128) is not None
    r.num_computed_tokens = 128
    placement = r.dcp_indexer_placement
    original = dict(placement.latent), dict(placement.indexer), cache.get_block_ids("live")
    pool = cache.coordinator.dcp_request_pool
    free = [rank.free_bundle_count for rank in pool.ranks]
    def fail(*args, **kwargs):
        raise RuntimeError("fail an existing request's indexer growth")
    monkeypatch.setattr(cache.coordinator.single_type_managers[1], "allocate_new_blocks", fail)
    with pytest.raises(RuntimeError, match="growth"):
        cache.allocate_slots(r, 2048)
    assert r.dcp_indexer_placement is placement
    assert (placement.latent, placement.indexer, cache.get_block_ids("live")) == original
    assert [rank.free_bundle_count for rank in pool.ranks] == free


@pytest.mark.parametrize("growth", [384, 8192], ids=["indexer", "latent"])
def test_block_construction_failure_is_retryable(monkeypatch, growth):
    from vllm.v1.core import dcp_indexer_pool as dcp

    cache = manager(monkeypatch, 64)
    r = req("live", 128)
    assert cache.allocate_slots(r, 128) is not None
    r.num_computed_tokens = 128
    p = r.dcp_indexer_placement
    pool = cache.coordinator.dcp_request_pool

    def snapshot():
        return (
            cache.get_block_ids("live"),
            dict(p.latent),
            dict(p.indexer),
            set(pool.groups),
            list(pool.holes),
            list(pool.load),
            [dict(rank._owners) for rank in pool.ranks],
            [{k: tuple(v) for k, v in pages.items()} for pages in pool.available],
        )

    before = snapshot()
    create = dcp.KVCacheBlock
    calls = 0

    def fail_second_page(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise MemoryError("injected block metadata exhaustion")
        return create(*args, **kwargs)

    with monkeypatch.context() as mp:
        mp.setattr(dcp, "KVCacheBlock", fail_second_page)
        with pytest.raises(MemoryError, match="metadata exhaustion"):
            cache.allocate_slots(r, growth)
    assert snapshot() == before
    assert cache.allocate_slots(r, growth) is not None
    cache.free(r)
    assert not pool.groups and all(rank.free_bundle_count == 64 for rank in pool.ranks)


def test_page_growth_does_not_copy_million_token_history(monkeypatch):
    cache = manager(monkeypatch, 2000)
    cache.max_model_len = 1024 * 1024
    r = req("long", 1_000_000)
    assert (
        cache.allocate_slots(
            r, 0, num_external_computed_tokens=1_000_000, dsa_compact_external_load=True
        )
        is not None
    )

    class CountCopies(list):
        copied = 0

        def __getitem__(self, index):
            result = super().__getitem__(index)
            if isinstance(index, slice):
                self.copied += len(result)
            return result

    indexer = cache.coordinator.single_type_managers[1]
    pages = CountCopies(indexer.req_to_blocks["long"])
    indexer.req_to_blocks["long"] = pages
    r.num_computed_tokens = 1_000_000
    assert cache.allocate_slots(r, 128) is not None
    assert len(pages) == 7814
    assert pages.copied == 0
