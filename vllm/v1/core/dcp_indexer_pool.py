# SPDX-License-Identifier: Apache-2.0
"""Scheduler-owned request placement for the two-group DCP decoder."""

from collections import Counter
from dataclasses import dataclass, field
from itertools import chain, count

from vllm.v1.core.dsa_shared_pool import DSASharedBlockOwner, DSASharedBundleAllocator
from vllm.v1.core.kv_cache_utils import KVCacheBlock

DCP_INDEXER_PLACEMENT_VERSION = 1

LATENT = DSASharedBlockOwner.LATENT
INDEXER = DSASharedBlockOwner.INDEXER


@dataclass
class RequestPlacement:
    owner: int
    epoch: int
    world: int
    latent: dict[int, tuple[int, ...]] = field(default_factory=dict)
    indexer: dict[int, int] = field(default_factory=dict)

    def latent_tables(self, blocks):
        null = (0,) * self.world
        if not blocks:
            return ((),) * self.world
        return tuple(zip(*(self.latent[b] if b else null for b in blocks), strict=True))

    def indexer_table(self, blocks):
        return [self.indexer[b] if b else 0 for b in blocks]


class DcpRequestPool:
    """G0 takes a whole parent on each rank. G1 suballocates one rank's pages."""

    def __init__(self, first_rank, world):
        self.layout = first_rank.layout
        self.ranks = [first_rank] + [
            DSASharedBundleAllocator(self.layout) for _ in range(world - 1)
        ]
        self.available = [{} for _ in self.ranks]  # parent -> free canonical pages
        self.holes = [0] * world
        self.load = [0] * world
        self.weights = {}
        self.placements = {}
        self.ids = count(1)  # Unique across groups, allocations and physical reuse.
        self.groups = {}
        self.journals = {}
        self.pressure = {}

    @property
    def free_bundle_count(self):
        return min(r.free_bundle_count for r in self.ranks)

    def choose(self, req, latent, indexer, weight=0):
        p = self.placements.get(req)
        if p is not None and latent == 0 and indexer == 0:
            return p.owner
        g = self.ranks[0].bundle_count_for_blocks(LATENT, latent)
        candidates = []
        free = [r.free_bundle_count for r in self.ranks]
        if min(free) >= g:
            eligible = range(len(free)) if p is None else (p.owner,)
            for rank in eligible:
                extra = self.ranks[rank].bundle_count_for_blocks(
                    INDEXER, max(0, indexer - self.holes[rank])
                )
                if g + extra <= free[rank]:
                    projected = max(
                        self.load[peer] + (weight if peer == rank else 0)
                        for peer in range(len(free))
                    )
                    candidates.append((projected, extra, rank))
        if candidates:
            return min(candidates)[2]
        if p is not None:
            self.pressure[req] = (p.owner, g, indexer)
        return None

    def victims(self, req, candidates):
        """Keep scheduling priority/order among releases that reduce the deficit."""
        if req not in self.pressure:
            return candidates
        owner, g, pages = self.pressure[req]
        b = self.layout.indexer_blocks_per_bundle
        free = [r.free_bundle_count for r in self.ranks]

        def deficit(f, h):
            return sum(max(0, g - x) for x in f) * b + max(
                0, pages - h - max(0, f[owner] - g) * b
            )

        before = deficit(free, self.holes[owner])
        helpful = []
        for candidate in candidates:
            p = self.placements.get(candidate.request_id)
            if p is None:
                continue
            common = len({self.groups[k][1] for k in p.latent if k in self.groups})
            after = [x + common for x in free]
            holes = self.holes[owner]
            used = Counter(self.groups[k][1] for k in p.indexer)
            for parent, n in used.items():
                released = len(self.available[p.owner].get(parent, ())) + n == b
                after[p.owner] += int(released)
                if p.owner == owner:
                    holes += n - b * int(released)
            if deficit(after, holes) < before:
                helpful.append(candidate)
        return helpful

    def bind(self, req, rank, weight):
        if req not in self.placements:
            self.placements[req] = RequestPlacement(
                rank, next(self.ids), len(self.ranks)
            )
        if self.placements[req].owner != rank:
            raise ValueError("A live DCP allocation cannot change owner")
        self.load[rank] += weight - self.weights.get(req, 0)
        self.weights[req] = weight
        self.pressure.pop(req, None)
        return self.placements[req]

    def forget(self, req):
        p = self.placements.pop(req, None)
        if p is not None:
            self.load[p.owner] -= self.weights.pop(req)
            self.pressure.pop(req, None)

    def allocate(self, req, kind, n):
        p = self.placements.get(req)
        if p is None or n < 0:
            raise ValueError("DCP allocation requires a bound request")
        if not n:
            return []
        result = []
        journal = self.journals.get(req)
        if kind == LATENT:
            bundles = self.ranks[0].bundle_count_for_blocks(kind, n)
            if bundles > self.free_bundle_count:
                raise ValueError("DCP latent allocation exceeds common capacity")
            parents = []
            try:
                for rank in self.ranks:
                    parents.append(rank.allocate(kind, bundles))
                for vector in zip(*parents, strict=True):
                    rows = [
                        self.layout.block_ids_for_bundle(kind, parent)
                        for parent in vector
                    ]
                    siblings = set()
                    for physical in zip(*rows, strict=True):
                        handle = next(self.ids)
                        result.append(KVCacheBlock(handle, ref_cnt=1))
                        siblings.add(handle)
                        p.latent[handle] = physical
                        self.groups[handle] = (p, vector, siblings)
                if journal is not None:
                    journal.append((kind, result))
            except BaseException:
                for block in result:
                    p.latent.pop(block.block_id, None)
                    self.groups.pop(block.block_id, None)
                for rank, acquired in zip(self.ranks, parents):
                    rank.free(kind, acquired)
                raise
        else:
            rank = p.owner
            partial = self.available[rank]
            needed = self.ranks[rank].bundle_count_for_blocks(
                kind, max(0, n - self.holes[rank])
            )
            existing = sorted(partial, key=lambda parent: len(partial[parent]))
            changed = {}
            new = self.ranks[rank].allocate(kind, needed)
            try:
                for parent in chain(existing, new):
                    if len(result) == n:
                        break
                    pages = partial.get(parent)
                    changed[parent] = pages
                    if pages is None:
                        pages = self.layout.block_ids_for_bundle(kind, parent)[::-1]
                    take = min(n - len(result), len(pages))
                    for i in range(1, take + 1):
                        handle = next(self.ids)
                        result.append(KVCacheBlock(handle, ref_cnt=1))
                        p.indexer[handle] = pages[-i]
                        self.groups[handle] = (p, parent, None)
                    if take == len(pages):
                        partial.pop(parent, None)
                    else:
                        partial[parent] = list(pages[:-take])
                if journal is not None:
                    journal.append((kind, result))
            except BaseException:
                for block in result:
                    p.indexer.pop(block.block_id, None)
                    self.groups.pop(block.block_id, None)
                for parent, pages in changed.items():
                    if pages is None:
                        partial.pop(parent, None)
                    else:
                        partial[parent] = pages
                self.ranks[rank].free(kind, new)
                raise
            self.holes[rank] += len(new) * self.layout.indexer_blocks_per_bundle - n
        return result

    def release(self, kind, blocks):
        live = [b for b in blocks if not b.is_null]
        ids = {b.block_id for b in live}
        if len(ids) != len(live) or any(
            h not in self.groups or (self.groups[h][2] is None) != (kind == INDEXER)
            for h in ids
        ):
            raise ValueError("Invalid DCP block release")
        for block in live:
            p, parent, siblings = self.groups.pop(block.block_id)
            block.ref_cnt = 0
            if kind == LATENT:
                # A save tracker can still contain a released column. Freeze
                # it as null so a later save cannot read its next occupant.
                p.latent[block.block_id] = (0,) * p.world
                siblings.remove(block.block_id)
                if not siblings:
                    for rank, physical in zip(self.ranks, parent, strict=True):
                        rank.free(kind, (physical,))
            else:
                rank = p.owner
                pages = self.available[rank].setdefault(parent, [])
                pages.append(p.indexer.pop(block.block_id))
                self.holes[rank] += 1
                if len(pages) == self.layout.indexer_blocks_per_bundle:
                    del self.available[rank][parent]
                    self.holes[rank] -= len(pages)
                    self.ranks[rank].free(kind, (parent,))

    def allocate_request(self, manager, request, demand, weight, args):
        req = request.request_id
        old = self.weights.get(req)
        if req in self.journals:
            raise RuntimeError("Reentrant DCP allocation for the same request")
        owner = self.choose(req, *demand, weight=weight)
        if owner is None:
            return None
        placement = self.bind(req, owner, weight)
        if not any(demand) and old is not None:
            return manager._allocate_slot_blocks(*args)
        # Only page growth/restore needs rollback. Compact G0 can fill holes
        # inside a table as well as append to it.
        tables = []
        for m in manager.coordinator.single_type_managers:
            for name in (
                "req_to_blocks",
                "num_cached_block",
                "_compact_external_prefix_tokens",
                "_compact_external_released_blocks",
            ):
                table = getattr(m, name, None)
                if table is not None:
                    value = table.get(req)
                    length = len(value) if isinstance(value, list) else None
                    tables.append((table, value, length, m._null_block))
        journal = self.journals[req] = []
        try:
            blocks = manager._allocate_slot_blocks(*args)
        except BaseException:
            new_latent_ids = set()
            for kind, allocated in reversed(journal):
                self.release(kind, allocated)
                if kind == LATENT:
                    for block in allocated:
                        new_latent_ids.add(block.block_id)
                        del placement.latent[block.block_id]
            for table, value, length, null_block in tables:
                if value is None:
                    table.pop(req, None)
                else:
                    if length is not None:
                        del value[length:]
                        if new_latent_ids:
                            # Compact G0 can also fill nulls inside the old table.
                            for i, block in enumerate(value):
                                if block.block_id in new_latent_ids:
                                    value[i] = null_block
                    table[req] = value
            if old is None:
                self.forget(req)
            else:
                self.bind(req, placement.owner, old)
            raise
        finally:
            del self.journals[req]
        request.dcp_indexer_placement = placement
        return blocks

    def prepare_output(self, output):
        """Send physical table deltas only when the existing protocol sends blocks."""
        placements = {}

        def translate(req, ids):
            p = self.placements[req]
            latent, indexer = ids
            placements[req] = (p.owner, p.epoch, p.latent_tables(latent))
            # The connector freezes rank-local G0 at save time. Its G1 slots
            # are canonical owner pages, including on non-writer participants.
            return latent, p.indexer_table(indexer)

        for row in output.scheduled_new_reqs:
            row.block_ids = translate(row.req_id, row.block_ids)
        cached = output.scheduled_cached_reqs
        for i, ids in enumerate(cached.new_block_ids):
            if ids is not None and (
                any(ids) or cached.req_ids[i] in cached.resumed_req_ids
            ):
                cached.new_block_ids[i] = translate(cached.req_ids[i], ids)
        if placements or output.finished_req_ids or output.preempted_req_ids:
            output.dcp_indexer_placements = placements


class DcpLogicalPool:
    """Block-pool surface used by the two MLA managers with prefix caching off."""

    dsa_shared_pool = True
    enable_caching = False

    def __init__(self, allocator, owner):
        self.allocator = allocator
        self.layout = allocator.layout
        self.owner = owner
        self.blocks_per_bundle = (
            self.layout.blocks_per_bundle(owner) if owner == LATENT else 1
        )
        self.null_block = KVCacheBlock(0, is_null=True)
        self.num_gpu_blocks = self.layout.row_count(owner)

    def get_request_blocks(self, req, n):
        return self.allocator.allocate(req, self.owner, n)

    def free_blocks(self, blocks):
        self.allocator.release(self.owner, blocks)

    def get_num_free_blocks(self):
        if self.owner == LATENT:
            return self.allocator.free_bundle_count * self.blocks_per_bundle
        return max(
            r.free_bundle_count * self.layout.indexer_blocks_per_bundle + h
            for r, h in zip(self.allocator.ranks, self.allocator.holes, strict=True)
        )

    def get_usage(self):
        return 1.0 - self.allocator.free_bundle_count / self.layout.capacity_bundles

    def take_events(self):
        return []

    def evict_blocks(self, block_ids):
        # No prefix hashes exist. The managers own release on abort/preemption.
        pass

    def reset_prefix_cache(self):
        return True
