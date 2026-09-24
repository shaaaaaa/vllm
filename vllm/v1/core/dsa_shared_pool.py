# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from math import gcd
from typing import Any


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _lcm(a: int, b: int) -> int:
    return a * b // gcd(a, b)


def dsa_block_pool_index(
    group_index: int,
    *,
    use_per_group_block_pools: bool,
    use_dsa_shared_block_pool: bool,
) -> int:
    """Map a KV cache group to the BlockPool/LogicalBlockPool it should use."""

    if use_per_group_block_pools or use_dsa_shared_block_pool:
        return group_index
    return 0


def dsa_scratch_blocks_for_topk(
    index_topk: int,
    block_size: int,
    num_rows: int = 1,
) -> int:
    """Number of latent blocks needed to hold sparse decode top-k rows."""

    if index_topk <= 0:
        raise ValueError("index_topk must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if num_rows <= 0:
        raise ValueError("num_rows must be positive")
    if index_topk % block_size:
        raise ValueError(
            "DSA index_topk must be an integer multiple of block_size: "
            f"index_topk={index_topk}, block_size={block_size}. Configure "
            "index_topk to N * block_size."
        )
    return num_rows * index_topk // block_size


def paired_bank_indexer_block_map(bundle_slots: int) -> tuple[int, ...]:
    """Map canonical BF16 bank IDs to compact C8 blocks, including null banks."""
    if type(bundle_slots) is not int or bundle_slots <= 0:
        raise ValueError("paired-bank slots must be a positive integer")
    t = bundle_slots
    return tuple(
        16 * ((b // 8) % t) + 8 * (b // (8 * t)) + b % 8 for b in range(16 * t)
    ) + tuple(16 * t + 2 * (b % t) + b // t for b in range(2 * t))


class DSASharedBlockOwner(str, Enum):
    LATENT = "latent"
    INDEXER = "indexer"


@dataclass(frozen=True)
class DSASharedBlockLayout:
    """Logical block-id mapping for the DSA shared bundle pool.

    Bundle slot 0 is reserved so block table padding can continue using block
    id 0. Real allocations start at bundle id 1.
    """

    latent_page_size_bytes: int
    indexer_page_size_bytes: int
    capacity_bundles: int
    k_nope_dim: int = 512
    k_pe_dim: int = 64
    indexer_dim: int = 128
    latent_element_size: int = 2
    indexer_element_size: int = 2
    indexer_scale_page_size_bytes: int = 0
    indexer_scale_layer_count: int | None = None
    paired_indexer_banks: bool = False
    backing_alignment_bytes: int = 0

    def __post_init__(self) -> None:
        if self.paired_indexer_banks != (self.indexer_bank_count == 2):
            raise ValueError("Indexer bank policy disagrees with layout implementation")
        if self.backing_alignment_bytes < 0:
            raise ValueError("backing alignment must be nonnegative")
        if self.latent_page_size_bytes <= 0 or self.indexer_page_size_bytes <= 0:
            raise ValueError("page sizes must be positive")
        if self.capacity_bundles <= 0:
            raise ValueError("capacity_bundles must be positive")
        if (
            min(self.latent_element_size, self.indexer_element_size) <= 0
            or self.indexer_scale_page_size_bytes < 0
        ):
            raise ValueError("invalid DSA element or scale size")
        if (
            self.indexer_dim * self.latent_page_size_bytes * self.indexer_element_size
        ) % (self.indexer_page_size_bytes * self.latent_element_size):
            raise ValueError("DSA page sizes do not share integral token geometry")
        if self.paired_indexer_banks and (
            self.latent_blocks_per_bundle != 2
            or self.nope_pages_per_bundle != 8
            or self.pe_pages_per_bundle != 1
            or self.indexer_scale_page_size_bytes != 256
        ):
            raise ValueError("Paired banks require the 128-token mixed indexer layout")
        if self.latent_dim != self.k_nope_dim + self.k_pe_dim:
            raise ValueError(
                "latent_page/indexer_page ratio does not match k_nope+k_pe dims"
            )
        if self.indexer_bank_count * (
            self.nope_pages_per_bundle + self.pe_pages_per_bundle
        ) != (self.indexer_blocks_per_bundle):
            raise ValueError("DSA bundle page split is inconsistent")

    @property
    def bundle_page_size_bytes(self) -> int:
        return _lcm(self.latent_page_size_bytes, self.indexer_page_size_bytes)

    @property
    def latent_blocks_per_bundle(self) -> int:
        return self.bundle_page_size_bytes // self.latent_page_size_bytes

    @property
    def indexer_blocks_per_bundle(self) -> int:
        return self.bundle_page_size_bytes // self.indexer_page_size_bytes

    @property
    def indexer_bank_count(self) -> int:
        return 1

    @property
    def latent_dim(self) -> int:
        return (
            self.indexer_dim
            * self.latent_page_size_bytes
            * self.indexer_element_size
            // (self.indexer_page_size_bytes * self.latent_element_size)
        )

    @property
    def slot_count(self) -> int:
        return self.capacity_bundles + 1

    @property
    def nope_pages_per_bundle(self) -> int:
        return (
            self.latent_blocks_per_bundle
            * self.k_nope_dim
            * self.latent_element_size
            // (self.indexer_dim * self.indexer_element_size)
        )

    @property
    def pe_pages_per_bundle(self) -> int:
        return (
            self.latent_blocks_per_bundle
            * self.k_pe_dim
            * self.latent_element_size
            // (self.indexer_dim * self.indexer_element_size)
        )

    @property
    def scale_bytes_per_bundle(self) -> int:
        return self.indexer_blocks_per_bundle * self.indexer_scale_page_size_bytes

    def allocation_bytes_per_bundle(
        self, latent_layers: int, indexer_layers: int
    ) -> int:
        """Charge key/latent slabs and the permanently allocated scale sidecars."""
        if not 0 < indexer_layers <= latent_layers:
            raise ValueError("DSA layer counts must satisfy 0 < indexer <= latent")
        scale_layers = self.indexer_scale_layer_count
        if scale_layers is None:
            scale_layers = indexer_layers
        if self.paired_indexer_banks and not 0 < scale_layers < indexer_layers:
            raise ValueError("Paired banks require both C8 and BF16 indexer owners")
        if not 0 <= scale_layers <= indexer_layers:
            raise ValueError("Indexer scale layer count exceeds physical owners")
        return (
            latent_layers
            + (indexer_layers - scale_layers if self.paired_indexer_banks else 0)
        ) * self.bundle_page_size_bytes + scale_layers * self.scale_bytes_per_bundle

    def allocation_overhead_bytes(self, latent_layers: int, indexer_layers: int) -> int:
        """Alignment retained by paired raw slabs and unpaired latent planes."""
        return (2 * latent_layers - indexer_layers) * self.backing_alignment_bytes

    def blocks_per_bundle(self, owner: DSASharedBlockOwner) -> int:
        if owner == DSASharedBlockOwner.LATENT:
            return self.latent_blocks_per_bundle
        if owner == DSASharedBlockOwner.INDEXER:
            return self.indexer_blocks_per_bundle
        raise AssertionError(f"Unexpected DSA owner: {owner}")

    def row_count(self, owner: DSASharedBlockOwner) -> int:
        return self.slot_count * self.blocks_per_bundle(owner)

    def block_ids_for_bundle(
        self,
        owner: DSASharedBlockOwner,
        bundle_id: int,
    ) -> tuple[int, ...]:
        if bundle_id <= 0 or bundle_id > self.capacity_bundles:
            raise ValueError(f"invalid bundle id {bundle_id}")
        if owner == DSASharedBlockOwner.LATENT:
            start = bundle_id * self.latent_blocks_per_bundle
            return tuple(start + i for i in range(self.latent_blocks_per_bundle))

        if owner == DSASharedBlockOwner.INDEXER:
            nope_start = bundle_id * self.nope_pages_per_bundle
            pe_start = (
                self.slot_count * self.nope_pages_per_bundle
                + bundle_id * self.pe_pages_per_bundle
            )
            return tuple(
                list(range(nope_start, nope_start + self.nope_pages_per_bundle))
                + list(range(pe_start, pe_start + self.pe_pages_per_bundle))
            )
        raise AssertionError(f"Unexpected DSA owner: {owner}")

    def bundle_id_for_block(
        self,
        owner: DSASharedBlockOwner,
        block_id: int,
    ) -> int:
        if block_id <= 0:
            raise ValueError(f"block id {block_id} is not a real DSA block")
        if owner == DSASharedBlockOwner.LATENT:
            return block_id // self.latent_blocks_per_bundle

        if owner == DSASharedBlockOwner.INDEXER:
            nope_slab_pages = self.slot_count * self.nope_pages_per_bundle
            if block_id < nope_slab_pages:
                return block_id // self.nope_pages_per_bundle
            return (block_id - nope_slab_pages) // self.pe_pages_per_bundle
        raise AssertionError(f"Unexpected DSA owner: {owner}")


class DSAPairedIndexerBlockLayout(DSASharedBlockLayout):
    """Mixed-precision bundle; keep the legacy allocation hot path unchanged."""

    @property
    def indexer_bank_count(self) -> int:
        return 2

    @property
    def indexer_blocks_per_bundle(self) -> int:
        return (
            self.indexer_bank_count
            * self.bundle_page_size_bytes
            // self.indexer_page_size_bytes
        )

    def block_ids_for_bundle(
        self,
        owner: DSASharedBlockOwner,
        bundle_id: int,
    ) -> tuple[int, ...]:
        if bundle_id <= 0 or bundle_id > self.capacity_bundles:
            raise ValueError(f"invalid bundle id {bundle_id}")
        if owner == DSASharedBlockOwner.LATENT:
            start = bundle_id * self.latent_blocks_per_bundle
            return tuple(start + i for i in range(self.latent_blocks_per_bundle))

        if owner == DSASharedBlockOwner.INDEXER:
            banks = tuple(
                bundle_id + bank * self.slot_count
                for bank in range(self.indexer_bank_count)
            )
            pe_base = (
                self.slot_count * self.indexer_bank_count * self.nope_pages_per_bundle
            )
            return tuple(
                block
                for base, width in (
                    (0, self.nope_pages_per_bundle),
                    (pe_base, self.pe_pages_per_bundle),
                )
                for bank in banks
                for block in range(base + bank * width, base + (bank + 1) * width)
            )
        raise AssertionError(f"Unexpected DSA owner: {owner}")

    def bundle_id_for_block(
        self,
        owner: DSASharedBlockOwner,
        block_id: int,
    ) -> int:
        if block_id <= 0:
            raise ValueError(f"block id {block_id} is not a real DSA block")
        if owner == DSASharedBlockOwner.LATENT:
            return block_id // self.latent_blocks_per_bundle

        if owner == DSASharedBlockOwner.INDEXER:
            nope_slab_pages = (
                self.slot_count * self.indexer_bank_count * self.nope_pages_per_bundle
            )
            if block_id < nope_slab_pages:
                return (block_id // self.nope_pages_per_bundle) % self.slot_count
            return (
                (block_id - nope_slab_pages) // self.pe_pages_per_bundle
            ) % self.slot_count
        raise AssertionError(f"Unexpected DSA owner: {owner}")


def dsa_shared_block_layout(
    latent_spec: Any,
    indexer_spec: Any,
    capacity_bundles: int = 1,
) -> DSASharedBlockLayout:
    """Resolve shared key geometry separately from indexer scale storage."""
    scale_bytes = int(getattr(indexer_spec, "indexer_scale_page_size_bytes", 0))
    latent_dims = getattr(latent_spec, "sparse_head_dim", None) or (512, 64)
    indexer_dims = getattr(indexer_spec, "sparse_head_dim", None) or (128,)
    layout_type = (
        DSAPairedIndexerBlockLayout
        if getattr(indexer_spec, "indexer_paired_banks", False)
        else DSASharedBlockLayout
    )
    return layout_type(
        latent_page_size_bytes=latent_spec.page_size_bytes,
        indexer_page_size_bytes=indexer_spec.page_size_bytes - scale_bytes,
        capacity_bundles=capacity_bundles,
        k_nope_dim=latent_dims[0],
        k_pe_dim=latent_dims[1],
        indexer_dim=indexer_dims[-1],
        latent_element_size=latent_spec.dtype.itemsize,
        indexer_element_size=(
            getattr(
                indexer_spec, "shared_indexer_key_dtype", indexer_spec.c8_k_cache_dtype
            )
            if scale_bytes
            else indexer_spec.dtype
        ).itemsize,
        indexer_scale_page_size_bytes=scale_bytes,
        indexer_scale_layer_count=getattr(
            indexer_spec, "indexer_scale_layer_count", None
        ),
        paired_indexer_banks=bool(getattr(indexer_spec, "indexer_paired_banks", False)),
        backing_alignment_bytes=int(
            getattr(indexer_spec, "shared_pool_alignment_bytes", 0)
        ),
    )


class DSASharedBundleAllocator:
    """Physical bundle allocator shared by DSA latent and indexer wrappers."""

    def __init__(self, layout: DSASharedBlockLayout) -> None:
        self.layout = layout
        self._owners: dict[int, DSASharedBlockOwner] = {}
        self._free_ranges: list[tuple[int, int]] = [
            (1, layout.capacity_bundles)
        ]

    @property
    def free_bundle_count(self) -> int:
        return sum(end - start + 1 for start, end in self._free_ranges)

    @property
    def free_range_count(self) -> int:
        return len(self._free_ranges)

    @property
    def largest_free_range(self) -> int:
        if not self._free_ranges:
            return 0
        return max(end - start + 1 for start, end in self._free_ranges)

    def owner_bundle_counts(self) -> Counter[DSASharedBlockOwner]:
        return Counter(self._owners.values())

    def bundle_count_for_blocks(
        self,
        owner: DSASharedBlockOwner,
        num_blocks: int,
    ) -> int:
        return _cdiv(num_blocks, self.layout.blocks_per_bundle(owner))

    def allocate(
        self,
        owner: DSASharedBlockOwner,
        num_bundles: int,
    ) -> tuple[int, ...]:
        if num_bundles <= 0:
            return ()
        if num_bundles > self.free_bundle_count:
            raise ValueError(
                f"Cannot get {num_bundles} DSA bundles from the shared pool"
            )
        chosen = self._take_from_best_contiguous_range(num_bundles)
        if chosen is None:
            chosen_list: list[int] = []
            while len(chosen_list) < num_bundles:
                need = num_bundles - len(chosen_list)
                start, end = self._free_ranges.pop(0)
                take = min(need, end - start + 1)
                chosen_list.extend(range(start, start + take))
                if start + take <= end:
                    self._free_ranges.insert(0, (start + take, end))
            chosen = tuple(chosen_list)
        for bundle_id in chosen:
            self._owners[bundle_id] = owner
        return chosen

    def free(
        self,
        owner: DSASharedBlockOwner,
        bundle_ids: Iterable[int],
    ) -> None:
        ids = tuple(bundle_ids)
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate DSA bundle ids in free: {ids}")
        for bundle_id in ids:
            actual = self._owners.get(bundle_id)
            if actual != owner:
                raise ValueError(
                    f"DSA bundle {bundle_id} is owned by {actual}, not {owner}"
                )
        for bundle_id in ids:
            del self._owners[bundle_id]
            self._insert_free_bundle(bundle_id)

    def _take_from_best_contiguous_range(
        self,
        num_bundles: int,
    ) -> tuple[int, ...] | None:
        best_idx: int | None = None
        best_len: int | None = None
        for idx, (start, end) in enumerate(self._free_ranges):
            length = end - start + 1
            if length >= num_bundles and (best_len is None or length < best_len):
                best_idx = idx
                best_len = length
        if best_idx is None:
            return None

        start, end = self._free_ranges[best_idx]
        chosen = tuple(range(start, start + num_bundles))
        if start + num_bundles <= end:
            self._free_ranges[best_idx] = (start + num_bundles, end)
        else:
            del self._free_ranges[best_idx]
        return chosen

    def _insert_free_bundle(self, bundle_id: int) -> None:
        self._free_ranges.append((bundle_id, bundle_id))
        self._free_ranges.sort()
        merged: list[tuple[int, int]] = []
        for start, end in self._free_ranges:
            if not merged or start > merged[-1][1] + 1:
                merged.append((start, end))
            else:
                prev_start, prev_end = merged[-1]
                merged[-1] = (prev_start, max(prev_end, end))
        self._free_ranges = merged
