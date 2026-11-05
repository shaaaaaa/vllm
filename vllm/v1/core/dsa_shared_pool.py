# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from math import gcd
from types import MappingProxyType
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
    PREFILL_RESERVED = "prefill_reserved"


class DSABlockAllocationMode(str, Enum):
    FULL_PARENT = "full_parent"
    PREFILL_CHILD = "prefill_child"


# Preserve the legacy fallback when the P-node feature is disabled.
LAYERWISE_PREFILL_BANK_COUNT = 2
LAYERWISE_PREFILL_BUNDLE_MULTIPLIER = 2


def layerwise_prefill_bank_count() -> int:
    """Return the P-node rotation depth, ignoring the setting on other nodes.

    Configure VLLM_ASCEND_LAYERWISE_PREFILL_NUM_BANKS before startup.
    The P-node default is four; valid values are positive integers.
    """
    if layerwise_prefill_bundle_multiplier() == 1:
        return LAYERWISE_PREFILL_BANK_COUNT
    name = "VLLM_ASCEND_LAYERWISE_PREFILL_NUM_BANKS"
    raw = os.environ.get(name, "4")
    try:
        count = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}") from None
    if count <= 0:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}")
    return count


def layerwise_prefill_bundle_multiplier() -> int:
    """Select the physical layout for the explicit P-node feature path.

    ``VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE`` is the cross-repository feature
    gate.  The cache layout must use the same gate as the scheduler and the
    LMCache connector; consulting a second, optional low-level DMA variable
    could allocate a legacy-sized slab for a P-node request.
    """
    raw = os.environ.get("VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE", "false")
    enabled = raw.strip().lower()
    if enabled not in {"true", "false"}:
        raise ValueError(
            "VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE must be 'true' or 'false', "
            f"got {raw!r}"
        )
    return LAYERWISE_PREFILL_BUNDLE_MULTIPLIER if enabled == "true" else 1


@dataclass(frozen=True)
class DSASharedBlockLayout:
    """Logical block-id mapping for the DSA shared bundle pool.

    Bundle slot 0 is reserved so block table padding can continue using block
    id 0. Real allocations start at bundle id 1.
    """

    latent_page_size_bytes: int
    indexer_page_size_bytes: int
    capacity_bundles: int
    bundle_multiplier: int = field(default=1, kw_only=True)
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
        if self.bundle_multiplier <= 0:
            raise ValueError("bundle_multiplier must be positive")
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
        return self.bundle_multiplier * _lcm(
            self.latent_page_size_bytes, self.indexer_page_size_bytes
        )

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
    *,
    prefill_child: bool = False,
) -> DSASharedBlockLayout:
    """Resolve shared key geometry separately from indexer scale storage."""
    scale_bytes = int(getattr(indexer_spec, "indexer_scale_page_size_bytes", 0))
    latent_dims = getattr(latent_spec, "sparse_head_dim", None) or (512, 64)
    indexer_dims = getattr(indexer_spec, "sparse_head_dim", None) or (128,)
    if prefill_child:
        # Temporal P banks keep canonical floating-point block IDs. C8 keys
        # occupy every other physical block, with scales for the full extent.
        key_size = indexer_spec.c8_k_cache_dtype.itemsize if scale_bytes else 1
        logical_size = indexer_spec.dtype.itemsize
        if scale_bytes and (logical_size != 2 or key_size != 1):
            raise ValueError("Prefill C8 requires BF16/FP16 logical and INT8 keys")
        logical_page = indexer_spec.block_size * indexer_dims[-1] * logical_size
        return DSASharedBlockLayout(
            latent_page_size_bytes=latent_spec.page_size_bytes,
            indexer_page_size_bytes=logical_page,
            capacity_bundles=capacity_bundles,
            bundle_multiplier=LAYERWISE_PREFILL_BUNDLE_MULTIPLIER,
            k_nope_dim=latent_dims[0],
            k_pe_dim=latent_dims[1],
            indexer_dim=indexer_dims[-1],
            latent_element_size=latent_spec.dtype.itemsize,
            indexer_element_size=logical_size,
            indexer_scale_page_size_bytes=(
                indexer_spec.block_size * 2 * (logical_size // key_size)
                if scale_bytes
                else 0
            ),
            indexer_scale_layer_count=1 if scale_bytes else 0,
            backing_alignment_bytes=int(
                getattr(indexer_spec, "shared_pool_alignment_bytes", 0)
            ),
        )
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
        self._free_ranges: list[tuple[int, int]] = [(1, layout.capacity_bundles)]

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


class PrefillLayerBundlePool:
    """Layer-agnostic child allocator backed by reserved DSA parents.

    A legacy parent represents the same bundle position in every local DSA
    layer.  Reserving one parent therefore materializes ``num_physical_slots``
    independently allocatable child bundles in the global slab.  Child bundle
    zero stays reserved for block-table padding.

    The pool exposes the subset of :class:`DSASharedBundleAllocator` used by
    ``DSASharedLogicalBlockPool``.  ``free_bundle_count`` includes children
    that can be materialized from currently free parents, so scheduler
    admission does not depend on eager parent reservation.
    """

    def __init__(
        self,
        parent_allocator: DSASharedBundleAllocator,
        num_physical_slots: int,
        *,
        bank_count: int | None = None,
    ) -> None:
        if bank_count is None:
            bank_count = layerwise_prefill_bank_count()
        if num_physical_slots <= 0:
            raise ValueError("num_physical_slots must be positive")
        if bank_count <= 0:
            raise ValueError("bank_count must be positive")
        self.parent_allocator = parent_allocator
        self.num_physical_slots = num_physical_slots
        self.bank_count = bank_count
        parent_layout = parent_allocator.layout
        self.parent_capacity = parent_layout.capacity_bundles
        self.layout = replace(
            parent_layout,
            capacity_bundles=self.parent_capacity * num_physical_slots,
        )
        self._owners: dict[int, DSASharedBlockOwner] = {}
        self._reserved_parents: set[int] = set()

    def child_bundle_id(self, physical_slot: int, parent_bundle_id: int) -> int:
        if physical_slot < 0 or physical_slot >= self.num_physical_slots:
            raise ValueError(f"invalid physical slot {physical_slot}")
        if parent_bundle_id <= 0 or parent_bundle_id > self.parent_capacity:
            raise ValueError(f"invalid parent bundle id {parent_bundle_id}")
        return physical_slot * self.parent_capacity + parent_bundle_id

    def parent_bundle_id(self, child_bundle_id: int) -> int:
        if child_bundle_id <= 0 or child_bundle_id > self.layout.capacity_bundles:
            raise ValueError(f"invalid child bundle id {child_bundle_id}")
        return (child_bundle_id - 1) % self.parent_capacity + 1

    def physical_slot(self, child_bundle_id: int) -> int:
        if child_bundle_id <= 0 or child_bundle_id > self.layout.capacity_bundles:
            raise ValueError(f"invalid child bundle id {child_bundle_id}")
        return (child_bundle_id - 1) // self.parent_capacity

    @property
    def reserved_parent_count(self) -> int:
        return len(self._reserved_parents)

    @property
    def materialized_free_bundle_count(self) -> int:
        return self.reserved_parent_count * self.num_physical_slots - len(self._owners)

    @property
    def free_bundle_count(self) -> int:
        return self.materialized_free_bundle_count + (
            self.parent_allocator.free_bundle_count * self.num_physical_slots
        )

    @property
    def free_range_count(self) -> int:
        # Child allocation is parent-packed rather than range based.  Expose a
        # stable metric-compatible value for existing shared-pool logging.
        return self.reserved_parent_count

    @property
    def largest_free_range(self) -> int:
        if not self.free_bundle_count:
            return 0
        return max(
            self.num_physical_slots,
            self.materialized_free_bundle_count,
        )

    def owner_bundle_counts(self) -> Counter[DSASharedBlockOwner]:
        return Counter(self._owners.values())

    def logical_bundle_count_for_blocks(
        self,
        owner: DSASharedBlockOwner,
        num_blocks: int,
    ) -> int:
        return _cdiv(num_blocks, self.layout.blocks_per_bundle(owner))

    def bundle_count_for_blocks(
        self,
        owner: DSASharedBlockOwner,
        num_blocks: int,
    ) -> int:
        """Return physical child bundles charged for a logical allocation."""
        return self.bank_count * self.logical_bundle_count_for_blocks(owner, num_blocks)

    def allocate_banks(
        self,
        owner: DSASharedBlockOwner,
        num_logical_bundles: int,
    ) -> tuple[tuple[int, ...], ...]:
        if owner == DSASharedBlockOwner.PREFILL_RESERVED:
            raise ValueError("PREFILL_RESERVED is not a child owner")
        if num_logical_bundles <= 0:
            return tuple(() for _ in range(self.bank_count))
        child_ids = self.allocate(owner, num_logical_bundles * self.bank_count)
        return tuple(
            child_ids[bank * num_logical_bundles : (bank + 1) * num_logical_bundles]
            for bank in range(self.bank_count)
        )

    def allocate(
        self,
        owner: DSASharedBlockOwner,
        num_bundles: int,
    ) -> tuple[int, ...]:
        if owner == DSASharedBlockOwner.PREFILL_RESERVED:
            raise ValueError("PREFILL_RESERVED is not a child owner")
        if num_bundles <= 0:
            return ()
        if num_bundles > self.free_bundle_count:
            raise ValueError(
                f"Cannot get {num_bundles} DSA child bundles from the "
                "layerwise prefill pool"
            )

        free_children = self._materialized_free_children()
        missing = num_bundles - len(free_children)
        if missing > 0:
            parents_needed = _cdiv(missing, self.num_physical_slots)
            parent_ids = self.parent_allocator.allocate(
                DSASharedBlockOwner.PREFILL_RESERVED,
                parents_needed,
            )
            self._reserved_parents.update(parent_ids)
            free_children = self._materialized_free_children()

        chosen = tuple(free_children[:num_bundles])
        if len(chosen) != num_bundles:
            raise AssertionError(
                "materialized DSA child capacity disagrees with admission"
            )
        for child_id in chosen:
            self._owners[child_id] = owner
        return chosen

    def free(
        self,
        owner: DSASharedBlockOwner,
        bundle_ids: Iterable[int],
    ) -> None:
        ids = tuple(bundle_ids)
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate DSA child bundle ids in free: {ids}")
        for child_id in ids:
            actual = self._owners.get(child_id)
            if actual != owner:
                raise ValueError(
                    f"DSA child bundle {child_id} is owned by {actual}, not {owner}"
                )
        affected_parents = {self.parent_bundle_id(child_id) for child_id in ids}
        for child_id in ids:
            del self._owners[child_id]

        releasable = tuple(
            parent_id
            for parent_id in sorted(affected_parents)
            if not self._parent_has_live_children(parent_id)
        )
        if releasable:
            self.parent_allocator.free(
                DSASharedBlockOwner.PREFILL_RESERVED,
                releasable,
            )
            self._reserved_parents.difference_update(releasable)

    def _materialized_free_children(self) -> list[int]:
        # Sort by parent first so allocations fill an existing reservation
        # before spreading into another parent.
        return [
            self.child_bundle_id(slot, parent_id)
            for parent_id in sorted(self._reserved_parents)
            for slot in range(self.num_physical_slots)
            if self.child_bundle_id(slot, parent_id) not in self._owners
        ]

    def _parent_has_live_children(self, parent_bundle_id: int) -> bool:
        return any(
            self.child_bundle_id(slot, parent_bundle_id) in self._owners
            for slot in range(self.num_physical_slots)
        )


@dataclass(frozen=True)
class PrefillLayerRef:
    """One DSA layer's identity across the three number spaces.

    ``kv_group`` and ``group_ordinal`` are the LMCache storage row identity,
    ``execution_index`` is the target/MTP execution order and ``bank`` is the
    rotating resident bank (``group_ordinal % bank_count``).  The two groups
    rotate independently: for GLM-5.2 layer 6 has LATENT ordinal 6 / bank 0
    and INDEXER ordinal 3 / bank 1.
    """

    layer_name: str
    kv_group: int
    group_ordinal: int
    execution_index: int
    bank: int


def build_prefill_layer_refs(
    latent_names: Sequence[str],
    indexer_names: Sequence[str],
    *,
    execution_order: Sequence[str] | None = None,
    bank_count: int | None = None,
) -> Mapping[str, PrefillLayerRef]:
    """Build the immutable per-layer bank mapping for layerwise prefill.

    The indexer layers must be exact siblings of latent layers (sharing the
    ``.self_attn`` module prefix).  Both groups get their own ordinal counter
    so an unequal shared-indexer topology (79 LATENT / 22 INDEXER) rotates the
    configured banks independently.
    """
    if bank_count is None:
        bank_count = layerwise_prefill_bank_count()
    if bank_count <= 0:
        raise ValueError("bank_count must be positive")
    if len(set(latent_names)) != len(latent_names):
        raise ValueError("latent layer names must be unique")
    if len(set(indexer_names)) != len(indexer_names):
        raise ValueError("indexer layer names must be unique")

    indexer_by_latent: dict[str, str] = {}
    latent_by_indexer: dict[str, str] = {}
    indexer_set = set(indexer_names)
    for latent_name in latent_names:
        sibling = latent_name.rsplit(".", 1)[0] + ".indexer.k_cache"
        if sibling in indexer_set:
            indexer_by_latent[latent_name] = sibling
            latent_by_indexer[sibling] = latent_name
    orphan_indexers = sorted(indexer_set - set(latent_by_indexer))
    if orphan_indexers:
        raise ValueError(
            f"indexer layers without an exact latent sibling: {orphan_indexers}"
        )

    order = list(execution_order) if execution_order is not None else list(latent_names)
    execution_index = {name: index for index, name in enumerate(order)}
    missing = [
        layer_name
        for layer_name in indexer_by_latent
        if layer_name not in execution_index
    ]
    if missing:
        raise ValueError(f"execution_order is missing latent layers: {missing}")

    refs: dict[str, PrefillLayerRef] = {}
    for ordinal, latent_name in enumerate(latent_names):
        refs[latent_name] = PrefillLayerRef(
            layer_name=latent_name,
            kv_group=0,
            group_ordinal=ordinal,
            execution_index=execution_index[latent_name],
            bank=ordinal % bank_count,
        )
    for ordinal, indexer_name in enumerate(indexer_names):
        refs[indexer_name] = PrefillLayerRef(
            layer_name=indexer_name,
            kv_group=1,
            group_ordinal=ordinal,
            execution_index=execution_index[latent_by_indexer[indexer_name]],
            bank=ordinal % bank_count,
        )
    return MappingProxyType(refs)
