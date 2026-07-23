# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from math import lcm

from vllm.logger import init_logger
from vllm.v1.core.block_pool import BlockPool, DSASharedLogicalBlockPool
from vllm.v1.core.dsa_shared_pool import (
    DSASharedBlockLayout,
    DSASharedBlockOwner,
    DSASharedBundleAllocator,
    dsa_block_pool_index,
    dsa_scratch_blocks_for_topk,
)
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashList,
    BlockHashListWithBlockSize,
    KVCacheBlock,
)
from vllm.v1.core.single_type_kv_cache_manager import (
    CrossAttentionManager,
    DSALatentManager,
    SingleTypeKVCacheManager,
    get_manager_for_kv_cache_spec,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    MLAAttentionSpec,
    dsa_shared_pool_enabled,
    dsa_two_groups_enabled,
)
from vllm.v1.request import Request

logger = init_logger(__name__)

# Throttle for the [KVSTARVE] admission-failure diagnostic (seconds).
_last_starve_log = [0.0]


def _get_dsa_pool_log_interval() -> float:
    raw = os.getenv("VLLM_ASCEND_DSA_POOL_LOG_INTERVAL", "5")
    try:
        return max(float(raw), 0.0)
    except ValueError:
        logger.warning(
            "Invalid VLLM_ASCEND_DSA_POOL_LOG_INTERVAL=%r; using 5 seconds.",
            raw,
        )
        return 5.0


class KVCacheCoordinator(ABC):
    """
    Coordinate the KV cache of different KV cache groups.
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        use_eagle: bool,
        enable_caching: bool,
        enable_kv_cache_events: bool,
        dcp_world_size: int,
        pcp_world_size: int,
        hash_block_size: int,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ):
        self.kv_cache_config = kv_cache_config
        self.max_model_len = max_model_len
        self.enable_caching = enable_caching
        self._dsa_pool_last_log = 0.0
        self._dsa_pool_log_interval = _get_dsa_pool_log_interval()
        self._dsa_last_starve: tuple[str, int, int, list[int]] | None = None

        # DSA two-group mode has two variants:
        # 1. legacy: one real BlockPool per group;
        # 2. shared bundle: two logical pools over one physical bundle pool.
        # Default mode: one normal pool shared by all groups.
        group_page_sizes = {
            g.kv_cache_spec.page_size_bytes for g in kv_cache_config.kv_cache_groups
        }
        self.use_dsa_shared_block_pool = (
            dsa_shared_pool_enabled()
            and dsa_two_groups_enabled()
            and len(kv_cache_config.kv_cache_groups) == 2
            and len(group_page_sizes) > 1
        )
        self.use_per_group_block_pools = (
            dsa_two_groups_enabled()
            and not self.use_dsa_shared_block_pool
            and len(kv_cache_config.kv_cache_groups) > 1
            and len(group_page_sizes) > 1
        )
        assert not (
            (self.use_per_group_block_pools or self.use_dsa_shared_block_pool)
            and enable_caching
        ), (
            "DSA two-group mode does not support prefix "
            "caching yet; launch with --no-enable-prefix-caching."
        )
        if self.use_dsa_shared_block_pool:
            latent_group = max(
                kv_cache_config.kv_cache_groups,
                key=lambda g: g.kv_cache_spec.page_size_bytes,
            )
            indexer_group = min(
                kv_cache_config.kv_cache_groups,
                key=lambda g: g.kv_cache_spec.page_size_bytes,
            )
            assert isinstance(latent_group.kv_cache_spec, MLAAttentionSpec)
            assert isinstance(indexer_group.kv_cache_spec, MLAAttentionSpec)
            layout = DSASharedBlockLayout(
                latent_page_size_bytes=latent_group.kv_cache_spec.page_size_bytes,
                indexer_page_size_bytes=indexer_group.kv_cache_spec.page_size_bytes,
                capacity_bundles=kv_cache_config.num_blocks,
            )
            self.dsa_shared_allocator = DSASharedBundleAllocator(layout)
            self.dsa_shared_num_layer_pairs = len(latent_group.layer_names)
            self.block_pools = []
            for group in kv_cache_config.kv_cache_groups:
                owner = (
                    DSASharedBlockOwner.LATENT
                    if group.kv_cache_spec.page_size_bytes
                    == latent_group.kv_cache_spec.page_size_bytes
                    else DSASharedBlockOwner.INDEXER
                )
                self.block_pools.append(
                    DSASharedLogicalBlockPool(self.dsa_shared_allocator, owner)
                )
            logger.info(
                "DSA shared KV bundle pool: %d bundles, latent=%d blocks/bundle, "
                "indexer=%d blocks/bundle.",
                kv_cache_config.num_blocks,
                layout.latent_blocks_per_bundle,
                layout.indexer_blocks_per_bundle,
            )
        else:
            num_pools = (
                len(kv_cache_config.kv_cache_groups)
                if self.use_per_group_block_pools
                else 1
            )
            pool_sizes = (
                kv_cache_config.num_blocks_per_group
                if self.use_per_group_block_pools
                and kv_cache_config.num_blocks_per_group is not None
                else [kv_cache_config.num_blocks] * num_pools
            )
            self.block_pools = [
                BlockPool(
                    pool_sizes[i],
                    enable_caching,
                    hash_block_size,
                    enable_kv_cache_events,
                    metrics_collector,
                )
                for i in range(num_pools)
            ]
            if self.use_per_group_block_pools:
                logger.info("Per-group KV block pools: %s blocks.", pool_sizes)
        self.block_pool = self.block_pools[0]

        # Needs special handling for find_longest_cache_hit if eagle is enabled
        self.use_eagle = use_eagle
        self.single_type_managers = tuple(
            get_manager_for_kv_cache_spec(
                kv_cache_spec=kv_cache_group.kv_cache_spec,
                block_pool=self.block_pools[
                    dsa_block_pool_index(
                        i,
                        use_per_group_block_pools=self.use_per_group_block_pools,
                        use_dsa_shared_block_pool=self.use_dsa_shared_block_pool,
                    )
                ],
                enable_caching=enable_caching,
                kv_cache_group_id=i,
                dcp_world_size=dcp_world_size,
                pcp_world_size=pcp_world_size,
            )
            for i, kv_cache_group in enumerate(self.kv_cache_config.kv_cache_groups)
        )
        if (
            self.use_per_group_block_pools or self.use_dsa_shared_block_pool
        ) and self.kv_cache_config.dsa_index_topk is not None:
            dsa_scratch_rows = self.kv_cache_config.dsa_num_speculative_tokens + 1
            for manager in self.single_type_managers:
                if isinstance(manager, DSALatentManager):
                    scratch_blocks = dsa_scratch_blocks_for_topk(
                        self.kv_cache_config.dsa_index_topk,
                        manager.block_size,
                        dsa_scratch_rows,
                    )
                    manager.scratch_blocks = scratch_blocks
                    logger.info(
                        "DSA latent scratch blocks: topk=%d spec_tokens=%d "
                        "rows=%d block_size=%d scratch_blocks=%d.",
                        self.kv_cache_config.dsa_index_topk,
                        self.kv_cache_config.dsa_num_speculative_tokens,
                        dsa_scratch_rows,
                        manager.block_size,
                        manager.scratch_blocks,
                    )

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: tuple[Sequence[KVCacheBlock], ...],
        num_encoder_tokens: int,
        total_computed_tokens: int,
        num_tokens_main_model: int,
        dsa_compact_external_load: bool = False,
    ) -> int:
        """
        Get the number of blocks needed to be allocated for the request.

        Args:
            request_id: The request ID.
            num_tokens: The total number of tokens that need a slot (including
                tokens that are already allocated).
            new_computed_blocks: The new computed blocks just hitting the
                prefix caching.
            num_encoder_tokens: The number of encoder tokens for allocating
                blocks for cross-attention.
            total_computed_tokens: Include both local and external tokens.
            num_tokens_main_model: The number of tokens for the main model (aka target
                model in spec decode). w/o spec decode, it is num_tokens;
                with spec decode, it is num_tokens - num_lookahead_tokens.

        Returns:
            The number of blocks to allocate.
        """
        return sum(
            self.get_num_blocks_to_allocate_per_group(
                request_id,
                num_tokens,
                new_computed_blocks,
                num_encoder_tokens,
                total_computed_tokens,
                num_tokens_main_model,
                dsa_compact_external_load,
            )
        )

    def get_num_blocks_to_allocate_per_group(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: tuple[Sequence[KVCacheBlock], ...],
        num_encoder_tokens: int,
        total_computed_tokens: int,
        num_tokens_main_model: int,
        dsa_compact_external_load: bool = False,
    ) -> list[int]:
        """Per-group variant of get_num_blocks_to_allocate (one entry per
        kv cache group / single-type manager)."""
        num_blocks_to_allocate = []
        for i, manager in enumerate(self.single_type_managers):
            if isinstance(manager, CrossAttentionManager):
                # For cross-attention, we issue a single static allocation
                # of blocks based on the number of encoder input tokens.
                num_blocks_to_allocate.append(
                    manager.get_num_blocks_to_allocate(
                        request_id, num_encoder_tokens, [], 0, num_encoder_tokens
                    )
                )
            else:
                if dsa_compact_external_load and isinstance(
                    manager, DSALatentManager
                ):
                    num_blocks_to_allocate.append(
                        manager.get_num_blocks_to_allocate_compact_external(
                            request_id,
                            num_tokens,
                            new_computed_blocks[i],
                            total_computed_tokens,
                            num_tokens_main_model,
                        )
                    )
                    continue
                num_blocks_to_allocate.append(
                    manager.get_num_blocks_to_allocate(
                        request_id,
                        num_tokens,
                        new_computed_blocks[i],
                        total_computed_tokens,
                        num_tokens_main_model,
                    )
                )
        return num_blocks_to_allocate

    def has_enough_free_blocks(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: tuple[Sequence[KVCacheBlock], ...],
        num_encoder_tokens: int,
        total_computed_tokens: int,
        num_tokens_main_model: int,
        dsa_compact_external_load: bool = False,
    ) -> bool:
        """Whether the allocation can be satisfied. With per-group block pools,
        EVERY group's demand must fit its own pool; with a single shared pool,
        the summed demand must fit that pool."""
        per_group = self.get_num_blocks_to_allocate_per_group(
            request_id,
            num_tokens,
            new_computed_blocks,
            num_encoder_tokens,
            total_computed_tokens,
            num_tokens_main_model,
            dsa_compact_external_load,
        )
        if self.use_per_group_block_pools:
            frees = [m.block_pool.get_num_free_blocks() for m in self.single_type_managers]
            ok = all(needed <= free for needed, free in zip(per_group, frees))
            if not ok and (time.monotonic() - _last_starve_log[0]) > 1.0:
                # [KVSTARVE] one-per-second snapshot of WHICH group blocked an
                # admission, with each group's demand vs free blocks. The group(s)
                # marked <<BLOCKED are the binding constraint for decode batch.
                _last_starve_log[0] = time.monotonic()
                parts = []
                for i, (needed, free, m) in enumerate(
                    zip(per_group, frees, self.single_type_managers)
                ):
                    spec = getattr(m, "kv_cache_spec", None)
                    page = getattr(spec, "page_size_bytes", "?")
                    flag = " <<BLOCKED" if needed > free else ""
                    parts.append(
                        f"g{i}({type(m).__name__},page={page}): "
                        f"need={needed} free={free}{flag}"
                    )
                logger.info(
                    "[KVSTARVE] req=%s not admitted | %s", request_id, "  ".join(parts)
                )
            return ok
        if self.use_dsa_shared_block_pool:
            needed_bundles = sum(
                manager.block_pool.get_num_bundles_to_allocate(needed)
                for manager, needed in zip(self.single_type_managers, per_group)
            )
            ok = needed_bundles <= self.dsa_shared_allocator.free_bundle_count
            if not ok:
                self._dsa_last_starve = (
                    request_id,
                    needed_bundles,
                    self.dsa_shared_allocator.free_bundle_count,
                    per_group,
                )
            if not ok and (time.monotonic() - _last_starve_log[0]) > 1.0:
                _last_starve_log[0] = time.monotonic()
                logger.info(
                    "[KVSTARVE] req=%s not admitted | need_bundles=%d free=%d "
                    "per_group_blocks=%s",
                    request_id,
                    needed_bundles,
                    self.dsa_shared_allocator.free_bundle_count,
                    per_group,
            )
            return ok
        return sum(per_group) <= self.block_pool.get_num_free_blocks()

    def maybe_log_dsa_shared_pool_usage(
        self,
        requests: dict[str, Request],
        running_count: int,
        waiting_count: int,
        max_running_count: int,
    ) -> None:
        if not self.use_dsa_shared_block_pool:
            return
        if self._dsa_pool_log_interval <= 0:
            return

        now = time.monotonic()
        if now - self._dsa_pool_last_log < self._dsa_pool_log_interval:
            return
        self._dsa_pool_last_log = now

        allocator = self.dsa_shared_allocator
        layout = allocator.layout
        capacity = layout.capacity_bundles
        free = allocator.free_bundle_count
        used = capacity - free
        owner_counts = allocator.owner_bundle_counts()
        latent_owner_bundles = owner_counts.get(DSASharedBlockOwner.LATENT, 0)
        indexer_owner_bundles = owner_counts.get(DSASharedBlockOwner.INDEXER, 0)

        latent_mgr = None
        indexer_mgr = None
        for manager in self.single_type_managers:
            owner = getattr(manager.block_pool, "owner", None)
            if owner == DSASharedBlockOwner.LATENT:
                latent_mgr = manager
            elif owner == DSASharedBlockOwner.INDEXER:
                indexer_mgr = manager

        latent_prefill_blocks = 0
        latent_prefill_bundles: set[int] = set()
        latent_postprefill_blocks = 0
        latent_postprefill_bundles: set[int] = set()
        latent_unknown_blocks = 0
        latent_unknown_bundles: set[int] = set()
        reqs_with_blocks: set[str] = set()

        if latent_mgr is not None:
            for request_id, blocks in latent_mgr.req_to_blocks.items():
                reqs_with_blocks.add(request_id)
                block_count = 0
                bundle_ids: set[int] = set()
                for block in blocks:
                    if block.is_null:
                        continue
                    block_count += 1
                    bundle_ids.add(
                        layout.bundle_id_for_block(
                            DSASharedBlockOwner.LATENT, block.block_id
                        )
                    )
                request = requests.get(request_id)
                if request is None:
                    latent_unknown_blocks += block_count
                    latent_unknown_bundles.update(bundle_ids)
                elif request.num_computed_tokens < request.num_prompt_tokens:
                    latent_prefill_blocks += block_count
                    latent_prefill_bundles.update(bundle_ids)
                else:
                    latent_postprefill_blocks += block_count
                    latent_postprefill_bundles.update(bundle_ids)

        indexer_blocks = 0
        indexer_bundles: set[int] = set()
        if indexer_mgr is not None:
            for request_id, blocks in indexer_mgr.req_to_blocks.items():
                reqs_with_blocks.add(request_id)
                for block in blocks:
                    if block.is_null:
                        continue
                    indexer_blocks += 1
                    indexer_bundles.add(
                        layout.bundle_id_for_block(
                            DSASharedBlockOwner.INDEXER, block.block_id
                        )
                    )

        bundle_bytes = (
            layout.bundle_page_size_bytes
            * getattr(self, "dsa_shared_num_layer_pairs", 1)
        )
        used_gib = used * bundle_bytes / 2**30
        total_gib = capacity * bundle_bytes / 2**30

        starve_req = "-"
        starve_need = 0
        starve_free = free
        starve_deficit = 0
        starve_per_group: list[int] = []
        if self._dsa_last_starve is not None:
            starve_req, starve_need, starve_free, starve_per_group = (
                self._dsa_last_starve
            )
            starve_deficit = max(starve_need - starve_free, 0)

        logger.info(
            "[DSA_POOL] bundles used=%d/%d free=%d usage=%.1f%% "
            "owner_latent=%d owner_indexer=%d free_ranges=%d largest_free=%d "
            "bytes_used=%.2f/%.2f GiB running=%d/%d waiting=%d reqs_with_blocks=%d "
            "latent_prefill_blocks=%d latent_prefill_bundles=%d "
            "latent_after_prefill_blocks=%d latent_after_prefill_bundles=%d "
            "latent_unknown_blocks=%d latent_unknown_bundles=%d "
            "indexer_blocks=%d indexer_bundles=%d "
            "last_starve_req=%s last_starve_need=%d "
            "last_starve_free=%d last_starve_deficit=%d "
            "last_starve_per_group_blocks=%s",
            used,
            capacity,
            free,
            100.0 * used / capacity if capacity else 0.0,
            latent_owner_bundles,
            indexer_owner_bundles,
            allocator.free_range_count,
            allocator.largest_free_range,
            used_gib,
            total_gib,
            running_count,
            max_running_count,
            waiting_count,
            len(reqs_with_blocks),
            latent_prefill_blocks,
            len(latent_prefill_bundles),
            latent_postprefill_blocks,
            len(latent_postprefill_bundles),
            latent_unknown_blocks,
            len(latent_unknown_bundles),
            indexer_blocks,
            len(indexer_bundles),
            starve_req,
            starve_need,
            starve_free,
            starve_deficit,
            starve_per_group,
        )

    def allocate_new_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: tuple[Sequence[KVCacheBlock], ...],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
        dsa_compact_external_load: bool = False,
    ) -> None:
        """
        Add the new computed blocks to the request. Optionally allocate new
            blocks for external computed tokens (if any).

        Args:
            request_id: The request ID.
            new_computed_blocks: The new computed blocks just hitting the
                prefix cache.
            num_local_computed_tokens: The number of local computed tokens.
            num_external_computed_tokens: The number of external computed tokens.
        """
        for i, manager in enumerate(self.single_type_managers):
            if dsa_compact_external_load and isinstance(manager, DSALatentManager):
                manager.allocate_new_computed_blocks_compact_external(
                    request_id,
                    new_computed_blocks[i],
                    num_local_computed_tokens,
                    num_external_computed_tokens,
                )
                continue
            manager.allocate_new_computed_blocks(
                request_id,
                new_computed_blocks[i],
                num_local_computed_tokens,
                num_external_computed_tokens,
            )
    def allocate_new_blocks(
        self,
        request_id: str,
        num_tokens: int,
        num_tokens_main_model: int,
        num_encoder_tokens: int = 0,
        dsa_compact_external_load: bool = False,
    ) -> tuple[list[KVCacheBlock], ...]:
        """
        Allocate new blocks for the request to give it at least `num_tokens`
        token slots.

        Args:
            request_id: The request ID.
            num_tokens: The total number of tokens that need a slot (including
                tokens that are already allocated).
            num_tokens_main_model: The number of tokens for the main model (aka target
                model in spec decode). w/o spec decode, it is num_tokens;
                with spec decode, it is num_tokens - num_lookahead_tokens.
            num_encoder_tokens: The number of encoder tokens for allocating
                blocks for cross-attention.

        Returns:
            The new allocated blocks.
        """
        new_blocks = []
        for manager in self.single_type_managers:
            if dsa_compact_external_load and isinstance(manager, DSALatentManager):
                blocks = manager.allocate_new_blocks_compact_external(
                    request_id, num_tokens, num_tokens_main_model
                )
            else:
                blocks = manager.allocate_new_blocks(
                    request_id,
                    num_encoder_tokens
                    if isinstance(manager, CrossAttentionManager)
                    else num_tokens,
                    num_tokens_main_model,
                )
            new_blocks.append(blocks)
        return tuple(new_blocks)

    def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:
        """
        Cache the blocks for the request.

        Args:
            request: The request.
            num_computed_tokens: The total number of tokens
                that need to be cached
                (including tokens that are already cached).
        """
        for manager in self.single_type_managers:
            manager.cache_blocks(request, num_computed_tokens)

    def free(self, request_id: str) -> None:
        """
        Free the blocks for the request.

        Args:
            request_id: The request ID.
        """
        for manager in self.single_type_managers:
            manager.free(request_id)

    def get_num_common_prefix_blocks(self, running_request_id: str) -> list[int]:
        """
        Get the number of common prefix blocks for all requests with allocated
        KV cache for each kv cache group.

        Args:
            running_request_id: The request ID of any running request, used to
                identify the common prefix blocks.

        Returns:
            list[int]: The number of common prefix blocks for each kv cache group.
        """
        return [
            manager.get_num_common_prefix_blocks(running_request_id)
            for manager in self.single_type_managers
        ]

    def remove_skipped_blocks(
        self,
        request_id: str,
        total_computed_tokens: int,
        num_prompt_tokens: int | None = None,
    ) -> None:
        """
        Remove the blocks that are no longer needed from `blocks` and replace
        the removed blocks with null_block.

        Args:
            request_id: The request ID.
            total_computed_tokens: The total number of computed tokens, including
                local computed tokens and external computed tokens.
            num_prompt_tokens: prompt length; only consumed by DSALatentManager
                (frees the prefill latent at end of prefill).
        """
        for manager in self.single_type_managers:
            if isinstance(manager, DSALatentManager):
                manager.remove_skipped_blocks(
                    request_id, total_computed_tokens, num_prompt_tokens
                )
            else:
                manager.remove_skipped_blocks(request_id, total_computed_tokens)

    def remove_saved_decode_window_blocks(
        self,
        request_id: str,
        committed_end: int,
    ) -> int:
        removed_blocks = 0
        for manager in self.single_type_managers:
            if isinstance(manager, DSALatentManager):
                removed_blocks += manager.remove_saved_decode_window_blocks(
                    request_id, committed_end
                )
        return removed_blocks

    def get_blocks(self, request_id: str) -> tuple[list[KVCacheBlock], ...]:
        """
        Get the blocks for the request.
        """
        return tuple(
            manager.req_to_blocks.get(request_id) or []
            for manager in self.single_type_managers
        )

    @abstractmethod
    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        pass

    def new_step_starts(self) -> None:
        """Called when a new step is started."""
        for manager in self.single_type_managers:
            manager.new_step_starts()


class KVCacheCoordinatorNoPrefixCache(KVCacheCoordinator):
    """
    KV cache coordinator to use if prefix caching is disabled or unsupported.
    In contrast to UnitaryKVCacheCoordinator and HybridKVCacheCoordinator,
    supports arbitrary numbers of KV cache groups (including 0 groups).
    Does not implement any features related to prefix caching.
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        use_eagle: bool,
        enable_kv_cache_events: bool,
        dcp_world_size: int,
        pcp_world_size: int,
        hash_block_size: int,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ):
        super().__init__(
            kv_cache_config,
            max_model_len,
            use_eagle,
            False,
            enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            hash_block_size=hash_block_size,
            metrics_collector=metrics_collector,
        )
        self.num_single_type_manager = len(self.single_type_managers)

    def get_num_common_prefix_blocks(self, running_request_id: str) -> list[int]:
        return [0] * self.num_single_type_manager

    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        blocks: tuple[list[KVCacheBlock], ...] = tuple(
            [] for _ in range(self.num_single_type_manager)
        )
        return blocks, 0


class UnitaryKVCacheCoordinator(KVCacheCoordinator):
    """
    KV cache coordinator for models with only one KV cache group. This is the
    case for models with only one KV cache type, e.g., all attention layers use
    full attention or all attention layers use sliding window attention.
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        use_eagle: bool,
        enable_caching: bool,
        enable_kv_cache_events: bool,
        dcp_world_size: int,
        pcp_world_size: int,
        hash_block_size: int,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ):
        super().__init__(
            kv_cache_config,
            max_model_len,
            use_eagle,
            enable_caching,
            enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            hash_block_size=hash_block_size,
            metrics_collector=metrics_collector,
        )
        self.kv_cache_spec = self.kv_cache_config.kv_cache_groups[0].kv_cache_spec
        self.block_size = self.kv_cache_spec.block_size
        self.dcp_world_size = dcp_world_size
        self.pcp_world_size = pcp_world_size
        if dcp_world_size > 1:
            self.block_size *= dcp_world_size
        if pcp_world_size > 1:
            self.block_size *= pcp_world_size
        # For models using only Mamba, block_size is set to max_model_len when
        # prefix caching is disabled, and hash_block_size validation is skipped.
        assert not enable_caching or (hash_block_size == self.block_size), (
            "UnitaryKVCacheCoordinator assumes hash_block_size == block_size"
        )
        assert len(self.kv_cache_config.kv_cache_groups) == 1, (
            "UnitaryKVCacheCoordinator assumes only one kv cache group"
        )

    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        hit_blocks = self.single_type_managers[0].find_longest_cache_hit(
            block_hashes=block_hashes,
            max_length=max_cache_hit_length,
            kv_cache_group_ids=[0],
            block_pool=self.block_pool,
            kv_cache_spec=self.kv_cache_spec,
            use_eagle=self.use_eagle,
            alignment_tokens=self.block_size,
            dcp_world_size=self.dcp_world_size,
            pcp_world_size=self.pcp_world_size,
        )
        return hit_blocks, len(hit_blocks[0]) * self.block_size


class HybridKVCacheCoordinator(KVCacheCoordinator):
    """
    KV cache coordinator for hybrid models with multiple KV cache types, and
    thus multiple kv cache groups.
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        use_eagle: bool,
        enable_caching: bool,
        enable_kv_cache_events: bool,
        dcp_world_size: int,
        pcp_world_size: int,
        hash_block_size: int,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ):
        super().__init__(
            kv_cache_config,
            max_model_len,
            use_eagle,
            enable_caching,
            enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            hash_block_size=hash_block_size,
            metrics_collector=metrics_collector,
        )
        # hash_block_size: the block size used to compute block hashes.
        # The actual block size usually equals hash_block_size, but in cases where
        # different KV cache groups have different block sizes, the actual block size
        # can be a multiple of hash_block_size.
        self.hash_block_size = hash_block_size
        assert all(
            g.kv_cache_spec.block_size % hash_block_size == 0
            for g in kv_cache_config.kv_cache_groups
        ), "block_size must be divisible by hash_block_size"
        assert dcp_world_size == 1, "DCP not support hybrid attn now."
        assert pcp_world_size == 1, "PCP not support hybrid attn now."
        self.verify_and_split_kv_cache_groups()

    def verify_and_split_kv_cache_groups(self) -> None:
        """
        Groups KV cache groups by their spec type for efficient batch processing
        during cache hit lookup.
        """
        attention_groups: list[
            tuple[KVCacheSpec, list[int], type[SingleTypeKVCacheManager]]
        ] = []

        for i, g in enumerate(self.kv_cache_config.kv_cache_groups):
            manager_cls = self.single_type_managers[i].__class__
            spec = g.kv_cache_spec

            # Try to find an existing group with the same spec
            for existing_spec, group_ids, existing_cls in attention_groups:
                if existing_spec == spec:
                    assert manager_cls is existing_cls, (
                        "Expected same manager class for identical KV cache specs."
                    )
                    group_ids.append(i)
                    break
            else:
                attention_groups.append((spec, [i], manager_cls))

        assert len(attention_groups) > 1, (
            "HybridKVCacheCoordinator requires at least two attention groups."
        )

        # Put full attention first: its efficient left-to-right scan provides
        # a tighter initial bound, reducing work for subsequent groups.
        self.attention_groups = sorted(
            attention_groups,
            key=lambda x: not isinstance(x[0], FullAttentionSpec),
        )

        # The LCM of the block sizes of all attention types.
        # The cache hit length must be a multiple of the LCM of the block sizes
        # to make sure the cache hit length is a multiple of the block size of
        # each attention type. Requiring this because we don't support partial
        # block cache hit yet.
        block_sizes = [spec.block_size for spec, _, _ in attention_groups]
        self.lcm_block_size = lcm(*block_sizes)

    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        """
        Find the longest cache hit using an iterative fixed-point algorithm.

        Each attention type either accepts the current candidate length or
        reduces it. If any type reduces the length, restart checks over all
        types. This converges because length monotonically decreases and is
        bounded below by 0.

        Args:
            block_hashes: The block hashes of the request.
            max_cache_hit_length: The maximum length of the cache hit.

        Returns:
            A tuple containing:
                - A tuple of the cache hit blocks for each single type manager.
                - The number of tokens of the longest cache hit.
        """

        def _get_block_hashes(kv_cache_spec: KVCacheSpec) -> BlockHashList:
            if kv_cache_spec.block_size == self.hash_block_size:
                return block_hashes
            return BlockHashListWithBlockSize(
                block_hashes, self.hash_block_size, kv_cache_spec.block_size
            )

        num_groups = len(self.kv_cache_config.kv_cache_groups)
        hit_length = max_cache_hit_length
        hit_blocks_by_group: list[list[KVCacheBlock] | None] = [None] * num_groups

        # Simple hybrid (1 full attn + 1 other): one iteration suffices.
        # Full attn is always first if it exists. This avoids EAGLE drops
        # being applied multiple times to non-full-attn groups.
        # FIXME (yifan): However, for complex hybrid models with multiple attn
        # groups, we still have the EAGLE spiral block dropping problem. See
        # discussion in issue https://github.com/vllm-project/vllm/issues/32802.
        is_simple_hybrid = len(self.attention_groups) == 2 and isinstance(
            self.attention_groups[0][0], FullAttentionSpec
        )

        while True:
            curr_hit_length = hit_length

            for spec, group_ids, manager_cls in self.attention_groups:
                is_full_attn = isinstance(spec, FullAttentionSpec)

                # Full attention: reuse cached blocks (downward-closed property)
                cached_blocks = hit_blocks_by_group[group_ids[0]]
                if is_full_attn and cached_blocks is not None:
                    # For full attention, we only need to compute the cache hit
                    # length once. Starting from the second iteration, if the
                    # curr_hit_length is reduced by other groups, we can simply
                    # keep the first (curr_hit_length // block_size) blocks from
                    # the last iteration.
                    num_blocks = curr_hit_length // spec.block_size
                    curr_hit_length = num_blocks * spec.block_size
                else:
                    hit_blocks = manager_cls.find_longest_cache_hit(
                        block_hashes=_get_block_hashes(spec),
                        max_length=curr_hit_length,
                        kv_cache_group_ids=group_ids,
                        block_pool=self.block_pool,
                        kv_cache_spec=spec,
                        use_eagle=self.use_eagle,
                        alignment_tokens=self.lcm_block_size,
                    )
                    curr_hit_length = len(hit_blocks[0]) * spec.block_size
                    for group_id, blocks in zip(group_ids, hit_blocks):
                        hit_blocks_by_group[group_id] = blocks

            if curr_hit_length >= hit_length:
                break
            hit_length = curr_hit_length
            # Simple hybrid: exit after one iteration
            if is_simple_hybrid:
                break

        # Truncate full attention blocks to final hit_length (if present)
        spec, group_ids, _ = self.attention_groups[0]
        if isinstance(spec, FullAttentionSpec):
            num_blocks = hit_length // spec.block_size
            for group_id in group_ids:
                if (blks := hit_blocks_by_group[group_id]) is not None:
                    del blks[num_blocks:]

        return tuple(
            blocks if blocks is not None else [] for blocks in hit_blocks_by_group
        ), hit_length


def get_kv_cache_coordinator(
    kv_cache_config: KVCacheConfig,
    max_model_len: int,
    use_eagle: bool,
    enable_caching: bool,
    enable_kv_cache_events: bool,
    dcp_world_size: int,
    pcp_world_size: int,
    hash_block_size: int,
    metrics_collector: KVCacheMetricsCollector | None = None,
) -> KVCacheCoordinator:
    if not enable_caching:
        return KVCacheCoordinatorNoPrefixCache(
            kv_cache_config,
            max_model_len,
            use_eagle,
            enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            hash_block_size=hash_block_size,
            metrics_collector=metrics_collector,
        )
    if len(kv_cache_config.kv_cache_groups) == 1:
        return UnitaryKVCacheCoordinator(
            kv_cache_config,
            max_model_len,
            use_eagle,
            enable_caching,
            enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            hash_block_size=hash_block_size,
            metrics_collector=metrics_collector,
        )
    return HybridKVCacheCoordinator(
        kv_cache_config,
        max_model_len,
        use_eagle,
        enable_caching,
        enable_kv_cache_events,
        dcp_world_size=dcp_world_size,
        pcp_world_size=pcp_world_size,
        hash_block_size=hash_block_size,
        metrics_collector=metrics_collector,
    )
