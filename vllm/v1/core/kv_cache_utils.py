# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV-Cache Utilities."""

import copy
import hashlib
import os
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, replace
from functools import partial
from math import lcm
from typing import Any, NewType, TypeAlias, overload

from vllm import envs
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils.hashing import sha256_cbor, xxhash_cbor
from vllm.utils.math_utils import cdiv
from vllm.utils.mem_utils import format_gib
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    MLAAttentionSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
    dsa_shared_pool_enabled,
    dsa_shrink_stage,
    dsa_two_groups_enabled,
)
from vllm.v1.request import Request
from vllm.v1.utils import tensor_data

# BlockHash represents the hash of a single KV-cache block used for
# prefix caching.  Treating it as a distinct type from `bytes` helps
# catch accidental misuse when passing around raw byte strings.
BlockHash = NewType("BlockHash", bytes)

# `BlockHashWithGroupId` combines a `BlockHash` with its KV cache group ID.
# It is represented as raw bytes for compactness and efficiency. The helper
# functions below pack/unpack the `BlockHash` and group id into/from the key.
BlockHashWithGroupId = NewType("BlockHashWithGroupId", bytes)

# ExternalBlockHash is used for reproducible prefix-cache block hashing.
# It's a union of `bytes` and `int` to keep backward compatibility
# after we default block hashing to use sha256 bytes.
ExternalBlockHash: TypeAlias = bytes | int


def make_block_hash_with_group_id(
    block_hash: BlockHash, group_id: int
) -> BlockHashWithGroupId:
    """Pack a `BlockHash` and group id into a `BlockHashWithGroupId`.

    The group id is encoded using 4 bytes in big-endian order and appended to
    the block hash bytes.  This representation avoids creating tuples while
    still allowing us to recover both components when needed.
    """
    return BlockHashWithGroupId(block_hash + group_id.to_bytes(4, "big", signed=False))


def get_block_hash(key: BlockHashWithGroupId) -> BlockHash:
    """Extract the `BlockHash` from a `BlockHashWithGroupId`."""
    return BlockHash(key[:-4])


def get_group_id(key: BlockHashWithGroupId) -> int:
    """Extract the group id from a `BlockHashWithGroupId`."""
    return int.from_bytes(key[-4:], "big", signed=False)


def maybe_convert_block_hash(hash_bytes: BlockHash) -> ExternalBlockHash:
    if not envs.VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES:
        return hash_bytes
    return int.from_bytes(hash_bytes, byteorder="big") & ((1 << 64) - 1)


logger = init_logger(__name__)

# The hash seed for the first block of any prefix block sequence.
#
# We use a random value to avoid hash collisions or PYTHONHASHSEED environment
# variable if set such that processes can share the seed if needed. This aligns
# with the behavior of Python's hash() function, which also uses a random seed
# if PYTHONHASHSEED is not set.
#
# The function `init_none_hash` initializes this variable globally.
NONE_HASH: BlockHash
_CBOR_HASH_FUNCTIONS = frozenset({sha256_cbor, xxhash_cbor})


def init_none_hash(hash_fn: Callable[[Any], bytes]):
    global NONE_HASH

    hash_seed = os.getenv("PYTHONHASHSEED")
    if hash_seed is None and hash_fn in _CBOR_HASH_FUNCTIONS:
        logger.warning(
            "PYTHONHASHSEED is not set. This will lead to non-reproducible "
            "block-hashes when using CBOR-based hash functions such as "
            "sha256_cbor or xxhash_cbor. Consider setting PYTHONHASHSEED to a "
            "fixed value for reproducibility."
        )

    if hash_seed is None:
        NONE_HASH = BlockHash(os.urandom(32))
    else:
        NONE_HASH = BlockHash(hash_fn(hash_seed))


@dataclass(slots=True)
class KVCacheBlock:
    """KV-cache block metadata."""

    # Block ID, ranging from 0 to num_gpu_blocks - 1.
    block_id: int
    # Reference count.
    ref_cnt: int = 0
    # The hash key (block hash + group id) of the block, only available
    # when the block is full and cached.
    _block_hash: BlockHashWithGroupId | None = None

    # Used to construct a doubly linked list for free blocks.
    # These two attributes should only be manipulated by FreeKVCacheBlockQueue.
    prev_free_block: "KVCacheBlock | None" = None
    next_free_block: "KVCacheBlock | None" = None

    # Whether the block is a null block that should never be cached.
    is_null: bool = False

    @property
    def block_hash(self) -> BlockHashWithGroupId | None:
        return self._block_hash

    @block_hash.setter
    def block_hash(self, block_hash: BlockHashWithGroupId):
        assert self.block_hash is None, (
            "The block already has a hash. This should not happen."
        )
        self._block_hash = block_hash

    def reset_hash(self):
        """Reset the block hash when the block is evicted."""
        self._block_hash = None

    def __repr__(self) -> str:
        # Use block_id instead of KVCacheBlock object to avoid calling __repr__
        # on KVCacheBlock object recursively.
        prev_block_id = self.prev_free_block.block_id if self.prev_free_block else None
        next_block_id = self.next_free_block.block_id if self.next_free_block else None
        return (
            f"KVCacheBlock(block_id={self.block_id}, "
            f"ref_cnt={self.ref_cnt}, "
            f"_block_hash={self._block_hash!r}, "
            f"prev_free_block={prev_block_id}, "
            f"next_free_block={next_block_id})"
        )


class FreeKVCacheBlockQueue:
    """This class organizes a list of KVCacheBlock objects to a doubly linked
    list of free blocks. We implement this class instead of using Python
    builtin deque to support removing a block in the middle of the queue
    in O(1) time. To close the performance gap to the builtin deque which is
    implemented in C++, this class does not allocate any Python objects when
    manipulating the linked list. Instead, this class manipulates the
    prev_free_block and next_free_block attributes of the given blocks.

    The queue is ordered by block ID in the beginning. When a block is allocated
    and then freed, it will be appended back with the eviction order:
    1. The least recent used block is at the front (LRU).
    2. If two blocks have the same last accessed time (allocated by the
       same sequence), the one with more hash tokens (the tail of a block
       chain) is at the front.
    Note that we maintain this order by reversing the block order when free
    blocks of a request. This operation is outside of this class.

    Args:
        blocks: A list of KVCacheBlock objects.
    """

    def __init__(self, blocks: list[KVCacheBlock]) -> None:
        self.num_free_blocks = len(blocks)

        # Initialize doubly links of consecutive blocks
        for i in range(self.num_free_blocks):
            if i > 0:
                blocks[i].prev_free_block = blocks[i - 1]
            if i < self.num_free_blocks - 1:
                blocks[i].next_free_block = blocks[i + 1]

        # Create a fake head and a tail block for the doubly linked list to
        # reduce branching in the code
        #
        # The implementation guaranteed that the fake head and tail
        # are NEVER got popped, so we could safely assume each real blocks
        # in the queue has prev and next blocks.
        self.fake_free_list_head = KVCacheBlock(block_id=-1)
        self.fake_free_list_tail = KVCacheBlock(block_id=-1)
        if self.num_free_blocks > 0:
            # Connect fake_head and fake_tail to the first and last block
            # respectively.
            self.fake_free_list_head.next_free_block = blocks[0]
            blocks[0].prev_free_block = self.fake_free_list_head
            self.fake_free_list_tail.prev_free_block = blocks[-1]
            blocks[-1].next_free_block = self.fake_free_list_tail
        else:
            # For empty list, simply connect the fake head and tail.
            self.fake_free_list_head.next_free_block = self.fake_free_list_tail
            self.fake_free_list_tail.prev_free_block = self.fake_free_list_head

    def popleft(self) -> KVCacheBlock:
        """Pop the first free block and reduce num_free_blocks by 1.

        Returns:
            The first free block.
        """
        if (
            self.fake_free_list_head.next_free_block is self.fake_free_list_tail
            or self.fake_free_list_head.next_free_block is None
        ):
            assert self.num_free_blocks == 0, (
                f"num_free_blocks ({self.num_free_blocks}) is out of sync "
                "with the free list."
            )
            raise ValueError("No free blocks available")

        first_block: KVCacheBlock = self.fake_free_list_head.next_free_block

        if first_block.next_free_block is None:
            # This should not happen if the block is from the free list.
            # It indicates a bug in the caller's logic.
            raise RuntimeError(
                "Invalid block found in popleft() "
                "which doesn't have a valid next_free_block"
            )

        # Connect fake_head and the next block of first_block (i.e. second block
        # or fake tail).
        self.fake_free_list_head.next_free_block = first_block.next_free_block
        first_block.next_free_block.prev_free_block = self.fake_free_list_head

        # Remove the block from the linked list.
        first_block.prev_free_block = first_block.next_free_block = None

        self.num_free_blocks -= 1
        return first_block

    def popleft_n(self, n: int) -> list[KVCacheBlock]:
        """Pop the first n free blocks and reduce num_free_blocks by n.

        Args:
            n: The number of blocks to pop.

        Returns:
            A list of n free blocks.
        """
        if n == 0:
            return []
        assert self.num_free_blocks >= n
        self.num_free_blocks -= n

        curr_block = self.fake_free_list_head.next_free_block
        # Pop n blocks from the head of the list
        ret = []
        for _ in range(n):
            assert curr_block is not None
            ret.append(curr_block)
            last_block = curr_block
            curr_block = curr_block.next_free_block
            # Reset prev_free_block and next_free_block of all popped blocks
            last_block.prev_free_block = None
            last_block.next_free_block = None

        if curr_block is not None:
            # The queue is not empty, connect the fake head to
            # the new first block.
            self.fake_free_list_head.next_free_block = curr_block
            curr_block.prev_free_block = self.fake_free_list_head
        return ret

    def remove(self, block: KVCacheBlock) -> None:
        """Remove a block in the free list and reduce num_free_blocks by 1.

        Args:
            block: The block to remove.
        """
        if block.prev_free_block is None or block.next_free_block is None:
            # This should not happen if the block is from the free list.
            # It indicates a bug in the caller's logic.
            raise RuntimeError(f"remove() called on an invalid block: {block}")

        # Link the previous block to the next block.
        block.prev_free_block.next_free_block = block.next_free_block
        # Link the next block to the previous block.
        block.next_free_block.prev_free_block = block.prev_free_block

        # Remove the block from the linked list.
        block.prev_free_block = block.next_free_block = None
        self.num_free_blocks -= 1

    def append(self, block: KVCacheBlock) -> None:
        """Put a block back into the free list and increase
        num_free_blocks by 1.

        Args:
            block: The block to append.
        """
        if self.fake_free_list_tail.prev_free_block is None:
            raise RuntimeError(
                "prev_free_block of fake_free_list_tail should always exist"
            )
        last_block: KVCacheBlock = self.fake_free_list_tail.prev_free_block

        # Connect the new block after the last block.
        last_block.next_free_block = block
        block.prev_free_block = last_block

        # Connect the fake tail after the new block.
        block.next_free_block = self.fake_free_list_tail
        self.fake_free_list_tail.prev_free_block = block

        self.num_free_blocks += 1

    def append_n(self, blocks: list[KVCacheBlock]) -> None:
        """Put a list of blocks back into the free list

        Args:
            blocks: The blocks to append.
        """
        if len(blocks) == 0:
            return

        last_block = self.fake_free_list_tail.prev_free_block
        assert last_block is not None, (
            "prev_free_block of fake_free_list_tail should always exist"
        )
        # Add inter-connections between consecutive blocks
        for block in blocks:
            block.prev_free_block = last_block
            last_block.next_free_block = block
            last_block = block

        # Connect the last block of <blocks> to the fake tail
        last_block.next_free_block = self.fake_free_list_tail
        self.fake_free_list_tail.prev_free_block = last_block

        self.num_free_blocks += len(blocks)

    def get_all_free_blocks(self) -> list[KVCacheBlock]:
        """Get all free blocks in the free list. Mainly used for testing.

        Returns:
            A list of free blocks.
        """
        ret = []
        if self.fake_free_list_head.next_free_block is None:
            raise RuntimeError(
                "next_free_block of fake_free_list_head should always exist"
            )
        # Start from the first block
        curr_block: KVCacheBlock = self.fake_free_list_head.next_free_block
        # As long as next_free_block is available, we haven't reached to
        # the fake tail yet.
        while curr_block.next_free_block is not None:
            ret.append(curr_block)
            curr_block = curr_block.next_free_block
        return ret


def need_extra_keys(request: Request) -> bool:
    """Check whether the blocks allocated to this request need extra hash keys.

    Args:
        request (Request): The request.

    Returns:
        bool: Whether blocks allocated to this request need extra hash keys.
    """

    # Multimodal requests need to include the MM hash.
    # LoRA requests need to include the LoRA name.
    # Request with provided cache salt need to include the salt.
    return (
        bool(request.mm_features)
        or (request.lora_request is not None)
        or (request.cache_salt is not None)
    )


def _gen_mm_extra_hash_keys(
    request: Request, start_token_idx: int, end_token_idx: int, start_mm_idx: int
) -> tuple[list[Any], int]:
    """Generate extra keys related to MultiModal request for block hash
    computation. For multi-modal inputs, the extra keys are
    (mm_hash, start_offset) that indicate a mm input contained in the
    block and its starting offset in the block tokens.

    Args:
        request: The request object.
        start_token_idx: The start token index of the block.
        end_token_idx: The end token index of the block.
        start_mm_idx: The start multi-modal index of the block.

    Returns:
        A tuple of extra keys and the next multi-modal index.
    """
    extra_keys: list[Any] = []

    mm_features = request.mm_features
    if not mm_features:
        return extra_keys, start_mm_idx

    # Note that we assume mm_features are sorted by mm_position.offset.
    # We do not need to check all mm inputs if the start token index is out of
    # range. This usually happens in the late prefill phase and decoding phase.
    last_pos = mm_features[-1].mm_position
    if last_pos.offset + last_pos.length < start_token_idx:
        return extra_keys, start_mm_idx

    # Support start_mm_idx == -1 to indicate the last mm input.
    if start_mm_idx < 0:
        assert -start_mm_idx <= len(mm_features)
        start_mm_idx = len(mm_features) + start_mm_idx

    curr_mm_idx = start_mm_idx
    while mm_features and curr_mm_idx < len(mm_features):
        mm_feature = mm_features[curr_mm_idx]
        assert mm_feature.identifier is not None
        offset = mm_feature.mm_position.offset
        length = mm_feature.mm_position.length
        if end_token_idx > offset:
            if start_token_idx > offset + length:
                # This block has passed the current mm input.
                curr_mm_idx += 1
                continue

            # The block contains the current mm input.
            extra_keys.append(mm_feature.identifier)

            if end_token_idx >= offset + length:
                # If this block contains the end of the current mm input,
                # move to the next mm input as this block may also contain
                # the next mm input.
                curr_mm_idx += 1
            else:
                # Otherwise this block is done with mm inputs.
                break
        else:
            # This block has not reached the current mm input.
            break
    return extra_keys, curr_mm_idx


def _gen_lora_extra_hash_keys(request: Request) -> list[str]:
    """Generate extra keys related to LoRA for block hash computation.

    Args:
        request: The request object.

    Returns:
        Return LoRA name of the request if it is a LoRA request. Return empty
        list otherwise.
    """
    if not request.lora_request:
        return []
    return [request.lora_request.lora_name]


def _gen_prompt_embeds_extra_hash_keys(
    request: Request, start_token_idx: int, end_token_idx: int
) -> list[bytes]:
    """Generate extra keys related to prompt embeds for block hash computation.

    Args:
        request: The request object.
        start_token_idx: The start token index of the block.
        end_token_idx: The end token index of the block.

    Returns:
        Return a stable hash of the block prompt embeddings if prompt embeds
        are present. Return empty list otherwise.
    """
    if request.prompt_embeds is None:
        return []
    block_range = (start_token_idx, end_token_idx)
    embeds_hash = request._prompt_embeds_per_block_hashes.get(block_range)
    if embeds_hash is None:
        block_prompt_embeds = request.prompt_embeds[start_token_idx:end_token_idx]
        # Hash prompt embeds once per block and cache on request
        embeds_hash = hashlib.sha256(tensor_data(block_prompt_embeds)).digest()
        request._prompt_embeds_per_block_hashes[block_range] = embeds_hash
    return [embeds_hash]


def generate_block_hash_extra_keys(
    request: Request, start_token_idx: int, end_token_idx: int, start_mm_idx: int
) -> tuple[tuple[Any, ...] | None, int]:
    """Generate extra keys for the block hash. The extra keys can come from
    the multi-modal inputs, request specific metadata (e.g., LoRA names), and
    hashed data from prompt embeddings.

    Args:
        request: The request object.
        start_token_idx: The start token index of the block.
        end_token_idx: The end token index of the block.
        start_mm_idx: The start multi-modal index of the block.

    Returns:
        A tuple of extra keys and the next multi-modal index.
    """
    mm_extra_keys: list[Any]
    mm_extra_keys, new_start_mm_idx = _gen_mm_extra_hash_keys(
        request, start_token_idx, end_token_idx, start_mm_idx
    )
    lora_extra_keys: list[str] = _gen_lora_extra_hash_keys(request)
    cache_salt_keys: list[str] = (
        [request.cache_salt] if (start_token_idx == 0 and request.cache_salt) else []
    )
    prompt_embeds_keys = _gen_prompt_embeds_extra_hash_keys(
        request, start_token_idx, end_token_idx
    )

    extra_keys: list[Any] = (
        lora_extra_keys + mm_extra_keys + cache_salt_keys + prompt_embeds_keys
    )

    if not extra_keys:
        return None, new_start_mm_idx

    return tuple(extra_keys), new_start_mm_idx


def hash_block_tokens(
    hash_function: Callable[[Any], bytes],
    parent_block_hash: BlockHash | None,
    curr_block_token_ids: Sequence[int],
    extra_keys: tuple[Any, ...] | None = None,
) -> BlockHash:
    """Computes a hash value corresponding to the contents of a block and
    the contents of the preceding block(s). The hash value is used for
    prefix caching. We use LRU cache for this function to avoid recomputing
    hash values for the same block contents.
    Args:
        hash_function: The hash function used to compute block hash.
        parent_block_hash: The hash of the parent block. None
            if this is the first block.
        curr_block_token_ids: A list of token ids in the current
            block. The current block is assumed to be full.
        extra_keys: Extra keys for the block.
    Returns:
        The hash value of the block and the token ids in the block.
        The entire tuple is used as the hash key of the block.
    """
    if not parent_block_hash:
        parent_block_hash = NONE_HASH

    curr_block_token_ids_tuple = tuple(curr_block_token_ids)
    return BlockHash(
        hash_function((parent_block_hash, curr_block_token_ids_tuple, extra_keys))
    )


def get_request_block_hasher(
    block_size: int,
    caching_hash_fn: Callable[[Any], bytes],
) -> Callable[[Request], list[BlockHash]]:
    """
    Returns a function which computes the list of un-computed block hashes
    of a request."""

    def request_block_hasher(request: Request) -> list[BlockHash]:
        start_token_idx = len(request.block_hashes) * block_size
        num_tokens = request.num_tokens

        if start_token_idx + block_size > num_tokens:
            # Early stop when there no new full blocks created.
            return []

        curr_mm_idx = 0
        if start_token_idx > 0:
            # Set curr_mm_idx = -1 to indicate the last mm input.
            # Note that since we reach to this branch only when the block is
            # completed with generated tokens, we only need to consider the
            # last mm input.
            curr_mm_idx = -1

        prev_block_hash_value = (
            request.block_hashes[-1] if request.block_hashes else None
        )
        new_block_hashes: list[BlockHash] = []
        while True:
            end_token_idx = start_token_idx + block_size
            if end_token_idx > num_tokens:
                # We only hash full blocks
                break

            # MM and LoRA requests need extra keys for block-hash computation.
            extra_keys, curr_mm_idx = generate_block_hash_extra_keys(
                request, start_token_idx, end_token_idx, curr_mm_idx
            )

            # Compute the hash of the current block
            block_tokens = request.all_token_ids[start_token_idx:end_token_idx]
            block_hash = hash_block_tokens(
                caching_hash_fn, prev_block_hash_value, block_tokens, extra_keys
            )

            new_block_hashes.append(block_hash)
            start_token_idx += block_size
            prev_block_hash_value = block_hash

        return new_block_hashes

    return request_block_hasher


def _check_enough_kv_cache_memory(
    available_memory: int,
    get_needed_memory: Callable[[], int],
    max_model_len: int,
    estimate_max_model_len: Callable[[int], int],
):
    if available_memory <= 0:
        raise ValueError(
            "No available memory for the cache blocks. "
            "Try increasing `gpu_memory_utilization` when initializing the engine. "
            "See https://docs.vllm.ai/en/latest/configuration/conserving_memory/ "
            "for more details."
        )

    needed_memory = get_needed_memory()

    if needed_memory > available_memory:
        estimated_max_len = estimate_max_model_len(available_memory)
        estimated_msg = ""
        if estimated_max_len > 0:
            estimated_msg = (
                "Based on the available memory, "
                f"the estimated maximum model length is {estimated_max_len}. "
            )

        raise ValueError(
            f"To serve at least one request with the models's max seq len "
            f"({max_model_len}), ({format_gib(needed_memory)} GiB KV "
            f"cache is needed, which is larger than the available KV cache "
            f"memory ({format_gib(available_memory)} GiB). {estimated_msg}"
            f"Try increasing `gpu_memory_utilization` or decreasing `max_model_len` "
            f"when initializing the engine. "
            f"See https://docs.vllm.ai/en/latest/configuration/conserving_memory/ "
            f"for more details."
        )


def max_memory_usage_bytes(
    vllm_config: VllmConfig, kv_cache_specs: Iterable[KVCacheSpec]
) -> int:
    """
    Get the maximum memory usage in bytes for the given KV cache specs.
    """
    return sum(spec.max_memory_usage_bytes(vllm_config) for spec in kv_cache_specs)


def estimate_max_model_len(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
    available_memory: int,
) -> int:
    """
    Estimates the maximum model length that can fit in the available memory
    using binary search.

    This function temporarily modifies max_model_len during estimation but
    restores the original value before returning, ensuring no side effects.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_spec: The kv cache spec of each attention layer in the model
        available_memory: Memory available for KV cache in bytes.

    Returns:
        The estimated maximum model length that can fit in the available memory.
    """
    # Save the original max_model_len to restore after estimation
    original_max_model_len = vllm_config.model_config.max_model_len

    # Define a function to check if a given model length fits in memory
    def fits_in_memory(model_len: int) -> bool:
        # Temporarily modify the max_model_len for this calculation
        vllm_config.model_config.max_model_len = model_len
        # Calculate memory needed for the given model length
        memory_needed = max_memory_usage_bytes(vllm_config, kv_cache_spec.values())
        return memory_needed <= available_memory

    try:
        # Binary search for the maximum model length
        left, right = 1, original_max_model_len

        # If even the smallest model length doesn't fit, return 0
        if not fits_in_memory(left):
            return 0

        # Binary search for the maximum model length that fits
        result = 1
        while left <= right:
            mid = (left + right) // 2
            if fits_in_memory(mid):
                result = mid
                left = mid + 1
            else:
                right = mid - 1
        return result
    finally:
        # Always restore the original max_model_len to avoid side effects
        vllm_config.model_config.max_model_len = original_max_model_len


def check_enough_kv_cache_memory(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
    available_memory: int,
):
    """
    Checks whether `available_memory` is enough for the KV cache to hold at
    least one request with the model's max_model_len.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_spec: The kv cache spec of each attention layer in the model
        available_memory: Memory available for KV cache in bytes.

    Raises:
        ValueError: If there is not enough memory available for the KV cache.
    """

    # No need to check for available memory if the kv_cache_spec is empty
    if kv_cache_spec:
        _check_enough_kv_cache_memory(
            available_memory,
            lambda: max_memory_usage_bytes(vllm_config, kv_cache_spec.values()),
            vllm_config.model_config.max_model_len,
            lambda am: estimate_max_model_len(vllm_config, kv_cache_spec, am),
        )


def create_kv_cache_group_specs(
    kv_cache_spec: dict[str, KVCacheSpec], grouped_layer_names: list[list[str]]
) -> list[KVCacheGroupSpec]:
    """
    Create KVCacheGroupSpec object for each kv cache group layer.
    The layers in the same group should share the same
    KVCacheSpec.

    Args:
        kv_cache_spec:
            A mapping from each layer name to its corresponding KVCacheSpec.
        grouped_layer_names:
            A list of kv cache groups, where each element is a list of layer
            names that belong to the same group and should share the same
            KVCacheSpec.
    Returns:
        A list of KVCacheGroupSpec objects, one for each group.
    """
    kv_cache_groups = []
    for layer_names_one_group in grouped_layer_names:
        layer_specs = [
            kv_cache_spec[layer_name] for layer_name in layer_names_one_group
        ]
        merged_layer_spec = layer_specs[0].merge(layer_specs)
        kv_cache_groups.append(
            KVCacheGroupSpec(layer_names_one_group, merged_layer_spec)
        )
    return kv_cache_groups


def is_kv_cache_spec_uniform(kv_cache_spec: dict[str, KVCacheSpec]) -> bool:
    """
    Whether all layers in the given KVCacheSpec have the same KV cache spec.
    Note that we regard FullAttentionSpec with and without sliding window as
    the same type.

    Args:
        kv_cache_spec: The kv cache spec of each attention layer in the model

    Returns:
        True if all layers have the same type, False otherwise.
    """

    if not kv_cache_spec:
        # Encoder-only models do not have KV cache, kv_cache_type can be
        # regarded as uniform.
        return True
    if dsa_two_groups_enabled():
        # DSA un-bundle: MLA latent (576) and indexer (128) must not merge into
        # one spec/group — they form separate groups with per-group block pools.
        head_sizes = {
            spec.head_size
            for spec in kv_cache_spec.values()
            if isinstance(spec, MLAAttentionSpec)
        }
        if len(head_sizes) > 1:
            return False
    try:
        kv_cache_spec_values = list(kv_cache_spec.values())
        _ = kv_cache_spec_values[0].merge(kv_cache_spec_values)
    except AssertionError:
        return False
    return True


def get_max_concurrency_for_kv_cache_config(
    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig
) -> float:
    """
    Get the maximum concurrency for the given KV cache configuration.
    """
    if (
        dsa_two_groups_enabled()
        and not dsa_shared_pool_enabled()
        and len(
            {
                g.kv_cache_spec.page_size_bytes
                for g in kv_cache_config.kv_cache_groups
            }
        )
        > 1
    ):
        # Per-group block pools: each pool has num_blocks blocks, and a request
        # needs cdiv(max_model_len, block_size) blocks FROM EACH pool. Concurrency
        # is limited by the pool needing the most blocks per request.
        max_blocks_per_req = max(
            cdiv(
                vllm_config.model_config.max_model_len, g.kv_cache_spec.block_size
            )
            for g in kv_cache_config.kv_cache_groups
        )
        return kv_cache_config.num_blocks / max_blocks_per_req
    if dsa_two_groups_enabled() and dsa_shared_pool_enabled() and len(
        {g.kv_cache_spec.page_size_bytes for g in kv_cache_config.kv_cache_groups}
    ) > 1:
        latent_group = max(
            kv_cache_config.kv_cache_groups,
            key=lambda g: g.kv_cache_spec.page_size_bytes,
        )
        indexer_group = min(
            kv_cache_config.kv_cache_groups,
            key=lambda g: g.kv_cache_spec.page_size_bytes,
        )
        bundle_page = lcm(
            latent_group.kv_cache_spec.page_size_bytes,
            indexer_group.kv_cache_spec.page_size_bytes,
        )
        max_blocks_per_req = cdiv(
            vllm_config.model_config.max_model_len,
            latent_group.kv_cache_spec.block_size,
        )
        bundles_per_req = cdiv(
            max_blocks_per_req,
            bundle_page // latent_group.kv_cache_spec.page_size_bytes,
        ) + cdiv(
            max_blocks_per_req,
            bundle_page // indexer_group.kv_cache_spec.page_size_bytes,
        )
        return kv_cache_config.num_blocks / bundles_per_req

    num_layer_per_group = max(
        len(group.layer_names) for group in kv_cache_config.kv_cache_groups
    )
    max_memory_usage_per_request = num_layer_per_group * max_memory_usage_bytes(
        vllm_config, (group.kv_cache_spec for group in kv_cache_config.kv_cache_groups)
    )
    memory_per_block = (
        kv_cache_config.kv_cache_groups[0].kv_cache_spec.page_size_bytes
        * num_layer_per_group
    )
    num_block_per_request = cdiv(max_memory_usage_per_request, memory_per_block)
    max_concurrency = kv_cache_config.num_blocks / num_block_per_request
    return max_concurrency


def may_override_num_blocks(vllm_config: VllmConfig, num_blocks: int) -> int:
    """
    Override the number of kv cache blocks if `num_gpu_blocks_override` is set.
    """
    if vllm_config.cache_config.num_gpu_blocks_override is not None:
        num_gpu_blocks_override = vllm_config.cache_config.num_gpu_blocks_override
        logger.info(
            "Overriding num_gpu_blocks=%d with num_gpu_blocks_override=%d",
            num_blocks,
            num_gpu_blocks_override,
        )
        num_blocks = num_gpu_blocks_override

    return num_blocks


def get_num_blocks(
    vllm_config: VllmConfig, num_layers: int, available_memory: int, page_size: int
) -> int:
    """
    Get the number of kv cache blocks.

    Args:
        vllm_config: The global VllmConfig
        num_layers: The number of layers
        available_memory: Memory available for KV cache in bytes.
        page_size: The page size of the KV cache.
    """
    num_blocks = int(available_memory // page_size // num_layers)
    num_blocks = max(num_blocks, 0)
    num_blocks = may_override_num_blocks(vllm_config, num_blocks)
    return num_blocks


def get_uniform_page_size(kv_cache_specs: Iterable[KVCacheSpec]) -> int:
    """
    Get the page size of the KV cache.
    """
    page_sizes = {layer.page_size_bytes for layer in kv_cache_specs}
    assert len(page_sizes) == 1
    return page_sizes.pop()


def _get_kv_cache_groups_uniform_spec(
    kv_cache_specs: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    """
    Generates the KV cache configuration for a model with the same KV cache
    spec for all layers.

    Args:
        kv_cache_specs: The kv cache spec of each attention layer in the model

    Returns:
        The generated KVCacheGroupSpecs
    """

    return create_kv_cache_group_specs(kv_cache_specs, [list(kv_cache_specs.keys())])


def _get_kv_cache_groups_uniform_type(
    spec: UniformTypeKVCacheSpecs,
) -> list[KVCacheGroupSpec]:
    """
    Generates the KV cache configuration for a model with one type of KV cache
    but different hidden sizes. All layers are merged into one group.

    Args:
        spec: The UniformTypeKVCacheSpecs of the model

    Returns:
        The generated KVCacheGroupSpecs
    """

    return [KVCacheGroupSpec(list(spec.kv_cache_specs.keys()), spec)]


def is_kv_cache_page_size_uniform(kv_cache_spec: dict[str, KVCacheSpec]) -> bool:
    """
    Whether all layers in the given KVCacheSpec have the same page size.
    Args:
        kv_cache_spec: The KVCacheSpec of each attention layer in the model

    Returns:
        True if all layers have the same page size, False otherwise.
    """

    page_sizes = {layer.page_size_bytes for layer in kv_cache_spec.values()}
    return len(page_sizes) == 1


def unify_kv_cache_spec_page_size(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> dict[str, KVCacheSpec]:
    """
    Unify the page size of the given KVCacheSpec. If the page size of all layers
    are the same, return the original KVCacheSpec. If not same, unify the page
    size by increasing the block size of layers with smaller page size. Raise
    NotImplementedError if failed to unify the page size.

    Args:
        kv_cache_spec: The KVCacheSpec of each attention layer in the model

    Returns:
        The updated KVCacheSpec with the same page_size_bytes.
    """
    page_sizes = {layer.page_size_bytes for layer in kv_cache_spec.values()}
    if len(page_sizes) <= 1:
        # All layers have the same page size, no need to unify.
        return kv_cache_spec

    max_page_size = max(page_sizes)
    new_kv_cache_spec = {}
    for layer_name, layer_spec in kv_cache_spec.items():
        if layer_spec.page_size_bytes == max_page_size:
            new_kv_cache_spec[layer_name] = layer_spec
        else:
            layer_page_size = layer_spec.page_size_bytes
            if max_page_size % layer_page_size != 0:
                raise NotImplementedError(
                    "The page size of the layer is not divisible by the "
                    "maximum page size. Cannot unify by adjusting block_size."
                )
            ratio = max_page_size // layer_page_size
            new_block_size = layer_spec.block_size * ratio
            new_spec = replace(layer_spec, block_size=new_block_size)
            assert new_spec.page_size_bytes == max_page_size
            new_kv_cache_spec[layer_name] = new_spec
    return new_kv_cache_spec


def is_kv_cache_type_attention_free(kv_cache_spec: dict[str, KVCacheSpec]) -> bool:
    # kv_cache_spec is an empty dict for attention free models
    return not kv_cache_spec


def _get_kv_cache_groups_uniform_page_size(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    """
    Generates the KV cache groups for hybrid models with multiple
    attention types but still with a uniform page size (physical memory per
    block per layer) for all layers.

    Detailed explanation about kv cache management of hybrid models:
    The layers in the models are repeated with some patterns, e.g., a model
    with 10 full attention layers and 20 sliding window attention layers can be
    regarded as repeating the pattern (1 * full, 2 * sw) 10 times.
    The KVCacheManager allocates different block tables for each of the 3 layers
    in the pattern, and repeats each of them 10 times to generate the
    block_table for the 30 layers in the model.
    Therefore, we can group the layers in the model into 3 kv_cache_groups, each
    of which contains 10 layers in the model.
    The KVCacheManager allocates the block_table for each group based on its
    kv_cache spec, and the model runner applies the block table to each layer
    in the group.
    For example:
    1. A model only uses full attention. The pattern is
    (num_hidden_layers * full), so there is only one group and the block table
    is shared by all layers. It is already handled by
    `_get_kv_cache_config_uniform_type`.
    2. A model with 10 full attention layers and 20 sliding window
    attention layers. There are 3 layers in the pattern (1 * full, 2 * sw), so
    there are 3 kv_cache_groups, each of which represents 10 layers.

    To simplify the implementation, we make the following assumptions:
    1. Physical memory per block: Must be the same across all KV cache groups.
    Breaking this assumption is non-trivial due to memory fragmentation concerns
    when allocating blocks of different sizes.
    2. Tokens per block (block_size): Currently, we directly use
    `CacheConfig.block_size` for all layers. It can be extended to vary by KV
    cache group, but within each KV cache group, all layers must share the same
    block size.
    3. Physical memory per token per layer: This property is decided by model
    config. Currently we only support models that have the same physical memory
    per token per layer for all layers. Can be relaxed with a simple extension,
    but still need to keep physical memory per block the same for all groups.
    4. Number of layers per group: Currently assumed the same for all layers.
    Can be relaxed with a simple extension, but still need to keep physical
    memory per block the same for all groups.
    5. Attention type within groups: All layers in a group must share the same
    attention type. One exception is that, when
    `--disable-hybrid-kv-cache-manager` is true, the single group for full
    attention layers may also include attention layers using sliding window or
    LLaMA 4 local attention. See `unify_hybrid_kv_cache_specs` for more details.
    6. Support for multiple attention types: The design for most components is
    general to an arbitrary number of attention types. But
    `find_longest_cache_hit` only supports one attention type or two
    types of full-attention plus exactly one another type. The general
    implementation of this function is feasible but we don't know how to
    implement it cleanly yet.

    As we assume tokens per block, physical memory per token per layer, and
    number of layers per group are the same now, we can ensure that physical
    memory per block is the same for all groups.

    Args:
        kv_cache_spec: The KVCacheSpec of each attention layer in the model
    Returns:
        The generated KVCacheGroupSpecs
    """
    # Group all layers by kv_cache_spec.
    # E.g., 2 full attention layers and 3 sliding window attention layers,
    # -> (full.0, full.1), (sw.0, sw.1, sw.2).
    same_type_layers: dict[KVCacheSpec, list[str]] = defaultdict(list)
    for layer_name, layer_spec in kv_cache_spec.items():
        same_type_layers[layer_spec].append(layer_name)

    # Split each group into smaller groups, to make the number of layers in each
    # group identical. Add padding to the last group of each type if necessary.
    # E.g., (full.0, full.1), (sw.0, sw.1, sw.2)
    # split to 3 groups with 2 layers each:
    # (full.0, full.1), (sw.0, sw.2), (sw.1, padding).
    # FIXME(Chen): At the moment of writing this code (2025-06-02), all
    # open-source hybrid model follows a n:1 pattern between different attention
    # types (e.g., Gemma3 5:1 between sw and full, LLaMA4 3:1 between local and
    # full), so we can use the "1" in the n:1 pattern as the group size, which
    # is the minimum number of layers among all attention types. Need a better
    # strategy if we want to support more complex patterns (e.g., 20 full + 30
    # sw, where the group size should be 10).
    min_num_layers = min([len(layers) for layers in same_type_layers.values()])
    group_size = min_num_layers
    max_num_layers = max([len(layers) for layers in same_type_layers.values()])
    if max_num_layers < min_num_layers * 1.5:
        # If the number of layers is not much larger than the minimum number of
        # layers, use the maximum number of layers as the group size to avoid
        # too many padding layers. A typical example is gpt-oss-20b + eagle,
        # with 12 sw + 13 full. We pad it to (13 sw, 13 full) instead of
        # (12 sw, 24 full). 1.5 is a heuristic to avoid too many padding
        # layers while accommodating speculative decoding drafters that add
        # extra layers to one attention type.
        group_size = max_num_layers
    grouped_layers = []
    for layers in same_type_layers.values():
        num_padding_layers = group_size - len(layers) % group_size
        if num_padding_layers != group_size:
            logger.warning(
                "Add %d padding layers, may waste at most %.2f%% KV cache memory",  # noqa
                num_padding_layers,
                num_padding_layers / len(layers) * 100,
            )
        num_groups = cdiv(len(layers), group_size)
        # In PP case, say if we have
        # - stage 0: full.0, sw.0, sw.1
        # - stage 1: full.1, sw.2, sw.3
        # We should have 3 groups: (full.0, full.1), (sw.0, sw.2), (sw.1, sw.3)
        # It can't be (full.0, full.1), (sw.0, sw.1), (sw.2, sw.3) because
        # the 3 groups in stage 0 will be (full.0), (sw.0, sw.1), (empty group)
        # and it will be padded to (full.0, padding), (sw.0, sw.1),
        # (padding, padding) to ensure the number of layers in each group is
        # the same and will cause memory waste.
        # To avoid this, we assign layers[i::num_groups] to the i-th group
        # instead of layers[i * group_size: (i + 1) * group_size]
        for i in range(num_groups):
            grouped_layers.append(layers[i::num_groups])
    return create_kv_cache_group_specs(kv_cache_spec, grouped_layers)


def get_kv_cache_config_from_groups(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> KVCacheConfig:
    """
    Generate the KV cache configuration from the KV cache groups and spec
    of each layer.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_groups: The KV cache groups
        available_memory: Memory available for KV cache in bytes
    Returns:
        The generated KVCacheConfig
    """
    hf_config = getattr(vllm_config.model_config, "hf_text_config", None)
    if hf_config is None:
        hf_config = getattr(vllm_config.model_config, "hf_config", None)
    dsa_index_topk = getattr(hf_config, "index_topk", None)
    dsa_num_speculative_tokens = max(
        int(getattr(vllm_config, "num_speculative_tokens", 0)), 0
    )
    dsa_sparse_rows = dsa_num_speculative_tokens + 1

    if len(kv_cache_groups) == 0:
        # Attention free models do not have KV cache.
        # Return num_blocks=1 as BlockPool always needs a null_block.
        return KVCacheConfig(
            num_blocks=1,
            kv_cache_tensors=[],
            kv_cache_groups=kv_cache_groups,
            dsa_index_topk=dsa_index_topk,
            dsa_num_speculative_tokens=dsa_num_speculative_tokens,
        )

    # Determine how model runners should initialize the KV cache tensors.
    if len(kv_cache_groups) == 1 and isinstance(
        kv_cache_groups[0].kv_cache_spec, UniformTypeKVCacheSpecs
    ):
        # Special case: all layers have the same type of KV cache but with
        # different hidden size. Allocate different amount of memory for each
        # layer based on its hidden size.
        num_blocks = (
            available_memory // kv_cache_groups[0].kv_cache_spec.page_size_bytes
        )
        num_blocks = may_override_num_blocks(vllm_config, num_blocks)
        per_layer_specs = kv_cache_groups[0].kv_cache_spec.kv_cache_specs
        kv_cache_tensors = [
            KVCacheTensor(
                size=per_layer_specs[layer_name].page_size_bytes * num_blocks,
                shared_by=[layer_name],
            )
            for layer_name in kv_cache_groups[0].layer_names
        ]
    elif dsa_two_groups_enabled() and len(
        {group.kv_cache_spec.page_size_bytes for group in kv_cache_groups}
    ) > 1:
        if dsa_shared_pool_enabled():
            if len(kv_cache_groups) != 2:
                raise ValueError("DSA shared pool expects exactly two KV groups.")
            latent_group = max(
                kv_cache_groups, key=lambda g: g.kv_cache_spec.page_size_bytes
            )
            indexer_group = min(
                kv_cache_groups, key=lambda g: g.kv_cache_spec.page_size_bytes
            )
            if len(latent_group.layer_names) != len(indexer_group.layer_names):
                raise ValueError(
                    "DSA shared pool expects one indexer layer per latent layer."
                )

            latent_page = latent_group.kv_cache_spec.page_size_bytes
            indexer_page = indexer_group.kv_cache_spec.page_size_bytes
            bundle_page = lcm(latent_page, indexer_page)
            latent_blocks_per_bundle = bundle_page // latent_page
            indexer_blocks_per_bundle = bundle_page // indexer_page
            num_layer_pairs = len(latent_group.layer_names)
            slot_count = available_memory // (num_layer_pairs * bundle_page)
            natural_bundles = slot_count - 1
            num_bundles = may_override_num_blocks(
                vllm_config,
                natural_bundles,
            )
            if num_bundles <= 0:
                raise ValueError(
                    "No available memory for DSA shared pool after reserving "
                    "bundle slot 0."
                )
            gpu_mem_util = getattr(
                vllm_config.cache_config, "gpu_memory_utilization", None
            )
            kv_mem_override = getattr(
                vllm_config.cache_config, "kv_cache_memory_bytes", None
            )
            nonshared_per_block_bytes = (
                len(latent_group.layer_names) * latent_page
                + len(indexer_group.layer_names) * indexer_page
            )
            natural_nonshared_blocks = (
                available_memory // nonshared_per_block_bytes
                if nonshared_per_block_bytes
                else 0
            )
            logger.info(
                "DSA shared sizing debug: gpu_memory_utilization=%s "
                "kv_cache_memory_bytes=%s available_kv=%.2f GiB "
                "latent_page=%.2f MiB indexer_page=%.2f MiB "
                "bundle_page=%.2f MiB latent_blocks_per_bundle=%d "
                "indexer_blocks_per_bundle=%d layer_pairs=%d "
                "natural_shared_bundles=%d natural_nonshared_blocks_per_group=%d.",
                gpu_mem_util,
                kv_mem_override,
                available_memory / 2**30,
                latent_page / 2**20,
                indexer_page / 2**20,
                bundle_page / 2**20,
                latent_blocks_per_bundle,
                indexer_blocks_per_bundle,
                num_layer_pairs,
                natural_bundles,
                natural_nonshared_blocks,
            )
            tensor_size = (num_bundles + 1) * bundle_page
            shared_pool_bytes = num_layer_pairs * tensor_size
            max_model_len = vllm_config.model_config.max_model_len
            max_num_seqs = vllm_config.scheduler_config.max_num_seqs
            latent_ctx_blocks = cdiv(
                max_model_len, latent_group.kv_cache_spec.block_size
            )
            indexer_ctx_blocks = cdiv(
                max_model_len, indexer_group.kv_cache_spec.block_size
            )
            if dsa_index_topk is not None and (
                int(dsa_index_topk) % latent_group.kv_cache_spec.block_size
            ):
                raise ValueError(
                    "DSA index_topk must be an integer multiple of block_size: "
                    f"index_topk={dsa_index_topk}, "
                    f"block_size={latent_group.kv_cache_spec.block_size}"
                )
            scratch_blocks = (
                int(dsa_index_topk)
                * dsa_sparse_rows
                // latent_group.kv_cache_spec.block_size
                if dsa_index_topk is not None
                else 0
            )
            full_context_bundles = cdiv(
                latent_ctx_blocks, latent_blocks_per_bundle
            ) + cdiv(indexer_ctx_blocks, indexer_blocks_per_bundle)
            sparse_decode_bundles = cdiv(
                scratch_blocks, latent_blocks_per_bundle
            ) + cdiv(indexer_ctx_blocks, indexer_blocks_per_bundle)
            full_context_capacity = (
                num_bundles // full_context_bundles
                if full_context_bundles
                else 0
            )
            sparse_decode_capacity = (
                num_bundles // sparse_decode_bundles
                if sparse_decode_bundles
                else 0
            )
            logger.info(
                "DSA shared pool budget: gpu_memory_utilization=%s "
                "kv_cache_memory_bytes=%s available_kv=%.2f GiB "
                "allocated=%.2f GiB fits_budget=%s bundles=%d "
                "bundle_page=%.2f MiB layer_pairs=%d.",
                gpu_mem_util,
                kv_mem_override,
                available_memory / 2**30,
                shared_pool_bytes / 2**30,
                shared_pool_bytes <= available_memory,
                num_bundles,
                bundle_page / 2**20,
                num_layer_pairs,
            )
            logger.info(
                "DSA shared pool capacity: max_model_len=%d max_num_seqs=%d "
                "index_topk=%s spec_tokens=%d scratch_rows=%d "
                "scratch_blocks=%d full_context_bundles_per_seq=%d "
                "full_context_capacity=%d sparse_decode_bundles_per_seq=%d "
                "sparse_decode_capacity=%d sparse_decode_enough_for_max_seqs=%s.",
                max_model_len,
                max_num_seqs,
                dsa_index_topk,
                dsa_num_speculative_tokens,
                dsa_sparse_rows,
                scratch_blocks,
                full_context_bundles,
                full_context_capacity,
                sparse_decode_bundles,
                sparse_decode_capacity,
                sparse_decode_capacity >= max_num_seqs,
            )

            indexer_by_name = set(indexer_group.layer_names)
            used_indexers: set[str] = set()
            pairs: list[tuple[str, str]] = []
            for latent_name in latent_group.layer_names:
                sibling = latent_name.rsplit(".", 1)[0] + ".indexer.k_cache"
                if sibling in indexer_by_name:
                    pairs.append((latent_name, sibling))
                    used_indexers.add(sibling)
            if len(pairs) != len(latent_group.layer_names):
                pairs = list(zip(latent_group.layer_names, indexer_group.layer_names))
                used_indexers = {indexer for _, indexer in pairs}
            if used_indexers != indexer_by_name:
                raise ValueError(
                    "DSA shared pool could not pair latent/indexer layers."
                )

            logger.info(
                "DSA shared pool sizing: bundles=%d, bundle_page=%.2f MiB, "
                "layer_pairs=%d, tensor_slots_include_null=%d.",
                num_bundles,
                bundle_page / 2**20,
                num_layer_pairs,
                num_bundles + 1,
            )
            kv_cache_tensors = [
                KVCacheTensor(size=tensor_size, shared_by=[latent, indexer])
                for latent, indexer in pairs
            ]
            return KVCacheConfig(
                num_blocks=num_bundles,
                kv_cache_tensors=kv_cache_tensors,
                kv_cache_groups=kv_cache_groups,
                dsa_index_topk=dsa_index_topk,
                dsa_num_speculative_tokens=dsa_num_speculative_tokens,
            )

        # DSA two-group mode: groups have different page sizes and each group is
        # backed by its OWN block pool. Every layer gets its own tensor of
        # group_num_blocks * its_own_page bytes.
        per_block_bytes = sum(
            len(group.layer_names) * group.kv_cache_spec.page_size_bytes
            for group in kv_cache_groups
        )
        natural_num_blocks = available_memory // per_block_bytes
        num_blocks = natural_num_blocks
        num_blocks = may_override_num_blocks(vllm_config, num_blocks)
        num_blocks_per_group = [num_blocks for _ in kv_cache_groups]
        gpu_mem_util = getattr(
            vllm_config.cache_config, "gpu_memory_utilization", None
        )
        kv_mem_override = getattr(
            vllm_config.cache_config, "kv_cache_memory_bytes", None
        )
        logger.info(
            "DSA nonshared sizing debug: gpu_memory_utilization=%s "
            "kv_cache_memory_bytes=%s available_kv=%.2f GiB "
            "per_block_bytes=%.2f MiB natural_blocks_per_group=%d.",
            gpu_mem_util,
            kv_mem_override,
            available_memory / 2**30,
            per_block_bytes / 2**20,
            natural_num_blocks,
        )

        if dsa_shrink_stage() == 2 and len(kv_cache_groups) == 2:
            # ===== TEST: indexer-priority pool sizing for a fixed target batch =====
            # Size the two pools to support DSA_TEST_BATCH concurrent decoders at
            # the actual request context, indexer-priority:
            #   indexer = batch * ctx_blocks            (resident full context)
            #   latent  = prefill_conc * ctx_blocks + batch * (scratch + gen_tail)
            # No clamping — it sizes for the target so the case runs at that batch.
            # Env knobs; revert after the experiment.
            block = kv_cache_groups[0].kv_cache_spec.block_size
            if dsa_index_topk is not None and int(dsa_index_topk) % block:
                raise ValueError(
                    "DSA index_topk must be an integer multiple of block_size: "
                    f"index_topk={dsa_index_topk}, block_size={block}"
                )
            scratch = (
                int(dsa_index_topk) * dsa_sparse_rows // block
                if dsa_index_topk is not None
                else 0
            )
            test_batch = int(os.getenv("DSA_TEST_BATCH", "11"))
            ctx_blocks = int(
                os.getenv("DSA_TEST_CTX_BLOCKS", str(cdiv(61056, block)))
            )  # ~60k prompt + 1k gen
            gen_blocks = cdiv(int(os.getenv("DSA_TEST_GEN_TOKENS", "1000")), block)
            prefill_conc = int(os.getenv("DSA_TEST_PREFILL_CONC", "1"))

            latent_blocks = (
                prefill_conc * ctx_blocks
                + test_batch * (scratch + gen_blocks)
            )
            idx_blocks = test_batch * ctx_blocks
            num_blocks_per_group = [latent_blocks, idx_blocks]
            num_blocks = max(num_blocks_per_group)

            latent_layers = len(kv_cache_groups[0].layer_names)
            latent_page = kv_cache_groups[0].kv_cache_spec.page_size_bytes
            idx_layers = len(kv_cache_groups[1].layer_names)
            idx_page = kv_cache_groups[1].kv_cache_spec.page_size_bytes
            need = (
                latent_layers * latent_blocks * latent_page
                + idx_layers * idx_blocks * idx_page
            )
            logger.info(
                "DSA TEST sizing: batch=%d ctx_blocks=%d -> latent=%d (%.2f GiB) "
                "indexer=%d (%.2f GiB) | need=%.2f avail=%.2f GiB%s",
                test_batch, ctx_blocks,
                latent_blocks, latent_layers * latent_blocks * latent_page / 2**30,
                idx_blocks, idx_layers * idx_blocks * idx_page / 2**30,
                need / 2**30, available_memory / 2**30,
                (
                    ""
                    if need <= available_memory
                    else "  [OVER BUDGET -> add --enforce-eager]"
                ),
            )

        nonshared_group_bytes = [
            len(group.layer_names)
            * group.kv_cache_spec.page_size_bytes
            * num_blocks_per_group[i]
            for i, group in enumerate(kv_cache_groups)
        ]
        nonshared_total_bytes = sum(nonshared_group_bytes)
        logger.info(
            "DSA nonshared actual pool usage: blocks_per_group=%s "
            "group_allocated_gib=%s total_allocated=%.2f GiB "
            "available_kv=%.2f GiB fits_budget=%s.",
            num_blocks_per_group,
            [round(bytes_ / 2**30, 4) for bytes_ in nonshared_group_bytes],
            nonshared_total_bytes / 2**30,
            available_memory / 2**30,
            nonshared_total_bytes <= available_memory,
        )

        kv_cache_tensors = [
            KVCacheTensor(
                size=group.kv_cache_spec.page_size_bytes * num_blocks_per_group[i],
                shared_by=[layer_name],
            )
            for i, group in enumerate(kv_cache_groups)
            for layer_name in group.layer_names
        ]
        return KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=kv_cache_tensors,
            kv_cache_groups=kv_cache_groups,
            num_blocks_per_group=num_blocks_per_group,
            dsa_index_topk=dsa_index_topk,
            dsa_num_speculative_tokens=dsa_num_speculative_tokens,
        )
    else:
        # General case:
        # We will have group_size memory pools, each is shared by one layer from
        # each group. As layers of different groups have different block table,
        # they will use different parts of the shared Tensor.
        # The memory layout for 3 groups (full.0, full.1), (sw.0, sw.2),
        # (sw.1, padding) will be: (group_size = 2)
        # full.0, sw.0, sw.1: share a Tensor with size=available_memory//2
        # full.1, sw.2: share another Tensor with size=available_memory//2
        group_size = max(len(group.layer_names) for group in kv_cache_groups)

        page_size = get_uniform_page_size(
            [group.kv_cache_spec for group in kv_cache_groups]
        )
        assert group_size > 0, "group_size must be greater than 0"
        num_blocks = get_num_blocks(
            vllm_config, group_size, available_memory, page_size
        )
        kv_cache_tensors = []
        for i in range(group_size):
            shared_by = []
            for j in range(len(kv_cache_groups)):
                if i < len(kv_cache_groups[j].layer_names):
                    shared_by.append(kv_cache_groups[j].layer_names[i])
            kv_cache_tensors.append(
                KVCacheTensor(size=page_size * num_blocks, shared_by=shared_by)
            )

    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=kv_cache_tensors,
        kv_cache_groups=kv_cache_groups,
        dsa_index_topk=dsa_index_topk,
        dsa_num_speculative_tokens=dsa_num_speculative_tokens,
    )


def unify_hybrid_kv_cache_specs(kv_cache_spec: dict[str, KVCacheSpec]):
    """
    This function tries to convert the KV cache specs to one type if the model
    is a hybrid model with multiple type of KV cache. It will convert all
    SlidingWindowSpec to FullAttentionSpec if both types are present.

    Args:
        kv_cache_spec: The kv cache spec of each attention layer in the model
    """

    if dsa_two_groups_enabled():
        head_sizes = {
            spec.head_size
            for spec in kv_cache_spec.values()
            if isinstance(spec, MLAAttentionSpec)
        }
        if len(head_sizes) > 1:
            # DSA two-group mode: latent and indexer INTENTIONALLY form two
            # groups with per-group block pools, even though a KV connector
            # disables the hybrid KV cache manager (the non-HMA connector only
            # consumes the latent group, block_ids[0]). Leave the specs alone.
            return

    if is_kv_cache_spec_uniform(
        kv_cache_spec
    ) or UniformTypeKVCacheSpecs.is_uniform_type(kv_cache_spec):
        return

    logger.warning(
        "Hybrid KV cache manager is disabled for this hybrid model, "
        "This means we do not enable any optimizations for saving KV cache "
        "memory (e.g., dropping the KV cache outside the sliding window). "
        "The compute of layers like sliding window is still saved."
    )

    has_full_attention = any(
        isinstance(spec, FullAttentionSpec) for spec in kv_cache_spec.values()
    )
    has_sliding_window = any(
        isinstance(spec, SlidingWindowSpec) for spec in kv_cache_spec.values()
    )
    has_chunked_local_attention = any(
        isinstance(spec, ChunkedLocalAttentionSpec) for spec in kv_cache_spec.values()
    )
    if has_full_attention and (has_sliding_window or has_chunked_local_attention):
        for layer_name, spec in kv_cache_spec.items():
            if isinstance(spec, SlidingWindowSpec):
                kv_cache_spec[layer_name] = FullAttentionSpec(
                    block_size=spec.block_size,
                    num_kv_heads=spec.num_kv_heads,
                    head_size=spec.head_size,
                    dtype=spec.dtype,
                    sliding_window=spec.sliding_window,
                    page_size_padded=spec.page_size_padded,
                )
            elif isinstance(spec, ChunkedLocalAttentionSpec):
                kv_cache_spec[layer_name] = FullAttentionSpec(
                    block_size=spec.block_size,
                    num_kv_heads=spec.num_kv_heads,
                    head_size=spec.head_size,
                    dtype=spec.dtype,
                    attention_chunk_size=spec.attention_chunk_size,
                    page_size_padded=spec.page_size_padded,
                )

    if not (
        is_kv_cache_spec_uniform(kv_cache_spec)
        or UniformTypeKVCacheSpecs.is_uniform_type(kv_cache_spec)
    ):
        raise ValueError(
            "Hybrid KV cache manager is disabled but failed to "
            "convert the KV cache specs to one unified type."
        )


def get_kv_cache_groups(
    vllm_config: VllmConfig, kv_cache_spec: dict[str, KVCacheSpec]
) -> list[KVCacheGroupSpec]:
    """
    Split the layers in the model into groups with the same KV cache spec.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_spec: The kv cache spec of each attention layer in the model

    Returns:
        The generated KVCacheGroups
    """
    if vllm_config.scheduler_config.disable_hybrid_kv_cache_manager:
        unify_hybrid_kv_cache_specs(kv_cache_spec)

    if is_kv_cache_type_attention_free(kv_cache_spec):
        # This returns an empty list to allow for the KVCacheManager to handle
        # attention free models.
        return []

    if is_kv_cache_spec_uniform(kv_cache_spec):
        # KV cache of all layers are the same, which is true for
        # most models. Allocate the same amount of memory for
        # each layer.
        return _get_kv_cache_groups_uniform_spec(kv_cache_spec)
    elif uniform_spec := UniformTypeKVCacheSpecs.from_specs(kv_cache_spec):
        # All layers need the same number of token slots (e.g., all layers are
        # full attention, or all layers are sliding window attention with the
        # same window size). Put all layers into one group.
        return _get_kv_cache_groups_uniform_type(uniform_spec)

    if dsa_two_groups_enabled() and not is_kv_cache_page_size_uniform(kv_cache_spec):
        # DSA two-group mode: latent (page 147456) and indexer (page 32768) keep
        # their true page sizes and are backed by per-group block pools, so no
        # page-size unification is needed. Sort groups by descending page size so
        # the latent group is group 0 (KV connectors that only handle the latent
        # consume block_ids[0]).
        groups = _get_kv_cache_groups_uniform_page_size(kv_cache_spec)
        groups.sort(key=lambda g: -g.kv_cache_spec.page_size_bytes)
        return groups

    # As KVCacheManager can only allocate memory of one size, we need to unify
    # the page size of the layers. For cases cannot be unified, this function
    # will raise an error.
    kv_cache_spec = unify_kv_cache_spec_page_size(kv_cache_spec)
    # Model contains multiple attention types, but KV cache of all layers
    # have the same physical memory per block per layer. Split the layers
    # into groups with the same number of layers, and thus same total page
    # size.
    return _get_kv_cache_groups_uniform_page_size(kv_cache_spec)


def generate_scheduler_kv_cache_config(
    kv_cache_configs: list[KVCacheConfig],
) -> KVCacheConfig:
    """
    Generate the KV cache configuration for the scheduler.
    """
    assert all(
        [cfg.num_blocks == kv_cache_configs[0].num_blocks for cfg in kv_cache_configs]
    )
    # All workers have the same kv_cache_config except layer names, so use
    # an arbitrary one to initialize the scheduler.
    cfg = copy.deepcopy(kv_cache_configs[0])
    for group in cfg.kv_cache_groups:
        if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs):
            # All layers in the UniformTypeKVCacheSpecs have the same type,
            # so use an arbitrary one to initialize the scheduler.
            group.kv_cache_spec = next(
                iter(group.kv_cache_spec.kv_cache_specs.values())
            )
    return cfg


def _report_kv_cache_config(
    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig
) -> None:
    """
    Log resolved KV cache configuration.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_config: The resolved KV cache configuration
    """
    min_block_size = min(
        [group.kv_cache_spec.block_size for group in kv_cache_config.kv_cache_groups]
    )

    # Log the KV cache size and maximum concurrency.
    if (
        dsa_two_groups_enabled()
        and not dsa_shared_pool_enabled()
        and len(
            {
                g.kv_cache_spec.page_size_bytes
                for g in kv_cache_config.kv_cache_groups
            }
        )
        > 1
    ):
        # Per-group block pools: each pool independently holds num_blocks blocks,
        # so the model supports num_blocks * block_size tokens (not divided by the
        # number of groups).
        num_tokens = kv_cache_config.num_blocks * min_block_size
    elif dsa_two_groups_enabled() and dsa_shared_pool_enabled() and len(
        {g.kv_cache_spec.page_size_bytes for g in kv_cache_config.kv_cache_groups}
    ) > 1:
        latent_group = max(
            kv_cache_config.kv_cache_groups,
            key=lambda g: g.kv_cache_spec.page_size_bytes,
        )
        bundle_page = lcm(
            *(
                g.kv_cache_spec.page_size_bytes
                for g in kv_cache_config.kv_cache_groups
            )
        )
        latent_blocks_per_bundle = (
            bundle_page // latent_group.kv_cache_spec.page_size_bytes
        )
        num_tokens = (
            kv_cache_config.num_blocks
            * latent_blocks_per_bundle
            * min_block_size
        )
    else:
        num_tokens = (
            kv_cache_config.num_blocks
            // len(kv_cache_config.kv_cache_groups)
            * min_block_size
        )
    dcp_size = vllm_config.parallel_config.decode_context_parallel_size
    pcp_size = vllm_config.parallel_config.prefill_context_parallel_size
    if pcp_size * dcp_size > 1:
        num_tokens *= pcp_size * dcp_size
        logger.info(
            "Multiplying the GPU KV cache size by the cp_world_size %d "
            "(pcp_world_size %d * dcp_world_size %d).",
            pcp_size * dcp_size,
            pcp_size,
            dcp_size,
        )
    num_tokens_str = f"{num_tokens:,}"
    logger.info_once("GPU KV cache size: %s tokens", num_tokens_str, scope="local")
    max_model_len_str = f"{vllm_config.model_config.max_model_len:,}"
    max_concurrency = get_max_concurrency_for_kv_cache_config(
        vllm_config, kv_cache_config
    )
    logger.info_once(
        "Maximum concurrency for %s tokens per request: %.2fx",
        max_model_len_str,
        max_concurrency,
        scope="local",
    )


def _max_memory_usage_bytes_from_groups(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
) -> int:
    """
    Calculate maximum memory usage in bytes from KV cache groups.

    This correctly accounts for padding in hybrid models. For example, if a
    model has 8 full attention layers and 9 sliding window layers, they will
    be padded to 9 full + 9 sliding window for uniform group sizes.
    """
    if not kv_cache_groups:
        return 0

    # UniformTypeKVCacheSpecs special case (single group, per-layer specs)
    if len(kv_cache_groups) == 1 and isinstance(
        kv_cache_groups[0].kv_cache_spec, UniformTypeKVCacheSpecs
    ):
        per_layer_specs = kv_cache_groups[0].kv_cache_spec.kv_cache_specs
        return sum(
            spec.max_memory_usage_bytes(vllm_config)
            for spec in per_layer_specs.values()
        )

    # DSA two-group mode (per-group block pools, different page sizes): one
    # max-length request needs blocks from EVERY group's pool. Per group:
    # blocks = cdiv(per-layer bytes for max_len, page); bytes = layers * blocks * page.
    if dsa_two_groups_enabled() and len(
        {g.kv_cache_spec.page_size_bytes for g in kv_cache_groups}
    ) > 1:
        if dsa_shared_pool_enabled():
            latent_group = max(
                kv_cache_groups, key=lambda g: g.kv_cache_spec.page_size_bytes
            )
            indexer_group = min(
                kv_cache_groups, key=lambda g: g.kv_cache_spec.page_size_bytes
            )
            bundle_page = lcm(
                latent_group.kv_cache_spec.page_size_bytes,
                indexer_group.kv_cache_spec.page_size_bytes,
            )
            blocks = cdiv(
                latent_group.kv_cache_spec.max_memory_usage_bytes(vllm_config),
                latent_group.kv_cache_spec.page_size_bytes,
            )
            bundles = cdiv(
                blocks,
                bundle_page // latent_group.kv_cache_spec.page_size_bytes,
            ) + cdiv(
                blocks,
                bundle_page // indexer_group.kv_cache_spec.page_size_bytes,
            )
            return len(latent_group.layer_names) * (bundles + 1) * bundle_page

        total = 0
        for group in kv_cache_groups:
            page = group.kv_cache_spec.page_size_bytes
            blocks = cdiv(group.kv_cache_spec.max_memory_usage_bytes(vllm_config), page)
            total += len(group.layer_names) * blocks * page
        return total

    # General case: group_size pools, each shared by one layer per group
    # Memory = group_size * page_size * blocks_for_max_len
    group_size = max(len(group.layer_names) for group in kv_cache_groups)
    page_size = get_uniform_page_size(
        [group.kv_cache_spec for group in kv_cache_groups]
    )
    any_spec = kv_cache_groups[0].kv_cache_spec
    blocks_needed = cdiv(any_spec.max_memory_usage_bytes(vllm_config), page_size)

    return group_size * page_size * blocks_needed


def _estimate_max_model_len_from_groups(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> int:
    """
    Binary search for the maximum model length that fits in available memory.
    Returns 0 if even 1 token doesn't fit.
    """
    original_max = vllm_config.model_config.max_model_len

    def fits(model_len: int) -> bool:
        vllm_config.model_config.max_model_len = model_len
        return (
            _max_memory_usage_bytes_from_groups(vllm_config, kv_cache_groups)
            <= available_memory
        )

    try:
        left, right = 1, original_max
        if not fits(left):
            return 0
        result = 1
        while left <= right:
            mid = (left + right) // 2
            if fits(mid):
                result = mid
                left = mid + 1
            else:
                right = mid - 1
        return result
    finally:
        vllm_config.model_config.max_model_len = original_max


def _auto_fit_max_model_len(
    vllm_config: VllmConfig,
    projected_groups_per_worker: list[list[KVCacheGroupSpec]],
    available_memory: list[int],
) -> None:
    """
    When max_model_len is set to -1, this function estimates the largest
    context length that can be supported with the available GPU memory.
    It uses binary search to find the maximum length that fits across all
    workers.

    Args:
        vllm_config: The global VllmConfig (will be modified in-place)
        projected_groups_per_worker: KV cache groups projected to each worker.
        available_memory: Memory available for KV cache in bytes for each
            worker.
    """
    original_max = vllm_config.model_config.max_model_len

    if all(not groups for groups in projected_groups_per_worker):
        # All workers have empty specs (attention-free model)
        logger.info_once(
            "Auto-fit max_model_len: attention-free model, "
            "using derived max_model_len=%d",
            original_max,
            scope="local",
        )
        return

    # Find the max_model_len that fits across all workers.
    auto_fit_max = original_max
    limiting_worker_mem = available_memory[0]
    for groups, avail_mem in zip(projected_groups_per_worker, available_memory):
        if not groups:
            continue
        worker_max = _estimate_max_model_len_from_groups(vllm_config, groups, avail_mem)
        if worker_max < auto_fit_max:
            auto_fit_max = worker_max
            limiting_worker_mem = avail_mem

    if auto_fit_max <= 0:
        raise ValueError(
            "Cannot auto-fit max_model_len: not enough GPU memory available "
            "to serve even a single token. Try increasing `gpu_memory_utilization`."
        )

    if auto_fit_max >= original_max:
        # The model's full context length fits in memory
        logger.info_once(
            "Auto-fit max_model_len: full model context length %d fits in "
            "available GPU memory",
            original_max,
            scope="local",
        )
    else:
        # Need to reduce max_model_len to fit in memory
        vllm_config.model_config.max_model_len = auto_fit_max
        logger.info_once(
            "Auto-fit max_model_len: reduced from %d to %d to fit in "
            "available GPU memory (%s GiB available for KV cache)",
            original_max,
            auto_fit_max,
            format_gib(limiting_worker_mem),
            scope="local",
        )


def _project_kv_cache_groups_to_worker(
    global_kv_cache_groups: list[KVCacheGroupSpec],
    worker_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    """
    Projects global KV cache groups onto a single worker's assigned layers.

    In pipeline parallelism, each worker only owns a subset of layers. This
    function filters the global groups to include only layers present on the
    given worker, adjusting UniformTypeKVCacheSpecs accordingly.

    Args:
        global_kv_cache_groups: The global KV cache groups for the whole model.
        worker_spec: The KV cache spec of each layer on this worker.

    Returns:
        The projected KV cache groups containing only this worker's layers.
    """
    projected_groups: list[KVCacheGroupSpec] = []
    for group in global_kv_cache_groups:
        worker_layer_names = [
            layer_name for layer_name in group.layer_names if layer_name in worker_spec
        ]
        group_spec = group.kv_cache_spec
        if worker_layer_names and isinstance(group_spec, UniformTypeKVCacheSpecs):
            group_spec = UniformTypeKVCacheSpecs(
                block_size=group_spec.block_size,
                kv_cache_specs={
                    layer_name: group_spec.kv_cache_specs[layer_name]
                    for layer_name in worker_layer_names
                },
            )
        projected_groups.append(KVCacheGroupSpec(worker_layer_names, group_spec))
    return projected_groups


def get_kv_cache_configs(
    vllm_config: VllmConfig,
    kv_cache_specs: list[dict[str, KVCacheSpec]],
    available_memory: list[int],
) -> list[KVCacheConfig]:
    """
    Generates the KV cache configurations for a model.
    Since we use a shared centralized controller for all workers, we need the
    `kv_cache_config` to be consistent across all workers to make sure
    the KV cache allocation can be applied to all workers. However, different
    workers may have different memory available, and different type of layers
    (when pipeline parallel is enabled). To handle the difference between
    workers, the current implementation is:
    1. Merge the KV cache specs of all workers to get the KVCacheSpecs for
       the whole model.
    2. Generate the KV cache groups based on the layer ratio of the whole model.
       This also handles spec unification for hybrid models.
    3. Handle auto-fit max_model_len and memory checks using per-worker
       projected groups to account for PP sharding.
    4. Generate the KV cache configs for each worker based on the KV cache
       grouping strategy. (This is reasonable because the layer ratio of
       different PP stages are similar.)
    5. Change the num_blocks of each worker to the smallest among all workers
       and shrink tensor sizes proportionally to avoid allocating unused memory.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_specs: List of dict[layer_name, KVCacheSpec] for each worker.
        available_memory: Memory available for KV cache in bytes for each
            worker.

    Returns:
        The generated KVCacheConfigs for each worker.
    """

    # Merge the KV cache specs of all workers. Different PP stages may have
    # different layer names, and different TP ranks of the same PP stage should
    # have the same KV cache spec.
    merged_kv_cache_specs: dict[str, KVCacheSpec] = {}
    for kv_cache_spec_one_worker in kv_cache_specs:
        for layer_name, layer_spec in kv_cache_spec_one_worker.items():
            if layer_name not in merged_kv_cache_specs:
                merged_kv_cache_specs[layer_name] = layer_spec
            else:
                assert merged_kv_cache_specs[layer_name] == layer_spec, (
                    "The KV cache specs for the same layer are different "
                    "across workers. This is not supported yet."
                )

    # Get global KV cache groups. This also handles spec unification for
    # hybrid models when disable_hybrid_kv_cache_manager is enabled.
    # After this call, merged_kv_cache_specs may be modified in-place.
    global_kv_cache_groups = get_kv_cache_groups(vllm_config, merged_kv_cache_specs)

    # If original_max_model_len was -1, automatically
    # determine the maximum model length that fits in available GPU memory.
    # We use per-worker projected groups to account for PP sharding.
    projected_groups_per_worker = [
        _project_kv_cache_groups_to_worker(global_kv_cache_groups, worker_spec)
        for worker_spec in kv_cache_specs
    ]

    if vllm_config.model_config.original_max_model_len == -1:
        _auto_fit_max_model_len(
            vllm_config, projected_groups_per_worker, available_memory
        )

    # Check if the available memory is enough per worker.
    for groups, avail_mem in zip(projected_groups_per_worker, available_memory):
        if not groups:
            continue
        _check_enough_kv_cache_memory(
            avail_mem,
            partial(_max_memory_usage_bytes_from_groups, vllm_config, groups),
            vllm_config.model_config.max_model_len,
            partial(_estimate_max_model_len_from_groups, vllm_config, groups),
        )

    kv_cache_configs: list[KVCacheConfig] = []
    for projected_groups, kv_cache_spec_one_worker, available_memory_one_worker in zip(
        projected_groups_per_worker, kv_cache_specs, available_memory
    ):
        assert sum(len(group.layer_names) for group in projected_groups) == len(
            kv_cache_spec_one_worker
        ), "Some layers are not assigned to any group."
        kv_cache_configs.append(
            get_kv_cache_config_from_groups(
                vllm_config, projected_groups, available_memory_one_worker
            )
        )

    # Make the block count uniform across ranks (so block ids are consistent):
    # take the cross-rank minimum and shrink the larger ranks' tensors to match.
    #
    # Two reconciliation modes:
    #   * Per-group (e.g. DSA two-group): each KV cache group has its OWN block
    #     count (num_blocks_per_group), and a tensor's byte size is a multiple of
    #     ITS group's count, not the scalar num_blocks (= max across groups).
    #     Each group must therefore be reconciled against its own cross-rank min,
    #     and a tensor shrunk by its own group's count. Reconciling with the
    #     scalar num_blocks would both mis-shrink and trip the divisibility
    #     assert whenever the per-group counts don't align (a tensor sized for
    #     latent_blocks is not a multiple of max(latent, indexer)).
    #   * DSA shared-bundle: tensor size is proportional to num_bundles + 1
    #     because bundle slot 0 is reserved for block-table padding.
    #   * Scalar (default single-pool): every tensor uses the one num_blocks.
    use_dsa_shared = all(
        dsa_two_groups_enabled()
        and dsa_shared_pool_enabled()
        and len(kv_cache_config.kv_cache_groups) == 2
        and len(
            {
                g.kv_cache_spec.page_size_bytes
                for g in kv_cache_config.kv_cache_groups
            }
        )
        > 1
        for kv_cache_config in kv_cache_configs
    )
    use_per_group = (not use_dsa_shared) and all(
        kv_cache_config.num_blocks_per_group is not None
        for kv_cache_config in kv_cache_configs
    )

    if use_per_group:
        num_groups = len(kv_cache_configs[0].num_blocks_per_group)
        assert all(
            len(c.num_blocks_per_group) == num_groups for c in kv_cache_configs
        ), "All ranks must agree on the number of KV cache groups."
        min_blocks_per_group = [
            min(c.num_blocks_per_group[g] for c in kv_cache_configs)
            for g in range(num_groups)
        ]
    else:
        min_num_blocks = min(
            kv_cache_config.num_blocks for kv_cache_config in kv_cache_configs
        )

    for kv_cache_config in kv_cache_configs:
        if use_per_group:
            old_per_group = kv_cache_config.num_blocks_per_group
            # Tensors are laid out group-by-group, one per layer in each group
            # (see get_kv_cache_config_from_groups), so walking them in step with
            # the groups gives each tensor its owning group's block count.
            tensors = iter(kv_cache_config.kv_cache_tensors)
            for g, group in enumerate(kv_cache_config.kv_cache_groups):
                old_g = old_per_group[g]
                new_g = min_blocks_per_group[g]
                for _ in group.layer_names:
                    tensor = next(tensors)
                    assert tensor.size % old_g == 0
                    if new_g != old_g:
                        tensor.size = tensor.size // old_g * new_g
            kv_cache_config.num_blocks_per_group = list(min_blocks_per_group)
            kv_cache_config.num_blocks = max(min_blocks_per_group)
        elif use_dsa_shared:
            num_blocks_old = kv_cache_config.num_blocks
            if num_blocks_old != min_num_blocks:
                kv_cache_config.num_blocks = min_num_blocks
                for tensor in kv_cache_config.kv_cache_tensors:
                    assert tensor.size % (num_blocks_old + 1) == 0
                    tensor.size = tensor.size // (num_blocks_old + 1) * (
                        min_num_blocks + 1
                    )
        else:
            num_blocks_old = kv_cache_config.num_blocks
            # Only shrink when this rank actually computed more blocks than the
            # global min (always equal for a single rank, e.g. TP=1).
            if num_blocks_old != min_num_blocks:
                kv_cache_config.num_blocks = min_num_blocks

                # Shrink tensor size proportionally
                for tensor in kv_cache_config.kv_cache_tensors:
                    assert tensor.size % num_blocks_old == 0
                    tensor.size = tensor.size // num_blocks_old * min_num_blocks

        if len(kv_cache_config.kv_cache_groups) > 0:
            _report_kv_cache_config(vllm_config, kv_cache_config)

    return kv_cache_configs


class BlockHashListWithBlockSize:
    """
    Convert block-hash granularity from `hash_block_size` to `target_block_size`.
    Used when KV cache groups have different block sizes: `hash_block_size`
    is the size used to compute the original `block_hashes`; `target_block_size`
    is the group's actual block size.

    Currently, only scaling up by an integer factor is supported (i.e.,
    `target_block_size` is a multiple of `hash_block_size`). Conversion is
    performed lazily on access for efficiency, by concatenating consecutive
    hashes at `hash_block_size` to form each hash at `target_block_size`.

    Example (`hash_block_size` = 16, `target_block_size` = 32):
    concatenating two 16-size hashes yields one 32-size hash:

    Block hashes with block_size 16:
    | Token Range | 0-15 | 16-31 | 32-47 | 48-63 |
    |-------------|------|-------|-------|-------|
    | Hash        | A    | B     | C     | D     |

    Block hashes with block_size 32:
    | Token Range | 0-31 | 32-63 |
    |-------------|------|-------|
    | Hash        | AB   | CD    |

    Args:
        block_hashes: Block hashes to convert, computed at `hash_block_size`.
        hash_block_size: Block size at which `block_hashes` were computed.
        target_block_size: Desired block size; must be a multiple of `hash_block_size`.
    """

    def __init__(
        self,
        block_hashes: list[BlockHash],
        hash_block_size: int,
        target_block_size: int,
    ):
        self.block_hashes = block_hashes
        assert target_block_size % hash_block_size == 0
        self.scale_factor = target_block_size // hash_block_size

    def __len__(self) -> int:
        return len(self.block_hashes) // self.scale_factor

    @overload
    def __getitem__(self, idx: int) -> BlockHash: ...

    @overload
    def __getitem__(self, idx: slice) -> list[BlockHash]: ...

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self._get_value_at(idx)

        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            return [self._get_value_at(i) for i in range(start, stop, step)]

        raise TypeError(f"Invalid index type: {type(idx)!r}")

    def __iter__(self) -> Iterator[BlockHash]:
        for i in range(len(self)):
            yield self._get_value_at(i)

    def _get_value_at(self, idx: int) -> BlockHash:
        base = idx * self.scale_factor
        end = base + self.scale_factor
        merged_hash: bytes = self.block_hashes[base]
        for i in range(base + 1, end):
            merged_hash += self.block_hashes[i]
        return BlockHash(merged_hash)


BlockHashList = list[BlockHash] | BlockHashListWithBlockSize
