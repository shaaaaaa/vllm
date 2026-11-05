# SPDX-License-Identifier: Apache-2.0
"""Run the actual compact allocator with a small CPU block pool."""

import ast
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


def manager():
    path = (
        Path(__file__).resolve().parents[3]
        / "vllm/v1/core/single_type_kv_cache_manager.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "DSALatentManager"
    )
    names = {
        "_blocks_per_bundle",
        "_round_up_to_bundle",
        "_round_down_to_bundle",
        "_compact_required_indices",
        "_get_num_missing_blocks",
        "get_num_blocks_to_allocate_compact_external",
        "allocate_new_computed_blocks_compact_external",
        "_allocate_compact_request_blocks",
        "allocate_new_blocks_compact_external",
    }
    cls.bases = []
    cls.body = [
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    module = ast.parse("from __future__ import annotations")
    module.body.append(cls)
    ns = dict(cdiv=lambda a, b: (a + b - 1) // b, logger=NS(info=lambda *a: None))
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), ns)
    instance = ns[cls.name]()
    issued = []

    def allocate(count):
        blocks = [NS(block_id=len(issued) + i + 1, is_null=False) for i in range(count)]
        issued.extend(blocks)
        return blocks

    instance.block_pool = NS(
        blocks_per_bundle=2, get_new_blocks=allocate, get_num_free_blocks=lambda: 1000
    )
    instance.block_size = 128
    instance.scratch_blocks = 32
    instance._null_block = NS(block_id=0, is_null=True)
    instance._compact_external_prefix_tokens = {}
    instance._compact_external_released_blocks = {}
    instance.num_cached_block = {}
    instance.req_to_blocks = defaultdict(list)
    instance._get_num_evictable_blocks = lambda blocks: 0
    return instance, issued


@pytest.mark.parametrize("length", [2, 127, 128, 129, 280, 1025, 4095, 4096, 4097])
def test_short_cold_admission_budget_matches_real_allocation(length):
    allocator, issued = manager()
    required = allocator.get_num_blocks_to_allocate_compact_external(
        "r", length, [], length - 1, length - 1
    )
    allocator.allocate_new_computed_blocks_compact_external("r", [], 0, length - 1)
    allocator.allocate_new_blocks_compact_external("r", length, length - 1)
    assert len(issued) == required == ((length + 255) // 256) * 2
    assert all(not block.is_null for block in allocator.req_to_blocks["r"])
    original = list(allocator.req_to_blocks["r"])
    # Growing beyond scratch retains every previously loaded slot.
    allocator.allocate_new_blocks_compact_external("r", 8193, 8192)
    assert allocator.req_to_blocks["r"][: len(original)] == original
    assert all(not block.is_null for block in allocator.req_to_blocks["r"])


def test_long_cold_admission_retains_sparse_scratch_and_null_middle():
    allocator, issued = manager()
    length = 131073
    required = allocator.get_num_blocks_to_allocate_compact_external(
        "r", length, [], length - 1, length - 1
    )
    allocator.allocate_new_computed_blocks_compact_external("r", [], 0, length - 1)
    allocator.allocate_new_blocks_compact_external("r", length, length - 1)
    assert len(issued) == required
    blocks = allocator.req_to_blocks["r"]
    assert all(not b.is_null for b in blocks[:32])
    assert any(b.is_null for b in blocks[32:-4])
    assert not blocks[1024].is_null
