# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks of production allocation logic without importing GPU extensions."""

import ast
import dataclasses
import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_node(path, name, namespace, parent=None):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    body = tree.body
    if parent:
        body = next(n for n in body if getattr(n, "name", None) == parent).body
    node = next(n for n in body if getattr(n, "name", None) == name)
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


@pytest.fixture
def code():
    path = ROOT / "vllm/v1/core/dsa_shared_pool.py"
    spec = importlib.util.spec_from_file_location("_dsa_isolation", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    namespace = dict(vars(module), dataclass=dataclasses.dataclass)
    load_node("vllm/v1/core/kv_cache_utils.py", "KVCacheBlock", namespace)
    load_node("vllm/v1/core/block_pool.py", "DSASharedLogicalBlockPool", namespace)
    yield namespace
    sys.modules.pop(spec.name, None)


def make_pool(code, prefill=False, owner="LATENT"):
    layout = code["DSASharedBlockLayout"](
        latent_page_size_bytes=576 * 128 * 2,
        indexer_page_size_bytes=128 * 128 * 2,
        capacity_bundles=8,
        bundle_multiplier=2 if prefill else 1,
    )
    allocator = code["DSASharedBundleAllocator"](layout)
    if prefill:
        allocator = code["PrefillLayerBundlePool"](allocator, 4)
    return code["DSASharedLogicalBlockPool"](
        allocator, getattr(code["DSASharedBlockOwner"], owner)
    )


@pytest.mark.parametrize("owner", ["LATENT", "INDEXER"])
def test_decode_partial_release_and_reallocation(code, owner):
    pool = make_pool(code, owner=owner)
    initial = pool.get_num_free_blocks()
    blocks = pool.get_new_blocks(1)
    assert all(b.bank_block_ids is None for b in blocks)
    assert len(blocks) == pool.blocks_per_bundle
    pool.free_blocks([pool.null_block, blocks[0]])
    assert pool.get_num_free_blocks() == initial - len(blocks)
    pool.free_blocks(reversed(blocks[1:]))
    assert pool.get_num_free_blocks() == initial
    assert [b.block_id for b in pool.get_new_blocks(1)] == [b.block_id for b in blocks]


def test_decode_release_keeps_legacy_refcount_semantics(code):
    pool = make_pool(code)
    initial = pool.get_num_free_blocks()
    blocks = pool.get_new_blocks(1)
    blocks[0].ref_cnt = 2
    # The full-parent baseline releases each reference sequentially, including
    # two references to the same block. P-only duplicate validation must not run.
    pool.free_blocks([blocks[0], blocks[0], *blocks[1:]])
    assert pool.get_num_free_blocks() == initial


def test_prefill_still_validates_both_banks_before_free(code):
    pool = make_pool(code, prefill=True)
    initial = pool.get_num_free_blocks()
    blocks = pool.get_new_blocks(1)
    assert all(len(b.bank_block_ids) == 2 for b in blocks)
    ids = [i for b in blocks for i in b.bank_block_ids]
    with pytest.raises(ValueError, match="duplicate"):
        pool.free_blocks([*blocks, blocks[0]])
    assert all(pool.blocks[i].ref_cnt == 1 for i in ids)
    pool.free_blocks(blocks)
    assert all(pool.blocks[i].ref_cnt == 0 for i in ids)
    assert pool.get_num_free_blocks() == initial


@pytest.mark.parametrize("prefill", [False, True])
@pytest.mark.parametrize("indexer_block_size", [64, 128, 256])
def test_concurrency_preserves_decode_formula(code, prefill, indexer_block_size):
    namespace = dict(
        code,
        cdiv=lambda x, y: (x + y - 1) // y,
        lcm=math.lcm,
        dsa_two_groups_enabled=lambda: True,
        dsa_shared_pool_enabled=lambda: True,
        layerwise_prefill_p_node_enabled=lambda: prefill,
        layerwise_prefill_bundle_multiplier=lambda: 2,
    )
    path = "vllm/v1/core/kv_cache_utils.py"
    for name in (
        "dsa_kv_residency_mode",
        "dsa_bundle_page_size_bytes",
        "dsa_required_bundles",
        "get_max_concurrency_for_kv_cache_config",
    ):
        load_node(path, name, namespace)
    latent = NS(
        kv_cache_spec=NS(page_size_bytes=576, block_size=128),
        layer_names=["l0", "l1", "l2", "l3"],
    )
    indexer = NS(kv_cache_spec=NS(page_size_bytes=128, block_size=indexer_block_size))
    config = NS(model_config=NS(max_model_len=4096))
    cache = NS(num_blocks=96, kv_cache_groups=[latent, indexer])
    result = namespace["get_max_concurrency_for_kv_cache_config"](config, cache)
    if prefill:
        expected = (96 * 4) / (
            2
            * (math.ceil(32 / 4) + math.ceil(math.ceil(4096 / indexer_block_size) / 18))
        )
    else:
        expected = 96 / (math.ceil(32 / 2) + math.ceil(32 / 9))
    assert result == expected


@pytest.mark.parametrize("prefill", [False, True])
def test_new_request_only_reads_bank_metadata_for_p(code, prefill):
    bank_calls = []
    banks = (([1], [2]), ([3], [4]))

    def bank_ids():
        bank_calls.append(True)
        assert prefill
        return banks

    namespace = dict(
        code, NewRequestData=NS(from_request=lambda *args, **kwargs: (args, kwargs))
    )
    make = load_node(
        "vllm/v1/core/sched/scheduler.py",
        "_make_new_request_data",
        namespace,
        "Scheduler",
    )
    scheduler = NS(
        kv_cache_manager=NS(coordinator=NS(layerwise_prefill_p_node=prefill))
    )
    args, kwargs = make(
        scheduler,
        "request",
        NS(get_block_ids=lambda: ([1], [2]), get_block_ids_by_bank=bank_ids),
        [10, 11],
    )
    assert args == ("request", ([1], [2]), [10, 11])
    assert len(bank_calls) == int(prefill)
    assert kwargs["block_ids_by_bank"] == (banks if prefill else None)


@pytest.mark.parametrize("prefill", [False, True])
def test_group_allocation_failure_rolls_back_only_p(code, prefill):
    class Manager:
        def __init__(self, fail=False):
            self.fail = fail
            self.req_to_blocks = {"req": ["old"]}
            self.freed = []
            self.block_pool = NS(free_blocks=lambda blocks: self.freed.extend(blocks))

        def allocate_new_blocks(self, req_id, num_tokens, main_tokens):
            assert (req_id, num_tokens, main_tokens) == ("req", 4096, 4095)
            if self.fail:
                raise ValueError("no space")
            self.req_to_blocks[req_id].append("new")
            return ["new"]

    namespace = dict(
        code,
        DSALatentManager=type("Latent", (), {}),
        CrossAttentionManager=type("Cross", (), {}),
    )
    allocate = load_node(
        "vllm/v1/core/kv_cache_coordinator.py",
        "allocate_new_blocks",
        namespace,
        "KVCacheCoordinator",
    )
    first, second = Manager(), Manager(fail=True)
    coordinator = NS(
        layerwise_prefill_p_node=prefill, single_type_managers=[first, second]
    )
    with pytest.raises(ValueError, match="no space"):
        allocate(coordinator, "req", 4096, 4095)
    assert first.req_to_blocks["req"] == (["old"] if prefill else ["old", "new"])
    assert first.freed == (["new"] if prefill else [])


@pytest.mark.parametrize("prefill", [False, True])
@pytest.mark.parametrize("resumed", [False, True])
def test_worker_decode_update_does_not_enter_bank_helper(code, prefill, resumed):
    path = "vllm/v1/worker/gpu_model_runner.py"
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    # Execute the actual worker dispatch, with a forbidden helper on D. Loading
    # the full runner would require CUDA even though this code only updates lists.
    branch = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.BoolOp)
        and "req_state.block_allocation_mode" in ast.unparse(node.test)
        and "new_block_allocation_mode" in ast.unparse(node.test)
        and any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id == "_update_request_kv_block_state"
            for child in ast.walk(node)
        )
    )
    mode = code["DSABlockAllocationMode"].PREFILL_CHILD if prefill else None
    state = NS(
        block_ids=([1], [10]),
        block_ids_by_bank=(([1], [10]), ([21], [30])) if prefill else None,
        block_allocation_mode=mode,
    )
    namespace = dict(
        code,
        req_id="req",
        req_state=state,
        req_index=None,
        new_block_ids=([2], [11]),
        new_block_ids_by_bank=(([2], [11]), ([22], [31])) if prefill else None,
        new_block_allocation_mode=mode,
        resumed_from_preemption=resumed,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("D entered bank metadata helper")

    if prefill:
        load_node(path, "_update_request_kv_block_state", namespace)
    else:
        namespace["_update_request_kv_block_state"] = forbidden
    exec(compile(ast.Module(body=[branch], type_ignores=[]), path, "exec"), namespace)
    assert state.block_ids == (([2], [11]) if resumed else ([1, 2], [10, 11]))
    if prefill:
        assert state.block_ids_by_bank[1] == (
            ([22], [31]) if resumed else ([21, 22], [30, 31])
        )
    else:
        assert state.block_ids_by_bank is None
