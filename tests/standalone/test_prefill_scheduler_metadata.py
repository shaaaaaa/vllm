# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only coverage of P-only scheduler metadata, using production methods."""

import ast
import itertools
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def api():
    path = ROOT / "vllm/v1/core/sched/scheduler.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Scheduler"
    )
    methods = [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef)
        and n.name in {"_make_new_request_data", "_make_cached_request_data"}
    ]
    ns = dict(
        itertools=itertools,
        DSABlockAllocationMode=NS(PREFILL_CHILD="prefill_child"),
        CachedRequestData=NS,
        NewRequestData=NS(
            from_request=lambda request, ids, tokens=None, **kw: NS(
                req_id=request.request_id,
                block_ids=ids,
                prefill_token_ids=tokens,
                block_ids_by_bank=kw.get("block_ids_by_bank"),
                block_allocation_mode=kw.get("block_allocation_mode"),
            )
        ),
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *methods,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), ns)
    return ns


class Blocks:
    def __init__(self, enabled, empty=False):
        self.enabled = enabled
        self.empty = empty
        self.bank_calls = 0

    def get_block_ids(self, allow_none=False):
        return None if self.empty and allow_none else ([7], [8])

    def get_block_ids_by_bank(self, allow_none=False):
        assert self.enabled, "non-P scheduler must not inspect bank metadata"
        self.bank_calls += 1
        return None if self.empty else (([7], [8]), ([17], [18]))

    def get_allocation_mode(self):
        pytest.fail("scheduler must not scan allocation mode a second time")


def runner(enabled):
    return NS(
        kv_cache_manager=NS(coordinator=NS(layerwise_prefill_p_node=enabled)),
        use_pp=False,
        prev_step_scheduled_req_ids={"running"},
    )


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("v2", [False, True])
def test_new_request(api, enabled, empty, v2):
    blocks = Blocks(enabled, empty)
    request = NS(request_id="new")
    tokens = [1, 2] if v2 else None
    result = api["_make_new_request_data"](runner(enabled), request, blocks, tokens)
    assert result.block_ids == ([7], [8])
    assert result.prefill_token_ids is tokens
    assert blocks.bank_calls == int(enabled)
    assert result.block_ids_by_bank == (
        (([7], [8]), ([17], [18])) if enabled and not empty else None
    )
    assert result.block_allocation_mode == (
        "prefill_child" if enabled and not empty else None
    )


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_running_and_resumed_requests(api, enabled, empty):
    requests = [
        NS(
            request_id=name,
            all_token_ids=[1, 2],
            num_computed_tokens=1,
            num_output_tokens=0,
            num_output_placeholders=0,
        )
        for name in ["running", "resumed"]
    ]
    blocks = {req.request_id: Blocks(enabled, empty) for req in requests}
    result = api["_make_cached_request_data"](
        runner(enabled), requests[:1], requests[1:], {}, {}, blocks
    )
    assert result.req_ids == ["running", "resumed"]
    assert result.resumed_req_ids == {"resumed"}
    assert result.all_token_ids == {"resumed": [1, 2]}
    assert all(b.bank_calls == int(enabled) for b in blocks.values())
    if enabled:
        banks = None if empty else (([7], [8]), ([17], [18]))
        assert result.new_block_ids_by_bank == [banks, banks]
        mode = None if empty else "prefill_child"
        assert result.new_block_allocation_modes == [mode, mode]
    else:
        assert result.new_block_ids_by_bank is None
        assert result.new_block_allocation_modes is None


def test_scheduler_uses_shared_new_request_builder():
    paths = [ROOT / "vllm/v1/core/sched/scheduler.py"]
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        schedules = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "schedule"
        ]
        assert schedules, path
        for schedule in schedules:
            calls = [
                n.func
                for n in ast.walk(schedule)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            ]
            assert any(n.attr == "_make_new_request_data" for n in calls), path
            assert not any(
                n.attr in {"get_allocation_mode", "get_block_ids_by_bank"}
                or (
                    n.attr == "from_request"
                    and isinstance(n.value, ast.Name)
                    and n.value.id == "NewRequestData"
                )
                for n in calls
            ), path


@pytest.mark.parametrize("enabled", [False, True])
def test_actual_blocks_are_scanned_once_only_for_p(api, enabled):
    path = ROOT / "vllm/v1/core/kv_cache_manager.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "KVCacheBlocks"
    )
    names = {"get_block_ids", "get_block_ids_by_bank", "get_allocation_mode"}
    methods = [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name in names and not n.decorator_list
    ]
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *methods,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), api)

    class Block:
        is_null = False

        def __init__(self, index):
            self.block_id = index
            self.bank_block_ids = (index, index + 1024) if enabled else None
            self.reads = 0

        @property
        def allocation_mode(self):
            self.reads += 1
            return "prefill_child" if enabled else "full_parent"

    rows = [Block(i) for i in range(1000)]
    blocks = NS(blocks=(rows,))
    for name in names:
        setattr(blocks, name, api[name].__get__(blocks))
    result = api["_make_new_request_data"](
        runner(enabled), NS(request_id="long"), blocks
    )
    assert result.block_ids == (list(range(1000)),)
    assert sum(block.reads for block in rows) == (1000 if enabled else 0)
    if enabled:
        assert result.block_ids_by_bank == (
            (list(range(1000)),),
            (list(range(1024, 2024)),),
        )
