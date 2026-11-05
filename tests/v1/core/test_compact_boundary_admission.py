# SPDX-License-Identifier: Apache-2.0
"""Execute admission expressions and real compact index planning on the CPU."""

import ast
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[3] / "vllm/v1/core"


def schedule_allocation(length=81665, **overrides):
    path = ROOT / "sched/scheduler.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    # Extract the existing contiguous admission segment, including its call to
    # allocate_slots, checking its actual arguments rather than a duplicate formula.
    body = next(
        n.body
        for n in ast.walk(tree)
        if isinstance(n, ast.For | ast.While)
        and any(
            isinstance(x, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "compact_external_load"
                for t in x.targets
            )
            for x in n.body
        )
    )
    first = next(
        i
        for i, n in enumerate(body)
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "compact_external_load"
            for t in n.targets
        )
    )
    last = next(
        i
        for i, n in enumerate(body)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "new_blocks" for t in n.targets)
        and i > first
    )
    request = NS(
        num_tokens=length,
        num_prompt_tokens=length,
        num_preemptions=0,
        num_computed_tokens=0,
        dsa_compact_allocated=False,
    )
    for key in tuple(overrides):
        if key.startswith("request_"):
            setattr(request, key[8:], overrides.pop(key))
    allocate = Mock(return_value=object())
    ns = dict(
        self=NS(block_size=128, kv_cache_manager=NS(allocate_slots=allocate)),
        request=request,
        request_id="r",
        dsa_compact_external_load=True,
        bootstrap_full_hit=False,
        load_kv_async=True,
        num_new_local_computed_tokens=0,
        num_external_computed_tokens=length - 1,
        effective_lookahead_tokens=0,
        num_new_tokens=0,
        new_computed_blocks=None,
        num_encoder_tokens=0,
        SERVING_PERF_ENABLED=False,
    )
    ns.update(overrides)
    exec(
        compile(
            ast.Module(body=body[first : last + 1], type_ignores=[]), str(path), "exec"
        ),
        ns,
    )
    assert allocate.call_args.args[1] == 0  # No model computation in async load.
    return allocate.call_args.kwargs, request


@pytest.mark.parametrize("length,lookahead", [(81664, 0), (81665, 1), (81666, 0)])
def test_scheduler_reserves_boundary_slot_without_scheduling_computation(
    length, lookahead
):
    kwargs, request = schedule_allocation(length)
    assert kwargs["num_lookahead_tokens"] == lookahead
    assert kwargs["num_external_computed_tokens"] == length - 1
    assert kwargs["delay_cache_blocks"] is True
    assert kwargs["dsa_compact_external_load"] is True
    assert request.num_computed_tokens == 0


@pytest.mark.parametrize(
    "override",
    [
        {"load_kv_async": False},
        {"dsa_compact_external_load": False},
        {"request_num_preemptions": 1},
        {"request_num_computed_tokens": 128},
        {"num_new_local_computed_tokens": 128},
        {"request_num_prompt_tokens": 81664},
        {"num_external_computed_tokens": 80000},
    ],
)
def test_unrelated_admission_keeps_original_lookahead(override):
    kwargs, _ = schedule_allocation(**override)
    assert kwargs["num_lookahead_tokens"] == 0


@pytest.mark.parametrize("scratch_rows", [1, 2])
@pytest.mark.parametrize("free", [80, 81, 88, 89, 211])
def test_boundary_compact_bundle_cost_and_exact_capacity(scratch_rows, free):
    path = ROOT / "single_type_kv_cache_manager.py"
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
    }
    cls = ast.ClassDef(
        name="Manager",
        bases=[],
        keywords=[],
        body=[
            n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names
        ],
        decorator_list=[],
    )
    module = ast.parse("from __future__ import annotations")
    module.body.append(cls)
    ns = dict(cdiv=lambda a, b: (a + b - 1) // b)
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), ns)
    manager = ns["Manager"]()
    manager.block_pool = NS(blocks_per_bundle=2)
    manager.block_size = 128
    manager.scratch_blocks = scratch_rows * 2048 // 128
    kwargs, _ = schedule_allocation()
    reusable = kwargs["num_external_computed_tokens"]
    allocation_end = reusable + kwargs["num_lookahead_tokens"]
    latent = manager._compact_required_indices(
        reusable, allocation_end, reusable, force_compact=True
    )
    index_blocks = (allocation_end + 127) // 128
    needed = (len(latent) + 1) // 2 + (index_blocks + 8) // 9
    assert allocation_end == 81665 and index_blocks == 639
    assert needed == (81 if scratch_rows == 1 else 89)
    assert set(range(manager.scratch_blocks)).issubset(latent)
    assert 638 in latent and len(latent) < 639
    # Execute the production shared-pool capacity branch.
    path = ROOT / "kv_cache_coordinator.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    method = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "has_enough_free_blocks"
    )
    module = ast.parse("from __future__ import annotations")
    module.body.append(method)
    import time

    ns = dict(time=time, _last_starve_log=[time.monotonic()], logger=Mock())
    exec(compile(module, str(path), "exec"), ns)
    coordinator = NS(
        get_num_blocks_to_allocate_per_group=lambda *args: [len(latent), index_blocks],
        use_per_group_block_pools=False,
        use_dsa_shared_block_pool=True,
        single_type_managers=[
            NS(block_pool=NS(get_num_bundles_to_allocate=lambda n: (n + 1) // 2)),
            NS(block_pool=NS(get_num_bundles_to_allocate=lambda n: (n + 8) // 9)),
        ],
        dsa_shared_allocator=NS(free_bundle_count=free),
    )
    admitted = ns[method.name](
        coordinator, "r", allocation_end, ([], []), 0, reusable, reusable, True
    )
    assert admitted == (needed <= free)
    assert coordinator.dsa_shared_allocator.free_bundle_count == free
