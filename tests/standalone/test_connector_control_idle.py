# SPDX-License-Identifier: Apache-2.0
"""Idle connector cleanup must not add a check to active scheduling."""

import ast
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[2]


def method(file, name):
    tree = ast.parse(file.read_text(encoding="utf-8"))
    node = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name
    )
    node.decorator_list = []
    prefix = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    ns = {}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[prefix, node], type_ignores=[])),
            str(file),
            "exec",
        ),
        ns,
    )
    return ns[name]


def test_active_scheduling_never_queries_connector_control():
    has_requests = method(ROOT / "vllm/v1/core/sched/interface.py", "has_requests")
    obj = NS(
        has_unfinished_requests=lambda: True,
        has_finished_requests=lambda: pytest.fail("active path consulted cleanup"),
        has_pending_connector_control=lambda: pytest.fail(
            "active path queried connector"
        ),
    )
    for _ in range(100):
        assert has_requests(obj)


def test_idle_work_follows_the_actual_multi_connector_release_queue():
    root = ROOT.parent
    impl_pending = method(
        root / "LMCache-NPU/lmcache/integration/vllm/vllm_v1_adapter.py",
        "has_pending_control",
    )
    dynamic_pending = method(
        root / "LMCache-NPU/lmcache/integration/vllm/lmcache_connector_v1.py",
        "has_pending_control",
    )
    multi_pending = method(
        root
        / "vllm-ascend/vllm_ascend/distributed/kv_transfer/ascend_multi_connector.py",
        "has_pending_control",
    )
    scheduler_pending = method(
        ROOT / "vllm/v1/core/sched/scheduler.py", "has_pending_connector_control"
    )
    has_requests = method(ROOT / "vllm/v1/core/sched/interface.py", "has_requests")
    impl = NS(_checkpoint_restore_releases=[("r", 1, 7)])
    impl.has_pending_control = lambda: impl_pending(impl)
    child = NS(_lmcache_engine=impl)
    child.has_pending_control = lambda: dynamic_pending(child)
    multi = NS(_connectors=[object(), child])
    multi.has_pending_control = lambda: multi_pending(multi)
    scheduler = NS(
        connector=multi,
        has_unfinished_requests=lambda: False,
        has_finished_requests=lambda: False,
    )
    scheduler.has_pending_connector_control = lambda: scheduler_pending(scheduler)
    assert has_requests(scheduler)
    impl._checkpoint_restore_releases.clear()
    assert not has_requests(scheduler)


def test_idle_scheduler_without_connector_remains_idle():
    pending = method(
        ROOT / "vllm/v1/core/sched/scheduler.py", "has_pending_connector_control"
    )
    assert not pending(NS(connector=None))
