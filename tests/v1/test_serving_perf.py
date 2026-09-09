# SPDX-License-Identifier: Apache-2.0
"""Startup gate and propagation contract for PD serving diagnostics."""

import runpy
from pathlib import Path

import pytest

from vllm import envs
from vllm.ray.ray_env import get_env_vars_to_copy


@pytest.mark.parametrize(
    "mode,enabled",
    [
        ("", False),
        ("0", False),
        (" FALSE ", False),
        ("no", False),
        ("off", False),
        ("1", True),
        (" Detail ", True),
        ("DEVICE", True),
    ],
)
def test_startup_perf_gate(monkeypatch, mode, enabled):
    monkeypatch.setenv("PD_SERVING_PERF", mode)
    namespace = runpy.run_path(
        str(Path(__file__).parents[2] / "vllm/v1/serving_perf.py")
    )
    assert namespace["SERVING_PERF_ENABLED"] is enabled


def test_perf_variable_is_propagated_to_ray_workers(monkeypatch):
    monkeypatch.setenv("PD_SERVING_PERF", "device")
    assert envs.PD_SERVING_PERF == "device"
    assert "PD_SERVING_PERF" in get_env_vars_to_copy()


def test_old_perf_variable_does_not_enable_timing(monkeypatch):
    monkeypatch.delenv("PD_SERVING_PERF", raising=False)
    monkeypatch.setenv("LMCACHE_COLD_START_PERF", "1")
    namespace = runpy.run_path(
        str(Path(__file__).parents[2] / "vllm/v1/serving_perf.py")
    )
    assert namespace["SERVING_PERF_ENABLED"] is False
