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
    assert namespace["SERVING_PERF_DETAILED"] is (
        mode.strip().lower() in ("detail", "device")
    )


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


@pytest.mark.parametrize(
    "mode,native,expected",
    [
        ("0", None, "WARNING"),
        ("1", None, "WARNING"),
        ("detail", None, None),
        ("device", None, None),
        ("1", "INFO", "INFO"),
        (None, None, None),
    ],
)
def test_native_log_default_precedes_library_init(monkeypatch, mode, native, expected):
    import os

    for key, value in (("PD_SERVING_PERF", mode), ("MC_LOG_LEVEL", native)):
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)
    runpy.run_path(str(Path(__file__).parents[2] / "vllm/env_override.py"))
    assert os.getenv("MC_LOG_LEVEL") == expected
