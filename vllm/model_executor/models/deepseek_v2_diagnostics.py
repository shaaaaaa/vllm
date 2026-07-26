# SPDX-License-Identifier: Apache-2.0
"""Exact, opt-in flight recorder for DeepSeek/GLM-5.1 forward diagnostics.

This module intentionally has no logging or host transfers in ``record_tensor``.
The model and attention implementation only clone device tensors while the
recorder is active.  The model runner performs the host transfer after forward
has completed.

The recorder is process-local. vLLM workers execute one model forward at a
time, so a small module-level state is preferable to changing model return
types or the production request protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class DeepseekV2Trace:
    metadata: dict[str, Any]
    tensors: list[tuple[str, torch.Tensor]] = field(default_factory=list)
    values: list[tuple[str, Any]] = field(default_factory=list)


_active_trace: DeepseekV2Trace | None = None


def begin_trace(metadata: dict[str, Any]) -> None:
    """Begin one trace, rejecting accidental nesting."""
    global _active_trace
    if _active_trace is not None:
        raise RuntimeError("DeepseekV2 diagnostic trace is already active")
    _active_trace = DeepseekV2Trace(metadata=dict(metadata))


def trace_is_active() -> bool:
    return _active_trace is not None


def record_tensor(label: str, tensor: torch.Tensor | None) -> None:
    """Snapshot a tensor exactly on its current device.

    ``clone`` is required because several attention/MoE paths reuse output
    buffers or update residual tensors in place.  Host transfer is deferred
    until ``end_trace`` so the forward is not synchronized at every boundary.
    """
    trace = _active_trace
    if trace is None or tensor is None:
        return
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{label}: expected torch.Tensor, got {type(tensor)!r}")
    trace.tensors.append((label, tensor.detach().clone()))


def record_value(label: str, value: Any) -> None:
    """Record small Python routing/configuration metadata."""
    trace = _active_trace
    if trace is not None:
        trace.values.append((label, value))


def end_trace() -> DeepseekV2Trace:
    """Finish and return the active trace."""
    global _active_trace
    if _active_trace is None:
        raise RuntimeError("No DeepseekV2 diagnostic trace is active")
    trace = _active_trace
    _active_trace = None
    return trace


def abort_trace() -> None:
    """Discard a partially collected trace after a forward exception."""
    global _active_trace
    _active_trace = None
