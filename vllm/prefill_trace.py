# SPDX-License-Identifier: Apache-2.0
"""Low-overhead host-side timestamp points for prefill diagnosis.

This module deliberately records point events only. It never queries an
accelerator event, synchronizes a device, or attributes elapsed time to the
Python function that happens to expose a synchronization boundary.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)

TRACE_ENV = "VLLM_PREFILL_TRACE"
TRACE_PREFIX_ENV = "VLLM_PREFILL_TRACE_REQUEST_PREFIXES"
DEFAULT_TRACE_PREFIXES = ("prefill-", "cmpl-prefill-")
LOG_PREFIX = "[PREFILL_TRACE] "


@dataclass(frozen=True)
class PrefillStep:
    request_id: str
    computed_tokens: int
    scheduled_tokens: int
    prompt_tokens: int


@cache
def enabled() -> bool:
    raw = os.getenv(TRACE_ENV, "false").strip().lower()
    if raw == "true":
        return True
    if raw == "false":
        return False
    raise ValueError(f"{TRACE_ENV} must be 'true' or 'false', got {raw!r}")


def request_enabled(request_id: str | None) -> bool:
    if not request_id or not enabled():
        return False
    raw_prefixes = os.getenv(TRACE_PREFIX_ENV)
    prefixes = (
        tuple(prefix.strip() for prefix in raw_prefixes.split(",") if prefix.strip())
        if raw_prefixes is not None
        else DEFAULT_TRACE_PREFIXES
    )
    return any(request_id.startswith(prefix) for prefix in prefixes)


def point(event: str, request_id: str, **fields: Any) -> int | None:
    """Log one wall-clock point without touching accelerator state."""
    if not request_enabled(request_id):
        return None
    return _emit(event, request_id, time.time_ns(), fields)


def point_at(
    event: str,
    request_id: str,
    unix_ns: int,
    **fields: Any,
) -> int | None:
    """Log a point captured earlier, once its request ID is known."""
    if not request_enabled(request_id):
        return None
    return _emit(event, request_id, unix_ns, fields)


def _emit(
    event: str,
    request_id: str,
    unix_ns: int,
    fields: dict[str, Any],
) -> int:
    payload = {
        "event": event,
        "unix_ns": unix_ns,
        "request": request_id,
        "pid": os.getpid(),
        **fields,
    }
    logger.info(
        "%s%s", LOG_PREFIX, json.dumps(payload, separators=(",", ":"), sort_keys=True)
    )
    return unix_ns


def points_at(
    events: Iterable[tuple[str, str, int | None, dict[str, Any]]],
) -> int:
    """Write multiple already-captured points with one log operation."""
    payloads = []
    for event, request_id, unix_ns, fields in events:
        if unix_ns is None or not request_enabled(request_id):
            continue
        payloads.append(
            {
                "event": event,
                "unix_ns": unix_ns,
                "request": request_id,
                "pid": os.getpid(),
                **fields,
            }
        )
    if payloads:
        logger.info(
            "%s%s",
            LOG_PREFIX,
            json.dumps(payloads, separators=(",", ":"), sort_keys=True),
        )
    return len(payloads)


def timestamp_ns() -> int | None:
    """Capture a host point only when tracing is enabled."""
    return time.time_ns() if enabled() else None


def prefill_steps(
    scheduler_output: SchedulerOutput,
    prompt_lens: dict[str, int],
) -> tuple[PrefillStep, ...]:
    """Extract scheduled prefill work and retain prompt lengths across chunks."""
    computed_by_req: dict[str, int] = {}
    for request in scheduler_output.scheduled_new_reqs:
        prompt_token_ids = request.prompt_token_ids
        if prompt_token_ids is None:
            prompt_token_ids = request.prefill_token_ids
        if prompt_token_ids is not None:
            prompt_lens[request.req_id] = len(prompt_token_ids)
        computed_by_req[request.req_id] = int(request.num_computed_tokens)

    cached = scheduler_output.scheduled_cached_reqs
    computed_by_req.update(
        (request_id, int(computed))
        for request_id, computed in zip(
            cached.req_ids,
            cached.num_computed_tokens,
            strict=True,
        )
    )

    steps = []
    for request_id, scheduled in scheduler_output.num_scheduled_tokens.items():
        computed = computed_by_req.get(request_id)
        prompt_len = prompt_lens.get(request_id)
        if computed is None or prompt_len is None or computed >= prompt_len:
            continue
        steps.append(
            PrefillStep(
                request_id=request_id,
                computed_tokens=computed,
                scheduled_tokens=int(scheduled),
                prompt_tokens=prompt_len,
            )
        )
    return tuple(steps)


def step_fields(step: PrefillStep, chunk_size: int) -> dict[str, int]:
    chunk = step.computed_tokens // chunk_size + 1
    total_chunks = (step.prompt_tokens + chunk_size - 1) // chunk_size
    return {
        "chunk": chunk,
        "total_chunks": total_chunks,
        "computed_tokens": step.computed_tokens,
        "scheduled_tokens": step.scheduled_tokens,
        "prompt_tokens": step.prompt_tokens,
    }
