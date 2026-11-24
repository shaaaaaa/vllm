# SPDX-License-Identifier: Apache-2.0
"""Per-engine progress from existing frontend counters, including stalled steps."""

import socket
import sys
import time
from typing import TYPE_CHECKING

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.metrics.stats import IterationStats, SchedulerStats

logger = init_logger(__name__)
_WINDOW_S = 10.0
_REMINDER_S = 60.0


def _host_pressure() -> str:
    """Read Linux host pressure only when emitting a slow-progress report."""
    readings = []
    if sys.platform.startswith("linux"):
        for resource in ("cpu", "memory", "io"):
            try:
                with open(f"/proc/pressure/{resource}", encoding="ascii") as stream:
                    # The first ("some") row gives the percentage of time stalled.
                    fields = stream.readline().split()
                if fields and fields[0] == "some":
                    value = next(
                        (f[6:] for f in fields if f.startswith("avg10=")), None
                    )
                    if value is not None:
                        readings.append(f"{resource}={float(value):.2f}")
            except (OSError, ValueError):
                continue
    return ",".join(readings) or "unavailable"


class DecodeProgress:
    def __init__(self, engine_index: int):
        self.engine_index = engine_index
        self.host = socket.gethostname()
        self.running = self.waiting = self.bad_windows = 0
        self.bad_since: float | None = None
        self.last_report: float | None = None
        self.last_token: float | None = None
        self._reset_window(time.monotonic())

    def _reset_window(self, now: float) -> None:
        self.started = self.accounted_at = now
        self.request_seconds = 0.0
        self.max_running = self.running
        self.tokens = self.prompt = self.batches = self.updates = 0
        self.has_decode_output = False

    def _account(self, now: float) -> None:
        self.request_seconds += self.running * (now - self.accounted_at)
        self.accounted_at = now

    def record(
        self, scheduler: "SchedulerStats | None", iteration: "IterationStats | None"
    ) -> None:
        if scheduler is not None:
            running, waiting = scheduler.num_running_reqs, scheduler.num_waiting_reqs
            if (running, waiting) != (self.running, self.waiting):
                now = time.monotonic()
                if not (self.running or self.waiting):
                    self._reset_window(now)
                    self.last_token = None
                self._account(now)
                self.running, self.waiting = running, waiting
                self.max_running = max(self.max_running, running)
                if not (running or waiting):
                    if self.last_report is not None and self.bad_since is not None:
                        logger.info(
                            "[DECODE_PROGRESS] dp=%d state=ended reason=queue_drained incident_s=%.1f",
                            self.engine_index,
                            now - self.bad_since,
                        )
                    self.last_report = None
                    self.bad_since = None
                    self.bad_windows = 0
            self.updates += 1
        if iteration is not None:
            self.batches += 1
            self.tokens += iteration.num_generation_tokens
            self.prompt += iteration.prompt_token_stats.computed
            # First tokens from prefill count toward generation_tokens as well.
            self.has_decode_output |= bool(iteration.inter_token_latencies_iter)
            if iteration.num_generation_tokens:
                self.last_token = time.monotonic()

    def log(self) -> None:
        if not (self.running or self.waiting):
            return
        now = time.monotonic()
        elapsed = now - self.started
        if elapsed < _WINDOW_S:
            return
        self._account(now)
        rate = self.tokens / self.request_seconds if self.request_seconds else 0.0
        target = 20 if self.max_running == 1 else 10
        reason = None
        if not self.tokens and not self.running and self.waiting:
            reason = "waiting_for_admission"
        elif not self.batches and not self.updates:
            reason = "no_engine_updates"
        elif not self.tokens and not self.prompt:
            reason = "no_generation_with_updates"
        elif self.tokens and self.has_decode_output and rate < target:
            reason = "mixed_prompt_and_decode" if self.prompt else "low_decode_rate"

        if reason and self.bad_since is None:
            self.bad_since = self.started
        incident_s = now - self.bad_since if self.bad_since is not None else 0.0
        if not reason:
            self.bad_since = None
        self.bad_windows = self.bad_windows + 1 if reason else 0
        if (
            reason
            and self.bad_windows >= 2
            and (self.last_report is None or now - self.last_report >= _REMINDER_S)
        ):
            state = "slow" if self.last_report is None else "reminder"
            self.last_report = now
        elif reason is None and self.last_report is not None:
            recovered = bool(self.tokens and self.has_decode_output)
            state = "recovered" if recovered else "progress"
            reason = "decode_rate_recovered" if recovered else "processing_prompt"
            self.last_report = None
        else:
            self._reset_window(now)
            return
        logger.info(
            "[DECODE_PROGRESS] dp=%d state=%s reason=%s incident_s=%.1f window_s=%.1f running=%d waiting=%d "
            "request_seconds=%.1f tokens_per_request_s=%.2f target=%d generation_tokens=%d "
            "computed_prompt_tokens=%d output_batches=%d scheduler_updates=%d last_token_age_s=%s "
            "host=%s host_psi_some_avg10_pct=%s",
            self.engine_index,
            state,
            reason,
            incident_s,
            elapsed,
            self.running,
            self.waiting,
            self.request_seconds,
            rate,
            target,
            self.tokens,
            self.prompt,
            self.batches,
            self.updates,
            "never" if self.last_token is None else round(now - self.last_token, 1),
            self.host,
            _host_pressure() if state in ("slow", "reminder") else "unmeasured",
        )
        self._reset_window(now)
