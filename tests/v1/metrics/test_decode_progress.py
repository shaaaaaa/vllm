# SPDX-License-Identifier: Apache-2.0
from io import StringIO
from types import SimpleNamespace as NS

import pytest

from vllm.v1.metrics import decode_progress


@pytest.fixture
def monitor(monkeypatch):
    clock = [0.0]
    lines = []
    monkeypatch.setattr(decode_progress, "time", NS(monotonic=lambda: clock[0]))
    monkeypatch.setattr(
        decode_progress, "logger", NS(info=lambda fmt, *args: lines.append(fmt % args))
    )
    progress = decode_progress.DecodeProgress(2)

    def tick(t, running=None, waiting=0, tokens=None, prompt=0, decode=True):
        clock[0] = t
        scheduler = (
            NS(num_running_reqs=running, num_waiting_reqs=waiting)
            if running is not None
            else None
        )
        iteration = (
            NS(
                num_generation_tokens=tokens,
                prompt_token_stats=NS(computed=prompt),
                inter_token_latencies_iter=[0.1] if decode and tokens else [],
            )
            if tokens is not None
            else None
        )
        progress.record(scheduler, iteration)
        progress.log()

    return tick, lines


def test_sustained_slowdown_reminder_and_recovery(monitor):
    tick, lines = monitor
    tick(0, 4)
    # DP2's three consecutive ten-second windows at 05:28 on November 24.
    for t, tokens in [(10, 39), (20, 8), (30, 31)]:
        tick(t, 4, tokens=tokens)
    assert len(lines) == 1 and "state=slow reason=low_decode_rate" in lines[0]
    assert "tokens_per_request_s=0.20" in lines[0]
    for t in range(40, 90, 10):
        tick(t)
    assert len(lines) == 2 and "state=reminder reason=no_engine_updates" in lines[1]
    tick(90, 4, tokens=600)
    assert "state=recovered" in lines[-1]
    tick(100, 0)
    assert len(lines) == 3


def test_waiting_without_completions_and_drain(monitor):
    tick, lines = monitor
    tick(0, 0, waiting=1)
    tick(10)
    tick(20)
    assert len(lines) == 1 and "waiting_for_admission" in lines[0]
    tick(25, 0)
    tick(40)
    assert "state=ended reason=queue_drained" in lines[-1]
    assert len(lines) == 2


def test_changing_load_uses_request_seconds_and_excludes_prompt_work(monitor):
    tick, lines = monitor
    tick(0, 1)
    tick(5, 4)
    tick(10, 4, tokens=250)  # 25 request-seconds, exactly 10 TPS.
    tick(20, 4, tokens=0, prompt=1024)
    tick(30, 4, tokens=0, prompt=1024)
    tick(40, 4, tokens=1)
    assert not lines
    tick(45, 2)
    tick(50, 2, tokens=6)
    assert len(lines) == 1 and "request_seconds=30.0" in lines[0]


def test_pressure_is_best_effort(monkeypatch):
    monkeypatch.setattr(decode_progress, "sys", NS(platform="linux"))

    def open_pressure(path, **kwargs):
        if path.endswith("/io"):
            raise PermissionError
        if path.endswith("/memory"):
            return StringIO("some avg10=invalid\n")
        return StringIO("some avg10=2.05 avg60=1.20 total=123\n")

    monkeypatch.setattr(decode_progress, "open", open_pressure, raising=False)
    assert decode_progress._host_pressure() == "cpu=2.05"
    monkeypatch.setattr(decode_progress, "sys", NS(platform="win32"))
    assert decode_progress._host_pressure() == "unavailable"


@pytest.mark.parametrize("mode", [None, "0", "1", "detail"])
def test_default_logger_registration_per_engine(monkeypatch, mode):
    from vllm.v1.metrics import loggers

    monkeypatch.setattr(loggers, "SERVING_PERF_CONFIGURED", mode is not None)
    monkeypatch.setattr(loggers, "SERVING_PERF_ENABLED", mode in ("1", "detail"))
    monkeypatch.setattr(loggers, "SERVING_PERF_DETAILED", mode == "detail")
    monkeypatch.setattr(loggers.PrometheusStatLogger, "__init__", lambda self, *a: None)
    config = NS(
        model_config=NS(runner_type="generate"),
        kv_transfer_config=None,
        observability_config=NS(cudagraph_metrics=False),
    )
    manager = loggers.StatLoggerManager(config, [0, 3], aggregate_engine_logging=True)
    actual = manager.stat_loggers
    assert isinstance(actual[-1], loggers.PrometheusStatLogger)
    assert len(actual) == {None: 2, "0": 1, "1": 2, "detail": 3}[mode]
    if mode is None:
        assert isinstance(actual[0], loggers.AggregatedLoggingStatLogger)
    if mode in ("1", "detail"):
        assert [
            s.progress.engine_index for s in actual[0].per_engine_stat_loggers.values()
        ] == [0, 3]
    if mode == "detail":
        assert isinstance(actual[1], loggers.AggregatedLoggingStatLogger)


def test_prefill_first_tokens_do_not_report_mixed_decode_or_recovery(monitor):
    tick, lines = monitor
    tick(0, 4)
    for t in (10, 20):
        tick(t, 4, tokens=4, prompt=16384, decode=False)
    assert lines == []
    for t in (30, 40):
        tick(t, 4, tokens=4, prompt=4096)
    assert len(lines) == 1 and "reason=mixed_prompt_and_decode" in lines[0]
    tick(50, 4, tokens=4, prompt=16384, decode=False)
    assert "state=progress reason=processing_prompt" in lines[-1]


@pytest.mark.parametrize("level", ["1", "detail"])
@pytest.mark.parametrize(
    ("runner_type", "prefill_node"), [("generate", "true"), ("pooling", "false")]
)
def test_decode_monitor_only_on_decode_workloads(
    monkeypatch, level, runner_type, prefill_node
):
    from vllm.v1.metrics import loggers

    monkeypatch.setenv("VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE", prefill_node)
    monkeypatch.setattr(loggers, "SERVING_PERF_CONFIGURED", True)
    monkeypatch.setattr(loggers, "SERVING_PERF_ENABLED", True)
    monkeypatch.setattr(loggers, "SERVING_PERF_DETAILED", level == "detail")
    monkeypatch.setattr(loggers.PrometheusStatLogger, "__init__", lambda self, *a: None)
    monkeypatch.setattr(
        loggers.AggregatedLoggingStatLogger, "__init__", lambda self, **k: None
    )
    config = NS(
        model_config=NS(runner_type=runner_type),
        kv_transfer_config=NS(kv_role="kv_both"),
    )
    actual = loggers.StatLoggerManager(
        config, [0, 1], aggregate_engine_logging=True
    ).stat_loggers
    expected = [loggers.AggregatedLoggingStatLogger] if level == "detail" else []
    assert list(map(type, actual)) == [*expected, loggers.PrometheusStatLogger]
