# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

from vllm.distributed.kv_transfer import diagnostics


def test_cold_perf_events_are_request_scoped_and_one_shot(monkeypatch):
    monkeypatch.setattr(diagnostics, "_COLD_PERF_ENABLED", True)
    diagnostics._cold_perf_request_ids.clear()
    diagnostics._cold_perf_emitted.clear()
    log = Mock()
    monkeypatch.setattr(diagnostics.logger, "info", log)

    diagnostics.mark_cold_perf_requests(("cold",))
    diagnostics.log_cold_perf_event(
        "decoder_first_schedule", request_id="other", once=True
    )
    diagnostics.log_cold_perf_event(
        "decoder_first_schedule", request_id="cold", once=True
    )
    diagnostics.log_cold_perf_event(
        "decoder_first_schedule", request_id="cold", once=True
    )

    log.assert_called_once()
    assert '"req_id":"cold"' in log.call_args.args[1]

    diagnostics.forget_cold_perf_request("cold")
    assert not diagnostics.is_cold_perf_request("cold")

    diagnostics.mark_cold_perf_requests(("cold",))
    diagnostics.log_cold_perf_event(
        "decoder_first_schedule", request_id="cold", once=True
    )
    assert log.call_count == 2
