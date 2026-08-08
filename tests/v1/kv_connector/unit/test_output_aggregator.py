# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.v1.outputs import (
    KVConnectorOutput,
    KVConnectorSaveCompletion,
    ModelRunnerOutput,
)

pytestmark = pytest.mark.cpu_test


class DummyModelRunnerOutput(ModelRunnerOutput):
    def __init__(
        self,
        finished_sending: set[str] | None = None,
        finished_recving: set[str] | None = None,
        invalid_block_ids: set[int] | None = None,
        decode_save_completions: list[KVConnectorSaveCompletion] | None = None,
        expected_finished_count: int = 0,
    ):
        self.kv_connector_output = KVConnectorOutput(
            finished_sending=finished_sending,
            finished_recving=finished_recving,
            invalid_block_ids=invalid_block_ids or set(),
            decode_save_completions=decode_save_completions or [],
            expected_finished_count=expected_finished_count,
        )

    def __repr__(self):
        return (
            f"DummyModelRunnerOutput("
            f"finished_sending={self.kv_connector_output.finished_sending},"
            f"finished_recving={self.kv_connector_output.finished_recving})"
            f"invalid_block_ids={self.kv_connector_output.invalid_block_ids})"
        )


def test_aggregate_workers_output():
    aggregator = KVOutputAggregator(expected_finished_count=2)

    output1 = DummyModelRunnerOutput()
    output2 = DummyModelRunnerOutput()

    aggregated = aggregator.aggregate([output1, output2])

    assert aggregated is output1
    aggregated = aggregated.kv_connector_output
    assert aggregated.finished_sending is None
    assert aggregated.finished_recving is None
    assert not aggregated.invalid_block_ids

    output1 = DummyModelRunnerOutput(
        finished_sending={"req1"}, finished_recving={"req2"}
    )
    output2 = DummyModelRunnerOutput(invalid_block_ids={1})

    aggregated = aggregator.aggregate([output1, output2])

    assert aggregated is output1
    aggregated = aggregated.kv_connector_output
    assert aggregated.finished_sending is None
    assert aggregated.finished_recving is None
    assert aggregated.invalid_block_ids == {1}

    output1 = DummyModelRunnerOutput(invalid_block_ids={2})
    output2 = DummyModelRunnerOutput(finished_sending={"req1"})

    aggregated = aggregator.aggregate([output1, output2])

    assert aggregated is output1
    aggregated = aggregated.kv_connector_output
    assert aggregated.finished_sending == {"req1"}
    assert aggregated.finished_recving is None
    assert aggregated.invalid_block_ids == {2}

    output1 = DummyModelRunnerOutput(invalid_block_ids={3, 4})
    output2 = DummyModelRunnerOutput(
        finished_recving={"req2"}, invalid_block_ids={4, 5}
    )

    aggregated = aggregator.aggregate([output1, output2])

    assert aggregated is output1
    aggregated = aggregated.kv_connector_output
    assert aggregated.finished_sending is None
    assert aggregated.finished_recving == {"req2"}
    assert aggregated.invalid_block_ids == {3, 4, 5}


def test_aggregate_workers_output_with_expected_finished_count():
    # We create the aggregator expecting to collect from 4 workers
    aggregator = KVOutputAggregator(expected_finished_count=4)
    assert aggregator._expected_finished_count == 4
    # Some request with default expected finished requests
    output1 = DummyModelRunnerOutput(finished_sending={"req1"})
    aggregated = aggregator.aggregate([output1])
    # still expecting to collect from 4 workers
    assert aggregator._send_remaining_count["req1"] == 3
    assert not aggregated.kv_connector_output.finished_sending
    assert not aggregated.kv_connector_output.finished_recving

    # Workers discover and find that in this setup they only need to
    # collect from 2
    output1 = DummyModelRunnerOutput(
        finished_sending={"req1"}, expected_finished_count=2
    )
    output2 = DummyModelRunnerOutput(
        finished_recving={"req2"}, expected_finished_count=2
    )
    output3 = DummyModelRunnerOutput(finished_recving={"req2"})
    # Req2 only needs 2 acks
    aggregated = aggregator.aggregate([output1, output2, output3])
    assert aggregated.kv_connector_output.expected_finished_count == 2

    assert not aggregated.kv_connector_output.finished_sending

    # Req2 is finished
    assert "req2" not in aggregator._recv_remaining_count
    assert aggregated.kv_connector_output.finished_recving == {"req2"}

    # Req1 is still waiting for 2 more acks (expected_finished_count has no effect)
    # NOTE: This is to showcase dynamic update. Workers are responsible for
    # ensuring "req1" termination in this case
    assert aggregator._send_remaining_count["req1"] == 2


def _decode_save_completion(
    *, worker_id: int, job_id: int, start: int, end: int, expected_count: int = 2
) -> KVConnectorSaveCompletion:
    return KVConnectorSaveCompletion(
        source="lmcache",
        request_id="req",
        generation=3,
        job_id=job_id,
        start=start,
        end=end,
        is_final=False,
        worker_id=worker_id,
        expected_count=expected_count,
    )


def test_decode_save_completion_waits_for_every_required_worker():
    aggregator = KVOutputAggregator(expected_finished_count=8)
    first_rank = DummyModelRunnerOutput(
        decode_save_completions=[
            _decode_save_completion(worker_id=0, job_id=2, start=256, end=512)
        ]
    )

    aggregated = aggregator.aggregate([first_rank])
    assert aggregated is not None
    assert aggregated.kv_connector_output.decode_save_completions == []

    second_rank = DummyModelRunnerOutput(
        decode_save_completions=[
            _decode_save_completion(worker_id=1, job_id=2, start=256, end=512)
        ]
    )
    aggregated = aggregator.aggregate([second_rank])
    assert aggregated is not None
    assert aggregated.kv_connector_output.decode_save_completions == [
        _decode_save_completion(
            worker_id=-1,
            job_id=2,
            start=256,
            end=512,
        )
    ]


def test_decode_save_completion_supports_single_saver_rank():
    aggregator = KVOutputAggregator(expected_finished_count=8)
    output = DummyModelRunnerOutput(
        decode_save_completions=[
            _decode_save_completion(
                worker_id=0,
                job_id=1,
                start=0,
                end=256,
                expected_count=1,
            )
        ]
    )

    aggregated = aggregator.aggregate([output])
    assert aggregated is not None
    [completion] = aggregated.kv_connector_output.decode_save_completions
    assert completion.worker_id == -1
    assert completion.expected_count == 1


def test_decode_save_completion_duplicate_is_idempotent():
    aggregator = KVOutputAggregator(expected_finished_count=8)
    completion = _decode_save_completion(
        worker_id=0,
        job_id=9,
        start=512,
        end=768,
        expected_count=1,
    )

    first = aggregator.aggregate(
        [DummyModelRunnerOutput(decode_save_completions=[completion, completion])]
    )
    assert first is not None
    assert len(first.kv_connector_output.decode_save_completions) == 1

    repeated = aggregator.aggregate(
        [DummyModelRunnerOutput(decode_save_completions=[completion])]
    )
    assert repeated is not None
    assert repeated.kv_connector_output.decode_save_completions == []


def test_kv_connector_output_merge_preserves_save_and_finish_outputs():
    completion = _decode_save_completion(
        worker_id=0,
        job_id=10,
        start=768,
        end=1024,
        expected_count=1,
    )
    merged = KVConnectorOutput.merge(
        KVConnectorOutput(
            finished_sending={"finished"},
            completed_decode_window_saves={"legacy": 512},
            decode_save_completions=[completion],
        ),
        KVConnectorOutput(
            finished_recving={"received"},
            invalid_block_ids={7},
            decode_save_completions=[completion],
        ),
    )

    assert merged.finished_sending == {"finished"}
    assert merged.finished_recving == {"received"}
    assert merged.invalid_block_ids == {7}
    assert merged.completed_decode_window_saves == {"legacy": 512}
    assert merged.decode_save_completions == [completion]
