"""Tests for the standalone DSA top-k tensor analyzer."""

from pathlib import Path

import pytest
import torch

from tools.analyze_dsa_topk import (
    LogFormatError,
    calculate_mtp_samples,
    calculate_overlap_samples,
    infer_base_seq_lengths,
    load_input_directory,
    parse_topk_tensor,
)


def _dump(
    directory: Path,
    *,
    layer: int,
    step: int,
    seq_len: int,
    values: list[list[int]],
    tp_rank: int = 0,
) -> Path:
    rows = len(values)
    topk = len(values[0])
    path = directory / (
        f"dsa_topk__layer-model_layers_{layer}_self_attn_attn"
        f"__tp-{tp_rank}__global-{tp_rank}__seq-{seq_len}"
        f"__step-{step:06d}__state-SpecDecoding__pid-401262"
        f"__rows-{rows}__k-{topk}.pt"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.tensor(values, dtype=torch.int64), path)
    return path


def test_loads_tp0_mtp_tensors_and_unions_rows(tmp_path):
    for step in range(3):
        for layer in range(2):
            _dump(
                tmp_path,
                layer=layer,
                step=step,
                seq_len=100 + step,
                values=[[step, step + 1], [step + 1, step + 2]],
            )
            _dump(
                tmp_path,
                layer=layer,
                step=step,
                seq_len=100 + step,
                values=[[999, 998], [997, 996]],
                tp_rank=1,
            )

    tensors, records = load_input_directory(tmp_path, tp_rank=0, topk=2, num_layers=2)

    assert len(tensors) == 6
    assert len(records) == 6
    assert records[0].positions == frozenset({0, 1, 2})
    assert records[0].row_count == 2
    assert records[-1].step == 2
    assert infer_base_seq_lengths(records) == {tmp_path.name: 100}
    assert calculate_mtp_samples(records, topk=2)[0] == [0.75, 0.75, 0.75]
    overlap = calculate_overlap_samples(records, max_previous_steps=2)
    assert overlap[(0, 1)] == pytest.approx([2 / 3, 2 / 3])
    assert overlap[(0, 2)] == pytest.approx([2 / 3])


def test_parses_filename_metadata(tmp_path):
    path = _dump(
        tmp_path,
        layer=64,
        step=412,
        seq_len=8601,
        values=[[1, 2], [2, 3]],
    )

    item = parse_topk_tensor(path, input_dir=tmp_path, topk=2, num_layers=78)

    assert item is not None
    assert item.layer_index == 64
    assert item.step == 412
    assert item.seq_len == 8601
    assert item.row_count == 2
    assert item.positions == ((1, 2), (2, 3))


def test_accepts_singleton_dimension_between_rows_and_topk(tmp_path):
    path = _dump(
        tmp_path,
        layer=64,
        step=412,
        seq_len=8601,
        values=[[1, 2], [2, 3]],
    )
    torch.save(torch.tensor([[[1, 2]], [[2, 3]]], dtype=torch.int64), path)

    item = parse_topk_tensor(path, input_dir=tmp_path, topk=2, num_layers=78)

    assert item is not None
    assert item.positions == ((1, 2), (2, 3))


def test_ignores_mtp_layer_beyond_configured_main_layers(tmp_path):
    path = _dump(
        tmp_path,
        layer=78,
        step=412,
        seq_len=8601,
        values=[[1, 2], [2, 3]],
    )

    item = parse_topk_tensor(path, input_dir=tmp_path, topk=2, num_layers=78)

    assert item is None


def test_uses_relative_parent_as_request_id(tmp_path):
    request_dir = tmp_path / "request-a"
    _dump(
        request_dir,
        layer=0,
        step=0,
        seq_len=100,
        values=[[1, 2], [2, 3]],
    )

    _, records = load_input_directory(tmp_path, tp_rank=0, topk=2, num_layers=1)

    assert records[0].request_id == "request-a"


def test_rejects_tensor_shape_that_disagrees_with_filename(tmp_path):
    path = _dump(
        tmp_path,
        layer=0,
        step=0,
        seq_len=100,
        values=[[1, 2], [2, 3]],
    )
    torch.save(torch.tensor([[1, 2]]), path)

    with pytest.raises(LogFormatError, match=r"expected \(2, 2\)"):
        parse_topk_tensor(path, input_dir=tmp_path, topk=2, num_layers=1)


def test_rejects_duplicate_rank0_step_layer(tmp_path):
    first = _dump(
        tmp_path,
        layer=0,
        step=0,
        seq_len=100,
        values=[[1, 2], [2, 3]],
    )
    duplicate = tmp_path / first.name.replace("__global-0", "__global-1")
    torch.save(torch.tensor([[1, 2], [2, 3]]), duplicate)

    with pytest.raises(LogFormatError, match="duplicate TP0 tensor"):
        load_input_directory(tmp_path, tp_rank=0, topk=2, num_layers=1)
