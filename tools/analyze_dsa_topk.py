#!/usr/bin/env python3
"""Analyze per-step DSA top-k tensors dumped by vLLM."""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

TOPK_FILE_RE = re.compile(
    r"^dsa_topk__layer-(?P<layer>.+?)__tp-(?P<tp>\d+)"
    r"__global-(?P<global_rank>\d+)__seq-(?P<seq_len>\d+)"
    r"__step-(?P<step>\d+)__state-(?P<state>.+?)__pid-(?P<pid>\d+)"
    r"__rows-(?P<rows>\d+)__k-(?P<k>\d+)\.pt$"
)
TENSOR_LAYER_INDEX_RE = re.compile(r"(?:^|_)layers_(\d+)(?:_|$)")
TOPK_LOG_RE = re.compile(
    r"\[DSA-TOPK\]\s+layer=(?P<layer>\S+)\s+row=(?P<row>\d+)\s+"
    r"req=(?P<req>\S+)\s+n_valid=(?P<n_valid>\d+)\s+"
    r"pos_head=\[(?P<positions>[^\]]*)\]"
)
TOPK_HEADER_RE = re.compile(
    r"\[DSA-TOPK\]\s+layer=(?P<layer>\S+)\s+row=(?P<row>\d+)\s+"
    r"req=(?P<req>\S+)\s+n_valid=(?P<n_valid>\d+)\s+pos_head=\["
)
LAYER_INDEX_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
TP_RANK_RE = re.compile(r"Worker_TP(\d+)")


class LogFormatError(RuntimeError):
    """Raised when strict reconstruction finds an inconsistent log."""

    def __init__(self, message: str, *, line_number: int | None = None):
        super().__init__(message)
        self.line_number = line_number


@dataclass(frozen=True)
class LogRow:
    line_number: int
    layer_name: str
    layer_index: int
    row: int
    request_id: str
    n_valid: int
    positions: tuple[int, ...]


@dataclass(frozen=True)
class UnionTopK:
    request_id: str
    step: int
    layer_name: str
    layer_index: int
    row_count: int
    positions: frozenset[int]
    seq_len: int | None = None


@dataclass(frozen=True)
class TensorTopK:
    path: Path
    request_id: str
    step: int
    seq_len: int
    layer_name: str
    layer_index: int
    tp_rank: int
    global_rank: int
    row_count: int
    topk: int
    positions: tuple[tuple[int, ...], ...]


def _fail(message: str, *, line_number: int | None = None) -> LogFormatError:
    prefix = f"line {line_number}: " if line_number is not None else ""
    return LogFormatError(prefix + message, line_number=line_number)


def _parse_log_rows(
    log_path: Path,
    *,
    tp_rank: int,
    topk: int,
    num_layers: int,
) -> list[LogRow]:
    rows: list[LogRow] = []
    with log_path.open("r", encoding="utf-8", errors="replace") as stream:
        for line_number, line in enumerate(stream, 1):
            if "[DSA-TOPK]" not in line:
                continue
            ranks = [int(value) for value in TP_RANK_RE.findall(line)]
            if not ranks or ranks[0] != tp_rank:
                continue
            if len(set(ranks)) > 1:
                continue
            headers = list(TOPK_HEADER_RE.finditer(line))
            if not headers:
                raise _fail(
                    "malformed [DSA-TOPK] header",
                    line_number=line_number,
                )
            if len(headers) > 1:
                continue
            if headers[0]["req"] == "?":
                continue
            grouped_headers: dict[tuple[str, int, str, int], list] = {}
            for header in headers:
                if header["req"] == "?":
                    continue
                key = (
                    header["layer"],
                    int(header["row"]),
                    header["req"],
                    int(header["n_valid"]),
                )
                grouped_headers.setdefault(key, []).append(header)

            for key, copies in grouped_headers.items():
                layer_name, row, request_id, n_valid = key
                if n_valid != topk:
                    raise _fail(
                        f"request {request_id!r} has n_valid={n_valid}, "
                        f"expected configured topk={topk}",
                        line_number=line_number,
                    )
                layer_match = LAYER_INDEX_RE.search(layer_name)
                if layer_match is None:
                    raise _fail(
                        f"cannot extract layer index from {layer_name!r}",
                        line_number=line_number,
                    )
                layer_index = int(layer_match.group(1))
                if not 0 <= layer_index < num_layers:
                    raise _fail(
                        f"layer index {layer_index} is outside [0, {num_layers})",
                        line_number=line_number,
                    )

                valid_positions: set[tuple[int, ...]] = set()
                errors: list[str] = []
                for header in copies:
                    match = TOPK_LOG_RE.match(line, header.start())
                    if match is None:
                        errors.append("record has no closing pos_head bracket")
                        continue
                    positions_text = match["positions"].strip()
                    try:
                        positions = (
                            tuple(
                                int(value.strip())
                                for value in positions_text.split(",")
                            )
                            if positions_text
                            else ()
                        )
                    except ValueError as error:
                        errors.append(f"invalid integer: {error}")
                        continue
                    if len(positions) != n_valid:
                        errors.append(
                            f"contains {len(positions)} values, expected {n_valid}"
                        )
                        continue
                    if any(position < 0 for position in positions):
                        errors.append("contains a negative top-k position")
                        continue
                    valid_positions.add(positions)

                if not valid_positions:
                    details = "; ".join(errors)
                    raise _fail(
                        f"request {request_id!r}, layer {layer_index}, "
                        f"row {row} has no complete copy: {details}",
                        line_number=line_number,
                    )
                if len(valid_positions) != 1:
                    raise _fail(
                        f"request {request_id!r}, layer {layer_index}, "
                        f"row {row} has conflicting complete copies",
                        line_number=line_number,
                    )
                rows.append(
                    LogRow(
                        line_number=line_number,
                        layer_name=layer_name,
                        layer_index=layer_index,
                        row=row,
                        request_id=request_id,
                        n_valid=n_valid,
                        positions=valid_positions.pop(),
                    )
                )
    if not rows:
        raise _fail(f"no [DSA-TOPK] records for Worker_TP{tp_rank} in {log_path}")
    return rows


def _format_log_context(log_path: Path, line_number: int) -> str:
    context: list[tuple[int, str]] = []
    first_line = max(1, line_number - 1)
    last_line = line_number + 1
    with log_path.open("r", encoding="utf-8", errors="replace") as stream:
        for current_line, text in enumerate(stream, 1):
            if current_line < first_line:
                continue
            if current_line > last_line:
                break
            context.append((current_line, text.rstrip("\r\n")))
    rendered = [f"log context: {log_path}"]
    for current_line, text in context:
        marker = ">" if current_line == line_number else " "
        rendered.append(f"{marker} {current_line}: {text}")
    return "\n".join(rendered)


def parse_log_rows(
    log_path: Path,
    *,
    tp_rank: int,
    topk: int,
    num_layers: int,
) -> list[LogRow]:
    try:
        return _parse_log_rows(
            log_path,
            tp_rank=tp_rank,
            topk=topk,
            num_layers=num_layers,
        )
    except LogFormatError as error:
        if error.line_number is None:
            raise
        context = _format_log_context(log_path, error.line_number)
        raise LogFormatError(
            f"{error}\n{context}",
            line_number=error.line_number,
        ) from error


def reconstruct_union_topk(
    rows: Sequence[LogRow],
    *,
    num_layers: int,
) -> list[UnionTopK]:
    """Group a complete layer cycle into one step and union its MTP rows."""

    # Split at every layer wrap, then validate each cycle.
    cycles: list[list[LogRow]] = []
    current: list[LogRow] = []
    previous_layer: int | None = None
    for row in rows:
        if previous_layer == num_layers - 1 and row.layer_index == 0:
            cycles.append(current)
            current = []
        elif previous_layer is not None and row.layer_index < previous_layer:
            raise _fail(
                f"layer wrapped from {previous_layer} to {row.layer_index}; "
                "only the final-layer to layer-0 wrap is valid",
                line_number=row.line_number,
            )
        current.append(row)
        previous_layer = row.layer_index
    if current:
        cycles.append(current)

    result: list[UnionTopK] = []
    request_steps: dict[str, int] = defaultdict(int)
    for cycle_index, cycle in enumerate(cycles):
        layer_rows: dict[int, list[LogRow]] = defaultdict(list)
        for row in cycle:
            layer_rows[row.layer_index].append(row)
        missing = sorted(set(range(num_layers)) - set(layer_rows))
        if missing:
            first_line = cycle[0].line_number
            raise _fail(
                f"step cycle {cycle_index} is incomplete; missing layers {missing}",
                line_number=first_line,
            )

        request_set: set[str] | None = None
        per_layer: dict[tuple[str, int], UnionTopK] = {}
        for layer_index in range(num_layers):
            entries = layer_rows[layer_index]
            names = {entry.layer_name for entry in entries}
            if len(names) != 1:
                raise _fail(
                    f"step cycle {cycle_index}, layer {layer_index} has "
                    f"inconsistent names {sorted(names)}",
                    line_number=entries[0].line_number,
                )
            requests_here = {
                entry.request_id for entry in entries if entry.request_id != "?"
            }
            if request_set is None:
                request_set = requests_here
                if not request_set:
                    raise _fail(
                        f"step cycle {cycle_index} has no real request in layer 0",
                        line_number=entries[0].line_number,
                    )
            elif requests_here != request_set:
                raise _fail(
                    f"step cycle {cycle_index}, layer {layer_index} request set "
                    f"{sorted(requests_here)} differs from layer 0 "
                    f"{sorted(request_set)}",
                    line_number=entries[0].line_number,
                )
            seen_rows: set[tuple[str, int]] = set()
            for request_id in sorted(requests_here):
                request_rows = [
                    entry for entry in entries if entry.request_id == request_id
                ]
                positions: set[int] = set()
                for entry in request_rows:
                    row_key = (request_id, entry.row)
                    if row_key in seen_rows:
                        raise _fail(
                            f"duplicate request/row {row_key} in step cycle "
                            f"{cycle_index}, layer {layer_index}",
                            line_number=entry.line_number,
                        )
                    seen_rows.add(row_key)
                    positions.update(entry.positions)
                per_layer[(request_id, layer_index)] = UnionTopK(
                    request_id=request_id,
                    step=-1,
                    layer_name=entries[0].layer_name,
                    layer_index=layer_index,
                    row_count=len(request_rows),
                    positions=frozenset(positions),
                )

        assert request_set is not None
        for request_id in sorted(request_set):
            step = request_steps[request_id]
            for layer_index in range(num_layers):
                item = per_layer[(request_id, layer_index)]
                result.append(
                    UnionTopK(
                        request_id=item.request_id,
                        step=step,
                        layer_name=item.layer_name,
                        layer_index=item.layer_index,
                        row_count=item.row_count,
                        positions=item.positions,
                    )
                )
            request_steps[request_id] += 1
    return result


def infer_base_seq_lengths(records: Sequence[UnionTopK]) -> dict[str, int]:
    bases: dict[str, int] = {}
    for item in records:
        if not item.positions:
            raise LogFormatError(
                f"request {item.request_id}, step {item.step}, layer "
                f"{item.layer_index} has an empty union top-k"
            )
        required = (
            item.seq_len - item.step
            if item.seq_len is not None
            else max(item.positions) - item.step + 1
        )
        bases[item.request_id] = max(bases.get(item.request_id, 0), required)
    return bases


def _summary(values: Iterable[float]) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        return {
            "mean": math.nan,
            "p10": math.nan,
            "p90": math.nan,
            "min": math.nan,
            "max": math.nan,
            "count": 0,
        }

    def percentile(percent: float) -> float:
        position = (len(ordered) - 1) * percent
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight

    return {
        "mean": sum(ordered) / len(ordered),
        "p10": percentile(0.10),
        "p90": percentile(0.90),
        "min": ordered[0],
        "max": ordered[-1],
        "count": len(ordered),
    }


def calculate_overlap_samples(
    records: Sequence[UnionTopK], max_previous_steps: int
) -> dict[tuple[int, int], list[float]]:
    by_series: dict[tuple[str, int], list[UnionTopK]] = defaultdict(list)
    for item in records:
        by_series[(item.request_id, item.layer_index)].append(item)
    samples: dict[tuple[int, int], list[float]] = defaultdict(list)
    for (_, layer_index), series in by_series.items():
        series.sort(key=lambda item: item.step)
        for index, current in enumerate(series):
            for previous_count in range(1, min(max_previous_steps, index) + 1):
                previous_union: set[int] = set()
                for previous in series[index - previous_count : index]:
                    previous_union.update(previous.positions)
                samples[(layer_index, previous_count)].append(
                    len(current.positions.intersection(previous_union))
                    / len(current.positions)
                )
    return samples


def calculate_region_samples(
    records: Sequence[UnionTopK],
    bases: dict[str, int],
    max_front_chunks: int,
    max_back_chunks: int,
    chunk_size: int,
) -> dict[tuple[int, int, int], list[float]]:
    samples: dict[tuple[int, int, int], list[float]] = defaultdict(list)
    for item in records:
        seq_len = bases[item.request_id] + item.step
        for front in range(1, max_front_chunks + 1):
            front_end = chunk_size * front
            for back in range(1, max_back_chunks + 1):
                back_start = max(0, seq_len - chunk_size * back)
                covered = sum(
                    position < front_end or position >= back_start
                    for position in item.positions
                )
                samples[(item.layer_index, front, back)].append(
                    covered / len(item.positions)
                )
    return samples


def calculate_mtp_samples(
    records: Sequence[UnionTopK], topk: int
) -> dict[int, list[float]]:
    samples: dict[int, list[float]] = defaultdict(list)
    for item in records:
        samples[item.layer_index].append(len(item.positions) / (topk * item.row_count))
    return samples


def _selected_layers(layer: str, records: Sequence[UnionTopK]) -> list[int]:
    available = sorted({item.layer_index for item in records})
    if layer == "all":
        return available
    try:
        index = int(layer)
    except ValueError:
        matches = {item.layer_index for item in records if item.layer_name == layer}
        if len(matches) != 1:
            raise ValueError(
                f"--layer {layer!r} did not match exactly one layer name"
            ) from None
        index = matches.pop()
    if index not in available:
        raise ValueError(f"--layer {index} is absent from reconstructed data")
    return [index]


def _aggregate_across_selected_layers(
    samples: dict[tuple[int, ...], list[float]],
    selected_layers: Sequence[int],
    coordinates: Iterable[tuple[int, ...]],
) -> dict[tuple[int, ...], dict[str, float]]:
    result = {}
    for coordinate in coordinates:
        if len(selected_layers) == 1:
            values = list(samples.get((selected_layers[0], *coordinate), []))
        else:
            # In all-layer mode the band describes variation between layers,
            # rather than mixing layer variation with step variation.
            values = [
                _summary(samples.get((layer_index, *coordinate), []))["mean"]
                for layer_index in selected_layers
                if samples.get((layer_index, *coordinate))
            ]
        result[coordinate] = _summary(values)
    return result


def _draw_band(
    axis,
    x: Sequence[int],
    summaries: Sequence[dict[str, float]],
    *,
    label: str,
) -> None:
    mean = [item["mean"] for item in summaries]
    p10 = [item["p10"] for item in summaries]
    p90 = [item["p90"] for item in summaries]
    minimum = [item["min"] for item in summaries]
    maximum = [item["max"] for item in summaries]
    line = axis.plot(x, mean, label=label)[0]
    axis.fill_between(x, p10, p90, color=line.get_color(), alpha=0.18)
    axis.plot(x, minimum, color=line.get_color(), alpha=0.35, linestyle=":")
    axis.plot(x, maximum, color=line.get_color(), alpha=0.35, linestyle=":")


def write_reconstructed_csv(path: Path, records: Sequence[UnionTopK]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "request_id",
                "step",
                "layer_index",
                "layer_name",
                "row_count",
                "union_size",
                "union_topk",
            ]
        )
        for item in records:
            writer.writerow(
                [
                    item.request_id,
                    item.step,
                    item.layer_index,
                    item.layer_name,
                    item.row_count,
                    len(item.positions),
                    " ".join(str(value) for value in sorted(item.positions)),
                ]
            )


def _write_summary_csv(
    path: Path,
    header: Sequence[str],
    rows: Iterable[Sequence[object]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([*header, "mean", "p10", "p90", "min", "max", "count"])
        writer.writerows(rows)


def _merge_partial_rank_rows(
    rows_by_rank: dict[int, list[LogRow]],
) -> tuple[list[LogRow], set[int]]:
    merged: dict[tuple[int, int, str, int], tuple[int, LogRow]] = {}
    used_ranks: set[int] = set()
    for rank, rank_rows in sorted(rows_by_rank.items()):
        step = 0
        previous_layer: int | None = None
        for row in rank_rows:
            if previous_layer is not None and row.layer_index < previous_layer:
                step += 1
            previous_layer = row.layer_index
            key = (step, row.layer_index, row.request_id, row.row)
            existing = merged.get(key)
            if existing is None:
                merged[key] = (rank, row)
                used_ranks.add(rank)
                continue
            existing_rank, existing_row = existing
            if (
                existing_row.layer_name != row.layer_name
                or existing_row.n_valid != row.n_valid
                or existing_row.positions != row.positions
            ):
                raise LogFormatError(
                    "conflicting clean TP copies for "
                    f"step={step}, layer={row.layer_index}, "
                    f"request={row.request_id!r}, row={row.row}: "
                    f"TP{existing_rank} line {existing_row.line_number} "
                    f"differs from TP{rank} line {row.line_number}"
                )
    ordered_rows = [
        item[1] for _, item in sorted(merged.items(), key=lambda pair: pair[0])
    ]
    return ordered_rows, used_ranks


def _load_single_log(
    log_path: Path,
    *,
    tp_rank: int | None,
    topk: int,
    num_layers: int,
) -> tuple[list[LogRow], list[UnionTopK], set[int]]:
    if tp_rank is None:
        ranks_in_file: set[int] = set()
        with log_path.open("r", encoding="utf-8", errors="replace") as stream:
            for line in stream:
                ranks_in_file.update(int(value) for value in TP_RANK_RE.findall(line))
        candidate_ranks = sorted(ranks_in_file)
        if not candidate_ranks:
            raise LogFormatError(f"{log_path} contains no Worker_TP rank prefixes")
    else:
        candidate_ranks = [tp_rank]

    if tp_rank is not None:
        rows = parse_log_rows(
            log_path,
            tp_rank=tp_rank,
            topk=topk,
            num_layers=num_layers,
        )
        records = reconstruct_union_topk(rows, num_layers=num_layers)
        return rows, records, {tp_rank}

    rows_by_rank: dict[int, list[LogRow]] = {}
    failures: list[tuple[int, Exception]] = []
    for candidate_rank in candidate_ranks:
        try:
            rows_by_rank[candidate_rank] = parse_log_rows(
                log_path,
                tp_rank=candidate_rank,
                topk=topk,
                num_layers=num_layers,
            )
        except (LogFormatError, ValueError) as error:
            failures.append((candidate_rank, error))
    if not rows_by_rank:
        summary = "\n".join(
            f"  TP{rank}: {str(error).splitlines()[0]}" for rank, error in failures
        )
        raise LogFormatError(
            f"{log_path}: no TP rank produced clean records:\n{summary}"
        )
    rows, used_ranks = _merge_partial_rank_rows(rows_by_rank)
    records = reconstruct_union_topk(rows, num_layers=num_layers)
    return rows, records, used_ranks


def _request_id(input_dir: Path, path: Path) -> str:
    relative_parent = path.parent.relative_to(input_dir)
    return (
        input_dir.name if relative_parent == Path(".") else relative_parent.as_posix()
    )


def parse_topk_tensor(
    path: Path,
    *,
    input_dir: Path,
    topk: int,
    num_layers: int,
) -> TensorTopK | None:
    """Load one rank-0 top-k dump; return None for other TP ranks."""

    match = TOPK_FILE_RE.fullmatch(path.name)
    if match is None:
        raise LogFormatError(f"malformed DSA top-k filename: {path.name}")
    tp_rank = int(match["tp"])
    if tp_rank != 0:
        return None

    layer_name = match["layer"]
    layer_match = TENSOR_LAYER_INDEX_RE.search(layer_name)
    if layer_match is None:
        raise LogFormatError(f"cannot extract layer index from {layer_name!r}")
    layer_index = int(layer_match.group(1))
    if layer_index >= num_layers:
        return None

    rows = int(match["rows"])
    file_topk = int(match["k"])
    if file_topk != topk:
        raise LogFormatError(
            f"{path} has k={file_topk}, expected configured topk={topk}"
        )
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("reading DSA top-k dumps requires PyTorch") from error
    tensor = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(tensor, torch.Tensor):
        raise LogFormatError(f"{path} does not contain a torch.Tensor")
    original_shape = tuple(tensor.shape)
    while tensor.ndim > 2 and tensor.shape[1] == 1:
        tensor = tensor.squeeze(1)
    if tensor.ndim != 2 or tuple(tensor.shape) != (rows, file_topk):
        raise LogFormatError(
            f"{path} has tensor shape {original_shape}, expected ({rows}, {file_topk})"
        )
    if tensor.is_floating_point() or tensor.is_complex():
        raise LogFormatError(f"{path} top-k indices must use an integer dtype")
    if torch.any(tensor < 0):
        raise LogFormatError(f"{path} contains a negative top-k position")

    return TensorTopK(
        path=path,
        request_id=_request_id(input_dir, path),
        step=int(match["step"]),
        seq_len=int(match["seq_len"]),
        layer_name=layer_name,
        layer_index=layer_index,
        tp_rank=tp_rank,
        global_rank=int(match["global_rank"]),
        row_count=rows,
        topk=file_topk,
        positions=tuple(tuple(int(value) for value in row) for row in tensor.tolist()),
    )


def load_input_directory(
    input_dir: Path,
    *,
    tp_rank: int | None,
    topk: int,
    num_layers: int,
) -> tuple[list[TensorTopK], list[UnionTopK]]:
    if not input_dir.is_dir():
        raise ValueError(f"input must be a directory: {input_dir}")
    if tp_rank not in (None, 0):
        raise ValueError("tensor dumps are analyzed from TP rank 0 only")
    paths = sorted(input_dir.rglob("dsa_topk__*.pt"))
    if not paths:
        raise ValueError(
            f"input directory contains no DSA top-k .pt files: {input_dir}"
        )

    tensors = [
        parsed
        for path in paths
        if (
            parsed := parse_topk_tensor(
                path,
                input_dir=input_dir,
                topk=topk,
                num_layers=num_layers,
            )
        )
        is not None
    ]
    if not tensors:
        raise LogFormatError(f"{input_dir} contains no TP rank 0 DSA top-k tensors")

    seen: set[tuple[str, int, int]] = set()
    records: list[UnionTopK] = []
    for item in sorted(
        tensors, key=lambda item: (item.request_id, item.step, item.layer_index)
    ):
        key = (item.request_id, item.step, item.layer_index)
        if key in seen:
            raise LogFormatError(
                "duplicate TP0 tensor for "
                f"request={item.request_id!r}, step={item.step}, "
                f"layer={item.layer_index}"
            )
        seen.add(key)
        records.append(
            UnionTopK(
                request_id=item.request_id,
                step=item.step,
                layer_name=item.layer_name,
                layer_index=item.layer_index,
                row_count=item.row_count,
                positions=frozenset(
                    position for row in item.positions for position in row
                ),
                seq_len=item.seq_len,
            )
        )
    return tensors, records


def analyze(args: argparse.Namespace) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "plotting requires matplotlib; install it with "
            "`python -m pip install matplotlib`"
        ) from error

    rows, records = load_input_directory(
        args.input_dir,
        tp_rank=args.tp_rank,
        topk=args.topk,
        num_layers=args.num_layers,
    )
    selected_layers = _selected_layers(args.layer, records)
    bases = infer_base_seq_lengths(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_reconstructed_csv(args.output_dir / "union_topk.csv", records)

    overlap = calculate_overlap_samples(records, args.n)
    overlap_summary = _aggregate_across_selected_layers(
        overlap, selected_layers, ((lag,) for lag in range(1, args.n + 1))
    )
    _write_summary_csv(
        args.output_dir / "previous_step_overlap.csv",
        ["previous_step_count"],
        (
            [
                lag,
                *(
                    overlap_summary[(lag,)][key]
                    for key in ("mean", "p10", "p90", "min", "max", "count")
                ),
            ]
            for lag in range(1, args.n + 1)
        ),
    )
    figure, axis = plt.subplots(figsize=(10, 6))
    x = list(range(1, args.n + 1))
    _draw_band(
        axis,
        x,
        [overlap_summary[(lag,)] for lag in x],
        label="all layers" if args.layer == "all" else f"layer {selected_layers[0]}",
    )
    axis.set(
        xlabel="previous step count",
        ylabel="union overlap rate",
        ylim=(0, 1),
        title="Current top-k overlap with previous-step union",
    )
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.output_dir / "previous_step_overlap.png", dpi=160)
    plt.close(figure)

    regions = calculate_region_samples(records, bases, args.u, args.v, args.chunk_size)
    region_summaries = {}
    for front in range(1, args.u + 1):
        region_summaries.update(
            _aggregate_across_selected_layers(
                regions,
                selected_layers,
                ((front, back) for back in range(1, args.v + 1)),
            )
        )
    _write_summary_csv(
        args.output_dir / "front_back_coverage.csv",
        ["front_chunk_count", "back_chunk_count"],
        (
            [
                front,
                back,
                *(
                    region_summaries[(front, back)][key]
                    for key in ("mean", "p10", "p90", "min", "max", "count")
                ),
            ]
            for front in range(1, args.u + 1)
            for back in range(1, args.v + 1)
        ),
    )
    figure, axis = plt.subplots(figsize=(11, 7))
    back_values = list(range(1, args.v + 1))
    for front in range(1, args.u + 1):
        _draw_band(
            axis,
            back_values,
            [region_summaries[(front, back)] for back in back_values],
            label=f"front={args.chunk_size}×{front}",
        )
    base_text = ", ".join(
        f"{request_id}={base}" for request_id, base in sorted(bases.items())
    )
    axis.set(
        xlabel="back chunk count (v)",
        ylabel="top-k coverage",
        ylim=(0, 1),
        title=f"Front/back union coverage; inferred base_seq_len: {base_text}",
    )
    axis.grid(alpha=0.3)
    axis.legend(ncol=max(1, min(4, args.u)))
    figure.tight_layout()
    figure.savefig(args.output_dir / "front_back_coverage.png", dpi=160)
    plt.close(figure)

    mtp = calculate_mtp_samples(records, args.topk)
    layers = sorted(mtp)
    mtp_summaries = [_summary(mtp[layer]) for layer in layers]
    _write_summary_csv(
        args.output_dir / "mtp_row_overlap_by_layer.csv",
        ["layer_index", "metric"],
        (
            row
            for layer, summary in zip(layers, mtp_summaries)
            for row in (
                [
                    layer,
                    "union_ratio",
                    *(
                        summary[key]
                        for key in ("mean", "p10", "p90", "min", "max", "count")
                    ),
                ],
                [
                    layer,
                    "overlap",
                    *(
                        (1 - summary[key])
                        if key in ("mean", "p10", "p90", "min", "max")
                        else summary[key]
                        for key in ("mean", "p90", "p10", "max", "min", "count")
                    ),
                ],
            )
        ),
    )
    figure, (union_axis, overlap_axis) = plt.subplots(
        2, 1, figsize=(12, 9), sharex=True
    )
    _draw_band(union_axis, layers, mtp_summaries, label="union ratio")
    overlap_summaries = [
        {
            key: (1 - value if key != "count" else value)
            for key, value in summary.items()
        }
        for summary in mtp_summaries
    ]
    # Complement reverses quantiles and extrema.
    for summary, union_summary in zip(overlap_summaries, mtp_summaries):
        summary["p10"] = 1 - union_summary["p90"]
        summary["p90"] = 1 - union_summary["p10"]
        summary["min"] = 1 - union_summary["max"]
        summary["max"] = 1 - union_summary["min"]
    _draw_band(overlap_axis, layers, overlap_summaries, label="overlap")
    union_axis.set(ylabel="|union| / (topk × rows)", ylim=(0, 1))
    overlap_axis.set(xlabel="layer index", ylabel="1 - union ratio", ylim=(0, 1))
    union_axis.grid(alpha=0.3)
    overlap_axis.grid(alpha=0.3)
    union_axis.legend()
    overlap_axis.legend()
    figure.suptitle("MTP row union ratio and overlap by layer")
    figure.tight_layout()
    figure.savefig(args.output_dir / "mtp_row_overlap_by_layer.png", dpi=160)
    plt.close(figure)

    with (args.output_dir / "inferred_base_seq_len.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["request_id", "inferred_base_seq_len"])
        writer.writerows(sorted(bases.items()))

    print(f"Loaded {len(rows)} TP0 top-k tensors")
    print(f"Built {len(records)} request/step/layer union top-k records")
    for request_id, base in sorted(bases.items()):
        print(f"inferred base_seq_len[{request_id}]={base}")
    print(f"Wrote results to {args.output_dir}")


def _parse_tp_rank(value: str) -> int | None:
    if value.lower() == "auto":
        return None
    try:
        rank = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "TP rank must be 'auto' or a non-negative integer"
        ) from error
    if rank < 0:
        raise argparse.ArgumentTypeError("TP rank must be non-negative")
    return rank


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_dir",
        type=Path,
        help="directory containing dsa_topk__*.pt tensor dumps",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("dsa_topk_stats"))
    parser.add_argument(
        "-n", type=int, default=10, help="maximum previous-step union window"
    )
    parser.add_argument(
        "-u", type=int, default=10, help="maximum number of front chunks"
    )
    parser.add_argument(
        "-v", type=int, default=10, help="maximum number of back chunks"
    )
    parser.add_argument(
        "--layer", default="all", help="'all', a layer index, or a complete layer name"
    )
    parser.add_argument("--num-layers", type=int, default=78)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument(
        "--tp-rank",
        default=0,
        type=_parse_tp_rank,
        metavar="auto|N",
        help="TP rank to use; tensor dumps currently support rank 0 only",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    for name in ("n", "u", "v", "num_layers", "topk", "chunk_size"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    analyze(args)


if __name__ == "__main__":
    main()
