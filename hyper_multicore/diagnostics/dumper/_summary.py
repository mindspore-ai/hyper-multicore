# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Compute compact summaries from owned Host snapshots."""

from __future__ import annotations

from collections.abc import Iterator
import csv
import json
import logging
from math import fsum, isfinite
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from hyper_multicore.diagnostics.dumper.api import _DumpInvocation
    from hyper_multicore.diagnostics.dumper._kernel import _KernelDumpTensorSnapshot
    from hyper_multicore.diagnostics.dumper._task import ElementRanges, _TaskDumpTensorSnapshot

_CHUNK_ELEMENTS = 256 * 1024
_PREVIEW_ELEMENTS = 4
_LOG_ROWS = 8
_LOGGER = logging.getLogger(__name__)
_FIELDS = (
    "rank", "window", "step", "invocation", "kernel", "direction", "level", "tags",
    "tensor", "ports", "capture_phase", "task_id", "owner", "core_type", "core_id",
    "stored_shape", "dtype", "range", "expert_rows", "source_offset_bytes", "stored_bytes",
    "full_bytes", "truncated", "elements", "nonfinite", "min", "max", "mean", "first_values", "valid_ranges",
)


def _summary_chunks(tensor: torch.Tensor, ranges: ElementRanges | None) -> Iterator[torch.Tensor]:
    """Iterate over captured elements without gathering the valid parts of a weight bank."""
    if tensor.is_complex():
        raise ValueError("dump summary requires a real-valued Tensor")
    flat = tensor.reshape(-1)
    if ranges is None:
        yield from flat.split(_CHUNK_ELEMENTS)
    else:
        for start, end in ranges:
            yield from flat[start:end].split(_CHUNK_ELEMENTS)


def _tensor_summary(tensor: torch.Tensor, ranges: ElementRanges | None = None) -> dict[str, Any]:
    """Reduce finite values in blocks to bound temporary Host allocations."""
    count = elements = 0
    first_values = []
    mins, maxes, means = [], [], []
    for chunk in _summary_chunks(tensor, ranges):
        elements += chunk.numel()
        if chunk.is_floating_point():
            values = chunk.to(torch.float64)
            values = values[torch.isfinite(values)] if values.numel() else values
        else:
            values = chunk
        if values.numel():
            count += values.numel()
            try:
                mins.append(values.min().item())
                maxes.append(values.max().item())
                means.append((values.to(torch.float64).mean().item(), values.numel()))
            except RuntimeError as error:
                raise ValueError(f"cannot summarize Host Tensor with dtype {tensor.dtype}: {error}") from error
        if len(first_values) < _PREVIEW_ELEMENTS:
            first_values.extend(chunk[:_PREVIEW_ELEMENTS - len(first_values)].tolist())
    return {
        "elements": elements, "nonfinite": elements - count,
        "min": min(mins) if count else None, "max": max(maxes) if count else None,
        "mean": fsum(mean * (n / count) for mean, n in means) if count else None,
        "first_values": [str(value) if isinstance(value, float) and not isfinite(value) else value
                         for value in first_values],
    }


def _kernel_summary_rows(
    snapshots: list[_KernelDumpTensorSnapshot],
    identity: dict[str, Any],
    expert_rows: list[tuple[int, int, int]],
    parameter_positions: tuple[int, ...],
) -> list[dict[str, Any]]:
    """Summarize each logical Kernel snapshot and retain all of its port aliases."""
    rows = []
    for snapshot in snapshots:
        graph_tensor = snapshot.graph_tensor
        tensor = snapshot.data
        row = {
            **identity, "tensor": graph_tensor.name, "capture_phase": snapshot.phase,
            "ports": [[port.stage, port.io, port.index] for port in graph_tensor.ports],
            "stored_shape": list(tensor.shape), "dtype": str(tensor.dtype).removeprefix("torch."),
            "range": "stored",
        }
        if graph_tensor.param_position in parameter_positions:
            valid_rows = expert_rows[-1][2] if expert_rows else 0
            if tensor.ndim < 1 or valid_rows > tensor.shape[0]:
                raise ValueError(f"expert rows exceed stored shape for Tensor {graph_tensor.name!r}")
            tensor = tensor[:valid_rows]
            row.update(range="expert_rows", expert_rows=expert_rows)
        rows.append({**row, **_tensor_summary(tensor)})
    return rows


def _task_summary_rows(
    snapshots: list[_TaskDumpTensorSnapshot], identity: dict[str, Any],
) -> list[dict[str, Any]]:
    """Summarize the bytes actually captured by each Task port."""
    rows = []
    for snapshot in snapshots:
        record, tensor = snapshot.record, snapshot.data
        rows.append({
            **identity, "tensor": record.tensor_name, "ports": [[record.stage, record.io, record.index]],
            "capture_phase": record.phase, "task_id": record.task_id, "owner": record.owner_id,
            "core_type": record.core_type, "core_id": record.core_id,
            "stored_shape": list(tensor.shape), "dtype": str(tensor.dtype).removeprefix("torch."),
            "range": record.coverage, "source_offset_bytes": record.source_offset_bytes,
            "stored_bytes": record.payload_bytes, "full_bytes": record.full_bytes, "truncated": record.truncated,
            **({"valid_ranges": list(record.valid_ranges)} if record.valid_ranges is not None else {}),
            **_tensor_summary(tensor, record.valid_ranges),
        })
    return rows


def _resolve_expert_rows(invocation: _DumpInvocation, limit: int | None) -> list[tuple[int, int, int]]:
    """Read component-owned row ranges only for Kernel snapshots eligible for a summary."""
    groups = invocation.row_groups
    if groups is not None:
        snapshots = invocation.kernel_snapshots if limit is None else invocation.kernel_snapshots[:limit]
        if any(s.graph_tensor.param_position in groups.parameter_positions for s in snapshots):
            return groups.resolve_rows()
    return []


def _summarize_invocation(
    invocation: _DumpInvocation, limit: int | None, expert_rows: list[tuple[int, int, int]],
) -> list[dict[str, Any]]:
    """Build context and statistics once for all selected summary outputs."""
    identity = {
        "rank": invocation.rank, "step": invocation.step, "invocation": invocation.invocation_id,
        "kernel": invocation.kernel_name, "direction": invocation.direction,
        "level": invocation.level.value, "tags": invocation.tags,
    }
    if invocation.task_snapshots:
        snapshots = invocation.task_snapshots
        return _task_summary_rows(snapshots if limit is None else snapshots[:limit], identity)
    snapshots = invocation.kernel_snapshots if limit is None else invocation.kernel_snapshots[:limit]
    groups = invocation.row_groups
    return _kernel_summary_rows(snapshots, identity, expert_rows, groups.parameter_positions if groups else ())


def _write_summary_csv(path: Path, rows: list[dict[str, Any]], window: int) -> None:
    """Place the complete CSV in the staged window only after a successful write."""
    with TemporaryDirectory(prefix=".summary.", dir=path.parent.parent) as directory:
        temporary = Path(directory) / path.name
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=_FIELDS)
            writer.writeheader()
            for row in sorted(rows, key=lambda entry: entry["invocation"]):
                writer.writerow({
                    **{key: json.dumps(value) if isinstance(value, (dict, list)) else value
                       for key, value in row.items()},
                    "window": window,
                })
        temporary.rename(path)


def _print_summary(rank: int, window: int, count: int, rows: list[dict[str, Any]]) -> None:
    """Print a bounded preview when requested, independent of the logger's level."""
    try:
        lines = [f"Dump summary rank={rank} window={window} showing={len(rows)}/{count}"]
        lines.extend(_format_summary_row(row) for row in rows)
        preview = "\n".join(lines)
    except MemoryError as error:
        _LOGGER.warning("Dump summary formatting failed for rank=%d window=%d: %s", rank, window, error)
        return
    try:
        print(preview, flush=True)
    except (ValueError, OSError, MemoryError) as error:
        _LOGGER.warning("Dump summary preview failed for rank=%d window=%d: %s", rank, window, error)


def _format_summary_row(row: dict[str, Any]) -> str:
    """Display the selected ports, then capture context and abbreviated values."""
    ports = ", ".join(f"{stage}.{io}[{index}]" for stage, io, index in row["ports"])
    context = f"  step={row['step']} invocation={row['invocation']} {ports}"
    if row["level"] == "task":
        owner = _format_summary_value(row["owner"])
        context += f" task={row['task_id']} owner={owner} worker={row['core_type']}/{row['core_id']}"
        if row["truncated"]:
            context += " truncated"
    stats = " ".join(f"{field}={_format_summary_value(row[field])}" for field in ("min", "max", "mean"))
    samples = ", ".join(_format_summary_value(value) for value in row["first_values"])
    return (
        f"{context} phase={row['capture_phase']} range={row['range']} shape={row['stored_shape']} {row['dtype']} "
        f"{stats} nonfinite={row['nonfinite']} values=[{samples}]"
    )


def _format_summary_value(value: int | float | str | None) -> str:
    """Round floats for the terminal while retaining exact values in CSV."""
    if isinstance(value, float):
        return f"{value:.6g}"
    return "-" if value is None else str(value)
