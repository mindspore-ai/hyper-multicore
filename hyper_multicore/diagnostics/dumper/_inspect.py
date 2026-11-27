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
"""Offline selection and comparison of saved Tensor snapshots."""

import json
from math import isfinite
from pathlib import Path
from typing import Any

import torch

from hyper_multicore.diagnostics.dumper._layout import _KERNEL_MANIFEST_VERSION, _TASK_MANIFEST_VERSION


def load_tensor(
    manifest_path: str | Path, *, invocation_id: int, tensor_name: str | None = None,
    port: tuple[str, str, int] | None = None, task_id: int | None = None,
    worker: tuple[str, int] | None = None,
) -> torch.Tensor:
    """Load one CPU Tensor from a trusted window manifest.

    Select by name or (stage, input/output, port index). When selecting a Task snapshot,
    task_id and (core type, core ID) can distinguish repeated ports.
    """
    if tensor_name is None and port is None:
        raise ValueError("select a Tensor by tensor_name or port")
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    version = manifest["schemaVersion"]
    is_task = manifest["level"] == "task"
    if version != _KERNEL_MANIFEST_VERSION and not (is_task and version == _TASK_MANIFEST_VERSION):
        raise ValueError(f"unsupported dump manifest schema {version} in {path}")
    invocation = next((call for call in manifest["invocations"] if call["invocationId"] == invocation_id), None)
    if invocation is None:
        raise ValueError(f"invocation {invocation_id} was not found in {path}")
    if is_task:
        matches = _task_files(invocation, tensor_name, port, task_id, worker)
    else:
        if task_id is not None or worker is not None:
            raise ValueError("task_id and worker apply to Task dump manifests")
        matches = _kernel_files(invocation, tensor_name, port)
    if len(matches) != 1:
        raise ValueError(f"expected one Tensor in invocation {invocation_id} in {path}, found {len(matches)}")
    return torch.load(path.parent / matches[0], map_location="cpu", weights_only=True)


def _kernel_files(
    invocation: dict[str, Any], tensor_name: str | None, port: tuple[str, str, int] | None,
) -> list[str]:
    """Select one logical Tensor even when several stage ports use it."""
    return [item["file"] for item in invocation["tensors"]
            if (tensor_name is None or item["name"] == tensor_name)
            and (port is None or any((p["stage"], p["io"], p["index"]) == port for p in item["ports"]))]


def _task_files(
    invocation: dict[str, Any], tensor_name: str | None, port: tuple[str, str, int] | None,
    task_id: int | None, worker: tuple[str, int] | None,
) -> list[str]:
    """Find Task files by operand identity and optional owner worker."""
    files = []
    for task in invocation["tasks"]:
        if task_id is not None and task["taskId"] != task_id:
            continue
        if worker is not None and (task["coreType"], task["coreId"]) != worker:
            continue
        files.extend(item["file"] for item in task["tensors"]
                     if (tensor_name is None or item["name"] == tensor_name)
                     and (port is None or (task["stage"], item["io"], item["index"]) == port))
    return files


def compare_tensors(reference: torch.Tensor, actual: torch.Tensor, *, rtol: float, atol: float) -> dict[str, Any]:
    """Compare already aligned CPU floating-point tensors with explicit tolerances.

    Relative error uses the reference magnitude. NaNs are mismatches, and equal
    signed infinities match. A mismatch involving a nonfinite value has infinite error.
    The first mismatch follows row-major index order and is reported separately
    from the maximum absolute error, which can fall within relative tolerance.
    """
    if reference.shape != actual.shape:
        raise ValueError(f"comparison requires aligned shapes, got {reference.shape} and {actual.shape}")
    for tensor in (reference, actual):
        if tensor.device.type != "cpu" or not tensor.is_floating_point():
            raise ValueError("comparison requires CPU floating-point tensors")
    if not (isfinite(rtol) and isfinite(atol) and rtol >= 0 and atol >= 0):
        raise ValueError("rtol and atol must be finite and nonnegative")
    shape = reference.shape
    reference, actual = reference.detach().flatten(), actual.detach().flatten()
    mismatches, worst_error, worst_offset = 0, 0.0, None
    first_mismatch = None
    elements_per_chunk = 256 * 1024
    for start in range(0, reference.numel(), elements_per_chunk):
        expected = reference[start:start + elements_per_chunk].to(torch.float64)
        observed = actual[start:start + elements_per_chunk].to(torch.float64)
        close = torch.isclose(observed, expected, rtol=rtol, atol=atol)
        mismatch = ~close
        count = int(mismatch.sum().item())
        if count and first_mismatch is None:
            first_mismatch = start + int(mismatch.to(torch.uint8).argmax().item())
        mismatches += count
        errors = (observed - expected).abs()
        errors.masked_fill_(torch.isnan(errors) & close, 0)
        errors.nan_to_num_(nan=float("inf"), posinf=float("inf"))
        error, offset = errors.max(dim=0)
        if float(error) > worst_error:
            worst_error, worst_offset = float(error), start + int(offset)
    result = {
        "elements": reference.numel(), "over_tolerance": mismatches, "max_abs_error": worst_error,
        "max_error_index": None, "reference_value": None, "actual_value": None,
        "first_mismatch_index": None, "first_mismatch_reference_value": None, "first_mismatch_actual_value": None,
    }
    if first_mismatch is not None:
        result.update(
            first_mismatch_index=_unravel_index(first_mismatch, shape),
            first_mismatch_reference_value=reference[first_mismatch].item(),
            first_mismatch_actual_value=actual[first_mismatch].item(),
        )
    if worst_offset is not None:
        result.update(
            max_error_index=_unravel_index(worst_offset, shape),
            reference_value=reference[worst_offset].item(),
            actual_value=actual[worst_offset].item(),
        )
    return result


def _unravel_index(offset: int, shape: torch.Size) -> tuple[int, ...]:
    """Express the flat offset in the aligned tensors' coordinate system."""
    indices = []
    for size in reversed(shape):
        offset, index = divmod(offset, size)
        indices.append(index)
    return tuple(reversed(indices))
