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
"""Whole-MegaKernel boundary snapshots and stage-only Tensor storage."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import torch

from hyper_multicore.diagnostics.dumper._layout import _SAFE_FILE_NAME, _stage_names


@dataclass(frozen=True)
class _TensorPort:
    """One internal operator port referring to a logical graph Tensor."""

    operator: str
    stage: str
    op_type: str
    io: str
    index: int


@dataclass(frozen=True)
class _GraphTensor:
    """One logical graph Tensor and its Kernel capture relationship."""

    name: str
    param_position: int
    declared_shape: tuple[int, ...]
    role: str
    producers: tuple[str, ...]
    consumers: tuple[str, ...]
    ports: tuple[_TensorPort, ...]


@dataclass
class _KernelDumpTensorSnapshot:
    """One Host-owned Tensor value plus export metadata."""

    graph_tensor: _GraphTensor
    phase: str
    original_dtype: str
    stored_dtype: str
    shape: tuple[int, ...]
    data: torch.Tensor


def _operator_type_name(operator: Any) -> str:
    op_type = operator.op_type
    value = getattr(op_type, "value", op_type)
    return str(value)


def _describe_graph_tensors(graph: Any) -> tuple[_GraphTensor, ...]:
    """Deduplicate ABI-bound graph tensors and classify their producers and consumers."""
    relationships: dict[tuple[str, int], dict[str, Any]] = {}
    operators = graph.topological_sort()
    stage_names = _stage_names(operators)
    for operator in operators:
        op_type = _operator_type_name(operator)
        for index, tensor in enumerate(operator.inputs):
            key = (tensor.name, int(tensor.param_position))
            relationship = relationships.setdefault(
                key,
                {
                    "shape": tuple(int(dim) for dim in tensor.shape),
                    "producers": [],
                    "consumers": [],
                    "ports": [],
                },
            )
            relationship["consumers"].append(operator.name)
            relationship["ports"].append(
                _TensorPort(operator.name, stage_names[operator.name], op_type, "input", index)
            )
        for index, tensor in enumerate(operator.outputs):
            key = (tensor.name, int(tensor.param_position))
            relationship = relationships.setdefault(
                key,
                {
                    "shape": tuple(int(dim) for dim in tensor.shape),
                    "producers": [],
                    "consumers": [],
                    "ports": [],
                },
            )
            relationship["producers"].append(operator.name)
            relationship["ports"].append(
                _TensorPort(operator.name, stage_names[operator.name], op_type, "output", index)
            )

    descriptions = []
    for (name, param_position), relationship in relationships.items():
        producers = tuple(relationship["producers"])
        consumers = tuple(relationship["consumers"])
        if not producers:
            role = "external_input"
        elif consumers:
            role = "stage_output"
        else:
            role = "final_output"
        descriptions.append(
            _GraphTensor(
                name=name,
                param_position=param_position,
                declared_shape=relationship["shape"],
                role=role,
                producers=producers,
                consumers=consumers,
                ports=tuple(relationship["ports"]),
            )
        )
    return tuple(sorted(descriptions, key=lambda tensor: (tensor.param_position, tensor.name)))


def _dtype_name(dtype: Any) -> str:
    return str(dtype).removeprefix("torch.")


def _snapshot_tensor(tensor: Any, graph_tensor: _GraphTensor, phase: str) -> _KernelDumpTensorSnapshot:
    if not torch.is_tensor(tensor):
        raise TypeError(
            f"dump Tensor {graph_tensor.name!r} at param_position={graph_tensor.param_position} "
            f"must map to torch.Tensor, got {type(tensor).__name__}"
        )
    host_tensor = tensor.detach().contiguous().cpu().clone()
    dtype_name = _dtype_name(host_tensor.dtype)
    return _KernelDumpTensorSnapshot(
        graph_tensor=graph_tensor,
        phase=phase,
        original_dtype=dtype_name,
        stored_dtype=dtype_name,
        shape=tuple(int(dim) for dim in tensor.shape),
        data=host_tensor,
    )


def _write_kernel_stage_files(
    snapshot: _KernelDumpTensorSnapshot,
    invocation_dir: Path,
    window_dir: Path,
    safe_name: str,
) -> tuple[Path, list[dict[str, Any]]]:
    """Save one Kernel dump Tensor snapshot under stages and hard-link its remaining ports."""
    metadata = []
    primary_path = None
    for port in snapshot.graph_tensor.ports:
        safe_stage = _SAFE_FILE_NAME.sub("_", port.stage)
        stage_dir = invocation_dir / "stages" / safe_stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        stage_file_name = f"{port.io}_{port.index:02d}_{safe_name}.pt"
        stage_path = stage_dir / stage_file_name
        if primary_path is None:
            torch.save(snapshot.data, stage_path)
            primary_path = stage_path
        else:
            try:
                os.link(primary_path, stage_path)
            except OSError as error:
                raise OSError(
                    f"failed to create stage-file hard link {stage_path} for {primary_path}"
                ) from error
        metadata.append(
            {
                "operator": port.operator,
                "stage": port.stage,
                "opType": port.op_type,
                "io": port.io,
                "index": port.index,
                "file": stage_path.relative_to(window_dir).as_posix(),
            }
        )
    if primary_path is None:
        boundary_dir = invocation_dir / "stages" / "__kernel_boundary__"
        boundary_dir.mkdir(parents=True, exist_ok=True)
        boundary_io = "input" if snapshot.phase == "before" else "output"
        file_name = f"{boundary_io}_{snapshot.graph_tensor.param_position:02d}_{safe_name}.pt"
        primary_path = boundary_dir / file_name
        torch.save(snapshot.data, primary_path)
    return primary_path, metadata
