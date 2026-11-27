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

from collections.abc import Collection
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import torch

from hyper_multicore.diagnostics.dumper._layout import _SAFE_FILE_NAME, _save_tensor, _stage_names


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
    data: torch.Tensor


def _operator_type_name(operator: Any) -> str:
    op_type = operator.op_type
    value = getattr(op_type, "value", op_type)
    return str(value)


def _describe_graph_tensors(
    graph: Any, *, stages: Collection[str] | None = None, io: Collection[str] | None = None,
) -> tuple[_GraphTensor, ...]:
    """Select stage ports while deriving each Tensor's capture phase from the full graph."""
    relationships: dict[tuple[str, int], dict[str, Any]] = {}
    operators = graph.topological_sort()
    stage_names = _stage_names(operators)
    for operator in operators:
        op_type = _operator_type_name(operator)
        stage = stage_names[operator.name]
        for port_io, tensors, relation in (
            ("input", operator.inputs, "consumers"),
            ("output", operator.outputs, "producers"),
        ):
            for index, tensor in enumerate(tensors):
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
                relationship[relation].append(operator.name)
                relationship["ports"].append(_TensorPort(operator.name, stage, op_type, port_io, index))

    descriptions = []
    for (name, param_position), relationship in relationships.items():
        ports = tuple(
            port for port in relationship["ports"]
            if (stages is None or port.stage in stages) and (io is None or port.io in io)
        )
        if not ports:
            continue
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
                ports=ports,
            )
        )
    return tuple(sorted(descriptions, key=lambda tensor: (tensor.param_position, tensor.name)))


def _snapshot_tensor(tensor: Any, graph_tensor: _GraphTensor, phase: str) -> _KernelDumpTensorSnapshot:
    if not torch.is_tensor(tensor):
        raise TypeError(
            f"dump Tensor {graph_tensor.name!r} at param_position={graph_tensor.param_position} "
            f"must map to torch.Tensor, got {type(tensor).__name__}"
        )
    host_tensor = tensor.detach().contiguous().cpu().clone()
    return _KernelDumpTensorSnapshot(graph_tensor, phase, host_tensor)


def _write_kernel_stage_files(
    snapshot: _KernelDumpTensorSnapshot,
    invocation_dir: Path,
    window_dir: Path,
    safe_name: str,
) -> list[dict[str, Any]]:
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
            _save_tensor(snapshot.data, stage_path)
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
    return metadata


def _write_kernel_snapshots(
    snapshots: list[_KernelDumpTensorSnapshot],
    invocation_dir: Path,
    window_dir: Path,
) -> dict[str, Any]:
    """Write logical tensor snapshots and their stage-port metadata."""
    tensor_metadata = []
    for snapshot in snapshots:
        safe_name = _SAFE_FILE_NAME.sub("_", snapshot.graph_tensor.name)
        port_metadata = _write_kernel_stage_files(
            snapshot,
            invocation_dir,
            window_dir,
            safe_name,
        )
        dtype_name = str(snapshot.data.dtype).removeprefix("torch.")
        tensor_metadata.append(
            {
                "name": snapshot.graph_tensor.name,
                "paramPosition": snapshot.graph_tensor.param_position,
                "role": snapshot.graph_tensor.role,
                "capturePhase": snapshot.phase,
                "producers": list(snapshot.graph_tensor.producers),
                "consumers": list(snapshot.graph_tensor.consumers),
                "ports": port_metadata,
                "declaredShape": list(snapshot.graph_tensor.declared_shape),
                "actualShape": list(snapshot.data.shape),
                "originalDtype": dtype_name,
                "storedDtype": dtype_name,
                "file": port_metadata[0]["file"],
            }
        )
    return {
        "tensorCount": len(tensor_metadata),
        "tensors": tensor_metadata,
    }
