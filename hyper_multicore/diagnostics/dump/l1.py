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
"""Host planning and decoding for task-level Torch MegaKernel Tensor dumps."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from math import prod
import struct
from typing import Any

import torch

from hyper_multicore.diagnostics.common.graph import operator_diagnostic_name
from hyper_multicore.scheduler.config import (
    RuntimeConfigC,
    TaskAiCoreType,
    TensorDescC,
)


L1_DUMP_MAGIC = 0x4C314450
L1_DUMP_VERSION = 2
L1_DUMP_PHASE_BEFORE = 1
L1_DUMP_PHASE_AFTER = 2
L1_DIAGNOSTIC_MODE = 2
L1_DUMP_STATUS_PENDING = 0
L1_DUMP_STATUS_CAPTURED = 1
L1_DUMP_HEADER = struct.Struct("<8I")
L1_DUMP_RECORD = struct.Struct("<10I24x")
L1_DUMP_ALIGNMENT = 64
L1_DUMP_RECORDS_OFFSET = 64
L1_DUMP_MAX_RECORDS = 4096

_CORE_TYPE_NAMES = {
    TaskAiCoreType.TASK_AICORE_CUBE: "AIC",
    TaskAiCoreType.TASK_AICORE_VECTOR: "AIV",
    TaskAiCoreType.TASK_AICORE_MIX: "MIX",
}
def _align_up(value: int, alignment: int = L1_DUMP_ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def _dtype_name(dtype: Any) -> str:
    return str(dtype).removeprefix("torch.")


@dataclass(frozen=True)
class L1TensorView:
    """Resolved task operand view copied by the common Device worker.

    Args:
        source_offset_bytes: Byte offset from the selected ABI parameter.
        shape: Logical task-local shape.
        note: Optional semantic note emitted in the manifest.
    """

    source_offset_bytes: int
    shape: tuple[int, ...]
    note: str = ""


@dataclass
class L1TensorViewContext:
    """Information available to an optional operator-specific view resolver."""

    operator: Any
    task_id: int
    owner_id: int | None
    task_desc: Any
    tensor_desc: Any
    tensor_spec: Any
    io: str
    index: int
    parameters: Sequence[Any]
    runtime_config: RuntimeConfigC
    rank: int
    stage: str
    default_view: L1TensorView
    parameter_cache: dict[int, Any]
    core_id: int = 0
    owner_context: Any = None

    def parameter_cpu(self, position: int) -> Any:
        """Return one ABI parameter on CPU, caching the debug-only transfer.

        Args:
            position: Zero-based position in the native ABI parameter list.
        """
        if position not in self.parameter_cache:
            parameter = self.parameters[position]
            if not torch.is_tensor(parameter):
                raise TypeError(
                    f"L1 dump parameter {position} must be torch.Tensor, got {type(parameter).__name__}"
                )
            self.parameter_cache[position] = parameter.detach().contiguous().cpu()
        return self.parameter_cache[position]

    def make_view(
        self,
        shape: Sequence[int],
        *,
        source_offset_bytes: int | None = None,
        note: str = "",
    ) -> L1TensorView:
        """Build a validated view relative to this operand's ABI parameter.

        Args:
            shape: Local contiguous shape of the selected task operand.
            source_offset_bytes: Byte offset relative to the bound ABI tensor's data pointer.
            note: Optional explanation retained in Host-side capture metadata.
        """
        resolved_shape = tuple(int(dim) for dim in shape)
        if any(dim < 0 for dim in resolved_shape):
            raise ValueError(f"L1 dump view shape must be non-negative, got {resolved_shape}")
        resolved_offset = self.default_view.source_offset_bytes
        if source_offset_bytes is not None:
            resolved_offset = source_offset_bytes
        if not isinstance(resolved_offset, int) or isinstance(resolved_offset, bool) or resolved_offset < 0:
            raise ValueError(f"L1 dump source offset must be a non-negative integer, got {resolved_offset!r}")
        return L1TensorView(resolved_offset, resolved_shape, note)


@dataclass(frozen=True)
class _L1TaskLocation:
    """Device worker assignment for one stable RuntimeConfig task ID."""

    task_id: int
    core_type: str
    core_id: int


@dataclass(frozen=True)
class _L1RecordPlan:
    """Host metadata and fixed Device payload slot for one task operand."""

    task_id: int
    task_index: int
    stage: str
    owner_id: int | None
    owner_label: str
    core_type: str
    core_id: int
    io: str
    index: int
    phase: str
    parameter_position: int
    tensor_name: str
    dtype: Any
    source_offset_bytes: int
    actual_shape: tuple[int, ...]
    payload_offset: int
    payload_bytes: int
    full_bytes: int
    truncated: bool
    note: str
    record_offset: int
    worker_id: int = 0xFFFFFFFF


@dataclass
class _PreparedL1Dump:
    """Runtime override, Device buffer and immutable record plan for one launch."""

    runtime_config: Any
    buffer: Any
    records: tuple[_L1RecordPlan, ...]
    total_bytes: int


@dataclass(frozen=True)
class _L1Snapshot:
    """One decoded task-level payload ready for Torch serialization."""

    record: _L1RecordPlan
    original_dtype: str
    stored_dtype: str
    stored_shape: tuple[int, ...]
    data: torch.Tensor


def _operator_stage_name(operator: Any) -> str:
    return operator_diagnostic_name(operator)


def _task_locations(
    runtime_config: RuntimeConfigC, *, vector_worker_stride: int = 2, broadcast_tasks: bool = False,
) -> dict[Any, _L1TaskLocation]:
    """Map scheduled task IDs to actual workers, preserving broadcast worker identity."""
    active_core_count = int(runtime_config.num_workers) if broadcast_tasks else int(runtime_config.num_workers) // 2
    if active_core_count <= 0:
        raise ValueError("L1 dump requires RuntimeConfig.num_workers to identify active AIC/AIV workers")
    schedules = (
        (
            TaskAiCoreType.TASK_AICORE_CUBE,
            runtime_config.cube_task_indices,
            int(runtime_config.task_index_num[0]),
        ),
        (
            TaskAiCoreType.TASK_AICORE_VECTOR,
            runtime_config.vector_task_indices,
            int(runtime_config.task_index_num[1]),
        ),
        (
            TaskAiCoreType.TASK_AICORE_MIX,
            runtime_config.mix_task_indices,
            int(runtime_config.task_index_num[2]),
        ),
    )
    locations = {}
    for core_type, task_indices, task_count in schedules:
        if task_count < 0 or task_count > len(task_indices):
            raise ValueError(f"L1 dump task schedule count is outside RuntimeConfig capacity: {task_count}")
        for schedule_index in range(task_count):
            task_id = int(task_indices[schedule_index])
            if broadcast_tasks:
                for worker_id in range(int(runtime_config.num_workers)):
                    locations[(task_id, worker_id)] = _L1TaskLocation(task_id, "AIV", worker_id)
                continue
            worker_count = (int(runtime_config.num_workers) // vector_worker_stride
                            if core_type == TaskAiCoreType.TASK_AICORE_VECTOR else active_core_count)
            location = _L1TaskLocation(
                task_id=task_id,
                core_type=_CORE_TYPE_NAMES[core_type],
                core_id=schedule_index % worker_count,
            )
            previous = locations.setdefault(task_id, location)
            if previous != location:
                raise ValueError(f"L1 dump task {task_id} appears in multiple Device schedules")
    return locations


def _split_dimension(operator: Any, io: str, index: int) -> int | None:
    """Return the configured split axis for a selected operator port."""
    split_spec = operator.split_spec
    if io == "input":
        for input_index, dimension in split_spec.split_inputs or ():
            if input_index == index:
                return None if dimension == -1 else int(dimension)
        return None
    if index >= len(split_spec.split_output_dims):
        return None
    dimension = int(split_spec.split_output_dims[index])
    return None if dimension == -1 else dimension


def _default_tensor_view(
    operator: Any,
    task_desc: Any,
    tensor_desc: Any,
    tensor_spec: Any,
    io: str,
    index: int,
    *, allow_strided: bool = False,
) -> L1TensorView:
    """Resolve a contiguous static task range and clip its final partial tile."""
    if tensor_desc.dynamic_shape:
        raise ValueError(
            f"L1 dump stage {_operator_stage_name(operator)!r} Tensor {tensor_spec.name!r} has dynamic shape; "
            "configure OperatorNode.diagnostic_view_resolver"
        )
    shape = [int(dim) for dim in tensor_spec.shape]
    split_dimension = _split_dimension(operator, io, index)
    if split_dimension is not None and shape:
        if not 0 <= split_dimension < len(shape):
            raise ValueError(
                f"L1 dump split dimension {split_dimension} is invalid for Tensor {tensor_spec.name!r} shape {shape}"
            )
        width = shape[split_dimension]
        count = min(width, int(task_desc.task_split_value))
        if count < width and prod(shape[:split_dimension]) > 1 and not allow_strided:
            raise ValueError("L1 default view cannot copy a non-contiguous split; configure diagnostic_view_resolver")
        row_elements = prod(shape[split_dimension + 1:])
        if row_elements and width:
            base_offset = int(tensor_desc.base_ptr_offset)
            if base_offset % row_elements and not allow_strided:
                raise ValueError("L1 default split offset is not row-aligned; configure diagnostic_view_resolver")
            start = base_offset // row_elements % width
            count = min(count, width - start)
        shape[split_dimension] = count
    return L1TensorView(
        source_offset_bytes=int(tensor_desc.base_ptr_offset) * int(tensor_desc.data_type),
        shape=tuple(shape),
    )


def _selected(config: Any, location: _L1TaskLocation, task_desc: Any, stage: str, owner_id: int | None) -> bool:
    """Apply task, owner, stage and worker selection predicates."""
    if config.stages is not None and stage not in config.stages:
        return False
    if config.task_ids is not None and location.task_id not in config.task_ids:
        return False
    if config.stage_task_indices is not None and int(task_desc.task_index) not in config.stage_task_indices:
        return False
    if config.owner_ids is not None and owner_id not in config.owner_ids:
        return False
    if config.core_types is not None and location.core_type not in config.core_types:
        return False
    return config.core_ids is None or location.core_id in config.core_ids


def _resolve_record_view(
    *,
    operator: Any,
    task_id: int,
    owner_id: int | None,
    task_desc: Any,
    tensor_desc: Any,
    tensor_spec: Any,
    io: str,
    index: int,
    parameters: Sequence[Any],
    runtime_config: RuntimeConfigC,
    rank: int,
    stage: str,
    parameter_cache: dict[int, Any],
    core_id: int = 0,
    owner_context: Any = None,
) -> L1TensorView:
    """Resolve one selected port's Host view before allocating its payload slot."""
    resolver = getattr(operator, "diagnostic_view_resolver", None)
    default_view = _default_tensor_view(
        operator,
        task_desc,
        tensor_desc,
        tensor_spec,
        io,
        index,
        allow_strided=resolver is not None,
    ) if not tensor_desc.dynamic_shape else L1TensorView(
        int(tensor_desc.base_ptr_offset) * int(tensor_desc.data_type),
        tuple(int(dim) for dim in tensor_spec.shape),
    )
    if resolver is None:
        if tensor_desc.dynamic_shape:
            raise ValueError(
                f"L1 dump stage {stage!r} Tensor {tensor_spec.name!r} needs a diagnostic_view_resolver"
            )
        return default_view
    context = L1TensorViewContext(
        operator=operator,
        task_id=task_id,
        owner_id=owner_id,
        task_desc=task_desc,
        tensor_desc=tensor_desc,
        tensor_spec=tensor_spec,
        io=io,
        index=index,
        parameters=parameters,
        runtime_config=runtime_config,
        rank=rank,
        stage=stage,
        default_view=default_view,
        parameter_cache=parameter_cache,
        core_id=core_id,
        owner_context=owner_context,
    )
    if not callable(resolver):
        raise TypeError(f"L1 dump diagnostic_view_resolver for stage {stage!r} must be callable")
    view = resolver(context)
    if not isinstance(view, L1TensorView):
        raise TypeError(
            f"L1 dump resolver for stage {stage!r} must return L1TensorView, got {type(view).__name__}"
        )
    return view


def _validate_view_bounds(
    parameter: Any,
    view: L1TensorView,
    tensor_name: str,
) -> tuple[Any, int]:
    """Check dtype, contiguity, alignment and the full source range against its ABI tensor."""
    if not torch.is_tensor(parameter):
        raise TypeError(f"L1 dump Tensor {tensor_name!r} must map to torch.Tensor, got {type(parameter).__name__}")
    if not parameter.is_contiguous():
        raise ValueError(f"L1 dump Tensor {tensor_name!r} ABI parameter must be contiguous")
    if (isinstance(view.source_offset_bytes, bool) or not isinstance(view.source_offset_bytes, int)
            or not 0 <= view.source_offset_bytes < 1 << 64):
        raise ValueError(f"L1 dump Tensor {tensor_name!r} source offset must be a non-negative uint64")
    if any(isinstance(dim, bool) or not isinstance(dim, int) or dim < 0 for dim in view.shape):
        raise ValueError(f"L1 dump Tensor {tensor_name!r} view shape must contain non-negative integers")
    element_size = int(parameter.element_size())
    if view.source_offset_bytes % element_size:
        raise ValueError(f"L1 dump Tensor {tensor_name!r} source offset must align to its element byte size")
    full_bytes = prod(view.shape) * element_size
    parameter_bytes = int(parameter.numel()) * element_size
    if view.source_offset_bytes + full_bytes > parameter_bytes:
        raise ValueError(
            f"L1 dump view for Tensor {tensor_name!r} exceeds its ABI parameter: "
            f"offset={view.source_offset_bytes}, bytes={full_bytes}, parameter_bytes={parameter_bytes}"
        )
    return parameter.dtype, full_bytes


def _operator_ports(operator: Any, task: Any) -> tuple[Any, ...]:
    """Resolve Host ABI ports without modifying the native TaskDesc arrays."""
    if operator.diagnostic_host_ports:
        def descriptors(specs: Any) -> list[TensorDescC]:
            """Build Host-only source descriptors for explicitly bound ABI ports.

            Args:
                specs: Native-bound graph operands requiring Host descriptors.
            """
            return [TensorDescC(input_position=spec.param_position, data_type=spec.dtype_size,
                                dynamic_shape=int(spec.is_dynamic)) for spec in specs]

        return (("input", descriptors(operator.inputs), operator.inputs, len(operator.inputs)),
                ("output", descriptors(operator.outputs), operator.outputs, len(operator.outputs)))
    return (("input", task.inputs, operator.inputs, int(task.num_inputs)),
            ("output", task.outputs, operator.outputs, int(task.num_outputs)))


@dataclass
class _L1PlanBuilder:
    """Own one launch's record list, parameter cache and monotonically assigned slots."""

    runtime: Any
    parameters: Sequence[Any]
    config: Any
    rank: int
    broadcast: bool
    records: list[_L1RecordPlan] = field(default_factory=list)
    parameter_cache: dict[int, Any] = field(default_factory=dict)
    next_payload_offset: int = L1_DUMP_RECORDS_OFFSET + L1_DUMP_MAX_RECORDS * L1_DUMP_RECORD.size

    def add(
        self, operator: Any, location: _L1TaskLocation, stage: str, owner_id: int | None,
        io: str, index: int, tensor_desc: Any, tensor_spec: Any,
    ) -> None:
        """Validate and append one selected operand, before allocating any payload.

        Args:
            operator: Graph operator associated with the selected task.
            location: Task identity and worker placement in the native schedule.
            stage: Diagnostic stage name associated with the operator.
            owner_id: Optional resolved business owner, such as an Expert ID.
            io: Input or output capture boundary.
            index: Zero-based operand index within the selected input or output list.
            tensor_desc: Native task operand descriptor containing source addressing information.
            tensor_spec: Graph operand contract containing its name and ABI position.
        """
        task_id = location.task_id
        task = self.runtime.host_config.all_tasks[task_id]
        position = int(tensor_desc.input_position)
        if position != int(tensor_spec.param_position):
            raise ValueError(f"L1 task {task_id} {io}{index} parameter position disagrees with graph Tensor")
        if not 0 <= position < len(self.parameters):
            raise ValueError(f"L1 task {task_id} requires parameter {position}, got {len(self.parameters)} parameters")
        parameter = self.parameters[position]
        if torch.is_tensor(parameter) and parameter.element_size() != int(tensor_desc.data_type):
            raise ValueError(f"L1 task {task_id} Tensor {tensor_spec.name!r} ABI element byte size mismatch")
        view = _resolve_record_view(
            operator=operator, task_id=task_id, owner_id=owner_id, task_desc=task, tensor_desc=tensor_desc,
            tensor_spec=tensor_spec, io=io, index=index, parameters=self.parameters,
            runtime_config=self.runtime.host_config, rank=self.rank, stage=stage,
            parameter_cache=self.parameter_cache, core_id=location.core_id,
            owner_context=getattr(getattr(self.runtime, "spec", None), "owner_context", None),
        )
        dtype, full_bytes = _validate_view_bounds(parameter, view, tensor_spec.name)
        if self.broadcast and full_bytes == 0:
            return  # No Device worker is launched for this empty row shard.
        payload_bytes = full_bytes
        if self.config.max_bytes_per_tensor is not None and payload_bytes > self.config.max_bytes_per_tensor:
            element_size = int(parameter.element_size())
            payload_bytes = self.config.max_bytes_per_tensor // element_size * element_size
            if payload_bytes <= 0:
                raise ValueError(f"max_bytes_per_tensor={self.config.max_bytes_per_tensor} cannot store one element")
        self.records.append(_L1RecordPlan(
            task_id=task_id, task_index=int(task.task_index), stage=stage, owner_id=owner_id,
            owner_label=self.runtime.owner_label, core_type=location.core_type, core_id=location.core_id,
            io=io, index=index, phase="before" if io == "input" else "after", parameter_position=position,
            tensor_name=tensor_spec.name, dtype=dtype, source_offset_bytes=view.source_offset_bytes,
            actual_shape=view.shape, payload_offset=self.next_payload_offset, payload_bytes=payload_bytes,
            full_bytes=full_bytes, truncated=payload_bytes < full_bytes, note=view.note,
            record_offset=L1_DUMP_RECORDS_OFFSET + len(self.records) * L1_DUMP_RECORD.size,
            worker_id=location.core_id if self.broadcast else 0xFFFFFFFF,
        ))
        self.next_payload_offset = _align_up(self.next_payload_offset + payload_bytes)
        if len(self.records) > L1_DUMP_MAX_RECORDS:
            raise ValueError(f"L1 dump record count exceeds {L1_DUMP_MAX_RECORDS}")


def _build_l1_record_plans(
    runtime: Any, graph: Any, parameters: Sequence[Any], config: Any, rank: int,
) -> tuple[_L1RecordPlan, ...]:
    """Select task/port pairs, then compile their independent snapshot slots."""
    runtime_config = runtime.host_config
    operators_by_stage = {}
    for operator in graph.topological_sort():
        stage = _operator_stage_name(operator)
        if stage in operators_by_stage:
            raise ValueError(f"duplicate L1 diagnostic stage name {stage!r}")
        operators_by_stage[stage] = operator
    spec = getattr(runtime, "spec", None)
    broadcast = getattr(spec, "execution_mode", "event_driven") == "aiv_pipeline"
    locations = _task_locations(runtime_config, vector_worker_stride=getattr(spec, "vector_worker_stride", 2),
                                broadcast_tasks=broadcast)
    builder = _L1PlanBuilder(runtime, parameters, config, rank, broadcast)
    for location in sorted(locations.values(), key=lambda item: (item.task_id, item.core_id)):
        task = runtime_config.all_tasks[location.task_id]
        stage = runtime.task_stage_names.get(location.task_id)
        operator = operators_by_stage.get(stage)
        if operator is None:
            continue
        owner_id = runtime.metadata.task_owner_ids.get(location.task_id)
        if not _selected(config, location, task, stage, owner_id):
            continue
        for io, descriptors, specs, count in _operator_ports(operator, task):
            if io not in config.io:
                continue
            if count > len(specs):
                raise ValueError(f"L1 task {location.task_id} declares too many {io} ports for its graph stage")
            for index in range(count):
                if config.tensor_names is None or specs[index].name in config.tensor_names:
                    builder.add(operator, location, stage, owner_id, io, index, descriptors[index], specs[index])
    if not builder.records:
        raise ValueError("L1 dump filters selected no task Tensor records")
    return tuple(builder.records)


def _pack_l1_buffer(records: tuple[_L1RecordPlan, ...]) -> bytearray:
    """Encode the versioned header, padded record table and independent payload slots."""
    payload_offset = min(record.payload_offset for record in records)
    total_bytes = _align_up(max(record.payload_offset + record.payload_bytes for record in records))
    raw_buffer = bytearray(total_bytes)
    L1_DUMP_HEADER.pack_into(
        raw_buffer,
        0,
        L1_DUMP_MAGIC,
        L1_DUMP_VERSION,
        1,
        len(records),
        L1_DUMP_RECORD.size,
        L1_DUMP_RECORDS_OFFSET,
        payload_offset,
        total_bytes,
    )
    phase_values = {"before": L1_DUMP_PHASE_BEFORE, "after": L1_DUMP_PHASE_AFTER}
    for record in records:
        source_low = record.source_offset_bytes & 0xFFFFFFFF
        source_high = record.source_offset_bytes >> 32
        L1_DUMP_RECORD.pack_into(
            raw_buffer,
            record.record_offset,
            record.task_id,
            phase_values[record.phase],
            record.parameter_position,
            source_low,
            source_high,
            record.payload_offset,
            record.payload_bytes,
            L1_DUMP_STATUS_PENDING,
            record.task_index,
            record.worker_id,
        )
    return raw_buffer


def _prepare_l1_dump(
    runtime: Any,
    graph: Any,
    parameters: Sequence[Any],
    config: Any,
    *,
    rank: int,
    device: Any,
) -> _PreparedL1Dump:
    """Compile selected task views and allocate one Device snapshot buffer."""
    if runtime is None:
        raise ValueError("L1 Tensor dump requires the prepared MegaKernel runtime")
    records = _build_l1_record_plans(runtime, graph, parameters, config, rank)
    budget_bytes = config.device_memory_budget_mb * 1024 * 1024
    total_bytes = _align_up(max(record.payload_offset + record.payload_bytes for record in records))
    if total_bytes >= 1 << 32:
        raise ValueError("L1 dump buffer exceeds the 32-bit payload-offset ABI")
    if total_bytes > budget_bytes:
        raise MemoryError(
            f"L1 dump requires {total_bytes} Device bytes, exceeding "
            f"device_memory_budget_mb={config.device_memory_budget_mb} ({budget_bytes} bytes)"
        )
    raw_buffer = _pack_l1_buffer(records)
    device_buffer = torch.frombuffer(raw_buffer, dtype=torch.uint8).clone().to(device=device)
    runtime_config = runtime.normal_tensor.clone()
    runtime_config[RuntimeConfigC.diagnostic_mode.offset] = L1_DIAGNOSTIC_MODE
    return _PreparedL1Dump(
        runtime_config=runtime_config,
        buffer=device_buffer,
        records=records,
        total_bytes=len(raw_buffer),
    )


def _payload_tensor(
    host_buffer: torch.Tensor,
    record: _L1RecordPlan,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Decode one payload slot as an owned CPU Tensor without dtype conversion."""
    if record.payload_bytes == 0:
        tensor = torch.empty((0,), dtype=record.dtype)
    else:
        payload = host_buffer.narrow(0, record.payload_offset, record.payload_bytes)
        tensor = payload.view(record.dtype).clone()
    stored_shape = (tensor.numel(),) if record.truncated else record.actual_shape
    return tensor.reshape(stored_shape), stored_shape


def _decode_l1_dump(prepared: _PreparedL1Dump) -> tuple[_L1Snapshot, ...]:
    """Copy the completed Device buffer to Host and decode all fixed slots."""
    host_buffer = prepared.buffer.detach().contiguous().cpu()
    header = tuple(int(value) for value in host_buffer[:L1_DUMP_HEADER.size].view(torch.int32).tolist())
    if header[:2] != (L1_DUMP_MAGIC, L1_DUMP_VERSION):
        raise ValueError(f"invalid L1 dump buffer header: magic={header[0]:#x}, version={header[1]}")
    snapshots = []
    for record in prepared.records:
        status_offset = record.record_offset + 7 * 4
        status = int(host_buffer[status_offset:status_offset + 4].view(torch.int32).item())
        if status != L1_DUMP_STATUS_CAPTURED:
            raise RuntimeError(
                f"L1 dump record was not captured: task_id={record.task_id}, stage={record.stage!r}, "
                f"io={record.io}, index={record.index}, status={status}, header_enabled={header[2]}"
            )
        data, stored_shape = _payload_tensor(host_buffer, record)
        dtype_name = _dtype_name(data.dtype)
        snapshots.append(
            _L1Snapshot(
                record=record,
                original_dtype=dtype_name,
                stored_dtype=dtype_name,
                stored_shape=stored_shape,
                data=data,
            )
        )
    return tuple(snapshots)
