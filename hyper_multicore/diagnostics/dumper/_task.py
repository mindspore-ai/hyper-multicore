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
"""Host planning, decoding and storage for task-level Torch MegaKernel Tensor dumps."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from math import prod
from pathlib import Path
import struct
from typing import Any

import torch

from hyper_multicore.diagnostics.common.call import OwnerIds
from hyper_multicore.diagnostics.common.runtime import DeviceDiagnosticMode, PreparedDiagnosticRuntime
from hyper_multicore.diagnostics.dumper._layout import _SAFE_FILE_NAME, _save_tensor, _stage_names
from hyper_multicore.scheduler.config import (
    RuntimeConfigC,
    TaskAiCoreType,
    TensorDescC,
)
from hyper_multicore.scheduler.worker_queues import EXPLICIT_QUEUE_LAYOUT, split_worker_queues


TASK_DUMP_MAGIC = 0x4C314450
TASK_DUMP_VERSION = 2
TASK_DUMP_PHASE_BEFORE = 1
TASK_DUMP_PHASE_AFTER = 2
TASK_DUMP_STATUS_PENDING = 0
TASK_DUMP_STATUS_CAPTURED = 1
TASK_DUMP_HEADER = struct.Struct("<8I")
TASK_DUMP_RECORD = struct.Struct("<10I24x")
TASK_DUMP_ALIGNMENT = 64
TASK_DUMP_RECORDS_OFFSET = 64
TASK_DUMP_MAX_RECORDS = 4096
_TASK_DUMP_ANY_WORKER = 0xFFFFFFFF

_CORE_TYPE_NAMES = {
    TaskAiCoreType.TASK_AICORE_CUBE: "AIC",
    TaskAiCoreType.TASK_AICORE_VECTOR: "AIV",
    TaskAiCoreType.TASK_AICORE_MIX: "MIX",
}
def _align_up(value: int, alignment: int = TASK_DUMP_ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


ElementRanges = tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class TaskDumpTensorView:
    """Continuous Tensor range selected for a Task dump port.

    Args:
        source_offset_bytes: Byte offset from the selected ABI parameter.
        shape: Logical task-local shape.
        note: Optional semantic note emitted in the manifest.
        coverage: Whether the view represents the planned task, a first tile,
            or a range whose relation to the executed task is unknown.
        valid_ranges: Optional sorted, disjoint half-open element ranges in the flattened view.
    """

    source_offset_bytes: int
    shape: tuple[int, ...]
    note: str = ""
    coverage: str = "task_view"
    valid_ranges: ElementRanges | None = None


class UnsupportedTaskDumpPort(ValueError):
    """A known operand for which Task dump cannot provide a reliable view."""


@dataclass
class TaskDumpTensorViewContext:
    """Per-port addressing context. owner_id is the schedule owner used for addressing.

    owner_ids maps schedule slots to business owners for this invocation, when supplied.
    """

    operator: Any
    task_id: int
    owner_id: int | None
    task_desc: Any
    tensor_desc: Any
    tensor_spec: Any
    io: str
    index: int
    parameters: Sequence[Any]
    diagnostic_runtime_config: RuntimeConfigC
    rank: int
    stage: str
    default_view: TaskDumpTensorView
    parameter_cache: dict[int, Any]
    core_id: int = 0
    owner_context: Any = None
    owner_ids: OwnerIds | None = None

    def parameter_cpu(self, position: int) -> Any:
        """Return one ABI parameter on CPU, caching the debug-only transfer.

        Args:
            position: Zero-based position in the native ABI parameter list.
        """
        if position not in self.parameter_cache:
            parameter = self.parameters[position]
            if not torch.is_tensor(parameter):
                raise TypeError(
                    f"Task dump parameter {position} must be torch.Tensor, got {type(parameter).__name__}"
                )
            self.parameter_cache[position] = parameter.detach().contiguous().cpu()
        return self.parameter_cache[position]

    def make_view(
        self,
        shape: Sequence[int],
        *,
        source_offset_bytes: int | None = None,
        note: str = "",
        coverage: str = "task_view",
        valid_ranges: ElementRanges | None = None,
    ) -> TaskDumpTensorView:
        """Build a validated view relative to this operand's ABI parameter.

        Args:
            shape: Local contiguous shape of the selected task operand.
            source_offset_bytes: Byte offset relative to the bound ABI tensor's data pointer.
            note: Optional explanation retained in Host-side capture metadata.
            coverage: Extent of the task covered by this view.
            valid_ranges: Optional flattened element intervals to include in statistics.
        """
        resolved_shape = tuple(int(dim) for dim in shape)
        if any(dim < 0 for dim in resolved_shape):
            raise ValueError(f"Task dump view shape must be non-negative, got {resolved_shape}")
        resolved_offset = self.default_view.source_offset_bytes
        if source_offset_bytes is not None:
            resolved_offset = source_offset_bytes
        if not isinstance(resolved_offset, int) or isinstance(resolved_offset, bool) or resolved_offset < 0:
            raise ValueError(f"Task dump source offset must be a non-negative integer, got {resolved_offset!r}")
        return TaskDumpTensorView(resolved_offset, resolved_shape, note, coverage, valid_ranges)


@dataclass(frozen=True)
class _TaskLocation:
    """Device worker assignment for one stable RuntimeConfig task ID."""

    task_id: int
    core_type: str
    core_id: int


@dataclass(frozen=True)
class _TaskDumpRecordPlan:
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
    capture_worker_id: int | None = None
    coverage: str = "task_view"
    valid_ranges: ElementRanges | None = None


@dataclass
class _PreparedTaskDump:
    """Runtime override, Device buffer and immutable record plan for one launch."""

    runtime_config_tensor: Any
    buffer: Any
    records: tuple[_TaskDumpRecordPlan, ...]
    total_bytes: int


@dataclass(frozen=True)
class _TaskDumpTensorSnapshot:
    """One decoded task-level payload ready for Torch serialization."""

    record: _TaskDumpRecordPlan
    data: torch.Tensor


def _iter_task_locations(
    runtime_config: RuntimeConfigC, *, vector_worker_stride: int = 2, broadcast_tasks: bool = False,
) -> Iterator[_TaskLocation]:
    """Yield the device worker assignments from the validated diagnostic schedule."""
    num_workers = runtime_config.num_workers
    cube_worker_count = num_workers // 2
    if (num_workers if broadcast_tasks else cube_worker_count) <= 0:
        raise ValueError("Task dump requires RuntimeConfig.num_workers to identify active AIC/AIV workers")
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
    if broadcast_tasks:
        for _, task_indices, task_count in schedules:
            for task_id in task_indices[:task_count]:
                for worker_id in range(num_workers):
                    yield _TaskLocation(task_id, "AIV", worker_id)
        return

    if runtime_config.queue_layout == EXPLICIT_QUEUE_LAYOUT:
        for (core_type, task_indices, _), queues in zip(schedules[:2], split_worker_queues(runtime_config)):
            for worker_id, queue in enumerate(queues):
                for position in queue.positions():
                    yield _TaskLocation(task_indices[position], _CORE_TYPE_NAMES[core_type], worker_id)
        return

    for core_type, task_indices, task_count in schedules:
        is_vector = core_type == TaskAiCoreType.TASK_AICORE_VECTOR
        worker_stride = vector_worker_stride if is_vector else 1
        worker_count = num_workers // worker_stride if is_vector else cube_worker_count
        for position, task_id in enumerate(task_indices[:task_count]):
            # Paired AIV scheduling assigns its logical slots to odd device worker IDs.
            worker_id = (position % worker_count) * worker_stride + (worker_stride - 1)
            yield _TaskLocation(task_id, _CORE_TYPE_NAMES[core_type], worker_id)


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
) -> TaskDumpTensorView:
    """Resolve a contiguous static task range and clip its final partial tile."""
    shape = [int(dim) for dim in tensor_spec.shape]
    split_dimension = _split_dimension(operator, io, index)
    if split_dimension is not None and shape:
        if not 0 <= split_dimension < len(shape):
            raise ValueError(
                f"Task dump split dimension {split_dimension} is invalid for Tensor {tensor_spec.name!r} shape {shape}"
            )
        width = shape[split_dimension]
        count = min(width, int(task_desc.task_split_value))
        if count < width and prod(shape[:split_dimension]) > 1 and not allow_strided:
            raise ValueError(
                "Task default view cannot copy a non-contiguous split; configure task_dump_tensor_view_resolver"
            )
        row_elements = prod(shape[split_dimension + 1:])
        if row_elements and width:
            base_offset = int(tensor_desc.base_ptr_offset)
            if base_offset % row_elements and not allow_strided:
                raise ValueError(
                    "Task default split offset is not row-aligned; configure task_dump_tensor_view_resolver"
                )
            start = base_offset // row_elements % width
            count = min(count, width - start)
        shape[split_dimension] = count
    return TaskDumpTensorView(
        source_offset_bytes=int(tensor_desc.base_ptr_offset) * int(tensor_desc.data_type),
        shape=tuple(shape),
    )


def _selected(config: Any, location: _TaskLocation, task_index: int, stage: str, owner_id: int | None) -> bool:
    """Apply task, owner, stage and worker selection predicates."""
    if config.stages is not None and stage not in config.stages:
        return False
    if config.task_ids is not None and location.task_id not in config.task_ids:
        return False
    if config.stage_task_indices is not None and task_index not in config.stage_task_indices:
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
    diagnostic_runtime_config: RuntimeConfigC,
    rank: int,
    stage: str,
    parameter_cache: dict[int, Any],
    core_id: int = 0,
    owner_context: Any = None,
    owner_ids: OwnerIds | None = None,
) -> TaskDumpTensorView:
    """Resolve one selected port's Host view before allocating its payload slot."""
    resolver = getattr(operator, "task_dump_tensor_view_resolver", None)
    default_view = _default_tensor_view(
        operator,
        task_desc,
        tensor_desc,
        tensor_spec,
        io,
        index,
        allow_strided=resolver is not None,
    ) if not tensor_desc.dynamic_shape else TaskDumpTensorView(
        int(tensor_desc.base_ptr_offset) * int(tensor_desc.data_type),
        tuple(int(dim) for dim in tensor_spec.shape),
    )
    if resolver is None:
        if tensor_desc.dynamic_shape:
            raise ValueError(
                f"Task dump stage {stage!r} Tensor {tensor_spec.name!r} needs a task_dump_tensor_view_resolver"
            )
        return default_view
    context = TaskDumpTensorViewContext(
        operator=operator,
        task_id=task_id,
        owner_id=owner_id,
        task_desc=task_desc,
        tensor_desc=tensor_desc,
        tensor_spec=tensor_spec,
        io=io,
        index=index,
        parameters=parameters,
        diagnostic_runtime_config=diagnostic_runtime_config,
        rank=rank,
        stage=stage,
        default_view=default_view,
        parameter_cache=parameter_cache,
        core_id=core_id,
        owner_context=owner_context,
        owner_ids=owner_ids,
    )
    if not callable(resolver):
        raise TypeError(f"Task dump task_dump_tensor_view_resolver for stage {stage!r} must be callable")
    view = resolver(context)
    if not isinstance(view, TaskDumpTensorView):
        raise TypeError(
            f"Task dump resolver for stage {stage!r} must return TaskDumpTensorView, got {type(view).__name__}"
        )
    return view


def _validate_view_bounds(
    parameter: Any,
    view: TaskDumpTensorView,
    tensor_name: str,
) -> tuple[Any, int]:
    """Check dtype, contiguity, alignment and the full source range against its ABI tensor."""
    if not torch.is_tensor(parameter):
        raise TypeError(f"Task dump Tensor {tensor_name!r} must map to torch.Tensor, got {type(parameter).__name__}")
    if not parameter.is_contiguous():
        raise ValueError(f"Task dump Tensor {tensor_name!r} ABI parameter must be contiguous")
    if (isinstance(view.source_offset_bytes, bool) or not isinstance(view.source_offset_bytes, int)
            or not 0 <= view.source_offset_bytes < 1 << 64):
        raise ValueError(f"Task dump Tensor {tensor_name!r} source offset must be a non-negative uint64")
    if any(isinstance(dim, bool) or not isinstance(dim, int) or dim < 0 for dim in view.shape):
        raise ValueError(f"Task dump Tensor {tensor_name!r} view shape must contain non-negative integers")
    element_size = int(parameter.element_size())
    if view.source_offset_bytes % element_size:
        raise ValueError(f"Task dump Tensor {tensor_name!r} source offset must align to its element byte size")
    full_bytes = prod(view.shape) * element_size
    parameter_bytes = int(parameter.numel()) * element_size
    if view.source_offset_bytes + full_bytes > parameter_bytes:
        raise ValueError(
            f"Task dump view for Tensor {tensor_name!r} exceeds its ABI parameter: "
            f"offset={view.source_offset_bytes}, bytes={full_bytes}, parameter_bytes={parameter_bytes}"
        )
    return parameter.dtype, full_bytes


def _stored_ranges(view: TaskDumpTensorView, stored_elements: int) -> ElementRanges | None:
    """Clip valid intervals to the captured prefix after validating the logical view."""
    if view.valid_ranges is None:
        return None
    ranges = []
    previous_end = 0
    full_elements = prod(view.shape)
    for start, end in view.valid_ranges:
        if not isinstance(start, int) or not isinstance(end, int) or not previous_end <= start < end <= full_elements:
            raise ValueError(f"Invalid Task dump element range [{start}, {end}) for shape {view.shape}")
        if start < stored_elements:
            ranges.append((start, min(end, stored_elements)))
        previous_end = end
    return tuple(ranges)


def _operator_ports(operator: Any, task: Any) -> tuple[Any, ...]:
    """Resolve Host ABI ports without modifying the native TaskDesc arrays."""
    if operator.diagnostic_host_ports:
        def descriptors(specs: Any) -> list[TensorDescC]:
            """Build Host-only source descriptors for explicitly bound ABI ports.

            Args:
                specs: Native-bound graph operands requiring Host descriptors.
            """
            return [TensorDescC(input_position=spec.param_position, data_type=spec.dtype_size,
                                dynamic_shape=1) for spec in specs]

        return (("input", descriptors(operator.inputs), operator.inputs),
                ("output", descriptors(operator.outputs), operator.outputs))
    return (("input", task.inputs, operator.inputs), ("output", task.outputs, operator.outputs))


class _TaskDumpCapacityExceeded(RuntimeError):
    """Selected Task snapshots cannot fit in one device capture buffer."""

    def __init__(self, reason: str, detail: str) -> None:
        """Keep the skip reason separate from the diagnostic message."""
        super().__init__(detail)
        self.reason = reason


@dataclass
class _TaskDumpPlanBuilder:
    """Own one launch's record list, parameter cache and monotonically assigned slots."""

    runtime: PreparedDiagnosticRuntime
    parameters: Sequence[Any]
    config: Any
    broadcast: bool
    owner_ids: OwnerIds | None = None
    records: list[_TaskDumpRecordPlan] = field(default_factory=list)
    parameter_cache: dict[int, Any] = field(default_factory=dict)
    next_payload_offset: int = TASK_DUMP_RECORDS_OFFSET + TASK_DUMP_MAX_RECORDS * TASK_DUMP_RECORD.size

    def add(
        self, operator: Any, location: _TaskLocation, io: str, index: int, tensor_desc: Any, tensor_spec: Any,
        *, stage: str, task_index: int, schedule_owner: int | None, owner_id: int | None,
    ) -> None:
        """Validate and append one selected operand, before allocating any payload.

        Args:
            operator: Graph operator associated with the selected task.
            location: Task identity and worker placement in the native schedule.
            stage: Diagnostic stage name associated with the operator.
            task_index: Stage-local index reported by the profiler.
            schedule_owner: Owner in the scheduled topology used for operand addressing.
            owner_id: Optional resolved business owner, such as an Expert ID.
            io: Input or output capture boundary.
            index: Zero-based operand index within the selected input or output list.
            tensor_desc: Native task operand descriptor containing source addressing information.
            tensor_spec: Graph operand contract containing its name and ABI position.
        """
        task_id = location.task_id
        task = self.runtime.diagnostic_runtime_config.all_tasks[task_id]
        position = int(tensor_desc.input_position)
        if not 0 <= position < len(self.parameters):
            raise ValueError(f"Task {task_id} requires parameter {position}, got {len(self.parameters)} parameters")
        parameter = self.parameters[position]
        if torch.is_tensor(parameter) and parameter.element_size() != int(tensor_desc.data_type):
            raise ValueError(f"Task {task_id} Tensor {tensor_spec.name!r} ABI element byte size mismatch")
        view = _resolve_record_view(
            operator=operator, task_id=task_id, owner_id=schedule_owner, task_desc=task, tensor_desc=tensor_desc,
            tensor_spec=tensor_spec, io=io, index=index, parameters=self.parameters,
            diagnostic_runtime_config=self.runtime.diagnostic_runtime_config, rank=self.runtime.rank, stage=stage,
            parameter_cache=self.parameter_cache, core_id=location.core_id,
            owner_context=self.runtime.spec.owner_context, owner_ids=self.owner_ids,
        )
        dtype, full_bytes = _validate_view_bounds(parameter, view, tensor_spec.name)
        if self.broadcast and full_bytes == 0:
            return  # No Device worker is launched for this empty row shard.
        payload_bytes = full_bytes
        if self.config.max_bytes_per_tensor is not None and payload_bytes > self.config.max_bytes_per_tensor:
            element_size = int(parameter.element_size())
            payload_bytes = self.config.max_bytes_per_tensor // element_size * element_size
            if payload_bytes <= 0:
                raise ValueError(
                    f"max_bytes_per_tensor={self.config.max_bytes_per_tensor} cannot store one element of "
                    f"Tensor {tensor_spec.name!r} ({dtype} requires {element_size} bytes per element)"
                )
        if len(self.records) == TASK_DUMP_MAX_RECORDS:
            raise _TaskDumpCapacityExceeded(
                "task_record_limit_exceeded", f"Task dump exceeds the record limit of {TASK_DUMP_MAX_RECORDS}",
            )
        self.records.append(_TaskDumpRecordPlan(
            task_id=task_id, task_index=task_index, stage=stage, owner_id=owner_id,
            owner_label=self.runtime.owner_label, core_type=location.core_type, core_id=location.core_id,
            io=io, index=index, phase="before" if io == "input" else "after", parameter_position=position,
            tensor_name=tensor_spec.name, dtype=dtype, source_offset_bytes=view.source_offset_bytes,
            actual_shape=view.shape, payload_offset=self.next_payload_offset, payload_bytes=payload_bytes,
            full_bytes=full_bytes, truncated=payload_bytes < full_bytes, note=view.note,
            record_offset=TASK_DUMP_RECORDS_OFFSET + len(self.records) * TASK_DUMP_RECORD.size,
            capture_worker_id=location.core_id if self.broadcast else None,
            coverage=view.coverage, valid_ranges=_stored_ranges(view, payload_bytes // parameter.element_size()),
        ))
        self.next_payload_offset = _align_up(self.next_payload_offset + payload_bytes)


def _build_task_record_plans(
    runtime: PreparedDiagnosticRuntime, parameters: Sequence[Any], config: Any, *, owner_ids: OwnerIds | None = None,
) -> tuple[_TaskDumpRecordPlan, ...]:
    """Select task/port pairs, then compile their independent snapshot slots."""
    metadata = runtime.metadata
    diagnostic_runtime_config = runtime.diagnostic_runtime_config
    operators = runtime.graph.topological_sort()
    stage_names = _stage_names(operators)
    operators_by_stage = {stage_names[operator.name]: operator for operator in operators}
    spec = runtime.spec
    broadcast = spec.execution_mode == "aiv_pipeline"
    locations = _iter_task_locations(diagnostic_runtime_config, vector_worker_stride=spec.vector_worker_stride,
                                     broadcast_tasks=broadcast)
    builder = _TaskDumpPlanBuilder(runtime, parameters, config, broadcast, owner_ids)
    for location in sorted(locations, key=lambda item: (item.task_id, item.core_id)):
        task = diagnostic_runtime_config.all_tasks[location.task_id]
        stage = metadata.task_stage_names.get(location.task_id)
        operator = operators_by_stage.get(stage)
        if operator is None:
            continue
        task_index = location.core_id if broadcast else int(task.task_index)
        schedule_owner = location.core_id if broadcast else metadata.task_owner_ids.get(location.task_id)
        owner_id = schedule_owner
        if owner_ids is not None and schedule_owner is not None:
            owner_id = owner_ids[schedule_owner]
            if owner_id is None:
                continue  # This invocation did not assign an expert to this schedule slot.
        if not _selected(config, location, task_index=task_index, stage=stage, owner_id=owner_id):
            continue
        for io, descriptors, specs in _operator_ports(operator, task):
            if io not in config.io:
                continue
            for index, tensor_spec in enumerate(specs):
                if config.tensor_names is None or tensor_spec.name in config.tensor_names:
                    builder.add(operator, location, io, index, descriptors[index], tensor_spec,
                                stage=stage, task_index=task_index, schedule_owner=schedule_owner, owner_id=owner_id)
    return tuple(builder.records)


def _pack_task_buffer(records: tuple[_TaskDumpRecordPlan, ...], *, total_bytes: int) -> bytearray:
    """Encode the versioned header, padded record table and independent payload slots."""
    payload_offset = min(record.payload_offset for record in records)
    raw_buffer = bytearray(total_bytes)
    TASK_DUMP_HEADER.pack_into(
        raw_buffer,
        0,
        TASK_DUMP_MAGIC,
        TASK_DUMP_VERSION,
        1,
        len(records),
        TASK_DUMP_RECORD.size,
        TASK_DUMP_RECORDS_OFFSET,
        payload_offset,
        total_bytes,
    )
    phase_values = {"before": TASK_DUMP_PHASE_BEFORE, "after": TASK_DUMP_PHASE_AFTER}
    for record in records:
        source_low = record.source_offset_bytes & 0xFFFFFFFF
        source_high = record.source_offset_bytes >> 32
        worker_id = _TASK_DUMP_ANY_WORKER if record.capture_worker_id is None else record.capture_worker_id
        TASK_DUMP_RECORD.pack_into(
            raw_buffer,
            record.record_offset,
            record.task_id,
            phase_values[record.phase],
            record.parameter_position,
            source_low,
            source_high,
            record.payload_offset,
            record.payload_bytes,
            TASK_DUMP_STATUS_PENDING,
            record.task_index,
            worker_id,
        )
    return raw_buffer


def _prepare_task_dump(
    runtime: PreparedDiagnosticRuntime, parameters: Sequence[Any], config: Any, *, owner_ids: OwnerIds | None = None,
) -> _PreparedTaskDump | None:
    """Allocate a Device snapshot buffer if any Task ports match the filters."""
    records = _build_task_record_plans(runtime, parameters, config, owner_ids=owner_ids)
    if not records:
        return None
    budget_bytes = config.device_memory_budget_mb * 1024 * 1024
    total_bytes = _align_up(max(record.payload_offset + record.payload_bytes for record in records))
    if total_bytes >= 1 << 32:
        raise _TaskDumpCapacityExceeded(
            "device_buffer_abi_limit_exceeded",
            f"Task dump device buffer requires {total_bytes} bytes, "
            f"exceeding the 32-bit ABI limit of {(1 << 32) - 1} bytes",
        )
    if total_bytes > budget_bytes:
        raise _TaskDumpCapacityExceeded(
            "device_buffer_budget_exceeded",
            f"Task dump device buffer requires {total_bytes} bytes, exceeding "
            f"device_memory_budget_mb={config.device_memory_budget_mb} ({budget_bytes} bytes)",
        )
    raw_buffer = _pack_task_buffer(records, total_bytes=total_bytes)
    device_buffer = torch.frombuffer(raw_buffer, dtype=torch.uint8).clone().to(runtime.runtime_config_tensor.device)
    # Records and execution must share the adapted task IDs and queues.
    task_dump_runtime_config = runtime.copy_diagnostic_runtime_config()
    task_dump_runtime_config.diagnostic_mode = DeviceDiagnosticMode.TASK_TENSOR_DUMP
    task_dump_runtime_config_tensor = runtime.tensor_from_config(task_dump_runtime_config)
    return _PreparedTaskDump(
        runtime_config_tensor=task_dump_runtime_config_tensor,
        buffer=device_buffer,
        records=records,
        total_bytes=total_bytes,
    )


def _payload_tensor(
    host_buffer: torch.Tensor,
    record: _TaskDumpRecordPlan,
) -> torch.Tensor:
    """Decode one payload slot as an owned CPU Tensor without dtype conversion."""
    if record.payload_bytes == 0:
        tensor = torch.empty((0,), dtype=record.dtype)
    else:
        payload = host_buffer.narrow(0, record.payload_offset, record.payload_bytes)
        tensor = payload.view(record.dtype).clone()
    stored_shape = (tensor.numel(),) if record.truncated else record.actual_shape
    return tensor.reshape(stored_shape)


def _decode_task_dump(prepared: _PreparedTaskDump) -> tuple[_TaskDumpTensorSnapshot, ...]:
    """Copy the completed Device buffer to Host and decode all fixed slots."""
    host_buffer = prepared.buffer.detach().contiguous().cpu()
    header = tuple(int(value) for value in host_buffer[:TASK_DUMP_HEADER.size].view(torch.int32).tolist())
    if header[:2] != (TASK_DUMP_MAGIC, TASK_DUMP_VERSION):
        raise ValueError(f"invalid Task dump buffer header: magic={header[0]:#x}, version={header[1]}")
    snapshots = []
    for record in prepared.records:
        status_offset = record.record_offset + 7 * 4
        status = int(host_buffer[status_offset:status_offset + 4].view(torch.int32).item())
        if status != TASK_DUMP_STATUS_CAPTURED:
            raise RuntimeError(
                f"Task dump record was not captured: task_id={record.task_id}, stage={record.stage!r}, "
                f"io={record.io}, index={record.index}, status={status}, header_enabled={header[2]}"
            )
        snapshots.append(_TaskDumpTensorSnapshot(record, _payload_tensor(host_buffer, record)))
    return tuple(snapshots)


def _write_task_snapshots(
    snapshots: list[_TaskDumpTensorSnapshot],
    invocation_dir: Path,
    window_dir: Path,
    device_buffer_bytes: int,
) -> dict[str, Any]:
    """Write task-local tensor snapshots with their native copy metadata."""
    snapshots_by_task: dict[tuple[int, str, int], list[_TaskDumpTensorSnapshot]] = {}
    for snapshot in snapshots:
        record = snapshot.record
        snapshots_by_task.setdefault((record.task_id, record.core_type, record.core_id), []).append(snapshot)

    task_metadata = []
    for key in sorted(snapshots_by_task):
        task_id = key[0]
        task_snapshots = snapshots_by_task[key]
        first = task_snapshots[0].record
        safe_stage = _SAFE_FILE_NAME.sub("_", first.stage)
        task_dir = invocation_dir / "stages" / safe_stage / f"task_{task_id:06d}"
        if first.capture_worker_id is not None:
            task_dir = task_dir.with_name(f"{task_dir.name}_worker_{first.core_id:03d}")
        task_dir.mkdir(parents=True, exist_ok=True)
        tensor_metadata = []
        for snapshot in task_snapshots:
            record = snapshot.record
            safe_name = _SAFE_FILE_NAME.sub("_", record.tensor_name)
            file_name = f"{record.io}_{record.index:02d}_{safe_name}.pt"
            output_path = task_dir / file_name
            _save_tensor(snapshot.data, output_path)
            dtype_name = str(snapshot.data.dtype).removeprefix("torch.")
            tensor_metadata.append(
                {
                    "name": record.tensor_name,
                    "paramPosition": record.parameter_position,
                    "io": record.io,
                    "index": record.index,
                    "capturePhase": record.phase,
                    "sourceOffsetBytes": record.source_offset_bytes,
                    "actualShape": list(record.actual_shape),
                    "storedShape": list(snapshot.data.shape),
                    "originalDtype": dtype_name,
                    "storedDtype": dtype_name,
                    "fullBytes": record.full_bytes,
                    "storedBytes": record.payload_bytes,
                    "truncated": record.truncated,
                    **({"coverage": record.coverage} if record.coverage != "task_view" else {}),
                    **({"validRanges": record.valid_ranges} if record.valid_ranges is not None else {}),
                    "note": record.note,
                    "file": output_path.relative_to(window_dir).as_posix(),
                }
            )
        task_metadata.append({
            "taskId": task_id,
            "stageTaskIndex": first.task_index,
            "stage": first.stage,
            "owner": {"label": first.owner_label, "id": first.owner_id},
            "coreType": first.core_type,
            "coreId": first.core_id,
            "tensorCount": len(tensor_metadata),
            "tensors": tensor_metadata,
        })
    return {
        "deviceBufferBytes": device_buffer_bytes,
        "taskCount": len(task_metadata),
        "tasks": task_metadata,
    }
