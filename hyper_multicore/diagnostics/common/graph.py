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
"""Business-independent graph contracts and diagnostic task identity."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from hyper_multicore.scheduler.config import RuntimeConfigC, TaskDescC, TaskType
from hyper_multicore.scheduler.graph import ComputeGraph, OperatorNode


@dataclass(frozen=True)
class DiagnosticSpec:
    """Stable kernel identity and optional business-owned task-owner resolver."""

    kernel_name: str
    owner_label: str = "Owner"
    owner_resolver: Callable[[OperatorNode, TaskDescC, Any], int | None] | None = None
    owner_context: Any = None
    execution_mode: str = "event_driven"
    vector_worker_stride: int = 2
    graph_adapter: Callable[[Any, RuntimeConfigC], ComputeGraph] | None = None
    active_vector_workers: int | None = None


@dataclass(frozen=True)
class GraphDiagnosticMetadata:
    """Host-only task identity, independent of trace or Tensor-dump formats."""

    kernel_name: str
    owner_label: str
    stage_names: Mapping[int, str]
    task_stage_ids: Mapping[int, int]
    task_stage_names: Mapping[int, str]
    task_owner_ids: Mapping[int, int | None]


def bind_graph_tensor_ports(
    graph: ComputeGraph,
    ports: Mapping[str, tuple[list[Any], list[Any]]], resolver: Callable[..., Any],
) -> ComputeGraph:
    """Bind real GM operands on a diagnostic-only copy of a scheduling graph.

    Native task builders may describe dependencies without encoding tensor
    operands. The business adapter supplies the actual input-list positions;
    neither the scheduling graph nor the normal Device runtime is modified.

    Args:
        graph: Business compute graph whose scheduling storage remains unchanged.
        ports: Mapping from operator names to actual native input and output operands.
        resolver: Business callback that resolves a selected operand's contiguous range.
    """
    graph = deepcopy(graph)
    for operator in graph.topological_sort():
        operator.inputs, operator.outputs = ports[operator.name]
        operator.param_positions = [tensor.param_position for tensor in operator.inputs + operator.outputs]
        operator.diagnostic_view_resolver = resolver
        operator.diagnostic_host_ports = True
    return graph


def operator_diagnostic_name(operator: OperatorNode) -> str:
    """Return an explicit diagnostic name or derive one from the graph node name.

    Args:
        operator: Graph operator associated with the selected task.
    """
    diagnostic_name = operator.diagnostic_name
    if diagnostic_name is not None:
        if not isinstance(diagnostic_name, str) or not diagnostic_name.strip():
            raise ValueError(
                f"OperatorNode.diagnostic_name must be a non-empty string when set: {diagnostic_name!r}"
            )
        return diagnostic_name.strip()
    if not isinstance(operator.name, str) or not operator.name.strip():
        raise ValueError(f"OperatorNode.name must be a non-empty string, got {operator.name!r}")
    stage_name = "".join(part[:1].upper() + part[1:] for part in operator.name.strip().split("_") if part)
    if not stage_name:
        raise ValueError(f"OperatorNode.name must contain a diagnostic display character: {operator.name!r}")
    return stage_name


def _scheduled_task_ids(runtime_config: RuntimeConfigC) -> set[int]:
    """Collect queue task IDs, including tasks outside RuntimeConfig.task_num."""
    task_ids = set()
    task_schedules = (
        (runtime_config.cube_task_indices, int(runtime_config.task_index_num[0])),
        (runtime_config.vector_task_indices, int(runtime_config.task_index_num[1])),
        (runtime_config.mix_task_indices, int(runtime_config.task_index_num[2])),
    )
    for task_indices, scheduled_task_count in task_schedules:
        if not 0 <= scheduled_task_count <= len(task_indices):
            raise ValueError(f"scheduled task count is outside RuntimeConfig capacity: {scheduled_task_count}")
        for schedule_index in range(scheduled_task_count):
            task_id = int(task_indices[schedule_index])
            if not 0 <= task_id < len(runtime_config.all_tasks):
                raise ValueError(f"scheduled task ID is outside RuntimeConfig capacity: {task_id}")
            task_ids.add(task_id)
    return task_ids


def _validate_parameter_position(position: int, parameter_count: int, label: str) -> None:
    if isinstance(position, bool) or not isinstance(position, int) or not 0 <= position < parameter_count:
        raise ValueError(f"{label} parameter position {position!r} is outside ABI parameter_count={parameter_count}")


def _validate_operator(operator: OperatorNode, parameter_count: int) -> None:
    """Validate graph operand names, shapes and native ABI bindings."""
    if isinstance(operator.task_num, bool) or not isinstance(operator.task_num, int) or operator.task_num < 0:
        raise ValueError(f"operator {operator.name!r} has invalid task_num={operator.task_num!r}")
    ports = operator.inputs + operator.outputs
    if len(operator.param_positions) != len(ports):
        raise ValueError(f"operator {operator.name!r} param_positions must match its input/output ports")
    for position, tensor_spec in zip(operator.param_positions, ports):
        _validate_parameter_position(tensor_spec.param_position, parameter_count, tensor_spec.name)
        if position != tensor_spec.param_position:
            raise ValueError(f"operator {operator.name!r} param_positions disagrees with Tensor {tensor_spec.name!r}")
        if not isinstance(tensor_spec.name, str) or not tensor_spec.name.strip():
            raise ValueError("TensorSpec.name must be a non-empty string")
        if any(isinstance(dim, bool) or not isinstance(dim, int) or dim < 0 for dim in tensor_spec.shape):
            raise ValueError(f"Tensor {tensor_spec.name!r} shape must contain non-negative integers")
        if (isinstance(tensor_spec.dtype_size, bool) or not isinstance(tensor_spec.dtype_size, int)
                or tensor_spec.dtype_size <= 0):
            raise ValueError(f"Tensor {tensor_spec.name!r} dtype_size must be a positive integer")


def _validate_task_ports(
    task: TaskDescC, operator: OperatorNode, task_id: int,
) -> None:
    """Check a scheduled task's operand descriptors against the graph contract."""
    if operator.diagnostic_host_ports:
        if not callable(operator.diagnostic_view_resolver):
            raise ValueError(f"native ABI stage {operator.name} requires a view resolver")
        return
    for io, descriptors, specs, count in (
        ("input", task.inputs, operator.inputs, int(task.num_inputs)),
        ("output", task.outputs, operator.outputs, int(task.num_outputs)),
    ):
        if count != len(specs) or count > len(descriptors):
            raise ValueError(f"task {task_id} stage {operator.name!r} {io} count does not match graph/ABI capacity")
        for index, spec in enumerate(specs):
            descriptor = descriptors[index]
            if int(descriptor.input_position) != spec.param_position:
                raise ValueError(f"task {task_id} {io}{index} parameter position disagrees with Tensor {spec.name!r}")
            if int(descriptor.data_type) != spec.dtype_size:
                raise ValueError(f"task {task_id} {io}{index} element byte size disagrees with Tensor {spec.name!r}")


def _validate_task_schedule(runtime_config: RuntimeConfigC, graph_task_count: int) -> None:
    """Reject missing, duplicated or unmapped compute tasks in Device queues."""
    queues = (runtime_config.cube_task_indices, runtime_config.vector_task_indices, runtime_config.mix_task_indices)
    scheduled_ids = _scheduled_task_ids(runtime_config)
    seen = set()
    for indices, count in zip(queues, runtime_config.task_index_num[:3]):
        for task_id in indices[:count]:
            if task_id in seen:
                raise ValueError(f"task {task_id} appears more than once in Device schedules")
            seen.add(task_id)
    missing = set(range(graph_task_count)) - scheduled_ids
    if missing:
        raise ValueError(f"graph tasks are absent from Device schedules: {sorted(missing)}")
    for task_id in scheduled_ids | set(range(int(runtime_config.task_num))):
        if task_id >= graph_task_count and runtime_config.all_tasks[task_id].task_type != TaskType.TASK_TERMINATE:
            raise ValueError(f"task {task_id} has no graph stage and is not a termination task")


def validate_mega_kernel_diagnostic_contract(
    graph: ComputeGraph,
    runtime_config: RuntimeConfigC,
    parameter_count: int,
) -> None:
    """Validate graph/task/ABI agreement before allocating Device diagnostic resources.

    Args:
        graph: Propagated graph; task IDs must follow contiguous topological ranges.
        runtime_config: Filled graph-sized Host configuration including Device queues.
        parameter_count: Number of launch binding entries. Device task ports
            must use real native input_list positions. Zero-task nodes may
            describe Host-visible results produced by an enclosing native call.

    Raises:
        TypeError: Invalid graph or runtime configuration.
        ValueError: Inconsistent graph/task/ABI mapping.

    Note:
        This read-only check does not call view resolvers or inspect Device Tensors.
        Actual offset/shape memory bounds are validated when L1 prepares a launch.
    """
    if not isinstance(graph, ComputeGraph):
        raise TypeError("graph must be ComputeGraph")
    if not isinstance(runtime_config, RuntimeConfigC) or not hasattr(runtime_config, "all_tasks"):
        raise TypeError("runtime_config must be an allocated graph-sized RuntimeConfigC")
    if isinstance(parameter_count, bool) or not isinstance(parameter_count, int) or parameter_count < 0:
        raise ValueError("parameter_count must be a non-negative integer")
    if runtime_config.task_num > len(runtime_config.all_tasks):
        raise ValueError("RuntimeConfig.task_num exceeds allocated task capacity")
    task_id = 0
    for operator in graph.topological_sort():
        _validate_operator(operator, parameter_count)
        if task_id + operator.task_num > runtime_config.task_num:
            raise ValueError(f"graph tasks for stage {operator.name!r} exceed RuntimeConfig.task_num")
        for current_task_id in range(task_id, task_id + operator.task_num):
            _validate_task_ports(runtime_config.all_tasks[current_task_id], operator, current_task_id)
        task_id += operator.task_num
    _validate_task_schedule(runtime_config, task_id)


def build_graph_metadata(
    graph: ComputeGraph, runtime_config: RuntimeConfigC, spec: DiagnosticSpec,
) -> GraphDiagnosticMetadata:
    """Resolve stable graph stage IDs and optional owners without modifying tasks.

    Args:
        graph: Business compute graph whose scheduling storage remains unchanged.
        runtime_config: Complete Host configuration describing native task queues.
        spec: Tool-neutral kernel identity and optional business owner callbacks.
    """
    for label, value in (("kernel_name", spec.kernel_name), ("owner_label", spec.owner_label)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} must be a non-empty string, got {value!r}")
    if spec.owner_resolver is not None and not callable(spec.owner_resolver):
        raise TypeError("owner_resolver must be callable or None")
    stages = {}
    task_stages = {}
    task_names = {}
    owners = {}
    task_id = 0
    for stage_id, operator in enumerate(graph.topological_sort()):
        name = operator_diagnostic_name(operator)
        stages[stage_id] = name
        for current_id in range(task_id, task_id + operator.task_num):
            owner = None
            if spec.owner_resolver is not None:
                owner = spec.owner_resolver(operator, runtime_config.all_tasks[current_id], spec.owner_context)
            if owner is not None and (isinstance(owner, bool) or not isinstance(owner, int)
                                      or not 0 <= owner < 0xFFFFFFFF):
                raise ValueError(f"diagnostic owner_id must be a uint32 integer or None, got {owner!r}")
            task_stages[current_id] = stage_id
            task_names[current_id] = name
            owners[current_id] = owner
        task_id += operator.task_num
    return GraphDiagnosticMetadata(
        spec.kernel_name.strip(), spec.owner_label.strip(),
        MappingProxyType(stages), MappingProxyType(task_stages),
        MappingProxyType(task_names), MappingProxyType(owners),
    )
