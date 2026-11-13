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
"""Build shared RuntimeConfigs for HyperMegaGate Route and RouteGrad."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import torch
import torch.distributed as dist

from hyper_multicore.diagnostics import MegaKernelDiagnostics, prepare_mega_kernel_diagnostics
from hyper_multicore.scheduler.config import (
    EVENT_INVALID_ID,
    INVALID_PROFILE_OWNER_ID,
    TaskAiCoreType,
    TaskDescC,
    TaskType,
    RuntimeConfigC,
)
from hyper_multicore.scheduler.runtime import allocate_runtime_config

from .diagnostics import (
    MEGA_GATE_ROUTE_GRAD_K1_STAGE_NAMES,
    MEGA_GATE_ROUTE_GRAD_STAGE_NAMES,
    MEGA_GATE_ROUTE_STAGE_NAMES,
    bind_pipeline_graph,
)


# AIV slot capacity shared with Route/RouteGrad tiling.
MEGA_GATE_AIV_WORKER_CAPACITY = 48
MEGA_GATE_ROUTE_TASK_TYPES = (
    TaskType.TASK_GATE_SOFTPLUS,
    TaskType.TASK_GATE_SQRT,
    TaskType.TASK_GATE_ADD_BIAS,
    TaskType.TASK_GATE_TOPK,
    TaskType.TASK_GATE_GATHER,
    TaskType.TASK_GATE_REDUCE_SUM,
    TaskType.TASK_GATE_ADD_EPSILON,
    TaskType.TASK_GATE_DIV,
    TaskType.TASK_GATE_MUL_SCALE,
    TaskType.TASK_GATE_CAST_INDEX,
)
MEGA_GATE_ROUTE_GRAD_TASK_TYPES = (
    TaskType.TASK_GATE_GRAD_MULS_SCALE,
    TaskType.TASK_GATE_GRAD_BROADCAST_DENOMINATOR,
    TaskType.TASK_GATE_GRAD_NEG,
    TaskType.TASK_GATE_GRAD_DIV_SELECTED,
    TaskType.TASK_GATE_GRAD_DIV_SELECTED_RATIO,
    TaskType.TASK_GATE_GRAD_MUL_CROSS,
    TaskType.TASK_GATE_GRAD_DIV_DIRECT,
    TaskType.TASK_GATE_GRAD_REDUCE_SUM,
    TaskType.TASK_GATE_GRAD_BROADCAST_ROW_SUM,
    TaskType.TASK_GATE_GRAD_ADD_SELECTED,
    TaskType.TASK_GATE_GRAD_ZEROS,
)
MEGA_GATE_ROUTE_GRAD_K1_TASK_TYPES = (
    TaskType.TASK_GATE_GRAD_MULS_SCALE,
    TaskType.TASK_GATE_GRAD_ZEROS,
)


@dataclass(frozen=True)
class MegaGatePlan:
    """Capacity-bounded RuntimeConfig shared by all input shapes on one NPU."""

    route_runtime: MegaKernelDiagnostics
    route_grad_runtime: MegaKernelDiagnostics
    route_grad_k1_runtime: MegaKernelDiagnostics
    vision_mask_placeholder: torch.Tensor


def _build_pipeline_runtime_config(
    task_types: tuple[TaskType, ...],
    stage_names: tuple[str, ...],
    *,
    kernel_name: str,
) -> RuntimeConfigC:
    """Create descriptors broadcast to every active token-row worker."""
    if len(task_types) != len(stage_names):
        raise RuntimeError("MegaGate task types and profile stages must have equal length")
    if not isinstance(kernel_name, str) or not kernel_name:
        raise ValueError("MegaGate kernel_name must be a non-empty string")
    runtime_config = allocate_runtime_config(len(task_types))
    runtime_config.task_num = len(task_types)
    runtime_config.num_workers = MEGA_GATE_AIV_WORKER_CAPACITY
    runtime_config.task_index_num[1] = len(task_types)
    for task_id, task_type in enumerate(task_types):
        task = TaskDescC()
        task.task_type = task_type
        task.task_aicore_type = TaskAiCoreType.TASK_AICORE_VECTOR
        task.trigger_event = EVENT_INVALID_ID
        task.dependent_event = EVENT_INVALID_ID
        task.task_index = 0
        task.task_split_num = MEGA_GATE_AIV_WORKER_CAPACITY
        task.tiling_data_offset = task_id
        task.profile_owner_id = INVALID_PROFILE_OWNER_ID
        runtime_config.all_tasks[task_id] = task
        runtime_config.vector_task_indices[task_id] = task_id
    return runtime_config


def build_mega_gate_plan(device: torch.device) -> MegaGatePlan:
    """Build an eventless Route runtime for one NPU device.

    Args:
        device: NPU device receiving the serialized RuntimeConfig.

    Returns:
        Immutable plan with normal and profiled broadcast stage descriptors.
    """
    if device.type != "npu":
        raise ValueError(f"MegaGate plan requires an NPU device, got {device}")
    rank = dist.get_rank() if dist.is_initialized() else 0

    def prepare(runtime_config: RuntimeConfigC, *, backward: bool = False) -> MegaKernelDiagnostics:
        """Prepare only ordinary storage; broadcast graph ports remain lazy.

        Args:
            runtime_config: Complete Host configuration describing native task queues.
            backward: Whether to prepare the RouteGrad pipeline instead of Route.
        """
        return prepare_mega_kernel_diagnostics(
            None, runtime_config, kernel_name="HyperMegaGateRouteGrad" if backward else "HyperMegaGateRoute",
            parameter_count=11, device=device, rank=rank, owner_label="TokenRowShard",
            execution_mode="aiv_pipeline", vector_worker_stride=1,
            graph_adapter=partial(bind_pipeline_graph, backward=backward),
        )

    return MegaGatePlan(
        route_runtime=prepare(_build_pipeline_runtime_config(
            MEGA_GATE_ROUTE_TASK_TYPES,
            MEGA_GATE_ROUTE_STAGE_NAMES,
            kernel_name="HyperMegaGateRoute",
        )),
        route_grad_runtime=prepare(_build_pipeline_runtime_config(
            MEGA_GATE_ROUTE_GRAD_TASK_TYPES,
            MEGA_GATE_ROUTE_GRAD_STAGE_NAMES,
            kernel_name="HyperMegaGateRouteGrad",
        ), backward=True),
        route_grad_k1_runtime=prepare(_build_pipeline_runtime_config(
            MEGA_GATE_ROUTE_GRAD_K1_TASK_TYPES,
            MEGA_GATE_ROUTE_GRAD_K1_STAGE_NAMES,
            kernel_name="HyperMegaGateRouteGrad",
        ), backward=True),
        vision_mask_placeholder=torch.zeros((1,), device=device, dtype=torch.bool),
    )
