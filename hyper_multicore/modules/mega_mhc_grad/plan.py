# Copyright 2026 Huawei Technologies Co., Ltd.
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
"""Shape-bound RuntimeConfig plan for HyperMegaMhcGrad."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch.distributed as dist

from hyper_multicore.diagnostics import MegaKernelDiagnostics, prepare_mega_kernel_diagnostics
from hyper_multicore.scheduler.builder import build_runtime_config
from hyper_multicore.scheduler.config import _validate_runtime_task_queues

from .diagnostics import bind_backward_graph
from .graph import (
    DEFAULT_GRAD_TOKEN_TILE,
    build_mega_mhc_grad_graph,
    order_vector_tasks_by_stage,
    resolve_grad_token_tile,
)


@dataclass(frozen=True)
class MegaMhcGradPlan:
    """Own one read-only backward schedule, without invocation state."""

    token_count: int
    hidden_size: int
    num_cube_cores: int
    num_vector_cores: int
    token_tile: int
    runtime: MegaKernelDiagnostics
    event_capacity: int


def build_mega_mhc_grad_plan(
    token_count: int,
    hidden_size: int,
    device: Any,
    num_cube_cores: int,
    num_vector_cores: int,
    token_tile: int = DEFAULT_GRAD_TOKEN_TILE,
) -> MegaMhcGradPlan:
    """Build and materialize the token-tiled fused backward schedule.

    Args:
        token_count: Total number of flattened tokens.
        hidden_size: Hidden dimension of every token.
        device: NPU device used to materialize runtime tensors.
        num_cube_cores: Number of Cube cores.
        num_vector_cores: Number of Vector cores.
        token_tile: Requested AIV token tile.

    Returns:
        Reusable backward plan bound to the requested shape and device.
    """
    token_tile = resolve_grad_token_tile(token_count, token_tile)
    graph, topology = build_mega_mhc_grad_graph(
        token_count,
        hidden_size,
        token_tile,
        num_vector_cores=num_vector_cores,
    )
    runtime_config = build_runtime_config(
        graph, topology, rank_id=0, num_cube_cores=num_cube_cores
    )
    order_vector_tasks_by_stage(runtime_config)
    _validate_runtime_task_queues(runtime_config, runtime_config.event_capacity)
    runtime = prepare_mega_kernel_diagnostics(
        graph, runtime_config, kernel_name="HyperMegaMhcGrad",
        parameter_count=33, device=device, owner_label="NativeKernel",
        rank=dist.get_rank() if dist.is_initialized() else 0,
        vector_worker_stride=1, graph_adapter=bind_backward_graph,
    )
    return MegaMhcGradPlan(
        token_count,
        hidden_size,
        num_cube_cores,
        num_vector_cores,
        token_tile,
        runtime,
        runtime_config.event_capacity,
    )
