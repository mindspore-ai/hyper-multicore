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
"""Shape-bound RuntimeConfig plan for HyperMegaMhc."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from hyper_multicore.diagnostics import MegaKernelDiagnostics, prepare_mega_kernel_diagnostics
from hyper_multicore.scheduler.builder import build_runtime_config
from hyper_multicore.scheduler.config import _validate_runtime_task_queues

from .diagnostics import bind_forward_graph
from .graph import DEFAULT_TOKEN_TILE, build_mega_mhc_graph, resolve_token_tile


@dataclass(frozen=True)
class MegaMhcPlan:
    """Own one read-only token-shape schedule, without invocation state."""

    token_count: int
    hidden_size: int
    num_cube_cores: int
    token_tile: int
    runtime: MegaKernelDiagnostics
    event_capacity: int

    @property
    def runtime_config(self) -> torch.Tensor:
        """Return the profiling-disabled RuntimeConfig tensor."""
        return self.runtime.runtime_config


def build_mega_mhc_plan(
    token_count: int,
    hidden_size: int,
    device: Any,
    num_cube_cores: int,
    token_tile: int = DEFAULT_TOKEN_TILE,
) -> MegaMhcPlan:
    """Build and materialize a token-owned HyperMegaMhc schedule.

    Args:
        token_count: Total flattened token count.
        hidden_size: Hidden dimension.
        device: Target NPU device.
        num_cube_cores: Number of Cube cores.
        token_tile: Preferred token count per task.
    """
    token_tile = resolve_token_tile(token_count, token_tile, num_cube_cores)
    graph, topology = build_mega_mhc_graph(
        token_count,
        hidden_size,
        num_cube_cores,
        token_tile,
    )
    runtime_config = build_runtime_config(
        graph,
        topology,
        rank_id=0,
        num_cube_cores=num_cube_cores,
    )
    _validate_runtime_task_queues(runtime_config, runtime_config.event_capacity)
    runtime = prepare_mega_kernel_diagnostics(
        graph, runtime_config, kernel_name="HyperMegaMhc",
        parameter_count=23, device=device, owner_label="TokenPartition",
        rank=dist.get_rank() if dist.is_initialized() else 0,
        vector_worker_stride=1, graph_adapter=bind_forward_graph,
    )
    return MegaMhcPlan(
        token_count=token_count,
        hidden_size=hidden_size,
        num_cube_cores=num_cube_cores,
        token_tile=token_tile,
        runtime=runtime,
        event_capacity=runtime_config.event_capacity,
    )
