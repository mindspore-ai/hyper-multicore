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
"""Generic MegaKernel integration without imports of concrete diagnostic tools."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch

from hyper_multicore.diagnostics.common.call import DiagnosticCall, LaunchContext
from hyper_multicore.diagnostics.common.graph import DiagnosticSpec, GraphDiagnosticMetadata
from hyper_multicore.diagnostics.common.runtime import PreparedDiagnosticRuntime
from hyper_multicore.diagnostics.common.session import _get_active_session
from hyper_multicore.scheduler.config import RuntimeConfigC, TaskDescC
from hyper_multicore.scheduler.graph import ComputeGraph, OperatorNode


class MegaKernelDiagnostics:
    """A reusable normal runtime and optional tool-independent launch boundary."""

    def __init__(self, runtime: PreparedDiagnosticRuntime) -> None:
        """Bind a normal runtime without evaluating diagnostic graph semantics."""
        self._runtime = runtime

    @property
    def runtime_config(self) -> torch.Tensor:
        """Return the original normal execution configuration."""
        return self._runtime.normal_tensor

    @property
    def is_active(self) -> bool:
        """Return whether an explicit diagnostic session is active."""
        session = _get_active_session()
        return session is not None and session.is_capture_enabled()

    @property
    def metadata(self) -> GraphDiagnosticMetadata:
        """Resolve shared graph metadata explicitly for diagnostics or inspection."""
        return self._runtime.metadata

    def launch(
        self, parameters: Sequence[torch.Tensor | None], *, direction: str,
        fallback_event_counters: torch.Tensor | None = None,
        tags: dict[str, str] | None = None,
    ) -> DiagnosticCall:
        """Select exactly one tool, or return an ordinary no-op context.

        Kernel adapters should bypass this method when is_active is false so
        ordinary execution need not create even a Host-side diagnostic call.

        Args:
            parameters: Complete native ABI bindings for this invocation.
            direction: Forward or backward invocation direction.
            fallback_event_counters: Existing execution-owned event counters, if any.
            tags: Optional invocation identity such as model layer and recompute phase.
        """
        context = LaunchContext(self._runtime, parameters, direction, fallback_event_counters, tags=tags)
        session = _get_active_session()
        if session is None or not session.is_capture_enabled():
            return DiagnosticCall(context.ordinary_arguments())
        if len(parameters) != self._runtime.parameter_count:
            raise ValueError(f"launch expects {self._runtime.parameter_count} ABI parameters, got {len(parameters)}")
        if direction not in ("forward", "backward"):
            raise ValueError(f"invalid MegaKernel direction {direction!r}")
        return session.prepare_call(context)


def prepare_mega_kernel_diagnostics(
    graph: ComputeGraph | None, runtime_config: RuntimeConfigC, *, kernel_name: str,
    parameter_count: int, device: torch.device | str, rank: int = 0, owner_label: str = "Owner",
    owner_resolver: Callable[[OperatorNode, TaskDescC, Any], int | None] | None = None,
    owner_context: Any = None,
    execution_mode: str = "event_driven", vector_worker_stride: int = 2,
    graph_adapter: Callable[[Any, RuntimeConfigC], ComputeGraph] | None = None,
    active_vector_workers: int | None = None,
) -> MegaKernelDiagnostics:
    """Prepare only normal execution storage and defer all diagnostic graph work.

    Args:
        graph: Completed scheduling graph; may be None if graph_adapter builds
            the graph of a native broadcast pipeline on first capture.
        runtime_config: Filled graph-sized Host configuration.
        kernel_name: Stable name shared by both tools.
        parameter_count: Number of launch binding entries. Device task positions
            must match input_list exactly; zero-task Host boundary nodes may
            describe results produced outside the MegaKernel.
        device: Device receiving the ordinary runtime configuration.
        rank: Distributed rank recorded by diagnostics.
        owner_label: Business display label, such as Expert, Head or Block.
        owner_resolver: Optional task-owner callback, never called on the normal path.
        owner_context: Business context passed unchanged to the owner callback.
        execution_mode: Event-driven queues or a broadcast ``aiv_pipeline``.
        vector_worker_stride: Two for paired MoE workers, one for all AIV workers.
        active_vector_workers: Optional active AIV count for a partial-worker policy.
        graph_adapter: Optional lazy business-owned binding of graph ports to the
            real native ABI. It receives a private Host config; ordinary storage
            and the caller's graph must remain unchanged.

    Returns:
        Adapter used by any MegaKernel; dynamic view resolvers remain dump-owned.
    """
    if graph is None and graph_adapter is None:
        raise ValueError("a compute graph or graph_adapter is required")
    if graph_adapter is not None and not callable(graph_adapter):
        raise TypeError("graph_adapter must be callable or None")
    resolved_device = torch.device(device)
    device_id = resolved_device.index
    if device_id is None:
        device_id = 0 if resolved_device.type == "cpu" else torch.get_device_module(resolved_device).current_device()
        resolved_device = torch.device(resolved_device.type, device_id)

    def tensor_factory(data: bytes) -> torch.Tensor:
        """Copy one runtime image onto the execution device.

        Args:
            data: Serialized runtime configuration bytes.
        """
        return torch.frombuffer(bytearray(data), dtype=torch.uint8).clone().to(device=resolved_device)

    if execution_mode not in ("event_driven", "aiv_pipeline"):
        raise ValueError(f"invalid execution mode: {execution_mode!r}")
    if vector_worker_stride not in (1, 2):
        raise ValueError("vector_worker_stride must be 1 or 2")
    runtime = PreparedDiagnosticRuntime(
        graph, runtime_config, DiagnosticSpec(
            kernel_name, owner_label, owner_resolver, owner_context,
            execution_mode, vector_worker_stride, graph_adapter, active_vector_workers,
        ),
        parameter_count=parameter_count, tensor_factory=tensor_factory, rank=rank, device_id=device_id,
    )
    return MegaKernelDiagnostics(runtime)
