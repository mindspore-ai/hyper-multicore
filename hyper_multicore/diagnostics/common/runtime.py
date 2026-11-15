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
"""Ordinary runtime storage and lazy, tool-neutral diagnostic metadata."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from enum import IntEnum
from threading import Lock
from typing import Any

from hyper_multicore.diagnostics.common.graph import (
    DiagnosticSpec, GraphDiagnosticMetadata, build_graph_metadata, validate_mega_kernel_diagnostic_contract,
)
from hyper_multicore.scheduler.config import RuntimeConfigC
from hyper_multicore.scheduler.graph import ComputeGraph
from hyper_multicore.scheduler.runtime import copy_runtime_config, serialize_runtime_config


class DiagnosticMode(IntEnum):
    """Device execution modes sharing one diagnostic_buffer ABI argument."""

    NORMAL = 0
    PROFILE = 1
    TENSOR_DUMP_L1 = 2


class PreparedDiagnosticRuntime:
    """Own the ordinary runtime; prepare diagnostic metadata only on demand."""

    def __init__(
        self, graph: ComputeGraph, runtime_config: RuntimeConfigC, spec: DiagnosticSpec, *,
        parameter_count: int, tensor_factory: Callable[[bytes], Any], rank: int, device_id: int,
    ) -> None:
        """Save the graph contract and create only the normal execution image."""
        self._graph = graph
        self.spec = spec
        self.parameter_count = parameter_count
        self.rank = rank
        self.device_id = device_id
        self.kernel_name = spec.kernel_name
        self.owner_label = spec.owner_label
        self._tensor_factory = tensor_factory
        self._host_config = copy_runtime_config(runtime_config)
        self._host_config.diagnostic_mode = DiagnosticMode.NORMAL
        self.normal_tensor = tensor_factory(serialize_runtime_config(self._host_config))
        self._metadata: GraphDiagnosticMetadata | None = None
        self._metadata_lock = Lock()
        self._diagnostic_config: RuntimeConfigC | None = None

    @property
    def graph(self) -> ComputeGraph:
        """Return the diagnostic graph after lazy business-owned ABI binding."""
        if self.spec.graph_adapter is not None and self._metadata is None:
            _ = self.metadata
        return self._graph

    @property
    def host_config(self) -> RuntimeConfigC:
        """Return the preserved Host configuration used to describe task operands."""
        if self.spec.graph_adapter is not None:
            _ = self.metadata
        return self._diagnostic_config if self._diagnostic_config is not None else self._host_config

    @property
    def metadata(self) -> GraphDiagnosticMetadata:
        """Validate and resolve graph identity only when a tool actually needs it."""
        if self._metadata is None:
            with self._metadata_lock:
                if self._metadata is None:
                    config = copy_runtime_config(self._host_config)
                    graph = self._graph
                    if self.spec.graph_adapter is not None:
                        graph = self.spec.graph_adapter(graph, config)
                    validate_mega_kernel_diagnostic_contract(graph, config, self.parameter_count)
                    metadata = build_graph_metadata(graph, config, self.spec)
                    self._graph = graph
                    self._diagnostic_config = config
                    self._metadata = metadata
        return self._metadata

    @property
    def task_stage_names(self) -> Mapping[int, str]:
        """Return the lazily resolved task-to-stage names."""
        return self.metadata.task_stage_names

    def copy_host_config(self) -> RuntimeConfigC:
        """Create an independent Host configuration for one tool's device variant."""
        config = self.host_config
        return copy_runtime_config(config)

    def tensor_from_config(self, runtime_config: RuntimeConfigC) -> Any:
        """Serialize a tool-owned configuration without changing the normal image.

        Args:
            runtime_config: Complete Host configuration describing native task queues.
        """
        return self._tensor_factory(serialize_runtime_config(runtime_config))
