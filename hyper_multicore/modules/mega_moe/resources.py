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
"""Precision-independent workspace, lease and distributed resource ownership."""
from collections.abc import Mapping
from dataclasses import replace
from typing import Any
import torch
import torch.distributed as dist
from hyper_multicore import _collective, shmem
from . import transport as moe_transport
from .heap_manager import _get_heap_manager, _workspace_layout
from .spec import MegaMoeSpec, bind_mega_moe_spec
from .workspace import MegaMoeWorkspace, _workspace_event_experts

class _MegaMoeExecutionResources:
    """Own one shape-bound plan and workspace in the shared SHMEM lifecycle."""

    @torch.inference_mode(False)
    def __init__(
        self,
        specification: dict[str, Any],
        tensor: torch.Tensor,
        *,
        expert_weight_layout: str,
        shared: bool,
        active_specifications: tuple[Mapping[str, Any], ...],
    ) -> None:
        """Bind resources once to the first NPU tensor."""
        bound_specification = {**specification, "expert_weight_layout": expert_weight_layout}
        self.spec = bind_mega_moe_spec(bound_specification, tensor)
        self._token_capacity = self.spec.local_num_tokens
        self._plans = {}
        if dist.is_initialized():
            self._validate_distributed_configuration(active_specifications)
        # Agree on optional stages before any HotReplica-only collective.
        placement = specification.get("hot_replica", False)
        replica_bf16_partials = False
        if placement:
            from hyper_multicore.modules.hot_replica.capability import require_capability  # pylint: disable=import-outside-toplevel
            replica_bf16_partials = require_capability(
                tensor, specification["ep_group"],
                dispatch_mode=self.spec.dispatch_mode,
            )
            if self.spec.architecture == 2201 and self.spec.dispatch_mode == "pull":
                self.spec = replace(self.spec, compact_pull=True)
        reservations = tuple((item, _workspace_layout(item, tensor.dtype)) for item in active_specifications)
        self.heap_manager = _get_heap_manager(self.spec.ep_group, tensor)
        added_specifications = self.heap_manager.reserve(
            reservations,
            allow_resize=any(item.get("dispatch_mode", "push") == "push" or item.get("auto_tokens", False)
                             for item in active_specifications),
        )
        attached = False
        self.plan = None
        self.workspace = None
        try:
            attached = self.heap_manager.attach_runtime()

            def construct() -> None:
                """Build and bind local resources before converging construction results."""
                transport = self.spec.transport_engine
                if transport in (None, "udma"):
                    config = shmem.debug_state()["config"]
                    active_udma = config["data_engine"] == "mte_udma"
                    if transport == "udma" and not active_udma:
                        raise ValueError("UDMA tasks require SHMEM initialized with mte_udma")
                    if transport is None:
                        transport = "udma" if active_udma else "mte"
                        self.spec = replace(self.spec, transport_engine=transport)
                if transport == "udma":
                    moe_transport.require_queue_backend()
                self.plan = self._initial_plan(tensor)
                self.workspace = MegaMoeWorkspace(
                    shared=shared, event_experts=_workspace_event_experts(specification) if placement else None,
                )
                self.workspace.transport_engine = self.spec.transport_engine
                self.workspace.capacity_floor = self.heap_manager.bind(self.workspace, self.spec, specification)
                self.workspace.heap_manager = self.heap_manager
                self.spec = self.heap_manager.workspace_spec(self.workspace)
                self._token_capacity = self.spec.local_num_tokens

            _collective.collective_call(
                construct, stage="MegaMoe resource construction", group=self.heap_manager.root_group,
            )
        except Exception:
            self.heap_manager.rollback_reservations(added_specifications, self.workspace)
            if attached and not self.heap_manager.workspaces:
                self.heap_manager.detach_runtime()
            raise
        self._closed = False
        self._workspace_closed = False
        self.placement = None
        if placement:
            from hyper_multicore.modules.hot_replica.mega_moe_adapter import PlacementResources  # pylint: disable=import-outside-toplevel
            self.placement = PlacementResources(self, specification, shared, bf16_partials=replica_bf16_partials)

    def _route_spec(self, experts: int | None) -> MegaMoeSpec:
        """Change the graph's logical width within the same physical workspace."""
        spec = self.spec
        if experts is None or experts == spec.num_experts:
            return spec
        limit = self.workspace.event_experts or spec.num_experts
        if not spec.num_experts < experts <= limit or experts % spec.ep_size:
            raise ValueError("HotReplica route geometry exceeds the shared workspace")
        return replace(spec, num_experts=experts)

    def ensure_capacity(self, maximum_received_slots: int) -> None:
        """Submit a larger layout for the validated route.

        Args:
            maximum_received_slots: Maximum destination load from the root count exchange.
        """
        self.heap_manager.ensure_capacity(self.workspace, maximum_received_slots)

    def lifecycle_signature(self) -> tuple[Any, ...]:
        """Identify allocations independently of process-local group addresses."""
        return (
            "mega_moe", self.placement is not None, self.spec.local_num_tokens, self.spec.hidden_size,
            self.spec.intermediate_size, self.spec.num_experts, self.spec.top_k,
            self.spec.receive_capacity, self.spec.ep_size, self.spec.dispatch_mode,
            self.spec.expert_weight_layout, self.spec.apply_probs_before_fc2, self.spec.partial_weight_grad_dtype,
            self.spec.activation_parameters,
            self.spec.precision, self.spec.transport_engine, self.spec.ready_depth,
            self.spec.schedule_policy, self.spec.lowering,
            self.spec.initial_capacity_factor, self.spec.capacity_growth_factor,
            self.spec.dispatch_split, self.spec.combine_split, self.spec.max_local_num_tokens, self.spec.auto_tokens,
        )

    def prepare_auto_workspace(self, tokens: int, received: int, tensor: torch.Tensor) -> None:
        """Commit physical capacity before taking a pull lease and publishing source rows.

        Args:
            tokens: Aligned group-maximum actual input length.
            received: Group-maximum received route rows.
            tensor: Input supplying device and dtype.
        """
        physical_spec = self.heap_manager.ensure_token_capacity(self.workspace, tokens, received)
        self.workspace.ensure(physical_spec, tensor.dtype, tensor.device)
        self._token_capacity = physical_spec.local_num_tokens
        if self.spec.dispatch_mode == "pull":
            self.workspace.claim()

    def can_close(self) -> bool:
        """Report whether neither an active call nor a backward graph needs buffers."""
        return ((self._workspace_closed or self.workspace.can_close())
                and (self.placement is None or self.placement.can_close()))

    def _validate_distributed_configuration(self, specifications: tuple[Mapping[str, Any], ...]) -> None:
        """Reject different allocation manifests before acquiring symmetric memory."""
        if self.spec.ep_size == 1:
            return
        fields = ("local_num_tokens", "max_local_num_tokens", "hidden_size", "intermediate_size",
                  "num_experts", "top_k", "initial_capacity_factor", "capacity_growth_factor", "dispatch_mode",
                  "dispatch_split", "combine_split", "activation_parameters",
                  "apply_probs_before_fc2", "partial_weight_grad_dtype", "expert_weight_layout",
                  "ep_size", "auto_tokens",
                  "precision", "transport_engine", "ready_depth", "schedule_policy", "lowering",
                  "batch_experts", "nominal_schedule", "hot_replica", "planner_key")
        # BF16 is the sole supported dtype; the literal tags also identify the protocol.
        manifest = ("dynamic_tokens_v3", "bfloat16", self.spec.expert_weight_layout, self.spec.architecture,
                    tuple(tuple(spec.get(key) for key in fields) for spec in specifications))
        # Resource groups must have the same order as their symmetric allocations.
        manifests = [None] * self.spec.ep_size
        dist.all_gather_object(manifests, manifest, group=self.spec.ep_group)
        if any(value != manifests[0] for value in manifests):
            raise ValueError("MegaMoe resource configurations differ across EP ranks")


    def close(self) -> None:
        """Release the workspace and leave the shared SHMEM lifecycle."""
        if self._closed:
            return
        if self.placement is not None:
            self.placement.close()
        if not self._workspace_closed:
            self.heap_manager.close_workspace(self.workspace)
            self._workspace_closed = True
        self._plans.clear()
        self._closed = True

    def _initial_plan(self, tensor):
        raise NotImplementedError("Precision backend must supply an initial plan")
