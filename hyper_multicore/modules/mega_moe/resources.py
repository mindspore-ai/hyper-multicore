# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
"""Precision-independent MoE storage, plan ownership and SHMEM lifecycle."""
from collections.abc import Mapping
from typing import Any
import torch
import torch.distributed as dist
from hyper_multicore import _collective, shmem
from .heap_manager import _get_heap_manager, _workspace_layout
from .spec import bind_mega_moe_spec
from .workspace import MegaMoeWorkspace

class _MegaMoeExecutionResources:
    """Own one shape-bound plan and workspace in the shared SHMEM lifecycle."""

    def __init__(
        self,
        specification: dict[str, Any],
        tensor: torch.Tensor,
        *,
        shared: bool,
        active_specifications: tuple[Mapping[str, Any], ...],
    ) -> None:
        """Bind resources once to the first NPU tensor."""
        self.spec = bind_mega_moe_spec(specification, tensor)
        self._token_capacity = self.spec.local_num_tokens
        self._plans = {}
        if dist.is_initialized():
            self._validate_distributed_configuration(active_specifications)
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
                if self.spec.transport_engine == "udma":
                    from .transport import require_queue_backend
                    require_queue_backend()
                    if shmem.debug_state()["config"]["data_engine"] != "mte_udma":
                        raise ValueError("UDMA tasks require SHMEM initialized with mte_udma")
                self.plan = self._initial_plan(tensor)
                self.workspace = MegaMoeWorkspace(shared=shared)
                self.workspace.transport_engine = self.spec.transport_engine
                self.workspace.capacity_floor = self.heap_manager.bind(self.workspace, self.spec, specification)
                self.workspace.heap_manager = self.heap_manager

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


    def _initial_plan(self, tensor):
        raise NotImplementedError("Precision backend must supply its initial plan")

    def ensure_capacity(self, maximum_received_slots: int) -> None:
        """Submit a larger layout, falling back to minimum capacity within an explicit budget.

        Args:
            maximum_received_slots: Maximum destination load from the root count exchange.
        """
        self.heap_manager.ensure_capacity(self.workspace, maximum_received_slots)

    def lifecycle_signature(self) -> tuple[Any, ...]:
        """Identify allocations independently of process-local group addresses."""
        return (
            "mega_moe", self.spec.local_num_tokens, self.spec.hidden_size,
            self.spec.intermediate_size, self.spec.num_experts, self.spec.top_k,
            self.spec.receive_capacity, self.spec.ep_size, self.spec.dispatch_mode,
            self.spec.expert_weight_layout,
            self.spec.initial_capacity_factor, self.spec.capacity_growth_factor,
            self.spec.dispatch_split, self.spec.combine_split, self.spec.max_local_num_tokens, self.spec.auto_tokens,
            self.spec.transport_engine, self.spec.ready_depth, self.spec.precision,
            self.spec.schedule_policy,
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
        return self._workspace_closed or self.workspace.can_close()

    def _validate_distributed_configuration(self, specifications: tuple[Mapping[str, Any], ...]) -> None:
        """Reject different allocation manifests before acquiring symmetric memory."""
        if self.spec.ep_size == 1:
            return
        fields = ("local_num_tokens", "max_local_num_tokens", "hidden_size", "intermediate_size",
                  "num_experts", "top_k", "initial_capacity_factor", "capacity_growth_factor", "dispatch_mode",
                  "dispatch_split", "combine_split", "swiglu_limit", "expert_weight_layout", "ep_size", "auto_tokens",
                  "transport_engine", "ready_depth", "precision", "schedule_policy")
        # BF16 is the sole supported dtype; the literal tags also identify the protocol.
        manifest = ("dynamic_tokens_v3", "bfloat16",
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
        if not self._workspace_closed:
            self.heap_manager.close_workspace(self.workspace)
            self._workspace_closed = True
        self._plans.clear()
        self._closed = True
