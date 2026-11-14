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
"""Model-facing Torch MegaMoe expert module."""

from __future__ import annotations

__all__ = ["MegaMoeExperts"]

import math
import os
import struct
from collections.abc import Mapping
from typing import Any, Literal

import torch
import torch.distributed as dist

from hyper_multicore import shmem

from ..module import MulticoreModule
from .transport import execution_options, requested_transport
from .schedule import MegaMoeScheduleOptions, resolve_schedule_options
from .spec import (
    _COMMUNICATION_SPLIT,
    _resolve_capacity_factors,
)
from .backends import get_backend
from .resources import _MegaMoeExecutionResources


ExpertWeightLayout = Literal["EIO", "EOI"]
_EXPERT_WEIGHT_LAYOUTS = frozenset(("EIO", "EOI"))


def _validate_swiglu_limit(swiglu_limit: float | None) -> None:
    """Validate a positive clamp value that survives float32 serialization."""
    if swiglu_limit is None:
        return
    valid_type = isinstance(swiglu_limit, (int, float)) and not isinstance(
        swiglu_limit, bool
    )
    try:
        encoded_limit = (
            struct.unpack("<f", struct.pack("<f", float(swiglu_limit)))[0]
            if valid_type
            else 0.0
        )
        valid_value = (
            valid_type
            and math.isfinite(swiglu_limit)
            and math.isfinite(encoded_limit)
            and encoded_limit > 0
        )
    except (OverflowError, TypeError, ValueError, struct.error):
        valid_value = False
    if not valid_value:
        raise ValueError(
            "swiglu_limit must be None or a finite positive float32-representable number, "
            f"got {swiglu_limit!r}."
        )


def _create_mega_moe_parameters(
    local_experts: int,
    hidden_size: int,
    intermediate_size: int,
    expert_weight_layout: ExpertWeightLayout,
) -> tuple[torch.nn.Parameter, torch.nn.Parameter]:
    """Create independently initialized local expert parameters."""
    if expert_weight_layout == "EOI":
        gate_up_shape = (local_experts, intermediate_size * 2, hidden_size)
        down_shape = (local_experts, hidden_size, intermediate_size)
    else:
        gate_up_shape = (local_experts, hidden_size, intermediate_size * 2)
        down_shape = (local_experts, intermediate_size, hidden_size)
    gate_up_weight = torch.nn.Parameter(
        torch.empty(gate_up_shape)
    )
    down_weight = torch.nn.Parameter(
        torch.empty(down_shape)
    )
    for expert_index in range(local_experts):
        torch.nn.init.xavier_uniform_(gate_up_weight[expert_index])
        torch.nn.init.xavier_uniform_(down_weight[expert_index])
    return gate_up_weight, down_weight


def _resolve_ep_size(ep_size: int | None, ep_group: Any | None) -> int:
    """Infer size from an explicit group without silently choosing WORLD."""
    if ep_size is not None and (not isinstance(ep_size, int) or isinstance(ep_size, bool) or ep_size <= 0):
        raise ValueError("ep_size must be a positive integer or None")
    if ep_group is not None:
        if not dist.is_initialized():
            raise ValueError("Initialize torch.distributed before passing ep_group")
        group_size = dist.get_world_size(ep_group)
        if ep_size is not None and ep_size != group_size:
            raise ValueError(f"ep_size={ep_size} does not match ep_group size={group_size}")
        return group_size
    if dist.is_initialized() and dist.get_world_size() > 1:
        raise ValueError("Pass ep_group explicitly for distributed MegaMoeExperts")
    return 1 if ep_size is None else ep_size


class MegaMoeExperts(MulticoreModule):
    """Execute Router-selected local experts with the Torch MegaMoe kernel.

    Serial model layers can call :meth:`share_execution_resources` before
    their first forward to share one automatically growing workspace while keeping
    independent parameters and optimizer state.
    """

    @classmethod
    def from_module(
        cls, module: torch.nn.Module, *, top_k: int, ep_group: Any | None = None, **execution_options: Any,
    ) -> MegaMoeExperts:
        """Convert materialized local GroupedExperts SwiGLU weights before optimizer construction.

        Args:
            module: Module exposing w1/w3 [local_E, I, H], w2 [local_E, H, I].
                DTensor local EP shards are accepted; FSDP weights must be materialized first.
            top_k: Experts selected by the existing router for each token.
            ep_group: Explicit EP group owning these local experts, or None for a standalone rank.
            **execution_options: Advanced MegaMoe options, including num_experts and expert_weight_layout.

        Returns:
            Experts with copied, converted Parameters. Optimizer state is not migrated.
        """
        if execution_options.get("precision", "bf16") != "bf16":
            raise ValueError("from_module converts BF16 weights; MXFP8 requires quantized physical weight roots")
        values = []
        for name in ("w1", "w3", "w2"):
            value = getattr(module, name, None)
            if hasattr(value, "to_local"):
                value = value.to_local()
            if not isinstance(value, torch.Tensor) or value.ndim != 3:
                raise ValueError(f"source {name} must be a materialized local rank-three expert tensor")
            if value.is_meta:
                raise ValueError(f"source {name} is meta; materialize and initialize/load weights before from_module")
            values.append(value)
        gate, up, down = values[0], values[1], values[2]
        if (up.shape != gate.shape or down.shape != (gate.shape[0], gate.shape[2], gate.shape[1])
                or any(value.device != gate.device or value.dtype != gate.dtype for value in values)
                or gate.requires_grad != up.requires_grad):
            raise ValueError("source SwiGLU expert shapes, dtypes, devices and gate/up gradient policies must match")
        ep_size = _resolve_ep_size(execution_options.get("ep_size"), ep_group)
        experts = execution_options.pop("num_experts", getattr(module, "num_experts", gate.shape[0] * ep_size))
        if experts != gate.shape[0] * ep_size:
            raise ValueError("source local expert count does not match num_experts and ep_group")
        result = cls(hidden_size=gate.shape[2], intermediate_size=gate.shape[1], num_experts=experts,
                     top_k=top_k, ep_group=ep_group, create_parameters=False, **execution_options)
        gate_up = torch.cat((gate.detach(), up.detach()), dim=1)
        down_data = down.detach()
        if result.expert_weight_layout == "EIO":
            gate_up = gate_up.transpose(1, 2)
            down_data = down_data.transpose(1, 2)
        result.gate_up_weight = torch.nn.Parameter(
            gate_up.contiguous(), requires_grad=gate.requires_grad)
        result.down_weight = torch.nn.Parameter(
            down_data.clone(memory_format=torch.contiguous_format),
            requires_grad=down.requires_grad)
        return result

    @property
    def _backend(self):
        return get_backend(self.precision)

    def _resource_signature(self, tensor: Any) -> Any:
        """Compare static configuration without process-local group identities."""
        configuration = tuple(sorted((name, value) for name, value in self._resource_group.specification.items()
                                     if name != "ep_group"))
        return (super()._resource_signature(tensor), configuration,
                self._resource_group.shared)

    def _register_runtime_release(self, manager: Any, resources: Any) -> None:
        """Register the Heap Manager's sole Runtime reference for final cleanup."""
        manager.runtime_release = resources.heap_manager.detach_runtime

    def __init__(
        self,
        *,
        local_num_tokens: int | None = None,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        swiglu_limit: float | None = None,
        ep_size: int | None = None,
        ep_group: Any | None = None,
        create_parameters: bool = True,
        expert_weight_layout: ExpertWeightLayout = "EIO",
        dispatch_mode: str | None = None,
        initial_capacity_factor: float | None = None,
        capacity_growth_factor: float | None = None,
        dispatch_split: int | None = None,
        combine_split: int | None = None,
        schedule: MegaMoeScheduleOptions | None = None,
        precision: Literal["bf16", "mxfp8"] = "bf16",
    ) -> None:
        """Initialize local expert parameters and a lazy execution owner.

        Args:
            hidden_size: Input and output hidden dimension.
            intermediate_size: SwiGLU intermediate dimension per expert.
            num_experts: Global routed-expert count.
            top_k: Experts selected for every token.
            initial_capacity_factor: Push initial receive rows as a multiple of local routed rows.
                Defaults to 1.25 for push; omitted for pull. Must be finite and at least 1.0.
            capacity_growth_factor: Push capacity multiplier on overflow, defaulting to 1.25.
                Must be finite and at least 1.0; 1.0 grows only to the current route demand.
                The resulting capacity is capped by the lossless route bound. Pull rejects explicit factors.
            dispatch_mode: Dispatch transport, either "push" (default) or "pull".
                Construct separate modules to switch modes; sharing requires equal modes.
            dispatch_split: Dispatch tile rows in forward and backward, default 128.
                Must be a positive multiple of 128.
            combine_split: Combine tile rows in forward and backward, with the same
                constraints and default as dispatch_split. Sharing requires equal splits.
            precision: BF16 (default) or MXFP8; both use the same forward API.
            schedule: Optional policy and local math-queue lookahead, resolved once.
                Communication remains FIFO. Configure before resource binding.
            swiglu_limit: Optional positive, finite float32-representable clamp
                limit for SwiGLU. The gate branch uses ``min(gate, limit)`` and
                the up branch is clamped to ``[-limit, limit]``. ``None``
                preserves the legacy unclamped path.
            ep_size: Optional compatibility check; normally inferred from explicit ep_group.
            create_parameters: Allocate owned weights; False requires explicit expert_weights each forward.
            expert_weight_layout: Logical expert-weight axis order. ``"EIO"`` means
                ``[experts, in_features, out_features]`` and preserves the historical layout. ``"EOI"`` means
                ``[experts, out_features, in_features]`` and accepts Torch linear storage directly. ``EOI`` weights
                must use dense contiguous storage for zero-copy execution.
            ep_group: Explicit EP group; expert ownership follows group-local rank order.
                Disjoint PP/DP groups bootstrap independently. Only one ordered EP membership
                can be active in SHMEM per process.
        """
        get_backend(precision)
        if precision == "mxfp8" and (local_num_tokens is None or create_parameters):
            raise ValueError("MXFP8 requires explicit local_num_tokens and create_parameters=False")
        implicit_mode = dispatch_mode is None
        dispatch_mode, dispatch_split, combine_split = execution_options(
            dispatch_mode, dispatch_split, combine_split, _COMMUNICATION_SPLIT)
        if implicit_mode and dispatch_mode == 'pull':
            initial_capacity_factor = capacity_growth_factor = None
        specification = {
            "local_num_tokens": _COMMUNICATION_SPLIT if local_num_tokens is None else local_num_tokens,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_experts": num_experts,
            "top_k": top_k,
            "initial_capacity_factor": initial_capacity_factor,
            "swiglu_limit": swiglu_limit,
            "ep_size": _resolve_ep_size(ep_size, ep_group),
            "ep_group": ep_group,
            "expert_weight_layout": expert_weight_layout,
            "dispatch_mode": dispatch_mode,
            "capacity_growth_factor": capacity_growth_factor,
            "dispatch_split": dispatch_split,
            "combine_split": combine_split,
            "schedule": schedule,
            "precision": precision,
        }
        specification["auto_tokens"] = local_num_tokens is None
        specification = self._resolve_specification(specification)
        compatibility_key = tuple(
            id(value) if name == "ep_group" else value for name, value in specification.items())
        super().__init__(
            resource_specification=specification,
            resource_compatibility_key=compatibility_key,
            resource_scope_key=("mega_moe", shmem.root_members(ep_group) if dist.is_initialized() else id(ep_group)),
        )
        self._initialize_module_state(specification, create_parameters=create_parameters)

    def _initialize_module_state(
        self, specification: Mapping[str, Any], *, create_parameters: bool,
    ) -> None:
        """Bind normalized configuration and initialize expert parameters."""
        self.precision = specification.get("precision", "bf16")
        self.hidden_size = specification["hidden_size"]
        self.intermediate_size = specification["intermediate_size"]
        self.num_experts = specification["num_experts"]
        self.top_k = specification["top_k"]
        self.expert_weight_layout = specification["expert_weight_layout"]
        self.initial_capacity_factor = specification["initial_capacity_factor"]
        self.capacity_growth_factor = specification["capacity_growth_factor"]
        self.dispatch_mode = specification["dispatch_mode"]
        self.dispatch_split = specification["dispatch_split"]
        self.combine_split = specification["combine_split"]
        self.swiglu_limit = specification["swiglu_limit"]
        self.ep_size = specification["ep_size"]
        self.local_experts = self.num_experts // self.ep_size
        self._ep_group = specification["ep_group"]
        no_copy = os.environ.get("HYPER_MULTICORE_BF16_NOGRAD_NO_COPY", "1")
        if no_copy not in ("0", "1"):
            raise ValueError("BF16 no-grad no-copy must be 0 or 1")
        self.nograd_no_copy = no_copy == "1"
        if self.precision == "mxfp8":
            no_clone = os.environ.get("HYPER_MULTICORE_MX_NOGRAD_NO_CLONE", "1")
            if no_clone not in ("0", "1"):
                raise ValueError("MX no-grad no-clone must be 0 or 1")
            self.nograd_no_clone = no_clone == "1"
        if create_parameters:
            self.gate_up_weight, self.down_weight = _create_mega_moe_parameters(
                self.local_experts, self.hidden_size, self.intermediate_size, self.expert_weight_layout,
            )
        else:
            self.register_parameter("gate_up_weight", None)
            self.register_parameter("down_weight", None)

    @classmethod
    def _resolve_specification(cls, specification: dict[str, Any]) -> dict[str, Any]:
        """Validate and normalize configuration before registering an execution owner."""
        specification = dict(specification)
        schedule = resolve_schedule_options(specification.pop("schedule", None),
                                            precision=specification.get("precision", "bf16"))
        expert_weight_layout = specification["expert_weight_layout"]
        if expert_weight_layout not in _EXPERT_WEIGHT_LAYOUTS:
            raise ValueError(
                "expert_weight_layout must be 'EIO' ([experts, in_features, out_features]) "
                "or 'EOI' ([experts, out_features, in_features]), "
                f"got {expert_weight_layout!r}."
            )
        initial_capacity_factor, capacity_growth_factor = _resolve_capacity_factors(
            specification["dispatch_mode"],
            specification["initial_capacity_factor"],
            specification["capacity_growth_factor"],
        )
        cls._validate_topology(
            local_num_tokens=specification["local_num_tokens"],
            hidden_size=specification["hidden_size"],
            intermediate_size=specification["intermediate_size"],
            num_experts=specification["num_experts"],
            top_k=specification["top_k"],
            swiglu_limit=specification["swiglu_limit"],
            ep_size=specification["ep_size"],
            dispatch_split=specification["dispatch_split"],
            combine_split=specification["combine_split"],
            auto_tokens=specification.get("auto_tokens", False),
        )
        swiglu_limit = specification["swiglu_limit"]
        if specification.get("precision") == "mxfp8":
            if specification["hidden_size"] % 64 or specification["intermediate_size"] % 64:
                raise ValueError("MX hidden/intermediate dimensions must be divisible by 64")
            if expert_weight_layout != "EIO" or swiglu_limit is not None:
                raise ValueError("MXFP8 requires EIO weights and unclamped activation")
        return {
            **specification,
            "transport_engine": requested_transport(),
            "ready_depth": schedule.queue_ready_lookahead,
            "schedule_policy": schedule.policy,
            "precision": specification.get("precision", "bf16"),
            "initial_capacity_factor": initial_capacity_factor,
            "capacity_growth_factor": capacity_growth_factor,
            "swiglu_limit": float(swiglu_limit) if swiglu_limit is not None else None,
        }

    @staticmethod
    def _validate_topology(
        *,
        local_num_tokens: int,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        swiglu_limit: float | None,
        ep_size: int,
        dispatch_split: int,
        combine_split: int,
        auto_tokens: bool = False,
    ) -> None:
        """Validate static shape and topology values before allocation."""
        values = {
            "local_num_tokens": local_num_tokens,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_experts": num_experts,
            "top_k": top_k,
            "ep_size": ep_size,
        }
        for name, value in values.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}.")
        if top_k > num_experts:
            raise ValueError(
                f"top_k ({top_k}) cannot exceed num_experts ({num_experts})."
            )
        if num_experts % ep_size:
            raise ValueError(
                f"num_experts ({num_experts}) must be divisible by ep_size ({ep_size})."
            )
        _validate_swiglu_limit(swiglu_limit)
        for name, split in (("dispatch_split", dispatch_split), ("combine_split", combine_split)):
            if (not isinstance(split, int) or isinstance(split, bool)
                    or split <= 0 or split % _COMMUNICATION_SPLIT):
                raise ValueError(
                    f"{name} must be a positive integer multiple of {_COMMUNICATION_SPLIT}, got {split!r}.")
            if not auto_tokens and local_num_tokens % split:
                raise ValueError(f"local_num_tokens ({local_num_tokens}) must be divisible by {name} ({split}).")

    def _root_group(self) -> Any:
        """Keep lifecycle coordination within this module's EP group."""
        return self._ep_group

    def _validate_tensors(self, hidden_states: torch.Tensor, expert_weights: tuple) -> None:
        """Validate activation and parameter metadata before acquiring resources."""
        if hidden_states.dtype != torch.bfloat16 or not getattr(hidden_states, "is_npu", False):
            raise TypeError("MegaMoeExperts requires BF16 NPU hidden states.")
        self._backend.validate_weights(self, expert_weights, hidden_states)

    def _validate_forward_inputs(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        tokens_per_expert: torch.Tensor | None,
        expert_weights: tuple,
    ) -> torch.Tensor:
        """Validate input shapes and return flattened local token states."""
        if hidden_states.ndim < 2 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                "hidden_states must have at least two dimensions and configured "
                f"hidden size {self.hidden_size}, got {tuple(hidden_states.shape)}."
            )
        hidden_flat = hidden_states.reshape(-1, self.hidden_size)
        route_shape = (hidden_flat.shape[0], self.top_k)
        if tuple(topk_ids.shape) != route_shape:
            raise ValueError(
                f"topk_ids must have shape {route_shape}, got {tuple(topk_ids.shape)}."
            )
        if tuple(topk_weights.shape) != route_shape:
            raise ValueError(
                f"topk_weights must have shape {route_shape}, "
                f"got {tuple(topk_weights.shape)}."
            )
        if tokens_per_expert is not None and tuple(tokens_per_expert.shape) != (
            self.num_experts,
        ):
            raise ValueError(
                f"tokens_per_expert must have shape ({self.num_experts},), "
                f"got {tuple(tokens_per_expert.shape)}."
            )
        self._validate_tensors(hidden_flat, expert_weights)
        return hidden_flat

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        tokens_per_expert: torch.Tensor | None = None,
        expert_weights: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Route token states through the configured experts.

        Args:
            hidden_states: Tensor ending in ``hidden_size``.
            topk_ids: Global expert IDs shaped ``[tokens, top_k]`` for the current input.
            topk_weights: Router weights with the same shape as ``topk_ids``.
            expert_weights: Current unsharded weights owned by an outer module.
                Noncontiguous EIO weights are copied with autograd tracking, never cached
                by this executor. EOI weights must be contiguous for zero-copy execution.
            tokens_per_expert: Optional trusted exact global-expert histogram.
                Supplying Router-produced counts skips histogram recomputation.

        Returns:
            Tensor matching the shape and dtype of ``hidden_states``.

        Raises:
            ValueError: EOI expert weights are noncontiguous.
            RuntimeError: The current NPU stream is being captured.

        Note:
            Automatic token management requires eager execution for both push
            and pull. NPU graph capture is unsupported, even with stable token
            counts and capacity.

            ``tokens_per_expert`` must describe the current ``topk_ids`` exactly.
            The steady-state path validates metadata but deliberately does not
            rebuild and compare the histogram.
        """
        weights = ((self.gate_up_weight, self.down_weight) if expert_weights is None else expert_weights)
        hidden_flat = self._validate_forward_inputs(
            hidden_states,
            topk_ids,
            topk_weights,
            tokens_per_expert,
            weights,
        )
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("MegaMoe push heap growth and dynamic tokens do not support graph capture")
        resources = self._get_execution_resources(hidden_flat)
        return self._backend.forward(self, hidden_states, hidden_flat, topk_ids, topk_weights,
                                     tokens_per_expert, weights, resources)

    def _create_execution_resources(
        self,
        tensor: torch.Tensor,
        *,
        shared: bool,
        active_specifications: tuple[Mapping[str, Any], ...],
    ) -> _MegaMoeExecutionResources:
        """Create the plan and workspace owned by this resource group."""
        return self._backend.Resources(
            self._resource_group.specification,
            tensor,
            shared=shared,
            active_specifications=active_specifications,
        )
