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

from collections.abc import Mapping
from typing import Any

import torch
import torch.distributed as dist

from hyper_multicore import shmem
from hyper_multicore.modules.mega_moe.numerics import (
    MegaMoeNumericsOptions, SiTUGLUParameters, SwiGLUParameters,
)
from hyper_multicore.experimental import MegaMoeTuningOptions

from ..module import MulticoreModule
from .backends import get_backend
from .invocation import prepare_expert_invocation
from .resources import _MegaMoeExecutionResources
from ..base_planner import BasePlanner
from .spec import (
    _COMMUNICATION_SPLIT,
    _resolve_capacity_factors,
)
from .placement_options import MegaMoePlacementOptions
from .transport import requested_transport
from .schedule import validate_schedule_options


class _InvalidExpertWeights(ValueError):
    """Expert metadata that must be rejected consistently across EP ranks."""


class _InvalidExpertWeightType(TypeError):
    """Expert dtype that must be rejected consistently across EP ranks."""


def _resolve_ep_size(ep_group: Any | None) -> int:
    """Infer size from an explicit group without silently choosing WORLD."""
    if ep_group is not None:
        if not dist.is_initialized():
            raise ValueError("Initialize torch.distributed before passing ep_group")
        return dist.get_world_size(ep_group)
    if dist.is_initialized() and dist.get_world_size() > 1:
        raise ValueError("Pass ep_group explicitly for distributed MegaMoeExperts")
    return 1


def _placement_planner(options: MegaMoePlacementOptions | None) -> BasePlanner | None:
    """Validate the placement strategy before allocating any EP resources."""
    if options is None:
        return None
    if not isinstance(options, MegaMoePlacementOptions):
        raise TypeError("expert_placement must be MegaMoePlacementOptions or None")
    if options.strategy == "none":
        if options.planner is not None:
            raise ValueError("placement strategy 'none' requires planner=None")
        return None
    if options.strategy != "adaptive":
        raise ValueError(f"Unsupported expert placement strategy: {options.strategy!r}")
    if not isinstance(options.planner, BasePlanner):
        raise TypeError("placement strategy 'adaptive' requires a BasePlanner instance")
    return options.planner


def _validate_precision_schedule(precision, scheduler):
    """Check precision-specific scheduling constraints."""
    if precision == "mxfp8" and (scheduler.batch_experts or scheduler.nominal_schedule
                                 or scheduler.lowering == "compact"):
        raise ValueError("MXFP8 requires dense per-expert lowering")


def _resolve_activation_parameters(
    activation: SwiGLUParameters | SiTUGLUParameters | None,
) -> SwiGLUParameters:
    """Resolve the default and validate the selected activation type."""
    activation_parameters = SwiGLUParameters() if activation is None else activation
    if isinstance(activation_parameters, SiTUGLUParameters):
        raise NotImplementedError("SiTUGLU execution is not implemented in MegaMoeExperts")
    if not isinstance(activation_parameters, SwiGLUParameters):
        raise TypeError("activation must be SwiGLUParameters, SiTUGLUParameters or None")
    return activation_parameters


def _resolve_expert_options(precision, activation, numerics_options, tuning_options):
    """Validate options that bind numerical semantics and scheduling for the module."""
    if numerics_options is not None and not isinstance(numerics_options, MegaMoeNumericsOptions):
        raise TypeError("numerics_options must be MegaMoeNumericsOptions or None")
    if tuning_options is not None and not isinstance(tuning_options, MegaMoeTuningOptions):
        raise TypeError("tuning_options must be MegaMoeTuningOptions or None")
    numerics = MegaMoeNumericsOptions() if numerics_options is None else numerics_options
    tuning = MegaMoeTuningOptions() if tuning_options is None else tuning_options
    if not isinstance(numerics.apply_probs_before_fc2, bool):
        raise TypeError("apply_probs_before_fc2 must be a boolean")
    if precision not in ("bf16", "mxfp8"):
        raise ValueError("MegaMoE precision must be bf16 or mxfp8")
    if numerics.partial_weight_grad_dtype not in (None, "native", "float32"):
        raise ValueError("partial_weight_grad_dtype must be native, float32 or None")
    scheduler = validate_schedule_options(tuning.scheduler)
    _validate_precision_schedule(precision, scheduler)
    activation_parameters = _resolve_activation_parameters(activation)
    if precision == "mxfp8" and (activation_parameters.clamp_limit is not None or numerics.apply_probs_before_fc2):
        raise ValueError("MXFP8 does not support clamp or before-FC2 probability weighting")
    return numerics, tuning, scheduler, activation_parameters


class MegaMoeExperts(MulticoreModule):
    """Execute Router-selected local experts with the Torch MegaMoe kernel.

    The caller owns expert parameters and passes their current local tensors to
    each forward. Serial layers may share a workspace before their first forward.
    """

    def _resource_signature(self, tensor: Any) -> Any:
        """Compare static configuration without process-local group identities."""
        configuration = tuple(sorted((name, value) for name, value in self._resource_group.specification.items()
                                     if name != "ep_group"))
        return (super()._resource_signature(tensor), configuration,
                self._resource_group.shared, self._pending_layout)

    @property
    def _backend(self):
        """Choose the private precision provider behind the common public contract."""
        return get_backend(self.precision)

    def _register_runtime_release(self, manager: Any, resources: Any) -> None:
        """Register the Heap Manager's sole Runtime reference for final cleanup."""
        manager.runtime_release = resources.heap_manager.detach_runtime

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        ep_group: Any | None = None,
        precision: str = "bf16",
        activation: SwiGLUParameters | SiTUGLUParameters | None = None,
        numerics_options: MegaMoeNumericsOptions | None = None,
        tuning_options: MegaMoeTuningOptions | None = None,
        expert_placement: MegaMoePlacementOptions | None = None,
    ) -> None:
        """Initialize a parameterless lazy execution owner.

        Args:
            hidden_size: Input and output hidden dimension.
            intermediate_size: Intermediate dimension per expert after activation.
            num_experts: Global routed-expert count.
            top_k: Experts selected for every token.
            ep_group: Explicit EP group; expert ownership follows group-local rank order.
                Disjoint PP/DP groups bootstrap independently. Only one ordered EP membership
                can be active in SHMEM per process. Its size determines the local expert count.
                May be omitted for a standalone single-rank process.
            precision: ``"bf16"`` for standard weights or ``"mxfp8"`` for external MX roots.
            activation: Immutable algorithm parameters. None selects ordinary SwiGLU.
                Passing SiTUGLUParameters raises NotImplementedError at construction.
            numerics_options: Probability placement and partial-gradient precision.
                None uses after-FC2 probability weighting and native partial dW.
            expert_placement: Placement strategy and planner. Adaptive placement
                requires A3 BF16 push execution and a multi-rank EP group.
            tuning_options: Public experimental tuning controls, imported from
                ``hyper_multicore.experimental``. None selects the default policy.
                Options are validated and copied into the instance configuration;
                they do not change the chosen probability or clamp semantics.

        Numerical choices and tuning options are fixed for this instance.
        Construct another executor to change them.
        """
        numerics, tuning, scheduler, activation_parameters = _resolve_expert_options(
            precision, activation, numerics_options, tuning_options,
        )
        planner = _placement_planner(expert_placement)
        if planner is not None and (precision != "bf16" or tuning.dispatch_mode != "push"):
            raise ValueError("HotReplica requires BF16 precision and push dispatch")
        source_bound = planner.source_token_bound if planner is not None else None
        planner_key = None
        if planner is not None:
            settings = planner.configuration_key()
            if not isinstance(settings, str):
                raise TypeError("Planner configuration_key() must return a string")
            planner_key = (type(planner).__module__, type(planner).__qualname__, settings)
        specification = {
            "local_num_tokens": _COMMUNICATION_SPLIT,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_experts": num_experts,
            "top_k": top_k,
            "partial_weight_grad_dtype": numerics.partial_weight_grad_dtype,
            "activation_parameters": activation_parameters,
            "apply_probs_before_fc2": numerics.apply_probs_before_fc2,
            "ep_size": _resolve_ep_size(ep_group),
            "ep_group": ep_group,
            "dispatch_mode": tuning.dispatch_mode,
            "initial_capacity_factor": tuning.initial_capacity_factor,
            "capacity_growth_factor": tuning.capacity_growth_factor,
            "dispatch_split": tuning.dispatch_split,
            "combine_split": tuning.combine_split,
            "precision": precision,
            "hot_replica": planner is not None,
            "planner_key": planner_key,
            "source_token_bound": source_bound,
            "transport_engine": requested_transport(),
            "ready_depth": scheduler.queue_ready_lookahead,
            "schedule_policy": scheduler.policy or "auto",
            "batch_experts": bool(scheduler.batch_experts),
            "nominal_schedule": bool(scheduler.nominal_schedule),
            "lowering": scheduler.lowering,
        }
        if planner is not None and specification["ep_size"] <= 1:
            raise ValueError("HotReplica requires an explicit EP group with more than one rank")
        if source_bound is not None:
            limit = (2**31 - 1) // max(hidden_size, top_k) // _COMMUNICATION_SPLIT * _COMMUNICATION_SPLIT
            if source_bound > limit:
                raise ValueError(f"HotReplica source_token_bound exceeds {limit} rows")
        specification["auto_tokens"] = True
        specification = self._resolve_specification(specification)
        compatibility_key = tuple(
            id(value) if name == "ep_group" else value for name, value in specification.items())
        super().__init__(
            resource_specification=specification,
            resource_compatibility_key=(compatibility_key, None),
            resource_scope_key=("mega_moe", shmem.root_members(ep_group) if dist.is_initialized() else id(ep_group)),
        )
        self._base_compatibility_key = compatibility_key
        self._planner = planner
        self._pending_layout = None
        self._initialize_module_state(specification)

    def _initialize_module_state(
        self, specification: Mapping[str, Any],
    ) -> None:
        """Bind normalized configuration and precompute valid weight shapes."""
        self.nograd_no_clone = True
        self.hidden_size = specification["hidden_size"]
        self.intermediate_size = specification["intermediate_size"]
        self.num_experts = specification["num_experts"]
        self.top_k = specification["top_k"]
        self.ep_size = specification["ep_size"]
        self.local_experts = self.num_experts // self.ep_size
        self._ep_group = specification["ep_group"]
        local, hidden, intermediate = self.local_experts, self.hidden_size, self.intermediate_size
        self._weight_shapes = {
            "EIO": ((local, hidden, intermediate * 2), (local, intermediate, hidden)),
            "EOI": ((local, intermediate * 2, hidden), (local, hidden, intermediate)),
        }

    @property
    def precision(self) -> str:
        """Return the precision fixed by this executor's resource configuration."""
        return self._resource_group.specification["precision"]

    @property
    def activation(self) -> SwiGLUParameters | SiTUGLUParameters:
        """Return the immutable activation parameters fixed for this executor."""
        return self._resource_group.specification["activation_parameters"]

    @property
    def apply_probs_before_fc2(self) -> bool:
        """Report the fixed position of routing-probability multiplication."""
        return self._resource_group.specification["apply_probs_before_fc2"]

    @classmethod
    def _resolve_specification(cls, specification: dict[str, Any]) -> dict[str, Any]:
        """Validate and normalize configuration before registering an execution owner."""
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
            ep_size=specification["ep_size"],
            dispatch_split=specification["dispatch_split"],
            combine_split=specification["combine_split"],
            auto_tokens=specification.get("auto_tokens", False),
        )
        return {
            **specification,
            "initial_capacity_factor": initial_capacity_factor,
            "capacity_growth_factor": capacity_growth_factor,
        }

    @staticmethod
    def _validate_topology(
        *,
        local_num_tokens: int,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
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

    def _infer_weight_layout(self, shape1: tuple[int, ...], shape2: tuple[int, ...]) -> str:
        """Classify the first pair using both logical shapes, including square matrices."""
        actual = (shape1, shape2)
        if actual == self._weight_shapes["EIO"]:
            return "EIO"
        if actual == self._weight_shapes["EOI"]:
            return "EOI"
        raise _InvalidExpertWeights(
            f"expert_weights must have EIO shapes {self._weight_shapes['EIO']} or "
            f"EOI shapes {self._weight_shapes['EOI']}, got {actual}.")

    @staticmethod
    def _validate_weight_storage(hidden_states, weight1, weight2, layout):
        """Validate device, dtype and layout without reading device values."""
        if (
            weight1.device != hidden_states.device
            or weight2.device != hidden_states.device
        ):
            raise _InvalidExpertWeights(
                "expert weights and hidden states must be on the same NPU."
            )
        if weight1.dtype != hidden_states.dtype or weight2.dtype != hidden_states.dtype:
            raise _InvalidExpertWeightType("expert weights and hidden states must all use BF16.")
        if layout == "EOI" and (
            not weight1.is_contiguous() or not weight2.is_contiguous()
        ):
            raise _InvalidExpertWeights("EOI expert weights must be contiguous for zero-copy execution.")

    def _validate_tensors(self, hidden_states: torch.Tensor, expert_weights: tuple) -> None:
        """Validate activation and parameter metadata before acquiring resources."""
        if hidden_states.dtype != torch.bfloat16 or not getattr(hidden_states, "is_npu", False):
            raise TypeError("MegaMoeExperts requires BF16 NPU hidden states.")
        try:
            weight1, weight2 = expert_weights
            shape1, shape2 = tuple(weight1.shape), tuple(weight2.shape)
        except (AttributeError, TypeError, ValueError) as error:
            raise _InvalidExpertWeights("expert_weights must contain two tensors") from error
        if self.precision == "mxfp8":
            try:
                self._backend.validate_weights(self, expert_weights, hidden_states)
            except (AttributeError, TypeError, ValueError, RuntimeError) as error:
                raise _InvalidExpertWeights(str(error)) from error
            if self._resource_group.resources is None:
                self._pending_layout = "EIO"
                self._resource_group.compatibility_key = (self._base_compatibility_key, "EIO")
            return
        resources = self._resource_group.resources
        if resources is None:
            layout = self._infer_weight_layout(shape1, shape2)
        else:
            layout = resources.spec.expert_weight_layout
            expected_weight1, expected_weight2 = self._weight_shapes[layout]
            if shape1 != expected_weight1:
                raise _InvalidExpertWeights(
                    f"gate_up_weight must have shape {expected_weight1}, got {shape1}.")
            if shape2 != expected_weight2:
                raise _InvalidExpertWeights(
                    f"down_weight must have shape {expected_weight2}, got {shape2}.")
        self._validate_weight_storage(hidden_states, weight1, weight2, layout)
        if resources is None:
            self._pending_layout = layout
            self._resource_group.compatibility_key = (self._base_compatibility_key, layout)

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

    def _prepare_forward_inputs(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        tokens_per_expert: torch.Tensor | None,
        expert_weights: tuple,
    ) -> tuple[torch.Tensor, bool]:
        """Defer rank-local weight errors until the binding or route collective."""
        try:
            return self._validate_forward_inputs(
                hidden_states, topk_ids, topk_weights, tokens_per_expert, expert_weights,
            ), False
        except (_InvalidExpertWeights, _InvalidExpertWeightType):
            if self.ep_size == 1:
                raise
            # Every rank must reach the existing binding/route collective before
            # rejecting a rank-local weight error; otherwise peers may hang.
            if self._resource_group.resources is None:
                self._pending_layout = None
                self._resource_group.compatibility_key = (self._base_compatibility_key, None)
            return hidden_states.reshape(-1, self.hidden_size), True

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        expert_weights: tuple[torch.Tensor, torch.Tensor],
        tokens_per_expert: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Route token states through the configured experts.

        Args:
            hidden_states: Tensor ending in ``hidden_size``.
            topk_ids: Global expert IDs shaped ``[tokens, top_k]`` for the current input.
            topk_weights: Router weights with the same shape as ``topk_ids``.
            expert_weights: Current local, unsharded gate/up and down weights owned by
                the outer module. BF16 accepts EIO ``([E,H,2I], [E,I,H])`` or EOI
                ``([E,2I,H], [E,H,I])``. Layout binds on first use; EOI must be
                contiguous. MXFP8 accepts EIO roots with packed column data and
                scales for forward, and packed row data and scales for backward.
            tokens_per_expert: Trusted exact global-expert histogram; skips histogram recomputation.

        Note:
            Returns the input shape and dtype. EOI noncontiguity raises ValueError;
            graph capture raises RuntimeError. Both transports require eager execution;
            trusted counts must match ``topk_ids`` without rebuilding the histogram.
        """
        weights = expert_weights
        hidden_flat, weight_error = self._prepare_forward_inputs(
            hidden_states, topk_ids, topk_weights, tokens_per_expert, weights,
        )
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("MegaMoe push heap growth and dynamic tokens do not support graph capture")
        resources = self._get_execution_resources(hidden_flat)
        with prepare_expert_invocation(hidden_flat, topk_ids, topk_weights, tokens_per_expert,
                                       weights, resources, weight_error=weight_error, planner=self._planner) as route:
            return self._backend.forward(
                self, hidden_states, hidden_flat, topk_ids, topk_weights, weights, resources, route,
            )

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
            expert_weight_layout=self._require_pending_layout(),
            shared=shared,
            active_specifications=active_specifications,
        )

    def _require_pending_layout(self) -> str:
        """Reject an invalid first weight pair after rank-wide binding agreement."""
        if self._pending_layout is None:
            raise ValueError("expert weights do not match a supported layout on all EP ranks")
        return self._pending_layout
