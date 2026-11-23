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
"""Isolated LongCat Flash MoE expert layer with fused permute/unpermute, zero experts and replicas."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import math
from typing import Any

import torch
import torch.distributed as dist

from hyper_multicore import shmem
from hyper_multicore._loader import load_torch_npu
from hyper_multicore.longcat._replica import (
    LongcatReplicaBinding,
    LongcatReplicaPlan,
    LongcatReplicaPool,
    ReplicaGranularity,
    plan_longcat_replicas,
)
from hyper_multicore.modules.mega_moe.backends.bf16.autograd import (
    OutputContext,
    TokenContext,
    _MegaMoeFunction,
    _launch_forward_kernel,
    _native_weight_operand,
    _prepare_forward_execution,
)
from hyper_multicore.modules.mega_moe.backends.bf16.plan import MegaMoePlan
from hyper_multicore.modules.mega_moe.backends.bf16.resources import BF16Resources
from hyper_multicore.modules.mega_moe.route import (
    _HOST_ROUTE_MAX_COUNTS,
    RouteMetadata,
    _compute_route_metadata,
    restore_topk_output,
)
from hyper_multicore.modules.mega_moe.schedule import validate_schedule_options
from hyper_multicore.modules.mega_moe.spec import (
    _COMMUNICATION_SPLIT,
    _align_capacity,
    _resolve_capacity_factors,
)
from hyper_multicore.modules.mega_moe.transport import requested_transport
from hyper_multicore.modules.mega_moe.workspace import MegaMoeWorkspace
from hyper_multicore.modules.module import MulticoreModule
from hyper_multicore.torch import ops as multicore_ops

__all__ = ["LongcatFlashExperts"]


def _resolve_ep_size(ep_size: int | None, ep_group: Any | None) -> int:
    """Validate explicit EP size against the active process group."""
    if ep_group is not None:
        if not dist.is_initialized():
            raise ValueError("Initialize torch.distributed before passing ep_group")
        actual = dist.get_world_size(ep_group)
    elif dist.is_initialized() and dist.get_world_size() > 1:
        raise ValueError("Pass ep_group explicitly for distributed LongcatFlashExperts")
    else:
        actual = 1
    if ep_size is not None:
        if isinstance(ep_size, bool) or not isinstance(ep_size, int) or ep_size <= 0:
            raise ValueError(f"ep_size must be a positive integer, got {ep_size!r}.")
        if ep_size != actual:
            raise ValueError(f"LongCat ep_size={ep_size} does not match expert group size={actual}")
    return actual


def _validate_configuration(
    *,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    zero_experts: int,
    top_k: int,
    ep_size: int,
    dispatch_mode: str,
    swiglu_precision: str,
    gate_up_padded: bool,
    partial_weight_grad_dtype: str | None,
    initial_token_capacity: int | None,
    dispatch_split: int,
    combine_split: int,
    replica_slots_per_rank: int,
    replica_granularity: str,
    replica_min_rows: int,
    replica_minimum_load_reduction: float | None,
) -> None:
    """Validate LongCat dimensions and options before allocating resources."""
    for name, value in (
        ("hidden_size", hidden_size),
        ("intermediate_size", intermediate_size),
        ("num_experts", num_experts),
        ("top_k", top_k),
        ("ep_size", ep_size),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    if isinstance(zero_experts, bool) or not isinstance(zero_experts, int) or zero_experts < 0:
        raise ValueError(f"zero_experts must be a nonnegative integer, got {zero_experts!r}.")
    if num_experts % ep_size or zero_experts % ep_size:
        raise ValueError("LongCat real and zero expert counts must be divisible by ep_size.")
    if top_k > num_experts + zero_experts:
        raise ValueError(
            f"top_k ({top_k}) cannot exceed num_experts + zero_experts ({num_experts + zero_experts})."
        )
    if dispatch_mode not in ("push", "pull"):
        raise ValueError(f"dispatch_mode must be push or pull, got {dispatch_mode!r}.")
    if swiglu_precision not in ("fused", "hf"):
        raise ValueError(f"swiglu_precision must be fused or hf, got {swiglu_precision!r}.")
    if not isinstance(gate_up_padded, bool):
        raise TypeError("gate_up_padded must be a boolean.")
    if partial_weight_grad_dtype not in (None, "native", "float32"):
        raise ValueError("partial_weight_grad_dtype must be native, float32 or None.")
    if initial_token_capacity is not None:
        if isinstance(initial_token_capacity, bool) or not isinstance(initial_token_capacity, int) or initial_token_capacity <= 0:
            raise ValueError(f"initial_token_capacity must be a positive integer, got {initial_token_capacity!r}.")
    for name, split in (("dispatch_split", dispatch_split), ("combine_split", combine_split)):
        if isinstance(split, bool) or not isinstance(split, int) or split <= 0 or split % _COMMUNICATION_SPLIT:
            raise ValueError(f"{name} must be a positive multiple of {_COMMUNICATION_SPLIT}, got {split!r}.")
    if isinstance(replica_slots_per_rank, bool) or not isinstance(replica_slots_per_rank, int) or replica_slots_per_rank < 0:
        raise ValueError(f"replica_slots_per_rank must be a nonnegative integer, got {replica_slots_per_rank!r}.")
    if replica_granularity not in ("expert", "token"):
        raise ValueError(f"replica_granularity must be expert or token, got {replica_granularity!r}.")
    if isinstance(replica_min_rows, bool) or not isinstance(replica_min_rows, int) or replica_min_rows < 0:
        raise ValueError(f"replica_min_rows must be a nonnegative integer, got {replica_min_rows!r}.")
    if replica_minimum_load_reduction is not None and (
        isinstance(replica_minimum_load_reduction, bool)
        or not isinstance(replica_minimum_load_reduction, (int, float))
        or not math.isfinite(replica_minimum_load_reduction)
        or not 0.0 <= float(replica_minimum_load_reduction) <= 1.0
    ):
        raise ValueError(
            "replica_minimum_load_reduction must be a finite number in [0.0, 1.0], "
            f"got {replica_minimum_load_reduction!r}."
        )


class _IdentityOnlyFunction(torch.autograd.Function):  # pylint: disable=abstract-method,arguments-differ
    """Fused unpermute/weighted passthrough when every EP rank routes strictly to zero experts."""

    @staticmethod
    def forward(
        ctx: Any,
        hidden_flat: torch.Tensor,
        router_weights: torch.Tensor,
        all_permuted: torch.Tensor,
        mapping: torch.Tensor,
    ) -> torch.Tensor:
        tokens = hidden_flat.shape[0]
        top_k = router_weights.shape[1]
        ctx.token_rows = tokens
        ctx.top_k = top_k
        ctx.topk_dtype = router_weights.dtype
        probs = router_weights.float().contiguous()
        ctx.save_for_backward(all_permuted, mapping, probs)
        if tokens == 0:
            return hidden_flat.new_empty((0, hidden_flat.shape[1]))
        return load_torch_npu().npu_moe_token_unpermute(all_permuted, mapping, probs=probs)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None, None, None]:
        all_permuted, mapping, probs = ctx.saved_tensors
        if ctx.token_rows == 0:
            grad_hidden = grad_output.new_empty((0, grad_output.shape[1])) if ctx.needs_input_grad[0] else None
            grad_weights = probs.new_empty((0, ctx.top_k), dtype=ctx.topk_dtype) if ctx.needs_input_grad[1] else None
            return grad_hidden, grad_weights, None, None
        grad_all = grad_output.new_empty(all_permuted.shape)
        grad_probs = torch.empty_like(probs)
        multicore_ops.mega_moe_unpermute_grad_out(all_permuted, grad_output, mapping, probs, grad_all, grad_probs)
        grad_hidden = (
            multicore_ops.moe_token_permute_grad(grad_all, mapping, ctx.token_rows, ctx.top_k)
            if ctx.needs_input_grad[0] else None
        )
        grad_weights = grad_probs.to(ctx.topk_dtype) if ctx.needs_input_grad[1] else None
        return grad_hidden, grad_weights, None, None


def _permute_all_nograd(
    hidden_flat: torch.Tensor,
    expert_ids_i32: torch.Tensor,
    total_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One-shot permute + histogram across real and identity experts."""
    tokens = hidden_flat.shape[0]
    top_k = expert_ids_i32.shape[1]
    if tokens == 0:
        return (
            hidden_flat.new_empty((0, hidden_flat.shape[1])),
            torch.empty((0,), dtype=torch.int32, device=hidden_flat.device),
            torch.zeros((total_experts,), dtype=torch.int32, device=hidden_flat.device),
        )
    rows, mapping, counts, _ = load_torch_npu().npu_moe_init_routing_v2(
        hidden_flat.contiguous(),
        expert_ids_i32,
        active_num=tokens * top_k,
        expert_num=total_experts,
        expert_tokens_num_type=1,
        expert_tokens_num_flag=True,
        active_expert_range=[0, total_experts],
        quant_mode=-1,
        row_idx_type=0,
    )
    return rows, mapping.reshape(-1).to(torch.int32).contiguous(), counts.to(torch.int32).contiguous()


def _execute_real_without_autograd(
    real_tokens: torch.Tensor,
    weight1: torch.Tensor,
    weight2: torch.Tensor,
    metadata: RouteMetadata,
    plan: MegaMoePlan,
    workspace: MegaMoeWorkspace,
    output_context: OutputContext,
    router_weights: torch.Tensor,
    *,
    workspace_claimed: bool = False,
    weight_binding: LongcatReplicaBinding | None = None,
) -> torch.Tensor:
    """Launch BF16 forward and restore weighted real+identity output directly while holding the workspace lease."""
    if workspace_claimed:
        if not workspace.in_use:
            raise RuntimeError("LongCat inference requires the caller-held workspace lease.")
    else:
        if not plan.spec.auto_tokens:
            workspace.ensure(plan.spec, real_tokens.dtype, real_tokens.device)
        workspace.claim()
    acquired = False
    try:
        if weight_binding is not None:
            weight1, weight2 = weight_binding.acquire((weight1, weight2))
            acquired = True
        execution = _prepare_forward_execution(
            workspace, plan, real_tokens, metadata.expert_capacity
        )
        _launch_forward_kernel(
            plan,
            metadata.for_kernel(plan.spec.hidden_size),
            execution.source,
            _native_weight_operand(weight1, plan),
            _native_weight_operand(weight2, plan),
            execution,
            stream=workspace.lease_stream,
        )
        real_rows = output_context.real_rows
        total_rows = output_context.total_rows
        if total_rows == 0:
            return real_tokens.new_empty((0, plan.spec.hidden_size))
        identity = output_context.identity_tokens
        mapping = output_context.unpermute_mapping
        if real_rows == 0:
            assert identity is not None
            return restore_topk_output(identity.contiguous(), mapping, router_weights)
        if identity is None or real_rows == total_rows:
            return restore_topk_output(execution.combine[:real_rows], mapping, router_weights)
        if execution.combine.shape[0] >= total_rows:
            execution.combine[real_rows:total_rows].copy_(identity)
            return restore_topk_output(execution.combine[:total_rows], mapping, router_weights)
        assembled = execution.combine.new_empty((total_rows, plan.spec.hidden_size))
        assembled[:real_rows].copy_(execution.combine[:real_rows])
        assembled[real_rows:].copy_(identity)
        return restore_topk_output(assembled, mapping, router_weights)
    finally:
        if acquired and weight_binding is not None:
            weight_binding.release()
        if not workspace_claimed:
            workspace.release(stream=workspace.lease_stream)


class LongcatFlashExperts(MulticoreModule):
    """Standalone LongCat Flash expert executor isolated from ``MegaMoeExperts``.

    Global expert IDs enumerate parameterized real experts ``[0, num_experts)``
    followed by identity (zero) experts ``[num_experts, num_experts + zero_experts)``.
    Forward executes:
      1. One-shot ``npu_moe_init_routing_v2`` over all real + identity slots,
         producing a contiguous real-expert prefix ``all_permuted[:R]`` and an
         identity-expert suffix ``all_permuted[R:]``.
      2. Optional load-balanced expert replica placement (``replica_slots_per_rank > 0``)
         with ``"expert"`` (whole-expert move, zero gradient rounding error) or
         ``"token"`` (token-level split replica) granularity strictly over real experts.
      3. ``HyperMegaMoe`` fused kernel strictly on real tokens (skipped entirely
         when every EP rank routes exclusively to identity experts).
      4. Concatenation ``[real_y, all_permuted[R:]]`` followed by one-shot fused
         ``npu_moe_token_unpermute(..., probs=router_weights.float())`` covering
         both real and identity expert FP32 weighting and gradients.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        zero_experts: int = 0,
        top_k: int,
        ep_size: int | None = None,
        ep_group: Any | None = None,
        dispatch_mode: str = "push",
        swiglu_precision: str = "fused",
        gate_up_padded: bool = False,
        partial_weight_grad_dtype: str | None = None,
        initial_token_capacity: int | None = None,
        initial_capacity_factor: float | None = None,
        capacity_growth_factor: float | None = None,
        dispatch_split: int = _COMMUNICATION_SPLIT,
        combine_split: int = _COMMUNICATION_SPLIT,
        replica_slots_per_rank: int = 0,
        replica_granularity: ReplicaGranularity = "expert",
        replica_min_rows: int = 1024,
        replica_minimum_load_reduction: float | None = None,
    ) -> None:
        """Initialize an isolated, parameterless lazy LongCat Flash expert executor."""
        resolved_ep = _resolve_ep_size(ep_size, ep_group)
        _validate_configuration(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            zero_experts=zero_experts,
            top_k=top_k,
            ep_size=resolved_ep,
            dispatch_mode=dispatch_mode,
            swiglu_precision=swiglu_precision,
            gate_up_padded=gate_up_padded,
            partial_weight_grad_dtype=partial_weight_grad_dtype,
            initial_token_capacity=initial_token_capacity,
            dispatch_split=dispatch_split,
            combine_split=combine_split,
            replica_slots_per_rank=replica_slots_per_rank,
            replica_granularity=replica_granularity,
            replica_min_rows=replica_min_rows,
            replica_minimum_load_reduction=replica_minimum_load_reduction,
        )
        init_factor, growth_factor = _resolve_capacity_factors(
            dispatch_mode, initial_capacity_factor, capacity_growth_factor
        )
        scheduler = validate_schedule_options(None)
        effective_real_top_k = min(top_k, num_experts)
        physical_experts = num_experts + resolved_ep * replica_slots_per_rank
        initial_tokens = (
            _COMMUNICATION_SPLIT
            if initial_token_capacity is None
            else max(_COMMUNICATION_SPLIT, _align_capacity(initial_token_capacity))
        )
        specification: dict[str, Any] = {
            "local_num_tokens": initial_tokens,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_experts": physical_experts,
            "top_k": effective_real_top_k,
            "partial_weight_grad_dtype": partial_weight_grad_dtype,
            "swiglu_limit": None,
            "clamp_after_activation": False,
            "swiglu_hf_bf16": swiglu_precision == "hf",
            "apply_probs_before_fc2": False,
            "ep_size": resolved_ep,
            "ep_group": ep_group,
            "dispatch_mode": dispatch_mode,
            "initial_capacity_factor": init_factor,
            "capacity_growth_factor": growth_factor,
            "dispatch_split": dispatch_split,
            "combine_split": combine_split,
            "precision": "bf16",
            "hot_replica": False,
            "planner_key": None,
            "source_token_bound": None,
            "transport_engine": requested_transport(),
            "ready_depth": scheduler.queue_ready_lookahead,
            "schedule_policy": scheduler.policy or "auto",
            "batch_experts": bool(scheduler.batch_experts),
            "nominal_schedule": bool(scheduler.nominal_schedule),
            "lowering": scheduler.lowering,
            "auto_tokens": True,
        }
        compatibility_key = tuple(
            id(value) if name == "ep_group" else value for name, value in specification.items()
        )
        super().__init__(
            resource_specification=specification,
            resource_compatibility_key=(compatibility_key, None),
            resource_scope_key=(
                "longcat_flash",
                shmem.root_members(ep_group) if dist.is_initialized() else id(ep_group),
            ),
        )
        self._base_compatibility_key = compatibility_key
        self._pending_layout: str | None = None
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.zero_experts = zero_experts
        self.total_experts = num_experts + zero_experts
        self.physical_experts = physical_experts
        self.top_k = top_k
        self.effective_real_top_k = effective_real_top_k
        self.ep_size = resolved_ep
        self.local_experts = num_experts // resolved_ep
        self.local_zero_experts = zero_experts // resolved_ep
        self._ep_group = ep_group
        self.dispatch_mode = dispatch_mode
        self.swiglu_precision = swiglu_precision
        self.gate_up_padded = gate_up_padded
        self.replica_slots_per_rank = replica_slots_per_rank
        self.replica_granularity = replica_granularity
        self.replica_min_rows = replica_min_rows
        self.replica_minimum_load_reduction = (
            None if replica_minimum_load_reduction is None else float(replica_minimum_load_reduction)
        )
        self._replica_pool = LongcatReplicaPool()
        self._invocation_counter = 0
        self._last_plan: MegaMoePlan | None = None
        self._last_replica_plan: LongcatReplicaPlan | None = None

        gate_experts = self.local_experts + (self.local_zero_experts if gate_up_padded else 0)
        self._weight_shapes = {
            "EOI": ((gate_experts, 2 * intermediate_size, hidden_size), (self.local_experts, hidden_size, intermediate_size)),
            "EIO": ((gate_experts, hidden_size, 2 * intermediate_size), (self.local_experts, intermediate_size, hidden_size)),
        }

    def close(self) -> None:
        """Release staged replica buffers and shared SHMEM execution resources."""
        self._replica_pool.close()
        super().close()

    def _root_group(self) -> Any:
        """Keep SHMEM lifecycle coordination within this module EP group."""
        return self._ep_group

    def _register_runtime_release(self, manager: Any, resources: Any) -> None:
        """Register the Heap Manager sole Runtime reference for final cleanup."""
        manager.runtime_release = resources.heap_manager.detach_runtime

    def _resource_signature(self, tensor: Any) -> Any:
        """Compare static configuration without process-local group identities."""
        configuration = tuple(
            sorted((name, value) for name, value in self._resource_group.specification.items() if name != "ep_group")
        )
        return (super()._resource_signature(tensor), configuration, self._resource_group.shared, self._pending_layout)

    def _create_execution_resources(
        self,
        tensor: Any,
        *,
        shared: bool,
        active_specifications: tuple[Mapping[str, Any], ...],
    ) -> BF16Resources:
        """Allocate BF16 plan and SHMEM workspace owned exclusively by LongcatFlashExperts."""
        if self._pending_layout is None:
            raise ValueError("expert weights do not match a supported EOI or EIO layout")
        return BF16Resources(
            self._resource_group.specification,
            tensor,
            expert_weight_layout=self._pending_layout,
            shared=shared,
            active_specifications=active_specifications,
        )

    def _validate_and_slice_weights(
        self,
        hidden_flat: torch.Tensor,
        expert_weights: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Validate caller-owned EOI/EIO weights and return the contiguous real-expert slice."""
        if not isinstance(expert_weights, tuple) or len(expert_weights) != 2:
            raise TypeError("expert_weights must be a (gate_up, down) tuple.")
        gate_up, down = expert_weights
        actual = (tuple(gate_up.shape), tuple(down.shape))
        resources = self._resource_group.resources
        if resources is None:
            if actual == self._weight_shapes["EOI"]:
                layout = "EOI"
            elif actual == self._weight_shapes["EIO"]:
                layout = "EIO"
            else:
                raise ValueError(
                    f"expert_weights must match EOI {self._weight_shapes[EOI]} or "
                    f"EIO {self._weight_shapes[EIO]}, got {actual}."
                )
        else:
            layout = resources.spec.expert_weight_layout
            if actual != self._weight_shapes[layout]:
                raise ValueError(
                    f"expert_weights must match bound {layout} shapes {self._weight_shapes[layout]}, got {actual}."
                )
        if gate_up.device != hidden_flat.device or down.device != hidden_flat.device:
            raise ValueError("expert weights and hidden states must be on the same NPU.")
        if gate_up.dtype != torch.bfloat16 or down.dtype != torch.bfloat16:
            raise TypeError("LongcatFlashExperts requires BF16 expert weights.")
        if layout == "EOI" and (not gate_up.is_contiguous() or not down.is_contiguous()):
            raise ValueError("EOI expert weights must be contiguous.")
        if resources is None:
            self._pending_layout = layout
            self._resource_group.compatibility_key = (self._base_compatibility_key, layout)
        real_gate_up = gate_up[: self.local_experts] if self.gate_up_padded and self.local_zero_experts else gate_up
        return real_gate_up, down

    def _validate_inputs(
        self,
        hidden: torch.Tensor,
        expert_ids: torch.Tensor,
        router_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Validate input shapes, devices and dtypes without host synchronization."""
        if hidden.ndim < 2 or hidden.shape[-1] != self.hidden_size:
            raise ValueError(
                f"hidden must have at least 2 dims ending in {self.hidden_size}, got {tuple(hidden.shape)}."
            )
        if hidden.dtype != torch.bfloat16 or not getattr(hidden, "is_npu", False):
            raise TypeError("LongcatFlashExperts requires BF16 NPU hidden states.")
        hidden_flat = hidden.reshape(-1, self.hidden_size)
        route_shape = (hidden_flat.shape[0], self.top_k)
        if tuple(expert_ids.shape) != route_shape or tuple(router_weights.shape) != route_shape:
            raise ValueError(
                f"expert_ids and router_weights must have shape {route_shape}, "
                f"got {tuple(expert_ids.shape)} and {tuple(router_weights.shape)}."
            )
        if expert_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"expert_ids must be int32 or int64, got {expert_ids.dtype}.")
        if not router_weights.is_floating_point():
            raise TypeError(f"router_weights must be floating point, got {router_weights.dtype}.")
        if expert_ids.device != hidden_flat.device or router_weights.device != hidden_flat.device:
            raise ValueError("hidden, expert_ids and router_weights must share one NPU device.")
        expert_ids_i32 = (
            expert_ids if expert_ids.dtype == torch.int32 and expert_ids.is_contiguous()
            else expert_ids.to(torch.int32).contiguous()
        )
        return hidden_flat, expert_ids_i32

    def _exchange_real_counts(
        self,
        real_counts: torch.Tensor,
        tokens: int,
    ) -> tuple[torch.Tensor, list[list[int]] | None, int, int, int, int, int, int, int, int, bool]:
        """All-gather per-rank token counts and real-expert histograms across the EP group."""
        header = torch.tensor([tokens], dtype=torch.int32, device=real_counts.device)
        payload = torch.cat((header, real_counts))
        if self.ep_size == 1:
            gathered = payload.unsqueeze(0)
            rank_id = 0
        else:
            gathered = torch.empty(
                (self.ep_size, payload.numel()), dtype=torch.int32, device=real_counts.device
            )
            dist.all_gather_into_tensor(gathered.reshape(-1), payload, group=self._ep_group)
            rank_id = dist.get_rank(self._ep_group)

        counts_by_source = gathered[:, 1:].contiguous()
        need_host_table = (
            self.replica_slots_per_rank > 0 and self.ep_size > 1
        ) or self.ep_size * self.num_experts <= _HOST_ROUTE_MAX_COUNTS
        if need_host_table:
            raw = gathered.tolist()
            tokens_host = [row[0] for row in raw]
            counts_host = [row[1:] for row in raw]
            source_real_totals = [sum(row) for row in counts_host]
            local_real_rows = source_real_totals[rank_id]
            total_real_across_ep = sum(source_real_totals)
            if total_real_across_ep == 0:
                return counts_by_source, counts_host, rank_id, 0, 0, 1, 0, _COMMUNICATION_SPLIT, 0, 0, False
            local = self.local_experts
            loads = [
                sum(sum(row[start : start + local]) for row in counts_host)
                for start in range(0, self.num_experts, local)
            ]
            max_received = max(loads)
            expert_capacity = max(1, loads[rank_id])
            max_tokens = max(tokens_host)
            max_per_expert = max(max(row) for row in counts_host)
            max_source_real = max(source_real_totals)
            start = rank_id * local
            all_experts_active = all(
                sum(row[expert] for row in counts_host) > 0 for expert in range(start, start + local)
            )
        else:
            counts_host = None
            local = self.local_experts
            source_totals = counts_by_source.sum(dim=1, dtype=torch.int32)
            loads = counts_by_source.reshape(self.ep_size, self.ep_size, local).sum(dim=(0, 2), dtype=torch.int32)
            local_expert_loads = counts_by_source[:, rank_id * local : (rank_id + 1) * local].sum(dim=0)
            summary = torch.stack(
                (
                    source_totals[rank_id],
                    source_totals.sum(dtype=torch.int32),
                    loads[rank_id],
                    loads.amax(),
                    gathered[:, 0].amax(),
                    counts_by_source.amax(),
                    source_totals.amax(),
                    (local_expert_loads > 0).all().to(torch.int32),
                )
            ).tolist()
            local_real_rows = int(summary[0])
            total_real_across_ep = int(summary[1])
            if total_real_across_ep == 0:
                return counts_by_source, None, rank_id, 0, 0, 1, 0, _COMMUNICATION_SPLIT, 0, 0, False
            expert_capacity = max(1, int(summary[2]))
            max_received = int(summary[3])
            max_tokens = int(summary[4])
            max_per_expert = int(summary[5])
            max_source_real = int(summary[6])
            all_experts_active = bool(summary[7])

        required_tokens = max(
            max_tokens,
            max_per_expert,
            (max_source_real + self.effective_real_top_k - 1) // self.effective_real_top_k,
        )
        plan_tokens = max(_COMMUNICATION_SPLIT, _align_capacity(required_tokens))
        return (
            counts_by_source,
            counts_host,
            rank_id,
            local_real_rows,
            total_real_across_ep,
            expert_capacity,
            max_received,
            plan_tokens,
            max_tokens,
            max_source_real,
            all_experts_active,
        )

    @staticmethod
    def _build_physical_slot_ids(
        rank_quotas: tuple[tuple[tuple[int, int], ...], ...],
        device: torch.device,
    ) -> torch.Tensor:
        """Expand per-logical-expert physical slot quotas into [R, 1] int32 IDs."""
        slots_list: list[int] = []
        counts_list: list[int] = []
        for expert_quotas in rank_quotas:
            for slot, count in expert_quotas:
                if count > 0:
                    slots_list.append(slot)
                    counts_list.append(count)
        slots_t = torch.tensor(slots_list, dtype=torch.int32, device=device)
        counts_t = torch.tensor(counts_list, dtype=torch.int64, device=device)
        return torch.repeat_interleave(slots_t, counts_t).reshape(-1, 1)

    def forward(
        self,
        hidden: torch.Tensor,
        expert_ids: torch.Tensor,
        router_weights: torch.Tensor,
        *,
        expert_weights: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Execute LongCat real and identity experts with optional replicas and fused permute/unpermute."""
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("LongcatFlashExperts dynamic routing does not support graph capture.")
        hidden_flat, expert_ids_i32 = self._validate_inputs(hidden, expert_ids, router_weights)
        real_gate_up, down = self._validate_and_slice_weights(hidden_flat, expert_weights)

        grad_enabled = torch.is_grad_enabled()
        with torch.no_grad():
            all_permuted, mapping, all_counts = _permute_all_nograd(
                hidden_flat, expert_ids_i32, self.total_experts
            )

        tokens = hidden_flat.shape[0]
        total_slots = all_permuted.shape[0]
        real_counts = all_counts[: self.num_experts]

        with torch.inference_mode(False), torch.no_grad():
            (
                counts_by_source,
                counts_host,
                rank_id,
                local_real_rows,
                total_real_across_ep,
                expert_capacity,
                max_received,
                plan_tokens,
                max_tokens,
                max_source_real,
                all_experts_active,
            ) = self._exchange_real_counts(real_counts, tokens)

        if total_real_across_ep == 0:
            self._last_replica_plan = None
            if grad_enabled:
                out = _IdentityOnlyFunction.apply(hidden_flat, router_weights, all_permuted, mapping)
                return out.reshape_as(hidden)
            return restore_topk_output(all_permuted, mapping, router_weights).reshape_as(hidden)

        replica_plan: LongcatReplicaPlan | None = None
        if self.replica_slots_per_rank > 0 and self.ep_size > 1 and counts_host is not None:
            reduction = (
                self.replica_minimum_load_reduction
                if self.replica_minimum_load_reduction is not None
                else (0.25 if grad_enabled else 0.4)
            )
            row_credit = (self.intermediate_size // 2) if grad_enabled else 0
            replica_plan = plan_longcat_replicas(
                counts_host,
                self.replica_slots_per_rank,
                granularity=self.replica_granularity,
                minimum_replica_rows=self.replica_min_rows,
                minimum_load_reduction=reduction,
                transfer_row_credit=row_credit,
            )
        self._last_replica_plan = replica_plan
        use_replicas = replica_plan is not None and not replica_plan.home_only

        resources = self._get_execution_resources(hidden_flat)
        pull = resources.spec.dispatch_mode == "pull"
        claimed = False
        weight_binding: LongcatReplicaBinding | None = None

        if local_real_rows == total_slots:
            real_x = all_permuted
            zero_x = None
        else:
            real_x = all_permuted[:local_real_rows]
            zero_x = all_permuted[local_real_rows:]

        try:
            with torch.inference_mode(False), torch.no_grad():
                if use_replicas:
                    assert replica_plan is not None
                    phys_counts_by_source = torch.tensor(
                        replica_plan.dispatch_counts, dtype=torch.int32, device=hidden_flat.device
                    )
                    phys_real_counts = phys_counts_by_source[rank_id].contiguous()
                    loads = list(replica_plan.physical_rank_loads)
                    expert_capacity = max(1, loads[rank_id])
                    max_received = max(loads)
                    max_per_expert = max(max(row) for row in replica_plan.dispatch_counts)
                    required_tokens = max(
                        max_tokens,
                        max_per_expert,
                        (max_source_real + self.effective_real_top_k - 1) // self.effective_real_top_k,
                    )
                    plan_tokens = max(_COMMUNICATION_SPLIT, _align_capacity(required_tokens))
                    phys_per_rank = self.local_experts + self.replica_slots_per_rank
                    start = rank_id * phys_per_rank
                    all_experts_active = all(
                        sum(row[p] for row in replica_plan.dispatch_counts) > 0
                        for p in range(start, start + phys_per_rank)
                    )
                    route_experts = self.physical_experts
                    route_spec = resources.spec
                    active_counts = phys_real_counts
                    active_counts_by_source = phys_counts_by_source
                    self._invocation_counter += 1
                    weight_binding = LongcatReplicaBinding(
                        self._replica_pool,
                        replica_plan,
                        rank_id,
                        self._ep_group,
                        self._invocation_counter,
                    )
                else:
                    route_experts = self.num_experts if self.replica_slots_per_rank > 0 else None
                    route_spec = (
                        replace(resources.spec, num_experts=self.num_experts)
                        if self.replica_slots_per_rank > 0
                        else resources.spec
                    )
                    active_counts = real_counts
                    active_counts_by_source = counts_by_source

                if not pull:
                    resources.workspace.wait_for_reuse()
                resources.prepare_auto_workspace(plan_tokens, max_received, hidden_flat)
                claimed = pull
                plan = resources.plan_for_tokens(
                    plan_tokens, hidden_flat.device, route_experts=route_experts
                )
                metadata = _compute_route_metadata(
                    active_counts, active_counts_by_source, None, route_spec, expert_capacity
                )
                metadata = metadata.with_active_experts(all_experts_active, tokens, plan_tokens)
            self._last_plan = plan

            if use_replicas and local_real_rows > 0:
                assert replica_plan is not None
                with torch.no_grad():
                    phys_slot_ids = self._build_physical_slot_ids(
                        replica_plan.source_quotas[rank_id], hidden_flat.device
                    )
                    phys_real_x, real_to_phys_map = load_torch_npu().npu_moe_token_permute(
                        real_x.contiguous(), phys_slot_ids
                    )
                    real_to_phys_map = real_to_phys_map.reshape(-1).to(torch.int32).contiguous()
                    if local_real_rows == total_slots:
                        active_mapping = real_to_phys_map.index_select(0, mapping)
                    else:
                        zero_ids = torch.arange(
                            local_real_rows, total_slots, dtype=torch.int32, device=hidden_flat.device
                        )
                        full_remap = torch.cat((real_to_phys_map, zero_ids), dim=0)
                        active_mapping = full_remap.index_select(0, mapping)
            else:
                phys_real_x = real_x
                active_mapping = mapping

            token_context = TokenContext(
                routed_tokens=phys_real_x,
                unpermute_mapping=active_mapping,
                token_rows=tokens,
                top_k=self.top_k,
                real_rows=local_real_rows,
                total_rows=total_slots,
            )
            output_context = OutputContext(
                unpermute_mapping=active_mapping,
                real_rows=local_real_rows,
                total_rows=total_slots,
                identity_tokens=zero_x,
            )

            if grad_enabled:
                output = _MegaMoeFunction.apply(
                    hidden_flat,
                    _native_weight_operand(real_gate_up, plan),
                    _native_weight_operand(down, plan),
                    metadata,
                    plan,
                    resources.workspace,
                    None,
                    router_weights,
                    claimed,
                    weight_binding,
                    token_context,
                    output_context,
                )
            else:
                output = _execute_real_without_autograd(
                    phys_real_x,
                    real_gate_up,
                    down,
                    metadata,
                    plan,
                    resources.workspace,
                    output_context,
                    router_weights,
                    workspace_claimed=claimed,
                    weight_binding=weight_binding,
                )
        finally:
            if claimed:
                resources.workspace.release()

        return output.reshape_as(hidden)
