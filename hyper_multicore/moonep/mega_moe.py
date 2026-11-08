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
"""Opt-in MoonEP placement with the original HP MegaMoE execution schedule."""
from __future__ import annotations

import math
import inspect
from functools import lru_cache
from typing import Any
import weakref

import torch

from hyper_multicore.modules.mega_moe.function import (
    execute_mega_moe_with_permutation, _execute_mega_moe_without_autograd,
)
from hyper_multicore.modules.mega_moe.heap_manager import _workspace_layout
from hyper_multicore.modules.mega_moe.module import MegaMoeExperts
from hyper_multicore.modules.mega_moe.route import prepare_topk_route, restore_topk_output
from hyper_multicore.modules.mega_moe.spec import _align_capacity
from .placement import build_placement, placement_is_profitable

from .shmem_transfer import execute_shared, get_expert_pool


_VIRTUAL_GROUPS: weakref.WeakKeyDictionary[Any, Any] = weakref.WeakKeyDictionary()


@lru_cache(maxsize=8)
def _route_statistics(counts: tuple[tuple[int, ...], ...]) -> tuple[int, int, int]:
    """Reuse integer summaries only; no tensors, placements or graph state are held."""
    totals = tuple(map(sum, zip(*counts)))
    return (max(totals, default=0), sum(value > 0 for value in totals),
            max((max(row, default=0) for row in counts), default=0))


class MoonEPExecution:
    """Share placement with ordinary EP while keeping MegaMoE's native schedule.

    Route selection precedes symmetric growth and source publication. Each
    invocation owns its placement; weights remain parameters of the original
    module and the shared bank is leased only while an executor uses its views.
    """

    def __init__(self, configuration: dict[str, Any], *, grad_reduce_dtype: str = "fp32", planner=None, transfer_backend=None) -> None:
        """Retain the original expert configuration and defer optional resources."""
        if grad_reduce_dtype not in ("fp32", "bf16"):
            raise ValueError("moonep_grad_reduce_dtype must be fp32 or bf16")
        self.planner = planner
        self.transfer_backend = transfer_backend
        try:
            self._planner_accepts_counts = planner is not None and "source_counts" in inspect.signature(planner).parameters
        except (TypeError, ValueError):
            self._planner_accepts_counts = False
        self.grad_reduce_dtype = torch.float32 if grad_reduce_dtype == "fp32" else torch.bfloat16
        self._virtual_executor = None
        self._virtual_executors = {}
        self._virtual_configuration = {
            **configuration, "num_experts": 2 * configuration["num_experts"], "create_parameters": False,
        }
        self._transfer_pool = None
        self.placement_statistics = {
            "native_balanced_calls": 0, "native_cost_skip_calls": 0,
            "native_no_copy_calls": 0, "moonep_calls": 0,
            "batch_experts_calls": 0, "batch_summary_unavailable_calls": 0,
        }

    def _get_virtual_executor(
        self, experts: MegaMoeExperts, hidden: torch.Tensor, plan_tokens: int, maximum_received: int,
    ) -> MegaMoeExperts:
        """Reserve the cold virtual workspace and expert bank in one transaction."""
        capacity = self._official_borrowed_capacity
        if capacity in self._virtual_executors:
            self._virtual_executor = self._virtual_executors[capacity]
            return self._virtual_executor
        configuration = {**self._virtual_configuration,
                         "num_experts": experts.ep_size * (experts.local_experts + capacity)}
        with torch.inference_mode(False), torch.no_grad():
            executor = MegaMoeExperts(**configuration)
        groups = _VIRTUAL_GROUPS.setdefault(experts._resource_group, {})
        target = groups.get(capacity)
        if target is None or target.closed or target.closing:
            target = executor._resource_group
            target.shared = experts._resource_group.shared
            groups[capacity] = target
        else:
            if target.compatibility_key != executor._resource_group.compatibility_key:
                raise ValueError("Shared MoonEP virtual plans require matching HP configuration")
            executor._move_to_resource_group(target)
        reservations = ()
        if target.resources is None:
            specification = target.specification
            layout = _workspace_layout(
                specification, hidden.dtype, capacity=_align_capacity(max(1, maximum_received)),
                token_capacity=plan_tokens if specification.get("auto_tokens", False) else None,
            )
            reservations = ((specification, layout),)
        with torch.inference_mode(False), torch.no_grad():
            self._transfer_pool = get_expert_pool(
                hidden, experts._resource_group.resources, experts.local_experts, self, reservations,
            )
        self._virtual_executors[capacity] = executor
        self._virtual_executor = executor
        return executor

    @staticmethod
    def _persistent_engine(flat: torch.Tensor, executor: MegaMoeExperts) -> Any:
        """Keep persistent state writable after an inference-only first call."""
        if executor._resource_group.resources is not None:
            return executor._get_execution_resources(flat)
        with torch.inference_mode(False), torch.no_grad():
            return executor._get_execution_resources(flat)

    def _prepare_native_route(
        self, flat: torch.Tensor, ids: torch.Tensor, probs: torch.Tensor,
        counts: torch.Tensor | None, weights: tuple[torch.Tensor, torch.Tensor],
        engine: Any, select_route: Any = None,
        validated_summary: tuple[int, int, int] | None = None,
        planned_counts: torch.Tensor | None = None,
    ) -> tuple[Any, Any, bool]:
        """Allocate only the route chosen after the existing count agreement."""
        pull = engine.spec.dispatch_mode == "pull"
        claimed = False

        def prepare_workspace(tokens: int, received: int) -> None:
            """Commit the chosen route before a pull view can reference SHMEM."""
            nonlocal claimed
            if engine.spec.auto_tokens:
                growth = (tokens > engine._token_capacity
                          or received > engine.workspace.capacity_floor and not pull
                          or engine.workspace.forward_event_counters is None)
                if growth:
                    with torch.inference_mode(False), torch.no_grad():
                        engine.prepare_auto_workspace(tokens, received, flat)
                else:
                    engine.prepare_auto_workspace(tokens, received, flat)
            elif pull:
                with torch.inference_mode(False), torch.no_grad():
                    engine.workspace.ensure(engine.spec, flat.dtype, flat.device)
                engine.workspace.claim()
            claimed = pull

        options = {"autograd_mask": (
            int(torch.is_grad_enabled()) | (int(flat.requires_grad) << 1)
            | (int(weights[0].requires_grad) << 2) | (int(weights[1].requires_grad) << 3)
            | (int(probs.requires_grad) << 4)
        )}
        if planned_counts is not None:
            options["planned_counts"] = planned_counts
        if validated_summary is not None:
            options["validated_summary"] = validated_summary
        if engine.spec.auto_tokens or select_route is not None:
            options["prepare_workspace"] = prepare_workspace
        elif pull:
            prepare_workspace(engine.spec.plan_tokens, engine.spec.receive_capacity)
        try:
            with torch.no_grad():
                route = prepare_topk_route(
                    flat, ids, probs, engine.spec, counts, workspace=engine.workspace,
                    select_placement=select_route, discard_rejected=select_route is not None,
                    with_source_permutation=True, **options,
                )
        except BaseException:
            if claimed:
                engine.workspace.release()
            raise
        return engine, route, claimed

    def _execute_native_route(
        self, flat: torch.Tensor, ids: torch.Tensor, probs: torch.Tensor,
        weights: tuple[torch.Tensor, torch.Tensor], engine: Any, route: Any,
        pull: bool, output_shape: torch.Size, *, batch_experts: bool = False, nominal_schedule: bool = False,
    ) -> torch.Tensor:
        """Use the original plan and kernels after selecting the ordinary route."""
        if not pull and not engine.spec.auto_tokens:
            with torch.inference_mode(False), torch.no_grad():
                engine.ensure_capacity(route.maximum_received_slots)
                engine.workspace.ensure(engine.spec, flat.dtype, flat.device)
        if not torch.is_grad_enabled():
            return _execute_mega_moe_without_autograd(
                probs, *weights, route,
                engine.plan_for_tokens(route.metadata.plan_tokens, flat.device,
                                       batch_experts=batch_experts, nominal_schedule=nominal_schedule),
                engine.workspace, workspace_claimed=pull).reshape(output_shape)
        output = execute_mega_moe_with_permutation(
            flat, ids, *weights, route, engine.plan_for_tokens(route.metadata.plan_tokens, flat.device,
                                       batch_experts=batch_experts, nominal_schedule=nominal_schedule),
            engine.workspace, topk_weights=probs if pull or engine.spec.apply_probs_before_fc2 else None,
            workspace_claimed=pull,
        )
        if not pull and not engine.spec.apply_probs_before_fc2:
            output = restore_topk_output(output, route.unpermute_mapping, probs)
        return output.reshape(output_shape)

    def _placement_is_profitable(
        self, engine: Any, maximum: int, target: int, weights: tuple[torch.Tensor, torch.Tensor], training: bool,
    ) -> bool:
        return placement_is_profitable(
            maximum, target, engine.spec.hidden_size, engine.spec.intermediate_size, engine.spec.num_cube_cores,
            sum(weight.numel() * weight.element_size() for weight in weights),
            # Count forward and input-gradient GEMMs for training. This conservative
            # estimate also covers frozen experts and stays stable under checkpointing.
            projection_flops=12 if training else 6,
            ep_size=engine.spec.ep_size, local_experts=engine.spec.local_experts,
            gradient_bytes=(sum(weight.numel() * self.grad_reduce_dtype.itemsize
                                for weight in weights if weight.requires_grad) if training else 0),
        )

    def forward(
        self, experts: MegaMoeExperts, hidden_states: torch.Tensor, flat: torch.Tensor,
        topk_ids: torch.Tensor, topk_weights: torch.Tensor, tokens_per_expert: torch.Tensor | None,
        weights: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Borrow only when rank imbalance is estimated to repay its transfer cost."""
        placement = None
        selected_tokens = 0
        batch_experts = False
        nominal_schedule = False
        engine = self._persistent_engine(flat, experts)

        def select_route(gathered: torch.Tensor, tokens: int, maximum: int, total: int | None, source_rows: tuple[int, ...], source_counts: tuple[tuple[int, ...], ...] | None, *, source_inverse=None) -> bool:
            """Choose placement using the existing validated count table."""
            nonlocal placement, selected_tokens, batch_experts, nominal_schedule
            selected_tokens = tokens
            target = (total + experts.ep_size - 1) // experts.ep_size if total is not None else tokens * experts.top_k
            if maximum <= target:
                self.placement_statistics["native_balanced_calls"] += 1
            elif not self._placement_is_profitable(engine, maximum, target, weights, experts.training):
                self.placement_statistics["native_cost_skip_calls"] += 1
            else:
                choose = build_placement if self.planner is None else self.planner
                planner_options = {"resources": engine, "source_rows": source_rows,
                                   "source_inverse": source_inverse}
                if self._planner_accepts_counts:
                    planner_options["source_counts"] = source_counts
                placement = choose(flat, topk_ids, gathered, max(1, target), engine.spec.rank_id,
                                   **planner_options)
                if placement is not None:
                    if source_counts is None:
                        # The existing large-table summary does not publish these
                        # statistics. Preserve pipelining rather than add a readback.
                        self.placement_statistics["batch_summary_unavailable_calls"] += 1
                    else:
                        largest_expert, active_experts, _ = _route_statistics(source_counts)
                        batch_experts = (largest_expert <= 4 * 128
                                         or active_experts <= experts.ep_size)
                        self.placement_statistics["batch_experts_calls"] += int(batch_experts)
                    return False
                self.placement_statistics["native_no_copy_calls"] += 1
            if source_counts is not None:
                largest_expert, active_experts, maximum_source = _route_statistics(source_counts)
                batch_experts = largest_expert <= 4 * 128 and active_experts > experts.ep_size
                slots = tokens * experts.top_k
                communication_lanes = max(1, math.ceil(slots / (engine.spec.num_experts * 128)))
                activation_lanes = max(1, math.ceil(slots / (engine.spec.local_experts * 128)))
                # Reuse the ordinary schedule for nonmoving large experts.
                # Nominal partitions also benefit sparse bounded expert queues;
                # forcing them for large experts can reduce device parallelism.
                nominal_schedule = (largest_expert <= 4 * 128
                                    and maximum_source <= 2 * 128 * communication_lanes
                                    and largest_expert <= 2 * 128 * activation_lanes)
                self.placement_statistics["batch_experts_calls"] += int(batch_experts)
            return True

        engine, route, claimed = self._prepare_native_route(
            flat, topk_ids, topk_weights, tokens_per_expert, weights, engine, select_route=select_route,
        )
        try:
            if placement is None:
                return self._execute_native_route(
                    flat, topk_ids, topk_weights, weights, engine, route,
                    engine.spec.dispatch_mode == "pull", hidden_states.shape,
                    batch_experts=batch_experts, nominal_schedule=nominal_schedule,
                )
        finally:
            if claimed:
                engine.workspace.release()
        self.placement_statistics["moonep_calls"] += 1
        self._official_borrowed_capacity = placement.borrowed_capacity
        counts = placement.counts_by_source[engine.spec.rank_id]
        maximum_received = (placement.receive_capacity if placement.borrowed_capacity not in self._virtual_executors else 0)
        return execute_shared(
            self, experts, hidden_states, placement.virtual_ids, topk_weights, weights,
            counts, placement.copies, selected_tokens, maximum_received,
            receive_bound=placement.receive_capacity, planned_counts=placement.counts_by_source,
            batch_experts=batch_experts,
        )

    def runtime_statistics(self) -> dict[str, Any]:
        """Return an independent host snapshot without synchronizing the device."""
        pool = self._transfer_pool
        return {
            "enabled": True,
            "placement": dict(self.placement_statistics),
            "transfer": None if pool is None else dict(pool.statistics),
            "bank_in_use": False if pool is None else pool.in_use,
        }

    def close(self) -> None:
        """Release HP plans and close the bank after its last invocation."""
        pool = self._transfer_pool
        if pool is not None:
            pool.owners.discard(self)
            if not pool.owners and not pool.graphs:
                pool.manager.close_workspace(pool)
        for executor in self._virtual_executors.values():
            executor.close()
        self._virtual_executors.clear()
