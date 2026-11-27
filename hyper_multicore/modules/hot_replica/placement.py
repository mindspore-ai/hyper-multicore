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
"""Validate custom quotas and materialize MegaMoE's virtual routes."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..base_planner import ExpertPlacement, PlacementRequest
from ..mega_moe.route_metadata import RouteMetadata
from ..mega_moe.route import _permute_topk_input
from ._planner.capacity import ExpertReplicaConfig
from ._planner.plan import ExpertExecutionPlan, physical_replica_slots


@dataclass(frozen=True)
class _LoweredPlacement:
    """Per-invocation route and weight-copy metadata owned by MegaMoE."""

    virtual_ids: torch.Tensor
    counts_by_source: torch.Tensor
    copies: torch.Tensor
    borrowed_capacity: int
    receive_capacity: int
    local_receive_capacity: int
    published_slots: int
    copies_by_rank: tuple[tuple[int, ...], ...]
    owner_rows: tuple[int, ...]
    route_metadata: RouteMetadata | None


def validate_quotas(result, request: PlacementRequest, *, slots: int) -> ExpertExecutionPlan | None:
    """Check custom quotas against the count exchange before binding resources."""
    if not isinstance(result, ExpertPlacement):
        raise TypeError("A planner must return ExpertPlacement or None")
    ranks = len(request.counts_by_source)
    copies = result.copies_by_rank
    if len(copies) != ranks:
        raise ValueError("Expert placement needs a copy row for every EP rank")
    width = len(copies[0])
    if width > slots or any(len(row) != width for row in copies):
        raise ValueError(f"Expert placement must use a uniform width within the {slots} available guest slots")
    # Require Python ints for the wire format, excluding bool and objects with custom arithmetic.
    # pylint: disable-next=unidiomatic-typecheck
    if any(type(expert) is not int for row in copies for expert in row):
        raise ValueError("Guest slots must identify integer logical experts")

    config = ExpertReplicaConfig(len(request.counts_by_source[0]), ranks, width)
    physical = tuple(physical_replica_slots(config, rank, row) for rank, row in enumerate(copies))
    plan = ExpertExecutionPlan(config, request.counts_by_source, physical, result.dispatch_counts)
    plan.validate()
    if max(plan.destination_loads) > 2**31 - 1:
        raise ValueError("Expert placement counts must fit the INT32 dispatch limit")
    home = config.home_experts
    # A zero-row guest has no producer. Use this same normalized table for
    # host agreement, device copy targets and the published slot range.
    active = tuple(row[:home] + tuple(expert if count else -1
                                     for expert, count in zip(row[home:], loads[home:]))
                   for row, loads in zip(physical, plan.destination_counts))
    if not any(expert >= 0 for row in active for expert in row[home:]):
        return None
    return ExpertExecutionPlan(config, request.counts_by_source, active, plan.dispatch_counts)


def remap_expert_ids(hidden, ids, sorted_ids, inverse):
    """Map stable logical occurrences to their physical expert IDs."""
    if (inverse is None and ids.device.type == "npu" and
            0 < ids.numel() <= 32768 and ids.shape[1] <= 8):
        _, inverse = _permute_topk_input(hidden.new_zeros((ids.shape[0], 1)), ids.int())
    if inverse is not None:
        return sorted_ids.index_select(0, inverse.flatten()).view_as(ids)
    order = torch.argsort(ids.flatten().float(), stable=True)
    remapped = torch.empty_like(sorted_ids)
    remapped.scatter_(0, order, sorted_ids)
    return remapped.view_as(ids)


def validate_lowered_ids(ids: torch.Tensor, virtual_ids: torch.Tensor, plan: ExpertExecutionPlan, rank: int) -> None:
    """Check the generated token order and all physical quotas on this source."""
    if (virtual_ids.shape != ids.shape or virtual_ids.device != ids.device
            or virtual_ids.dtype not in (torch.int32, torch.int64)):
        raise ValueError("Lowered IDs must be integers on the source Top-K shape and device")
    physical = virtual_ids.flatten().to(device="cpu", dtype=torch.int64)
    if physical.numel() and (int(physical.min()) < 0 or int(physical.max()) >= plan.config.physical_experts):
        raise ValueError("Lowered IDs exceed the physical expert range")
    logical = torch.tensor(plan.physical_to_logical, dtype=torch.int64)
    if not torch.equal(logical[physical], ids.flatten().to(device="cpu", dtype=torch.int64)):
        raise ValueError("Lowered IDs changed the logical expert at a source Top-K position")
    histogram = physical.bincount(minlength=plan.config.physical_experts)
    if tuple(histogram.tolist()) != plan.dispatch_counts[rank]:
        raise ValueError("Lowered IDs disagree with the source's physical dispatch quotas")
