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
"""Lightweight tuning options for the public MegaMoe expert module."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class MegaMoeSchedulerOptions:
    """Experimental lowering, ordering and bounded local ready selection.

    Args:
        policy: On arch35, ``None`` uses ``"two_expert_pipeline"`` to interleave
            neighboring experts. ``"builder_order"`` keeps the builder's math order.
            On arch22, use ``None`` to let lowering choose the schedule.
        queue_ready_lookahead: ``0`` runs FIFO, ``1`` checks only the head, and
            ``8`` selects the first ready task within eight slots. ``None`` uses
            FIFO on arch35 and a 256-slot Vector window on arch22 compact.
        batch_experts: On arch22 BF16, ``True`` batches GMMs by stage and disables
            per-expert pipeline ordering. ``None`` is equivalent to ``False``.
        nominal_schedule: On arch22 BF16 compact, ``True`` limits the lane count
            using nominal rows per group. ``None`` is equivalent to ``False``.
        lowering: ``"auto"`` selects compact for eligible arch22 BF16 Push plans;
            ``"dense"`` uses fixed partitions and ``"compact"`` requires eligibility.
            On arch35, both precisions use dense lowering.
    """
    policy: str | None = None
    queue_ready_lookahead: int | None = None
    batch_experts: bool | None = None
    nominal_schedule: bool | None = None
    lowering: Literal["auto", "dense", "compact"] = "auto"


@dataclass(frozen=True)
class MegaMoeTuningOptions:
    """Select experimental execution policies without changing model semantics.

    Pass this immutable value to ``MegaMoeExperts(tuning_options=...)``.
    The executor validates and normalizes its fields during construction; all
    corresponding EP ranks must use the same effective configuration. Fields,
    accepted values and default policies may change between releases.

    Args:
        dispatch_mode: ``"push"`` uses PUT dispatch; ``"pull"`` uses GET dispatch.
            Both modes use PUT for combine.
        initial_capacity_factor: Initial push receive capacity multiplier. ``None``
            selects 1.25 for push. Pull rejects explicit capacity factors.
        capacity_growth_factor: Push capacity growth multiplier. ``None`` selects
            1.25 for push; 1.0 grows to the aligned current demand. Explicit
            factors must be finite numbers at least 1, excluding bool.
        dispatch_split: Dispatch tile rows in forward and backward; a positive
            integer multiple of 128, excluding bool.
        combine_split: Combine tile rows, with the same constraints as dispatch.
        scheduler: Optional per-instance math scheduling and lowering options.
    """

    dispatch_mode: Literal["push", "pull"] = "push"
    initial_capacity_factor: float | None = None
    capacity_growth_factor: float | None = None
    dispatch_split: int = 128
    combine_split: int = 128
    scheduler: MegaMoeSchedulerOptions | None = None
