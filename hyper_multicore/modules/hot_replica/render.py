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
"""Render immutable expert quotas into executor slot geometry."""

import numpy as np

from ._planner.plan import ExpertExecutionPlan


def render_plan(
    plan: ExpertExecutionPlan, rank: int, borrowed: int,
) -> tuple[np.ndarray, np.ndarray, tuple[tuple[int, ...], ...]]:
    """Trim empty tails and retain the source's stable logical expert order."""
    config = plan.config
    home, ranks, full = config.home_experts, config.ep_size, config.slots_per_rank
    width = home + borrowed
    owners = tuple(row[home:width] for row in plan.slot_to_logical)
    table = np.frombuffer(plan.dispatch_counts_bytes, dtype=np.int32).reshape(ranks, ranks, full)
    table = table[:, :, :width].copy().reshape(ranks, -1)
    logical = np.asarray(plan.slot_to_logical, dtype=np.int32)[:, :width].reshape(-1)
    slots = np.argsort(logical, kind="stable").astype(np.int32, copy=False)
    return np.repeat(slots, table[rank, slots]), table, owners
