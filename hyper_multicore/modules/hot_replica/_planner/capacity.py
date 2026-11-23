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
"""Backend-independent bounds for resident hot-expert replicas."""

from __future__ import annotations

from dataclasses import dataclass


def _integer(value: int, name: str, minimum: int = 1) -> int:
    """Validate one integer configuration value without accepting booleans."""
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return value


def align_capacity(rows: int, alignment: int = 128) -> int:
    """Round a nonnegative receive-row count to the backend alignment."""
    _integer(rows, "rows", 0)
    _integer(alignment, "alignment")
    return (rows + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class ExpertReplicaConfig:
    """Logical ownership and per-rank resident guest budget.

    Args:
        num_experts: Logical expert count, divisible by ep_size.
        ep_size: Number of expert-parallel ranks.
        replica_slots_per_rank: Extra non-parameter weight slots on each rank.
    """

    num_experts: int
    ep_size: int
    replica_slots_per_rank: int

    def __post_init__(self) -> None:
        """Reject incompatible topology before allocating or communicating."""
        _integer(self.num_experts, "num_experts")
        _integer(self.ep_size, "ep_size")
        _integer(self.replica_slots_per_rank, "replica_slots_per_rank", 0)
        if self.num_experts % self.ep_size:
            raise ValueError("num_experts must be divisible by ep_size")

    @property
    def home_experts(self) -> int:
        """Return the number of optimizer-owned experts on each rank."""
        return self.num_experts // self.ep_size

    @property
    def slots_per_rank(self) -> int:
        """Return the uniform home-plus-guest execution shape."""
        return self.home_experts + self.replica_slots_per_rank

    @property
    def physical_experts(self) -> int:
        """Return the backend's physical route-ID range."""
        return self.ep_size * self.slots_per_rank

    def maximum_receive_rows(self, local_num_tokens: int, top_k: int, *, alignment: int = 128) -> int:
        """Bound every distinct-TopK route without requiring full preallocation.

        The planner retains floor(B * moved_rows / H) of each balanced transfer,
        with at most one incoming owner per rank. Integer
        truncation costs at most ep_size - 1 rows. B >= H admits exact balance.
        """
        _integer(local_num_tokens, "local_num_tokens")
        _integer(top_k, "top_k")
        if top_k > self.num_experts:
            raise ValueError("top_k must not exceed num_experts")
        home = self.home_experts
        budget = min(self.replica_slots_per_rank, home)
        average = local_num_tokens * top_k
        upper = self.ep_size * local_num_tokens * min(top_k, home)
        if budget == home:
            upper = average
        elif budget:
            mixed = ((home - budget) * upper + budget * average + home - 1) // home
            upper = min(upper, mixed + self.ep_size - 1)
        return align_capacity(upper, alignment)
