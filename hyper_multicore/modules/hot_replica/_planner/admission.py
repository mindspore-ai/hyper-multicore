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
"""Route-based admission of optional expert replicas."""

import math
from typing import Optional

_BASIS_POINTS = 10000
_DEFAULT_MINIMUM_LOAD_REDUCTION = 0.4


def reduction_basis_points(value: float) -> int:
    """Round an admission threshold upwards for identical integer CPU/device decisions.

    Args:
        value: Required reduction of the maximum received rows, between zero and one.

    Returns:
        Integer basis points; zero preserves the original placement policy.
    """
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not 0 <= value <= 1 or not math.isfinite(value)):
        raise ValueError("replica_minimum_load_reduction must be finite and between zero and one")
    return math.ceil(value * _BASIS_POINTS)


def resolve_replica_admission(value: Optional[float]) -> tuple[float, int]:
    """Use a fractional route-load threshold without disabling a workload's budget.

    Args:
        value: Explicit threshold, or None for the default relative reduction.

    Returns:
        Required fractional reduction and a zero absolute-row threshold.

    Note:
        This is a placement heuristic, not a latency guarantee. Capacity-required
        replicas remain enabled regardless of the optional-copy threshold.
    """
    resolved = _DEFAULT_MINIMUM_LOAD_REDUCTION if value is None else value
    reduction_basis_points(resolved)
    return resolved, 0


def optional_peak_gain_is_sufficient(
    peak: int, candidate_peak: int, admission: int, minimum_saved_rows: int,
) -> bool:
    """Apply the same integer gain threshold to bounds and completed placements."""
    saved = peak - candidate_peak
    return saved >= minimum_saved_rows and saved * _BASIS_POINTS >= peak * admission
