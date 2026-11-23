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
"""Row and kernel element descriptors for one MegaMoE route."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import torch


@dataclass(frozen=True)
class RouteMetadata:
    """Exact offsets and counts consumed by the low-level native ABI."""

    dispatch_src_off: torch.Tensor
    dispatch_target_off: torch.Tensor
    dispatch_size: torch.Tensor
    combine_src_off: torch.Tensor
    combine_target_off: torch.Tensor
    combine_size: torch.Tensor
    group_list: torch.Tensor
    expert_capacity: int
    local_num_tokens: int | None = None
    plan_tokens: int | None = None
    all_experts_active: bool = False
    _elements: tuple[int, RouteMetadata] | None = field(default=None, repr=False, compare=False)

    def for_kernel(self, hidden_size: int) -> RouteMetadata:
        """Convert row descriptors once; keep the public row-unit contract unchanged."""
        if hidden_size == 1:
            return self
        if self._elements is not None and self._elements[0] == hidden_size:
            return self._elements[1]
        return replace(
            self,
            dispatch_src_off=self.dispatch_src_off * hidden_size,
            dispatch_target_off=self.dispatch_target_off * hidden_size,
            dispatch_size=self.dispatch_size * hidden_size,
            combine_src_off=self.combine_src_off * hidden_size,
            combine_target_off=self.combine_target_off * hidden_size,
            combine_size=self.combine_size * hidden_size,
            _elements=None,
        )

    def record_stream(self, stream: Any) -> None:
        """Retain row descriptors and cached element descriptors on their consumer stream."""
        views = (self,) if self._elements is None else (self, self._elements[1])
        for view in views:
            for value in vars(view).values():
                if isinstance(value, torch.Tensor):
                    value.record_stream(stream)

    def with_active_experts(self, active: bool, tokens: int, plan_tokens: int) -> RouteMetadata:
        """Bind source rows and activity to both immutable descriptor views."""
        native = self._elements
        return replace(
            self, all_experts_active=active, local_num_tokens=tokens, plan_tokens=plan_tokens,
            _elements=(native[0], replace(native[1], all_experts_active=active)) if native else None,
        )
