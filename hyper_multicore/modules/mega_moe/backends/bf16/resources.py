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
"""BF16 plan selection over the shared MoE resource owner."""
from dataclasses import replace
from typing import Any
import torch
from hyper_multicore.scheduler.config import mega_moe_event_capacity
from ...resources import _MegaMoeExecutionResources
from .plan import build_mega_moe_plan

class BF16Resources(_MegaMoeExecutionResources):
    """Own and reuse the BF16 execution plans for one workspace."""
    def _initial_plan(self, tensor: torch.Tensor) -> Any:
        """Precision implementations supply plans while sharing the resource lifecycle."""
        dtype = torch.float32 if self.spec.partial_weight_grad_dtype == "float32" else None
        options = {} if dtype is None else {"weight_grad_dtype": dtype}
        return build_mega_moe_plan(self.spec, tensor.device, **options) if not self.spec.dynamic_tokens else None

    def plan_for_tokens(
        self, tokens: int, device: Any, *, weight_grad_dtype: torch.dtype | None = None,
        route_experts: int | None = None, batch_experts: bool = False, nominal_schedule: bool = False,
    ) -> Any:
        """Reuse one covering plan per geometric bucket within committed capacity.

        Args:
            tokens: EP group maximum input rows, already aligned to 128.
            device: Execution device used for the parameters and workspace.
            weight_grad_dtype: Optional FP32 native output for partial expert weight
                gradients. Autograd and the caller control the final gradient dtype.

        Returns:
            Immutable plan covering the requested rows without enlarging storage.
        """
        if weight_grad_dtype is None and self.spec.partial_weight_grad_dtype == "float32":
            weight_grad_dtype = torch.float32
        if tokens > self._token_capacity:
            raise RuntimeError(f"MegaMoe schedule exceeds committed token capacity: "
                               f"requested={tokens}, capacity={self._token_capacity}")
        bucket = 1 << (max(128, tokens) - 1).bit_length()
        spec = self.spec
        key = bucket if weight_grad_dtype is None else (bucket, weight_grad_dtype)
        if route_experts is not None:
            if not 0 < route_experts <= spec.num_experts or route_experts % spec.ep_size:
                raise ValueError("HotReplica route geometry exceeds the shared workspace")
            spec = replace(
                spec, num_experts=route_experts,
                batch_experts=spec.batch_experts or batch_experts,
                nominal_schedule=spec.nominal_schedule or nominal_schedule,
            )
            key = (key, route_experts, spec.batch_experts, spec.nominal_schedule)
        plan = self._plans.get(key)
        if plan is None or plan.spec.plan_tokens < tokens:
            schedule_tokens = min(bucket, self._token_capacity)
            options = {} if weight_grad_dtype is None else {"weight_grad_dtype": weight_grad_dtype}
            plan = build_mega_moe_plan(replace(spec, schedule_tokens=schedule_tokens), device, **options)
            # INT32 route bounds allow at most 25 buckets starting at 128 rows.
            # A growing partial bucket replaces its plan; backward keeps the old object.
            if spec.num_experts != self.spec.num_experts:
                counters = 4 * mega_moe_event_capacity(spec.num_experts, spec.ep_size)
                plan = replace(plan, event_counter_bytes=counters)
            self._plans[key] = plan
        return plan
