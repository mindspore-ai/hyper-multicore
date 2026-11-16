# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
"""BF16 plan selection over the shared MoE resource owner."""
from dataclasses import replace
from typing import Any
import torch
from ...resources import _MegaMoeExecutionResources
from .plan import build_mega_moe_plan

class BF16Resources(_MegaMoeExecutionResources):
    def _initial_plan(self, tensor: torch.Tensor) -> Any:
        """Precision implementations supply plans while sharing the resource lifecycle."""
        return build_mega_moe_plan(self.spec, tensor.device) if not self.spec.dynamic_tokens else None

    def plan_for_tokens(self, tokens: int, device: Any) -> Any:
        """Reuse one covering plan per geometric bucket within committed capacity.

        Args:
            tokens: EP group maximum input rows, already aligned to 128.
            device: Execution device used for the parameters and workspace.

        Returns:
            Immutable plan covering the requested rows without enlarging storage.
        """
        if weight_grad_dtype is None and self.spec.partial_weight_grad_dtype == "float32":
            weight_grad_dtype = torch.float32
        if tokens > self._token_capacity:
            raise RuntimeError(f"MegaMoe schedule exceeds committed token capacity: "
                               f"requested={tokens}, capacity={self._token_capacity}")
        bucket = 1 << (max(128, tokens) - 1).bit_length()
        plan = self._plans.get(bucket)
        if plan is None or plan.spec.plan_tokens < tokens:
            schedule_tokens = min(bucket, self._token_capacity)
            plan = build_mega_moe_plan(replace(
                self.spec, schedule_tokens=schedule_tokens,
                dispatch_split=min(self.spec.dispatch_split, schedule_tokens),
                combine_split=min(self.spec.combine_split, schedule_tokens)), device)
            # INT32 route bounds allow at most 25 buckets starting at 128 rows.
            # A growing partial bucket replaces its plan; backward keeps the old object.
            self._plans[bucket] = plan
        return plan
