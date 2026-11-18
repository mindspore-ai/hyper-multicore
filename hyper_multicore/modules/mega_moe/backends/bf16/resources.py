# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
"""BF16 plan selection over the shared MoE resource owner."""
from dataclasses import replace
from typing import Any
import torch
from ...resources import _MegaMoeExecutionResources
from .plan import build_mega_moe_plan

class BF16Resources(_MegaMoeExecutionResources):
    """Own and reuse the BF16 execution plans for one workspace."""
    def _initial_plan(self, tensor: torch.Tensor) -> Any:
        """Precision implementations supply plans while sharing the resource lifecycle."""
        dtype = torch.float32 if self.spec.partial_weight_grad_dtype == "float32" else None
        options = {} if dtype is None else {"weight_grad_dtype": dtype}
        return build_mega_moe_plan(self.spec, tensor.device, **options) if not self.spec.dynamic_tokens else None

    def plan_for_tokens(self, tokens: int, device: Any, *, weight_grad_dtype: torch.dtype | None = None) -> Any:
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
        key = bucket if weight_grad_dtype is None else (bucket, weight_grad_dtype)
        plan = self._plans.get(key)
        if plan is None or plan.spec.plan_tokens < tokens:
            schedule_tokens = min(bucket, self._token_capacity)
            options = {} if weight_grad_dtype is None else {"weight_grad_dtype": weight_grad_dtype}
            plan = build_mega_moe_plan(replace(self.spec, schedule_tokens=schedule_tokens), device, **options)
            # INT32 route bounds allow at most 25 buckets starting at 128 rows.
            # A growing partial bucket replaces its plan; backward keeps the old object.
            self._plans[key] = plan
        return plan
