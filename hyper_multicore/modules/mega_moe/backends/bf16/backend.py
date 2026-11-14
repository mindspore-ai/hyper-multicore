# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
"""BF16 backend: use standard tensor weights and the BF16 plan/ABI."""
from .resources import BF16Resources as Resources
from .execution import forward


def validate_weights(module, expert_weights, hidden):
    weight1, weight2 = expert_weights
    if weight1 is None or weight2 is None:
        raise ValueError("Parameterless MegaMoe requires explicit expert_weights")
    if module.expert_weight_layout == "EOI":
        expected_weight1 = (module.local_experts, module.intermediate_size * 2, module.hidden_size)
        expected_weight2 = (module.local_experts, module.hidden_size, module.intermediate_size)
    else:
        expected_weight1 = (module.local_experts, module.hidden_size, module.intermediate_size * 2)
        expected_weight2 = (module.local_experts, module.intermediate_size, module.hidden_size)
    if tuple(weight1.shape) != expected_weight1:
        raise ValueError(
            f"gate_up_weight must have shape {expected_weight1}, got {tuple(weight1.shape)}."
        )
    if tuple(weight2.shape) != expected_weight2:
        raise ValueError(
            f"down_weight must have shape {expected_weight2}, got {tuple(weight2.shape)}."
        )
    if (
        weight1.device != hidden.device
        or weight2.device != hidden.device
    ):
        raise ValueError(
            "expert weights and hidden states must be on the same NPU."
        )
    if weight1.dtype != hidden.dtype or weight2.dtype != hidden.dtype:
        raise TypeError("expert weights and hidden states must all use BF16.")
    if module.expert_weight_layout == "EOI" and (
        not weight1.is_contiguous() or not weight2.is_contiguous()
    ):
        raise ValueError("EOI expert weights must be contiguous for zero-copy execution.")
