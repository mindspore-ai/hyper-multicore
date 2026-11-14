# Copyright 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""CPU FP32 weight accumulation from the same rounded BF16 expert operands."""
import torch
import torch.nn.functional as F


def fp32_expert_reference(
    tokens: torch.Tensor, gate_up: torch.Tensor, down: torch.Tensor, layout: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate one unclipped, unit-probability expert and its mean-square-loss dW.

    Projections, activation, upstream gradient and activation gradient cross the
    native BF16 stores. Only the final two weight-gradient matmuls accumulate in
    FP32; no BF16 parameter .grad is used as an FP32 reference.
    """
    if layout == "EOI":
        gate_up, down = gate_up.T, down.T
    elif layout != "EIO":
        raise ValueError(f"Unknown expert weight layout: {layout}")
    tokens, gate_up, down = (value.detach().cpu().float() for value in (tokens, gate_up, down))
    projection = (tokens @ gate_up).bfloat16().float()
    gate, up = projection.chunk(2, dim=-1)
    silu = F.silu(gate)
    activation = (silu * up).bfloat16().float()
    output = (activation @ down).bfloat16()
    grad_output = (2 * output.float() / output.numel()).bfloat16().float()
    grad_activation = (grad_output @ down.T).bfloat16().float()
    sigmoid = gate.sigmoid()
    grad_gate = (grad_activation * up * sigmoid * (1 + gate * (1 - sigmoid))).bfloat16().float()
    grad_up = (grad_activation * silu).bfloat16().float()
    grad_projection = torch.cat((grad_gate, grad_up), dim=-1)
    grad_gate_up = tokens.T @ grad_projection
    grad_down = activation.T @ grad_output
    if layout == "EOI":
        grad_gate_up, grad_down = grad_gate_up.T, grad_down.T
    return output, grad_gate_up, grad_down
