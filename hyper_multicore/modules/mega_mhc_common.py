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
"""Shared runtime contracts for HyperMegaMhc forward and backward."""

from __future__ import annotations

import torch

from hyper_multicore._core_spec import CoreSpec


MHC_STREAMS = 4
MHC_MAPPINGS = MHC_STREAMS * MHC_STREAMS + 2 * MHC_STREAMS
MHC_SINKHORN_ITERS = 20
# The forward RMSNorm normal path has a fixed 12,288-element UB allocation.
MHC_FORWARD_MAX_HIDDEN_SIZE = 12288
MHC_GRAD_MAX_HIDDEN_SIZE = 5760
MHC_GRAD_MIN_TOKENS = 40

MegaMhcOutputs = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def validate_mega_mhc_inputs(
    previous_output: torch.Tensor,
    residual: torch.Tensor,
    previous_pre_mix: torch.Tensor,
    previous_post_mix: torch.Tensor,
    previous_residual_mix: torch.Tensor,
    phi: torch.Tensor,
    alpha: torch.Tensor,
    bias: torch.Tensor,
    norm_weight: torch.Tensor,
    num_iters: int,
) -> None:
    """Validate the fixed-four-stream mHC input shapes and iteration count.

    Args:
        previous_output: Previous block output.
        residual: Previous residual streams.
        previous_pre_mix: Previous pre-mix coefficients.
        previous_post_mix: Previous post-mix coefficients.
        previous_residual_mix: Previous residual-mix coefficients.
        phi: Projection weights.
        alpha: Mapping scale parameters.
        bias: Mapping bias parameters.
        norm_weight: Shifted RMSNorm weight.
        num_iters: Sinkhorn iteration count.

    Raises:
        ValueError: If an input shape or iteration count is unsupported.
    """
    if not isinstance(residual, torch.Tensor):
        raise TypeError("residual must be a Tensor.")
    if residual.ndim < 3 or residual.shape[-2] != MHC_STREAMS:
        raise ValueError(f"residual must end in [{MHC_STREAMS}, H], got {tuple(residual.shape)}.")
    leading_shape = tuple(residual.shape[:-2])
    hidden_size = residual.shape[-1]
    expected_shapes = {
        "previous_output": (*leading_shape, hidden_size),
        "previous_pre_mix": (*leading_shape, MHC_STREAMS),
        "previous_post_mix": (*leading_shape, MHC_STREAMS),
        "previous_residual_mix": (*leading_shape, MHC_STREAMS, MHC_STREAMS),
        "phi": (MHC_MAPPINGS, MHC_STREAMS * hidden_size),
        "alpha": (3,),
        "bias": (MHC_MAPPINGS,),
        "norm_weight": (hidden_size,),
    }
    values = {
        "previous_output": previous_output,
        "previous_pre_mix": previous_pre_mix,
        "previous_post_mix": previous_post_mix,
        "previous_residual_mix": previous_residual_mix,
        "phi": phi,
        "alpha": alpha,
        "bias": bias,
        "norm_weight": norm_weight,
    }
    for name, expected in expected_shapes.items():
        if not isinstance(values[name], torch.Tensor):
            raise TypeError(f"{name} must be a Tensor.")
        actual = tuple(values[name].shape)
        if actual != expected:
            raise ValueError(f"{name} must have shape {expected}, got {actual}.")
    if num_iters != MHC_SINKHORN_ITERS:
        raise ValueError(f"initial HyperMegaMhc requires num_iters={MHC_SINKHORN_ITERS}, got {num_iters}.")


def validate_mega_mhc_hidden_size(hidden_size: int, *, need_backward: bool) -> None:
    """Reject hidden dimensions outside the selected native execution contract.

    Args:
        hidden_size: Hidden dimension of each residual stream.
        need_backward: Whether the matching native backward must be supported.
    """
    if not isinstance(hidden_size, int) or isinstance(hidden_size, bool) or hidden_size <= 0:
        raise ValueError(f"hidden_size must be a positive integer, got {hidden_size!r}.")
    if hidden_size % 128:
        raise ValueError(f"hidden_size must be divisible by 128, got {hidden_size}.")
    if need_backward and hidden_size > MHC_GRAD_MAX_HIDDEN_SIZE:
        raise ValueError(
            f"HyperMegaMhc training requires hidden_size <= {MHC_GRAD_MAX_HIDDEN_SIZE}, got {hidden_size}."
        )
    if hidden_size > MHC_FORWARD_MAX_HIDDEN_SIZE:
        raise ValueError(
            f"HyperMegaMhc forward requires hidden_size <= {MHC_FORWARD_MAX_HIDDEN_SIZE}, got {hidden_size}."
        )


def validate_mega_mhc_tensor(
    name: str,
    value: torch.Tensor,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    """Validate one activation, parameter, gradient, or cache tensor.

    Args:
        name: Input name used in diagnostic messages.
        value: Tensor to validate.
        shape: Exact expected shape.
        dtype: Expected dtype for this particular input.
        device: Common execution device.
    """
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a Tensor.")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}.")
    if value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {value.dtype}.")
    if value.device != device:
        raise ValueError(f"{name} must be on {device}, got {value.device}.")


def validate_mega_mhc_native_inputs(
    previous_output: torch.Tensor,
    residual: torch.Tensor,
    previous_pre_mix: torch.Tensor,
    previous_post_mix: torch.Tensor,
    previous_residual_mix: torch.Tensor,
    phi: torch.Tensor,
    alpha: torch.Tensor,
    bias: torch.Tensor,
    norm_weight: torch.Tensor,
    *,
    num_iters: int = MHC_SINKHORN_ITERS,
    need_backward: bool,
    token_tile: int,
) -> None:
    """Validate production inputs without constraining test-only Torch references.

    Args:
        previous_output: Previous block output.
        residual: Previous residual streams.
        previous_pre_mix: Previous pre-mix coefficients.
        previous_post_mix: Previous post-mix coefficients.
        previous_residual_mix: Previous residual-mix coefficients.
        phi: Projection weights.
        alpha: Mapping scale parameters.
        bias: Mapping bias parameters.
        norm_weight: Shifted RMSNorm weight.
        num_iters: Sinkhorn iteration count.
        need_backward: Whether to enforce the backward hidden-size limit.
        token_tile: Requested tokens per scheduler task.
    """
    validate_mega_mhc_inputs(
        previous_output, residual, previous_pre_mix, previous_post_mix, previous_residual_mix,
        phi, alpha, bias, norm_weight, num_iters,
    )
    validate_mega_mhc_hidden_size(residual.shape[-1], need_backward=need_backward)
    if not isinstance(token_tile, int) or isinstance(token_tile, bool) or token_tile <= 0:
        raise ValueError(f"token_tile must be a positive integer, got {token_tile!r}.")
    if residual.numel() == 0:
        raise ValueError("HyperMegaMhc requires at least one token.")
    if residual.device.type != "npu":
        raise TypeError(f"HyperMegaMhc requires NPU tensors, got {residual.device}.")
    inputs = {
        "previous_output": (previous_output, torch.bfloat16),
        "residual": (residual, torch.bfloat16),
        "previous_pre_mix": (previous_pre_mix, torch.float32),
        "previous_post_mix": (previous_post_mix, torch.float32),
        "previous_residual_mix": (previous_residual_mix, torch.float32),
        "phi": (phi, torch.float32),
        "alpha": (alpha, torch.float32),
        "bias": (bias, torch.float32),
        "norm_weight": (norm_weight, torch.bfloat16),
    }
    for name, (value, dtype) in inputs.items():
        validate_mega_mhc_tensor(name, value, tuple(value.shape), dtype, residual.device)


def validate_mega_mhc_core_spec(core_spec: CoreSpec) -> None:
    """Require the fixed 1:2 topology used by the MHC mixed kernels."""
    if core_spec.num_vector_cores != 2 * core_spec.num_cube_cores:
        raise RuntimeError(
            "HyperMegaMhc requires the KERNEL_TYPE_MIX_AIC_1_2 core ratio, "
            f"but Device limits report cube_core_num={core_spec.num_cube_cores} and "
            f"vector_core_num={core_spec.num_vector_cores}."
        )


def validate_mega_mhc_token_count(token_count: int, num_vector_cores: int) -> None:
    """Reject shapes too small for native backward reductions.

    Args:
        token_count: Flattened token count.
        num_vector_cores: Number of Vector cores.
    """
    minimum = max(MHC_GRAD_MIN_TOKENS, num_vector_cores)
    if token_count < minimum:
        raise ValueError(
            f"HyperMegaMhc training requires at least {minimum} tokens, "
            f"got tokens={token_count}, vector_cores={num_vector_cores}."
        )
