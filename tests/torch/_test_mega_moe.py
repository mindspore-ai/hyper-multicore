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

"""Public MegaMoe experts forward/backward system-test workers."""

from __future__ import annotations

import gc
import json
import os
import statistics
import time
from dataclasses import dataclass, fields
from pathlib import Path

# The launcher activates the Ascend runtime and multicore payload before importing this
# worker. Framework imports stay out of the ``test_mega_moe.py`` launcher.
import torch
import torch.distributed as dist
import torch_npu  # noqa: F401  # pylint: disable=unused-import

from hyper_multicore import MegaMoeExperts, shmem
from hyper_multicore.modules.mega_moe.plan import _build_runtime_artifacts
from tests.common.multicore.moe_reference import TorchExperts

_RTOL = 2e-2
_ATOL = 2e-3


def _move_to_test_device(module: torch.nn.Module) -> torch.nn.Module:
    """Move a runtime Torch module to the worker device and dtype."""
    return module.to(device=DEVICE, dtype=torch.bfloat16)


@dataclass(frozen=True)
class MoeShape:
    """Minimal public-layer test shape."""

    local_num_tokens: int = 128
    hidden_size: int = 512
    intermediate_size: int = 128
    num_experts: int = 4
    top_k: int = 2
    ep_size: int = 2

    @property
    def local_experts(self) -> int:
        """Return expert weights owned by each EP rank."""
        return self.num_experts // self.ep_size


@dataclass(frozen=True)
class _RunResult:
    """Forward and backward tensors compared across the two backends."""

    output: torch.Tensor
    input_grad: torch.Tensor
    route_weight_grad: torch.Tensor
    gate_up_weight_grad: torch.Tensor
    down_weight_grad: torch.Tensor
    gate_up_weight: torch.Tensor
    down_weight: torch.Tensor


def _init_dist() -> tuple[int, torch.device]:
    """Bind the local NPU and initialize the EP process group."""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="hccl")
    world_size = dist.get_world_size()
    expected_world_size = int(os.getenv("HP_MEGA_MOE_WORLD_SIZE", "2"))
    if world_size != expected_world_size:
        raise ValueError(
            f"worker requires {expected_world_size} ranks, got {world_size}."
        )
    return dist.get_rank(), torch.device("npu", local_rank)


RANK, DEVICE = _init_dist()


def make_balanced_route(
    shape: MoeShape,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the balanced token-major route and its exact local histogram."""
    positions = torch.arange(
        shape.local_num_tokens * shape.top_k,
        dtype=torch.int64,
        device=DEVICE,
    ).view(shape.local_num_tokens, shape.top_k)
    topk_ids = (positions + RANK * shape.top_k).remainder(shape.num_experts)
    base_weights = torch.arange(
        shape.top_k,
        0,
        -1,
        dtype=torch.float32,
        device=DEVICE,
    )
    topk_weights = (
        (base_weights / base_weights.sum())
        .expand(
            shape.local_num_tokens,
            -1,
        )
        .clone()
    )
    routed_slots = shape.local_num_tokens * shape.top_k
    if routed_slots % shape.num_experts:
        raise ValueError(
            "balanced route requires routed slots divisible by num_experts, "
            f"got routed_slots={routed_slots}, num_experts={shape.num_experts}."
        )
    tokens_per_expert = torch.full(
        (shape.num_experts,),
        routed_slots // shape.num_experts,
        dtype=torch.int32,
        device=DEVICE,
    )
    return topk_ids.to(torch.int32), topk_weights, tokens_per_expert


def _make_fixed_route(
    shape: MoeShape,
    pattern: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build one deterministic route and the Router-owned histogram."""
    if pattern == "balanced":
        return make_balanced_route(shape)
    positions = torch.arange(
        shape.local_num_tokens * shape.top_k,
        dtype=torch.int64,
        device=DEVICE,
    ).view(shape.local_num_tokens, shape.top_k)
    if pattern == "zero_token_experts":
        topk_ids = positions.remainder(shape.ep_size) * shape.local_experts
    elif pattern == "skew":
        token_rows = torch.arange(
            shape.local_num_tokens,
            dtype=torch.int64,
            device=DEVICE,
        )
        second_expert = torch.where(
            token_rows.remainder(8) == 0,
            torch.full_like(token_rows, shape.local_experts),
            torch.ones_like(token_rows),
        )
        topk_ids = torch.stack((torch.zeros_like(token_rows), second_expert), dim=1)
    elif pattern == "single_destination":
        topk_ids = positions.remainder(shape.local_experts)
    else:
        raise ValueError(f"unsupported route pattern {pattern!r}.")
    topk_weights = torch.full(
        (shape.local_num_tokens, shape.top_k),
        1.0 / shape.top_k,
        dtype=torch.float32,
        device=DEVICE,
    )
    tokens_per_expert = torch.bincount(
        topk_ids.reshape(-1),
        minlength=shape.num_experts,
    ).to(torch.int32)
    return topk_ids.to(torch.int32), topk_weights, tokens_per_expert


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Return the local value of a plain tensor or DTensor."""
    return tensor


def new_common_moe(shape: MoeShape) -> TorchExperts:
    """Build the independent, owner-sharded Torch correctness oracle."""
    torch.manual_seed(2026)
    torch.npu.manual_seed(2026)
    common_moe = TorchExperts(
        dim=shape.hidden_size,
        hidden_dim=shape.intermediate_size,
        num_experts=shape.num_experts,
        group=dist.group.WORLD,
    )
    common_moe = _move_to_test_device(common_moe)
    return common_moe


def _copy_common_weights_to_mega(
    common_moe: TorchExperts,
    mega: MegaMoeExperts,
) -> None:
    """Copy the local common-MoE expert shard into MegaMoe's packed layout."""
    gate_weight = _local_tensor(common_moe.w1)
    up_weight = _local_tensor(common_moe.w3)
    down_weight = _local_tensor(common_moe.w2)
    with torch.no_grad():
        if mega.expert_weight_layout == "EOI":
            mega.gate_up_weight.copy_(torch.cat((gate_weight, up_weight), dim=1))
            mega.down_weight.copy_(down_weight)
        else:
            mega.gate_up_weight.copy_(
                torch.cat(
                    (gate_weight.transpose(1, 2), up_weight.transpose(1, 2)),
                    dim=-1,
                )
            )
            mega.down_weight.copy_(down_weight.transpose(1, 2))


def new_layers(
    shape: MoeShape,
    *,
    initial_capacity_factor: float | None = None,
    capacity_growth_factor: float | None = None,
    dispatch_mode: str | None = None,
    expert_weight_layout: str = "EIO",
) -> tuple[MegaMoeExperts, torch.nn.Module]:
    """Construct parameter-aligned MegaMoe and common expert layers.

    Args:
        shape: Local token and expert dimensions.
        initial_capacity_factor: Optional initial push receive factor.
        dispatch_mode: MegaMoe dispatch transport.
        capacity_growth_factor: Optional push growth multiplier.
        expert_weight_layout: Public expert parameter layout.

    Returns:
        MegaMoe and common expert modules with identical local parameters.
    """
    common_moe = new_common_moe(shape)
    mode = dispatch_mode or os.getenv("HP_MEGA_MOE_DISPATCH_MODE", "push")
    mega = MegaMoeExperts(
        hidden_size=shape.hidden_size,
        intermediate_size=shape.intermediate_size,
        num_experts=shape.num_experts,
        top_k=shape.top_k,
        initial_capacity_factor=initial_capacity_factor if mode == "push" else None,
        capacity_growth_factor=capacity_growth_factor if mode == "push" else None,
        dispatch_mode=mode,
        expert_weight_layout=expert_weight_layout,
        ep_size=shape.ep_size,
        ep_group=dist.group.WORLD,
    ).to(device=DEVICE, dtype=torch.bfloat16)
    _copy_common_weights_to_mega(common_moe, mega)
    return mega, common_moe


def make_data(shape: MoeShape) -> tuple[torch.Tensor, torch.Tensor]:
    """Create rank-specific input and upstream gradient tensors."""
    torch.manual_seed(30_000 + RANK)
    torch.npu.manual_seed(30_000 + RANK)
    hidden_states = torch.randn(
        shape.local_num_tokens,
        shape.hidden_size,
        dtype=torch.float32,
        device=DEVICE,
    ).to(torch.bfloat16)
    grad_output = torch.randn(
        shape.local_num_tokens,
        shape.hidden_size,
        dtype=torch.float32,
        device=DEVICE,
    ).to(torch.bfloat16)
    return hidden_states, grad_output


def forward_layer(
    layer: MegaMoeExperts | TorchExperts,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    tokens_per_expert: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run either backend with the same route and weight expert outputs once."""
    if isinstance(layer, MegaMoeExperts):
        return layer(
            hidden_states, topk_ids, topk_weights,
            tokens_per_expert=tokens_per_expert,
        )
    return layer(hidden_states, topk_ids, topk_weights)


def run_layer(
    layer: MegaMoeExperts | torch.nn.Module,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    tokens_per_expert: torch.Tensor | None,
    grad_output: torch.Tensor,
    *,
    accumulation_steps: int = 1,
) -> _RunResult:
    """Accumulate F+B microbatches, take one SGD step, and detach compared tensors."""
    layer.zero_grad(set_to_none=True)
    hidden = hidden_states.detach().clone().requires_grad_(True)
    route_weights = topk_weights.detach().clone().requires_grad_(True)
    first_weight_gradients = ()
    for microbatch in range(accumulation_steps):
        output = forward_layer(
            layer,
            hidden,
            topk_ids,
            route_weights,
            tokens_per_expert=tokens_per_expert,
        )
        output.backward(grad_output)
        if accumulation_steps > 1:
            gradients = expert_weight_gradients(layer)
            if microbatch == 0:
                first_weight_gradients = tuple(gradient.detach().clone() for gradient in gradients)
            else:
                for name, gradient, first in zip(("gate/up", "down"), gradients, first_weight_gradients):
                    assert_close(f"{name} accumulated gradient", gradient, first * (microbatch + 1))
    if isinstance(layer, MegaMoeExperts):
        for weight in (layer.gate_up_weight, layer.down_weight):
            assert weight.grad.stride() == weight.stride(), (
                f"rank={RANK}: expected gradient stride={weight.stride()}, actual={weight.grad.stride()}."
            )
    optimizer = torch.optim.SGD(layer.parameters(), lr=1e-3)
    optimizer.step()
    torch.npu.synchronize(DEVICE)
    gate_up_weight_grad, down_weight_grad = expert_weight_gradients(layer)
    gate_up_weight, down_weight = expert_weights(layer)
    gradients = {
        "input": hidden.grad,
        "route weight": route_weights.grad,
        "gate/up weight": gate_up_weight_grad,
        "down weight": down_weight_grad,
    }
    missing = [name for name, gradient in gradients.items() if gradient is None]
    if missing:
        raise AssertionError(f"rank={RANK}: missing gradients {missing}.")
    return _RunResult(
        output=output.detach().clone(),
        input_grad=hidden.grad.detach().clone(),
        route_weight_grad=route_weights.grad.detach().clone(),
        gate_up_weight_grad=gate_up_weight_grad.detach().clone(),
        down_weight_grad=down_weight_grad.detach().clone(),
        gate_up_weight=gate_up_weight.detach().clone(),
        down_weight=down_weight.detach().clone(),
    )


def expert_weight_gradients(
    layer: MegaMoeExperts | torch.nn.Module,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Return expert gradients in MegaMoe's packed local layout."""
    if isinstance(layer, MegaMoeExperts):
        if layer.expert_weight_layout == "EOI":
            gate_up_grad = layer.gate_up_weight.grad
            down_grad = layer.down_weight.grad
            return (
                None if gate_up_grad is None else gate_up_grad.transpose(1, 2),
                None if down_grad is None else down_grad.transpose(1, 2),
            )
        return layer.gate_up_weight.grad, layer.down_weight.grad
    if not isinstance(layer, TorchExperts):
        raise TypeError(f"unsupported expert layer type {type(layer).__name__}.")
    gate_grad = layer.w1.grad
    up_grad = layer.w3.grad
    down_grad = layer.w2.grad
    if gate_grad is None or up_grad is None or down_grad is None:
        return None, None
    return (
        torch.cat(
            (
                _local_tensor(gate_grad).transpose(1, 2),
                _local_tensor(up_grad).transpose(1, 2),
            ),
            dim=-1,
        ),
        _local_tensor(down_grad).transpose(1, 2),
    )


def expert_weights(
    layer: MegaMoeExperts | torch.nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return expert weights in MegaMoe's packed local layout."""
    if isinstance(layer, MegaMoeExperts):
        if layer.expert_weight_layout == "EOI":
            return (
                layer.gate_up_weight.transpose(1, 2),
                layer.down_weight.transpose(1, 2),
            )
        return layer.gate_up_weight, layer.down_weight
    if not isinstance(layer, TorchExperts):
        raise TypeError(f"unsupported expert layer type {type(layer).__name__}.")
    gate_weight = _local_tensor(layer.w1)
    up_weight = _local_tensor(layer.w3)
    down_weight = _local_tensor(layer.w2)
    return (
        torch.cat(
            (gate_weight.transpose(1, 2), up_weight.transpose(1, 2)),
            dim=-1,
        ),
        down_weight.transpose(1, 2),
    )


def assert_finite(name: str, tensor: torch.Tensor) -> None:
    """Require every compared result to be finite."""
    finite = bool(torch.isfinite(tensor).all().item())
    assert finite, (
        f"rank={RANK}: {name} contains non-finite values, shape={tuple(tensor.shape)}."
    )


def assert_close(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    """Compare BF16 common-MoE and MegaMoe results with diagnostic maxima."""
    if torch.allclose(actual, expected, rtol=_RTOL, atol=_ATOL):
        return
    difference = (actual.float() - expected.float()).abs()
    maximum_absolute = float(difference.max().item())
    maximum_relative = float(
        (difference / expected.float().abs().clamp_min(1e-12)).max().item()
    )
    raise AssertionError(
        f"rank={RANK}: {name} differs from common MoE at rtol={_RTOL}, "
        f"atol={_ATOL}, max_abs={maximum_absolute}, max_rel={maximum_relative}."
    )


def _assert_active_expert_gradients(
    shape: MoeShape,
    local_counts: torch.Tensor,
    result: _RunResult,
    backend: str,
) -> None:
    """Require useful dW for every expert in the balanced route."""
    global_counts = local_counts.clone()
    dist.all_reduce(global_counts, op=dist.ReduceOp.SUM)
    expert_start = RANK * shape.local_experts
    for local_idx in range(shape.local_experts):
        global_idx = expert_start + local_idx
        count = int(global_counts[global_idx].item())
        assert count > 0, (
            f"rank={RANK}: balanced route did not activate expert {global_idx}."
        )
        for name, gradient in (
            ("gate_up_weight_grad", result.gate_up_weight_grad[local_idx]),
            ("down_weight_grad", result.down_weight_grad[local_idx]),
        ):
            nonzero = int(torch.count_nonzero(gradient).item())
            assert nonzero > 0, (
                f"rank={RANK}: {backend} active expert {global_idx} {name} is all zero, "
                f"route_count={count}."
            )


def assert_results_close(common_result: _RunResult, mega_result: _RunResult) -> None:
    """Compare every forward and backward result from two expert backends."""
    for result_field in fields(_RunResult):
        field_name = result_field.name
        common_tensor = getattr(common_result, field_name)
        mega_tensor = getattr(mega_result, field_name)
        assert mega_tensor.shape == common_tensor.shape, (
            f"rank={RANK}: {field_name} shape mismatch: "
            f"mega={tuple(mega_tensor.shape)}, common={tuple(common_tensor.shape)}."
        )
        assert_finite(f"common {field_name}", common_tensor)
        assert_finite(f"MegaMoe {field_name}", mega_tensor)
        assert_close(field_name, mega_tensor, common_tensor)


def test_mega_moe_level0_balanced() -> None:
    """Compare EIO/EOI routed-expert F+B and optimizer updates with common MoE."""
    shape = MoeShape()
    hidden_states, grad_output = make_data(shape)
    mega, common_moe = new_layers(shape)
    try:
        for pattern in ("balanced", "zero_token_experts", "skew", "single_destination"):
            topk_ids, topk_weights, tokens_per_expert = _make_fixed_route(
                shape,
                pattern,
            )
            common_result = run_layer(
                common_moe,
                hidden_states,
                topk_ids,
                topk_weights,
                tokens_per_expert,
                grad_output,
            )
            mega_result = run_layer(
                mega,
                hidden_states,
                topk_ids,
                topk_weights,
                tokens_per_expert,
                grad_output,
            )
            assert_results_close(common_result, mega_result)
            if pattern == "balanced":
                _assert_active_expert_gradients(
                    shape,
                    tokens_per_expert,
                    common_result,
                    "common",
                )
                _assert_active_expert_gradients(
                    shape,
                    tokens_per_expert,
                    mega_result,
                    "MegaMoe",
                )
    finally:
        mega.close()

    # Reuse the bounded-capacity scenario to cover TorchTitan-native EOI storage
    # in the same process group as the default EIO routing cases.
    bounded, bounded_common = new_layers(shape, initial_capacity_factor=1.5, expert_weight_layout="EOI")
    try:
        for pattern in ("balanced", "zero_token_experts", "skew", "single_destination"):
            ids, probabilities, counts = _make_fixed_route(shape, pattern)
            if pattern == "balanced":
                weights = (bounded.gate_up_weight, bounded.down_weight)
                output = forward_layer(bounded, hidden_states, ids, probabilities, tokens_per_expert=counts)
                gradients = torch.autograd.grad(output, weights, grad_output)
                for weight, gradient in zip(weights, gradients):
                    assert gradient.stride() == weight.stride(), (
                        f"rank={RANK}: native gradient stride expected={weight.stride()}, "
                        f"actual={gradient.stride()}."
                    )
            accumulation_steps = 2 if pattern == "balanced" else 1
            reference = run_layer(bounded_common, hidden_states, ids, probabilities, counts, grad_output,
                                  accumulation_steps=accumulation_steps)
            actual = run_layer(bounded, hidden_states, ids, probabilities, counts, grad_output,
                               accumulation_steps=accumulation_steps)
            assert_results_close(reference, actual)
    finally:
        bounded.close()


def performance_shape() -> MoeShape:
    """Return the fixed representative four-card performance shape."""
    return MoeShape(
        local_num_tokens=4096,
        hidden_size=5120,
        intermediate_size=1792,
        num_experts=32,
        top_k=8,
        ep_size=4,
    )


def _new_performance_layer(
    shape: MoeShape,
    backend: str,
) -> MegaMoeExperts | torch.nn.Module:
    """Construct one seeded backend without retaining its comparison peer."""
    if backend == "common":
        return new_common_moe(shape)
    if backend not in {"mega_moe", "mega_moe_external_shared"}:
        raise ValueError(
            f"unsupported performance backend {backend!r}."
        )
    def env_flag(name: str) -> bool:
        """Read the explicit benchmark switches independently of runtime defaults."""
        return os.getenv(name, "0").lower() in {"1", "true", "yes", "on"}

    forward_shared = env_flag("HP_MEGA_MOE_FORWARD_SHARED_EXPERT")
    backward_shared = env_flag("HP_MEGA_MOE_BACKWARD_SHARED_EXPERT")
    if backend == "mega_moe_external_shared":
        forward_shared = backward_shared = False
    shared_intermediate = int(os.getenv("HP_MEGA_MOE_SHARED_INTERMEDIATE_SIZE", "0"))
    include_shared = shared_intermediate > 0
    return MegaMoeExperts(
        hidden_size=shape.hidden_size,
        intermediate_size=shape.intermediate_size,
        num_experts=shape.num_experts,
        top_k=shape.top_k,
        initial_capacity_factor=1.5 if os.getenv("HP_MEGA_MOE_DISPATCH_MODE", "push") == "push" else None,
        dispatch_mode=os.getenv("HP_MEGA_MOE_DISPATCH_MODE", "push"),
        ep_size=shape.ep_size,
        ep_group=dist.group.WORLD,
        shared_intermediate_size=shared_intermediate if include_shared else 0,
        forward_shared_expert=forward_shared,
        backward_shared_expert=backward_shared,
        **({"forward_fusion_mode": "legacy", "backward_fusion_mode": "legacy"}
           if backend == "mega_moe_external_shared" else {}),
    ).to(
        device=DEVICE,
        dtype=torch.bfloat16,
    )


def rank_max_latency(elapsed_ms: float) -> float:
    """Return one F+B latency reduced by maximum across all ranks."""
    latency = torch.tensor(elapsed_ms, dtype=torch.float32, device=DEVICE)
    dist.all_reduce(latency, op=dist.ReduceOp.MAX)
    return float(latency.cpu().item())


def _timed_fwd_bwd(
    layer: MegaMoeExperts | torch.nn.Module,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    grad_output: torch.Tensor,
    *,
    validate_gradients: bool = False,
) -> tuple[float, float]:
    """Time one synchronized layer forward and backward operation."""
    layer.zero_grad(set_to_none=True)
    hidden = hidden_states.detach().requires_grad_(True)
    route_weights = topk_weights.detach().requires_grad_(True)
    dist.barrier()
    torch.npu.synchronize(DEVICE)
    start_event = torch.npu.Event(enable_timing=True)
    end_event = torch.npu.Event(enable_timing=True)
    start = time.perf_counter()
    start_event.record()
    output = forward_layer(
        layer,
        hidden,
        topk_ids,
        route_weights,
        tokens_per_expert=tokens_per_expert,
    )
    output.backward(grad_output)
    end_event.record()
    torch.npu.synchronize(DEVICE)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    gate_up_weight_grad, down_weight_grad = expert_weight_gradients(layer)
    gradients = (
        ("input", hidden.grad),
        ("route weight", route_weights.grad),
        ("gate/up weight", gate_up_weight_grad),
        ("down weight", down_weight_grad),
    )
    if isinstance(layer, MegaMoeExperts) and layer.shared_intermediate_size:
        gradients += (("shared gate/up weight", layer.shared_gate_up_weight.grad),
                      ("shared down weight", layer.shared_down_weight.grad))
    missing = [name for name, gradient in gradients if gradient is None]
    if missing:
        raise AssertionError(
            f"rank={RANK}: performance F+B missed gradients {missing}."
        )
    if validate_gradients:
        finite = torch.ones((), dtype=torch.int32, device=DEVICE)
        for _, gradient in gradients:
            finite = torch.minimum(
                finite,
                torch.isfinite(gradient).all().to(torch.int32),
            )
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not bool(finite.cpu().item()):
            raise AssertionError(
                f"rank={RANK}: performance F+B produced non-finite gradients."
            )
    return rank_max_latency(elapsed_ms), rank_max_latency(start_event.elapsed_time(end_event))


def _performance_result(
    backend: str,
    shape: MoeShape,
    warmup_steps: int,
    latencies: list[float],
    device_latencies: list[float],
) -> dict:
    """Build one routed-expert performance result."""
    return {
        "backend": backend,
        "topology": {
            "world_size": dist.get_world_size(),
            "ep": shape.ep_size,
        },
        "shape": {
            "local_num_tokens": shape.local_num_tokens,
            "hidden_size": shape.hidden_size,
            "intermediate_size": shape.intermediate_size,
            "num_experts": shape.num_experts,
            "top_k": shape.top_k,
            "dtype": "bfloat16",
        },
        "warmup_steps": warmup_steps,
        "measured_steps": len(latencies),
        "validation": {"gradients_present": True, "gradients_finite": True},
        "shared_expert": {
            "intermediate_size": int(os.getenv("HP_MEGA_MOE_SHARED_INTERMEDIATE_SIZE", "0")),
            "forward": os.getenv("HP_MEGA_MOE_FORWARD_SHARED_EXPERT", "0"),
            "backward": os.getenv("HP_MEGA_MOE_BACKWARD_SHARED_EXPERT", "0"),
            "external_reference": backend == "mega_moe_external_shared",
        },
        "route_fusion": {
            "forward": os.getenv("HYPER_MULTICORE_MEGA_MOE_FORWARD_FUSION_MODE", "default"),
            "backward": os.getenv("HYPER_MULTICORE_MEGA_MOE_BACKWARD_FUSION_MODE", "default"),
        },
        "rank_max_fwd_bwd_ms": {
            "mean": statistics.mean(latencies),
            "median": statistics.median(latencies),
            "minimum": min(latencies),
            "maximum": max(latencies),
            "samples": latencies,
        },
        "rank_max_npu_event_ms": {
            "mean": statistics.mean(device_latencies),
            "median": statistics.median(device_latencies),
            "minimum": min(device_latencies),
            "maximum": max(device_latencies),
            "samples": device_latencies,
        },
        "device_healthy": True,
    }


def _record_performance_schedule(result: dict, layer: MegaMoeExperts, backend: str, shape: MoeShape) -> None:
    """Describe the exact graph selected by the measured public-layer invocation."""
    resources = layer._resource_group.resources
    plan = resources.plan_for_tokens(shape.local_num_tokens, DEVICE)
    result["route_fusion"] = {
        "forward": plan.spec.forward_fusion_mode,
        "backward": plan.spec.backward_fusion_mode,
    }
    result["shared_expert"].update({
        "forward": plan.spec.forward_shared_expert,
        "backward": plan.spec.backward_shared_expert,
        "external_reference": backend == "mega_moe_external_shared",
    })
    result["integration"] = {
        "reuse_backward_dispatch": plan.reuse_backward_dispatch,
        "forward_shared_sum_in_kernel": plan.spec.forward_unpermute_fused and (
            plan.spec.forward_shared_expert or plan.spec.backward_shared_expert),
        "backward_shared_sum_in_kernel": (
            plan.spec.backward_unpermute_fused and plan.spec.backward_shared_expert),
        "shared_weight_gradient_layout": "EOI" if plan.spec.backward_shared_expert else "autograd",
    }
    _, fwd_config, _, bwd_config = _build_runtime_artifacts(plan.spec)
    result["schedule"] = {
        direction: {
            # Both worker queues include termination; forward task_num does not.
            "total_tasks": sum(int(count) for count in config.task_index_num[:2]),
            "aic_tasks": int(config.task_index_num[0]),
            "aiv_tasks": int(config.task_index_num[1]),
        }
        for direction, config in (("forward", fwd_config), ("backward", bwd_config))
    }


def _record_performance_memory(result: dict, memory_values: list[int]) -> None:
    """Aggregate captured memory metrics without counting reporting allocations."""
    torch.npu.synchronize(DEVICE)
    memory = torch.tensor(memory_values, dtype=torch.int64, device=DEVICE)
    dist.all_reduce(memory, op=dist.ReduceOp.MAX)
    names = ("peak_allocated", "peak_reserved", "peak_device_used_sampled", "shmem_heap",
             "current_allocated", "current_reserved", "baseline_allocated", "baseline_reserved",
             "peak_allocated_increment")
    result["memory_bytes"] = dict(zip(names, (int(value) for value in memory.cpu().tolist())))
    result["memory_measurement"] = {
        "scope": "steady_state_fwd_bwd_after_gradient_validation_and_warmup",
        "reset_after": "clear_parameter_gradients_collect_and_synchronize",
        "validation_cache_released_before_warmup": True,
        "aggregation": "maximum_across_ranks_for_each_metric",
        "increment": "maximum_across_ranks_of_local_peak_minus_local_baseline",
        "device_used_sampling": "before_measurement_and_after_each_synchronized_fwd_bwd",
        "device_used_is_continuous_peak": False,
    }


def _benchmark_performance_backend(
    backend: str,
    shape: MoeShape,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    grad_output: torch.Tensor,
    warmup_steps: int,
    measured_steps: int,
) -> dict:
    """Warm up and measure one routed-expert backend."""
    torch.manual_seed(99173 + RANK)
    layer = _new_performance_layer(shape, backend)
    try:
        def sample_memory() -> tuple[int, int]:
            """Sample synchronized device use and the current symmetric heap."""
            torch.npu.synchronize(DEVICE)
            free_bytes, total_bytes = torch.npu.mem_get_info(DEVICE)
            heap_bytes = (shmem.debug_state().get("config") or {}).get("heap_size_bytes", 0)
            return total_bytes - free_bytes, heap_bytes

        _timed_fwd_bwd(
            layer,
            hidden_states,
            topk_ids,
            topk_weights,
            tokens_per_expert,
            grad_output,
            validate_gradients=True,
        )
        # Finite checks allocate full-sized temporary masks. Release their cached
        # blocks before warmup so they affect neither live nor reserved peaks.
        layer.zero_grad(set_to_none=True)
        gc.collect()
        torch.npu.synchronize(DEVICE)
        torch.npu.empty_cache()
        for _ in range(warmup_steps):
            _timed_fwd_bwd(
                layer, hidden_states, topk_ids, topk_weights, tokens_per_expert, grad_output,
            )
        layer.zero_grad(set_to_none=True)
        gc.collect()
        torch.npu.synchronize(DEVICE)
        baseline_allocated = torch.npu.memory_allocated(DEVICE)
        baseline_reserved = torch.npu.memory_reserved(DEVICE)
        peak_device_used, peak_shmem_heap = sample_memory()
        torch.npu.reset_peak_memory_stats(DEVICE)
        latencies = []
        device_latencies = []
        for _ in range(measured_steps):
            latency, device_latency = _timed_fwd_bwd(
                layer,
                hidden_states,
                topk_ids,
                topk_weights,
                tokens_per_expert,
                grad_output,
            )
            used_bytes, heap_bytes = sample_memory()
            peak_device_used = max(peak_device_used, used_bytes)
            peak_shmem_heap = max(peak_shmem_heap, heap_bytes)
            latencies.append(latency)
            device_latencies.append(device_latency)
        # Capture allocator statistics before diagnostic graph construction and
        # the reporting reduction allocate their own temporary tensors.
        peak_allocated = torch.npu.max_memory_allocated(DEVICE)
        memory_values = [
            peak_allocated, torch.npu.max_memory_reserved(DEVICE),
            peak_device_used, peak_shmem_heap, torch.npu.memory_allocated(DEVICE),
            torch.npu.memory_reserved(DEVICE), baseline_allocated, baseline_reserved,
            peak_allocated - baseline_allocated,
        ]
        result = _performance_result(backend, shape, warmup_steps, latencies, device_latencies)
        if isinstance(layer, MegaMoeExperts):
            _record_performance_schedule(result, layer, backend, shape)
        _record_performance_memory(result, memory_values)
        return result
    finally:
        if isinstance(layer, MegaMoeExperts):
            layer.close()


def test_mega_moe_fwd_bwd_performance() -> None:
    """Compare selected reference and MegaMoe F+B backends in one distributed launch."""
    warmup_steps = int(os.getenv("HP_MEGA_MOE_PERF_WARMUP", "3"))
    measured_steps = int(os.getenv("HP_MEGA_MOE_PERF_MEASURED", "5"))
    if warmup_steps < 0 or measured_steps <= 0:
        raise ValueError(
            f"invalid performance steps: warmup={warmup_steps}, measured={measured_steps}."
        )

    shape = performance_shape()
    topk_ids, topk_weights, tokens_per_expert = make_balanced_route(shape)
    hidden_states, grad_output = make_data(shape)
    results = {}
    shared_enabled = int(os.getenv("HP_MEGA_MOE_SHARED_INTERMEDIATE_SIZE", "0")) > 0
    backends = ("mega_moe_external_shared", "mega_moe") if shared_enabled else ("common", "mega_moe")
    if os.getenv("HP_MEGA_MOE_PERF_REVERSE") == "1":
        backends = tuple(reversed(backends))
    for backend in backends:
        results[backend] = _benchmark_performance_backend(
            backend,
            shape,
            hidden_states,
            topk_ids,
            topk_weights,
            tokens_per_expert,
            grad_output,
            warmup_steps,
            measured_steps,
        )
        gc.collect()
        torch.npu.empty_cache()
        dist.barrier()

    ratio = None
    if shared_enabled:
        reference_median = results["mega_moe_external_shared"]["rank_max_fwd_bwd_ms"]["median"]
        mega_median = results["mega_moe"]["rank_max_fwd_bwd_ms"]["median"]
        ratio = mega_median / reference_median
    else:
        common_median = results["common"]["rank_max_fwd_bwd_ms"]["median"]
        mega_median = results["mega_moe"]["rank_max_fwd_bwd_ms"]["median"]
        ratio = mega_median / common_median
    if RANK == 0:
        result_path = os.getenv("HP_MEGA_MOE_PERF_RESULT")
        if not result_path:
            raise ValueError("HP_MEGA_MOE_PERF_RESULT must name the output JSON path.")
        result = {
            "order": list(results),
            "results": results,
            "mega_over_reference": ratio,
            "performance_acceptance": False,
            "reference_kind": ("external_shared_legacy_megamoe" if shared_enabled
                               else "independent_torch_correctness_oracle"),
        }
        path = Path(result_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"PERF_RESULT_JSON={path}")
        print(json.dumps(result, sort_keys=True))
    dist.barrier()
