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
"""Numerical regression for independently fused local shared experts."""

import itertools
import math
import os

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401  # pylint: disable=unused-import

from hyper_multicore import MegaMoeExperts


def _route_ids(tokens, experts, top_k, device, pattern):
    """Build deterministic balanced, random and empty-rank routes."""
    if pattern == "random":
        ids = torch.rand(tokens, experts, device=device).topk(top_k, dim=-1).indices.int()
    else:
        ids = torch.arange(tokens * top_k, device=device).view(tokens, top_k).remainder(experts).int()
        if pattern in {"zero_experts", "skewed"}:
            ids[:] = torch.arange(top_k, device=device)
            if pattern == "skewed":
                ids[-16:] += top_k
        elif pattern != "balanced":
            raise ValueError(f"unknown shared route pattern: {pattern}")
    return ids


def _forward(layer, x, ids, probabilities, counts, weights):
    """Use identical route and shared operands for every lifecycle probe."""
    return layer(x, ids, probabilities, tokens_per_expert=counts, expert_weights=weights[:2],
                 shared_gate_up_weight=weights[2], shared_down_weight=weights[3])


def _check_retained(layer, inputs, output, gradients, upstream):
    """Require exact repeatability and independent pending-forward storage."""
    x, _, probabilities, _, weights = inputs
    variables = (x, probabilities, *weights)
    gradients = tuple(value.detach().clone() for value in gradients)
    for repeat in range(3):
        repeated = torch.autograd.grad(output, variables, upstream, retain_graph=repeat < 2)
        torch.npu.synchronize()
        for observed, reference in zip(repeated, gradients):
            torch.testing.assert_close(observed, reference, rtol=0, atol=0,
                                       msg=lambda message: f"repeat {repeat}: {message}")
    snapshot = output.detach().clone()
    another = _forward(layer, *inputs)
    torch.autograd.grad(another, variables, upstream)
    torch.testing.assert_close(output, snapshot, rtol=0, atol=0)
    pending = [_forward(layer, *inputs) for _ in range(2)]
    for value in reversed(pending):
        observed = torch.autograd.grad(value, variables, upstream)
        for gradient, reference in zip(observed, gradients):
            torch.testing.assert_close(gradient, reference, rtol=0, atol=0)


def _check_empty_experts(actual, local_counts):
    """Empty routed experts must not return stale workspace gradients."""
    for expert, count in enumerate(local_counts):
        if count == 0:
            for gradient in actual[3:5]:
                assert torch.count_nonzero(gradient[expert]) == 0, "nonzero empty-expert gradient"


def _compare_results(actual, expected, names, rank, shared_forward, shared_backward):
    """Apply the original external threshold to every output and gradient."""
    for name, observed, reference in zip(names, actual, expected):
        difference = (observed.float() - reference.float()).abs()
        maximum = difference.max().item() if difference.numel() else 0.0
        rms = difference.square().mean().sqrt().item() if difference.numel() else 0.0
        reference_max = reference.float().abs().max().item() if reference.numel() else 0.0
        print(f"SHARED_ERROR rank={rank} forward={shared_forward} backward={shared_backward} "
              f"tensor={name} max_abs={maximum:.8g} rms={rms:.8g} "
              f"reference_max={reference_max:.8g}", flush=True)
        torch.testing.assert_close(observed, reference, rtol=5e-2, atol=1e-2,
                                   msg=lambda message: f"shared {shared_forward}/{shared_backward} {name}: {message}")


def _switch_cases():
    """Keep cold-start order tests separate from the complete switch matrix."""
    modes = (("legacy", "permute", "unpermute", "both")
             if os.getenv("HP_MEGA_MOE_TEST_ROUTE_MATRIX") == "1" else ("legacy",))
    cases = itertools.product(modes, modes, (False, True), (False, True))
    if os.getenv("HP_MEGA_MOE_TEST_FULL_ONLY") == "1":
        cases = (("legacy", "legacy", False, False), ("both", "both", True, True))
        if os.getenv("HP_MEGA_MOE_TEST_FULL_FIRST") == "1":
            cases = tuple(reversed(cases))
    return cases


def test_shared_direction_switches():
    """Compare forward and all gradients against the external shared branch."""
    local = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local)
    dist.init_process_group("hccl")
    device = torch.device("npu", local)
    torch.manual_seed(81130 + local)
    representative = os.getenv("HP_MEGA_MOE_SHARED_REPRESENTATIVE") == "1"
    tokens, hidden, intermediate, shared = (4096, 5120, 1792, 1792) if representative else (128, 5120, 128, 1792)
    experts, top_k = (32, 8) if representative else (4, 2)
    world = dist.get_world_size()
    source = torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16)
    probabilities = torch.rand(tokens, top_k, device=device)
    probabilities /= probabilities.sum(-1, keepdim=True)
    pattern = os.getenv("HP_MEGA_MOE_SHARED_ROUTE_PATTERN", "balanced" if representative else "random")
    ids = _route_ids(tokens, experts, top_k, device, pattern)
    counts = torch.bincount(ids.flatten().long(), minlength=experts).int()
    global_counts = counts.clone()
    dist.all_reduce(global_counts)
    local_counts = global_counts.chunk(world)[local].cpu()
    upstream = torch.randn_like(source) / math.sqrt(hidden)
    parameters = (
        torch.randn(experts // world, hidden, 2 * intermediate, device=device, dtype=source.dtype) / math.sqrt(hidden),
        torch.randn(experts // world, intermediate, hidden, device=device, dtype=source.dtype)
        / math.sqrt(intermediate),
        torch.randn(2 * shared, hidden, device=device, dtype=source.dtype) / math.sqrt(hidden),
        torch.randn(hidden, shared, device=device, dtype=source.dtype) / math.sqrt(shared),
    )
    expected = None
    names = ("output", "dx", "dprob", "routed_dw1", "routed_dw2", "shared_dw1", "shared_dw2")
    try:
        for forward, backward, shared_forward, shared_backward in _switch_cases():
            retained = (forward, backward, shared_forward, shared_backward) == ("both", "both", True, True)
            layer = MegaMoeExperts(
                hidden_size=hidden, intermediate_size=intermediate,
                num_experts=experts, top_k=top_k, ep_size=world, ep_group=dist.group.WORLD,
                create_parameters=False, shared_intermediate_size=shared,
                fusion_mode="full" if retained else None,
                forward_fusion_mode=forward, backward_fusion_mode=backward,
                forward_shared_expert=shared_forward, backward_shared_expert=shared_backward,
            )
            x = source.detach().clone().requires_grad_()
            p = probabilities.detach().clone().requires_grad_()
            weights = tuple(value.detach().clone().requires_grad_() for value in parameters)
            output = gradients = None
            try:
                inputs = (x, ids, p, counts, weights)
                output = _forward(layer, *inputs)
                torch.npu.synchronize()
                gradients = torch.autograd.grad(output, (x, p, *weights), upstream, retain_graph=retained)
                torch.npu.synchronize()
                if retained:
                    _check_retained(layer, inputs, output, gradients, upstream)
                actual = tuple(value.detach().cpu() for value in (output, *gradients))
                _check_empty_experts(actual, local_counts)
                if expected is None:
                    expected = actual
                    for value in actual:
                        assert torch.isfinite(value).all(), "non-finite reference"
                else:
                    _compare_results(actual, expected, names, local, shared_forward, shared_backward)
                print(f"SHARED_PASS rank={local} route={forward}/{backward} "
                      f"forward={shared_forward} backward={shared_backward}", flush=True)
            finally:
                del output, gradients
                layer.close()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    test_shared_direction_switches()
