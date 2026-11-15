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

"""Two-rank numerical checks for the ordinary MoE example's EP dispatcher."""

import os
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch_npu

from examples import _native
from examples.moe.experts import NativeExperts
from tests.common.multicore.moe_reference import TorchExperts

_HIDDEN = 128
_INTERMEDIATE = 128
_EXPERTS = 4
_TOKENS = 128


def _legacy_permute(inputs, counts, ep_size, local_experts):
    """Retain the pre-fusion permutation for the optional-score comparison."""
    # pylint: disable-next=protected-access
    indices, local_counts = _native._generate_permute_indices(counts, local_experts, ep_size)
    return inputs.shape, inputs[indices], indices, local_counts, False


def _route(pattern, rank, device):
    """Build balanced, sparse and single-destination routes."""
    rows = torch.arange(_TOKENS, device=device, dtype=torch.int64)
    if pattern == "balanced":
        return torch.stack(((rows + rank) % _EXPERTS, (rows + rank + 1) % _EXPERTS), dim=1)
    if pattern == "sparse":
        return torch.stack((rows * 0, rows * 0 + 2), dim=1)
    if pattern == "single_destination":
        return torch.stack((rows * 0, rows * 0 + 1), dim=1)
    raise ValueError(pattern)


def _make_experts(device):
    """Give both independent implementations the same local expert weights."""
    torch.manual_seed(207)
    reference = TorchExperts(_HIDDEN, _INTERMEDIATE, _EXPERTS, group=dist.group.WORLD).to(
        device=device, dtype=torch.bfloat16,
    )
    source = SimpleNamespace(
        gate_up_weight=torch.cat((reference.w1.transpose(1, 2), reference.w3.transpose(1, 2)), dim=-1),
        down_weight=reference.w2.transpose(1, 2),
    )
    config = SimpleNamespace(hidden_size=_HIDDEN, intermediate_size=_INTERMEDIATE, num_experts=_EXPERTS)
    native = NativeExperts(config, source, dist.group.WORLD, device)
    return native, reference


def _compare_full_experts(native, reference, rank, device):
    """Compare outputs and all gradients against replicated independent experts."""
    for pattern in ("balanced", "sparse", "single_destination"):
        native.zero_grad(set_to_none=True)
        reference.zero_grad(set_to_none=True)
        torch.manual_seed(301 + rank)
        hidden = torch.randn(_TOKENS, _HIDDEN, dtype=torch.bfloat16, device=device)
        weights = torch.rand(_TOKENS, 2, dtype=torch.float32, device=device)
        gradient = torch.randn_like(hidden)
        ids = _route(pattern, rank, device)
        native_hidden = hidden.detach().clone().requires_grad_()
        reference_hidden = hidden.detach().clone().requires_grad_()
        native_weights = weights.detach().clone().requires_grad_()
        reference_weights = weights.detach().clone().requires_grad_()

        expected = reference(reference_hidden, ids, reference_weights)
        with (patch.object(torch_npu, "npu_moe_token_permute", wraps=torch_npu.npu_moe_token_permute) as permute,
              patch.object(torch_npu, "npu_moe_token_unpermute", wraps=torch_npu.npu_moe_token_unpermute) as unpermute):
            actual = native(native_hidden, ids, native_weights)
            expected_calls = 1 if pattern == "single_destination" and rank == 1 else 2
            assert permute.call_count == expected_calls
            assert unpermute.call_count == expected_calls
        expected.backward(gradient)
        actual.backward(gradient)
        torch.npu.synchronize()
        pairs = [(actual, expected), (native_hidden.grad, reference_hidden.grad),
                 (native_weights.grad, reference_weights.grad)]
        for name in ("w1", "w2", "w3"):
            pairs.append((getattr(native.module, name).grad, getattr(reference, name).grad))
        for got, want in pairs:
            torch.testing.assert_close(got, want, rtol=2e-2, atol=2e-3, msg=lambda text, label=pattern: f"{label}: {text}")


def _compare_optional_scores(module, rank, device):
    """Check FP32 optional score values and gradients through the EP all-to-alls."""
    counts = torch.tensor([5, 0, 2, 3] if rank == 0 else [0, 2, 5, 3], dtype=torch.int64, device=device)
    torch.manual_seed(701 + rank)
    hidden = torch.randn(10, _HIDDEN, dtype=torch.bfloat16, device=device)
    scores = torch.rand(10, dtype=torch.float32, device=device)
    gradient = torch.randn_like(hidden)
    outcomes = []
    for legacy in (True, False):
        module.zero_grad(set_to_none=True)
        x = hidden.detach().clone().requires_grad_()
        p = scores.detach().clone().requires_grad_()
        with patch.object(_native, "_permute", _legacy_permute) if legacy else nullcontext():
            output = module(x, counts, p)
            output.backward(gradient)
        outcomes.append((output.detach(), x.grad, p.grad,
                         *(parameter.grad.clone() for parameter in module.parameters())))
    torch.npu.synchronize()
    for expected, actual in zip(*outcomes):
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-3)

def test_moe_baseline_ep_permutation():
    """Run source and expert permutation with BF16 kernels across two ranks."""
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group("hccl")
    try:
        device = torch.device("npu", local_rank)
        rank = dist.get_rank()
        native, reference = _make_experts(device)
        _compare_full_experts(native, reference, rank, device)
        _compare_optional_scores(native.module, rank, device)
    finally:
        dist.destroy_process_group()
