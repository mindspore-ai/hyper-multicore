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
"""Dynamic route/shared fusion with retained graphs and empty EP participants."""

import math
import os

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401  # pylint: disable=unused-import

from hyper_multicore import MegaMoeExperts
from tests.common.multicore._mega_moe_utils import write_evidence
from tests.torch._test_mega_moe_shared import (
    _check_empty_experts, _check_retained, _compare_results, _forward, _route_ids,
)


_HIDDEN, _INTERMEDIATE, _SHARED = 5120, 128, 256
_NAMES = ("output", "dx", "dprob", "routed_dw1", "routed_dw2", "shared_dw1", "shared_dw2")
_SWITCHES = (
    ("legacy", "legacy", False, False),
    ("both", "both", True, True),
    ("permute", "legacy", True, False),
    ("unpermute", "legacy", False, True),
    ("legacy", "permute", False, True),
    ("legacy", "unpermute", True, False),
    ("both", "permute", False, False),
    ("unpermute", "both", True, True),
)
_CASES = (
    ("pending_before_growth", (128, 128), "balanced"),
    ("zero_input", (0, 257), "balanced"),
    ("ragged", (1, 129), "random"),
    ("growth_zero_receiver", (257, 257), "zero_experts"),
    ("zero_receiver", (129, 129), "zero_experts"),
    ("larger", (512, 512), "balanced"),
    ("zero_all", (0, 0), "balanced"),
    ("revisit", (128, 128), "random"),
)


def _batches(rank, device):
    """Create identical real-token inputs for the reference and fused executions."""
    result = []
    for index, (name, lengths, pattern) in enumerate(_CASES):
        torch.manual_seed(91600 + index * 2 + rank)
        tokens = lengths[rank]
        x = torch.randn(tokens, _HIDDEN, device=device, dtype=torch.bfloat16)
        probabilities = torch.rand(tokens, 2, device=device)
        probabilities /= probabilities.sum(-1, keepdim=True)
        ids = _route_ids(tokens, 4, 2, device, pattern)
        counts = torch.bincount(ids.flatten().long(), minlength=4).int()
        global_counts = counts.clone()
        dist.all_reduce(global_counts)
        upstream = torch.randn_like(x) / math.sqrt(_HIDDEN)
        result.append((x, ids, probabilities, None if name in {"ragged", "zero_input"} else counts,
                       upstream, global_counts.chunk(2)[rank].cpu()))
    return result


def _evaluate(layer, inputs, output, upstream, local_counts, retained):
    """Snapshot every gradient before later schedules can reuse scratch buffers."""
    x, _, probabilities, _, weights = inputs
    gradients = torch.autograd.grad(output, (x, probabilities, *weights), upstream, retain_graph=retained)
    torch.npu.synchronize()
    actual = tuple(value.detach().cpu().clone() for value in (output, *gradients))
    for value in actual:
        assert torch.isfinite(value).all(), "non-finite dynamic output or gradient"
    _check_empty_experts(actual, local_counts)
    if retained:
        _check_retained(layer, inputs, output, gradients, upstream)
    return actual


def _run_sequence(layer, batches, weights, retained):
    """Keep a graph live through source growth before skew inflates receive capacity."""
    results = {}
    pending = None
    for index, batch in enumerate(batches):
        source, ids, probabilities, counts, upstream, local_counts = batch
        inputs = (source.detach().clone().requires_grad_(), ids,
                  probabilities.detach().clone().requires_grad_(), counts, weights)
        output = _forward(layer, *inputs)
        torch.npu.synchronize()
        if index == 0:
            capacity = layer._resource_group.resources._token_capacity  # pylint: disable=protected-access
            pending = (inputs, output, upstream, local_counts, output.detach().clone(), capacity)
            continue
        results[index] = _evaluate(layer, inputs, output, upstream, local_counts, retained)
        if index == 1:
            old_inputs, old_output, old_upstream, old_counts, snapshot, capacity = pending
            grown = layer._resource_group.resources._token_capacity  # pylint: disable=protected-access
            assert grown > capacity, "larger input did not grow the source reservation"
            torch.testing.assert_close(old_output, snapshot, rtol=0, atol=0)
            results[0] = _evaluate(layer, old_inputs, old_output, old_upstream, old_counts, retained)
            pending = None
    return results


def test_dynamic_shared_fusion():
    """Compare automatic mixed/full fusion with legacy routing and external shared math."""
    local = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local)
    dist.init_process_group("hccl")
    if dist.get_world_size() != 2:
        raise ValueError("dynamic shared fusion requires two ranks")
    device = torch.device("npu", local)
    batches = _batches(local, device)
    torch.manual_seed(91730 + local)
    parameters = (
        torch.randn(2, _HIDDEN, 2 * _INTERMEDIATE, device=device, dtype=torch.bfloat16) / math.sqrt(_HIDDEN),
        torch.randn(2, _INTERMEDIATE, _HIDDEN, device=device, dtype=torch.bfloat16) / math.sqrt(_INTERMEDIATE),
        torch.randn(2 * _SHARED, _HIDDEN, device=device, dtype=torch.bfloat16) / math.sqrt(_HIDDEN),
        torch.randn(_HIDDEN, _SHARED, device=device, dtype=torch.bfloat16) / math.sqrt(_SHARED),
    )
    expected = None
    mode = os.getenv("HP_MEGA_MOE_DISPATCH_MODE", "push")
    try:
        for forward, backward, shared_forward, shared_backward in _SWITCHES:
            full = (forward, backward, shared_forward, shared_backward) == ("both", "both", True, True)
            layer = MegaMoeExperts(
                hidden_size=_HIDDEN, intermediate_size=_INTERMEDIATE, num_experts=4, top_k=2,
                ep_size=2, ep_group=dist.group.WORLD, dispatch_mode=mode, create_parameters=False,
                shared_intermediate_size=_SHARED, fusion_mode="full" if full else None,
                forward_fusion_mode=forward, backward_fusion_mode=backward,
                forward_shared_expert=shared_forward, backward_shared_expert=shared_backward,
            )
            weights = tuple(value.detach().clone().requires_grad_() for value in parameters)
            try:
                results = _run_sequence(layer, batches, weights, expected is not None)
                if expected is None:
                    expected = results
                else:
                    for index, actual in sorted(results.items()):
                        _compare_results(actual, expected[index], _NAMES, local, shared_forward, shared_backward)
                        print(f"DYNAMIC_SHARED_PASS rank={local} transport={mode} case={_CASES[index][0]} "
                              f"route={forward}/{backward} shared={shared_forward}/{shared_backward}", flush=True)
            finally:
                layer.close()
        write_evidence({"rank": dist.get_rank(), "passed": True, "dispatch_mode": mode,
                        "switches": _SWITCHES,
                        "cases": [{"name": name, "lengths": lengths, "pattern": pattern}
                                  for name, lengths, pattern in _CASES]})
    finally:
        dist.destroy_process_group()
