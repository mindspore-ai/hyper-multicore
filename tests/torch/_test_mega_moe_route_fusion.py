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
"""Exercise indexed pull with short tails, empty receivers and retained graphs."""

import itertools
import math
import os

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401  # pylint: disable=unused-import

from hyper_multicore import MegaMoeExperts
from tests.common.multicore._mega_moe_utils import write_evidence


_MODES = ("legacy", "permute", "unpermute", "both")


def _new_layer(forward: str, backward: str) -> MegaMoeExperts:
    """Keep only four dispatch tasks so no earlier vector task provides a TPipe."""
    return MegaMoeExperts(
        hidden_size=5120, intermediate_size=128,
        num_experts=4, top_k=2, ep_size=2, ep_group=dist.group.WORLD,
        dispatch_mode="pull", create_parameters=False,
        forward_fusion_mode=forward, backward_fusion_mode=backward,
    )


def _evaluate(module, inputs: tuple, ids: torch.Tensor, parameters: tuple) -> tuple:
    """Compare repeated gradients while the original forward state remains live."""
    source, probabilities, upstream = inputs
    hidden = source.detach().clone().requires_grad_()
    weights = probabilities.detach().clone().requires_grad_()
    counts = torch.bincount(ids.flatten().long(), minlength=4).int()
    with torch.no_grad():
        inference_output = module(hidden, ids, weights, tokens_per_expert=counts, expert_weights=parameters)
        inference_snapshot = inference_output.clone()
        module(hidden * 0.5, ids, weights, tokens_per_expert=counts, expert_weights=parameters)
        torch.testing.assert_close(inference_output, inference_snapshot, rtol=0, atol=0)
    output = module(hidden, ids, weights, tokens_per_expert=counts, expert_weights=parameters)
    torch.testing.assert_close(output, inference_output, rtol=2e-2, atol=2e-3)
    variables = (hidden, weights, *parameters)
    gradients = torch.autograd.grad(output, variables, upstream, retain_graph=True)
    actual = (output.detach().cpu(), *(value.detach().cpu() for value in gradients))
    repeated = torch.autograd.grad(output, variables, upstream)
    for index, (first, second) in enumerate(zip(actual[1:], repeated)):
        torch.testing.assert_close(first, second.cpu(), rtol=2e-2, atol=2e-3,
                                   msg=lambda message, label=f"retained gradient {index}": f"{label}: {message}")
    return actual


def _check_pattern(inputs: tuple, ids: torch.Tensor, parameters: tuple, pattern: str) -> list[dict]:
    """Use the contiguous pull path as the routing-fusion numerical control."""
    reference = _new_layer("legacy", "legacy")
    try:
        expected = _evaluate(reference, inputs, ids, parameters)
    finally:
        reference.close()
    results = []
    pairs = list(itertools.product(_MODES, repeat=2))
    pairs.remove(("permute", "legacy"))
    for forward, backward in [("permute", "legacy"), *pairs]:
        module = _new_layer(forward, backward)
        try:
            actual = _evaluate(module, inputs, ids, parameters)
            for index, (observed, control) in enumerate(zip(actual, expected)):
                torch.testing.assert_close(observed, control, rtol=2e-2, atol=2e-3,
                                           msg=lambda message, label=f"{pattern} {forward}/{backward} tensor {index}":
                                           f"{label}: {message}")
            results.append({"pattern": pattern, "forward": forward, "backward": backward})
        finally:
            module.close()
    return results


def test_indexed_pull_fusion() -> None:
    """Feature: Indexed pull route fusion.
    Description: Run all mode pairs with short tails, an empty receiver and retained backward.
    Expectation: Outputs and all four gradients match contiguous pull without an active transport TPipe.
    """
    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(local)
    if not dist.is_initialized():
        dist.init_process_group("hccl")
    if dist.get_world_size() != 2:
        raise ValueError("indexed pull regression requires two ranks")
    device = torch.device("npu", local)
    torch.manual_seed(71027 + dist.get_rank())
    source = torch.randn((128, 5120), device=device, dtype=torch.bfloat16)
    parameters = (
        (torch.randn((2, 5120, 256), device=device, dtype=torch.bfloat16) / math.sqrt(5120)).requires_grad_(),
        (torch.randn((2, 128, 5120), device=device, dtype=torch.bfloat16) / math.sqrt(128)).requires_grad_(),
    )
    upstream = torch.randn_like(source) / math.sqrt(5120)
    probabilities = torch.rand((128, 2), device=device)
    probabilities /= probabilities.sum(1, keepdim=True)
    routes = (
        ("random", torch.rand((128, 4), device=device).topk(2, dim=1).indices.int()),
        ("single_destination", torch.arange(2, device=device, dtype=torch.int32).expand(128, -1).contiguous()),
    )
    try:
        cases = []
        for pattern, ids in routes:
            cases.extend(_check_pattern((source, probabilities, upstream), ids, parameters, pattern))
        write_evidence({"rank": dist.get_rank(), "passed": True, "cases": cases})
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    test_indexed_pull_fusion()
