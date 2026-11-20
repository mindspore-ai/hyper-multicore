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
"""Small actual-device tests that force placement only to exercise edge geometries."""

from functools import partial
import gc
import itertools
import json
import os
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint
from hyper_multicore import MegaMoePlacementOptions, MegaMoeExperts, MegaMoeNumericsOptions, SwiGLUParameters
from hyper_multicore.experimental import MegaMoeTuningOptions
from hyper_multicore.modules.hot_replica.planner import HotReplicaPlanner
from hyper_multicore.modules.hot_replica import mega_moe_adapter


def relative(got: torch.Tensor, expected: torch.Tensor) -> float:
    """Measure one entire tensor relative to the reference norm."""
    a, b = got.detach().float(), expected.detach().float()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))



def _call_experts(op, ids, hidden, probs, weight1, weight2):
    """Execute one fixed owner parameter set and its selected route."""
    return op(hidden, ids, probs, expert_weights=(weight1, weight2))

def _execute_graph(call, leaves, local_dy, mode):
    """Exercise the requested graph lifecycle with ordinary checkpointing."""
    stream = torch.npu.Stream() if mode == "cross_stream" else torch.npu.current_stream()
    stream.wait_stream(torch.npu.current_stream())
    with patch.object(mega_moe_adapter, "placement_is_profitable", return_value=True), patch.object(
        mega_moe_adapter, "placement_receive_limit", side_effect=lambda target, *_: target
    ):
        if mode == "inference_first":
            with torch.inference_mode():
                initial = call(*leaves)
            assert torch.isfinite(initial).all()
            del initial
        output = (
            checkpoint(call, *leaves, use_reentrant=mode == "reentrant")
            if mode in ("reentrant", "nonreentrant")
            else call(*leaves)
        )
        if mode == "interleaved":
            other = call(leaves[0] * 0.7, leaves[1], leaves[2] * 0.9, leaves[3] * 1.1)
            other.backward(local_dy)
            for leaf in leaves:
                leaf.grad = None
            del other
        if mode == "cross_stream":
            stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            output.backward(local_dy, retain_graph=mode == "retain")
            if mode == "retain":
                for leaf in leaves:
                    leaf.grad = None
                output.backward(local_dy)
        torch.npu.current_stream().wait_stream(stream)
    observed = [output, *(leaf.grad for leaf in leaves)]
    return output, observed


def _check_joint_result(op, observed, reference, config):
    """Require real transfers and compare all differentiable tensor outputs."""
    owner = op._resource_group.resources.placement  # pylint: disable=protected-access
    assert owner.statistics["placements"] > 0 and owner.pool is not None
    assert owner.pool.statistics["gradient_returns"] > 0
    errors = [relative(a, b) for a, b in zip(observed, reference)]
    assert all(error < 0.025 for error in errors), (config, errors)
    return {"config": config, "relative_errors": errors, "transfers": dict(owner.pool.statistics)}


def _configurations():
    """Allow one graph-lifecycle case to be rerun without changing its checks."""
    configs = list(itertools.product(("EIO", "EOI"), (False, True), ("native", "float32"), (128, 512)))
    configs += [
        ("EIO", True, "float32", 128, mode)
        for mode in (
            "reentrant",
            "nonreentrant",
            "retain",
            "cross_stream",
            "interleaved",
            "inference_first",
            "empty_source",
        )
    ]
    mode = os.environ.get("EDGE_MODE")
    return [c for c in configs if len(c) == 5 and c[4] == mode] if mode else configs


def main() -> None:
    """Run every requested geometry and graph lifecycle on two NPU ranks."""
    rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(rank)
    dist.init_process_group("hccl")
    device = torch.device("npu", rank)
    records = []
    try:
        torch.manual_seed(91 + rank)
        tokens, h, i, local_experts, k = 257, 256, 128, 4, 2
        values = (
            torch.randn(tokens, h, device=device, dtype=torch.bfloat16) * 0.2,
            torch.rand(tokens, k, device=device).softmax(-1),
            torch.randn(local_experts, h, 2 * i, device=device, dtype=torch.bfloat16) * 0.02,
            torch.randn(local_experts, i, h, device=device, dtype=torch.bfloat16) * 0.02,
        )
        ids = torch.arange(k, device=device, dtype=torch.int32).expand(tokens, k).contiguous()
        dy = torch.randn_like(values[0]) * 0.1
        configs = _configurations()
        for number, config in enumerate(configs):
            layout, before, dw, split, *modes = config
            mode = next(iter(modes), "ordinary")
            reference = None
            for enabled in (False, True):
                op = MegaMoeExperts(
                    hidden_size=h,
                    intermediate_size=i,
                    num_experts=local_experts * 2,
                    top_k=k,
                    ep_group=dist.group.WORLD,
                    activation=SwiGLUParameters(
                        clamp_limit=0.15,
                        clamp_after_silu=number % 2 == 0,
                    ),
                    numerics_options=MegaMoeNumericsOptions(
                        apply_probs_before_fc2=before,
                        partial_weight_grad_dtype=dw,
                    ),
                    expert_placement=MegaMoePlacementOptions("adaptive", HotReplicaPlanner()) if enabled else None,
                    tuning_options=MegaMoeTuningOptions(
                        dispatch_split=split, combine_split=split,
                    ),
                )
                leaves = [v.detach().clone().requires_grad_() for v in values]
                if layout == "EOI":
                    leaves[2:] = [v.transpose(1, 2).contiguous().detach().requires_grad_() for v in leaves[2:]]
                if mode == "empty_source" and rank == 0:
                    leaves[:2] = [v[:0].detach().requires_grad_() for v in leaves[:2]]
                local_ids = ids[: leaves[0].shape[0]]
                local_dy = dy[: leaves[0].shape[0]]

                call = partial(_call_experts, op, local_ids)

                output, observed = _execute_graph(call, leaves, local_dy, mode)
                if enabled:
                    records.append(_check_joint_result(op, observed, reference, config))
                else:
                    reference = [a.detach().clone() for a in observed]
                del output, observed, leaves
                gc.collect()
                op.close()
                del op
            if rank == 0:
                print("EDGE_PASS", number, config, flush=True)
        Path(os.environ["EDGE_OUT"]).mkdir(exist_ok=True, parents=True)
        (Path(os.environ["EDGE_OUT"]) / f"rank{rank}.json").write_text(json.dumps(records))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
