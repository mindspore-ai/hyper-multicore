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

import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint
from hyper_multicore import MegaMoeExperts, MegaMoeNumericsOptions, SwiGLUParameters
from hyper_multicore.experimental import MegaMoeTuningOptions, wrap_checkpoint
from hyper_multicore.modules.hot_replica.pipeline import _band_layout, projection_shapes
from tests.torch._test_hot_replica import SkewPlanner


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
    if mode == "reentrant":
        call = wrap_checkpoint(
            call, lambda: tuple(weight._version for weight in leaves[2:]),  # pylint: disable=protected-access
            prepare_replay=True,
        )
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


def _check_joint_result(op, planner, observed, reference, config):
    """Require real transfers and compare all differentiable tensor outputs."""
    owner = op._resource_group.resources.placement  # pylint: disable=protected-access
    assert owner.statistics["placements"] > 0 and owner.pool is not None
    transfers = owner.pool.statistics
    assert transfers["remote_gradient_returns"] + transfers.get("fused_gradient_returns", 0) > 0
    if config[-1] == "reentrant":
        assert owner.statistics["route_replays"] == 1
        assert owner.statistics["skipped_recomputes"] == 1
    if config[-1] == "multiple_guests":
        placement = planner.last_placement
        assert max(sum(expert >= 0 for expert in row) for row in placement.copies_by_rank) >= 2
        assert transfers.get("fused_gradient_returns", 0) > 0
        spec = owner.parent.spec
        bands = [_band_layout(shape)[1]
                 for shape in projection_shapes(spec.hidden_size, spec.intermediate_size, spec.expert_weight_layout)]
        assert min(bands) > 1
    errors = [relative(a, b) for a, b in zip(observed, reference)]
    assert all(error < 0.025 for error in errors), (config, errors)
    return {"config": config, "relative_errors": errors, "transfers": dict(owner.pool.statistics)}


def _check_shared_bank(device, ranks):
    """Share a bank across two independent executors that still hold backward graphs."""
    h, i, tokens, home = 256, 128, 128, 2
    ids = torch.tensor([0, 1], device=device, dtype=torch.int32).expand(tokens, -1).contiguous()
    probs = torch.rand(tokens, 2, dtype=torch.float32, device=device).softmax(-1)
    hidden = torch.randn(tokens, h, dtype=torch.bfloat16, device=device) * 0.1
    w13 = torch.randn(home, h, 2 * i, dtype=torch.bfloat16, device=device) * 0.02
    w2 = torch.randn(home, i, h, dtype=torch.bfloat16, device=device) * 0.02
    dy = torch.randn_like(hidden) * 0.1
    main = torch.npu.current_stream()
    other = torch.npu.Stream(device=device)

    def run(policy):
        experts = [
            MegaMoeExperts(
                hidden_size=h, intermediate_size=i, num_experts=home * ranks, top_k=2,
                ep_group=dist.group.WORLD,
                tuning_options=MegaMoeTuningOptions(expert_placement=SkewPlanner(), replica_bank=policy),
                numerics_options=MegaMoeNumericsOptions(partial_weight_grad_dtype="float32"),
            )
            for _ in range(2)
        ]
        layers = [
            tuple(v.detach().requires_grad_() for v in (hidden, probs, w13, w2)),
            tuple(v.detach().requires_grad_() for v in (
                hidden * 0.5, probs, w13.transpose(1, 2).contiguous() * 0.7,
                w2.transpose(1, 2).contiguous() * 1.4,
            )),
        ]
        first = _call_experts(experts[0], ids, *layers[0])
        other.wait_stream(main)
        with torch.npu.stream(other):
            second = _call_experts(experts[1], ids, *layers[1])
            second.backward(dy)
        main.wait_stream(other)
        first.backward(dy, retain_graph=True)
        retained = tuple(leaf.grad.detach().clone() for leaf in layers[0])
        for leaf in layers[0]:
            leaf.grad = None
        first.backward(dy)
        assert all(torch.equal(leaf.grad, saved) for leaf, saved in zip(layers[0], retained))

        observed = tuple((output.detach(), *(leaf.grad for leaf in layer))
                         for output, layer in zip((first, second), layers))
        owners = [expert._resource_group.resources.placement for expert in experts]  # pylint: disable=protected-access
        first_pool, second_pool = (owner.pool for owner in owners)
        assert first_pool is not None and second_pool is not None
        assert first_pool is second_pool if policy == "shared" else first_pool is not second_pool
        pools = {id(pool): pool for pool in (first_pool, second_pool)}
        evidence = {
            "pools": len(pools),
            "bank_bytes": sum(pool.allocation_bytes for pool in pools.values()),
            "backward_resident_hits": sum(pool.statistics.get("fused_backward_resident_hits", 0)
                                          for pool in pools.values()),
            "backward_weight_bytes": sum(pool.statistics.get("fused_backward_weight_bytes", 0)
                                         for pool in pools.values()),
        }
        del first, second
        gc.collect()
        experts[0].close()
        assert not second_pool.closed
        experts[1].close()
        assert all(pool.closed for pool in pools.values())
        return observed, evidence

    dedicated, dedicated_report = run("dedicated")
    shared, shared_report = run("shared")
    assert all(torch.equal(a, b) for each_a, each_b in zip(dedicated, shared) for a, b in zip(each_a, each_b))
    assert shared_report["bank_bytes"] < dedicated_report["bank_bytes"]
    return {"shared": shared_report, "dedicated": dedicated_report, "bitwise": True}


def _configurations(ranks):
    """Allow one graph-lifecycle case to be rerun without changing its checks."""
    if ranks == 4:
        return [("EIO", True, "float32", 128, "multiple_guests")]
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


def test_placement_edge_cases() -> None:
    """Run every requested geometry and graph lifecycle on the assigned NPU ranks."""
    rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(rank)
    dist.init_process_group("hccl")
    device = torch.device("npu", rank)
    records = []
    try:
        torch.manual_seed(91 + rank)
        ranks = dist.get_world_size()
        tokens, h, i, local_experts, k = (257, 2048, 1024, 6, 2) if ranks == 4 else (257, 256, 128, 4, 2)
        values = (
            torch.randn(tokens, h, device=device, dtype=torch.bfloat16) * 0.2,
            torch.rand(tokens, k, device=device).softmax(-1),
            torch.randn(local_experts, h, 2 * i, device=device, dtype=torch.bfloat16) * 0.02,
            torch.randn(local_experts, i, h, device=device, dtype=torch.bfloat16) * 0.02,
        )
        ids = torch.arange(k, device=device, dtype=torch.int32).expand(tokens, k).contiguous()
        if ranks == 4:
            ids = torch.arange(tokens*k, device=device, dtype=torch.int32).view(tokens, k) % local_experts
        dy = torch.randn_like(values[0]) * 0.1
        configs = _configurations(ranks)
        for number, config in enumerate(configs):
            layout, before, dw, split, *modes = config
            mode = next(iter(modes), "ordinary")
            reference = None
            for enabled in (False, True):
                planner = SkewPlanner() if enabled else None
                op = MegaMoeExperts(
                    hidden_size=h,
                    intermediate_size=i,
                    num_experts=local_experts * ranks,
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
                    tuning_options=MegaMoeTuningOptions(
                        dispatch_split=split, combine_split=split, expert_placement=planner,
                    ),
                )
                leaves = [v.detach().clone().requires_grad_() for v in values]
                if layout == "EOI":
                    leaves[2:] = [v.transpose(1, 2).contiguous().detach().requires_grad_() for v in leaves[2:]]
                elif mode in ("inference_first", "reentrant"):
                    leaves[2:] = [v.transpose(1, 2).contiguous().transpose(1, 2).detach().requires_grad_()
                                  for v in leaves[2:]]
                    assert all(not value.is_contiguous() for value in leaves[2:])
                if mode == "empty_source" and rank == 0:
                    leaves[:2] = [v[:0].detach().requires_grad_() for v in leaves[:2]]
                local_ids = ids[: leaves[0].shape[0]]
                local_dy = dy[: leaves[0].shape[0]]

                call = partial(_call_experts, op, local_ids)

                output, observed = _execute_graph(call, leaves, local_dy, mode)
                if enabled:
                    records.append(_check_joint_result(op, planner, observed, reference, config))
                else:
                    reference = [a.detach().clone() for a in observed]
                del output, observed, leaves
                gc.collect()
                op.close()
                del op
            if rank == 0:
                print("EDGE_PASS", number, config, flush=True)
        output_dir = Path(os.environ["EDGE_OUT"])
        output_dir.mkdir(exist_ok=True, parents=True)
        if ranks == 2:
            shared_bank = _check_shared_bank(device, ranks)
            (output_dir / f"shared-banks-rank{rank}.json").write_text(json.dumps(shared_bank))
        (output_dir / f"rank{rank}.json").write_text(json.dumps(records))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    test_placement_edge_cases()
