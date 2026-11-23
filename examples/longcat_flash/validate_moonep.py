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
"""Atlas A3 validation and benchmark script for LongcatFlashExperts + MoonEP (HotReplica)."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import sys
from typing import Any

import torch
import torch.distributed as dist

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.longcat_flash.native_reference import NativeLongcatExperts
from examples.longcat_flash.validate import (
    benchmark,
    compare,
    execute,
    inputs,
    routes,
    values,
)
from hyper_multicore import BasePlanner, MegaMoePlacementOptions, shmem
from hyper_multicore.longcat import LongcatFlashExperts
from hyper_multicore.modules.base_planner import ExpertPlacement, PlacementRequest
from hyper_multicore.modules.hot_replica.placement import build_placement
from hyper_multicore.modules.hot_replica.planner import HotReplicaPlanner


class SkewPlanner(BasePlanner):
    """Deterministic test planner that borrows experts whenever load exceeds balanced average."""

    def __init__(self, *, source_token_bound: int | None = None) -> None:
        """Bind optional per-rank training token bound."""
        super().__init__(source_token_bound=source_token_bound)

    @torch.no_grad()
    def plan(self, request: PlacementRequest) -> ExpertPlacement | None:
        """Trigger placement whenever max rank load exceeds the uniform average."""
        ranks = request.counts_by_source.shape[0]
        target = max(1, (request.total_rows + ranks - 1) // ranks)
        if request.maximum_received <= target:
            return None
        return build_placement(
            request.hidden,
            request.ids,
            request.counts_by_source,
            target,
            request.rank,
            source_inverse=request.source_inverse,
        )


def parse_args() -> argparse.Namespace:
    """Parse CLI options for MoonEP + LongcatFlashExperts validation on Atlas A3."""
    parser = argparse.ArgumentParser(
        description="Validate and benchmark LongcatFlashExperts with MoonEP (HotReplica) on Atlas A3."
    )
    parser.add_argument("--check", choices=("accuracy", "benchmark"), default="accuracy")
    parser.add_argument("--planner", choices=("hot_replica", "skew"), default="hot_replica")
    parser.add_argument("--source-token-bound", type=int, default=None)
    parser.add_argument("--swiglu-precision", choices=("fused", "hf"), default="fused")
    parser.add_argument("--exercise-edge-routes", action="store_true",
                        help="Cycle normal, hot, empty_source, and all-identity steps.")
    parser.add_argument("--ep", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--initial-token-capacity", type=int, default=None)
    parser.add_argument("--hidden", type=int, default=6144)
    parser.add_argument("--intermediate", type=int, default=2048)
    parser.add_argument("--real", type=int, default=None)
    parser.add_argument("--zero", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--no-gate-up-padded", dest="gate_up_padded", action="store_false", default=True)
    parser.add_argument("--route-fixture", choices=("router", "hot"), default="hot")
    parser.add_argument("--updates", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--compare-baseline", action="store_true",
                        help="In benchmark mode, also run baseline without MoonEP for speedup comparison.")
    parser.add_argument("--strict", action="store_true",
                        help="Require exact 0.0 error on all tensors (note: single-expert splits may have BF16 dW rounding).")
    parser.add_argument("--dw-rel-l2-tol", type=float, default=1e-2,
                        help="Relative L2 tolerance for dW1/dW2 when MoonEP splits an expert across ranks.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.real is None:
        args.real = 8 * args.ep
    if args.zero is None:
        args.zero = 4 * args.ep
    return args


def build_planner(args: argparse.Namespace) -> BasePlanner:
    """Create the requested MoonEP placement planner."""
    if args.planner == "skew":
        return SkewPlanner(source_token_bound=args.source_token_bound)
    return HotReplicaPlanner(source_token_bound=args.source_token_bound)


def build_executor(args: argparse.Namespace, *, enable_moonep: bool) -> LongcatFlashExperts:
    """Construct LongcatFlashExperts with or without MoonEP adaptive placement."""
    placement = (
        MegaMoePlacementOptions("adaptive", build_planner(args))
        if enable_moonep
        else None
    )
    return LongcatFlashExperts(
        hidden_size=args.hidden,
        intermediate_size=args.intermediate,
        num_experts=args.real,
        zero_experts=args.zero,
        top_k=args.top_k,
        ep_size=args.ep,
        ep_group=dist.group.WORLD,
        dispatch_mode="push",
        swiglu_precision=args.swiglu_precision,
        gate_up_padded=args.gate_up_padded,
        initial_token_capacity=args.initial_token_capacity,
        expert_placement=placement,
    )


def _step_passed(comparison: dict[str, Any], *, strict: bool, dw_rel_l2_tol: float) -> bool:
    """Check exact alignment on activations/probs and exact or bounded alignment on split dW."""
    for name, metric in comparison.items():
        if metric.get("both_unused", False):
            continue
        if strict or name in ("output", "input_gradient", "probability_gradient"):
            if not metric.get("strict_pass", False):
                return False
        else:
            if not metric.get("strict_pass", False) and float(metric.get("rel_l2", float("inf"))) > dw_rel_l2_tol:
                return False
    return True


def _placement_summary(layer: LongcatFlashExperts) -> dict[str, Any] | None:
    """Summarize the last MoonEP placement selected on this rank."""
    placement = layer._last_placement  # pylint: disable=protected-access
    if placement is None:
        return None
    return {
        "borrowed_capacity": placement.borrowed_capacity,
        "receive_capacity": placement.receive_capacity,
        "local_receive_capacity": placement.local_receive_capacity,
        "transfer_experts": placement.transfer_experts,
        "published_slots": placement.published_slots,
        "virtual_experts": int(placement.counts_by_source.shape[1]),
    }


def _placement_stats(layer: LongcatFlashExperts) -> dict[str, int] | None:
    """Read cumulative PlacementResources statistics if initialized."""
    group = getattr(layer, "_resource_group", None)
    resources = getattr(group, "resources", None) if group is not None else None
    placement_res = getattr(resources, "placement", None) if resources is not None else None
    return dict(placement_res.statistics) if placement_res is not None else None


def run_accuracy(
    args: argparse.Namespace,
    data: dict[str, torch.Tensor],
    layer: LongcatFlashExperts,
    rank: int,
) -> dict[str, Any]:
    """Validate LongcatFlashExperts + MoonEP against NativeLongcatExperts."""
    reference = NativeLongcatExperts(
        args.real,
        args.zero,
        args.ep,
        dist.group.WORLD,
        args.swiglu_precision,
        gate_up_padded=args.gate_up_padded,
    )
    other = {
        key: value.detach().clone().requires_grad_(True) if key in ("gate", "down") else value
        for key, value in data.items()
    }
    records = []
    for step in range(args.updates):
        kind, ids = routes(args, data, step, rank)
        ref_output = execute(reference, data, ids, backward=True)
        expected = {
            name: None if value is None else value.detach().cpu()
            for name, value in values(data, ref_output).items()
        }
        out = execute(layer, other, ids, backward=True)
        comparison = compare(values(other, out), expected)
        ok = _step_passed(comparison, strict=args.strict, dw_rel_l2_tol=args.dw_rel_l2_tol)
        state = torch.tensor([1 if ok else 0], dtype=torch.int32, device="npu")
        dist.all_reduce(state, op=dist.ReduceOp.MIN)
        records.append({
            "step": step,
            "routes": kind,
            "placement": _placement_summary(layer),
            "comparison": comparison,
            "all_ranks_pass": bool(state.item()),
        })
        if not records[-1]["all_ranks_pass"]:
            break
    return {
        "records": records,
        "placement_statistics": _placement_stats(layer),
        "all_steps_pass": len(records) == args.updates and all(r["all_ranks_pass"] for r in records),
    }


def main() -> None:
    """Run distributed MoonEP + LongcatFlashExperts validation or benchmark on Atlas A3."""
    args = parse_args()
    import torch_npu  # noqa: F401  # pylint: disable=import-outside-toplevel,unused-import

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    world_size = int(os.environ.get("WORLD_SIZE", str(args.ep)))
    if world_size != args.ep:
        raise ValueError(f"WORLD_SIZE ({world_size}) must equal --ep ({args.ep}).")
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size)

    try:
        data = inputs(args, rank)
        layer = build_executor(args, enable_moonep=True)
        try:
            if args.check == "accuracy":
                result: dict[str, Any] = run_accuracy(args, data, layer, rank)
            else:
                moonep_bench = benchmark(args, data, layer)
                result = {
                    "moonep": moonep_bench,
                    "placement_statistics": _placement_stats(layer),
                }
        finally:
            layer.close()

        if args.check == "benchmark" and args.compare_baseline:
            baseline_layer = build_executor(args, enable_moonep=False)
            try:
                base_bench = benchmark(args, data, baseline_layer)
                result["baseline"] = base_bench
                result["speedup"] = (
                    base_bench["median_rank_max_ms"] / result["moonep"]["median_rank_max_ms"]
                )
            finally:
                baseline_layer.close()

        shmem_after_close = dataclasses.asdict(shmem.status())
        report = {
            "rank": rank,
            "check": args.check,
            "planner": args.planner,
            "source_token_bound": args.source_token_bound,
            "swiglu_precision": args.swiglu_precision,
            "gate_up_padded": args.gate_up_padded,
            "shape": {
                "ep": args.ep,
                "tokens": args.tokens,
                "hidden": args.hidden,
                "intermediate": args.intermediate,
                "real": args.real,
                "zero": args.zero,
                "top_k": args.top_k,
            },
            "result": result,
            "shmem_after_close": shmem_after_close,
        }
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / f"rank{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
        if rank == 0:
            stats = result["placement_statistics"]
            if args.check == "accuracy":
                passed = result["all_steps_pass"]
                state = shmem_after_close["state"]
                print(f"[MoonEP accuracy] planner={args.planner} all_steps_pass={passed} stats={stats} shmem={state}")
            else:
                med = result["moonep"]["median_rank_max_ms"]
                if "baseline" in result:
                    bmed = result["baseline"]["median_rank_max_ms"]
                    speedup = result["speedup"]
                    print(
                        f"[MoonEP benchmark] planner={args.planner} baseline={bmed:.3f}ms "
                        f"moonep={med:.3f}ms speedup={speedup:.2f}x stats={stats}"
                    )
                else:
                    print(f"[MoonEP benchmark] planner={args.planner} moonep={med:.3f}ms stats={stats}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
