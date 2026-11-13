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

"""Single-NPU dense Transformer benchmark comparing only mHC implementations."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch_npu  # Registers the optional NPU backend used by this executable example.

from examples._optimizer import Float16OptimizerWithFloat16Params
from examples._training import _compare_tensor_maps, build_optimizer
from examples.mhc.model import MhcTransformerConfig, MhcTransformerModel
from hyper_multicore.modules.mega_mhc_common import validate_mega_mhc_token_count
from hyper_multicore._core_spec import get_device_core_spec
from hyper_multicore.modules.mega_mhc_common import validate_mega_mhc_core_spec

_RTOL = 2e-2
_ATOL = 2e-3
_DTYPE = torch.bfloat16


@dataclass
class _Workload:
    """One local dense model, independent optimizer state and common input batch."""

    model: MhcTransformerModel
    optimizer: Float16OptimizerWithFloat16Params
    input_ids: torch.Tensor
    labels: torch.Tensor
    device: torch.device


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse single-device model dimensions and matched training settings.

    Args:
        argv: Optional arguments; None reads the executable command line.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--intermediate-size", type=int, default=2048)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--measured-steps", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--output", default="output/mhc_transformer_benchmark.json")
    args = parser.parse_args(argv)
    for name in ("tokens", "hidden_size", "intermediate_size", "num_layers", "warmup_steps", "measured_steps"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.device_index < 0:
        raise ValueError("device_index must be nonnegative")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    return args


def _tensor_state(model: MhcTransformerModel, *, gradients: bool) -> dict[str, torch.Tensor]:
    """Read every required dense parameter or gradient without layout conversion."""
    result = {}
    for name, parameter in model.named_parameters():
        if gradients and not parameter.requires_grad:
            continue
        value = parameter.grad if gradients else parameter
        if value is None:
            raise RuntimeError(f"missing dense model gradient for {name}")
        result[name] = value.detach()
    return result


def _optimizer_step(workload: _Workload, *, capture_gradients: bool = False) -> tuple:
    """Run a complete local forward/backward/AdamW step without communication."""
    workload.optimizer.zero_grad(set_to_none=True)
    output = workload.model(workload.input_ids, labels=workload.labels)
    loss = output["loss"]
    if loss is None:
        raise RuntimeError("dense model did not return a training loss")
    loss.backward()
    gradients = ({name: value.clone() for name, value in _tensor_state(workload.model, gradients=True).items()}
                 if capture_gradients else None)
    workload.optimizer.step()
    return loss.detach(), output["logits"].detach(), gradients


def _compare_mhc_component(
    models: dict[str, MhcTransformerModel], config: MhcTransformerConfig, seed: int, device: torch.device,
) -> dict[str, Any]:
    """Check fused mHC outputs and all input/parameter gradients on identical inputs."""
    hidden = config.hidden_size
    tokens = config.max_seq_len
    generator = torch.Generator(device="cpu").manual_seed(seed + 20_000 + 0)
    shapes = ((1, tokens, hidden), (1, tokens, 4, hidden), (1, tokens, 4),
              (1, tokens, 4), (1, tokens, 4, 4))
    dtypes = (_DTYPE, _DTYPE, torch.float32, torch.float32, torch.float32)
    inputs = tuple(torch.randn(shape, generator=generator).to(device=device, dtype=dtype)
                   for shape, dtype in zip(shapes, dtypes))
    probes = tuple(torch.randn(shape, generator=generator).to(device=device, dtype=dtype) / tokens
                   for shape, dtype in zip((shapes[1], shapes[2], shapes[3], shapes[4], shapes[0]),
                                           (_DTYPE, torch.float32, torch.float32, torch.float32, _DTYPE)))
    forwards, gradients = {}, {}
    for backend in ("reference", "common", "mega"):
        boundary = models["common" if backend == "reference" else backend].layers[0].attention_mhc
        state = tuple(tensor.detach().clone().requires_grad_() for tensor in inputs)
        outputs = boundary.reference_forward(*state) if backend == "reference" else boundary(*state)
        values = torch.autograd.grad(outputs, (*state, *boundary.parameters()), grad_outputs=probes)
        forwards[backend] = {str(index): output.detach() for index, output in enumerate(outputs)}
        gradients[backend] = {str(index): value.detach() for index, value in enumerate(values)}
    comparisons = {}
    for backend in ("common", "mega"):
        comparisons[f"{backend}_forward"] = _compare_tensor_maps(
            f"mHC {backend} forward", forwards["reference"], forwards[backend],
            device, rtol=_RTOL, atol=_ATOL,
        )
        comparisons[f"{backend}_gradients"] = _compare_tensor_maps(
            f"mHC {backend} gradients", gradients["reference"], gradients[backend],
            device, rtol=_RTOL, atol=_ATOL,
        )
    return comparisons


def _validate_models(workloads: dict[str, _Workload]) -> dict[str, Any]:
    """Compare complete dense-model outputs, all gradients and updated parameters."""
    device = workloads["common"].device
    models = {name: workload.model for name, workload in workloads.items()}
    compare = _compare_tensor_maps
    initial = compare("initial parameters", *[_tensor_state(models[name], gradients=False)
                                             for name in ("common", "mega")], device, rtol=0, atol=0)
    values = {}
    for name, workload in workloads.items():
        before = workload.model.lm_head.weight.detach().clone()
        loss, logits, gradients = _optimizer_step(workload, capture_gradients=True)
        if not bool(torch.isfinite(loss).all().item()) or any(
                not bool(torch.isfinite(value).all().item()) for value in gradients.values()):
            raise RuntimeError(f"{name} dense training step produced nonfinite values")
        if torch.equal(before, workload.model.lm_head.weight):
            raise RuntimeError(f"{name} dense optimizer did not update the vocabulary head")
        values[name] = ({"loss": loss, "logits": logits}, gradients)
    return {
        "scope": "full_dense_model_with_selected_mhc",
        "initial_parameters": initial,
        "forward": compare("forward", values["common"][0], values["mega"][0], device, rtol=_RTOL, atol=_ATOL),
        "gradients": compare("gradients", values["common"][1], values["mega"][1], device, rtol=_RTOL, atol=_ATOL),
        "updated_parameters": compare("updated parameters", *[_tensor_state(models[name], gradients=False)
                                                             for name in ("common", "mega")],
                                      device, rtol=_RTOL, atol=_ATOL),
    }


def _timed_step(workload: _Workload) -> float:
    """Measure a complete local optimizer step with explicit device boundaries."""
    torch.npu.synchronize(workload.device)
    started = time.perf_counter()
    _optimizer_step(workload)
    torch.npu.synchronize(workload.device)
    return (time.perf_counter() - started) * 1000


def _measure(workload: _Workload, args: argparse.Namespace) -> dict[str, Any]:
    """Exclude validation and warmup from the steady-state optimizer-step samples."""
    for _ in range(args.warmup_steps - 1):
        _timed_step(workload)
    values = [_timed_step(workload) for _ in range(args.measured_steps)]
    return {"steady_state_optimizer_step_ms": {
        "median": statistics.median(values), "minimum": min(values), "maximum": max(values), "samples": values,
    }}


def _build_workloads(args: argparse.Namespace) -> tuple[MhcTransformerConfig, dict[str, _Workload]]:
    """Build identical local models without initializing a process group or SHMEM."""
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or (dist.is_initialized() and dist.get_world_size() != 1):
        raise ValueError("the dense mHC benchmark uses one device; launch with python, not multi-rank torchrun")
    torch.npu.set_device(args.device_index)
    device = torch.device("npu", args.device_index)
    core_spec = get_device_core_spec(args.device_index)
    validate_mega_mhc_core_spec(core_spec)
    vector_cores = core_spec.num_vector_cores
    validate_mega_mhc_token_count(args.tokens, vector_cores)
    config = MhcTransformerConfig(hidden_size=args.hidden_size, intermediate_size=args.intermediate_size,
                                  num_layers=args.num_layers, max_seq_len=args.tokens)
    generator = torch.Generator().manual_seed(args.seed + 10000)
    inputs = torch.randint(config.vocab_size, (1, config.max_seq_len), generator=generator).to(device)
    result = {}
    for name in ("common", "mega"):
        torch.manual_seed(args.seed)
        model = MhcTransformerModel(config, use_mega_mhc=name == "mega").to(device=device, dtype=_DTYPE)
        result[name] = _Workload(model, build_optimizer(model, learning_rate=args.learning_rate,
                                                       weight_decay=args.weight_decay), inputs, inputs.clone(), device)
    return config, result


def main(argv: list[str] | None = None) -> int:
    """Validate the mHC boundary and complete dense training, then time both implementations.

    Args:
        argv: Optional command-line arguments.
    """
    args = parse_args(argv)
    config, workloads = _build_workloads(args)
    models = {name: workload.model for name, workload in workloads.items()}
    component = _compare_mhc_component(models, config, args.seed, workloads["common"].device)
    accuracy = _validate_models(workloads)
    measurements = {name: _measure(workload, args) for name, workload in workloads.items()}
    medians = {name: value["steady_state_optimizer_step_ms"]["median"] for name, value in measurements.items()}
    report = {
        "architecture": "dense_transformer_with_mhc", "topology": {"devices": 1, "distributed": False},
        "shape": vars(config), "optimizer": {"name": "AdamW", "fp32_main_parameters": True},
        "mhc_component": component, "accuracy": accuracy, "backends": measurements,
        "warmup_steps": args.warmup_steps, "measured_steps": args.measured_steps,
        "comparison": {"metric": "complete local optimizer-step milliseconds", "order": ["common", "mega"],
                       "mega_over_common": medians["mega"] / medians["common"]},
        "runtime": {"torch": str(torch.__version__), "torch_npu": torch_npu.__version__},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
