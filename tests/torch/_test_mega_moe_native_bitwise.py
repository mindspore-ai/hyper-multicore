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
"""Bitwise Native EP/MegaMoe acceptance at the routed-expert boundary on A3."""

from dataclasses import asdict, dataclass
import os

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401  # pylint: disable=unused-import

from examples.moe.experts import NativeExperts
from tests.common.multicore._mega_moe_utils import write_evidence
from tests.common.multicore.owned_mega_moe import OwnedMegaMoeExperts

_STEPS = 4


@dataclass(frozen=True)
class _Shape:
    """Keep the validated A3 geometry explicit rather than implying arbitrary-shape parity."""

    hidden_size: int = 5120
    intermediate_size: int = 1792
    num_experts: int = 64
    top_k: int = 8
    local_num_tokens: int = 4096
    ep_size: int = 8


def _disable_determinism() -> dict:
    """Disable Torch/NPU determinism and record the available runtime observation."""
    assert os.getenv("HCCL_DETERMINISTIC") is None, "launcher must unset HCCL_DETERMINISTIC"
    assert os.getenv("HCCL_DETERMINISTIC_OP_EXPANSION") is None
    torch.use_deterministic_algorithms(False)
    # Older PTA branches expose only the Torch switch; do not invent a readback for them.
    if hasattr(torch.npu, "set_deterministic_level"):
        torch.npu.set_deterministic_level(0)
    get_level = getattr(torch.npu, "_get_deterministic_level", None)
    level = get_level() if get_level is not None else None
    assert not torch.are_deterministic_algorithms_enabled()
    assert level in (None, 0), f"NPU determinism was not disabled: level={level}"
    return {"torch_deterministic": False, "npu_deterministic_level": level,
            "hccl_deterministic_env": os.getenv("HCCL_DETERMINISTIC")}


def _expert_tensors(layer: torch.nn.Module, *, gradients: bool = False) -> tuple:
    """Map Native w1/w3/w2 into the same packed EIO axes without changing its math."""
    if isinstance(layer, NativeExperts):
        weights = (layer.module.w1, layer.module.w3, layer.module.w2)
        gate, up, down = (weight.grad if gradients else weight for weight in weights)
        assert all(value is not None for value in (gate, up, down)), "missing Native expert gradients"
        return torch.cat((gate, up), dim=1).transpose(1, 2).contiguous(), down.transpose(1, 2).contiguous()
    weights = (layer.gate_up_weight, layer.down_weight)
    return tuple(weight.grad if gradients else weight for weight in weights)


def _make_layers(shape: _Shape, device: torch.device) -> tuple:
    """Copy identical local shards into the actual Native EP and the public executor."""
    torch.manual_seed(20261114 + dist.get_rank())
    mega = OwnedMegaMoeExperts(
        hidden_size=shape.hidden_size, intermediate_size=shape.intermediate_size,
        num_experts=shape.num_experts, top_k=shape.top_k, ep_group=dist.group.WORLD,
        dispatch_mode=os.environ["HP_MEGA_MOE_DISPATCH_MODE"],
    ).to(device=device, dtype=torch.bfloat16)
    generator = torch.Generator().manual_seed(8114 + dist.get_rank())
    with torch.no_grad():
        for parameter in mega.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.01)
    native = NativeExperts(shape, mega, dist.group.WORLD, device)
    return native, mega


def _make_data(shape: _Shape, step: int, device: torch.device) -> tuple:
    """Supply fixed BF16 inputs/upstream and independent FP32 routing leaves, without attention."""
    generator = torch.Generator().manual_seed(41000 + 31 * step + dist.get_rank())
    hidden = (torch.randn(shape.local_num_tokens, shape.hidden_size, generator=generator) * 0.2).to(
        device=device, dtype=torch.bfloat16,
    )
    ids = torch.rand(shape.local_num_tokens, shape.num_experts, generator=generator).topk(shape.top_k, dim=-1)
    probabilities = torch.rand(shape.local_num_tokens, shape.top_k, generator=generator)
    probabilities = (probabilities / probabilities.sum(dim=-1, keepdim=True)).to(device)
    upstream = (torch.randn(shape.local_num_tokens, shape.hidden_size, generator=generator) * 0.1).to(
        device=device, dtype=torch.bfloat16,
    )
    return hidden, ids.indices.to(device=device, dtype=torch.int32), probabilities, upstream


def _snapshot(layer: torch.nn.Module, data: tuple) -> dict:
    """Retain every output/gradient element on CPU for byte comparisons, including signed zero."""
    layer.zero_grad(set_to_none=True)
    hidden, ids, probabilities, upstream = data
    hidden = hidden.detach().clone().requires_grad_(True)
    probabilities = probabilities.detach().clone().requires_grad_(True)
    output = layer(hidden, ids, probabilities)
    output.backward(upstream)
    torch.npu.synchronize()
    gate_up_grad, down_grad = _expert_tensors(layer, gradients=True)
    tensors = {"output": output, "input_grad": hidden.grad, "route_weight_grad": probabilities.grad,
               "gate_up_weight_grad": gate_up_grad, "down_weight_grad": down_grad}
    assert all(tensor is not None for tensor in tensors.values()), "missing routed-expert gradient"
    return {name: tensor.detach().cpu().contiguous() for name, tensor in tensors.items()}


def _tensor_failure(name: str, actual: torch.Tensor, expected: torch.Tensor) -> str | None:
    """Compare finite tensors by their bytes; a numerical tolerance cannot hide a regression."""
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return (f"{name}: metadata mismatch: actual={actual.shape}/{actual.dtype}, "
                f"expected={expected.shape}/{expected.dtype}")
    if not bool(torch.isfinite(actual).all()) or not bool(torch.isfinite(expected).all()):
        return f"{name}: nonfinite values"
    actual_bits = actual.contiguous().reshape(-1).view(torch.uint8)
    expected_bits = expected.contiguous().reshape(-1).view(torch.uint8)
    if not torch.equal(actual_bits, expected_bits):
        differences = int(torch.count_nonzero(actual_bits != expected_bits))
        return f"{name}: {differences}/{actual_bits.numel()} bytes differ ({actual.dtype}, shape={tuple(actual.shape)})"
    return None


def _assert_bitwise(phase: str, actual: dict, expected: dict) -> dict:
    """Converge failures before raising, so every rank stops before the next MoE collective."""
    assert actual.keys() == expected.keys()
    local_failures = [message for name in expected
                      if (message := _tensor_failure(name, actual[name], expected[name])) is not None]
    failures = [None] * dist.get_world_size()
    dist.all_gather_object(failures, local_failures)
    assert not any(failures), f"{phase}: bitwise comparison failed by rank: {dict(enumerate(failures))}"
    return {"phase": phase, "tensors": list(expected), "bitwise": True, "finite": True}


def _make_optimizer(layer: torch.nn.Module) -> tuple:
    """Use independent FP32 packed masters and the same ordinary AdamW on both sides."""
    masters = [torch.nn.Parameter(weight.detach().float().contiguous().clone()) for weight in _expert_tensors(layer)]
    optimizer = torch.optim.AdamW(masters, lr=1e-4, betas=(0.9, 0.95), eps=1e-8,
                                 weight_decay=0.01, foreach=False, fused=False)
    return masters, optimizer


@torch.no_grad()
def _update(layer: torch.nn.Module, state: tuple, snapshot: dict) -> dict:
    """Update masters, write BF16 model weights, and expose all moments for comparison."""
    masters, optimizer = state
    for master, name in zip(masters, ("gate_up_weight_grad", "down_weight_grad")):
        master.grad = snapshot[name].to(device=master.device, dtype=torch.float32)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    gate_up, down = (master.to(torch.bfloat16) for master in masters)
    if isinstance(layer, NativeExperts):
        gate, up = gate_up.split(layer.module.w1.shape[1], dim=-1)
        layer.module.w1.copy_(gate.transpose(1, 2))
        layer.module.w3.copy_(up.transpose(1, 2))
        layer.module.w2.copy_(down.transpose(1, 2))
    else:
        layer.gate_up_weight.copy_(gate_up)
        layer.down_weight.copy_(down)
    weights = _expert_tensors(layer)
    result = {f"bf16_weight_{index}": weight.detach().cpu().contiguous() for index, weight in enumerate(weights)}
    for index, master in enumerate(masters):
        result[f"fp32_master_{index}"] = master.detach().cpu()
        for name, value in optimizer.state[master].items():
            result[f"{name}_{index}"] = value.detach().cpu()
    return result


def _run_steps(native: NativeExperts, mega: OwnedMegaMoeExperts, shape: _Shape, device: torch.device) -> list:
    """Check Native A/A before comparing Mega, then check complete AdamW state each step."""
    native_state, mega_state = _make_optimizer(native), _make_optimizer(mega)
    results = []
    for step in range(_STEPS):
        data = _make_data(shape, step, device)
        reference = _snapshot(native, data)
        repeat = _snapshot(native, data)
        control = _assert_bitwise(f"step={step} Native A/A", repeat, reference)
        del repeat
        actual = _snapshot(mega, data)
        precision = _assert_bitwise(f"step={step} Native/Mega", actual, reference)
        expected_update = _update(native, native_state, reference)
        actual_update = _update(mega, mega_state, actual)
        update = _assert_bitwise(f"step={step} AdamW", actual_update, expected_update)
        results.append({"step": step, "native_repeat": control, "native_mega": precision, "optimizer": update})
        print(f"rank={dist.get_rank()} step={step}: Native A/A, Mega F/B and AdamW bitwise", flush=True)
    return results


def test_mega_moe_native_bitwise() -> None:
    """Require pure BF16 routed experts and AdamW to align with the actual Native EP on every rank."""
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    determinism = _disable_determinism()
    dist.init_process_group("hccl")
    try:
        shape = _Shape()
        assert dist.get_world_size() == shape.ep_size
        device = torch.device("npu", local_rank)
        native, mega = _make_layers(shape, device)
        try:
            steps = _run_steps(native, mega, shape, device)
        finally:
            mega.close()
        write_evidence({
            "rank": dist.get_rank(), "shape": asdict(shape), "dtype": "bfloat16",
            "dispatch_mode": os.environ["HP_MEGA_MOE_DISPATCH_MODE"], "determinism": determinism,
            "baseline": "examples.moe.experts.NativeExperts", "boundary": "routed experts only",
            "attention": False, "norm": False, "shared_experts": False, "b": 0, "swiglu_limit": None,
            "optimizer": "FP32 master AdamW", "steps": steps, "bitwise": True,
        })
    finally:
        dist.destroy_process_group()
