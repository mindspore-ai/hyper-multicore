"""Device regressions requested by the MegaMoE review."""
import ctypes
import os
from unittest.mock import patch

import torch
import torch.nn.functional as F

from hyper_multicore import MegaMoeExperts
from hyper_multicore.modules.mega_moe import function as function_module
from hyper_multicore.modules.mega_moe import module as module_module
from hyper_multicore.modules.mega_moe.plan import build_mega_moe_plan
from hyper_multicore.scheduler.config import RuntimeConfigC
from tests.common.mark_utils import arg_mark
from tests.common.multicore._test_env import prepare_multicore_test_environment

_LAYOUTS = ("EIO", "EOI")


def _layer(layout: str, **options):
    return MegaMoeExperts(
        hidden_size=128, intermediate_size=512, num_experts=32, top_k=2,
        ep_size=1, apply_probs_before_fc2=True, expert_weight_layout=layout, **options,
    )


def _header(plan) -> RuntimeConfigC:
    raw = plan.fwd_runtime_config.detach().cpu().numpy().tobytes()
    return RuntimeConfigC.from_buffer_copy(raw[:ctypes.sizeof(RuntimeConfigC)])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="onecard", essential_mark="essential")
def test_compact_header_uses_normal_ready_queue() -> None:
    """Compact graphs publish ready_queue=1 in NORMAL mode for both layouts and clamp modes."""
    prepare_multicore_test_environment()
    torch.npu.set_device(0)
    device = torch.device("npu", 0)
    for layout in _LAYOUTS:
        for clamp in (False, True):
            options = {"clamp_after_activation": clamp} if clamp else {}
            if clamp:
                options["swiglu_limit"] = 0.5
            layer = _layer(layout, **options).to(device=device, dtype=torch.bfloat16)
            try:
                tokens = 393216
                x = torch.randn(tokens, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
                ids = torch.randint(0, 32, (tokens, 2), device=device, dtype=torch.int32)
                probs = torch.softmax(torch.randn(tokens, 2, device=device, dtype=torch.float32), dim=-1)
                out = layer(x, ids, probs)
                resources = layer._get_execution_resources(x.reshape(-1, 128))  # pylint: disable=protected-access
                header = _header(resources.plan_for_tokens(tokens, device))
                assert int(header.vector_ready_queue) == 1
                assert int(header.diagnostic_mode) == 0
                out.float().square().mean().backward()
                assert x.grad is not None and torch.isfinite(x.grad).all()
                assert torch.count_nonzero(x.grad).item() > 0
            finally:
                layer.close()


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="onecard", essential_mark="essential")
def test_fp32_native_weight_gradients_before_autograd_cast() -> None:
    """Capture FP32 native dW for both layouts and keep cold-expert rows exactly zero."""
    prepare_multicore_test_environment()
    torch.npu.set_device(0)
    device = torch.device("npu", 0)
    for layout in _LAYOUTS:
        layer = _layer(layout).to(device=device, dtype=torch.bfloat16)
        try:
            tokens = 4096
            x = torch.randn(tokens, 128, device=device, dtype=torch.bfloat16)
            # Route every row to expert 0 so every other expert is cold.
            ids = torch.zeros((tokens, 2), device=device, dtype=torch.int32)
            ids[:, 1] = 1
            probs = torch.softmax(torch.randn(tokens, 2, device=device, dtype=torch.float32), dim=-1)

            def fp32_plan(spec, dev, **kwargs):
                return build_mega_moe_plan(spec, dev, weight_grad_dtype=torch.float32)

            os.environ["HP_MEGA_MOE_CAPTURE_DW"] = "1"
            function_module._CAPTURED_WEIGHT_GRADS = None  # pylint: disable=protected-access
            try:
                with patch.object(module_module, "build_mega_moe_plan", side_effect=fp32_plan):
                    out = layer(x, ids, probs)
                    out.float().square().mean().backward()
            finally:
                os.environ.pop("HP_MEGA_MOE_CAPTURE_DW", None)
            captured = function_module._CAPTURED_WEIGHT_GRADS  # pylint: disable=protected-access
            assert captured is not None
            assert captured[0].dtype == torch.float32
            assert captured[1].dtype == torch.float32
            assert torch.count_nonzero(captured[0]).item() > 0
            # Experts 0 and 1 are active; every later row must be exactly zero.
            assert torch.count_nonzero(captured[0][2:]).item() == 0
            assert torch.count_nonzero(captured[1][2:]).item() == 0
        finally:
            layer.close()


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="onecard", essential_mark="essential")
def test_reentrant_no_grad_then_grad() -> None:
    """A no-grad forward must not freeze the later grad-enabled invocation."""
    prepare_multicore_test_environment()
    torch.npu.set_device(0)
    device = torch.device("npu", 0)
    layer = _layer("EIO").to(device=device, dtype=torch.bfloat16)
    try:
        tokens = 4096
        x = torch.randn(tokens, 128, device=device, dtype=torch.bfloat16)
        ids = torch.randint(0, 32, (tokens, 2), device=device, dtype=torch.int32)
        probs = torch.softmax(torch.randn(tokens, 2, device=device, dtype=torch.float32), dim=-1)
        with torch.no_grad():
            first = layer(x, ids, probs)
        states = [first.requires_grad]
        second = layer(x, ids, probs)
        states.append(second.requires_grad)
        assert states == [False, True]
        second.float().square().mean().backward()
        assert layer.gate_up_weight.grad is not None
        assert torch.isfinite(layer.gate_up_weight.grad).all()
    finally:
        layer.close()


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="onecard", essential_mark="essential")
def test_fp32_native_dw_matches_cpu_fp32_reference() -> None:
    """Small top-1 expert keeps native FP32 dW equal to a CPU FP32 accumulation."""
    prepare_multicore_test_environment()
    torch.npu.set_device(0)
    device = torch.device("npu", 0)
    layer = MegaMoeExperts(
        hidden_size=128, intermediate_size=64, num_experts=2, top_k=1,
        ep_size=1, expert_weight_layout="EIO",
    ).to(device=device, dtype=torch.bfloat16)
    try:
        tokens = 128
        x = torch.randn(tokens, 128, device=device, dtype=torch.bfloat16)
        ids = torch.zeros((tokens, 1), device=device, dtype=torch.int32)
        probs = torch.ones((tokens, 1), device=device, dtype=torch.float32)

        def fp32_plan(spec, dev, **kwargs):
            return build_mega_moe_plan(spec, dev, weight_grad_dtype=torch.float32)

        os.environ["HP_MEGA_MOE_CAPTURE_DW"] = "1"
        function_module._CAPTURED_WEIGHT_GRADS = None  # pylint: disable=protected-access
        try:
            with patch.object(module_module, "build_mega_moe_plan", side_effect=fp32_plan):
                out = layer(x, ids, probs)
                out.float().square().mean().backward()
        finally:
            os.environ.pop("HP_MEGA_MOE_CAPTURE_DW", None)
        captured = function_module._CAPTURED_WEIGHT_GRADS  # pylint: disable=protected-access
        assert captured is not None

        w1 = layer.gate_up_weight.detach().cpu().requires_grad_()
        w2 = layer.down_weight.detach().cpu().requires_grad_()
        x_cpu = x.cpu()
        projection = x_cpu @ w1[0].transpose(0, 1)
        gate, up = projection.chunk(2, dim=-1)
        activation = F.silu(gate) * up
        reference = activation @ w2[0]
        reference.float().square().mean().backward()
        errors = [
            (captured[0][0].cpu() - w1.grad.float()).abs().max().item(),
            (captured[1][0].cpu() - w2.grad.float()).abs().max().item(),
        ]
        assert errors[0] <= 5e-2, errors
        assert errors[1] <= 5e-2, errors
    finally:
        layer.close()
