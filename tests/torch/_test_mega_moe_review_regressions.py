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

"""Device regressions requested by the MegaMoE review."""
import ctypes
from functools import partial
import os
from typing import Any
from unittest.mock import patch

import torch
from torch.utils.checkpoint import checkpoint

from hyper_multicore.modules.mega_moe import function as function_module
from hyper_multicore.modules.mega_moe import module as module_module
from hyper_multicore.modules.mega_moe.plan import MegaMoePlan, build_mega_moe_plan
from hyper_multicore.modules.mega_moe.spec import MegaMoeSpec
from hyper_multicore.scheduler.config import RuntimeConfigC
from tests.common.mark_utils import arg_mark
from tests.common.multicore._test_env import prepare_multicore_test_environment
from tests.common.multicore.owned_mega_moe import OwnedMegaMoeExperts
from tests.common.multicore.fp32_expert_reference import fp32_expert_reference

_LAYOUTS = ("EIO", "EOI")


def _layer(layout: str, **options: Any) -> OwnedMegaMoeExperts:
    """Own test weights while exercising the public external-weight executor."""
    return OwnedMegaMoeExperts(
        hidden_size=128, intermediate_size=512, num_experts=32, top_k=2,
        ep_group=None, apply_probs_before_fc2=True, expert_weight_layout=layout, **options,
    )


def _header(plan: MegaMoePlan) -> RuntimeConfigC:
    """Read the runtime header uploaded by the selected plan."""
    raw = plan.fwd_runtime_config.detach().cpu().numpy().tobytes()
    return RuntimeConfigC.from_buffer_copy(raw[:ctypes.sizeof(RuntimeConfigC)])


def _checkpoint_forward(
    layer: OwnedMegaMoeExperts, ids: torch.Tensor, states: list[bool],
    tokens: torch.Tensor, probabilities: torch.Tensor,
) -> torch.Tensor:
    """Observe both real checkpoint invocations without capturing loop variables."""
    states.append(torch.is_grad_enabled())
    return layer(tokens, ids, probabilities)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="onecard", essential_mark="essential")
def test_compact_header_uses_normal_ready_queue() -> None:
    """Compact graphs publish ready_queue=1 in NORMAL mode for both layouts and clamp modes."""
    prepare_multicore_test_environment()
    torch.npu.set_device(0)
    device = torch.device("npu", 0)
    for layout in _LAYOUTS:
        for clamp in (False, True):
            options = {"clamp_after_silu": clamp} if clamp else {}
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

            def fp32_plan(spec: MegaMoeSpec, dev: torch.device, **_kwargs: Any) -> MegaMoePlan:
                """Request native FP32 partial weight gradients."""
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
    """Real reentrant replay must retain outputs and reproduce all ordinary gradients."""
    prepare_multicore_test_environment()
    torch.npu.set_device(0)
    device = torch.device("npu", 0)
    for mode in ("push", "pull"):
        for layout in _LAYOUTS:
            layer = _layer(layout, dispatch_mode=mode).to(device=device, dtype=torch.bfloat16)
            try:
                x = torch.randn(128, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
                ids = torch.randint(0, 32, (128, 2), device=device, dtype=torch.int32)
                probs = torch.softmax(torch.randn(128, 2, device=device), dim=-1).requires_grad_()
                ordinary = layer(x, ids, probs)
                ordinary.float().square().mean().backward()
                expected = [value.detach().clone() for value in (
                    ordinary, x.grad, probs.grad, layer.gate_up_weight.grad, layer.down_weight.grad)]
                layer.zero_grad(set_to_none=True)
                replay_x = x.detach().clone().requires_grad_()
                replay_probs = probs.detach().clone().requires_grad_()
                states = []

                forward = partial(_checkpoint_forward, layer, ids, states)
                output = checkpoint(forward, replay_x, replay_probs, use_reentrant=True)
                retained = output.detach().clone()
                output.float().square().mean().backward()
                assert states == [False, True], states
                actual = (output, replay_x.grad, replay_probs.grad,
                          layer.gate_up_weight.grad, layer.down_weight.grad)
                for observed, reference in zip(actual, expected):
                    torch.testing.assert_close(observed, reference, rtol=2e-2, atol=2e-3)
                torch.testing.assert_close(output, retained, rtol=0, atol=0)
            finally:
                layer.close()


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="onecard", essential_mark="essential")
def test_fp32_native_dw_matches_cpu_fp32_reference() -> None:
    """Compare each active expert to rounded BF16 operands with FP32 dW accumulation."""
    prepare_multicore_test_environment()
    torch.npu.set_device(0)
    device = torch.device("npu", 0)
    for layout in _LAYOUTS:
        # Non-square gate/up weights expose accidental EIO transposition.
        layer = OwnedMegaMoeExperts(
            hidden_size=128, intermediate_size=128, num_experts=2, top_k=1,
            ep_group=None, expert_weight_layout=layout,
        ).to(device=device, dtype=torch.bfloat16)
        try:
            with torch.no_grad():
                layer.gate_up_weight.fill_(1 / 16)
                layer.down_weight.fill_(1 / 32)
            x = torch.full((128, 128), 1 / 16, device=device, dtype=torch.bfloat16)
            ids = torch.zeros((128, 1), device=device, dtype=torch.int32)
            probs = torch.ones((128, 1), device=device, dtype=torch.float32)

            def fp32_plan(spec: MegaMoeSpec, dev: torch.device, **_kwargs: Any) -> MegaMoePlan:
                """Request native FP32 partial weight gradients."""
                return build_mega_moe_plan(spec, dev, weight_grad_dtype=torch.float32)

            with patch.dict(os.environ, HP_MEGA_MOE_CAPTURE_DW="1"):
                function_module._CAPTURED_WEIGHT_GRADS = None  # pylint: disable=protected-access
                with patch.object(module_module, "build_mega_moe_plan", side_effect=fp32_plan):
                    out = layer(x, ids, probs)
                    out.float().square().mean().backward()
            captured = function_module._CAPTURED_WEIGHT_GRADS  # pylint: disable=protected-access
            assert captured is not None
            expected_output, first, second = fp32_expert_reference(
                x, layer.gate_up_weight[0], layer.down_weight[0], layout)
            torch.testing.assert_close(out.cpu(), expected_output, rtol=0, atol=0)
            for observed, reference in zip(captured, (first, second)):
                assert observed.dtype == torch.float32
                assert observed[0].shape == reference.shape
                assert torch.count_nonzero(reference).item() > 0
                torch.testing.assert_close(observed[0].cpu(), reference, rtol=1e-5, atol=1e-7)
                assert torch.count_nonzero(observed[1:]).item() == 0
        finally:
            layer.close()
