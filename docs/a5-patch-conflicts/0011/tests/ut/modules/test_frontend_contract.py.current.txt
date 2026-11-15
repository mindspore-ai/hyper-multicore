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
"""CPU checks for model-facing MegaKernel migration, named state and precision policies."""

import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import torch

from hyper_multicore import MegaGate, MegaMhc, MegaMoeExperts
from hyper_multicore.experimental import MegaMoeTuningOptions
from hyper_multicore.modules.mega_gate import module as gate_module
from hyper_multicore.modules.mega_moe import module as moe_module
from tests.common.mark_utils import arg_mark
from tests.common.multicore.mega_mhc_golden import torch_mega_mhc


class TestFrontendMigration(unittest.TestCase):
    """Check original Parameter ownership and mathematical configuration at replacement boundaries."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_ep_size_follows_explicit_subgroup_and_rejects_implicit_world(self) -> None:
        """Feature: MegaKernel frontend migration.
        Description: Ep size follows explicit subgroup and rejects implicit world.
        Expectation: A subgroup determines local expert count; a global process group is never guessed.
        """
        options = {"hidden_size": 16, "intermediate_size": 8, "num_experts": 4, "top_k": 2}
        group = object()
        for tuning_options in (None, MegaMoeTuningOptions(dispatch_mode="pull")):
            for size in (1, 2):
                with (self.subTest(tuning_options=tuning_options, group_size=size),
                      patch.object(moe_module.dist, "is_initialized", return_value=True),
                      patch.object(moe_module.dist, "get_world_size",
                                   side_effect=lambda selected=None, n=size: 8 if selected is None else n) as get_size,
                      patch.object(moe_module.shmem, "root_members", return_value=tuple(range(2, 2 + size)))):
                    layer = MegaMoeExperts(**options, ep_group=group, tuning_options=tuning_options)
                    self.addCleanup(layer.close)
                    get_size.assert_called_once_with(group)
                    self.assertEqual(layer.ep_size, size)
                    self.assertEqual(layer.local_experts, 4 // size)
                    self.assertIs(layer._root_group(), group)  # pylint: disable=protected-access
                    with self.assertRaisesRegex(TypeError, "ep_size"):
                        # pylint: disable-next=unexpected-keyword-arg
                        MegaMoeExperts(**options, ep_group=group, ep_size=size, tuning_options=tuning_options)
                    with self.assertRaisesRegex(ValueError, "ep_group explicitly"):
                        MegaMoeExperts(**options, tuning_options=tuning_options)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_ep_group_requires_initialization_and_defaults_to_single_rank(self) -> None:
        """Explicit groups require distributed initialization; standalone owners infer EP1."""
        options = {"hidden_size": 16, "intermediate_size": 8, "num_experts": 4, "top_k": 2}
        for tuning_options in (None, MegaMoeTuningOptions(dispatch_mode="pull")):
            with (self.subTest(tuning_options=tuning_options),
                  patch.object(moe_module.dist, "is_initialized", return_value=False),
                  patch.object(moe_module.dist, "get_world_size") as get_size):
                layer = MegaMoeExperts(**options, tuning_options=tuning_options)
                self.addCleanup(layer.close)
                self.assertEqual((layer.ep_size, layer.local_experts), (1, 4))
                with self.assertRaisesRegex(ValueError, "Initialize torch.distributed"):
                    MegaMoeExperts(**options, ep_group=object(), tuning_options=tuning_options)
                get_size.assert_not_called()
            with (patch.object(moe_module.dist, "is_initialized", return_value=True),
                  patch.object(moe_module.dist, "get_world_size", return_value=1),
                  patch.object(moe_module.shmem, "root_members", return_value=(0,))):
                layer = MegaMoeExperts(**options, tuning_options=tuning_options)
                self.addCleanup(layer.close)
                self.assertEqual((layer.ep_size, layer.local_experts), (1, 4))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_moe_executor_preserves_external_parameter_ownership(self) -> None:
        """Feature: MegaKernel frontend migration.
        Description: Supply an existing model's packed local parameters to a parameterless executor.
        Expectation: Expert and optimizer state stay with the original owner.
        """
        owner = torch.nn.Module()
        owner.gate_up = torch.nn.Parameter(torch.randn(4, 16, 16, dtype=torch.bfloat16))
        owner.down = torch.nn.Parameter(torch.randn(4, 16, 8, dtype=torch.bfloat16))
        executor = MegaMoeExperts(hidden_size=16, intermediate_size=8, num_experts=4, top_k=2)
        self.addCleanup(executor.close)
        hidden = SimpleNamespace(dtype=torch.bfloat16, is_npu=True, device=torch.device("cpu"))
        executor._validate_tensors(hidden, (owner.gate_up, owner.down))  # pylint: disable=protected-access
        self.assertEqual(executor._pending_layout, "EOI")  # pylint: disable=protected-access
        self.assertEqual(list(executor.parameters()), [])
        self.assertEqual(set(owner.state_dict()), {"gate_up", "down"})
        self.assertFalse(hasattr(MegaMoeExperts, "from_module"))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_moe_migration_passes_tuning_options_and_rejects_old_keywords(self) -> None:
        """Feature: MegaMoe migration tuning.
        Description: Configure pull execution through the experimental options and reject an old flat keyword.
        Expectation: The configuration reaches the resource specification and the old keyword is rejected.
        """
        converted = MegaMoeExperts(hidden_size=16, intermediate_size=8, num_experts=4, top_k=2,
                                    tuning_options=MegaMoeTuningOptions(dispatch_mode="pull", dispatch_split=256))
        self.addCleanup(converted.close)
        self.assertIs(type(converted), MegaMoeExperts)
        specification = converted._resource_group.specification  # pylint: disable=protected-access
        self.assertEqual((specification["dispatch_mode"], specification["dispatch_split"]), ("pull", 256))
        self.assertIsNone(specification["initial_capacity_factor"])
        with self.assertRaisesRegex(TypeError, "dispatch_mode"):
            # This negative test deliberately passes a removed public keyword.
            # pylint: disable-next=unexpected-keyword-arg
            MegaMoeExperts(hidden_size=16, intermediate_size=8, num_experts=4, top_k=2, dispatch_mode="pull")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_gate_migration_and_gradient_parity(self) -> None:
        """Feature: MegaKernel frontend migration.
        Description: Gate migration and gradient parity.
        Expectation: A migrated router keeps Parameters and scoring instead of initializing a new gate.
        """
        source = MegaGate(hidden_size=8, num_experts=4, top_k=2, scoring_func="sigmoid",
                          routed_scaling_factor=1.7)
        source.bias.requires_grad_(False)
        target = MegaGate.from_module(source)
        self.assertIs(target.weight, source.weight)
        self.assertIs(target.bias, source.bias)
        hidden = torch.randn(2, 3, 8, requires_grad=True)
        actual, expected = target(hidden), source(hidden)
        self.assertIs(type(actual), tuple)
        self.assertEqual(tuple(actual[2].shape), (6, 2))
        for got, want in zip(actual, expected):
            torch.testing.assert_close(got, want, rtol=0, atol=0)
        probe = torch.randn_like(actual[1])
        first = torch.autograd.grad((actual[1] * probe).sum(), (hidden, target.weight))
        second = torch.autograd.grad((expected[1] * probe).sum(), (hidden, source.weight))
        for got, want in zip(first, second):
            torch.testing.assert_close(got, want, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "vision_enabled=True"):
            target(hidden, image_mask=torch.ones(2, 3, dtype=torch.bool))
        for source_vision in (False, True):
            original = MegaGate(hidden_size=8, num_experts=4, top_k=2, scoring_func="sigmoid",
                                routed_scaling_factor=1.0, vision_enabled=source_vision)
            for configured in (None, False, True):
                options = {} if configured is None else {"vision_enabled": configured}
                with self.subTest(source_vision=source_vision, configured=configured):
                    if source_vision and configured is False:
                        with self.assertRaisesRegex(ValueError, "source bias_vl"):
                            MegaGate.from_module(original, **options)
                        continue
                    migrated = MegaGate.from_module(original, **options)
                    self.assertIs(migrated.weight, original.weight)
                    self.assertEqual(migrated.bias_vl is not None,
                                     source_vision if configured is None else configured)
                    if source_vision:
                        self.assertIs(migrated.bias_vl, original.bias_vl)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_dtype_conversion_preserves_control_values_and_existing_gradients(self) -> None:
        """Feature: MegaKernel frontend migration.
        Description: Dtype conversion preserves control values and existing gradients.
        Expectation: Global BF16 conversion must never round an FP32 predictor and then cast it back.
        """
        model = torch.nn.Module()
        model.mhc = MegaMhc(128)
        model.gate = MegaGate(hidden_size=128, num_experts=4, top_k=2, scoring_func="sqrtsoftplus",
                              routed_scaling_factor=1.0, vision_enabled=True)
        protected = [model.mhc.phi, model.mhc.alpha, model.mhc.bias, model.gate.bias, model.gate.bias_vl]
        originals = []
        with torch.no_grad():
            for parameter in protected:
                parameter.copy_(torch.randn_like(parameter))
                parameter.grad = torch.randn_like(parameter)
                originals.append((torch.clone(parameter), parameter.grad.clone()))
        model.bfloat16()
        self.assertEqual(model.mhc.norm_weight.dtype, torch.bfloat16)
        self.assertEqual(model.gate.weight.dtype, torch.bfloat16)
        for parameter, (value, gradient) in zip(protected, originals):
            self.assertEqual(parameter.dtype, torch.float32)
            self.assertEqual(parameter.grad.dtype, torch.float32)
            torch.testing.assert_close(parameter, value, rtol=0, atol=0)
            torch.testing.assert_close(parameter.grad, gradient, rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_mhc_migration_keeps_parameters_and_gradients(self) -> None:
        """Feature: MegaKernel frontend migration.
        Description: Compare migrated and source modules with the same five input tensors.
        Expectation: Preserve Parameter identities, outputs and every input/parameter gradient.
        """
        source = MegaMhc(128, hc_eps=2e-6, norm_eps=3e-6)
        target = MegaMhc.from_module(source)
        self.assertIs(type(target), MegaMhc)
        for name, parameter in source.named_parameters():
            self.assertIs(getattr(target, name), parameter)
        self.assertEqual(target.hc_eps, source.hc_eps)
        state = (torch.randn(2, 3, 128, dtype=torch.bfloat16), torch.randn(2, 3, 4, 128, dtype=torch.bfloat16),
                 torch.rand(2, 3, 4), torch.rand(2, 3, 4), torch.rand(2, 3, 4, 4))
        state = tuple(value.requires_grad_() for value in state)

        def golden(*args: torch.Tensor, token_tile: int, **kwargs: Any) -> tuple[torch.Tensor, ...]:
            """Use the independent Torch oracle while accepting the execution-only tile option."""
            del token_tile
            return torch_mega_mhc(*args, **kwargs)

        with patch("hyper_multicore.modules.mega_mhc.module.hyper_mega_mhc", side_effect=golden):
            migrated = target(*state)
            original = source(*state)
        self.assertIs(type(migrated), tuple)
        probe = [torch.randn_like(value) for value in migrated]
        inputs = (*state, *target.parameters())
        actual = torch.autograd.grad(sum((value * weight).sum() for value, weight in zip(migrated, probe)), inputs)
        expected = torch.autograd.grad(sum((value * weight).sum() for value, weight in zip(original, probe)), inputs)
        for got, want in zip((*migrated, *actual), (*original, *expected)):
            torch.testing.assert_close(got, want, rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_gate_dispatch_preserves_scoring_device_and_empty_input_rules(self) -> None:
        """Feature: MegaGate implementation selection.
        Description: Exercise forward selection without loading native plans or running NPU operations.
        Expectation: Only nonempty NPU sqrtsoftplus uses native Route; all other inputs use Torch.
        """
        result = (torch.ones(1, 4), torch.ones(1, 2), torch.zeros(1, 2, dtype=torch.int64))
        cases = (("sqrtsoftplus", "npu", 8, True), ("sqrtsoftplus", "npu", 0, False),
                 ("sqrtsoftplus", "cpu", 8, False), ("softmax", "npu", 8, False),
                 ("sigmoid", "npu", 8, False))
        for scoring, device, elements, native in cases:
            with self.subTest(scoring=scoring, device=device, elements=elements):
                gate = MegaGate(hidden_size=8, num_experts=4, top_k=2, scoring_func=scoring, routed_scaling_factor=1.0)
                hidden = SimpleNamespace(device=SimpleNamespace(type=device), numel=lambda count=elements: count)
                with (patch.object(gate, "_torch_forward", return_value=result) as ordinary,
                      patch.object(gate, "_plan") as plan,
                      patch.object(gate_module, "mega_gate", return_value=result) as fused):
                    self.assertIs(gate(hidden), result)
                self.assertEqual(fused.call_count, int(native))
                self.assertEqual(plan.call_count, int(native))
                self.assertEqual(ordinary.call_count, int(not native))
