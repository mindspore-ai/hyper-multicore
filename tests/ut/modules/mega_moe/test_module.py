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
"""Unit tests for the model-facing MegaMoe module."""

import inspect
import struct
import unittest
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from functools import partial
from itertools import product
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, PropertyMock, patch

import torch

from hyper_multicore import MegaMoeNumericsOptions, _automatic, _collective
from hyper_multicore.experimental import MegaMoeTuningOptions
from hyper_multicore.modules.mega_moe import module as mega_moe_module
from hyper_multicore.modules.mega_moe.backward.graph import build_backward_graph
from hyper_multicore.modules.mega_moe.backward.gen_runtime_data import build_config_for_rank
from hyper_multicore.modules.mega_moe.backward.storage import can_reuse_backward_dispatch
from hyper_multicore.modules.mega_moe.backward.tiling_tables import (
    get_act_grad_tiling_bytes, get_gate_grad_tiling_bytes, get_w1_grad_tiling_bytes, get_w2_grad_tiling_bytes,
)
from hyper_multicore.modules.mega_moe.forward.graph import build_forward_graph
from hyper_multicore.modules.mega_moe.forward.tiling_tables import get_down_proj_tiling_bytes, get_up_proj_tiling_bytes
from hyper_multicore.modules.mega_moe.module import MegaMoeExperts
from hyper_multicore.modules.mega_moe.spec import MegaMoeSpec
from hyper_multicore.scheduler.config import TaskSplitValue
from tests.common.mark_utils import arg_mark


class TestMegaMoeExperts(unittest.TestCase):
    """Validate the public API and execution-resource lifecycle."""

    # pylint: disable=protected-access

    @staticmethod
    @contextmanager
    def _two_rank_group() -> Iterator[Any]:
        """Construct CPU test owners from an explicit mocked two-rank EP group."""
        group = object()
        with (patch.object(mega_moe_module.dist, "is_initialized", return_value=True),
              patch.object(mega_moe_module.dist, "get_world_size", return_value=2),
              patch.object(mega_moe_module.shmem, "root_members", return_value=(0, 1))):
            yield group

    @staticmethod
    def _cache_resources(capacity: int, *, automatic: bool = False, mode: str = "push") -> Any:
        """Create a lazy resource with an explicit committed source reservation."""
        resources = mega_moe_module._MegaMoeExecutionResources.__new__(mega_moe_module._MegaMoeExecutionResources)
        resources.spec = MegaMoeSpec(capacity, 16, 8, 4, 2, None, capacity * 2, 1, None, 0, 24,
                                    max_local_num_tokens=None if automatic else capacity,
                                    auto_tokens=automatic, dispatch_mode=mode)
        resources._token_capacity = capacity
        resources._plans = {}
        return resources

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_fp32_weight_gradient_plan_has_a_separate_cache_entry(self) -> None:
        """FP32 gradient plans cannot change a retained ordinary BF16 plan."""
        resources = self._cache_resources(256, automatic=True)
        def build(spec: MegaMoeSpec, _: Any, **options: Any) -> SimpleNamespace:
            """Record the gradient output dtype chosen for each cached plan."""
            return SimpleNamespace(spec=spec, weight_grad_dtype=options.get("weight_grad_dtype"))
        with patch.object(mega_moe_module, "build_mega_moe_plan", side_effect=build) as builder:
            ordinary = resources.plan_for_tokens(128, "cpu")
            precise = resources.plan_for_tokens(128, "cpu", weight_grad_dtype=torch.float32)
            self.assertIs(resources.plan_for_tokens(128, "cpu"), ordinary)
            self.assertIs(resources.plan_for_tokens(128, "cpu", weight_grad_dtype=torch.float32), precise)
        self.assertIsNone(ordinary.weight_grad_dtype)
        self.assertEqual(precise.weight_grad_dtype, torch.float32)
        self.assertEqual(builder.call_count, 2)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_automatic_constructor_accepts_shapes_without_token_arguments(self) -> None:
        """Feature: Automatic token API.
        Description: Construct both transports without predicting token counts and reject removed keywords.
        Expectation: Start lazily at 128 rows and accept empty, aligned and arbitrary real input shapes.
        """
        for mode in ("push", "pull"):
            with self.subTest(mode=mode):
                layer = MegaMoeExperts(
                    hidden_size=16, intermediate_size=8, num_experts=4, top_k=2,
                    tuning_options=MegaMoeTuningOptions(dispatch_mode=mode))
                self.addCleanup(layer.close)
                self.assertFalse(hasattr(layer, "local_num_tokens"))
                self.assertFalse(hasattr(layer, "max_local_num_tokens"))
                self.assertTrue(layer._resource_group.specification["auto_tokens"])
                self.assertEqual(layer._resource_group.specification["local_num_tokens"], 128)
                self.assertIsNone(layer._resource_group.resources)
                parameters = (torch.empty(4, 16, 16, dtype=torch.bfloat16),
                              torch.empty(4, 8, 16, dtype=torch.bfloat16))
                with patch.object(torch.Tensor, "is_npu", new_callable=PropertyMock, return_value=True, create=True):
                    for rows in (0, 1, 127, 128, 129, 257, 4097):
                        result = layer._validate_forward_inputs(
                            torch.empty(1, rows, 16, dtype=torch.bfloat16),
                            torch.zeros(rows, 2, dtype=torch.int32), torch.ones(rows, 2), None, parameters)
                        self.assertEqual(tuple(result.shape), (rows, 16))
        for keyword in ("local_num_tokens", "max_local_num_tokens"):
            with self.subTest(keyword=keyword), self.assertRaisesRegex(TypeError, keyword):
                MegaMoeExperts(hidden_size=16, intermediate_size=8, num_experts=4, top_k=2, **{keyword: 128})

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_dynamic_cycles_retain_bucket_plans_and_reserved_specification(self) -> None:
        """Feature: Geometric token plan retention.
        Description: Replay seven buckets plus nearby and wide length sequences after warmup.
        Expectation: Each covering plan is built once and retained plans keep the committed reservation.
        """
        for lengths, expected in (
            ((128, 256, 512, 1024, 2048, 4096, 8192), 7),
            ((3200, 3712, 3328, 3968, 3456, 4096, 3584, 3840), 1),
            ((8192, 8064, 7936, 7808, 7680), 1), ((128, 1152, 2176, 3200, 4224), 4),
        ):
            with self.subTest(lengths=lengths):
                resources = self._cache_resources(8192)
                with patch.object(mega_moe_module, "build_mega_moe_plan",
                                  side_effect=lambda spec, device: SimpleNamespace(spec=spec)) as build:
                    first = [resources.plan_for_tokens(tokens, "cpu") for tokens in lengths]
                    for _ in range(3):
                        for tokens, retained in zip(lengths, first):
                            self.assertIs(resources.plan_for_tokens(tokens, "cpu"), retained)
                    self.assertEqual((build.call_count, len(resources._plans)), (expected, expected))
                for plan in first:
                    self.assertEqual((plan.spec.routed_slots, plan.spec.receive_capacity), (16384, 16384))
                    self.assertEqual(resources.spec, replace(plan.spec, schedule_tokens=None))


    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_partial_and_small_buckets_cover_inputs_without_expanding_storage(self) -> None:
        """Feature: bounded token schedule coverage.

        Description: Exercise zero, tiny inputs and non-power-of-two public token limits.
        Expectation: Every plan covers its input within the original reservation and tiny inputs use 128 rows.
        """
        for public_bound, reserved in ((1, 128), (129, 256), (257, 384), (8000, 8064), (8193, 8320)):
            with self.subTest(public_bound=public_bound):
                resources = self._cache_resources(reserved)
                resources.spec = replace(resources.spec, max_local_num_tokens=public_bound)
                with patch.object(mega_moe_module, "build_mega_moe_plan",
                                  side_effect=lambda spec, device: SimpleNamespace(spec=spec)):
                    for requested in (0, 128, reserved):
                        plan = resources.plan_for_tokens(requested, "cpu")
                        self.assertGreaterEqual(plan.spec.plan_tokens, max(128, requested))
                        self.assertLessEqual(plan.spec.plan_tokens, reserved)
                        self.assertEqual(plan.spec.plan_tokens % 128, 0)
                        self.assertEqual(plan.spec.routed_slots, reserved * 2)
                    self.assertEqual(resources.plan_for_tokens(1, "cpu").spec.plan_tokens, 128)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_automatic_plan_uses_committed_capacity_in_both_transports(self) -> None:
        """Feature: automatic token schedule coverage.

        Description: Grow a 128-row reservation in both transports and native weight layouts.
        Expectation: Plans retain the layout and cover 4096 rows without changing the original specification.
        """
        for mode, layout in product(("push", "pull"), ("EIO", "EOI")):
            with self.subTest(mode=mode, layout=layout):
                resources = self._cache_resources(128, automatic=True, mode=mode)
                resources.spec = replace(resources.spec, expert_weight_layout=layout)
                resources.workspace = Mock()
                resources.heap_manager = Mock()
                physical = replace(resources.spec, local_num_tokens=4096, receive_capacity=32768)
                resources.heap_manager.ensure_token_capacity.return_value = physical
                tensor = torch.empty(1, 16, dtype=torch.bfloat16)
                resources.prepare_auto_workspace(4096, 32768, tensor)
                with patch.object(mega_moe_module, "build_mega_moe_plan",
                                  side_effect=lambda spec, device: SimpleNamespace(spec=spec)):
                    plan = resources.plan_for_tokens(4096, "cpu")
                    self.assertEqual(plan.spec.plan_tokens, 4096)
                    self.assertEqual(plan.spec.expert_weight_layout, layout)
                resources.workspace.ensure.assert_called_once_with(physical, tensor.dtype, tensor.device)
                self.assertEqual(resources.workspace.claim.call_count, int(mode == "pull"))
                self.assertEqual(resources.spec.local_num_tokens, 128)
                self.assertEqual(resources._token_capacity, 4096)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_partial_bucket_upgrade_is_retryable_and_keeps_retained_plans(self) -> None:
        """Feature: Immutable partial bucket replacement.
        Description: Fail a larger plan build, then retry it while backward retains the old plan.
        Expectation: Failure preserves the cache; successful retry replaces only the cached reference.
        """
        resources = self._cache_resources(128, automatic=True)
        resources._token_capacity = 384
        with patch.object(mega_moe_module, "build_mega_moe_plan",
                          side_effect=lambda spec, device: SimpleNamespace(spec=spec)):
            retained = resources.plan_for_tokens(384, "cpu")
            resources._token_capacity = 512
            with (patch.object(mega_moe_module, "build_mega_moe_plan", side_effect=RuntimeError("build failed")),
                  self.assertRaisesRegex(RuntimeError, "build failed")):
                resources.plan_for_tokens(512, "cpu")
            self.assertIs(resources._plans[512], retained)
            self.assertEqual(retained.spec.plan_tokens, 384)
            expanded = resources.plan_for_tokens(512, "cpu")
            self.assertIsNot(expanded, retained)
            self.assertEqual((retained.spec.plan_tokens, expanded.spec.plan_tokens), (384, 512))
            self.assertIs(resources.plan_for_tokens(384, "cpu"), expanded)
            self.assertEqual(len(resources._plans), 1)
            self.assertEqual(resources.spec.local_num_tokens, 128)


    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_unprepared_capacity_is_rejected_before_building(self) -> None:
        """Feature: committed token capacity enforcement.

        Description: Fail workspace preparation before requesting a larger token schedule.
        Expectation: The previous capacity is retained and no undersized plan is constructed.
        """
        resources = self._cache_resources(128, automatic=True)
        resources.workspace = Mock()
        resources.workspace.ensure.side_effect = RuntimeError("workspace failed")
        resources.heap_manager = Mock()
        resources.heap_manager.ensure_token_capacity.return_value = replace(resources.spec, local_num_tokens=4096)
        with self.assertRaisesRegex(RuntimeError, "workspace failed"):
            resources.prepare_auto_workspace(4096, 8192, torch.empty(1, 16))
        self.assertEqual(resources._token_capacity, 128)
        with (patch.object(mega_moe_module, "build_mega_moe_plan") as build,
              self.assertRaisesRegex(RuntimeError, "requested=4096, capacity=128")):
            resources.plan_for_tokens(4096, "cpu")
        build.assert_not_called()

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_native_int32_limit_bounds_the_number_of_cached_plans(self) -> None:
        """Feature: bounded geometric plan cache.

        Description: Replay every geometric level within the native INT32 source row limit.
        Expectation: Each level is constructed once and no cached plan exceeds the reservation.
        """
        capacity = ((2**31 - 1) // 16 // 128) * 128
        resources = self._cache_resources(capacity)
        lengths = [2**power for power in range(7, 27)] + [capacity]
        with patch.object(mega_moe_module, "build_mega_moe_plan",
                          side_effect=lambda spec, device: SimpleNamespace(spec=spec)) as build:
            for tokens in lengths * 3:
                resources.plan_for_tokens(tokens, "cpu")
            self.assertEqual(build.call_count, 21)
            self.assertEqual(len(resources._plans), 21)
            self.assertLessEqual(max(plan.spec.plan_tokens for plan in resources._plans.values()), capacity)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_resource_consensus_rejects_configuration_and_wire_mismatches(self) -> None:
        """Feature: Distributed resource and automatic count agreement.
        Description: Inject a different manifest, legacy wire version or incompatible expert-weight layout.
        Expectation: Every mismatch fails before symmetric allocation or count exchange.
        """
        resources = mega_moe_module._MegaMoeExecutionResources.__new__(mega_moe_module._MegaMoeExecutionResources)
        resources.spec = SimpleNamespace(ep_size=2, ep_group=object(), expert_weight_layout="EOI")
        for mismatch in ("configuration", "wire", "layout"):
            def gather(output: list, value: Any, kind: str = mismatch, **kwargs: Any) -> None:
                """Change one peer agreement field while preserving all unrelated metadata."""
                self.assertIs(kwargs["group"], resources.spec.ep_group)
                if kind == "layout":
                    self.assertEqual(value[2], "EOI")
                    peer = (value[0], value[1], "EIO", value[3])
                else:
                    peer = ("dynamic_tokens_v1", *value[1:]) if kind == "wire" else ("different",)
                output[:] = [value, peer]

            with (self.subTest(mismatch=mismatch),
                  patch.object(mega_moe_module.dist, "all_gather_object", side_effect=gather),
                  self.assertRaisesRegex(ValueError, "configurations differ")):
                resources._validate_distributed_configuration(
                    ({"local_num_tokens": 128, "auto_tokens": True, "expert_weight_layout": "EOI"},))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_precision_options_participate_in_rank_consensus(self) -> None:
        """Reject a rank using a different clamp or routing-probability definition."""
        resources = mega_moe_module._MegaMoeExecutionResources.__new__(mega_moe_module._MegaMoeExecutionResources)
        resources.spec = SimpleNamespace(ep_size=2, ep_group=object(), expert_weight_layout="EOI")
        specification = {"local_num_tokens": 128, "auto_tokens": True, "expert_weight_layout": "EOI",
                         "swiglu_limit": 0.5, "clamp_after_activation": False, "apply_probs_before_fc2": False}

        def capture(output: list, manifest: Any, *, peer_manifest: list, **_kwargs: Any) -> None:
            """Record the independently normalized peer manifest."""
            peer_manifest.append(manifest)
            output[:] = [manifest, manifest]

        def mismatch(output: list, manifest: Any, *, peer_manifest: list, **_kwargs: Any) -> None:
            """Converge the two ranks' different mathematical configurations."""
            output[:] = [manifest, peer_manifest[0]]

        for field, value in (("swiglu_limit", 7.0), ("clamp_after_activation", True),
                             ("apply_probs_before_fc2", True)):
            peer_manifest = []
            with patch.object(mega_moe_module.dist, "all_gather_object",
                              side_effect=partial(capture, peer_manifest=peer_manifest)):
                resources._validate_distributed_configuration(({**specification, field: value},))
            with (self.subTest(field=field),
                  patch.object(mega_moe_module.dist, "all_gather_object",
                               side_effect=partial(mismatch, peer_manifest=peer_manifest)),
                  self.assertRaisesRegex(ValueError, "configurations differ")):
                resources._validate_distributed_configuration((specification,))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_constructor_defaults_to_growing_push(self) -> None:
        """Feature: Push capacity defaults.
        Description: Construct a layer without explicit capacity factors.
        Expectation: Resolve both factors to 1.25 without creating expert weights.
        """
        experts = MegaMoeExperts(hidden_size=16, intermediate_size=8,
                                 num_experts=4, top_k=2)
        self.addCleanup(experts.close)
        specification = experts._resource_group.specification
        self.assertEqual((specification["dispatch_mode"], specification["initial_capacity_factor"],
                          specification["capacity_growth_factor"]), ("push", 1.25, 1.25))
        self.assertEqual(list(experts.parameters()), [])
        self.assertIsNone(experts.swiglu_limit)
        self.assertFalse(experts.clamp_after_silu)
        self.assertFalse(experts.apply_probs_before_fc2)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_gmm_tiling_preserves_split_when_dimensions_collide(self) -> None:
        """Keep native M/N/K fields and complete per-core rows when replacing dimensions."""
        row_bytes = struct.calcsize("<504I")
        for builder, dimensions in (
            (get_up_proj_tiling_bytes, (4096, 8192, 4096)),
            (get_down_proj_tiling_bytes, (4096, 4096, 4096)),
            (get_act_grad_tiling_bytes, (4096, 4096, 4096)),
            (get_gate_grad_tiling_bytes, (4096, 4096, 8192)),
            (get_w1_grad_tiling_bytes, (4096, 8192, 4096)),
            (get_w2_grad_tiling_bytes, (4096, 4096, 4096)),
        ):
            for cores in (20, 24, 32):
                with self.subTest(builder=builder.__name__, cores=cores):
                    data = builder(4096, hidden_size=4096, intermediate_size=4096,
                                   num_groups=4, num_cube_cores=cores)
                    self.assertEqual(len(data), cores * row_bytes)
                    self.assertEqual(data, data[:row_bytes] * cores)
                    words = struct.unpack("<504I", data[:row_bytes])
                    self.assertEqual(words[:2], (4, cores))
                    self.assertEqual(words[14], 4096)
                    matmul_start = words.index(393216) - 19
                    self.assertEqual(words[matmul_start], cores)
                    self.assertEqual(words[matmul_start + 1:matmul_start + 4], dimensions)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_options_replace_flat_numerical_and_tuning_keywords(self) -> None:
        """Feature: Model-facing MegaMoe interface.
        Description: Inspect the signature and try the former tuning keywords.
        Expectation: Execution choices use the typed extension point instead of flat constructor keywords.
        """
        names = ("dispatch_mode", "initial_capacity_factor", "capacity_growth_factor",
                 "dispatch_split", "combine_split")
        layer = MegaMoeExperts(hidden_size=16, intermediate_size=8, num_experts=4, top_k=2)
        self.addCleanup(layer.close)
        for name in names:
            with self.subTest(name=name):
                self.assertNotIn(name, inspect.signature(MegaMoeExperts).parameters)
                self.assertFalse(hasattr(layer, name))
                with self.assertRaisesRegex(TypeError, name):
                    MegaMoeExperts(hidden_size=16, intermediate_size=8, num_experts=4, top_k=2, **{name: 128})
        self.assertNotIn("_tuning", inspect.signature(MegaMoeExperts).parameters)
        self.assertFalse(hasattr(layer, "_tuning"))
        self.assertIn("numerics_options", inspect.signature(MegaMoeExperts).parameters)
        self.assertIn("tuning_options", inspect.signature(MegaMoeExperts).parameters)
        self.assertNotIn("ep_size", inspect.signature(MegaMoeExperts).parameters)
        for name in ("expert_weight_layout", "create_parameters", "clamp_after_activation",
                     "swiglu_limit", "clamp_after_silu", "apply_probs_before_fc2", "execution_options",
                     "weight_grad_dtype", "batch_experts", "nominal_schedule"):
            with self.subTest(name=name), self.assertRaisesRegex(TypeError, name):
                MegaMoeExperts(hidden_size=16, intermediate_size=8, num_experts=4, top_k=2, **{name: None})
        self.assertEqual(inspect.signature(MegaMoeExperts.forward).parameters["expert_weights"].default,
                         inspect.Parameter.empty)
        self.assertEqual(list(layer.parameters()), [])

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_weight_layout_is_inferred_once_and_checked_for_shared_layers(self) -> None:
        """Identify both layouts from both tensors, including single-weight square shapes."""
        hidden = SimpleNamespace(dtype=torch.bfloat16, is_npu=True, device=torch.device("cpu"))
        for hidden_size, intermediate_size in ((16, 8), (8, 8), (12, 8)):
            for layout in ("EIO", "EOI"):
                with self.subTest(hidden=hidden_size, intermediate=intermediate_size, layout=layout):
                    layers = [MegaMoeExperts(hidden_size=hidden_size, intermediate_size=intermediate_size,
                                             num_experts=4, top_k=2) for _ in range(2)]
                    MegaMoeExperts.share_execution_resources(layers)
                    group = layers[0]._resource_group
                    shapes = layers[0]._weight_shapes
                    weights = tuple(torch.empty(shape, dtype=torch.bfloat16) for shape in shapes[layout])
                    try:
                        layers[0]._validate_tensors(hidden, weights)
                        self.assertEqual(layers[0]._pending_layout, layout)
                        self.assertEqual(group.compatibility_key[-1], layout)
                        group.resources = SimpleNamespace(spec=SimpleNamespace(expert_weight_layout=layout))
                        fresh = tuple(torch.empty(shape, dtype=torch.bfloat16) for shape in shapes[layout])
                        layers[1]._validate_tensors(hidden, fresh)
                        other = "EOI" if layout == "EIO" else "EIO"
                        wrong = tuple(torch.empty(shape, dtype=torch.bfloat16) for shape in shapes[other])
                        with self.assertRaisesRegex(ValueError, "must have shape"):
                            layers[1]._validate_tensors(hidden, wrong)
                    finally:
                        group.resources = None
                        for layer in layers:
                            layer.close()

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_tuning_options_control_specification_and_resource_sharing(self) -> None:
        """Feature: Experimental execution options.
        Description: Construct tuned push/pull owners and compare each resource compatibility key.
        Expectation: Every choice reaches the specification and differing choices prevent workspace sharing.
        """
        default = MegaMoeExperts(hidden_size=16, intermediate_size=8, num_experts=4, top_k=2)
        self.addCleanup(default.close)
        variants = (
            ({"dispatch_mode": "pull"}, ("pull", None, None, 128, 128)),
            ({"initial_capacity_factor": 1.5}, ("push", 1.5, 1.25, 128, 128)),
            ({"capacity_growth_factor": 2.0}, ("push", 1.25, 2.0, 128, 128)),
            ({"dispatch_split": 256}, ("push", 1.25, 1.25, 256, 128)),
            ({"combine_split": 384}, ("push", 1.25, 1.25, 128, 384)),
        )
        names = ("dispatch_mode", "initial_capacity_factor", "capacity_growth_factor",
                 "dispatch_split", "combine_split")
        for options, values in variants:
            with self.subTest(options=options):
                layer = MegaMoeExperts(hidden_size=16, intermediate_size=8, num_experts=4, top_k=2,
                                       tuning_options=MegaMoeTuningOptions(**options))
                self.addCleanup(layer.close)
                for key, value in zip(names, values):
                    self.assertEqual(layer._resource_group.specification[key], value)
                with self.assertRaises(ValueError):
                    MegaMoeExperts.share_execution_resources([default, layer])

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_default_options_share_the_same_module_and_resources(self) -> None:
        """Feature: One model-facing executor type.
        Description: Compare implicit, explicit and normalized numerical and tuning defaults.
        Expectation: Equivalent configurations share resources without changing model or optimizer state.
        """
        options = {"hidden_size": 16, "intermediate_size": 8, "num_experts": 4, "top_k": 2}
        ordinary = MegaMoeExperts(**options)
        tuned = MegaMoeExperts(**options, numerics_options=MegaMoeNumericsOptions(),
                               tuning_options=MegaMoeTuningOptions())
        peer = MegaMoeExperts(**options, numerics_options=None, tuning_options=MegaMoeTuningOptions(
            initial_capacity_factor=1.25, capacity_growth_factor=1.25))
        for layer in (ordinary, tuned, peer):
            self.addCleanup(layer.close)
        self.assertIs(type(tuned), MegaMoeExperts)
        self.assertEqual(ordinary._resource_group.specification, tuned._resource_group.specification)
        self.assertEqual(tuple(ordinary.state_dict()), tuple(tuned.state_dict()))
        tuned.load_state_dict(ordinary.state_dict())
        for name, value in ordinary.state_dict().items():
            torch.testing.assert_close(tuned.state_dict()[name], value)
        MegaMoeExperts.share_execution_resources([ordinary, tuned, peer])
        self.assertIs(ordinary._resource_group, tuned._resource_group)
        self.assertIs(tuned._resource_group, peer._resource_group)
        binding = SimpleNamespace(dtype=torch.bfloat16)
        self.assertEqual(ordinary._resource_signature(binding), tuned._resource_signature(binding))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_precision_options_are_read_only_and_separate_resource_groups(self) -> None:
        """Keep construction-time mathematics fixed throughout resource and graph lifetimes."""
        options = {"hidden_size": 16, "intermediate_size": 8, "num_experts": 4, "top_k": 2}
        ordinary = MegaMoeExperts(**options)
        self.addCleanup(ordinary.close)
        for limit, clamp_after_silu, apply_probs_before_fc2 in product((None, 0.5), (False, True), (False, True)):
            with self.subTest(limit=limit, clamp=clamp_after_silu, probs=apply_probs_before_fc2):
                layer = MegaMoeExperts(**options, numerics_options=MegaMoeNumericsOptions(
                    swiglu_limit=limit, clamp_after_silu=clamp_after_silu,
                    apply_probs_before_fc2=apply_probs_before_fc2))
                self.addCleanup(layer.close)
                specification = layer._resource_group.specification
                for name, value, field in (("swiglu_limit", limit, "swiglu_limit"),
                                           ("clamp_after_silu", clamp_after_silu, "clamp_after_activation"),
                                           ("apply_probs_before_fc2", apply_probs_before_fc2,
                                            "apply_probs_before_fc2")):
                    self.assertEqual(getattr(layer, name), value)
                    self.assertEqual(specification[field], value)
                    with self.assertRaises(AttributeError):
                        setattr(layer, name, value)
                if limit is not None or clamp_after_silu or apply_probs_before_fc2:
                    with self.assertRaisesRegex(ValueError, "identical configuration"):
                        MegaMoeExperts.share_execution_resources([ordinary, layer])

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_numerical_flags_require_booleans_and_options_require_their_types(self) -> None:
        """Reject truthy substitutes and untyped options before registering execution resources."""
        options = {"hidden_size": 16, "intermediate_size": 8, "num_experts": 4, "top_k": 2}
        for name in ("clamp_after_silu", "apply_probs_before_fc2"):
            for value in (None, 0, 1, "true", [], {}, torch.tensor(True)):
                with self.subTest(name=name, value=value), self.assertRaisesRegex(TypeError, name):
                    MegaMoeExperts(**options, numerics_options=MegaMoeNumericsOptions(**{name: value}))
        for value in ({}, {"dispatch_mode": "pull"}, "pull", object()):
            with self.subTest(options=value), self.assertRaisesRegex(TypeError, "tuning_options"):
                MegaMoeExperts(**options, tuning_options=value)
        for value in ({}, {"swiglu_limit": 7.0}, "default", object(), MegaMoeTuningOptions()):
            with self.subTest(options=value), self.assertRaisesRegex(TypeError, "numerics_options"):
                MegaMoeExperts(**options, numerics_options=value)
        with self.assertRaisesRegex(TypeError, "tuning_options"):
            MegaMoeExperts(**options, tuning_options=MegaMoeNumericsOptions())

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_eoi_layout_is_validated_and_propagated_to_native_graphs(self) -> None:
        """Feature: Native EOI expert-weight layout.
        Description: Construct automatic-token experts and build both native graphs with Torch linear storage.
        Expectation: Preserve EOI operand shapes, transpose flags and rejection of noncontiguous weights.
        """
        with self._two_rank_group() as group:
            experts = MegaMoeExperts(
                hidden_size=12, intermediate_size=8, num_experts=4, top_k=2, ep_group=group,
            ).to(dtype=torch.bfloat16)
        self.addCleanup(experts.close)
        hidden_states = SimpleNamespace(dtype=torch.bfloat16, is_npu=True, device=torch.device("cpu"))
        weights = (torch.empty(2, 16, 12, dtype=torch.bfloat16),
                   torch.empty(2, 12, 8, dtype=torch.bfloat16))
        experts._validate_tensors(hidden_states, weights)
        self.assertEqual(experts._pending_layout, "EOI")

        topology = TaskSplitValue(tp=1, ep=2, all_expert_num=4, top_k=2, seq_size=128)
        forward = build_forward_graph(
            topology,
            hidden_size=12,
            intermediate_size=8,
            expert_weight_layout="EOI",
        )
        backward = build_backward_graph(
            topology,
            hidden_size=12,
            intermediate_size=8,
            expert_weight_layout="EOI",
        )
        self.assertTrue(forward.get_op("up_proj").inputs[1].transpose)
        self.assertTrue(forward.get_op("down_proj").inputs[1].transpose)
        self.assertFalse(backward.get_op("act_grad").inputs[1].transpose)
        self.assertFalse(backward.get_op("gate_grad").inputs[1].transpose)

        backward.propagate_splits(topology)
        runtime = build_config_for_rank(backward, topology, 0)
        for name, positions, shape, tiling_builder in (
            ("w1_grad", [10, 17], [2, 16, 12], get_w1_grad_tiling_bytes),
            ("w2_grad", [0, 5], [2, 12, 8], get_w2_grad_tiling_bytes),
        ):
            op = backward.get_op(name)
            self.assertEqual(op.task_num, 48, msg=f"{name} tasks: expected=48, actual={op.task_num}")
            self.assertEqual(op.outputs[0].shape, shape,
                             msg=f"{name} output: expected={shape}, actual={op.outputs[0].shape}")
            task = next(task for task in runtime.all_tasks[:runtime.task_num]
                        if task.outputs[0].input_position == op.outputs[0].param_position)
            actual_positions = [task.inputs[i].input_position for i in range(2)]
            actual_transpose = [task.inputs[i].transpose_flag for i in range(2)]
            self.assertEqual(actual_positions, positions,
                             msg=f"{name} operands: expected={positions}, actual={actual_positions}")
            self.assertEqual(actual_transpose, [1, 0],
                             msg=f"{name} transpose: expected=[1, 0], actual={actual_transpose}")
            tiling = tiling_builder(op.split_value, hidden_size=12, intermediate_size=8,
                                   num_groups=2, expert_weight_layout="EOI")
            words = struct.unpack(f"<{len(tiling) // 4}I", tiling)
            matmul_start = words.index(393216) - 19
            axes = list(words[matmul_start + 1:matmul_start + 3])
            self.assertEqual(axes, shape[1:], msg=f"{name} tiling M/N: expected={shape[1:]}, actual={axes}")
        self.assertTrue(can_reuse_backward_dispatch(runtime, 2, 24),
                        msg=f"Expected safe dispatch reuse, actual={can_reuse_backward_dispatch(runtime, 2, 24)}")

        hidden_states = SimpleNamespace(dtype=torch.bfloat16, is_npu=True, device=torch.device("cpu"))
        noncontiguous_gate_up = torch.empty(2, 12, 16, dtype=torch.bfloat16).transpose(1, 2)
        noncontiguous_down = torch.empty(2, 8, 12, dtype=torch.bfloat16).transpose(1, 2)
        with self.assertRaisesRegex(ValueError, "EOI expert weights must be contiguous"):
            experts._validate_tensors(hidden_states, (noncontiguous_gate_up, noncontiguous_down))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_eio_is_the_default_native_graph_layout(self) -> None:
        """Feature: Default EIO expert-weight layout.
        Description: Build forward/backward graphs with the default layout and reject unknown layouts.
        Expectation: Keep the original operand shapes and transpose interpretation.
        """
        topology = TaskSplitValue(tp=1, ep=2, all_expert_num=4, top_k=2, seq_size=128)
        forward = build_forward_graph(topology, hidden_size=12, intermediate_size=8)
        backward = build_backward_graph(topology, hidden_size=12, intermediate_size=8)

        self.assertFalse(forward.get_op("up_proj").inputs[1].transpose)
        self.assertFalse(forward.get_op("down_proj").inputs[1].transpose)
        self.assertTrue(backward.get_op("act_grad").inputs[1].transpose)
        self.assertTrue(backward.get_op("gate_grad").inputs[1].transpose)

        for builder in (build_forward_graph, build_backward_graph):
            with self.subTest(builder=builder.__name__):
                with self.assertRaisesRegex(
                    ValueError, "expert_weight_layout must be 'EIO' or 'EOI', got 'invalid'",
                    msg=f"Expected invalid layout rejection from {builder.__name__}",
                ):
                    builder(topology, expert_weight_layout="invalid")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_constructor_rejects_invalid_static_values_before_allocation(self) -> None:
        """Feature: MegaMoe public API and resource lifecycle.

        Description: Reject invalid capacity and token split during construction.
        Expectation: Invalid topology, alignment and capacity factors fail before parameter allocation.
        """
        for overrides, message in (
            ({"initial_capacity_factor": 0.999}, "initial_capacity_factor"),
            ({"num_experts": 35}, "divisible"),
            ({"dispatch_mode": "invalid"}, "dispatch_mode"),
            ({"capacity_growth_factor": 0.999}, "capacity_growth_factor"),
            ({"initial_capacity_factor": float("nan")}, "initial_capacity_factor"),
            ({"capacity_growth_factor": float("inf")}, "capacity_growth_factor"),
            ({"initial_capacity_factor": True}, "initial_capacity_factor"),
            ({"capacity_growth_factor": 1.5, "dispatch_mode": "pull"}, "push"),
            ({"dispatch_split": 129}, "dispatch_split"),
            ({"combine_split": 0}, "combine_split"),
            ({"dispatch_split": True}, "dispatch_split"),
        ):
            with (
                self.subTest(overrides=overrides),
                self._two_rank_group() as group,
                self.assertRaisesRegex(ValueError, message),
            ):
                MegaMoeExperts(
                    hidden_size=16,
                    intermediate_size=8,
                    num_experts=overrides.get("num_experts", 4),
                    top_k=2,
                    ep_group=group,
                    tuning_options=MegaMoeTuningOptions(
                        dispatch_mode=overrides.get("dispatch_mode", "push"),
                        initial_capacity_factor=overrides.get("initial_capacity_factor"),
                        capacity_growth_factor=overrides.get("capacity_growth_factor"),
                        dispatch_split=overrides.get("dispatch_split", 128),
                        combine_split=overrides.get("combine_split", 128),
                    ),
                )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_constructor_validates_and_records_swiglu_limit(self) -> None:
        """Feature: validate the model-facing SwiGLU clamp option.

        Description: Construct experts with one valid limit and several invalid
            or non-float32-representable values.
        Expectation: The valid limit is retained and invalid limits fail before
            parameter allocation.
        """
        experts = MegaMoeExperts(
            hidden_size=16,
            intermediate_size=8,
            num_experts=4,
            top_k=2,
            numerics_options=MegaMoeNumericsOptions(swiglu_limit=10),
        )
        try:
            self.assertEqual(experts.swiglu_limit, 10.0)
            self.assertEqual(
                experts._resource_group.specification["swiglu_limit"],
                10.0,
            )
        finally:
            experts.close()

        for invalid_limit in (
            0,
            -1,
            1e-50,
            1e39,
            float("nan"),
            float("inf"),
            True,
            "10",
        ):
            with (
                self.subTest(swiglu_limit=invalid_limit),
                self.assertRaisesRegex(ValueError, "swiglu_limit"),
            ):
                MegaMoeExperts(
                    hidden_size=16,
                    intermediate_size=8,
                    num_experts=4,
                    top_k=2,
                    numerics_options=MegaMoeNumericsOptions(swiglu_limit=invalid_limit),
                )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_forward_passes_router_inputs_and_restores_shape(self) -> None:
        """Feature: forward passes router inputs and restores shape.
        Description: Execute forward with router metadata and caller-shaped inputs.
        Expectation: Preserve Router inputs, expert parameters and the caller's shape.
        """
        with self._two_rank_group() as group:
            experts = MegaMoeExperts(
                hidden_size=16, intermediate_size=8, num_experts=4, top_k=2, ep_group=group,
            ).to(dtype=torch.bfloat16)
        self.addCleanup(experts.close)
        hidden_states = torch.arange(2048, dtype=torch.bfloat16).reshape(2, 64, 16)
        topk_ids = torch.zeros((128, 2), dtype=torch.int32)
        topk_weights = torch.full((128, 2), 0.5)
        expert_weights = (torch.nn.Parameter(torch.empty(2, 16, 16, dtype=torch.bfloat16)),
                          torch.nn.Parameter(torch.empty(2, 8, 16, dtype=torch.bfloat16)))
        tokens_per_expert = torch.tensor([256, 0, 0, 0], dtype=torch.int32)
        expected = hidden_states.reshape(128, 16) + 1
        resources = SimpleNamespace(spec=SimpleNamespace(auto_tokens=False, dynamic_tokens=False, dispatch_mode="push",
                                                         apply_probs_before_fc2=False),
                                    plan=object(), workspace=object(), ensure_capacity=Mock(),
                                    plan_for_tokens=Mock())
        route = SimpleNamespace(
            routed_tokens=object(), metadata=SimpleNamespace(plan_tokens=128),
            unpermute_mapping=object(), maximum_received_slots=512
        )
        resources.plan_for_tokens.return_value = resources.plan
        expert_output = object()

        with (
            patch.object(
                torch.Tensor, "is_npu", new_callable=PropertyMock, return_value=True, create=True
            ),
            patch.object(torch, "npu", Mock(is_current_stream_capturing=Mock(return_value=False)), create=True),
            patch.object(experts, "_get_execution_resources", return_value=resources),
            patch.object(
                mega_moe_module, "prepare_topk_route", return_value=route
            ) as mock_prepare,
            patch.object(
                mega_moe_module, "execute_mega_moe_with_permutation", return_value=expert_output
            ) as mock_execute,
            patch.object(
                mega_moe_module, "restore_topk_output", return_value=expected
            ) as mock_restore,
        ):
            actual = experts(
                hidden_states,
                topk_ids,
                topk_weights,
                expert_weights=expert_weights,
                tokens_per_expert=tokens_per_expert,
            )

        torch.testing.assert_close(actual, expected.reshape(hidden_states.shape))
        hidden_flat = mock_prepare.call_args.args[0]
        torch.testing.assert_close(hidden_flat, hidden_states.reshape(128, 16))
        mock_prepare.assert_called_once_with(
            hidden_flat, topk_ids, topk_weights, resources.spec, tokens_per_expert,
            workspace=resources.workspace, autograd_mask=13,
        )
        resources.ensure_capacity.assert_called_once_with(512)
        mock_execute.assert_called_once_with(
            hidden_flat,
            topk_ids,
            expert_weights[0],
            expert_weights[1],
            route,
            resources.plan,
            resources.workspace,
            topk_weights=None,
            workspace_claimed=False,
        )
        mock_restore.assert_called_once_with(
            expert_output, route.unpermute_mapping, topk_weights
        )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_forward_uses_experimental_pull_configuration(self) -> None:
        """Feature: Experimental pull dispatch.
        Description: Execute a mocked automatic-token forward with explicit pull options.
        Expectation: Prepare and release the pull workspace and pass routing weights to the kernel.
        """
        experts = MegaMoeExperts(
            hidden_size=16, intermediate_size=8, num_experts=4, top_k=2,
            tuning_options=MegaMoeTuningOptions(dispatch_mode="pull"),
        ).to(dtype=torch.bfloat16)
        self.addCleanup(experts.close)
        hidden = torch.ones(2, 16, dtype=torch.bfloat16)
        ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
        weights = torch.ones(2, 2)
        route = SimpleNamespace(metadata=SimpleNamespace(plan_tokens=128))
        workspace = Mock()
        resources = SimpleNamespace(
            spec=SimpleNamespace(auto_tokens=True, apply_probs_before_fc2=False,
                                 dispatch_mode=experts._resource_group.specification["dispatch_mode"]),
            workspace=workspace, prepare_auto_workspace=Mock(), plan_for_tokens=Mock(return_value=object()),
        )

        def _prepare(*_args: Any, **kwargs: Any) -> Any:
            """Run the capacity callback as the real route preparation does."""
            kwargs["prepare_workspace"](128, 4)
            return route

        with (
            patch.object(torch.Tensor, "is_npu", new_callable=PropertyMock, return_value=True, create=True),
            patch.object(torch, "npu", Mock(is_current_stream_capturing=Mock(return_value=False)), create=True),
            patch.object(experts, "_get_execution_resources", return_value=resources),
            patch.object(mega_moe_module, "prepare_topk_route", side_effect=_prepare) as mock_prepare,
            patch.object(mega_moe_module, "execute_mega_moe_with_permutation", return_value=hidden) as mock_execute,
            patch.object(mega_moe_module, "restore_topk_output") as mock_restore,
        ):
            expert_weights = (torch.empty(4, 16, 16, dtype=torch.bfloat16),
                              torch.empty(4, 8, 16, dtype=torch.bfloat16))
            actual = experts(hidden, ids, weights, expert_weights=expert_weights)
        torch.testing.assert_close(actual, hidden)
        resources.prepare_auto_workspace.assert_called_once_with(128, 4, mock_prepare.call_args.args[0])
        self.assertIs(mock_execute.call_args.kwargs["topk_weights"], weights)
        self.assertTrue(mock_execute.call_args.kwargs["workspace_claimed"])
        workspace.release.assert_called_once_with()
        mock_restore.assert_not_called()

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_push_before_fc2_passes_probabilities_without_weighting_outputs_again(self) -> None:
        """Keep pre-FC2 probabilities in the native path and skip the weighted output restore."""
        experts = MegaMoeExperts(hidden_size=16, intermediate_size=8, num_experts=4, top_k=2,
                                 numerics_options=MegaMoeNumericsOptions(apply_probs_before_fc2=True))
        self.addCleanup(experts.close)
        hidden = torch.ones(2, 16, dtype=torch.bfloat16)
        ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
        probabilities = torch.full((2, 2), 0.5)
        expert_weights = (torch.empty(4, 16, 16, dtype=torch.bfloat16),
                          torch.empty(4, 8, 16, dtype=torch.bfloat16))
        route = SimpleNamespace(metadata=SimpleNamespace(plan_tokens=128), maximum_received_slots=4)
        resources = SimpleNamespace(
            spec=SimpleNamespace(auto_tokens=False, dispatch_mode="push", apply_probs_before_fc2=True),
            workspace=object(), ensure_capacity=Mock(), plan_for_tokens=Mock(return_value=object()),
        )
        with (
            patch.object(torch.Tensor, "is_npu", new_callable=PropertyMock, return_value=True, create=True),
            patch.object(torch, "npu", Mock(is_current_stream_capturing=Mock(return_value=False)), create=True),
            patch.object(experts, "_get_execution_resources", return_value=resources),
            patch.object(mega_moe_module, "prepare_topk_route", return_value=route),
            patch.object(mega_moe_module, "execute_mega_moe_with_permutation", return_value=hidden) as execute,
            patch.object(mega_moe_module, "restore_topk_output") as restore,
        ):
            actual = experts(hidden, ids, probabilities, expert_weights=expert_weights)
        torch.testing.assert_close(actual, hidden)
        self.assertIs(execute.call_args.kwargs["topk_weights"], probabilities)
        self.assertFalse(execute.call_args.kwargs["workspace_claimed"])
        resources.ensure_capacity.assert_called_once_with(4)
        restore.assert_not_called()

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_forward_rejects_invalid_weights_before_resource_creation(self) -> None:
        """Feature: MegaMoe public API and resource lifecycle.

        Description: Reject invalid expert weights before initializing native resources.
        Expectation: Invalid external expert tensors fail before any execution resources are created.
        """
        experts = MegaMoeExperts(
            hidden_size=16,
            intermediate_size=8,
            num_experts=4,
            top_k=2,
        ).to(dtype=torch.bfloat16)
        self.addCleanup(experts.close)
        with (
            patch.object(
                torch.Tensor, "is_npu", new_callable=PropertyMock, return_value=True, create=True
            ),
            patch.object(
                experts, "_create_execution_resources"
            ) as mock_create_resources,
            self.assertRaisesRegex(ValueError, "expert_weights must have EIO shapes"),
        ):
            experts(
                torch.zeros((128, 16), dtype=torch.bfloat16),
                torch.zeros((128, 2), dtype=torch.int32),
                torch.full((128, 2), 0.5),
                expert_weights=(torch.zeros((4, 16, 16), dtype=torch.bfloat16),
                                torch.zeros((4, 15, 8), dtype=torch.bfloat16)),
            )
        with (patch.object(torch.Tensor, "is_npu", new_callable=PropertyMock, return_value=True, create=True),
              patch.object(experts, "_create_execution_resources", mock_create_resources)):
            for weights, error in ((None, ValueError),
                                   ((torch.zeros(4, 16, 16), torch.zeros(4, 8, 16)), TypeError)):
                with self.assertRaises(error):
                    experts(torch.zeros(128, 16, dtype=torch.bfloat16), torch.zeros(128, 2),
                            torch.ones(128, 2), expert_weights=weights)
        mock_create_resources.assert_not_called()

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    @patch.object(MegaMoeExperts, "_register_runtime_release", return_value=None)
    def test_shared_layers_create_once_and_close_after_last_owner(
        self,
        mock_register_runtime_release: Mock,
    ) -> None:
        """Feature: MegaMoe public API and resource lifecycle.

        Description: Keep parameters independent while one serial resource group is shared.
        Expectation: Serial owners share one resource while retaining their independent Parameters.
        """
        layers = [
            MegaMoeExperts(
                hidden_size=16,
                intermediate_size=8,
                num_experts=4,
                top_k=2,
            )
            for _ in range(4)
        ]
        resources = Mock()
        input_tensor = SimpleNamespace(device="npu:0", dtype="bfloat16")

        other = MegaMoeExperts(
            hidden_size=16, intermediate_size=8, num_experts=4, top_k=2,
            tuning_options=MegaMoeTuningOptions(dispatch_mode="pull"))
        self.addCleanup(other.close)
        with self.assertRaisesRegex(ValueError, "identical configuration"):
            MegaMoeExperts.share_execution_resources([layers[0], other])
        other.close()
        MegaMoeExperts.share_execution_resources(layers)
        shared_group = layers[0]._resource_group
        with patch.object(
            MegaMoeExperts,
            "_create_execution_resources",
            return_value=resources,
        ) as mock_create_resources:
            resolved = [
                layer._get_execution_resources(input_tensor) for layer in layers
            ]
        mock_register_runtime_release.assert_called_once()
        self.assertIs(mock_register_runtime_release.call_args.args[-1], resources)

        self.assertTrue(shared_group.shared)
        self.assertEqual(len(shared_group.members), len(layers))
        self.assertTrue(all(layer._resource_group is shared_group for layer in layers))
        self.assertTrue(all(value is resources for value in resolved))
        mock_create_resources.assert_called_once()
        self.assertTrue(mock_create_resources.call_args.kwargs["shared"])
        self.assertEqual(
            len(mock_create_resources.call_args.kwargs["active_specifications"]),
            1,
        )

        for layer in layers[:-1]:
            layer.close()
        resources.close.assert_not_called()
        layers[-1].close()
        layers[-1].close()
        resources.close.assert_called_once_with()

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_execution_resource_uses_manager_owned_runtime(self) -> None:
        """Feature: execution resource uses manager-owned Runtime.
        Description: Construct and close one execution resource twice.
        Expectation: Reserve and attach through the manager while close only removes the workspace.
        """
        root_group = object()
        bound_spec = SimpleNamespace(ep_group=root_group, ep_size=1, local_num_tokens=128,
                                     dispatch_mode="push", max_local_num_tokens=None,
                                     dynamic_tokens=False, auto_tokens=False)
        workspace = Mock()
        manager = Mock(root_group=root_group, heap_bytes=1024, epoch=0, workspaces=[], growth_records=[])
        manager.reserve.return_value = ({},)
        manager.attach_runtime.return_value = True
        manager.bind.return_value = 128
        close_order = []
        manager.close_workspace.side_effect = lambda _workspace: close_order.append("workspace")

        with (
            patch.object(
                mega_moe_module,
                "bind_mega_moe_spec",
                return_value=bound_spec,
            ),
            patch.object(mega_moe_module, "_get_heap_manager", return_value=manager),
            patch.object(
                mega_moe_module, "build_mega_moe_plan", return_value=object()
            ) as mock_build_plan,
            patch.object(mega_moe_module, "MegaMoeWorkspace", return_value=workspace),
            patch.object(_automatic._CLEANUP, "root_group", object()),
            patch.object(_collective, "exchange", wraps=_collective.exchange) as exchange,
        ):
            resources = mega_moe_module._MegaMoeExecutionResources(  # pylint: disable=protected-access
                {},
                SimpleNamespace(device="npu:0", dtype=torch.bfloat16, element_size=lambda: 2),
                expert_weight_layout="EIO",
                shared=False,
                active_specifications=(),
            )
            manager.reserve.assert_called_once()
            manager.attach_runtime.assert_called_once_with()
            exchange.assert_called_once()
            self.assertIs(exchange.call_args.kwargs["group"], root_group)
            mock_build_plan.assert_called_once_with(bound_spec, "npu:0")
            resources.close()
            resources.close()

        manager.close_workspace.assert_called_once_with(workspace)
        manager.detach_runtime.assert_not_called()
        self.assertEqual(close_order, ["workspace"])

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_execution_resource_construction_failure_rolls_back_manager_state(self) -> None:
        """Feature: construction failure rolls back manager state.
        Description: Fail plan creation after attaching the sole Runtime reference.
        Expectation: Remove new reservations and detach a newly acquired unused Runtime.
        """
        root_group = object()
        bound_spec = SimpleNamespace(ep_group=root_group, ep_size=1, local_num_tokens=128,
                                     dispatch_mode="push", max_local_num_tokens=None,
                                     dynamic_tokens=False, auto_tokens=False)

        manager = Mock(heap_bytes=1024, epoch=0, workspaces=[])
        manager.reserve.return_value = ({},)
        manager.attach_runtime.return_value = True
        with (
            patch.object(
                mega_moe_module,
                "bind_mega_moe_spec",
                return_value=bound_spec,
            ),
            patch.object(mega_moe_module, "_get_heap_manager", return_value=manager),
            patch.object(
                mega_moe_module,
                "build_mega_moe_plan",
                side_effect=RuntimeError("plan failed"),
            ),
            self.assertRaisesRegex(RuntimeError, "plan failed"),
        ):
            mega_moe_module._MegaMoeExecutionResources(  # pylint: disable=protected-access
                {},
                SimpleNamespace(device="npu:0", dtype=torch.bfloat16, element_size=lambda: 2),
                expert_weight_layout="EIO",
                shared=False,
                active_specifications=(),
            )

        manager.rollback_reservations.assert_called_once()
        manager.detach_runtime.assert_called_once_with()

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_construction_failure_after_restart_preserves_committed_heap(self) -> None:
        """Feature: MegaMoe public API and resource lifecycle.

        Description: Roll back only the new reservation after a successful Heap restart.
        Expectation: A failed lazy bind preserves the previously committed heap and its existing Runtime user.
        """
        root_group = object()
        specification = {"name": "new resource"}
        bound_spec = SimpleNamespace(ep_group=root_group, ep_size=1, local_num_tokens=128,
                                     dispatch_mode="push", max_local_num_tokens=None,
                                     dynamic_tokens=False, auto_tokens=False)
        growth_record = {"epoch": 1, "heap_bytes": 4096}
        manager = Mock(heap_bytes=2048, epoch=0, workspaces=[object()], growth_records=[])
        manager.attach_runtime.return_value = False

        def reserve(*_args: object, **_kwargs: object) -> tuple[dict]:
            """Simulate a committed restart while adding the new reservation."""
            manager.heap_bytes = 4096
            manager.epoch = 1
            manager.growth_records.append(growth_record)
            return (specification,)

        manager.reserve.side_effect = reserve
        with (
            patch.object(mega_moe_module, "bind_mega_moe_spec", return_value=bound_spec),
            patch.object(mega_moe_module, "_workspace_layout", return_value=object()),
            patch.object(mega_moe_module, "_get_heap_manager", return_value=manager),
            patch.object(mega_moe_module, "build_mega_moe_plan", side_effect=RuntimeError("plan failed")),
            self.assertRaisesRegex(RuntimeError, "plan failed"),
        ):
            mega_moe_module._MegaMoeExecutionResources(  # pylint: disable=protected-access
                specification,
                SimpleNamespace(device="npu:0", dtype=torch.bfloat16, element_size=lambda: 2),
                expert_weight_layout="EIO",
                shared=False,
                active_specifications=(specification,),
            )

        manager.rollback_reservations.assert_called_once_with((specification,), None)
        manager.detach_runtime.assert_not_called()
        self.assertEqual((manager.heap_bytes, manager.epoch), (4096, 1))
        self.assertEqual(manager.growth_records, [growth_record])

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_workspace_close_failure_is_retryable(self) -> None:
        """Feature: MegaMoe public API and resource lifecycle.

        Description: Keep the resource open until manager-owned workspace close succeeds.
        Expectation: A failed close remains open and succeeds on a later retry.
        """
        resources = mega_moe_module._MegaMoeExecutionResources.__new__(  # pylint: disable=protected-access
            mega_moe_module._MegaMoeExecutionResources  # pylint: disable=protected-access
        )
        resources.workspace = Mock()
        resources._workspace_closed = False  # pylint: disable=protected-access
        resources.heap_manager = Mock()
        resources._plans = OrderedDict()
        resources._closed = False  # pylint: disable=protected-access

        resources.heap_manager.close_workspace.side_effect = RuntimeError("close failed")
        with self.assertRaisesRegex(RuntimeError, "close failed"):
            resources.close()

        self.assertFalse(resources._closed)  # pylint: disable=protected-access

        resources.heap_manager.close_workspace.side_effect = None
        resources.close()
        self.assertEqual(resources.heap_manager.close_workspace.call_count, 2)


if __name__ == "__main__":
    unittest.main()
