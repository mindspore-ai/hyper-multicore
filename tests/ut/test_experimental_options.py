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
"""Check the immutable customer-facing experimental tuning options."""

import unittest
from dataclasses import FrozenInstanceError, replace

from hyper_multicore.experimental import MegaMoeTuningOptions
from tests.common.mark_utils import arg_mark


class TestMegaMoeTuningOptions(unittest.TestCase):
    """Keep tuning overrides immutable and separate from model mathematics."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_layer_specific_options_do_not_mutate_shared_defaults(self) -> None:
        """Create an independent override without changing an existing layer's options."""
        default = MegaMoeTuningOptions()
        tuned = replace(default, dispatch_mode="pull", dispatch_split=256)
        self.assertEqual((default.dispatch_mode, default.dispatch_split), ("push", 128))
        self.assertEqual((tuned.dispatch_mode, tuned.dispatch_split), ("pull", 256))
        for options in (default, tuned):
            with self.subTest(options=options), self.assertRaises(FrozenInstanceError):
                options.dispatch_mode = "push"

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_model_semantics_and_low_level_plan_fields_are_not_tuning_options(self) -> None:
        """Reject fields belonging to the model contract or the internal plan builder."""
        for name in ("swiglu_limit", "clamp_after_silu", "apply_probs_before_fc2",
                     "weight_grad_dtype", "batch_experts", "nominal_schedule"):
            with self.subTest(name=name), self.assertRaisesRegex(TypeError, name):
                MegaMoeTuningOptions(**{name: True})
