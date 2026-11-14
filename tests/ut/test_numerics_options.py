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
"""Check stable, immutable numerical options independently of the executor."""

import os
import subprocess
import sys
import unittest
from dataclasses import FrozenInstanceError, replace

from hyper_multicore import MegaMoeNumericsOptions
from tests.common.mark_utils import arg_mark


class TestMegaMoeNumericsOptions(unittest.TestCase):
    """Keep a model's numerical definition distinct from execution tuning."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_numerics_options_import_without_the_executor_or_npu_backend(self) -> None:
        """Load a numerical configuration without loading the executor or NPU backend."""
        code = """
import sys
from hyper_multicore import MegaMoeNumericsOptions
from hyper_multicore.modules.mega_moe.numerics import MegaMoeNumericsOptions as OwnedNumerics
assert MegaMoeNumericsOptions is OwnedNumerics
assert MegaMoeNumericsOptions.__module__ == "hyper_multicore.modules.mega_moe.numerics"
options = MegaMoeNumericsOptions()
assert options.swiglu_limit is None
assert options.clamp_after_silu is False
assert options.apply_probs_before_fc2 is False
assert 'torch_npu' not in sys.modules
assert 'hyper_multicore.modules.mega_moe.module' not in sys.modules
"""
        result = subprocess.run(
            [sys.executable, "-c", code], check=False, capture_output=True, text=True,
            env={**os.environ, "TORCH_DEVICE_BACKEND_AUTOLOAD": "0"},
        )
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_layer_specific_numerics_preserve_immutable_defaults(self) -> None:
        """Derive a layer's numerical configuration without mutating a shared default."""
        default = MegaMoeNumericsOptions()
        adjusted = replace(default, swiglu_limit=7.0, clamp_after_silu=True, apply_probs_before_fc2=True)
        self.assertEqual(default, MegaMoeNumericsOptions())
        self.assertEqual((adjusted.swiglu_limit, adjusted.clamp_after_silu, adjusted.apply_probs_before_fc2),
                         (7.0, True, True))
        for options in (default, adjusted):
            for name in ("swiglu_limit", "clamp_after_silu", "apply_probs_before_fc2"):
                with self.subTest(options=options, name=name), self.assertRaises(FrozenInstanceError):
                    setattr(options, name, getattr(options, name))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_tuning_and_internal_plan_fields_are_not_numerical_options(self) -> None:
        """Keep the stable numerical object free of scheduling and unfinished gradient policies."""
        for name in ("dispatch_mode", "initial_capacity_factor", "capacity_growth_factor", "dispatch_split",
                     "combine_split", "weight_grad_dtype", "batch_experts", "nominal_schedule"):
            with self.subTest(name=name), self.assertRaisesRegex(TypeError, name):
                MegaMoeNumericsOptions(**{name: True})
