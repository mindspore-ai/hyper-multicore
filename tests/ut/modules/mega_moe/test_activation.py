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
"""Public activation parameters and their resource registration contract."""

from dataclasses import FrozenInstanceError, replace
import pickle
from unittest import TestCase
from unittest.mock import patch

from hyper_multicore import MegaMoeExperts, MegaMoeNumericsOptions, SiTUGLUParameters, SwiGLUParameters
from hyper_multicore.modules import module as multicore_module


_EXPERT_OPTIONS = {"hidden_size": 16, "intermediate_size": 8, "num_experts": 4, "top_k": 2}


class TestActivation(TestCase):
    """Verify the activation interface without initializing NPU resources."""

    def test_parameters_are_immutable_picklable_and_replaceable(self):
        """Keep activation configs immutable and usable in resource manifests."""
        for parameters in (SwiGLUParameters(), SiTUGLUParameters(beta=7.0)):
            self.assertEqual(pickle.loads(pickle.dumps(parameters)), parameters)
            self.assertEqual(hash(replace(parameters)), hash(parameters))
            with self.assertRaises(FrozenInstanceError):
                parameters.unused = True
        original = SwiGLUParameters()
        self.assertEqual(replace(original, clamp_limit=3.0).clamp_limit, 3.0)
        self.assertIsNone(original.clamp_limit)
        with self.assertRaises(TypeError):
            SwiGLUParameters(1.0)  # pylint: disable=too-many-function-args
        with self.assertRaises(TypeError):
            SiTUGLUParameters()  # pylint: disable=missing-kwoa

    def test_parameter_boundaries(self):
        """Check SiTUGLU beta limits and the disabled linear branch."""
        for value in (True, '1', None, float('nan'), float('inf'), float('-inf'), 1e39):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    SiTUGLUParameters(beta=value)
                with self.assertRaises(ValueError):
                    SiTUGLUParameters(beta=1.0, linear_beta=value)
        for value in (0.0, -1.0, 1e-50):
            with self.assertRaises(ValueError):
                SiTUGLUParameters(beta=value)
        with self.assertRaises(ValueError):
            SiTUGLUParameters(beta=1.0, linear_beta=1e-50)
        for value in (0.0, -7.0, -1e39, 2.0):
            self.assertEqual(SiTUGLUParameters(beta=1.0, linear_beta=value).linear_beta, value)

    def test_default_and_explicit_swiglu_share_resources(self):
        """Match implicit and explicit defaults in shared execution resources."""
        ordinary = MegaMoeExperts(**_EXPERT_OPTIONS)
        self.addCleanup(ordinary.close)
        explicit = MegaMoeExperts(**_EXPERT_OPTIONS, activation=SwiGLUParameters())
        self.addCleanup(explicit.close)
        self.assertEqual(ordinary.activation, explicit.activation)
        MegaMoeExperts.share_execution_resources([ordinary, explicit])
        for name in ('swiglu_limit', 'clamp_after_silu'):
            self.assertFalse(hasattr(ordinary, name))
            with self.assertRaises(TypeError):
                MegaMoeNumericsOptions(**{name: True})
        for value in ('swiglu', {}, object(), MegaMoeNumericsOptions()):
            with self.assertRaisesRegex(TypeError, 'activation'):
                MegaMoeExperts(**_EXPERT_OPTIONS, activation=value)

    def test_mxfp8_accepts_pre_silu_clamp(self):
        """Expose the GroupQuant clamp through the normal executor constructor."""
        options = {**_EXPERT_OPTIONS, "hidden_size": 128, "intermediate_size": 64, "precision": "mxfp8"}
        activation = SwiGLUParameters(clamp_limit=0.5)
        experts = MegaMoeExperts(**options, activation=activation)
        self.addCleanup(experts.close)
        self.assertEqual(experts.activation, activation)

        with self.assertRaisesRegex(ValueError, "clamp only before SiLU"):
            MegaMoeExperts(**options, activation=replace(activation, clamp_after_silu=True))
        with self.assertRaisesRegex(ValueError, "before-FC2"):
            MegaMoeExperts(**options, activation=activation,
                           numerics_options=MegaMoeNumericsOptions(apply_probs_before_fc2=True))

    def test_unsupported_activation_fails_before_resource_registration(self):
        """Reject an unsupported activation before creating a resource group."""
        activation = SiTUGLUParameters(beta=3.0)
        with patch.object(multicore_module, '_RESOURCE_MANAGER') as manager:
            for precision in ('bf16', 'mxfp8'):
                with self.subTest(precision=precision):
                    with self.assertRaisesRegex(NotImplementedError, 'SiTUGLU execution is not implemented'):
                        MegaMoeExperts(**_EXPERT_OPTIONS, precision=precision, activation=activation)
            manager.create_group.assert_not_called()
