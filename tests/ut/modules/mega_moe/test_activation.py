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
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import torch

from hyper_multicore import MegaMoeExperts, MegaMoeNumericsOptions, SiTUGLUParameters, SwiGLUParameters
from hyper_multicore.modules import module as multicore_module
from hyper_multicore.modules.mega_moe import spec as mega_moe_spec


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

    def test_situglu_layers_share_only_with_matching_parameters(self):
        """Share a workspace across compatible SiTUGLU layers before execution."""
        options = {**_EXPERT_OPTIONS, "intermediate_size": 16}
        layers = [MegaMoeExperts(**options, activation=SiTUGLUParameters(beta=4.0, linear_beta=value))
                  for value in (25.0, 25.0, 20.0)]
        for layer in layers:
            self.addCleanup(layer.close)
        first, compatible, incompatible = layers
        with self.assertRaisesRegex(ValueError, "identical configuration"):
            MegaMoeExperts.share_execution_resources(layers)
        MegaMoeExperts.share_execution_resources((first, compatible))
        self.assertIs(first._resource_group, compatible._resource_group)  # pylint: disable=protected-access
        self.assertIsNot(first._resource_group, incompatible._resource_group)  # pylint: disable=protected-access

    def test_situglu_is_bound_to_bf16(self):
        """Preserve SiTUGLU parameters and reject MXFP8 before registration."""
        activation = SiTUGLUParameters(beta=3.0)
        experts = MegaMoeExperts(**{**_EXPERT_OPTIONS, 'intermediate_size': 16}, activation=activation)
        self.addCleanup(experts.close)
        self.assertEqual(experts.activation, activation)
        with patch.object(multicore_module, '_RESOURCE_MANAGER') as manager:
            with self.assertRaisesRegex(ValueError, 'SiTUGLU requires BF16'):
                MegaMoeExperts(**_EXPERT_OPTIONS, precision='mxfp8', activation=activation)
            manager.create_group.assert_not_called()

    def test_situglu_binds_on_arch22_and_arch35(self):
        """Use one BF16 specification for both supported device architectures."""
        activation = SiTUGLUParameters(beta=4, linear_beta=25)
        specification = {
            **_EXPERT_OPTIONS, "intermediate_size": 16, "activation_parameters": activation,
            "ep_group": None, "ep_size": 1, "local_num_tokens": 128,
            "initial_capacity_factor": None,
        }
        tensor = SimpleNamespace(device="npu:0", dtype=torch.bfloat16)
        for architecture, cores in ((2201, 24), (3510, 32)):
            with (self.subTest(architecture=architecture),
                  patch.object(mega_moe_spec, "query_npu_arch", return_value=architecture),
                  patch.object(mega_moe_spec, "get_device_core_spec",
                               return_value=SimpleNamespace(num_cube_cores=cores))):
                bound = mega_moe_spec.bind_mega_moe_spec(specification, tensor)
                self.assertEqual(bound.architecture, architecture)
                self.assertEqual(bound.activation_parameters, activation)
                for linear_beta in (1e-40, 1e38):
                    invalid = {**specification, "activation_parameters":
                               SiTUGLUParameters(beta=4, linear_beta=linear_beta)}
                    with self.assertRaisesRegex(ValueError, "linear_beta and its reciprocal"):
                        mega_moe_spec.bind_mega_moe_spec(invalid, tensor)
