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

"""Unit tests for native Torch adapter registration diagnostics."""

from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


from hyper_multicore._loader import NativeComponentUnavailableError
from hyper_multicore import _loader
from hyper_multicore.torch import ops
from tests.common.mark_utils import arg_mark


# Registration-cache and alias-ABI checks require Torch's private operator schema and loader.
class TestTorchOps(unittest.TestCase):
    """Native initialization is cached, but Torch itself is imported normally."""

    # pylint: disable=protected-access

    def setUp(self) -> None:
        """Isolate backend registration from the native adapter schema checks."""
        _loader._loaded_kinds.clear()
        _loader._native_failure = None
        patcher = patch.object(ops, "load_torch_npu")
        self.backend = patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self) -> None:
        """Never retain a mocked native-registration cache between tests."""
        _loader._loaded_kinds.clear()
        _loader._native_failure = None

    @staticmethod
    def _backward_op(alias: str = "a") -> SimpleNamespace:
        """Build a real schema for the mutable receive and expert input-gradient slots."""
        middle = ", ".join(f"Tensor input_{index}" for index in range(1, 12))
        schema = ops.torch._C.parse_schema(
            f"mega_moe_grad(Tensor(a!) dispatch, {middle}, Tensor({alias}!) gate_dx) -> ()"
        )
        return SimpleNamespace(default=SimpleNamespace(_schema=schema))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_adapter_failure_reports_original_cause(self):
        """Feature: Native adapter load diagnostics.
        Description: Register the backend and fail the actual adapter library load.
        Expectation: Preserve the failed library and original ABI error.
        """
        with (
            patch.object(ops, "get_multicore_paths", return_value=(Path("vendor"), Path("bad.so"))),
            patch.object(ops, "preload_vendor_library"),
            patch.object(ops.torch.ops, "load_library", side_effect=OSError("bad ABI")),
            self.assertRaisesRegex(NativeComponentUnavailableError, "ADAPTER-LOAD-FAILED.*bad ABI"),
        ):
            ops._load_native()

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_successful_registration_is_cached(self):
        """Feature: Cached native registration.
        Description: Load the backend and adapter twice, then reject mismatched native schemas.
        Expectation: Register once per cache lifetime and retain transport and alias validation.
        """
        with (
            patch.object(ops, "get_multicore_paths", return_value=(Path("vendor"), Path("good.so"))),
            patch.object(ops, "preload_vendor_library") as preload,
            patch.object(ops.torch.ops, "load_library") as load,
            patch.object(ops.torch.ops.hyper_multicore, "mega_moe_transport_version",
                         return_value=1, create=True) as version,
            patch.object(ops.torch.ops.hyper_multicore, "mega_moe_grad", self._backward_op(), create=True) as backward,
        ):
            ops._load_native()
            route_inputs = tuple(object() for _ in range(6))
            grad_inputs = tuple(object() for _ in range(8))
            with (
                patch.object(ops.torch.ops.hyper_multicore, "mega_gate_route", create=True) as route,
                patch.object(ops.torch.ops.hyper_multicore, "mega_gate_route_grad", create=True) as grad,
            ):
                self.assertIs(ops._mega_gate_route(
                    *route_inputs, top_k=6, routed_scaling_factor=1.0, use_vision_bias=False,
                ), route.return_value)
                self.assertIs(ops._mega_gate_route_grad(
                    *grad_inputs, top_k=6, routed_scaling_factor=1.0,
                ), grad.return_value)
                route.assert_called_once_with(*route_inputs, 6, 1.0, False)
                grad.assert_called_once_with(*grad_inputs, 6, 1.0)
            ops._load_native()
            self.backend.assert_called_once_with()
            preload.assert_called_once_with(Path("vendor"))
            load.assert_called_once_with("good.so")
            for protocol, alias, message in ((0, "a", "transport ABI"), (1, "e", "storage reuse")):
                _loader._loaded_kinds.clear()
                _loader._native_failure = None
                version.return_value = protocol
                backward.default._schema = self._backward_op(alias).default._schema
                with self.assertRaisesRegex(NativeComponentUnavailableError, message):
                    ops._load_native()

    def test_import_and_gate_bridge_keep_shared_adapter_lazy(self):
        """Importing the shared wrappers and selecting them must not load native code."""
        code = """
import torch
import torch_npu
from unittest.mock import patch
with patch.object(torch.ops, 'load_library') as load:
    from hyper_multicore.torch import ops
    from hyper_multicore.modules.mega_gate import function
    assert function._native_ops() is ops
    assert hasattr(ops, '_mega_gate_route') and hasattr(ops, '_mega_gate_route_grad')
    load.assert_not_called()
"""
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
