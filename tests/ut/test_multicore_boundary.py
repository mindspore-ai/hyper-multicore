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

"""Verify component exports and absence of framework dispatch."""

import importlib
import os
import subprocess
import sys
import unittest
from pathlib import Path

import hyper_multicore as multicore
from tests.common.mark_utils import arg_mark


class TestMulticoreBoundary(unittest.TestCase):
    """Check the component contract without initializing a native runtime."""


    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_business_symbols_are_exported_by_standalone_package(self):
        """Verify the Multicore public export boundary.

        Feature: Multicore public exports.
        Description: Inspect the HyperMulticore public package symbols.
        Expectation: The package exports its business modules and diagnostic helpers.
        """
        self.assertCountEqual(multicore.__all__, [
            "MegaMhc", "MegaGate", "MegaMoeExperts", "MegaKernelDiagnostics",
            "prepare_mega_kernel_diagnostics", "validate_mega_kernel_diagnostic_contract",
        ])
        self.assertTrue(callable(multicore.prepare_mega_kernel_diagnostics))
        for name in ("MegaMhc", "MegaGate", "MegaMoeExperts"):
            self.assertTrue(callable(getattr(multicore, name)))
        for name in ("MhcState", "MegaMhcOutput", "GateOutput", "GateBackend"):
            self.assertFalse(hasattr(multicore, name))
        self.assertNotIn("hyper_multicore_shmem_torch", sys.modules)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_mhc_exposes_only_canonical_module_name(self) -> None:
        """Feature: Single public mHC module name.
        Description: Import all business classes with the NPU backend blocked, then inspect mHC exports.
        Expectation: CPU imports need no HCCL; backend failures propagate and the canonical mHC identity is preserved.
        """
        code = """
import importlib.abc
import importlib.util
import sys

class RejectNpuBackend(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch_npu":
            spec = importlib.util.spec_from_loader(fullname, self, is_package=True)
            spec.submodule_search_locations = ["/cpu-test-unavailable-torch-npu"]
            return spec
        return None

    def exec_module(self, module):
        raise ImportError("CPU import guard: libhccl.so unavailable")

sys.meta_path.insert(0, RejectNpuBackend())
if sys.argv[1] == "backward":
    backward = importlib.import_module("hyper_multicore.modules.mega_mhc_grad.function")
    assert callable(backward.hyper_mega_mhc_grad)
from hyper_multicore import MegaGate, MegaMhc, MegaMoeExperts
from hyper_multicore._loader import load_torch_npu, NativeComponentUnavailableError
assert MegaGate.__name__ == "MegaGate"
assert MegaMhc.__name__ == "MegaMhc"
assert MegaMoeExperts.__name__ == "MegaMoeExperts"
import torch
from hyper_multicore.torch import ops
router = MegaGate(hidden_size=4, num_experts=2, top_k=1, scoring_func="sigmoid", routed_scaling_factor=1.0)
assert router(torch.ones(1, 2, 4))[2].shape == (2, 1)
assert MegaMhc(128).norm_weight.shape == (128,)
experts = MegaMoeExperts(hidden_size=4, intermediate_size=4, num_experts=2, top_k=1)
try:
    experts(torch.ones(1, 2, 4, dtype=torch.bfloat16), torch.zeros(2, 1, dtype=torch.int32), torch.ones(2, 1))
except TypeError as error:
    assert "NPU hidden states" in str(error)
else:
    raise AssertionError("CPU expert execution must reject inputs before native initialization")
experts.close()
assert "torch_npu" not in sys.modules
for initialize in (ops._load_native,):
    try:
        initialize()
    except NativeComponentUnavailableError as error:
        assert "PTA-IMPORT-FAILED" in str(error)
        assert "libhccl.so" in str(error)
        assert isinstance(error.__cause__, ImportError)
    else:
        raise AssertionError("Backend registration must fail when HCCL is unavailable")
assert load_torch_npu.cache_info().currsize == 0
"""
        for order in ("forward", "backward"):
            result = subprocess.run(
                [sys.executable, "-c", code, order], check=False, capture_output=True, text=True,
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "TORCH_DEVICE_BACKEND_AUTOLOAD": "0"},
            )
            self.assertEqual(result.returncode, 0, msg=f"{order}: {result.stdout}{result.stderr}")
        package = importlib.import_module("hyper_multicore.modules.mega_mhc")
        implementation = importlib.import_module("hyper_multicore.modules.mega_mhc.module")
        for namespace in (multicore, package, implementation):
            with self.subTest(namespace=namespace.__name__):
                self.assertIs(namespace.MegaMhc, multicore.MegaMhc)
                self.assertNotIn("HyperMegaMhc", getattr(namespace, "__all__", ()))
                self.assertFalse(hasattr(namespace, "HyperMegaMhc"))
