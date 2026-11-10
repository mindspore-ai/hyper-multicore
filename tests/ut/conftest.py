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
"""Allow multicore Host unit tests to collect without an installed NPU backend."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from unittest import mock

import pytest


def _unmocked_npu_call(*args, **kwargs):
    raise AssertionError("CPU multicore UT reached an NPU API without an explicit test mock")


def pytest_configure(config: pytest.Config) -> None:
    """Install fail-closed NPU placeholders only when the optional backend is absent.

    Args:
        config: Pytest configuration used to install a fail-closed CPU-only NPU stub.
    """
    # `python -m pytest` from tests/ must not import tests/torch as the framework.
    tests_directory = Path(__file__).resolve().parents[1]
    sys.path[:] = [path for path in sys.path if Path(path or ".").resolve() != tests_directory]
    torch = importlib.import_module("torch")

    if "torch_npu" in sys.modules or importlib.util.find_spec("torch_npu") is not None:
        return
    backend = ModuleType("torch_npu")
    npu = ModuleType("torch.npu")
    for name in ("synchronize", "current_device", "current_stream", "Event", "Stream", "stream", "get_device_name",
                 "get_device_limit", "is_current_stream_capturing"):
        setattr(npu, name, mock.Mock(side_effect=_unmocked_npu_call))
    backend.npu = npu
    for name in ("npu_moe_token_permute", "npu_moe_token_unpermute", "npu_moe_token_permute_grad_v2"):
        setattr(backend, name, mock.Mock(side_effect=_unmocked_npu_call))
    patcher = mock.patch.dict(sys.modules, {"torch_npu": backend})
    patcher.start()
    config.add_cleanup(patcher.stop)
    npu_patcher = mock.patch.object(torch, "npu", npu, create=True)
    npu_patcher.start()
    config.add_cleanup(npu_patcher.stop)
    tensor_patcher = mock.patch.object(torch.Tensor, "is_npu", property(lambda tensor: False), create=True)
    tensor_patcher.start()
    config.add_cleanup(tensor_patcher.stop)
