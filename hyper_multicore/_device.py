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
"""Cached device capabilities shared by execution and placement kernels."""
import ctypes
from functools import lru_cache

import torch

from hyper_multicore._loader import query_npu_arch, validate_native_compatibility


@lru_cache(maxsize=None)
def device_ub_bytes(device_index: int) -> int | None:
    """Look up this visible device's packaged UB size without initializing CANN tiling."""
    capacities = validate_native_compatibility()["build_info"].get("device_ub_bytes")
    if not capacities:
        return None
    by_soc = capacities.get(str(query_npu_arch(device_index)))
    if not by_soc:
        return None
    name = ctypes.CDLL("libascendcl.so", mode=ctypes.RTLD_LOCAL).aclrtGetSocName
    name.argtypes = []
    name.restype = ctypes.c_char_p
    with torch.npu.device(device_index):
        soc = name()
    return by_soc.get(soc.decode("ascii", errors="replace").lower()) if soc else None


def device_core_limit(device: torch.device, name: str) -> int:
    """Read a positive core limit from the active NPU runtime configuration."""
    index = device.index if device.index is not None else torch.npu.current_device()
    cores = torch.npu.get_device_limit(index).get(name)
    if not isinstance(cores, int) or isinstance(cores, bool) or cores <= 0:
        raise RuntimeError(
            f"torch.npu.get_device_limit() returned an invalid {name} "
            f"for NPU {index}: {cores!r}."
        )
    return cores
