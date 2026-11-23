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
"""Load the private HotReplica communication and count kernels."""

import ctypes
import importlib
import importlib.util
from functools import lru_cache
from pathlib import Path
import sys
from types import ModuleType

from hyper_multicore._loader import select_native_shmem, validate_native_compatibility
from hyper_multicore._native_manifest import HOT_REPLICA_BF16_TARGETS, SHMEM_DEPENDENCIES


@lru_cache(maxsize=1)
def _selection():
    selected = validate_native_compatibility()
    backend, sdk = select_native_shmem(selected)
    _, key = HOT_REPLICA_BF16_TARGETS[backend]
    artifacts = selected["variant"].get(key)
    if artifacts is None:
        raise RuntimeError(f"Selected HMC adapter has no {backend} BF16 HotReplica capability")
    return selected["root"], sdk["shmem_root"], artifacts


def native_library_path(filename: str) -> Path:
    """Resolve HotReplica from the framework variant and active SHMEM backend."""
    root, _, artifacts = _selection()
    key = {
        "_hot_replica_shared_bridge.so": "bridge",
        "libhmc_hot_replica_kernels.so": "kernels",
    }[filename]
    return root / artifacts[key]


@lru_cache(maxsize=1)
def _load_kernels():
    """Load the selected private dependencies without initializing a SHMEM root."""

    root, sdk, _ = _selection()
    root = root / sdk
    handles = tuple(ctypes.CDLL(str(root / name), mode=ctypes.RTLD_LOCAL) for name in SHMEM_DEPENDENCIES)
    return handles, ctypes.CDLL(str(native_library_path("libhmc_hot_replica_kernels.so")), mode=ctypes.RTLD_GLOBAL)


def load_native_extension(name: str) -> ModuleType:
    """Load only the matching, integrity-validated framework host bridge."""
    module_name = __package__ + "." + name
    library = native_library_path(name + ".so").resolve()
    cached = sys.modules.get(module_name)
    if cached is not None:
        if Path(getattr(cached, "__file__", "")).resolve() != library:
            raise RuntimeError(f"HotReplica extension already loaded from another artifact: {module_name}")
        return cached
    _load_kernels()
    spec = importlib.util.spec_from_file_location(module_name, library)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load HotReplica extension: {module_name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[module_name] = module
    return module
