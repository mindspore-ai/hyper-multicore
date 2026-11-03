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
"""Locate and load the component-owned multicore payload."""

from __future__ import annotations

import ctypes
from importlib import metadata, util
import json
from functools import lru_cache
import importlib
import os
from pathlib import Path
import platform
import sys
from types import ModuleType
from contextlib import contextmanager
from threading import RLock

from hyper_multicore._native_manifest import check_integrity, read_manifest, resource_path, select_variant

_VENDOR_NAME = "hyper_multicore_nn"
_NATIVE_LOCK = RLock()
_selection = None
_native_failure = None
_loaded_kinds = set()


class NativeComponentUnavailableError(RuntimeError):
    """Report an unavailable or incorrectly activated multicore component."""


def diagnostic_context() -> str:
    """Read best-effort environment/build metadata without importing either framework."""
    actual = {"python": sys.executable, "python_abi": f"cp{sys.version_info.major}{sys.version_info.minor}",
              "host_arch": platform.machine(), "cann_version": "unknown"}
    for package in ("torch", "torch-npu"):
        try:
            actual[package] = metadata.version(package)
        except Exception:  # Diagnostics must not replace the original import/load failure.
            actual[package] = "unavailable"
    try:
        for line in (Path(os.environ["ASCEND_HOME_PATH"]) / "opp/version.info").read_text().splitlines():
            key, separator, value = line.partition("=")
            if separator and key.strip() == "Version":
                actual["cann_version"] = value.strip()
                break
    except Exception:
        pass
    try:
        info = json.loads((_component_root().parent / "build_info.json").read_text())
        matrix = [{key: variant.get(key) for key in ("id", "torch_version", "pta_version", "torch_cxx11_abi")}
                  for variant in info.get("variants", [])]
    except Exception:
        matrix = "unavailable"
    return f"runtime={actual} packaged_matrix={matrix}"


@lru_cache(maxsize=1)
def load_torch_npu() -> ModuleType:
    """Register the optional NPU backend only when an operation needs it.

    Import errors propagate so an unusable backend cannot masquerade as successful
    native initialization. Failed imports are not cached and may be retried.
    """
    try:
        metadata.version("torch-npu")
    except metadata.PackageNotFoundError as error:
        raise NativeComponentUnavailableError(
            f"[HYPER-MULTICORE-PTA-NOT-INSTALLED] Python={sys.executable}; install torch-npu "
            f"for your Torch/CANN environment. Inspect hyper_multicore_build_info for packaged branches. {diagnostic_context()}"
        ) from error
    try:
        return importlib.import_module("torch_npu")
    except (ImportError, OSError, RuntimeError) as error:
        raise NativeComponentUnavailableError(
            f"[HYPER-MULTICORE-PTA-IMPORT-FAILED] {diagnostic_context()}: {error}"
        ) from error


def require_multicore_environment() -> Path:
    """Require the packaged custom OPP environment prepared by ``set_env.bash``."""
    component_root = _component_root()
    vendor_root = component_root / "vendors" / _VENDOR_NAME
    if not vendor_root.is_dir():
        raise NativeComponentUnavailableError(
            f"[HP-NATIVE-PAYLOAD-MISSING] component=multicore vendor={vendor_root}. "
            "Inspect the current build log; for source/PYTHONPATH development, run "
            "./build.sh."
        )
    vendor_root = vendor_root.resolve()
    op_api_root = (vendor_root / "op_api" / "lib").resolve()
    shmem_root = (component_root.parent / "shmem" / "lib").resolve()
    required_library_roots = (op_api_root, shmem_root, shmem_root / "shmem")
    missing_variables = []
    if not _environment_contains_path("ASCEND_CUSTOM_OPP_PATH", vendor_root):
        missing_variables.append("ASCEND_CUSTOM_OPP_PATH")
    if not all(_environment_contains_path("LD_LIBRARY_PATH", path) for path in required_library_roots):
        missing_variables.append("LD_LIBRARY_PATH")
    if missing_variables:
        set_env_script = component_root / "set_env.bash"
        missing_text = ",".join(missing_variables)
        if any(module in sys.modules for module in ("torch", "torch_npu")):
            raise NativeComponentUnavailableError(
                "[HP-NATIVE-OPP-ACTIVATION-TOO-LATE] component=multicore "
                f"missing={missing_text}. Torch/torch_npu has already been imported; "
                f"exit the current Python process, run source {set_env_script}, and start a new process."
            )
        raise NativeComponentUnavailableError(
            "[HP-NATIVE-OPP-NOT-ACTIVATED] component=multicore "
            f"missing={missing_text}. Run source {set_env_script} before starting Python with multicore."
        )
    return vendor_root.resolve()


def get_multicore_paths() -> tuple[Path, Path]:
    """Return the vendor root and shared Torch adapter after ABI validation."""
    vendor_root = require_multicore_environment()
    selected = validate_native_compatibility()
    return vendor_root, resource_path(selected["root"], selected["variant"]["torch_so"])


def _runtime_identity() -> dict:
    """Read branch and ABI selectors without comparing external binary fingerprints."""
    try:
        pta = metadata.version("torch-npu")
    except metadata.PackageNotFoundError as error:
        raise NativeComponentUnavailableError(
            f"[HYPER-MULTICORE-PTA-NOT-INSTALLED] Python={sys.executable}; "
            "install matching Torch/PTA/CANN; inspect hyper_multicore_build_info."
        ) from error
    import torch

    actual = {"torch_version": metadata.version("torch"), "pta_version": pta,
              "python_abi": f"cp{sys.version_info.major}{sys.version_info.minor}",
              "host_arch": platform.machine(), "torch_cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
              "cann_version": "unknown"}
    try:
        version_file = Path(os.environ["ASCEND_HOME_PATH"]) / "opp/version.info"
        for line in version_file.read_text().splitlines():
            name, separator, value = line.partition("=")
            if separator and name.strip() == "Version":
                actual["cann_version"] = value.strip()
                break
    except (OSError, KeyError):
        pass
    return actual


def validate_native_compatibility() -> dict:
    """Choose one internally validated adapter pair; external post/CANN values do not gate it."""
    global _selection
    actual = _runtime_identity()  # Import the framework outside our native-registration lock.
    with _NATIVE_LOCK:
        if _selection is not None:
            return _selection
        root = _component_root().parent
        try:
            info = json.loads((root / "build_info.json").read_text(encoding="utf-8"))
            manifest = read_manifest(root)
            check_integrity(root, info, manifest)
            variant = select_variant(manifest, actual)
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            raise NativeComponentUnavailableError(
                f"[HYPER-MULTICORE-ABI-MISMATCH] root={root} runtime={actual}: {error}"
            ) from error
        _selection = {"root": root, "variant": variant, "actual": actual, "build_info": info}
        return _selection


def native_ready(kind: str) -> bool:
    """Check terminal state even when a previously loaded adapter takes its fast path."""
    with _NATIVE_LOCK:
        if _native_failure is not None:
            raise NativeComponentUnavailableError(
                f"[HYPER-MULTICORE-INITIALIZATION-FAILED] Restart the process: {_native_failure}"
            ) from _native_failure
        return kind in _loaded_kinds


@contextmanager
def native_loading(kind: str):
    """Serialize initial registration and make partial native initialization failures terminal."""
    global _native_failure
    with _NATIVE_LOCK:
        if _native_failure is not None:
            raise NativeComponentUnavailableError(
                f"[HYPER-MULTICORE-INITIALIZATION-FAILED] Restart the process: {_native_failure}"
            ) from _native_failure
        if kind in _loaded_kinds:
            yield False
            return
        try:
            yield True
        except BaseException as error:
            _native_failure = error
            raise
        else:
            _loaded_kinds.add(kind)


def native_context() -> str:
    """Describe the fixed adapter and runtime without hiding the original loader failure."""
    if _selection is None:
        return diagnostic_context()
    record = next(v for v in _selection["build_info"]["variants"] if v["id"] == _selection["variant"]["id"])
    return f"runtime={_selection['actual']} adapter={_selection['variant']} built={record}"


def preload_vendor_library(vendor_root: Path) -> None:
    """Load the exact ACLNN library without exposing its dependency symbols globally."""
    library_path = vendor_root / "op_api" / "lib" / "libcust_opapi.so"
    try:
        # CANN libraries can export identically named globals. Global promotion
        # can bind their independent destructors to the same object at exit.
        # ACLNN entry points are resolved through the vendor library handle.
        ctypes.CDLL(str(library_path), mode=ctypes.RTLD_LOCAL)
    except OSError as error:
        raise NativeComponentUnavailableError(
            f"[HP-NATIVE-VENDOR-LOAD-FAILED] library={library_path} {native_context()} error={error}."
        ) from error


def _component_root() -> Path:
    """Return the installed or source-build multicore payload root."""
    packaged = Path(__file__).resolve().parent / "lib"
    if packaged.is_dir():
        return packaged
    repository_root = Path(__file__).resolve().parents[1]
    if (repository_root / "setup.py").is_file():
        source_build = (
            repository_root / "build" / "native" / "payload" / "hyper_multicore" / "lib"
        )
        if source_build.is_dir():
            return source_build
    return packaged


def _environment_contains_path(variable: str, expected: Path) -> bool:
    """Return whether an environment path list contains ``expected``."""
    configured = [value for value in os.environ.get(variable, "").split(os.pathsep) if value]
    return str(expected.resolve()) in {str(Path(value).resolve()) for value in configured}
