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
"""Shared environment helpers for multicore system tests."""

import os
import shutil
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CANN_SET_ENV = "/usr/local/Ascend/cann/set_env.sh"
HP_ACTIVATION_SCRIPT = "hyper_multicore_set_env.bash"
HP_ACTIVATION_VARIABLES = ("ASCEND_CUSTOM_OPP_PATH", "LD_LIBRARY_PATH")
INHERITED_RANK_VARIABLES = ("RANK_TABLE_FILE", "RANK_ID", "RANK_SIZE")


def _active_multicore_vendor_roots() -> list[Path]:
    """Return packaged vendor roots activated in the current environment."""
    vendor_roots = [
        Path(value).resolve()
        for value in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(os.pathsep)
        if value and Path(value).name == "hyper_multicore_nn"
    ]
    library_roots = {
        Path(value).resolve()
        for value in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
        if value
    }
    return [
        vendor_root
        for vendor_root in vendor_roots
        if (vendor_root / "op_api" / "lib" / "libcust_opapi.so").is_file()
        and (vendor_root / "op_api" / "lib").resolve() in library_roots
        and (vendor_root.parents[1].parent / "shmem" / "lib").resolve() in library_roots
        and (vendor_root.parents[1].parent / "shmem" / "lib" / "shmem").resolve() in library_roots
    ]


def multicore_test_environment_is_active() -> bool:
    """Return whether the current process already has the packaged OPP paths."""
    return bool(_active_multicore_vendor_roots())


def multicore_adapter_is_available() -> bool:
    """Return whether an activated payload advertises its native adapter inventory.

    An existing manifest is considered available even when it is empty. The
    worker then loads the payload and reports corruption instead of turning a
    broken build into a skip.

    Returns:
        Whether any active multicore payload advertises native build resources.

    """
    for vendor_root in _active_multicore_vendor_roots():
        payload = vendor_root.parents[2]
        if (payload / "adapters/manifest.json").is_file() or (payload / "build_info.json").is_file():
            return True
    return False


def multicore_activation_scripts() -> list[Path]:
    """Return source-build and wheel activation candidates in priority order.

    System-test workers import the checkout through an explicit ``PYTHONPATH``.
    Prefer its matching native payload when it exists so a separately installed
    wheel cannot activate a different vendor tree for the source Python code.
    """
    scripts = []
    source_script = (
        PROJECT_ROOT
        / "build"
        / "native"
        / "payload"
        / "hyper_multicore"
        / "lib"
        / "set_env.bash"
    )
    if source_script.is_file() and source_script not in scripts:
        scripts.append(source_script)
    installed_script = shutil.which(HP_ACTIVATION_SCRIPT)
    if installed_script and Path(installed_script) not in scripts:
        scripts.append(Path(installed_script))
    return scripts


def _activation_vendor_root(activation_script: Path) -> Path | None:
    """Return the directly owned vendor root for a payload activation script."""
    vendor_root = activation_script.parent / "vendors" / "hyper_multicore_nn"
    return vendor_root.resolve() if vendor_root.is_dir() else None


def prepare_multicore_test_environment() -> None:
    """Validate CANN and activate the wheel or PYTHONPATH multicore environment."""
    required_cann_variables = ("ASCEND_HOME_PATH", "ASCEND_OPP_PATH", "ASCEND_AICPU_PATH")
    missing_cann_variables = [name for name in required_cann_variables if not os.environ.get(name)]
    if missing_cann_variables:
        missing_text = ",".join(missing_cann_variables)
        raise RuntimeError(
            f"[HP-NATIVE-CANN-ENV-NOT-ACTIVATED] missing={missing_text}. "
            f"Run before pytest: source {DEFAULT_CANN_SET_ENV}"
        )

    activation_scripts = multicore_activation_scripts()
    active_vendor_roots = _active_multicore_vendor_roots()
    preferred_vendor_root = (
        _activation_vendor_root(activation_scripts[0]) if activation_scripts else None
    )
    if active_vendor_roots and (
        preferred_vendor_root is None or preferred_vendor_root in active_vendor_roots
    ):
        return
    if not activation_scripts:
        raise RuntimeError(
            f"[HP-NATIVE-ACTIVATION-SCRIPT-NOT-FOUND] {HP_ACTIVATION_SCRIPT} is absent from PATH and the "
            "source-build payload is unavailable. Run ./build.sh, then activate the wheel or PYTHONPATH payload."
        )
    activation_errors = []
    for activation_script in activation_scripts:
        activation_command = f"source {activation_script}"
        result = subprocess.run(
            ["bash", "-c", 'source "$1" >/dev/null && env -0', "bash", str(activation_script)],
            check=False,
            capture_output=True,
        )
        if result.returncode != 0:
            error = result.stderr.decode("utf-8", errors="replace").strip()
            activation_errors.append(f"command={activation_command} error={error}")
            continue
        activated_environment = {}
        for entry in result.stdout.split(b"\0"):
            name, separator, value = entry.partition(b"=")
            if separator:
                activated_environment[name.decode("utf-8")] = value.decode("utf-8")
        for variable in HP_ACTIVATION_VARIABLES:
            if variable in activated_environment:
                os.environ[variable] = activated_environment[variable]
        if multicore_test_environment_is_active():
            return
        activation_errors.append(f"command={activation_command} error=vendor paths were not activated")

    raise RuntimeError(
        "[HP-NATIVE-OPP-ACTIVATION-FAILED] " + "; ".join(activation_errors)
    )


@contextmanager
def without_inherited_rank_environment() -> Iterator[None]:
    """Prevent local distributed cases from inheriting stale cluster rank metadata."""
    inherited = {
        variable: os.environ[variable]
        for variable in INHERITED_RANK_VARIABLES
        if variable in os.environ
    }
    for variable in INHERITED_RANK_VARIABLES:
        os.environ.pop(variable, None)
    try:
        yield
    finally:
        for variable in INHERITED_RANK_VARIABLES:
            os.environ.pop(variable, None)
        os.environ.update(inherited)
