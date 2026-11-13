#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Private SHMEM SDK preparation for Multicore builds.

function hp_prepare_shmem_sdk() {
    local project_root=$1
    local soc_list=$2
    local soc_type
    local device_arch
    local selection
    selection=$("${PYTHON_BIN:-python}" "${project_root}/hyper_multicore/_build/soc_selection.py" \
        "${soc_list}" --groups) || return 10
    if [[ "${selection}" == *$'\n'* ]]; then
        echo "ERROR: Prepare one SHMEM backend at a time." >&2
        return 10
    fi
    read -r device_arch soc_type _ _ soc_list <<< "${selection}"
    local native_jobs=$3
    local native_root="${project_root}/build/native"
    local python_bin="${PYTHON_BIN:-python}"
    local locked_source_dir
    local locked_commit
    local prepared_dependency
    local wrapper="${project_root}/hyper_multicore/shmem/cmake/shmem_wrapper"

    if [[ -z "${ASCEND_HOME_PATH:-}" || ! -d "${ASCEND_HOME_PATH}" ]]; then
        echo "HP_NATIVE_REASON_CODE=CANN_ENV_NOT_CONFIGURED"
        echo "ERROR: ASCEND_HOME_PATH must identify the selected CANN installation." >&2
        echo "       Source the required CANN set_env.sh before invoking the native build." >&2
        return 3
    fi
    local cann_version_file="${ASCEND_HOME_PATH}/opp/version.info"
    if ! command -v "${python_bin}" >/dev/null 2>&1; then
        echo "HP_NATIVE_REASON_CODE=PYTHON_BUILD_DEPENDENCY_NOT_FOUND"
        echo "ERROR: The selected Python executable is unavailable: ${python_bin}." >&2
        return 4
    fi
    python_bin=$(command -v "${python_bin}")
    local host_arch
    host_arch=$(uname -m)
    local sdk_work_root="${HP_SHMEM_WORK_ROOT:-${native_root}/sdk/${host_arch}/${device_arch}}"
    if ! "${python_bin}" - "${project_root}" "${sdk_work_root}" <<'PYWORKSPACE'
from pathlib import Path
import sys

sys.path.insert(0, str(Path(sys.argv[1]) / "hyper_multicore/_build"))
from _hyper_multicore_bootstrap import load_module

load_module("build_workspace").reset_workspace(Path(sys.argv[2]), Path(sys.argv[1]))
PYWORKSPACE
    then
        echo "HP_NATIVE_REASON_CODE=SHMEM_WORKSPACE_INVALID"
        echo "ERROR: SDK workspace must be owned by this checkout and contain no symlinked ancestors." >&2
        return 4
    fi
    for required_tool in awk bisheng cmake gcc g++ git make readlink sha256sum tar; do
        if ! command -v "${required_tool}" >/dev/null 2>&1; then
            echo "HP_NATIVE_REASON_CODE=BUILD_TOOL_NOT_FOUND"
            echo "ERROR: Required SHMEM SDK build tool not found on PATH: ${required_tool}." >&2
            return 4
        fi
    done

    local cann_root
    local bisheng_path
    local gcc_path
    local gxx_path
    local shmem_lock_digest
    local toolchain_key
    local source_dir
    local build_dir
    local install_root
    cann_root=$(cd "${ASCEND_HOME_PATH}" && pwd -P)
    bisheng_path=$("${python_bin}" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' \
        "$(command -v bisheng)")
    gcc_path=$("${python_bin}" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' \
        "$(command -v gcc)")
    gxx_path=$("${python_bin}" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' \
        "$(command -v g++)")
    if ! shmem_lock_digest=$("${python_bin}" -c '
import hashlib
import json
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    dependencies = json.load(source)["components"]["multicore"]
lock = {"shmem": dependencies["shmem"]}
if sys.argv[2] == "Ascend950":
    lock["nlohmann_json"] = dependencies["nlohmann_json"]
encoded = json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()
print(hashlib.sha256(encoded).hexdigest())
' "${project_root}/hyper_multicore/_build/dependencies.lock.json" "${soc_type}"); then
        echo "HP_NATIVE_REASON_CODE=SHMEM_LOCK_INVALID"
        echo "ERROR: Cannot read the locked SHMEM dependency contract." >&2
        return 4
    fi
    toolchain_key=$(
        {
            printf '%s\n' \
                "schema=3" \
                "soc_type=${soc_type}" \
                "cann_root=${cann_root}" \
                "bisheng=${bisheng_path}" \
                "host_arch=${host_arch}" \
                "gcc=${gcc_path}" \
                "gxx=${gxx_path}" \
                "shmem_lock=${shmem_lock_digest}"
            "${bisheng_path}" --version 2>&1 || true
            "${gcc_path}" --version 2>&1 || true
            "${gxx_path}" --version 2>&1 || true
            if [[ -f "${cann_version_file}" ]]; then
                sha256sum "${cann_version_file}"
            else
                printf '%s\n' "cann_version_info=missing"
            fi
            sha256sum "${project_root}/hyper_multicore/cmake/hardening.cmake" \
                "${project_root}/hyper_multicore/shmem/cmake/shmem_wrapper/CMakeLists.txt" \
                "${project_root}/hyper_multicore/shmem/_build/shmem_sdk.sh"
        } | sha256sum | awk '{print substr($1, 1, 16)}'
    )
    if [[ ! "${toolchain_key}" =~ ^[0-9a-f]{16}$ ]]; then
        echo "HP_NATIVE_REASON_CODE=SHMEM_TOOLCHAIN_KEY_INVALID"
        echo "ERROR: Cannot derive a stable SHMEM CANN toolchain identity." >&2
        return 4
    fi
    source_dir="${sdk_work_root}/shmem-source"
    build_dir="${sdk_work_root}/shmem"
    install_root="${sdk_work_root}/shmem-install"

    if ! prepared_dependency=$("${python_bin}" "${project_root}/hyper_multicore/_build/prepare_dependencies.py" \
        --dependency shmem); then
        echo "HP_NATIVE_REASON_CODE=SHMEM_SOURCE_NOT_PREPARED"
        return 6
    fi
    locked_source_dir=$("${python_bin}" -c 'import json,sys; print(json.loads(sys.argv[1])["shmem"]["source_dir"])' \
        "${prepared_dependency}") || return 6
    locked_commit=$("${python_bin}" -c 'import json,sys; print(json.loads(sys.argv[1])["shmem"]["commit"])' \
        "${prepared_dependency}") || return 6

    mkdir -p "${source_dir}" "${build_dir}"
    if ! (set -o pipefail; git -C "${locked_source_dir}" archive --format=tar "${locked_commit}" \
        | tar -xf - -C "${source_dir}"); then
        echo "HP_NATIVE_REASON_CODE=SHMEM_SOURCE_NOT_PREPARED"
        return 6
    fi
    if [[ ! -d "${source_dir}/src" || ! -f "${source_dir}/CMakeLists.txt" ]]; then
        echo "HP_NATIVE_REASON_CODE=SHMEM_SOURCE_NOT_PREPARED"
        echo "ERROR: Locked SHMEM commit lacks SDK sources." >&2
        return 6
    fi
    if ! "${python_bin}" - "${project_root}" "${source_dir}" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

project_root, source_dir = (Path(value).resolve() for value in sys.argv[1:])
lock_path = project_root / "hyper_multicore/_build/dependencies.lock.json"
lock = json.loads(lock_path.read_text(encoding="utf-8"))["components"]["multicore"]["shmem"]
environment = os.environ.copy()
environment["GIT_CEILING_DIRECTORIES"] = str(source_dir.parent)
for adapter in lock.get("patches", []):
    adapter_path = project_root / adapter["path"]
    actual_hash = hashlib.sha256(adapter_path.read_bytes()).hexdigest()
    if actual_hash != adapter["sha256"]:
        raise ValueError(f"SHMEM adapter hash mismatch: {adapter_path}")
    for options in (("--check",), (), ("--check", "--reverse")):
        subprocess.run(
            ["git", "apply", *options, str(adapter_path)],
            cwd=source_dir,
            env=environment,
            check=True,
        )
PY
    then
        echo "HP_NATIVE_REASON_CODE=SHMEM_ADAPTER_FAILED"
        echo "ERROR: Cannot apply the locked SHMEM compatibility adapter." >&2
        return 6
    fi

    if [[ "${soc_type}" == Ascend950 ]]; then
        if ! prepared_dependency=$("${python_bin}" "${project_root}/hyper_multicore/_build/prepare_dependencies.py" \
            --dependency nlohmann_json); then
            echo "HP_NATIVE_REASON_CODE=SHMEM_SOURCE_NOT_PREPARED"
            return 6
        fi
        if ! "${python_bin}" - "${prepared_dependency}" "${source_dir}" <<'PY'
import json
from pathlib import Path
import subprocess
import sys

dependency = json.loads(sys.argv[1])["nlohmann_json"]
header = subprocess.check_output([
    "git", "-C", dependency["source_dir"], "show",
    dependency["commit"] + ":single_include/nlohmann/json.hpp",
])
target = Path(sys.argv[2]) / "3rdparty/json/single_include/nlohmann/json.hpp"
target.parent.mkdir(parents=True, exist_ok=True)
target.write_bytes(header)
PY
        then
            echo "HP_NATIVE_REASON_CODE=SHMEM_SOURCE_NOT_PREPARED"
            return 6
        fi
    fi

    echo "INFO: preparing pinned CANN SHMEM SDK"
    echo "  locked source: ${locked_source_dir}"
    echo "  isolated source: ${source_dir}"
    echo "  CANN toolchain: ${cann_root} (${bisheng_path})"
    echo "  host toolchain: ${host_arch} (${gcc_path}, ${gxx_path})"
    echo "  soc family: ${soc_type}"
    echo "  install root: ${install_root}"

    cmake -S "${wrapper}" \
        -B "${build_dir}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DHP_SHMEM_SOURCE_DIR="${source_dir}" \
        -DUSE_CXX11_ABI=ON \
        -DUSE_UNIT_TEST=OFF \
        -DUSE_EXAMPLES=OFF \
        -DBUILD_PYTHON=OFF \
        -DCMAKE_DISABLE_FIND_PACKAGE_MPI=TRUE \
        -DACLSHMEM_RDMA_SUPPORT=OFF \
        -DSOC_TYPE="${soc_type}"
    cmake --build "${build_dir}" --parallel "${native_jobs}"
    cmake --install "${build_dir}" --prefix "${install_root}/shmem"

    HP_SHMEM_INSTALL_ROOT="${install_root}"
    HP_SHMEM_TOOLCHAIN_KEY="${toolchain_key}"
    HP_SHMEM_SOC_TYPE="${soc_type}"
    export HP_SHMEM_INSTALL_ROOT HP_SHMEM_TOOLCHAIN_KEY HP_SHMEM_SOC_TYPE
}
