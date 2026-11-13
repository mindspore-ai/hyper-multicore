#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
NATIVE_ROOT="${PROJECT_ROOT}/build/native"
HOST_ARCH=$(uname -m)
COMMON_ROOT="${NATIVE_ROOT}/common/${HOST_ARCH}"
WORK_ROOT="${HP_SHMEM_WORK_ROOT:-${NATIVE_ROOT}/sdk/${HOST_ARCH}}"
COMPONENT_ROOT="${HP_COMPONENT_ROOT:-${COMMON_ROOT}/payload}"
SOC_LIST=""
NATIVE_JOBS="$(nproc)"

function show_help() {
    cat <<EOF
Usage:
  bash hyper_multicore/shmem/build.sh [OPTIONS]

Prepare private SDK/ops resources. Use the root build.sh for bindings and wheels.

Options:
  --soc-list VALUE    Comma-separated CANN SoC IDs. Default: ascend910b,ascend910_93,ascend950.
  --jobs VALUE        Parallel build jobs. Default: nproc.
  --clean             Accepted; selected SDK/ops workspaces are rebuilt on every run.
  -h, --help          Show this help message.
EOF
}

function fail() {
    local reason_code=$1
    local message=$2
    local exit_code=${3:-1}
    trap - ERR
    echo "HP_NATIVE_REASON_CODE=${reason_code}"
    echo "ERROR: ${message}" >&2
    exit "${exit_code}"
}

function require_value() {
    local option_name=$1
    local option_value=${2:-}
    if [[ -z "${option_value}" || "${option_value}" == --* ]]; then
        fail "INVALID_SELECTION" "${option_name} requires a value." 2
    fi
}


while [[ $# -gt 0 ]]; do
    case "$1" in
        --soc-list=*)
            require_value "--soc-list" "${1#*=}"
            SOC_LIST="${1#*=}"
            shift
            ;;
        --soc-list)
            require_value "$1" "${2:-}"
            SOC_LIST="$2"
            shift 2
            ;;
        --jobs=*)
            require_value "--jobs" "${1#*=}"
            NATIVE_JOBS="${1#*=}"
            shift
            ;;
        --jobs)
            require_value "$1" "${2:-}"
            NATIVE_JOBS="$2"
            shift 2
            ;;
        --clean)
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            fail "INVALID_SELECTION" "Unknown option '$1'." 2
            ;;
    esac
done

if ! [[ "${NATIVE_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
    fail "INVALID_SELECTION" "--jobs must be a positive integer, got '${NATIVE_JOBS}'." 2
fi
[[ "$(uname -s)" == Linux ]] || fail "UNSUPPORTED_HOST" "Native builds require Linux/CANN/PTA." 3


CURRENT_REASON_CODE="SHMEM_BUILD_FAILED"
PYTHON_BIN="${PYTHON_BIN:-python}"
function report_unhandled_error() {
    local exit_code=$?
    trap - ERR
    echo "HP_NATIVE_REASON_CODE=${CURRENT_REASON_CODE}"
    echo "ERROR: SHMEM build failed unexpectedly with exit ${exit_code}." >&2
    exit "${exit_code}"
}
trap report_unhandled_error ERR

SOC_ARGS=()
if [[ -n "${SOC_LIST}" ]]; then SOC_ARGS+=("${SOC_LIST}"); fi
SOC_SELECTION=$("${PYTHON_BIN}" "${PROJECT_ROOT}/hyper_multicore/_build/soc_selection.py" "${SOC_ARGS[@]}") ||
    fail "UNSUPPORTED_SOC_SELECTION" "Invalid --soc-list=${SOC_LIST}." 10
read -r SOC_LIST _ <<< "${SOC_SELECTION}"

if [[ "${HP_NATIVE_STAGE:-}" != common ]]; then
    exec "${PYTHON_BIN}" - "${PROJECT_ROOT}" "${SOC_LIST}" "${NATIVE_JOBS}" <<'PYSHMEM'
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys

root = Path(sys.argv[1])
native = root / "build/native"
native.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(root / "hyper_multicore/_build"))
from _hyper_multicore_bootstrap import load_module

workspace = load_module("build_workspace")
signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
try:
    with workspace.build_lock(native, "shmem"):
        common = native / "common" / platform.machine()
        workspace.check_workspace(common, root)
        workspace.claim_workspace(common, root)
        process = subprocess.Popen(
            ["bash", str(root / "hyper_multicore/shmem/build.sh"), "--soc-list", sys.argv[2], "--jobs", sys.argv[3]],
            env={**os.environ, "HP_NATIVE_STAGE": "common"}, start_new_session=True,
        )
        try:
            status = process.wait()
        finally:
            load_module("command_runner").stop_stages((process,))
except KeyboardInterrupt:
    sys.exit(130)
sys.exit(status)
PYSHMEM
fi

cd "${PROJECT_ROOT}"
rm -rf "${COMPONENT_ROOT}/shmem"
mkdir -p "${COMPONENT_ROOT}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    fail "PYTHON_BUILD_DEPENDENCY_NOT_FOUND" \
        "The selected Python executable is unavailable: ${PYTHON_BIN}." 4
fi
PYTHON_BIN=$(command -v "${PYTHON_BIN}")
for required_tool in cmake find gcc g++ grep make readelf readlink sed; do
    if ! command -v "${required_tool}" >/dev/null 2>&1; then
        fail "BUILD_TOOL_NOT_FOUND" \
            "Required SHMEM build tool not found on PATH: ${required_tool}." 4
    fi
done
source "${PROJECT_ROOT}/hyper_multicore/shmem/_build/shmem_sdk.sh"
SHMEM_SDK_MANIFEST="${HP_SHMEM_HANDOFF:-${COMMON_ROOT}/sdk.json}"
mkdir -p "$(dirname "${SHMEM_SDK_MANIFEST}")"
"${PYTHON_BIN}" -c 'import json,sys; from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps({"architectures": {}}))' \
    "${SHMEM_SDK_MANIFEST}"

function build_backend() {
    local backend=$1
    local sdk_soc_type=$2
    local compile_soc=$3
    local backend_socs=$4
    export HP_SHMEM_WORK_ROOT="${WORK_ROOT}/${backend}"
    CURRENT_REASON_CODE="SHMEM_SDK_BUILD_FAILED"
    hp_prepare_shmem_sdk "${PROJECT_ROOT}" "${compile_soc}" "${NATIVE_JOBS}"

    SHMEM_INSTALL_ROOT="${HP_SHMEM_INSTALL_ROOT}"
    SHMEM_SOURCE_DIR="${SHMEM_INSTALL_ROOT}/shmem"
    export SHMEM_HOME_PATH="${SHMEM_INSTALL_ROOT}"
    export SHMEM_SOURCE_DIR

    OPS_INSTALL_DIR="${COMPONENT_ROOT}/shmem"

    # Build the private SHMEM ops library against the prepared SDK.
    CCSRC_OPS_BUILD_DIR="${HP_SHMEM_WORK_ROOT}/ccsrc-ops"
    CURRENT_REASON_CODE="SHMEM_OPS_BUILD_FAILED"
    cmake -S "${PROJECT_ROOT}/hyper_multicore/shmem/ccsrc/ops" \
        -B "${CCSRC_OPS_BUILD_DIR}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DHP_SHMEM_SOC_TYPE="${HP_SHMEM_SOC_TYPE}" \
        -DHP_SHMEM_RESOURCE_ARCH="${backend}" \
        -DCMAKE_INSTALL_PREFIX="${OPS_INSTALL_DIR}"
    cmake --build "${CCSRC_OPS_BUILD_DIR}" --parallel "${NATIVE_JOBS}"
    cmake --install "${CCSRC_OPS_BUILD_DIR}"
    SHMEM_OPS_LIB="${OPS_INSTALL_DIR}/lib/${backend}/libhyper_multicore_shmem_ops.so"
    if [[ ! -s "${SHMEM_OPS_LIB}" ]]; then
        fail "EXPECTED_ARTIFACT_MISSING" \
            "SHMEM ops kernel library was not installed: ${SHMEM_OPS_LIB}." 9
    fi

    CURRENT_REASON_CODE="SHMEM_ELF_VALIDATION_FAILED"
    for private_library in \
        libhyper_multicore_shmem.so \
        libhyper_multicore_shmem_utils.so \
        aclshmem_bootstrap_uid.so \
        aclshmem_bootstrap_config_store.so; do
        private_library_path="${OPS_INSTALL_DIR}/lib/${backend}/shmem/${private_library}"
        if [[ ! -s "${private_library_path}" ]]; then
            fail "EXPECTED_ARTIFACT_MISSING" "Private SHMEM library is missing: ${private_library_path}." 11
        fi
        if ! readelf -d "${private_library_path}" | grep -Eq "\(SONAME\).*\[${private_library}\]"; then
            fail "SHMEM_PRIVATE_SONAME_INVALID" \
                "Private SHMEM SONAME does not match ${private_library}: ${private_library_path}." 11
        fi
    done

    SHMEM_UTILS_LIB="${OPS_INSTALL_DIR}/lib/${backend}/shmem/libhyper_multicore_shmem_utils.so"
    if ! readelf -Ws "${SHMEM_UTILS_LIB}" | grep -E '[[:space:]]hpshmem_log$' >/dev/null; then
        fail "SHMEM_PRIVATE_LOG_API_MISSING" \
            "Private SHMEM utility library does not export hpshmem_log: ${SHMEM_UTILS_LIB}." 11
    fi

    while IFS= read -r component_library; do
        dynamic_section=$(readelf -d "${component_library}")
        if grep -Eq '\(NEEDED\).*\[(libshmem|libshmem_utils|libshmem_bootstrap_[^]]*)\.so\]' \
            <<< "${dynamic_section}"; then
            fail "GENERIC_SHMEM_DT_NEEDED_FOUND" \
                "Component library depends on a generic SHMEM SONAME: ${component_library}." 11
        fi
        if grep -Eq '(RPATH|RUNPATH)' <<< "${dynamic_section}"; then
            fail "RUNPATH_FOUND" \
                "Component library must not contain RPATH/RUNPATH: ${component_library}." 11
        fi
    done < <(find "${OPS_INSTALL_DIR}/lib/${backend}" -type f -name '*.so' -print)

    "${PYTHON_BIN}" - "${SHMEM_SDK_MANIFEST}" "${backend}" "${HP_SHMEM_INSTALL_ROOT}" \
        "${HP_SHMEM_TOOLCHAIN_KEY}" "${backend_socs}" "${sdk_soc_type}" <<'PYHANDOFF'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
handoff = json.loads(path.read_text())
handoff["architectures"][sys.argv[2]] = {
    "install_root": sys.argv[3], "toolchain_key": sys.argv[4],
    "sdk_soc_type": sys.argv[6], "soc_list": sys.argv[5].split(","),
}
path.write_text(json.dumps(handoff))
PYHANDOFF
}

SHMEM_GROUPS=$("${PYTHON_BIN}" "${PROJECT_ROOT}/hyper_multicore/_build/soc_selection.py" --groups "${SOC_LIST}")
while read -r backend sdk_soc_type compile_soc _registered_socs backend_socs; do
    build_backend "${backend}" "${sdk_soc_type}" "${compile_soc}" "${backend_socs}"
done <<< "${SHMEM_GROUPS}"

trap - ERR
echo "INFO: SHMEM build completed: ${SOC_LIST}"
echo "  component: ${COMPONENT_ROOT}"
