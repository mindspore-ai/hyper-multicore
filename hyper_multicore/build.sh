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
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
source "${PROJECT_ROOT}/scripts/native/cann_version.sh"
BUILD_DIR="${PROJECT_ROOT}/build"
NATIVE_ROOT="${BUILD_DIR}/native"
COMMON_ROOT="${NATIVE_ROOT}/common/$(uname -m)"
WORK_ROOT="${HP_NATIVE_WORK_ROOT:-${COMMON_ROOT}/work}"
COMPONENT_ROOT="${HP_COMPONENT_ROOT:-${COMMON_ROOT}/payload}"
OUTPUT_ROOT="${COMPONENT_ROOT}/lib"
SOC_LIST=""
NATIVE_JOBS="$(nproc)"
CLEAN="off"
PHASE=""
DEVICE_ARCH=""
CURRENT_REASON_CODE="MULTICORE_BUILD_FAILED"
PYTHON_BIN="${PYTHON_BIN:-python}"

function show_help() {
    cat <<EOF
Usage:
  bash hyper_multicore/build.sh [OPTIONS]

Options:
  --soc-list VALUE    Comma-separated CANN SoC IDs. Default: ascend910b,ascend910_93,ascend950.
  --jobs VALUE        Parallel build jobs. Default: nproc.
  --clean             Remove multicore work and install outputs before building.
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

function report_unhandled_error() {
    local exit_code=$?
    trap - ERR
    echo "HP_NATIVE_REASON_CODE=${CURRENT_REASON_CODE}"
    echo "ERROR: multicore build failed unexpectedly with exit ${exit_code}." >&2
    exit "${exit_code}"
}
trap report_unhandled_error ERR

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
        --phase|--arch)
            require_value "$1" "${2:-}"
            if [[ "$1" == --phase ]]; then PHASE=$2; else DEVICE_ARCH=$2; fi
            shift 2
            ;;
        --stage-only)
            shift
            ;;
        --clean)
            CLEAN="on"
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

function validate_multicore_build_log() {
    local build_log=$1
    if grep -Eiv '^\[ERROR\] ATRACE\(.*get platform info failed, drvErr=4\.$' "${build_log}" | \
            grep -Eiq \
                '\[ERROR\]|ERROR REASON|CMake Error|fatal error:|:[0-9]+(:[0-9]+)?:[[:space:]]+error:|ninja: build stopped|(gmake|make)(\[[0-9]+\])?: \*\*\*'; then
        fail "CANN_VENDOR_BUILD_LOG_ERROR" \
            "Fatal compiler or CANN message found in ${build_log}." 12
    fi
}

function require_nonempty_artifact() {
    local artifact=$1
    local description=$2
    if [[ ! -s "${artifact}" ]]; then
        fail "CANN_VENDOR_ARTIFACT_MISSING" \
            "Missing or empty ${description}: ${artifact}." 12
    fi
}

function validate_shared_library_security() {
    local library=$1
    local dynamic_section
    local program_headers
    dynamic_section=$(readelf -d "${library}")
    program_headers=$(readelf -W -l "${library}")
    if grep -Eq '(RPATH|RUNPATH)' <<< "${dynamic_section}"; then
        fail "RUNPATH_FOUND" "Shared library must not contain RPATH/RUNPATH: ${library}." 20
    fi
    if ! grep -q 'GNU_RELRO' <<< "${program_headers}"; then
        fail "RELRO_MISSING" "Shared library is missing GNU_RELRO: ${library}." 20
    fi
    if ! grep -Eq '\(BIND_NOW\)|FLAGS[^]]*BIND_NOW|FLAGS_1[^]]*NOW' <<< "${dynamic_section}"; then
        fail "BIND_NOW_MISSING" "Shared library is missing BIND_NOW: ${library}." 20
    fi
    if grep -Eq 'GNU_STACK.*RWE' <<< "${program_headers}"; then
        fail "EXECUTABLE_STACK_FOUND" "Shared library has an executable stack: ${library}." 20
    fi
    if grep -Eq '\(TEXTREL\)' <<< "${dynamic_section}"; then
        fail "TEXT_RELOCATION_FOUND" "Shared library contains text relocations: ${library}." 20
    fi
    if readelf -W -S "${library}" | grep -q '\.symtab'; then
        fail "STATIC_SYMBOL_TABLE_FOUND" "Release shared library is not stripped: ${library}." 20
    fi
}

function validate_multicore_vendor() {
    local vendor_root=$1
    local soc_list=$2
    local build_log=${3:-}
    local kernel_layout=${4:-series}
    local kernel_id
    local library
    local dynamic_section
    local ldd_output
    local op_name
    local required_symbol
    local soc
    local artifact
    local symbol_table
    local -a forbidden_files=()
    local -a libraries=()
    local -a artifacts=()
    local -a validation_socs=()
    local -a required_symbols=(
        aclnnHyperMegaMoe
        aclnnHyperMegaMoeGetWorkspaceSize
        aclnnHyperMegaMoeGrad
        aclnnHyperMegaMoeGradGetWorkspaceSize
        aclnnHyperMegaMhc
        aclnnHyperMegaMhcGetWorkspaceSize
        aclnnHyperMegaGateRoute
        aclnnHyperMegaGateRouteGetWorkspaceSize
        aclnnHyperMegaGateRouteGrad
        aclnnHyperMegaGateRouteGradGetWorkspaceSize
    )

    if [[ -n "${build_log}" ]]; then
        validate_multicore_build_log "${build_log}"
    fi
    mapfile -t libraries < <(find "${vendor_root}" -type f -name 'libcust_opapi.so' -print)
    if [[ ${#libraries[@]} -ne 1 ]]; then
        fail "CANN_VENDOR_LIBRARY_INVALID" \
            "Expected one libcust_opapi.so under ${vendor_root}, found ${#libraries[@]}." 12
    fi
    library=${libraries[0]}
    symbol_table=$(nm -D --defined-only "${library}" | awk '{print $NF}')
    for required_symbol in "${required_symbols[@]}"; do
        if ! grep -Fxq "${required_symbol}" <<< "${symbol_table}"; then
            fail "CANN_VENDOR_SYMBOL_MISSING" \
                "Required symbol ${required_symbol} is missing from ${library}." 12
        fi
    done
    if grep -Eq 'aclnnMegaMoe|aclnnHyperMegaMoE|HyperMegaMoE' <<< "${symbol_table}"; then
        fail "CANN_VENDOR_SYMBOL_CONFLICT" \
            "Forbidden or case-conflicting MegaMoe symbol found in ${library}." 12
    fi
    mapfile -t forbidden_files < <(find "${vendor_root}" -type f \
        \( -name '*.cpp' -o -name '*.h' -o -name '*.ini' -o -name '*.json' \
           -o -name '*.py' -o -name '*.txt' \) \
        -exec grep -El 'aclnnMegaMoe|aclnnHyperMegaMoE|HyperMegaMoE' {} +)
    if [[ ${#forbidden_files[@]} -ne 0 ]]; then
        fail "CANN_VENDOR_IDENTITY_CONFLICT" \
            "Forbidden or case-conflicting MegaMoe identity found in ${forbidden_files[0]}." 12
    fi

    IFS=',' read -r -a validation_socs <<< "${soc_list}"
    for soc in "${validation_socs[@]}"; do
        kernel_id=${soc}
        if [[ "${kernel_layout}" == "architecture" ]]; then
            read -r kernel_id _ <<< "$("${PYTHON_BIN}" "${PROJECT_ROOT}/hyper_multicore/_build/soc_selection.py" --groups "${soc}")"
        fi
        for op_name in hyper_mega_moe hyper_mega_moe_grad hyper_mega_mhc hyper_mega_mhc_grad \
                hyper_mega_gate_route hyper_mega_gate_route_grad; do
            mapfile -t artifacts < <(
                find "${vendor_root}/op_impl/ai_core/tbe/kernel/${kernel_id}/${op_name}" \
                    -maxdepth 1 -type f -name '*.o' -print 2>/dev/null
            )
            if [[ ${#artifacts[@]} -eq 0 ]]; then
                fail "CANN_VENDOR_ARTIFACT_MISSING" \
                    "Missing ${soc} ${op_name} kernel object under ${vendor_root}." 12
            fi
            for artifact in "${artifacts[@]}"; do
                require_nonempty_artifact "${artifact}" "${soc} ${op_name} kernel object"
            done
            mapfile -t artifacts < <(
                find "${vendor_root}/op_impl/ai_core/tbe/kernel/${kernel_id}/${op_name}" \
                    -maxdepth 1 -type f -name '*.json' -print 2>/dev/null
            )
            if [[ ${#artifacts[@]} -eq 0 ]]; then
                fail "CANN_VENDOR_ARTIFACT_MISSING" \
                    "Missing ${soc} ${op_name} kernel metadata under ${vendor_root}." 12
            fi
            for artifact in "${artifacts[@]}"; do
                require_nonempty_artifact "${artifact}" "${soc} ${op_name} kernel metadata"
            done
        done
        require_nonempty_artifact \
            "${vendor_root}/op_impl/ai_core/tbe/config/${soc}/aic-${soc}-ops-info.json" \
            "${soc} CANN op info config"
        require_nonempty_artifact \
            "${vendor_root}/op_impl/ai_core/tbe/kernel/config/${soc}/binary_info_config.json" \
            "${soc} binary index"
        for op_name in hyper_mega_moe hyper_mega_moe_grad hyper_mega_mhc hyper_mega_mhc_grad \
                hyper_mega_gate_route hyper_mega_gate_route_grad; do
            require_nonempty_artifact \
                "${vendor_root}/op_impl/ai_core/tbe/kernel/config/${soc}/${op_name}.json" \
                "${soc} ${op_name} binary config"
        done
    done

    dynamic_section=$(readelf -d "${library}")
    if ! grep -Eq '\(SONAME\).*\[libcust_opapi\.so\]' <<< "${dynamic_section}"; then
        fail "CANN_VENDOR_SONAME_INVALID" \
            "Expected libcust_opapi.so SONAME in ${library}." 12
    fi
    if grep -Eq '(RPATH|RUNPATH)' <<< "${dynamic_section}"; then
        fail "CANN_VENDOR_RUNPATH_FOUND" \
            "CANN vendor library must not contain RPATH/RUNPATH: ${library}." 12
    fi
    ldd_output=$(ldd -r "${library}" 2>&1)
    if grep -Eq 'not found|undefined symbol:' <<< "${ldd_output}"; then
        fail "CANN_VENDOR_RUNTIME_LINK_FAILED" \
            "Unresolved runtime dependency found for ${library}." 12
    fi
    echo "INFO: validated multicore vendor ${vendor_root} for ${soc_list}"
}

SOC_ARGS=()
if [[ -n "${SOC_LIST}" ]]; then SOC_ARGS+=("${SOC_LIST}"); fi
SOC_SELECTION=$("${PYTHON_BIN}" "${PROJECT_ROOT}/hyper_multicore/_build/soc_selection.py" "${SOC_ARGS[@]}") ||
    fail "UNSUPPORTED_SOC_SELECTION" "Invalid --soc-list=${SOC_LIST}." 10
read -r SOC_LIST _ <<< "${SOC_SELECTION}"

if [[ "${HP_NATIVE_STAGE:-}" != "common" ]]; then
    BUILD_ARGS=(--native-only --soc-list "${SOC_LIST}" --jobs "${NATIVE_JOBS}")
    if [[ "${CLEAN}" == "on" ]]; then BUILD_ARGS+=(--clean); fi
    exec "${PYTHON_BIN}" "${PROJECT_ROOT}/hyper_multicore/_build/build_native.py" "${BUILD_ARGS[@]}"
fi

CANN_SOC_LIST="${SOC_LIST}"

cd "${PROJECT_ROOT}"
if [[ -z "${ASCEND_HOME_PATH:-}" || ! -d "${ASCEND_HOME_PATH}" ]]; then
    fail "CANN_ENV_NOT_CONFIGURED" \
        "ASCEND_HOME_PATH must identify the selected CANN installation; source its set_env.sh first." 3
fi
CANN_VERSION_FILE="${ASCEND_HOME_PATH}/opp/version.info"
CANN_VERSION=$(hp_read_cann_version "${ASCEND_HOME_PATH}")
CANN_VERSION=${CANN_VERSION:-unknown}

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    fail "PYTHON_BUILD_DEPENDENCY_NOT_FOUND" \
        "The selected Python executable is unavailable: ${PYTHON_BIN}." 4
fi
PYTHON_BIN=$(command -v "${PYTHON_BIN}")
required_tools=(awk asc_opc bash cmake find gcc g++ git grep ld ldd make nm readelf readlink)
required_tools+=(sed sha256sum sort tee)
for required_tool in "${required_tools[@]}"; do
    if ! command -v "${required_tool}" >/dev/null 2>&1; then
        fail "BUILD_TOOL_NOT_FOUND" "Required multicore build tool not found on PATH: ${required_tool}." 4
    fi
done
MULTICORE_VENDOR_CMAKE="${PROJECT_ROOT}/hyper_multicore/cmake/vendor"

function calculate_vendor_fingerprint() {
    local cann_soc=$1
    local cann_root
    local bisheng_path
    local bisheng_version
    local asc_opc_path
    local gcc_path
    local gxx_path
    local ld_path
    local cmake_path
    local make_path
    cann_root=$(cd "${ASCEND_HOME_PATH}" && pwd -P)
    bisheng_path=$(readlink -f "$(command -v bisheng)")
    bisheng_version=$("${bisheng_path}" --version 2>&1 | sed -n '1p')
    asc_opc_path=$(readlink -f "$(command -v asc_opc)")
    gcc_path=$(readlink -f "$(command -v gcc)")
    gxx_path=$(readlink -f "$(command -v g++)")
    ld_path=$(readlink -f "$(command -v ld)")
    cmake_path=$(readlink -f "$(command -v cmake)")
    make_path=$(readlink -f "$(command -v make)")
    (
        cd "${PROJECT_ROOT}"
        {
            printf '%s\n' \
                "schema=1" \
                "soc=${cann_soc}" \
                "cann=${CANN_VERSION}" \
                "cann_root=${cann_root}" \
                "bisheng=${bisheng_path}:${bisheng_version}" \
                "asc_opc=${asc_opc_path}" \
                "gcc=${gcc_path}:$(gcc -dumpmachine):$(gcc -dumpfullversion -dumpversion)" \
                "gxx=${gxx_path}:$(g++ -dumpmachine):$(g++ -dumpfullversion -dumpversion)" \
                "ld=${ld_path}:$(ld --version | sed -n '1p')" \
                "cmake=${cmake_path}:$(cmake --version | sed -n '1p')" \
                "make=${make_path}:$(make --version | sed -n '1p')" \
                "env_CC=${CC-}" \
                "env_CXX=${CXX-}" \
                "env_CFLAGS=${CFLAGS-}" \
                "env_CXXFLAGS=${CXXFLAGS-}" \
                "env_CPPFLAGS=${CPPFLAGS-}" \
                "env_LDFLAGS=${LDFLAGS-}" \
                "env_CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE-}" \
                "multicore_lock=${MULTICORE_LOCK_DIGEST}" \
                "shmem_headers=${SHMEM_HEADERS_IDENTITY}" \
                "arch=$(uname -m)"
            if [[ -f "${CANN_VERSION_FILE}" ]]; then
                sha256sum "${CANN_VERSION_FILE}"
            else
                printf '%s\n' "cann_version_info=missing"
            fi
            sha256sum "${asc_opc_path}"
            sha256sum \
                hyper_multicore/build.sh \
                hyper_multicore/_build/assemble_multicore_source.py \
                hyper_multicore/_build/merge_multicore_vendors.py \
                hyper_multicore/cmake/hardening.cmake \
                hyper_multicore/shmem/_build/shmem_sdk.sh
            while IFS= read -r -d '' source_file; do
                sha256sum "${source_file}"
            done < <(
                find hyper_multicore/ops hyper_multicore/cmake/vendor \
                    hyper_multicore/shmem/cmake/shmem_wrapper \
                    -type f -print0 | sort -z
            )
        } | sha256sum | awk '{print $1}'
    )
}

function vendor_cache() {
    "${PYTHON_BIN}" -c 'import sys; sys.path.insert(0, sys.argv.pop(1)); from _hyper_multicore_bootstrap import load_module; load_module("vendor_cache").main()' \
        "${PROJECT_ROOT}/hyper_multicore/_build" "$@"
}

SHMEM_SDK_MANIFEST="${HP_SHMEM_HANDOFF:-${COMMON_ROOT}/sdk.json}"
ARCH_GROUPS=$("${PYTHON_BIN}" "${PROJECT_ROOT}/hyper_multicore/_build/soc_selection.py" --groups "${CANN_SOC_LIST}")

function vendor_paths() {
    SHMEM_INSTALL_ROOT=$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["architectures"][sys.argv[2]]["install_root"])' \
        "${SHMEM_SDK_MANIFEST}" "${device_arch}")
    export ASCEND_SHMEM_HOME_PATH="${SHMEM_INSTALL_ROOT}"
    export SHMEM_HOME_PATH="${SHMEM_INSTALL_ROOT}"
    ASSEMBLY_ROOT="${WORK_ROOT}/source-assembly/${device_arch}"
    SOURCE_ROOT="${ASSEMBLY_ROOT}/source"
    VENDOR_BUILD_ROOT="${WORK_ROOT}/vendor-build/${device_arch}"
    VENDOR_STAGE_ROOT="${VENDOR_BUILD_ROOT}/stage"
    VENDOR_CANDIDATE="${VENDOR_STAGE_ROOT}/packages/vendors/hyper_multicore_nn"
    KERNEL_LOG="${WORK_ROOT}/logs/cann-vendor-build-${device_arch}.log"
    VENDOR_REQUEST="${WORK_ROOT}/vendor-request-${device_arch}.json"
    REUSED_VENDOR="${WORK_ROOT}/vendor-reuse-${device_arch}/hyper_multicore_nn"
}

function prepare_common() {
    cd "${PROJECT_ROOT}"
    if [[ "${CLEAN}" == "on" ]]; then
        rm -rf "${WORK_ROOT}" "${COMPONENT_ROOT}"
    fi
    rm -rf "${COMPONENT_ROOT}"
    mkdir -p "${OUTPUT_ROOT}"
    SHMEM_ARGS=(--soc-list "${SOC_LIST}" --jobs "${NATIVE_JOBS}")
    if [[ "${CLEAN}" == "on" ]]; then
        SHMEM_ARGS+=(--clean)
    fi
    bash "${SCRIPT_DIR}/shmem/build.sh" "${SHMEM_ARGS[@]}"

    CURRENT_REASON_CODE="MULTICORE_DEPENDENCY_PREPARATION_FAILED"
    PREPARED_DEPENDENCIES=$("${PYTHON_BIN}" "${PROJECT_ROOT}/hyper_multicore/_build/prepare_dependencies.py" \
        --dependency ops_nn \
        --dependency ops_transformer)
    OPS_NN_SOURCE_DIR=$("${PYTHON_BIN}" -c 'import json,sys; print(json.loads(sys.argv[1])["ops_nn"]["source_dir"])' \
        "${PREPARED_DEPENDENCIES}")
    OPS_TRANSFORMER_SOURCE_DIR=$("${PYTHON_BIN}" -c 'import json,sys; print(json.loads(sys.argv[1])["ops_transformer"]["source_dir"])' \
        "${PREPARED_DEPENDENCIES}")

    CURRENT_REASON_CODE="SHMEM_SDK_HANDOFF_FAILED"
    # SHMEM publishes this before the vendor work root is fingerprinted.
    SHMEM_HEADERS_IDENTITY=$("${PYTHON_BIN}" - "${SHMEM_SDK_MANIFEST}" <<'PYHEADERS'
import hashlib
import json
from pathlib import Path
import sys

backends = json.loads(Path(sys.argv[1]).read_text())["architectures"]
identities = set()
for backend in backends.values():
    sdk = Path(backend["install_root"]) / "shmem"
    required = ["include/shmem.h", "lib/libhyper_multicore_shmem.so", "lib/libhyper_multicore_shmem_utils.so",
                "lib/aclshmem_bootstrap_uid.so", "lib/aclshmem_bootstrap_config_store.so"]
    if any(not (sdk / name).is_file() for name in required):
        raise ValueError(f"Full SHMEM SDK is missing at {sdk}")
    headers = {str(path.relative_to(sdk)): hashlib.sha256(path.read_bytes()).hexdigest()
               for folder in ("include", "src/device") for path in sorted((sdk / folder).rglob("*")) if path.is_file()}
    if not headers:
        raise ValueError(f"SHMEM SDK headers are missing at {sdk}")
    identities.add(hashlib.sha256(json.dumps(headers, sort_keys=True).encode()).hexdigest())
if len(identities) != 1:
    raise ValueError("SHMEM SDK headers differ across backends")
print(identities.pop())
PYHEADERS
    )

    if ! MULTICORE_LOCK_DIGEST=$("${PYTHON_BIN}" -c '
import hashlib
import json
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    lock = json.load(source)["components"]["multicore"]
encoded = json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()
print(hashlib.sha256(encoded).hexdigest())
    ' "${PROJECT_ROOT}/hyper_multicore/_build/dependencies.lock.json"); then
        fail "MULTICORE_LOCK_INVALID" "Cannot read the locked multicore dependency contract." 9
    fi
    if [[ ! "${MULTICORE_LOCK_DIGEST}" =~ ^[0-9a-f]{64}$ ]]; then
        fail "MULTICORE_LOCK_INVALID" "Cannot read the locked multicore dependency contract." 9
    fi

    mkdir -p "${WORK_ROOT}/logs"

    echo "INFO: building unified multicore CANN vendor"
    echo "  CANN: ${ASCEND_HOME_PATH}"
    echo "  SOCs: ${CANN_SOC_LIST}"
    echo "  SHMEM SDKs: ${SHMEM_SDK_MANIFEST}"

    COMMON_HOST_INPUT_IDENTITY=$(calculate_vendor_fingerprint "common-host-input")
    PENDING_ARCHES=()
    while read -r device_arch _sdk_soc_type cann_soc registered_socs _selected_socs; do
        vendor_paths
        CACHE_ARGS=(prepare --root "${PROJECT_ROOT}" --native "${HP_VENDOR_CACHE_ROOT}"
                    --arch "${device_arch}" --sdk "${SHMEM_INSTALL_ROOT}/shmem" --request "${VENDOR_REQUEST}")
        if [[ "${HP_VENDOR_CACHE_CLEAN:-}" == "1" ]]; then CACHE_ARGS+=(--clean); fi
        vendor_cache "${CACHE_ARGS[@]}"
        if [[ -d "${REUSED_VENDOR}" ]] && (validate_multicore_vendor "${REUSED_VENDOR}" "${cann_soc}"); then
            continue
        fi
        PENDING_ARCHES+=("${device_arch}")
    done <<< "${ARCH_GROUPS}"
}

case "${PHASE}" in
    prepare)
        prepare_common >&2
        "${PYTHON_BIN}" - "${OPS_NN_SOURCE_DIR}" "${OPS_TRANSFORMER_SOURCE_DIR}" \
            "${COMMON_HOST_INPUT_IDENTITY}" "${PENDING_ARCHES[@]}" <<'PYCONTEXT'
import json
import sys
print(json.dumps({"ops_nn_source": sys.argv[1], "ops_transformer_source": sys.argv[2],
                  "host_input_identity": sys.argv[3], "pending": sys.argv[4:]}))
PYCONTEXT
        exit 0
        ;;
    vendor)
        export PIP_NO_BUILD_ISOLATION=false
        export PIP_NO_INDEX=1
        while read -r device_arch _sdk_soc_type cann_soc registered_socs _selected_socs; do
            if [[ "${device_arch}" != "${DEVICE_ARCH}" ]]; then continue; fi
            vendor_paths
            VENDOR_STARTED=${EPOCHREALTIME}
            rm -rf "${ASSEMBLY_ROOT}" "${VENDOR_BUILD_ROOT}"
            CURRENT_REASON_CODE="MULTICORE_SOURCE_ASSEMBLY_FAILED"
            "${PYTHON_BIN}" "${PROJECT_ROOT}/hyper_multicore/_build/assemble_multicore_source.py" \
                --ops-nn-source "${HP_OPS_NN_SOURCE}" \
                --ops-transformer-source "${HP_OPS_TRANSFORMER_SOURCE}" \
                --work-dir "${ASSEMBLY_ROOT}"

            echo "INFO: building isolated ${cann_soc} vendor input from ${SOURCE_ROOT}"
            CURRENT_REASON_CODE="CANN_VENDOR_BUILD_FAILED"
            set +e
            (
                set -e
                cmake -S "${MULTICORE_VENDOR_CMAKE}" \
                    -B "${VENDOR_BUILD_ROOT}" \
                    -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}" \
                    -DASCEND_CANN_PACKAGE_PATH="${ASCEND_HOME_PATH}" \
                    -DASCEND_COMPUTE_UNIT="${cann_soc}" \
                    -DHP_MULTICORE_REGISTERED_SOCS="${registered_socs//,/;}" \
                    -DASCEND_PYTHON_EXECUTABLE="${PYTHON_BIN}" \
                    -DHP_MULTICORE_SOURCE_ROOT="${SOURCE_ROOT}" \
                    -DHP_SHMEM_SDK_ROOT="${SHMEM_INSTALL_ROOT}/shmem" \
                    -DHP_VENDOR_STAGE_ROOT="${VENDOR_STAGE_ROOT}"
                cmake --build "${VENDOR_BUILD_ROOT}" --parallel "${NATIVE_JOBS}"
                cmake --build "${VENDOR_BUILD_ROOT}" --parallel "${NATIVE_JOBS}" \
                    --target binary
                cmake --install "${VENDOR_BUILD_ROOT}"
            ) 2>&1 | tee "${KERNEL_LOG}"
            cann_build_status=${PIPESTATUS[0]}
            set -e
            if [[ ${cann_build_status} -ne 0 ]]; then
                fail "CANN_VENDOR_BUILD_FAILED" \
                    "${cann_soc} CANN vendor build exited ${cann_build_status}; inspect ${KERNEL_LOG}." \
                    "${cann_build_status}"
            fi

            if [[ ! -d "${VENDOR_CANDIDATE}" ]]; then
                fail "UNIFIED_VENDOR_NOT_FOUND" \
                    "Expected ${cann_soc} hyper_multicore_nn package at ${VENDOR_CANDIDATE}." 12
            fi
            CURRENT_REASON_CODE="CANN_VENDOR_VALIDATION_FAILED"
            validate_multicore_vendor "${VENDOR_CANDIDATE}" "${cann_soc}" "${KERNEL_LOG}"
            if [[ -n "${HP_VENDOR_CACHE_ROOT:-}" ]]; then
                vendor_cache built --request "${VENDOR_REQUEST}" --candidate "${VENDOR_CANDIDATE}" --started "${VENDOR_STARTED}"
            fi
            exit 0
        done <<< "${ARCH_GROUPS}"
        fail "UNSUPPORTED_SOC_SELECTION" "Architecture ${DEVICE_ARCH} is not selected." 10
        ;;
    merge)
        PER_SOC_VENDOR_INPUTS=()
        PER_SOC_HOST_INPUT_IDENTITIES=()
        while read -r device_arch _sdk_soc_type cann_soc registered_socs _selected_socs; do
            vendor_paths
            candidate=$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["candidate"])' "${VENDOR_REQUEST}")
            PER_SOC_VENDOR_INPUTS+=("--input" "${cann_soc}=${candidate}")
            PER_SOC_HOST_INPUT_IDENTITIES+=("--host-input-identity" "${cann_soc}=${HP_VENDOR_HOST_IDENTITY}")
        done <<< "${ARCH_GROUPS}"
        VENDOR_ROOT="${OUTPUT_ROOT}/vendors/hyper_multicore_nn"
        mkdir -p "$(dirname "${VENDOR_ROOT}")"
        CURRENT_REASON_CODE="CANN_VENDOR_MERGE_FAILED"
        "${PYTHON_BIN}" "${PROJECT_ROOT}/hyper_multicore/_build/merge_multicore_vendors.py" \
            "${PER_SOC_VENDOR_INPUTS[@]}" \
            "${PER_SOC_HOST_INPUT_IDENTITIES[@]}" \
            --soc-list "${CANN_SOC_LIST}" \
            --output "${VENDOR_ROOT}"

        CURRENT_REASON_CODE="CANN_VENDOR_VALIDATION_FAILED"
        validate_multicore_vendor "${VENDOR_ROOT}" "${CANN_SOC_LIST}" "" architecture
        cp "${PROJECT_ROOT}/hyper_multicore/set_env.bash" "${OUTPUT_ROOT}/set_env.bash"

        while IFS= read -r packaged_library; do
            validate_shared_library_security "${packaged_library}"
        done < <(find "${COMPONENT_ROOT}" -type f -name '*.so' -print)
        echo "INFO: common multicore resources completed: ${COMPONENT_ROOT}"
        ;;
    *) fail "INVALID_SELECTION" "Unknown internal build phase: ${PHASE}." 2 ;;
esac
