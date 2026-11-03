# Copyright 2026 Huawei Technologies Co., Ltd.
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

function(hp_detect_torch_build_contract python_executable output_prefix)
    if(NOT EXISTS "${python_executable}")
        message(FATAL_ERROR
            "Python executable does not exist: '${python_executable}'")
    endif()
    if(NOT CMAKE_CXX_COMPILER_LOADED)
        message(FATAL_ERROR
            "Torch build-contract detection requires the CXX language")
    endif()

    string(CONCAT torch_contract_code
        "from importlib.util import find_spec\n"
        "from pathlib import Path\n"
        "torch_spec = find_spec('torch')\n"
        "if torch_spec is None or not torch_spec.submodule_search_locations:\n"
        "    raise RuntimeError('Torch package is not installed')\n"
        "npu_spec = find_spec('torch_npu')\n"
        "if npu_spec is None or not npu_spec.submodule_search_locations:\n"
        "    raise RuntimeError('TorchNPU package is not installed')\n"
        "import torch\n"
        "values = (\n"
        "    str(Path(next(iter(torch_spec.submodule_search_locations))).resolve()),\n"
        "    str(Path(next(iter(npu_spec.submodule_search_locations))).resolve()),\n"
        "    str(int(torch.compiled_with_cxx11_abi())),\n"
        ")\n"
        "if any('\\n' in value or '\\r' in value or ';' in value for value in values):\n"
        "    raise RuntimeError(f'Torch build contract contains an unsupported character: {values!r}')\n"
        "print('\\n'.join(values))")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E env TORCH_DEVICE_BACKEND_AUTOLOAD=0
                ${python_executable} -c "${torch_contract_code}"
        RESULT_VARIABLE torch_contract_status
        OUTPUT_VARIABLE torch_contract
        ERROR_VARIABLE torch_contract_error
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(NOT torch_contract_status EQUAL 0)
        message(FATAL_ERROR
            "Cannot determine the active TorchNPU build contract: "
            "${torch_contract_error}")
    endif()

    string(REPLACE "\r\n" "\n" torch_contract "${torch_contract}")
    string(REPLACE "\n" ";" torch_contract_fields "${torch_contract}")
    list(LENGTH torch_contract_fields torch_contract_field_count)
    if(NOT torch_contract_field_count EQUAL 3)
        message(FATAL_ERROR
            "TorchNPU build contract must contain three fields; "
            "got '${torch_contract}'")
    endif()
    list(GET torch_contract_fields 0 torch_package_root)
    list(GET torch_contract_fields 1 torch_npu_package_root)
    list(GET torch_contract_fields 2 torch_cxx11_abi)
    if(NOT torch_cxx11_abi MATCHES "^[01]$")
        message(FATAL_ERROR
            "Torch reports an invalid C++11 ABI value: '${torch_cxx11_abi}'")
    endif()

    if(TARGET torch)
        message(FATAL_ERROR
            "Torch build-contract detection cannot replace an existing torch target")
    endif()

    set(torch_config_dir "${torch_package_root}/share/cmake/Torch")
    set(caffe2_config_dir "${torch_package_root}/share/cmake/Caffe2")
    foreach(package IN ITEMS Torch Caffe2)
        set(config_file "${torch_package_root}/share/cmake/${package}/${package}Config.cmake")
        if(NOT EXISTS "${config_file}")
            message(FATAL_ERROR "The active Torch package is missing a CMake config: '${config_file}'")
        endif()
    endforeach()
    set(had_torch_install_prefix FALSE)
    if(DEFINED ENV{TORCH_INSTALL_PREFIX})
        set(had_torch_install_prefix TRUE)
        set(saved_torch_install_prefix "$ENV{TORCH_INSTALL_PREFIX}")
    endif()
    unset(ENV{TORCH_INSTALL_PREFIX})
    set(Torch_DIR "${torch_config_dir}")
    set(Caffe2_DIR "${caffe2_config_dir}")
    set(TORCH_LIBRARY "TORCH_LIBRARY-NOTFOUND")
    find_package(Torch CONFIG REQUIRED
        PATHS "${torch_config_dir}"
        NO_DEFAULT_PATH)
    if(had_torch_install_prefix)
        set(ENV{TORCH_INSTALL_PREFIX} "${saved_torch_install_prefix}")
    else()
        unset(ENV{TORCH_INSTALL_PREFIX})
    endif()
    if(NOT TARGET torch)
        message(FATAL_ERROR "TorchConfig did not create the torch target")
    endif()

    get_target_property(torch_interface_features
        torch INTERFACE_COMPILE_FEATURES)
    get_target_property(torch_legacy_standard torch CXX_STANDARD)
    get_target_property(torch_interface_options
        torch INTERFACE_COMPILE_OPTIONS)

    set(torch_cxx_standard "")
    foreach(torch_feature IN LISTS torch_interface_features)
        if(torch_feature MATCHES "^cxx_std_([0-9]+)$")
            set(torch_candidate_standard "${CMAKE_MATCH_1}")
            if(torch_cxx_standard STREQUAL "" OR
              torch_candidate_standard GREATER torch_cxx_standard)
                set(torch_cxx_standard "${torch_candidate_standard}")
            endif()
        endif()
    endforeach()
    if(NOT torch_cxx_standard STREQUAL "")
        set(torch_contract_source "INTERFACE_COMPILE_FEATURES")
    elseif(torch_legacy_standard MATCHES "^[0-9]+$")
        set(torch_cxx_standard "${torch_legacy_standard}")
        set(torch_contract_source "CXX_STANDARD")
    else()
        message(FATAL_ERROR
            "TorchConfig does not publish a recognized C++ standard: "
            "INTERFACE_COMPILE_FEATURES='${torch_interface_features}', "
            "CXX_STANDARD='${torch_legacy_standard}', "
            "INTERFACE_COMPILE_OPTIONS='${torch_interface_options}'")
    endif()

    set(torch_required_feature "cxx_std_${torch_cxx_standard}")
    if(torch_cxx_standard LESS 17)
        message(FATAL_ERROR
            "TorchConfig requires at least C++17; "
            "reported C++${torch_cxx_standard}")
    endif()
    if(NOT "${torch_required_feature}" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
        message(FATAL_ERROR
            "TorchConfig requires unsupported compile feature "
            "'${torch_required_feature}'")
    endif()
    message(STATUS
        "HyperMulticore Torch contract: source=${torch_contract_source}, "
        "minimum=C++${torch_cxx_standard}")

    set(${output_prefix}_PACKAGE_ROOT
        "${torch_package_root}" PARENT_SCOPE)
    set(${output_prefix}_NPU_PACKAGE_ROOT
        "${torch_npu_package_root}" PARENT_SCOPE)
    set(${output_prefix}_CXX_STANDARD
        "${torch_cxx_standard}" PARENT_SCOPE)
    set(${output_prefix}_CXX11_ABI
        "${torch_cxx11_abi}" PARENT_SCOPE)
endfunction()

function(hp_apply_torch_cxx_standard target_name torch_cxx_standard)
    target_compile_features(${target_name} PRIVATE
        "cxx_std_${torch_cxx_standard}")
endfunction()

function(hp_detect_op_plugin_version torch_root torch_npu_root output_name)
    set(torch_version_header
        "${torch_root}/include/torch/csrc/api/include/torch/version.h")
    set(plugin_version_header
        "${torch_npu_root}/include/third_party/op-plugin/op_plugin/utils/Version.h")
    foreach(required_header IN ITEMS "${torch_version_header}" "${plugin_version_header}")
        if(NOT EXISTS "${required_header}")
            message(FATAL_ERROR "Missing framework version header: ${required_header}")
        endif()
    endforeach()
    file(STRINGS "${torch_version_header}" forwarded_version
        REGEX "^#include <torch/headeronly/version.h>")
    if(forwarded_version)
        set(torch_version_header "${torch_root}/include/torch/headeronly/version.h")
        if(NOT EXISTS "${torch_version_header}")
            message(FATAL_ERROR "Missing forwarded Torch version header: ${torch_version_header}")
        endif()
    endif()
    foreach(component IN ITEMS MAJOR MINOR)
        file(STRINGS "${torch_version_header}" definition
            REGEX "^#define TORCH_VERSION_${component} [0-9]+")
        list(LENGTH definition definition_count)
        if(NOT definition_count EQUAL 1)
            message(FATAL_ERROR "Unrecognized Torch ${component} version in ${torch_version_header}")
        endif()
        string(REGEX REPLACE "^#define TORCH_VERSION_${component} ([0-9]+).*" "\\1"
            torch_${component} "${definition}")
    endforeach()
    set(symbol "V${torch_MAJOR}R${torch_MINOR}")
    file(STRINGS "${plugin_version_header}" definition
        REGEX "^#define ${symbol} [0-9]+")
    if(NOT definition)
        message(FATAL_ERROR "Installed op-plugin has no version selector ${symbol}")
    endif()
    message(STATUS "HyperMulticore op-plugin version: ${symbol}")
    set(${output_name} "${symbol}" PARENT_SCOPE)
endfunction()
