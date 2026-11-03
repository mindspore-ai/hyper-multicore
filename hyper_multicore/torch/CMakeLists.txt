cmake_minimum_required(VERSION 3.18)
project(hyper_multicore_multicore_torch LANGUAGES CXX)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
include(${CMAKE_CURRENT_LIST_DIR}/../cmake/hardening.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/../cmake/torch_compat.cmake)

if(NOT HP_MULTICORE_VENDOR_ROOT OR
    NOT EXISTS "${HP_MULTICORE_VENDOR_ROOT}/op_api/include")
    message(FATAL_ERROR
        "HP_MULTICORE_VENDOR_ROOT must identify a complete "
        "hyper_multicore_nn vendor")
endif()

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

hp_detect_torch_build_contract("${Python3_EXECUTABLE}" TORCH)
hp_detect_op_plugin_version("${TORCH_PACKAGE_ROOT}" "${TORCH_NPU_PACKAGE_ROOT}" TORCH_OP_PLUGIN_VERSION)

set(TORCH_INCLUDE_ROOT "${TORCH_PACKAGE_ROOT}/include")
set(TORCH_NPU_INCLUDE_ROOT "${TORCH_NPU_PACKAGE_ROOT}/include")
set(OP_PLUGIN_ROOT "${TORCH_NPU_INCLUDE_ROOT}/third_party/op-plugin")
set(TORCH_LINK_LIBRARIES
    "${TORCH_PACKAGE_ROOT}/lib/libc10.so"
    "${TORCH_PACKAGE_ROOT}/lib/libtorch.so"
    "${TORCH_PACKAGE_ROOT}/lib/libtorch_cpu.so"
    "${TORCH_NPU_PACKAGE_ROOT}/lib/libtorch_npu.so")

foreach(required_path IN ITEMS
    ${Python3_INCLUDE_DIRS}
    "${TORCH_INCLUDE_ROOT}"
    "${TORCH_INCLUDE_ROOT}/torch/csrc/api/include"
    "${TORCH_NPU_INCLUDE_ROOT}"
    "${OP_PLUGIN_ROOT}/op_plugin/include/npu_cpp_extension.h"
    ${TORCH_LINK_LIBRARIES})
    if(NOT EXISTS "${required_path}")
        message(FATAL_ERROR
            "The active Torch environment is missing a build input: "
            "${required_path}")
    endif()
endforeach()

add_library(hyper_multicore_torch SHARED
    csrc/registration.cpp
    csrc/registration_mhc.cpp
    csrc/mega_mhc.cpp
    csrc/mega_moe.cpp
    csrc/mega_moe_grad.cpp
    csrc/moe_token_permute_grad.cpp
    csrc/moe_token_permute_out.cpp
    csrc/mega_moe_unpermute.cpp
    csrc/mega_gate_registration.cpp
    csrc/mega_gate.cpp)
hp_apply_torch_cxx_standard(
    hyper_multicore_torch "${TORCH_CXX_STANDARD}")
# Match the active op-plugin compatibility headers.
target_compile_definitions(hyper_multicore_torch PRIVATE
    _GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI}
    CURRENT_VERSION=${TORCH_OP_PLUGIN_VERSION})
target_include_directories(hyper_multicore_torch PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Python3_INCLUDE_DIRS}
    ${TORCH_INCLUDE_ROOT}
    ${TORCH_INCLUDE_ROOT}/torch/csrc/api/include
    ${TORCH_NPU_INCLUDE_ROOT}
    ${TORCH_NPU_INCLUDE_ROOT}/third_party/acl/inc
    ${OP_PLUGIN_ROOT}
    ${OP_PLUGIN_ROOT}/op_plugin/include
    ${HP_MULTICORE_VENDOR_ROOT}/op_api/include)
target_link_libraries(hyper_multicore_torch PRIVATE
    ${TORCH_LINK_LIBRARIES})
hp_enable_elf_hardening(hyper_multicore_torch)

install(TARGETS hyper_multicore_torch DESTINATION .)
install(CODE [[
    file(RPATH_REMOVE
        FILE "${CMAKE_INSTALL_PREFIX}/libhyper_multicore_torch.so")
]])
