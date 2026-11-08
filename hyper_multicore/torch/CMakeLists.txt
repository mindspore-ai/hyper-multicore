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

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

hp_detect_torch_build_contract("${Python3_EXECUTABLE}" TORCH)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/torch_cxx_standard.txt" "${TORCH_CXX_STANDARD}")
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

# Compile existing AscendC utility kernels separately from framework C++ headers.
# Both supported products (910B and 910_93) use the dav-2201 device architecture.
set(CANN_ROOT "$ENV{ASCEND_HOME_PATH}")
find_program(ASCEND_BISHENG bisheng HINTS "${CANN_ROOT}/tools/bisheng_compiler/bin" REQUIRED)
find_library(ASCENDC_HOST_RUNTIME ascendc_runtime
    PATHS "${CANN_ROOT}/lib64" "${CANN_ROOT}/aarch64-linux/lib64" REQUIRED NO_DEFAULT_PATH)
add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/mega_moe_utilities.o"
    COMMAND "${ASCEND_BISHENG}" -x asc --npu-arch=dav-2201 --cce-auto-sync=off
        -std=c++17 -O2 -fPIC -c "${CMAKE_CURRENT_SOURCE_DIR}/csrc/mega_moe_utilities.asc"
        -o "${CMAKE_CURRENT_BINARY_DIR}/mega_moe_utilities.o"
        "-I${CANN_ROOT}/asc" "-I${CANN_ROOT}/asc/include"
        "-I${CANN_ROOT}/asc/include/interface" "-I${CANN_ROOT}/asc/include/utils"
        "-I${CANN_ROOT}/asc/impl" "-I${CANN_ROOT}/include"
        "-I${CANN_ROOT}/include/runtime" "-I${CANN_ROOT}/pkg_inc"
        "-I${CANN_ROOT}/pkg_inc/runtime"
    DEPENDS csrc/mega_moe_utilities.asc VERBATIM)
set_source_files_properties("${CMAKE_CURRENT_BINARY_DIR}/mega_moe_utilities.o"
    PROPERTIES GENERATED TRUE EXTERNAL_OBJECT TRUE)

add_library(hyper_multicore_torch SHARED
    csrc/mega_moe_utilities.cpp
    "${CMAKE_CURRENT_BINARY_DIR}/mega_moe_utilities.o"
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
target_link_directories(hyper_multicore_torch PRIVATE "${CANN_ROOT}/lib64")
target_include_directories(hyper_multicore_torch BEFORE PRIVATE "${CANN_ROOT}/include")
target_link_libraries(hyper_multicore_torch PRIVATE
    ${TORCH_LINK_LIBRARIES}
    "${ASCENDC_HOST_RUNTIME}" profapi mmpa c_sec unified_dlog ascendcl runtime)
hp_enable_elf_hardening(hyper_multicore_torch)

install(TARGETS hyper_multicore_torch DESTINATION .)
install(CODE [[
    file(RPATH_REMOVE
        FILE "${CMAKE_INSTALL_PREFIX}/libhyper_multicore_torch.so")
]])
