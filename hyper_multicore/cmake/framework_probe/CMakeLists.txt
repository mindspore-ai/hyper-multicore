cmake_minimum_required(VERSION 3.18)
project(hyper_multicore_framework_probe LANGUAGES CXX)
include(${CMAKE_CURRENT_LIST_DIR}/../torch_compat.cmake)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
hp_detect_torch_build_contract("${Python3_EXECUTABLE}" TORCH)
hp_detect_op_plugin_version("${TORCH_PACKAGE_ROOT}" "${TORCH_NPU_PACKAGE_ROOT}" TORCH_OP_PLUGIN_VERSION)
add_library(hyper_multicore_framework_probe SHARED probe.cpp)
hp_apply_torch_cxx_standard(hyper_multicore_framework_probe "${TORCH_CXX_STANDARD}")
target_compile_definitions(hyper_multicore_framework_probe PRIVATE
    _GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI} CURRENT_VERSION=${TORCH_OP_PLUGIN_VERSION})
target_include_directories(hyper_multicore_framework_probe PRIVATE
    ${Python3_INCLUDE_DIRS} ${TORCH_PACKAGE_ROOT}/include
    ${TORCH_PACKAGE_ROOT}/include/torch/csrc/api/include
    ${TORCH_NPU_PACKAGE_ROOT}/include ${TORCH_NPU_PACKAGE_ROOT}/include/third_party/acl/inc
    ${TORCH_NPU_PACKAGE_ROOT}/include/third_party/op-plugin
    $ENV{ASCEND_HOME_PATH}/include ${CMAKE_CURRENT_LIST_DIR}/../../torch/csrc)
target_link_libraries(hyper_multicore_framework_probe PRIVATE
    ${TORCH_PACKAGE_ROOT}/lib/libc10.so ${TORCH_PACKAGE_ROOT}/lib/libtorch.so
    ${TORCH_PACKAGE_ROOT}/lib/libtorch_cpu.so ${TORCH_NPU_PACKAGE_ROOT}/lib/libtorch_npu.so)

target_link_libraries(hyper_multicore_framework_probe PRIVATE Python3::Python)
target_link_options(hyper_multicore_framework_probe PRIVATE -Wl,--no-undefined)
