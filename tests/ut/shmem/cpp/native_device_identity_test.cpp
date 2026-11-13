/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

#include "acl/acl.h"
#include "cann/host.h"

namespace {

using hyper_multicore::shmem::cann::host::query_device_identity;
using hyper_multicore::shmem::runtime::ErrorCode;

using hyper_multicore::shmem::BackendSupportsDevice;
#ifdef HP_SHMEM_ASCEND950
static_assert(BackendSupportsDevice(3510));
static_assert(!BackendSupportsDevice(2201));
#else
static_assert(BackendSupportsDevice(2201));
static_assert(!BackendSupportsDevice(3510));
#endif
static_assert(!BackendSupportsDevice(0));
static_assert(!BackendSupportsDevice(3002));

constexpr int32_t kDeviceIndex = 3;
aclError g_device_status = ACL_SUCCESS;
aclError g_arch_status = ACL_SUCCESS;
int64_t g_npu_arch = 3510;
char g_device_soc_name[64] = {};
bool g_null_soc = false;

void check(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}

void test_identity_values() {
  const struct {
    int64_t arch;
    const char *soc;
  } devices[] = {{2201, "Ascend910B4"}, {2201, "Ascend910_9391"}, {3510, "Ascend950PR_9599"},
                 {3510, "Ascend950DT_9591"}, {3510, "FutureDeviceName"}};
  for (const auto &device : devices) {
    g_npu_arch = device.arch;
    std::copy_n(device.soc, std::strlen(device.soc) + 1, g_device_soc_name);
    const auto identity = query_device_identity();
    check(identity.ok(), "device identity query failed");
    g_device_soc_name[0] = '!';
    check(identity.value().device_index == kDeviceIndex, "wrong device index");
    check(identity.value().npu_arch == device.arch, "wrong NPU architecture");
    check(identity.value().device_soc_name == device.soc, "SoC name was rewritten or not owned");
  }
}

void test_query_failures() {
  g_device_status = ACL_ERROR_UNINITIALIZE;
  const auto device_failure = query_device_identity();
  check(!device_failure.ok(), "device query failure was ignored");
  check(device_failure.error().error_code == ErrorCode::CannError, "wrong device error category");
  check(device_failure.error().cann_error_code == g_device_status, "device error code was lost");
  g_device_status = ACL_SUCCESS;

  g_arch_status = ACL_ERROR_INVALID_PARAM;
  const auto arch_failure = query_device_identity();
  check(!arch_failure.ok(), "architecture query failure was ignored");
  check(arch_failure.error().error_code == ErrorCode::CannError, "wrong architecture error category");
  check(arch_failure.error().cann_error_code == g_arch_status, "architecture error code was lost");
  check(arch_failure.error().message.find("aclrtGetDeviceInfo(NPU_ARCH)") != std::string::npos,
        "failed architecture query was not identified");
  g_arch_status = ACL_SUCCESS;

  g_device_soc_name[0] = '\0';
  for (bool null_soc : {false, true}) {
    g_null_soc = null_soc;
    const auto empty_soc = query_device_identity();
    check(!empty_soc.ok(), "empty SoC name was accepted");
    check(empty_soc.error().error_code == ErrorCode::UnsupportedCapability, "wrong empty SoC error category");
  }
}

}  // namespace

extern "C" aclError aclrtGetDevice(int32_t *device_id) {
  *device_id = kDeviceIndex;
  return g_device_status;
}

extern "C" aclError aclrtGetDeviceInfo(uint32_t device_id, aclrtDevAttr attr, int64_t *value) {
  if (device_id != kDeviceIndex || attr != ACL_DEV_ATTR_NPU_ARCH) return ACL_ERROR_INVALID_PARAM;
  *value = g_npu_arch;
  return g_arch_status;
}

extern "C" const char *aclrtGetSocName() {
  return g_null_soc ? nullptr : g_device_soc_name;
}

int main() {
  try {
    test_identity_values();
    test_query_failures();
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
