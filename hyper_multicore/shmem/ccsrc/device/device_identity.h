/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cstdint>
#include <string>

namespace hyper_multicore::shmem {

inline constexpr int64_t kAscend910NpuArch = 2201;
inline constexpr int64_t kAscend950NpuArch = 3510;

#ifdef HP_SHMEM_ASCEND950
inline constexpr int64_t kBackendNpuArch = kAscend950NpuArch;
inline constexpr char kBackendSocType[] = "Ascend950";
#else
inline constexpr int64_t kBackendNpuArch = kAscend910NpuArch;
inline constexpr char kBackendSocType[] = "Ascend910B";
#endif

/** @brief Reject a device generation that cannot use the linked private SHMEM backend. */
inline constexpr bool BackendSupportsDevice(int64_t npu_arch) {
  return npu_arch == kBackendNpuArch;
}

/** @brief Current ACL device and its owned SoC identity observed during Runtime initialization. */
struct DeviceIdentity {
  int32_t device_index;
  int64_t npu_arch;
  std::string device_soc_name;
};

}  // namespace hyper_multicore::shmem
