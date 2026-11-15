// Copyright 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#ifndef HYPER_MULTICORE_OPS_COMMON_MEGA_MOE_PROBABILITY_WORKSPACE_H_
#define HYPER_MULTICORE_OPS_COMMON_MEGA_MOE_PROBABILITY_WORKSPACE_H_
#include "kernel_operator.h"
#include "probability_abi.h"

namespace MulticoreRuntime {
__aicore__ inline GM_ADDR ProbabilityPlane(GM_ADDR workspace, int64_t stride, uint32_t plane) {
  return workspace + static_cast<int64_t>(plane) * stride * ProbabilityLayout::ELEMENT_BYTES;
}

}  // namespace MulticoreRuntime
#endif  // HYPER_MULTICORE_OPS_COMMON_MEGA_MOE_PROBABILITY_WORKSPACE_H_
