// Copyright 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#ifndef HYPER_MULTICORE_OPS_COMMON_GROUPED_MATMUL_TASK_OPTIONS_H_
#define HYPER_MULTICORE_OPS_COMMON_GROUPED_MATMUL_TASK_OPTIONS_H_
#include <cstdint>

namespace MulticoreRuntime {
// Mirrors the option definitions in tasks/gmm.py.
enum class GroupedMatmulTaskMode : uint32_t {
  PER_EXPERT = 0,
  SINGLE_GROUP = 1,
  BATCHED = 2,
  BATCHED_READY = 3,
};
}  // namespace MulticoreRuntime
#endif  // HYPER_MULTICORE_OPS_COMMON_GROUPED_MATMUL_TASK_OPTIONS_H_
