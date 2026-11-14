// Copyright 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#ifndef HYPER_MULTICORE_OPS_COMMON_SWIGLU_TASK_OPTIONS_H_
#define HYPER_MULTICORE_OPS_COMMON_SWIGLU_TASK_OPTIONS_H_
#include <cstdint>

namespace MulticoreRuntime {
// Mirrors the option definitions in tasks/swiglu.py.
enum class SwiGluClampPosition : uint32_t {
  BEFORE_SILU = 0,
  AFTER_SILU = 1,
};

enum class SwiGluTaskIteration : uint32_t { TILED = 0, PERSISTENT_LANES = 2 };
}  // namespace MulticoreRuntime
#endif  // HYPER_MULTICORE_OPS_COMMON_SWIGLU_TASK_OPTIONS_H_
