// Copyright 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#ifndef HYPER_MULTICORE_OPS_COMMON_MEGA_MOE_PROBABILITY_ABI_H_
#define HYPER_MULTICORE_OPS_COMMON_MEGA_MOE_PROBABILITY_ABI_H_
#include <cstdint>

namespace MulticoreRuntime {
// Mirrors modules/mega_moe/probability.py; cross-language tests pin every value.
// Strides and plane offsets count FP32 scalar rows, never hidden elements or bytes.
struct ProbabilityLayout {
  static constexpr uint32_t PLANES = 4;
  static constexpr uint32_t ELEMENT_BYTES = 4;
  static constexpr uint32_t FORWARD_POSITION = 16;
  static constexpr uint32_t BACKWARD_POSITION = 25;
  static constexpr uint32_t SOURCE = 0;
  static constexpr uint32_t RECEIVED = 1;
  static constexpr uint32_t RECEIVED_GRADIENT = 2;
  static constexpr uint32_t RETURNED_GRADIENT = 3;
};

}  // namespace MulticoreRuntime
#endif  // HYPER_MULTICORE_OPS_COMMON_MEGA_MOE_PROBABILITY_ABI_H_
