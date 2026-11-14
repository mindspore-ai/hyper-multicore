// Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#ifndef HYPER_MULTICORE_BF16_LOGICAL_LANE_H_
#define HYPER_MULTICORE_BF16_LOGICAL_LANE_H_
#include <cstdint>

#if defined(__CCE_AICORE__) || defined(__NPU_ARCH__)
#define MULTICORE_BF16_LANE_INLINE __aicore__ inline
#else
#define MULTICORE_BF16_LANE_INLINE inline
#endif

namespace MulticoreBf16 {
MULTICORE_BF16_LANE_INLINE uint32_t RotateLogicalCube(uint32_t lane, uint32_t expert,
                                                    uint64_t blocks, uint32_t cubes) {
  // A local permutation preserves tile ownership even when expert sizes differ.
  const uint32_t shift = static_cast<uint32_t>((expert * (blocks % cubes)) % cubes);
  return (lane + cubes - shift) % cubes;
}
}  // namespace MulticoreBf16
#undef MULTICORE_BF16_LANE_INLINE
#endif
