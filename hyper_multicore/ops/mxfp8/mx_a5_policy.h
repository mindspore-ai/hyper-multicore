// Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#ifndef HYPER_MULTICORE_MX_A5_POLICY_H_
#define HYPER_MULTICORE_MX_A5_POLICY_H_
#include "mx_gmm_descriptor.h"

#ifdef __NPU_ARCH__
#define MX_POLICY_INLINE __aicore__ inline
#else
#define MX_POLICY_INLINE inline
#endif

namespace MulticoreMx {
struct MxA5Resources {
  uint32_t scale_k;
  uint8_t stages;
  uint64_t l1_bytes;
};

MX_POLICY_INLINE MxA5Resources SelectMxA5Resources(const MxGmmDescriptor& r) {
  // Match the qualified A5 T6 scale capacity and official three-buffer predicate.
  const uint32_t scale = r.k < 8192 ? r.k : 8192;
  const uint64_t a = static_cast<uint64_t>(r.base_m) * ((r.k_al1 + 63) / 64 * 64);
  const uint64_t b = static_cast<uint64_t>(r.base_n) * ((r.k_bl1 + 63) / 64 * 64);
  const uint64_t scales = static_cast<uint64_t>(r.base_m + r.base_n) * ((scale + 63) / 64) * 2;
  const uint64_t common = a + b + scales;
  const bool triple = r.layout != static_cast<uint32_t>(MxLayout::TN)
      && common + a <= 262144 && common + b <= 262144;
  return {scale, static_cast<uint8_t>(triple ? 3 : 2), 2 * common + (triple ? a + b : 0)};
}

MX_POLICY_INLINE uint32_t RotateMxLogicalCube(uint32_t lane, uint32_t expert,
                                             uint64_t blocks, uint32_t cubes) {
  const uint32_t shift = static_cast<uint32_t>((expert * (blocks % cubes)) % cubes);
  return (lane + cubes - shift) % cubes;
}
}  // namespace MulticoreMx
#undef MX_POLICY_INLINE
#endif
