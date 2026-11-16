// Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#ifndef HYPER_MULTICORE_BF16_GMM_PROBLEM_H_
#define HYPER_MULTICORE_BF16_GMM_PROBLEM_H_
#include <cstdint>

#if defined(__CCE_AICORE__) || defined(__NPU_ARCH__)
#define MULTICORE_BF16_INLINE __aicore__ inline
#else
#define MULTICORE_BF16_INLINE inline
#endif

namespace MulticoreBf16 {
struct Problem {
  uint64_t m, n, k;
  uint64_t a, b, c;
};

MULTICORE_BF16_INLINE Problem MakeProblem(uint64_t m, uint64_t n, uint64_t k, uint64_t expert,
                                        uint64_t start, uint64_t rows, bool trans_a) {
  if (trans_a) return {m, n, rows, start * m, start * n, expert * m * n};
  return {rows, n, k, start * k, expert * k * n, start * n};
}
struct TailSplit {
  uint64_t m, n;
};

MULTICORE_BF16_INLINE TailSplit SplitTail(uint64_t m, uint64_t n, uint64_t base_m,
                                        uint64_t base_n, uint64_t cubes) {
  const uint64_t tiles = ((m + base_m - 1) / base_m) * ((n + base_n - 1) / base_n);
  TailSplit split{1, 1};
  const uint64_t tail = tiles <= cubes ? 0 : tiles % cubes;
  if (tail == 0) return split;
  // Match arch35 CalcTailBasicBlock: split the final wave before adding idle Cube slots.
  while ((split.m + 1) * split.n * tail <= cubes) {
    ++split.m;
    if (split.m * (split.n + 1) * tail <= cubes) ++split.n;
  }
  return split;
}

}  // namespace MulticoreBf16
#undef MULTICORE_BF16_INLINE
#endif
