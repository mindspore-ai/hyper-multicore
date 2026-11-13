#pragma once
#include <cstdint>

#if defined(__CCE_AICORE__) || defined(__NPU_ARCH__)
#define MULTICORE_BF16_INLINE __aicore__ inline
#else
#define MULTICORE_BF16_INLINE inline
#endif

namespace MulticoreBf16 {
template <typename Tiling>
MULTICORE_BF16_INLINE Tiling MakeSwiGluGradTiling(int64_t rows, int64_t columns) {
  return {rows, columns, rows, columns, rows, columns, 248 * 1024, 1, 1, 1};
}
}  // namespace MulticoreBf16
#undef MULTICORE_BF16_INLINE
