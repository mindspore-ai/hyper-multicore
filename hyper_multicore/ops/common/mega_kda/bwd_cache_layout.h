/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0.
 */
#ifndef HYPER_MEGA_KDA_BWD_CACHE_LAYOUT_H
#define HYPER_MEGA_KDA_BWD_CACHE_LAYOUT_H

#include <stdint.h>

namespace MegaKdaCache {

struct Layout {
  uint64_t qNormOff;
  uint64_t kNormOff;
  uint64_t vConvOff;
  uint64_t qRstdOff;
  uint64_t kRstdOff;
  uint64_t gCumOff;
  uint64_t mInvOff;
  uint64_t aqkOff;
  uint64_t wOff;
  uint64_t uOff;
  uint64_t qgOff;
  uint64_t kgOff;
  uint64_t expGLastOff;
  uint64_t hChunksOff;
  uint64_t vNewOff;
  uint64_t oAttnOff;
  uint64_t oRstdOff;
  uint64_t haloQkvOff;
  uint64_t aRankOff;
  uint64_t totalBytes;
};

__aicore__ inline uint64_t Align512(uint64_t x) {
  return (x + 511ULL) & ~511ULL;
}

__aicore__ inline Layout ComputeLayout(uint32_t seqLocal, uint32_t numHeads,
                                       uint32_t headDim, uint32_t chunkSize) {
  Layout L{};
  const uint64_t S = static_cast<uint64_t>(seqLocal);
  const uint64_t H = static_cast<uint64_t>(numHeads);
  const uint64_t d = static_cast<uint64_t>(headDim);
  const uint64_t C = static_cast<uint64_t>(chunkSize);
  const uint64_t NT = S / C;
  const uint64_t hc = H * NT;
  uint64_t off = 0ULL;
#define KDA_NEXT_OFF(FIELD, BYTES) \
  do {                             \
    L.FIELD = off;                 \
    off = Align512(off + (BYTES)); \
  } while (0)
  KDA_NEXT_OFF(qNormOff,    H * S * d * 2ULL);
  KDA_NEXT_OFF(kNormOff,    H * S * d * 2ULL);
  KDA_NEXT_OFF(vConvOff,    H * S * d * 2ULL);
  KDA_NEXT_OFF(qRstdOff,    H * S * 4ULL);
  KDA_NEXT_OFF(kRstdOff,    H * S * 4ULL);
  KDA_NEXT_OFF(gCumOff,     hc * C * d * 4ULL);
  KDA_NEXT_OFF(mInvOff,     hc * C * C * 2ULL);
  KDA_NEXT_OFF(aqkOff,      hc * C * C * 2ULL);
  KDA_NEXT_OFF(wOff,        hc * C * d * 2ULL);
  KDA_NEXT_OFF(uOff,        hc * C * d * 2ULL);
  KDA_NEXT_OFF(qgOff,       hc * C * d * 2ULL);
  KDA_NEXT_OFF(kgOff,       hc * C * d * 2ULL);
  KDA_NEXT_OFF(expGLastOff, hc * d * 4ULL);
  KDA_NEXT_OFF(hChunksOff,  hc * d * d * 2ULL);
  KDA_NEXT_OFF(vNewOff,     hc * C * d * 2ULL);
  KDA_NEXT_OFF(oAttnOff,    hc * C * d * 2ULL);
  KDA_NEXT_OFF(oRstdOff,    H * S * 4ULL);
  KDA_NEXT_OFF(haloQkvOff,  3ULL * 3ULL * H * d * 2ULL);
  KDA_NEXT_OFF(aRankOff,    H * d * d * 4ULL);
#undef KDA_NEXT_OFF
  L.totalBytes = off;
  return L;
}

}  // namespace MegaKdaCache

#endif  // HYPER_MEGA_KDA_BWD_CACHE_LAYOUT_H
