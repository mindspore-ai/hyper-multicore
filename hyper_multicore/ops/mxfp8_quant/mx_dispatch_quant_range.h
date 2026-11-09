// Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#ifndef HYPER_MULTICORE_MX_DISPATCH_QUANT_RANGE_H_
#define HYPER_MULTICORE_MX_DISPATCH_QUANT_RANGE_H_
#include <cstdint>
namespace MulticoreMx {
struct MxQuantRange { int64_t begin, end; };

__aicore__ inline MxQuantRange MxFragmentInterior(int64_t begin, int64_t end, int64_t total) {
  const int64_t aligned_begin = (begin + 63) / 64 * 64;
  const int64_t aligned_end = end == total ? end : end / 64 * 64;
  return aligned_begin < aligned_end ? MxQuantRange{aligned_begin, aligned_end} : MxQuantRange{0, 0};
}

__aicore__ inline bool MxFragmentOwnsBlock(int64_t begin, int64_t end,
                                        int64_t source_begin, int64_t source_end, int64_t split) {
  return source_begin <= begin && end <= source_end && begin < end
    && (begin - source_begin) / split == (end - 1 - source_begin) / split;
}
}  // namespace MulticoreMx
#endif
