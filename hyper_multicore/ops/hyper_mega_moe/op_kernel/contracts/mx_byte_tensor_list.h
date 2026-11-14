// Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#ifndef HYPER_MULTICORE_MX_BYTE_TENSOR_LIST_H_
#define HYPER_MULTICORE_MX_BYTE_TENSOR_LIST_H_
#include <cstdint>

#ifdef __NPU_ARCH__
#define MX_BYTE_LIST_INLINE __aicore__ inline
#else
#define MX_BYTE_LIST_INLINE inline
#endif
namespace MulticoreMx {
MX_BYTE_LIST_INLINE bool IsMxByteListHeaderValid(uint64_t pointer_offset, uint32_t count) {
  // Each folded UINT8 vector contributes one rank word and one byte-length word.
  return count > 0 && count <= 64 && pointer_offset == sizeof(uint64_t) * (1 + 2 * count);
}

MX_BYTE_LIST_INLINE bool IsMxByteVectorShapeValid(uint64_t rank_word, uint64_t bytes, uint64_t expected_bytes) {
  // The upper rank-word bits are CANN descriptor metadata, not the list length.
  return (rank_word & 0xffffffffULL) == 1 && bytes == expected_bytes;
}
}  // namespace MulticoreMx
#undef MX_BYTE_LIST_INLINE
#endif
