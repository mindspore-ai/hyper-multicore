// Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#ifndef HYPER_MULTICORE_TRANSPORT_CREDITS_H_
#define HYPER_MULTICORE_TRANSPORT_CREDITS_H_
#include <cstdint>
namespace MulticoreRuntime {
#ifdef __NPU_ARCH__
__aicore__ inline
#else
inline
#endif
uint32_t UdmaQpForWorker(uint32_t worker, uint32_t qps) {
  return worker / (64U / qps);
}
#ifdef __NPU_ARCH__
__aicore__ inline
#else
inline
#endif
uint32_t EmptyFragmentCredits(uint32_t size, uint32_t tile, uint32_t fragments, uint32_t index) {
  // Only the first empty fragment publishes the whole empty suffix; real transfers retain ADD1 after quiet.
  const uint32_t first_empty = (static_cast<uint64_t>(size) + tile - 1) / tile;
  return index == first_empty ? fragments - first_empty : 0;
}
}  // namespace MulticoreRuntime
#endif
