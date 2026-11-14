// Copyright 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#ifndef HYPER_MULTICORE_OPS_RUNTIME_VECTOR_READY_QUEUE_H_
#define HYPER_MULTICORE_OPS_RUNTIME_VECTOR_READY_QUEUE_H_
#include <cstdint>

namespace MulticoreRuntime {
// Queue owns execution/recording. Selection and worker placement are identical
// for ordinary, profiled and tensor-dump invocations.
template <typename Queue>
__aicore__ inline void RunVectorReadyQueue(Queue &queue, uint32_t first, uint32_t tasks, uint32_t stride) {
  constexpr uint32_t words = 4;
  constexpr uint32_t bits = 64 * words;
  if (first >= tasks) return;
  const uint32_t slots = (tasks - first + stride - 1) / stride;
  uint64_t completed[words] = {};
  uint32_t head = 0;
  while (head < slots) {
    const uint32_t head_bit = head % bits;
    if ((completed[head_bit / 64] & (1ULL << (head_bit % 64))) != 0) {
      completed[head_bit / 64] &= ~(1ULL << (head_bit % 64));
      ++head;
      continue;
    }
    bool issued = false;
    const uint32_t limit = head + bits < slots ? head + bits : slots;
    for (uint32_t index = head; index < limit; ++index) {
      const uint32_t bit = index % bits;
      if ((completed[bit / 64] & (1ULL << (bit % 64))) != 0) continue;
      const uint32_t task = queue.TaskAt(first + index * stride);
      if (!queue.IsReady(task)) continue;
      queue.Execute(task);
      completed[bit / 64] |= 1ULL << (bit % 64);
      issued = true;
      break;
    }
    if (!issued) queue.Wait(queue.TaskAt(first + head * stride));
  }
}
}  // namespace MulticoreRuntime
#endif
