// Copyright 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#ifndef HYPER_MULTICORE_OPS_RUNTIME_WORKER_READY_QUEUE_H_
#define HYPER_MULTICORE_OPS_RUNTIME_WORKER_READY_QUEUE_H_
#include <cstdint>

namespace MulticoreRuntime {
// Selection is independent of core kind, placement, waiting and diagnostics.
// The bitmap tracks completed slots in this worker's view, never other workers.
template <typename Queue>
__aicore__ inline void RunWorkerQueue(Queue &queue, uint32_t first, uint32_t end,
                                    uint32_t stride, uint32_t depth) {
  if (first >= end) return;
  const uint32_t slots = (end - first + stride - 1) / stride;
  uint64_t completed[4] = {};
  uint32_t head = 0;
  while (head < slots) {
    uint32_t selected = head;
    if (depth != 0) {
      const uint32_t head_bit = head % 256;
      if (completed[head_bit / 64] & (1ULL << (head_bit % 64))) {
        completed[head_bit / 64] &= ~(1ULL << (head_bit % 64));
        ++head;
        continue;
      }
      selected = slots;
      const uint32_t limit = slots - head < depth ? slots : head + depth;
      for (uint32_t i = head; i < limit; ++i) {
        const uint32_t bit = i % 256;
        if (completed[bit / 64] & (1ULL << (bit % 64))) continue;
        if (queue.IsReady(queue.TaskAt(first + i * stride))) { selected = i; break; }
      }
      if (selected == slots) { queue.Wait(queue.TaskAt(first + head * stride)); continue; }
    }
    queue.Execute(queue.TaskAt(first + selected * stride));
    if (depth == 0) ++head;
    else {
      const uint32_t bit = selected % 256;
      completed[bit / 64] |= 1ULL << (bit % 64);
    }
  }
}
}  // namespace MulticoreRuntime
#endif
