// Copyright 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "shmem.h"

namespace MulticoreMoE {
// Generation-separated source readiness and read completion are an MoE
// distributed-buffer protocol, independent of task queue selection.
struct InvocationProtocol {
  template <class Worker>
  __aicore__ static inline void Publish(Worker &worker, const MulticoreRuntime::ReadyHandshakeMeta &meta,
                                        bool completion) {
#if !defined(__DAV_CUBE__) && !defined(__DAV_C220_CUBE__)
    using namespace MulticoreRuntime;
    int64_t ep = getExtraValueFromTiling(worker.input_list[Worker::TILING_IDX], 1);
    int64_t rank = getExtraValueFromTiling(worker.input_list[Worker::TILING_IDX], 0) % ep;
    constexpr uint32_t ready_stride = DATA_CACHE_LINE_SIZE / INT32_T_SIZE;
    __gm__ int32_t *ready = (__gm__ int32_t *)(worker.input_list[Worker::EVENT_IDX]) + worker.runtime_event_capacity;
    if (completion) {
      ready += (ep + 1) * ready_stride;
    }

    GlobalTensor<int32_t> ready_state;
    ready_state.SetGlobalBuffer(ready, static_cast<uint32_t>((ep + 1) * ready_stride));
    uint32_t generation_index = static_cast<uint32_t>(ep) * ready_stride;
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
      ready_state[generation_index]);
    int32_t previous_generation = ready_state.GetValue(generation_index);
    if (previous_generation == INT32_MAX) {
      AscendC::Trap();
    }
    int32_t generation = previous_generation + 1;
    ready_state.SetValue(generation_index, generation);
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
      ready_state[generation_index]);
    PipeBarrier<PIPE_ALL>();

    uint32_t round = 0;
    for (int64_t distance = 1; distance < ep; distance *= 2, ++round) {
      int64_t target = (rank + distance) % ep;
      __gm__ int32_t *round_ready = ready + round * ready_stride;
      aclshmemx_signal_op(round_ready, generation, ACLSHMEM_SIGNAL_SET, static_cast<int>(target));
      aclshmem_signal_wait_until(round_ready, ACLSHMEM_CMP_GE, generation);
    }
    if (!completion) {
      worker.TriggerEvent(meta.ready_event);
    }
#endif
  }
};
}  // namespace MulticoreMoE
