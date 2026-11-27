/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0.
 */
#include <cstddef>

#include "kda_compute.h"
#include "runtime/worker_kernel.h"
#include "shmem.h"

namespace MulticoreKda {

using MulticoreRuntime::KernelWorkerBase;
using MulticoreRuntime::ReadyHandshakeMeta;
using MulticoreRuntime::RuntimeStorageBoundsPolicy;
using MulticoreRuntime::TaskDesc;
using MulticoreRuntime::TaskType;
using MulticoreRuntime::VectorWorkerPolicy;

struct KdaInvocationProtocol {
  template <class Worker>
  __aicore__ static inline void Publish(Worker &worker, const ReadyHandshakeMeta &meta, bool completion) {
#if !defined(__DAV_CUBE__) && !defined(__DAV_C220_CUBE__)
    const uint32_t cpSize = worker.GetTiling()->cpSize;
    const uint32_t cpRank = worker.GetTiling()->cpRank;
    if (cpSize <= 1U) {
      if (!completion && meta.ready_event != 0U) {
        worker.TriggerEvent(meta.ready_event);
      }
      return;
    }
    constexpr uint32_t readyStride = MulticoreRuntime::DATA_CACHE_LINE_SIZE / MulticoreRuntime::INT32_T_SIZE;
    __gm__ int32_t *ready =
        reinterpret_cast<__gm__ int32_t *>(worker.input_list[Worker::EVENT_IDX]) + worker.runtime_event_capacity;
    if (completion) {
      ready += static_cast<uint64_t>(cpSize + 1U) * readyStride;
    }

    GlobalTensor<int32_t> readyState;
    readyState.SetGlobalBuffer(ready, static_cast<uint32_t>((cpSize + 1U) * readyStride));
    const uint32_t genIdx = cpSize * readyStride;
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(readyState[genIdx]);
    const int32_t prevGen = readyState.GetValue(genIdx);
    if (prevGen == INT32_MAX) {
      AscendC::Trap();
    }
    const int32_t gen = prevGen + 1;
    readyState.SetValue(genIdx, gen);
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(readyState[genIdx]);
    PipeBarrier<PIPE_ALL>();

    uint32_t round = 0;
    for (uint32_t dist = 1; dist < cpSize; dist <<= 1, ++round) {
      const int target = static_cast<int>((cpRank + dist) % cpSize);
      __gm__ int32_t *roundReady = ready + static_cast<uint64_t>(round) * readyStride;
      aclshmemx_signal_op(roundReady, gen, ACLSHMEM_SIGNAL_SET, target);
      aclshmem_signal_wait_until(roundReady, ACLSHMEM_CMP_GE, gen);
    }
    if (!completion) {
      worker.TriggerEvent(meta.ready_event);
    }
#endif
  }
};

class HyperMegaKdaWorkerKernel
    : public KernelWorkerBase<HyperMegaKdaWorkerKernel, VectorWorkerPolicy::ALL,
                              RuntimeStorageBoundsPolicy::GRAPH_OWNED, KdaInvocationProtocol> {
 public:
  static constexpr uint32_t TILING_IDX = IDX_TILING;
  static constexpr uint32_t EVENT_IDX = IDX_EVENT;
  static constexpr uint32_t DIAGNOSTIC_IDX = IDX_DIAG;
  static constexpr size_t RUNTIME_BOUNDS_OFFSET = offsetof(HyperMegaKdaLaunchTilingData, runtimeConfigBytes);

  __aicore__ inline void SetTiling(const HyperMegaKdaLaunchTilingData *t) { tiling_ = t; }
  __aicore__ inline const HyperMegaKdaLaunchTilingData *GetTiling() const { return tiling_; }

  __aicore__ inline uint32_t GetReadyGroupSize() const {
    return (tiling_->cpSize > 1U) ? tiling_->cpSize : 1U;
  }

  __aicore__ inline void ExecuteComputeKernel(TaskDesc taskDesc) {
    const HyperMegaKdaLaunchTilingData *t = tiling_;
    switch (static_cast<TaskType>(taskDesc.task_type)) {
      case TaskType::TASK_KDA_PROJ_BATCH:
        ExecuteProjBatch(taskDesc, this->input_list, t);
        break;
      case TaskType::TASK_KDA_SHORT_CONV:
        ExecuteShortConv(taskDesc, this->input_list, t);
        break;
      case TaskType::TASK_KDA_PRE_GATE:
        ExecutePreGate(taskDesc, this->input_list, t);
        break;
      case TaskType::TASK_KDA_INTRA_CUBE:
        ExecuteIntraCube(taskDesc, this->input_list, t);
        break;
      case TaskType::TASK_KDA_TRI_DIAG16:
        ExecuteTriDiag16(taskDesc, this->input_list, t);
        break;
      case TaskType::TASK_KDA_WY_CUBE:
        ExecuteWyCube(taskDesc, this->input_list, t);
        break;
      case TaskType::TASK_KDA_FWD_H_AIC:
        ExecuteFwdHAic(taskDesc, this->input_list, t);
        break;
      case TaskType::TASK_KDA_FWD_H_AIV:
        ExecuteFwdHAiv(taskDesc, this->input_list, t);
        break;
      case TaskType::TASK_KDA_FINALIZE_O:
        ExecuteFinalizeO(taskDesc, this->input_list, t);
        break;
      case TaskType::TASK_KDA_GATED_RMSNORM:
        ExecuteGatedRmsNorm(taskDesc, this->input_list, t);
        break;
      case TaskType::TASK_KDA_INTER_SUMMARY_AIC:
        ExecuteInterSummaryAic(taskDesc, this->input_list, t);
        break;
      case TaskType::TASK_KDA_INTER_SUMMARY_AIV:
        ExecuteInterSummaryAiv(taskDesc, this->input_list, t);
        break;
      case TaskType::TASK_KDA_SCAN_MERGE_AIC:
        ExecuteScanMergeAic(taskDesc, this->input_list, t);
        break;
      case TaskType::TASK_KDA_SCAN_MERGE_AIV:
        ExecuteScanMergeAiv(taskDesc, this->input_list, t);
        break;
      case TaskType::TASK_SHMEM_PUT_MEM_SIGNAL:
        ExecuteShmemPut(taskDesc, this->input_list, t);
        break;
      default:
        break;
    }
  }

 private:
  const HyperMegaKdaLaunchTilingData *tiling_{nullptr};
};

__aicore__ inline void RunHyperMegaKdaWorker(uint32_t workerId, __gm__ uint8_t *runtimeConfig, GM_ADDR *inputList,
                                             const HyperMegaKdaLaunchTilingData *tilingData) {
  HyperMegaKdaWorkerKernel worker;
  worker.SetTiling(tilingData);
  worker.Init(workerId, runtimeConfig, inputList);
  worker.Process();
}

}  // namespace MulticoreKda
