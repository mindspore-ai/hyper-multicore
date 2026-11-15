/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file worker_kernel.h
 * @brief Shared CRTP base class for AIC/AIV scheduling loop and event synchronization.
 *
 * Each op derives KernelWorker : KernelWorkerBase<KernelWorker> and supplies:
 *   - static constexpr uint32_t TILING_IDX  — index into input_list for tiling params
 *   - static constexpr uint32_t EVENT_IDX   — index into input_list for all_event_counters
 *   - void ExecuteComputeKernel(TaskDesc)    — op-specific task dispatch switch
 *   - static constexpr uint32_t DIAGNOSTIC_IDX — index into input_list for the shared profiling/dump buffer
 *
 * Compute kernels (ExecuteMatmul, ExecuteShmemPutMem, etc.) are NOT part of this class;
 * they belong in each op's worker_kernel.cpp.
 */

#ifndef HYPER_MULTICORE_CORE_MULTICORE_OPS_RUNTIME_WORKER_KERNEL_H_
#define HYPER_MULTICORE_CORE_MULTICORE_OPS_RUNTIME_WORKER_KERNEL_H_

#include "kernel_operator.h"
#include "shmem.h"
#include "runtime_config.hpp"
#include "runtime_storage_bounds.h"
#include "cycle_trace_recorder.h"
#include "l1_tensor_dump.h"
#include "vector_ready_queue.h"


using namespace AscendC;  // NOLINT(build/namespaces)

#if (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510 && defined(__DAV_CUBE__)) || \
    ((!defined(__NPU_ARCH__) || __NPU_ARCH__ != 3510) && defined(__DAV_C220_CUBE__))
#define HMC_RUNTIME_CUBE
#endif

namespace MulticoreRuntime {

constexpr uint32_t VECTOR_WORKER_STRIDE = 2;
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
// GetSystemCycle uses the fixed system counter, independent of AI Core clocks.
constexpr int64_t SYSTEM_CYCLES_PER_US = 1000;
constexpr int64_t CUBE_EVENT_REFRESH_INTERVAL_US = 50;
constexpr int64_t VECTOR_EVENT_REFRESH_INTERVAL_US = 150;
#else
constexpr int64_t EVENT_REFRESH_TIME_UNIT_CYCLES = 50;
constexpr int64_t CUBE_EVENT_REFRESH_INTERVAL_UNITS = 50;     // 2,500 system cycles.
constexpr int64_t VECTOR_EVENT_REFRESH_INTERVAL_UNITS = 150;  // 7,500 system cycles.
#endif
// The selected SwiGLU tiling uses baseRowLen=19; smaller dynamic tails must lower it.
constexpr int64_t SWIGLU_DYNAMIC_TAIL_BASE_ROW_LIMIT = 19;
constexpr int64_t READY_SIGNAL_RADIX = 2;

enum class VectorWorkerPolicy : uint32_t {
  LEGACY_ODD = 0,
  ALL = 1,
};

enum class RuntimeStorageBoundsPolicy : uint32_t {
  NATIVE_TILING = 0,
  GRAPH_OWNED = 1,
};

template <typename Derived, VectorWorkerPolicy WorkerPolicy = VectorWorkerPolicy::LEGACY_ODD,
          RuntimeStorageBoundsPolicy StorageBoundsPolicy = RuntimeStorageBoundsPolicy::NATIVE_TILING>
class KernelWorkerBase {
 public:
  __aicore__ inline KernelWorkerBase() {}

  static constexpr uint32_t PROFILE_DESC_WAIT_DEPENDENCY = 0x10000;
  static constexpr uint32_t PROFILE_DESC_TRIGGER_EVENT = 0x10007;
  static constexpr uint32_t PROFILE_DESC_TASK_TYPE_BASE = 0x20000;

  __aicore__ inline void Init(uint32_t worker_id, __gm__ uint8_t *runtimeConfigPtr, GM_ADDR *input_list) {
    this->worker_id_ = worker_id;
    this->runtimeConfigPtr = runtimeConfigPtr;

    uint64_t runtime_bytes = 0;
    uint64_t event_bytes = 0;
    uint32_t ep_size = 1;
    uint64_t local_experts = 0;
    if constexpr (StorageBoundsPolicy == RuntimeStorageBoundsPolicy::NATIVE_TILING) {
      runtime_bytes = getExtraValueFromTiling(input_list[Derived::TILING_IDX], 6);
      event_bytes = getExtraValueFromTiling(input_list[Derived::TILING_IDX], 7);
      ep_size = static_cast<uint32_t>(getExtraValueFromTiling(input_list[Derived::TILING_IDX], 1));
      local_experts = getExtraValueFromTiling(input_list[Derived::TILING_IDX], 2) / ep_size;
    } else {
      auto bounds = reinterpret_cast<__gm__ RuntimeStorageBounds *>(
        input_list[Derived::TILING_IDX] + Derived::RUNTIME_BOUNDS_OFFSET);
      runtime_bytes = bounds->runtime_config_bytes;
      event_bytes = bounds->event_counter_bytes;
    }
    if (!isRuntimeStorageValid(runtimeConfigPtr, runtime_bytes, event_bytes, ep_size, local_experts)) {
      AscendC::Trap();
    }
    this->runtime_task_capacity = getRuntimeTaskCapacity(runtimeConfigPtr);
    this->runtime_event_capacity = getRuntimeEventCapacity(runtimeConfigPtr);
#ifdef HMC_RUNTIME_CUBE
    // The worker's scratch region is fixed for the lifetime of this task graph.
    this->grouped_matmul_group_list_offset_ = getGroupedMatmulGroupListOffsetById(runtimeConfigPtr, worker_id);
#endif
    all_event_counters.SetGlobalBuffer((__gm__ int32_t *)(input_list[Derived::EVENT_IDX]), runtime_event_capacity);

    all_event_num_triggers.SetGlobalBuffer((__gm__ int32_t *)(runtimeConfigPtr + getAllEventNumTriggersOffset()),
                                           runtime_event_capacity);
    vector_task_indexs.SetGlobalBuffer(
      (__gm__ int32_t *)(runtimeConfigPtr + getVectorTaskIndexsOffset(runtimeConfigPtr)), runtime_task_capacity);
    cube_task_indexs.SetGlobalBuffer((__gm__ int32_t *)(runtimeConfigPtr + getCubeTaskIndexsOffset(runtimeConfigPtr)),
                                     runtime_task_capacity);
    atomic_add_values.SetGlobalBuffer((__gm__ int32_t *)(runtimeConfigPtr + getAtomicAddValuesOffset(runtimeConfigPtr)),
                                      ATOMIC_ADD_VALUE_LEN);

    this->input_list = input_list;
    this->task_num = getTaskNum(this->runtimeConfigPtr);
    this->vector_task_num = getTaskIndexNumByTaskType(this->runtimeConfigPtr, TaskAiCoreType::TASK_AICORE_VECTOR);
    this->cube_task_num = getTaskIndexNumByTaskType(this->runtimeConfigPtr, TaskAiCoreType::TASK_AICORE_CUBE);
    // RuntimeConfig is operator-independent; native tiling buffers are not.
    // Reading a fixed native-tiling offset here corrupts the round-robin stride
    // for operators whose sixth field is not the AIC count (for example mHC).
    this->vector_num = getNumWorkers(this->runtimeConfigPtr);
    this->core_num = this->vector_num / 2;
    const auto header = reinterpret_cast<__gm__ RuntimeHeader *>(runtimeConfigPtr);
    ready_queue_ = header->vector_ready_queue == 1;
    this->active_vector_num = (WorkerPolicy == VectorWorkerPolicy::ALL || ready_queue_)
                              ? this->vector_num : this->vector_num / 2;
  }

  __aicore__ inline void Process() {
    ReadyHandshakeMeta meta;
    bool has_ready = LoadReadyHandshakeMeta(&meta);
    pull_protocol_ = meta.completion_event != 0;
    if (has_ready) {
      WaitForDeviceReady(meta);
    }
    const auto mode = getDiagnosticMode(this->runtimeConfigPtr);
    CycleTraceRecorder recorder;
    if (mode == DiagnosticMode::PROFILE) {
      recorder.Init(this->worker_id_, this->input_list[Derived::DIAGNOSTIC_IDX],
                    getAicProfileRecordCapacity(this->runtimeConfigPtr),
                    getAivProfileRecordCapacity(this->runtimeConfigPtr),
                    getCubeWorkerCapacity(this->runtimeConfigPtr));
    }
    ProcessTasks(mode, recorder);
    if (meta.completion_event != 0) {
      CompleteDeviceReads(meta);
    }
  }

 protected:
  // Share the compute call site across diagnostics. Replicating the entire
  // inlined task dispatcher for each mode exceeds the AIV scalar stack limit.
  __aicore__ inline void ProcessTasks(DiagnosticMode mode, CycleTraceRecorder &recorder) {
#ifdef HMC_RUNTIME_CUBE
    uint32_t block_idx = this->worker_id_;
    const uint32_t count = this->cube_task_num;
    const uint32_t stride = this->core_num;
#else
    if (ready_queue_) {
      ProcessVectorReady(mode, recorder);
      return;
    }
    uint32_t block_idx = GetVectorScheduleIndex();
    if (block_idx == INVALID_VECTOR_SCHEDULE_INDEX) return;
    const uint32_t count = this->vector_task_num;
    const uint32_t stride = this->active_vector_num;
#endif
    for (; block_idx < count; block_idx += stride) {
      ExecuteTask(GetTaskIndex(block_idx), mode, recorder);
    }
  }

  __aicore__ inline bool IsQueuedTaskReady(TaskId task_id) {
    const uint32_t dependent = *reinterpret_cast<__gm__ uint32_t *>(runtimeConfigPtr +
      getAllTasksOffset(runtimeConfigPtr) + task_id * sizeof(TaskDesc) + offsetof(TaskDesc, dependent_event));
    if (dependent == EVENT_INVALID_ID) {
      return true;
    }
    if (dependent >= runtime_event_capacity) {
      AscendC::Trap();
    }
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
      all_event_counters[dependent]);
    return all_event_counters.GetValue(dependent) >= all_event_num_triggers.GetValue(dependent);
  }

  struct ReadyExecution {
    KernelWorkerBase &worker;
    DiagnosticMode mode;
    CycleTraceRecorder &recorder;
    __aicore__ inline TaskId TaskAt(uint32_t index) { return worker.GetTaskIndex(index); }
    __aicore__ inline bool IsReady(TaskId task) { return worker.IsQueuedTaskReady(task); }
    __aicore__ inline void Execute(TaskId task) {
      worker.ExecuteTask(task, mode, recorder);
    }
    __aicore__ inline void Wait(TaskId task) {
      const uint32_t dependent = *reinterpret_cast<__gm__ uint32_t *>(worker.runtimeConfigPtr +
        getAllTasksOffset(worker.runtimeConfigPtr) + task * sizeof(TaskDesc) + offsetof(TaskDesc, dependent_event));
      worker.WaitForDependency(dependent, 5);
    }
  };

  __aicore__ inline void ProcessVectorReady(DiagnosticMode mode, CycleTraceRecorder &recorder) {
    const uint32_t first = GetVectorScheduleIndex();
    if (first == INVALID_VECTOR_SCHEDULE_INDEX) return;
    ReadyExecution execution{*this, mode, recorder};
    RunVectorReadyQueue(execution, first, this->vector_task_num, this->active_vector_num);
  }

  __aicore__ inline bool LoadReadyHandshakeMeta(ReadyHandshakeMeta *meta) {
    getReadyHandshakeMeta(this->runtimeConfigPtr, meta);
    return meta->ready_event != 0;
  }

  __aicore__ inline void PublishDeviceReady(const ReadyHandshakeMeta &meta, bool completion = false) {
#ifndef HMC_RUNTIME_CUBE
    int64_t ep = getExtraValueFromTiling(input_list[Derived::TILING_IDX], 1);
    int64_t rank = getExtraValueFromTiling(input_list[Derived::TILING_IDX], 0) % ep;
    constexpr uint32_t ready_stride = DATA_CACHE_LINE_SIZE / INT32_T_SIZE;
    __gm__ int32_t *ready = (__gm__ int32_t *)(input_list[Derived::EVENT_IDX]) + runtime_event_capacity;
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
    for (int64_t distance = 1; distance < ep; distance *= READY_SIGNAL_RADIX, ++round) {
      int64_t target = (rank + distance) % ep;
      __gm__ int32_t *round_ready = ready + round * ready_stride;
      aclshmemx_signal_op(round_ready, generation, ACLSHMEM_SIGNAL_SET, static_cast<int>(target));
      aclshmem_signal_wait_until(round_ready, ACLSHMEM_CMP_GE, generation);
    }
    if (!completion) {
      TriggerEvent(meta.ready_event);
    }
#endif
  }

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
  __aicore__ inline void CompleteDeviceReads(const ReadyHandshakeMeta &meta) {
#ifndef HMC_RUNTIME_CUBE
    if (!IsActiveVectorWorker()) {
      return;
    }
#endif
    TriggerEvent(meta.completion_event);
#ifndef HMC_RUNTIME_CUBE
    if (this->worker_id_ == 1) {
      WaitForDependency(meta.completion_event);
      if (meta.ready_event != 0) {
        PublishDeviceReady(meta, true);
      }
    }
#endif
  }
#else
  __aicore__ inline void CompleteDeviceReads(const ReadyHandshakeMeta &meta) {
#ifndef HMC_RUNTIME_CUBE
    if (this->worker_id_ % 2 == 0) {
      return;
    }
#endif
    TriggerEvent(meta.completion_event);
#ifndef HMC_RUNTIME_CUBE
    if (this->worker_id_ == 1) {
      WaitForDependency(meta.completion_event);
      if (meta.ready_event != 0) {
        PublishDeviceReady(meta, true);
      }
    }
#endif
  }
#endif


#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
  __aicore__ inline void WaitForDeviceReady(const ReadyHandshakeMeta &meta) {
#ifdef HMC_RUNTIME_CUBE
    WaitForDependency(meta.ready_event);
#else
    if (!IsActiveVectorWorker()) {
      return;
    }
    if (this->worker_id_ == 1) {
      PublishDeviceReady(meta);
    }
    WaitForDependency(meta.ready_event);
#endif
  }
#else
  __aicore__ inline void WaitForDeviceReady(const ReadyHandshakeMeta &meta) {
#ifdef HMC_RUNTIME_CUBE
    WaitForDependency(meta.ready_event);
#else
    if (this->active_vector_num != this->vector_num && this->worker_id_ % VECTOR_WORKER_STRIDE == 0) {
      return;
    }
    const uint32_t leader = this->active_vector_num == this->vector_num ? 0 : 1;
    if (this->worker_id_ == leader) {
      PublishDeviceReady(meta);
    }
    WaitForDependency(meta.ready_event);
#endif
  }
#endif

  __aicore__ inline TaskId GetTaskIndex(uint32_t task_id) {
#ifdef HMC_RUNTIME_CUBE
    return cube_task_indexs.GetValue(task_id);
#else
    return vector_task_indexs.GetValue(task_id);
#endif
  }

  __aicore__ inline uint32_t GetVectorScheduleIndex() {
    if (WorkerPolicy == VectorWorkerPolicy::ALL || ready_queue_) {
      return this->worker_id_ < this->active_vector_num ? this->worker_id_ : INVALID_VECTOR_SCHEDULE_INDEX;
    }
    return this->worker_id_ % 2 == 0 ? INVALID_VECTOR_SCHEDULE_INDEX : this->worker_id_ / 2;
  }

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
  __aicore__ inline bool IsActiveVectorWorker() const {
    if (WorkerPolicy == VectorWorkerPolicy::ALL || ready_queue_) {
      return this->worker_id_ < this->active_vector_num;
    }
    return this->worker_id_ % VECTOR_WORKER_STRIDE != 0;
  }
#endif

  __aicore__ inline void AtomicAddForAllEventCounters(uint32_t event_index) {
#if defined(HMC_RUNTIME_CUBE) && defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
    // A5 cannot publish an L1 buffer to GM; preserve the original scalar atomic completion path.
    PipeBarrier<PIPE_ALL>();
    (void)AscendC::AtomicAdd<int32_t>(this->all_event_counters.GetPhyAddr(event_index), 1);
    PipeBarrier<PIPE_ALL>();
#else
    TPipe eventPipe;
#ifdef HMC_RUNTIME_CUBE
    TBuf<AscendC::TPosition::A1> eventBuffer;
#else
    TBuf<AscendC::TPosition::VECOUT> eventBuffer;
#endif
    eventPipe.InitBuffer(eventBuffer, DISPATCH_TOKEN_UB_SIZE);
    LocalTensor<int32_t> localSet = eventBuffer.GetWithOffset<int32_t>(EXP_TOKEN_COUNT_FLAG_CNT, 0);
#ifdef HMC_RUNTIME_CUBE
    SyncFunc<AscendC::HardEvent::S_MTE2>();
    DataCopy(localSet, this->atomic_add_values, EXP_TOKEN_COUNT_FLAG_CNT);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    AscendC::SetAtomicAdd<int32_t>();
    SyncFunc<AscendC::HardEvent::S_MTE2>();
    DataCopy(this->all_event_counters[event_index], localSet, EXP_TOKEN_COUNT_FLAG_CNT);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    AscendC::SetAtomicNone();
#else
    localSet.SetValue(0, 1);
    for (int32_t i = 1; i < EXP_TOKEN_COUNT_FLAG_CNT; ++i) {
      localSet.SetValue(i, 0);
    }
    AscendC::SetAtomicAdd<int32_t>();
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopy(this->all_event_counters[event_index], localSet, EXP_TOKEN_COUNT_FLAG_CNT);
    SyncFunc<AscendC::HardEvent::MTE3_S>();
    AscendC::SetAtomicNone();
#endif
    eventBuffer.FreeTensor(localSet);
    eventPipe.Destroy();
#endif
  }

  __aicore__ inline void CaptureL1TaskTensors(TaskId task_id, uint32_t phase) {
    MulticoreRuntime::CaptureL1TaskTensors(input_list[Derived::DIAGNOSTIC_IDX], input_list, task_id, phase, worker_id_);
  }

  __aicore__ inline void LoadTaskDescriptor(TaskId task_id, TaskDesc &task_desc) {
    if (task_id >= runtime_task_capacity) {
      AscendC::Trap();
    }
    getTaskDesc(this->runtimeConfigPtr, &task_desc, task_id);
    if ((task_desc.dependent_event != EVENT_INVALID_ID && task_desc.dependent_event >= runtime_event_capacity) ||
        static_cast<uint64_t>(task_desc.trigger_event) + ATOMIC_ADD_VALUE_LEN > runtime_event_capacity) {
      AscendC::Trap();
    }
  }

  __aicore__ inline void ExecuteTask(TaskId task_id, DiagnosticMode mode, CycleTraceRecorder &recorder) {
    TaskDesc task_desc;
    LoadTaskDescriptor(task_id, task_desc);
    const bool profile = mode == DiagnosticMode::PROFILE;
    const bool dump = mode == DiagnosticMode::TENSOR_DUMP_L1;
    const uint32_t owner_id = profile ? getTaskProfileOwnerId(this->runtimeConfigPtr, task_id) : 0;
    if (task_desc.dependent_event != EVENT_INVALID_ID) {
      const uint64_t start = profile ? recorder.Now() : 0;
      WaitForDependency(task_desc.dependent_event, task_desc.extra_value_2);
      if (profile) {
        recorder.Record(Derived::PROFILE_DESC_WAIT_DEPENDENCY, task_id, task_desc.task_index, owner_id,
                        start, recorder.Now());
      }
    }
    if (dump) CaptureL1TaskTensors(task_id, L1_DUMP_PHASE_BEFORE);
    const uint64_t compute_start = profile ? recorder.Now() : 0;
    static_cast<Derived *>(this)->ExecuteComputeKernel(task_desc);
    if (profile) {
      const uint64_t compute_end = recorder.Now();
      uint32_t desc_id = getTaskProfileDescId(this->runtimeConfigPtr, task_id);
      if (desc_id == PROFILE_DESC_INVALID_ID) {
        desc_id = PROFILE_DESC_TASK_TYPE_BASE + static_cast<uint32_t>(task_desc.task_type);
      }
      recorder.Record(desc_id, task_id, task_desc.task_index, owner_id, compute_start, compute_end);
    }
    if (dump) {
      PipeBarrier<PIPE_ALL>();
      CaptureL1TaskTensors(task_id, L1_DUMP_PHASE_AFTER);
    }
    if (task_desc.task_type != TaskType::TASK_SHMEM_PUT_MEM_SIGNAL) {
      const uint64_t start = profile ? recorder.Now() : 0;
      TriggerEvent(task_desc.trigger_event);
      if (profile) {
        recorder.Record(Derived::PROFILE_DESC_TRIGGER_EVENT, task_id, task_desc.task_index, owner_id,
                        start, recorder.Now());
      }
    }
  }

  __aicore__ inline void WaitForDependency(uint32_t event_index, uint32_t poll_interval_us = 0) {
    // Pull producers signal locally after MTE3 completion. Observe short-lived
    // GMM/SwiGLU dependencies promptly while retaining a bounded refresh rate.
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
#ifdef HMC_RUNTIME_CUBE
    const uint32_t default_poll_interval_us = pull_protocol_ ? 10 : CUBE_EVENT_REFRESH_INTERVAL_US;
#else
    const uint32_t default_poll_interval_us = pull_protocol_ ? 30 : VECTOR_EVENT_REFRESH_INTERVAL_US;
#endif
#else
#ifdef HMC_RUNTIME_CUBE
    const uint32_t default_poll_interval_us = pull_protocol_ ? 10 : CUBE_EVENT_REFRESH_INTERVAL_UNITS;
#else
    const uint32_t default_poll_interval_us = pull_protocol_ ? 30 : VECTOR_EVENT_REFRESH_INTERVAL_UNITS;
#endif
#endif
    if (poll_interval_us == 0) {
      poll_interval_us = default_poll_interval_us;
    }
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
    constexpr int64_t cycles_per_us = SYSTEM_CYCLES_PER_US;
#else
    constexpr int64_t cycles_per_us = EVENT_REFRESH_TIME_UNIT_CYCLES;
#endif
    int32_t needed = all_event_num_triggers.GetValue(event_index);
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
      all_event_counters[event_index]);
    int32_t current = all_event_counters.GetValue(event_index);
    PipeBarrier<PIPE_ALL>();
    int64_t previous_cycle = AscendC::GetSystemCycle();
    while (current < needed) {
      int64_t elapsed_cycles = AscendC::GetSystemCycle() - previous_cycle;
      if (elapsed_cycles / cycles_per_us > poll_interval_us) {
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
          all_event_counters[event_index]);
        current = all_event_counters.GetValue(event_index);
        previous_cycle = AscendC::GetSystemCycle();
      }
    }
  }

  __aicore__ inline void TriggerEvent(uint32_t event_index) { AtomicAddForAllEventCounters(event_index); }

  bool pull_protocol_ = false;
  uint32_t worker_id_ = 0;
#if !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510)
  bool ready_queue_ = false;
#endif
  uint32_t grouped_matmul_group_list_offset_ = 0;
  uint32_t task_num = 0;
  uint32_t runtime_task_capacity = 0;
  uint32_t runtime_event_capacity = 0;
  int32_t vector_task_num = 0;
  int32_t cube_task_num = 0;

  GlobalTensor<int32_t> all_event_counters;
  GlobalTensor<int32_t> all_event_num_triggers;
  GlobalTensor<int32_t> atomic_add_values;
  GlobalTensor<int32_t> vector_task_indexs;
  GlobalTensor<int32_t> cube_task_indexs;

  __gm__ uint8_t *runtimeConfigPtr;
  GM_ADDR *input_list = nullptr;
  int64_t core_num = 0;
  int64_t vector_num = 0;
  uint32_t active_vector_num = 0;
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
  bool ready_queue_ = false;
#endif
  static constexpr uint32_t INVALID_VECTOR_SCHEDULE_INDEX = 0xFFFFFFFF;
};

}  // namespace MulticoreRuntime

#undef HMC_RUNTIME_CUBE
#endif  // HYPER_MULTICORE_CORE_MULTICORE_OPS_RUNTIME_WORKER_KERNEL_H_
