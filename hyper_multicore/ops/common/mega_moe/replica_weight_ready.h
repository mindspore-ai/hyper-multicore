// Copyright 2026 Huawei Technologies Co., Ltd. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#ifndef HMC_MOE_REPLICA_WEIGHT_READY_H
#define HMC_MOE_REPLICA_WEIGHT_READY_H

#include "kernel_operator.h"
#include "common/mega_moe/replica_trace.h"
#include "runtime/runtime_config.hpp"

namespace MulticoreMoE {

// Appended after the intact runtime and its explicit worker queues.
struct ReplicaLaunch {
  uint64_t magic, home, epoch;
  uint64_t ready[2], local[2], guest[2];
  uint64_t gradient;
};
constexpr uint64_t kReplicaLaunch = 0x4d4f4f4e45505752ULL;
static_assert(sizeof(ReplicaLaunch) == 80);

template <bool Producer = false>
__aicore__ inline void WaitReplicaEpoch(uint64_t address, uint64_t epoch) {
  const auto word = reinterpret_cast<const __gm__ volatile int32_t *>(address);
  AscendC::GlobalTensor<int32_t> flag;
  flag.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(address), 1);
  uint64_t refreshed = 0;
  do {
    // A producer can remain busy for many tiles. Keep waiting lanes from
    // continuously invalidating its cache line during that interval.
    const uint64_t now = Producer ? AscendC::GetSystemCycle() : 0;
    if (!Producer || now - refreshed >= 50) {
      AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                        AscendC::DcciDst::CACHELINE_OUT>(flag);
      refreshed = now;
    }
  } while (*word < static_cast<int32_t>(epoch));
}

__aicore__ inline const __gm__ ReplicaLaunch *GetReplicaLaunch(GM_ADDR runtime, GM_ADDR tiling) {
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
  using MulticoreRuntime::ATOMIC_ADD_VALUE_LEN;
  using MulticoreRuntime::RuntimeHeader;
  using MulticoreRuntime::WorkerQueue;
  using MulticoreRuntime::getAtomicAddValuesOffset;
  using MulticoreRuntime::getExtraValueFromTiling;
  using MulticoreRuntime::usesExplicitWorkerQueues;
  const auto header = reinterpret_cast<const __gm__ RuntimeHeader *>(runtime);
  uint64_t offset = getAtomicAddValuesOffset(runtime) + ATOMIC_ADD_VALUE_LEN * sizeof(int32_t);
  if (usesExplicitWorkerQueues(runtime)) {
    offset += (header->num_workers / 2 + header->num_workers) * sizeof(WorkerQueue);
  }
  offset = (offset + 7) & ~7ULL;
  constexpr uint64_t replica_bytes = sizeof(ReplicaLaunch);
  if (getExtraValueFromTiling(tiling, 6) == offset + replica_bytes) {
    const auto record = reinterpret_cast<const __gm__ ReplicaLaunch *>(runtime + offset);
    if (record->magic == kReplicaLaunch) return record;
  }
#endif
  return nullptr;
}

__aicore__ inline GM_ADDR ReplicaWeight(GM_ADDR ordinary, uint64_t matrix_bytes, uint32_t expert,
                                       uint32_t projection, const __gm__ ReplicaLaunch *launch,
                                       bool single_group, MulticoreRuntime::CycleTraceRecorder *trace,
                                       uint32_t profile_owner) {
  if (launch == nullptr) return ordinary;
  if (!single_group || launch->epoch == 0 || launch->epoch > 0x7fffffff) AscendC::Trap();
  if (expert < launch->home) {
    return reinterpret_cast<GM_ADDR>(launch->local[projection] + expert * matrix_bytes);
  }
  const uint32_t slot = expert - launch->home;
  const auto start = ReplicaTraceCycle(trace);
  WaitReplicaEpoch(launch->ready[projection] + slot * MulticoreRuntime::DATA_CACHE_LINE_SIZE, launch->epoch);
  RecordReplicaStage(trace, ReplicaTraceStage::WEIGHT_WAIT, projection, start, ReplicaTraceCycle(trace),
                     slot, profile_owner);
  return reinterpret_cast<GM_ADDR>(launch->guest[projection] + slot * matrix_bytes);
}

}  // namespace MulticoreMoE

#endif  // HMC_MOE_REPLICA_WEIGHT_READY_H
