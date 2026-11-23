// Copyright 2026 Huawei Technologies Co., Ltd. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#ifndef HMC_MOE_REPLICA_WEIGHT_READY_H
#define HMC_MOE_REPLICA_WEIGHT_READY_H

#include "kernel_operator.h"
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

constexpr uint32_t kA5ReplicaHelpers = 4;
struct A5ReplicaLaunch {
  ReplicaLaunch base;
  uint64_t prefetch;
};
static_assert(sizeof(A5ReplicaLaunch) == 88);

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
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201 || __NPU_ARCH__ == 3510)
  using namespace MulticoreRuntime;
  const auto header = reinterpret_cast<const __gm__ RuntimeHeader *>(runtime);
  uint64_t offset = getAtomicAddValuesOffset(runtime) + ATOMIC_ADD_VALUE_LEN * sizeof(int32_t);
  if (usesExplicitWorkerQueues(runtime)) {
    offset += (header->num_workers / 2 + header->num_workers) * sizeof(WorkerQueue);
  }
  offset = (offset + 7) & ~7ULL;
#if __NPU_ARCH__ == 3510
  constexpr uint64_t replica_bytes = sizeof(A5ReplicaLaunch);
#else
  constexpr uint64_t replica_bytes = sizeof(ReplicaLaunch);
#endif
  if (getExtraValueFromTiling(tiling, 6) == offset + replica_bytes) {
    const auto record = reinterpret_cast<const __gm__ ReplicaLaunch *>(runtime + offset);
    if (record->magic == kReplicaLaunch) return record;
  }
#endif
  return nullptr;
}

__aicore__ inline GM_ADDR ReplicaWeight(GM_ADDR ordinary, uint64_t matrix_bytes, uint32_t expert,
                                       uint32_t projection, const __gm__ ReplicaLaunch *launch,
                                       bool single_group) {
  if (launch == nullptr) return ordinary;
  if (!single_group || launch->epoch == 0 || launch->epoch > 0x7fffffff) AscendC::Trap();
  if (expert < launch->home) {
    return reinterpret_cast<GM_ADDR>(launch->local[projection] + expert * matrix_bytes);
  }
  const uint32_t slot = expert - launch->home;
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
  const uint32_t parts = kA5ReplicaHelpers;
#else
  const uint32_t parts = 1;
#endif
  for (uint32_t part = 0; part < parts; ++part) {
    WaitReplicaEpoch(launch->ready[projection] + (slot * parts + part) * MulticoreRuntime::DATA_CACHE_LINE_SIZE,
                     launch->epoch);
  }
  return reinterpret_cast<GM_ADDR>(launch->guest[projection] + slot * matrix_bytes);
}

}  // namespace MulticoreMoE

#endif  // HMC_MOE_REPLICA_WEIGHT_READY_H
