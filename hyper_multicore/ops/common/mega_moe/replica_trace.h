// Copyright 2026 Huawei Technologies Co., Ltd. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#ifndef HMC_MOE_REPLICA_TRACE_H
#define HMC_MOE_REPLICA_TRACE_H

#include "runtime/cycle_trace_recorder.h"

namespace MulticoreMoE {
// Each pair is W13 then W2. Keep in sync with MegaMoE's diagnostic stage names.
enum class ReplicaTraceStage : uint32_t {
  WEIGHT_WAIT = 0x10100,
  OWNER_REDUCE = 0x10102,
  LONGEST_READY_WAIT = 0x10104,
  OWNER_DRAIN = 0x10106,
  ACK_PUBLISH = 0x10108,
  GUEST_ACK_WAIT = 0x1010a,
};

__aicore__ inline uint64_t ReplicaTraceCycle(const MulticoreRuntime::CycleTraceRecorder *trace) {
  // These boundaries bracket scalar flag polls or operations with their own completion.
  return trace == nullptr ? 0 : static_cast<uint64_t>(AscendC::GetSystemCycle());
}

__aicore__ inline void RecordReplicaStage(MulticoreRuntime::CycleTraceRecorder *trace, ReplicaTraceStage stage,
                                          uint32_t projection, uint64_t start, uint64_t end,
                                          uint32_t index = 0, uint32_t owner = UINT32_MAX) {
  if (trace == nullptr) return;
  trace->Record(static_cast<uint32_t>(stage) + projection, UINT32_MAX, index, owner, start, end);
}

struct ReplicaGradientTrace {
  MulticoreRuntime::CycleTraceRecorder *recorder;
  uint32_t projection;
  uint32_t physical_base;
  uint64_t longest_start = 0;
  uint64_t longest_end = 0;
  uint32_t longest_band = 0;
  uint32_t longest_owner = 0;

  __aicore__ inline uint64_t Now() const {
    return ReplicaTraceCycle(recorder);
  }

  __aicore__ inline void ObserveReadyWait(uint64_t start, uint32_t band, uint32_t owner) {
    if (recorder == nullptr) return;
    const auto end = Now();
    if (longest_start == 0 || end - start > longest_end - longest_start) {
      longest_start = start;
      longest_end = end;
      longest_band = band;
      longest_owner = owner;
    }
  }

  __aicore__ inline void Record(ReplicaTraceStage stage, uint64_t start, uint64_t end,
                                uint32_t index = 0, uint32_t owner = UINT32_MAX) const {
    RecordReplicaStage(recorder, stage, projection, start, end, index, owner);
  }

  __aicore__ inline void RecordReduction(uint64_t start, uint64_t end) const {
    Record(ReplicaTraceStage::OWNER_REDUCE, start, end);
    if (longest_start != 0) {
      Record(ReplicaTraceStage::LONGEST_READY_WAIT, longest_start, longest_end,
             longest_band, physical_base + longest_owner);
    }
  }
};
}  // namespace MulticoreMoE

#endif  // HMC_MOE_REPLICA_TRACE_H
