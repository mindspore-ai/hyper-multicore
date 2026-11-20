// Copyright 2026 Huawei Technologies Co., Ltd. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "common/mega_moe/replica_weight_ready.h"
#include "data_plane/rma.h"

namespace MulticoreMoE {
struct A5ReplicaPrefetch {
  uint64_t slots, backward, rank, ranks, credit, elements[2], edges;
};
struct A5ReplicaWeightEdge {
  uint64_t peer, slot, expert;
};
static_assert(sizeof(A5ReplicaPrefetch) == 64);
static_assert(sizeof(A5ReplicaWeightEdge) == 24);

__aicore__ inline void PutA5ReplicaWeightMte(GM_ADDR destination, GM_ADDR source, uint64_t bytes, int32_t peer) {
  namespace shmem = hyper_multicore::shmem;
  uint64_t ub_address, ub_bytes;
  if (!shmem::cann::device::query_mte_resources(peer, ub_address, ub_bytes) || ub_bytes < 32 ||
      ub_bytes > UINT32_MAX) AscendC::Trap();
  auto *ub = reinterpret_cast<__ubuf__ uint8_t *>(ub_address);
  // The generic SDK Put can select UDMA QP0 even in an explicit MTE schedule.
  // Replica helpers must not share the token workers' UDMA queues.
  constexpr uint32_t max_bytes = 1U << 28;
  for (uint64_t offset = 0; offset < bytes; offset += max_bytes) {
    const uint32_t chunk = bytes - offset < max_bytes ? static_cast<uint32_t>(bytes - offset) : max_bytes;
    shmem::cann::device::mte_put_nbi(destination + offset, source + offset, ub,
                                    static_cast<uint32_t>(ub_bytes), chunk, peer, EVENT_ID0);
    shmem::cann::device::mte_quiet();
  }
}

__aicore__ inline void PrefetchA5ReplicaWeights(const __gm__ ReplicaLaunch *launch, uint32_t worker) {
  if (launch == nullptr || worker % 2 != 0 || worker / 2 >= kA5ReplicaHelpers) return;
  const auto addr = reinterpret_cast<const __gm__ A5ReplicaLaunch *>(launch)->prefetch;
  if (addr == 0) return;
  const auto record = reinterpret_cast<const __gm__ A5ReplicaPrefetch *>(addr);
  if (record->rank >= record->ranks || record->slots == 0 || record->slots > launch->home ||
      record->edges > record->ranks * record->slots || launch->epoch == 0 ||
      launch->epoch > 0x7fffffff || record->backward > 1) AscendC::Trap();
  namespace plane = hyper_multicore::shmem::data_plane;
  constexpr auto line = MulticoreRuntime::DATA_CACHE_LINE_SIZE;
  const uint32_t part = worker / 2;
  const auto incoming = reinterpret_cast<const __gm__ uint64_t *>(record + 1);
  const auto edges = reinterpret_cast<const __gm__ A5ReplicaWeightEdge *>(incoming + record->slots);
  if (part == 0) {
    // Credit every incoming edge before waiting for any outgoing edge.
    for (uint32_t slot = 0; slot < record->slots; ++slot) {
      const auto peer = incoming[slot];
      if (peer == UINT64_MAX) continue;
      if (peer >= record->ranks) AscendC::Trap();
      const auto credit = record->credit + (record->rank * launch->home + slot) * line;
      plane::signal(reinterpret_cast<__gm__ int32_t *>(credit), launch->epoch, plane::SignalOp::Set, peer);
    }
  }
  for (uint32_t index = 0; index < 2; ++index) {
    const uint32_t projection = record->backward ? 1 - index : index;
    const uint64_t elements = record->elements[projection];
    // Cut at 32-byte boundaries so different lanes never share a transfer block.
    if (elements == 0 || elements % 16 != 0) AscendC::Trap();
    const uint64_t first = (elements / 16 * part / kA5ReplicaHelpers) * 16;
    const uint64_t end = (elements / 16 * (part + 1) / kA5ReplicaHelpers) * 16;
    for (uint32_t i = 0; i < record->edges; ++i) {
      const auto peer = edges[i].peer, slot = edges[i].slot, expert = edges[i].expert;
      if (peer >= record->ranks || slot >= record->slots || expert >= launch->home) AscendC::Trap();
      const auto credit = record->credit + (peer * launch->home + slot) * line;
      WaitReplicaEpoch<true>(credit, launch->epoch);
      auto dst = reinterpret_cast<GM_ADDR>(launch->guest[projection] + (slot * elements + first) * 2);
      auto src = reinterpret_cast<GM_ADDR>(launch->local[projection] + (expert * elements + first) * 2);
      PutA5ReplicaWeightMte(dst, src, (end - first) * sizeof(bfloat16_t), static_cast<int32_t>(peer));
      const auto ready = launch->ready[projection] + (slot * kA5ReplicaHelpers + part) * line;
      plane::signal(reinterpret_cast<__gm__ int32_t *>(ready), launch->epoch, plane::SignalOp::Set, peer);
    }
  }
}
}  // namespace MulticoreMoE
