// Copyright 2026 Huawei Technologies Co., Ltd. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#ifndef HMC_MOE_REPLICA_GRADIENT_BANDS_H
#define HMC_MOE_REPLICA_GRADIENT_BANDS_H

#include "common/mega_moe/replica_weight_ready.h"
#include "runtime/gradient_band_producer.h"
#include "runtime/replica_gradient.h"
#include "shmem.h"

namespace MulticoreMoE {
struct ReplicaGradientBandRecord {
  uint64_t home, slots, epoch, rows, columns, band_rows, bands;
  uint64_t owner, guest, owner_flags, guest_flags, owned;
  uint64_t scratch, rank, ack, done;
};
struct ReplicaGradientBandEdge {
  uint64_t peer, slot, owner, last;
};
static_assert(sizeof(ReplicaGradientBandRecord) == 128);
static_assert(sizeof(ReplicaGradientBandEdge) == 32);
constexpr uint64_t kFp32ReplicaGradientTable = 2;
constexpr uint64_t kBf16ReplicaGradientTable = 3;

__aicore__ inline bool UsesBf16ReplicaPartials(const __gm__ ReplicaLaunch *launch) {
  return launch != nullptr && launch->gradient != 0 &&
      *reinterpret_cast<const __gm__ uint64_t *>(launch->gradient) == kBf16ReplicaGradientTable;
}

__aicore__ inline uint64_t GradientBandOwnerScratch(const __gm__ ReplicaGradientBandRecord *record,
                                                     uint32_t expert) {
  const auto edges = reinterpret_cast<const __gm__ ReplicaGradientBandEdge *>(record + 1);
  const auto mapping = reinterpret_cast<const __gm__ uint64_t *>(edges + record->owned);
  return mapping[expert];
}

__aicore__ inline const __gm__ uint64_t *GradientBandGuestOwners(
    const __gm__ ReplicaGradientBandRecord *record) {
  const auto edges = reinterpret_cast<const __gm__ ReplicaGradientBandEdge *>(record + 1);
  return reinterpret_cast<const __gm__ uint64_t *>(edges + record->owned) + record->home;
}

__aicore__ inline __gm__ ReplicaGradientBandRecord *GetReplicaGradientBands(
    const __gm__ ReplicaLaunch *launch, uint32_t projection) {
  if (launch == nullptr || launch->gradient == 0) return nullptr;
  auto table = reinterpret_cast<__gm__ uint64_t *>(launch->gradient);
  if ((table[0] != kFp32ReplicaGradientTable && table[0] != kBf16ReplicaGradientTable) ||
      table[1] != 3 || table[2] < 19) AscendC::Trap();
  return reinterpret_cast<__gm__ ReplicaGradientBandRecord *>(table + table[projection + 1]);
}

__aicore__ inline uint64_t GradientBandExpertFlags(const __gm__ ReplicaGradientBandRecord *record,
                                                   uint32_t expert, uint32_t cubes) {
  if (expert >= record->home + record->slots || record->bands == 0 || record->bands > 64 ||
      record->band_rows == 0 || record->epoch == 0 || record->epoch > 0x7fffffff) AscendC::Trap();
  const uint64_t stride = record->bands * cubes * MulticoreRuntime::DATA_CACHE_LINE_SIZE;
  return expert < record->home ? record->owner_flags + expert * stride
                              : record->guest_flags + (expert - record->home) * stride;
}

#if !defined(__DAV_CUBE__) && !defined(__DAV_C220_CUBE__)
__aicore__ inline void ValidateReplicaBandGeometry(const __gm__ ReplicaGradientBandRecord *record, uint32_t ep) {
  if (record->band_rows == 0 || record->rows == 0 || record->columns == 0 || record->bands > 64 ||
      record->bands != (record->rows + record->band_rows - 1) / record->band_rows ||
      record->owned > ep * record->slots) AscendC::Trap();
}

__aicore__ inline void FinishReplicaBandReads(const __gm__ ReplicaGradientBandRecord *record,
                                             const __gm__ ReplicaGradientBandEdge *edges,
                                             uint32_t worker, uint32_t cubes, const ReplicaGradientTrace &trace) {
  using MulticoreRuntime::DATA_CACHE_LINE_SIZE;
  using MulticoreRuntime::StoreReplicaCompletion;
  // Advertise reuse only after every local consumer has completed its stores.
  StoreReplicaCompletion(reinterpret_cast<GM_ADDR>(record->done + worker * DATA_CACHE_LINE_SIZE), record->epoch);
  if (worker != 0) return;
  const auto drain_start = trace.Now();
  for (uint32_t lane = 0; lane < cubes; ++lane) {
    WaitReplicaEpoch(record->done + lane * DATA_CACHE_LINE_SIZE, record->epoch);
  }
  const auto drained = trace.Now();
  for (uint32_t index = 0; index < record->owned; ++index) {
    const uint64_t line = record->rank * record->home + edges[index].slot;
    aclshmemx_signal_op(reinterpret_cast<__gm__ int32_t *>(record->ack + line * DATA_CACHE_LINE_SIZE),
                       record->epoch, ACLSHMEM_SIGNAL_SET, edges[index].peer);
  }
  const auto published = trace.Now();
  trace.Record(ReplicaTraceStage::OWNER_DRAIN, drain_start, drained);
  trace.Record(ReplicaTraceStage::ACK_PUBLISH, drained, published);
}
#endif


__aicore__ inline void ReturnReplicaGradientBands(const __gm__ ReplicaGradientBandRecord *record,
                                                 GM_ADDR groups_address, uint32_t worker, uint32_t cubes, uint32_t ep,
                                                 ReplicaGradientTrace &trace) {
#if !defined(__DAV_CUBE__) && !defined(__DAV_C220_CUBE__)
  using MulticoreRuntime::AddReplicaGradient;
  using MulticoreRuntime::DATA_CACHE_LINE_SIZE;
  if (record == nullptr || record->owned == 0) return;
  ValidateReplicaBandGeometry(record, ep);
  const auto edges = reinterpret_cast<const __gm__ ReplicaGradientBandEdge *>(record + 1);
  const auto expert_groups = reinterpret_cast<const __gm__ int64_t *>(groups_address);
  const uint32_t groups = record->bands < 4 ? record->bands : 4;
  const uint32_t group = worker % groups, lane = worker / groups;
  const uint32_t lanes = (cubes - 1 - group) / groups + 1;
  const uint64_t matrix_elements = record->rows * record->columns;
  const auto reduce_start = trace.Now();
  for (uint32_t band = group; band < record->bands; band += groups) {
    const uint64_t offset = band * record->band_rows * record->columns;
    const uint64_t length = record->band_rows * record->columns;
    const uint64_t elements = matrix_elements - offset < length ? matrix_elements - offset : length;
    for (uint32_t i = 0; i < record->owned; ++i) {
      const auto peer = edges[i].peer, slot = edges[i].slot, owner = edges[i].owner;
      if (peer >= ep || slot >= record->slots || owner >= record->home) AscendC::Trap();
      const int64_t owner_start = owner == 0 ? 0 : expert_groups[owner - 1];
      // A fully moved expert has a zeroed owner output and may have no scheduled TN task.
      const bool owner_produces = expert_groups[owner] > owner_start;
      const auto local = GradientBandExpertFlags(record, owner, cubes);
      const auto guest = GradientBandExpertFlags(record, record->home + slot, cubes);
      auto mapped = hyper_multicore::shmem::data_plane::remote_ptr(reinterpret_cast<__gm__ void *>(guest), peer);
      if (mapped == nullptr) AscendC::Trap();
      const auto wait_start = trace.Now();
      for (uint32_t cube = 0; cube < cubes; ++cube) {
        const uint64_t signal = (band * cubes + cube) * DATA_CACHE_LINE_SIZE;
        if (owner_produces) WaitReplicaEpoch<true>(local + signal, record->epoch);
        WaitReplicaEpoch<true>(reinterpret_cast<uint64_t>(mapped) + signal, record->epoch);
      }
      trace.ObserveReadyWait(wait_start, band, owner);
      const uint64_t scratch = GradientBandOwnerScratch(record, owner);
      if (scratch == 0 || record->scratch == 0) AscendC::Trap();
      auto dst = reinterpret_cast<GM_ADDR>(
          record->scratch + ((scratch - 1) * matrix_elements + offset) * sizeof(float));
      auto src = reinterpret_cast<GM_ADDR>(record->guest + (slot * matrix_elements + offset) * sizeof(float));
      if (edges[i].last != 0) {
        auto final = reinterpret_cast<GM_ADDR>(record->owner + (owner * matrix_elements + offset) * sizeof(bfloat16_t));
        AddReplicaGradient<true>(dst, src, elements, peer, lane, lanes, final);
      } else {
        AddReplicaGradient(dst, src, elements, peer, lane, lanes);
      }
    }
  }
  const auto reduce_end = trace.Now();
  FinishReplicaBandReads(record, edges, worker, cubes, trace);
  trace.RecordReduction(reduce_start, reduce_end);
#endif
}

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201 && !defined(__DAV_CUBE__) && !defined(__DAV_C220_CUBE__)
__aicore__ inline uint32_t WaitForBf16BandPartials(const __gm__ ReplicaGradientBandRecord *record,
                                                 const __gm__ ReplicaGradientBandEdge *edges, uint32_t first,
                                                 uint64_t owner, bool owner_produces, uint64_t signal,
                                                 uint32_t cubes, uint32_t ep) {
  using MulticoreRuntime::DATA_CACHE_LINE_SIZE;
  if (owner_produces) {
    const auto flags = GradientBandExpertFlags(record, owner, cubes);
    for (uint32_t cube = 0; cube < cubes; ++cube) {
      WaitReplicaEpoch<true>(flags + signal + cube * DATA_CACHE_LINE_SIZE, record->epoch);
    }
  }
  uint32_t last = first;
  do {
    if (last >= record->owned || edges[last].owner != owner ||
        edges[last].peer >= ep || edges[last].slot >= record->slots) AscendC::Trap();
    const auto flags = GradientBandExpertFlags(record, record->home + edges[last].slot, cubes);
    const auto remote = hyper_multicore::shmem::data_plane::remote_ptr(
        reinterpret_cast<__gm__ void *>(flags), edges[last].peer);
    if (remote == nullptr) AscendC::Trap();
    for (uint32_t cube = 0; cube < cubes; ++cube) {
      WaitReplicaEpoch<true>(reinterpret_cast<uint64_t>(remote) + signal + cube * DATA_CACHE_LINE_SIZE, record->epoch);
    }
  } while (edges[last++].last == 0);
  return last;
}
#endif

__aicore__ inline void ReturnBf16ReplicaGradientBands(const __gm__ ReplicaGradientBandRecord *record,
                                                     GM_ADDR groups_address, uint32_t worker,
                                                     uint32_t cubes, uint32_t ep, ReplicaGradientTrace &trace) {
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201 && !defined(__DAV_CUBE__) && !defined(__DAV_C220_CUBE__)
  using MulticoreRuntime::DATA_CACHE_LINE_SIZE;
  if (record == nullptr || record->owned == 0) return;
  ValidateReplicaBandGeometry(record, ep);
  if (record->scratch != 0) AscendC::Trap();
  const auto edges = reinterpret_cast<const __gm__ ReplicaGradientBandEdge *>(record + 1);
  const auto expert_groups = reinterpret_cast<const __gm__ int64_t *>(groups_address);
  const uint32_t groups = record->bands < 4 ? record->bands : 4;
  const uint32_t group = worker % groups, lane = worker / groups;
  const uint32_t lanes = (cubes - 1 - group) / groups + 1;
  const uint64_t matrix = record->rows * record->columns;
  constexpr uint32_t tile = 4096;
  AscendC::TPipe pipe;
  AscendC::TBuf<AscendC::TPosition::VECCALC> buffer;
  pipe.InitBuffer(buffer, tile * (2 * sizeof(float) + 2 * sizeof(bfloat16_t)));
  auto accum = buffer.Get<float>();
  auto partial = accum[tile];
  auto native_home = accum[2 * tile].ReinterpretCast<bfloat16_t>();
  auto native_guest = native_home[tile];
  const auto owner_base = reinterpret_cast<__gm__ bfloat16_t *>(record->owner);
  const auto guest_base = reinterpret_cast<__gm__ bfloat16_t *>(record->guest);
  AscendC::DataCopyPadExtParams<bfloat16_t> padding{false, 0, 0, 0};

  const auto reduce_start = trace.Now();
  for (uint32_t band = group; band < record->bands; band += groups) {
    const uint64_t start = band * record->band_rows * record->columns;
    const uint64_t length = record->band_rows * record->columns;
    const uint64_t end = start + length < matrix ? start + length : matrix;
    const uint64_t signal = band * cubes * DATA_CACHE_LINE_SIZE;
    // The host groups edges by owner and retains the original peer/slot order
    // inside each group. Keep the FP32 sum in UB through all its contributors.
    for (uint32_t first = 0; first < record->owned;) {
      const uint64_t owner = edges[first].owner;
      if (owner >= record->home) AscendC::Trap();
      const int64_t owner_start = owner == 0 ? 0 : expert_groups[owner - 1];
      const bool owner_produces = expert_groups[owner] > owner_start;
      const auto wait_start = trace.Now();
      const uint32_t last = WaitForBf16BandPartials(record, edges, first, owner, owner_produces, signal, cubes, ep);
      trace.ObserveReadyWait(wait_start, band, owner);

      for (uint64_t offset = start + lane * tile; offset < end; offset += lanes * tile) {
        const uint32_t count = end - offset < tile ? end - offset : tile;
        AscendC::DataCopyExtParams transfer{1, count * static_cast<uint32_t>(sizeof(bfloat16_t)), 0, 0, 0};
        AscendC::GlobalTensor<bfloat16_t> home;
        home.SetGlobalBuffer(owner_base + owner * matrix + offset, count);
        if (owner_produces) AscendC::DataCopyPad(native_home, home, transfer, padding);
        for (uint32_t edge = first; edge < last; ++edge) {
          auto remote = hyper_multicore::shmem::data_plane::remote_ptr(
              reinterpret_cast<__gm__ void *>(guest_base + edges[edge].slot * matrix + offset), edges[edge].peer);
          if (remote == nullptr) AscendC::Trap();
          AscendC::GlobalTensor<bfloat16_t> guest;
          guest.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(remote), count);
          AscendC::DataCopyPad(native_guest, guest, transfer, padding);
          AscendC::PipeBarrier<PIPE_ALL>();
          if (edge == first) {
            if (owner_produces) AscendC::Cast(accum, native_home, AscendC::RoundMode::CAST_NONE, count);
            else AscendC::Duplicate(accum, 0.0f, count);
          }
          AscendC::Cast(partial, native_guest, AscendC::RoundMode::CAST_NONE, count);
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::Add(accum, accum, partial, count);
          AscendC::PipeBarrier<PIPE_ALL>();
        }
        AscendC::Cast(native_home, accum, AscendC::RoundMode::CAST_RINT, count);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::SetAtomicNone();
        AscendC::DataCopyPad(home, native_home, transfer);
        AscendC::PipeBarrier<PIPE_ALL>();
      }
      first = last;
    }
  }
  pipe.Destroy();
  const auto reduce_end = trace.Now();
  FinishReplicaBandReads(record, edges, worker, cubes, trace);
  trace.RecordReduction(reduce_start, reduce_end);
#endif
}

__aicore__ inline void WaitReplicaGradientAcks(const __gm__ ReplicaGradientBandRecord *record,
                                             const ReplicaGradientTrace &trace) {
#if !defined(__DAV_CUBE__) && !defined(__DAV_C220_CUBE__)
  using MulticoreRuntime::DATA_CACHE_LINE_SIZE;
  const auto owners = GradientBandGuestOwners(record);
  for (uint32_t slot = 0; slot < record->slots; ++slot) {
    if (owners[slot] != UINT64_MAX) {
      const uint64_t line = owners[slot] * record->home + slot;
      const auto start = trace.Now();
      WaitReplicaEpoch(record->ack + line * DATA_CACHE_LINE_SIZE, record->epoch);
      trace.Record(ReplicaTraceStage::GUEST_ACK_WAIT, start, trace.Now(), slot,
                   trace.physical_base + record->home + slot);
    }
  }
#endif
}

}  // namespace MulticoreMoE

#endif  // HMC_MOE_REPLICA_GRADIENT_BANDS_H
