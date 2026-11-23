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
__aicore__ inline void FinishReplicaBandReads(const __gm__ ReplicaGradientBandRecord *record,
                                             const __gm__ ReplicaGradientBandEdge *edges,
                                             uint32_t worker, uint32_t cubes) {
  using namespace MulticoreRuntime;
  // Advertise reuse only after every local consumer has completed its stores.
  StoreReplicaCompletion(reinterpret_cast<GM_ADDR>(record->done + worker * DATA_CACHE_LINE_SIZE), record->epoch);
  if (worker != 0) return;
  for (uint32_t lane = 0; lane < cubes; ++lane) {
    WaitReplicaEpoch(record->done + lane * DATA_CACHE_LINE_SIZE, record->epoch);
  }
  for (uint32_t index = 0; index < record->owned; ++index) {
    const uint64_t line = record->rank * record->home + edges[index].slot;
    aclshmemx_signal_op(reinterpret_cast<__gm__ int32_t *>(record->ack + line * DATA_CACHE_LINE_SIZE),
                       record->epoch, ACLSHMEM_SIGNAL_SET, edges[index].peer);
  }
}
#endif


__aicore__ inline void ReturnReplicaGradientBands(const __gm__ ReplicaGradientBandRecord *record,
                                                 GM_ADDR groups_address, uint32_t worker, uint32_t cubes, uint32_t ep) {
#if !defined(__DAV_CUBE__) && !defined(__DAV_C220_CUBE__)
  using namespace MulticoreRuntime;
  if (record == nullptr || record->owned == 0) return;
  if (record->band_rows == 0 || record->rows == 0 || record->columns == 0 || record->bands > 64 ||
      record->bands != (record->rows + record->band_rows - 1) / record->band_rows ||
      record->owned > ep * record->slots) AscendC::Trap();
  const auto edges = reinterpret_cast<const __gm__ ReplicaGradientBandEdge *>(record + 1);
  const auto expert_groups = reinterpret_cast<const __gm__ int64_t *>(groups_address);
  const uint32_t groups = record->bands < 4 ? record->bands : 4;
  const uint32_t group = worker % groups, lane = worker / groups;
  const uint32_t lanes = (cubes - 1 - group) / groups + 1;
  const uint64_t matrix_elements = record->rows * record->columns;
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
      for (uint32_t cube = 0; cube < cubes; ++cube) {
        const uint64_t signal = (band * cubes + cube) * DATA_CACHE_LINE_SIZE;
        if (owner_produces) WaitReplicaEpoch<true>(local + signal, record->epoch);
        WaitReplicaEpoch<true>(reinterpret_cast<uint64_t>(mapped) + signal, record->epoch);
      }
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
  FinishReplicaBandReads(record, edges, worker, cubes);
#endif
}

__aicore__ inline void ReturnBf16ReplicaGradientBands(const __gm__ ReplicaGradientBandRecord *record,
                                                     GM_ADDR groups_address, uint32_t worker,
                                                     uint32_t cubes, uint32_t ep) {
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201 && !defined(__DAV_CUBE__) && !defined(__DAV_C220_CUBE__)
  using namespace AscendC;
  using namespace MulticoreRuntime;
  if (record == nullptr || record->owned == 0) return;
  if (record->band_rows == 0 || record->rows == 0 || record->columns == 0 || record->bands > 64 ||
      record->bands != (record->rows + record->band_rows - 1) / record->band_rows ||
      record->owned > ep * record->slots || record->scratch != 0) AscendC::Trap();
  const auto edges = reinterpret_cast<const __gm__ ReplicaGradientBandEdge *>(record + 1);
  const auto expert_groups = reinterpret_cast<const __gm__ int64_t *>(groups_address);
  const uint32_t groups = record->bands < 4 ? record->bands : 4;
  const uint32_t group = worker % groups, lane = worker / groups;
  const uint32_t lanes = (cubes - 1 - group) / groups + 1;
  const uint64_t matrix = record->rows * record->columns;
  constexpr uint32_t tile = 4096;
  TPipe pipe;
  TBuf<TPosition::VECCALC> buffer;
  pipe.InitBuffer(buffer, tile * (2 * sizeof(float) + 2 * sizeof(bfloat16_t)));
  auto accum = buffer.Get<float>();
  auto partial = accum[tile];
  auto native_home = accum[2 * tile].ReinterpretCast<bfloat16_t>();
  auto native_guest = native_home[tile];
  const auto owner_base = reinterpret_cast<__gm__ bfloat16_t *>(record->owner);
  const auto guest_base = reinterpret_cast<__gm__ bfloat16_t *>(record->guest);
  DataCopyPadExtParams<bfloat16_t> padding{false, 0, 0, 0};

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
          WaitReplicaEpoch<true>(reinterpret_cast<uint64_t>(remote) + signal + cube * DATA_CACHE_LINE_SIZE,
                                 record->epoch);
        }
      } while (edges[last++].last == 0);

      for (uint64_t offset = start + lane * tile; offset < end; offset += lanes * tile) {
        const uint32_t count = end - offset < tile ? end - offset : tile;
        DataCopyExtParams transfer{1, count * static_cast<uint32_t>(sizeof(bfloat16_t)), 0, 0, 0};
        GlobalTensor<bfloat16_t> home;
        home.SetGlobalBuffer(owner_base + owner * matrix + offset, count);
        if (owner_produces) DataCopyPad(native_home, home, transfer, padding);
        for (uint32_t edge = first; edge < last; ++edge) {
          auto remote = hyper_multicore::shmem::data_plane::remote_ptr(
              reinterpret_cast<__gm__ void *>(guest_base + edges[edge].slot * matrix + offset), edges[edge].peer);
          if (remote == nullptr) AscendC::Trap();
          GlobalTensor<bfloat16_t> guest;
          guest.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(remote), count);
          DataCopyPad(native_guest, guest, transfer, padding);
          PipeBarrier<PIPE_ALL>();
          if (edge == first) {
            if (owner_produces) Cast(accum, native_home, RoundMode::CAST_NONE, count);
            else Duplicate(accum, 0.0f, count);
          }
          Cast(partial, native_guest, RoundMode::CAST_NONE, count);
          PipeBarrier<PIPE_V>();
          Add(accum, accum, partial, count);
          PipeBarrier<PIPE_ALL>();
        }
        Cast(native_home, accum, RoundMode::CAST_RINT, count);
        PipeBarrier<PIPE_ALL>();
        SetAtomicNone();
        DataCopyPad(home, native_home, transfer);
        PipeBarrier<PIPE_ALL>();
      }
      first = last;
    }
  }
  pipe.Destroy();
  FinishReplicaBandReads(record, edges, worker, cubes);
#endif
}

__aicore__ inline void WaitReplicaGradientAcks(const __gm__ ReplicaGradientBandRecord *record) {
#if !defined(__DAV_CUBE__) && !defined(__DAV_C220_CUBE__)
  using namespace MulticoreRuntime;
  const auto owners = GradientBandGuestOwners(record);
  for (uint32_t slot = 0; slot < record->slots; ++slot) {
    if (owners[slot] != UINT64_MAX) {
      const uint64_t line = owners[slot] * record->home + slot;
      WaitReplicaEpoch(record->ack + line * DATA_CACHE_LINE_SIZE, record->epoch);
    }
  }
#endif
}

}  // namespace MulticoreMoE

#endif  // HMC_MOE_REPLICA_GRADIENT_BANDS_H
