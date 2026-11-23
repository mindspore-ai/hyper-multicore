/**
 * Copyright 2026 Huawei Technologies Co., Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef HYPER_MULTICORE_REPLICA_GRADIENT_UDMA_H
#define HYPER_MULTICORE_REPLICA_GRADIENT_UDMA_H

#include "runtime_config.hpp"
#include "replica_gradient.h"
#include "shmem.h"

namespace MulticoreRuntime {

__aicore__ inline uint32_t ReplicaGradientChunkSize(int64_t elements, int64_t offset, uint32_t chunk_elements) {
  return static_cast<uint32_t>(elements - offset < chunk_elements ? elements - offset : chunk_elements);
}

__aicore__ inline void ValidateReplicaGradientUdmaLayout(GM_ADDR scratch, uint64_t chunk_elements, uint64_t workers) {
  if (scratch == nullptr || reinterpret_cast<uint64_t>(scratch) % DATA_CACHE_LINE_SIZE != 0 ||
      chunk_elements == 0 || chunk_elements % 8 != 0 || chunk_elements > (1U << 19) ||
      workers != 4) {
    AscendC::Trap();
  }
}

// A caller must reserve every (peer, qp) for this helper alone.
// Source/slot iteration remains in the caller's existing deterministic order.
__aicore__ inline void AddReplicaGradientUdma(GM_ADDR output, GM_ADDR source, int64_t elements,
                                             int32_t peer, int32_t worker, int32_t workers,
                                             GM_ADDR scratch, uint32_t chunk_elements, uint32_t qp) {
#if !defined(__DAV_CUBE__) && defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
  ValidateReplicaGradientUdmaLayout(scratch, chunk_elements, workers);
  if (peer < 0 || peer >= aclshmem_n_pes() || elements <= 0 || worker < 0 || worker >= workers) {
    AscendC::Trap();
  }
  if (peer == aclshmem_my_pe() ||
      (aclshmemi_get_state()->topo_list[peer] & ACLSHMEM_TRANSPORT_UDMA) == 0) {
    AddReplicaGradient(output, source, elements, peer, worker, workers);
    return;
  }
  if (qp >= aclshmemi_udma_qp_info_fetch()->qp_num) {
    AscendC::Trap();
  }
  const int64_t stride = static_cast<int64_t>(workers) * chunk_elements;
  int64_t offset = static_cast<int64_t>(worker) * chunk_elements;
  if (offset >= elements) {
    return;
  }
  // Keep network banks unchanged while amortizing arithmetic fences over 64 KiB tiles.
  constexpr uint32_t tile_elements = 16384;
  AscendC::TPipe pipe;
  AscendC::TBuf<AscendC::TPosition::VECCALC> arithmetic;
  AscendC::TBuf<AscendC::TPosition::VECOUT> wqe;
  pipe.InitBuffer(arithmetic, 2 * tile_elements * sizeof(float));
  pipe.InitBuffer(wqe, ACLSHMEM_UDMA_MTE_STAGING_UB_SIZE);
  auto local = arithmetic.Get<float>();
  auto fetched = local[tile_elements];
  auto *wqe_bytes = reinterpret_cast<__ubuf__ uint8_t *>(wqe.Get<uint8_t>().GetPhyAddr());
  auto *remote = reinterpret_cast<__gm__ uint8_t *>(source);
  auto *staging = reinterpret_cast<__gm__ float *>(scratch) +
                  static_cast<int64_t>(worker) * 2 * chunk_elements;
  const uint32_t initial = ReplicaGradientChunkSize(elements, offset, chunk_elements);
  aclshmemx_udma_qp_get_nbi<uint8_t>(reinterpret_cast<__gm__ uint8_t *>(staging),
      remote + offset * sizeof(float), wqe_bytes, initial * sizeof(float), peer, qp, EVENT_ID0);
  uint32_t bank = 0;
  while (offset < elements) {
    aclshmemx_udma_qp_quiet(peer, qp);
    const int64_t next = offset + stride;
    const uint32_t count = ReplicaGradientChunkSize(elements, offset, chunk_elements);
    if (next < elements) {
      const uint32_t next_count = ReplicaGradientChunkSize(elements, next, chunk_elements);
      aclshmemx_udma_qp_get_nbi<uint8_t>(
          reinterpret_cast<__gm__ uint8_t *>(staging + (1 - bank) * chunk_elements),
          remote + next * sizeof(float), wqe_bytes, next_count * sizeof(float), peer, qp, EVENT_ID0);
    }
    AscendC::GlobalTensor<float> home, guest;
    home.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(output) + offset, count);
    guest.SetGlobalBuffer(staging + bank * chunk_elements, count);
    for (uint32_t tile = 0; tile < count; tile += tile_elements) {
      const uint32_t size = count - tile < tile_elements ? count - tile : tile_elements;
      AscendC::DataCopyExtParams params{1, size * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
      AscendC::DataCopyPadExtParams<float> padding{false, 0, 0, 0};
      AscendC::DataCopyPad(local, home[tile], params, padding);
      AscendC::DataCopyPad(fetched, guest[tile], params, padding);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
      AscendC::Add(local, local, fetched, size);
      AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
      AscendC::SetAtomicNone();
      AscendC::DataCopyPad(home[tile], local, params);
      AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
      AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
    bank = 1 - bank;
    offset = next;
  }
  pipe.Destroy();
#else
  AddReplicaGradient(output, source, elements, peer, worker, workers);
#endif
}

}  // namespace MulticoreRuntime
#endif  // HYPER_MULTICORE_REPLICA_GRADIENT_UDMA_H
