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
#ifndef HYPER_MULTICORE_MULTICORE_GET_MEM_H
#define HYPER_MULTICORE_MULTICORE_GET_MEM_H

#include "kernel_operator.h"
#include "data_plane/rma.h"
#include "shmem.h"
#include "transport_credits.h"

template <typename T, bool UseUdma = false>
__aicore__ inline void PullToLocal(GM_ADDR destination, GM_ADDR source, int64_t elements, int source_pe) {
  if (elements == 0) {
    return;
  }
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
  if constexpr (UseUdma) {
    if (source_pe != aclshmem_my_pe()) {
      const uint32_t worker = AscendC::GetBlockIdx();
      if (worker % 2 == 0 || worker / 2 >= 32) {
        AscendC::Trap();
      }
      const uint32_t qp_count = aclshmemi_udma_qp_info_fetch()->qp_num;
      if (qp_count != 1 && qp_count != 8 && qp_count != 16 && qp_count != 32) AscendC::Trap();
      const uint32_t qp = MulticoreRuntime::UdmaQpForWorker(worker, qp_count);
      AscendC::TPipe pipe;
      AscendC::TBuf<AscendC::TPosition::VECOUT> staging;
      pipe.InitBuffer(staging, ACLSHMEM_UDMA_MTE_STAGING_UB_SIZE);
      auto scratch = reinterpret_cast<__ubuf__ uint8_t *>(staging.Get<uint8_t>().GetPhyAddr());
      auto local = reinterpret_cast<__gm__ uint8_t *>(destination);
      auto remote = reinterpret_cast<__gm__ uint8_t *>(source);
      uint64_t remaining = elements * sizeof(T);
      while (remaining > 0) {
        const uint32_t bytes = remaining < (1ULL << 28) ? remaining : (1ULL << 28);
        aclshmemx_udma_qp_get_nbi<uint8_t>(local, remote, scratch, bytes, source_pe, qp, 0);
        // Local expert credit is published by the worker only after all reads complete.
        aclshmemx_udma_qp_quiet(source_pe, qp);
        local += bytes;
        remote += bytes;
        remaining -= bytes;
      }
      return;
    }
    // Self traffic does not need the remote-read burst cap used by legacy callers.
    hyper_multicore::shmem::data_plane::get_pipelined<T>(destination, source, elements, source_pe, 0);
    return;
  }
#endif
  hyper_multicore::shmem::data_plane::get_pipelined<T>(destination, source, elements, source_pe);
}

#endif  // HYPER_MULTICORE_MULTICORE_GET_MEM_H
