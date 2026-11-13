/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cstdint>

#include "cann/device.h"

namespace hyper_multicore::shmem::data_plane::mte_rma {

/**
 * @brief Issue an explicit MTE NBI Put using caller-owned UB and Event resources.
 *
 * @param remote_dst Symmetric GM destination represented by its local address.
 * @param local_src Local GM source owned by the caller.
 * @param ub_scratch Caller-owned temporary UB address.
 * @param ub_scratch_bytes UB capacity in Byte.
 * @param bytes Transfer length in Byte.
 * @param target_pe Destination in CANN Root WORLD coordinates.
 * @param event_id Caller-owned MTE synchronization Event ID.
 * @pre The consumer validates arguments, MTE reachability and resource reuse dependencies.
 * @note Return means issued, not completed. The consumer must eventually call quiet().
 */
__aicore__ inline void put_nbi(__gm__ uint8_t *remote_dst, __gm__ const uint8_t *local_src,
                               __ubuf__ uint8_t *ub_scratch, uint32_t ub_scratch_bytes, uint32_t bytes,
                               int32_t target_pe, uint32_t event_id) {
  cann::device::mte_put_nbi(remote_dst, local_src, ub_scratch, ub_scratch_bytes, bytes, target_pe, event_id);
}

/**
 * @brief Copy local GM to a symmetric destination using the CANN-owned double buffer.
 * @pre The calling AIV exclusively owns the CANN UB range and EVENT_ID0/1.
 * @note Zero elements is a no-op. Call quiet() before publishing completion to a peer.
 */
template <typename T>
__aicore__ inline void put_pipelined(GM_ADDR destination, GM_ADDR source, int64_t elements, int32_t target_pe) {
  if (elements == 0) {
    return;
  }
  uint64_t copy_ub;
  uint64_t copy_ub_size;
  if (!cann::device::query_mte_resources(target_pe, copy_ub, copy_ub_size)) {
    AscendC::Trap();
  }
  __ubuf__ T *ping_buff = reinterpret_cast<__ubuf__ T *>(copy_ub);
  __ubuf__ T *pong_buff = reinterpret_cast<__ubuf__ T *>(copy_ub + copy_ub_size / 2);
  auto ptr = cann::device::remote_ptr(reinterpret_cast<__gm__ T *>(destination), target_pe);
  __gm__ T *remote_ptr = reinterpret_cast<__gm__ T *>(ptr);
  uint64_t block_size = copy_ub_size / 2 / sizeof(T) * sizeof(T);
  uint64_t remain = (elements * sizeof(T)) % block_size;

  uint64_t repeat_times = (elements * sizeof(T)) / block_size;
  uint64_t repeat_elem = block_size / sizeof(T);
  AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
  AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
  for (uint64_t i = 0; i < repeat_times; i++) {
    AscendC::TEventID EVENT_ID = (i & 1) ? EVENT_ID0 : EVENT_ID1;
    __ubuf__ T *buf = (i & 1) ? ping_buff : pong_buff;
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
    cann::device::copy_gm_to_ub(buf, reinterpret_cast<__gm__ T *>(source) + i * repeat_elem, block_size);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
    cann::device::copy_ub_to_gm(remote_ptr + i * repeat_elem, buf, block_size);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
  }
  if (remain > 0) {
    AscendC::TEventID EVENT_ID = (repeat_times & 1) ? EVENT_ID0 : EVENT_ID1;
    __ubuf__ T *buf = (repeat_times & 1) ? ping_buff : pong_buff;
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
    cann::device::copy_gm_to_ub(buf, reinterpret_cast<__gm__ T *>(source) + repeat_times * repeat_elem, remain);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
    cann::device::copy_ub_to_gm(remote_ptr + repeat_times * repeat_elem, buf, remain);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
  }
  AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
  AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
}

/** @brief Complete explicit MTE NBI operations previously issued by the calling AIV. */
__aicore__ inline void quiet() { cann::device::mte_quiet(); }

}  // namespace hyper_multicore::shmem::data_plane::mte_rma
