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
#include "data_plane/sync.h"

namespace hyper_multicore::shmem::data_plane {

/** @brief Return the calling PE's Root rank. */
ACLSHMEM_DEVICE int32_t my_pe() { return cann::device::my_pe(); }

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
/** @brief Return the initialized number of UDMA QPs per peer. */
ACLSHMEM_DEVICE uint32_t udma_qp_count() { return cann::device::udma_qp_count(); }

/**
 * @brief Complete a GET on the caller-selected, exclusively owned QP.
 * @note Return makes the local data available before the worker publishes completion credits.
 */
template <typename T>
__aicore__ inline void get_udma(GM_ADDR destination, GM_ADDR source, int64_t elements, int32_t source_pe, uint32_t qp) {
  if (elements != 0) {
    cann::device::get_udma<T>(destination, source, elements, source_pe, qp);
  }
}
#endif

/**
 * @brief Translate one local symmetric GM address to the corresponding address on a target Root PE.
 *
 * @param local_symmetric_address Symmetric GM address in the calling PE's address space.
 * @param target_pe Destination in CANN Root WORLD coordinates.
 * @return Corresponding symmetric GM address on target_pe.
 */
ACLSHMEM_DEVICE __gm__ void *remote_ptr(__gm__ void *local_symmetric_address, int32_t target_pe) {
  return cann::device::remote_ptr(local_symmetric_address, target_pe);
}

/**
 * @brief Synchronize-copy local GM bytes to a symmetric address on one target Root PE.
 *
 * @param remote_dst Symmetric GM destination represented by its local address.
 * @param local_src Local GM source owned by the caller.
 * @param bytes Transfer length in Byte; zero is a no-op.
 * @param target_pe Destination in CANN Root WORLD coordinates.
 * @pre The consumer validates addresses, length, target PE and MTE reachability.
 * @note One nonzero call maps to one CANN call. Return means the data is remote-visible.
 */
__aicore__ inline void put(__gm__ void *remote_dst, __gm__ const void *local_src, uint32_t bytes, int32_t target_pe) {
  if (bytes != 0) {
    cann::device::put(remote_dst, local_src, bytes, target_pe);
  }
}

/**
 * @brief Synchronize-copy bytes from a symmetric address on one Root PE to local GM.
 *
 * @param local_dst Local GM destination owned by the caller.
 * @param remote_src Symmetric GM source represented by its local address.
 * @param bytes Transfer length in Byte; zero is a no-op.
 * @param target_pe Source in CANN Root WORLD coordinates.
 * @pre The consumer validates addresses, length, target PE and MTE reachability.
 * @note One nonzero call maps to one CANN call. Return makes local_dst available to later work.
 */
__aicore__ inline void get(__gm__ void *local_dst, __gm__ const void *remote_src, uint32_t bytes, int32_t target_pe) {
  if (bytes != 0) {
    cann::device::get(local_dst, remote_src, bytes, target_pe);
  }
}

/**
 * @brief Synchronize-copy one data chunk and then update a Signal on the same target Root PE.
 *
 * @param remote_dst Symmetric GM destination represented by its local address.
 * @param local_src Local GM source owned by the caller.
 * @param bytes Transfer length in Byte; zero skips the data transfer but still updates the Signal.
 * @param remote_signal Symmetric GM address of the remote Signal word.
 * @param signal_value Value used by the Signal operation.
 * @param signal_op Set or Add operation performed after the Put completes.
 * @param target_pe Destination in CANN Root WORLD coordinates.
 * @pre For a multi-chunk message, the consumer uses put() for preceding chunks and calls put_signal() only for the
 * final chunk, so the logical message updates its Signal exactly once.
 * @note Return means the data is remote-visible before the Signal update becomes visible.
 */
__aicore__ inline void put_signal(__gm__ void *remote_dst, __gm__ const void *local_src, uint32_t bytes,
                                  __gm__ int32_t *remote_signal, int32_t signal_value, SignalOp signal_op,
                                  int32_t target_pe) {
  put(remote_dst, local_src, bytes, target_pe);
  signal(remote_signal, signal_value, signal_op, target_pe);
}

/**
 * @brief Complete a double-buffered MTE GET with a caller-selected burst cap.
 *
 * @param destination Local GM destination owned by the caller.
 * @param source Symmetric GM source represented by its local address.
 * @param elements Number of T elements to transfer; zero is a no-op.
 * @param source_pe Source in CANN Root WORLD coordinates.
 * @param max_block_bytes Burst cap in bytes; zero uses each allocated UB half. The default preserves remote-read policy.
 * @pre One worker exclusively owns both halves of the CANN-provided UB range and EVENT_ID0/1 for the call duration.
 * @note Return means the final MTE3 write to destination has completed.
 */
template <typename T>
__aicore__ inline void get_pipelined(GM_ADDR destination, GM_ADDR source, int64_t elements, int source_pe,
                                    uint64_t max_block_bytes = 4 * 1024) {
  if (elements == 0) {
    return;
  }
  uint64_t ub_address;
  uint64_t ub_bytes;
  if (!cann::device::query_mte_resources(source_pe, ub_address, ub_bytes)) {
    AscendC::Trap();
  }
  auto *remote = reinterpret_cast<__gm__ T *>(cann::device::remote_ptr(source, source_pe));
  if (remote == nullptr) {
    AscendC::Trap();
  }
  auto *local = reinterpret_cast<__gm__ T *>(destination);
  uint64_t block_bytes = ub_bytes / 2;
  // Remote callers retain small bursts; local copies can use their full double buffer.
  if (max_block_bytes != 0 && block_bytes > max_block_bytes) {
    block_bytes = max_block_bytes;
  }
  uint64_t total_bytes = static_cast<uint64_t>(elements) * sizeof(T);
  AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
  AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
  for (uint64_t offset = 0, block = 0; offset < total_bytes; offset += block_bytes, ++block) {
    AscendC::TEventID event = (block & 1) ? EVENT_ID1 : EVENT_ID0;
    auto *buffer = reinterpret_cast<__ubuf__ T *>(ub_address + (block & 1) * block_bytes);
    uint64_t bytes = total_bytes - offset < block_bytes ? total_bytes - offset : block_bytes;
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event);
    cann::device::copy_gm_to_ub(buffer, remote + offset / sizeof(T), bytes);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(event);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(event);
    cann::device::copy_ub_to_gm(local + offset / sizeof(T), buffer, bytes);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event);
  }
  AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
  AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
}

}  // namespace hyper_multicore::shmem::data_plane
