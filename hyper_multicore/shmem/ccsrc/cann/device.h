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

#include "cann/signal.h"
#include "kernel_operator.h"
#include "shmem.h"

namespace hyper_multicore::shmem::cann::device {

/** @brief Return the calling PE's Root rank. */
ACLSHMEM_DEVICE int32_t my_pe() { return aclshmem_my_pe(); }

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
/** @brief Return the initialized number of UDMA QPs per peer. */
ACLSHMEM_DEVICE uint32_t udma_qp_count() { return aclshmemi_udma_qp_info_fetch()->qp_num; }

/** @brief Complete a GET on a caller-owned QP, waiting after each SDK-sized chunk. */
template <typename T>
ACLSHMEM_DEVICE void get_udma(GM_ADDR destination, GM_ADDR source, int64_t elements, int32_t source_pe, uint32_t qp) {
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
    aclshmemx_udma_qp_quiet(source_pe, qp);
    local += bytes;
    remote += bytes;
    remaining -= bytes;
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
  return aclshmem_ptr(local_symmetric_address, target_pe);
}

/** @brief Order earlier SHMEM operations before later SHMEM operations issued by the calling AIV. */
ACLSHMEM_DEVICE void fence() { aclshmem_fence(); }

ACLSHMEM_DEVICE constexpr int32_t ToCannCompareOp(CompareOp comparison) {
  switch (comparison) {
    case CompareOp::Equal:
      return ACLSHMEM_CMP_EQ;
    case CompareOp::NotEqual:
      return ACLSHMEM_CMP_NE;
    case CompareOp::Greater:
      return ACLSHMEM_CMP_GT;
    case CompareOp::GreaterEqual:
      return ACLSHMEM_CMP_GE;
    case CompareOp::Less:
      return ACLSHMEM_CMP_LT;
    case CompareOp::LessEqual:
      return ACLSHMEM_CMP_LE;
  }
  return ACLSHMEM_CMP_EQ;
}

/**
 * @brief Synchronize-copy local GM bytes to a symmetric address on one target Root PE.
 *
 * @param remote_dst Symmetric GM destination represented by its local address.
 * @param local_src Local GM source owned by the caller.
 * @param bytes Transfer length in Byte.
 * @param target_pe Destination in CANN Root WORLD coordinates.
 * @note The caller validates all arguments. Return means the data is remote-visible.
 */
ACLSHMEM_DEVICE void put(__gm__ void *remote_dst, __gm__ const void *local_src, uint32_t bytes, int32_t target_pe) {
  aclshmem_putmem(remote_dst, const_cast<__gm__ void *>(local_src), bytes, target_pe);
}

/**
 * @brief Synchronize-copy bytes from a symmetric address on one Root PE to local GM.
 *
 * @param local_dst Local GM destination owned by the caller.
 * @param remote_src Symmetric GM source represented by its local address.
 * @param bytes Transfer length in Byte.
 * @param target_pe Source in CANN Root WORLD coordinates.
 * @note The caller validates all arguments. Return makes local_dst available to later work in the calling Kernel.
 */
ACLSHMEM_DEVICE void get(__gm__ void *local_dst, __gm__ const void *remote_src, uint32_t bytes, int32_t target_pe) {
  aclshmem_getmem(local_dst, const_cast<__gm__ void *>(remote_src), bytes, target_pe);
}

/**
 * @brief Apply one Set or Add operation to a symmetric Signal on a target Root PE.
 *
 * @param remote_signal Symmetric GM address of the remote Signal word.
 * @param value Value used by the operation.
 * @param operation Set or Add.
 * @param target_pe Destination in CANN Root WORLD coordinates.
 * @note Return means the Signal update is remote-visible.
 */
ACLSHMEM_DEVICE void signal(__gm__ int32_t *remote_signal, int32_t value, SignalOp operation, int32_t target_pe) {
  aclshmemx_signal_op(remote_signal, value, operation == SignalOp::Add ? ACLSHMEM_SIGNAL_ADD : ACLSHMEM_SIGNAL_SET,
                      target_pe);
}

/** @brief Set one symmetric Signal on a target Root PE. */
ACLSHMEM_DEVICE void signal_set(__gm__ int32_t *remote_signal, int32_t value, int32_t target_pe) {
  signal(remote_signal, value, SignalOp::Set, target_pe);
}

/**
 * @brief Atomically add to one symmetric Signal on a target Root PE.
 *
 * @param remote_signal Symmetric GM address of the remote Signal word.
 * @param value Value added to the Signal word.
 * @param target_pe Destination in CANN Root WORLD coordinates.
 * @note Return means the Signal update is remote-visible.
 */
ACLSHMEM_DEVICE void signal_add(__gm__ int32_t *remote_signal, int32_t value, int32_t target_pe) {
  signal(remote_signal, value, SignalOp::Add, target_pe);
}

/**
 * @brief Block until one local Signal satisfies a comparison.
 *
 * @param local_signal Local GM address of the Signal word; no PE translation is performed.
 * @param compare Comparison applied between the observed Signal and value.
 * @param value Comparison value.
 * @return Signal value observed when the comparison is satisfied.
 */
ACLSHMEM_DEVICE int32_t signal_wait(__gm__ int32_t *local_signal, CompareOp comparison, int32_t value) {
  return aclshmem_signal_wait_until(local_signal, ToCannCompareOp(comparison), value);
}

/**
 * @brief Complete a collective device barrier over CANN Root WORLD.
 *
 * @note Every Root PE must participate. The caller satisfies CANN MIX Kernel and FFTS requirements and does not mix
 * this protocol with AscendC SyncAll.
 */
ACLSHMEM_DEVICE void barrier() { aclshmem_barrier(ACLSHMEM_TEAM_WORLD); }

/**
 * @brief Issue an explicit MTE NBI Put to one target Root PE.
 *
 * @param remote_dst Symmetric GM destination represented by its local address.
 * @param local_src Local GM source owned by the caller.
 * @param ub_scratch Caller-owned temporary UB address.
 * @param ub_scratch_bytes UB capacity in Byte.
 * @param bytes Transfer length in Byte.
 * @param target_pe Destination in CANN Root WORLD coordinates.
 * @param event_id Caller-owned MTE synchronization Event ID.
 * @note Return means issued, not completed. The caller owns UB/Event synchronization and eventual quiet.
 */
ACLSHMEM_DEVICE void mte_put_nbi(__gm__ uint8_t *remote_dst, __gm__ const uint8_t *local_src,
                                 __ubuf__ uint8_t *ub_scratch, uint32_t ub_scratch_bytes, uint32_t bytes,
                                 int32_t target_pe, uint32_t event_id) {
  aclshmemx_mte_put_nbi<uint8_t>(remote_dst, const_cast<__gm__ uint8_t *>(local_src), ub_scratch, ub_scratch_bytes,
                                 bytes, target_pe, event_id);
}

/** @brief Complete explicit MTE NBI operations previously issued by the calling AIV. */
ACLSHMEM_DEVICE void mte_quiet() { aclshmemx_mte_quiet(); }

/**
 * @brief Query the CANN-owned MTE resources for one target Root PE.
 *
 * @param target_pe Target in CANN Root WORLD coordinates.
 * @param ub_address Receives the CANN-owned UB base address available to the calling AIV.
 * @param ub_bytes Receives the CANN-owned UB capacity in Byte.
 * @return Whether the target supports MTE transport.
 */
ACLSHMEM_DEVICE bool query_mte_resources(int32_t target_pe, uint64_t &ub_address, uint64_t &ub_bytes) {
  auto *state = aclshmemi_get_state();
  ub_address = static_cast<uint64_t>(state->mte_config.aclshmem_ub);
  ub_bytes = state->mte_config.ub_size;
  return (state->topo_list[target_pe] & ACLSHMEM_TRANSPORT_MTE) != 0;
}

/** @brief Copy one contiguous byte range from GM to caller-owned UB. */
template <typename T>
ACLSHMEM_DEVICE void copy_gm_to_ub(__ubuf__ T *destination, __gm__ T *source, uint64_t bytes) {
  aclshmemi_copy_gm2ub(destination, source, bytes);
}

/** @brief Copy one contiguous byte range from caller-owned UB to GM. */
template <typename T>
ACLSHMEM_DEVICE void copy_ub_to_gm(__gm__ T *destination, __ubuf__ T *source, uint64_t bytes) {
  aclshmemi_copy_ub2gm(destination, source, bytes);
}

}  // namespace hyper_multicore::shmem::cann::device
