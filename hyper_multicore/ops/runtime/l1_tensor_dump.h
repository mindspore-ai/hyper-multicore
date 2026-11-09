/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MULTICORE_SCHEDULER_L1_TENSOR_DUMP_H
#define MULTICORE_SCHEDULER_L1_TENSOR_DUMP_H

#include "kernel_operator.h"

namespace MulticoreRuntime {

using namespace AscendC;  // NOLINT(build/namespaces)

constexpr uint32_t L1_DUMP_MAGIC = 0x4C314450;
constexpr uint32_t L1_DUMP_VERSION = 2;
constexpr uint32_t L1_DUMP_MAX_RECORDS = 4096;
constexpr uint32_t L1_DUMP_PHASE_BEFORE = 1;
constexpr uint32_t L1_DUMP_PHASE_AFTER = 2;
constexpr uint32_t L1_DUMP_STATUS_PENDING = 0;
constexpr uint32_t L1_DUMP_STATUS_CAPTURED = 1;

struct L1DumpHeader {
  uint32_t magic;
  uint32_t version;
  uint32_t enabled;
  uint32_t record_count;
  uint32_t record_stride;
  uint32_t records_offset;
  uint32_t payload_offset;
  uint32_t total_bytes;
};

struct L1DumpRecord {
  uint32_t task_id;
  uint32_t phase;
  uint32_t parameter_position;
  uint32_t source_offset_low;
  uint32_t source_offset_high;
  uint32_t payload_offset;
  uint32_t payload_bytes;
  uint32_t status;
  uint32_t task_index;
  uint32_t worker_id;
  uint32_t padding[6];  // A core must not write a cache line containing another core's status.
};

static_assert(sizeof(L1DumpHeader) == 32, "L1DumpHeader ABI must remain 32 bytes");
static_assert(sizeof(L1DumpRecord) == 64, "L1DumpRecord ABI must remain one 64-byte data-cache line");

template <uint32_t SharedParameterBase = 0, uint32_t SharedParameterCount = 0>
__aicore__ inline GM_ADDR ResolveL1ParameterAddress(GM_ADDR *parameters, uint32_t position, GM_ADDR shared_table) {
  if constexpr (SharedParameterCount != 0) {
    if (position >= SharedParameterBase) {
      if (position - SharedParameterBase >= SharedParameterCount || shared_table == nullptr) {
        AscendC::Trap();
      }
      // Shared sidecars are logical diagnostic ports, not entries in the native input list.
      return reinterpret_cast<GM_ADDR>(
        reinterpret_cast<__gm__ uint64_t *>(shared_table)[position - SharedParameterBase + 1]);
    }
  }
  return parameters[position];
}

__aicore__ inline bool ShouldSkipL1Record(GlobalTensor<uint32_t> &record_words, uint32_t task_id, uint32_t phase,
                                         uint32_t worker_id) {
  return record_words.GetValue(0) != task_id || record_words.GetValue(1) != phase ||
         record_words.GetValue(7) != L1_DUMP_STATUS_PENDING ||
         (record_words.GetValue(9) != 0xFFFFFFFF && record_words.GetValue(9) != worker_id);
}

template <uint32_t SharedParameterBase = 0, uint32_t SharedParameterCount = 0>
__aicore__ inline void CaptureL1TaskTensors(GM_ADDR buffer, GM_ADDR *parameters, uint32_t task_id, uint32_t phase,
                                            uint32_t worker_id, GM_ADDR shared_table = nullptr) {
  __gm__ uint8_t *dump_buffer = (__gm__ uint8_t *)(buffer);
  GlobalTensor<uint32_t> header_words;
  header_words.SetGlobalBuffer((__gm__ uint32_t *)(dump_buffer), sizeof(L1DumpHeader) / sizeof(uint32_t));
  DataCacheCleanAndInvalid<uint32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(header_words[0]);
  PipeBarrier<PIPE_ALL>();
  if (header_words.GetValue(0) != L1_DUMP_MAGIC || header_words.GetValue(1) != L1_DUMP_VERSION ||
      header_words.GetValue(2) == 0) {
    return;
  }
  uint32_t record_count = header_words.GetValue(3);
  uint32_t record_stride = header_words.GetValue(4);
  uint32_t records_offset = header_words.GetValue(5);
  uint32_t payload_offset = header_words.GetValue(6);
  uint32_t total_bytes = header_words.GetValue(7);
  uint64_t records_end = static_cast<uint64_t>(records_offset) + static_cast<uint64_t>(record_count) * record_stride;
  if (record_count > L1_DUMP_MAX_RECORDS || record_stride != sizeof(L1DumpRecord) ||
      records_offset < sizeof(L1DumpHeader) || records_end > payload_offset || payload_offset > total_bytes) {
    AscendC::Trap();
  }
  for (uint32_t record_index = 0; record_index < record_count; ++record_index) {
    __gm__ uint8_t *record_address = dump_buffer + records_offset + record_index * record_stride;
    GlobalTensor<uint32_t> record_words;
    record_words.SetGlobalBuffer((__gm__ uint32_t *)(record_address), sizeof(L1DumpRecord) / sizeof(uint32_t));
    DataCacheCleanAndInvalid<uint32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(record_words[0]);
    PipeBarrier<PIPE_ALL>();
    if (ShouldSkipL1Record(record_words, task_id, phase, worker_id)) {
      continue;
    }
    uint64_t source_offset =
      static_cast<uint64_t>(record_words.GetValue(3)) | (static_cast<uint64_t>(record_words.GetValue(4)) << 32);
    uint32_t parameter_position = record_words.GetValue(2);
    uint32_t record_payload_offset = record_words.GetValue(5);
    uint32_t payload_bytes = record_words.GetValue(6);
    if (record_payload_offset < payload_offset ||
        static_cast<uint64_t>(record_payload_offset) + payload_bytes > total_bytes) {
      AscendC::Trap();
    }
    if (payload_bytes != 0) {
      GlobalTensor<uint8_t> source_bytes;
      GlobalTensor<uint8_t> destination_bytes;
      GM_ADDR source = ResolveL1ParameterAddress<SharedParameterBase, SharedParameterCount>(
        parameters, parameter_position, shared_table);
      source_bytes.SetGlobalBuffer((__gm__ uint8_t *)(source) + source_offset, payload_bytes);
      destination_bytes.SetGlobalBuffer(dump_buffer + record_payload_offset, payload_bytes);
      DataCacheCleanAndInvalid<uint8_t, CacheLine::ENTIRE_DATA_CACHE>(source_bytes);
      PipeBarrier<PIPE_ALL>();
      for (uint32_t byte_index = 0; byte_index < payload_bytes; ++byte_index) {
        destination_bytes.SetValue(byte_index, source_bytes.GetValue(byte_index));
      }
      DataCacheCleanAndInvalid<uint8_t, CacheLine::ENTIRE_DATA_CACHE>(destination_bytes);
      PipeBarrier<PIPE_ALL>();
    }
    record_words.SetValue(7, L1_DUMP_STATUS_CAPTURED);
    DataCacheCleanAndInvalid<uint32_t, CacheLine::SINGLE_CACHE_LINE>(record_words[0]);
    PipeBarrier<PIPE_ALL>();
  }
}

}  // namespace MulticoreRuntime

#endif  // MULTICORE_SCHEDULER_L1_TENSOR_DUMP_H
