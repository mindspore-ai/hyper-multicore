// Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#ifndef HYPER_MULTICORE_MX_QUANT_DESCRIPTOR_H_
#define HYPER_MULTICORE_MX_QUANT_DESCRIPTOR_H_
#include <cstddef>
#include <cstdint>

namespace MulticoreMx {
constexpr uint32_t MX_FP8_E4M3FN_GE_TYPE = 36;
constexpr uint32_t MX_QUANT_MAGIC = 0x4D585154;
constexpr uint32_t MX_QUANT_TASK = 143;
constexpr uint32_t MX_SWIGLU_TASK = 145;
constexpr uint32_t MX_SWIGLU_GRAD_TASK = 146;
constexpr uint32_t MX_SWIGLU_ROW_TASK = 147;
constexpr uint32_t MX_DISPATCH_QUANT_TASK = 153;

struct MxQuantDescriptor {
  uint32_t magic;
  uint16_t version, input_bytes;
  uint32_t columns, expert_index, worker_index, worker_count;
  uint32_t block_rows, block_columns, task_type, size;
  uint64_t reserved[3];
};
static_assert(sizeof(MxQuantDescriptor) == 64, "MX quant descriptor size mismatch");
static_assert(offsetof(MxQuantDescriptor, columns) == 8, "MX quant column offset mismatch");
static_assert(offsetof(MxQuantDescriptor, reserved) == 40, "MX quant reserved offset mismatch");

#ifdef __NPU_ARCH__
__aicore__
#endif
inline bool IsMxQuantDescriptorValid(const MxQuantDescriptor& r) {
  if (r.task_type == MX_DISPATCH_QUANT_TASK) {
    const uint64_t roles = r.reserved[2];
    return r.magic == MX_QUANT_MAGIC && r.version == 2 && r.size == sizeof(r) && r.input_bytes == 2
      && r.columns > 0 && r.columns % 64 == 0 && r.worker_count > 0 && r.worker_index < r.worker_count
      && r.block_rows == 64 && r.block_columns == 256 && r.reserved[1] > 0 && r.reserved[1] % 64 == 0
      && (roles & 255) < 64 && ((roles >> 8) & 255) < 64
      && ((roles >> 16) & 65535) > 0 && ((roles >> 32) & 65535) > r.expert_index && (roles >> 48) == 0;
  }
  const bool supported = r.task_type == MX_QUANT_TASK || r.task_type == MX_SWIGLU_TASK
                        || r.task_type == MX_SWIGLU_GRAD_TASK || r.task_type == MX_SWIGLU_ROW_TASK;
  const uint32_t block_columns = r.task_type == MX_SWIGLU_GRAD_TASK ? 128 : 256;
  return r.magic == MX_QUANT_MAGIC && r.version == 1 && r.size == sizeof(r) && r.input_bytes == 2
    && r.columns > 0 && r.columns % 64 == 0 && r.worker_count > 0 && r.worker_index < r.worker_count
    && supported && (r.task_type != MX_SWIGLU_GRAD_TASK || r.columns % 128 == 0)
    && r.block_rows == (r.task_type == MX_SWIGLU_ROW_TASK ? 16 : 64)
    && r.block_columns == block_columns;
}
}  // namespace MulticoreMx
#endif
