// Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#ifndef HYPER_MULTICORE_MX_QUANT_DESCRIPTOR_H_
#define HYPER_MULTICORE_MX_QUANT_DESCRIPTOR_H_
#include <cfloat>
#include <cstddef>
#include <cstdint>

namespace MulticoreMx {
constexpr uint32_t MX_FP8_E4M3FN_GE_TYPE = 36;
constexpr uint32_t MX_QUANT_MAGIC = 0x4D585154;
constexpr uint32_t MX_QUANT_TASK = 143;
constexpr uint32_t MX_SWIGLU_TASK = 145;
constexpr uint32_t MX_SWIGLU_GRAD_TASK = 146;

struct MxQuantDescriptor {
  uint32_t magic;
  uint16_t version, input_bytes;
  uint32_t columns, expert_index, worker_index, worker_count;
  uint32_t block_rows, block_columns, task_type, size;
  float clamp_limit;
  uint32_t reserved;
  uint64_t padding[2];
};
static_assert(sizeof(MxQuantDescriptor) == 64, "MX quant descriptor size mismatch");
static_assert(offsetof(MxQuantDescriptor, columns) == 8, "MX quant column offset mismatch");
static_assert(offsetof(MxQuantDescriptor, clamp_limit) == 40, "MX quant clamp offset mismatch");

#ifdef __NPU_ARCH__
__aicore__
#endif
inline bool IsMxQuantDescriptorValid(const MxQuantDescriptor& r) {
  const bool supported = r.task_type == MX_QUANT_TASK || r.task_type == MX_SWIGLU_TASK
                        || r.task_type == MX_SWIGLU_GRAD_TASK;
  const uint32_t block_columns = r.task_type == MX_SWIGLU_GRAD_TASK ? 128 : 256;
  return r.magic == MX_QUANT_MAGIC && r.version == 2 && r.size == sizeof(r) && r.input_bytes == 2
    && r.columns > 0 && r.columns % 64 == 0 && r.worker_count > 0 && r.worker_index < r.worker_count
    && supported && (r.task_type != MX_SWIGLU_GRAD_TASK || r.columns % 128 == 0)
    && r.block_rows == 64 && r.block_columns == block_columns
    && r.clamp_limit >= 0.0f && r.clamp_limit <= FLT_MAX
    && (r.task_type != MX_QUANT_TASK || r.clamp_limit == 0.0f);
}
}  // namespace MulticoreMx
#endif
