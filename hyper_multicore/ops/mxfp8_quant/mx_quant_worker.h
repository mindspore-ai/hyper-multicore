// Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#ifndef HYPER_MULTICORE_MX_QUANT_WORKER_H_
#define HYPER_MULTICORE_MX_QUANT_WORKER_H_

#include "kernel_operator.h"
#include "runtime/runtime_config.hpp"
#include "mx_quant_descriptor.h"
#include "mx_dispatch_quant_range.h"
#define HYPER_MULTICORE_MX_LOGICAL_VECTOR
#ifndef TPL_SCALE_ALG_0
#define TPL_SCALE_ALG_0 0
#endif
#include "dynamic/arch35/dynamic_mx_quant_with_dual_axis_base.h"
#include "swiglu/arch35/swiglu_mx_quant_with_dual_axis_regbase.h"
#include "swiglu_grad/arch35/swiglu_backward_mx_quant_with_dual_axis_regbase.h"

#ifdef HYPER_MULTICORE_MX_ROW_ONLY
#include "mx_swiglu_row_only_worker.h"
#endif

namespace MulticoreMx {
__aicore__ inline int64_t MxQuantCeil(int64_t value, int64_t block) {
  return (value + block - 1) / block;
}

__aicore__ inline void ExecuteMxQuantRange(const MxQuantDescriptor& r, GM_ADDR source, GM_ADDR gradient,
                                         GM_ADDR row, GM_ADDR row_scale, GM_ADDR column, GM_ADDR column_scale,
                                         int64_t expert_start, int64_t row_begin, int64_t count) {
  if (count == 0) return;
  const int64_t start = expert_start + row_begin;
  const int64_t width = r.columns;
  const int64_t source_width = r.task_type == MX_SWIGLU_TASK ? 2 * width : width;
  source += start * source_width * sizeof(bfloat16_t);
  row += start * width;
  column += start * width;
  row_scale += start * (width / 32);
  column_scale += (expert_start / 64 + r.expert_index + row_begin / 64) * width * 2;
#if (__NPU_ARCH__ == 3510)
  const int64_t overflow_mode = AscendC::GetCtrlSpr<60, 60>();
#endif
  AscendC::TPipe pipe;
  if (r.task_type == MX_QUANT_TASK || r.task_type == MX_DISPATCH_QUANT_TASK) {
    DynamicMxQuantWithDualAxisTilingData tiling{};
    tiling.totalCoreNum = tiling.usedCoreNum = r.worker_count;
    tiling.roundMode = 4;
    tiling.dstType = MX_FP8_E4M3FN_GE_TYPE;
    tiling.scaleAlg = 1;
    tiling.blockSize = 32;
    tiling.dim0 = 1;
    tiling.dimNeg2 = count;
    tiling.dimNeg1 = width;
    tiling.blockW = r.block_columns;
    tiling.splitBlockH = r.block_rows;
    tiling.dimNeg2Tail = (count - 1) % r.block_rows + 1;
    tiling.dimNeg1Tail = (width - 1) % r.block_columns + 1;
    tiling.dimNeg2SplitBlockNum = MxQuantCeil(count, r.block_rows);
    tiling.dimNeg1BlockNum = MxQuantCeil(width, r.block_columns);
    tiling.blockCountPerBatch = tiling.dimNeg2SplitBlockNum * tiling.dimNeg1BlockNum;
    tiling.headCoreNum = tiling.blockCountPerBatch % r.worker_count;
    tiling.blockPerHeadCore = MxQuantCeil(tiling.blockCountPerBatch, r.worker_count);
    tiling.blockPerTailCore = tiling.blockCountPerBatch / r.worker_count;
    tiling.dimNeg2IsOdd = MxQuantCeil(count, 32) % 2;
    tiling.scale1ColCountPerBatch = width / 32;
    tiling.scale2RowCountPerBatch = MxQuantCeil(count, 64) * 2;
    DynamicMxQuantWithDualAxis::DynamicMxQuantWithDualAxisBase<
      bfloat16_t, fp8_e4m3fn_t, fp8_e4m3fn_t, AscendC::RoundMode::CAST_RINT, 1>
      op(&tiling, &pipe, r.worker_index);
    op.Init(source, row, row_scale, column, column_scale);
    op.Process();
  } else if (r.task_type == MX_SWIGLU_TASK) {
    SwigluMxQuantWithDualAxisTilingData tiling{};
    tiling.usedCoreNum = r.worker_count;
    tiling.dstType = MX_FP8_E4M3FN_GE_TYPE;
    tiling.activateLeft = 1;
    tiling.dimM = count;
    tiling.dimN = width;
    tiling.numGroups = 1;
    tiling.blockW = r.block_columns;
    tiling.splitBlockH = r.block_rows;
    tiling.dimNBlockNum = MxQuantCeil(width, r.block_columns);
    tiling.dimNTail = (width - 1) % r.block_columns + 1;
    SwigluMxQuantWithDualAxis::SwigluMxQuantWithDualAxisBase<
      bfloat16_t, fp8_e4m3fn_t, 1, AscendC::RoundMode::CAST_RINT, 1, 0>
      op(&tiling, &pipe, r.worker_index);
    op.Init(source, nullptr, row, row_scale, column, column_scale);
    op.Process();
  } else {
    const int64_t half_width = width / 2;
    gradient += start * half_width * sizeof(bfloat16_t);
    SwigluBackwardMxQuantWithDualAxisTilingData tiling{};
    tiling.usedCoreNum = r.worker_count;
    tiling.dstType = MX_FP8_E4M3FN_GE_TYPE;
    tiling.activateLeft = 1;
    tiling.dimM = count;
    tiling.dimN = half_width;
    tiling.numGroups = 1;
    tiling.blockW = r.block_columns;
    tiling.splitBlockH = r.block_rows;
    tiling.dimNBlockNum = MxQuantCeil(half_width, r.block_columns);
    tiling.dimNTail = (half_width - 1) % r.block_columns + 1;
    tiling.yGradRowStride = half_width;
    tiling.dimGradX = width;
    tiling.dimBatch = 1;
    SwigluBackwardMxQuantWithDualAxis::SwigluBackwardMxQuantWithDualAxisBase<
      bfloat16_t, fp8_e4m3fn_t, 1, AscendC::RoundMode::CAST_RINT, 1, 0>
      op(&tiling, &pipe, r.worker_index);
    op.Init(source, gradient, nullptr, row, row_scale, column, column_scale);
    op.Process();
  }
#if (__NPU_ARCH__ == 3510)
  AscendC::SetCtrlSpr<60, 60>(overflow_mode);
#endif
}

__aicore__ inline void ExecuteMxDispatchQuant(const MxQuantDescriptor& r, GM_ADDR* input_list,
                                             GM_ADDR source, GM_ADDR group_list, GM_ADDR row, GM_ADDR row_scale,
                                             GM_ADDR column, GM_ADDR column_scale) {
  AscendC::GlobalTensor<int64_t> groups, offsets;
  AscendC::GlobalTensor<int32_t> sizes;
  groups.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(group_list));
  const uint64_t roles = r.reserved[2];
  offsets.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(input_list[roles & 255]));
  sizes.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(input_list[(roles >> 8) & 255]));
  groups.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
  offsets.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
  sizes.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
  const int64_t ep = (roles >> 16) & 65535;
  const int64_t experts = (roles >> 32) & 65535;
  const int64_t start = r.expert_index == 0 ? 0 : groups.GetValue(r.expert_index - 1);
  const int64_t count = groups.GetValue(r.expert_index) - start;
  const int64_t split = r.reserved[1];
  if (count == 0) return;
  if (r.reserved[0] != UINT64_MAX) {
    for (int64_t peer = 0; peer < ep; ++peer) {
      const int64_t index = peer * experts + r.expert_index;
      const int64_t source_begin = offsets.GetValue(index) / r.columns - start;
      const int64_t source_end = source_begin + sizes.GetValue(index) / r.columns;
      const int64_t begin = source_begin + static_cast<int64_t>(r.reserved[0]) * split;
      if (begin >= source_end) continue;
      const int64_t end = begin + split < source_end ? begin + split : source_end;
      const auto range = MxFragmentInterior(begin, end, count);
      ExecuteMxQuantRange(r, source, nullptr, row, row_scale, column, column_scale,
                          start, range.begin, range.end - range.begin);
    }
  } else {
    for (int64_t begin = static_cast<int64_t>(r.worker_index) * 64; begin < count;
         begin += static_cast<int64_t>(r.worker_count) * 64) {
      const int64_t end = begin + 64 < count ? begin + 64 : count;
      bool owned = false;
      for (int64_t peer = 0; peer < ep; ++peer) {
        const int64_t index = peer * experts + r.expert_index;
        const int64_t source_begin = offsets.GetValue(index) / r.columns - start;
        const int64_t source_end = source_begin + sizes.GetValue(index) / r.columns;
        owned = owned || MxFragmentOwnsBlock(begin, end, source_begin, source_end, split);
      }
      if (!owned) {
        MxQuantDescriptor single = r;
        single.worker_index = 0;
        single.worker_count = 1;
        ExecuteMxQuantRange(single, source, nullptr, row, row_scale, column, column_scale,
                            start, begin, end - begin);
      }
    }
  }
}

__aicore__ inline void ExecuteMxQuant(const MxQuantDescriptor& r, GM_ADDR source, GM_ADDR gradient,
                                     GM_ADDR group_list, GM_ADDR row, GM_ADDR row_scale,
                                     GM_ADDR column, GM_ADDR column_scale) {
  AscendC::GlobalTensor<int64_t> groups;
  groups.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(group_list));
  groups.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
  const int64_t start = r.expert_index == 0 ? 0 : groups.GetValue(r.expert_index - 1);
  const int64_t count = groups.GetValue(r.expert_index) - start;
  ExecuteMxQuantRange(r, source, gradient, row, row_scale, column, column_scale, start, 0, count);
}

__aicore__ inline void ExecuteMxQuantTask(const MulticoreRuntime::TaskDesc& task, GM_ADDR* input_list) {
  const bool backward = static_cast<uint32_t>(task.task_type) == MX_SWIGLU_GRAD_TASK;
  const uint32_t group_slot = backward ? 2 : 1;
  AscendC::GlobalTensor<uint64_t> records;
  records.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(
    input_list[task.inputs[group_slot + 1].input_position]));
  MxQuantDescriptor record;
  auto values = reinterpret_cast<uint64_t*>(&record);
  for (uint32_t index = 0; index < sizeof(record) / sizeof(uint64_t); ++index)
    values[index] = records.GetValue(task.extra_value_0 * (sizeof(record) / sizeof(uint64_t)) + index);
  if (record.task_type == MX_DISPATCH_QUANT_TASK) {
    if (!IsMxQuantDescriptorValid(record)) AscendC::Trap();
    ExecuteMxDispatchQuant(record, input_list, input_list[task.inputs[0].input_position],
      input_list[task.inputs[1].input_position], input_list[task.outputs[0].input_position],
      input_list[task.outputs[1].input_position], input_list[task.outputs[2].input_position],
      input_list[task.outputs[3].input_position]);
    return;
  }
#ifdef HYPER_MULTICORE_MX_ROW_ONLY
  if (record.task_type == MX_SWIGLU_ROW_TASK) {
    ExecuteMxSwigluRowOnly(record, input_list[task.inputs[0].input_position],
      input_list[task.inputs[1].input_position], input_list[task.outputs[0].input_position],
      input_list[task.outputs[1].input_position]);
    return;
  }
#endif
  ExecuteMxQuant(record, input_list[task.inputs[0].input_position],
    backward ? input_list[task.inputs[1].input_position] : nullptr,
    input_list[task.inputs[group_slot].input_position], input_list[task.outputs[0].input_position],
    input_list[task.outputs[1].input_position], input_list[task.outputs[2].input_position],
    input_list[task.outputs[3].input_position]);
}
}  // namespace MulticoreMx
#endif
