// Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#ifndef HYPER_MULTICORE_MX_SWIGLU_ROW_ONLY_WORKER_H_
#define HYPER_MULTICORE_MX_SWIGLU_ROW_ONLY_WORKER_H_
#include "swiglu_row/arch35/swiglu_mx_quant_tiling_data.h"
#include "swiglu_row/arch35/swiglu_mx_quant_last_last.h"

namespace MulticoreMx {
__aicore__ inline void ExecuteMxSwigluRowOnly(const MxQuantDescriptor& r, GM_ADDR source,
                                             GM_ADDR group_list, GM_ADDR row, GM_ADDR row_scale) {
  AscendC::GlobalTensor<int64_t> groups;
  groups.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(group_list));
  groups.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
  const int64_t start = r.expert_index == 0 ? 0 : groups.GetValue(r.expert_index - 1);
  const int64_t count = groups.GetValue(r.expert_index) - start;
  if (count == 0) return;
  source += start * 2 * r.columns * sizeof(bfloat16_t);
  row += start * r.columns;
  row_scale += start * (r.columns / 32);
  // The pinned row-only primitive partitions contiguous rows among logical workers.
  // A 16x256 UB block keeps its unchanged double-buffered allocation bounded.
  const int64_t block_rows = r.block_rows;
  SwigluMxQuantTilingData tiling{};
  tiling.usedCoreNum = r.worker_count;
  tiling.inputDim1 = count;
  tiling.inputDim2 = 2 * r.columns;
  tiling.outputDim2 = r.columns;
  tiling.maxBasicNumUbDim1 = block_rows;
  tiling.maxBasicNumUbDim2 = 1;
  tiling.tailCoreBasicNumDim1 = count / r.worker_count;
  tiling.frontCoreNum = count % r.worker_count;
  const int64_t front_rows = tiling.tailCoreBasicNumDim1 + 1;
  const int64_t tail_rows = tiling.tailCoreBasicNumDim1;
  tiling.frontCoreLoopTimes = (front_rows + block_rows - 1) / block_rows;
  tiling.frontCoreLastLoopBasicNum = (front_rows - 1) % block_rows + 1;
  tiling.tailCoreLoopTimes = (tail_rows + block_rows - 1) / block_rows;
  tiling.tailCoreLastLoopBasicNum = tail_rows == 0 ? 0 : (tail_rows - 1) % block_rows + 1;
  tiling.ubLoopPerRow = (r.columns + 255) / 256;
  tiling.ubTailPerRow = (r.columns - 1) % 256 + 1;
  tiling.activateLeft = 1;
  tiling.swigluMode = 0;
  tiling.roundMode = 1;
  tiling.scaleAlg = 1;
#if (__NPU_ARCH__ == 3510)
  const int64_t overflow_mode = AscendC::GetCtrlSpr<60, 60>();
#endif
  AscendC::TPipe pipe;
  SwigluMxQuant::SwigluMxQuantLastLast<bfloat16_t, fp8_e4m3fn_t, int64_t, false> op;
  op.Init(source, nullptr, row, row_scale, nullptr, &tiling, &pipe, r.worker_index);
  op.Process();
#if (__NPU_ARCH__ == 3510)
  AscendC::SetCtrlSpr<60, 60>(overflow_mode);
#endif
}
}  // namespace MulticoreMx
#endif
