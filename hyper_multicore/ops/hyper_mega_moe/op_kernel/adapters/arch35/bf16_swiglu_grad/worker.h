// Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#pragma once
#include "task_tiling.h"
#include "swi_glu_grad/arch35/swi_glu_grad_regbase_tiling.h"
#include "swi_glu_grad/arch35/swi_glu_grad_move_align.h"
namespace MulticoreBf16 {
__aicore__ inline void ExecuteSwiGluGrad(GM_ADDR grad, GM_ADDR x, GM_ADDR out,
                                        int64_t rows, int64_t columns) {
  const auto tiling = MakeSwiGluGradTiling<GluBaseTilingData>(rows, columns);
  AscendC::TPipe pipe;
  SwiGluGrad::SwiGluGradMoveAlignKernel<bfloat16_t> kernel(pipe);
  kernel.Init(grad, x, out, tiling, 0);
  kernel.Process();
  AscendC::PipeBarrier<PIPE_ALL>();
}
}  // namespace MulticoreBf16
