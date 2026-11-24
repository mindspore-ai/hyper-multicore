/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * Licensed under CANN Open Software License Agreement Version 1.0.
 */
#ifndef HYPER_MULTICORE_OPS_COMMON_MEGA_MOE_SITU_GLU_H_
#define HYPER_MULTICORE_OPS_COMMON_MEGA_MOE_SITU_GLU_H_
#include <cstdint>
#include <cstddef>
#include "kernel_operator.h"
#include "lib/activation/sigmoid.h"
#include "lib/math/tanh.h"

// The upstream compute classes consume this task-local tiling view.
struct SituGluTilingData {
  int64_t coreNumAll;
  int64_t dimBatchSize;
  int64_t dim2H;
  int64_t isLongH;
  int64_t ubMaxPair;
  float beta;
  float linearBeta;
  int64_t activateLeft;
};
using SituGluGradTilingData = SituGluTilingData;
#include "../../situ_glu/situ_glu.hpp"
#include "../../situ_glu_grad/situ_glu_grad.hpp"

namespace MulticoreRuntime {
// The host serializes exactly two little-endian FP32 values per activation plan.
struct SituParameters {
  float beta;
  float linearBeta;
};
static_assert(sizeof(SituParameters) == 8 && offsetof(SituParameters, linearBeta) == 4);

__aicore__ inline SituParameters ReadSituParameters(GM_ADDR record) {
  const auto *params = reinterpret_cast<__gm__ const SituParameters *>(record);
  return {params->beta, params->linearBeta};
}

template <bool Backward>
__aicore__ inline void RunSiTUGlu(GM_ADDR input, GM_ADDR gradient, GM_ADDR output, int64_t rows, int64_t width,
                                  const SituParameters &params) {
  constexpr int64_t tilePairs = 2048;
  SituGluTilingData tiling = {1, rows, 2 * width, width > tilePairs, tilePairs, params.beta, params.linearBeta, 1};
  AscendC::TPipe pipe;
  if constexpr (Backward) {
    SituGluGradOps::SituGluGradBase<bfloat16_t> kernel(&pipe);
    kernel.Init(gradient, input, output, &tiling, 0);
    kernel.Process();
  } else {
    SituGluOps::SituGluBase<bfloat16_t> kernel(&pipe);
    kernel.Init(input, output, &tiling, 0);
    kernel.Process();
  }
  AscendC::PipeBarrier<PIPE_ALL>();
}
}  // namespace MulticoreRuntime
#endif  // HYPER_MULTICORE_OPS_COMMON_MEGA_MOE_SITU_GLU_H_
