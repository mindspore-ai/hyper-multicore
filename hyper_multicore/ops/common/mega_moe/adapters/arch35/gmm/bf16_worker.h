// Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
// Task binding only; numerical blocks and their tile loop are upstream Blaze.
#pragma once
#include "blaze/gemm/kernel/kernel_grouped_matmul.h"
#include "common/mega_moe/gmm_logical_workers.h"
#include "bf16_problem.h"

namespace MulticoreBf16 {
template <class LayoutA, class LayoutB, class Output = bfloat16_t>
__aicore__ inline void RunStandardGmm(const Problem& p, GM_ADDR a, GM_ADDR b, GM_ADDR c,
                                     uint32_t lane, uint32_t cores, uint32_t expert) {
  using Blaze::Gemm::KernelGroupedMmadNoQuant;
  using Blaze::Gemm::MatmulMultiBlockBasic;
  using Blaze::Gemm::MatmulOutputMode;
  namespace Block = Blaze::Gemm::Block;
  using Row = asc::te::nd_ext_layout_ptn;
  using Policy = MatmulMultiBlockBasic<0, 0, KernelGroupedMmadNoQuant, 0, MatmulOutputMode::OVERWRITE>;
  using Mmad = Block::BlockMmad<Policy, bfloat16_t, LayoutA, bfloat16_t, LayoutB,
                               Output, Row, float, Row>;
  using Shape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
  using Kernel = Blaze::Gemm::Kernel::GemmUniversal<Shape, Mmad, Blaze::Epilogue::Block::BlockEpilogueEmpty,
                                       Block::BlockSchedulerGmmNoQuant>;
  // Basic arch35 tiling: BM/BN/BK=256/256/64, stepKa/stepKb=4.
  typename Mmad::Params mm{a, b, c, nullptr, nullptr, nullptr,
      256, 256, 256, 256, 256, 64, 2, 1, nullptr, 0};
  const auto tail = SplitTail(p.m, p.n, 256, 256, cores);
  typename Block::BlockSchedulerGmmNoQuant::Params schedule{
      256, 256, tail.m, tail.n, 1, 1, -1, 1, static_cast<int64_t>(p.m),
      true, true, true, Mmad::TRANS_B, false, sizeof(bfloat16_t)};
  typename Kernel::Params params{{p.m, p.n, p.k, 1}, mm, {}, schedule,
                                 {1, -1, 0, 1, 1, 1, 0, 0}};
  struct Mapping {
    uint32_t expert;
    __aicore__ inline uint32_t operator()(uint32_t lane, uint32_t cores, uint64_t blocks) const {
      return Mmad::TRANS_A ? MulticoreGmm::RotateLogicalCube(lane, expert, blocks, cores) : lane;
    }
  };
  AscendC::TPipe pipe;
  Kernel kernel;
  kernel.RunSingleProblem(params, lane, cores, Mapping{expert});
  AscendC::PipeBarrier<PIPE_ALL>();
}

template <class WeightOutput = bfloat16_t, class Task>
__aicore__ inline void ExecuteGmmTask(const Task& task, GM_ADDR* inputs, uint32_t cores,
                                      const TCubeTiling& tiling) {
  if ASCEND_IS_AIV { return; }
  const uint32_t expert = task.task_index / cores, lane = task.task_index % cores;
  AscendC::GlobalTensor<int64_t> groups;
  groups.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(inputs[task.inputs[2].input_position]));
  const int64_t start = expert ? groups.GetValue(expert - 1) : 0;
  const int64_t end = groups.GetValue(expert);
  if (start < 0 || end < start) AscendC::Trap();
  const bool ta = task.inputs[0].transpose_flag == 1, tb = task.inputs[1].transpose_flag == 1;
  const auto p = MakeProblem(tiling.M, tiling.N, tiling.Ka, expert, start, end - start, ta);
  auto a = inputs[task.inputs[0].input_position] + p.a * sizeof(bfloat16_t);
  auto b = inputs[task.inputs[1].input_position] + p.b * sizeof(bfloat16_t);
  // TN writes partial dW; NN/NT retain the activation output dtype.
  const uint32_t output_bytes = ta ? sizeof(WeightOutput) : sizeof(bfloat16_t);
  if (task.outputs[0].data_type != output_bytes) AscendC::Trap();
  auto c = inputs[task.outputs[0].input_position] + p.c * output_bytes;
  if (start == end) {
    if (ta) {
      // Empty expert still owns a complete zero weight gradient. Cache lines are disjoint.
      AscendC::GlobalTensor<uint32_t> output;
      const uint64_t words = p.m * p.n * output_bytes / sizeof(uint32_t);
      output.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(c), words);
      for (uint64_t line = lane; line < (words + 31) / 32; line += cores) {
        for (uint64_t i = line * 32; i < (line + 1) * 32 && i < words; ++i) output.SetValue(i, 0);
        AscendC::DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                         AscendC::DcciDst::CACHELINE_OUT>(output[line * 32]);
      }
    }
    return;
  }
  AscendC::AscendCUtils::SetOverflow(1);
  if (ta) {
    RunStandardGmm<asc::te::dn_ext_layout_ptn, asc::te::nd_ext_layout_ptn, WeightOutput>(
        p, a, b, c, lane, cores, expert);
  } else if (tb) {
    RunStandardGmm<asc::te::nd_ext_layout_ptn, asc::te::dn_ext_layout_ptn>(p, a, b, c, lane, cores, expert);
  } else {
    RunStandardGmm<asc::te::nd_ext_layout_ptn, asc::te::nd_ext_layout_ptn>(p, a, b, c, lane, cores, expert);
  }
}
}  // namespace MulticoreBf16
