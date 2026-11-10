// Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#ifndef HYPER_MULTICORE_BF16_GMM_CGMCT_WORKER_H_
#define HYPER_MULTICORE_BF16_GMM_CGMCT_WORKER_H_
#include "kernel_operator.h"
#include "../cgmct/kernel/kernel_grouped_matmul.h"
#include "problem.h"

namespace MulticoreBf16 {
template <class LayoutA, class LayoutB>
__aicore__ inline void RunCgmct(const Problem& problem, GM_ADDR a, GM_ADDR b, GM_ADDR c,
                               TCubeTiling& tiling, uint32_t logical_cube, uint32_t cubes) {
  using namespace Cgmct::Gemm;
  using Shape = AscendC::Shape<_0, _0, _0>;
  using Builder = Block::BlockGroupedMatmulBuilder<bfloat16_t, LayoutA, bfloat16_t, LayoutB,
      bfloat16_t, layout::RowMajor, float, layout::RowMajor, Shape, Shape,
      GroupedMatmulAswtScheduler, MatmulMultiBlockBias<>>;
  using Kernel = Kernel::KernelGroupedMatmul<MatmulShape, Builder,
      Block::BlockEpilogueEmpty, GroupedMatmulAswtScheduler>;
  tiling.M = problem.m;
  tiling.N = problem.n;
  tiling.Ka = tiling.Kb = problem.k;
  tiling.singleCoreM = problem.m;
  tiling.singleCoreN = problem.n;
  tiling.singleCoreK = problem.k;
  const auto tail = SplitTail(problem.m, problem.n, tiling.baseM, tiling.baseN, cubes);
  typename Kernel::GMMTiling group{1, 0, 0, 1, 1, 1, 0, tail.m, tail.n, 0};
  group.matmulTiling = &tiling;
  typename Kernel::Params params{{1, 1, 1, 1}, {a, b, c, nullptr, nullptr}, {}, group};
  AscendC::TPipe pipe;
  Kernel kernel;
  kernel.problemShape_ = {static_cast<int64_t>(problem.m), static_cast<int64_t>(problem.n),
                         static_cast<int64_t>(problem.k), 1};
  kernel.blockNum_ = cubes;
  kernel.aGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(a));
  kernel.bGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(b));
  kernel.cGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(c));
  typename Kernel::BlockMmadOp block;
  block.SetSubBlockIdx(0);
  block.Init(&tiling, &pipe);
  kernel.IterateOneGroup(params, block, logical_cube, 0);
  block.End();
  AscendC::PipeBarrier<PIPE_ALL>();
}

template <class Task>
__aicore__ inline void ExecuteCgmctTask(const Task& task, GM_ADDR* inputs, uint32_t cubes) {
  if ASCEND_IS_AIV { return; }
  const uint32_t expert = task.task_index / cubes;
  AscendC::GlobalTensor<int64_t> groups;
  groups.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(inputs[task.inputs[2].input_position]));
  const uint64_t start = expert == 0 ? 0 : groups.GetValue(expert - 1);
  const uint64_t rows = groups.GetValue(expert) - start;
  if (rows == 0) return;
  AscendC::AscendCUtils::SetOverflow(1);
  GET_TILING_DATA_MEMBER(GMMTilingData, mmTilingData, tiling, inputs[task.tiling_data_position]);
  const bool trans_a = task.inputs[0].transpose_flag == 1;
  const bool trans_b = task.inputs[1].transpose_flag == 1;
  const auto problem = MakeProblem(tiling.M, tiling.N, tiling.Ka, expert, start, rows, trans_a);
  GM_ADDR a = inputs[task.inputs[0].input_position] + problem.a * sizeof(bfloat16_t);
  GM_ADDR b = inputs[task.inputs[1].input_position] + problem.b * sizeof(bfloat16_t);
  GM_ADDR c = inputs[task.outputs[0].input_position] + problem.c * sizeof(bfloat16_t);
  if (trans_a) {
    RunCgmct<Cgmct::Gemm::layout::ColumnMajor, Cgmct::Gemm::layout::RowMajor>(
        problem, a, b, c, tiling, task.task_index % cubes, cubes);
  } else if (trans_b) {
    RunCgmct<Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::ColumnMajor>(
        problem, a, b, c, tiling, task.task_index % cubes, cubes);
  } else {
    RunCgmct<Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::RowMajor>(
        problem, a, b, c, tiling, task.task_index % cubes, cubes);
  }
}
}  // namespace MulticoreBf16
#endif
