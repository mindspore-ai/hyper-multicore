// Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#ifndef HYPER_MULTICORE_BF16_GMM_BLAZE_WORKER_H_
#define HYPER_MULTICORE_BF16_GMM_BLAZE_WORKER_H_
#include "blaze/gemm/block/block_mmad_matmul_basic.h"
#include "blaze/gemm/block/block_scheduler_grouped_matmul.h"
#include "logical_lane.h"
#include "problem.h"

namespace MulticoreBf16 {
template <class LayoutA, class LayoutB, bool RotateLanes = false>
__aicore__ inline void RunBlaze(const Problem& problem, GM_ADDR a, GM_ADDR b, GM_ADDR c,
                               const TCubeTiling& tiling, uint32_t logical_cube, uint32_t cubes,
                               uint32_t expert = 0) {
  using namespace Blaze::Gemm;
  using RowMajor = asc::te::nd_ext_layout_ptn;
  using Policy = MatmulMultiBlockBasic<0, 0, KernelGroupedMmadNoQuant, 0, MatmulOutputMode::OVERWRITE>;
  using Mmad = Block::BlockMmad<Policy, bfloat16_t, LayoutA, bfloat16_t, LayoutB,
                               bfloat16_t, RowMajor, float, RowMajor>;
  const uint64_t k_l1 = tiling.baseK * Blaze::Gemm::Max(Blaze::Gemm::Min(tiling.stepKa, tiling.stepKb), 1);
  typename Mmad::Params params{a, b, c, nullptr, nullptr, nullptr,
      static_cast<uint64_t>(tiling.baseM), static_cast<uint64_t>(tiling.baseN), k_l1,
      static_cast<uint32_t>(tiling.baseM), static_cast<uint32_t>(tiling.baseN),
      static_cast<uint32_t>(tiling.baseK), 2, static_cast<uint16_t>(tiling.dbL0C), nullptr, 0};
  const auto tail = SplitTail(problem.m, problem.n, tiling.baseM, tiling.baseN, cubes);
  typename Block::BlockSchedulerGmmNoQuant::Params schedule_params{
      tiling.baseM, tiling.baseN, tail.m, tail.n, 1, 1, -1, 1, static_cast<int64_t>(problem.m),
      true, true, true, Mmad::TRANS_B, false, sizeof(bfloat16_t)};
  Block::BlockSchedulerGmmNoQuant schedule(schedule_params);
  typename Mmad::TupleShape shape{static_cast<int64_t>(problem.m), static_cast<int64_t>(problem.n),
                                 static_cast<int64_t>(problem.k), 1};
  schedule.UpdateNextGroup(shape);
  if constexpr (RotateLanes) {
    logical_cube = RotateLogicalCube(logical_cube, expert, schedule.GetBlockNums(), cubes);
  }
  using MakeA = asc::te::frame_layout_format<LayoutA, asc::te::layout_trait_default<bfloat16_t>>;
  using MakeB = asc::te::frame_layout_format<LayoutB, asc::te::layout_trait_default<bfloat16_t>>;
  using MakeC = asc::te::frame_layout_format<RowMajor, asc::te::layout_trait_default<bfloat16_t>>;
  using MakeBias = asc::te::frame_layout_format<RowMajor, asc::te::layout_trait_default<float>>;
  auto gm_a = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(
      reinterpret_cast<__gm__ bfloat16_t*>(a)), MakeA{}(problem.m, problem.k));
  auto gm_b = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(
      reinterpret_cast<__gm__ bfloat16_t*>(b)), MakeB{}(problem.k, problem.n));
  auto gm_c = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(
      reinterpret_cast<__gm__ bfloat16_t*>(c)), MakeC{}(problem.m, problem.n));
  auto gm_bias = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(
      static_cast<__gm__ float*>(nullptr)), MakeBias{}(int64_t{1}, static_cast<int64_t>(problem.n)));
  AscendC::TPipe pipe;
  Mmad block;
  block.Init(params);
  for (int64_t tile = logical_cube; tile < schedule.GetBlockNums(); tile += cubes) {
    auto block_shape = schedule.template GetBlockShape<Mmad::TRANS_B, bfloat16_t>(tile);
    auto coordinate = schedule.GetBlockCoord(tile);
    const int64_t m = asc::te::get<0>(block_shape), n = asc::te::get<1>(block_shape);
    const int64_t m_offset = asc::te::get<0>(coordinate), n_offset = asc::te::get<1>(coordinate);
    auto block_a = gm_a.slice(asc::te::make_coord(m_offset, int64_t{0}),
                              asc::te::make_shape(m, static_cast<int64_t>(problem.k)));
    auto block_b = gm_b.slice(asc::te::make_coord(int64_t{0}, n_offset),
                              asc::te::make_shape(static_cast<int64_t>(problem.k), n));
    auto block_c = gm_c.slice(asc::te::make_coord(m_offset, n_offset), asc::te::make_shape(m, n));
    auto block_bias = gm_bias.slice(asc::te::make_coord(int64_t{0}, n_offset), asc::te::make_shape(int64_t{1}, n));
    block(block_a, block_b, block_bias, block_c, block_shape);
  }
  AscendC::PipeBarrier<PIPE_ALL>();
}

template <bool RotateLanes = false, class Task>
__aicore__ inline void ExecuteBlazeTask(const Task& task, GM_ADDR* inputs, uint32_t cubes) {
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
    RunBlaze<asc::te::dn_ext_layout_ptn, asc::te::nd_ext_layout_ptn, RotateLanes>(
        problem, a, b, c, tiling, task.task_index % cubes, cubes, expert);
  } else if (trans_b) {
    RunBlaze<asc::te::nd_ext_layout_ptn, asc::te::dn_ext_layout_ptn, RotateLanes>(
        problem, a, b, c, tiling, task.task_index % cubes, cubes, expert);
  } else {
    RunBlaze<asc::te::nd_ext_layout_ptn, asc::te::nd_ext_layout_ptn, RotateLanes>(
        problem, a, b, c, tiling, task.task_index % cubes, cubes, expert);
  }
}
}  // namespace MulticoreBf16
#endif
