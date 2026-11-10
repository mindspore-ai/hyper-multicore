// Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#ifndef HYPER_MULTICORE_MX_GMM_WORKER_H_
#define HYPER_MULTICORE_MX_GMM_WORKER_H_

#include "kernel_operator.h"
#include "mx_gmm_descriptor.h"
#include "mx_launch_descriptor.h"
#include "mx_a5_policy.h"
#include "mx_blaze/gemm/block/block_mmad_qgmm_mx.h"
#include "mx_blaze/gemm/kernel/kernel_qgmm_mx.h"

namespace MulticoreMx {

template <class LayoutA, class LayoutB, class Output>
__aicore__ inline void ExecuteMxGmmMatrix(const MxGmmDescriptor& r, GM_ADDR x, GM_ADDR w,
                                        GM_ADDR xs, GM_ADDR ws, GM_ADDR y) {
  using namespace Blaze::Gemm;
  using RowMajor = asc::te::nd_ext_layout_ptn;
  using Policy = GroupedMatmulWithScaleMx<0, false, KernelGroupedMmadWithScaleMx>;
  using Mmad = Block::BlockMmad<Policy, fp8_e4m3fn_t, LayoutA, fp8_e4m3fn_t, LayoutB,
                                Output, RowMajor, float, RowMajor>;
  using Shape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
  using MxKernel = Kernel::GemmUniversal<Shape, Mmad, Blaze::Epilogue::Block::BlockEpilogueEmpty,
                                        Block::BlockSchedulerGmmSwatWithTailSplit>;
  const auto resource = SelectMxA5Resources(r);
  if (resource.l1_bytes > 524288) AscendC::Trap();
  typename MxKernel::GMMTiling t{1, r.m, r.n, r.k, r.base_m, r.base_n, r.base_k,
      r.k_al1, r.k_bl1, resource.scale_k, resource.scale_k, 0, static_cast<uint8_t>(r.db_l0c),
      resource.stages, static_cast<int8_t>(r.layout == static_cast<uint32_t>(MxLayout::TN) ? 2 : 0), 0, 1};
  typename MxKernel::Params params{{r.m, r.n, r.k, 1}, {x, w, y, nullptr, xs, ws}, {}, nullptr, t};
  // Preserve the original task-local pipe and drain before publishing task completion.
  AscendC::TPipe pipe;
  MxKernel kernel;
  kernel.RunSingleExpert(params, r.cube_index, r.cube_count, r.group_list_offset / sizeof(int64_t));
}

template <class Output>
__aicore__ inline void ExecuteMxGmmLayout(const MxGmmDescriptor& r, GM_ADDR x, GM_ADDR w,
                                        GM_ADDR xs, GM_ADDR ws, GM_ADDR y) {
  if (r.layout == static_cast<uint32_t>(MxLayout::NN)) {
    ExecuteMxGmmMatrix<asc::te::nd_ext_layout_ptn, asc::te::nd_ext_layout_ptn, Output>(r, x, w, xs, ws, y);
  } else if (r.layout == static_cast<uint32_t>(MxLayout::NT)) {
    ExecuteMxGmmMatrix<asc::te::nd_ext_layout_ptn, asc::te::dn_ext_layout_ptn, Output>(r, x, w, xs, ws, y);
  } else {
    ExecuteMxGmmMatrix<asc::te::dn_ext_layout_ptn, asc::te::nd_ext_layout_ptn, Output>(r, x, w, xs, ws, y);
  }
}

__aicore__ inline void ZeroMxWeightGradient(const MxGmmDescriptor& r, GM_ADDR y) {
  const uint64_t words = static_cast<uint64_t>(r.m) * r.n *
      (r.output_dtype == static_cast<uint32_t>(MxOutput::BF16) ? 2 : 4) / sizeof(uint32_t);
  AscendC::GlobalTensor<uint32_t> output;
  output.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(y), words);
  // One logical worker owns each complete cache line, including its write-back.
  for (uint64_t line = r.cube_index; line < (words + 15) / 16; line += r.cube_count) {
    for (uint64_t i = line * 16; i < (line + 1) * 16 && i < words; ++i) {
      output.SetValue(i, 0);
    }
    AscendC::DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                     AscendC::DcciDst::CACHELINE_OUT>(output[line * 16]);
  }
}

__aicore__ inline void ExecuteMxGmm(const MxGmmDescriptor& r, GM_ADDR x, GM_ADDR w,
                                   GM_ADDR xs, GM_ADDR ws, GM_ADDR y) {
  if (r.m == 0 || r.k == 0) {
    if (r.layout == static_cast<uint32_t>(MxLayout::TN)) {
      ZeroMxWeightGradient(r, y);
    }
    return;
  }
  if (r.output_dtype == static_cast<uint32_t>(MxOutput::BF16)) {
    ExecuteMxGmmLayout<bfloat16_t>(r, x, w, xs, ws, y);
  } else {
    ExecuteMxGmmLayout<float>(r, x, w, xs, ws, y);
  }
}

// The task caller supplies current physical storage; descriptors never retain pointers.
__aicore__ inline void CheckMxRange(uint64_t offset, uint64_t bytes, uint64_t capacity) {
  if (offset > capacity || bytes > capacity - offset) AscendC::Trap();
}

__aicore__ inline MxGmmDescriptor LoadMxGmmDescriptor(const __gm__ MxGmmDescriptor* source) {
  MxGmmDescriptor r;
  r.magic = source->magic;
  r.version = source->version;
  r.size = source->size;
  r.layout = source->layout;
  r.flags = source->flags;
  r.x_scale_position = source->x_scale_position;
  r.w_scale_position = source->w_scale_position;
  r.cube_index = source->cube_index;
  r.cube_count = source->cube_count;
  r.m = source->m;
  r.n = source->n;
  r.k = source->k;
  r.base_m = source->base_m;
  r.base_n = source->base_n;
  r.base_k = source->base_k;
  r.k_al1 = source->k_al1;
  r.k_bl1 = source->k_bl1;
  r.scale_k = source->scale_k;
  r.db_l0c = source->db_l0c;
  r.output_dtype = source->output_dtype;
  r.x_offset = source->x_offset;
  r.w_offset = source->w_offset;
  r.y_offset = source->y_offset;
  r.x_scale_offset = source->x_scale_offset;
  r.w_scale_offset = source->w_scale_offset;
  r.group_list_offset = source->group_list_offset;
  return r;
}

template <class Task>
__aicore__ inline void ExecuteMxGmmTask(const Task& task, GM_ADDR* inputs,
                                       const __gm__ MxLaunchDescriptor* launch) {
  CheckMxRange(static_cast<uint64_t>(task.extra_value_0) * sizeof(MxGmmDescriptor),
               sizeof(MxGmmDescriptor), launch->buffer_bytes[task.inputs[3].input_position]);
  auto table = reinterpret_cast<__gm__ MxGmmDescriptor*>(inputs[task.inputs[3].input_position]);
  MxGmmDescriptor r = LoadMxGmmDescriptor(table + task.extra_value_0);
  if (!IsMxGmmDescriptorValid(r) || r.x_scale_position >= launch->buffer_count ||
      r.w_scale_position >= launch->buffer_count || (r.flags & ~MX_GMM_DYNAMIC_GROUP) != 0) {
    AscendC::Trap();
  }
  if (r.flags & MX_GMM_DYNAMIC_GROUP) {
    CheckMxRange(r.group_list_offset, sizeof(int64_t),
                 launch->buffer_bytes[task.inputs[2].input_position]);
    auto groups = reinterpret_cast<__gm__ int64_t*>(inputs[task.inputs[2].input_position]);
    const uint64_t expert = r.group_list_offset / sizeof(int64_t);
    const uint64_t end = groups[expert];
    const uint64_t start = expert == 0 ? 0 : groups[expert - 1];
    const uint64_t maximum = r.layout == static_cast<uint32_t>(MxLayout::TN) ? r.k : r.m;
    if (end < start || end - start > maximum) AscendC::Trap();
    ResolveMxGmmGroup(r, start, end);
  }
  const uint64_t m = r.m, n = r.n, k = r.k;
  const uint64_t output_bytes = r.output_dtype == static_cast<uint32_t>(MxOutput::BF16) ? 2 : 4;
  CheckMxRange(r.x_offset, m * k, launch->buffer_bytes[task.inputs[0].input_position]);
  CheckMxRange(r.w_offset, n * k, launch->buffer_bytes[task.inputs[1].input_position]);
  CheckMxRange(r.y_offset, m * n * output_bytes, launch->buffer_bytes[task.outputs[0].input_position]);
  CheckMxRange(r.x_scale_offset, m * ((k + 63) / 64) * 2, launch->buffer_bytes[r.x_scale_position]);
  CheckMxRange(r.w_scale_offset, n * ((k + 63) / 64) * 2, launch->buffer_bytes[r.w_scale_position]);
  ExecuteMxGmm(r, inputs[task.inputs[0].input_position] + r.x_offset,
               inputs[task.inputs[1].input_position] + r.w_offset,
               inputs[r.x_scale_position] + r.x_scale_offset,
               inputs[r.w_scale_position] + r.w_scale_offset,
               inputs[task.outputs[0].input_position] + r.y_offset);
}
}  // namespace MulticoreMx
#endif
