/* Copyright 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * Licensed under the Apache License, Version 2.0. */
#pragma once
#include "data_plane/rma.h"
#include "runtime/runtime_config.hpp"

namespace MulticoreMoE {
/** Translate expert routing descriptors; transport mechanisms remain in SHMEM. */
template <typename T>
__aicore__ inline void ExecuteTransfer(const MulticoreRuntime::TaskDesc &task, GM_ADDR *inputs,
                                      GM_ADDR metadata, GM_ADDR events) {
  using namespace MulticoreRuntime;
  namespace plane = hyper_multicore::shmem::data_plane;
  auto topology = reinterpret_cast<__gm__ int64_t *>(metadata);
  const int64_t ep = topology[1], experts = topology[2], hidden = topology[3];
  if (ep <= 0 || experts <= 0 || experts % ep || hidden <= 0 || task.task_split_num % experts) AscendC::Trap();
  const int64_t fragments = task.task_split_num / experts;
  if (fragments <= 0) AscendC::Trap();
  const int32_t peer = task.task_index / ((experts / ep) * fragments);
  if (peer >= ep || task.extra_value_2 > 1) AscendC::Trap();
  const auto engine = static_cast<plane::TransferEngine>(task.extra_value_2);
  const bool pull = task.task_type == TaskType::TASK_SHMEM_GET_MEM;
  auto sizes = reinterpret_cast<__gm__ int32_t *>(inputs[task.inputs[3].input_position]);
  const int64_t total = sizes[task.inputs[3].base_ptr_offset];
  if (total < 0) AscendC::Trap();
  const uint64_t extent = static_cast<uint64_t>(task.task_split_value) * hidden;
  if (extent == 0) AscendC::Trap();
  const uint64_t tile = task.task_index % fragments;
  const uint64_t valid_tiles = total / extent + (total % extent != 0);
  if (tile < valid_tiles) {
    const uint64_t first = tile * extent;
    const int64_t count = total - first < extent ? total - first : extent;
    auto destination_offsets = reinterpret_cast<__gm__ int64_t *>(inputs[task.inputs[0].input_position]);
    auto source_offsets = reinterpret_cast<__gm__ int64_t *>(inputs[task.inputs[2].input_position]);
    const int64_t destination = destination_offsets[task.inputs[0].base_ptr_offset];
    const int64_t source = source_offsets[task.inputs[2].base_ptr_offset];
    const uint64_t limit = ~uint64_t{0} / sizeof(T);
    if (first > limit || source < 0 || destination < 0 || static_cast<uint64_t>(source) > limit - first ||
        static_cast<uint64_t>(destination) > limit - first) AscendC::Trap();
    auto dst = inputs[task.outputs[0].input_position] + (destination + first) * sizeof(T);
    auto src = inputs[task.inputs[1].input_position] + (source + first) * sizeof(T);
    if (pull) plane::get_complete<T>(dst, src, count, peer, engine);
    else plane::put_complete<T>(dst, src, count, peer, engine);
  }
  // Empty Push fragments still satisfy the original multi-source ADD contract.
  // Pull completion is published locally by the runtime after this function returns.
  if (!pull) plane::signal(reinterpret_cast<__gm__ int32_t *>(events) + task.trigger_event,
                           1, plane::SignalOp::Add, peer);
}
}  // namespace MulticoreMoE
