// Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#ifndef HYPER_MULTICORE_MX_TASK_GRAPH_WORKER_H
#define HYPER_MULTICORE_MX_TASK_GRAPH_WORKER_H
#include "kernel_operator.h"
#include "mxfp8/mx_launch_descriptor.h"
#include "mxfp8/mx_byte_tensor_list.h"
#include "runtime/worker_kernel.h"
#ifdef __DAV_CUBE__
#include "mxfp8/mx_gmm_worker.h"
#else
#include "mxfp8_quant/mx_quant_worker.h"
#define DTYPE_DISPATCH_TARGET bfloat16_t
#include "put_mem_signal/put_mem_signal_kernel.cpp"
#endif

using namespace AscendC;
namespace MulticoreRuntime {
template <VectorWorkerPolicy Policy>
class MxTaskWorker : public KernelWorkerBase<MxTaskWorker<Policy>, Policy,
                                           RuntimeStorageBoundsPolicy::NATIVE_TILING> {
 public:
  static constexpr uint32_t EVENT_IDX = 65;
  static constexpr uint32_t DIAGNOSTIC_IDX = 66;
  static constexpr uint32_t TILING_IDX = 67;
  __aicore__ inline void ExecuteComputeKernel(TaskDesc task) {
    auto launch = reinterpret_cast<__gm__ MulticoreMx::MxLaunchDescriptor*>(this->input_list[TILING_IDX]);
    if (task.num_inputs > 4 || task.num_outputs > 4) Trap();
    for (uint32_t i=0;i<task.num_inputs;++i) {
      if (task.inputs[i].input_position >= launch->buffer_count) Trap();
    }
    for (uint32_t i=0;i<task.num_outputs;++i) {
      if (task.outputs[i].input_position >= launch->buffer_count) Trap();
    }
    switch (static_cast<uint32_t>(task.task_type)) {
      case 0:
      case 10: break;
#ifdef __DAV_CUBE__
      case 142: MulticoreMx::ExecuteMxGmmTask(task, this->input_list, launch); break;
#else
#ifdef HYPER_MULTICORE_MX_ROW_ONLY
      case 147:
#endif
      case 143:
      case 145:
      case 146: MulticoreMx::ExecuteMxQuantTask(task, this->input_list); break;
      case 153: MulticoreMx::ExecuteMxQuantTask(task, this->input_list); break;
      case 105: ExecuteShmemPutMem(task); break;
      case 141: this->template ExecuteShmemGetMem<bfloat16_t, true>(task); break;
#endif
      default: Trap();
    }
  }
 private:
#ifndef __DAV_CUBE__
  __aicore__ inline void ExecuteShmemPutMem(TaskDesc task_desc) {
    GM_ADDR signal = this->input_list[EVENT_IDX];

    int64_t ep = getExtraValueFromTiling(this->input_list[TILING_IDX], 1);
    int64_t expert_num = getExtraValueFromTiling(this->input_list[TILING_IDX], 2);
    int64_t expert_num_single_rank = expert_num / ep;
    int64_t hidden_size = getExtraValueFromTiling(this->input_list[TILING_IDX], 3);

    int64_t rank_id = (*(__gm__ int64_t *)(this->input_list[TILING_IDX])) / ep * ep;
    int64_t single_expert_task_num = task_desc.task_split_num / expert_num;
    int64_t target_pe = rank_id + task_desc.task_index / (expert_num_single_rank * single_expert_task_num);

    int64_t target_offset_ = *((__gm__ int64_t *)(this->input_list[task_desc.inputs[0].input_position] +
                                                  task_desc.inputs[0].base_ptr_offset * task_desc.inputs[0].data_type));
    int64_t src_offset_ = *((__gm__ int64_t *)(this->input_list[task_desc.inputs[2].input_position] +
                                               task_desc.inputs[2].base_ptr_offset * task_desc.inputs[2].data_type));
    int32_t size_ = *((__gm__ int32_t *)(this->input_list[task_desc.inputs[3].input_position] +
                                         task_desc.inputs[3].base_ptr_offset * task_desc.inputs[3].data_type));
    int32_t send_data_size_ = task_desc.task_split_value * hidden_size;

    int32_t value = task_desc.task_index % single_expert_task_num;
    int32_t start = value * task_desc.task_split_value * hidden_size;
    int32_t end = (value + 1) * task_desc.task_split_value * hidden_size;
    if (start >= size_) {
      const uint32_t credits = task_desc.extra_value_1 == 1
        ? EmptyFragmentCredits(size_, send_data_size_, single_expert_task_num, value) : 1;
      if (credits != 0) {
        aclshmemx_signal_op(reinterpret_cast<__gm__ int32_t *>(signal) + task_desc.trigger_event,
                            credits, ACLSHMEM_SIGNAL_ADD, target_pe);
      }
      return;
    } else {
      if (end > size_) {
        send_data_size_ = size_ - start;
      }
    }
    target_offset_ = target_offset_ + static_cast<int64_t>(start);
    src_offset_ = src_offset_ + static_cast<int64_t>(start);
    send_data_size_ = static_cast<int64_t>(send_data_size_);

    put_mem_signal_kernel(this->input_list[task_desc.outputs[0].input_position] +
                            task_desc.outputs[0].base_ptr_offset * task_desc.outputs[0].data_type,
                          target_offset_,
                          this->input_list[task_desc.inputs[1].input_position] +
                            task_desc.inputs[1].base_ptr_offset * task_desc.inputs[1].data_type,
                          src_offset_, send_data_size_, signal, static_cast<int64_t>(task_desc.trigger_event), 1,
                          nullptr, 1, target_pe, false);
  }
#endif
};
}
#include "mx_planned_worker.h"
template <MulticoreRuntime::VectorWorkerPolicy Policy>
__aicore__ inline void RunMxTaskGraph(GM_ADDR buffers, GM_ADDR runtime, GM_ADDR events,
                                    GM_ADDR diagnostic, GM_ADDR launch_metadata) {
  auto launch = reinterpret_cast<__gm__ MulticoreMx::MxLaunchDescriptor*>(launch_metadata);
  const bool roles = launch->version == 6 && launch->reserved == 5;
  if (launch->magic != MulticoreMx::MX_LAUNCH_MAGIC ||
      (!roles && (launch->version != 1 || launch->reserved != 0)) ||
      launch->buffer_count == 0 || launch->buffer_count > 64 ||
      !MulticoreMx::IsMxLaunchTopologyValid(launch->ep, launch->experts)) Trap();
  auto list_words = reinterpret_cast<__gm__ uint64_t*>(buffers);
  const uint64_t pointer_offset = list_words[0];
  if (!MulticoreMx::IsMxByteListHeaderValid(pointer_offset, launch->buffer_count)) Trap();
  auto data_pointers = list_words + pointer_offset / sizeof(uint64_t);
  GM_ADDR inputs[68] = {};
  for (uint32_t i=0;i<launch->buffer_count;++i) {
    if (!MulticoreMx::IsMxByteVectorShapeValid(
          list_words[1 + 2 * i], list_words[2 + 2 * i], launch->buffer_bytes[i])) Trap();
    inputs[i] = reinterpret_cast<GM_ADDR>(data_pointers[i]);
  }
  inputs[64] = runtime;
  inputs[65] = events;
  inputs[66] = diagnostic;
  inputs[67] = launch_metadata;
  if (roles) {
#if __NPU_ARCH__ != 3510
    Trap();
#endif
    MulticoreRuntime::ReadyHandshakeMeta meta;
    MulticoreRuntime::getReadyHandshakeMeta(runtime, &meta);
    const uint32_t protocol = *reinterpret_cast<__gm__ uint32_t*>(runtime +
        offsetof(MulticoreRuntime::RuntimeHeader, protocol_version));
    const bool pull = meta.completion_event != 0;
    if (launch->cube_count != 32 || MulticoreRuntime::getNumWorkers(runtime) != 64 ||
        protocol != (pull ? MulticoreRuntime::RUNTIME_PROTOCOL_VERSION : 0) ||
        (pull && launch->version != 6)) Trap();
    if (*reinterpret_cast<__gm__ int32_t*>(runtime + MulticoreRuntime::getTaskIndexNumOffset(runtime) +
        3 * sizeof(int32_t)) != -1) Trap();
    MulticoreRuntime::MxPlannedWorker worker;
    worker.Init(GetBlockIdx(), runtime, inputs);
    worker.ProcessPlanned();
    return;
  }
  if constexpr (Policy != MulticoreRuntime::VectorWorkerPolicy::LEGACY_ODD) {
#if __NPU_ARCH__ != 3510
    Trap();
#endif
    MulticoreRuntime::ReadyHandshakeMeta meta;
    MulticoreRuntime::getReadyHandshakeMeta(runtime, &meta);
    if (meta.completion_event != 0 ||
        *reinterpret_cast<__gm__ uint32_t*>(runtime +
            offsetof(MulticoreRuntime::RuntimeHeader, protocol_version)) != 0) Trap();
  }
  if constexpr (Policy == MulticoreRuntime::VectorWorkerPolicy::SINGLE_VECTOR) {
    if (launch->cube_count != 32 || MulticoreRuntime::getNumWorkers(runtime) != 64) Trap();
#ifndef __DAV_CUBE__
    if (GetSubBlockNum() != 1) Trap();
#endif
  }
  MulticoreRuntime::MxTaskWorker<Policy> worker;
  worker.Init(GetBlockIdx(), runtime, inputs);
  worker.Process();
}
#endif
