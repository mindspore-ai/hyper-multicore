/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file worker_kernel.cpp
 * @brief MoE-FFN backward — op-specific compute kernels + KernelWorker specialization.
 */

#include <cstddef>

#include "kernel_operator.h"
#include "swi_glu/swi_glu.cpp"
#include "swi_glu_grad/swi_glu_grad.cpp"
#include "grouped_matmul/grouped_matmul.cpp"
#include "put_mem_signal/put_mem_signal_kernel.cpp"
#include "runtime/worker_kernel.h"

using namespace AscendC;

namespace MulticoreRuntime {

class KernelWorker : public KernelWorkerBase<KernelWorker> {
 public:
  // input_list layout for hyper_mega_moe_grad (backward):
  //   [33] = tiling params  [34] = all_event_counters  [35] = diagnostic_buffer
  //   [25] = gmm_workspace  [26] = swi_glu_grad_workspace
  static constexpr uint32_t TILING_IDX              = 33;
  static constexpr uint32_t EVENT_IDX               = 34;
  static constexpr uint32_t DIAGNOSTIC_IDX          = 35;
  static constexpr uint32_t SHARED_PTRS_IDX = 36;
  static constexpr uint32_t WORKSPACE_IDX           = 25;
  static constexpr uint32_t SWIGLU_GRAD_WORKSPACE_IDX = 26;
  static constexpr uint32_t SWIGLU_DYNAMIC_FIELDS_OFFSET =
    static_cast<uint32_t>(offsetof(SwiGluTilingData, rowLen));
  static constexpr int64_t SWIGLU_DYNAMIC_FIELDS_BYTES =
    offsetof(SwiGluTilingData, baseRowLen) + sizeof(uint32_t) - SWIGLU_DYNAMIC_FIELDS_OFFSET;
  static constexpr uint32_t GMM_BASE_M_OFFSET = static_cast<uint32_t>(
    offsetof(GMMTilingData, gmmBaseParams) + offsetof(GMMBaseParams, m));
  static constexpr uint32_t GMM_MATMUL_M_OFFSET = static_cast<uint32_t>(
    offsetof(GMMTilingData, mmTilingData) + offsetof(TCubeTiling, M));
  static constexpr int64_t GMM_MATMUL_M_FIELDS_BYTES =
    offsetof(TCubeTiling, singleCoreM) + sizeof(int32_t) - offsetof(TCubeTiling, M);
  static constexpr uint32_t GMM_MATMUL_K_OFFSET = static_cast<uint32_t>(
    offsetof(GMMTilingData, mmTilingData) + offsetof(TCubeTiling, Ka));
  static constexpr int64_t GMM_MATMUL_K_FIELDS_BYTES =
    offsetof(TCubeTiling, singleCoreK) + sizeof(int32_t) - offsetof(TCubeTiling, Ka);

  __aicore__ inline void CaptureL1TaskTensors(TaskId task_id, uint32_t phase) {
    MulticoreRuntime::CaptureL1TaskTensors<35, 12>(
      input_list[DIAGNOSTIC_IDX], input_list, task_id, phase, worker_id_, input_list[SHARED_PTRS_IDX]);
  }

  __aicore__ inline void ExecuteComputeKernel(TaskDesc task_desc) {
    switch (task_desc.task_type) {
      case TaskType::TASK_BEGIN_TASK_GRAPH:
        break;
      case TaskType::TASK_MATMUL:
        ExecuteMatmul(task_desc);
        break;
      case TaskType::TASK_GROUPED_MATMUL:
        ExecuteGroupedMatmul(task_desc);
        break;
#ifndef __DAV_C220_CUBE__
      case TaskType::TASK_ADD_CUSTOM:
        ExecuteSharedResidualAdd(task_desc);
        break;
#endif
      case TaskType::TASK_SHMEM_PUT_MEM_SIGNAL:
        ExecuteShmemPutMem(task_desc);
        break;
      case TaskType::TASK_SHMEM_GET_MEM:
        ExecuteShmemGetMem<DTYPE_DISPATCH_TARGET>(task_desc);
        break;
      case TaskType::TASK_SWI_GLU_GRAD:
        ExecuteSwiGluGrad(task_desc);
        break;
      case TaskType::TASK_UNPERMUTE_REDUCE_GRAD:
        ExecuteUnpermuteReduceGrad(task_desc);
        break;
#ifndef __DAV_C220_CUBE__
      case TaskType::TASK_WEIGHTED_UNPERMUTE:
        if (task_desc.extra_value_1 != 0) {
          AcquireRemoteWrites();
        }
        break;
      case TaskType::TASK_TOPK_WEIGHT_GRAD:
        ExecuteTopKWeightGrad(task_desc);
        break;
#endif
      default:
        break;
    }
  }

 private:
  __aicore__ inline float GetSwiGluClampLimit(const TaskDesc &task_desc) {
    union {
      uint32_t bits;
      float value;
    } encoded = {};
    encoded.bits = task_desc.extra_value_0;
    return encoded.value;
  }

  __aicore__ inline bool getTransposeData(uint32_t data) { return data == 1; }

  __aicore__ inline void ExecuteMatmul(const TaskDesc &task_desc) {}

#ifndef __DAV_C220_CUBE__
  __aicore__ inline void ExecuteSharedResidualAdd(const TaskDesc &task_desc) {
    if (task_desc.task_split_value == 0) {
      return;
    }
    auto *shared = reinterpret_cast<__gm__ DTYPE_DISPATCH_TARGET *>(SharedTensorAddress(task_desc.inputs[0]));
    auto *total = reinterpret_cast<__gm__ DTYPE_DISPATCH_TARGET *>(SharedTensorAddress(task_desc.inputs[1]));
    auto *state = aclshmemi_get_state();
    uint64_t scratch = state->mte_config.aclshmem_ub + state->mte_config.ub_size;
    constexpr uint32_t chunk = 4096;
    constexpr uint32_t bytes = chunk * sizeof(DTYPE_DISPATCH_TARGET);
    auto *value_ub = reinterpret_cast<__ubuf__ DTYPE_DISPATCH_TARGET *>(scratch);
    auto *output_ub = reinterpret_cast<__ubuf__ DTYPE_DISPATCH_TARGET *>(scratch + bytes);
    auto value = MakeLocalTensorFromUb<DTYPE_DISPATCH_TARGET>(scratch, chunk);
    auto output = MakeLocalTensorFromUb<DTYPE_DISPATCH_TARGET>(scratch + bytes, chunk);
    auto accumulator = MakeLocalTensorFromUb<float>(scratch + 2 * bytes, chunk);
    auto temporary = MakeLocalTensorFromUb<float>(scratch + 2 * bytes + chunk * sizeof(float), chunk);
    uint64_t remaining = static_cast<uint64_t>(task_desc.inputs[0].dim[1]) * task_desc.task_split_value;
    for (uint64_t offset = 0; offset < remaining; offset += chunk) {
      uint32_t elements = remaining - offset < chunk ? remaining - offset : chunk;
      aclshmemi_copy_gm2ub(output_ub, total + offset, elements * sizeof(DTYPE_DISPATCH_TARGET));
      AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
      AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
      AddRoundedSharedResidual(shared + offset, value_ub, value, output, accumulator, temporary, elements);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
      AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
      aclshmemi_copy_ub2gm(total + offset, output_ub, elements * sizeof(DTYPE_DISPATCH_TARGET));
      AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
      AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
    }
  }

  __aicore__ inline void AcquireRemoteWrites() {
    AscendC::GlobalTensor<uint8_t> cache;
    cache.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(input_list[EVENT_IDX]), 1);
    __asm__ __volatile__("");
    AscendC::DataCacheCleanAndInvalid<uint8_t, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(
      cache);
    __asm__ __volatile__("");
  }
#endif

  template <typename T>
  __aicore__ inline LocalTensor<T> MakeLocalTensorFromUb(uint64_t address, uint32_t elements) {
    LocalTensor<T> local;
    TBuffAddr buffer_address;
    buffer_address.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
    buffer_address.bufferAddr = address;
#if ASCENDC_CPU_DEBUG
    buffer_address.dataLen = elements * sizeof(T);
    buffer_address.absAddr = reinterpret_cast<uint8_t *>(address);
#endif
    local.SetAddr(buffer_address);
    local.SetSize(elements);
    return local;
  }

  __aicore__ inline void ExecuteUnpermuteReduceGrad(const TaskDesc &task_desc) {
    int64_t hidden_size = getExtraValueFromTiling(input_list[TILING_IDX], 3);
    int64_t top_k = static_cast<int64_t>(task_desc.extra_value_0);
    int64_t row_begin = static_cast<int64_t>(task_desc.task_index) * task_desc.task_split_value;
    int64_t row_end = row_begin + task_desc.task_split_value;
    int64_t total_rows = task_desc.outputs[0].dim[0];
    if (row_end > total_rows) {
      row_end = total_rows;
    }
    if (row_begin >= row_end) {
      return;
    }

    __gm__ int32_t *route_index = (__gm__ int32_t *)(input_list[task_desc.inputs[1].input_position]);
    __gm__ DTYPE_DISPATCH_TARGET *src =
      (__gm__ DTYPE_DISPATCH_TARGET *)(input_list[task_desc.inputs[0].input_position]);
    __gm__ DTYPE_DISPATCH_TARGET *dst =
      reinterpret_cast<__gm__ DTYPE_DISPATCH_TARGET *>(SharedTensorAddress(task_desc.outputs[0]));
    __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();
    uint64_t reduce_ub = device_state->mte_config.aclshmem_ub + device_state->mte_config.ub_size;
    int64_t chunk_elems = RouteChunkElements(reduce_ub, 4096, 3 * sizeof(DTYPE_DISPATCH_TARGET) + 3 * sizeof(float));
    uint64_t value_bytes = chunk_elems * sizeof(DTYPE_DISPATCH_TARGET);
    __ubuf__ DTYPE_DISPATCH_TARGET *value_ping_ub = reinterpret_cast<__ubuf__ DTYPE_DISPATCH_TARGET *>(reduce_ub);
    __ubuf__ DTYPE_DISPATCH_TARGET *value_pong_ub =
      reinterpret_cast<__ubuf__ DTYPE_DISPATCH_TARGET *>(reduce_ub + value_bytes);
    __ubuf__ DTYPE_DISPATCH_TARGET *output_ub =
      reinterpret_cast<__ubuf__ DTYPE_DISPATCH_TARGET *>(reduce_ub + value_bytes * 2);
    LocalTensor<DTYPE_DISPATCH_TARGET> value_ping =
      MakeLocalTensorFromUb<DTYPE_DISPATCH_TARGET>(reduce_ub, chunk_elems);
    LocalTensor<DTYPE_DISPATCH_TARGET> value_pong =
      MakeLocalTensorFromUb<DTYPE_DISPATCH_TARGET>(reduce_ub + value_bytes, chunk_elems);
    LocalTensor<DTYPE_DISPATCH_TARGET> output =
      MakeLocalTensorFromUb<DTYPE_DISPATCH_TARGET>(reduce_ub + value_bytes * 2, chunk_elems);
    LocalTensor<float> accumulator = MakeLocalTensorFromUb<float>(reduce_ub + value_bytes * 3, chunk_elems);
    LocalTensor<float> temporary_ping =
      MakeLocalTensorFromUb<float>(reduce_ub + value_bytes * 3 + chunk_elems * sizeof(float), chunk_elems);
    LocalTensor<float> temporary_pong =
      MakeLocalTensorFromUb<float>(reduce_ub + value_bytes * 3 + chunk_elems * sizeof(float) * 2, chunk_elems);

    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
    for (int64_t row = row_begin; row < row_end; ++row) {
      for (int64_t col = 0; col < hidden_size; col += chunk_elems) {
        int64_t elements = hidden_size - col < chunk_elems ? hidden_size - col : chunk_elems;
        uint64_t bytes = elements * sizeof(DTYPE_DISPATCH_TARGET);
        int64_t first_row = route_index[row * top_k];
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        aclshmemi_copy_gm2ub(value_ping_ub, src + first_row * hidden_size + col, bytes);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        if (top_k > 1) {
          int64_t second_row = route_index[row * top_k + 1];
          AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
          aclshmemi_copy_gm2ub(value_pong_ub, src + second_row * hidden_size + col, bytes);
          AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        }
        Cast(accumulator, value_ping, RoundMode::CAST_NONE, elements);
        PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        for (int64_t k = 1; k < top_k; ++k) {
          bool pong = (k & 1) != 0;
          AscendC::TEventID event_id = pong ? EVENT_ID1 : EVENT_ID0;
          LocalTensor<DTYPE_DISPATCH_TARGET> value = pong ? value_pong : value_ping;
          LocalTensor<float> temporary = pong ? temporary_pong : temporary_ping;
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event_id);
          if (k + 1 < top_k) {
            bool next_pong = ((k + 1) & 1) != 0;
            AscendC::TEventID next_event_id = next_pong ? EVENT_ID1 : EVENT_ID0;
            __ubuf__ DTYPE_DISPATCH_TARGET *next_value_ub = next_pong ? value_pong_ub : value_ping_ub;
            int64_t next_row = route_index[row * top_k + k + 1];
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(next_event_id);
            aclshmemi_copy_gm2ub(next_value_ub, src + next_row * hidden_size + col, bytes);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(next_event_id);
          }
          Cast(temporary, value, RoundMode::CAST_NONE, elements);
          PipeBarrier<PIPE_V>();
          Add(accumulator, accumulator, temporary, elements);
          PipeBarrier<PIPE_V>();
          AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event_id);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        Cast(output, accumulator, RoundMode::CAST_RINT, elements);
        PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
        aclshmemi_copy_ub2gm(dst + row * hidden_size + col, output_ub, bytes);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
      }
    }
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
  }

#ifndef __DAV_C220_CUBE__
  __aicore__ inline void RunTopKWeightGrad(__gm__ DTYPE_DISPATCH_TARGET *expert_output,
                                           __gm__ DTYPE_DISPATCH_TARGET *grad_output, __gm__ int32_t *route_index,
                                           __gm__ float *grad_topk_weight, int64_t hidden_size, int64_t top_k,
                                           int64_t token_begin, int64_t token_end) {
    // ReduceSum also fetches events through GetTPipePtr on route-only workers.
    TPipe reduce_pipe;
    __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();
    uint64_t dot_ub = device_state->mte_config.aclshmem_ub + device_state->mte_config.ub_size;
    int64_t chunk_elems = RouteChunkElements(dot_ub, 2048, 3 * sizeof(DTYPE_DISPATCH_TARGET) + 3 * sizeof(float));
    uint64_t value_bytes = chunk_elems * sizeof(DTYPE_DISPATCH_TARGET);
    uint64_t float_bytes = chunk_elems * sizeof(float);
    __ubuf__ DTYPE_DISPATCH_TARGET *expert_ping_ub = reinterpret_cast<__ubuf__ DTYPE_DISPATCH_TARGET *>(dot_ub);
    __ubuf__ DTYPE_DISPATCH_TARGET *expert_pong_ub =
      reinterpret_cast<__ubuf__ DTYPE_DISPATCH_TARGET *>(dot_ub + value_bytes);
    __ubuf__ DTYPE_DISPATCH_TARGET *grad_ub =
      reinterpret_cast<__ubuf__ DTYPE_DISPATCH_TARGET *>(dot_ub + value_bytes * 2);
    LocalTensor<DTYPE_DISPATCH_TARGET> expert_ping = MakeLocalTensorFromUb<DTYPE_DISPATCH_TARGET>(dot_ub, chunk_elems);
    LocalTensor<DTYPE_DISPATCH_TARGET> expert_pong =
      MakeLocalTensorFromUb<DTYPE_DISPATCH_TARGET>(dot_ub + value_bytes, chunk_elems);
    LocalTensor<DTYPE_DISPATCH_TARGET> grad_local =
      MakeLocalTensorFromUb<DTYPE_DISPATCH_TARGET>(dot_ub + value_bytes * 2, chunk_elems);
    LocalTensor<float> expert_float = MakeLocalTensorFromUb<float>(dot_ub + value_bytes * 3, chunk_elems);
    LocalTensor<float> grad_float = MakeLocalTensorFromUb<float>(dot_ub + value_bytes * 3 + float_bytes, chunk_elems);
    LocalTensor<float> reduce_work =
      MakeLocalTensorFromUb<float>(dot_ub + value_bytes * 3 + float_bytes * 2, chunk_elems);
    LocalTensor<float> reduce_out = MakeLocalTensorFromUb<float>(dot_ub + value_bytes * 3 + float_bytes * 3, 8);

    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
    for (int64_t token = token_begin; token < token_end; ++token) {
      int32_t routed_rows[16];
      float dots[16];
      for (int64_t k = 0; k < top_k; ++k) {
        routed_rows[k] = route_index[token * top_k + k];
        dots[k] = 0.0f;
      }
      for (int64_t col = 0; col < hidden_size; col += chunk_elems) {
        int64_t elements = hidden_size - col;
        if (elements > chunk_elems) {
          elements = chunk_elems;
        }
        uint64_t bytes = elements * sizeof(DTYPE_DISPATCH_TARGET);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        aclshmemi_copy_gm2ub(grad_ub, grad_output + token * hidden_size + col, bytes);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
        Cast(grad_float, grad_local, RoundMode::CAST_NONE, elements);
        PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        aclshmemi_copy_gm2ub(expert_ping_ub, expert_output + static_cast<int64_t>(routed_rows[0]) * hidden_size + col,
                             bytes);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        for (int64_t k = 0; k < top_k; ++k) {
          bool pong = (k & 1) != 0;
          AscendC::TEventID event_id = pong ? EVENT_ID1 : EVENT_ID0;
          LocalTensor<DTYPE_DISPATCH_TARGET> expert = pong ? expert_pong : expert_ping;
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event_id);
          if (k + 1 < top_k) {
            bool next_pong = ((k + 1) & 1) != 0;
            AscendC::TEventID next_event_id = next_pong ? EVENT_ID1 : EVENT_ID0;
            __ubuf__ DTYPE_DISPATCH_TARGET *next_expert_ub = next_pong ? expert_pong_ub : expert_ping_ub;
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(next_event_id);
            aclshmemi_copy_gm2ub(next_expert_ub,
                                 expert_output + static_cast<int64_t>(routed_rows[k + 1]) * hidden_size + col, bytes);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(next_event_id);
          }
          Cast(expert_float, expert, RoundMode::CAST_NONE, elements);
          PipeBarrier<PIPE_V>();
          Mul(expert_float, expert_float, grad_float, elements);
          PipeBarrier<PIPE_V>();
          ReduceSum(reduce_out, expert_float, reduce_work, elements);
          SyncFunc<AscendC::HardEvent::V_S>();
          dots[k] += reduce_out.GetValue(0);
          AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event_id);
        }
      }
      for (int64_t k = 0; k < top_k; ++k) {
        grad_topk_weight[token * top_k + k] = dots[k];
      }
    }
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
    reduce_pipe.Destroy();
  }

  __aicore__ inline void ExecuteTopKWeightGrad(const TaskDesc &task_desc) {
    int64_t hidden_size = getExtraValueFromTiling(input_list[TILING_IDX], 3);
    int64_t top_k = static_cast<int64_t>(task_desc.extra_value_0);
    int64_t token_begin = static_cast<int64_t>(task_desc.task_index) * task_desc.task_split_value;
    int64_t token_end = token_begin + task_desc.task_split_value;
    int64_t token_count = task_desc.outputs[0].dim[0];
    if (token_end > token_count) {
      token_end = token_count;
    }
    if (token_begin >= token_end) {
      return;
    }
    __gm__ DTYPE_DISPATCH_TARGET *expert_output =
      (__gm__ DTYPE_DISPATCH_TARGET *)(input_list[task_desc.inputs[0].input_position]);
    __gm__ DTYPE_DISPATCH_TARGET *grad_output =
      (__gm__ DTYPE_DISPATCH_TARGET *)(input_list[task_desc.inputs[1].input_position]);
    __gm__ int32_t *route_index = (__gm__ int32_t *)(input_list[task_desc.inputs[2].input_position]);
    __gm__ float *grad_topk_weight = (__gm__ float *)(input_list[task_desc.outputs[0].input_position]);
    RunTopKWeightGrad(expert_output, grad_output, route_index, grad_topk_weight, hidden_size, top_k, token_begin,
                      token_end);
  }
#endif

  __aicore__ inline void cacheWriteThrough(__gm__ uint8_t *sourceAddr, int64_t length) {
    if (length <= 0) {
      return;
    }
    __gm__ uint8_t *start =
      (__gm__ uint8_t *)((int64_t)sourceAddr / AscendC::CACHE_LINE_SIZE * AscendC::CACHE_LINE_SIZE);
    __gm__ uint8_t *end =
      (__gm__ uint8_t *)(((int64_t)sourceAddr + length - 1) / AscendC::CACHE_LINE_SIZE * AscendC::CACHE_LINE_SIZE);
    AscendC::GlobalTensor<uint8_t> global;
    global.SetGlobalBuffer(start);
    for (uint32_t i = 0; i <= end - start; i += AscendC::CACHE_LINE_SIZE) {
      __asm__ __volatile__("");
      AscendC::DataCacheCleanAndInvalid<uint8_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                        AscendC::DcciDst::CACHELINE_OUT>(global[i]);
      __asm__ __volatile__("");
    }
  }

  __aicore__ inline void ExecuteSwiGluGrad(const TaskDesc &task_desc) {
    if (task_desc.inputs[0].tensor_type == 2) {
      if (task_desc.task_split_value == 0) {
        return;
      }
      swi_glu_grad(SharedTensorAddress(task_desc.inputs[0]), SharedTensorAddress(task_desc.inputs[1]),
                   SharedTensorAddress(task_desc.outputs[0]), input_list[SWIGLU_GRAD_WORKSPACE_IDX],
                   input_list[task_desc.tiling_data_position] + task_desc.tiling_data_offset,
                   GetSwiGluClampLimit(task_desc), true);
      return;
    }
    if (task_desc.inputs[0].dynamic_shape == 1 || task_desc.outputs[0].dynamic_shape == 1) {
      DynamicData dynamic_data;
      getDynamicData(this->runtimeConfigPtr, &(dynamic_data));
      if (dynamic_data.dynamic_type == DynamicType::DYNAMIC_DSV3_MOE) {
        GM_ADDR grouped_list = input_list[dynamic_data.dynamic_input_position];
        GlobalTensor<int64_t> grouped_list_tensor;
        int64_t ep = getExtraValueFromTiling(input_list[TILING_IDX], 1);
        int64_t expert_num = getExtraValueFromTiling(input_list[TILING_IDX], 2);
        int64_t expert_num_single_rank = expert_num / ep;
        grouped_list_tensor.SetGlobalBuffer((__gm__ int64_t *)(grouped_list), expert_num_single_rank);

        uint32_t grouped_list_shape = dynamic_data.dynamic_group_size;
        uint32_t per_task_num = task_desc.task_split_num / grouped_list_shape;
        uint32_t data_index = task_desc.task_index / per_task_num;
        uint32_t task_index = task_desc.task_index % per_task_num;
        int64_t start = 0;
        int64_t end = grouped_list_tensor.GetValue(data_index);
        if (data_index != 0) {
          start = grouped_list_tensor.GetValue(data_index - 1);
        }
        int64_t current_seq_start = task_index * task_desc.task_split_value;
        int64_t current_seq_end = (task_index + 1) * task_desc.task_split_value;
        int64_t base_ptr_offset = start + current_seq_start;

        if (base_ptr_offset >= end) {
          return;
        }
        int64_t input_0_offset = task_desc.inputs[0].dynamic_shape == 1
                                   ? base_ptr_offset * task_desc.inputs[0].dim[1] * task_desc.inputs[0].data_type
                                   : task_desc.inputs[0].base_ptr_offset * task_desc.inputs[0].data_type;
        int64_t input_1_offset = task_desc.inputs[1].dynamic_shape == 1
                                   ? base_ptr_offset * task_desc.inputs[1].dim[1] * task_desc.inputs[1].data_type
                                   : task_desc.inputs[1].base_ptr_offset * task_desc.inputs[1].data_type;
        int64_t output_0_offset = task_desc.outputs[0].dynamic_shape == 1
                                    ? base_ptr_offset * task_desc.outputs[0].dim[1] * task_desc.outputs[0].data_type
                                    : task_desc.outputs[0].base_ptr_offset * task_desc.outputs[0].data_type;
        if (start + current_seq_end <= end) {
          swi_glu_grad(input_list[task_desc.inputs[0].input_position] + input_0_offset,
                       input_list[task_desc.inputs[1].input_position] + input_1_offset,
                       input_list[task_desc.outputs[0].input_position] + output_0_offset,
                       input_list[SWIGLU_GRAD_WORKSPACE_IDX],
                       input_list[task_desc.tiling_data_position] + task_desc.tiling_data_offset,
                       GetSwiGluClampLimit(task_desc));
          return;
        } else {
          GM_ADDR tiling_data_addr = input_list[task_desc.tiling_data_position] + 80 * (AscendC::GetBlockIdx() + 1);
          __gm__ SwiGluTilingData *tilingdata_data = reinterpret_cast<__gm__ SwiGluTilingData *>(tiling_data_addr);
          tilingdata_data->rowLen = end - (start + current_seq_start);
          if (tilingdata_data->rowLen == 0) {
            return;
          }
          tilingdata_data->baseRowLen = tilingdata_data->rowLen < SWIGLU_DYNAMIC_TAIL_BASE_ROW_LIMIT
                                           ? tilingdata_data->rowLen : SWIGLU_DYNAMIC_TAIL_BASE_ROW_LIMIT;
          cacheWriteThrough(tiling_data_addr + SWIGLU_DYNAMIC_FIELDS_OFFSET, SWIGLU_DYNAMIC_FIELDS_BYTES);
          PipeBarrier<PIPE_ALL>();

          swi_glu_grad(input_list[task_desc.inputs[0].input_position] + input_0_offset,
                       input_list[task_desc.inputs[1].input_position] + input_1_offset,
                       input_list[task_desc.outputs[0].input_position] + output_0_offset,
                       input_list[SWIGLU_GRAD_WORKSPACE_IDX],
                       tiling_data_addr, GetSwiGluClampLimit(task_desc));
        }
      }
      return;
    }
  }

  __aicore__ inline void ExecuteGroupedMatmul(const TaskDesc &task_desc) {
    if (task_desc.inputs[0].tensor_type == 2) {
      if (task_desc.inputs[0].dim[0] == 0) {
        return;
      }
      GM_ADDR tiling_address = input_list[task_desc.tiling_data_position] + task_desc.tiling_data_offset +
                               2016 * AscendC::GetBlockIdx();
      grouped_matmul(SharedTensorAddress(task_desc.inputs[0]), SharedTensorAddress(task_desc.inputs[1]),
                     nullptr, nullptr, nullptr, nullptr, nullptr, SharedTensorAddress(task_desc.inputs[2]),
                     nullptr, SharedTensorAddress(task_desc.outputs[0]), input_list[WORKSPACE_IDX], tiling_address,
                     task_desc.inputs[0].transpose_flag == 1, task_desc.inputs[1].transpose_flag == 1);
      return;
    }
    if (task_desc.inputs[0].dynamic_shape == 1 || task_desc.inputs[1].dynamic_shape == 1) {
      DynamicData dynamic_data;
      getDynamicData(this->runtimeConfigPtr, &(dynamic_data));
      if (dynamic_data.dynamic_type == DynamicType::DYNAMIC_DSV3_MOE) {
        GM_ADDR grouped_list = input_list[task_desc.inputs[2].input_position] +
                               task_desc.inputs[2].base_ptr_offset * task_desc.inputs[2].data_type;
        GlobalTensor<int64_t> grouped_list_tensor;
        int64_t ep = getExtraValueFromTiling(input_list[TILING_IDX], 1);
        int64_t expert_num = getExtraValueFromTiling(input_list[TILING_IDX], 2);
        int64_t expert_num_single_rank = expert_num / ep;
        grouped_list_tensor.SetGlobalBuffer((__gm__ int64_t *)(grouped_list), expert_num_single_rank);

        GM_ADDR grouped_list_real = this->runtimeConfigPtr + this->grouped_matmul_group_list_offset_;
        GlobalTensor<int64_t> grouped_list_tensor_real;
        grouped_list_tensor_real.SetGlobalBuffer((__gm__ int64_t *)(grouped_list_real), expert_num_single_rank);
        uint32_t data_index = task_desc.task_index / this->core_num;
        int64_t value = grouped_list_tensor.GetValue(data_index);
        int64_t start = 0;
        if (data_index != 0) {
          value = value - grouped_list_tensor.GetValue(data_index - 1);
          start = grouped_list_tensor.GetValue(data_index - 1);
        }
        if (value == 0) {
          return;
        }
        for (uint32_t i = 0; i < expert_num_single_rank; i++) {
          if (i > data_index) {
            grouped_list_tensor_real.SetValue(i, value);
          } else if (i == data_index) {
            grouped_list_tensor_real.SetValue(i, value);
          } else {
            grouped_list_tensor_real.SetValue(i, 0);
          }
        }
        cacheWriteThrough(grouped_list_real, expert_num_single_rank * sizeof(int64_t));

        GM_ADDR tiling_data_addr = input_list[task_desc.tiling_data_position] + 2016 * AscendC::GetBlockIdx();
        __gm__ GMMTilingData *tilingdata_data = reinterpret_cast<__gm__ GMMTilingData *>(tiling_data_addr);
        if (getTransposeData(task_desc.inputs[0].transpose_flag)) {
          tilingdata_data->mmTilingData.Ka = value;
          tilingdata_data->mmTilingData.Kb = value;
          tilingdata_data->mmTilingData.singleCoreK = value;
          cacheWriteThrough(tiling_data_addr + GMM_MATMUL_K_OFFSET, GMM_MATMUL_K_FIELDS_BYTES);
        } else {
          tilingdata_data->gmmBaseParams.m = value;
          tilingdata_data->mmTilingData.M = value;
          tilingdata_data->mmTilingData.singleCoreM = value;
          cacheWriteThrough(tiling_data_addr + GMM_BASE_M_OFFSET, sizeof(uint32_t));
          cacheWriteThrough(tiling_data_addr + GMM_MATMUL_M_OFFSET, GMM_MATMUL_M_FIELDS_BYTES);
        }
        PipeBarrier<PIPE_ALL>();

        int64_t input_0_offset = task_desc.inputs[0].dynamic_shape == 1
                                   ? start * task_desc.inputs[0].dim[1] * task_desc.inputs[0].data_type
                                   : task_desc.inputs[0].base_ptr_offset * task_desc.inputs[0].data_type;
        int64_t input_1_offset = task_desc.inputs[1].dynamic_shape == 1
                                   ? start * task_desc.inputs[1].dim[1] * task_desc.inputs[1].data_type
                                   : task_desc.inputs[1].base_ptr_offset * task_desc.inputs[1].data_type;
        int64_t output_0_offset = task_desc.outputs[0].dynamic_shape == 1
                                    ? start * task_desc.outputs[0].dim[1] * task_desc.outputs[0].data_type
                                    : task_desc.outputs[0].base_ptr_offset * task_desc.outputs[0].data_type;

        grouped_matmul(input_list[task_desc.inputs[0].input_position] + input_0_offset,
                       input_list[task_desc.inputs[1].input_position] + input_1_offset, nullptr, nullptr, nullptr,
                       nullptr, nullptr, grouped_list_real, nullptr,
                       input_list[task_desc.outputs[0].input_position] + output_0_offset, input_list[WORKSPACE_IDX],
                       tiling_data_addr, getTransposeData(task_desc.inputs[0].transpose_flag),
                       getTransposeData(task_desc.inputs[1].transpose_flag));
      }
      return;
    }
  }

  __aicore__ inline void ExecuteShmemPutMem(const TaskDesc &task_desc) {
    if (task_desc.extra_value_0 != 0xFFFFFFFFU) {
      ExecuteRoutedShmemPutMem(task_desc);
      return;
    }
    GM_ADDR signal = input_list[EVENT_IDX];

    int64_t ep = getExtraValueFromTiling(input_list[TILING_IDX], 1);
    int64_t expert_num = getExtraValueFromTiling(input_list[TILING_IDX], 2);
    int64_t expert_num_single_rank = expert_num / ep;
    int64_t hidden_size = getExtraValueFromTiling(input_list[TILING_IDX], 3);

    int64_t rank_id = (*(__gm__ int64_t *)(input_list[TILING_IDX])) / ep * ep;
    int64_t single_expert_task_num = task_desc.task_split_num / expert_num;
    int64_t target_pe = rank_id + task_desc.task_index / (expert_num_single_rank * single_expert_task_num);

    int64_t target_offset_ = *((__gm__ int64_t *)(input_list[task_desc.inputs[0].input_position] +
                                                  task_desc.inputs[0].base_ptr_offset * task_desc.inputs[0].data_type));
    int64_t src_offset_ = *((__gm__ int64_t *)(input_list[task_desc.inputs[2].input_position] +
                                               task_desc.inputs[2].base_ptr_offset * task_desc.inputs[2].data_type));
    int32_t size_ = *((__gm__ int32_t *)(input_list[task_desc.inputs[3].input_position] +
                                         task_desc.inputs[3].base_ptr_offset * task_desc.inputs[3].data_type));
    int32_t send_data_size_ = task_desc.task_split_value * hidden_size;

    int32_t value = task_desc.task_index % single_expert_task_num;
    int32_t start = value * task_desc.task_split_value * hidden_size;
    int32_t end = (value + 1) * task_desc.task_split_value * hidden_size;
    if (start >= size_) {
      aclshmemx_signal_op(reinterpret_cast<__gm__ int32_t *>(signal) + task_desc.trigger_event, 1, ACLSHMEM_SIGNAL_ADD,
                          target_pe);
      return;
    } else {
      if (end > size_) {
        send_data_size_ = size_ - start;
      }
    }
    target_offset_ = target_offset_ + static_cast<int64_t>(start);
    src_offset_ = src_offset_ + static_cast<int64_t>(start);
    send_data_size_ = static_cast<int64_t>(send_data_size_);

    put_mem_signal_kernel(input_list[task_desc.outputs[0].input_position] +
                            task_desc.outputs[0].base_ptr_offset * task_desc.outputs[0].data_type,
                          target_offset_,
                          input_list[task_desc.inputs[1].input_position] +
                            task_desc.inputs[1].base_ptr_offset * task_desc.inputs[1].data_type,
                          src_offset_, send_data_size_, signal, static_cast<int64_t>(task_desc.trigger_event), 1,
                          nullptr, 1, target_pe, false);
  }

  __aicore__ inline void PublishRouteDispatch(const TaskDesc &task, int64_t target_pe) {
    shmem_data_plane::fence();
    shmem_data_plane::signal(reinterpret_cast<__gm__ int32_t *>(input_list[EVENT_IDX]) + task.trigger_event, 1,
                             shmem_data_plane::SignalOp::Add, target_pe);
  }

  __aicore__ inline void LoadWeightedRouteRow(__gm__ DTYPE_DISPATCH_TARGET *source,
                                               __gm__ int32_t *indices, int64_t route_row, int64_t hidden_size,
                                               int64_t top_k, __ubuf__ DTYPE_DISPATCH_TARGET *buffer,
                                               AscendC::TEventID event) {
    int64_t token = indices[route_row] / top_k;
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event);
    aclshmemi_copy_gm2ub(buffer, source + token * hidden_size, hidden_size * sizeof(DTYPE_DISPATCH_TARGET));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event);
  }

  __aicore__ inline void PutWeightedRouteRows(__gm__ DTYPE_DISPATCH_TARGET *source,
                                              __gm__ DTYPE_DISPATCH_TARGET *destination, __gm__ int32_t *indices,
                                              __gm__ float *weights, int64_t route_begin, int64_t rows,
                                              int64_t hidden_size, int64_t top_k) {
    constexpr int64_t ROW_ELEMENTS = 8192;
    uint64_t scratch = aclshmemi_get_state()->mte_config.aclshmem_ub;
    uint64_t value_bytes = ROW_ELEMENTS * sizeof(DTYPE_DISPATCH_TARGET);
    if (scratch + 2 * value_bytes + ROW_ELEMENTS * sizeof(float) > ROUTE_UB_BYTES) {
      AscendC::Trap();
    }
    __ubuf__ DTYPE_DISPATCH_TARGET *ping_ub = reinterpret_cast<__ubuf__ DTYPE_DISPATCH_TARGET *>(scratch);
    __ubuf__ DTYPE_DISPATCH_TARGET *pong_ub =
      reinterpret_cast<__ubuf__ DTYPE_DISPATCH_TARGET *>(scratch + value_bytes);
    LocalTensor<DTYPE_DISPATCH_TARGET> ping = MakeLocalTensorFromUb<DTYPE_DISPATCH_TARGET>(scratch, ROW_ELEMENTS);
    LocalTensor<DTYPE_DISPATCH_TARGET> pong =
      MakeLocalTensorFromUb<DTYPE_DISPATCH_TARGET>(scratch + value_bytes, ROW_ELEMENTS);
    LocalTensor<float> temporary = MakeLocalTensorFromUb<float>(scratch + 2 * value_bytes, ROW_ELEMENTS);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    LoadWeightedRouteRow(source, indices, route_begin, hidden_size, top_k, ping_ub, EVENT_ID0);
    for (int64_t row = 0; row < rows; ++row) {
      bool use_pong = (row & 1) != 0;
      AscendC::TEventID event = use_pong ? EVENT_ID1 : EVENT_ID0;
      LocalTensor<DTYPE_DISPATCH_TARGET> value = use_pong ? pong : ping;
      __ubuf__ DTYPE_DISPATCH_TARGET *value_ub = use_pong ? pong_ub : ping_ub;
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event);
      if (row + 1 < rows) {
        LoadWeightedRouteRow(source, indices, route_begin + row + 1, hidden_size, top_k,
                             use_pong ? ping_ub : pong_ub, use_pong ? EVENT_ID0 : EVENT_ID1);
      }
      Cast(temporary, value, RoundMode::CAST_NONE, hidden_size);
      PipeBarrier<PIPE_V>();
      float weight = weights[indices[route_begin + row]];
      Muls(temporary, temporary, weight, hidden_size);
      PipeBarrier<PIPE_V>();
      Cast(value, temporary, RoundMode::CAST_RINT, hidden_size);
      AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(event);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(event);
      aclshmemi_copy_ub2gm(destination + row * hidden_size, value_ub, hidden_size * sizeof(DTYPE_DISPATCH_TARGET));
      AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event);
    }
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
  }

  __aicore__ inline void ExecuteRoutedShmemPutMem(const TaskDesc &task_desc) {
    int64_t ep = getExtraValueFromTiling(input_list[TILING_IDX], 1);
    int64_t expert_num = getExtraValueFromTiling(input_list[TILING_IDX], 2);
    int64_t local_experts = expert_num / ep;
    int64_t hidden_size = getExtraValueFromTiling(input_list[TILING_IDX], 3);
    int64_t top_k = static_cast<int64_t>(task_desc.inputs[1].dim[3]);
    int64_t tasks_per_expert = task_desc.task_split_num / expert_num;
    int64_t target_pe = task_desc.task_index / (local_experts * tasks_per_expert);

    int32_t size = *((__gm__ int32_t *)(input_list[task_desc.inputs[3].input_position] +
                                        task_desc.inputs[3].base_ptr_offset * task_desc.inputs[3].data_type));
    int64_t tile = task_desc.task_index % tasks_per_expert;
    int64_t start = tile * task_desc.task_split_value * hidden_size;
    if (start >= size) {
      PublishRouteDispatch(task_desc, target_pe);
#ifndef __DAV_C220_CUBE__
      if (task_desc.extra_value_2 != 0) {
        this->TriggerEvent(task_desc.extra_value_2);
      }
#endif
      return;
    }

    int64_t target_offset = *((__gm__ int64_t *)(input_list[task_desc.inputs[0].input_position] +
                                                 task_desc.inputs[0].base_ptr_offset * task_desc.inputs[0].data_type));
    int64_t src_offset = *((__gm__ int64_t *)(input_list[task_desc.inputs[2].input_position] +
                                              task_desc.inputs[2].base_ptr_offset * task_desc.inputs[2].data_type));
    int64_t send_elems = task_desc.task_split_value * hidden_size;
    if (start + send_elems > size) {
      send_elems = size - start;
    }
    int64_t rows = send_elems / hidden_size;
    int64_t logical_src_row = (src_offset + start) / hidden_size;
    int64_t target_base = target_offset + start;

    __gm__ int32_t *route_index = (__gm__ int32_t *)(input_list[task_desc.extra_value_0]);
    __gm__ float *route_weight = (__gm__ float *)(input_list[task_desc.extra_value_1]);
    __gm__ DTYPE_DISPATCH_TARGET *src =
      (__gm__ DTYPE_DISPATCH_TARGET *)(input_list[task_desc.inputs[1].input_position]);
    __gm__ DTYPE_DISPATCH_TARGET *target =
      (__gm__ DTYPE_DISPATCH_TARGET *)(input_list[task_desc.outputs[0].input_position] +
                                       task_desc.outputs[0].base_ptr_offset * task_desc.outputs[0].data_type);
    __gm__ DTYPE_DISPATCH_TARGET *dst =
      (__gm__ DTYPE_DISPATCH_TARGET *)(shmem_data_plane::remote_ptr(target + target_base, target_pe));

    if (hidden_size <= 8192) {
      PutWeightedRouteRows(src, dst, route_index, route_weight, logical_src_row, rows, hidden_size, top_k);
      PublishRouteDispatch(task_desc, target_pe);
      if (task_desc.extra_value_2 != 0) {
        this->TriggerEvent(task_desc.extra_value_2);
      }
      return;
    }

    __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();
    uint64_t copy_ub = device_state->mte_config.aclshmem_ub;
    constexpr int64_t chunk_elems = 8192;
    __ubuf__ DTYPE_DISPATCH_TARGET *value_ub = reinterpret_cast<__ubuf__ DTYPE_DISPATCH_TARGET *>(copy_ub);
    LocalTensor<DTYPE_DISPATCH_TARGET> value = MakeLocalTensorFromUb<DTYPE_DISPATCH_TARGET>(copy_ub, chunk_elems);
    LocalTensor<float> temporary =
      MakeLocalTensorFromUb<float>(copy_ub + chunk_elems * sizeof(DTYPE_DISPATCH_TARGET), chunk_elems);

    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    for (int64_t row = 0; row < rows; ++row) {
      int64_t route_row = logical_src_row + row;
      int32_t route_slot = route_index[route_row];
      int32_t token_row = route_slot / top_k;
      float weight = route_weight[route_slot];
      for (int64_t col = 0; col < hidden_size; col += chunk_elems) {
        int64_t elements = hidden_size - col;
        if (elements > chunk_elems) {
          elements = chunk_elems;
        }
        uint64_t bytes = elements * sizeof(DTYPE_DISPATCH_TARGET);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        aclshmemi_copy_gm2ub(value_ub, src + static_cast<int64_t>(token_row) * hidden_size + col, bytes);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        Cast(temporary, value, RoundMode::CAST_NONE, elements);
        PipeBarrier<PIPE_V>();
        Muls(temporary, temporary, weight, elements);
        PipeBarrier<PIPE_V>();
        Cast(value, temporary, RoundMode::CAST_RINT, elements);
        PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        aclshmemi_copy_ub2gm(dst + row * hidden_size + col, value_ub, bytes);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
      }
    }
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    PublishRouteDispatch(task_desc, target_pe);
#ifndef __DAV_C220_CUBE__
    if (task_desc.extra_value_2 != 0) {
      this->TriggerEvent(task_desc.extra_value_2);
    }
#endif
  }
};

}  // namespace MulticoreRuntime

extern "C" inline __aicore__ void worker_kernel(uint32_t worker_id, __gm__ uint8_t *runtimeConfigPtr,
                                                GM_ADDR *input_list) {
  MulticoreRuntime::KernelWorker worker;
  worker.Init(worker_id, runtimeConfigPtr, input_list);
  worker.Process();
}
