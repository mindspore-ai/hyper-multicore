/**
 * Copyright 2024 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "data_plane/mte_rma.h"
#include "data_plane/sync.h"
#include "kernel_operator.h"

using namespace AscendC;

namespace shmem_data_plane = hyper_multicore::shmem::data_plane;

namespace hyper_multicore {

template <typename T>
class PutMemSignalKernel {
 public:
  __aicore__ inline PutMemSignalKernel() {}
  __aicore__ inline void Init(GM_ADDR target, int64_t target_offset, GM_ADDR src, int64_t src_offset, int64_t size,
                              GM_ADDR signal, int64_t signal_offset, int32_t signal_value, int64_t signal_op,
                              int64_t target_pe, bool non_blocking, GM_ADDR workspace) {
    // Initialize pointers and parameters
    target_ = reinterpret_cast<__gm__ T *>(target);
    src_ = reinterpret_cast<__gm__ T *>(src);
    signal_ = reinterpret_cast<__gm__ int32_t *>(signal);
    target_offset_ = target_offset;
    src_offset_ = src_offset;
    size_ = size;
    signal_offset_ = signal_offset;
    signal_value_ = signal_value;
    signal_op_ = signal_op;
    target_pe_ = target_pe;
    (void)non_blocking;

    // UB tiling set
    aiv_idx_ = 0;
    aiv_num_ = 1;

    syncGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(workspace), 256);
    pipe.InitBuffer(localwork, 1, sizeof(int32_t) * aiv_num_);
  }

  __aicore__ inline void Process() {
    // Empty tiles still satisfy remote dependencies without resolving data addresses.
    if (size_ == 0) {
      shmem_data_plane::mte_rma::quiet();
      shmem_data_plane::signal(signal_ + signal_offset_, signal_value_,
                               signal_op_ == 1 ? shmem_data_plane::SignalOp::Add : shmem_data_plane::SignalOp::Set,
                               target_pe_);
      return;
    }
    // Perform data transfer
    auto size_per_core = size_ / aiv_num_;
    auto target_ptr = target_ + target_offset_ + aiv_idx_ * size_per_core;
    auto src_ptr = src_ + src_offset_ + aiv_idx_ * size_per_core;
    if (aiv_idx_ == aiv_num_ - 1) {
      size_per_core = size_ - size_per_core * aiv_idx_;
    }
    shmem_data_plane::mte_rma::put_pipelined<T>(reinterpret_cast<GM_ADDR>(target_ptr),
                                               reinterpret_cast<GM_ADDR>(src_ptr), size_per_core, target_pe_);

    // Write signal after data transfer
    auto signal_ptr = signal_ + signal_offset_;
    shmem_data_plane::mte_rma::quiet();
    // Sync ensure corresponding tasks are done
    if (aiv_num_ > 1) {
      AscendC::LocalTensor<int32_t> workLocal = localwork.AllocTensor<int32_t>();
      AscendC::SyncAll(syncGlobal, workLocal, aiv_num_);
      localwork.FreeTensor(workLocal);
    }

    if (aiv_idx_ == 0) {
      shmem_data_plane::signal(signal_ptr, signal_value_,
                               signal_op_ == 1 ? shmem_data_plane::SignalOp::Add : shmem_data_plane::SignalOp::Set,
                               target_pe_);
    }
  }

 private:
  __gm__ T *target_;
  __gm__ T *src_;
  __gm__ int32_t *signal_;
  AscendC::TPipe pipe;

  int64_t target_offset_ = 0;
  int64_t src_offset_ = 0;
  int64_t size_ = 0;
  int64_t signal_offset_ = 0;
  int32_t signal_value_ = 0;
  int64_t signal_op_ = 0;
  int64_t target_pe_ = 0;
  AscendC::GlobalTensor<int32_t> syncGlobal;
  AscendC::TQue<AscendC::TPosition::VECIN, 1> localwork;
  // ub
  uint32_t aiv_idx_ = 0;
  uint32_t aiv_num_ = 0;
};

}  // namespace hyper_multicore

extern "C" inline __aicore__ void put_mem_signal_kernel(GM_ADDR target, int64_t target_offset, GM_ADDR src,
                                                        int64_t src_offset, int64_t size, GM_ADDR signal,
                                                        int64_t signal_offset, int64_t signal_value, GM_ADDR workspace,
                                                        int64_t signal_op, int64_t target_pe, bool non_blocking) {
  hyper_multicore::PutMemSignalKernel<DTYPE_DISPATCH_TARGET> op;
  op.Init(target, target_offset, src, src_offset, size, signal, signal_offset, signal_value, signal_op, target_pe,
          non_blocking, workspace);
  op.Process();
}
