// Copyright 2026 Huawei Technologies Co., Ltd
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ============================================================================
#include "acl/acl.h"
#include "kernel_operator.h"  // NOLINT(build/include_subdir): flat CANN SDK header.
#define HOT_COUNT_DEVICE __aicore__ inline
#define HOT_COUNT_LOCAL __ubuf__
#include "./count_algorithm.h"

namespace {
constexpr int kBlocks = 32;
constexpr int kMaximumExperts = 512;
constexpr int kMaximumTopK = 8;
constexpr int kTileRows = 128;
constexpr int kStrideAlignment = 64;
constexpr int kMaximumStride = 576;

template <AscendC::HardEvent Event>
__aicore__ inline void SyncCountPipe() {
  AscendC::SetFlag<Event>(EVENT_ID0);
  AscendC::WaitFlag<Event>(EVENT_ID0);
}
}  // namespace

inline __gm__ struct OpSystemRunCfg g_opSystemRunCfg{0};

template <typename InputT, int FixedTopK = 0>
__global__ __aicore__ void replica_count_partials(GM_ADDR input, GM_ADDR partials, int64_t rows, int top_k, int experts,
                                                  int stride) {
  AscendC::TPipe pipe;
  AscendC::TBuf<AscendC::TPosition::VECCALC> tile_buffer, count_buffer;
  pipe.InitBuffer(tile_buffer, kTileRows * kMaximumTopK * sizeof(InputT));
  pipe.InitBuffer(count_buffer, kMaximumStride * sizeof(int64_t));
  auto tile = tile_buffer.Get<InputT>();
  auto memory = reinterpret_cast<__ubuf__ InputT *>(tile.GetPhyAddr());
  auto local_counts = count_buffer.Get<int64_t>();
  auto counts = reinterpret_cast<__ubuf__ int64_t *>(local_counts.GetPhyAddr());
  for (int expert = 0; expert < stride; ++expert) counts[expert] = 0;
  const int64_t per_block = (rows + kBlocks - 1) / kBlocks;
  const int64_t first = AscendC::GetBlockIdx() * per_block;
  const int64_t last = rows < first + per_block ? rows : first + per_block;
  AscendC::GlobalTensor<InputT> source;
  AscendC::GlobalTensor<int64_t> output;
  source.SetGlobalBuffer(reinterpret_cast<__gm__ InputT *>(input));
  output.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(partials));
  for (int64_t offset = first; offset < last; offset += kTileRows) {
    const int length = last - offset < kTileRows ? static_cast<int>(last - offset) : kTileRows;
    SyncCountPipe<AscendC::HardEvent::S_MTE2>();
    AscendC::DataCopyExtParams copy{1, static_cast<uint32_t>(length * top_k * sizeof(InputT)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<InputT> padding{false, 0, 0, 0};
    AscendC::DataCopyPad(tile, source[offset * top_k], copy, padding);
    SyncCountPipe<AscendC::HardEvent::MTE2_S>();
    for (int row = 0; row < length; ++row) {
      if constexpr (FixedTopK != 0) {
        hot::CountFixedRow<FixedTopK>(memory + row * FixedTopK, counts, experts);
      } else {
        hot::CountRow(memory + row * top_k, counts, top_k, experts);
      }
    }
  }
  // Each producer owns an aligned row, so it can publish its complete partial histogram with one copy.
  const int base = AscendC::GetBlockIdx() * stride;
  SyncCountPipe<AscendC::HardEvent::S_MTE3>();
  AscendC::DataCopy(output[base], local_counts, stride);
  SyncCountPipe<AscendC::HardEvent::MTE3_S>();
}

__global__ __aicore__ void replica_route_payload_merge(GM_ADDR partials, GM_ADDR result, int64_t rows,
                                                       int experts, int stride, int autograd_mask) {
  constexpr int kColumns = 32;
  const int first = AscendC::GetBlockIdx() * kColumns;
  const int length = experts + 3 - first < kColumns ? experts + 3 - first : kColumns;
  const int header = first == 0 ? 3 : 0;
  const int first_count = first == 0 ? 0 : first - 3;
  const int columns = length - header;
  const int padded_columns = (columns + 3) / 4 * 4;
  AscendC::TPipe pipe;
  AscendC::TBuf<AscendC::TPosition::VECCALC> input_buffer, output_buffer, flag_buffer;
  pipe.InitBuffer(input_buffer, kBlocks * kColumns * sizeof(int64_t));
  pipe.InitBuffer(output_buffer, kColumns * sizeof(int32_t));
  pipe.InitBuffer(flag_buffer, kBlocks * 32);
  auto local = input_buffer.Get<int64_t>();
  auto reduced = output_buffer.Get<int32_t>();
  auto flags = flag_buffer.Get<int64_t>();
  AscendC::GlobalTensor<int64_t> source;
  AscendC::GlobalTensor<int32_t> output;
  source.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(partials));
  output.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(result));
  AscendC::DataCopyExtParams copy{kBlocks, static_cast<uint32_t>(columns * sizeof(int64_t)),
      static_cast<uint32_t>((stride - columns) * sizeof(int64_t)), 0, 0};
  AscendC::DataCopyPadExtParams<int64_t> padding{false, 0, 0, 0};
  AscendC::DataCopyPad(local, source[first_count], copy, padding);
  if (first == 0) {
    AscendC::DataCopyExtParams flag_copy{kBlocks, sizeof(int64_t),
        static_cast<uint32_t>((stride - 1) * sizeof(int64_t)), 0, 0};
    AscendC::DataCopyPad(flags, source[experts], flag_copy, padding);
  }
  SyncCountPipe<AscendC::HardEvent::MTE2_S>();
  auto memory = reinterpret_cast<__ubuf__ int64_t *>(local.GetPhyAddr());
  auto values = reinterpret_cast<__ubuf__ int32_t *>(reduced.GetPhyAddr());
  for (int column = 0; column < columns; ++column) {
    int64_t value = 0;
    for (int block = 0; block < kBlocks; ++block) value += memory[block * padded_columns + column];
    values[header + column] = static_cast<int32_t>(value);
  }
  if (first == 0) {
    values[0] = 0;
    values[1] = static_cast<int32_t>(rows);
    values[2] = autograd_mask;
    auto invalid = reinterpret_cast<__ubuf__ int64_t *>(flags.GetPhyAddr());
    for (int block = 0; block < kBlocks; ++block) {
      if (invalid[block * 4] != 0) values[3] = -1;
    }
  }
  SyncCountPipe<AscendC::HardEvent::S_MTE3>();
  AscendC::DataCopyExtParams publish{1, static_cast<uint32_t>(length * sizeof(int32_t)), 0, 0, 0};
  AscendC::DataCopyPad(output[first], reduced, publish);
  SyncCountPipe<AscendC::HardEvent::MTE3_S>();
}

extern "C" int launch_replica_route_payload(void *stream, void *input, void *partials, void *result, int64_t rows,
                                            int top_k, int experts, int stride, int autograd_mask, int input_bytes) {
  if (rows < 0 || top_k < 1 || top_k > kMaximumTopK || rows > INT32_MAX / top_k || experts < 1 ||
      experts > kMaximumExperts || stride != (experts + kStrideAlignment) / kStrideAlignment * kStrideAlignment ||
      (input_bytes != sizeof(int32_t) && input_bytes != sizeof(int64_t))) {
    return ACL_ERROR_INVALID_PARAM;
  }
  if (input_bytes == sizeof(int32_t)) {
    if (top_k == 8) {
      replica_count_partials<int32_t, 8><<<kBlocks, nullptr, stream>>>(static_cast<uint8_t *>(input),
          static_cast<uint8_t *>(partials), rows, top_k, experts, stride);
    } else {
      replica_count_partials<int32_t><<<kBlocks, nullptr, stream>>>(static_cast<uint8_t *>(input),
          static_cast<uint8_t *>(partials), rows, top_k, experts, stride);
    }
  } else {
    if (top_k == 8) {
      replica_count_partials<int64_t, 8><<<kBlocks, nullptr, stream>>>(static_cast<uint8_t *>(input),
          static_cast<uint8_t *>(partials), rows, top_k, experts, stride);
    } else {
      replica_count_partials<int64_t><<<kBlocks, nullptr, stream>>>(static_cast<uint8_t *>(input),
          static_cast<uint8_t *>(partials), rows, top_k, experts, stride);
    }
  }
  const int status = aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL);
  if (status) return status;
  replica_route_payload_merge<<<(experts + 3 + 31) / 32, nullptr, stream>>>(static_cast<uint8_t *>(partials),
      static_cast<uint8_t *>(result), rows, experts, stride, autograd_mask);
  return aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL);
}

extern "C" int replica_counts_abi_version() { return 2; }
