// Copyright 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#include "acl/acl.h"
#include "kernel_operator.h"
#define HOT_COUNT_DEVICE __aicore__ inline
#define HOT_COUNT_LOCAL __ubuf__
#include "./count_algorithm.h"

inline __gm__ struct OpSystemRunCfg g_opSystemRunCfg{0};

namespace {
constexpr int kExperts = 128;
constexpr int kTopK = 8;
constexpr int kBlocks = 32;
constexpr int kStride = 136;
constexpr int kTileTokens = 128;
constexpr int kTile = kTileTokens * kTopK;
constexpr int kMaximumTokens = 16384;
constexpr int kSortedCapacity = (kMaximumTokens / kBlocks) * kTopK + kExperts * 7;

template <AscendC::HardEvent Event>
__aicore__ inline void Sync() {
  AscendC::SetFlag<Event>(EVENT_ID0);
  AscendC::WaitFlag<Event>(EVENT_ID0);
}

__aicore__ inline int BoundedExpert(int64_t expert) {
  return expert < 0 ? 0 : (expert >= kExperts ? kExperts - 1 : static_cast<int>(expert));
}

template <typename InputT, bool ProduceInverse>
__global__ __aicore__ void CountOrIndexReplicaRows(GM_ADDR ids, GM_ADDR hist, GM_ADDR inverse, GM_ADDR sorted, int64_t rows) {
  AscendC::TPipe pipe;
  AscendC::TBuf<AscendC::TPosition::VECCALC> ids_buffer, count_buffer, second_buffer;
  AscendC::TBuf<AscendC::TPosition::VECCALC> sort_buffer, local_pointer_buffer, inverse_buffer;
  pipe.InitBuffer(ids_buffer, kTile * sizeof(InputT));
  pipe.InitBuffer(count_buffer, kStride * sizeof(int32_t));
  if constexpr (ProduceInverse) {
    pipe.InitBuffer(second_buffer, kBlocks * kStride * sizeof(int32_t));
    pipe.InitBuffer(sort_buffer, kSortedCapacity * sizeof(int32_t));
    pipe.InitBuffer(local_pointer_buffer, kExperts * 2 * sizeof(int32_t));
    if constexpr (sizeof(InputT) != sizeof(int32_t)) {
      pipe.InitBuffer(inverse_buffer, kTile * sizeof(int32_t));
    }
  }
  auto id_tile = ids_buffer.Get<InputT>();
  const auto input = reinterpret_cast<__ubuf__ const InputT *>(id_tile.GetPhyAddr());
  auto count_tile = count_buffer.Get<int32_t>();
  auto count = reinterpret_cast<__ubuf__ int32_t *>(count_tile.GetPhyAddr());

  AscendC::GlobalTensor<int32_t> histogram, index, stable_rows;
  AscendC::GlobalTensor<InputT> src;
  AscendC::LocalTensor<int32_t> inverse_tile;
  AscendC::LocalTensor<int32_t> block_sorted;
  __ubuf__ int32_t *order_values = nullptr;
  __ubuf__ int32_t *block_write = nullptr;
  __ubuf__ int32_t *global_starts = nullptr;
  histogram.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(hist));
  src.SetGlobalBuffer(reinterpret_cast<__gm__ InputT *>(ids));
  if constexpr (ProduceInverse) {
    index.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(inverse));
    stable_rows.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(sorted));
    block_sorted = sort_buffer.Get<int32_t>();
    order_values = reinterpret_cast<__ubuf__ int32_t *>(block_sorted.GetPhyAddr());
    block_write = reinterpret_cast<__ubuf__ int32_t *>(local_pointer_buffer.Get<int32_t>().GetPhyAddr());
    global_starts = block_write + kExperts;
    if constexpr (sizeof(InputT) == sizeof(int32_t)) {
      inverse_tile = ids_buffer.Get<int32_t>();
    } else {
      inverse_tile = inverse_buffer.Get<int32_t>();
    }
  }
  const int block = AscendC::GetBlockIdx();
  const int64_t block_rows = (rows + kBlocks - 1) / kBlocks;
  const int64_t first = block * block_rows;
  const int64_t last = rows < first + block_rows ? rows : first + block_rows;
  if constexpr (ProduceInverse) {
    if (first >= last) return;
    // Each output block owns a contiguous source span. Earlier blocks precede
    // it for each expert. All rows of earlier experts precede that span.
    auto all_tile = second_buffer.Get<int32_t>();
    AscendC::DataCopy(all_tile, histogram, kBlocks * kStride);
    Sync<AscendC::HardEvent::MTE2_S>();
    auto all = reinterpret_cast<__ubuf__ int32_t *>(all_tile.GetPhyAddr());
    int earlier_experts = 0;
    int local_until = 0;
    for (int expert = 0; expert < kExperts; ++expert) {
      int total = 0;
      int prior_blocks = 0;
      for (int i = 0; i < kBlocks; ++i) {
        const int n = all[i * kStride + expert];
        total += n;
        prior_blocks += i < block ? n : 0;
      }
      count[expert] = earlier_experts + prior_blocks;
      earlier_experts += total;
      block_write[expert] = local_until;
      global_starts[expert] = count[expert];
      const int n = all[block * kStride + expert];
      local_until += (n + 7) / 8 * 8;
    }
  } else {
    for (int i = 0; i < kStride; ++i) count[i] = 0;
  }

  for (int64_t pos = first; pos < last; pos += kTileTokens) {
    const int length = last - pos < kTileTokens ? static_cast<int>(last - pos) : kTileTokens;
    Sync<AscendC::HardEvent::S_MTE2>();
    AscendC::DataCopy(id_tile, src[pos * kTopK], length * kTopK);
    Sync<AscendC::HardEvent::MTE2_S>();
    if constexpr (ProduceInverse) {
      auto inverse_values = reinterpret_cast<__ubuf__ int32_t *>(inverse_tile.GetPhyAddr());
      for (int j = 0; j < length * kTopK; ++j) {
        const int expert = BoundedExpert(input[j]);
        inverse_values[j] = count[expert]++;
        order_values[block_write[expert]++] = static_cast<int32_t>(pos + j / kTopK);
      }
      Sync<AscendC::HardEvent::S_MTE3>();
      AscendC::DataCopy(index[pos * kTopK], inverse_tile, length * kTopK);
      Sync<AscendC::HardEvent::MTE3_S>();
    } else {
      for (int row = 0; row < length; ++row) {
        hot::CountFixedRow<kTopK>(input + row * kTopK, count, kExperts);
      }
    }
  }
  if constexpr (ProduceInverse) {
    int local = 0;
    Sync<AscendC::HardEvent::S_MTE3>();
    for (int expert = 0; expert < kExperts; ++expert) {
      const int n = count[expert] - global_starts[expert];
      if (n == 0) continue;
      AscendC::DataCopyExtParams copy{1, static_cast<uint32_t>(n * sizeof(int32_t)), 0, 0, 0};
      AscendC::DataCopyPad(stable_rows[global_starts[expert]], block_sorted[local], copy);
      local += (n + 7) / 8 * 8;
    }
    Sync<AscendC::HardEvent::MTE3_S>();
  } else {
    Sync<AscendC::HardEvent::S_MTE3>();
    AscendC::DataCopy(histogram[block * kStride], count_tile, kStride);
    Sync<AscendC::HardEvent::MTE3_S>();
  }
}

__global__ __aicore__ void BuildOrderedRoutePayload(GM_ADDR hist, GM_ADDR output, int rows, int flags) {
  constexpr int kChunk = 32;
  AscendC::TPipe pipe;
  AscendC::TBuf<AscendC::TPosition::VECCALC> hist_buffer, result_buffer;
  pipe.InitBuffer(hist_buffer, kBlocks * kStride * sizeof(int32_t));
  pipe.InitBuffer(result_buffer, kChunk * sizeof(int32_t));
  AscendC::GlobalTensor<int32_t> histogram, payload;
  histogram.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(hist));
  payload.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(output));
  auto hist_tile = hist_buffer.Get<int32_t>();
  auto result = result_buffer.Get<int32_t>();
  AscendC::DataCopy(hist_tile, histogram, kBlocks * kStride);
  Sync<AscendC::HardEvent::MTE2_S>();
  auto all = reinterpret_cast<__ubuf__ int32_t *>(hist_tile.GetPhyAddr());
  auto out = reinterpret_cast<__ubuf__ int32_t *>(result.GetPhyAddr());
  const int first = AscendC::GetBlockIdx() * kChunk;
  const int len = kExperts + 3 - first < kChunk ? kExperts + 3 - first : kChunk;
  for (int col = 0; col < len; ++col) {
    const int field = first + col;
    if (field < 3) {
      out[col] = field == 0 ? 0 : (field == 1 ? rows : flags);
      continue;
    }
    int total = 0;
    int invalid = 0;
    for (int block = 0; block < kBlocks; ++block) {
      total += all[block * kStride + field - 3];
      if (field == 3) invalid |= all[block * kStride + kExperts];
    }
    out[col] = invalid ? -1 : total;
  }
  Sync<AscendC::HardEvent::S_MTE3>();
  AscendC::DataCopyExtParams copy{1, static_cast<uint32_t>(len * sizeof(int32_t)), 0, 0, 0};
  AscendC::DataCopyPad(payload[first], result, copy);
  Sync<AscendC::HardEvent::MTE3_S>();
}
}  // namespace

template <typename InputT>
static int LaunchOrderedReplicaRoute(void *stream, void *ids, void *hist, void *payload,
                                    void *inverse, void *sorted, int rows, int flags) {
  auto src = static_cast<uint8_t *>(ids);
  auto histogram = static_cast<uint8_t *>(hist);
  CountOrIndexReplicaRows<InputT, false><<<kBlocks, nullptr, stream>>>(src, histogram, nullptr, nullptr, rows);
  int status = aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL);
  if (status != ACL_SUCCESS) return status;
  BuildOrderedRoutePayload<<<(kExperts + 3 + 31) / 32, nullptr, stream>>>(
      histogram, static_cast<uint8_t *>(payload), rows, flags);
  status = aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL);
  if (status != ACL_SUCCESS) return status;
  CountOrIndexReplicaRows<InputT, true><<<kBlocks, nullptr, stream>>>(
      src, histogram, static_cast<uint8_t *>(inverse), static_cast<uint8_t *>(sorted), rows);
  return aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL);
}

extern "C" int launch_a3_ordered_replica_route(void *stream, void *ids, void *hist, void *payload,
                                               void *inverse, void *sorted, int rows, int flags,
                                               int input_bytes) {
  if (rows < 1 || rows > kMaximumTokens) return ACL_ERROR_INVALID_PARAM;
  if (input_bytes == sizeof(int32_t)) {
    return LaunchOrderedReplicaRoute<int32_t>(stream, ids, hist, payload, inverse, sorted, rows, flags);
  }
  if (input_bytes == sizeof(int64_t)) {
    return LaunchOrderedReplicaRoute<int64_t>(stream, ids, hist, payload, inverse, sorted, rows, flags);
  }
  return ACL_ERROR_INVALID_PARAM;
}
