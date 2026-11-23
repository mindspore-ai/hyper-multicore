// Copyright 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
// Return unquantized expert gradients and accumulate them at their owners.
#include <cstdint>
#include "acl/acl.h"
#include "kernel_operator.h"  // NOLINT(build/include_subdir)
#include "shmem.h"            // NOLINT(build/include_subdir)
#include "csrc/launch.h"

#ifndef ASCEND_IS_AIC
#define ASCEND_IS_AIC (false)
#endif

using namespace AscendC;  // NOLINT(build/namespaces)

namespace {
constexpr int32_t kElements = 8192;
constexpr int32_t kScratch = 64;
constexpr int32_t kReservedUb = 188 * 1024;
constexpr int32_t kFirstContribution = kScratch + kElements * sizeof(float);
constexpr int32_t kIndexOffset = kFirstContribution + kElements * sizeof(float);
constexpr int32_t kNativeHomeOffset = kIndexOffset + kElements * sizeof(float);
constexpr int32_t kTilePairStride = kElements * (2 * sizeof(float) + sizeof(bfloat16_t));
constexpr int32_t kBufferedIndexOffset = kScratch + 2 * kTilePairStride;

template <typename T>
__aicore__ inline LocalTensor<T> LocalAt(int32_t offset, TPosition position) {
    LocalTensor<T> local;
    local.address_.logicPos = static_cast<uint8_t>(position);
    local.address_.bufferAddr = offset;
    local.address_.dataLen = kElements * sizeof(T);
    return local;
}

template <typename T>
__aicore__ inline void LoadOwnerAsync(LocalTensor<T> local, __gm__ T* source, int32_t elements) {
    GlobalTensor<T> global;
    global.SetGlobalBuffer(source);
    DataCopyExtParams copy{1, static_cast<uint32_t>(elements * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padding{false, 0, 0, 0};
    DataCopyPad(local, global, copy, padding);
}

template <typename T>
__aicore__ inline void LoadOwner(LocalTensor<T> local, __gm__ T* source, int32_t elements) {
    LoadOwnerAsync(local, source, elements);
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline void StoreOwnerAsync(__gm__ T* destination, LocalTensor<T> local, int32_t elements) {
    GlobalTensor<T> global;
    global.SetGlobalBuffer(destination);
    DataCopyPad(global, local, DataCopyExtParams(1, elements * sizeof(T), 0, 0, 0));
}

template <typename T>
__aicore__ inline void StoreOwner(__gm__ T* destination, LocalTensor<T> local, int32_t elements) {
    GlobalTensor<T> global;
    global.SetGlobalBuffer(destination);
    DataCopyExtParams copy{1, static_cast<uint32_t>(elements * sizeof(T)), 0, 0, 0};
    PipeBarrier<PIPE_ALL>();
    DataCopyPad(global, local, copy);
    PipeBarrier<PIPE_ALL>();
}
__aicore__ inline void BuildOwnerLists(__gm__ int32_t* plan, __ubuf__ int32_t* heads,
                                     __ubuf__ int32_t* next, int32_t experts,
                                     int32_t ranks, int32_t slots, int32_t rank) {
    const int32_t first_expert = rank * experts;
    for (int32_t expert = 0; expert < experts; ++expert) {
        heads[expert] = -1;
    }
    // Build source-ordered lists once per worker, rather than once per tile.
    for (int32_t rb = ranks * slots - 1; rb >= 0; --rb) {
        const int32_t expert = plan[rb] - first_expert;
        if (expert >= 0 && expert < experts) {
            next[rb] = heads[expert];
            heads[expert] = rb;
        }
    }
}
__aicore__ inline void ReadContribution(__gm__ float* remote, int64_t matrix, int32_t slots,
                                        int32_t entry, int64_t begin, int32_t elements, int32_t buffer) {
    const int32_t offset = buffer == 0 ? kFirstContribution : kIndexOffset;
    const TEventID ready = buffer == 0 ? EVENT_ID0 : EVENT_ID1;
    const int64_t peer_offset = static_cast<int64_t>(entry % slots) * matrix + begin;
    aclshmemx_mte_get_nbi(reinterpret_cast<__ubuf__ float*>(offset), remote + peer_offset,
                          elements, entry / slots, ready);
    SetFlag<HardEvent::MTE2_V>(ready);
}

template <bool NativeHome>
__aicore__ inline void LoadSinglePeerTile(GM_ADDR source, __gm__ float* remote, int64_t expert_offset,
                                         int64_t remote_offset, int64_t matrix, int64_t tile,
                                         int32_t peer, int32_t buffer) {
    const int64_t begin = tile * kElements;
    const int32_t elements = matrix - begin < kElements ? matrix - begin : kElements;
    const int32_t offset = kScratch + buffer * kTilePairStride;
    const TEventID event = buffer == 0 ? EVENT_ID0 : EVENT_ID1;
    if constexpr (NativeHome) {
        auto native = LocalAt<bfloat16_t>(offset + 8 * kElements, TPosition::VECIN);
        LoadOwnerAsync(native, reinterpret_cast<__gm__ bfloat16_t*>(source) + expert_offset + begin, elements);
    } else {
        auto accum = LocalAt<float>(offset, TPosition::VECIN);
        LoadOwnerAsync(accum, reinterpret_cast<__gm__ float*>(source) + expert_offset + begin, elements);
    }
    aclshmemx_mte_get_nbi(reinterpret_cast<__ubuf__ float*>(offset + 4 * kElements),
                          remote + remote_offset + begin, elements, peer, event);
    SetFlag<HardEvent::MTE2_V>(event);
}

template <bool NativeHome>
__aicore__ inline void ReturnSinglePeer(GM_ADDR source, GM_ADDR output, __gm__ float* remote,
                                        int32_t expert, int32_t entry, int64_t matrix, int32_t slots,
                                        int32_t worker, int32_t workers, int64_t tiles, bool bf16) {
    if (worker >= tiles) return;
    const int64_t expert_offset = static_cast<int64_t>(expert) * matrix;
    const int64_t remote_offset = static_cast<int64_t>(entry % slots) * matrix;
    const int32_t peer = entry / slots;
    int32_t buffer = 0;
    LoadSinglePeerTile<NativeHome>(source, remote, expert_offset, remote_offset, matrix, worker, peer, buffer);
    for (int64_t tile = worker; tile < tiles; tile += workers) {
        const TEventID event = buffer == 0 ? EVENT_ID0 : EVENT_ID1;
        const int64_t following = tile + workers;
        WaitFlag<HardEvent::MTE2_V>(event);
        if (following < tiles) {
            const TEventID other = buffer == 0 ? EVENT_ID1 : EVENT_ID0;
            // A prior output must leave this UB slot before the next MTE2.
            if (tile != worker) WaitFlag<HardEvent::MTE3_MTE2>(other);
            LoadSinglePeerTile<NativeHome>(source, remote, expert_offset, remote_offset, matrix,
                                           following, peer, 1 - buffer);
        }
        const int64_t begin = tile * kElements;
        const int32_t elements = matrix - begin < kElements ? matrix - begin : kElements;
        const int32_t offset = kScratch + buffer * kTilePairStride;
        auto accum = LocalAt<float>(offset, TPosition::VECIN);
        auto contribution = LocalAt<float>(offset + 4 * kElements, TPosition::VECIN);
        if constexpr (NativeHome) {
            auto native = LocalAt<bfloat16_t>(offset + 8 * kElements, TPosition::VECIN);
            Cast(accum, native, RoundMode::CAST_NONE, elements);
            PipeBarrier<PIPE_V>();
        }
        Add(accum, accum, contribution, elements);
        if (bf16) {
            auto native = LocalAt<bfloat16_t>(offset + 8 * kElements, TPosition::VECOUT);
            PipeBarrier<PIPE_V>();
            Cast(native, accum, RoundMode::CAST_RINT, elements);
            SetFlag<HardEvent::V_MTE3>(event);
            WaitFlag<HardEvent::V_MTE3>(event);
            StoreOwnerAsync(reinterpret_cast<__gm__ bfloat16_t*>(output) + expert_offset + begin, native, elements);
        } else {
            SetFlag<HardEvent::V_MTE3>(event);
            WaitFlag<HardEvent::V_MTE3>(event);
            StoreOwnerAsync(reinterpret_cast<__gm__ float*>(output) + expert_offset + begin, accum, elements);
        }
        if (following + workers < tiles) SetFlag<HardEvent::MTE3_MTE2>(event);
        buffer = 1 - buffer;
    }
    PipeBarrier<PIPE_ALL>();
}

template <bool NativeHome, bool Buffered>
__aicore__ inline void ReturnOwnerGradients(uint64_t ffts, GM_ADDR source_address,
                             GM_ADDR output_address, GM_ADDR slots_address,
                             GM_ADDR plan_address, int32_t experts,
                             int64_t matrix, int32_t ranks, int32_t slots,
                             int32_t rank, bool bf16, bool publish, bool finish) {
    util_set_ffts_config(ffts);
    if ASCEND_IS_AIC {
        return;
    }
    if (publish) aclshmemx_barrier_all_vec();
    auto* source = reinterpret_cast<__gm__ float*>(source_address);
    auto* output = reinterpret_cast<__gm__ float*>(output_address);
    auto* remote = reinterpret_cast<__gm__ float*>(slots_address);
    auto* plan = reinterpret_cast<__gm__ int32_t*>(plan_address);
    auto* heads = reinterpret_cast<__ubuf__ int32_t*>(Buffered ? kBufferedIndexOffset : kIndexOffset);
    auto* next = heads + (experts + 7) / 8 * 8;
    BuildOwnerLists(plan, heads, next, experts, ranks, slots, rank);
    const int32_t sub_blocks = static_cast<int32_t>(GetSubBlockNum());
    const int32_t worker = sub_blocks > 0 ? GetBlockIdx() / sub_blocks : 0;
    const int32_t workers = GetBlockNum();
    const int64_t tiles = (matrix + kElements - 1) / kElements;
    auto accumulator = LocalAt<float>(kScratch, TPosition::VECIN);

    for (int32_t expert = 0; expert < experts; ++expert) {
        // Owned native gradients can remain in place when no peer contributes.
        // Every worker still participates in both root barriers.
        if (source_address == output_address && heads[expert] < 0) {
            continue;
        }
        if constexpr (Buffered) {
            const int32_t head = heads[expert];
            if (head >= 0 && next[head] < 0) {
                ReturnSinglePeer<NativeHome>(source_address, output_address, remote, expert, head, matrix,
                                              slots, worker, workers, tiles, bf16);
                continue;
            }
        }
        for (int64_t tile = worker; tile < tiles; tile += workers) {
            const int64_t begin = tile * kElements;
            const int32_t elements = matrix - begin < kElements ? matrix - begin : kElements;
            const int64_t offset = static_cast<int64_t>(expert) * matrix + begin;
            const int32_t head = heads[expert];
            if constexpr (NativeHome) {
                auto native = LocalAt<bfloat16_t>(kNativeHomeOffset, TPosition::VECIN);
                auto* native_source = reinterpret_cast<__gm__ bfloat16_t*>(source_address);
                LoadOwner(native, native_source + offset, elements);
                if (head < 0) {
                    StoreOwner(reinterpret_cast<__gm__ bfloat16_t*>(output_address) + offset, native, elements);
                    continue;
                }
                ReadContribution(remote, matrix, slots, head, begin, elements, 0);
                Cast(accumulator, native, RoundMode::CAST_NONE, elements);
                PipeBarrier<PIPE_V>();
            } else {
                LoadOwner(accumulator, source + offset, elements);
                if constexpr (Buffered) {
                    if (head >= 0) ReadContribution(remote, matrix, slots, head, begin, elements, 0);
                }
            }
            int32_t buffer = 0;
            for (int32_t entry = head; entry >= 0; entry = next[entry]) {
                const TEventID ready = buffer == 0 ? EVENT_ID0 : EVENT_ID1;
                const int32_t following = next[entry];
                if constexpr (!Buffered) {
                    ReadContribution(remote, matrix, slots, entry, begin, elements, 0);
                }
                WaitFlag<HardEvent::MTE2_V>(ready);
                if constexpr (Buffered) {
                    if (following >= 0) {
                        const TEventID other = buffer == 0 ? EVENT_ID1 : EVENT_ID0;
                        // Before a slot is reused, wait for its preceding vector Add.
                        if (entry != head) WaitFlag<HardEvent::V_MTE2>(other);
                        ReadContribution(remote, matrix, slots, following, begin, elements, 1 - buffer);
                    }
                }
                const int32_t contribution_offset = buffer == 0 ? kFirstContribution : kIndexOffset;
                auto contribution = LocalAt<float>(contribution_offset, TPosition::VECIN);
                Add(accumulator, accumulator, contribution, elements);
                if constexpr (Buffered) {
                    if (following >= 0 && next[following] >= 0) SetFlag<HardEvent::V_MTE2>(ready);
                    PipeBarrier<PIPE_V>();
                    buffer = 1 - buffer;
                } else {
                    PipeBarrier<PIPE_ALL>();
                }
            }
            if (bf16) {
                auto cast = LocalAt<bfloat16_t>(kScratch + 4*kElements, TPosition::VECOUT);
                Cast(cast, accumulator, RoundMode::CAST_RINT, elements);
                PipeBarrier<PIPE_ALL>();
                GlobalTensor<bfloat16_t> dst;
                dst.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(output_address) + offset);
                DataCopyPad(dst, cast, DataCopyExtParams(1, elements * sizeof(bfloat16_t), 0, 0, 0));
                PipeBarrier<PIPE_ALL>();
            } else {
                StoreOwner(output + offset, accumulator, elements);
            }
        }
    }
    aclshmem_quiet();
    // All readers must complete before a later backward overwrites the slots.
    if (finish) aclshmemx_barrier_all_vec();
}
}  // namespace

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__
void HotReplicaSharedGradReturnFp32(uint64_t ffts, GM_ADDR source, GM_ADDR output, GM_ADDR slots,
                                  GM_ADDR plan, int32_t experts, int64_t matrix, int32_t ranks,
                                  int32_t borrowed, int32_t rank, bool bf16, bool publish, bool finish) {
    ReturnOwnerGradients<false, false>(ffts, source, output, slots, plan, experts, matrix,
                               ranks, borrowed, rank, bf16, publish, finish);
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__
void HotReplicaSharedGradReturnBuffered(uint64_t ffts, GM_ADDR source, GM_ADDR output, GM_ADDR slots,
                                       GM_ADDR plan, int32_t experts, int64_t matrix, int32_t ranks,
                                       int32_t borrowed, int32_t rank, bool native_home, bool bf16,
                                       bool publish, bool finish) {
    if (native_home) {
        ReturnOwnerGradients<true, true>(ffts, source, output, slots, plan, experts, matrix,
                                         ranks, borrowed, rank, true, publish, finish);
    } else {
        ReturnOwnerGradients<false, true>(ffts, source, output, slots, plan, experts, matrix,
                                          ranks, borrowed, rank, bf16, publish, finish);
    }
}

extern "C" ASCEND_HOT_REPLICA_EXPORT bool hmc_hot_replica_buffered_plan_fits(
    int32_t experts, int32_t ranks, int32_t borrowed) {
    const int64_t entries = (static_cast<int64_t>(experts) + 7) / 8 * 8 +
                            static_cast<int64_t>(ranks) * borrowed;
    return experts > 0 && ranks > 0 && borrowed > 0 &&
           kBufferedIndexOffset + entries * sizeof(int32_t) <= kReservedUb;
}

template <bool Buffered>
static int LaunchGradientReturn(
    void* stream, const void* source, void* output, void* slots,
    const int32_t* plan, int32_t experts, int32_t hidden, int32_t projection,
    int32_t ranks, int32_t rank, int32_t blocks, int32_t borrowed, bool native_home, bool bf16,
    bool publish, bool finish) {
    const int64_t index_entries = (experts + 7) / 8 * 8 + ranks * borrowed;
    if (stream == nullptr || source == nullptr || output == nullptr ||
        slots == nullptr || plan == nullptr || experts <= 0 || hidden <= 0 ||
        projection <= 0 || ranks != aclshmem_n_pes() || rank < 0 ||
        rank >= ranks || blocks <= 0 ||
        (Buffered ? !hmc_hot_replica_buffered_plan_fits(experts, ranks, borrowed) :
                    kIndexOffset + index_entries * sizeof(int32_t) > kReservedUb)) {
        return ACL_ERROR_INVALID_PARAM;
    }
    auto* source_bytes = reinterpret_cast<uint8_t*>(const_cast<void*>(source));
    auto* output_bytes = reinterpret_cast<uint8_t*>(output);
    auto* slot_bytes = reinterpret_cast<uint8_t*>(slots);
    auto* plan_bytes = reinterpret_cast<uint8_t*>(const_cast<int32_t*>(plan));
    const int64_t matrix = static_cast<int64_t>(hidden) * projection;
    if constexpr (Buffered) {
        HotReplicaSharedGradReturnBuffered<<<blocks, nullptr, stream>>>(
            shmemx_get_ffts_config(), source_bytes, output_bytes, slot_bytes, plan_bytes,
            experts, matrix, ranks, borrowed, rank, native_home, bf16, publish, finish);
    } else {
        HotReplicaSharedGradReturnFp32<<<blocks, nullptr, stream>>>(
            shmemx_get_ffts_config(), source_bytes, output_bytes, slot_bytes, plan_bytes,
            experts, matrix, ranks, borrowed, rank, bf16, publish, finish);
    }
    return aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL);
}

extern "C" ASCEND_HOT_REPLICA_EXPORT int hmc_hot_replica_grad_return_fp32(
    void* stream, const void* source, void* output, void* slots,
    const int32_t* plan, int32_t experts, int32_t hidden, int32_t projection,
    int32_t ranks, int32_t rank, int32_t blocks, int32_t borrowed, bool bf16, bool publish, bool finish) {
    return LaunchGradientReturn<false>(stream, source, output, slots, plan, experts, hidden, projection,
                                       ranks, rank, blocks, borrowed, false, bf16, publish, finish);
}

extern "C" ASCEND_HOT_REPLICA_EXPORT int hmc_hot_replica_grad_return_buffered(
    void* stream, const void* source, void* output, void* slots,
    const int32_t* plan, int32_t experts, int32_t hidden, int32_t projection,
    int32_t ranks, int32_t rank, int32_t blocks, int32_t borrowed, bool native_home, bool bf16,
    bool publish, bool finish) {
    return LaunchGradientReturn<true>(stream, source, output, slots, plan, experts, hidden, projection,
                                      ranks, rank, blocks, borrowed, native_home, bf16, publish, finish);
}
