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
constexpr int32_t kIndexOffset = kScratch + kElements * 8;

template <typename T>
__aicore__ inline LocalTensor<T> LocalAt(int32_t offset, TPosition position) {
    LocalTensor<T> local;
    local.address_.logicPos = static_cast<uint8_t>(position);
    local.address_.bufferAddr = offset;
    local.address_.dataLen = kElements * sizeof(T);
    return local;
}

template <typename T>
__aicore__ inline void LoadOwner(LocalTensor<T> local, __gm__ T* source, int32_t elements) {
    GlobalTensor<T> global;
    global.SetGlobalBuffer(source);
    DataCopyExtParams copy{1, static_cast<uint32_t>(elements * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padding{false, 0, 0, 0};
    DataCopyPad(local, global, copy, padding);
    PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void StoreOwner(__gm__ float* destination,
                                LocalTensor<float> local, int32_t elements) {
    GlobalTensor<float> global;
    global.SetGlobalBuffer(destination);
    DataCopyExtParams copy{1, static_cast<uint32_t>(elements * sizeof(float)), 0, 0, 0};
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
}  // namespace

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__
void HotReplicaSharedGradReturn(uint64_t ffts, GM_ADDR source_address,
                             GM_ADDR output_address, GM_ADDR slots_address,
                             GM_ADDR plan_address, int32_t experts,
                             int64_t matrix, int32_t ranks, int32_t slots,
                             int32_t rank, bool native_home, bool bf16, bool publish, bool finish) {
    util_set_ffts_config(ffts);
    if ASCEND_IS_AIC {
        return;
    }
    if (publish) aclshmemx_barrier_all_vec();
    auto* source = reinterpret_cast<__gm__ float*>(source_address);
    auto* output = reinterpret_cast<__gm__ float*>(output_address);
    auto* remote = reinterpret_cast<__gm__ float*>(slots_address);
    auto* plan = reinterpret_cast<__gm__ int32_t*>(plan_address);
    auto* heads = reinterpret_cast<__ubuf__ int32_t*>(kIndexOffset);
    auto* next = heads + (experts + 7) / 8 * 8;
    BuildOwnerLists(plan, heads, next, experts, ranks, slots, rank);
    const int32_t sub_blocks = static_cast<int32_t>(GetSubBlockNum());
    const int32_t worker = sub_blocks > 0 ? GetBlockIdx() / sub_blocks : 0;
    const int32_t workers = GetBlockNum();
    const int64_t tiles = (matrix + kElements - 1) / kElements;
    auto accumulator = LocalAt<float>(kScratch, TPosition::VECIN);
    auto contribution = LocalAt<float>(kScratch + 4*kElements, TPosition::VECIN);
    auto native = contribution.ReinterpretCast<bfloat16_t>();
    auto* native_source = reinterpret_cast<__gm__ bfloat16_t*>(source_address);
    for (int32_t expert = 0; expert < experts; ++expert) {
        // Owned native gradients can remain in place when no peer contributes.
        // Every worker still participates in both root barriers.
        if (source_address == output_address && heads[expert] < 0) {
            continue;
        }
        for (int64_t tile = worker; tile < tiles; tile += workers) {
            const int64_t begin = tile * kElements;
            const int32_t elements = matrix - begin < kElements ? matrix - begin : kElements;
            const int64_t offset = static_cast<int64_t>(expert) * matrix + begin;
            if (native_home) {
                LoadOwner(native, native_source + offset, elements);
                Cast(accumulator, native, RoundMode::CAST_NONE, elements);
                // The peer read reuses the native input tile after conversion.
                PipeBarrier<PIPE_ALL>();
            } else {
                LoadOwner(accumulator, source + offset, elements);
            }
            if (heads[expert] >= 0) {
                for (int32_t rb = heads[expert]; rb >= 0; rb = next[rb]) {
                    const int32_t peer = rb / slots;
                    const int32_t slot = rb % slots;
                    const int64_t peer_offset = static_cast<int64_t>(slot) * matrix + begin;
                    aclshmemx_mte_get_nbi(
                        reinterpret_cast<__ubuf__ float*>(kScratch + 4*kElements),
                        remote + peer_offset, elements, peer, EVENT_ID0);
                    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                    Add(accumulator, accumulator, contribution, elements);
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

extern "C" ASCEND_HOT_REPLICA_EXPORT int hmc_hot_replica_grad_return(
    void* stream, const void* source, void* output, void* slots,
    const int32_t* plan, int32_t experts, int32_t hidden, int32_t projection,
    int32_t ranks, int32_t rank, int32_t blocks, int32_t borrowed,
    bool native_home, bool bf16, bool publish, bool finish) {
    const int64_t index_entries = (static_cast<int64_t>(experts) + 7) / 8 * 8 +
                                  static_cast<int64_t>(ranks) * borrowed;
    if (stream == nullptr || source == nullptr || output == nullptr ||
        slots == nullptr || plan == nullptr || experts <= 0 || hidden <= 0 ||
        projection <= 0 || ranks != aclshmem_n_pes() || rank < 0 ||
        rank >= ranks || blocks <= 0 || borrowed < 0 || borrowed > experts || (native_home && !bf16) ||
        kIndexOffset + index_entries * sizeof(int32_t) > kReservedUb) {
        return ACL_ERROR_INVALID_PARAM;
    }
    HotReplicaSharedGradReturn<<<blocks, nullptr, stream>>>(
        shmemx_get_ffts_config(),
        reinterpret_cast<uint8_t*>(const_cast<void*>(source)),
        reinterpret_cast<uint8_t*>(output), reinterpret_cast<uint8_t*>(slots),
        reinterpret_cast<uint8_t*>(const_cast<int32_t*>(plan)), experts,
        static_cast<int64_t>(hidden) * projection, ranks, borrowed, rank, native_home, bf16, publish, finish);
    return aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL);
}
