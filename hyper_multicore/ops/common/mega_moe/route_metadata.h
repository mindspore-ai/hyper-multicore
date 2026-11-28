// Copyright 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include "kernel_operator.h"
#include "route_validation.h"

// Shared lowering of the current [source,expert] count table. The exact-count
// and bounded entries use the same offsets, units, partitioning and stores.
namespace hyper_multicore::moe_route {
using namespace AscendC;

__aicore__ inline int64_t Aligned(int64_t n) { return (n + 7) / 8 * 8; }
template <typename T>
__aicore__ inline void Load(LocalTensor<T> local, GM_ADDR address, int64_t count) {
    GlobalTensor<T> global;
    global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(address));
    DataCopyExtParams copy{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> pad{false, 0, 0, 0};
    DataCopyPad(local, global, copy, pad);
    SetFlag<HardEvent::MTE2_S>(0);
    WaitFlag<HardEvent::MTE2_S>(0);
}

template <typename T>
__aicore__ inline void Store(GM_ADDR address, LocalTensor<T> local, int64_t count) {
    GlobalTensor<T> global;
    global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(address));
    SetFlag<HardEvent::S_MTE3>(0);
    WaitFlag<HardEvent::S_MTE3>(0);
    DataCopyExtParams copy{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
    DataCopyPad(global, local, copy);
    SetFlag<HardEvent::MTE3_S>(0);
    WaitFlag<HardEvent::MTE3_S>(0);
}

// The row prefix is computed once. Reuse its scratch for the element units
// after the row store completes; no larger UB contract or second count scan.
template <typename T>
__aicore__ inline void StoreUnits(GM_ADDR rows, LocalTensor<T> values, int64_t count,
                                 int32_t hidden, int64_t unit_bytes) {
    Store(rows, values, count);
    if (hidden != 1) {
        for (int64_t i = 0; i < count; ++i) values.SetValue(i, values.GetValue(i) * hidden);
        Store(rows + unit_bytes, values, count);
    }
}

// Independent integer accumulators shorten the scalar dependency chain.
// Every count still reaches the caller's validation policy; int64 sums retain
// the full count domain, including malformed rows rejected at SourceTotal.
template <typename Validation>
__aicore__ inline int64_t SumCounts(LocalTensor<int32_t> counts, int64_t base,
    int64_t step, int32_t begin, int32_t end, int32_t hidden, Validation& validation) {
    int64_t a = 0, b = 0, c = 0, d = 0;
    int32_t i = begin;
    for (; i + 3 < end; i += 4) {
        const int32_t n0 = counts.GetValue(base + i * step);
        const int32_t n1 = counts.GetValue(base + (i + 1) * step);
        const int32_t n2 = counts.GetValue(base + (i + 2) * step);
        const int32_t n3 = counts.GetValue(base + (i + 3) * step);
        validation.Count(n0, hidden);
        validation.Count(n1, hidden);
        validation.Count(n2, hidden);
        validation.Count(n3, hidden);
        a += n0; b += n1; c += n2; d += n3;
    }
    for (; i < end; ++i) {
        const int32_t n = counts.GetValue(base + i * step);
        validation.Count(n, hidden);
        a += n;
    }
    return (a + b) + (c + d);
}

template <bool Summarize, typename Validation>
__global__ __aicore__ void Metadata(GM_ADDR input, GM_ADDR offsets, GM_ADDR sizes,
    int32_t p, int32_t e, int32_t rank, int32_t hidden, int32_t stride, int32_t group,
    int32_t programs, int32_t bytes, int32_t row_stride, int32_t header,
    Validation validation, GM_ADDR summary = nullptr) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    TBuf<TPosition::VECCALC> buffer;
    pipe.InitBuffer(buffer, bytes);
    auto memory = buffer.Get<int32_t>();
    const int64_t span = (p - 1) * row_stride + e + header;
    auto counts = memory;
    auto out = memory[Aligned(span)].ReinterpretCast<int64_t>();
    auto length = memory[Aligned(span) + 2 * stride];
    auto groups = memory[Aligned(span) + 3 * stride].ReinterpretCast<int64_t>();
    const int32_t local = e / p;
    const int32_t block = GetBlockIdx();
    const int64_t offset_unit_bytes = static_cast<int64_t>(4 * stride + local) * 8;
    const int64_t size_unit_bytes = static_cast<int64_t>(2 * stride) * 4;
    if (block < programs) {
        const int32_t begin = block * group;
        const int32_t owned = p - begin < group ? p - begin : group;
        // Summary also consumes destination columns. Other entries need only
        // their source rows; preserve the global mask separately on later groups.
        const int32_t prefix_words = header && begin ? 8 : 0;
        if constexpr (Summarize) {
            Load(counts, input, span);
        } else {
            Load(counts[prefix_words], input + static_cast<int64_t>(begin) * row_stride * 4,
                 static_cast<int64_t>(owned - 1) * row_stride + e + header);
            if (prefix_words) Load(counts, input, header);
        }
        for (int32_t source = begin; source < begin + owned; ++source) {
            const int64_t base = Summarize ? static_cast<int64_t>(source) * row_stride :
                prefix_words + static_cast<int64_t>(source - begin) * row_stride;
            validation.Source(counts, base);
            int64_t prefix = 0;
            int32_t minimum = INT32_MAX, maximum = INT32_MIN;
            if (Summarize || source == rank) {
                for (int32_t expert = 0; expert < e; ++expert) {
                    const int32_t n = counts.GetValue(base + header + expert);
                    validation.Count(n, hidden);
                    if constexpr (Summarize) {
                        minimum = n < minimum ? n : minimum;
                        maximum = n > maximum ? n : maximum;
                    }
                    out.SetValue(expert, prefix);
                    if (source == rank) length.SetValue(expert, n);
                    prefix += n;
                }
                if constexpr (!Summarize) {
                    for (int32_t j = 0; j < local; ++j)
                        groups.SetValue(j, out.GetValue(rank * local + j));
                }
            } else {
                // Remote sources publish only this rank's expert slice.
                const int32_t first = rank * local;
                prefix = SumCounts(counts, base + header, 1, 0, first, hidden, validation);
                for (int32_t j = 0; j < local; ++j) {
                    const int32_t n = counts.GetValue(base + header + first + j);
                    validation.Count(n, hidden);
                    groups.SetValue(j, prefix);
                    prefix += n;
                }
                prefix += SumCounts(counts, base + header, 1, first + local, e, hidden, validation);
            }
            validation.SourceTotal(prefix);
            if constexpr (Summarize) {
                int64_t received = 0;
                bool active = true;
                for (int32_t j = 0; j < local; ++j) {
                    int64_t expert_rows = 0;
                    for (int32_t peer = 0; peer < p; ++peer) {
                        expert_rows += counts.GetValue(peer * row_stride + header + source * local + j);
                    }
                    received += expert_rows;
                    active = active && expert_rows > 0;
                }
                for (int32_t j = 0; j < 3; ++j) groups.SetValue(j, counts.GetValue(base + j));
                groups.SetValue(3, prefix);
                groups.SetValue(4, minimum);
                groups.SetValue(5, maximum);
                groups.SetValue(6, received);
                groups.SetValue(7, active);
                Store(summary + static_cast<int64_t>(source) * 8 * sizeof(int64_t), groups, 8);
            }
            if constexpr (Summarize) {
                for (int32_t j = 0; j < local; ++j) groups.SetValue(j, out.GetValue(rank * local + j));
            }
            StoreUnits(offsets + (3 * stride + source * local) * 8, groups, local, hidden, offset_unit_bytes);
            if (source == rank) {
                StoreUnits(offsets, out, e, hidden, offset_unit_bytes);
                StoreUnits(sizes, length, e, hidden, size_unit_bytes);
            }
        }
    } else {
        const int32_t begin = (block - programs) * group;
        const int32_t owned = p - begin < group ? p - begin : group;
        // Load just this owner's destination columns. DataCopyPad rounds each
        // local row to 32 bytes; external row stride and padding stay explicit.
        const int32_t width = owned * local;
        const int64_t local_stride = Aligned(width);
        const bool compact_columns = programs > 1 && p * local_stride <= Aligned(span);
        if (compact_columns) {
            GlobalTensor<int32_t> global;
            global.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(input) + header + begin * local);
            const DataCopyExtParams copy{static_cast<uint16_t>(p), static_cast<uint32_t>(width * 4),
                                        static_cast<uint32_t>((row_stride - width) * 4), 0, 0};
            const DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
            DataCopyPad(counts, global, copy, pad);
            SetFlag<HardEvent::MTE2_S>(0);
            WaitFlag<HardEvent::MTE2_S>(0);
        } else {
            // Tiny unpadded matrices can require more row padding than the
            // existing scratch contract. Keep their original packed traversal.
            Load(counts, input, span);
        }
        for (int32_t dest = begin; dest < p && dest < begin + group; ++dest) {
            int64_t prefix = 0;
            for (int32_t j = 0; j < local; ++j) {
                int64_t source_prefix = 0;
                const int64_t step = compact_columns ? local_stride : row_stride;
                const int64_t base = compact_columns ? (dest - begin) * local + j :
                    header + dest * local + j;
                if (dest == rank) {
                    for (int32_t source = 0; source < p; ++source) {
                        const int32_t n = counts.GetValue(base + source * step);
                        out.SetValue(source * local + j, prefix + source_prefix);
                        length.SetValue(source * local + j, n);
                        source_prefix += n;
                    }
                } else {
                    // Source blocks validate all counts. This owner needs two
                    // integer reductions, rather than the remote full prefix.
                    ExactRouteValidation unchecked;
                    source_prefix = SumCounts(counts, base, step, 0, rank, hidden, unchecked);
                    groups.SetValue(j, prefix + source_prefix);
                    source_prefix += SumCounts(counts, base, step, rank, p, hidden, unchecked);
                }
                prefix += source_prefix;
                if (dest == rank) groups.SetValue(j, prefix);
            }
            validation.DestinationTotal(prefix);
            if (dest == rank) {
                // GMM group_list always remains in row units in both views.
                Store(offsets + 4 * stride * 8, groups, local);
                if (hidden != 1) Store(offsets + offset_unit_bytes + 4 * stride * 8, groups, local);
            }
            if (dest == rank) {
                for (int32_t j = 0; j < local; ++j) groups.SetValue(j, out.GetValue(rank * local + j));
            }
            StoreUnits(offsets + (stride + dest * local) * 8, groups, local, hidden, offset_unit_bytes);
            if (dest == rank) {
                StoreUnits(offsets + 2 * stride * 8, out, e, hidden, offset_unit_bytes);
                StoreUnits(sizes + stride * 4, length, e, hidden, size_unit_bytes);
            }
        }
    }
}
}  // namespace hyper_multicore::moe_route
