// Copyright 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include "kernel_operator.h"

namespace hyper_multicore::moe_route {
using AscendC::LocalTensor;

// The exact-count path validates the returned summary on Host. Its device
// metadata traversal carries no receive bound or per-count checks.
struct ExactRouteValidation {
    __aicore__ inline void Source(LocalTensor<int32_t>, int64_t) const {}
    __aicore__ inline void Count(int32_t, int32_t) const {}
    __aicore__ inline void SourceTotal(int64_t) const {}
    __aicore__ inline void DestinationTotal(int64_t) const {}
};

// Only the explicitly enabled no-D2H entry constructs this policy. Every
// source and destination is checked by its existing metadata owner block;
// dependent expert work waits for the complete metadata kernel.
struct BoundedRouteValidation {
    int64_t capacity;
    int32_t tokens;
    int32_t topk;

    __aicore__ inline void Source(LocalTensor<int32_t> counts, int64_t base) const {
        const int32_t mask = counts.GetValue(2);
        if (mask < 0 || mask >= 32 || counts.GetValue(base) != 0 ||
            counts.GetValue(base + 1) != tokens || counts.GetValue(base + 2) != mask) AscendC::Trap();
    }
    __aicore__ inline void Count(int32_t n, int32_t hidden) const {
        if (n < 0 || n > tokens || static_cast<int64_t>(n) * hidden > INT32_MAX) AscendC::Trap();
    }
    __aicore__ inline void SourceTotal(int64_t total) const {
        if (total != static_cast<int64_t>(tokens) * topk) AscendC::Trap();
    }
    __aicore__ inline void DestinationTotal(int64_t total) const {
        if (total > capacity) AscendC::Trap();
    }
};
}  // namespace hyper_multicore::moe_route
