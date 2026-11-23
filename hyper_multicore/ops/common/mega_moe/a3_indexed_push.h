// Copyright 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <stdint.h>

namespace MulticoreMoE {
constexpr uint32_t A3_INDEXED_TOP_K = 8;
constexpr uint32_t A3_INDEXED_MAX_TOKENS = 16384;
constexpr int64_t A3_INDEXED_SOURCE_BITS = 17;
constexpr int64_t A3_INDEXED_SOURCE_MASK = (int64_t{1} << 16) - 1;
constexpr int64_t A3_INDEXED_ROUTE_FENCED = int64_t{1} << 16;
constexpr int32_t A3_INDEXED_TILING_SOURCE_ROWS = 8;
constexpr int32_t A3_INDEXED_TILING_ROUTE_FENCED = 9;
}  // namespace MulticoreMoE
