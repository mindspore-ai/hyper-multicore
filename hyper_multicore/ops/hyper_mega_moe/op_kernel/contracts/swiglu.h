// Copyright (c) 2026 Huawei Technologies Co., Ltd.
#pragma once
#include <cstdint>
namespace MulticoreMoE {
// The selected standard SwiGLU tiling limits dynamic baseRowLen to 19.
constexpr int64_t kSwiGluDynamicTailBaseRows = 19;
}
