// Copyright 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
extern "C" int hmc_hot_replica_allocate(void*, void*, void*, void*, void*, void*,
    int64_t, int64_t, int64_t, int64_t, void*, uint32_t, int64_t);
extern "C" int hmc_hot_replica_render(void*, void*, void*, void*, void*, void*, void*,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);
