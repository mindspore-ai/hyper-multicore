// Copyright 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>

#if defined(__GNUC__)
#define ASCEND_HOT_REPLICA_EXPORT __attribute__((visibility("default")))
#else
#define ASCEND_HOT_REPLICA_EXPORT
#endif

extern "C" ASCEND_HOT_REPLICA_EXPORT int hmc_hot_replica_prefetch_weight_kernel_do(
    void* stream, uint8_t* home, uint8_t* slots, uint8_t* copies,
    int64_t experts, int64_t hidden, int64_t projection, int64_t local_slots,
    int64_t blocks, bool publish, bool finish);
