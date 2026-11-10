/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sync.h
 * \brief Cross-core synchronization primitives for AIC<->AIV handshake.
 */
#pragma once

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "c_api/sync/sync.h"

// bisheng --asc-aicore-lang 混合编译（host/device 双边）时，host 不定义 __CCE_AICORE__ /  __NPU_ARCH__，
// 编译器内置 pipe_t 枚举中 PIPE_FIX 成员被条件编译裁剪（cce_aicore_intrinsics.h 中
// PIPE_FIX 的定义条件即 __CCE_AICORE__ >= 210），而本文件的模板默认实参
// （pipe_t Pipe = PIPE_FIX）在 host 同样需要解析，导致上层程序混合编译报
// "use of undeclared identifier 'PIPE_FIX'"。
// 因此守卫条件必须用 ifndef __CCE_AICORE__ 或 __NPU_ARCH__ 表示非device编译：
// 此时若未定义PIPE_FIX 依旧缺失，则添加新的定义兜底。
// device （__NPU_ARCH__ 已定义）与 UT（显式 -D__CCE_AICORE__=310 且 stub_fun.h
// 提供完整枚举）均使用真实 PIPE_FIX 枚举成员。取值与 cce_aicore_intrinsics.h 定义一致
// （PIPE_FIX = 10）。必须位于上述头文件 include 之后，保证 pipe_t 类型已定义。
#ifndef __NPU_ARCH__
#ifndef PIPE_FIX
#define PIPE_FIX (pipe_t)(10)
#endif
#endif

namespace Blaze {
namespace Gemm {
namespace Sync {

// 同步模式与 syncID 定义集中于此。
constexpr uint16_t SYNC_MODE_BLOCK = 0; // 对应 asc_sync_block_arrive / asc_sync_block_wait
constexpr uint16_t SYNC_MODE_INTER = 2; // 对应 asc_sync_inter_arrive / asc_sync_inter_wait
constexpr uint16_t SYNC_MODE_INTRA = 4; // 对应 asc_sync_intra_arrive / asc_sync_intra_wait

constexpr uint16_t MIX_AIC_SYNC_AIV_FLAG = 0;
constexpr uint16_t MIX_AIV_SYNC_AIC_FLAG = 1;
constexpr uint16_t MIX_FLAG_ID_MAX = 16;

constexpr uint16_t NPU_ARCH_950 = 950;
constexpr uint16_t NPU_ARCH_960 = 960;

namespace detail {

template <uint16_t Mode, pipe_t Pipe>
__aicore__ inline void Arrive(uint16_t id)
{
    if constexpr (Mode == SYNC_MODE_INTRA) { // 默认 Mode = 4
        asc_sync_intra_arrive(Pipe, id);
    } else if constexpr (Mode == SYNC_MODE_INTER) { // Mode = 2
        asc_sync_inter_arrive(Pipe, id);
    } else {
        asc_sync_block_arrive(Pipe, id);
    }
}

template <uint16_t Mode, pipe_t Pipe>
__aicore__ inline void Wait(uint16_t id)
{
    if constexpr (Mode == SYNC_MODE_INTRA) {
        asc_sync_intra_wait(Pipe, id);
    } else if constexpr (Mode == SYNC_MODE_INTER) {
        asc_sync_inter_wait(Pipe, id);
    } else {
        asc_sync_block_wait(Pipe, id);
    }
}

} // namespace detail

template <uint16_t Mode = SYNC_MODE_INTRA, pipe_t Pipe = PIPE_FIX, uint16_t NpuArch = NPU_ARCH_950>
__aicore__ inline void NotifyVector(uint16_t id = MIX_AIC_SYNC_AIV_FLAG, bool isMixCV1V2 = false)
{
    detail::Arrive<Mode, Pipe>(id);
    if constexpr (NpuArch == NPU_ARCH_950) {
        if (isMixCV1V2) {
            detail::Arrive<Mode, Pipe>(id + MIX_FLAG_ID_MAX);
        }
    }
}

template <uint16_t Mode = SYNC_MODE_INTRA, pipe_t Pipe = PIPE_FIX, uint16_t NpuArch = NPU_ARCH_950>
__aicore__ inline void WaitForVector(uint16_t id = MIX_AIV_SYNC_AIC_FLAG, bool isMixCV1V2 = false)
{
    detail::Wait<Mode, Pipe>(id);
    if constexpr (NpuArch == NPU_ARCH_950) {
        if (isMixCV1V2) {
            detail::Wait<Mode, Pipe>(id + MIX_FLAG_ID_MAX);
        }
    }
}

template <uint16_t Mode = SYNC_MODE_INTRA, pipe_t Pipe = PIPE_V>
__aicore__ inline void NotifyCube(uint16_t id = MIX_AIV_SYNC_AIC_FLAG)
{
    detail::Arrive<Mode, Pipe>(id);
}

template <uint16_t Mode = SYNC_MODE_INTRA, pipe_t Pipe = PIPE_V>
__aicore__ inline void WaitForCube(uint16_t id = MIX_AIC_SYNC_AIV_FLAG)
{
    detail::Wait<Mode, Pipe>(id);
}

} // namespace Sync
} // namespace Gemm
} // namespace Blaze
