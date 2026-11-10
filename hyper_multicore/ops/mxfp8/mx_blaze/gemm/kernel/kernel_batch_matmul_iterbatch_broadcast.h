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
 * \file kernel_batch_matmul_iterbatch_broadcast.h
 * \brief GemmUniversal partial specialization for IterBatch-Broadcast path
 *        Combines BmmBroadcast % operator with iterbatch L1/L0 pipelining
 *        Uses BlockSchedulerIterBatchBroadcast for batch grouping
 *        Uses BlockMmad<MatmulIterBatchBroadcast<ABc,BBc>> for iterbatch MMAD
 *        Broadcast side determined at compile time via DispatchPolicy template params
 *        Kernel does gmA/gmB Slice for broadcast mapping before calling MMAD block
 */

#pragma once

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif

#include "mxfp8/mx_blaze/epilogue/block/block_epilogue_empty.h"
#include "mxfp8/mx_blaze/gemm/block/block_mmad.h"
#include "mxfp8/mx_blaze/gemm/block/block_mmad_iterbatch_broadcast.h"
#include "mxfp8/mx_blaze/gemm/block/block_scheduler_iterbatch_broadcast.h"
#include "mxfp8/mx_blaze/gemm/utils/common_utils.h"
#include "kernel_universal.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Kernel {

template <class ProblemShape_, class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>
class GemmUniversal<ProblemShape_, BlockMmad_, BlockEpilogue_, BlockScheduler_,
                    AscendC::Std::enable_if_t<AscendC::Std::is_same_v<
                        KernelIterBatchBroadcast, typename BlockMmad_::DispatchPolicy::ScheduleType>>> {
public:
    __aicore__ inline GemmUniversal() {}
    __aicore__ inline ~GemmUniversal() {}

    using BlockMmad = BlockMmad_;
    using ProblemShape = ProblemShape_;
    using BlockScheduler = BlockScheduler_;
    using BlockEpilogue = BlockEpilogue_;
    using DispatchPolicy = typename BlockMmad::DispatchPolicy;
    using BlockMmadParams = typename BlockMmad::Params;
    using BlockEpilogueParams = typename BlockEpilogue::Params;
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using CType = typename BlockMmad::CType;
    using BiasType = typename BlockMmad::BiasType;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;
    using LayoutC = typename BlockMmad::LayoutC;
    using LayoutBias = typename BlockMmad::LayoutBias;
    using TupleShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using MakeLayoutA = asc::te::frame_layout_format<LayoutA, AscendC::Std::Int<asc::te::c0_element<AType>>>;
    using MakeLayoutB = asc::te::frame_layout_format<LayoutB, AscendC::Std::Int<asc::te::c0_element<BType>>>;
    using MakeLayoutC = asc::te::frame_layout_format<LayoutC, AscendC::Std::Int<asc::te::c0_element<CType>>>;
    using MakeLayoutBias = asc::te::frame_layout_format<LayoutBias, AscendC::Std::Int<asc::te::c0_element<BiasType>>>;
    using BlockSchedulerParams = typename Block::BlockSchedulerIterBatchBroadcast<ProblemShape>::Params;
    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        BlockEpilogueParams epilogueParams;
        BlockSchedulerParams schedulerParams;
        Params() = default;
    };

    __aicore__ inline void Init(Params const& params)
    {
        m_ = static_cast<uint64_t>(asc::te::get<MNK_M>(params.problemShape));
        n_ = static_cast<uint64_t>(asc::te::get<MNK_N>(params.problemShape));
        k_ = static_cast<uint64_t>(asc::te::get<MNK_K>(params.problemShape));
        b_ = static_cast<uint64_t>(params.schedulerParams.cBatchDim0) *
             static_cast<uint64_t>(params.schedulerParams.cBatchDim1) *
             static_cast<uint64_t>(params.schedulerParams.cBatchDim2) *
             static_cast<uint64_t>(params.schedulerParams.cBatchDim3);
        aGmAddr_ = reinterpret_cast<__gm__ AType*>(params.mmadParams.aGmAddr);
        bGmAddr_ = reinterpret_cast<__gm__ BType*>(params.mmadParams.bGmAddr);
        cGmAddr_ = reinterpret_cast<__gm__ CType*>(params.mmadParams.cGmAddr);
        if (params.mmadParams.biasGmAddr != nullptr) {
            isBias_ = true;
            biasGmAddr_ = reinterpret_cast<__gm__ BiasType*>(params.mmadParams.biasGmAddr);
        }
    }

    __aicore__ inline void operator()(Params const& params)
    {
        if ASCEND_IS_AIV {
            return;
        }
        BlockMmad blockMmad;
        int64_t curBlockIdx = AscendC::GetBlockIdx();
        int64_t coreNums = AscendC::GetBlockNum();
        Init(params);
        const BlockSchedulerParams& schP = params.schedulerParams;
        Block::BlockSchedulerIterBatchBroadcast<ProblemShape> bs(params.problemShape, schP);
        int64_t blockNums = bs.GetBlockNums();
        int64_t realCoreNums = bs.GetCoreNums(coreNums);
        if (curBlockIdx >= realCoreNums) {
            return;
        }
        bool broadcastSingleBatch = bs.IsBroadcastSideSingleBatch();
        bool aSingleBatch = A_BROADCAST && broadcastSingleBatch;
        bool bSingleBatch = B_BROADCAST && broadcastSingleBatch;
        Blaze::Gemm::SetHF32(params.schedulerParams.isHf32);
        BlockMmadParams mmadParams = params.mmadParams;
        mmadParams.aBroadcastSingleBatch = aSingleBatch;
        mmadParams.bBroadcastSingleBatch = bSingleBatch;
        blockMmad.Init(mmadParams);
        uint64_t totalABatches = static_cast<uint64_t>(
            params.schedulerParams.aBatchDim0 * params.schedulerParams.aBatchDim1 * params.schedulerParams.aBatchDim2 *
            params.schedulerParams.aBatchDim3);
        uint64_t totalBBatches = static_cast<uint64_t>(
            params.schedulerParams.bBatchDim0 * params.schedulerParams.bBatchDim1 * params.schedulerParams.bBatchDim2 *
            params.schedulerParams.bBatchDim3);
        uint64_t totalCBatches = static_cast<uint64_t>(
            params.schedulerParams.cBatchDim0 * params.schedulerParams.cBatchDim1 * params.schedulerParams.cBatchDim2 *
            params.schedulerParams.cBatchDim3);

        auto layoutA3D = asc::te::make_frame_layout<LayoutA>(totalABatches, m_, k_);
        auto layoutB3D = asc::te::make_frame_layout<LayoutB>(totalBBatches, k_, n_);
        auto layoutC3D = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(totalCBatches, m_, n_);
        auto layoutBias = MakeLayoutBias{}(1L, n_);
        auto gmA = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(aGmAddr_), layoutA3D);
        auto gmB = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(bGmAddr_), layoutB3D);
        auto gmC = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(cGmAddr_), layoutC3D);
        auto gmBias = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(biasGmAddr_), layoutBias);
        for (int64_t blockIdx = curBlockIdx; blockIdx < blockNums; blockIdx += coreNums) {
            auto blockShape = bs.GetBlockShape(blockIdx, blockNums);
            auto blockCoord = bs.GetBlockCoord(blockIdx);
            uint64_t startBatchIdx = static_cast<uint64_t>(asc::te::get<MNK_B>(blockCoord));
            uint64_t curIterBatchL1 = static_cast<uint64_t>(asc::te::get<MNK_B>(blockShape));
            uint64_t aGmStartBatch = static_cast<uint64_t>(bs.ComputeABroadcastIndex(startBatchIdx));
            uint64_t bGmStartBatch = static_cast<uint64_t>(bs.ComputeBBroadcastIndex(startBatchIdx));
            uint64_t al1Count = aSingleBatch ? 1UL : curIterBatchL1;
            uint64_t bl1Count = bSingleBatch ? 1UL : curIterBatchL1;
            uint64_t agmStart = A_BROADCAST ? aGmStartBatch : startBatchIdx;
            uint64_t bgmStart = B_BROADCAST ? bGmStartBatch : startBatchIdx;
            auto gmASlice = gmA.slice(asc::te::make_coord(agmStart, asc::te::make_coord(0, 0)),
                                      asc::te::make_shape(al1Count, asc::te::make_shape(m_, k_)));
            auto gmBSlice = gmB.slice(asc::te::make_coord(bgmStart, asc::te::make_coord(0, 0)),
                                      asc::te::make_shape(bl1Count, asc::te::make_shape(k_, n_)));
            auto gmCSlice = gmC.slice(asc::te::make_coord(startBatchIdx, asc::te::make_coord(0, 0)),
                                      asc::te::make_shape(curIterBatchL1, asc::te::make_shape(m_, n_)));
            blockMmad(gmASlice, gmBSlice, gmBias, gmCSlice, curIterBatchL1);
        }
        Blaze::Gemm::UnsetHF32(params.schedulerParams.isHf32);
    }

private:
    static constexpr bool A_BROADCAST = DispatchPolicy::A_BROADCAST;
    static constexpr bool B_BROADCAST = DispatchPolicy::B_BROADCAST;
    bool isBias_ = false;
    uint64_t m_{1};
    uint64_t n_{1};
    uint64_t k_{1};
    uint64_t b_{1};
    __gm__ AType* aGmAddr_;
    __gm__ BType* bGmAddr_;
    __gm__ CType* cGmAddr_;
    __gm__ BiasType* biasGmAddr_ = nullptr;
};

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
