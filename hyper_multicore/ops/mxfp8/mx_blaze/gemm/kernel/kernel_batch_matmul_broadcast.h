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
 * \file kernel_batch_matmul_broadcast.h
 * \brief
 */

#pragma once

#include "kernel_basic_intf.h"

#include "mxfp8/mx_blaze/epilogue/block/block_epilogue_empty.h"
#include "mxfp8/mx_blaze/gemm/block/block_mmad.h"
#include "mxfp8/mx_blaze/gemm/block/block_mmad_matmul_basic.h"
#include "mxfp8/mx_blaze/gemm/utils/common_utils.h"
#include "kernel_universal.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Kernel {

template <class ProblemShape_, class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>
class GemmUniversal<ProblemShape_, BlockMmad_, BlockEpilogue_, BlockScheduler_,
                    AscendC::Std::enable_if_t<AscendC::Std::is_same_v<
                        KernelMmadMultiBlockBmmBroadcast, typename BlockMmad_::DispatchPolicy::ScheduleType>>> {
public:
    __aicore__ inline GemmUniversal() {}
    __aicore__ inline ~GemmUniversal() {}

    using BlockMmad = BlockMmad_;
    using ProblemShape = ProblemShape_;
    using BlockScheduler = BlockScheduler_;
    using BlockEpilogue = BlockEpilogue_;

    // mmad
    using BlockMmadParams = typename BlockMmad::Params;
    using BlockEpilogueParams = typename BlockEpilogue::Params;
    using BlockSchedulerParams = typename BlockScheduler::Params;
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

    struct BatchInfo {
        uint32_t aBatchDim0 = 1UL;
        uint32_t bBatchDim0 = 1UL;
        uint32_t aBatchDim1 = 1UL;
        uint32_t bBatchDim1 = 1UL;
        uint32_t cBatchDim1 = 1UL;
        uint32_t aBatchDim2 = 1UL;
        uint32_t bBatchDim2 = 1UL;
        uint32_t cBatchDim2 = 1UL;
        uint32_t aBatchDim3 = 1UL;
        uint32_t bBatchDim3 = 1UL;
        uint32_t cBatchDim3 = 1UL;
        uint32_t biasBatchDimAll = 1UL;
    };

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        BlockEpilogueParams epilogueParams;
        BlockSchedulerParams schParams;
        BatchInfo batchInfo;
        Params() = default;
    };

    __aicore__ inline void operator()(Params const& params)
    {
        if ASCEND_IS_AIV {
            return;
        }
        Init(params);

        // 初始化blockScheduler
        BlockScheduler bs(params.problemShape, params.schParams);
        int64_t curBlockIdx = AscendC::GetBlockIdx();
        int64_t realCoreNums = bs.GetCoreNums(); // 实际需要的核数
        if (curBlockIdx >= realCoreNums) {
            return;
        }

        Blaze::Gemm::SetHF32(params.schParams.isHf32);

        BlockMmad blockMmad;
        blockMmad.Init(params.mmadParams);

        uint64_t totalABatches = static_cast<uint64_t>(params.batchInfo.aBatchDim0) * params.batchInfo.aBatchDim1 *
                                 params.batchInfo.aBatchDim2 * params.batchInfo.aBatchDim3;
        uint64_t totalBBatches = static_cast<uint64_t>(params.batchInfo.bBatchDim0) * params.batchInfo.bBatchDim1 *
                                 params.batchInfo.bBatchDim2 * params.batchInfo.bBatchDim3;

        auto layoutA = MakeLayoutA{}(totalABatches, m_, k_);
        auto layoutB = MakeLayoutB{}(totalBBatches, k_, n_);
        auto layoutC = MakeLayoutC{}(batch_, m_, n_);
        auto layoutBias = MakeLayoutBias{}(params.batchInfo.biasBatchDimAll, 1L, n_);
        // A,B,C Gm Tensor
        auto gmA = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(aGmAddr_), layoutA);
        auto gmB = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(bGmAddr_), layoutB);
        auto gmC = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(cGmAddr_), layoutC);
        auto gmBias = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(biasGmAddr_), layoutBias);

        uint64_t preBatchIdx = batch_;
        int64_t totalBlockNums = bs.GetBlockNums(); // 切分总块数
        int64_t coreNums = AscendC::GetBlockNum();  // 实际启用的物理核数
        // Process tiles in ping-pong mode
        for (int64_t tileIdx = curBlockIdx; tileIdx < totalBlockNums; tileIdx += coreNums) {
            auto tileShape = bs.template GetBlockShape<TRANS_B, BType>(tileIdx); // 非全载
            auto tileCoord = bs.GetBlockCoord(tileIdx);                          // (m, n, k, b)
            auto coordM = asc::te::get<MNK_M>(tileCoord);
            auto coordN = asc::te::get<MNK_N>(tileCoord);
            auto shapeM = asc::te::get<MNK_M>(tileShape);
            auto shapeN = asc::te::get<MNK_N>(tileShape);
            auto shapeK = asc::te::get<MNK_K>(tileShape);
            curBatchIdx_ = static_cast<uint64_t>(asc::te::get<MNK_B>(tileCoord));

            if (preBatchIdx != curBatchIdx_) {
                UpdateBatchIndex(params.batchInfo);
                preBatchIdx = curBatchIdx_;
            }
            // Block offset
            auto subTensorA = gmA.slice(AscendC::MakeCoord(batchAIndex_, AscendC::MakeCoord(coordM, 0L)),
                                        AscendC::MakeShape(1L, AscendC::MakeShape(shapeM, shapeK)));
            auto gmBlockA = asc::te::squeeze<0>(subTensorA);
            auto subTensorB = gmB.slice(AscendC::MakeCoord(batchBIndex_, AscendC::MakeCoord(0L, coordN)),
                                        AscendC::MakeShape(1L, AscendC::MakeShape(shapeK, shapeN)));
            auto gmBlockB = asc::te::squeeze<0>(subTensorB);
            auto subTensorC = gmC.slice(AscendC::MakeCoord(curBatchIdx_, AscendC::MakeCoord(coordM, coordN)),
                                        AscendC::MakeShape(1L, AscendC::MakeShape(shapeM, shapeN)));
            auto gmBlockC = asc::te::squeeze<0>(subTensorC);
            uint64_t biasBatchIdx = params.batchInfo.biasBatchDimAll == 1UL ? 0UL : curBatchIdx_;
            auto subTensorBias = gmBias.slice(AscendC::MakeCoord(biasBatchIdx, AscendC::MakeCoord(0L, coordN)),
                                              AscendC::MakeShape(1L, AscendC::MakeShape(1L, shapeN)));
            auto gmBlockBias = asc::te::squeeze<0>(subTensorBias);
            blockMmad(gmBlockA, gmBlockB, gmBlockBias, gmBlockC, tileShape);
        }

        Blaze::Gemm::UnsetHF32(params.schParams.isHf32);
    }

private:
    __aicore__ inline void Init(Params const& params)
    {
        auto blockMmadParams = params.mmadParams;
        m_ = static_cast<uint64_t>(asc::te::get<MNK_M>(params.problemShape));
        n_ = static_cast<uint64_t>(asc::te::get<MNK_N>(params.problemShape));
        k_ = static_cast<uint64_t>(asc::te::get<MNK_K>(params.problemShape));
        batch_ = static_cast<uint64_t>(AscendC::Std::max(asc::te::get<MNK_B>(params.problemShape), 1L));
        aGmAddr_ = reinterpret_cast<__gm__ AType*>(blockMmadParams.aGmAddr);
        bGmAddr_ = reinterpret_cast<__gm__ BType*>(blockMmadParams.bGmAddr);
        cGmAddr_ = reinterpret_cast<__gm__ CType*>(blockMmadParams.cGmAddr);
        biasGmAddr_ = reinterpret_cast<__gm__ BiasType*>(blockMmadParams.biasGmAddr);
    }

    __aicore__ inline void UpdateBatchIndex(BatchInfo const& batchInfo)
    {
        uint64_t cBatchStrideDim1 = static_cast<uint64_t>(batchInfo.cBatchDim2) * batchInfo.cBatchDim3;
        uint64_t cBatchStrideDim0 = static_cast<uint64_t>(batchInfo.cBatchDim1) * cBatchStrideDim1;
        uint64_t batchC0Index = curBatchIdx_ / cBatchStrideDim0;
        uint64_t batchC1Index = curBatchIdx_ % cBatchStrideDim0 / cBatchStrideDim1;
        uint64_t batchC2Index = curBatchIdx_ % cBatchStrideDim1 / batchInfo.cBatchDim3;
        uint64_t batchC3Index = curBatchIdx_ % batchInfo.cBatchDim3;

        uint64_t batchA0Index = batchC0Index % batchInfo.aBatchDim0;
        uint64_t batchA1Index = batchC1Index % batchInfo.aBatchDim1;
        uint64_t batchA2Index = batchC2Index % batchInfo.aBatchDim2;
        uint64_t batchA3Index = batchC3Index % batchInfo.aBatchDim3;
        batchAIndex_ = batchA0Index;
        batchAIndex_ = batchAIndex_ * batchInfo.aBatchDim1 + batchA1Index;
        batchAIndex_ = batchAIndex_ * batchInfo.aBatchDim2 + batchA2Index;
        batchAIndex_ = batchAIndex_ * batchInfo.aBatchDim3 + batchA3Index;

        uint64_t batchB0Index = batchC0Index % batchInfo.bBatchDim0;
        uint64_t batchB1Index = batchC1Index % batchInfo.bBatchDim1;
        uint64_t batchB2Index = batchC2Index % batchInfo.bBatchDim2;
        uint64_t batchB3Index = batchC3Index % batchInfo.bBatchDim3;
        batchBIndex_ = batchB0Index;
        batchBIndex_ = batchBIndex_ * batchInfo.bBatchDim1 + batchB1Index;
        batchBIndex_ = batchBIndex_ * batchInfo.bBatchDim2 + batchB2Index;
        batchBIndex_ = batchBIndex_ * batchInfo.bBatchDim3 + batchB3Index;
    }

    template <typename TensorA, typename TensorB>
    __aicore__ inline void SetL2Cache(TensorA& gmA, TensorB& gmB, uint32_t l2CacheMode)
    {
        if (l2CacheMode == ALL_L2_CACHE_DISABLE || l2CacheMode == B_L2_CACHE_DISABLE) {
            gmB.set_l2_cache_hint(asc::te::cache_mode::disable);
        }
        if (l2CacheMode == ALL_L2_CACHE_DISABLE || l2CacheMode == A_L2_CACHE_DISABLE) {
            gmA.set_l2_cache_hint(asc::te::cache_mode::disable);
        }
    }

private:
    static constexpr bool TRANS_B = BlockMmad::TRANS_B;
    __gm__ AType* aGmAddr_;
    __gm__ BType* bGmAddr_;
    __gm__ CType* cGmAddr_;
    __gm__ BiasType* biasGmAddr_ = nullptr; // 可选输入，直接初始化

    uint64_t curBatchIdx_ = {0};
    uint64_t batchAIndex_ = {0};
    uint64_t batchBIndex_ = {0};
    uint64_t m_{1};
    uint64_t n_{1};
    uint64_t k_{1};
    uint64_t batch_{1};
};

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
