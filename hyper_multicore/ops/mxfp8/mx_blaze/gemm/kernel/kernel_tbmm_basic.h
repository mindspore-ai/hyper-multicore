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
 * \file kernel_tbmm_basic.h
 * \brief
 */

#pragma once

#include "kernel_basic_intf.h"

#include "mxfp8/mx_blaze/gemm/block/block_mmad.h"
#include "mxfp8/mx_blaze/gemm/block/block_mmad_matmul_basic.h"
#include "mxfp8/mx_blaze/gemm/utils/common_utils.h"
#include "mxfp8/mx_blaze/gemm/utils/layout_utils.h"
#include "kernel_universal.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Kernel {

template <class ProblemShape_, class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>
class GemmUniversal<ProblemShape_, BlockMmad_, BlockEpilogue_, BlockScheduler_,
                    AscendC::Std::enable_if_t<AscendC::Std::is_same_v<
                        KernelMmadMultiBlockTBMM, typename BlockMmad_::DispatchPolicy::ScheduleType>>> {
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
    using BlockShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using MakeLayoutB = asc::te::frame_layout_format<LayoutB, AscendC::Std::Int<asc::te::c0_element<BType>>>;
    using MakeLayoutBias = asc::te::frame_layout_format<LayoutBias, AscendC::Std::Int<asc::te::c0_element<BiasType>>>;

    static_assert(AscendC::Std::is_one_of_v<
                      AscendC::Std::tuple<AType, BType, CType, BiasType>, AscendC::Std::tuple<half, half, half, half>,
                      AscendC::Std::tuple<half, half, half, float>, AscendC::Std::tuple<half, half, signed char, half>,
                      AscendC::Std::tuple<half, half, signed char, float>,
                      AscendC::Std::tuple<bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t>,
                      AscendC::Std::tuple<float, float, float, float>>,
                  "Unsupported (AType, BType, CType, BiasType) combination");
    static_assert(!AscendC::Std::is_one_of_v<LayoutA, asc::te::nz_layout_ptn, asc::te::zn_layout_ptn> &&
                      !AscendC::Std::is_one_of_v<LayoutC, asc::te::nz_layout_ptn, asc::te::zn_layout_ptn>,
                  "LayoutA and LayoutC cannot be nz_layout_ptn or zn_layout_ptn");

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        BlockEpilogueParams epilogueParams;
        BlockSchedulerParams schParams;
        Params() = default;
    };

    __aicore__ inline void operator()(Params& params)
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

        if (params.schParams.isHf32) {
            AscendC::SetHF32Mode(1);
            AscendC::SetHF32TransMode(1);
        }

        BlockMmad blockMmad;
        blockMmad.Init(params.mmadParams);

        MatmulProcess(params, blockMmad, bs, curBlockIdx, AscendC::GetBlockNum(), bs.GetBlockNums());

        UnsetHf32();
    }

private:
    __aicore__ inline void MatmulProcess(Params const& params, BlockMmad& blockMmad, BlockScheduler& bs,
                                         int64_t curBlockIdx, int64_t coreNums, int64_t totalBlockNums)
    {
        uint64_t batchStrideA = TRANS_BATCH_A ? k_ : m_ * k_;
        uint64_t mStrideA = TRANS_BATCH_A ? batch_ * k_ : k_;

        auto layoutA = MakeNDBatchLayout<AType>(batch_, m_, k_, batchStrideA, mStrideA);
        auto layoutB = MakeLayoutB{}(batch_, k_, n_);
        auto layoutC = MakeNDBatchLayout<CType>(batch_, m_, n_, n_, batch_ * n_);

        uint64_t innerBatch = Blaze::Gemm::CeilDiv(batch_, batchSplitFactor_);
        auto splitBatchLayoutC = asc::te::make_pattern_layout<
            asc::te::nd_layout_ptn, asc::te::layout_trait<CType, AscendC::Std::Int<asc::te::c0_element<CType>>>>(
            asc::te::make_shape(batchSplitFactor_, innerBatch, asc::te::make_shape(m_, n_)),
            asc::te::make_stride(m_ * innerBatch * n_, n_, asc::te::make_stride(innerBatch * n_, asc::te::_1{})));

        auto layoutBias = MakeLayoutBias{}(1L, n_);
        // A,B,C Gm Tensor
        auto gmA = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(aGmAddr_), layoutA);
        auto gmB = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(bGmAddr_), layoutB);
        auto gmC = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(cGmAddr_), layoutC);
        auto splitBatchGmC = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(cGmAddr_),
                                                  splitBatchLayoutC);
        auto gmBias = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(biasGmAddr_), layoutBias);

        // 使能双页表
        SetL2Cache(gmA, gmB, params.schParams.l2CacheMode);

        // Process tiles in ping-pong mode
        for (int64_t blockIdx = curBlockIdx; blockIdx < totalBlockNums; blockIdx += coreNums) {
            auto blockShape = bs.template GetBlockShape<TRANS_B, BType>(blockIdx); // 非全载
            auto blockCoord = bs.GetBlockCoord(blockIdx);                          // (m, n, k, b)
            auto coordM = asc::te::get<MNK_M>(blockCoord);
            auto coordN = asc::te::get<MNK_N>(blockCoord);
            auto shapeM = asc::te::get<MNK_M>(blockShape);
            auto shapeN = asc::te::get<MNK_N>(blockShape);
            auto shapeK = asc::te::get<MNK_K>(blockShape);
            curBatchIdx_ = asc::te::get<MNK_B>(blockCoord);

            // Block offset
            auto subTensorA = gmA.slice(AscendC::MakeCoord(curBatchIdx_, AscendC::MakeCoord(coordM, 0L)),
                                        AscendC::MakeShape(1L, AscendC::MakeShape(shapeM, shapeK)));
            auto gmBlockA = asc::te::squeeze<0>(subTensorA);
            auto subTensorB = gmB.slice(AscendC::MakeCoord(curBatchIdx_, AscendC::MakeCoord(0L, coordN)),
                                        AscendC::MakeShape(1L, AscendC::MakeShape(shapeK, shapeN)));
            auto gmBlockB = asc::te::squeeze<0>(subTensorB);
            auto subTensorC = gmC.slice(AscendC::MakeCoord(curBatchIdx_, AscendC::MakeCoord(coordM, coordN)),
                                        AscendC::MakeShape(1L, AscendC::MakeShape(shapeM, shapeN)));
            auto gmBlockC = asc::te::squeeze<0>(subTensorC);

            auto outerBatchIdx = curBatchIdx_ / innerBatch;
            auto innerBatchIdx = curBatchIdx_ % innerBatch;
            auto splitBatchSubTensorC = splitBatchGmC.slice(
                AscendC::MakeCoord(outerBatchIdx, innerBatchIdx, AscendC::MakeCoord(coordM, coordN)),
                AscendC::MakeShape(asc::te::_1{}, asc::te::_1{}, AscendC::MakeShape(shapeM, shapeN)));
            auto splitBatchGmBlockC = asc::te::squeeze<0, 1>(splitBatchSubTensorC);

            auto gmBlockBias = gmBias.slice(AscendC::MakeCoord(0L, coordN), AscendC::MakeShape(1L, shapeN));

            __gm__ uint64_t* gmScalePtr = nullptr;
            if (scaleGmAddr_ != nullptr) {
                gmScalePtr = scaleGmAddr_ + curBatchIdx_ * n_ + coordN;
            }
            if (batchSplitFactor_ > 1) {
                blockMmad(gmBlockA, gmBlockB, gmBlockBias, splitBatchGmBlockC, blockShape, gmScalePtr);
            } else {
                blockMmad(gmBlockA, gmBlockB, gmBlockBias, gmBlockC, blockShape, gmScalePtr);
            }
        }
    }

    template <typename T>
    __aicore__ inline auto MakeNDBatchLayout(uint64_t batch, uint64_t row, uint64_t col, uint64_t batchStride,
                                             uint64_t colStride)
    {
        return asc::te::make_pattern_layout<asc::te::nd_layout_ptn,
                                            asc::te::layout_trait<T, AscendC::Std::Int<asc::te::c0_element<T>>>>(
            asc::te::make_shape(batch, asc::te::make_shape(row, col)),
            asc::te::make_stride(batchStride, asc::te::make_stride(colStride, asc::te::_1{})));
    }

    __aicore__ inline void Init(Params const& params)
    {
        auto blockMmadParams = params.mmadParams;
        m_ = static_cast<uint64_t>(asc::te::get<MNK_M>(params.problemShape));
        n_ = static_cast<uint64_t>(asc::te::get<MNK_N>(params.problemShape));
        k_ = static_cast<uint64_t>(asc::te::get<MNK_K>(params.problemShape));
        batch_ = static_cast<uint64_t>(asc::te::get<MNK_B>(params.problemShape));
        batchSplitFactor_ = static_cast<uint64_t>(asc::te::get<MNK_SplitB>(params.problemShape));
        aGmAddr_ = reinterpret_cast<__gm__ AType*>(blockMmadParams.aGmAddr);
        bGmAddr_ = reinterpret_cast<__gm__ BType*>(blockMmadParams.bGmAddr);
        cGmAddr_ = reinterpret_cast<__gm__ CType*>(blockMmadParams.cGmAddr);
        biasGmAddr_ = reinterpret_cast<__gm__ BiasType*>(blockMmadParams.biasGmAddr);
        scaleGmAddr_ = reinterpret_cast<__gm__ uint64_t*>(blockMmadParams.scaleGmAddr);
    }

    __aicore__ inline void UnsetHf32() { AscendC::SetHF32Mode(0); }

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
    static constexpr uint64_t NON_CONTIGUOUS_TYPE = BlockMmad::NON_CONTIGUOUS_TYPE;
    static constexpr bool TRANS_BATCH_A = (NON_CONTIGUOUS_TYPE ==
                                           static_cast<uint64_t>(NoContiguousType::NON_CONTIGUOUS_TYPE_PERM_X1));
    __gm__ AType* aGmAddr_;
    __gm__ BType* bGmAddr_;
    __gm__ CType* cGmAddr_;
    __gm__ BiasType* biasGmAddr_ = nullptr;  // 可选输入，直接初始化
    __gm__ uint64_t* scaleGmAddr_ = nullptr; // 可选输入，int8输出量化scale

    uint64_t curBatchIdx_ = {0};
    uint64_t m_{1};
    uint64_t n_{1};
    uint64_t k_{1};
    uint64_t batch_{1};
    uint64_t batchSplitFactor_{1};
};

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
