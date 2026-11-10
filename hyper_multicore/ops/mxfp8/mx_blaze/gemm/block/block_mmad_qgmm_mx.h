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
 * \file block_mmad_qgmm_mx.h
 * \brief QGMM MX tensor_api block mmad implementation with independent kAL1 / kBL1 scheduling.
 */

#pragma once

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif

#include "tensor_api/tensor.h"
#include "mxfp8/mx_blaze/gemm/utils/common_utils.h"
#include "mxfp8/mx_blaze/gemm/policy/dispatch_policy.h"
#include "mxfp8/mx_blaze/gemm/block/block_mmad.h"
#include "mxfp8/mx_blaze/gemm/utils/layout_utils.h"
#include "mxfp8/mx_blaze/gemm/tile/compute.h"
#include "mxfp8/mx_blaze/gemm/tile/datamove.h"
#include "mxfp8/mx_blaze/gemm/tile/tile_trait.h"

namespace Blaze {
namespace Gemm {
namespace Block {

template <uint64_t FullLoadMode_, bool AtomicAdd_, class ScheduleType_, class AType_, class LayoutA_, class BType_,
          class LayoutB_, class CType_, class LayoutC_, class BiasType_, class LayoutBias_>
class BlockMmad<GroupedMatmulWithScaleMx<FullLoadMode_, AtomicAdd_, ScheduleType_>, AType_, LayoutA_, BType_, LayoutB_,
                CType_, LayoutC_, BiasType_, LayoutBias_> {
public:
    using AType = AType_;
    using BType = BType_;
    using CType = CType_;
    using LayoutA = LayoutA_;
    // LayoutB_ supports both legacy B layout input and tuple<LayoutB, LayoutScaleB> input.
    // The tuple form carries the scaleB GM layout, for example nn_layout_ptn for scale NZ.
    static constexpr bool isTuple = AscendC::Std::is_tuple_v<LayoutB_>;
    using LayoutBSelect = AscendC::Std::conditional_t<isTuple, LayoutB_, AscendC::Std::tuple<LayoutB_, LayoutB_>>;
    using LayoutB = typename AscendC::Std::tuple_element<0, LayoutBSelect>::type;
    using NDLayoutScaleB = AscendC::Std::conditional_t<IsTrans<LayoutB>::value, asc::te::scaleb_dn_layout_ptn,
                                                       asc::te::scaleb_nd_layout_ptn>;
    using LayoutScaleBSelect = AscendC::Std::conditional_t<isTuple, LayoutB_,
                                                           AscendC::Std::tuple<LayoutB, NDLayoutScaleB>>;
    using LayoutScaleB = typename AscendC::Std::tuple_element<1, LayoutScaleBSelect>::type;
    using LayoutC = LayoutC_;
    using BiasType = BiasType_;
    using LayoutBias = LayoutBias_;
    using DispatchPolicy = GroupedMatmulWithScaleMx<FullLoadMode_, AtomicAdd_, ScheduleType_>;
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;

    static constexpr bool TRANS_A = IsTrans<LayoutA>::value;
    static constexpr bool TRANS_B = IsTrans<LayoutB>::value;
    static constexpr bool IS_FP4_TYPE = IsFp4<AType>();
    static_assert(DOUBLE_BUFFER_COUNT == 2, "QGMM MX block mmad only supports double buffer.");
    static constexpr uint64_t SPLIT_M_ALIGN = 2;
    static constexpr uint64_t CONCAT_N_FACTOR = 2UL;
    static constexpr uint16_t CONCAT_MATRIX_NUM = 2U;

    struct Params {
        GM_ADDR aGmAddr{nullptr};
        GM_ADDR bGmAddr{nullptr};
        GM_ADDR cGmAddr{nullptr};
        GM_ADDR biasGmAddr{nullptr};
        GM_ADDR scaleAGmAddr{nullptr};
        GM_ADDR scaleBGmAddr{nullptr};
    };

    struct L1Params {
        uint64_t kAL1;     // A matrix L1 K-axis split size.
        uint64_t kBL1;     // B matrix L1 K-axis split size.
        uint64_t scaleKL1; // Shared ScaleA/ScaleB L1 K-axis split size.
    };

    struct MmadParams {
        BlockShape tileShapeL0;
        L1Params l1Params;
        bool isBias;
        bool enableL0cPingPong;
        uint8_t l1BufferStage{DOUBLE_BUFFER_COUNT};
    };

    template <typename TensorScaleAL1, typename TensorScaleBL1>
    struct ScalePair {
        TensorScaleAL1 scaleA;
        TensorScaleBL1 scaleB;
    };

    __aicore__ inline BlockMmad()
    {
#pragma unroll
        for (uint8_t i = 0; i < MTE1_MTE2_EVENT_ID_NUM; ++i) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(i);
        }
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(M_MTE1_FLAG_0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(M_MTE1_FLAG_1);
        AscendC::SetMMLayoutTransform(true);
    }

    __aicore__ inline ~BlockMmad()
    {
#pragma unroll
        for (uint8_t i = 0; i < MTE1_MTE2_EVENT_ID_NUM; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(i);
        }
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(M_MTE1_FLAG_0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(M_MTE1_FLAG_1);
        AscendC::SetMMLayoutTransform(false);
    }

    __aicore__ inline void Init(const ProblemShape& problemShape, const MmadParams& params)
    {
        m_ = asc::te::get<MNK_M>(problemShape);
        n_ = asc::te::get<MNK_N>(problemShape);
        k_ = asc::te::get<MNK_K>(problemShape);
        kAL1_ = params.l1Params.kAL1;
        kBL1_ = params.l1Params.kBL1;
        baseM_ = asc::te::get<MNK_M>(params.tileShapeL0);
        baseN_ = asc::te::get<MNK_N>(params.tileShapeL0);
        baseK_ = asc::te::get<MNK_K>(params.tileShapeL0);
        isBias_ = params.isBias;
        enableL0cPingPong_ = params.enableL0cPingPong;
        l1BufferStage_ = params.l1BufferStage == TRIPLE_BUFFER_COUNT ? TRIPLE_BUFFER_COUNT : DOUBLE_BUFFER_COUNT;
        orderAL1BL1_ = kAL1_ >= kBL1_;
        scaleKL1_ = Max(params.l1Params.scaleKL1, 1UL);

        constexpr uint64_t sizeShift = IS_FP4_TYPE ? 1UL : 0UL;
        scaleKL1Span_ = CeilDiv(scaleKL1_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        aL1OneBuffer_ = (baseM_ * CeilAlign(kAL1_, MXFP_DIVISOR_SIZE)) >> sizeShift;
        bL1OneBuffer_ = (baseN_ * CeilAlign(kBL1_, MXFP_DIVISOR_SIZE)) >> sizeShift;
        concatBL1OneBuffer_ = (baseN_ * CONCAT_N_FACTOR * CeilAlign(kBL1_, MXFP_DIVISOR_SIZE)) >> sizeShift;
        scaleAL1OneBuffer_ = baseM_ * scaleKL1Span_;
        scaleBL1OneBuffer_ = baseN_ * scaleKL1Span_;
        if (isBias_) {
            biasL1OneBuffer_ = baseN_ * sizeof(BiasType);
        }

        InitL1BufferOffsets();
    }

    __aicore__ inline void UpdateParamsForNextProblem(const ProblemShape& problemShape)
    {
        m_ = asc::te::get<MNK_M>(problemShape);
        n_ = asc::te::get<MNK_N>(problemShape);
        k_ = asc::te::get<MNK_K>(problemShape);
    }

private:
    template <typename TensorAL1, typename TensorScaleAL1, typename CopyL12L0A, typename CopyL12L0ScaleA>
    __aicore__ inline auto CopyNoBiasAToL0(const TensorAL1& tensorAL1, const TensorScaleAL1& tensorBlockScaleAL1,
                                           const CopyL12L0A& copyL12L0A, const CopyL12L0ScaleA& copyL12L0ScaleA,
                                           uint64_t curM, uint64_t curKL0, uint64_t curKL0ScaleSpan,
                                           uint64_t scaleOffsetA, uint64_t kaL1Offset, uint64_t kL0Offset,
                                           uint64_t l0Offset)
    {
        auto tensorAL0 = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l0a, AType>(l0Offset),
            asc::te::make_frame_layout<asc::te::nz_layout_ptn, AscendC::Std::Int<C0_SIZE>>(curM, curKL0));
        auto tensorBlockAL1 = tensorAL1.slice(asc::te::make_coord(0, kaL1Offset + kL0Offset),
                                              asc::te::make_shape(curM, curKL0));
        asc::te::copy(copyL12L0A, tensorAL0, tensorBlockAL1);

        auto tensorScaleAL0 = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l0scalea, fp8_e8m0_t>(l0Offset >> 4),
            asc::te::make_frame_layout<asc::te::zz_layout_ptn, AscendC::Std::Int<SCALE_C0>>(curM, curKL0ScaleSpan));
        asc::te::copy(copyL12L0ScaleA, tensorScaleAL0,
                      tensorBlockScaleAL1.slice(asc::te::make_coord(0, scaleOffsetA),
                                                asc::te::make_shape(curM, curKL0ScaleSpan)));
        return tensorAL0;
    }

    template <typename TensorBL1, typename TensorScaleBL1, typename CopyL12L0B, typename CopyL12L0ScaleB>
    __aicore__ inline auto CopyNoBiasBToL0(const TensorBL1& tensorBL1, const TensorScaleBL1& tensorBlockScaleBL1,
                                           const CopyL12L0B& copyL12L0B, const CopyL12L0ScaleB& copyL12L0ScaleB,
                                           uint64_t curN, uint64_t curKL0, uint64_t curKL0ScaleSpan,
                                           uint64_t scaleOffsetB, uint64_t kbL1Offset, uint64_t kL0Offset,
                                           uint64_t l0Offset)
    {
        auto tensorBL0 = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l0b, BType>(l0Offset),
            asc::te::make_frame_layout<asc::te::zn_layout_ptn, AscendC::Std::Int<C0_SIZE>>(curKL0, curN));
        auto tensorBlockBL1 = tensorBL1.slice(asc::te::make_coord(kbL1Offset + kL0Offset, 0),
                                              asc::te::make_shape(curKL0, curN));
        asc::te::copy(copyL12L0B, tensorBL0, tensorBlockBL1);

        auto tensorScaleBL0 = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l0scaleb, fp8_e8m0_t>(l0Offset >> 4),
            asc::te::make_frame_layout<asc::te::nn_layout_ptn, AscendC::Std::Int<SCALE_C0>>(curKL0ScaleSpan, curN));
        asc::te::copy(copyL12L0ScaleB, tensorScaleBL0,
                      tensorBlockScaleBL1.slice(asc::te::make_coord(scaleOffsetB, 0),
                                                asc::te::make_shape(curKL0ScaleSpan, curN)));
        return tensorBL0;
    }

    template <typename TensorL0C, typename TensorAL0, typename TensorBL0, typename MmadAtom>
    __aicore__ inline void RunNoBiasMmad(TensorL0C tensorL0C, TensorAL0 tensorAL0, TensorBL0 tensorBL0,
                                         MmadAtom mmADAtom, uint64_t curM, uint64_t curN, uint64_t curKL0,
                                         uint64_t absKOffset)
    {
        asc::te::mmad_params mmADParams;
        mmADParams.m = static_cast<uint16_t>(curM);
        mmADParams.k = static_cast<uint16_t>(CeilAlign(curKL0, MXFP_DIVISOR_SIZE));
        mmADParams.n = static_cast<uint16_t>(curN);
        mmADParams.unit_flag = (absKOffset + curKL0 >= k_) ? asc::te::unit_flag_mode::enable_update :
                                                             asc::te::unit_flag_mode::enable_keep;
        mmADParams.init_with_zero = (absKOffset == 0);
        asc::te::mmad(mmADAtom.with(mmADParams), tensorL0C, tensorAL0, tensorBL0);
    }

    template <typename TensorL0C, typename TensorAL1, typename TensorBL1, typename TensorScaleAL1,
              typename TensorScaleBL1>
    __aicore__ inline void IterateNoBias(TensorL0C tensorL0C, TensorAL1 tensorAL1, TensorBL1 tensorBL1,
                                         TensorScaleAL1 tensorScaleAL1, TensorScaleBL1 tensorScaleBL1, uint64_t curM,
                                         uint64_t curN, uint64_t curGmAKL1, uint64_t curGmBKL1, uint64_t absKStartA,
                                         uint64_t absKStartB, uint64_t kaL1Offset, uint64_t kbL1Offset)
    {
        uint64_t l1Ka = NEED_A_SET_L1_K_ZERO ? CeilAlign(curGmAKL1, MXFP_DIVISOR_SIZE) : curGmAKL1;
        uint64_t l1Kb = NEED_B_SET_L1_K_ZERO ? CeilAlign(curGmBKL1, MXFP_DIVISOR_SIZE) : curGmBKL1;
        uint64_t minPadKL1 = Min(l1Ka, l1Kb);
        uint64_t minGmKL1 = Min(curGmAKL1, curGmBKL1);
        uint64_t scaleBaseA = GetScaleOffset(absKStartA % scaleKL1_);
        uint64_t scaleBaseB = GetScaleOffset(absKStartB % scaleKL1_);
        auto tensorBlockScaleAL1 = tensorScaleAL1.slice(asc::te::make_coord(0, scaleBaseA),
                                                        asc::te::make_shape(curM, GetScaleSpan(curGmAKL1)));
        auto tensorBlockScaleBL1 = tensorScaleBL1.slice(asc::te::make_coord(scaleBaseB, 0),
                                                        asc::te::make_shape(GetScaleSpan(curGmBKL1), curN));
        auto copyL12L0A = asc::te::make_copy(asc::te::copy_l1_to_l0a{});
        auto copyL12L0B = asc::te::make_copy(asc::te::copy_l1_to_l0b{});
        auto copyL12L0ScaleA = asc::te::make_copy(asc::te::copy_l1_to_l0scalea{});
        auto copyL12L0ScaleB = asc::te::make_copy(asc::te::copy_l1_to_l0scaleb{});
        constexpr auto mmadAtom = asc::te::make_mmad(asc::te::mmad_operation{}, Tile::MmadTraitMX{});
        const uint64_t halfL0Size = AscendC::TOTAL_L0A_SIZE / DOUBLE_BUFFER_COUNT;

        for (uint64_t kL0Offset = 0; kL0Offset < minGmKL1; kL0Offset += baseK_) {
            uint64_t curKL0 = (kL0Offset + baseK_ > minPadKL1) ? (minPadKL1 - kL0Offset) : baseK_;
            uint64_t absKOffset = absKStartA + kaL1Offset + kL0Offset;
            uint64_t scaleOffsetA = GetScaleOffset(kaL1Offset + kL0Offset);
            uint64_t scaleOffsetB = GetScaleOffset(kbL1Offset + kL0Offset);
            uint64_t curKL0ScaleSpan = GetScaleSpan(curKL0);
            uint64_t l0Offset = halfL0Size * (l0PingPong_ & 1UL);
            uint16_t mte1WaitMFlag = static_cast<uint16_t>((l0PingPong_ & 1UL) + M_MTE1_FLAG_0);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(mte1WaitMFlag);

            auto tensorAL0 = CopyNoBiasAToL0(tensorAL1, tensorBlockScaleAL1, copyL12L0A, copyL12L0ScaleA, curM, curKL0,
                                             curKL0ScaleSpan, scaleOffsetA, kaL1Offset, kL0Offset, l0Offset);
            auto tensorBL0 = CopyNoBiasBToL0(tensorBL1, tensorBlockScaleBL1, copyL12L0B, copyL12L0ScaleB, curN, curKL0,
                                             curKL0ScaleSpan, scaleOffsetB, kbL1Offset, kL0Offset, l0Offset);

            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0PingPong_ & 1UL);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0PingPong_ & 1UL);

            RunNoBiasMmad(tensorL0C, tensorAL0, tensorBL0, mmadAtom, curM, curN, curKL0, absKOffset);

            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(mte1WaitMFlag);
            l0PingPong_++;
        }
    }

public:
    template <typename TensorA, typename TensorB, typename TensorScaleA, typename TensorScaleB, typename TensorBias,
              typename TensorC>
    __aicore__ inline void operator()(const TensorA& gmA, const TensorB& gmB, const TensorScaleA& gmScaleA,
                                      const TensorScaleB& gmScaleB, const TensorBias& gmBias, const TensorC& gmC,
                                      const BlockShape& blockShape)
    {
        RunImpl(gmA, gmB, gmScaleA, gmScaleB, gmBias, gmC, blockShape);
    }

    template <typename TensorA, typename TensorB, typename TensorScaleA, typename TensorScaleB, typename TensorBias,
              typename TensorC>
    __aicore__ inline void operator()(const TensorA& gmA, const TensorB& gmBFirst, const TensorB& gmBSecond,
                                      const TensorScaleA& gmScaleA, const TensorScaleB& gmScaleBFirst,
                                      const TensorScaleB& gmScaleBSecond, const TensorBias& gmBias,
                                      const TensorC& ubCFirst, const TensorC& ubCSecond, const BlockShape& blockShape)
    {
        static_assert(AscendC::Std::is_same_v<asc::te::get_mem_location<TensorC>, asc::te::location::ub>,
                      "Dual MMAD output tensors must be located in UB.");
        RunImpl(gmA, gmBFirst, gmScaleA, gmScaleBFirst, gmBias, ubCFirst, blockShape);
        RunImpl(gmA, gmBSecond, gmScaleA, gmScaleBSecond, gmBias, ubCSecond, blockShape);
    }

    template <typename TensorA, typename TensorB, typename TensorScaleA, typename TensorScaleB, typename TensorUbC>
    __aicore__ inline void operator()(TensorA gmA, TensorB gmBLeft, TensorScaleA gmScaleA, TensorScaleB gmScaleBLeft,
                                      TensorUbC& ubC, const BlockShape& singleShape)
    {
        RunConcatN(gmA, gmBLeft, gmScaleA, gmScaleBLeft, ubC, singleShape);
    }

private:
    static constexpr bool NEED_A_SET_L1_K_ZERO = AscendC::Std::is_one_of_v<AType, fp8_e5m2_t, fp8_e4m3fn_t>;
    static constexpr bool NEED_B_SET_L1_K_ZERO = AscendC::Std::is_one_of_v<BType, fp8_e5m2_t, fp8_e4m3fn_t> ||
                                                 (AscendC::Std::is_one_of_v<BType, fp4x2_e2m1_t, fp4x2_e1m2_t> &&
                                                  !TRANS_B);
    static constexpr uint64_t C0_SIZE = IS_FP4_TYPE ? C0_SIZE_B4 : C0_SIZE_B8;
    static constexpr uint8_t L1_OPERAND_NUM = 2;
    static constexpr uint8_t MTE1_MTE2_EVENT_ID_NUM = TRIPLE_BUFFER_COUNT * L1_OPERAND_NUM + SCALE_BUFFER_NUM;

    using MakeLayoutAL1 = AscendC::Std::conditional_t<
        TRANS_A, asc::te::frame_layout_format<asc::te::zn_layout_ptn, AscendC::Std::Int<C0_SIZE>>,
        asc::te::frame_layout_format<asc::te::nz_layout_ptn, AscendC::Std::Int<C0_SIZE>>>;
    using MakeLayoutBL1 = AscendC::Std::conditional_t<
        TRANS_B, asc::te::frame_layout_format<asc::te::zn_layout_ptn, AscendC::Std::Int<C0_SIZE>>,
        asc::te::frame_layout_format<asc::te::nz_layout_ptn, AscendC::Std::Int<C0_SIZE>>>;

    __aicore__ inline void InitL1BufferOffsets()
    {
        constexpr uint64_t halfL1Offset = AscendC::TOTAL_L1_SIZE >> 1;
        l1BufferAOffset_[0] = 0UL;
        l1BufferBOffset_[0] = aL1OneBuffer_;
        l1BufferAOffset_[1] = halfL1Offset;
        l1BufferBOffset_[1] = halfL1Offset + aL1OneBuffer_;
#pragma unroll
        for (int32_t bufferId = 0; bufferId < static_cast<int32_t>(SCALE_BUFFER_NUM); ++bufferId) {
            l1BufferScaleAOffset_[bufferId] = l1BufferBOffset_[bufferId] + bL1OneBuffer_;
            l1BufferScaleBOffset_[bufferId] = l1BufferScaleAOffset_[bufferId] + scaleAL1OneBuffer_;
            l1BufferBiasOffset_[bufferId] = l1BufferScaleBOffset_[bufferId] + scaleBL1OneBuffer_;
        }
        if (l1BufferStage_ == TRIPLE_BUFFER_COUNT) {
            l1BufferAOffset_[2] = l1BufferBiasOffset_[0] + biasL1OneBuffer_; // 2为l1BufferAOffset_数组的最后一位下标
            l1BufferBOffset_[2] = l1BufferBiasOffset_[1] + biasL1OneBuffer_; // 2为l1BufferBOffset_数组的最后一位下标
        }
    }

    __aicore__ inline uint64_t GetL1BufId(uint64_t loopCnt) const
    {
        return l1BufferStage_ == TRIPLE_BUFFER_COUNT ? loopCnt % TRIPLE_BUFFER_COUNT : (loopCnt & 1UL);
    }

    __aicore__ inline uint64_t GetScaleSpan(uint64_t kSpan) const
    {
        return CeilDiv(kSpan, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
    }

    __aicore__ inline uint64_t GetScaleOffset(uint64_t kOffset) const
    {
        return (kOffset >> MXFP_DIVISOR_SHIFT) << MXFP_MULTI_BASE_SHIFT;
    }

    __aicore__ inline bool NeedBias(uint64_t absKOffset) const { return isBias_ && absKOffset == 0; }

    template <typename TensorScaleA, typename TensorScaleB>
    __aicore__ inline auto CopyScalesInL1(const TensorScaleA& gmScaleA, const TensorScaleB& gmScaleB, uint64_t curM,
                                          uint64_t curN, uint64_t kL1Offset, uint64_t scaleL1BufId)
    {
        auto layoutScaleAL1 = asc::te::make_frame_layout<asc::te::zz_layout_ptn, AscendC::Std::Int<SCALE_C0>>(
            curM, scaleKL1Span_);
        auto tensorScaleAL1 = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l1, fp8_e8m0_t>(l1BufferScaleAOffset_[scaleL1BufId]),
            layoutScaleAL1);
        auto layoutScaleBL1 = asc::te::make_frame_layout<asc::te::nn_layout_ptn, AscendC::Std::Int<SCALE_C0>>(
            scaleKL1Span_, curN);
        auto tensorScaleBL1 = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l1, fp8_e8m0_t>(l1BufferScaleBOffset_[scaleL1BufId]),
            layoutScaleBL1);

        if (kL1Offset % scaleKL1_ == 0) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(static_cast<uint16_t>(l1BufferStage_ * SCALE_BUFFER_NUM) +
                                                             scaleL1BufId);
            uint64_t curScaleKL1 = scaleKL1_;
            if (kL1Offset + curScaleKL1 > k_) {
                curScaleKL1 = k_ - kL1Offset;
            }
            auto copyScaleGM2L1 = asc::te::make_copy(asc::te::copy_gm_to_l1{});
            auto gmBlockScaleA = gmScaleA.slice(asc::te::make_coord(0, GetScaleOffset(kL1Offset)),
                                                asc::te::make_shape(curM, GetScaleSpan(curScaleKL1)));
            asc::te::copy(copyScaleGM2L1, tensorScaleAL1, gmBlockScaleA);

            auto gmBlockScaleB = gmScaleB.slice(asc::te::make_coord(GetScaleOffset(kL1Offset), 0),
                                                asc::te::make_shape(GetScaleSpan(curScaleKL1), curN));
            asc::te::copy(copyScaleGM2L1, tensorScaleBL1, gmBlockScaleB);
        }
        return ScalePair<decltype(tensorScaleAL1), decltype(tensorScaleBL1)>{tensorScaleAL1, tensorScaleBL1};
    }

    template <typename TensorScaleA, typename TensorScaleB>
    __aicore__ inline auto CopyConcatScalesInL1(TensorScaleA gmScaleA, TensorScaleB gmScaleBLeft, uint64_t curM,
                                                uint64_t curN, uint64_t kL1Offset, uint64_t scaleL1BufId)
    {
        const uint64_t concatN = curN * CONCAT_N_FACTOR;
        const uint64_t concatScaleAOffset = l1BufferBOffset_[scaleL1BufId] + concatBL1OneBuffer_;
        const uint64_t concatScaleBOffset = concatScaleAOffset + scaleAL1OneBuffer_;
        auto layoutScaleAL1 = asc::te::make_frame_layout<asc::te::zz_layout_ptn, AscendC::Std::Int<SCALE_C0>>(
            curM, scaleKL1Span_);
        auto tensorScaleAL1 = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l1, fp8_e8m0_t>(concatScaleAOffset), layoutScaleAL1);
        auto layoutScaleBL1 = asc::te::make_frame_layout<asc::te::nn_layout_ptn, AscendC::Std::Int<SCALE_C0>>(
            scaleKL1Span_, concatN);
        auto tensorScaleBL1 = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l1, fp8_e8m0_t>(concatScaleBOffset), layoutScaleBL1);

        if (kL1Offset % scaleKL1_ == 0) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(static_cast<uint16_t>(l1BufferStage_ * SCALE_BUFFER_NUM) +
                                                             scaleL1BufId);
            uint64_t curScaleKL1 = scaleKL1_;
            if (kL1Offset + curScaleKL1 > k_) {
                curScaleKL1 = k_ - kL1Offset;
            }
            const uint64_t scaleOffset = GetScaleOffset(kL1Offset);
            const uint64_t scaleSpan = GetScaleSpan(curScaleKL1);
            auto copyScaleGM2L1 = asc::te::make_copy(asc::te::copy_gm_to_l1{});
            auto gmBlockScaleA = gmScaleA.slice(asc::te::make_coord(0, scaleOffset),
                                                asc::te::make_shape(curM, scaleSpan));
            asc::te::copy(copyScaleGM2L1, tensorScaleAL1, gmBlockScaleA);

            auto gmBlockScaleBLeft = gmScaleBLeft.slice(asc::te::make_coord(scaleOffset, 0),
                                                        asc::te::make_shape(scaleSpan, curN));
            auto copyConcatGM2L1 = asc::te::make_copy(Blaze::Gemm::Tile::CopyConcatGM2L1{});
            asc::te::copy(copyConcatGM2L1.with(Blaze::Gemm::Tile::CopyConcatGM2L1Params{n_, k_}), tensorScaleBL1,
                          gmBlockScaleBLeft);
        }
        return ScalePair<decltype(tensorScaleAL1), decltype(tensorScaleBL1)>{tensorScaleAL1, tensorScaleBL1};
    }

    template <typename TensorA>
    __aicore__ inline auto CopyAInL1(const TensorA& gmA, uint64_t curM, uint64_t curGmAKL1, uint64_t aL1BufId,
                                     uint64_t kL1Offset)
    {
        auto copyGM2L1 = asc::te::make_copy(asc::te::copy_gm_to_l1{});
        uint64_t l1K = curGmAKL1;
        if constexpr (NEED_A_SET_L1_K_ZERO) {
            l1K = CeilAlign(curGmAKL1, MXFP_DIVISOR_SIZE);
        }
        auto layoutAL1 = MakeLayoutAL1{}(curM, l1K);
        auto tensorAL1 = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l1, AType>(l1BufferAOffset_[aL1BufId]), layoutAL1);
        auto gmBlockA = gmA.slice(asc::te::make_coord(0, kL1Offset), asc::te::make_shape(curM, curGmAKL1));
        if constexpr (NEED_A_SET_L1_K_ZERO) {
            Blaze::Gemm::Tile::PadMxKAL1::PadZero(tensorAL1, gmBlockA);
        }
        asc::te::copy(copyGM2L1, tensorAL1, gmBlockA);
        return tensorAL1;
    }

    template <typename TensorB>
    __aicore__ inline auto CopyBInL1(const TensorB& gmB, uint64_t curN, uint64_t curGmBKL1, uint64_t bL1BufId,
                                     uint64_t kL1Offset)
    {
        auto copyGM2L1 = asc::te::make_copy(asc::te::copy_gm_to_l1{});
        uint64_t l1K = curGmBKL1;
        if constexpr (NEED_B_SET_L1_K_ZERO) {
            l1K = CeilAlign(curGmBKL1, MXFP_DIVISOR_SIZE);
        }
        auto layoutBL1 = MakeLayoutBL1{}(l1K, curN);
        auto tensorBL1 = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l1, BType>(l1BufferBOffset_[bL1BufId]), layoutBL1);
        auto gmBlockB = gmB.slice(asc::te::make_coord(kL1Offset, 0), asc::te::make_shape(curGmBKL1, curN));
        if constexpr (NEED_B_SET_L1_K_ZERO) {
            Blaze::Gemm::Tile::PadMxKBL1::PadZero(tensorBL1, gmBlockB);
        }
        asc::te::copy(copyGM2L1, tensorBL1, gmBlockB);
        return tensorBL1;
    }

    template <typename TensorB>
    __aicore__ inline auto CopyConcatBInL1(TensorB gmBLeft, uint64_t curN, uint64_t curGmBKL1, uint64_t bL1BufId,
                                           uint64_t kL1Offset)
    {
        uint64_t l1K = curGmBKL1;
        if constexpr (NEED_B_SET_L1_K_ZERO) {
            l1K = CeilAlign(curGmBKL1, MXFP_DIVISOR_SIZE);
        }
        const uint64_t concatN = curN * CONCAT_N_FACTOR;
        auto layoutBL1 = MakeLayoutBL1{}(l1K, concatN);
        auto tensorBL1 = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l1, BType>(l1BufferBOffset_[bL1BufId]), layoutBL1);
        auto gmBlockBLeft = gmBLeft.slice(asc::te::make_coord(kL1Offset, 0), asc::te::make_shape(curGmBKL1, curN));
        if constexpr (NEED_B_SET_L1_K_ZERO) {
            Blaze::Gemm::Tile::PadMxKBL1::PadZero(tensorBL1, gmBlockBLeft);
        }
        auto copyConcatGM2L1 = asc::te::make_copy(Blaze::Gemm::Tile::CopyConcatGM2L1{});
        asc::te::copy(copyConcatGM2L1.with(Blaze::Gemm::Tile::CopyConcatGM2L1Params{n_, k_}), tensorBL1, gmBlockBLeft);
        return tensorBL1;
    }

    template <typename TensorBias>
    __aicore__ inline auto CopyBiasInL1(const TensorBias& gmBias, uint64_t curN, uint64_t biasBufId, bool needCopyBias)
    {
        uint64_t biasNAlign = CeilAlign(curN, BLOCK_CUBE);
        auto layoutBiasL1 = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(1UL, biasNAlign);
        auto tensorBiasL1 = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l1, BiasType>(l1BufferBiasOffset_[biasBufId]), layoutBiasL1);
        if (isBias_ && needCopyBias) {
            auto copyGM2L1 = asc::te::make_copy(asc::te::copy_gm_to_l1{});
            asc::te::copy(copyGM2L1, tensorBiasL1, gmBias);
        }
        return tensorBiasL1;
    }

    template <typename TensorL0C, typename TensorAL1, typename TensorBL1, typename TensorScaleAL1,
              typename TensorScaleBL1, typename TensorBiasL1>
    __aicore__ inline void Iterate(TensorL0C tensorL0C, TensorAL1 tensorAL1, TensorBL1 tensorBL1,
                                   TensorScaleAL1 tensorScaleAL1, TensorScaleBL1 tensorScaleBL1,
                                   TensorBiasL1 tensorBiasL1, uint64_t curM, uint64_t curN, uint64_t curGmAKL1,
                                   uint64_t curGmBKL1, uint64_t scaleL1BufId, uint64_t absKStartA, uint64_t absKStartB,
                                   uint64_t kaL1Offset, uint64_t kbL1Offset)
    {
        uint64_t l1Ka = NEED_A_SET_L1_K_ZERO ? CeilAlign(curGmAKL1, MXFP_DIVISOR_SIZE) : curGmAKL1;
        uint64_t l1Kb = NEED_B_SET_L1_K_ZERO ? CeilAlign(curGmBKL1, MXFP_DIVISOR_SIZE) : curGmBKL1;

        // Match the actual A/B L1 layouts built by conditional MX K padding, especially for split-K tails.
        uint64_t minPadKL1 = Min(l1Ka, l1Kb);
        uint64_t minGmKL1 = Min(curGmAKL1, curGmBKL1);
        uint64_t scaleBaseA = GetScaleOffset(absKStartA % scaleKL1_);
        uint64_t scaleBaseB = GetScaleOffset(absKStartB % scaleKL1_);
        auto tensorBlockScaleAL1 = tensorScaleAL1.slice(asc::te::make_coord(0, scaleBaseA),
                                                        asc::te::make_shape(curM, GetScaleSpan(curGmAKL1)));
        auto tensorBlockScaleBL1 = tensorScaleBL1.slice(asc::te::make_coord(scaleBaseB, 0),
                                                        asc::te::make_shape(GetScaleSpan(curGmBKL1), curN));
        uint64_t biasNAlign = CeilAlign(curN, BLOCK_CUBE);
        auto layoutBt = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(1UL, biasNAlign);
        auto tensorBt = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::bias, float>(baseN_ * scaleL1BufId * sizeof(float)), layoutBt);

        auto copyL12L0A = asc::te::make_copy(asc::te::copy_l1_to_l0a{});
        auto copyL12L0B = asc::te::make_copy(asc::te::copy_l1_to_l0b{});
        auto copyL12L0ScaleA = asc::te::make_copy(asc::te::copy_l1_to_l0scalea{});
        auto copyL12L0ScaleB = asc::te::make_copy(asc::te::copy_l1_to_l0scaleb{});
        auto copyL12BT = asc::te::make_copy(asc::te::copy_l1_to_biastable{});
        constexpr auto mmadAtom = asc::te::make_mmad(asc::te::mmad_operation{}, Tile::MmadTraitMX{});
        const uint64_t halfL0Size = AscendC::TOTAL_L0A_SIZE / DOUBLE_BUFFER_COUNT;

        // Walk the current L1 K strip in baseK L0 tiles, copying A/B/scale to L0 before each MMAD.
        for (uint64_t kL0Offset = 0; kL0Offset < minGmKL1; kL0Offset += baseK_) {
            uint64_t curKL0 = (kL0Offset + baseK_ > minPadKL1) ? (minPadKL1 - kL0Offset) : baseK_;
            uint64_t absKOffset = absKStartA + kaL1Offset + kL0Offset;
            uint64_t scaleOffsetA = GetScaleOffset(kaL1Offset + kL0Offset);
            uint64_t scaleOffsetB = GetScaleOffset(kbL1Offset + kL0Offset);
            uint64_t curKL0ScaleSpan = GetScaleSpan(curKL0);
            bool needBiasThisIter = NeedBias(absKOffset);
            uint64_t l0Offset = halfL0Size * (l0PingPong_ & 1UL);
            uint16_t mte1WaitMFlag = static_cast<uint16_t>((l0PingPong_ & 1UL) + M_MTE1_FLAG_0);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(mte1WaitMFlag);

            auto tensorAL0 = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::l0a, AType>(l0Offset),
                asc::te::make_frame_layout<asc::te::nz_layout_ptn, AscendC::Std::Int<C0_SIZE>>(curM, curKL0));
            auto tensorBlockAL1 = tensorAL1.slice(asc::te::make_coord(0, kaL1Offset + kL0Offset),
                                                  asc::te::make_shape(curM, curKL0));
            asc::te::copy(copyL12L0A, tensorAL0, tensorBlockAL1);

            auto tensorScaleAL0 = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::l0scalea, fp8_e8m0_t>(l0Offset >> 4),
                asc::te::make_frame_layout<asc::te::zz_layout_ptn, AscendC::Std::Int<SCALE_C0>>(curM, curKL0ScaleSpan));
            asc::te::copy(copyL12L0ScaleA, tensorScaleAL0,
                          tensorBlockScaleAL1.slice(asc::te::make_coord(0, scaleOffsetA),
                                                    asc::te::make_shape(curM, curKL0ScaleSpan)));

            if (needBiasThisIter) {
                asc::te::copy(copyL12BT, tensorBt, tensorBiasL1);
            }

            auto tensorBL0 = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::l0b, BType>(l0Offset),
                asc::te::make_frame_layout<asc::te::zn_layout_ptn, AscendC::Std::Int<C0_SIZE>>(curKL0, curN));
            auto tensorBlockBL1 = tensorBL1.slice(asc::te::make_coord(kbL1Offset + kL0Offset, 0),
                                                  asc::te::make_shape(curKL0, curN));
            asc::te::copy(copyL12L0B, tensorBL0, tensorBlockBL1);

            auto tensorScaleBL0 = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::l0scaleb, fp8_e8m0_t>(l0Offset >> 4),
                asc::te::make_frame_layout<asc::te::nn_layout_ptn, AscendC::Std::Int<SCALE_C0>>(curKL0ScaleSpan, curN));
            asc::te::copy(copyL12L0ScaleB, tensorScaleBL0,
                          tensorBlockScaleBL1.slice(asc::te::make_coord(scaleOffsetB, 0),
                                                    asc::te::make_shape(curKL0ScaleSpan, curN)));

            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0PingPong_ & 1UL);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0PingPong_ & 1UL);

            asc::te::mmad_params mmadParams;
            mmadParams.m = static_cast<uint16_t>(curM);
            mmadParams.k = static_cast<uint16_t>(CeilAlign(curKL0, MXFP_DIVISOR_SIZE));
            mmadParams.n = static_cast<uint16_t>(curN);
            mmadParams.unit_flag = (absKOffset + curKL0 >= k_) ? asc::te::unit_flag_mode::enable_update :
                                                                 asc::te::unit_flag_mode::enable_keep;
            mmadParams.init_with_zero = !needBiasThisIter && (absKOffset == 0);
            if (needBiasThisIter) {
                asc::te::mmad(mmadAtom.with(mmadParams), tensorL0C, tensorAL0, tensorBL0, tensorBt);
            } else {
                asc::te::mmad(mmadAtom.with(mmadParams), tensorL0C, tensorAL0, tensorBL0);
            }

            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(mte1WaitMFlag);
            l0PingPong_++;
        }
    }

    template <typename TensorA, typename TensorB, typename TensorScaleA, typename TensorScaleB, typename TensorBias,
              typename TensorC>
    __aicore__ inline void RunImpl(const TensorA& gmA, const TensorB& gmB, const TensorScaleA& gmScaleA,
                                   const TensorScaleB& gmScaleB, const TensorBias& gmBias, const TensorC& tensorC,
                                   const BlockShape& blockShape)
    {
        constexpr bool isUbOutput = AscendC::Std::is_same_v<asc::te::get_mem_location<TensorC>, asc::te::location::ub>;
        constexpr bool isGmOutput = AscendC::Std::is_same_v<asc::te::get_mem_location<TensorC>, asc::te::location::gm>;
        static_assert(isUbOutput || isGmOutput, "Block MMAD output tensor must be located in GM or UB.");

        // Current output tile shape.
        uint64_t curM = asc::te::get<MNK_M>(blockShape);
        uint64_t curN = asc::te::get<MNK_N>(blockShape);
        const uint64_t halfL0CSize = AscendC::TOTAL_L0C_SIZE / DOUBLE_BUFFER_COUNT;
        uint64_t l0cOffset = (l0cPingPong_ & 1UL) * halfL0CSize;
        uint64_t l0cLayoutM = curM;
        if constexpr (isUbOutput) {
            // DUAL_DST_SPLIT_M requires an even Fixpipe mSize. Keep curM unchanged for GM slices and MMAD,
            // and expose one padding row only through the L0C-to-UB copy layout when the logical M is odd.
            l0cLayoutM = CeilAlign(curM, SPLIT_M_ALIGN);
        }
        auto layoutL0C = asc::te::frame_layout_format<asc::te::nz_layout_ptn, AscendC::Std::Int<C0_SIZE_L0C>>{}(
            l0cLayoutM, curN);
        auto tensorL0C = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0c, float>(l0cOffset),
                                              layoutL0C);

        if (orderAL1BL1_) {
            // A-major: kAL1 >= kBL1, so A is reused across the inner B K strips.
            for (uint64_t kOuter = 0; kOuter < k_; kOuter += kAL1_) {
                uint64_t scaleL1BufId = scaleLoopCnt_ & 1UL;
                uint64_t aL1BufId = GetL1BufId(aL1LoopCnt_);
                uint64_t nextKOuter = kOuter + kAL1_;
                uint64_t curGmAKL1 = (nextKOuter > k_) ? (k_ - kOuter) : kAL1_;
                // Copy scales once per scaleKL1 window; this may be a no-op inside the window.
                auto scalePair = CopyScalesInL1(gmScaleA, gmScaleB, curM, curN, kOuter, scaleL1BufId);
                auto& tensorScaleAL1 = scalePair.scaleA;
                auto& tensorScaleBL1 = scalePair.scaleB;
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(aL1BufId);
                auto tensorAL1 = CopyAInL1(gmA, curM, curGmAKL1, aL1BufId, kOuter);
                // Bias is consumed only by the first effective K tile.
                auto tensorBiasL1 = CopyBiasInL1(gmBias, curN, scaleL1BufId, NeedBias(kOuter));

                // Walk B K strips under the current A strip, then stage A/B/scale to L0 and MMAD.
                for (uint64_t kInner = kOuter; kInner < Min(kOuter + kAL1_, k_); kInner += kBL1_) {
                    uint64_t bL1BufId = GetL1BufId(bL1LoopCnt_);
                    uint64_t curGmBKL1 = (kInner + kBL1_ > k_) ? (k_ - kInner) : kBL1_;
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(static_cast<uint16_t>(l1BufferStage_) + bL1BufId);
                    auto tensorBL1 = CopyBInL1(gmB, curN, curGmBKL1, bL1BufId, kInner);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(bL1BufId);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(bL1BufId);
                    Iterate(tensorL0C, tensorAL1, tensorBL1, tensorScaleAL1, tensorScaleBL1, tensorBiasL1, curM, curN,
                            curGmAKL1, curGmBKL1, scaleL1BufId, kOuter, kInner, kInner - kOuter, 0);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(static_cast<uint16_t>(l1BufferStage_) + bL1BufId);
                    bL1LoopCnt_++;
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(aL1BufId);
                // Release the scale buffer when the copied scale window has been fully consumed.
                if ((nextKOuter % scaleKL1_) == 0 || nextKOuter >= k_) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(
                        static_cast<uint16_t>(l1BufferStage_ * SCALE_BUFFER_NUM) + scaleL1BufId);
                    scaleLoopCnt_++;
                }
                aL1LoopCnt_++;
            }
        } else {
            // B-major: kBL1 > kAL1, so B is reused across the inner A K strips.
            for (uint64_t kOuter = 0; kOuter < k_; kOuter += kBL1_) {
                uint64_t scaleL1BufId = scaleLoopCnt_ & 1UL;
                uint64_t bL1BufId = GetL1BufId(bL1LoopCnt_);
                uint64_t nextKOuter = kOuter + kBL1_;
                uint64_t curGmBKL1 = (nextKOuter > k_) ? (k_ - kOuter) : kBL1_;
                // Copy scales once per scaleKL1 window; this may be a no-op inside the window.
                auto scalePair = CopyScalesInL1(gmScaleA, gmScaleB, curM, curN, kOuter, scaleL1BufId);
                auto& tensorScaleAL1 = scalePair.scaleA;
                auto& tensorScaleBL1 = scalePair.scaleB;
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(static_cast<uint16_t>(l1BufferStage_) + bL1BufId);
                auto tensorBL1 = CopyBInL1(gmB, curN, curGmBKL1, bL1BufId, kOuter);
                // Bias is consumed only by the first effective K tile.
                auto tensorBiasL1 = CopyBiasInL1(gmBias, curN, scaleL1BufId, NeedBias(kOuter));

                // Walk A K strips under the current B strip, then stage A/B/scale to L0 and MMAD.
                for (uint64_t kInner = kOuter; kInner < Min(kOuter + kBL1_, k_); kInner += kAL1_) {
                    uint64_t aL1BufId = GetL1BufId(aL1LoopCnt_);
                    uint64_t curGmAKL1 = (kInner + kAL1_ > k_) ? (k_ - kInner) : kAL1_;
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(aL1BufId);
                    auto tensorAL1 = CopyAInL1(gmA, curM, curGmAKL1, aL1BufId, kInner);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(aL1BufId);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(aL1BufId);
                    Iterate(tensorL0C, tensorAL1, tensorBL1, tensorScaleAL1, tensorScaleBL1, tensorBiasL1, curM, curN,
                            curGmAKL1, curGmBKL1, scaleL1BufId, kInner, kOuter, 0, kInner - kOuter);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(aL1BufId);
                    aL1LoopCnt_++;
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(static_cast<uint16_t>(l1BufferStage_) + bL1BufId);
                // Release the scale buffer when the copied scale window has been fully consumed.
                if ((nextKOuter % scaleKL1_) == 0 || nextKOuter >= k_) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(
                        static_cast<uint16_t>(l1BufferStage_ * SCALE_BUFFER_NUM) + scaleL1BufId);
                    scaleLoopCnt_++;
                }
                bL1LoopCnt_++;
            }
        }

        if constexpr (isUbOutput) {
            auto copyL0C2UB = asc::te::make_copy(asc::te::copy_l0c_to_ub{},
                                                 Blaze::Gemm::Tile::CopyL0C2UBTraitMixSplitM{});
            asc::te::copy(copyL0C2UB.with(asc::te::l0c_to_ub_params(asc::te::unit_flag_mode::enable_update)), tensorC,
                          tensorL0C);
        } else {
            auto copyL0C2GM = asc::te::make_copy(asc::te::copy_l0c_to_gm{});
            asc::te::copy(copyL0C2GM.with(asc::te::l0c_to_gm_params(asc::te::unit_flag_mode::enable_update)), tensorC,
                          tensorL0C);
        }
        if (enableL0cPingPong_) {
            l0cPingPong_++;
        }
    }

    template <typename TensorUbC, typename TensorL0C>
    __aicore__ inline void CopyConcatL0C2SingleUb(TensorUbC& ubC, TensorL0C tensorL0C, uint64_t, uint64_t)
    {
        auto copyL0C2UBSplitM = asc::te::make_copy(asc::te::copy_l0c_to_ub{},
                                                   Blaze::Gemm::Tile::CopyL0C2UBTraitSplitM{});
        asc::te::copy(copyL0C2UBSplitM.with(asc::te::l0c_to_ub_params(asc::te::unit_flag_mode::enable_update)), ubC,
                      tensorL0C);
    }

    template <typename TensorL0C, typename TensorA, typename TensorB, typename TensorScaleA, typename TensorScaleB>
    __aicore__ inline void RunConcatNWithAOuter(TensorL0C tensorL0C, TensorA gmA, TensorB gmBLeft,
                                                TensorScaleA gmScaleA, TensorScaleB gmScaleBLeft, uint64_t curM,
                                                uint64_t curN, uint64_t concatN)
    {
        for (uint64_t kOuter = 0; kOuter < k_; kOuter += kAL1_) {
            uint64_t scaleL1BufId = scaleLoopCnt_ & 1UL;
            uint64_t aL1BufId = GetL1BufId(aL1LoopCnt_);
            uint64_t nextKOuter = kOuter + kAL1_;
            uint64_t curGmAKL1 = (nextKOuter > k_) ? (k_ - kOuter) : kAL1_;
            auto scalePair = CopyConcatScalesInL1(gmScaleA, gmScaleBLeft, curM, curN, kOuter, scaleL1BufId);
            auto& tensorScaleAL1 = scalePair.scaleA;
            auto& tensorScaleBL1 = scalePair.scaleB;
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(aL1BufId);
            auto tensorAL1 = CopyAInL1(gmA, curM, curGmAKL1, aL1BufId, kOuter);

            for (uint64_t kInner = kOuter; kInner < Min(kOuter + kAL1_, k_); kInner += kBL1_) {
                uint64_t bL1BufId = GetL1BufId(bL1LoopCnt_);
                uint64_t curGmBKL1 = (kInner + kBL1_ > k_) ? (k_ - kInner) : kBL1_;
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(static_cast<uint16_t>(l1BufferStage_) + bL1BufId);
                auto tensorBL1 = CopyConcatBInL1(gmBLeft, curN, curGmBKL1, bL1BufId, kInner);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(bL1BufId);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(bL1BufId);
                IterateNoBias(tensorL0C, tensorAL1, tensorBL1, tensorScaleAL1, tensorScaleBL1, curM, concatN, curGmAKL1,
                              curGmBKL1, kOuter, kInner, kInner - kOuter, 0);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(static_cast<uint16_t>(l1BufferStage_) + bL1BufId);
                bL1LoopCnt_++;
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(aL1BufId);
            if ((nextKOuter % scaleKL1_) == 0 || nextKOuter >= k_) {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(
                    static_cast<uint16_t>(l1BufferStage_ * SCALE_BUFFER_NUM) + scaleL1BufId);
                scaleLoopCnt_++;
            }
            aL1LoopCnt_++;
        }
    }

    template <typename TensorL0C, typename TensorA, typename TensorB, typename TensorScaleA, typename TensorScaleB>
    __aicore__ inline void RunConcatNWithBOuter(TensorL0C tensorL0C, TensorA gmA, TensorB gmBLeft,
                                                TensorScaleA gmScaleA, TensorScaleB gmScaleBLeft, uint64_t curM,
                                                uint64_t curN, uint64_t concatN)
    {
        for (uint64_t kOuter = 0; kOuter < k_; kOuter += kBL1_) {
            uint64_t scaleL1BufId = scaleLoopCnt_ & 1UL;
            uint64_t bL1BufId = GetL1BufId(bL1LoopCnt_);
            uint64_t nextKOuter = kOuter + kBL1_;
            uint64_t curGmBKL1 = (nextKOuter > k_) ? (k_ - kOuter) : kBL1_;
            auto scalePair = CopyConcatScalesInL1(gmScaleA, gmScaleBLeft, curM, curN, kOuter, scaleL1BufId);
            auto& tensorScaleAL1 = scalePair.scaleA;
            auto& tensorScaleBL1 = scalePair.scaleB;
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(static_cast<uint16_t>(l1BufferStage_) + bL1BufId);
            auto tensorBL1 = CopyConcatBInL1(gmBLeft, curN, curGmBKL1, bL1BufId, kOuter);

            for (uint64_t kInner = kOuter; kInner < Min(kOuter + kBL1_, k_); kInner += kAL1_) {
                uint64_t aL1BufId = GetL1BufId(aL1LoopCnt_);
                uint64_t curGmAKL1 = (kInner + kAL1_ > k_) ? (k_ - kInner) : kAL1_;
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(aL1BufId);
                auto tensorAL1 = CopyAInL1(gmA, curM, curGmAKL1, aL1BufId, kInner);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(aL1BufId);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(aL1BufId);
                IterateNoBias(tensorL0C, tensorAL1, tensorBL1, tensorScaleAL1, tensorScaleBL1, curM, concatN, curGmAKL1,
                              curGmBKL1, kInner, kOuter, 0, kInner - kOuter);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(aL1BufId);
                aL1LoopCnt_++;
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(static_cast<uint16_t>(l1BufferStage_) + bL1BufId);
            if ((nextKOuter % scaleKL1_) == 0 || nextKOuter >= k_) {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(
                    static_cast<uint16_t>(l1BufferStage_ * SCALE_BUFFER_NUM) + scaleL1BufId);
                scaleLoopCnt_++;
            }
            bL1LoopCnt_++;
        }
    }

    template <typename TensorA, typename TensorB, typename TensorScaleA, typename TensorScaleB, typename TensorUbC>
    __aicore__ inline void RunConcatN(TensorA gmA, TensorB gmBLeft, TensorScaleA gmScaleA, TensorScaleB gmScaleBLeft,
                                      TensorUbC& ubC, const BlockShape& singleShape)
    {
        uint64_t curM = asc::te::get<MNK_M>(singleShape);
        uint64_t curN = asc::te::get<MNK_N>(singleShape);
        const uint64_t concatN = curN * CONCAT_N_FACTOR;
        const uint64_t halfL0CSize = AscendC::TOTAL_L0C_SIZE / DOUBLE_BUFFER_COUNT;
        uint64_t l0cOffset = (l0cPingPong_ & 1UL) * halfL0CSize;
        uint64_t l0cLayoutM = curM;
        l0cLayoutM = CeilAlign(curM, SPLIT_M_ALIGN);
        auto layoutL0C = asc::te::frame_layout_format<asc::te::nz_layout_ptn, AscendC::Std::Int<C0_SIZE_L0C>>{}(
            l0cLayoutM, concatN);
        auto tensorL0C = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0c, float>(l0cOffset),
                                              layoutL0C);

        if (orderAL1BL1_) {
            RunConcatNWithAOuter(tensorL0C, gmA, gmBLeft, gmScaleA, gmScaleBLeft, curM, curN, concatN);
        } else {
            RunConcatNWithBOuter(tensorL0C, gmA, gmBLeft, gmScaleA, gmScaleBLeft, curM, curN, concatN);
        }

        CopyConcatL0C2SingleUb(ubC, tensorL0C, curM, curN);
        if (enableL0cPingPong_) {
            l0cPingPong_++;
        }
    }

private:
    uint64_t m_{0};
    uint64_t n_{0};
    uint64_t k_{0};
    uint64_t kAL1_{1};
    uint64_t kBL1_{1};
    uint64_t scaleKL1_{1};
    uint64_t scaleKL1Span_{1};
    uint64_t baseM_{16};
    uint64_t baseN_{16};
    uint64_t baseK_{16};
    uint64_t aL1LoopCnt_{0};
    uint64_t bL1LoopCnt_{0};
    uint64_t scaleLoopCnt_{0};
    uint64_t l0PingPong_{0};
    uint64_t l0cPingPong_{0};
    uint64_t aL1OneBuffer_{0};
    uint64_t bL1OneBuffer_{0};
    uint64_t concatBL1OneBuffer_{0};
    uint64_t scaleAL1OneBuffer_{0};
    uint64_t scaleBL1OneBuffer_{0};
    uint64_t biasL1OneBuffer_{0};
    uint64_t l1BufferStage_{DOUBLE_BUFFER_COUNT};
    uint64_t l1BufferAOffset_[TRIPLE_BUFFER_COUNT] = {0, 0, 0};
    uint64_t l1BufferBOffset_[TRIPLE_BUFFER_COUNT] = {0, 0, 0};
    uint64_t l1BufferScaleAOffset_[2] = {0, 0};
    uint64_t l1BufferScaleBOffset_[2] = {0, 0};
    uint64_t l1BufferBiasOffset_[2] = {0, 0};
    bool orderAL1BL1_{false};
    bool isBias_{false};
    bool enableL0cPingPong_{false};
};

} // namespace Block
} // namespace Gemm
} // namespace Blaze
