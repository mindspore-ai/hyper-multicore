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
 * \file block_scheduler_gmm_swat_with_tail_split.h
 * \brief
 */

#pragma once

#include "mxfp8/mx_blaze/gemm/utils/common_utils.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Block {
constexpr int64_t GMM_WINDOW_LEN = 4; // Sliding window length along the M tile axis.
constexpr int64_t INNER_AXIS_MIN_SPLIT_VAL = 128;

class BlockSchedulerGmmSwatWithTailSplit {
public:
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    // M block size, N block size, M split offset, N split offset.
    using BlockShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = asc::te::coord<int64_t, int64_t, int64_t, int64_t>;
    struct BlockInfo {
        BlockShape blockShape{};
        BlockCoord blockCoord{};
    };

    struct MxGroupParams {
        uint32_t groupIdx{0};
        int64_t groupEndOffset{0};
        int64_t perGroupBOffset{0};
        ProblemShape problemShape{};
        bool singleW{true};
        bool transA{false};
        bool fp4A{false};
        bool fp4B{false};
    };

    struct MxGroupInfo {
        int64_t aOffset{0};
        int64_t bOffset{0};
        int64_t scaleAOffset{0};
        int64_t scaleBOffset{0};
        int64_t biasOffset{0};
        int64_t outputOffset{0};
        int64_t outputScaleOffset{0};
        int64_t inputScaleK{0};
    };

    struct MxOutputOffsets {
        int64_t outputOffset{0};
        int64_t outputScaleOffset{0};
    };

    struct MxBlockInfo {
        BlockShape blockShape{};
        int64_t mOffset{0};
        int64_t nOffset{0};
        MxOutputOffsets outputOffsets{};
    };

    struct SwigluGroupParams {
        uint32_t groupIdx{0};
        int64_t groupMEndOffset{0};
        ProblemShape problemShape{};
        bool singleW{true};
    };

    struct SwigluGroupInfo {
        int64_t aOffset{0};
        int64_t bOffset{0};
        int64_t scaleAOffset{0};
        int64_t scaleBOffset{0};
        int64_t outputOffset{0};
        int64_t outputScaleOffset{0};
        int64_t inputScaleK{0};
    };

    struct SwigluOutputOffsets {
        int64_t outputOffset{0};
        int64_t outputScaleOffset{0};
    };

    struct SwigluBlockInfo {
        BlockShape blockShape{};
        BlockShape epilogueBlockShape{};
        int64_t aMOffset{0};
        int64_t bLeftNOffset{0};
        int64_t bRightNOffset{0};
        SwigluOutputOffsets outputOffsets{};
    };

    struct Params {
        int32_t baseM{0};
        int32_t baseN{0};
    };

    __aicore__ inline BlockSchedulerGmmSwatWithTailSplit(const Params& params)
        : baseM_(params.baseM), baseN_(params.baseN)
    {
        blockNum_ = AscendC::GetBlockNum();
        blockIdx_ = AscendC::GetBlockIdx() / AscendC::GetTaskRation();
        endBlockIdx_ = blockNum_ > 0 ? blockNum_ - 1 : 0;
    }

    __aicore__ inline BlockSchedulerGmmSwatWithTailSplit(int32_t baseM, int32_t baseN, int32_t baseK)
        : baseM_(baseM), baseN_(baseN)
    {
        (void)baseK;
        blockNum_ = AscendC::GetBlockNum();
        blockIdx_ = AscendC::GetBlockIdx() / AscendC::GetTaskRation();
        endBlockIdx_ = blockNum_ > 0 ? blockNum_ - 1 : 0;
    }

    __aicore__ inline void SetLogicalWorkers(uint32_t index, uint32_t count)
    {
        blockIdx_ = index;
        blockNum_ = count;
        endBlockIdx_ = count - 1;
    }

    __aicore__ inline void UpdateNextProblem(const ProblemShape& problemShape)
    {
        // Scheduler maps only M/N blocks; K is kept for shape consistency and does not drive split-K here.
        k_ = asc::te::get<MNK_K>(problemShape);
        if (m_ != asc::te::get<MNK_M>(problemShape) || n_ != asc::te::get<MNK_N>(problemShape)) {
            m_ = asc::te::get<MNK_M>(problemShape);
            n_ = asc::te::get<MNK_N>(problemShape);
            mBlockNums_ = CeilDiv(m_, static_cast<int64_t>(baseM_));
            nBlockNums_ = CeilDiv(n_, static_cast<int64_t>(baseN_));
            mBaseTail_ = m_ - (mBlockNums_ - 1) * baseM_;
            nBaseTail_ = n_ - (nBlockNums_ - 1) * baseN_;
            totalBlockNums_ = mBlockNums_ * nBlockNums_;
            mainMWindow_ = GMM_WINDOW_LEN < mBlockNums_ ? GMM_WINDOW_LEN : mBlockNums_;
            mainRow_ = mBlockNums_ / mainMWindow_ - 1;
            tailWindow_ = mBlockNums_ - mainMWindow_ * mainRow_;
        }
        roundIdx_ = 0;
        round_ = CeilDiv(totalBlockNums_, static_cast<int64_t>(blockNum_));
        // Continue from the next physical core after the previous group to balance grouped problems.
        startBlockIdx_ = endBlockIdx_ == blockNum_ - 1 ? 0 : (endBlockIdx_ + 1);
        // The last physical core that owns a block before optional tail splitting.
        endBlockIdx_ = (totalBlockNums_ + startBlockIdx_ - 1) % blockNum_;
        tailBlockNums_ = Min(static_cast<int64_t>(endBlockIdx_ + 1), totalBlockNums_);
        // Adjust the per-core round count when this group wraps around the physical core range.
        if (startBlockIdx_ > endBlockIdx_ && (blockIdx_ > endBlockIdx_ && blockIdx_ < startBlockIdx_)) {
            round_ -= 1;
        } else if (startBlockIdx_ <= endBlockIdx_ && (blockIdx_ > endBlockIdx_ || blockIdx_ < startBlockIdx_)) {
            round_ -= 1;
        }
    }

    __aicore__ inline void UpdateBaseM(uint32_t baseM) { baseM_ = baseM; }

    __aicore__ inline void SetTailAlign(uint32_t mTailAlign, uint32_t nTailAlign)
    {
        mTailAlign_ = mTailAlign;
        nTailAlign_ = nTailAlign;
    }

    __aicore__ inline void UpdateTailTile(uint32_t mTailCnt, uint32_t nTailCnt)
    {
        mTailCnt_ = mTailCnt;
        nTailCnt_ = nTailCnt;
        tailCnt_ = mTailCnt_ * nTailCnt_;
        int64_t newEndBlockIdx = endBlockIdx_ + tailBlockNums_ * (tailCnt_ - 1);
        if (blockIdx_ > endBlockIdx_ && blockIdx_ <= newEndBlockIdx) {
            round_ += 1;
        }
        if (blockIdx_ > newEndBlockIdx) {
            mTailCnt_ = 1;
            nTailCnt_ = 1;
            tailCnt_ = 1;
            tailBlockBase_ = 0;
        } else if (tailCnt_ > 1) {
            // Base physical block of the original tail block window before M/N tail splitting.
            tailBlockBase_ = endBlockIdx_ + 1 - tailBlockNums_;
        } else {
            tailBlockBase_ = 0;
        }
        endBlockIdx_ = newEndBlockIdx;
    }

    __aicore__ inline void UpdateTailTile()
    {
        // Use the idle cores after normal blocks to split the final M/N tail block.
        if (tailBlockNums_ == 0) {
            return;
        }
        int64_t remainBlockNums = (AscendC::GetBlockNum() - endBlockIdx_ - 1) / tailBlockNums_ + 1;
        if (remainBlockNums <= 1) {
            return;
        }

        // Tail split granularity is bounded by cube alignment and the caller's tail alignment.
        int64_t mMin = Min(static_cast<int64_t>(BLOCK_CUBE), mTailAlign_);
        int64_t nMin = Min(static_cast<int64_t>(BLOCK_CUBE), nTailAlign_);

        int64_t mBlockNums = Min(CeilDiv(static_cast<int64_t>(mBaseTail_), mMin), remainBlockNums);
        int64_t nBlockNums = Min(CeilDiv(static_cast<int64_t>(nBaseTail_), nMin), remainBlockNums);
        while (mBlockNums * nBlockNums > remainBlockNums) {
            if (mBlockNums >= nBlockNums) {
                mBlockNums -= 1;
            } else {
                nBlockNums -= 1;
            }
        }
        UpdateTailTile(mBlockNums, nBlockNums);
    }

    __aicore__ inline bool GetNextBlockCoord(BlockCoord& blockCoord)
    {
        if (round_ == 0 || roundIdx_ > round_ - 1) {
            return false;
        }
        int64_t newBlockIdx = static_cast<int64_t>(blockIdx_);
        if (roundIdx_ == round_ - 1 && tailCnt_ > 1) {
            newBlockIdx = (tailBlockBase_ + ((newBlockIdx - tailBlockBase_) / tailCnt_) * tailCnt_) / tailCnt_;
        }
        int64_t index = newBlockIdx + roundIdx_ * blockNum_;
        if (blockIdx_ < startBlockIdx_) {
            index += blockNum_ - startBlockIdx_;
        } else if (tailCnt_ > 1 && endBlockIdx_ + 1 >= tailCnt_ * totalBlockNums_) {
            index -= (tailBlockBase_ + ((startBlockIdx_ - tailBlockBase_) / tailCnt_) * tailCnt_) / tailCnt_;
        } else {
            index -= startBlockIdx_;
        }

        // SWAT order walks a short M window first, then N; odd rows reverse N for locality.
        int64_t rowIdx = index / nBlockNums_ / mainMWindow_;
        if (rowIdx < mainRow_) {
            AscendC::Std::get<MNK_M>(blockCoord) = rowIdx * mainMWindow_ + index % mainMWindow_;
            AscendC::Std::get<MNK_N>(blockCoord) = (index / mainMWindow_) % nBlockNums_;
        } else {
            rowIdx = mainRow_;
            int64_t tailIndex = index - mainRow_ * mainMWindow_ * nBlockNums_;
            AscendC::Std::get<MNK_M>(blockCoord) = mainRow_ * mainMWindow_ + tailIndex % tailWindow_;
            AscendC::Std::get<MNK_N>(blockCoord) = (tailIndex / tailWindow_) % nBlockNums_;
        }
        if (rowIdx & 1) {
            AscendC::Std::get<MNK_N>(blockCoord) = nBlockNums_ - 1 - asc::te::get<MNK_N>(blockCoord);
        }
        roundIdx_++;
        return true;
    }

    __aicore__ inline BlockShape GetBlockShape(const BlockCoord& blockCoord)
    {
        int64_t singleCoreM = asc::te::get<MNK_M>(blockCoord) != (mBlockNums_ - 1) ? baseM_ : mBaseTail_;
        int64_t singleCoreN = asc::te::get<MNK_N>(blockCoord) != (nBlockNums_ - 1) ? baseN_ : nBaseTail_;
        // Return offsets only for the final split tail block; bias and split-K are handled by the kernel/block.
        if (tailCnt_ == 1 || roundIdx_ < round_) {
            return {singleCoreM, singleCoreN, 0, 0};
        }

        int64_t singleCoreMSplit = (singleCoreM + mTailCnt_ - 1) / mTailCnt_;
        int64_t singleCoreNSplit = (singleCoreN + nTailCnt_ - 1) / nTailCnt_;
        singleCoreMSplit = CeilAlign(singleCoreMSplit, mTailAlign_);
        singleCoreNSplit = CeilAlign(singleCoreNSplit, nTailAlign_);

        int64_t mSplitIdx = (blockIdx_ % tailCnt_) % mTailCnt_;
        int64_t nSplitIdx = (blockIdx_ % tailCnt_) / mTailCnt_;
        int64_t mSplitAddrOffset = mSplitIdx * singleCoreMSplit;
        int64_t nSplitAddrOffset = nSplitIdx * singleCoreNSplit;
        if (mSplitAddrOffset >= singleCoreM || nSplitAddrOffset >= singleCoreN) {
            return {0, 0, 0, 0};
        }
        singleCoreM = Min(singleCoreM - mSplitAddrOffset, singleCoreMSplit);
        singleCoreN = Min(singleCoreN - nSplitAddrOffset, singleCoreNSplit);
        return {singleCoreM, singleCoreN, mSplitAddrOffset, nSplitAddrOffset};
    }

    __aicore__ inline bool GetNextBlock(BlockInfo& blockInfo)
    {
        BlockCoord blockIndex;
        if (!GetNextBlockCoord(blockIndex)) {
            return false;
        }

        // Convert the internal SWAT tile index and tail-split offsets to an element coordinate and a true MNKB shape.
        const BlockShape legacyBlockShape = GetBlockShape(blockIndex);
        const int64_t blockM = asc::te::get<MNK_M>(legacyBlockShape);
        const int64_t blockN = asc::te::get<MNK_N>(legacyBlockShape);
        const int64_t mSplitOffset = asc::te::get<MNK_K>(legacyBlockShape);
        const int64_t nSplitOffset = asc::te::get<MNK_B>(legacyBlockShape);
        const int64_t mOffset = asc::te::get<MNK_M>(blockIndex) * baseM_ + mSplitOffset;
        const int64_t nOffset = asc::te::get<MNK_N>(blockIndex) * baseN_ + nSplitOffset;

        blockInfo.blockShape = BlockShape{blockM, blockN, k_, 1};
        blockInfo.blockCoord = BlockCoord{mOffset, nOffset, 0, 0};
        return true;
    }

    __aicore__ inline void UpdateMxGroup(const MxGroupParams& params)
    {
        const int64_t groupIdx = static_cast<int64_t>(params.groupIdx);
        const int64_t problemM = asc::te::get<MNK_M>(params.problemShape);
        const int64_t problemN = asc::te::get<MNK_N>(params.problemShape);
        const int64_t problemK = asc::te::get<MNK_K>(params.problemShape);
        const int64_t previousSplitOffset = params.groupEndOffset - (params.transA ? problemK : problemM);
        const int64_t inputScaleK = GetScaleK(problemK);
        const int64_t outputScaleN = GetScaleK(problemN);

        if (!params.transA) {
            mxGroupInfo_.aOffset = previousSplitOffset * problemK;
            if (params.fp4A) {
                mxGroupInfo_.aOffset >>= 1;
            }
            mxGroupInfo_.bOffset = params.singleW ? params.perGroupBOffset * groupIdx : 0;
            if (params.singleW && params.fp4B) {
                mxGroupInfo_.bOffset >>= 1;
            }
            mxGroupInfo_.scaleAOffset = previousSplitOffset * inputScaleK;
            mxGroupInfo_.scaleBOffset = params.singleW ? groupIdx * problemN * inputScaleK : 0;
            mxGroupInfo_.outputOffset = previousSplitOffset * problemN;
            mxGroupInfo_.outputScaleOffset = previousSplitOffset * outputScaleN;
        } else {
            mxGroupInfo_.aOffset = problemM * previousSplitOffset;
            if (params.fp4A) {
                mxGroupInfo_.aOffset >>= 1;
            }
            mxGroupInfo_.bOffset = problemN * previousSplitOffset;
            const int64_t scaleGroupOffset = previousSplitOffset >> MXFP_DIVISOR_SHIFT;
            const int64_t scaleStart = (scaleGroupOffset + groupIdx) << MXFP_MULTI_BASE_SHIFT;
            mxGroupInfo_.scaleAOffset = problemM * scaleStart;
            mxGroupInfo_.scaleBOffset = problemN * scaleStart;
            mxGroupInfo_.outputOffset = groupIdx * problemM * problemN;
            mxGroupInfo_.outputScaleOffset = groupIdx * problemM * outputScaleN;
        }
        mxGroupInfo_.biasOffset = groupIdx * problemN;
        mxGroupInfo_.inputScaleK = inputScaleK;
        mxOutputN_ = problemN;
        mxOutputScaleN_ = outputScaleN;
    }

    __aicore__ inline const MxGroupInfo& GetMxGroupInfo() const { return mxGroupInfo_; }

    __aicore__ inline bool GetNextMxBlock(MxBlockInfo& mxBlockInfo)
    {
        BlockInfo blockInfo;
        if (!GetNextBlock(blockInfo)) {
            return false;
        }
        mxBlockInfo.blockShape = blockInfo.blockShape;
        mxBlockInfo.mOffset = asc::te::get<MNK_M>(blockInfo.blockCoord);
        mxBlockInfo.nOffset = asc::te::get<MNK_N>(blockInfo.blockCoord);
        mxBlockInfo.outputOffsets.outputOffset = mxBlockInfo.mOffset * mxOutputN_ + mxBlockInfo.nOffset;
        mxBlockInfo.outputOffsets.outputScaleOffset = mxBlockInfo.mOffset * mxOutputScaleN_ +
                                                      CeilDiv(mxBlockInfo.nOffset,
                                                              static_cast<int64_t>(MXFP_DIVISOR_SIZE)) *
                                                          MXFP_MULTI_BASE_SIZE;
        return true;
    }

    __aicore__ inline void UpdateSwigluGroup(const SwigluGroupParams& params)
    {
        const int64_t groupIdx = static_cast<int64_t>(params.groupIdx);
        const int64_t problemM = asc::te::get<MNK_M>(params.problemShape);
        const int64_t problemN = asc::te::get<MNK_N>(params.problemShape);
        const int64_t problemK = asc::te::get<MNK_K>(params.problemShape);
        const int64_t mPrefixOffset = params.groupMEndOffset - problemM;
        const int64_t inputScaleK = GetScaleK(problemK);
        swigluOutputN_ = problemN / SWIGLU_SPLIT_COUNT;
        swigluOutputScaleK_ = GetScaleK(swigluOutputN_);

        swigluGroupInfo_.aOffset = mPrefixOffset * problemK;
        swigluGroupInfo_.bOffset = params.singleW ? groupIdx * problemN * problemK : 0;
        swigluGroupInfo_.scaleAOffset = mPrefixOffset * inputScaleK;
        swigluGroupInfo_.scaleBOffset = params.singleW ? groupIdx * problemN * inputScaleK : 0;
        swigluGroupInfo_.outputOffset = mPrefixOffset * swigluOutputN_;
        swigluGroupInfo_.outputScaleOffset = mPrefixOffset * swigluOutputScaleK_;
        swigluGroupInfo_.inputScaleK = inputScaleK;
    }

    __aicore__ inline const SwigluGroupInfo& GetSwigluGroupInfo() const { return swigluGroupInfo_; }

    __aicore__ inline bool GetNextSwigluBlock(SwigluBlockInfo& swigluBlockInfo)
    {
        BlockInfo blockInfo;
        if (!GetNextBlock(blockInfo)) {
            return false;
        }

        // Resolve all SwiGLU-specific input coordinates and epilogue offsets before returning to the kernel.
        const int64_t blockM = asc::te::get<MNK_M>(blockInfo.blockShape);
        const int64_t blockN = asc::te::get<MNK_N>(blockInfo.blockShape);
        const int64_t mOffset = asc::te::get<MNK_M>(blockInfo.blockCoord);
        const int64_t nOffset = asc::te::get<MNK_N>(blockInfo.blockCoord);

        swigluBlockInfo.blockShape = blockInfo.blockShape;
        swigluBlockInfo.epilogueBlockShape = BlockShape{blockM, blockN, 0, 0};
        swigluBlockInfo.aMOffset = mOffset;
        swigluBlockInfo.bLeftNOffset = nOffset;
        swigluBlockInfo.bRightNOffset = nOffset + swigluOutputN_;
        swigluBlockInfo.outputOffsets.outputOffset = mOffset * swigluOutputN_ + nOffset;
        swigluBlockInfo.outputOffsets.outputScaleOffset = mOffset * swigluOutputScaleK_ +
                                                          nOffset / MX_OUTPUT_SCALE_BLOCK_SIZE;
        return true;
    }

    __aicore__ inline int64_t GetEndBlockIdx() const { return endBlockIdx_; }

private:
    static constexpr int64_t SWIGLU_SPLIT_COUNT = 2;
    static constexpr int64_t MX_OUTPUT_SCALE_BLOCK_SIZE = MXFP_DIVISOR_SIZE / MXFP_MULTI_BASE_SIZE;

    __aicore__ inline int64_t GetScaleK(int64_t k) const
    {
        return ((k + MXFP_DIVISOR_SIZE - 1) >> MXFP_DIVISOR_SHIFT) << MXFP_MULTI_BASE_SHIFT;
    }

    SwigluGroupInfo swigluGroupInfo_{};
    MxGroupInfo mxGroupInfo_{};
    int64_t mxOutputN_{0};
    int64_t mxOutputScaleN_{0};
    int64_t swigluOutputN_{0};
    int64_t swigluOutputScaleK_{0};
    int64_t mBlockNums_{0};
    int64_t nBlockNums_{0};
    int64_t totalBlockNums_{0};
    int64_t m_{0};
    int64_t n_{0};
    int64_t k_{0};
    int64_t mTailCnt_{1};
    int64_t nTailCnt_{1};
    int64_t mTailAlign_{1};
    int64_t nTailAlign_{1};
    int64_t tailCnt_{1};
    int64_t tailBlockNums_{0};
    int64_t tailBlockBase_{0};
    int64_t mainMWindow_{0};
    int64_t tailWindow_{0};
    int64_t mainRow_{0};
    int64_t round_{0};
    int64_t roundIdx_{0};
    int32_t baseM_{0};
    int32_t baseN_{0};
    int32_t mBaseTail_{0};
    int32_t nBaseTail_{0};
    uint32_t blockNum_{0};
    uint32_t blockIdx_{0};
    uint32_t startBlockIdx_{0};
    uint32_t endBlockIdx_{0};
};

} // namespace Block
} // namespace Gemm
} // namespace Blaze
