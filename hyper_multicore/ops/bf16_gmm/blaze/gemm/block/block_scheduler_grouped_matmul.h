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
 * \file block_scheduler_grouped_matmul.h
 * \brief Block scheduler for non-quant grouped matmul.
 */

#pragma once

#include "../utils/common_utils.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Block {

class BlockSchedulerGmmNoQuant {
public:
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = asc::te::coord<int64_t, int64_t, int64_t, int64_t>;
    using GroupCoord = asc::te::coord<int64_t, int64_t, int64_t, int64_t>;

    struct Params {
        int32_t baseM{0};
        int32_t baseN{0};
        uint64_t mTailCnt{1};
        uint64_t nTailCnt{1};
        uint32_t mTailAlign{1};
        uint32_t nTailAlign{1};
        int32_t groupType{-1};
        uint32_t groupNum{0};
        int64_t initialM{0};
        bool singleX{false};
        bool singleWeight{false};
        bool singleY{false};
        bool transB{false};
        bool weightNz{false};
        uint32_t weightElementSize{1};
    };

    __aicore__ inline explicit BlockSchedulerGmmNoQuant(const Params& params)
        : baseM_(params.baseM),
          baseN_(params.baseN),
          mTailCnt_(params.mTailCnt),
          nTailCnt_(params.nTailCnt),
          mTailAlign_(params.mTailAlign),
          nTailAlign_(params.nTailAlign),
          groupType_(params.groupType),
          groupNum_(params.groupNum),
          initialM_(params.initialM),
          singleX_(params.singleX),
          singleWeight_(params.singleWeight),
          singleY_(params.singleY),
          transB_(params.transB),
          weightNz_(params.weightNz),
          weightElementSize_(params.weightElementSize)
    {
        blockNum_ = static_cast<int64_t>(AscendC::GetBlockNum());
    }

    // Keep the block scheduling interface aligned with BlockSchedulerMatmulBasic.
    __aicore__ inline int64_t GetBlockNums()
    {
        // The virtual interval [groupStartBlock_, groupEndBlock_) preserves the cross-group start core.
        return groupEndBlock_;
    }

    __aicore__ inline int64_t GetFirstBlockIdx(int64_t blockIdx)
    {
        // Map the physical core to its first task in the current group's virtual interval.
        return blockIdx >= groupStartBlock_ ? blockIdx : blockIdx + blockNum_;
    }

    __aicore__ inline int64_t GetCoreNums() { return Min(groupEndBlock_, blockNum_); }

    template <bool TransB_ = false, class BType_>
    __aicore__ inline BlockShape GetBlockShape(int64_t blockIdx)
    {
        UpdateBlockInfo(blockIdx);
        return blockShape_;
    }

    __aicore__ inline BlockCoord GetBlockCoord(int blockIdx)
    {
        UpdateBlockInfo(static_cast<int64_t>(blockIdx));
        return blockCoord_;
    }

    // GMM-specific group parsing and continuous-storage offset interfaces stay in the scheduler.
    __aicore__ inline int64_t GetSplitValue(int64_t groupValue, uint32_t groupListType)
    {
        if (groupType_ == -1) {
            return 0;
        }
        if (groupListType == GROUP_LIST_TYPE_OFFSET) {
            const int64_t splitValue = groupValue - groupListOffset_;
            groupListOffset_ = groupValue;
            return splitValue;
        }
        return groupValue;
    }

    // Advance the output offset without updating block scheduling state.
    __aicore__ inline int64_t UpdateNextOutputOffset(const ProblemShape& problemShape)
    {
        const int64_t currentCOffset = singleY_ ? nextCOffset_ : 0;
        const int64_t problemM = Max(asc::te::get<MNK_M>(problemShape), static_cast<int64_t>(0));
        const int64_t problemN = Max(asc::te::get<MNK_N>(problemShape), static_cast<int64_t>(0));
        nextCOffset_ += problemM * problemN;
        return currentCOffset;
    }

    __aicore__ inline GroupCoord UpdateNextGroup(const ProblemShape& problemShape)
    {
        // Offset state and block scheduling state advance together, including groups without valid tiles.
        auto groupCoord = UpdateGroupOffset(problemShape);
        UpdateNextProblem(problemShape, NeedTailSplit(asc::te::get<MNK_M>(problemShape)));
        return groupCoord;
    }

private:
    static constexpr int64_t WINDOW_LEN = 4;
    static constexpr uint32_t GROUP_LIST_TYPE_OFFSET = 0;

    struct SplitBlockInfo {
        int64_t blockM{0};
        int64_t blockN{0};
        int64_t mOffset{0};
        int64_t nOffset{0};
    };

    struct TaskInfo {
        int64_t tileIndex{0};
        int64_t tailSplitIndex{0};
        bool isTailSplit{false};
    };

    __aicore__ inline bool NeedTailSplit(int64_t problemM) const
    {
        return groupType_ != 0 || groupNum_ != 1 || problemM == initialM_;
    }

    __aicore__ inline GroupCoord UpdateGroupOffset(const ProblemShape& problemShape)
    {
        const int64_t problemM = Max(asc::te::get<MNK_M>(problemShape), static_cast<int64_t>(0));
        const int64_t problemN = Max(asc::te::get<MNK_N>(problemShape), static_cast<int64_t>(0));
        const int64_t problemK = Max(asc::te::get<MNK_K>(problemShape), static_cast<int64_t>(0));

        auto groupCoord = GroupCoord{singleX_ ? nextAOffset_ : 0, singleWeight_ ? nextBOffset_ : 0,
                                     singleWeight_ ? nextBiasOffset_ : 0, singleY_ ? nextCOffset_ : 0};

        nextAOffset_ += problemM * problemK;
        nextBOffset_ += GetWeightSize(problemN, problemK);
        nextBiasOffset_ += problemN;
        nextCOffset_ += problemM * problemN;
        return groupCoord;
    }

    __aicore__ inline int64_t GetWeightSize(int64_t n, int64_t k) const
    {
        if (!weightNz_) {
            return n * k;
        }
        constexpr int64_t outerBlockSize = 16;
        constexpr int64_t blockByteSize = 32;
        const int64_t elementSize = Max(static_cast<int64_t>(weightElementSize_), static_cast<int64_t>(1));
        const int64_t c0Size = Max(blockByteSize / elementSize, static_cast<int64_t>(1));
        if (transB_) {
            return CeilAlign(n, outerBlockSize) * CeilAlign(k, c0Size);
        }
        return CeilAlign(n, c0Size) * CeilAlign(k, outerBlockSize);
    }

    __aicore__ inline void UpdateNextProblem(const ProblemShape& problemShape, bool tailSplit)
    {
        mTailCnt_ = Max(mTailCnt_, static_cast<uint64_t>(1));
        nTailCnt_ = Max(nTailCnt_, static_cast<uint64_t>(1));
        mTailAlign_ = Max(mTailAlign_, static_cast<uint32_t>(1));
        nTailAlign_ = Max(nTailAlign_, static_cast<uint32_t>(1));
        m_ = asc::te::get<MNK_M>(problemShape);
        n_ = asc::te::get<MNK_N>(problemShape);
        k_ = asc::te::get<MNK_K>(problemShape);
        mTileNum_ = 0;
        nTileNum_ = 0;
        logicalTileNum_ = 0;
        totalTileNum_ = 0;
        tailCnt_ = 1;
        tailWaveBase_ = 0;
        mainWindow_ = 0;
        mainRow_ = 0;
        tailWindow_ = 0;
        groupStartBlock_ = nextGroupStartBlock_;
        groupEndBlock_ = groupStartBlock_;
        lastBlockIdx_ = -1;
        blockShape_ = BlockShape{};
        blockCoord_ = BlockCoord{};
        if (m_ <= 0 || n_ <= 0 || k_ <= 0 || baseM_ <= 0 || baseN_ <= 0 || blockNum_ <= 0) {
            return;
        }

        mTileNum_ = CeilDiv(m_, static_cast<int64_t>(baseM_));
        nTileNum_ = CeilDiv(n_, static_cast<int64_t>(baseN_));
        logicalTileNum_ = mTileNum_ * nTileNum_;
        totalTileNum_ = logicalTileNum_;
        const int64_t tailTileNum = logicalTileNum_ % blockNum_;
        tailWaveBase_ = logicalTileNum_ - tailTileNum;
        if (tailTileNum > 0 && (mTailCnt_ > 1 || nTailCnt_ > 1) && tailSplit) {
            tailCnt_ = mTailCnt_ * nTailCnt_;
            totalTileNum_ += (tailCnt_ - 1) * tailTileNum;
        }

        mainWindow_ = Min(WINDOW_LEN, mTileNum_);
        mainRow_ = mTileNum_ / mainWindow_ - 1;
        tailWindow_ = mTileNum_ - mainWindow_ * mainRow_;
        groupEndBlock_ = groupStartBlock_ + totalTileNum_;
        nextGroupStartBlock_ = groupEndBlock_ % blockNum_;
    }

    __aicore__ inline void UpdateBlockInfo(int64_t blockIdx)
    {
        if (lastBlockIdx_ == blockIdx) {
            return;
        }
        lastBlockIdx_ = blockIdx;
        blockShape_ = BlockShape{0, 0, k_, 1};
        blockCoord_ = BlockCoord{0, 0, 0, 0};
        if (blockIdx < groupStartBlock_ || blockIdx >= groupEndBlock_) {
            return;
        }

        const TaskInfo taskInfo = GetTaskInfo(blockIdx - groupStartBlock_);
        const BlockCoord tileCoord = GetTileCoord(taskInfo.tileIndex);
        const SplitBlockInfo splitBlock = GetSplitBlockInfo(tileCoord, taskInfo);
        const int64_t mOffset = asc::te::get<MNK_M>(tileCoord) * baseM_ + splitBlock.mOffset;
        const int64_t nOffset = asc::te::get<MNK_N>(tileCoord) * baseN_ + splitBlock.nOffset;
        blockShape_ = BlockShape{splitBlock.blockM, splitBlock.blockN, k_, 1};
        blockCoord_ = BlockCoord{mOffset, nOffset, 0, 0};
    }

    __aicore__ inline TaskInfo GetTaskInfo(int64_t relativeTask) const
    {
        // Main-wave tasks map one-to-one; each logical tile in the last incomplete wave expands to tailCnt_ tasks.
        if (tailCnt_ == 1 || relativeTask < tailWaveBase_) {
            return {relativeTask, 0, false};
        }
        const int64_t tailTask = relativeTask - tailWaveBase_;
        return {tailWaveBase_ + tailTask / static_cast<int64_t>(tailCnt_), tailTask % static_cast<int64_t>(tailCnt_),
                true};
    }

    __aicore__ inline BlockCoord GetTileCoord(int64_t index) const
    {
        int64_t rowIdx = index / nTileNum_ / mainWindow_;
        int64_t mTileIdx = 0;
        int64_t nTileIdx = 0;
        if (rowIdx < mainRow_) {
            mTileIdx = rowIdx * mainWindow_ + index % mainWindow_;
            nTileIdx = (index / mainWindow_) % nTileNum_;
        } else {
            rowIdx = mainRow_;
            const int64_t tailIndex = index - mainRow_ * mainWindow_ * nTileNum_;
            mTileIdx = mainRow_ * mainWindow_ + tailIndex % tailWindow_;
            nTileIdx = (tailIndex / tailWindow_) % nTileNum_;
        }
        if (rowIdx & 1) {
            nTileIdx = nTileNum_ - 1 - nTileIdx;
        }
        return {mTileIdx, nTileIdx, 0, 0};
    }

    __aicore__ inline SplitBlockInfo GetSplitBlockInfo(const BlockCoord& tileCoord, const TaskInfo& taskInfo) const
    {
        const int64_t mTileIdx = asc::te::get<MNK_M>(tileCoord);
        const int64_t nTileIdx = asc::te::get<MNK_N>(tileCoord);
        int64_t blockM = mTileIdx == mTileNum_ - 1 ? m_ - (mTileNum_ - 1) * baseM_ : baseM_;
        int64_t blockN = nTileIdx == nTileNum_ - 1 ? n_ - (nTileNum_ - 1) * baseN_ : baseN_;
        if (!taskInfo.isTailSplit) {
            return {blockM, blockN, 0, 0};
        }

        const int64_t mSplit = CeilAlign(CeilDiv(blockM, static_cast<int64_t>(mTailCnt_)),
                                         static_cast<int64_t>(mTailAlign_));
        const int64_t nSplit = CeilAlign(CeilDiv(blockN, static_cast<int64_t>(nTailCnt_)),
                                         static_cast<int64_t>(nTailAlign_));
        const int64_t effectiveMTailCnt = CeilDiv(blockM, mSplit);
        const int64_t effectiveNTailCnt = CeilDiv(blockN, nSplit);
        const int64_t mSplitIdx = taskInfo.tailSplitIndex % effectiveMTailCnt;
        const int64_t nSplitIdx = taskInfo.tailSplitIndex / effectiveMTailCnt;
        const int64_t mSplitOffset = mSplitIdx * mSplit;
        const int64_t nSplitOffset = nSplitIdx * nSplit;
        if (nSplitIdx >= effectiveNTailCnt || mSplitOffset >= blockM || nSplitOffset >= blockN) {
            return {0, 0, 0, 0};
        }
        blockM = Min(blockM - mSplitOffset, mSplit);
        blockN = Min(blockN - nSplitOffset, nSplit);
        return {blockM, blockN, mSplitOffset, nSplitOffset};
    }

    int64_t m_{0};
    int64_t n_{0};
    int64_t k_{0};
    int64_t mTileNum_{0};
    int64_t nTileNum_{0};
    int64_t logicalTileNum_{0};
    int64_t totalTileNum_{0};
    int64_t tailWaveBase_{0};
    int64_t mainWindow_{0};
    int64_t tailWindow_{0};
    int64_t mainRow_{0};
    int64_t blockNum_{0};
    int32_t baseM_{0};
    int32_t baseN_{0};
    uint64_t mTailCnt_{1};
    uint64_t nTailCnt_{1};
    uint64_t tailCnt_{1};
    uint32_t mTailAlign_{1};
    uint32_t nTailAlign_{1};
    int32_t groupType_{-1};
    uint32_t groupNum_{0};
    int64_t initialM_{0};
    int64_t groupStartBlock_{0};
    int64_t groupEndBlock_{0};
    int64_t nextGroupStartBlock_{0};
    int64_t lastBlockIdx_{-1};
    BlockShape blockShape_{};
    BlockCoord blockCoord_{};
    int64_t groupListOffset_{0};
    int64_t nextAOffset_{0};
    int64_t nextBOffset_{0};
    int64_t nextBiasOffset_{0};
    int64_t nextCOffset_{0};
    bool singleX_{false};
    bool singleWeight_{false};
    bool singleY_{false};
    bool transB_{false};
    bool weightNz_{false};
    uint32_t weightElementSize_{1};
};

} // namespace Block
} // namespace Gemm
} // namespace Blaze
