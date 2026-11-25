/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0.
 *
 * Unified Zero-Overhead Fast Path for HyperMegaKda (CP=1 + CP>1 ALLGATHER & PREFIX_SUM):
 *   Phase 0:   SHMEM Startup Dissemination Barrier + 36KB Causal Conv Halo P2P (CP>1 only)
 *   Phase 1:   40-AIV Contiguous 4-Head Row-Tiled ShortConv1d + SiLU + Q/K L2Norm (with Halo)
 *   Phase 2:   20-AIC <-> 40-AIV Double-Buffered L2-Ring-Buffer FusedPrepare
 *              (PreGate + IntraCube + TriDiag16 + Cube Neumann/Schur WY in ONE pass!)
 *   Phase 3A:  Pass-1: Rank 0 runs 1-pass FusedFwdH+FinalizeO+GatedRMSNorm & publishes B_0;
 *              Middle ranks (0 < r < cp-1) run 1:2 InterSummary (B_r, M_r); Rank cp-1 skips!
 *   Phase 3B:  SHMEM State Exchange & 1:2 Cube-Accelerated Merge (ALLGATHER 1-round or PREFIX_SUM log2(CP)-round)
 *   Phase 3C:  Pass-2 (rank >= 1): 1:2 FusedFwdH+FinalizeO+Overlapped GatedRMSNorm from S_excl -> IDX_OUT!
 *   Phase 4:   SHMEM Completion Dissemination Barrier (CP>1 only)
 *
 * ZERO HBM writes for wsSubIn, wsSubOut, wsAkk, wsVbeta, wsKbg, wsH, wsVnew, wsO!
 */
#ifndef HYPER_MEGA_KDA_CP1_FAST_H
#define HYPER_MEGA_KDA_CP1_FAST_H

#include "kda_compute.h"
#include "common/mega_kda/bwd_cache_layout.h"
#include "shmem.h"

namespace MulticoreKda {

// L2-resident per-lane scratch layout for Phase 2 FusedPrepare (144 KB/lane, 2 lanes, 2 slots = 576 KB/core)
constexpr uint64_t CP1_PREP_OFF_SUBIN = 0ULL;             // 288 * 128 * 2 = 73,728 B
constexpr uint64_t CP1_PREP_OFF_SUBOUT = 74ULL * 1024ULL; // 5120 * 4      = 20,480 B
constexpr uint64_t CP1_PREP_OFF_AKK = 96ULL * 1024ULL;    // 64 * 64 * 2   =  8,192 B
constexpr uint64_t CP1_PREP_OFF_KBG = 104ULL * 1024ULL;   // 64 * 128 * 2  = 16,384 B
constexpr uint64_t CP1_PREP_OFF_VBETA = 120ULL * 1024ULL; // 64 * 128 * 2  = 16,384 B
constexpr uint64_t CP1_PREP_LANE_BYTES = 144ULL * 1024ULL;
constexpr uint64_t CP1_PREP_SLOT_BYTES = 2ULL * CP1_PREP_LANE_BYTES;
constexpr uint64_t CP1_PREP_CORE_BYTES = 2ULL * CP1_PREP_SLOT_BYTES;

// L2-resident per-core scratch layout for Phase 3 Fused FwdH+FinalizeO+GatedRMSNorm (176 KB <= 320 KB stride)
constexpr uint64_t CP1_FWD_OFF_WS = 0ULL;                 // 64 * 128 * 4  = 32 KB (FP32)
constexpr uint64_t CP1_FWD_OFF_DS = 32ULL * 1024ULL;      // 128 * 128 * 4 = 64 KB (FP32)
constexpr uint64_t CP1_FWD_OFF_H = 96ULL * 1024ULL;       // 128 * 128 * 2 = 32 KB (BF16)
constexpr uint64_t CP1_FWD_OFF_VN = 128ULL * 1024ULL;     // 64 * 128 * 2  = 16 KB (BF16)
constexpr uint64_t CP1_FWD_OFF_O0 = 144ULL * 1024ULL;     // 64 * 128 * 2  = 16 KB (BF16)
constexpr uint64_t CP1_FWD_OFF_O1 = 160ULL * 1024ULL;     // 64 * 128 * 2  = 16 KB (BF16)

// ---------------------------------------------------------------------------
// SHMEM helpers for CP > 1 Fast Path (AIV only)
// ---------------------------------------------------------------------------
__aicore__ inline void RunFastShmemPutSet(GM_ADDR rb, GM_ADDR sb, GM_ADDR ev, int64_t signalOff,
                                          int32_t gen, int64_t dstByteOff, int64_t srcByteOff,
                                          int64_t nBytes, int64_t dstPe, bool publish) {
  hyper_multicore::PutMemSignalKernel<uint16_t> op;
  op.Init(rb, dstByteOff / 2, sb, srcByteOff / 2, nBytes / 2, ev, signalOff, gen, 0 /*Set*/, dstPe, false, nullptr);
  op.Process(publish);
}

__aicore__ inline int32_t RunFastDisseminationBarrier(GM_ADDR evAddr, const HyperMegaKdaLaunchTilingData *t,
                                                      bool completion) {
#if !defined(__DAV_CUBE__) && !defined(__DAV_C220_CUBE__)
  const uint32_t cpSize = t->cpSize;
  const uint32_t cpRank = t->cpRank;
  constexpr uint32_t readyStride = 16U;  // 64B cache line
  const uint64_t totalEvInts = t->eventCounterBytes / sizeof(int32_t);
  const uint64_t readyBase = totalEvInts - 2ULL * static_cast<uint64_t>(cpSize + 1U) * readyStride;
  __gm__ int32_t *ready = reinterpret_cast<__gm__ int32_t *>(evAddr) + readyBase;
  if (completion) {
    ready += static_cast<uint64_t>(cpSize + 1U) * readyStride;
  }

  GlobalTensor<int32_t> readyState;
  readyState.SetGlobalBuffer(ready, static_cast<uint32_t>((cpSize + 1U) * readyStride));
  const uint32_t genIdx = cpSize * readyStride;
  DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(readyState[genIdx]);
  const int32_t prevGen = readyState.GetValue(genIdx);
  if (prevGen == INT32_MAX) {
    AscendC::Trap();
  }
  const int32_t gen = prevGen + 1;
  readyState.SetValue(genIdx, gen);
  DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(readyState[genIdx]);
  PipeBarrier<PIPE_ALL>();

  uint32_t round = 0;
  for (uint32_t dist = 1; dist < cpSize; dist <<= 1, ++round) {
    const int target = static_cast<int>((cpRank + dist) % cpSize);
    __gm__ int32_t *roundReady = ready + static_cast<uint64_t>(round) * readyStride;
    aclshmemx_signal_op(roundReady, gen, ACLSHMEM_SIGNAL_SET, target);
    aclshmem_signal_wait_until(roundReady, ACLSHMEM_CMP_GE, gen);
  }
  return gen;
#else
  (void)evAddr;
  (void)t;
  (void)completion;
  return 0;
#endif
}

__aicore__ inline int32_t ReadCurrentReadyGen(GM_ADDR evAddr, const HyperMegaKdaLaunchTilingData *t) {
#if !defined(__DAV_CUBE__) && !defined(__DAV_C220_CUBE__)
  const uint32_t cpSize = t->cpSize;
  constexpr uint32_t readyStride = 16U;
  const uint64_t totalEvInts = t->eventCounterBytes / sizeof(int32_t);
  const uint64_t readyBase = totalEvInts - 2ULL * static_cast<uint64_t>(cpSize + 1U) * readyStride;
  GlobalTensor<int32_t> readyState;
  readyState.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(evAddr) + readyBase,
                             static_cast<uint32_t>((cpSize + 1U) * readyStride));
  const uint32_t genIdx = cpSize * readyStride;
  DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(readyState[genIdx]);
  return readyState.GetValue(genIdx);
#else
  (void)evAddr;
  (void)t;
  return 0;
#endif
}

// ---------------------------------------------------------------------------
// Phase 0.5 (AIV 0): Direct 3-token Causal Conv Halo P2P (rank -> rank + 1)
// Layout in sendHaloOff / recvHaloOff: [3(Q,K,V), 3(rows), totalHeadDim] BF16
// ---------------------------------------------------------------------------
__aicore__ inline void RunCpFastHaloExchange(uint32_t wid, int32_t gen, GM_ADDR *inputList,
                                             const HyperMegaKdaLaunchTilingData *t) {
#if !defined(__DAV_CUBE__) && !defined(__DAV_C220_CUBE__)
  if (wid == 0U) {
    const uint32_t totalHeadDim = t->numHeads * t->headDim;
    const uint32_t elems3 = 3U * totalHeadDim;  // 6144 BF16 (12 KB) for D=2048; 24 KB for D=4096
    GM_ADDR sb = inputList[IDX_SEND_BUF];
    GM_ADDR rb = inputList[IDX_RECV_BUF];
    GM_ADDR ev = inputList[IDX_EVENT];

    if (t->cpRank + 1U < t->cpSize) {
      TPipe pipe;
      TBuf<TPosition::VECCALC> buf;
      pipe.InitBuffer(buf, elems3 * 2U + 64U);
      LocalTensor<bfloat16_t> tmp = buf.GetWithOffset<bfloat16_t>(elems3, 0);
      const uint32_t directIdxs[3] = {IDX_WQ, IDX_WK, IDX_WV};
      for (uint32_t w = 0; w < 3U; ++w) {
        GlobalTensor<bfloat16_t> sGm, dGm;
        sGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(inputList[directIdxs[w]]) +
                                static_cast<uint64_t>(t->seqLocal - 3U) * totalHeadDim,
                            elems3);
        dGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(sb + t->sendHaloOff) + w * elems3, elems3);
        DataCopy(tmp, sGm, elems3);
        PipeBarrier<PIPE_ALL>();
        DataCopy(dGm, tmp, elems3);
        PipeBarrier<PIPE_ALL>();
      }
      pipe.Destroy();
      RunFastShmemPutSet(rb, sb, ev, 0 /*haloSig*/, gen, static_cast<int64_t>(t->recvHaloOff),
                         static_cast<int64_t>(t->sendHaloOff), static_cast<int64_t>(3ULL * elems3 * 2ULL),
                         static_cast<int64_t>(t->cpRank + 1U), true);
    }
    if (t->cpRank > 0U) {
      aclshmem_signal_wait_until(reinterpret_cast<__gm__ int32_t *>(ev), ACLSHMEM_CMP_GE, gen);
      if (t->saveForBwd != 0U) {
        const auto cacheL = MegaKdaCache::ComputeLayout(t->seqLocal, t->numHeads, t->headDim, t->chunkSize);
        TPipe hPipe;
        TBuf<TPosition::VECCALC> hBuf;
        hPipe.InitBuffer(hBuf, elems3 * 2U + 64U);
        LocalTensor<bfloat16_t> hTmp = hBuf.GetWithOffset<bfloat16_t>(elems3, 0);
        for (uint32_t w = 0; w < 3U; ++w) {
          GlobalTensor<bfloat16_t> sGm, dGm;
          sGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(rb + t->recvHaloOff) + w * elems3, elems3);
          dGm.SetGlobalBuffer(
              reinterpret_cast<__gm__ bfloat16_t *>(inputList[IDX_BWD_CACHE] + cacheL.haloQkvOff) + w * elems3,
              elems3);
          DataCopy(hTmp, sGm, elems3);
          PipeBarrier<PIPE_ALL>();
          DataCopy(dGm, hTmp, elems3);
          PipeBarrier<PIPE_ALL>();
        }
        hPipe.Destroy();
      }
    }
  }
  PipeBarrier<PIPE_ALL>();
  SyncAll();
#else
  (void)wid;
  (void)gen;
  (void)inputList;
  (void)t;
#endif
}

// ---------------------------------------------------------------------------
// Phase 1 (AIV): 40-AIV Contiguous 4-Head Row-Tiled ShortConv1d + SiLU + L2Norm
// Supports both CP=1 (hasHalo=false) and CP>1 State-Passing (hasHalo=cpRank>0)!
// ---------------------------------------------------------------------------
__aicore__ inline void RunCp1ShortConv(uint32_t wid, GM_ADDR *inputList, const HyperMegaKdaLaunchTilingData *t,
                                       bool hasHalo = false) {
  if ASCEND_IS_AIV {
    const uint32_t numAiv = 2U * t->numCubeCores;
    const uint32_t H = t->numHeads;
    const uint32_t d = t->headDim;  // 128
    const uint32_t S = t->seqLocal;
    const uint32_t totalHeadDim = H * d;
    GM_ADDR ws = inputList[IDX_WORKSPACE];

    const uint32_t hTile = ((H & 3U) == 0U && (H / 4U) <= numAiv) ? 4U : 1U;
    const uint32_t numGroups = H / hTile;
    const uint32_t slicesPerGroup = numAiv / numGroups;
    const uint32_t activeAivs = numGroups * slicesPerGroup;

    const uint32_t cols = hTile * d;
    constexpr uint32_t BT = 16U;
    constexpr uint32_t IN_ROWS = BT + 3U;

    TPipe pipe;
    TBuf<TPosition::VECCALC> buf;
    const uint32_t uW = 0;
    const uint32_t uInBf16 = uW + 4U * cols * 4U;
    const uint32_t uInFp = uInBf16 + ((IN_ROWS * cols * 2U + 31U) & ~31U);
    const uint32_t uAcc = uInFp + IN_ROWS * cols * 4U;
    const uint32_t uTmp = uAcc + BT * cols * 4U;
    const uint32_t uOutBf16 = uTmp + BT * cols * 4U;
    const uint32_t uNorm = uOutBf16 + BT * cols * 2U;
    const uint32_t uNormBrcb = uNorm + BT * hTile * 4U;
    pipe.InitBuffer(buf, uNormBrcb + BT * hTile * 8U * 4U + 64U);

    LocalTensor<float> wFp = buf.GetWithOffset<float>(4U * cols, uW);
    LocalTensor<bfloat16_t> inBf16 = buf.GetWithOffset<bfloat16_t>(IN_ROWS * cols, uInBf16);
    LocalTensor<float> inFp = buf.GetWithOffset<float>(IN_ROWS * cols, uInFp);
    LocalTensor<float> acc = buf.GetWithOffset<float>(BT * cols, uAcc);
    LocalTensor<float> tmp = buf.GetWithOffset<float>(BT * cols, uTmp);
    LocalTensor<bfloat16_t> outBf16 = buf.GetWithOffset<bfloat16_t>(BT * cols, uOutBf16);
    LocalTensor<float> normVec = buf.GetWithOffset<float>(BT * hTile, uNorm);
    LocalTensor<float> normBrcb = buf.GetWithOffset<float>(BT * hTile * 8U, uNormBrcb);

    // AIV 0 publishes 16x16 BF16 identity to wsEye16Off once before SyncAll
    if (wid == 0U) {
      Duplicate(acc, 0.0f, 256);
      PipeBarrier<PIPE_ALL>();
      for (uint32_t r = 0; r < 16U; ++r) {
        acc.SetValue(r * 16U + r, 1.0f);
      }
      PipeBarrier<PIPE_ALL>();
      Cast(outBf16, acc, RoundMode::CAST_RINT, 256);
      PipeBarrier<PIPE_ALL>();
      GlobalTensor<bfloat16_t> eyeGm;
      eyeGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + t->wsEye16Off), 256);
      DataCopy(eyeGm, outBf16, 256);
      PipeBarrier<PIPE_ALL>();
      if (t->saveForBwd != 0U) {
        Duplicate(acc, 0.0f, 4096);
        PipeBarrier<PIPE_ALL>();
        for (uint32_t r = 0; r < 64U; ++r) {
          acc.SetValue(r * 64U + r, 1.0f);
        }
        PipeBarrier<PIPE_ALL>();
        Cast(outBf16, acc, RoundMode::CAST_RINT, 4096);
        PipeBarrier<PIPE_ALL>();
        GlobalTensor<bfloat16_t> eye64Gm;
        eye64Gm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + t->wsEye16Off + 512ULL), 4096);
        DataCopy(eye64Gm, outBf16, 4096);
        PipeBarrier<PIPE_ALL>();
      }
    }

    if (wid < activeAivs) {
      const uint32_t g4 = wid / slicesPerGroup;
      const uint32_t sIdx = wid % slicesPerGroup;
      const uint32_t h0 = g4 * hTile;
      const uint32_t rowsPerSlice = (((S + slicesPerGroup - 1U) / slicesPerGroup) + 7U) & ~7U;
      const uint32_t rStart = sIdx * rowsPerSlice;
      const uint32_t rEnd = (rStart + rowsPerSlice <= S) ? (rStart + rowsPerSlice) : S;

      if (rStart < rEnd) {
        const uint32_t directIdxs[3] = {IDX_WQ, IDX_WK, IDX_WV};
        const uint32_t wConvIdxs[3] = {IDX_WCONV_Q, IDX_WCONV_K, IDX_WCONV_V};
        const uint64_t dstWsOffs[3] = {t->wsQHatOff, t->wsKHatOff, t->wsVConvOff};

        const uint16_t inBlockLen = static_cast<uint16_t>((cols * 2U) / 32U);
        const uint16_t inSrcStride = static_cast<uint16_t>(((totalHeadDim - cols) * 2U) / 32U);
        const uint16_t headBlockLen = static_cast<uint16_t>((d * 2U) / 32U);
        const uint16_t headSrcStride = static_cast<uint16_t>(((cols - d) * 2U) / 32U);
        const uint8_t fpRowBlocks = static_cast<uint8_t>((cols * 4U) / 32U);

        for (uint32_t which = 0; which < 3U; ++which) {
          {
            GlobalTensor<bfloat16_t> wGm;
            wGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(inputList[wConvIdxs[which]]) +
                                    static_cast<uint64_t>(h0) * d * 4ULL,
                                cols * 4U);
            DataCopy(inBf16, wGm, cols * 4U);
            PipeBarrier<PIPE_ALL>();
            Cast(tmp, inBf16, RoundMode::CAST_NONE, cols * 4U);
            PipeBarrier<PIPE_ALL>();
            for (uint32_t k = 0; k < cols; ++k) {
              wFp.SetValue(0U * cols + k, tmp.GetValue(k * 4U + 0U));
              wFp.SetValue(1U * cols + k, tmp.GetValue(k * 4U + 1U));
              wFp.SetValue(2U * cols + k, tmp.GetValue(k * 4U + 2U));
              wFp.SetValue(3U * cols + k, tmp.GetValue(k * 4U + 3U));
            }
            PipeBarrier<PIPE_ALL>();
          }

          __gm__ bfloat16_t *srcBase =
              reinterpret_cast<__gm__ bfloat16_t *>(inputList[directIdxs[which]]) + static_cast<uint64_t>(h0) * d;
          __gm__ bfloat16_t *haloBase =
              hasHalo ? (reinterpret_cast<__gm__ bfloat16_t *>(inputList[IDX_RECV_BUF] + t->recvHaloOff) +
                         static_cast<uint64_t>(which) * 3ULL * totalHeadDim + static_cast<uint64_t>(h0) * d)
                      : nullptr;
          __gm__ bfloat16_t *dstBase = reinterpret_cast<__gm__ bfloat16_t *>(ws + dstWsOffs[which]);

          for (uint32_t bStart = rStart; bStart < rEnd; bStart += BT) {
            const uint32_t curT = (bStart + BT <= rEnd) ? BT : (rEnd - bStart);
            if (bStart < 3U) {
              const uint32_t pad = 3U - bStart;
              const uint32_t validRows = curT + bStart;
              if (hasHalo) {
                GlobalTensor<bfloat16_t> hGm;
                hGm.SetGlobalBuffer(haloBase + static_cast<uint64_t>(bStart) * totalHeadDim,
                                    static_cast<uint64_t>(pad) * totalHeadDim);
                DataCopyParams hp{static_cast<uint16_t>(pad), inBlockLen, inSrcStride, 0};
                DataCopy(inBf16[0], hGm, hp);
                GlobalTensor<bfloat16_t> sGm;
                sGm.SetGlobalBuffer(srcBase, static_cast<uint64_t>(validRows) * totalHeadDim);
                DataCopyParams cp{static_cast<uint16_t>(validRows), inBlockLen, inSrcStride, 0};
                DataCopy(inBf16[pad * cols], sGm, cp);
                PipeBarrier<PIPE_ALL>();
                Cast(inFp, inBf16, RoundMode::CAST_NONE, (curT + 3U) * cols);
                PipeBarrier<PIPE_V>();
              } else {
                Duplicate(inFp, 0.0f, pad * cols);
                GlobalTensor<bfloat16_t> sGm;
                sGm.SetGlobalBuffer(srcBase, static_cast<uint64_t>(validRows) * totalHeadDim);
                DataCopyParams cp{static_cast<uint16_t>(validRows), inBlockLen, inSrcStride, 0};
                DataCopy(inBf16[pad * cols], sGm, cp);
                PipeBarrier<PIPE_ALL>();
                Cast(inFp[pad * cols], inBf16[pad * cols], RoundMode::CAST_NONE, validRows * cols);
                PipeBarrier<PIPE_V>();
              }
            } else {
              GlobalTensor<bfloat16_t> sGm;
              sGm.SetGlobalBuffer(srcBase + static_cast<uint64_t>(bStart - 3U) * totalHeadDim,
                                  static_cast<uint64_t>(curT + 3U) * totalHeadDim);
              DataCopyParams cp{static_cast<uint16_t>(curT + 3U), inBlockLen, inSrcStride, 0};
              DataCopy(inBf16, sGm, cp);
              PipeBarrier<PIPE_ALL>();
              Cast(inFp, inBf16, RoundMode::CAST_NONE, (curT + 3U) * cols);
              PipeBarrier<PIPE_V>();
            }

            BinaryRepeatParams bMul{1, 1, 1, fpRowBlocks, fpRowBlocks, 0};
            for (uint32_t col = 0; col < cols; col += 64U) {
              Mul(acc[col], inFp[0U * cols + col], wFp[0U * cols + col], 64, static_cast<uint8_t>(curT), bMul);
            }
            PipeBarrier<PIPE_V>();
            for (uint32_t tap = 1U; tap < 4U; ++tap) {
              for (uint32_t col = 0; col < cols; col += 64U) {
                Mul(tmp[col], inFp[tap * cols + col], wFp[tap * cols + col], 64, static_cast<uint8_t>(curT), bMul);
              }
              PipeBarrier<PIPE_V>();
              Add(acc, acc, tmp, curT * cols);
              PipeBarrier<PIPE_V>();
            }

            Muls(tmp, acc, -1.0f, curT * cols);
            PipeBarrier<PIPE_V>();
            Mins(tmp, tmp, 80.0f, curT * cols);
            PipeBarrier<PIPE_V>();
            Maxs(tmp, tmp, -80.0f, curT * cols);
            PipeBarrier<PIPE_V>();
            Exp(tmp, tmp, curT * cols);
            PipeBarrier<PIPE_V>();
            Adds(tmp, tmp, 1.0f, curT * cols);
            PipeBarrier<PIPE_V>();
            Div(acc, acc, tmp, curT * cols);
            PipeBarrier<PIPE_V>();

            if (which == 0U || which == 1U) {
              const uint32_t nVec = curT * hTile;
              Mul(tmp, acc, acc, curT * cols);
              PipeBarrier<PIPE_V>();
              BinaryRepeatParams foldP{1, 1, 1, 16, 16, 16};
              Add(tmp, tmp, tmp[64], 64, static_cast<uint8_t>(nVec), foldP);
              PipeBarrier<PIPE_V>();
              WholeReduceSum(normVec, tmp, 64, static_cast<int32_t>(nVec), 1, 1, 16);
              PipeBarrier<PIPE_V>();
              Adds(normVec, normVec, 1e-6f, nVec);
              PipeBarrier<PIPE_V>();
              Sqrt(normVec, normVec, nVec);
              PipeBarrier<PIPE_V>();
              Duplicate(tmp, 1.0f, nVec);
              PipeBarrier<PIPE_V>();
              Div(normVec, tmp, normVec, nVec);
              PipeBarrier<PIPE_V>();
              Brcb(normBrcb, normVec, static_cast<uint8_t>((nVec + 7U) / 8U), {1, 8});
              PipeBarrier<PIPE_V>();
              BinaryRepeatParams scaleP{1, 1, 0, 16, 16, 1};
              Mul(acc, acc, normBrcb, 64, static_cast<uint8_t>(nVec), scaleP);
              Mul(acc[64], acc[64], normBrcb, 64, static_cast<uint8_t>(nVec), scaleP);
              PipeBarrier<PIPE_V>();
              if (t->saveForBwd != 0U) {
                const auto cacheL = MegaKdaCache::ComputeLayout(S, H, d, t->chunkSize);
                const uint64_t rstdBaseOff = (which == 0U) ? cacheL.qRstdOff : cacheL.kRstdOff;
                __gm__ float *rstdBase = reinterpret_cast<__gm__ float *>(inputList[IDX_BWD_CACHE] + rstdBaseOff);
                PipeBarrier<PIPE_ALL>();
                for (uint32_t hi = 0; hi < hTile; ++hi) {
                  for (uint32_t tr = 0; tr < curT; ++tr) {
                    tmp.SetValue(hi * curT + tr, normVec.GetValue(tr * hTile + hi));
                  }
                }
                PipeBarrier<PIPE_ALL>();
                for (uint32_t hi = 0; hi < hTile; ++hi) {
                  GlobalTensor<float> rGm;
                  rGm.SetGlobalBuffer(rstdBase + static_cast<uint64_t>(h0 + hi) * S + bStart, curT);
                  DataCopy(rGm, tmp[hi * curT], curT);
                }
                PipeBarrier<PIPE_ALL>();
              }
            }

            Cast(outBf16, acc, RoundMode::CAST_RINT, curT * cols);
            PipeBarrier<PIPE_ALL>();

            for (uint32_t hi = 0; hi < hTile; ++hi) {
              GlobalTensor<bfloat16_t> dGm;
              dGm.SetGlobalBuffer(dstBase + (static_cast<uint64_t>(h0 + hi) * S + bStart) * d, curT * d);
              if (hTile == 1U) {
                DataCopy(dGm, outBf16, curT * d);
              } else {
                DataCopyParams stP{static_cast<uint16_t>(curT), headBlockLen, headSrcStride, 0};
                DataCopy(dGm, outBf16[hi * d], stP);
              }
            }
            if (t->saveForBwd != 0U) {
              const auto cacheL = MegaKdaCache::ComputeLayout(S, H, d, t->chunkSize);
              const uint64_t cacheOffs[3] = {cacheL.qNormOff, cacheL.kNormOff, cacheL.vConvOff};
              __gm__ bfloat16_t *cDstBase = reinterpret_cast<__gm__ bfloat16_t *>(inputList[IDX_BWD_CACHE] + cacheOffs[which]);
              for (uint32_t hi = 0; hi < hTile; ++hi) {
                GlobalTensor<bfloat16_t> cGm;
                cGm.SetGlobalBuffer(cDstBase + (static_cast<uint64_t>(h0 + hi) * S + bStart) * d, curT * d);
                if (hTile == 1U) {
                  DataCopy(cGm, outBf16, curT * d);
                } else {
                  DataCopyParams stP{static_cast<uint16_t>(curT), headBlockLen, headSrcStride, 0};
                  DataCopy(cGm, outBf16[hi * d], stP);
                }
              }
            }
            PipeBarrier<PIPE_ALL>();
          }
        }
      }
    }

    pipe.Destroy();
    PipeBarrier<PIPE_ALL>();
    SyncAll();
  }
}

// ---------------------------------------------------------------------------
// Phase 2: 20-AIC <-> 40-AIV Double-Buffered L2-Ring-Buffer FusedPrepare
// ---------------------------------------------------------------------------
__aicore__ inline void RunCp1FusedPrepareAiv(uint32_t wid, GM_ADDR *inputList,
                                             const HyperMegaKdaLaunchTilingData *t) {
  if ASCEND_IS_AIV {
    const uint32_t core = wid >> 1;
    const uint32_t lane = wid & 1U;
    const uint32_t numCube = t->numCubeCores;
    const uint32_t H = t->numHeads;
    const uint32_t d = t->headDim;
    const uint32_t C = t->chunkSize;
    const uint32_t numChunks = t->numChunks;
    const uint32_t numPairs = H >> 1;
    const uint32_t totalJobs = numChunks * numPairs;
    const uint32_t gRowDim = H * d;
    const uint32_t betaStride = H;
    const float scale = BitsToFloat(t->scaleBits);
    const float lowerBound = BitsToFloat(t->lowerBoundBits);
    GM_ADDR ws = inputList[IDX_WORKSPACE];
    GM_ADDR prepCoreBase = ws + t->wsSubInOff + static_cast<uint64_t>(core) * CP1_PREP_CORE_BYTES;

    __gm__ bfloat16_t *gRawAll = reinterpret_cast<__gm__ bfloat16_t *>(inputList[IDX_WFB]);
    __gm__ bfloat16_t *betaRawAll = reinterpret_cast<__gm__ bfloat16_t *>(inputList[IDX_WB]);

    TPipe pipe;
    TBuf<TPosition::VECCALC> buf;
    const uint32_t uALog = 0;
    const uint32_t uMaskStrict = 256;
    const uint32_t uMaskIncl = uMaskStrict + 1024;
    const uint32_t uBetaMask = uMaskIncl + 1024;
    const uint32_t uDtBias = uBetaMask + 256;
    const uint32_t uWork = 4096;

    const uint32_t uGCum = uWork;
    const uint32_t uQFp = uGCum + C * d * 4;
    const uint32_t uKFp = uQFp + C * d * 4;
    const uint32_t uS1 = uKFp + C * d * 4;
    const uint32_t uS2 = uS1 + 32 * d * 4;
    const uint32_t uBfTmp = uS2 + 32 * d * 4;
    const uint32_t uBetaRawFp = uBfTmp + C * d * 2;
    const uint32_t uBetaVec = uBetaRawFp + C * 64 * 4;
    const uint32_t uBetaTmp = uBetaVec + C * 4;
    const uint32_t uBetaBrcb = uBetaTmp + C * 4;
    const uint32_t uGRef = uBetaBrcb + C * 8 * 4;

    const uint32_t uMatA = uWork;
    const uint32_t uMatAqk = uMatA + C * C * 4;
    const uint32_t uSubOut = uMatAqk + C * C * 4;
    const uint32_t uBfOut = uSubOut + 5120 * 4;

    pipe.InitBuffer(buf, uGRef + d * 4 + 64U);

    LocalTensor<float> aArr = buf.GetWithOffset<float>(64, uALog);
    LocalTensor<float> maskStrict = buf.GetWithOffset<float>(256, uMaskStrict);
    LocalTensor<float> maskIncl = buf.GetWithOffset<float>(256, uMaskIncl);
    LocalTensor<float> betaMask = buf.GetWithOffset<float>(64, uBetaMask);
    LocalTensor<float> dtBias = buf.GetWithOffset<float>(d, uDtBias);

    LocalTensor<float> gCum = buf.GetWithOffset<float>(C * d, uGCum);
    LocalTensor<float> qFp = buf.GetWithOffset<float>(C * d, uQFp);
    LocalTensor<float> kFp = buf.GetWithOffset<float>(C * d, uKFp);
    LocalTensor<float> s1 = buf.GetWithOffset<float>(32 * d, uS1);
    LocalTensor<float> s2 = buf.GetWithOffset<float>(32 * d, uS2);
    LocalTensor<bfloat16_t> bfTmp = buf.GetWithOffset<bfloat16_t>(C * d, uBfTmp);
    LocalTensor<float> betaRawFp = buf.GetWithOffset<float>(C * 64, uBetaRawFp);
    LocalTensor<float> betaVec = buf.GetWithOffset<float>(C, uBetaVec);
    LocalTensor<float> betaTmp = buf.GetWithOffset<float>(C, uBetaTmp);
    LocalTensor<float> betaBrcb = buf.GetWithOffset<float>(C * 8, uBetaBrcb);
    LocalTensor<float> gRef = buf.GetWithOffset<float>(d, uGRef);

    LocalTensor<float> matA = buf.GetWithOffset<float>(C * C, uMatA);
    LocalTensor<float> matAqk = buf.GetWithOffset<float>(C * C, uMatAqk);
    LocalTensor<float> subOut = buf.GetWithOffset<float>(5120, uSubOut);
    LocalTensor<bfloat16_t> bfOut = buf.GetWithOffset<bfloat16_t>(C * C, uBfOut);

    {
      const uint32_t hAligned = (H + 7U) & ~7U;
      GlobalTensor<float> aLogGm;
      aLogGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(inputList[IDX_ALOG]), hAligned);
      DataCopy(aArr, aLogGm, hAligned);
      Duplicate(maskStrict, 0.0f, 256);
      Duplicate(maskIncl, 0.0f, 256);
      PipeBarrier<PIPE_ALL>();
      Exp(aArr, aArr, hAligned);
      for (uint32_t r = 0; r < 16U; ++r) {
        for (uint32_t j = 0; j < r; ++j) {
          maskStrict.SetValue(r * 16U + j, 1.0f);
          maskIncl.SetValue(r * 16U + j, 1.0f);
        }
        maskIncl.SetValue(r * 16U + r, 1.0f);
      }
      PipeBarrier<PIPE_ALL>();
    }

    const uint16_t blockLen = static_cast<uint16_t>((d * 2U) / 32U);
    const uint16_t gSrcStride = static_cast<uint16_t>(((gRowDim - d) * 2U) / 32U);

    uint32_t jobCount = 0;
    for (uint32_t j = 0, job = core; job < totalJobs; ++j, job += numCube) {
      ++jobCount;
      const uint32_t c = job / numPairs;
      const uint32_t pair = job % numPairs;
      const uint32_t h = (pair << 1) + lane;
      const uint32_t slot = j & 1U;
      const uint64_t tStart = static_cast<uint64_t>(c) * C;
      const uint64_t hcIdx = static_cast<uint64_t>(h) * numChunks + c;

      if (j >= 2U) {
        CrossCoreWaitFlag<0x2, PIPE_MTE3>(6U + slot);
      }

      GM_ADDR laneBase = prepCoreBase + slot * CP1_PREP_SLOT_BYTES + lane * CP1_PREP_LANE_BYTES;
      __gm__ bfloat16_t *subInPtr = reinterpret_cast<__gm__ bfloat16_t *>(laneBase + CP1_PREP_OFF_SUBIN);
      __gm__ float *subOutPtr = reinterpret_cast<__gm__ float *>(laneBase + CP1_PREP_OFF_SUBOUT);
      __gm__ bfloat16_t *akkPtr = reinterpret_cast<__gm__ bfloat16_t *>(laneBase + CP1_PREP_OFF_AKK);
      __gm__ bfloat16_t *kbgPtr = reinterpret_cast<__gm__ bfloat16_t *>(laneBase + CP1_PREP_OFF_KBG);
      __gm__ bfloat16_t *vbPtr = reinterpret_cast<__gm__ bfloat16_t *>(laneBase + CP1_PREP_OFF_VBETA);

      const float A = aArr.GetValue(h);
      {
        GlobalTensor<float> dtGm;
        dtGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(inputList[IDX_DTBIAS]) + static_cast<uint64_t>(h) * d, d);
        DataCopy(dtBias, dtGm, d);
      }

      {
        const uint32_t betaElems = C * betaStride;
        GlobalTensor<bfloat16_t> bGm;
        bGm.SetGlobalBuffer(betaRawAll + tStart * betaStride, betaElems);
        DataCopy(bfTmp, bGm, betaElems);
        Duplicate(betaMask, 0.0f, 64);
        PipeBarrier<PIPE_ALL>();
        betaMask.SetValue(h, 1.0f);
        Cast(betaRawFp, bfTmp, RoundMode::CAST_NONE, betaElems);
        PipeBarrier<PIPE_ALL>();
        if ((betaStride & 7U) == 0U && betaStride <= 64U) {
          const uint8_t repStride = static_cast<uint8_t>(betaStride / 8U);
          Mul(betaRawFp, betaRawFp, betaMask, static_cast<uint64_t>(betaStride), static_cast<uint8_t>(C),
              BinaryRepeatParams{1, 1, 1, repStride, repStride, 0});
          PipeBarrier<PIPE_V>();
          WholeReduceSum(betaVec, betaRawFp, static_cast<int32_t>(betaStride), static_cast<int32_t>(C), 1, 1,
                         static_cast<uint16_t>(repStride));
          PipeBarrier<PIPE_V>();
        } else {
          for (uint32_t r = 0; r < C; ++r) {
            betaVec.SetValue(r, betaRawFp.GetValue(r * betaStride + h));
          }
          PipeBarrier<PIPE_ALL>();
        }
        Muls(betaTmp, betaVec, -1.0f, C);
        PipeBarrier<PIPE_V>();
        Mins(betaTmp, betaTmp, 80.0f, C);
        PipeBarrier<PIPE_V>();
        Maxs(betaTmp, betaTmp, -80.0f, C);
        PipeBarrier<PIPE_V>();
        Exp(betaTmp, betaTmp, C);
        PipeBarrier<PIPE_V>();
        Adds(betaTmp, betaTmp, 1.0f, C);
        PipeBarrier<PIPE_V>();
        Duplicate(betaVec, 1.0f, C);
        PipeBarrier<PIPE_V>();
        Div(betaVec, betaVec, betaTmp, C);
        PipeBarrier<PIPE_V>();
        Brcb(betaBrcb, betaVec, static_cast<uint8_t>(C / 8U), {1, 8});
        PipeBarrier<PIPE_ALL>();
      }

      __gm__ bfloat16_t *gRawBase = gRawAll + static_cast<uint64_t>(h) * d;
      for (uint32_t half = 0; half < 2U; ++half) {
        GlobalTensor<bfloat16_t> gGm;
        gGm.SetGlobalBuffer(gRawBase + (tStart + half * 32ULL) * gRowDim, 32ULL * gRowDim);
        DataCopyParams cp{32, blockLen, gSrcStride, 0};
        DataCopy(bfTmp, gGm, cp);
        PipeBarrier<PIPE_ALL>();
        Cast(s1, bfTmp, RoundMode::CAST_NONE, 32 * d);
        PipeBarrier<PIPE_V>();
        BinaryRepeatParams bAdd{1, 1, 1, 16, 16, 0};
        Add(s1, s1, dtBias, 64, 32, bAdd);
        Add(s1[64], s1[64], dtBias[64], 64, 32, bAdd);
        PipeBarrier<PIPE_V>();
        if (t->hasLowerBound) {
          Muls(s1, s1, -A, 32 * d);
          PipeBarrier<PIPE_V>();
          Mins(s1, s1, 80.0f, 32 * d);
          PipeBarrier<PIPE_V>();
          Maxs(s1, s1, -80.0f, 32 * d);
          PipeBarrier<PIPE_V>();
          Exp(s1, s1, 32 * d);
          PipeBarrier<PIPE_V>();
          Adds(s1, s1, 1.0f, 32 * d);
          PipeBarrier<PIPE_V>();
          Duplicate(s2, lowerBound, 32 * d);
          PipeBarrier<PIPE_V>();
          Div(gCum[half * 32 * d], s2, s1, 32 * d);
        } else {
          Maxs(s2, s1, 0.0f, 32 * d);
          Abs(s1, s1, 32 * d);
          PipeBarrier<PIPE_V>();
          Muls(s1, s1, -1.0f, 32 * d);
          PipeBarrier<PIPE_V>();
          Exp(s1, s1, 32 * d);
          PipeBarrier<PIPE_V>();
          Adds(s1, s1, 1.0f, 32 * d);
          PipeBarrier<PIPE_V>();
          Ln(s1, s1, 32 * d);
          PipeBarrier<PIPE_V>();
          Add(s1, s2, s1, 32 * d);
          PipeBarrier<PIPE_V>();
          Muls(gCum[half * 32 * d], s1, -A, 32 * d);
        }
        PipeBarrier<PIPE_V>();
      }

      Adds(qFp[0], gCum[0], 0.0f, 1 * d);
      Add(qFp[1 * d], gCum[1 * d], gCum[0], 63 * d);
      PipeBarrier<PIPE_V>();
      Adds(gCum[0], qFp[0], 0.0f, 2 * d);
      Add(gCum[2 * d], qFp[2 * d], qFp[0], 62 * d);
      PipeBarrier<PIPE_V>();
      Adds(qFp[0], gCum[0], 0.0f, 4 * d);
      Add(qFp[4 * d], gCum[4 * d], gCum[0], 60 * d);
      PipeBarrier<PIPE_V>();
      Adds(gCum[0], qFp[0], 0.0f, 8 * d);
      Add(gCum[8 * d], qFp[8 * d], qFp[0], 56 * d);
      PipeBarrier<PIPE_V>();
      Adds(qFp[0], gCum[0], 0.0f, 16 * d);
      Add(qFp[16 * d], gCum[16 * d], gCum[0], 48 * d);
      PipeBarrier<PIPE_V>();
      Adds(gCum[0], qFp[0], 0.0f, 32 * d);
      Add(gCum[32 * d], qFp[32 * d], qFp[0], 32 * d);
      PipeBarrier<PIPE_V>();

      Mins(gRef, gCum[(C - 1) * d], 80.0f, d);
      PipeBarrier<PIPE_V>();
      Maxs(gRef, gRef, -80.0f, d);
      PipeBarrier<PIPE_V>();
      Exp(gRef, gRef, d);
      PipeBarrier<PIPE_ALL>();
      {
        GlobalTensor<float> egGm;
        egGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ws + t->wsExpGLastOff) + hcIdx * d, d);
        DataCopy(egGm, gRef, d);
        if (t->saveForBwd != 0U) {
          const auto cacheL = MegaKdaCache::ComputeLayout(t->seqLocal, H, d, C);
          GlobalTensor<float> gcCacheGm, egCacheGm;
          gcCacheGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(inputList[IDX_BWD_CACHE] + cacheL.gCumOff) + hcIdx * C * d, C * d);
          egCacheGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(inputList[IDX_BWD_CACHE] + cacheL.expGLastOff) + hcIdx * d, d);
          DataCopy(gcCacheGm, gCum, C * d);
          DataCopy(egCacheGm, gRef, d);
        }
      }

      {
        GlobalTensor<bfloat16_t> vGm, vbGm;
        vGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + t->wsVConvOff) +
                                (static_cast<uint64_t>(h) * t->seqEff + tStart) * d,
                            C * d);
        vbGm.SetGlobalBuffer(vbPtr, C * d);
        DataCopy(bfTmp, vGm, C * d);
        PipeBarrier<PIPE_ALL>();
        Cast(qFp, bfTmp, RoundMode::CAST_NONE, C * d);
        PipeBarrier<PIPE_V>();
        BinaryRepeatParams bScale{1, 1, 0, 16, 16, 1};
        Mul(qFp, qFp, betaBrcb, 64, C, bScale);
        Mul(qFp[64], qFp[64], betaBrcb, 64, C, bScale);
        PipeBarrier<PIPE_V>();
        Cast(bfTmp, qFp, RoundMode::CAST_RINT, C * d);
        PipeBarrier<PIPE_ALL>();
        DataCopy(vbGm, bfTmp, C * d);
        PipeBarrier<PIPE_ALL>();
      }

      {
        GlobalTensor<bfloat16_t> qGm, kGm;
        qGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + t->wsQHatOff) +
                                (static_cast<uint64_t>(h) * t->seqEff + tStart) * d,
                            C * d);
        kGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + t->wsKHatOff) +
                                (static_cast<uint64_t>(h) * t->seqEff + tStart) * d,
                            C * d);
        DataCopy(bfTmp, qGm, C * d);
        PipeBarrier<PIPE_ALL>();
        Cast(qFp, bfTmp, RoundMode::CAST_NONE, C * d);
        PipeBarrier<PIPE_ALL>();
        DataCopy(bfTmp, kGm, C * d);
        PipeBarrier<PIPE_ALL>();
        Cast(kFp, bfTmp, RoundMode::CAST_NONE, C * d);
        PipeBarrier<PIPE_ALL>();
      }

      for (uint32_t half = 0; half < 2U; ++half) {
        const uint32_t rOff = half * 32 * d;
        Maxs(s1, gCum[rOff], -80.0f, 32 * d);
        PipeBarrier<PIPE_V>();
        Exp(s1, s1, 32 * d);
        PipeBarrier<PIPE_V>();

        Mul(s2, qFp[rOff], s1, 32 * d);
        PipeBarrier<PIPE_V>();
        Muls(s2, s2, scale, 32 * d);
        PipeBarrier<PIPE_V>();
        Cast(bfTmp[0], s2, RoundMode::CAST_RINT, 32 * d);

        Mul(s2, kFp[rOff], s1, 32 * d);
        PipeBarrier<PIPE_V>();
        BinaryRepeatParams bScale{1, 1, 0, 16, 16, 1};
        Mul(s2, s2, betaBrcb[half * 32 * 8], 64, 32, bScale);
        Mul(s2[64], s2[64], betaBrcb[half * 32 * 8], 64, 32, bScale);
        PipeBarrier<PIPE_V>();
        Cast(bfTmp[32 * d], s2, RoundMode::CAST_RINT, 32 * d);
        PipeBarrier<PIPE_ALL>();

        GlobalTensor<bfloat16_t> qgGm, kbgGm;
        qgGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + t->wsQgOff) + hcIdx * C * d + rOff, 32 * d);
        kbgGm.SetGlobalBuffer(kbgPtr + rOff, 32 * d);
        DataCopy(qgGm, bfTmp[0], 32 * d);
        DataCopy(kbgGm, bfTmp[32 * d], 32 * d);
        if (t->saveForBwd != 0U) {
          const auto cacheL = MegaKdaCache::ComputeLayout(t->seqLocal, H, d, C);
          GlobalTensor<bfloat16_t> qgCacheGm;
          qgCacheGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(inputList[IDX_BWD_CACHE] + cacheL.qgOff) + hcIdx * C * d + rOff, 32 * d);
          DataCopy(qgCacheGm, bfTmp[0], 32 * d);
        }
        PipeBarrier<PIPE_ALL>();
      }

      Adds(gRef, gCum[(C - 1) * d], 0.0f, d);
      PipeBarrier<PIPE_V>();
      for (uint32_t half = 0; half < 2U; ++half) {
        const uint32_t rOff = half * 32 * d;
        BinaryRepeatParams bSub{1, 1, 1, 16, 0, 16};
        Sub(s1, gRef, gCum[rOff], 64, 32, bSub);
        Sub(s1[64], gRef[64], gCum[rOff + 64], 64, 32, bSub);
        PipeBarrier<PIPE_V>();
        Mins(s1, s1, 80.0f, 32 * d);
        PipeBarrier<PIPE_V>();
        Maxs(s1, s1, -80.0f, 32 * d);
        PipeBarrier<PIPE_V>();
        Exp(s1, s1, 32 * d);
        PipeBarrier<PIPE_V>();
        Mul(s2, kFp[rOff], s1, 32 * d);
        PipeBarrier<PIPE_V>();
        Cast(bfTmp[rOff], s2, RoundMode::CAST_RINT, 32 * d);
        PipeBarrier<PIPE_V>();
      }
      PipeBarrier<PIPE_ALL>();
      {
        GlobalTensor<bfloat16_t> kgGm;
        kgGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + t->wsKgOff) + hcIdx * C * d, C * d);
        DataCopy(kgGm, bfTmp, C * d);
        if (t->saveForBwd != 0U) {
          const auto cacheL = MegaKdaCache::ComputeLayout(t->seqLocal, H, d, C);
          GlobalTensor<bfloat16_t> kgCacheGm;
          kgCacheGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(inputList[IDX_BWD_CACHE] + cacheL.kgOff) + hcIdx * C * d, C * d);
          DataCopy(kgCacheGm, bfTmp, C * d);
        }
        PipeBarrier<PIPE_ALL>();
      }

      for (uint32_t a = 0; a < 4U; ++a) {
        const uint32_t base = a * 16U;
        const uint32_t dRowOff = a * 48U;
        Adds(gRef, gCum[(base + 7U) * d], 0.0f, d);
        PipeBarrier<PIPE_V>();

        BinaryRepeatParams bSub{1, 1, 1, 16, 16, 0};
        Sub(s2, gCum[base * d], gRef, 64, 16, bSub);
        Sub(s2[64], gCum[base * d + 64], gRef[64], 64, 16, bSub);
        PipeBarrier<PIPE_V>();
        Mins(s2, s2, 40.0f, 16 * d);
        PipeBarrier<PIPE_V>();
        Maxs(s2, s2, -40.0f, 16 * d);
        PipeBarrier<PIPE_V>();
        Exp(s2, s2, 16 * d);
        PipeBarrier<PIPE_V>();

        Mul(s1, kFp[base * d], s2, 16 * d);
        PipeBarrier<PIPE_V>();
        BinaryRepeatParams bScale{1, 1, 0, 16, 16, 1};
        Mul(s1, s1, betaBrcb[base * 8], 64, 16, bScale);
        Mul(s1[64], s1[64], betaBrcb[base * 8], 64, 16, bScale);
        PipeBarrier<PIPE_V>();
        Muls(s1, s1, -1.0f, 16 * d);
        PipeBarrier<PIPE_V>();
        Cast(bfTmp[0], s1, RoundMode::CAST_RINT, 16 * d);

        Mul(s1, qFp[base * d], s2, 16 * d);
        PipeBarrier<PIPE_V>();
        Muls(s1, s1, scale, 16 * d);
        PipeBarrier<PIPE_V>();
        Cast(bfTmp[16 * d], s1, RoundMode::CAST_RINT, 16 * d);

        Div(s1, kFp[base * d], s2, 16 * d);
        PipeBarrier<PIPE_V>();
        Cast(bfTmp[32 * d], s1, RoundMode::CAST_RINT, 16 * d);
        PipeBarrier<PIPE_ALL>();

        GlobalTensor<bfloat16_t> dGm;
        dGm.SetGlobalBuffer(subInPtr + dRowOff * d, 48 * d);
        DataCopy(dGm, bfTmp, 48 * d);
        PipeBarrier<PIPE_ALL>();

        if (a >= 1U) {
          const uint32_t offBBase[3] = {192U, 208U, 240U};
          const uint32_t cols = a * 16U;
          for (uint32_t j0 = 0; j0 < cols; j0 += 32U) {
            const uint32_t cur = (j0 + 32U <= cols) ? 32U : (cols - j0);
            BinaryRepeatParams bSubB{1, 1, 1, 16, 0, 16};
            Sub(s2, gRef, gCum[j0 * d], 64, static_cast<uint8_t>(cur), bSubB);
            Sub(s2[64], gRef[64], gCum[j0 * d + 64], 64, static_cast<uint8_t>(cur), bSubB);
            PipeBarrier<PIPE_V>();
            Maxs(s2, s2, -40.0f, cur * d);
            PipeBarrier<PIPE_V>();
            Exp(s2, s2, cur * d);
            PipeBarrier<PIPE_V>();
            Mul(s1, kFp[j0 * d], s2, cur * d);
            PipeBarrier<PIPE_V>();
            Cast(bfTmp[j0 * d], s1, RoundMode::CAST_RINT, cur * d);
            PipeBarrier<PIPE_V>();
          }
          PipeBarrier<PIPE_ALL>();
          GlobalTensor<bfloat16_t> bGm;
          bGm.SetGlobalBuffer(subInPtr + offBBase[a - 1U] * d, cols * d);
          DataCopy(bGm, bfTmp, cols * d);
          PipeBarrier<PIPE_ALL>();
        }
      }

      CrossCoreSetFlag<0x2, PIPE_MTE3>(0U + slot);
      CrossCoreWaitFlag<0x2, PIPE_MTE2>(2U + slot);

      {
        GlobalTensor<float> subGm;
        GlobalTensor<bfloat16_t> akkGm, aqkGm;
        subGm.SetGlobalBuffer(subOutPtr, 5120);
        akkGm.SetGlobalBuffer(akkPtr, C * C);
        aqkGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + t->wsAqkOff) + hcIdx * C * C, C * C);

        DataCopy(subOut, subGm, 5120);
        Duplicate(matA, 0.0f, C * C);
        Duplicate(matAqk, 0.0f, C * C);
        PipeBarrier<PIPE_ALL>();

        for (uint32_t a = 0; a < 4U; ++a) {
          const uint32_t dOff = 3072U + a * 512U;
          Mul(subOut[dOff], subOut[dOff], maskStrict, 256);
          Mul(subOut[dOff + 256], subOut[dOff + 256], maskIncl, 256);
        }
        PipeBarrier<PIPE_V>();

        for (uint32_t a = 0; a < 4U; ++a) {
          const uint32_t dOff = 3072U + a * 512U;
          const uint32_t base = a * 16U;
          UnaryRepeatParams uPack{1, 1, 8, 2};
          Adds(matA[base * C + base], subOut[dOff], 0.0f, 16, 16, uPack);
          Adds(matAqk[base * C + base], subOut[dOff + 256], 0.0f, 16, 16, uPack);
        }

        const uint32_t subOffs[3] = {0, 512, 1536};
        for (uint32_t a = 1U; a <= 3U; ++a) {
          const uint32_t cols = a * 16U;
          const uint32_t off = subOffs[a - 1U];
          UnaryRepeatParams uOff{1, 1, 8, static_cast<uint8_t>(cols / 8U)};
          Adds(matA[a * 16 * C], subOut[off], 0.0f, static_cast<uint64_t>(cols), 16, uOff);
          Adds(matAqk[a * 16 * C], subOut[off + 16 * cols], 0.0f, static_cast<uint64_t>(cols), 16, uOff);
        }
        PipeBarrier<PIPE_V>();

        Cast(bfOut, matAqk, RoundMode::CAST_RINT, C * C);
        PipeBarrier<PIPE_ALL>();
        DataCopy(aqkGm, bfOut, C * C);
        if (t->saveForBwd != 0U) {
          const auto cacheL = MegaKdaCache::ComputeLayout(t->seqLocal, H, d, C);
          GlobalTensor<bfloat16_t> aqkCacheGm;
          aqkCacheGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(inputList[IDX_BWD_CACHE] + cacheL.aqkOff) + hcIdx * C * C, C * C);
          DataCopy(aqkCacheGm, bfOut, C * C);
        }
        PipeBarrier<PIPE_ALL>();

        Cast(bfOut, matA, RoundMode::CAST_RINT, C * C);
        PipeBarrier<PIPE_ALL>();
        DataCopy(akkGm, bfOut, C * C);
        PipeBarrier<PIPE_ALL>();
      }

      CrossCoreSetFlag<0x2, PIPE_MTE3>(4U + slot);
    }

    if (jobCount >= 2U) {
      CrossCoreWaitFlag<0x2, PIPE_MTE3>(6U + ((jobCount - 2U) & 1U));
    }
    if (jobCount >= 1U) {
      CrossCoreWaitFlag<0x2, PIPE_MTE3>(6U + ((jobCount - 1U) & 1U));
    }

    pipe.Destroy();
    PipeBarrier<PIPE_ALL>();
    SyncAll();
  }
}

__aicore__ inline void RunCp1FusedPrepareAic(uint32_t core, GM_ADDR *inputList,
                                             const HyperMegaKdaLaunchTilingData *t) {
  if ASCEND_IS_AIC {
    const uint32_t numCube = t->numCubeCores;
    const uint32_t H = t->numHeads;
    const uint32_t d = t->headDim;
    const uint32_t C = t->chunkSize;
    const uint32_t numChunks = t->numChunks;
    const uint32_t numPairs = H >> 1;
    const uint32_t totalJobs = numChunks * numPairs;
    GM_ADDR ws = inputList[IDX_WORKSPACE];
    GM_ADDR prepCoreBase = ws + t->wsSubInOff + static_cast<uint64_t>(core) * CP1_PREP_CORE_BYTES;

    KdaRawCubePipe cube;
    cube.Init(72 * 1024, 16 * 1024, 32 * 1024, 64 * 1024);
    auto l1Akk = cube.L1At(0);
    auto l1Tmp = cube.L1At(8 * 1024);
    auto l1Vb = cube.L1At(16 * 1024);
    auto l1Kbg = cube.L1At(32 * 1024);
    auto l1Eye = cube.L1At(48 * 1024);
    auto l1Work = cube.L1At(50 * 1024);
    auto l1Eye64 = cube.L1At(64 * 1024);
    auto A = cube.L0A();
    auto B = cube.L0B();
    auto C0 = cube.L0C();

    bool eyeLoaded = false;

    for (uint32_t j = 0, job = core; job < totalJobs; ++j, job += numCube) {
      const uint32_t c = job / numPairs;
      const uint32_t pair = job % numPairs;
      const uint32_t slot = j & 1U;
      GM_ADDR slotBase = prepCoreBase + slot * CP1_PREP_SLOT_BYTES;

      CrossCoreWaitFlag<0x2, PIPE_MTE2>(0U + slot);

      if (!eyeLoaded) {
        GlobalTensor<bfloat16_t> eyeGm;
        eyeGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + t->wsEye16Off), 256);
        cube.CopyGmToL1Nz(l1Eye, eyeGm, 16, 16, 16, 16);
        if (t->saveForBwd != 0U) {
          GlobalTensor<bfloat16_t> eye64Gm;
          eye64Gm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + t->wsEye16Off + 512ULL), 4096);
          cube.CopyGmToL1Nz(l1Eye64, eye64Gm, 64, 64, 64, 64);
        }
        cube.WaitMte2ToMte1();
        eyeLoaded = true;
      }

      for (uint32_t lane = 0; lane < 2U; ++lane) {
        GM_ADDR laneBase = slotBase + lane * CP1_PREP_LANE_BYTES;
        __gm__ bfloat16_t *inPtr = reinterpret_cast<__gm__ bfloat16_t *>(laneBase + CP1_PREP_OFF_SUBIN);
        __gm__ float *outPtr = reinterpret_cast<__gm__ float *>(laneBase + CP1_PREP_OFF_SUBOUT);

        RunIntraRawTile(cube, l1Akk, l1Vb, A, B, C0, inPtr + 48ULL * d, inPtr + 192ULL * d, outPtr + 0ULL, d, 16);
        RunIntraRawTile(cube, l1Akk, l1Vb, A, B, C0, inPtr + 96ULL * d, inPtr + 208ULL * d, outPtr + 512ULL, d, 32);
        RunIntraRawTile(cube, l1Akk, l1Vb, A, B, C0, inPtr + 144ULL * d, inPtr + 240ULL * d, outPtr + 1536ULL, d, 48);
        for (uint32_t a = 0; a < 4U; ++a) {
          const uint64_t dRowOff = a * 48ULL;
          RunIntraRawTile(cube, l1Akk, l1Vb, A, B, C0, inPtr + dRowOff * d, inPtr + (dRowOff + 32ULL) * d,
                          outPtr + 3072ULL + a * 512ULL, d, 16);
        }
      }
      CrossCoreSetFlag<0x2, PIPE_FIX>(2U + slot);

      CrossCoreWaitFlag<0x2, PIPE_MTE2>(4U + slot);

      for (uint32_t lane = 0; lane < 2U; ++lane) {
        const uint32_t h = (pair << 1) + lane;
        const uint64_t hcIdx = static_cast<uint64_t>(h) * numChunks + c;
        GM_ADDR laneBase = slotBase + lane * CP1_PREP_LANE_BYTES;

        GlobalTensor<bfloat16_t> akkGm, vbGm, kbgGm, wGm;
        GlobalTensor<float> uGm;
        akkGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(laneBase + CP1_PREP_OFF_AKK), C * C);
        vbGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(laneBase + CP1_PREP_OFF_VBETA), C * d);
        kbgGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(laneBase + CP1_PREP_OFF_KBG), C * d);
        uGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ws + t->wsUOff) + hcIdx * C * d, C * d);
        wGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + t->wsWOff) + hcIdx * C * d, C * d);

        cube.CopyGmToL1Nz(l1Akk, akkGm, 64, 64, 64, 64);
        cube.CopyGmToL1Nz(l1Vb, vbGm, 64, d, d, 64);
        cube.CopyGmToL1Nz(l1Kbg, kbgGm, 64, d, d, 64);
        cube.WaitMte2ToMte1();

        for (uint32_t a = 0; a < 4U; ++a) {
          cube.InvertDiag16InL1(l1Akk, a * 16U, l1Eye, l1Work);
        }
        cube.SubMatmul64ToL1(l1Akk, 16, 16, l1Akk, 16, 0, l1Tmp, 0, 0, 16, 16, 16, 64);
        cube.SubMatmul64ToL1(l1Tmp, 0, 0, l1Akk, 0, 0, l1Akk, 16, 0, 16, 16, 16, 64);
        cube.SubMatmul64ToL1(l1Akk, 48, 48, l1Akk, 48, 32, l1Tmp, 0, 0, 16, 16, 16, 64);
        cube.SubMatmul64ToL1(l1Tmp, 0, 0, l1Akk, 32, 32, l1Akk, 48, 32, 16, 16, 16, 64);
        cube.SubMatmul64ToL1(l1Akk, 32, 32, l1Akk, 32, 0, l1Tmp, 0, 0, 32, 32, 32, 64);
        cube.SubMatmul64ToL1(l1Tmp, 0, 0, l1Akk, 0, 0, l1Akk, 32, 0, 32, 32, 32, 64);

        if (t->saveForBwd != 0U) {
          const auto cacheL = MegaKdaCache::ComputeLayout(t->seqLocal, H, d, C);
          GlobalTensor<bfloat16_t> mInvCacheGm;
          mInvCacheGm.SetGlobalBuffer(
              reinterpret_cast<__gm__ bfloat16_t *>(inputList[IDX_BWD_CACHE] + cacheL.mInvOff) + hcIdx * C * C,
              C * C);
          cube.LoadL0A_NoT(A, l1Akk, 64, 64, 64);
          cube.LoadL0B_NoT(B, l1Eye64, 64, 64, 64);
          cube.WaitMte1ToM();
          cube.RunMmad(C0, A, B, 64, 64, 64, true);
          cube.WaitMToMte1();
          cube.WaitMToFix();
          cube.StoreBf16Gm(mInvCacheGm, C0, 64, 64, 64);
          cube.WaitFixToM();
        }

        cube.LoadL0A_NoT(A, l1Akk, 64, 64, 64);
        cube.LoadL0B_NoT(B, l1Vb, 64, d, 64);
        cube.WaitMte1ToM();
        cube.RunMmad(C0, A, B, 64, d, 64, true);
        cube.WaitMToMte1();
        cube.WaitMToFix();
        cube.StoreFp32Gm(uGm, C0, 64, d, d);
        if (t->saveForBwd != 0U) {
          const auto cacheL = MegaKdaCache::ComputeLayout(t->seqLocal, H, d, C);
          GlobalTensor<bfloat16_t> uCacheGm;
          uCacheGm.SetGlobalBuffer(
              reinterpret_cast<__gm__ bfloat16_t *>(inputList[IDX_BWD_CACHE] + cacheL.uOff) + hcIdx * C * d, C * d);
          cube.StoreBf16Gm(uCacheGm, C0, 64, d, d);
        }
        cube.WaitFixToM();

        cube.LoadL0A_NoT(A, l1Akk, 64, 64, 64);
        cube.LoadL0B_NoT(B, l1Kbg, 64, d, 64);
        cube.WaitMte1ToMte2();
        cube.WaitMte1ToM();
        cube.RunMmad(C0, A, B, 64, d, 64, true);
        cube.WaitMToMte1();
        cube.WaitMToFix();
        cube.StoreBf16Gm(wGm, C0, 64, d, d);
        if (t->saveForBwd != 0U) {
          const auto cacheL = MegaKdaCache::ComputeLayout(t->seqLocal, H, d, C);
          GlobalTensor<bfloat16_t> wCacheGm;
          wCacheGm.SetGlobalBuffer(
              reinterpret_cast<__gm__ bfloat16_t *>(inputList[IDX_BWD_CACHE] + cacheL.wOff) + hcIdx * C * d, C * d);
          cube.StoreBf16Gm(wCacheGm, C0, 64, d, d);
        }
        cube.WaitFixToM();
      }
      CrossCoreSetFlag<0x2, PIPE_FIX>(6U + slot);
    }

    cube.Destroy();
  }
}

// ---------------------------------------------------------------------------
// Phase 3: 1:2 AIC <-> 2xAIV Fused FwdH + FinalizeO + Overlapped GatedRMSNorm
// Supports: CP=1 (initFromExcl=false, publishRank0State=false),
//           CP>1 Rank 0 (initFromExcl=false, publishRank0State=true),
//           CP>1 Rank>=1 Pass-2 (initFromExcl=true, publishRank0State=false)!
// ---------------------------------------------------------------------------
__aicore__ inline void RunCp1FusedFwdHAic(uint32_t core, GM_ADDR *inputList,
                                          const HyperMegaKdaLaunchTilingData *t) {
  if ASCEND_IS_AIC {
    const uint32_t numCube = t->numCubeCores;
    const uint32_t H = t->numHeads;
    const uint32_t d = t->headDim;
    const uint32_t C = t->chunkSize;
    const uint32_t numChunks = t->numChunks;
    GM_ADDR ws = inputList[IDX_WORKSPACE];
    GM_ADDR scBase = ws + t->wsFwdHScratchOff + static_cast<uint64_t>(core) * t->wsFwdHScratchStride;

    GlobalTensor<float> wsGm, dsGm;
    GlobalTensor<bfloat16_t> hGm, vnGm, oGm0, oGm1;
    wsGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(scBase + CP1_FWD_OFF_WS), C * d);
    dsGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(scBase + CP1_FWD_OFF_DS), d * d);
    hGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(scBase + CP1_FWD_OFF_H), d * d);
    vnGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(scBase + CP1_FWD_OFF_VN), C * d);
    oGm0.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(scBase + CP1_FWD_OFF_O0), C * d);
    oGm1.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(scBase + CP1_FWD_OFF_O1), C * d);

    KdaRawCubePipe cube;
    cube.Init(80 * 1024, 32 * 1024, 32 * 1024, 64 * 1024);
    auto l1H = cube.L1At(0);
    auto l1W = cube.L1At(32 * 1024);
    auto l1Qg = cube.L1At(48 * 1024);
    auto l1Vn = cube.L1At(64 * 1024);
    auto A = cube.L0A();
    auto B = cube.L0B();
    auto C0 = cube.L0C();

    for (uint32_t h = core; h < H; h += numCube) {
      for (uint32_t c = 0; c < numChunks; ++c) {
        const uint64_t hcIdx = static_cast<uint64_t>(h) * numChunks + c;
        GlobalTensor<bfloat16_t> wGm, qgGm, aqkGm, kgGm;
        wGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + t->wsWOff) + hcIdx * C * d, C * d);
        qgGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + t->wsQgOff) + hcIdx * C * d, C * d);
        aqkGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + t->wsAqkOff) + hcIdx * C * C, C * C);
        kgGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + t->wsKgOff) + hcIdx * C * d, C * d);

        CrossCoreWaitFlag<0x2, PIPE_MTE2>(0);

        cube.CopyGmToL1Nz(l1H, hGm, d, d, d, d);
        cube.CopyGmToL1Nz(l1W, wGm, 64, d, d, 64);
        cube.CopyGmToL1Nz(l1Qg, qgGm, 64, d, d, 64);
        cube.WaitMte2ToMte1();

        cube.LoadL0A_NoT(A, l1W, 64, d, 64);
        cube.LoadL0B_NoT(B, l1H, d, d, d);
        cube.WaitMte1ToM();
        cube.RunMmad(C0, A, B, 64, d, d, true);
        cube.WaitMToMte1();
        cube.WaitMToFix();
        cube.StoreFp32Gm(wsGm, C0, 64, d, d);
        cube.WaitFixToM();
        CrossCoreSetFlag<0x2, PIPE_FIX>(1);

        cube.LoadL0A_NoT(A, l1Qg, 64, d, 64);
        cube.LoadL0B_NoT(B, l1H, d, d, d);
        cube.WaitMte1ToMte2();
        cube.WaitMte1ToM();
        cube.RunMmad(C0, A, B, 64, d, d, true);
        cube.WaitMToMte1();

        cube.CopyGmToL1Nz(l1W, aqkGm, 64, 64, 64, 64);
        cube.CopyGmToL1Nz(l1Qg, kgGm, 64, d, d, 64);

        CrossCoreWaitFlag<0x2, PIPE_MTE2>(2);
        cube.CopyGmToL1Nz(l1Vn, vnGm, 64, d, d, 64);
        cube.WaitMte2ToMte1();

        cube.LoadL0A_NoT(A, l1W, 64, 64, 64);
        cube.LoadL0B_NoT(B, l1Vn, 64, d, 64);
        cube.WaitMte1ToM();
        cube.RunMmad(C0, A, B, 64, d, 64, false);
        cube.WaitMToMte1();
        cube.WaitMToFix();
        cube.StoreBf16Gm((c & 1U) ? oGm1 : oGm0, C0, 64, d, d);
        cube.WaitFixToM();

        cube.LoadL0A_Trans(A, l1Qg, d, 64);
        cube.LoadL0B_NoT(B, l1Vn, 64, d, 64);
        cube.WaitMte1ToMte2();
        cube.WaitMte1ToM();
        cube.RunMmad(C0, A, B, d, d, 64, true);
        cube.WaitMToMte1();
        cube.WaitMToFix();
        cube.StoreFp32Gm(dsGm, C0, d, d, d);
        cube.WaitFixToM();
        CrossCoreSetFlag<0x2, PIPE_FIX>(3);
      }
    }
    cube.Destroy();
  }
}

__aicore__ inline void RunCp1GatedRmsNorm32(
    LocalTensor<float> &wFp, LocalTensor<bfloat16_t> &oBf, LocalTensor<bfloat16_t> &gBf,
    LocalTensor<float> &oFp, LocalTensor<float> &gFp, LocalTensor<float> &tmp,
    LocalTensor<float> &rmsVec, LocalTensor<float> &rmsBrcb,
    __gm__ bfloat16_t *oSrcPtr, __gm__ bfloat16_t *gOutPtr, __gm__ bfloat16_t *dstPtr,
    uint32_t d, uint32_t totalHeadDim, float eps,
    __gm__ bfloat16_t *oAttnCacheDst = nullptr, __gm__ float *oRstdCacheDst = nullptr) {
  constexpr uint32_t R = 32U;
  const float invD = 1.0f / static_cast<float>(static_cast<int32_t>(d));
  const uint16_t blockLen = static_cast<uint16_t>((d * 2U) / 32U);
  const uint16_t stride = static_cast<uint16_t>(((totalHeadDim - d) * 2U) / 32U);

  GlobalTensor<bfloat16_t> oGm, gGm, dGm;
  oGm.SetGlobalBuffer(oSrcPtr, R * d);
  gGm.SetGlobalBuffer(gOutPtr, static_cast<uint64_t>(R) * totalHeadDim);
  dGm.SetGlobalBuffer(dstPtr, static_cast<uint64_t>(R) * totalHeadDim);

  DataCopy(oBf, oGm, R * d);
  DataCopyParams inPG{R, blockLen, stride, 0};
  DataCopy(gBf, gGm, inPG);
  PipeBarrier<PIPE_ALL>();
  if (oAttnCacheDst != nullptr) {
    GlobalTensor<bfloat16_t> oaGm;
    oaGm.SetGlobalBuffer(oAttnCacheDst, R * d);
    DataCopy(oaGm, oBf, R * d);
    PipeBarrier<PIPE_ALL>();
  }

  Cast(oFp, oBf, RoundMode::CAST_NONE, R * d);
  Cast(gFp, gBf, RoundMode::CAST_NONE, R * d);
  PipeBarrier<PIPE_V>();

  Mul(tmp, oFp, oFp, R * d);
  PipeBarrier<PIPE_V>();
  BinaryRepeatParams foldP{1, 1, 1, 16, 16, 16};
  Add(tmp, tmp, tmp[64], 64, R, foldP);
  PipeBarrier<PIPE_V>();
  WholeReduceSum(rmsVec, tmp, 64, static_cast<int32_t>(R), 1, 1, 16);
  PipeBarrier<PIPE_V>();
  Muls(rmsVec, rmsVec, invD, R);
  PipeBarrier<PIPE_V>();
  Adds(rmsVec, rmsVec, eps, R);
  PipeBarrier<PIPE_V>();
  Sqrt(rmsVec, rmsVec, R);
  PipeBarrier<PIPE_V>();
  Duplicate(tmp, 1.0f, R);
  PipeBarrier<PIPE_V>();
  Div(rmsVec, tmp, rmsVec, R);
  PipeBarrier<PIPE_V>();
  if (oRstdCacheDst != nullptr) {
    PipeBarrier<PIPE_ALL>();
    GlobalTensor<float> orGm;
    orGm.SetGlobalBuffer(oRstdCacheDst, R);
    DataCopy(orGm, rmsVec, R);
    PipeBarrier<PIPE_ALL>();
  }
  Brcb(rmsBrcb, rmsVec, static_cast<uint8_t>(R / 8U), {1, 8});
  PipeBarrier<PIPE_V>();
  BinaryRepeatParams scaleP{1, 1, 0, 16, 16, 1};
  Mul(oFp, oFp, rmsBrcb, 64, R, scaleP);
  Mul(oFp[64], oFp[64], rmsBrcb, 64, R, scaleP);
  PipeBarrier<PIPE_V>();
  BinaryRepeatParams wMul{1, 1, 1, 16, 16, 0};
  Mul(oFp, oFp, wFp, 64, R, wMul);
  Mul(oFp[64], oFp[64], wFp[64], 64, R, wMul);
  PipeBarrier<PIPE_V>();

  Muls(gFp, gFp, -1.0f, R * d);
  PipeBarrier<PIPE_V>();
  Mins(gFp, gFp, 80.0f, R * d);
  PipeBarrier<PIPE_V>();
  Maxs(gFp, gFp, -80.0f, R * d);
  PipeBarrier<PIPE_V>();
  Exp(gFp, gFp, R * d);
  PipeBarrier<PIPE_V>();
  Adds(gFp, gFp, 1.0f, R * d);
  PipeBarrier<PIPE_V>();
  Div(oFp, oFp, gFp, R * d);
  PipeBarrier<PIPE_V>();

  Cast(oBf, oFp, RoundMode::CAST_RINT, R * d);
  PipeBarrier<PIPE_ALL>();
  DataCopyParams outP{R, blockLen, 0, stride};
  DataCopy(dGm, oBf, outP);
  PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void RunCp1FusedFwdHAiv(uint32_t wid, GM_ADDR *inputList,
                                          const HyperMegaKdaLaunchTilingData *t,
                                          bool initFromExcl = false, bool publishRank0State = false) {
  if ASCEND_IS_AIV {
    const uint32_t core = wid >> 1;
    const uint32_t sub = wid & 1U;
    const uint32_t numCube = t->numCubeCores;
    const uint32_t H = t->numHeads;
    const uint32_t d = t->headDim;
    const uint32_t C = t->chunkSize;
    const uint32_t halfC = C >> 1;
    const uint32_t halfD = d >> 1;
    const uint32_t numChunks = t->numChunks;
    const uint32_t totalHeadDim = H * d;
    const float eps = BitsToFloat(t->normEpsBits);
    GM_ADDR ws = inputList[IDX_WORKSPACE];
    GM_ADDR scBase = ws + t->wsFwdHScratchOff + static_cast<uint64_t>(core) * t->wsFwdHScratchStride;

    __gm__ float *wsPtr = reinterpret_cast<__gm__ float *>(scBase + CP1_FWD_OFF_WS);
    __gm__ float *dsPtr = reinterpret_cast<__gm__ float *>(scBase + CP1_FWD_OFF_DS);
    __gm__ bfloat16_t *hPtr = reinterpret_cast<__gm__ bfloat16_t *>(scBase + CP1_FWD_OFF_H);
    __gm__ bfloat16_t *vnPtr = reinterpret_cast<__gm__ bfloat16_t *>(scBase + CP1_FWD_OFF_VN);
    __gm__ bfloat16_t *oPtr0 = reinterpret_cast<__gm__ bfloat16_t *>(scBase + CP1_FWD_OFF_O0);
    __gm__ bfloat16_t *oPtr1 = reinterpret_cast<__gm__ bfloat16_t *>(scBase + CP1_FWD_OFF_O1);
    __gm__ bfloat16_t *gOutAll = reinterpret_cast<__gm__ bfloat16_t *>(inputList[IDX_WGB]);
    __gm__ bfloat16_t *outAll = reinterpret_cast<__gm__ bfloat16_t *>(inputList[IDX_OUT]);

    TPipe pipe;
    TBuf<TPosition::VECCALC> buf;
    const uint32_t uState = 0;
    const uint32_t uWNorm = uState + halfD * d * 4;
    const uint32_t uWork = uWNorm + d * 4;

    const uint32_t uB1 = uWork;
    const uint32_t uB2 = uB1 + halfD * d * 4;
    const uint32_t uBf = uB2 + halfC * d * 4;
    const uint32_t uExpG = uBf + halfD * d * 2;
    const uint32_t uExpGBrcb = uExpG + halfD * 4;

    const uint32_t uOBf = uWork;
    const uint32_t uGBf = uOBf + halfC * d * 2;
    const uint32_t uOFp = uGBf + halfC * d * 2;
    const uint32_t uGFp = uOFp + halfC * d * 4;
    const uint32_t uTmp = uGFp + halfC * d * 4;
    const uint32_t uRms = uTmp + halfC * d * 4;
    const uint32_t uRmsBrcb = uRms + halfC * 4;

    pipe.InitBuffer(buf, uExpGBrcb + halfD * 8 * 4 + 64U);

    LocalTensor<float> sUb = buf.GetWithOffset<float>(halfD * d, uState);
    LocalTensor<float> wNormFp = buf.GetWithOffset<float>(d, uWNorm);

    LocalTensor<float> b1 = buf.GetWithOffset<float>(halfD * d, uB1);
    LocalTensor<float> b2 = buf.GetWithOffset<float>(halfC * d, uB2);
    LocalTensor<bfloat16_t> bfOut = buf.GetWithOffset<bfloat16_t>(halfD * d, uBf);
    LocalTensor<float> expG = buf.GetWithOffset<float>(halfD, uExpG);
    LocalTensor<float> expGBrcb = buf.GetWithOffset<float>(halfD * 8, uExpGBrcb);

    LocalTensor<bfloat16_t> oBf = buf.GetWithOffset<bfloat16_t>(halfC * d, uOBf);
    LocalTensor<bfloat16_t> gBf = buf.GetWithOffset<bfloat16_t>(halfC * d, uGBf);
    LocalTensor<float> oFp = buf.GetWithOffset<float>(halfC * d, uOFp);
    LocalTensor<float> gFp = buf.GetWithOffset<float>(halfC * d, uGFp);
    LocalTensor<float> tmp = buf.GetWithOffset<float>(halfC * d, uTmp);
    LocalTensor<float> rmsVec = buf.GetWithOffset<float>(halfC, uRms);
    LocalTensor<float> rmsBrcb = buf.GetWithOffset<float>(halfC * 8, uRmsBrcb);

    {
      GlobalTensor<bfloat16_t> wGm;
      wGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(inputList[IDX_ONORMW]), d);
      DataCopy(oBf, wGm, d);
      PipeBarrier<PIPE_ALL>();
      Cast(wNormFp, oBf, RoundMode::CAST_NONE, d);
      PipeBarrier<PIPE_ALL>();
    }

    const uint32_t vRowOff = sub * halfC;
    const uint32_t sRowOff = sub * halfD;

    for (uint32_t h = core; h < H; h += numCube) {
      const uint32_t g = h / t->headsPerGroup;
      const uint32_t hInG = h % t->headsPerGroup;
      const uint64_t groupOff = g * 2ULL * t->sendScanGroupHalfBytes;
      const uint64_t headElemOff = static_cast<uint64_t>(hInG) * d * d + sRowOff * d;

      if (initFromExcl) {
        GlobalTensor<float> exGm;
        exGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ws + t->wsScanExclOff + groupOff) + headElemOff,
                             halfD * d);
        DataCopy(sUb, exGm, halfD * d);
        PipeBarrier<PIPE_ALL>();
      } else {
        Duplicate(sUb, 0.0f, halfD * d);
        PipeBarrier<PIPE_V>();
      }
      Cast(bfOut, sUb, RoundMode::CAST_RINT, halfD * d);
      PipeBarrier<PIPE_ALL>();
      {
        GlobalTensor<bfloat16_t> hGm;
        hGm.SetGlobalBuffer(hPtr + static_cast<uint64_t>(sRowOff) * d, halfD * d);
        DataCopy(hGm, bfOut, halfD * d);
        if (t->saveForBwd != 0U) {
          const auto cacheL = MegaKdaCache::ComputeLayout(t->seqLocal, H, d, C);
          const uint64_t hc0 = static_cast<uint64_t>(h) * numChunks;
          GlobalTensor<bfloat16_t> hcCacheGm;
          hcCacheGm.SetGlobalBuffer(
              reinterpret_cast<__gm__ bfloat16_t *>(inputList[IDX_BWD_CACHE] + cacheL.hChunksOff) +
                  (hc0 * d + sRowOff) * d,
              halfD * d);
          DataCopy(hcCacheGm, bfOut, halfD * d);
        }
        PipeBarrier<PIPE_ALL>();
      }
      CrossCoreSetFlag<0x2, PIPE_MTE3>(0);

      for (uint32_t c = 0; c < numChunks; ++c) {
        const uint64_t hcIdx = static_cast<uint64_t>(h) * numChunks + c;

        CrossCoreWaitFlag<0x2, PIPE_MTE2>(1);

        {
          GlobalTensor<float> uGm, wsGm;
          GlobalTensor<bfloat16_t> vnGm;
          uGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ws + t->wsUOff) + hcIdx * C * d + vRowOff * d,
                              halfC * d);
          wsGm.SetGlobalBuffer(wsPtr + vRowOff * d, halfC * d);
          vnGm.SetGlobalBuffer(vnPtr + vRowOff * d, halfC * d);

          DataCopy(b1, uGm, halfC * d);
          DataCopy(b2, wsGm, halfC * d);
          PipeBarrier<PIPE_ALL>();
          Sub(b1, b1, b2, halfC * d);
          PipeBarrier<PIPE_V>();
          Cast(bfOut, b1, RoundMode::CAST_RINT, halfC * d);
          PipeBarrier<PIPE_ALL>();
          DataCopy(vnGm, bfOut, halfC * d);
          if (t->saveForBwd != 0U) {
            const auto cacheL = MegaKdaCache::ComputeLayout(t->seqLocal, H, d, C);
            GlobalTensor<bfloat16_t> vnCacheGm;
            vnCacheGm.SetGlobalBuffer(
                reinterpret_cast<__gm__ bfloat16_t *>(inputList[IDX_BWD_CACHE] + cacheL.vNewOff) +
                    (hcIdx * C + vRowOff) * d,
                halfC * d);
            DataCopy(vnCacheGm, bfOut, halfC * d);
          }
          PipeBarrier<PIPE_ALL>();
        }
        CrossCoreSetFlag<0x2, PIPE_MTE3>(2);

        if (c > 0U) {
          const uint32_t prevC = c - 1U;
          const uint64_t rAbs = static_cast<uint64_t>(prevC) * C + vRowOff;
          __gm__ bfloat16_t *oSrc = ((prevC & 1U) ? oPtr1 : oPtr0) + vRowOff * d;
          __gm__ bfloat16_t *oaDst = nullptr;
          __gm__ float *orDst = nullptr;
          if (t->saveForBwd != 0U) {
            const auto cacheL = MegaKdaCache::ComputeLayout(t->seqLocal, H, d, C);
            const uint64_t prevHc = static_cast<uint64_t>(h) * numChunks + prevC;
            oaDst = reinterpret_cast<__gm__ bfloat16_t *>(inputList[IDX_BWD_CACHE] + cacheL.oAttnOff) +
                    (prevHc * C + vRowOff) * d;
            orDst = reinterpret_cast<__gm__ float *>(inputList[IDX_BWD_CACHE] + cacheL.oRstdOff) +
                    static_cast<uint64_t>(h) * t->seqLocal + rAbs;
          }
          RunCp1GatedRmsNorm32(
              wNormFp, oBf, gBf, oFp, gFp, tmp, rmsVec, rmsBrcb,
              oSrc, gOutAll + rAbs * totalHeadDim + static_cast<uint64_t>(h) * d,
              outAll + rAbs * totalHeadDim + static_cast<uint64_t>(h) * d,
              d, totalHeadDim, eps, oaDst, orDst);
        }

        CrossCoreWaitFlag<0x2, PIPE_MTE2>(3);

        if (c + 1U < numChunks || publishRank0State) {
          GlobalTensor<float> egGm, dsGm;
          egGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ws + t->wsExpGLastOff) + hcIdx * d + sRowOff, halfD);
          dsGm.SetGlobalBuffer(dsPtr + sRowOff * d, halfD * d);
          DataCopy(expG, egGm, halfD);
          DataCopy(b1, dsGm, halfD * d);
          PipeBarrier<PIPE_ALL>();

          Brcb(expGBrcb, expG, static_cast<uint8_t>(halfD / 8U), {1, 8});
          PipeBarrier<PIPE_V>();
          BinaryRepeatParams sScale{1, 1, 0, 16, 16, 1};
          Mul(sUb, sUb, expGBrcb, 64, static_cast<uint8_t>(halfD), sScale);
          Mul(sUb[64], sUb[64], expGBrcb, 64, static_cast<uint8_t>(halfD), sScale);
          PipeBarrier<PIPE_V>();
          Add(sUb, sUb, b1, halfD * d);
          PipeBarrier<PIPE_V>();

          if (c + 1U < numChunks) {
            Cast(bfOut, sUb, RoundMode::CAST_RINT, halfD * d);
            PipeBarrier<PIPE_ALL>();
            GlobalTensor<bfloat16_t> hGm;
            hGm.SetGlobalBuffer(hPtr + static_cast<uint64_t>(sRowOff) * d, halfD * d);
            DataCopy(hGm, bfOut, halfD * d);
            if (t->saveForBwd != 0U) {
              const auto cacheL = MegaKdaCache::ComputeLayout(t->seqLocal, H, d, C);
              GlobalTensor<bfloat16_t> hcCacheGm;
              hcCacheGm.SetGlobalBuffer(
                  reinterpret_cast<__gm__ bfloat16_t *>(inputList[IDX_BWD_CACHE] + cacheL.hChunksOff) +
                      ((hcIdx + 1ULL) * d + sRowOff) * d,
                  halfD * d);
              DataCopy(hcCacheGm, bfOut, halfD * d);
            }
            PipeBarrier<PIPE_ALL>();
            CrossCoreSetFlag<0x2, PIPE_MTE3>(0);
          } else {
            // Rank 0 final chunk: publish FP32 B_0 to comm_send_buf + sendScanOff!
            PipeBarrier<PIPE_ALL>();
            GlobalTensor<float> sendBGm;
            sendBGm.SetGlobalBuffer(
                reinterpret_cast<__gm__ float *>(inputList[IDX_SEND_BUF] + t->sendScanOff + groupOff) + headElemOff,
                halfD * d);
            DataCopy(sendBGm, sUb, halfD * d);
            PipeBarrier<PIPE_ALL>();
          }
        }
      }

      {
        const uint32_t lastC = numChunks - 1U;
        const uint64_t rAbs = static_cast<uint64_t>(lastC) * C + vRowOff;
        __gm__ bfloat16_t *oSrc = ((lastC & 1U) ? oPtr1 : oPtr0) + vRowOff * d;
        __gm__ bfloat16_t *oaDst = nullptr;
        __gm__ float *orDst = nullptr;
        if (t->saveForBwd != 0U) {
          const auto cacheL = MegaKdaCache::ComputeLayout(t->seqLocal, H, d, C);
          const uint64_t lastHc = static_cast<uint64_t>(h) * numChunks + lastC;
          oaDst = reinterpret_cast<__gm__ bfloat16_t *>(inputList[IDX_BWD_CACHE] + cacheL.oAttnOff) +
                  (lastHc * C + vRowOff) * d;
          orDst = reinterpret_cast<__gm__ float *>(inputList[IDX_BWD_CACHE] + cacheL.oRstdOff) +
                  static_cast<uint64_t>(h) * t->seqLocal + rAbs;
        }
        RunCp1GatedRmsNorm32(
            wNormFp, oBf, gBf, oFp, gFp, tmp, rmsVec, rmsBrcb,
            oSrc, gOutAll + rAbs * totalHeadDim + static_cast<uint64_t>(h) * d,
            outAll + rAbs * totalHeadDim + static_cast<uint64_t>(h) * d,
            d, totalHeadDim, eps, oaDst, orDst);
      }
    }

    pipe.Destroy();
  }
}

// ---------------------------------------------------------------------------
// Entry 1: CP=1 Dedicated Fast Path
// ---------------------------------------------------------------------------
__aicore__ inline void RunCp1FastPath(uint32_t blockIdx, GM_ADDR *inputList,
                                      const HyperMegaKdaLaunchTilingData *t) {
  if ASCEND_IS_AIV {
    RunCp1ShortConv(blockIdx, inputList, t, false);
    RunCp1FusedPrepareAiv(blockIdx, inputList, t);
    RunCp1FusedFwdHAiv(blockIdx, inputList, t, false, false);
  }
  if ASCEND_IS_AIC {
    RunCp1FusedPrepareAic(blockIdx, inputList, t);
    RunCp1FusedFwdHAic(blockIdx, inputList, t);
  }
}

// ---------------------------------------------------------------------------
// Entry 2: CP>1 State-Passing (ALLGATHER / PREFIX_SUM) Fused Fast Path!
// ---------------------------------------------------------------------------
__aicore__ inline void RunStatePassingCpFastPath(uint32_t blockIdx, GM_ADDR *inputList,
                                                 const HyperMegaKdaLaunchTilingData *t) {
  const uint32_t cp = t->cpSize;
  const uint32_t rank = t->cpRank;
  const uint32_t H = t->numHeads;
  const uint32_t numCube = t->numCubeCores;
  const uint32_t G = t->headGroups;
  const bool isPrefix = (t->cpMode == CP_MODE_PREFIX_SUM);

  if ASCEND_IS_AIV {
    const uint32_t wid = blockIdx;
    const uint32_t core = wid >> 1;
    GM_ADDR sb = inputList[IDX_SEND_BUF];
    GM_ADDR rb = inputList[IDX_RECV_BUF];
    GM_ADDR ev = inputList[IDX_EVENT];

    // Phase 0: Startup SHMEM Ready Dissemination Barrier
    int32_t gen = 0;
    if (wid == 0U) {
      gen = RunFastDisseminationBarrier(ev, t, false);
    }
    PipeBarrier<PIPE_ALL>();
    SyncAll();
    if (wid != 0U) {
      gen = ReadCurrentReadyGen(ev, t);
    }

    // Phase 0.5: 36 KB Causal Conv Halo P2P (rank -> rank + 1)
    RunCpFastHaloExchange(wid, gen, inputList, t);

    // Phase 1: 40-AIV Contiguous 4-Head Row-Tiled ShortConv1d + SiLU + L2Norm (with Halo)
    RunCp1ShortConv(wid, inputList, t, (rank > 0U));

    // Phase 2: 20-AIC <-> 40-AIV Double-Buffered L2-Ring-Buffer FusedPrepare
    RunCp1FusedPrepareAiv(wid, inputList, t);

    // Phase 3A (Pass 1): Rank 0 runs 1-pass FusedFwdH+FinalizeO+GatedRMSNorm & publishes B_0;
    //                    Middle ranks run 1:2 InterSummary (B_r, M_r); Last rank skips!
    if (rank == 0U) {
      RunCp1FusedFwdHAiv(wid, inputList, t, false, true);
    } else if (rank + 1U < cp) {
      for (uint32_t h = core; h < H; h += numCube) {
        TaskDesc td{};
        td.task_index = h;
        ExecuteInterSummaryAiv(td, inputList, t);
        if (t->saveForBwd != 0U) {
          const auto cacheL = MegaKdaCache::ComputeLayout(t->seqLocal, H, t->headDim, t->chunkSize);
          const uint32_t sub = wid & 1U;
          const uint32_t d = t->headDim;
          const uint32_t halfD = d >> 1;
          const uint32_t g = h / t->headsPerGroup;
          const uint32_t hInG = h % t->headsPerGroup;
          const uint64_t groupOff = g * 2ULL * t->sendScanGroupHalfBytes;
          const uint64_t headElemOff = static_cast<uint64_t>(hInG) * d * d + sub * halfD * d;
          TPipe aPipe;
          TBuf<TPosition::VECCALC> aBuf;
          aPipe.InitBuffer(aBuf, halfD * d * 4U + 64U);
          LocalTensor<float> aTmp = aBuf.GetWithOffset<float>(halfD * d, 0);
          GlobalTensor<float> srcA, dstA;
          srcA.SetGlobalBuffer(
              reinterpret_cast<__gm__ float *>(inputList[IDX_WORKSPACE] + t->wsScanSelfOff + groupOff +
                                               t->sendScanGroupHalfBytes) + headElemOff,
              halfD * d);
          dstA.SetGlobalBuffer(
              reinterpret_cast<__gm__ float *>(inputList[IDX_BWD_CACHE] + cacheL.aRankOff) +
                  (static_cast<uint64_t>(h) * d + sub * halfD) * d,
              halfD * d);
          DataCopy(aTmp, srcA, halfD * d);
          PipeBarrier<PIPE_ALL>();
          DataCopy(dstA, aTmp, halfD * d);
          PipeBarrier<PIPE_ALL>();
          aPipe.Destroy();
        }
      }
    }
    PipeBarrier<PIPE_ALL>();
    SyncAll();

    // Phase 3B: SHMEM State Exchange & 1:2 Cube-Accelerated Merge
    if (!isPrefix) {
      // ALLGATHER (LASP-2 1-round): parallel AIVs send to each dst > rank!
      if (rank + 1U < cp) {
        const uint32_t dst = rank + 1U + wid;
        if (dst < cp) {
          const bool hasM = (rank > 0U);
          const int64_t nBytes = static_cast<int64_t>(
              hasM ? (2ULL * t->sendScanGroupHalfBytes) : t->sendScanGroupHalfBytes);
          const int64_t sigOff = static_cast<int64_t>((1U + rank) * 16U);
          for (uint32_t g = 0; g < G; ++g) {
            const uint64_t groupOff = g * 2ULL * t->sendScanGroupHalfBytes;
            const int64_t srcOff = static_cast<int64_t>(t->sendScanOff + groupOff);
            const int64_t dstOff =
                static_cast<int64_t>(t->recvScanOff + rank * t->sendScanSlotBytes + groupOff);
            RunFastShmemPutSet(rb, sb, ev, sigOff, gen, dstOff, srcOff, nBytes, static_cast<int64_t>(dst),
                               (g + 1U == G));
          }
        }
      }
      if (rank >= 1U && wid == 0U) {
        for (uint32_t src = 0; src < rank; ++src) {
          aclshmem_signal_wait_until(reinterpret_cast<__gm__ int32_t *>(ev) + (1U + src) * 16U,
                                     ACLSHMEM_CMP_GE, gen);
        }
      }
      PipeBarrier<PIPE_ALL>();
      SyncAll();

      if (rank >= 1U) {
        for (uint32_t h = core; h < H; h += numCube) {
          TaskDesc td{};
          td.task_index = h;
          td.extra_value_0 = 0U;  // ALLGATHER_REDUCE
          td.extra_value_1 = 0U;
          ExecuteScanMergeAiv(td, inputList, t);
        }
        PipeBarrier<PIPE_ALL>();
        SyncAll();
      }
    } else {
      // PREFIX_SUM (Exclusive Hillis-Steele Tree-Scan): K = ceil(log2(CP)) rounds!
      for (uint32_t k = 0; k < t->scanRounds; ++k) {
        const uint32_t stride = 1U << k;
        if (rank + stride < cp && wid == 0U) {
          const uint32_t dst = rank + stride;
          uint32_t srcPingPong = 0U;
          bool hasM = false;
          if (rank == 0U) {
            srcPingPong = 0U;
            hasM = false;
          } else if (k == 0U) {
            srcPingPong = 0U;
            hasM = (rank >= stride);
          } else if (rank < (1U << (k - 1U))) {
            uint32_t lastK = 0U;
            while ((1U << (lastK + 1U)) <= rank) {
              ++lastK;
            }
            srcPingPong = (lastK + 1U) & 1U;
            hasM = false;
          } else {
            srcPingPong = k & 1U;
            hasM = (rank >= stride);
          }
          const int64_t nBytes = static_cast<int64_t>(
              hasM ? (2ULL * t->sendScanGroupHalfBytes) : t->sendScanGroupHalfBytes);
          const int64_t sigOff = static_cast<int64_t>((1U + k) * 16U);
          for (uint32_t g = 0; g < G; ++g) {
            const uint64_t groupOff = g * 2ULL * t->sendScanGroupHalfBytes;
            const int64_t srcOff =
                static_cast<int64_t>(t->sendScanOff + srcPingPong * t->sendScanSlotBytes + groupOff);
            const int64_t dstOff =
                static_cast<int64_t>(t->recvScanOff + k * t->sendScanSlotBytes + groupOff);
            RunFastShmemPutSet(rb, sb, ev, sigOff, gen, dstOff, srcOff, nBytes, static_cast<int64_t>(dst),
                               (g + 1U == G));
          }
        }
        if (rank >= stride && wid == 0U) {
          aclshmem_signal_wait_until(reinterpret_cast<__gm__ int32_t *>(ev) + (1U + k) * 16U,
                                     ACLSHMEM_CMP_GE, gen);
        }
        PipeBarrier<PIPE_ALL>();
        SyncAll();

        if (rank >= stride) {
          for (uint32_t h = core; h < H; h += numCube) {
            TaskDesc td{};
            td.task_index = h;
            td.extra_value_0 = 1U;  // PREFIX_ROUND
            td.extra_value_1 = k;
            ExecuteScanMergeAiv(td, inputList, t);
          }
          PipeBarrier<PIPE_ALL>();
          SyncAll();
        }
      }
    }

    // Phase 3C (Pass 2, rank >= 1): Fused FwdH + FinalizeO + Overlapped GatedRMSNorm -> IDX_OUT!
    if (rank >= 1U) {
      RunCp1FusedFwdHAiv(wid, inputList, t, true, false);
    }

    // Phase 4: Completion SHMEM Dissemination Barrier
    PipeBarrier<PIPE_ALL>();
    SyncAll();
    if (wid == 0U) {
      RunFastDisseminationBarrier(ev, t, true);
    }
  }

  if ASCEND_IS_AIC {
    const uint32_t core = blockIdx;

    // Phase 2: FusedPrepare AIC
    RunCp1FusedPrepareAic(core, inputList, t);

    // Phase 3A (Pass 1): Rank 0 FusedFwdH; Middle ranks InterSummary
    if (rank == 0U) {
      RunCp1FusedFwdHAic(core, inputList, t);
    } else if (rank + 1U < cp) {
      for (uint32_t h = core; h < H; h += numCube) {
        TaskDesc td{};
        td.task_index = h;
        ExecuteInterSummaryAic(td, inputList, t);
      }
    }

    // Phase 3B: ScanMerge AIC
    if (!isPrefix) {
      if (rank >= 1U) {
        for (uint32_t h = core; h < H; h += numCube) {
          TaskDesc td{};
          td.task_index = h;
          td.extra_value_0 = 0U;
          td.extra_value_1 = 0U;
          ExecuteScanMergeAic(td, inputList, t);
        }
      }
    } else {
      for (uint32_t k = 0; k < t->scanRounds; ++k) {
        const uint32_t stride = 1U << k;
        if (rank >= stride) {
          for (uint32_t h = core; h < H; h += numCube) {
            TaskDesc td{};
            td.task_index = h;
            td.extra_value_0 = 1U;
            td.extra_value_1 = k;
            ExecuteScanMergeAic(td, inputList, t);
          }
        }
      }
    }

    // Phase 3C (Pass 2, rank >= 1): FusedFwdH AIC
    if (rank >= 1U) {
      RunCp1FusedFwdHAic(core, inputList, t);
    }
  }
}

}  // namespace MulticoreKda

#endif  // HYPER_MEGA_KDA_CP1_FAST_H
