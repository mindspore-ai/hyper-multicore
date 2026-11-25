/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0.
 *
 * MegaKDA-Grad: Unified 4-Phase L2-Resident Fused Backward Engine
 * Supports CP=1 and CP>1 (allgather & prefix_sum) with dynamic core-count scaling
 * (Ascend 910B 20-AIC/40-AIV, Ascend 910C 24-AIC/48-AIV).
 *
 *   Phase B0: (CP>1) SHMEM Ready Gen-Barrier + 2N-AIV GatedRMSNorm_bwd
 *   Phase B1: N-AIC <-> 2N-AIV 1:2 FusedBwdH + (CP>1) Reverse Suffix-Scan (AllGather / Tree-Scan)
 *   Phase B2: N-AIC <-> 2N-AIV Double-Buffered L2-Ring FusedBwdIntraWyGate + Q/K L2Norm_bwd
 *   Phase B3: 2N-AIV SiLU_bwd + (CP>1) 36KB Reverse Conv Halo + ShortConv1d_bwd + Weight Reduction + Gen-Barrier
 */
#ifndef HYPER_MEGA_KDA_BWD_FAST_H
#define HYPER_MEGA_KDA_BWD_FAST_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "kda_cube_raw.h"
#include "common/mega_kda/bwd_cache_layout.h"
#include "shmem.h"

namespace MulticoreKdaBwd {

using namespace AscendC;

// IO array indices (matching hyper_mega_kda_grad.cpp)
constexpr uint32_t B_DO_NORMED   = 0;
constexpr uint32_t B_Q_IN        = 1;
constexpr uint32_t B_K_IN        = 2;
constexpr uint32_t B_V_IN        = 3;
constexpr uint32_t B_G_IN        = 4;
constexpr uint32_t B_BETA_IN     = 5;
constexpr uint32_t B_G_OUT       = 6;
constexpr uint32_t B_WCONV_Q     = 7;
constexpr uint32_t B_WCONV_K     = 8;
constexpr uint32_t B_WCONV_V     = 9;
constexpr uint32_t B_A_LOG       = 10;
constexpr uint32_t B_DT_BIAS     = 11;
constexpr uint32_t B_O_NORM_W    = 12;
constexpr uint32_t B_BWD_CACHE   = 13;
constexpr uint32_t B_SEND_BUF    = 14;
constexpr uint32_t B_RECV_BUF    = 15;
constexpr uint32_t B_EVENT       = 16;
constexpr uint32_t B_DQ_IN       = 17;
constexpr uint32_t B_DK_IN       = 18;
constexpr uint32_t B_DV_IN       = 19;
constexpr uint32_t B_DG_IN       = 20;
constexpr uint32_t B_DBETA_IN    = 21;
constexpr uint32_t B_DG_OUT      = 22;
constexpr uint32_t B_DW_CONV     = 23;
constexpr uint32_t B_DA_LOG      = 24;
constexpr uint32_t B_DDT_BIAS    = 25;
constexpr uint32_t B_DO_NORM_W   = 26;
constexpr uint32_t B_WORKSPACE   = 27;
constexpr uint32_t B_TILING      = 28;

// Phase B1 per-core L2 scratch (256 KB/core)
constexpr uint64_t BWDH_OFF_DH       = 0ULL;               // 128*128*2 = 32 KB (BF16)
constexpr uint64_t BWDH_OFF_DHDELTA  = 32ULL * 1024ULL;    // 128*128*4 = 64 KB (FP32)
constexpr uint64_t BWDH_OFF_DVN      = 96ULL * 1024ULL;    // 64*128*2  = 16 KB (BF16)
constexpr uint64_t BWDH_OFF_NEGDVN   = 112ULL * 1024ULL;   // 64*128*2  = 16 KB (BF16)
constexpr uint64_t BWDH_CORE_BYTES   = 256ULL * 1024ULL;

// Phase B2 double-buffered L2 ring (256 KB/lane, 2 lanes, 2 slots = 1024 KB/core)
// Holds BOTH 3 off-diag strips AND 4 diag-16 blocks simultaneously -> zero extra CrossCore syncs!
constexpr uint64_t BWDR_OFF_DKBG     = 0ULL;               // 64*128*2   = 16 KB (BF16)
constexpr uint64_t BWDR_OFF_DVBETA   = 16ULL * 1024ULL;    // 64*128*2   = 16 KB (BF16)
constexpr uint64_t BWDR_OFF_DAKKRAW  = 32ULL * 1024ULL;    // 64*64*4    = 16 KB (FP32)
constexpr uint64_t BWDR_OFF_KREF     = 48ULL * 1024ULL;    // (96+64)*128*2  = 40 KB (BF16: off-diag 24KB + diag 16KB)
constexpr uint64_t BWDR_OFF_QKBREF   = 88ULL * 1024ULL;    // (96+128)*128*2 = 56 KB (BF16: off-diag 24KB + diag 32KB)
constexpr uint64_t BWDR_OFF_DASTRIP  = 144ULL * 1024ULL;   // (3072+2048)*2  = 10 KB (BF16, padded to 12 KB)
constexpr uint64_t BWDR_OFF_TQBK     = 156ULL * 1024ULL;   // (96+128)*128*2 = 56 KB (BF16: off-diag 24KB + diag 32KB)
constexpr uint64_t BWDR_OFF_TKNEG    = 212ULL * 1024ULL;   // (96+64)*128*2  = 40 KB (BF16: off-diag 24KB + diag 16KB)
constexpr uint64_t BWDR_LANE_BYTES   = 256ULL * 1024ULL;
constexpr uint64_t BWDR_SLOT_BYTES   = 2ULL * BWDR_LANE_BYTES;
constexpr uint64_t BWDR_CORE_BYTES   = 2ULL * BWDR_SLOT_BYTES;

__aicore__ inline float BitsToFloatBwd(uint32_t u) {
  union { uint32_t i; float f; } v;
  v.i = u;
  return v.f;
}

// ---------------------------------------------------------------------------
// SHMEM helpers (generation-separated dissemination barrier + put_signal)
// ---------------------------------------------------------------------------
__aicore__ inline int32_t BwdShmemGenBarrier(uint32_t wid, GM_ADDR evBuf, uint64_t eventCounterBytes,
                                             uint32_t cpSize, uint32_t cpRank, bool isCompletion) {
  int32_t gen = 1;
#if !defined(__DAV_CUBE__) && !defined(__DAV_C220_CUBE__)
  if (wid == 0U && cpSize > 1U) {
    constexpr uint64_t kStride = 16ULL;  // 64B cache line
    const uint64_t totalInts = eventCounterBytes / 4ULL;
    const uint64_t base = totalInts - 2ULL * static_cast<uint64_t>(cpSize + 1U) * kStride;
    __gm__ int32_t *ready = reinterpret_cast<__gm__ int32_t *>(evBuf) + base;
    if (isCompletion) {
      ready += static_cast<uint64_t>(cpSize + 1U) * kStride;
    }
    GlobalTensor<int32_t> st;
    st.SetGlobalBuffer(ready, static_cast<uint32_t>((cpSize + 1U) * kStride));
    const uint32_t genIdx = cpSize * kStride;
    TPipe pipe;
    TBuf<TPosition::VECCALC> buf;
    pipe.InitBuffer(buf, 64);
    LocalTensor<int32_t> lv = buf.Get<int32_t>();
    DataCopy(lv, st[genIdx], 8);
    PipeBarrier<PIPE_ALL>();
    gen = lv.GetValue(0) + 1;
    lv.SetValue(0, gen);
    PipeBarrier<PIPE_ALL>();
    DataCopy(st[genIdx], lv, 8);
    PipeBarrier<PIPE_ALL>();
    pipe.Destroy();

    uint32_t round = 0U;
    for (uint32_t dist = 1U; dist < cpSize; dist <<= 1, ++round) {
      const int target = static_cast<int>((cpRank + dist) % cpSize);
      __gm__ int32_t *rSig = ready + static_cast<uint64_t>(round) * kStride;
      aclshmemx_signal_op(rSig, gen, ACLSHMEM_SIGNAL_SET, target);
      aclshmem_signal_wait_until(rSig, ACLSHMEM_CMP_GE, gen);
    }
  }
#else
  (void)wid; (void)evBuf; (void)eventCounterBytes; (void)cpSize; (void)cpRank; (void)isCompletion;
#endif
  return gen;
}

__aicore__ inline int32_t BwdReadReadyGen(uint32_t wid, GM_ADDR evBuf, uint64_t eventCounterBytes, uint32_t cpSize) {
  int32_t gen = 1;
#if !defined(__DAV_CUBE__) && !defined(__DAV_C220_CUBE__)
  if (cpSize > 1U) {
    constexpr uint64_t kStride = 16ULL;
    const uint64_t totalInts = eventCounterBytes / 4ULL;
    const uint64_t base = totalInts - 2ULL * static_cast<uint64_t>(cpSize + 1U) * kStride;
    GlobalTensor<int32_t> st;
    st.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(evBuf) + base,
                       static_cast<uint32_t>((cpSize + 1U) * kStride));
    const uint32_t genIdx = cpSize * kStride;
    TPipe pipe;
    TBuf<TPosition::VECCALC> buf;
    pipe.InitBuffer(buf, 64);
    LocalTensor<int32_t> lv = buf.Get<int32_t>();
    DataCopy(lv, st[genIdx], 8);
    PipeBarrier<PIPE_ALL>();
    gen = lv.GetValue(0);
    pipe.Destroy();
  }
#endif
  (void)wid;
  return gen;
}

__aicore__ inline void BwdShmemPutSet(GM_ADDR recvBuf, GM_ADDR sendBuf, GM_ADDR evBuf,
                                      int64_t sigIdx, int32_t sigVal,
                                      int64_t dstByteOff, int64_t srcByteOff,
                                      int64_t numBytes, int64_t dstRank, bool doSignal) {
#if !defined(__DAV_CUBE__) && !defined(__DAV_C220_CUBE__)
  aclshmem_putmem_nbi(reinterpret_cast<__gm__ uint8_t *>(recvBuf) + dstByteOff,
                      reinterpret_cast<__gm__ uint8_t *>(sendBuf) + srcByteOff,
                      static_cast<size_t>(numBytes), static_cast<int>(dstRank));
  aclshmem_quiet();
  if (doSignal) {
    aclshmemx_signal_op(reinterpret_cast<__gm__ int32_t *>(evBuf) + sigIdx,
                        sigVal, ACLSHMEM_SIGNAL_SET, static_cast<int>(dstRank));
  }
#else
  (void)recvBuf; (void)sendBuf; (void)evBuf; (void)sigIdx; (void)sigVal;
  (void)dstByteOff; (void)srcByteOff; (void)numBytes; (void)dstRank; (void)doSignal;
#endif
}

// ---------------------------------------------------------------------------
// Phase B0 (AIV): 2N-AIV GatedRMSNorm_bwd -> d_g_out, wsDO (head-chunk), partial d_o_norm_w
// ---------------------------------------------------------------------------
__aicore__ inline void RunPhaseB0RmsNormBwd(uint32_t wid, GM_ADDR *io, const HyperMegaKdaGradTilingData *t) {
  if ASCEND_IS_AIV {
    const uint32_t numAiv = 2U * t->numCubeCores;
    const uint32_t S = t->seqLocal;
    const uint32_t H = t->numHeads;
    const uint32_t d = t->headDim;
    const uint32_t C = t->chunkSize;
    const uint32_t numChunks = t->numChunks;
    const uint32_t D = H * d;
    constexpr uint32_t R = 32U;
    const float invD = 1.0f / static_cast<float>(static_cast<int32_t>(d));
    const auto cacheL = MegaKdaCache::ComputeLayout(S, H, d, C);

    __gm__ bfloat16_t *doNormAll = reinterpret_cast<__gm__ bfloat16_t *>(io[B_DO_NORMED]);
    __gm__ bfloat16_t *gOutAll   = reinterpret_cast<__gm__ bfloat16_t *>(io[B_G_OUT]);
    __gm__ bfloat16_t *dgOutAll  = reinterpret_cast<__gm__ bfloat16_t *>(io[B_DG_OUT]);
    __gm__ bfloat16_t *oAttnAll  = reinterpret_cast<__gm__ bfloat16_t *>(io[B_BWD_CACHE] + cacheL.oAttnOff);
    __gm__ float *oRstdAll       = reinterpret_cast<__gm__ float *>(io[B_BWD_CACHE] + cacheL.oRstdOff);
    __gm__ bfloat16_t *dOAll     = reinterpret_cast<__gm__ bfloat16_t *>(io[B_WORKSPACE] + t->wsDOOff);

    TPipe pipe;
    TBuf<TPosition::VECCALC> buf;
    const uint32_t uWNorm = 0;
    const uint32_t uDwAcc = uWNorm + d * 4U;
    const uint32_t uRstd  = uDwAcc + d * 4U;
    const uint32_t uCVec  = uRstd + 128U;
    const uint32_t uBrcb  = uCVec + 128U;
    const uint32_t uDoBf  = 2560U;
    const uint32_t uGoBf  = uDoBf + R * d * 2U;
    const uint32_t uOaBf  = uGoBf + R * d * 2U;
    const uint32_t uDoFp  = uOaBf + R * d * 2U;
    const uint32_t uGoFp  = uDoFp + R * d * 4U;
    const uint32_t uOaFp  = uGoFp + R * d * 4U;
    const uint32_t uTmp   = uOaFp + R * d * 4U;
    pipe.InitBuffer(buf, uTmp + R * d * 4U + 64U);

    LocalTensor<float> wNormFp   = buf.GetWithOffset<float>(d, uWNorm);
    LocalTensor<float> dwAcc     = buf.GetWithOffset<float>(d, uDwAcc);
    LocalTensor<float> rstdVec   = buf.GetWithOffset<float>(R, uRstd);
    LocalTensor<float> cVec      = buf.GetWithOffset<float>(R, uCVec);
    LocalTensor<float> brcb      = buf.GetWithOffset<float>(R * 8U, uBrcb);
    LocalTensor<bfloat16_t> doBf = buf.GetWithOffset<bfloat16_t>(R * d, uDoBf);
    LocalTensor<bfloat16_t> goBf = buf.GetWithOffset<bfloat16_t>(R * d, uGoBf);
    LocalTensor<bfloat16_t> oaBf = buf.GetWithOffset<bfloat16_t>(R * d, uOaBf);
    LocalTensor<float> doFp      = buf.GetWithOffset<float>(R * d, uDoFp);
    LocalTensor<float> goFp      = buf.GetWithOffset<float>(R * d, uGoFp);
    LocalTensor<float> oaFp      = buf.GetWithOffset<float>(R * d, uOaFp);
    LocalTensor<float> tmp       = buf.GetWithOffset<float>(R * d, uTmp);

    {
      GlobalTensor<bfloat16_t> wGm;
      wGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(io[B_O_NORM_W]), d);
      DataCopy(doBf, wGm, d);
      Duplicate(dwAcc, 0.0f, d);
      PipeBarrier<PIPE_ALL>();
      Cast(wNormFp, doBf, RoundMode::CAST_NONE, d);
      PipeBarrier<PIPE_V>();
    }

    const uint16_t blockLen = static_cast<uint16_t>((d * 2U) / 32U);
    const uint16_t stride = static_cast<uint16_t>(((D - d) * 2U) / 32U);
    const uint32_t totalBlks = H * numChunks * 2U;

    for (uint32_t blk = wid; blk < totalBlks; blk += numAiv) {
      const uint32_t half = blk & 1U;
      const uint32_t hcIdx = blk >> 1;
      const uint32_t h = hcIdx / numChunks;
      const uint32_t c = hcIdx % numChunks;
      const uint64_t rAbs = static_cast<uint64_t>(c) * C + half * R;
      const uint64_t rowOff = rAbs * D + static_cast<uint64_t>(h) * d;
      const uint64_t hcOff = (static_cast<uint64_t>(hcIdx) * C + half * R) * d;

      GlobalTensor<bfloat16_t> doGm, goGm, oaGm, dgoGm, doOutGm;
      GlobalTensor<float> rsGm;
      doGm.SetGlobalBuffer(doNormAll + rowOff, static_cast<uint64_t>(R) * D);
      goGm.SetGlobalBuffer(gOutAll + rowOff, static_cast<uint64_t>(R) * D);
      dgoGm.SetGlobalBuffer(dgOutAll + rowOff, static_cast<uint64_t>(R) * D);
      oaGm.SetGlobalBuffer(oAttnAll + hcOff, R * d);
      doOutGm.SetGlobalBuffer(dOAll + hcOff, R * d);
      rsGm.SetGlobalBuffer(oRstdAll + static_cast<uint64_t>(h) * S + rAbs, R);

      DataCopyParams strP{R, blockLen, stride, 0};
      DataCopy(doBf, doGm, strP);
      DataCopy(goBf, goGm, strP);
      DataCopy(oaBf, oaGm, R * d);
      DataCopy(rstdVec, rsGm, R);
      PipeBarrier<PIPE_ALL>();

      Cast(doFp, doBf, RoundMode::CAST_NONE, R * d);
      Cast(goFp, goBf, RoundMode::CAST_NONE, R * d);
      Cast(oaFp, oaBf, RoundMode::CAST_NONE, R * d);
      PipeBarrier<PIPE_V>();

      // s = sigmoid(g_out) -> goFp
      Muls(goFp, goFp, -1.0f, R * d);
      PipeBarrier<PIPE_V>();
      Mins(goFp, goFp, 80.0f, R * d);
      PipeBarrier<PIPE_V>();
      Maxs(goFp, goFp, -80.0f, R * d);
      PipeBarrier<PIPE_V>();
      Exp(goFp, goFp, R * d);
      PipeBarrier<PIPE_V>();
      Adds(goFp, goFp, 1.0f, R * d);
      PipeBarrier<PIPE_V>();
      Duplicate(tmp, 1.0f, R * d);
      PipeBarrier<PIPE_V>();
      Div(goFp, tmp, goFp, R * d);
      PipeBarrier<PIPE_V>();

      // do_hat = d_o_normed * s -> doFp; tmp = do_hat * (1 - s)
      Mul(doFp, doFp, goFp, R * d);
      Muls(tmp, goFp, -1.0f, R * d);
      PipeBarrier<PIPE_V>();
      Adds(tmp, tmp, 1.0f, R * d);
      PipeBarrier<PIPE_V>();
      Mul(tmp, doFp, tmp, R * d);
      PipeBarrier<PIPE_V>();

      // goFp = o_norm = o_attn * o_rstd
      Brcb(brcb, rstdVec, static_cast<uint8_t>(R / 8U), {1, 8});
      PipeBarrier<PIPE_V>();
      BinaryRepeatParams scaleP{1, 1, 0, 16, 16, 1};
      Mul(goFp, oaFp, brcb, 64, R, scaleP);
      Mul(goFp[64], oaFp[64], brcb, 64, R, scaleP);
      PipeBarrier<PIPE_V>();

      // dg_out = tmp * o_norm * w_norm
      BinaryRepeatParams wMul{1, 1, 1, 16, 16, 0};
      Mul(tmp, tmp, goFp, R * d);
      PipeBarrier<PIPE_V>();
      Mul(tmp, tmp, wNormFp, 64, R, wMul);
      Mul(tmp[64], tmp[64], wNormFp[64], 64, R, wMul);
      PipeBarrier<PIPE_V>();
      Cast(goBf, tmp, RoundMode::CAST_RINT, R * d);
      PipeBarrier<PIPE_ALL>();
      DataCopyParams outP{R, blockLen, 0, stride};
      DataCopy(dgoGm, goBf, outP);
      PipeBarrier<PIPE_ALL>();

      // dw_norm += sum_r(do_hat * o_norm)
      Mul(tmp, doFp, goFp, R * d);
      PipeBarrier<PIPE_V>();
      for (uint32_t st = 16U; st >= 1U; st >>= 1) {
        Add(tmp, tmp, tmp[st * d], st * d);
        PipeBarrier<PIPE_V>();
      }
      Add(dwAcc, dwAcc, tmp, d);
      PipeBarrier<PIPE_V>();

      // dw_do = do_hat * w_norm -> doFp
      Mul(doFp, doFp, wNormFp, 64, R, wMul);
      Mul(doFp[64], doFp[64], wNormFp[64], 64, R, wMul);
      PipeBarrier<PIPE_V>();

      // c_rms = mean_d(dw_do * o_attn)
      Mul(tmp, doFp, oaFp, R * d);
      PipeBarrier<PIPE_V>();
      BinaryRepeatParams foldP{1, 1, 1, 16, 16, 16};
      Add(tmp, tmp, tmp[64], 64, R, foldP);
      PipeBarrier<PIPE_V>();
      WholeReduceSum(cVec, tmp, 64, static_cast<int32_t>(R), 1, 1, 16);
      PipeBarrier<PIPE_V>();
      Mul(cVec, cVec, rstdVec, R);
      PipeBarrier<PIPE_V>();
      Mul(cVec, cVec, rstdVec, R);
      PipeBarrier<PIPE_V>();
      Muls(cVec, cVec, invD, R);
      PipeBarrier<PIPE_V>();
      Brcb(brcb, cVec, static_cast<uint8_t>(R / 8U), {1, 8});
      PipeBarrier<PIPE_V>();

      Mul(tmp, oaFp, brcb, 64, R, scaleP);
      Mul(tmp[64], oaFp[64], brcb, 64, R, scaleP);
      PipeBarrier<PIPE_V>();
      Sub(doFp, doFp, tmp, R * d);
      PipeBarrier<PIPE_V>();

      Brcb(brcb, rstdVec, static_cast<uint8_t>(R / 8U), {1, 8});
      PipeBarrier<PIPE_V>();
      Mul(doFp, doFp, brcb, 64, R, scaleP);
      Mul(doFp[64], doFp[64], brcb, 64, R, scaleP);
      PipeBarrier<PIPE_V>();
      Cast(doBf, doFp, RoundMode::CAST_RINT, R * d);
      PipeBarrier<PIPE_ALL>();
      DataCopy(doOutGm, doBf, R * d);
      PipeBarrier<PIPE_ALL>();
    }

    GlobalTensor<float> pnGm;
    pnGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(io[B_WORKSPACE] + t->wsPartialNormOff) + wid * d, d);
    PipeBarrier<PIPE_ALL>();
    DataCopy(pnGm, dwAcc, d);
    PipeBarrier<PIPE_ALL>();
    pipe.Destroy();
  }
}

// ---------------------------------------------------------------------------
// Phase B1 (AIC + AIV 1:2): FusedBwdH (supports summaryOnly for CP>2 middle ranks)
// ---------------------------------------------------------------------------
__aicore__ inline void RunPhaseB1BwdHAic(uint32_t core, GM_ADDR *io, const HyperMegaKdaGradTilingData *t,
                                         bool summaryOnly) {
  if ASCEND_IS_AIC {
    const uint32_t numCube = t->numCubeCores;
    const uint32_t S = t->seqLocal;
    const uint32_t H = t->numHeads;
    const uint32_t d = t->headDim;
    const uint32_t C = t->chunkSize;
    const uint32_t numChunks = t->numChunks;
    const auto cacheL = MegaKdaCache::ComputeLayout(S, H, d, C);

    GM_ADDR ws = io[B_WORKSPACE];
    GM_ADDR bc = io[B_BWD_CACHE];
    GM_ADDR scBase = ws + t->wsBwdHScratchOff + static_cast<uint64_t>(core) * BWDH_CORE_BYTES;

    MulticoreKda::KdaRawCubePipe cube;
    cube.Init(176 * 1024, 32 * 1024, 32 * 1024, 64 * 1024);
    auto l1DO     = cube.L1At(0);
    auto l1Aqk    = cube.L1At(16 * 1024);
    auto l1Kg     = cube.L1At(24 * 1024);
    auto l1DH     = cube.L1At(40 * 1024);
    auto l1Vnew   = cube.L1At(72 * 1024);
    auto l1Hc     = cube.L1At(88 * 1024);
    auto l1Qg     = cube.L1At(120 * 1024);
    auto l1W      = cube.L1At(136 * 1024);
    auto l1NegDvn = cube.L1At(152 * 1024);
    auto A = cube.L0A();
    auto B = cube.L0B();
    auto C0 = cube.L0C();

    GlobalTensor<bfloat16_t> dhGm, dvnScGm, negDvnGm;
    GlobalTensor<float> dhDeltaGm;
    dhGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(scBase + BWDH_OFF_DH), d * d);
    dvnScGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(scBase + BWDH_OFF_DVN), C * d);
    negDvnGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(scBase + BWDH_OFF_NEGDVN), C * d);
    dhDeltaGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(scBase + BWDH_OFF_DHDELTA), d * d);

    for (uint32_t h = core; h < H; h += numCube) {
      for (int32_t ci = static_cast<int32_t>(numChunks) - 1; ci >= 0; --ci) {
        const uint32_t c = static_cast<uint32_t>(ci);
        const uint64_t hcIdx = static_cast<uint64_t>(h) * numChunks + c;

        CrossCoreWaitFlag<0x2, PIPE_MTE2>(0);

        GlobalTensor<bfloat16_t> doGm, aqkGm, kgGm, vnGm, hcGm, qgGm, wGm;
        doGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + t->wsDOOff) + hcIdx * C * d, C * d);
        aqkGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(bc + cacheL.aqkOff) + hcIdx * C * C, C * C);
        kgGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(bc + cacheL.kgOff) + hcIdx * C * d, C * d);
        qgGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(bc + cacheL.qgOff) + hcIdx * C * d, C * d);
        wGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(bc + cacheL.wOff) + hcIdx * C * d, C * d);

        cube.CopyGmToL1Nz(l1DO, doGm, 64, d, d, 64);
        cube.CopyGmToL1Nz(l1Aqk, aqkGm, 64, 64, 64, 64);
        cube.CopyGmToL1Nz(l1Kg, kgGm, 64, d, d, 64);
        cube.CopyGmToL1Nz(l1DH, dhGm, d, d, d, d);
        cube.CopyGmToL1Nz(l1Qg, qgGm, 64, d, d, 64);
        cube.CopyGmToL1Nz(l1W, wGm, 64, d, d, 64);
        if (!summaryOnly) {
          vnGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(bc + cacheL.vNewOff) + hcIdx * C * d, C * d);
          hcGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(bc + cacheL.hChunksOff) + hcIdx * d * d, d * d);
          cube.CopyGmToL1Nz(l1Vnew, vnGm, 64, d, d, 64);
          cube.CopyGmToL1Nz(l1Hc, hcGm, d, d, d, d);
        }
        cube.WaitMte2ToMte1();

        // 1+2. dv_new = Aqk^T @ dO + kg @ dH
        cube.LoadL0A_Trans(A, l1Aqk, 64, 64);
        cube.LoadL0B_NoT(B, l1DO, 64, d, 64);
        cube.WaitMte1ToM();
        cube.RunMmad(C0, A, B, 64, d, 64, true);
        cube.WaitMToMte1();

        cube.LoadL0A_NoT(A, l1Kg, 64, d, 64);
        cube.LoadL0B_NoT(B, l1DH, d, d, d);
        cube.WaitMte1ToM();
        cube.RunMmad(C0, A, B, 64, d, d, false);
        cube.WaitMToMte1();
        cube.WaitMToFix();
        cube.StoreBf16Gm(dvnScGm, C0, 64, d, d);
        if (!summaryOnly) {
          GlobalTensor<bfloat16_t> dvnOutGm;
          dvnOutGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + t->wsDvNewOff) + hcIdx * C * d, C * d);
          cube.StoreBf16Gm(dvnOutGm, C0, 64, d, d);
        }
        cube.WaitFixToM();
        CrossCoreSetFlag<0x2, PIPE_FIX>(1);

        if (!summaryOnly) {
          GlobalTensor<bfloat16_t> dAqkGm, dqgGm, dkgGm;
          dAqkGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + t->wsDAqkOff) + hcIdx * C * C, C * C);
          dqgGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + t->wsDqgOff) + hcIdx * C * d, C * d);
          dkgGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + t->wsDkgOff) + hcIdx * C * d, C * d);

          // 3. dAqk = dO @ V_new^T
          cube.LoadL0A_NoT(A, l1DO, 64, d, 64);
          cube.LoadL0B_Trans(B, l1Vnew, d, 64);
          cube.WaitMte1ToM();
          cube.RunMmad(C0, A, B, 64, 64, d, true);
          cube.WaitMToMte1();
          cube.WaitMToFix();
          cube.StoreBf16Gm(dAqkGm, C0, 64, 64, 64);
          cube.WaitFixToM();

          // 4. dqg = dO @ H_c^T
          cube.LoadL0A_NoT(A, l1DO, 64, d, 64);
          cube.LoadL0B_Trans(B, l1Hc, d, d);
          cube.WaitMte1ToM();
          cube.RunMmad(C0, A, B, 64, d, d, true);
          cube.WaitMToMte1();
          cube.WaitMToFix();
          cube.StoreBf16Gm(dqgGm, C0, 64, d, d);
          cube.WaitFixToM();

          // 5. dkg = V_new @ dH^T
          cube.LoadL0A_NoT(A, l1Vnew, 64, d, 64);
          cube.LoadL0B_Trans(B, l1DH, d, d);
          cube.WaitMte1ToM();
          cube.RunMmad(C0, A, B, 64, d, d, true);
          cube.WaitMToMte1();
          cube.WaitMToFix();
          cube.StoreBf16Gm(dkgGm, C0, 64, d, d);
          cube.WaitFixToM();
        }

        CrossCoreWaitFlag<0x2, PIPE_MTE2>(2);
        cube.CopyGmToL1Nz(l1NegDvn, negDvnGm, 64, d, d, 64);
        cube.WaitMte2ToMte1();

        if (!summaryOnly) {
          // 6. dw = (-dv_new) @ H_c^T
          GlobalTensor<bfloat16_t> dwGm;
          dwGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + t->wsDwOff) + hcIdx * C * d, C * d);
          cube.LoadL0A_NoT(A, l1NegDvn, 64, d, 64);
          cube.LoadL0B_Trans(B, l1Hc, d, d);
          cube.WaitMte1ToM();
          cube.RunMmad(C0, A, B, 64, d, d, true);
          cube.WaitMToMte1();
          cube.WaitMToFix();
          cube.StoreBf16Gm(dwGm, C0, 64, d, d);
          cube.WaitFixToM();
        }

        // 7+8. dH_delta = qg^T @ dO + w^T @ (-dv_new)
        cube.LoadL0A_Trans(A, l1Qg, d, 64);
        cube.LoadL0B_NoT(B, l1DO, 64, d, 64);
        cube.WaitMte1ToM();
        cube.RunMmad(C0, A, B, d, d, 64, true);
        cube.WaitMToMte1();

        cube.LoadL0A_Trans(A, l1W, d, 64);
        cube.LoadL0B_NoT(B, l1NegDvn, 64, d, 64);
        cube.WaitMte1ToM();
        cube.RunMmad(C0, A, B, d, d, 64, false);
        cube.WaitMToMte1();
        cube.WaitMToFix();
        cube.StoreFp32Gm(dhDeltaGm, C0, d, d, d);
        cube.WaitFixToM();
        cube.WaitMte1ToMte2();

        CrossCoreSetFlag<0x2, PIPE_FIX>(3);
      }
    }
    cube.Destroy();
  }
}

__aicore__ inline void RunPhaseB1BwdHAiv(uint32_t wid, GM_ADDR *io, const HyperMegaKdaGradTilingData *t,
                                         bool initFromExcl, bool publishSummary, bool summaryOnly) {
  if ASCEND_IS_AIV {
    const uint32_t core = wid >> 1;
    const uint32_t sub = wid & 1U;
    const uint32_t numCube = t->numCubeCores;
    const uint32_t S = t->seqLocal;
    const uint32_t H = t->numHeads;
    const uint32_t d = t->headDim;
    const uint32_t C = t->chunkSize;
    const uint32_t numChunks = t->numChunks;
    const uint32_t halfD = d >> 1;
    const uint32_t halfC = C >> 1;
    const uint32_t sRowOff = sub * halfD;
    const uint32_t vRowOff = sub * halfC;
    const auto cacheL = MegaKdaCache::ComputeLayout(S, H, d, C);

    GM_ADDR ws = io[B_WORKSPACE];
    GM_ADDR bc = io[B_BWD_CACHE];
    GM_ADDR scBase = ws + t->wsBwdHScratchOff + static_cast<uint64_t>(core) * BWDH_CORE_BYTES;

    __gm__ bfloat16_t *dhPtr      = reinterpret_cast<__gm__ bfloat16_t *>(scBase + BWDH_OFF_DH);
    __gm__ float *dhDeltaPtr      = reinterpret_cast<__gm__ float *>(scBase + BWDH_OFF_DHDELTA);
    __gm__ bfloat16_t *dvnPtr     = reinterpret_cast<__gm__ bfloat16_t *>(scBase + BWDH_OFF_DVN);
    __gm__ bfloat16_t *negDvnPtr  = reinterpret_cast<__gm__ bfloat16_t *>(scBase + BWDH_OFF_NEGDVN);

    TPipe pipe;
    TBuf<TPosition::VECCALC> buf;
    const uint32_t uDH    = 0;
    const uint32_t uB1    = uDH + halfD * d * 4U;
    const uint32_t uB2    = uB1 + halfD * d * 4U;
    const uint32_t uBf    = uB2 + halfD * d * 4U;
    const uint32_t uExpG  = uBf + halfD * d * 2U;
    const uint32_t uDgVec = uExpG + 256U;
    const uint32_t uBrcb  = uDgVec + 256U;
    pipe.InitBuffer(buf, uBrcb + halfD * 8U * 4U + 64U);

    LocalTensor<float> dHUb    = buf.GetWithOffset<float>(halfD * d, uDH);
    LocalTensor<float> b1      = buf.GetWithOffset<float>(halfD * d, uB1);
    LocalTensor<float> b2      = buf.GetWithOffset<float>(halfD * d, uB2);
    LocalTensor<bfloat16_t> bf = buf.GetWithOffset<bfloat16_t>(halfD * d, uBf);
    LocalTensor<float> expG    = buf.GetWithOffset<float>(halfD, uExpG);
    LocalTensor<float> dgVec   = buf.GetWithOffset<float>(halfD, uDgVec);
    LocalTensor<float> brcb    = buf.GetWithOffset<float>(halfD * 8U, uBrcb);

    for (uint32_t h = core; h < H; h += numCube) {
      const uint64_t hStateOff = (static_cast<uint64_t>(h) * d + sRowOff) * d;
      if (initFromExcl) {
        GlobalTensor<float> exGm;
        exGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ws + t->wsRevExclOff) + hStateOff, halfD * d);
        DataCopy(dHUb, exGm, halfD * d);
        PipeBarrier<PIPE_ALL>();
      } else {
        Duplicate(dHUb, 0.0f, halfD * d);
        PipeBarrier<PIPE_V>();
      }
      Cast(bf, dHUb, RoundMode::CAST_RINT, halfD * d);
      PipeBarrier<PIPE_ALL>();
      {
        GlobalTensor<bfloat16_t> dhGm;
        dhGm.SetGlobalBuffer(dhPtr + static_cast<uint64_t>(sRowOff) * d, halfD * d);
        DataCopy(dhGm, bf, halfD * d);
        PipeBarrier<PIPE_ALL>();
      }
      CrossCoreSetFlag<0x2, PIPE_MTE3>(0);

      for (int32_t ci = static_cast<int32_t>(numChunks) - 1; ci >= 0; --ci) {
        const uint32_t c = static_cast<uint32_t>(ci);
        const uint64_t hcIdx = static_cast<uint64_t>(h) * numChunks + c;

        CrossCoreWaitFlag<0x2, PIPE_MTE2>(1);
        {
          GlobalTensor<bfloat16_t> srcVn, dstNeg;
          srcVn.SetGlobalBuffer(dvnPtr + vRowOff * d, halfC * d);
          dstNeg.SetGlobalBuffer(negDvnPtr + vRowOff * d, halfC * d);
          DataCopy(bf, srcVn, halfC * d);
          PipeBarrier<PIPE_ALL>();
          Cast(b1, bf, RoundMode::CAST_NONE, halfC * d);
          PipeBarrier<PIPE_V>();
          Muls(b1, b1, -1.0f, halfC * d);
          PipeBarrier<PIPE_V>();
          Cast(bf, b1, RoundMode::CAST_RINT, halfC * d);
          PipeBarrier<PIPE_ALL>();
          DataCopy(dstNeg, bf, halfC * d);
          PipeBarrier<PIPE_ALL>();
        }
        CrossCoreSetFlag<0x2, PIPE_MTE3>(2);

        if (!summaryOnly) {
          GlobalTensor<bfloat16_t> hcGm;
          GlobalTensor<float> dgGm;
          hcGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(bc + cacheL.hChunksOff) +
                                   (hcIdx * d + sRowOff) * d,
                               halfD * d);
          dgGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ws + t->wsDgammaOff) + hcIdx * d + sRowOff, halfD);
          DataCopy(bf, hcGm, halfD * d);
          PipeBarrier<PIPE_ALL>();
          Cast(b2, bf, RoundMode::CAST_NONE, halfD * d);
          PipeBarrier<PIPE_V>();
          Mul(b2, b2, dHUb, halfD * d);
          PipeBarrier<PIPE_V>();
          BinaryRepeatParams foldP{1, 1, 1, 16, 16, 16};
          Add(b2, b2, b2[64], 64, static_cast<uint8_t>(halfD), foldP);
          PipeBarrier<PIPE_V>();
          WholeReduceSum(dgVec, b2, 64, static_cast<int32_t>(halfD), 1, 1, 16);
          PipeBarrier<PIPE_ALL>();
          DataCopy(dgGm, dgVec, halfD);
          PipeBarrier<PIPE_ALL>();
        }

        CrossCoreWaitFlag<0x2, PIPE_MTE2>(3);

        if (ci > 0 || publishSummary) {
          GlobalTensor<float> egGm, dsGm;
          egGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(bc + cacheL.expGLastOff) + hcIdx * d + sRowOff, halfD);
          dsGm.SetGlobalBuffer(dhDeltaPtr + static_cast<uint64_t>(sRowOff) * d, halfD * d);
          DataCopy(expG, egGm, halfD);
          DataCopy(b1, dsGm, halfD * d);
          PipeBarrier<PIPE_ALL>();

          Brcb(brcb, expG, static_cast<uint8_t>(halfD / 8U), {1, 8});
          PipeBarrier<PIPE_V>();
          BinaryRepeatParams sScale{1, 1, 0, 16, 16, 1};
          Mul(dHUb, dHUb, brcb, 64, static_cast<uint8_t>(halfD), sScale);
          Mul(dHUb[64], dHUb[64], brcb, 64, static_cast<uint8_t>(halfD), sScale);
          PipeBarrier<PIPE_V>();
          Add(dHUb, dHUb, b1, halfD * d);
          PipeBarrier<PIPE_V>();

          if (ci > 0) {
            Cast(bf, dHUb, RoundMode::CAST_RINT, halfD * d);
            PipeBarrier<PIPE_ALL>();
            GlobalTensor<bfloat16_t> dhGm;
            dhGm.SetGlobalBuffer(dhPtr + static_cast<uint64_t>(sRowOff) * d, halfD * d);
            DataCopy(dhGm, bf, halfD * d);
            PipeBarrier<PIPE_ALL>();
            CrossCoreSetFlag<0x2, PIPE_MTE3>(0);
          } else {
            GlobalTensor<float> selfDGm, sendDGm;
            selfDGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ws + t->wsRevSelfOff) + hStateOff, halfD * d);
            sendDGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(io[B_SEND_BUF] + t->sendScanOff) + hStateOff,
                                    halfD * d);
            PipeBarrier<PIPE_ALL>();
            DataCopy(selfDGm, dHUb, halfD * d);
            DataCopy(sendDGm, dHUb, halfD * d);
            if (summaryOnly) {
              GlobalTensor<float> srcAGm, selfAGm, sendAGm;
              srcAGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(bc + cacheL.aRankOff) + hStateOff, halfD * d);
              selfAGm.SetGlobalBuffer(
                  reinterpret_cast<__gm__ float *>(ws + t->wsRevSelfOff + t->scanHalfBytes) + hStateOff, halfD * d);
              sendAGm.SetGlobalBuffer(
                  reinterpret_cast<__gm__ float *>(io[B_SEND_BUF] + t->sendScanOff + t->scanHalfBytes) + hStateOff,
                  halfD * d);
              DataCopy(b1, srcAGm, halfD * d);
              PipeBarrier<PIPE_ALL>();
              DataCopy(selfAGm, b1, halfD * d);
              DataCopy(sendAGm, b1, halfD * d);
            }
            PipeBarrier<PIPE_ALL>();
          }
        }
      }
    }
    pipe.Destroy();
  }
}

// ---------------------------------------------------------------------------
// Phase B1.5: 1:2 Cube-Accelerated Reverse State Merge
// ---------------------------------------------------------------------------
__aicore__ inline void RunBwdMatmul128Aic(uint32_t core, GM_ADDR *io, const HyperMegaKdaGradTilingData *t,
                                          bool transLhs) {
  if ASCEND_IS_AIC {
    const uint32_t numCube = t->numCubeCores;
    const uint32_t H = t->numHeads;
    const uint32_t d = t->headDim;
    GM_ADDR ws = io[B_WORKSPACE];

    MulticoreKda::KdaRawCubePipe cube;
    cube.Init(64 * 1024, 32 * 1024, 32 * 1024, 64 * 1024);
    auto l1A = cube.L1At(0);
    auto l1B = cube.L1At(32 * 1024);
    auto A = cube.L0A();
    auto B = cube.L0B();
    auto C0 = cube.L0C();

    for (uint32_t h = core; h < H; h += numCube) {
      GM_ADDR tmpBase = ws + t->wsRevTmpOff + static_cast<uint64_t>(h) * 3ULL * d * d * 4ULL;
      GlobalTensor<float> outGm;
      GlobalTensor<bfloat16_t> lhsGm, rhsGm;
      outGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(tmpBase), d * d);
      lhsGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(tmpBase + d * d * 4ULL), d * d);
      rhsGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(tmpBase + d * d * 4ULL) + d * d, d * d);

      CrossCoreWaitFlag<0x2, PIPE_MTE2>(0);
      cube.CopyGmToL1Nz(l1A, lhsGm, d, d, d, d);
      cube.CopyGmToL1Nz(l1B, rhsGm, d, d, d, d);
      cube.WaitMte2ToMte1();
      if (transLhs) {
        cube.LoadL0A_Trans(A, l1A, d, d);
      } else {
        cube.LoadL0A_NoT(A, l1A, d, d, d);
      }
      cube.LoadL0B_NoT(B, l1B, d, d, d);
      cube.WaitMte1ToM();
      cube.RunMmad(C0, A, B, d, d, d, true);
      cube.WaitMToMte1();
      cube.WaitMToFix();
      cube.StoreFp32Gm(outGm, C0, d, d, d);
      cube.WaitFixToM();
      cube.WaitMte1ToMte2();
      CrossCoreSetFlag<0x2, PIPE_FIX>(1);
    }
    cube.Destroy();
  }
}

__aicore__ inline void RunBwdMatmul128Aiv(uint32_t wid, GM_ADDR *io, const HyperMegaKdaGradTilingData *t,
                                          GM_ADDR lhsBase, GM_ADDR rhsBase, GM_ADDR addBase, GM_ADDR dstBase) {
  if ASCEND_IS_AIV {
    const uint32_t core = wid >> 1;
    const uint32_t sub = wid & 1U;
    const uint32_t numCube = t->numCubeCores;
    const uint32_t H = t->numHeads;
    const uint32_t d = t->headDim;
    const uint32_t halfD = d >> 1;
    GM_ADDR ws = io[B_WORKSPACE];

    TPipe pipe;
    TBuf<TPosition::VECCALC> buf;
    pipe.InitBuffer(buf, halfD * d * 4U * 2U + halfD * d * 2U + 64U);
    LocalTensor<float> f1 = buf.GetWithOffset<float>(halfD * d, 0);
    LocalTensor<float> f2 = buf.GetWithOffset<float>(halfD * d, halfD * d * 4U);
    LocalTensor<bfloat16_t> bf = buf.GetWithOffset<bfloat16_t>(halfD * d, halfD * d * 8U);

    for (uint32_t h = core; h < H; h += numCube) {
      const uint64_t hOff = (static_cast<uint64_t>(h) * d + sub * halfD) * d;
      GM_ADDR tmpBase = ws + t->wsRevTmpOff + static_cast<uint64_t>(h) * 3ULL * d * d * 4ULL;
      GlobalTensor<float> lGm, rGm, outTmpGm, addGm, dstGm;
      GlobalTensor<bfloat16_t> lBfGm, rBfGm;
      lGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(lhsBase) + hOff, halfD * d);
      rGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(rhsBase) + hOff, halfD * d);
      lBfGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(tmpBase + d * d * 4ULL) + sub * halfD * d, halfD * d);
      rBfGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(tmpBase + d * d * 4ULL) + d * d + sub * halfD * d,
                            halfD * d);
      outTmpGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(tmpBase) + sub * halfD * d, halfD * d);
      dstGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dstBase) + hOff, halfD * d);

      DataCopy(f1, lGm, halfD * d);
      DataCopy(f2, rGm, halfD * d);
      PipeBarrier<PIPE_ALL>();
      Cast(bf, f1, RoundMode::CAST_RINT, halfD * d);
      PipeBarrier<PIPE_ALL>();
      DataCopy(lBfGm, bf, halfD * d);
      PipeBarrier<PIPE_ALL>();
      Cast(bf, f2, RoundMode::CAST_RINT, halfD * d);
      PipeBarrier<PIPE_ALL>();
      DataCopy(rBfGm, bf, halfD * d);
      PipeBarrier<PIPE_ALL>();
      CrossCoreSetFlag<0x2, PIPE_MTE3>(0);

      CrossCoreWaitFlag<0x2, PIPE_MTE2>(1);
      DataCopy(f1, outTmpGm, halfD * d);
      if (addBase != nullptr) {
        addGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(addBase) + hOff, halfD * d);
        DataCopy(f2, addGm, halfD * d);
        PipeBarrier<PIPE_ALL>();
        Add(f1, f1, f2, halfD * d);
        PipeBarrier<PIPE_V>();
      } else {
        PipeBarrier<PIPE_ALL>();
      }
      PipeBarrier<PIPE_ALL>();
      DataCopy(dstGm, f1, halfD * d);
      PipeBarrier<PIPE_ALL>();
    }
    pipe.Destroy();
  }
}

__aicore__ inline void RunBwdCopyStateAiv(uint32_t wid, GM_ADDR srcBase, GM_ADDR dstBase, uint64_t totalFloats,
                                          uint32_t numAiv) {
  if ASCEND_IS_AIV {
    const uint64_t tileF = 4096ULL;
    const uint64_t numTiles = (totalFloats + tileF - 1ULL) / tileF;
    TPipe pipe;
    TBuf<TPosition::VECCALC> buf;
    pipe.InitBuffer(buf, tileF * 4U + 64U);
    LocalTensor<float> tmp = buf.GetWithOffset<float>(tileF, 0);
    for (uint64_t ti = wid; ti < numTiles; ti += numAiv) {
      const uint64_t off = ti * tileF;
      const uint32_t cur = static_cast<uint32_t>((off + tileF <= totalFloats) ? tileF : (totalFloats - off));
      GlobalTensor<float> sGm, dGm;
      sGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(srcBase) + off, cur);
      dGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dstBase) + off, cur);
      DataCopy(tmp, sGm, cur);
      PipeBarrier<PIPE_ALL>();
      DataCopy(dGm, tmp, cur);
      PipeBarrier<PIPE_ALL>();
    }
    pipe.Destroy();
  }
}

// ---------------------------------------------------------------------------
// Phase B2 (AIC + AIV Double-Buffered L2-Ring): WY_bwd + Intra_bwd + dG + Gate_bwd + Beta_bwd + L2Norm_bwd
// ---------------------------------------------------------------------------
__aicore__ inline void RunPhaseB2Aic(uint32_t core, GM_ADDR *io, const HyperMegaKdaGradTilingData *t) {
  if ASCEND_IS_AIC {
    const uint32_t numCube = t->numCubeCores;
    const uint32_t S = t->seqLocal;
    const uint32_t H = t->numHeads;
    const uint32_t d = t->headDim;
    const uint32_t C = t->chunkSize;
    const uint32_t numChunks = t->numChunks;
    const uint32_t numPairs = H >> 1;
    const uint32_t totalJobs = numChunks * numPairs;
    const auto cacheL = MegaKdaCache::ComputeLayout(S, H, d, C);

    GM_ADDR ws = io[B_WORKSPACE];
    GM_ADDR bc = io[B_BWD_CACHE];
    GM_ADDR ringCore = ws + t->wsBwdRingOff + static_cast<uint64_t>(core) * BWDR_CORE_BYTES;

    MulticoreKda::KdaRawCubePipe cube;
    cube.Init(128 * 1024, 32 * 1024, 32 * 1024, 64 * 1024);
    auto l1MInv    = cube.L1At(0);
    auto l1Dw      = cube.L1At(8 * 1024);
    auto l1DvNew   = cube.L1At(24 * 1024);
    auto l1W       = cube.L1At(40 * 1024);
    auto l1U       = cube.L1At(56 * 1024);
    auto l1Dkbg    = cube.L1At(72 * 1024);
    auto l1Dvbeta  = cube.L1At(88 * 1024);
    auto l1DAStrip = cube.L1At(0);
    auto l1KRef    = cube.L1At(8 * 1024);
    auto l1QKbRef  = cube.L1At(24 * 1024);
    auto A = cube.L0A();
    auto B = cube.L0B();
    auto C0 = cube.L0C();

    for (uint32_t j = 0, job = core; job < totalJobs; ++j, job += numCube) {
      const uint32_t slot = j & 1U;
      const uint32_t c = job / numPairs;
      const uint32_t pair = job % numPairs;
      GM_ADDR slotBase = ringCore + slot * BWDR_SLOT_BYTES;

      if (j >= 2U) {
        CrossCoreWaitFlag<0x2, PIPE_MTE2>(6U + slot);
      }

      // Step 1: WY_bwd + dAkk_raw
      for (uint32_t lane = 0; lane < 2U; ++lane) {
        const uint32_t h = (pair << 1) + lane;
        const uint64_t hcIdx = static_cast<uint64_t>(h) * numChunks + c;
        GM_ADDR laneBase = slotBase + lane * BWDR_LANE_BYTES;

        GlobalTensor<bfloat16_t> mInvGm, dwGm, dvnGm, wGm, uGm, dkbgGm, dvbGm;
        GlobalTensor<float> dAkkGm;
        mInvGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(bc + cacheL.mInvOff) + hcIdx * C * C, C * C);
        dwGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + t->wsDwOff) + hcIdx * C * d, C * d);
        dvnGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + t->wsDvNewOff) + hcIdx * C * d, C * d);
        wGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(bc + cacheL.wOff) + hcIdx * C * d, C * d);
        uGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(bc + cacheL.uOff) + hcIdx * C * d, C * d);
        dkbgGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(laneBase + BWDR_OFF_DKBG), C * d);
        dvbGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(laneBase + BWDR_OFF_DVBETA), C * d);
        dAkkGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(laneBase + BWDR_OFF_DAKKRAW), C * C);

        cube.CopyGmToL1Nz(l1MInv, mInvGm, 64, 64, 64, 64);
        cube.CopyGmToL1Nz(l1Dw, dwGm, 64, d, d, 64);
        cube.CopyGmToL1Nz(l1DvNew, dvnGm, 64, d, d, 64);
        cube.CopyGmToL1Nz(l1W, wGm, 64, d, d, 64);
        cube.CopyGmToL1Nz(l1U, uGm, 64, d, d, 64);
        cube.WaitMte2ToMte1();

        cube.LoadL0A_Trans(A, l1MInv, 64, 64);
        cube.LoadL0B_NoT(B, l1Dw, 64, d, 64);
        cube.WaitMte1ToM();
        cube.RunMmad(C0, A, B, 64, d, 64, true);
        cube.WaitMToMte1();
        cube.WaitMToFix();
        cube.StoreBf16Gm(dkbgGm, C0, 64, d, d);
        cube.StoreBf16L1Nz(l1Dkbg, C0, 64, d, 64);
        cube.WaitFixToM();

        cube.LoadL0A_Trans(A, l1MInv, 64, 64);
        cube.LoadL0B_NoT(B, l1DvNew, 64, d, 64);
        cube.WaitMte1ToM();
        cube.RunMmad(C0, A, B, 64, d, 64, true);
        cube.WaitMToMte1();
        cube.WaitMToFix();
        cube.StoreBf16Gm(dvbGm, C0, 64, d, d);
        cube.StoreBf16L1Nz(l1Dvbeta, C0, 64, d, 64);
        cube.WaitFixToMte1();
        cube.WaitFixToM();

        cube.LoadL0A_NoT(A, l1Dkbg, 64, d, 64);
        cube.LoadL0B_Trans(B, l1W, d, 64);
        cube.WaitMte1ToM();
        cube.RunMmad(C0, A, B, 64, 64, d, true);
        cube.WaitMToMte1();

        cube.LoadL0A_NoT(A, l1Dvbeta, 64, d, 64);
        cube.LoadL0B_Trans(B, l1U, d, 64);
        cube.WaitMte1ToM();
        cube.RunMmad(C0, A, B, 64, 64, d, false);
        cube.WaitMToMte1();
        cube.WaitMToFix();
        cube.StoreFp32Gm(dAkkGm, C0, 64, 64, 64);
        cube.WaitFixToM();
        cube.WaitMte1ToMte2();
      }
      CrossCoreSetFlag<0x2, PIPE_FIX>(0U + slot);

      CrossCoreWaitFlag<0x2, PIPE_MTE2>(2U + slot);

      // Step 2: 6 off-diag 16x16 strip matmuls per lane
      const uint32_t kRefRowOffs[3] = {0U, 16U, 48U};
      const uint32_t dAStrElemOffs[3] = {0U, 32U * 16U, 32U * 16U + 32U * 32U};
      for (uint32_t lane = 0; lane < 2U; ++lane) {
        GM_ADDR laneBase = slotBase + lane * BWDR_LANE_BYTES;
        __gm__ bfloat16_t *kRefBase   = reinterpret_cast<__gm__ bfloat16_t *>(laneBase + BWDR_OFF_KREF);
        __gm__ bfloat16_t *qKbRefBase = reinterpret_cast<__gm__ bfloat16_t *>(laneBase + BWDR_OFF_QKBREF);
        __gm__ bfloat16_t *dAStrBase  = reinterpret_cast<__gm__ bfloat16_t *>(laneBase + BWDR_OFF_DASTRIP);
        __gm__ bfloat16_t *tQbkBase   = reinterpret_cast<__gm__ bfloat16_t *>(laneBase + BWDR_OFF_TQBK);
        __gm__ bfloat16_t *tKnegBase  = reinterpret_cast<__gm__ bfloat16_t *>(laneBase + BWDR_OFF_TKNEG);

        for (uint32_t a = 1U; a <= 3U; ++a) {
          const uint32_t cols = a * 16U;
          const uint32_t kRowOff = kRefRowOffs[a - 1U];
          const uint32_t qRowOff = (a - 1U) * 32U;
          const uint32_t daOff = dAStrElemOffs[a - 1U];

          GlobalTensor<bfloat16_t> daGm, krGm, qkrGm, tqOutGm, tkOutGm;
          daGm.SetGlobalBuffer(dAStrBase + daOff, 32U * cols);
          krGm.SetGlobalBuffer(kRefBase + kRowOff * d, cols * d);
          qkrGm.SetGlobalBuffer(qKbRefBase + qRowOff * d, 32U * d);
          tqOutGm.SetGlobalBuffer(tQbkBase + qRowOff * d, 32U * d);
          tkOutGm.SetGlobalBuffer(tKnegBase + kRowOff * d, cols * d);

          cube.CopyGmToL1Nz(l1DAStrip, daGm, 32U, cols, cols, 32U);
          cube.CopyGmToL1Nz(l1KRef, krGm, cols, d, d, cols);
          cube.CopyGmToL1Nz(l1QKbRef, qkrGm, 32U, d, d, 32U);
          cube.WaitMte2ToMte1();

          cube.LoadL0A_NoT(A, l1DAStrip, 32U, cols, 32U);
          cube.LoadL0B_NoT(B, l1KRef, cols, d, cols);
          cube.WaitMte1ToM();
          cube.RunMmad(C0, A, B, 32U, d, cols, true);
          cube.WaitMToMte1();
          cube.WaitMToFix();
          cube.StoreBf16Gm(tqOutGm, C0, 32U, d, d);
          cube.WaitFixToM();

          cube.LoadL0A_Trans(A, l1DAStrip, cols, 32U);
          cube.LoadL0B_NoT(B, l1QKbRef, 32U, d, 32U);
          cube.WaitMte1ToM();
          cube.RunMmad(C0, A, B, cols, d, 32U, true);
          cube.WaitMToMte1();
          cube.WaitMToFix();
          cube.StoreBf16Gm(tkOutGm, C0, cols, d, d);
          cube.WaitFixToM();
          cube.WaitMte1ToMte2();
        }

        const float lbAic = BitsToFloatBwd(t->lowerBoundBits);
        if (t->hasLowerBound != 0U && lbAic >= -5.5f) {
          for (uint32_t a = 0; a < 4U; ++a) {
            const uint32_t kRowOff = 96U + a * 16U;
            const uint32_t qRowOff = 96U + a * 32U;
            const uint32_t daOff = 3072U + a * 512U;

            GlobalTensor<bfloat16_t> daGm, krGm, qkrGm, tqOutGm, tkOutGm;
            daGm.SetGlobalBuffer(dAStrBase + daOff, 32U * 16U);
            krGm.SetGlobalBuffer(kRefBase + kRowOff * d, 16U * d);
            qkrGm.SetGlobalBuffer(qKbRefBase + qRowOff * d, 32U * d);
            tqOutGm.SetGlobalBuffer(tQbkBase + qRowOff * d, 32U * d);
            tkOutGm.SetGlobalBuffer(tKnegBase + kRowOff * d, 16U * d);

            cube.CopyGmToL1Nz(l1DAStrip, daGm, 32U, 16U, 16U, 32U);
            cube.CopyGmToL1Nz(l1KRef, krGm, 16U, d, d, 16U);
            cube.CopyGmToL1Nz(l1QKbRef, qkrGm, 32U, d, d, 32U);
            cube.WaitMte2ToMte1();

            cube.LoadL0A_NoT(A, l1DAStrip, 32U, 16U, 32U);
            cube.LoadL0B_NoT(B, l1KRef, 16U, d, 16U);
            cube.WaitMte1ToM();
            cube.RunMmad(C0, A, B, 32U, d, 16U, true);
            cube.WaitMToMte1();
            cube.WaitMToFix();
            cube.StoreBf16Gm(tqOutGm, C0, 32U, d, d);
            cube.WaitFixToM();

            cube.LoadL0A_Trans(A, l1DAStrip, 16U, 32U);
            cube.LoadL0B_NoT(B, l1QKbRef, 32U, d, 32U);
            cube.WaitMte1ToM();
            cube.RunMmad(C0, A, B, 16U, d, 32U, true);
            cube.WaitMToMte1();
            cube.WaitMToFix();
            cube.StoreBf16Gm(tkOutGm, C0, 16U, d, d);
            cube.WaitFixToM();
            cube.WaitMte1ToMte2();
          }
        }
      }
      CrossCoreSetFlag<0x2, PIPE_FIX>(4U + slot);
    }
    // Drain final in-flight slot flags so CrossCore counters remain 100% balanced!
    uint32_t jobCount = 0U;
    if (core < totalJobs) {
      jobCount = (totalJobs - 1U - core) / numCube + 1U;
    }
    if (jobCount >= 2U) {
      CrossCoreWaitFlag<0x2, PIPE_MTE2>(6U + ((jobCount - 2U) & 1U));
    }
    if (jobCount >= 1U) {
      CrossCoreWaitFlag<0x2, PIPE_MTE2>(6U + ((jobCount - 1U) & 1U));
    }
    cube.Destroy();
  }
}

__aicore__ inline void RunPhaseB2Aiv(uint32_t wid, GM_ADDR *io, const HyperMegaKdaGradTilingData *t) {
  if ASCEND_IS_AIV {
    const uint32_t core = wid >> 1;
    const uint32_t lane = wid & 1U;
    const uint32_t numCube = t->numCubeCores;
    const uint32_t S = t->seqLocal;
    const uint32_t H = t->numHeads;
    const uint32_t d = t->headDim;
    const uint32_t C = t->chunkSize;
    const uint32_t D = H * d;
    const uint32_t numChunks = t->numChunks;
    const uint32_t numPairs = H >> 1;
    const uint32_t totalJobs = numChunks * numPairs;
    const float scale = BitsToFloatBwd(t->scaleBits);
    const float lowerBound = BitsToFloatBwd(t->lowerBoundBits);
    const auto cacheL = MegaKdaCache::ComputeLayout(S, H, d, C);

    GM_ADDR ws = io[B_WORKSPACE];
    GM_ADDR bc = io[B_BWD_CACHE];
    GM_ADDR ringCore = ws + t->wsBwdRingOff + static_cast<uint64_t>(core) * BWDR_CORE_BYTES;

    const uint64_t aLogStrideF = ((H + 7U) & ~7U);
    __gm__ float *pALogPtr = reinterpret_cast<__gm__ float *>(ws + t->wsPartialALogOff) + wid * aLogStrideF;
    __gm__ float *pDtBiasPtr = reinterpret_cast<__gm__ float *>(ws + t->wsPartialDtBiasOff) + wid * D;

    TPipe pipe;
    TBuf<TPosition::VECCALC> buf;
    // Peak UB = 185.2 KB < 192 KB
    const uint32_t uGCum     = 0;
    const uint32_t uDAqk     = 32 * 1024;
    const uint32_t uDAkk     = 48 * 1024;
    const uint32_t uAcc1     = 64 * 1024;
    const uint32_t uAcc2     = 96 * 1024;
    const uint32_t uWork     = 128 * 1024;
    const uint32_t uBf       = 160 * 1024;
    const uint32_t uSmall    = 176 * 1024;
    const uint32_t uBeta     = uSmall;
    const uint32_t uBetaBrcb = uBeta + 256U;
    const uint32_t uGRef     = uBetaBrcb + 2048U;
    const uint32_t uDtBias   = uGRef + 512U;
    const uint32_t uDbTot    = uDtBias + 512U;
    const uint32_t uRstd     = uDbTot + 256U;
    const uint32_t uALogVec  = uRstd + 256U;
    const uint32_t uDgGamma  = uALogVec + 256U;
    const uint32_t uRowTmp   = uDgGamma + 512U;
    const uint32_t uScal     = uRowTmp + 2048U;
    const uint32_t uScalBrcb = uScal + 256U;
    const uint32_t uEnd      = uScalBrcb + 2048U;
    pipe.InitBuffer(buf, uEnd + 64U);

    LocalTensor<float> gCum     = buf.GetWithOffset<float>(C * d, uGCum);
    LocalTensor<float> dAqkFp   = buf.GetWithOffset<float>(C * C, uDAqk);
    LocalTensor<float> dAkkFp   = buf.GetWithOffset<float>(C * C, uDAkk);
    LocalTensor<float> tmpFull  = buf.GetWithOffset<float>(C * d, uDAqk);  // alias of dAqkFp+dAkkFp (32 KB)
    LocalTensor<float> acc1     = buf.GetWithOffset<float>(C * d, uAcc1);
    LocalTensor<float> acc2     = buf.GetWithOffset<float>(C * d, uAcc2);
    LocalTensor<float> work     = buf.GetWithOffset<float>(C * d, uWork);
    LocalTensor<bfloat16_t> bf  = buf.GetWithOffset<bfloat16_t>(C * d, uBf);
    LocalTensor<float> betaFp   = buf.GetWithOffset<float>(C, uBeta);
    LocalTensor<float> betaBrcb = buf.GetWithOffset<float>(C * 8U, uBetaBrcb);
    LocalTensor<float> gRef     = buf.GetWithOffset<float>(d, uGRef);
    LocalTensor<float> dtBias   = buf.GetWithOffset<float>(d, uDtBias);
    LocalTensor<float> dbTot    = buf.GetWithOffset<float>(C, uDbTot);
    LocalTensor<float> rstdFp   = buf.GetWithOffset<float>(C, uRstd);
    LocalTensor<float> aLogAll  = buf.GetWithOffset<float>(64, uALogVec);
    LocalTensor<float> dgGamma  = buf.GetWithOffset<float>(d, uDgGamma);
    LocalTensor<float> rowTmp   = buf.GetWithOffset<float>(512, uRowTmp);
    LocalTensor<float> scalVec  = buf.GetWithOffset<float>(C, uScal);
    LocalTensor<float> scalBrcb = buf.GetWithOffset<float>(C * 8U, uScalBrcb);

    // Zero per-AIV partial d_a_log & d_dt_bias accumulators in workspace
    {
      Duplicate(work, 0.0f, C * d);
      PipeBarrier<PIPE_ALL>();
      GlobalTensor<float> pAGm;
      pAGm.SetGlobalBuffer(pALogPtr, static_cast<uint32_t>(aLogStrideF));
      DataCopy(pAGm, work, static_cast<uint32_t>(aLogStrideF));
      for (uint32_t off = 0; off < D; off += C * d) {
        const uint32_t cur = (off + C * d <= D) ? (C * d) : (D - off);
        GlobalTensor<float> pBGm;
        pBGm.SetGlobalBuffer(pDtBiasPtr + off, cur);
        DataCopy(pBGm, work, cur);
      }
      PipeBarrier<PIPE_ALL>();
      Duplicate(aLogAll, 0.0f, 64);
      PipeBarrier<PIPE_ALL>();
      GlobalTensor<float> alGm;
      alGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(io[B_A_LOG]), H);
      DataCopy(aLogAll, alGm, ((H + 7U) & ~7U));
      PipeBarrier<PIPE_ALL>();
      Exp(aLogAll, aLogAll, 64);
      PipeBarrier<PIPE_ALL>();
    }

    const uint16_t dBlockLen = static_cast<uint16_t>((d * 2U) / 32U);
    const uint16_t dRowStride = static_cast<uint16_t>(((D - d) * 2U) / 32U);
    const uint32_t kRefRowOffs[3] = {0U, 16U, 48U};
    const uint32_t dAStrElemOffs[3] = {0U, 32U * 16U, 32U * 16U + 32U * 32U};

    for (uint32_t j = 0, job = core; job < totalJobs; ++j, job += numCube) {
      const uint32_t slot = j & 1U;
      const uint32_t c = job / numPairs;
      const uint32_t pair = job % numPairs;
      const uint32_t h = (pair << 1) + lane;
      const uint64_t hcIdx = static_cast<uint64_t>(h) * numChunks + c;
      const uint64_t tStart = static_cast<uint64_t>(c) * C;
      GM_ADDR laneBase = ringCore + slot * BWDR_SLOT_BYTES + lane * BWDR_LANE_BYTES;

      CrossCoreWaitFlag<0x2, PIPE_MTE2>(0U + slot);

      // Load gCum, dAqk (mask incl), dAkk_raw (negate & mask strict), beta, dtBias
      {
        GlobalTensor<float> gcGm, dAkkRawGm, dtGm;
        GlobalTensor<bfloat16_t> dAqkGm, betaGm;
        gcGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(bc + cacheL.gCumOff) + hcIdx * C * d, C * d);
        dAkkRawGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(laneBase + BWDR_OFF_DAKKRAW), C * C);
        dAqkGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + t->wsDAqkOff) + hcIdx * C * C, C * C);
        dtGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(io[B_DT_BIAS]) + static_cast<uint64_t>(h) * d, d);
        DataCopy(gCum, gcGm, C * d);
        DataCopy(dAkkFp, dAkkRawGm, C * C);
        DataCopy(bf, dAqkGm, C * C);
        DataCopy(dtBias, dtGm, d);
        PipeBarrier<PIPE_ALL>();
        Cast(dAqkFp, bf, RoundMode::CAST_NONE, C * C);
        PipeBarrier<PIPE_V>();
        Muls(dAkkFp, dAkkFp, -1.0f, C * C);
        PipeBarrier<PIPE_ALL>();
        betaGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(io[B_BETA_IN]) + tStart * H, C * H);
        DataCopy(bf, betaGm, C * H);
        PipeBarrier<PIPE_ALL>();
        Cast(work, bf, RoundMode::CAST_NONE, C * H);
        PipeBarrier<PIPE_ALL>();
        for (uint32_t r = 0; r < C; ++r) {
          betaFp.SetValue(r, work.GetValue(r * H + h));
        }
        PipeBarrier<PIPE_ALL>();
        Muls(betaFp, betaFp, -1.0f, C);
        PipeBarrier<PIPE_V>();
        Mins(betaFp, betaFp, 80.0f, C);
        PipeBarrier<PIPE_V>();
        Maxs(betaFp, betaFp, -80.0f, C);
        PipeBarrier<PIPE_V>();
        Exp(betaFp, betaFp, C);
        PipeBarrier<PIPE_V>();
        Adds(betaFp, betaFp, 1.0f, C);
        PipeBarrier<PIPE_V>();
        Duplicate(scalVec, 1.0f, C);
        PipeBarrier<PIPE_V>();
        Div(betaFp, scalVec, betaFp, C);
        PipeBarrier<PIPE_V>();
        Brcb(betaBrcb, betaFp, static_cast<uint8_t>(C / 8U), {1, 8});
        PipeBarrier<PIPE_V>();
      }

      // Load qNorm -> acc1, kNorm -> acc2 for Step 1 off-diag prep
      {
        GlobalTensor<bfloat16_t> qGm, kGm;
        qGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(bc + cacheL.qNormOff) +
                                (static_cast<uint64_t>(h) * S + tStart) * d, C * d);
        kGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(bc + cacheL.kNormOff) +
                                (static_cast<uint64_t>(h) * S + tStart) * d, C * d);
        DataCopy(bf, qGm, C * d);
        PipeBarrier<PIPE_ALL>();
        Cast(acc1, bf, RoundMode::CAST_NONE, C * d);
        PipeBarrier<PIPE_ALL>();
        DataCopy(bf, kGm, C * d);
        PipeBarrier<PIPE_ALL>();
        Cast(acc2, bf, RoundMode::CAST_NONE, C * d);
        PipeBarrier<PIPE_ALL>();
      }

      // Step 1: Prepare safe off-diag operands (a = 1, 2, 3)
      __gm__ bfloat16_t *kRefBase   = reinterpret_cast<__gm__ bfloat16_t *>(laneBase + BWDR_OFF_KREF);
      __gm__ bfloat16_t *qKbRefBase = reinterpret_cast<__gm__ bfloat16_t *>(laneBase + BWDR_OFF_QKBREF);
      __gm__ bfloat16_t *dAStrBase  = reinterpret_cast<__gm__ bfloat16_t *>(laneBase + BWDR_OFF_DASTRIP);

      for (uint32_t a = 1U; a <= 3U; ++a) {
        const uint32_t rA = a * 16U;
        const uint32_t cols = rA;
        const uint32_t kRowOff = kRefRowOffs[a - 1U];
        const uint32_t qRowOff = (a - 1U) * 32U;
        const uint32_t daOff = dAStrElemOffs[a - 1U];

        Adds(gRef, gCum[rA * d], 0.0f, d);
        PipeBarrier<PIPE_V>();

        BinaryRepeatParams subRef{1, 1, 1, 16, 0, 16};
        Sub(work, gRef, gCum, 64, static_cast<uint8_t>(cols), subRef);
        Sub(work[64], gRef[64], gCum[64], 64, static_cast<uint8_t>(cols), subRef);
        PipeBarrier<PIPE_V>();
        Mins(work, work, 0.0f, cols * d);
        PipeBarrier<PIPE_V>();
        Maxs(work, work, -80.0f, cols * d);
        PipeBarrier<PIPE_V>();
        Exp(work, work, cols * d);
        PipeBarrier<PIPE_V>();
        Mul(work, work, acc2, cols * d);
        PipeBarrier<PIPE_V>();
        Cast(bf, work, RoundMode::CAST_RINT, cols * d);
        PipeBarrier<PIPE_ALL>();
        {
          GlobalTensor<bfloat16_t> krGm;
          krGm.SetGlobalBuffer(kRefBase + kRowOff * d, cols * d);
          DataCopy(krGm, bf, cols * d);
          PipeBarrier<PIPE_ALL>();
        }

        BinaryRepeatParams subRow{1, 1, 1, 16, 16, 0};
        Sub(work, gCum[rA * d], gRef, 64, 16, subRow);
        Sub(work[64], gCum[rA * d + 64], gRef[64], 64, 16, subRow);
        PipeBarrier<PIPE_V>();
        Mins(work, work, 0.0f, 16U * d);
        PipeBarrier<PIPE_V>();
        Maxs(work, work, -80.0f, 16U * d);
        PipeBarrier<PIPE_V>();
        Exp(work, work, 16U * d);
        PipeBarrier<PIPE_V>();

        Mul(work[16U * d], acc1[rA * d], work, 16U * d);
        PipeBarrier<PIPE_V>();
        Muls(work[16U * d], work[16U * d], scale, 16U * d);
        PipeBarrier<PIPE_V>();

        Mul(work[32U * d], acc2[rA * d], work, 16U * d);
        PipeBarrier<PIPE_V>();
        BinaryRepeatParams bScale{1, 1, 0, 16, 16, 1};
        Mul(work[32U * d], work[32U * d], betaBrcb[rA * 8U], 64, 16, bScale);
        Mul(work[32U * d + 64U], work[32U * d + 64U], betaBrcb[rA * 8U], 64, 16, bScale);
        PipeBarrier<PIPE_V>();
        Cast(bf, work[16U * d], RoundMode::CAST_RINT, 32U * d);
        PipeBarrier<PIPE_ALL>();
        {
          GlobalTensor<bfloat16_t> qkrGm;
          qkrGm.SetGlobalBuffer(qKbRefBase + qRowOff * d, 32U * d);
          DataCopy(qkrGm, bf, 32U * d);
          PipeBarrier<PIPE_ALL>();
        }

        UnaryRepeatParams packP{1, 1, static_cast<uint8_t>(cols / 8U), 8};
        Adds(work[0], dAqkFp[rA * C], 0.0f, static_cast<uint64_t>(cols), 16, packP);
        Adds(work[16U * cols], dAkkFp[rA * C], 0.0f, static_cast<uint64_t>(cols), 16, packP);
        PipeBarrier<PIPE_V>();
        Cast(bf, work, RoundMode::CAST_RINT, 32U * cols);
        PipeBarrier<PIPE_ALL>();
        {
          GlobalTensor<bfloat16_t> daGm;
          daGm.SetGlobalBuffer(dAStrBase + daOff, 32U * cols);
          DataCopy(daGm, bf, 32U * cols);
          PipeBarrier<PIPE_ALL>();
        }
      }

      const bool useCubeDiag = (t->hasLowerBound != 0U) && (lowerBound >= -5.5f);
      if (useCubeDiag) {
        // Prepare 4x(16x16) diag blocks using midpoint reference rMid = base + 8 (|dG| <= 40, BF16-safe!)
        BinaryRepeatParams subRef{1, 1, 1, 16, 0, 16};
        BinaryRepeatParams subRow{1, 1, 1, 16, 16, 0};
        BinaryRepeatParams bScale{1, 1, 0, 16, 16, 1};
        UnaryRepeatParams pack16P{1, 1, 2, 8};

        for (uint32_t a = 0; a < 4U; ++a) {
          const uint32_t base = a * 16U;
          const uint32_t rMid = base + 8U;
          const uint32_t kRowOff = 96U + a * 16U;
          const uint32_t qRowOff = 96U + a * 32U;
          const uint32_t daOff = 3072U + a * 512U;

          Adds(gRef, gCum[rMid * d], 0.0f, d);
          PipeBarrier<PIPE_V>();

          // kRefDiag = k[base..base+16] * exp(G[rMid] - G[j])
          Sub(work, gRef, gCum[base * d], 64, 16, subRef);
          Sub(work[64], gRef[64], gCum[base * d + 64U], 64, 16, subRef);
          PipeBarrier<PIPE_V>();
          Mins(work, work, 45.0f, 16U * d);
          PipeBarrier<PIPE_V>();
          Maxs(work, work, -80.0f, 16U * d);
          PipeBarrier<PIPE_V>();
          Exp(work, work, 16U * d);
          PipeBarrier<PIPE_V>();
          Mul(work, work, acc2[base * d], 16U * d);
          PipeBarrier<PIPE_V>();
          Cast(bf, work, RoundMode::CAST_RINT, 16U * d);
          PipeBarrier<PIPE_ALL>();
          {
            GlobalTensor<bfloat16_t> krGm;
            krGm.SetGlobalBuffer(kRefBase + kRowOff * d, 16U * d);
            DataCopy(krGm, bf, 16U * d);
            PipeBarrier<PIPE_ALL>();
          }

          // qKbRefDiag: [scale * q * exp(G[i] - G[rMid]); beta * k * exp(G[i] - G[rMid])]
          Sub(work, gCum[base * d], gRef, 64, 16, subRow);
          Sub(work[64], gCum[base * d + 64U], gRef[64], 64, 16, subRow);
          PipeBarrier<PIPE_V>();
          Mins(work, work, 45.0f, 16U * d);
          PipeBarrier<PIPE_V>();
          Maxs(work, work, -80.0f, 16U * d);
          PipeBarrier<PIPE_V>();
          Exp(work, work, 16U * d);
          PipeBarrier<PIPE_V>();
          Mul(work[16U * d], acc1[base * d], work, 16U * d);
          PipeBarrier<PIPE_V>();
          Muls(work[16U * d], work[16U * d], scale, 16U * d);
          PipeBarrier<PIPE_V>();
          Mul(work[32U * d], acc2[base * d], work, 16U * d);
          PipeBarrier<PIPE_V>();
          Mul(work[32U * d], work[32U * d], betaBrcb[base * 8U], 64, 16, bScale);
          Mul(work[32U * d + 64U], work[32U * d + 64U], betaBrcb[base * 8U], 64, 16, bScale);
          PipeBarrier<PIPE_V>();
          Cast(bf, work[16U * d], RoundMode::CAST_RINT, 32U * d);
          PipeBarrier<PIPE_ALL>();
          {
            GlobalTensor<bfloat16_t> qkrGm;
            qkrGm.SetGlobalBuffer(qKbRefBase + qRowOff * d, 32U * d);
            DataCopy(qkrGm, bf, 32U * d);
            PipeBarrier<PIPE_ALL>();
          }

          // dADiag [32, 16]: STRICT tril(-2) (i >= j + 2) for BOTH dAqk and dAkk!
          // Both i == j (rowTmp[0..63]) and i == j + 1 (rowTmp[64..255]) are computed in exact FP32
          // after Step 2B so dG = q*dq + k*dk_pos - k*dk_neg achieves full FP32 cancellation accuracy!
          Adds(work[0], dAqkFp[base * C + base], 0.0f, 16, 16, pack16P);
          Adds(work[256], dAkkFp[base * C + base], 0.0f, 16, 16, pack16P);
          PipeBarrier<PIPE_ALL>();
          for (uint32_t r = 0; r < 16U; ++r) {
            rowTmp.SetValue(base + r, work.GetValue(r * 16U + r) * scale);
            if (r + 1U < 16U) {
              const float sqk = work.GetValue((r + 1U) * 16U + r) * scale;
              const float skk = work.GetValue(256U + (r + 1U) * 16U + r);
              const float bNext = betaFp.GetValue(base + r + 1U);
              rowTmp.SetValue(64U + base + r, sqk);
              rowTmp.SetValue(128U + base + r, skk);
              rowTmp.SetValue(192U + base + r, skk * bNext);
            } else {
              rowTmp.SetValue(64U + base + r, 0.0f);
              rowTmp.SetValue(128U + base + r, 0.0f);
              rowTmp.SetValue(192U + base + r, 0.0f);
            }
            const uint32_t colStart = (r > 0U) ? (r - 1U) : 0U;
            for (uint32_t col = colStart; col < 16U; ++col) {
              work.SetValue(r * 16U + col, 0.0f);
              work.SetValue(256U + r * 16U + col, 0.0f);
            }
          }
          PipeBarrier<PIPE_ALL>();
          Cast(bf, work, RoundMode::CAST_RINT, 512U);
          PipeBarrier<PIPE_ALL>();
          {
            GlobalTensor<bfloat16_t> daGm;
            daGm.SetGlobalBuffer(dAStrBase + daOff, 512U);
            DataCopy(daGm, bf, 512U);
            PipeBarrier<PIPE_ALL>();
          }
        }
      }
      CrossCoreSetFlag<0x2, PIPE_MTE3>(2U + slot);

      if (!useCubeDiag) {
        // Overlap with AIC Step 2: pack 4x(16x16) lower-tri diag blocks column-major into tmpFull[0..2048)
        Duplicate(acc1, 0.0f, 2048U);
        PipeBarrier<PIPE_ALL>();
        for (uint32_t a = 0; a < 4U; ++a) {
          const uint32_t base = a * 16U;
          for (uint32_t jj = 0; jj < 16U; ++jj) {
            const uint32_t jRow = base + jj;
            const uint32_t colBase = (a * 16U + jj) * 16U;
            acc1.SetValue(colBase + jj, dAqkFp.GetValue(jRow * C + jRow) * scale);
            for (uint32_t ii = jj + 1U; ii < 16U; ++ii) {
              const uint32_t iRow = base + ii;
              acc1.SetValue(colBase + ii, dAqkFp.GetValue(iRow * C + jRow) * scale);
              acc1.SetValue(1024U + colBase + ii, dAkkFp.GetValue(iRow * C + jRow));
            }
          }
        }
        PipeBarrier<PIPE_ALL>();
        Adds(tmpFull[0], acc1[0], 0.0f, 2048U);
        PipeBarrier<PIPE_V>();
      }

      CrossCoreWaitFlag<0x2, PIPE_MTE2>(4U + slot);

      Duplicate(acc1, 0.0f, C * d);
      Duplicate(acc2, 0.0f, C * d);
      Duplicate(work, 0.0f, C * d);
      PipeBarrier<PIPE_V>();

      // 2A. Rescale & accumulate AIC off-diag outputs
      __gm__ bfloat16_t *tQbkBase  = reinterpret_cast<__gm__ bfloat16_t *>(laneBase + BWDR_OFF_TQBK);
      __gm__ bfloat16_t *tKnegBase = reinterpret_cast<__gm__ bfloat16_t *>(laneBase + BWDR_OFF_TKNEG);

      for (uint32_t a = 1U; a <= 3U; ++a) {
        const uint32_t rA = a * 16U;
        const uint32_t cols = rA;
        const uint32_t kRowOff = kRefRowOffs[a - 1U];
        const uint32_t qRowOff = (a - 1U) * 32U;

        Adds(gRef, gCum[rA * d], 0.0f, d);
        PipeBarrier<PIPE_V>();

        GlobalTensor<bfloat16_t> tqGm, tkGm;
        tqGm.SetGlobalBuffer(tQbkBase + qRowOff * d, 32U * d);
        tkGm.SetGlobalBuffer(tKnegBase + kRowOff * d, cols * d);
        DataCopy(bf, tqGm, 32U * d);
        PipeBarrier<PIPE_ALL>();

        BinaryRepeatParams subRow{1, 1, 1, 16, 16, 0};
        Sub(acc1[rA * d], gCum[rA * d], gRef, 64, 16, subRow);
        Sub(acc1[rA * d + 64], gCum[rA * d + 64], gRef[64], 64, 16, subRow);
        PipeBarrier<PIPE_V>();
        Mins(acc1[rA * d], acc1[rA * d], 0.0f, 16U * d);
        PipeBarrier<PIPE_V>();
        Maxs(acc1[rA * d], acc1[rA * d], -80.0f, 16U * d);
        PipeBarrier<PIPE_V>();
        Exp(acc1[rA * d], acc1[rA * d], 16U * d);
        PipeBarrier<PIPE_V>();

        Cast(acc2[rA * d], bf[16U * d], RoundMode::CAST_NONE, 16U * d);
        PipeBarrier<PIPE_V>();
        Mul(acc2[rA * d], acc2[rA * d], acc1[rA * d], 16U * d);
        Cast(work[48U * d], bf[0], RoundMode::CAST_NONE, 16U * d);
        PipeBarrier<PIPE_V>();
        Mul(acc1[rA * d], acc1[rA * d], work[48U * d], 16U * d);
        PipeBarrier<PIPE_V>();
        Muls(acc1[rA * d], acc1[rA * d], scale, 16U * d);
        PipeBarrier<PIPE_ALL>();

        DataCopy(bf, tkGm, cols * d);
        PipeBarrier<PIPE_ALL>();
        for (uint32_t b0 = 0; b0 < cols; b0 += 16U) {
          BinaryRepeatParams subRef{1, 1, 1, 16, 0, 16};
          Sub(acc1[0], gRef, gCum[b0 * d], 64, 16, subRef);
          Sub(acc1[64], gRef[64], gCum[b0 * d + 64], 64, 16, subRef);
          PipeBarrier<PIPE_V>();
          Mins(acc1[0], acc1[0], 0.0f, 16U * d);
          PipeBarrier<PIPE_V>();
          Maxs(acc1[0], acc1[0], -80.0f, 16U * d);
          PipeBarrier<PIPE_V>();
          Exp(acc1[0], acc1[0], 16U * d);
          PipeBarrier<PIPE_V>();
          Cast(work[48U * d], bf[b0 * d], RoundMode::CAST_NONE, 16U * d);
          PipeBarrier<PIPE_V>();
          Mul(work[48U * d], work[48U * d], acc1[0], 16U * d);
          PipeBarrier<PIPE_V>();
          Add(work[b0 * d], work[b0 * d], work[48U * d], 16U * d);
          PipeBarrier<PIPE_V>();
        }
        PipeBarrier<PIPE_ALL>();
      }
      Duplicate(acc1[0], 0.0f, 16U * d);
      Duplicate(work[48U * d], 0.0f, 16U * d);
      PipeBarrier<PIPE_V>();

      // 2B. 4x(16x16) Diagonal-Block Intra_bwd:
      //   - Fast Cube Path (lower_bound >= -5.5): just 4x(16x128) vector rescales of AIC outputs!
      //   - Fallback AIV Path (unbounded g_t): batched FP32 vector Diag-16 (zero overflow!)
      if (useCubeDiag) {
        BinaryRepeatParams subRow{1, 1, 1, 16, 16, 0};
        BinaryRepeatParams subRef{1, 1, 1, 16, 0, 16};
        for (uint32_t a = 0; a < 4U; ++a) {
          const uint32_t base = a * 16U;
          const uint32_t rMid = base + 8U;
          const uint32_t kRowOff = 96U + a * 16U;
          const uint32_t qRowOff = 96U + a * 32U;

          Adds(gRef, gCum[rMid * d], 0.0f, d);
          PipeBarrier<PIPE_ALL>();

          GlobalTensor<bfloat16_t> tqGm, tkGm;
          tqGm.SetGlobalBuffer(tQbkBase + qRowOff * d, 32U * d);
          tkGm.SetGlobalBuffer(tKnegBase + kRowOff * d, 16U * d);
          DataCopy(bf, tqGm, 32U * d);
          PipeBarrier<PIPE_ALL>();

          // exp(G[i] - G[rMid]) -> tmpFull[0..16*d)
          Sub(tmpFull[0], gCum[base * d], gRef, 64, 16, subRow);
          Sub(tmpFull[64], gCum[base * d + 64U], gRef[64], 64, 16, subRow);
          PipeBarrier<PIPE_V>();
          Mins(tmpFull[0], tmpFull[0], 45.0f, 16U * d);
          PipeBarrier<PIPE_V>();
          Maxs(tmpFull[0], tmpFull[0], -80.0f, 16U * d);
          PipeBarrier<PIPE_V>();
          Exp(tmpFull[0], tmpFull[0], 16U * d);
          PipeBarrier<PIPE_V>();

          // acc1[base..base+16] += scale * exp(G[i] - G[rMid]) * tQbkDiag[0..16]
          Cast(tmpFull[16U * d], bf[0], RoundMode::CAST_NONE, 16U * d);
          PipeBarrier<PIPE_V>();
          Mul(tmpFull[16U * d], tmpFull[16U * d], tmpFull[0], 16U * d);
          PipeBarrier<PIPE_V>();
          Axpy(acc1[base * d], tmpFull[16U * d], scale, 16U * d);
          PipeBarrier<PIPE_V>();

          // acc2[base..base+16] += exp(G[i] - G[rMid]) * tQbkDiag[16..32]
          Cast(tmpFull[16U * d], bf[16U * d], RoundMode::CAST_NONE, 16U * d);
          PipeBarrier<PIPE_V>();
          Mul(tmpFull[16U * d], tmpFull[16U * d], tmpFull[0], 16U * d);
          PipeBarrier<PIPE_V>();
          Add(acc2[base * d], acc2[base * d], tmpFull[16U * d], 16U * d);
          PipeBarrier<PIPE_ALL>();

          // work[base..base+16] += exp(G[rMid] - G[j]) * tKnegDiag[0..16]
          DataCopy(bf, tkGm, 16U * d);
          PipeBarrier<PIPE_ALL>();
          Sub(tmpFull[0], gRef, gCum[base * d], 64, 16, subRef);
          Sub(tmpFull[64], gRef[64], gCum[base * d + 64U], 64, 16, subRef);
          PipeBarrier<PIPE_V>();
          Mins(tmpFull[0], tmpFull[0], 45.0f, 16U * d);
          PipeBarrier<PIPE_V>();
          Maxs(tmpFull[0], tmpFull[0], -80.0f, 16U * d);
          PipeBarrier<PIPE_V>();
          Exp(tmpFull[0], tmpFull[0], 16U * d);
          PipeBarrier<PIPE_V>();
          Cast(tmpFull[16U * d], bf[0], RoundMode::CAST_NONE, 16U * d);
          PipeBarrier<PIPE_V>();
          Mul(tmpFull[16U * d], tmpFull[16U * d], tmpFull[0], 16U * d);
          PipeBarrier<PIPE_V>();
          Add(work[base * d], work[base * d], tmpFull[16U * d], 16U * d);
          PipeBarrier<PIPE_ALL>();
        }
        // Add exact FP32 i==j diagonal AND i==j+1 sub-diagonal (reusing kDGm/qDGm in bf!)
        {
          GlobalTensor<bfloat16_t> qDGm, kDGm;
          qDGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(bc + cacheL.qNormOff) +
                                   (static_cast<uint64_t>(h) * S + tStart) * d, C * d);
          kDGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(bc + cacheL.kNormOff) +
                                   (static_cast<uint64_t>(h) * S + tStart) * d, C * d);
          BinaryRepeatParams diagS{1, 1, 0, 16, 16, 1};

          // --- Pass K: kNorm in bf ---
          Brcb(scalBrcb, rowTmp[0], static_cast<uint8_t>(C / 8U), {1, 8});
          DataCopy(bf, kDGm, C * d);
          PipeBarrier<PIPE_ALL>();
          // i == j: acc1 += dAqkDiag * kNorm
          Cast(tmpFull, bf, RoundMode::CAST_NONE, C * d);
          PipeBarrier<PIPE_V>();
          Mul(tmpFull, tmpFull, scalBrcb, 64, C, diagS);
          Mul(tmpFull[64], tmpFull[64], scalBrcb, 64, C, diagS);
          PipeBarrier<PIPE_V>();
          Add(acc1, acc1, tmpFull, C * d);
          PipeBarrier<PIPE_V>();

          // i == j + 1 (2 halves of 31 pairs: j=0..30 and j=32..62; boundary j=31 has weight 0)
          for (uint32_t h0 = 0; h0 < 64U; h0 += 32U) {
            Sub(tmpFull[0], gCum[(h0 + 1U) * d], gCum[h0 * d], 31U * d);
            PipeBarrier<PIPE_V>();
            Mins(tmpFull[0], tmpFull[0], 0.0f, 31U * d);
            PipeBarrier<PIPE_V>();
            Maxs(tmpFull[0], tmpFull[0], -80.0f, 31U * d);
            PipeBarrier<PIPE_V>();
            Exp(tmpFull[0], tmpFull[0], 31U * d);
            PipeBarrier<PIPE_V>();

            // tmpFull[32*d..] = k[j] * eSub -> acc1[j+1] += subQk * (k[j]*eSub); acc2[j+1] += subKk * (k[j]*eSub)
            Cast(tmpFull[32U * d], bf[h0 * d], RoundMode::CAST_NONE, 31U * d);
            PipeBarrier<PIPE_V>();
            Mul(tmpFull[32U * d], tmpFull[32U * d], tmpFull[0], 31U * d);
            PipeBarrier<PIPE_V>();

            Brcb(scalBrcb, rowTmp[64U + h0], 4, {1, 8});
            PipeBarrier<PIPE_V>();
            // Use tmpFull[0] temporarily? No, keep eSub in tmpFull[0] for k[j+1]! Multiply in place?
            // Wait: save k[j]*eSub in tmpFull[32*d], scale into acc1 via extra? We have 2 halves,
            // so do subKk*beta * k[j+1] * eSub first using tmpFull[0], OR compute k[j]*eSub twice (cheap!).
            // Better: do work[j] += k[j+1]*eSub*subKkBeta AFTER acc1/acc2 by overwriting tmpFull[0] last!
            // Actually: for acc1 & acc2 we need tmpFull[32*d] twice; just re-cast k[j] (31*128 = 16 cycles!).
            Mul(tmpFull[32U * d], tmpFull[32U * d], scalBrcb, 64, 31, diagS);
            Mul(tmpFull[32U * d + 64U], tmpFull[32U * d + 64U], scalBrcb, 64, 31, diagS);
            PipeBarrier<PIPE_V>();
            Add(acc1[(h0 + 1U) * d], acc1[(h0 + 1U) * d], tmpFull[32U * d], 31U * d);
            PipeBarrier<PIPE_V>();

            Cast(tmpFull[32U * d], bf[h0 * d], RoundMode::CAST_NONE, 31U * d);
            PipeBarrier<PIPE_V>();
            Mul(tmpFull[32U * d], tmpFull[32U * d], tmpFull[0], 31U * d);
            PipeBarrier<PIPE_V>();
            Brcb(scalBrcb, rowTmp[128U + h0], 4, {1, 8});
            PipeBarrier<PIPE_V>();
            Mul(tmpFull[32U * d], tmpFull[32U * d], scalBrcb, 64, 31, diagS);
            Mul(tmpFull[32U * d + 64U], tmpFull[32U * d + 64U], scalBrcb, 64, 31, diagS);
            PipeBarrier<PIPE_V>();
            Add(acc2[(h0 + 1U) * d], acc2[(h0 + 1U) * d], tmpFull[32U * d], 31U * d);
            PipeBarrier<PIPE_V>();

            // work[j] += k[j+1] * eSub * (subKk * beta[j+1])
            Cast(tmpFull[32U * d], bf[(h0 + 1U) * d], RoundMode::CAST_NONE, 31U * d);
            PipeBarrier<PIPE_V>();
            Mul(tmpFull[32U * d], tmpFull[32U * d], tmpFull[0], 31U * d);
            PipeBarrier<PIPE_V>();
            Brcb(scalBrcb, rowTmp[192U + h0], 4, {1, 8});
            PipeBarrier<PIPE_V>();
            Mul(tmpFull[32U * d], tmpFull[32U * d], scalBrcb, 64, 31, diagS);
            Mul(tmpFull[32U * d + 64U], tmpFull[32U * d + 64U], scalBrcb, 64, 31, diagS);
            PipeBarrier<PIPE_V>();
            Add(work[h0 * d], work[h0 * d], tmpFull[32U * d], 31U * d);
            PipeBarrier<PIPE_V>();
          }
          PipeBarrier<PIPE_ALL>();

          // --- Pass Q: qNorm in bf ---
          Brcb(scalBrcb, rowTmp[0], static_cast<uint8_t>(C / 8U), {1, 8});
          DataCopy(bf, qDGm, C * d);
          PipeBarrier<PIPE_ALL>();
          // i == j: work += dAqkDiag * qNorm
          Cast(tmpFull, bf, RoundMode::CAST_NONE, C * d);
          PipeBarrier<PIPE_V>();
          Mul(tmpFull, tmpFull, scalBrcb, 64, C, diagS);
          Mul(tmpFull[64], tmpFull[64], scalBrcb, 64, C, diagS);
          PipeBarrier<PIPE_V>();
          Add(work, work, tmpFull, C * d);
          PipeBarrier<PIPE_V>();

          // i == j + 1: work[j] += q[j+1] * eSub * subQk
          for (uint32_t h0 = 0; h0 < 64U; h0 += 32U) {
            Sub(tmpFull[0], gCum[(h0 + 1U) * d], gCum[h0 * d], 31U * d);
            PipeBarrier<PIPE_V>();
            Mins(tmpFull[0], tmpFull[0], 0.0f, 31U * d);
            PipeBarrier<PIPE_V>();
            Maxs(tmpFull[0], tmpFull[0], -80.0f, 31U * d);
            PipeBarrier<PIPE_V>();
            Exp(tmpFull[0], tmpFull[0], 31U * d);
            PipeBarrier<PIPE_V>();
            Cast(tmpFull[32U * d], bf[(h0 + 1U) * d], RoundMode::CAST_NONE, 31U * d);
            PipeBarrier<PIPE_V>();
            Mul(tmpFull[32U * d], tmpFull[32U * d], tmpFull[0], 31U * d);
            PipeBarrier<PIPE_V>();
            Brcb(scalBrcb, rowTmp[64U + h0], 4, {1, 8});
            PipeBarrier<PIPE_V>();
            Mul(tmpFull[32U * d], tmpFull[32U * d], scalBrcb, 64, 31, diagS);
            Mul(tmpFull[32U * d + 64U], tmpFull[32U * d + 64U], scalBrcb, 64, 31, diagS);
            PipeBarrier<PIPE_V>();
            Add(work[h0 * d], work[h0 * d], tmpFull[32U * d], 31U * d);
            PipeBarrier<PIPE_V>();
          }
          PipeBarrier<PIPE_ALL>();
        }
      } else {
        LocalTensor<float> dAqkDiag = buf.GetWithOffset<float>(1024U, uDAqk);
        LocalTensor<float> dAkkDiag = buf.GetWithOffset<float>(1024U, uDAqk + 4096U);
        LocalTensor<float> q16Fp    = buf.GetWithOffset<float>(16U * d, uDAqk + 8192U);
        LocalTensor<float> k16Fp    = buf.GetWithOffset<float>(16U * d, uDAqk + 16384U);
        LocalTensor<float> exp16Fp  = buf.GetWithOffset<float>(16U * d, uDAqk + 24576U);
        LocalTensor<float> prod16A  = buf.GetWithOffset<float>(16U * d, uBf);
        LocalTensor<float> prod16B  = buf.GetWithOffset<float>(16U * d, uBf + 8192U);
        BinaryRepeatParams rowBrcbP{1, 1, 1, 16, 16, 0};
        BinaryRepeatParams scalP{1, 1, 0, 16, 16, 1};

        for (uint32_t a = 0; a < 4U; ++a) {
          const uint32_t base = a * 16U;
          GlobalTensor<bfloat16_t> qBlkGm, kBlkGm;
          qBlkGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(bc + cacheL.qNormOff) +
                                     (static_cast<uint64_t>(h) * S + tStart + base) * d, 16U * d);
          kBlkGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(bc + cacheL.kNormOff) +
                                     (static_cast<uint64_t>(h) * S + tStart + base) * d, 16U * d);
          PipeBarrier<PIPE_ALL>();
          DataCopy(bf[0], qBlkGm, 16U * d);
          DataCopy(bf[16U * d], kBlkGm, 16U * d);
          PipeBarrier<PIPE_ALL>();
          Cast(q16Fp, bf[0], RoundMode::CAST_NONE, 16U * d);
          Cast(k16Fp, bf[16U * d], RoundMode::CAST_NONE, 16U * d);
          PipeBarrier<PIPE_V>();

          for (uint32_t jj = 0; jj < 16U; ++jj) {
            const uint32_t jRow = base + jj;
            const uint32_t cnt = 16U - jj;
            const uint32_t off = jj * d;
            const uint32_t colOff = (a * 16U + jj) * 16U;

            // exp16Fp[jj..15] = exp(clamp(G[jRow..base+15] - G[jRow], -80, 0))
            Sub(exp16Fp[off], gCum[jRow * d], gCum[jRow * d], 64, static_cast<uint8_t>(cnt), rowBrcbP);
            Sub(exp16Fp[off + 64U], gCum[jRow * d + 64U], gCum[jRow * d + 64U], 64, static_cast<uint8_t>(cnt), rowBrcbP);
            PipeBarrier<PIPE_V>();
            Mins(exp16Fp[off], exp16Fp[off], 0.0f, cnt * d);
            PipeBarrier<PIPE_V>();
            Maxs(exp16Fp[off], exp16Fp[off], -80.0f, cnt * d);
            PipeBarrier<PIPE_V>();
            Exp(exp16Fp[off], exp16Fp[off], cnt * d);
            PipeBarrier<PIPE_V>();

            // prod16A[jj..15] = exp16Fp[jj..15] * k16Fp[jj]
            Mul(prod16A[off], exp16Fp[off], k16Fp[jj * d], 64, static_cast<uint8_t>(cnt), rowBrcbP);
            Mul(prod16A[off + 64U], exp16Fp[off + 64U], k16Fp[jj * d + 64U], 64, static_cast<uint8_t>(cnt), rowBrcbP);
            PipeBarrier<PIPE_V>();

            // Broadcast wQk[0..15] -> scalBrcb
            Brcb(scalBrcb, dAqkDiag[colOff], 2, {1, 8});
            PipeBarrier<PIPE_V>();
            // acc1[jRow..base+15] += prod16A * wQk
            Mul(prod16B[off], prod16A[off], scalBrcb[jj * 8U], 64, static_cast<uint8_t>(cnt), scalP);
            Mul(prod16B[off + 64U], prod16A[off + 64U], scalBrcb[jj * 8U], 64, static_cast<uint8_t>(cnt), scalP);
            PipeBarrier<PIPE_V>();
            Add(acc1[jRow * d], acc1[jRow * d], prod16B[off], cnt * d);
            PipeBarrier<PIPE_V>();

            // prod16B[jj..15] = q16Fp[jj..15] * wQk (for dk_neg_intra)
            Mul(prod16B[off], q16Fp[off], scalBrcb[jj * 8U], 64, static_cast<uint8_t>(cnt), scalP);
            Mul(prod16B[off + 64U], q16Fp[off + 64U], scalBrcb[jj * 8U], 64, static_cast<uint8_t>(cnt), scalP);
            PipeBarrier<PIPE_V>();

            if (cnt > 1U) {
              // Broadcast wKk[0..15] -> scalBrcb (entry jj is 0.0f)
              Brcb(scalBrcb, dAkkDiag[colOff], 2, {1, 8});
              PipeBarrier<PIPE_V>();
              // acc2[jRow..base+15] += prod16A * wKk
              Mul(prod16A[off], prod16A[off], scalBrcb[jj * 8U], 64, static_cast<uint8_t>(cnt), scalP);
              Mul(prod16A[off + 64U], prod16A[off + 64U], scalBrcb[jj * 8U], 64, static_cast<uint8_t>(cnt), scalP);
              PipeBarrier<PIPE_V>();
              Add(acc2[jRow * d], acc2[jRow * d], prod16A[off], cnt * d);
              PipeBarrier<PIPE_V>();

              // prod16A = k16Fp * wKk * beta -> add into prod16B
              Mul(prod16A[off], k16Fp[off], scalBrcb[jj * 8U], 64, static_cast<uint8_t>(cnt), scalP);
              Mul(prod16A[off + 64U], k16Fp[off + 64U], scalBrcb[jj * 8U], 64, static_cast<uint8_t>(cnt), scalP);
              PipeBarrier<PIPE_V>();
              Mul(prod16A[off], prod16A[off], betaBrcb[jRow * 8U], 64, static_cast<uint8_t>(cnt), scalP);
              Mul(prod16A[off + 64U], prod16A[off + 64U], betaBrcb[jRow * 8U], 64, static_cast<uint8_t>(cnt), scalP);
              PipeBarrier<PIPE_V>();
              Add(prod16B[off], prod16B[off], prod16A[off], cnt * d);
              PipeBarrier<PIPE_V>();
            }

            // work[jRow] += sum_{r=0..cnt-1} prod16B[jj+r] * exp16Fp[jj+r]
            Mul(prod16B[off], prod16B[off], exp16Fp[off], cnt * d);
            PipeBarrier<PIPE_V>();
            uint32_t rem = cnt;
            while (rem > 1U) {
              const uint32_t half = rem >> 1;
              const uint32_t odd = rem & 1U;
              Add(prod16B[off], prod16B[off], prod16B[off + (half + odd) * d], half * d);
              PipeBarrier<PIPE_V>();
              rem = half + odd;
            }
            Add(work[jRow * d], work[jRow * d], prod16B[off], d);
            PipeBarrier<PIPE_V>();
          }
        }
        PipeBarrier<PIPE_ALL>();
      }

      // 2C. Combine inter-chunk terms, compute d_q/k/v_silu, d_beta_in, dG, reverse cumsum, kda_gate_bwd
      {
        Mins(tmpFull, gCum, 0.0f, C * d);
        PipeBarrier<PIPE_V>();
        Maxs(tmpFull, tmpFull, -80.0f, C * d);
        PipeBarrier<PIPE_V>();
        Exp(tmpFull, tmpFull, C * d);
        PipeBarrier<PIPE_V>();

        GlobalTensor<float> stashDkNegGm, dGGm;
        stashDkNegGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(laneBase + BWDR_OFF_TQBK), C * d);
        dGGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(laneBase + BWDR_OFF_KREF), C * d);
        PipeBarrier<PIPE_ALL>();
        DataCopy(stashDkNegGm, work, C * d);
        PipeBarrier<PIPE_ALL>();

        // (i) dq_tot = dq_intra + scale * exp(G) * dqg -> acc1
        GlobalTensor<bfloat16_t> dqgGm;
        dqgGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + t->wsDqgOff) + hcIdx * C * d, C * d);
        DataCopy(bf, dqgGm, C * d);
        PipeBarrier<PIPE_ALL>();
        Cast(work, bf, RoundMode::CAST_NONE, C * d);
        PipeBarrier<PIPE_V>();
        Mul(work, work, tmpFull, C * d);
        PipeBarrier<PIPE_V>();
        Axpy(acc1, work, scale, C * d);
        PipeBarrier<PIPE_ALL>();

        // (ii) Q L2Norm_bwd + stash dG_q = qNorm * dq_tot -> dGGm
        GlobalTensor<bfloat16_t> qGm;
        GlobalTensor<float> qrGm;
        qGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(bc + cacheL.qNormOff) +
                                (static_cast<uint64_t>(h) * S + tStart) * d, C * d);
        qrGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(bc + cacheL.qRstdOff) +
                                 static_cast<uint64_t>(h) * S + tStart, C);
        DataCopy(bf, qGm, C * d);
        DataCopy(rstdFp, qrGm, C);
        PipeBarrier<PIPE_ALL>();
        Cast(work, bf, RoundMode::CAST_NONE, C * d);
        PipeBarrier<PIPE_V>();
        Mul(work, work, acc1, C * d);
        PipeBarrier<PIPE_ALL>();
        DataCopy(dGGm, work, C * d);
        PipeBarrier<PIPE_ALL>();
        BinaryRepeatParams foldP{1, 1, 1, 16, 16, 16};
        Add(work, work, work[64], 64, C, foldP);
        PipeBarrier<PIPE_V>();
        WholeReduceSum(scalVec, work, 64, static_cast<int32_t>(C), 1, 1, 16);
        PipeBarrier<PIPE_V>();
        Brcb(scalBrcb, scalVec, static_cast<uint8_t>(C / 8U), {1, 8});
        PipeBarrier<PIPE_V>();
        Cast(work, bf, RoundMode::CAST_NONE, C * d);
        PipeBarrier<PIPE_V>();
        BinaryRepeatParams sMul{1, 1, 0, 16, 16, 1};
        Mul(work, work, scalBrcb, 64, C, sMul);
        Mul(work[64], work[64], scalBrcb, 64, C, sMul);
        PipeBarrier<PIPE_V>();
        Sub(acc1, acc1, work, C * d);
        PipeBarrier<PIPE_V>();
        Brcb(scalBrcb, rstdFp, static_cast<uint8_t>(C / 8U), {1, 8});
        PipeBarrier<PIPE_V>();
        Mul(acc1, acc1, scalBrcb, 64, C, sMul);
        Mul(acc1[64], acc1[64], scalBrcb, 64, C, sMul);
        PipeBarrier<PIPE_V>();
        Cast(bf, acc1, RoundMode::CAST_RINT, C * d);
        PipeBarrier<PIPE_ALL>();
        {
          GlobalTensor<bfloat16_t> dqSiluGm;
          dqSiluGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + t->wsDSiluOff) +
                                       tStart * D + static_cast<uint64_t>(h) * d,
                                   static_cast<uint64_t>(C) * D);
          DataCopyParams stP{static_cast<uint16_t>(C), dBlockLen, 0, dRowStride};
          DataCopy(dqSiluGm, bf, stP);
          PipeBarrier<PIPE_ALL>();
        }

        // (iii) dbk_intra + exp(G) * dkbg -> acc2; db_k -> dbTot; dk_pos = beta * acc2 -> acc2
        GlobalTensor<bfloat16_t> dkbgGm;
        dkbgGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(laneBase + BWDR_OFF_DKBG), C * d);
        DataCopy(bf, dkbgGm, C * d);
        PipeBarrier<PIPE_ALL>();
        Cast(work, bf, RoundMode::CAST_NONE, C * d);
        PipeBarrier<PIPE_V>();
        Mul(work, work, tmpFull, C * d);
        PipeBarrier<PIPE_V>();
        Add(acc2, acc2, work, C * d);
        PipeBarrier<PIPE_ALL>();

        GlobalTensor<bfloat16_t> kGm;
        kGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(bc + cacheL.kNormOff) +
                                (static_cast<uint64_t>(h) * S + tStart) * d, C * d);
        DataCopy(bf, kGm, C * d);
        PipeBarrier<PIPE_ALL>();
        Cast(work, bf, RoundMode::CAST_NONE, C * d);
        PipeBarrier<PIPE_V>();
        Mul(acc1, work, acc2, C * d);
        PipeBarrier<PIPE_V>();
        Add(acc1, acc1, acc1[64], 64, C, foldP);
        PipeBarrier<PIPE_V>();
        WholeReduceSum(dbTot, acc1, 64, static_cast<int32_t>(C), 1, 1, 16);
        PipeBarrier<PIPE_V>();

        Mul(acc2, acc2, betaBrcb, 64, C, sMul);
        Mul(acc2[64], acc2[64], betaBrcb, 64, C, sMul);
        PipeBarrier<PIPE_ALL>();

        DataCopy(tmpFull, dGGm, C * d);
        PipeBarrier<PIPE_ALL>();
        Mul(acc1, work, acc2, C * d);
        PipeBarrier<PIPE_V>();
        Add(tmpFull, tmpFull, acc1, C * d);
        PipeBarrier<PIPE_V>();

        // (iv) exp(G_last - G) * dkg -> acc1; sum(kg*dkg) + expGLast*dgamma -> tmpFull[63]
        Adds(gRef, gCum[(C - 1U) * d], 0.0f, d);
        PipeBarrier<PIPE_V>();
        BinaryRepeatParams subLast{1, 1, 1, 16, 0, 16};
        Sub(acc1, gRef, gCum, 64, C, subLast);
        Sub(acc1[64], gRef[64], gCum[64], 64, C, subLast);
        PipeBarrier<PIPE_V>();
        Mins(acc1, acc1, 0.0f, C * d);
        PipeBarrier<PIPE_V>();
        Maxs(acc1, acc1, -80.0f, C * d);
        PipeBarrier<PIPE_V>();
        Exp(acc1, acc1, C * d);
        PipeBarrier<PIPE_ALL>();

        GlobalTensor<bfloat16_t> dkgGm;
        dkgGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + t->wsDkgOff) + hcIdx * C * d, C * d);
        DataCopy(bf, dkgGm, C * d);
        PipeBarrier<PIPE_ALL>();
        Cast(work, bf, RoundMode::CAST_NONE, C * d);
        PipeBarrier<PIPE_V>();
        Mul(acc1, acc1, work, C * d);
        PipeBarrier<PIPE_ALL>();

        DataCopy(bf, kGm, C * d);
        DataCopy(gCum, stashDkNegGm, C * d);
        PipeBarrier<PIPE_ALL>();
        Cast(work, bf, RoundMode::CAST_NONE, C * d);
        PipeBarrier<PIPE_V>();

        Mul(work, work, acc1, C * d);
        PipeBarrier<PIPE_V>();
        for (uint32_t st = 32U; st >= 1U; st >>= 1) {
          Add(work, work, work[st * d], st * d);
          PipeBarrier<PIPE_V>();
        }
        PipeBarrier<PIPE_ALL>();
        {
          GlobalTensor<float> egGm, dgamGm;
          egGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(bc + cacheL.expGLastOff) + hcIdx * d, d);
          dgamGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ws + t->wsDgammaOff) + hcIdx * d, d);
          DataCopy(gRef, egGm, d);
          DataCopy(dgGamma, dgamGm, d);
          PipeBarrier<PIPE_ALL>();
          Mul(dgGamma, dgGamma, gRef, d);
          PipeBarrier<PIPE_V>();
          Add(dgGamma, dgGamma, work[0], d);
          PipeBarrier<PIPE_V>();
          Add(tmpFull[(C - 1U) * d], tmpFull[(C - 1U) * d], dgGamma, d);
          PipeBarrier<PIPE_V>();
        }

        Add(gCum, gCum, acc1, C * d);
        PipeBarrier<PIPE_V>();
        Cast(work, bf, RoundMode::CAST_NONE, C * d);
        PipeBarrier<PIPE_V>();
        Mul(acc1, work, gCum, C * d);
        PipeBarrier<PIPE_V>();
        Sub(tmpFull, tmpFull, acc1, C * d);
        PipeBarrier<PIPE_V>();
        Add(acc2, acc2, gCum, C * d);
        PipeBarrier<PIPE_ALL>();

        // (v) K L2Norm_bwd -> write dk_silu to wsDSiluOff + S*D
        GlobalTensor<float> krGm;
        krGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(bc + cacheL.kRstdOff) +
                                 static_cast<uint64_t>(h) * S + tStart, C);
        DataCopy(rstdFp, krGm, C);
        PipeBarrier<PIPE_ALL>();
        Mul(acc1, work, acc2, C * d);
        PipeBarrier<PIPE_V>();
        Add(acc1, acc1, acc1[64], 64, C, foldP);
        PipeBarrier<PIPE_V>();
        WholeReduceSum(scalVec, acc1, 64, static_cast<int32_t>(C), 1, 1, 16);
        PipeBarrier<PIPE_V>();
        Brcb(scalBrcb, scalVec, static_cast<uint8_t>(C / 8U), {1, 8});
        PipeBarrier<PIPE_V>();
        Mul(work, work, scalBrcb, 64, C, sMul);
        Mul(work[64], work[64], scalBrcb, 64, C, sMul);
        PipeBarrier<PIPE_V>();
        Sub(acc2, acc2, work, C * d);
        PipeBarrier<PIPE_V>();
        Brcb(scalBrcb, rstdFp, static_cast<uint8_t>(C / 8U), {1, 8});
        PipeBarrier<PIPE_V>();
        Mul(acc2, acc2, scalBrcb, 64, C, sMul);
        Mul(acc2[64], acc2[64], scalBrcb, 64, C, sMul);
        PipeBarrier<PIPE_V>();
        Cast(bf, acc2, RoundMode::CAST_RINT, C * d);
        PipeBarrier<PIPE_ALL>();
        {
          GlobalTensor<bfloat16_t> dkSiluGm;
          dkSiluGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + t->wsDSiluOff) +
                                       (static_cast<uint64_t>(S) + tStart) * D + static_cast<uint64_t>(h) * d,
                                   static_cast<uint64_t>(C) * D);
          DataCopyParams stP{static_cast<uint16_t>(C), dBlockLen, 0, dRowStride};
          DataCopy(dkSiluGm, bf, stP);
          PipeBarrier<PIPE_ALL>();
        }

        // (vi) dv_tot = beta * dvbeta -> wsDSiluOff + 2*S*D; d_beta_in -> head-major wsDOOff
        {
          GlobalTensor<bfloat16_t> dvbGm, vGm;
          dvbGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(laneBase + BWDR_OFF_DVBETA), C * d);
          vGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(bc + cacheL.vConvOff) +
                                  (static_cast<uint64_t>(h) * S + tStart) * d, C * d);
          DataCopy(bf, dvbGm, C * d);
          PipeBarrier<PIPE_ALL>();
          Cast(acc1, bf, RoundMode::CAST_NONE, C * d);
          PipeBarrier<PIPE_ALL>();
          DataCopy(bf, vGm, C * d);
          PipeBarrier<PIPE_ALL>();
          Cast(work, bf, RoundMode::CAST_NONE, C * d);
          PipeBarrier<PIPE_V>();
          Mul(acc2, work, acc1, C * d);
          PipeBarrier<PIPE_V>();
          Add(acc2, acc2, acc2[64], 64, C, foldP);
          PipeBarrier<PIPE_V>();
          WholeReduceSum(scalVec, acc2, 64, static_cast<int32_t>(C), 1, 1, 16);
          PipeBarrier<PIPE_V>();
          Add(dbTot, dbTot, scalVec, C);
          PipeBarrier<PIPE_V>();

          Muls(scalVec, betaFp, -1.0f, C);
          PipeBarrier<PIPE_V>();
          Adds(scalVec, scalVec, 1.0f, C);
          PipeBarrier<PIPE_V>();
          Mul(scalVec, scalVec, betaFp, C);
          PipeBarrier<PIPE_V>();
          Mul(dbTot, dbTot, scalVec, C);
          PipeBarrier<PIPE_V>();

          GlobalTensor<float> dbHeadGm;
          dbHeadGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ws + t->wsDOOff) +
                                       static_cast<uint64_t>(h) * S + tStart, C);
          PipeBarrier<PIPE_ALL>();
          DataCopy(dbHeadGm, dbTot, C);
          PipeBarrier<PIPE_ALL>();

          Mul(acc1, acc1, betaBrcb, 64, C, sMul);
          Mul(acc1[64], acc1[64], betaBrcb, 64, C, sMul);
          PipeBarrier<PIPE_V>();
          Cast(bf, acc1, RoundMode::CAST_RINT, C * d);
          PipeBarrier<PIPE_ALL>();
          GlobalTensor<bfloat16_t> dvSiluGm;
          dvSiluGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + t->wsDSiluOff) +
                                       (2ULL * S + tStart) * D + static_cast<uint64_t>(h) * d,
                                   static_cast<uint64_t>(C) * D);
          DataCopyParams stP{static_cast<uint16_t>(C), dBlockLen, 0, dRowStride};
          DataCopy(dvSiluGm, bf, stP);
          PipeBarrier<PIPE_ALL>();
        }

        // (vii) 6-step parallel REVERSE inclusive cumsum on tmpFull (dG -> dg_step)
        Adds(acc1[63 * d], tmpFull[63 * d], 0.0f, 1 * d);
        Add(acc1[0], tmpFull[0], tmpFull[1 * d], 63 * d);
        PipeBarrier<PIPE_V>();
        Adds(tmpFull[62 * d], acc1[62 * d], 0.0f, 2 * d);
        Add(tmpFull[0], acc1[0], acc1[2 * d], 62 * d);
        PipeBarrier<PIPE_V>();
        Adds(acc1[60 * d], tmpFull[60 * d], 0.0f, 4 * d);
        Add(acc1[0], tmpFull[0], tmpFull[4 * d], 60 * d);
        PipeBarrier<PIPE_V>();
        Adds(tmpFull[56 * d], acc1[56 * d], 0.0f, 8 * d);
        Add(tmpFull[0], acc1[0], acc1[8 * d], 56 * d);
        PipeBarrier<PIPE_V>();
        Adds(acc1[48 * d], tmpFull[48 * d], 0.0f, 16 * d);
        Add(acc1[0], tmpFull[0], tmpFull[16 * d], 48 * d);
        PipeBarrier<PIPE_V>();
        Adds(tmpFull[32 * d], acc1[32 * d], 0.0f, 32 * d);
        Add(tmpFull[0], acc1[0], acc1[32 * d], 32 * d);
        PipeBarrier<PIPE_ALL>();

        // (viii) kda_gate_bwd -> d_g_in + partial d_a_log, d_dt_bias
        const float A = aLogAll.GetValue(h);
        GlobalTensor<bfloat16_t> gInGm, dgOutGm;
        gInGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(io[B_G_IN]) +
                                  tStart * D + static_cast<uint64_t>(h) * d,
                              static_cast<uint64_t>(C) * D);
        dgOutGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(io[B_DG_IN]) +
                                    tStart * D + static_cast<uint64_t>(h) * d,
                                static_cast<uint64_t>(C) * D);
        DataCopyParams inPG{static_cast<uint16_t>(C), dBlockLen, dRowStride, 0};
        DataCopy(bf, gInGm, inPG);
        PipeBarrier<PIPE_ALL>();
        Cast(acc1, bf, RoundMode::CAST_NONE, C * d);
        PipeBarrier<PIPE_V>();
        BinaryRepeatParams bAdd{1, 1, 1, 16, 16, 0};
        Add(acc1, acc1, dtBias, 64, C, bAdd);
        Add(acc1[64], acc1[64], dtBias[64], 64, C, bAdd);
        PipeBarrier<PIPE_V>();

        if (t->hasLowerBound != 0U) {
          Muls(work, acc1, -A, C * d);
          PipeBarrier<PIPE_V>();
          Mins(work, work, 80.0f, C * d);
          PipeBarrier<PIPE_V>();
          Maxs(work, work, -80.0f, C * d);
          PipeBarrier<PIPE_V>();
          Exp(work, work, C * d);
          PipeBarrier<PIPE_V>();
          Adds(work, work, 1.0f, C * d);
          PipeBarrier<PIPE_V>();
          Duplicate(acc2, 1.0f, C * d);
          PipeBarrier<PIPE_V>();
          Div(work, acc2, work, C * d);
          PipeBarrier<PIPE_V>();
          Muls(acc2, work, -1.0f, C * d);
          PipeBarrier<PIPE_V>();
          Adds(acc2, acc2, 1.0f, C * d);
          PipeBarrier<PIPE_V>();
          Mul(acc2, acc2, work, C * d);
          PipeBarrier<PIPE_V>();
          Muls(acc2, acc2, lowerBound * A, C * d);
          PipeBarrier<PIPE_V>();
          Mul(work, tmpFull, acc2, C * d);
          PipeBarrier<PIPE_V>();
          Mul(acc1, work, acc1, C * d);
          PipeBarrier<PIPE_V>();
        } else {
          Maxs(acc2, acc1, 0.0f, C * d);
          Abs(gCum, acc1, C * d);
          PipeBarrier<PIPE_V>();
          Muls(gCum, gCum, -1.0f, C * d);
          PipeBarrier<PIPE_V>();
          Exp(gCum, gCum, C * d);
          PipeBarrier<PIPE_V>();
          Adds(gCum, gCum, 1.0f, C * d);
          PipeBarrier<PIPE_V>();
          Ln(gCum, gCum, C * d);
          PipeBarrier<PIPE_V>();
          Add(gCum, acc2, gCum, C * d);
          PipeBarrier<PIPE_V>();
          Muls(gCum, gCum, -A, C * d);
          PipeBarrier<PIPE_V>();

          Muls(work, acc1, -1.0f, C * d);
          PipeBarrier<PIPE_V>();
          Mins(work, work, 80.0f, C * d);
          PipeBarrier<PIPE_V>();
          Maxs(work, work, -80.0f, C * d);
          PipeBarrier<PIPE_V>();
          Exp(work, work, C * d);
          PipeBarrier<PIPE_V>();
          Adds(work, work, 1.0f, C * d);
          PipeBarrier<PIPE_V>();
          Duplicate(acc2, 1.0f, C * d);
          PipeBarrier<PIPE_V>();
          Div(work, acc2, work, C * d);
          PipeBarrier<PIPE_V>();
          Muls(work, work, -A, C * d);
          PipeBarrier<PIPE_V>();
          Mul(work, work, tmpFull, C * d);
          PipeBarrier<PIPE_V>();
          Mul(acc1, tmpFull, gCum, C * d);
          PipeBarrier<PIPE_V>();
        }

        Cast(bf, work, RoundMode::CAST_RINT, C * d);
        PipeBarrier<PIPE_ALL>();
        DataCopyParams outPG{static_cast<uint16_t>(C), dBlockLen, 0, dRowStride};
        DataCopy(dgOutGm, bf, outPG);
        PipeBarrier<PIPE_ALL>();

        for (uint32_t st = 32U; st >= 1U; st >>= 1) {
          Add(work, work, work[st * d], st * d);
          Add(acc1, acc1, acc1[st * d], st * d);
          PipeBarrier<PIPE_V>();
        }
        PipeBarrier<PIPE_ALL>();
        {
          GlobalTensor<float> pbGm;
          pbGm.SetGlobalBuffer(pDtBiasPtr + static_cast<uint64_t>(h) * d, d);
          DataCopy(gRef, pbGm, d);
          PipeBarrier<PIPE_ALL>();
          Add(gRef, gRef, work[0], d);
          PipeBarrier<PIPE_ALL>();
          DataCopy(pbGm, gRef, d);
          PipeBarrier<PIPE_ALL>();
        }
        Add(acc1[0], acc1[0], acc1[64], 64);
        PipeBarrier<PIPE_V>();
        WholeReduceSum(scalVec, acc1[0], 64, 1, 1, 1, 16);
        PipeBarrier<PIPE_ALL>();
        {
          GlobalTensor<float> paGm;
          paGm.SetGlobalBuffer(pALogPtr, static_cast<uint32_t>(aLogStrideF));
          DataCopy(rowTmp, paGm, static_cast<uint32_t>(aLogStrideF));
          PipeBarrier<PIPE_ALL>();
          rowTmp.SetValue(h, rowTmp.GetValue(h) + scalVec.GetValue(0));
          PipeBarrier<PIPE_ALL>();
          DataCopy(paGm, rowTmp, static_cast<uint32_t>(aLogStrideF));
          PipeBarrier<PIPE_ALL>();
        }
      }

      CrossCoreSetFlag<0x2, PIPE_MTE3>(6U + slot);
    }
    pipe.Destroy();
  }
}

// ---------------------------------------------------------------------------
// Phase B3 (AIV): SiLU_bwd + Reverse 3-Token Halo + ShortConv1d_bwd + Epilogue Reductions
// ---------------------------------------------------------------------------
__aicore__ inline void RunPhaseB3ConvBwdAndReduce(uint32_t wid, int32_t gen, GM_ADDR *io,
                                                  const HyperMegaKdaGradTilingData *t) {
  if ASCEND_IS_AIV {
    const uint32_t numAiv = 2U * t->numCubeCores;
    const uint32_t S = t->seqLocal;
    const uint32_t H = t->numHeads;
    const uint32_t d = t->headDim;
    const uint32_t C = t->chunkSize;
    const uint32_t numChunks = t->numChunks;
    const uint32_t D = H * d;
    const uint32_t cp = t->cpSize;
    const uint32_t rank = t->cpRank;
    const auto cacheL = MegaKdaCache::ComputeLayout(S, H, d, C);
    GM_ADDR ws = io[B_WORKSPACE];
    GM_ADDR bc = io[B_BWD_CACHE];

    const uint32_t hTile = ((H & 3U) == 0U && (H / 4U) <= numAiv) ? 4U : 1U;
    const uint32_t numGroups = H / hTile;
    const uint32_t slicesPerGroup = numAiv / numGroups;
    const uint32_t activeAivs = numGroups * slicesPerGroup;
    const uint32_t cols = hTile * d;
    constexpr uint32_t BT = 16U;

    const uint32_t inIdxs[3]    = {B_Q_IN, B_K_IN, B_V_IN};
    const uint32_t wConvIdxs[3] = {B_WCONV_Q, B_WCONV_K, B_WCONV_V};
    const uint32_t dInIdxs[3]   = {B_DQ_IN, B_DK_IN, B_DV_IN};

    // Step B3.1: Recompute 4-tap u_t on-the-fly & apply SiLU_bwd -> wsDuOff
    if (wid < activeAivs) {
      const uint32_t g4 = wid / slicesPerGroup;
      const uint32_t sIdx = wid % slicesPerGroup;
      const uint32_t cOff = g4 * cols;
      const uint32_t rowsPerSlice = (((S + slicesPerGroup - 1U) / slicesPerGroup) + 7U) & ~7U;
      const uint32_t rStart = sIdx * rowsPerSlice;
      const uint32_t rEnd = (rStart + rowsPerSlice <= S) ? (rStart + rowsPerSlice) : S;

      if (rStart < rEnd) {
        TPipe pipe;
        TBuf<TPosition::VECCALC> buf;
        const uint32_t uXPad = 0;
        const uint32_t uW4   = uXPad + (BT + 3U) * cols * 2U + 64U;
        const uint32_t uU    = uW4 + 4U * cols * 4U;
        const uint32_t uS    = uU + BT * cols * 4U;
        const uint32_t uTmp  = uS + BT * cols * 4U;
        const uint32_t uOut  = uTmp + BT * cols * 4U;
        pipe.InitBuffer(buf, uOut + BT * cols * 2U + 64U);

        LocalTensor<bfloat16_t> xPad  = buf.GetWithOffset<bfloat16_t>((BT + 3U) * cols, uXPad);
        LocalTensor<float> w4         = buf.GetWithOffset<float>(4U * cols, uW4);
        LocalTensor<float> uFp        = buf.GetWithOffset<float>(BT * cols, uU);
        LocalTensor<float> sFp        = buf.GetWithOffset<float>(BT * cols, uS);
        LocalTensor<float> tmp        = buf.GetWithOffset<float>(BT * cols, uTmp);
        LocalTensor<bfloat16_t> outBf = buf.GetWithOffset<bfloat16_t>(BT * cols, uOut);

        const uint16_t blockLen = static_cast<uint16_t>((cols * 2U) / 32U);
        const uint16_t rowStride = static_cast<uint16_t>(((D - cols) * 2U) / 32U);
        const uint8_t fpRowBlk = static_cast<uint8_t>((cols * 4U) / 32U);
        BinaryRepeatParams wMul{1, 1, 1, fpRowBlk, fpRowBlk, 0};

        for (uint32_t which = 0; which < 3U; ++which) {
          {
            GlobalTensor<bfloat16_t> wGm;
            wGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(io[wConvIdxs[which]]) +
                                    static_cast<uint64_t>(cOff) * 4U, cols * 4U);
            DataCopy(outBf, wGm, cols * 4U);
            PipeBarrier<PIPE_ALL>();
            Cast(tmp, outBf, RoundMode::CAST_NONE, cols * 4U);
            PipeBarrier<PIPE_ALL>();
            for (uint32_t ch = 0; ch < cols; ++ch) {
              for (uint32_t tau = 0; tau < 4U; ++tau) {
                w4.SetValue(tau * cols + ch, tmp.GetValue(ch * 4U + tau));
              }
            }
            PipeBarrier<PIPE_ALL>();
          }

          __gm__ bfloat16_t *xBase = reinterpret_cast<__gm__ bfloat16_t *>(io[inIdxs[which]]);
          __gm__ bfloat16_t *haloBase =
              reinterpret_cast<__gm__ bfloat16_t *>(bc + cacheL.haloQkvOff) + static_cast<uint64_t>(which) * 3ULL * D;
          __gm__ bfloat16_t *dSiluBase =
              reinterpret_cast<__gm__ bfloat16_t *>(ws + t->wsDSiluOff) + static_cast<uint64_t>(which) * S * D;
          __gm__ bfloat16_t *duBase =
              reinterpret_cast<__gm__ bfloat16_t *>(ws + t->wsDuOff) + static_cast<uint64_t>(which) * S * D;

          for (uint32_t bStart = rStart; bStart < rEnd; bStart += BT) {
            const uint32_t curT = (bStart + BT <= rEnd) ? BT : (rEnd - bStart);
            if (bStart >= 3U) {
              GlobalTensor<bfloat16_t> xGm;
              xGm.SetGlobalBuffer(xBase + static_cast<uint64_t>(bStart - 3U) * D + cOff,
                                  static_cast<uint64_t>(curT + 3U) * D);
              DataCopyParams cpP{static_cast<uint16_t>(curT + 3U), blockLen, rowStride, 0};
              DataCopy(xPad, xGm, cpP);
            } else {
              const uint32_t padZ = 3U - bStart;
              if (rank > 0U && cp > 1U) {
                GlobalTensor<bfloat16_t> hGm;
                hGm.SetGlobalBuffer(haloBase + static_cast<uint64_t>(bStart) * D + cOff,
                                    static_cast<uint64_t>(padZ) * D);
                DataCopyParams cpH{static_cast<uint16_t>(padZ), blockLen, rowStride, 0};
                DataCopy(xPad, hGm, cpH);
              } else {
                Duplicate(tmp, 0.0f, padZ * cols);
                PipeBarrier<PIPE_V>();
                Cast(xPad, tmp, RoundMode::CAST_RINT, padZ * cols);
                PipeBarrier<PIPE_ALL>();
              }
              GlobalTensor<bfloat16_t> xGm;
              xGm.SetGlobalBuffer(xBase + cOff, static_cast<uint64_t>(bStart + curT) * D);
              DataCopyParams cpX{static_cast<uint16_t>(bStart + curT), blockLen, rowStride, 0};
              DataCopy(xPad[padZ * cols], xGm, cpX);
            }
            PipeBarrier<PIPE_ALL>();

            Duplicate(uFp, 0.0f, curT * cols);
            PipeBarrier<PIPE_V>();
            for (uint32_t tau = 0; tau < 4U; ++tau) {
              Cast(tmp, xPad[tau * cols], RoundMode::CAST_NONE, curT * cols);
              PipeBarrier<PIPE_V>();
              for (uint32_t c0 = 0; c0 < cols; c0 += 64U) {
                Mul(tmp[c0], tmp[c0], w4[tau * cols + c0], 64, static_cast<uint8_t>(curT), wMul);
              }
              PipeBarrier<PIPE_V>();
              Add(uFp, uFp, tmp, curT * cols);
              PipeBarrier<PIPE_V>();
            }

            // s = sigmoid(u) -> sFp; deriv = s * (1 + u * (1 - s)) -> sFp
            Muls(sFp, uFp, -1.0f, curT * cols);
            PipeBarrier<PIPE_V>();
            Mins(sFp, sFp, 80.0f, curT * cols);
            PipeBarrier<PIPE_V>();
            Maxs(sFp, sFp, -80.0f, curT * cols);
            PipeBarrier<PIPE_V>();
            Exp(sFp, sFp, curT * cols);
            PipeBarrier<PIPE_V>();
            Adds(sFp, sFp, 1.0f, curT * cols);
            PipeBarrier<PIPE_V>();
            Duplicate(tmp, 1.0f, curT * cols);
            PipeBarrier<PIPE_V>();
            Div(sFp, tmp, sFp, curT * cols);
            PipeBarrier<PIPE_V>();

            Muls(tmp, sFp, -1.0f, curT * cols);
            PipeBarrier<PIPE_V>();
            Adds(tmp, tmp, 1.0f, curT * cols);
            PipeBarrier<PIPE_V>();
            Mul(tmp, tmp, uFp, curT * cols);
            PipeBarrier<PIPE_V>();
            Adds(tmp, tmp, 1.0f, curT * cols);
            PipeBarrier<PIPE_V>();
            Mul(sFp, sFp, tmp, curT * cols);
            PipeBarrier<PIPE_V>();

            GlobalTensor<bfloat16_t> dsGm, duGm;
            dsGm.SetGlobalBuffer(dSiluBase + static_cast<uint64_t>(bStart) * D + cOff,
                                 static_cast<uint64_t>(curT) * D);
            duGm.SetGlobalBuffer(duBase + static_cast<uint64_t>(bStart) * D + cOff,
                                 static_cast<uint64_t>(curT) * D);
            DataCopyParams inP{static_cast<uint16_t>(curT), blockLen, rowStride, 0};
            DataCopy(outBf, dsGm, inP);
            PipeBarrier<PIPE_ALL>();
            Cast(tmp, outBf, RoundMode::CAST_NONE, curT * cols);
            PipeBarrier<PIPE_V>();
            Mul(tmp, tmp, sFp, curT * cols);
            PipeBarrier<PIPE_V>();
            Cast(outBf, tmp, RoundMode::CAST_RINT, curT * cols);
            PipeBarrier<PIPE_ALL>();
            DataCopyParams outP{static_cast<uint16_t>(curT), blockLen, 0, rowStride};
            DataCopy(duGm, outBf, outP);
            PipeBarrier<PIPE_ALL>();
          }
        }
        pipe.Destroy();
      }
    }
    PipeBarrier<PIPE_ALL>();
    SyncAll();

    // Step B3.2: Reverse 3-token SHMEM Halo (rank > 0 sends du[0..2] to rank - 1)
    if (cp > 1U) {
#if !defined(__DAV_CUBE__) && !defined(__DAV_C220_CUBE__)
      if (wid == 0U) {
        const uint32_t elems3 = 3U * D;
        GM_ADDR sb = io[B_SEND_BUF];
        GM_ADDR rb = io[B_RECV_BUF];
        GM_ADDR ev = io[B_EVENT];
        if (rank > 0U) {
          TPipe hPipe;
          TBuf<TPosition::VECCALC> hBuf;
          hPipe.InitBuffer(hBuf, elems3 * 2U + 64U);
          LocalTensor<bfloat16_t> hTmp = hBuf.GetWithOffset<bfloat16_t>(elems3, 0);
          for (uint32_t w = 0; w < 3U; ++w) {
            GlobalTensor<bfloat16_t> sGm, dGm;
            sGm.SetGlobalBuffer(
                reinterpret_cast<__gm__ bfloat16_t *>(ws + t->wsDuOff) + static_cast<uint64_t>(w) * S * D, elems3);
            dGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(sb + t->sendHaloOff) + w * elems3, elems3);
            DataCopy(hTmp, sGm, elems3);
            PipeBarrier<PIPE_ALL>();
            DataCopy(dGm, hTmp, elems3);
            PipeBarrier<PIPE_ALL>();
          }
          hPipe.Destroy();
          BwdShmemPutSet(rb, sb, ev, 0, gen, static_cast<int64_t>(t->recvHaloOff),
                         static_cast<int64_t>(t->sendHaloOff), static_cast<int64_t>(3ULL * elems3 * 2ULL),
                         static_cast<int64_t>(rank - 1U), true);
        }
        if (rank + 1U < cp) {
          aclshmem_signal_wait_until(reinterpret_cast<__gm__ int32_t *>(ev), ACLSHMEM_CMP_GE, gen);
        }
      }
#endif
      PipeBarrier<PIPE_ALL>();
      SyncAll();
    }

    // Step B3.3: ShortConv1d_bwd -> d_q_in, d_k_in, d_v_in AND partial d_w_conv
    if (wid < activeAivs) {
      const uint32_t g4 = wid / slicesPerGroup;
      const uint32_t sIdx = wid % slicesPerGroup;
      const uint32_t cOff = g4 * cols;
      const uint32_t rowsPerSlice = (((S + slicesPerGroup - 1U) / slicesPerGroup) + 7U) & ~7U;
      const uint32_t rStart = sIdx * rowsPerSlice;
      const uint32_t rEnd = (rStart + rowsPerSlice <= S) ? (rStart + rowsPerSlice) : S;

      TPipe pipe;
      TBuf<TPosition::VECCALC> buf;
      const uint32_t uDuPad = 0;
      const uint32_t uXPad  = uDuPad + (BT + 3U) * cols * 2U + 64U;
      const uint32_t uW4    = uXPad + (BT + 3U) * cols * 2U + 64U;
      const uint32_t uDwAcc = uW4 + 4U * cols * 4U;
      const uint32_t uAcc   = uDwAcc + 4U * cols * 4U;
      const uint32_t uTmp   = uAcc + BT * cols * 4U;
      const uint32_t uOut   = uTmp + BT * cols * 4U;
      pipe.InitBuffer(buf, uOut + BT * cols * 2U + 64U);

      LocalTensor<bfloat16_t> duPad = buf.GetWithOffset<bfloat16_t>((BT + 3U) * cols, uDuPad);
      LocalTensor<bfloat16_t> xPad  = buf.GetWithOffset<bfloat16_t>((BT + 3U) * cols, uXPad);
      LocalTensor<float> w4         = buf.GetWithOffset<float>(4U * cols, uW4);
      LocalTensor<float> dwAcc      = buf.GetWithOffset<float>(4U * cols, uDwAcc);
      LocalTensor<float> acc        = buf.GetWithOffset<float>(BT * cols, uAcc);
      LocalTensor<float> tmp        = buf.GetWithOffset<float>(BT * cols, uTmp);
      LocalTensor<bfloat16_t> outBf = buf.GetWithOffset<bfloat16_t>(BT * cols, uOut);

      const uint16_t blockLen = static_cast<uint16_t>((cols * 2U) / 32U);
      const uint16_t rowStride = static_cast<uint16_t>(((D - cols) * 2U) / 32U);
      const uint8_t fpRowBlk = static_cast<uint8_t>((cols * 4U) / 32U);
      BinaryRepeatParams wMul{1, 1, 1, fpRowBlk, fpRowBlk, 0};

      for (uint32_t which = 0; which < 3U; ++which) {
        Duplicate(dwAcc, 0.0f, 4U * cols);
        PipeBarrier<PIPE_V>();

        if (rStart < rEnd) {
          {
            GlobalTensor<bfloat16_t> wGm;
            wGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(io[wConvIdxs[which]]) +
                                    static_cast<uint64_t>(cOff) * 4U, cols * 4U);
            DataCopy(outBf, wGm, cols * 4U);
            PipeBarrier<PIPE_ALL>();
            Cast(tmp, outBf, RoundMode::CAST_NONE, cols * 4U);
            PipeBarrier<PIPE_ALL>();
            for (uint32_t ch = 0; ch < cols; ++ch) {
              for (uint32_t tau = 0; tau < 4U; ++tau) {
                w4.SetValue(tau * cols + ch, tmp.GetValue(ch * 4U + tau));
              }
            }
            PipeBarrier<PIPE_ALL>();
          }

          __gm__ bfloat16_t *xBase = reinterpret_cast<__gm__ bfloat16_t *>(io[inIdxs[which]]);
          __gm__ bfloat16_t *dxBase = reinterpret_cast<__gm__ bfloat16_t *>(io[dInIdxs[which]]);
          __gm__ bfloat16_t *duBase =
              reinterpret_cast<__gm__ bfloat16_t *>(ws + t->wsDuOff) + static_cast<uint64_t>(which) * S * D;
          __gm__ bfloat16_t *fwdHaloBase =
              reinterpret_cast<__gm__ bfloat16_t *>(bc + cacheL.haloQkvOff) + static_cast<uint64_t>(which) * 3ULL * D;
          __gm__ bfloat16_t *revHaloBase =
              reinterpret_cast<__gm__ bfloat16_t *>(io[B_RECV_BUF] + t->recvHaloOff) +
              static_cast<uint64_t>(which) * 3ULL * D;

          for (uint32_t bStart = rStart; bStart < rEnd; bStart += BT) {
            const uint32_t curT = (bStart + BT <= rEnd) ? BT : (rEnd - bStart);

            if (bStart + curT + 3U <= S) {
              GlobalTensor<bfloat16_t> duGm;
              duGm.SetGlobalBuffer(duBase + static_cast<uint64_t>(bStart) * D + cOff,
                                   static_cast<uint64_t>(curT + 3U) * D);
              DataCopyParams cpD{static_cast<uint16_t>(curT + 3U), blockLen, rowStride, 0};
              DataCopy(duPad, duGm, cpD);
            } else {
              const uint32_t avail = S - bStart;
              const uint32_t tail = (bStart + curT + 3U) - S;
              GlobalTensor<bfloat16_t> duGm;
              duGm.SetGlobalBuffer(duBase + static_cast<uint64_t>(bStart) * D + cOff,
                                   static_cast<uint64_t>(avail) * D);
              DataCopyParams cpD{static_cast<uint16_t>(avail), blockLen, rowStride, 0};
              DataCopy(duPad, duGm, cpD);
              if (rank + 1U < cp && cp > 1U) {
                GlobalTensor<bfloat16_t> rhGm;
                rhGm.SetGlobalBuffer(revHaloBase + cOff, static_cast<uint64_t>(tail) * D);
                DataCopyParams cpH{static_cast<uint16_t>(tail), blockLen, rowStride, 0};
                DataCopy(duPad[avail * cols], rhGm, cpH);
              } else {
                Duplicate(tmp, 0.0f, tail * cols);
                PipeBarrier<PIPE_V>();
                Cast(duPad[avail * cols], tmp, RoundMode::CAST_RINT, tail * cols);
                PipeBarrier<PIPE_ALL>();
              }
            }

            if (bStart >= 3U) {
              GlobalTensor<bfloat16_t> xGm;
              xGm.SetGlobalBuffer(xBase + static_cast<uint64_t>(bStart - 3U) * D + cOff,
                                  static_cast<uint64_t>(curT + 3U) * D);
              DataCopyParams cpX{static_cast<uint16_t>(curT + 3U), blockLen, rowStride, 0};
              DataCopy(xPad, xGm, cpX);
            } else {
              const uint32_t padZ = 3U - bStart;
              if (rank > 0U && cp > 1U) {
                GlobalTensor<bfloat16_t> hGm;
                hGm.SetGlobalBuffer(fwdHaloBase + static_cast<uint64_t>(bStart) * D + cOff,
                                    static_cast<uint64_t>(padZ) * D);
                DataCopyParams cpH{static_cast<uint16_t>(padZ), blockLen, rowStride, 0};
                DataCopy(xPad, hGm, cpH);
              } else {
                Duplicate(tmp, 0.0f, padZ * cols);
                PipeBarrier<PIPE_V>();
                Cast(xPad, tmp, RoundMode::CAST_RINT, padZ * cols);
                PipeBarrier<PIPE_ALL>();
              }
              GlobalTensor<bfloat16_t> xGm;
              xGm.SetGlobalBuffer(xBase + cOff, static_cast<uint64_t>(bStart + curT) * D);
              DataCopyParams cpX{static_cast<uint16_t>(bStart + curT), blockLen, rowStride, 0};
              DataCopy(xPad[padZ * cols], xGm, cpX);
            }
            PipeBarrier<PIPE_ALL>();

            // dx[t] = sum_{tau=0..3} duPad[t + tau] * w4[3 - tau]
            Duplicate(acc, 0.0f, curT * cols);
            PipeBarrier<PIPE_V>();
            for (uint32_t tau = 0; tau < 4U; ++tau) {
              Cast(tmp, duPad[tau * cols], RoundMode::CAST_NONE, curT * cols);
              PipeBarrier<PIPE_V>();
              const uint32_t wIdx = 3U - tau;
              for (uint32_t c0 = 0; c0 < cols; c0 += 64U) {
                Mul(tmp[c0], tmp[c0], w4[wIdx * cols + c0], 64, static_cast<uint8_t>(curT), wMul);
              }
              PipeBarrier<PIPE_V>();
              Add(acc, acc, tmp, curT * cols);
              PipeBarrier<PIPE_V>();
            }
            Cast(outBf, acc, RoundMode::CAST_RINT, curT * cols);
            PipeBarrier<PIPE_ALL>();
            {
              GlobalTensor<bfloat16_t> dxGm;
              dxGm.SetGlobalBuffer(dxBase + static_cast<uint64_t>(bStart) * D + cOff,
                                   static_cast<uint64_t>(curT) * D);
              DataCopyParams outP{static_cast<uint16_t>(curT), blockLen, 0, rowStride};
              DataCopy(dxGm, outBf, outP);
              PipeBarrier<PIPE_ALL>();
            }

            // dwAcc[tau] += sum_{t=0..curT-1} duPad[t] * xPad[t + tau]
            Cast(acc, duPad[0], RoundMode::CAST_NONE, curT * cols);
            PipeBarrier<PIPE_V>();
            for (uint32_t tau = 0; tau < 4U; ++tau) {
              Cast(tmp, xPad[tau * cols], RoundMode::CAST_NONE, curT * cols);
              PipeBarrier<PIPE_V>();
              Mul(tmp, tmp, acc, curT * cols);
              PipeBarrier<PIPE_V>();
              for (uint32_t st = curT >> 1; st >= 1U; st >>= 1) {
                Add(tmp, tmp, tmp[st * cols], st * cols);
                PipeBarrier<PIPE_V>();
              }
              Add(dwAcc[tau * cols], dwAcc[tau * cols], tmp[0], cols);
              PipeBarrier<PIPE_V>();
            }
            PipeBarrier<PIPE_ALL>();
          }
        }

        // Transpose dwAcc [4, cols] -> tmp [cols, 4] and store to wsPartialWConvOff
        PipeBarrier<PIPE_ALL>();
        for (uint32_t ch = 0; ch < cols; ++ch) {
          for (uint32_t tau = 0; tau < 4U; ++tau) {
            tmp.SetValue(ch * 4U + tau, dwAcc.GetValue(tau * cols + ch));
          }
        }
        PipeBarrier<PIPE_ALL>();
        {
          GlobalTensor<float> pwGm;
          pwGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ws + t->wsPartialWConvOff) +
                                   ((static_cast<uint64_t>(sIdx) * 3ULL + which) * D + cOff) * 4ULL,
                               cols * 4U);
          DataCopy(pwGm, tmp, cols * 4U);
          PipeBarrier<PIPE_ALL>();
        }
      }
      pipe.Destroy();
    }
    PipeBarrier<PIPE_ALL>();
    SyncAll();

    // Step B3.4: Epilogue Reductions + d_beta_in [H, S] -> [S, H] Transpose
    {
      TPipe pipe;
      TBuf<TPosition::VECCALC> buf;
      pipe.InitBuffer(buf, 4096U * 4U * 2U + 4096U * 2U + 64U);
      LocalTensor<float> acc = buf.GetWithOffset<float>(4096U, 0);
      LocalTensor<float> tmp = buf.GetWithOffset<float>(4096U, 4096U * 4U);
      LocalTensor<bfloat16_t> bf = buf.GetWithOffset<bfloat16_t>(4096U, 4096U * 8U);

      // 1. Reduce d_o_norm_weight [numAiv, d] -> [d]
      if (wid == 0U) {
        Duplicate(acc, 0.0f, d);
        PipeBarrier<PIPE_ALL>();
        __gm__ float *pnBase = reinterpret_cast<__gm__ float *>(ws + t->wsPartialNormOff);
        for (uint32_t a = 0; a < numAiv; ++a) {
          GlobalTensor<float> sGm;
          sGm.SetGlobalBuffer(pnBase + a * d, d);
          DataCopy(tmp, sGm, d);
          PipeBarrier<PIPE_ALL>();
          Add(acc, acc, tmp, d);
          PipeBarrier<PIPE_ALL>();
        }
        GlobalTensor<float> dGm;
        dGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(io[B_DO_NORM_W]), d);
        DataCopy(dGm, acc, d);
        PipeBarrier<PIPE_ALL>();
      }

      // 2. Reduce d_a_log [numAiv, aLogStrideF] -> [H]
      if (wid == 1U || numAiv == 1U) {
        const uint32_t aStride = static_cast<uint32_t>((H + 7U) & ~7U);
        Duplicate(acc, 0.0f, aStride);
        PipeBarrier<PIPE_ALL>();
        __gm__ float *paBase = reinterpret_cast<__gm__ float *>(ws + t->wsPartialALogOff);
        for (uint32_t a = 0; a < numAiv; ++a) {
          GlobalTensor<float> sGm;
          sGm.SetGlobalBuffer(paBase + static_cast<uint64_t>(a) * aStride, aStride);
          DataCopy(tmp, sGm, aStride);
          PipeBarrier<PIPE_ALL>();
          Add(acc, acc, tmp, aStride);
          PipeBarrier<PIPE_ALL>();
        }
        GlobalTensor<float> dGm;
        dGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(io[B_DA_LOG]), H);
        DataCopy(dGm, acc, (H >= 8U && (H & 7U) == 0U) ? H : aStride);
        PipeBarrier<PIPE_ALL>();
      }

      // 3. Reduce d_dt_bias [numAiv, D] -> [D]
      __gm__ float *pbBase = reinterpret_cast<__gm__ float *>(ws + t->wsPartialDtBiasOff);
      for (uint32_t h = wid; h < H; h += numAiv) {
        Duplicate(acc, 0.0f, d);
        PipeBarrier<PIPE_ALL>();
        for (uint32_t a = 0; a < numAiv; ++a) {
          GlobalTensor<float> sGm;
          sGm.SetGlobalBuffer(pbBase + static_cast<uint64_t>(a) * D + static_cast<uint64_t>(h) * d, d);
          DataCopy(tmp, sGm, d);
          PipeBarrier<PIPE_ALL>();
          Add(acc, acc, tmp, d);
          PipeBarrier<PIPE_ALL>();
        }
        GlobalTensor<float> dGm;
        dGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(io[B_DDT_BIAS]) + static_cast<uint64_t>(h) * d, d);
        DataCopy(dGm, acc, d);
        PipeBarrier<PIPE_ALL>();
      }

      // 4. Reduce d_w_conv [slicesPerGroup, 3*D*4] -> [3*D*4]
      const uint64_t totalWConv = 3ULL * D * 4ULL;
      const uint64_t wTile = 2048ULL;
      const uint64_t nWTiles = (totalWConv + wTile - 1ULL) / wTile;
      __gm__ float *pwBase = reinterpret_cast<__gm__ float *>(ws + t->wsPartialWConvOff);
      for (uint64_t ti = wid; ti < nWTiles; ti += numAiv) {
        const uint64_t off = ti * wTile;
        const uint32_t cur = static_cast<uint32_t>((off + wTile <= totalWConv) ? wTile : (totalWConv - off));
        Duplicate(acc, 0.0f, cur);
        PipeBarrier<PIPE_ALL>();
        for (uint32_t s = 0; s < slicesPerGroup; ++s) {
          GlobalTensor<float> sGm;
          sGm.SetGlobalBuffer(pwBase + static_cast<uint64_t>(s) * totalWConv + off, cur);
          DataCopy(tmp, sGm, cur);
          PipeBarrier<PIPE_ALL>();
          Add(acc, acc, tmp, cur);
          PipeBarrier<PIPE_ALL>();
        }
        GlobalTensor<float> dGm;
        dGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(io[B_DW_CONV]) + off, cur);
        DataCopy(dGm, acc, cur);
        PipeBarrier<PIPE_ALL>();
      }

      // 5. Transpose d_beta_in from head-major [H, S] FP32 (wsDOOff) -> [S, H] BF16 (io[B_DBETA_IN])
      __gm__ float *dbHeadBase = reinterpret_cast<__gm__ float *>(ws + t->wsDOOff);
      __gm__ bfloat16_t *dbOutBase = reinterpret_cast<__gm__ bfloat16_t *>(io[B_DBETA_IN]);
      for (uint32_t c = wid; c < numChunks; c += numAiv) {
        const uint64_t tStart = static_cast<uint64_t>(c) * C;
        for (uint32_t h = 0; h < H; ++h) {
          GlobalTensor<float> sGm;
          sGm.SetGlobalBuffer(dbHeadBase + static_cast<uint64_t>(h) * S + tStart, C);
          DataCopy(tmp[h * C], sGm, C);
        }
        PipeBarrier<PIPE_ALL>();
        for (uint32_t r = 0; r < C; ++r) {
          for (uint32_t h = 0; h < H; ++h) {
            acc.SetValue(r * H + h, tmp.GetValue(h * C + r));
          }
        }
        PipeBarrier<PIPE_ALL>();
        Cast(bf, acc, RoundMode::CAST_RINT, C * H);
        PipeBarrier<PIPE_ALL>();
        GlobalTensor<bfloat16_t> dGm;
        dGm.SetGlobalBuffer(dbOutBase + tStart * H, C * H);
        DataCopy(dGm, bf, C * H);
        PipeBarrier<PIPE_ALL>();
      }

      pipe.Destroy();
    }
  }
}

// ---------------------------------------------------------------------------
// Top-Level Entry: Unified CP=1 and CP>1 (allgather / prefix_sum) Backward
// ---------------------------------------------------------------------------
__aicore__ inline void RunHyperMegaKdaGrad(uint32_t blockIdx, GM_ADDR *io, const HyperMegaKdaGradTilingData *t) {
  const uint32_t cp = t->cpSize;
  const uint32_t rank = t->cpRank;
  const uint32_t H = t->numHeads;
  const uint32_t d = t->headDim;
  const uint32_t numCube = t->numCubeCores;
  const uint32_t numAiv = 2U * numCube;
  const bool isPrefix = (t->cpMode == 2U);
  const uint64_t stateFloats = static_cast<uint64_t>(H) * d * d;
  GM_ADDR ws = io[B_WORKSPACE];
  GM_ADDR sb = io[B_SEND_BUF];
  GM_ADDR rb = io[B_RECV_BUF];
  GM_ADDR ev = io[B_EVENT];

  if ASCEND_IS_AIV {
    const uint32_t wid = blockIdx;
    int32_t gen = 1;
    if (cp > 1U) {
      BwdShmemGenBarrier(wid, ev, t->eventCounterBytes, cp, rank, false);
      PipeBarrier<PIPE_ALL>();
      SyncAll();
      gen = BwdReadReadyGen(wid, ev, t->eventCounterBytes, cp);
    }

    // Phase B0: GatedRMSNorm_bwd -> d_g_out, wsDO, partial d_o_norm_w
    RunPhaseB0RmsNormBwd(wid, io, t);
    PipeBarrier<PIPE_ALL>();
    SyncAll();

    // Phase B1: Inter-chunk BwdH + (CP>1) Reverse Suffix-Scan
    if (cp == 1U) {
      RunPhaseB1BwdHAiv(wid, io, t, false, false, false);
      PipeBarrier<PIPE_ALL>();
      SyncAll();
    } else {
      if (rank + 1U == cp) {
        RunPhaseB1BwdHAiv(wid, io, t, false, true, false);
      } else if (rank > 0U) {
        RunPhaseB1BwdHAiv(wid, io, t, false, true, true);
      }
      PipeBarrier<PIPE_ALL>();
      SyncAll();

      if (!isPrefix) {
        // ALLGATHER (1-round parallel reverse put): rank > 0 sends (D_r, [A_r]) to all dst < rank
        if (rank > 0U) {
          const uint32_t dst = wid;
          if (dst < rank) {
            const bool hasA = (rank + 1U < cp);
            const int64_t nBytes = static_cast<int64_t>(hasA ? t->scanSlotBytes : t->scanHalfBytes);
            const int64_t sigOff = static_cast<int64_t>((1U + rank) * 16U);
            const int64_t srcOff = static_cast<int64_t>(t->sendScanOff);
            const int64_t dstOff = static_cast<int64_t>(t->recvScanOff + rank * t->scanSlotBytes);
            BwdShmemPutSet(rb, sb, ev, sigOff, gen, dstOff, srcOff, nBytes, static_cast<int64_t>(dst), true);
          }
        }
        if (rank + 1U < cp && wid == 0U) {
#if !defined(__DAV_CUBE__) && !defined(__DAV_C220_CUBE__)
          for (uint32_t src = rank + 1U; src < cp; ++src) {
            aclshmem_signal_wait_until(reinterpret_cast<__gm__ int32_t *>(ev) + (1U + src) * 16U,
                                       ACLSHMEM_CMP_GE, gen);
          }
#endif
        }
        PipeBarrier<PIPE_ALL>();
        SyncAll();

        if (rank + 1U < cp) {
          GM_ADDR lastD = rb + t->recvScanOff + static_cast<uint64_t>(cp - 1U) * t->scanSlotBytes;
          RunBwdCopyStateAiv(wid, lastD, ws + t->wsRevExclOff, stateFloats, numAiv);
          PipeBarrier<PIPE_ALL>();
          SyncAll();
          for (int32_t k = static_cast<int32_t>(cp) - 2; k > static_cast<int32_t>(rank); --k) {
            GM_ADDR slotK = rb + t->recvScanOff + static_cast<uint64_t>(k) * t->scanSlotBytes;
            RunBwdMatmul128Aiv(wid, io, t, slotK + t->scanHalfBytes, ws + t->wsRevExclOff, slotK,
                               ws + t->wsRevExclOff);
            PipeBarrier<PIPE_ALL>();
            SyncAll();
          }
        }
      } else {
        // PREFIX_SUM (O(log2(CP))-round Reverse Hillis-Steele Exclusive Suffix Tree-Scan)
        bool hasExcl = false;
        for (uint32_t k = 0; k < t->scanRounds; ++k) {
          const uint32_t dist = 1U << k;
          if (rank >= dist && wid == 0U) {
            const uint32_t dst = rank - dist;
            const int64_t sigOff = static_cast<int64_t>((1U + k) * 16U);
            const int64_t srcOff = static_cast<int64_t>(t->sendScanOff);
            const int64_t dstOff = static_cast<int64_t>(t->recvScanOff + k * t->scanSlotBytes);
            BwdShmemPutSet(rb, sb, ev, sigOff, gen, dstOff, srcOff, static_cast<int64_t>(t->scanSlotBytes),
                           static_cast<int64_t>(dst), true);
          }
          if (rank + dist < cp && wid == 0U) {
#if !defined(__DAV_CUBE__) && !defined(__DAV_C220_CUBE__)
            aclshmem_signal_wait_until(reinterpret_cast<__gm__ int32_t *>(ev) + (1U + k) * 16U,
                                       ACLSHMEM_CMP_GE, gen);
#endif
          }
          PipeBarrier<PIPE_ALL>();
          SyncAll();

          if (rank + dist < cp) {
            GM_ADDR peerSlot = rb + t->recvScanOff + static_cast<uint64_t>(k) * t->scanSlotBytes;
            if (!hasExcl) {
              RunBwdCopyStateAiv(wid, peerSlot, ws + t->wsRevExclOff, 2ULL * stateFloats, numAiv);
              hasExcl = true;
            } else {
              RunBwdMatmul128Aiv(wid, io, t, ws + t->wsRevExclOff + t->scanHalfBytes, peerSlot,
                                 ws + t->wsRevExclOff, ws + t->wsRevExclOff);
              PipeBarrier<PIPE_ALL>();
              SyncAll();
              if (k + 1U < t->scanRounds) {
                RunBwdMatmul128Aiv(wid, io, t, peerSlot + t->scanHalfBytes,
                                   ws + t->wsRevExclOff + t->scanHalfBytes, nullptr,
                                   ws + t->wsRevExclOff + t->scanHalfBytes);
              }
            }
            PipeBarrier<PIPE_ALL>();
            SyncAll();

            if (k + 1U < t->scanRounds && rank > 0U) {
              RunBwdMatmul128Aiv(wid, io, t, ws + t->wsRevSelfOff + t->scanHalfBytes, peerSlot,
                                 ws + t->wsRevSelfOff, ws + t->wsRevSelfOff);
              PipeBarrier<PIPE_ALL>();
              SyncAll();
              RunBwdMatmul128Aiv(wid, io, t, peerSlot + t->scanHalfBytes,
                                 ws + t->wsRevSelfOff + t->scanHalfBytes, nullptr,
                                 ws + t->wsRevSelfOff + t->scanHalfBytes);
              PipeBarrier<PIPE_ALL>();
              SyncAll();
              RunBwdCopyStateAiv(wid, ws + t->wsRevSelfOff, sb + t->sendScanOff, 2ULL * stateFloats, numAiv);
              PipeBarrier<PIPE_ALL>();
              SyncAll();
            }
          }
        }
      }

      if (rank + 1U < cp) {
        RunPhaseB1BwdHAiv(wid, io, t, true, false, false);
      }
      PipeBarrier<PIPE_ALL>();
      SyncAll();
    }

    // Phase B2: Double-Buffered L2-Ring WY_bwd + Intra_bwd + dG + Gate_bwd + Beta_bwd + L2Norm_bwd
    RunPhaseB2Aiv(wid, io, t);
    PipeBarrier<PIPE_ALL>();
    SyncAll();

    // Phase B3: SiLU_bwd + Reverse Halo + ShortConv1d_bwd + Epilogue Reductions
    RunPhaseB3ConvBwdAndReduce(wid, gen, io, t);
    PipeBarrier<PIPE_ALL>();
    SyncAll();

    if (cp > 1U) {
      BwdShmemGenBarrier(wid, ev, t->eventCounterBytes, cp, rank, true);
      PipeBarrier<PIPE_ALL>();
      SyncAll();
    }
  }

  if ASCEND_IS_AIC {
    const uint32_t core = blockIdx;
    if (cp > 1U) {
      SyncAll();  // Match ready barrier
    }
    SyncAll();  // Match Phase B0

    if (cp == 1U) {
      RunPhaseB1BwdHAic(core, io, t, false);
      SyncAll();
    } else {
      if (rank + 1U == cp) {
        RunPhaseB1BwdHAic(core, io, t, false);
      } else if (rank > 0U) {
        RunPhaseB1BwdHAic(core, io, t, true);
      }
      SyncAll();

      if (!isPrefix) {
        SyncAll();  // Match SHMEM wait
        if (rank + 1U < cp) {
          SyncAll();  // Match copy lastD
          for (int32_t k = static_cast<int32_t>(cp) - 2; k > static_cast<int32_t>(rank); --k) {
            RunBwdMatmul128Aic(core, io, t, true);
            SyncAll();
          }
        }
      } else {
        bool hasExcl = false;
        for (uint32_t k = 0; k < t->scanRounds; ++k) {
          const uint32_t dist = 1U << k;
          SyncAll();  // Match SHMEM round wait
          if (rank + dist < cp) {
            if (!hasExcl) {
              hasExcl = true;
            } else {
              RunBwdMatmul128Aic(core, io, t, true);
              SyncAll();
              if (k + 1U < t->scanRounds) {
                RunBwdMatmul128Aic(core, io, t, false);
              }
            }
            SyncAll();

            if (k + 1U < t->scanRounds && rank > 0U) {
              RunBwdMatmul128Aic(core, io, t, true);
              SyncAll();
              RunBwdMatmul128Aic(core, io, t, false);
              SyncAll();
              SyncAll();  // Match stage copy
            }
          }
        }
      }

      if (rank + 1U < cp) {
        RunPhaseB1BwdHAic(core, io, t, false);
      }
      SyncAll();
    }

    // Phase B2 AIC
    RunPhaseB2Aic(core, io, t);
    SyncAll();

    // Match Phase B3 SyncAlls:
    SyncAll();  // Match Step B3.1
    if (cp > 1U) {
      SyncAll();  // Match Step B3.2 Reverse Halo
    }
    SyncAll();  // Match Step B3.3
    SyncAll();  // Match Step B3.4 Epilogue
    if (cp > 1U) {
      SyncAll();  // Match completion barrier
    }
  }
}

}  // namespace MulticoreKdaBwd

#endif  // HYPER_MEGA_KDA_BWD_FAST_H
