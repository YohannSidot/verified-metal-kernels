# Copyright 2026 Yohann Sidot (harness); portions adapted from Kapitsa, Copyright 2026 Nace AI.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
# with the License. The full text is in LICENSE-APACHE-2.0.txt, shipped with this file, and at
# http://www.apache.org/licenses/LICENSE-2.0. Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# either express or implied. Kapitsa's credit to MLX (MIT License) is reproduced in ../THIRD_PARTY_NOTICES.txt.
# This file was written by AI models (Claude) under the author's direction.
#
# Portions adapted from Kapitsa (https://github.com/nace-ai/kapitsa, Copyright 2026 Nace AI, Apache License 2.0),
# read as text at commit
# ea7625f5a41992980c3adb5f6a4542cee7ff7416, are marked "adapted from Kapitsa". Changes made to them: transcribed into
# mx.fast.metal_kernel bodies (MLX's generated signatures instead of buffer indices), parameters passed as small
# uint32 arrays instead of constant structs, segments stacked into one matrix, bfloat read as its 16 raw bits,
# simdgroup counts as constants, routing indices read from device memory instead of a threadgroup copy, a grid z (or y)
# index added to run several token rows in one dispatch.
#   QMVF_*          <- src/kernels/qmvf.metal (qmvf_core, qmvf_multi): R1 of the LM head and of the router
#   QMV2_ROWS       <- src/kernels/quant.h (qmv2_rows)
#   MOE_GU_R1       <- src/kernels/moe.metal (moe_gate_up3_impl<2>, moe_gate_up3_r2)
#   MOE_DOWN_R1     <- src/kernels/moe.metal (moe_down2_impl, moe_down2)
#   PREP_SRC        <- src/kernels/verify8.metal (x8_prep2_impl<false>)
#   MOE8E_GU_SRC    <- src/kernels/verify8.metal (moe8e_gu_t<2, 8>, moe8e_gu_r2)
#   MOE8E_DOWN_SRC  <- src/kernels/verify8.metal (moe8e_down_t<2>, moe8e_down_r2)
#   COMBINE_SRC     <- src/kernels/verify8.metal (moe8_combine)
#   QMM8T_SRC       <- src/kernels/tensor/verify8t.metal (qmm8t_impl; qmm8tp_b4_nt32 for the LM head)
"""One verify cycle of Kapitsa's Qwen3.6-35B-A3B, op by op: the plain step's arithmetic (R1), today's 8-row verify on an
M5 (R2), and a multi-row form whose every row carries R1's bits (K).  Our transcriptions, compiled by MLX with
math_mode "fast"; synthetic weights in the pack's layout; nothing from Kapitsa is run.

Classes:
  lm_head  248320 x 2048, 4-bit g64, final RMSNorm folded.  R1 qmvf_b4_t1_r4_norm; R2 x8_prep2 + qmm8tp_b4_nt32;
           K the qmvf source at 2 token rows per thread, 8 weight rows per simdgroup (M >= 4), R1 on a grid (M <= 3).
  moe      one MoE block's experts: 8 routed + the shared expert, gate|up 2 x 512 x 2048 and down 2048 x 512, 4-bit g64,
           routing given (the same indices and weights for every arm).  R1 moe_gate_up3_r2 + moe_down2 on the
           normalized fp32 row; R2 x8_prep2 + moe8e_gu_r2 + moe8e_down_r2 + moe8_combine (the M5 default, emu 2,
           every expert and the shared one on 2-row tiles); K R1's two kernels with a grid index over the rows.
  router   router 256 + shared-expert gate 1, x 2048, 8-bit g64, post-attention RMSNorm folded, normalized row written.
           R1 qmvf_b8_t1_r4_norm (+ xn); K R1 on an M-row grid (k_nt2: 2 token rows per thread, 8 weight rows).
           R2 (x8_prep2 two-term + qmm8x2_b8_nf1, simdgroup MMA) is not transcribed.
Modes: --check (bitwise, R1 alone per row), --bench (interleaved rounds), --quick, --classes, --json.
"""
import argparse
import json
import platform
import random
import statistics
import subprocess
import time

import numpy as np
import mlx.core as mx

D = 2048
EPS = 1e-6
MATH = {"math_mode": "fast"}
VOCAB = 248320
NE, TOPK, MI = 256, 8, 512
MT = 72  # moe8e tile list capacity (verify8.metal kV8SM)

COMMON = r"""
inline __attribute__((always_inline)) float kl_bf(ushort b) { return as_type<float>(uint(b) << 16); }
inline __attribute__((always_inline)) float kl_silu(float x) { return x / (1.0f + metal::fast::exp(-x)); }
"""

# ---- adapted from Kapitsa src/kernels/qmvf.metal (qmvf_core; NORM true, GS 64, NM 1) ----
QMVF_HEADER = COMMON + r"""
template <int BITS, int NT, int ROWS>
inline __attribute__((always_inline)) void kl_qmvf_core(const device uint32_t* W, const device ushort* S,
    const device ushort* B, const device float* x, uint x_ld, uint t0, uint R, uint K, uint N,
    const device ushort* nw, uint row0, uint lane, thread float (&acc)[ROWS][NT], thread float (&ss)[NT]) {
  constexpr int GS = 64;
  constexpr int PER_WORD = 32 / BITS;
  constexpr int VPL = 16;
  constexpr int WORDS = VPL / PER_WORD;
  constexpr uint MASK = (1u << BITS) - 1u;
  const uint wpr = K / PER_WORD, gpr = K / GS;
  uint rows[ROWS];
  for (int r = 0; r < ROWS; ++r) rows[r] = min(row0 + r, N - 1);
  for (int r = 0; r < ROWS; ++r)
    for (int t = 0; t < NT; ++t) acc[r][t] = 0.0f;
  for (int t = 0; t < NT; ++t) ss[t] = 0.0f;
  uint xo[NT];
  for (int t = 0; t < NT; ++t) xo[t] = min(t0 + uint(t), R - 1) * x_ld;
  for (uint k0 = lane * VPL; k0 < K; k0 += 32 * VPL) {
    float xv[NT][VPL];
    float xs[NT];
    float4 nv[VPL / 4];
    for (int i = 0; i < VPL / 4; ++i) {
      const uint k = k0 + i * 4;
      nv[i] = float4(kl_bf(nw[k]), kl_bf(nw[k + 1]), kl_bf(nw[k + 2]), kl_bf(nw[k + 3]));
    }
    for (int t = 0; t < NT; ++t) {
      xs[t] = 0.0f;
      for (int i = 0; i < VPL / 4; ++i) {
        float4 v = ((const device float4*)(x + xo[t] + k0))[i];
        ss[t] += dot(v, v);
        v *= nv[i];
        xs[t] += v.x + v.y + v.z + v.w;
        for (int e = 0; e < 4; ++e)
          xv[t][i * 4 + e] = v[e] * float(1.0f / float(1u << (BITS * ((i * 4 + e) % PER_WORD))));
      }
    }
    const uint g = k0 / GS, wo = k0 / PER_WORD;
    for (int r = 0; r < ROWS; ++r) {
      const device uint32_t* wp = W + rows[r] * wpr + wo;
      uint32_t wv[WORDS];
      if (WORDS == 4) {
        uint4 v = *(const device uint4*)wp;
        wv[0] = v.x; wv[1] = v.y; wv[2] = v.z; wv[3] = v.w;
      } else {
        uint2 v = *(const device uint2*)wp;
        wv[0] = v.x; wv[1] = v.y;
      }
      float d[NT];
      for (int t = 0; t < NT; ++t) d[t] = 0.0f;
      for (int i = 0; i < WORDS; ++i)
        for (int j = 0; j < PER_WORD; ++j) {
          const float wf = float(wv[i] & (MASK << (BITS * j)));
          for (int t = 0; t < NT; ++t) d[t] += wf * xv[t][i * PER_WORD + j];
        }
      const float sc = kl_bf(S[rows[r] * gpr + g]), bi = kl_bf(B[rows[r] * gpr + g]);
      for (int t = 0; t < NT; ++t) acc[r][t] += sc * d[t] + bi * xs[t];
    }
  }
}
"""
# qmvf_multi's epilogue (inv after the dot, xn written by token-row group 0 / simdgroup 0 of the first block)
QMVF_BODY = r"""
  const uint lane = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  const uint nsg = 2;
  const uint K = prm[0], R = prm[1], N = prm[2];
  const float eps = epsv[0];
  const uint t0 = threadgroup_position_in_grid.x * NT;
  const uint row0 = (threadgroup_position_in_grid.y * nsg + sg) * ROWS;
  if (row0 >= N) return;
  float acc[ROWS][NT], ss[NT];
  kl_qmvf_core<BITS, NT, ROWS>(w, s, b, x, K, t0, R, K, N, nw, row0, lane, acc, ss);
  for (int t = 0; t < NT; ++t) {
    float inv = metal::precise::rsqrt(simd_sum(ss[t]) / float(K) + eps);
    for (int r = 0; r < ROWS; ++r) {
      float v = simd_sum(acc[r][t]);
      v *= inv;
      const uint row = row0 + r;
      if (lane == 0 && row < N && t0 + uint(t) < R) y[(t0 + t) * N + row] = v;
    }
    if (XN && threadgroup_position_in_grid.y == 0 && sg == 0 && t0 + uint(t) < R)
      for (uint k = lane; k < K; k += 32) xn[(t0 + t) * K + k] = x[(t0 + t) * K + k] * inv * kl_bf(nw[k]);
  }
"""

# ---- adapted from Kapitsa src/kernels/quant.h (qmv2_rows) and src/kernels/moe.metal (moe_gate_up3_r2, moe_down2) ----
MOE_R1_HEADER = COMMON + r"""
template <int BITS, int VPL, int ROWS, int GS>
inline __attribute__((always_inline)) void kl_qmv2_rows(const device uint32_t* w, const device ushort* scales,
    const device ushort* biases, threadgroup const float* xs, uint N, uint K, uint row0, uint lane, thread float* acc) {
  constexpr int PER_WORD = 32 / BITS;
  constexpr uint MASK = (1u << BITS) - 1u;
  constexpr int WORDS = VPL / PER_WORD;
  const uint words_per_row = K / PER_WORD;
  const uint groups_per_row = K / GS;
  uint rows[ROWS];
  for (int r = 0; r < ROWS; ++r) {
    acc[r] = 0.0f;
    rows[r] = min(row0 + r, N - 1);
  }
  for (uint k0 = lane * VPL; k0 < K; k0 += 32 * VPL) {
    float xv[VPL];
    float xsum = 0.0f;
    for (int i = 0; i < VPL / 4; ++i) {
      float4 v = ((threadgroup const float4*)(xs + k0))[i];
      xsum += v.x + v.y + v.z + v.w;
      for (int e = 0; e < 4; ++e) xv[i * 4 + e] = v[e] * float(1.0f / float(1u << (BITS * ((i * 4 + e) % PER_WORD))));
    }
    const uint g = k0 / GS, wo = k0 / PER_WORD;
    for (int r = 0; r < ROWS; ++r) {
      const device uint32_t* wp = w + rows[r] * words_per_row + wo;
      uint32_t wv[WORDS];
      if (WORDS == 4) {
        uint4 v = *(const device uint4*)wp;
        wv[0] = v.x; wv[1] = v.y; wv[2] = v.z; wv[3] = v.w;
      } else if (WORDS == 2) {
        uint2 v = *(const device uint2*)wp;
        wv[0] = v.x; wv[1] = v.y;
      } else {
        for (int i = 0; i < WORDS; ++i) wv[i] = wp[i];
      }
      float d = 0.0f;
      for (int i = 0; i < WORDS; ++i)
        for (int j = 0; j < PER_WORD; ++j) d += float(wv[i] & (MASK << (BITS * j))) * xv[i * PER_WORD + j];
      acc[r] += kl_bf(scales[rows[r] * groups_per_row + g]) * d + kl_bf(biases[rows[r] * groups_per_row + g]) * xsum;
    }
  }
  for (int r = 0; r < ROWS; ++r) acc[r] = simd_sum(acc[r]);
}
"""
# grid (I / 16 threadgroups, k + 1 slots, rows) x 256 threads; token row = grid z (R1 alone: one row)
MOE_GU_R1 = r"""
  constexpr int ROWS = 2;
  threadgroup float xs[4096];
  const uint3 tgp = threadgroup_position_in_grid;
  const uint tid = thread_index_in_threadgroup, sg = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
  const uint ntg = 256, nsg = 8;
  const uint Dm = prm[0], I = prm[1], SI = prm[2], k = prm[3];
  const uint t = tgp.z, slot = tgp.y;
  const bool shared = slot == k;
  for (uint i = tid; i < Dm / 4; i += ntg) ((threadgroup float4*)xs)[i] = ((const device float4*)(xn + t * Dm))[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint N = shared ? SI : I;
  const uint row0 = (tgp.x * nsg + sg) * ROWS;
  if (row0 >= N) return;
  const uint K = Dm, wpr = K / 8, gpr = K / 64;
  const device uint32_t *G = gw, *U = uw;
  const device ushort *Gs = gsc, *Gb = gbi, *Us = usc, *Ub = ubi;
  if (shared) {
    G = sgw; Gs = sgs; Gb = sgb; U = suw; Us = sus; Ub = sub;
  } else {
    const uint e = idx[t * k + slot];
    G += e * N * wpr; U += e * N * wpr;
    Gs += e * N * gpr; Gb += e * N * gpr; Us += e * N * gpr; Ub += e * N * gpr;
  }
  float ag[ROWS], au[ROWS];
  kl_qmv2_rows<4, 16, ROWS, 64>(G, Gs, Gb, xs, N, K, row0, lane, ag);
  kl_qmv2_rows<4, 16, ROWS, 64>(U, Us, Ub, xs, N, K, row0, lane, au);
  if (lane == 0)
    for (int r = 0; r < ROWS; ++r)
      if (row0 + r < N) h[(t * (k + 1) + slot) * max(I, SI) + row0 + r] = kl_silu(ag[r]) * au[r];
"""
# grid (D / 16 threadgroups, rows) x 256 threads; y = x + the block (Kapitsa: x[row] += tot)
MOE_DOWN_R1 = r"""
  constexpr int ROWS = 2;
  threadgroup float hs[9 * 512];
  const uint3 tgp = threadgroup_position_in_grid;
  const uint tid = thread_index_in_threadgroup, sg = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
  const uint ntg = 256, nsg = 8;
  const uint Dm = prm[0], I = prm[1], SI = prm[2], k = prm[3];
  const uint t = tgp.y;
  const uint hstride = max(I, SI);
  const device float* hr = h + t * (k + 1) * hstride;
  for (uint i = tid; i < (k + 1) * hstride / 4; i += ntg) ((threadgroup float4*)hs)[i] = ((const device float4*)hr)[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint row0 = (tgp.x * nsg + sg) * ROWS;
  if (row0 >= Dm) return;
  float tot[ROWS] = {0.0f, 0.0f};
  for (uint slot = 0; slot <= k; ++slot) {
    const bool shared = slot == k;
    const uint K = shared ? SI : I;
    const uint wpr = K / 8, gpr = K / 64;
    const device uint32_t* W = shared ? sdw : dw + idx[t * k + slot] * Dm * wpr;
    const device ushort* S = shared ? sds : dsc + idx[t * k + slot] * Dm * gpr;
    const device ushort* B = shared ? sdb : dbi + idx[t * k + slot] * Dm * gpr;
    float a[ROWS];
    kl_qmv2_rows<4, 16, ROWS, 64>(W, S, B, hs + slot * hstride, Dm, K, row0, lane, a);
    const float ws = wts[t * (k + 1) + slot];
    for (int r = 0; r < ROWS; ++r) tot[r] += ws * a[r];
  }
  if (lane == 0)
    for (int r = 0; r < ROWS; ++r)
      if (row0 + r < Dm) y[t * Dm + row0 + r] = x[t * Dm + row0 + r] + tot[r];
"""

# ---- adapted from Kapitsa src/kernels/qmv.metal: qmv2_impl<4, 16, 2, 64, ADD true, NORM false> (qmv2_b4_g64_r2_add);
# grid (N / 16 threadgroups, rows) x 256 threads; token row = grid y (R1 alone: one row)
QMV2_ADD_SRC = r"""
  constexpr int ROWS = 2;
  threadgroup float xs[4096];
  const uint tgp = threadgroup_position_in_grid.x, t = threadgroup_position_in_grid.y;
  const uint tid = thread_index_in_threadgroup, sg = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
  const uint ntg = 256, nsg = 8;
  const uint K = prm[0], N = prm[1];
  const device float* xr = x + t * K;
  for (uint i = tid; i < K / 4; i += ntg) ((threadgroup float4*)xs)[i] = ((const device float4*)xr)[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint row0 = (tgp * nsg + sg) * ROWS;
  if (row0 >= N) return;
  float acc[ROWS];
  kl_qmv2_rows<4, 16, ROWS, 64>(w, s, b, xs, N, K, row0, lane, acc);
  for (int r = 0; r < ROWS; ++r) {
    const uint row = row0 + r;
    if (lane == 0 && row < N) y[t * N + row] = acc[r] + resid[t * N + row];
  }
"""

# K for o_proj / GDN out: qmv2_rows' arithmetic with x read from device memory (as quant.h's qmv2_rows_dx) and NT token
# rows per thread sharing each decoded weight word (the qmvf layout applied to qmv2); NT 2, ROWS 4, 4 simdgroups.
QMV2_ADD_NT_SRC = r"""
  const uint tgp = threadgroup_position_in_grid.x, tb = threadgroup_position_in_grid.y;
  const uint sg = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
  const uint nsg = NSG;
  const uint K = prm[0], N = prm[1], R = prm[2];
  constexpr int BITS = 4, VPL = 16, GS = 64, PER_WORD = 8, WORDS = 2;
  constexpr uint MASK = 15u;
  const uint row0 = (tgp * nsg + sg) * ROWS;
  if (row0 >= N) return;
  const uint wpr = K / PER_WORD, gpr = K / GS;
  uint rows[ROWS];
  float acc[ROWS][NT];
  for (int r = 0; r < ROWS; ++r) {
    rows[r] = min(row0 + r, N - 1);
    for (int t = 0; t < NT; ++t) acc[r][t] = 0.0f;
  }
  uint xo[NT];
  for (int t = 0; t < NT; ++t) xo[t] = min(tb * NT + uint(t), R - 1) * K;
  for (uint k0 = lane * VPL; k0 < K; k0 += 32 * VPL) {
    float xv[NT][VPL];
    float xsum[NT];
    for (int t = 0; t < NT; ++t) {
      xsum[t] = 0.0f;
      for (int i = 0; i < VPL / 4; ++i) {
        float4 v = ((const device float4*)(x + xo[t] + k0))[i];
        xsum[t] += v.x + v.y + v.z + v.w;
        for (int e = 0; e < 4; ++e) xv[t][i * 4 + e] = v[e] * float(1.0f / float(1u << (BITS * ((i * 4 + e) % PER_WORD))));
      }
    }
    const uint g = k0 / GS, wo = k0 / PER_WORD;
    for (int r = 0; r < ROWS; ++r) {
      const device uint32_t* wp = w + rows[r] * wpr + wo;
      uint2 v = *(const device uint2*)wp;
      uint32_t wv[WORDS] = {v.x, v.y};
      float d[NT];
      for (int t = 0; t < NT; ++t) d[t] = 0.0f;
      for (int i = 0; i < WORDS; ++i)
        for (int j = 0; j < PER_WORD; ++j) {
          const float wf = float(wv[i] & (MASK << (BITS * j)));
          for (int t = 0; t < NT; ++t) d[t] += wf * xv[t][i * PER_WORD + j];
        }
      const float sc = kl_bf(s[rows[r] * gpr + g]), bi = kl_bf(b[rows[r] * gpr + g]);
      for (int t = 0; t < NT; ++t) acc[r][t] += sc * d[t] + bi * xsum[t];
    }
  }
  for (int t = 0; t < NT; ++t)
    for (int r = 0; r < ROWS; ++r) {
      float v = simd_sum(acc[r][t]);
      const uint row = row0 + r, tt = tb * NT + t;
      if (lane == 0 && row < N && tt < R) y[tt * N + row] = v + resid[tt * N + row];
    }
"""

# ---- adapted from Kapitsa src/kernels/verify8.metal: x8_prep2_impl<false> (norm, prepu 1, write_t 1, no lo / xn) ----
PREP_SRC = r"""
  threadgroup half stg[8][256];
  threadgroup float inv_s[8];
  const uint tg = threadgroup_position_in_grid.x;
  const uint tid = thread_index_in_threadgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  const uint lane = thread_index_in_simdgroup;
  const uint K = prm[0], R = prm[1];
  const float eps = epsv[0];
  const uint t = sg;
  const bool live = t < R;
  const device float* xr = x + t * K;
  if (NORM) {
    float ss = 0.0f;
    if (live && K % 512 == 0) {
      float4 a0 = float4(0.0f), a1 = float4(0.0f), a2 = float4(0.0f), a3 = float4(0.0f);
      for (uint i = lane; i < K / 4; i += 128) {
        const float4 v0 = ((const device float4*)xr)[i], v1 = ((const device float4*)xr)[i + 32];
        const float4 v2 = ((const device float4*)xr)[i + 64], v3 = ((const device float4*)xr)[i + 96];
        a0 = fma(v0, v0, a0);
        a1 = fma(v1, v1, a1);
        a2 = fma(v2, v2, a2);
        a3 = fma(v3, v3, a3);
      }
      const float4 a = (a0 + a1) + (a2 + a3);
      ss = (a.x + a.y) + (a.z + a.w);
    } else if (live) {
      for (uint i = lane; i < K / 4; i += 32) {
        const float4 v = ((const device float4*)xr)[i];
        ss += dot(v, v);
      }
    }
    ss = simd_sum(ss);
    if (lane == 0) inv_s[t] = metal::precise::rsqrt(ss / float(K) + eps);
  }
  const uint k0 = tg * 256 + lane * 8;
  float v[8];
  if (live) {
    const float4 a = *(const device float4*)(xr + k0), b = *(const device float4*)(xr + k0 + 4);
    v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
  } else {
    for (int i = 0; i < 8; ++i) v[i] = 0.0f;
  }
  if (NORM) {
    simdgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = inv_s[t];
    for (int i = 0; i < 8; ++i) v[i] = v[i] * inv * as_type<float>(uint(nw[k0 + i]) << 16);
  }
  float sh = 0.0f;
  for (int i = 0; i < 8; ++i) {
    const half h = half(v[i]);
    stg[t][lane * 8 + i] = h;
    sh += float(h);
  }
  sh += simd_shuffle_xor(sh, 1u);
  sh += simd_shuffle_xor(sh, 2u);
  sh += simd_shuffle_xor(sh, 4u);
  if ((lane & 7) == 0) sums[(tg * 4 + lane / 8) * 8 + t] = sh;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint gi = tid / 64, l2 = (tid % 64) / 2, hf = tid % 2;
  const uint q = l2 >> 2;
  const uint fm = (q & 4) | ((l2 >> 1) & 3), fn = ((q & 2) << 1) | ((l2 & 1) << 1);
  {
    half o[8];
    for (int jj = 0; jj < 8; ++jj) {
      const uint j = hf * 8 + jj, f = j / 2, e = j % 2;
      const uint kk = 16 * (fm / 2) + 8 * (f / 4) + (f % 4) + 4 * (fm % 2);
      o[jj] = stg[fn + e][gi * 64 + kk];
    }
    device half* dst = table + ((tg * 4 + gi) * 32 + l2) * 16 + hf * 8;
    *(device uint4*)dst = uint4(as_type<uint>(half2(o[0], o[1])), as_type<uint>(half2(o[2], o[3])),
                                as_type<uint>(half2(o[4], o[5])), as_type<uint>(half2(o[6], o[7])));
  }
  {
    const uint tt = tid / 32, s0 = (tid % 32) * 8, gb = s0 & ~63u;
    half o[8];
    for (int i = 0; i < 8; ++i) {
      const uint sl = (s0 + i) & 63;
      const uint c = (sl >> 2) & 3, jb = (sl >> 4) & 3, kq = sl & 3;
      o[i] = stg[tt][gb + 16 * c + 8 * (jb >> 1) + 2 * kq + (jb & 1)];
    }
    *(device uint4*)(table_t + tt * K + tg * 256 + s0) =
        uint4(as_type<uint>(half2(o[0], o[1])), as_type<uint>(half2(o[2], o[3])), as_type<uint>(half2(o[4], o[5])),
              as_type<uint>(half2(o[6], o[7])));
  }
"""

# ---- adapted from Kapitsa src/kernels/verify8.metal: moe8e_gu_t<2, 8>, moe8e_down_t<2>, moe8_combine ----
MOE8E_HEADER = COMMON + r"""
constant constexpr uint kV8 = 8;
constant constexpr uint kMT = 72;
constant constexpr float kV8Sub = 16777216.0f;
inline uint2 kl_v8_lane(uint lane) {
  const uint q = lane >> 2;
  return uint2((q & 4) | ((lane >> 1) & 3), ((q & 2) << 1) | ((lane & 1) << 1));
}
inline uint kl_v8_lane_of(uint fm, uint fn) {
  return ((fm >> 2) << 4) | ((fn >> 2) << 3) | ((fm & 3) << 1) | ((fn >> 1) & 1);
}
inline uint2 kl_v8_fj(uint kk) { return uint2(4 * ((kk >> 3) & 1) + (kk & 3), 2 * (kk >> 4) + ((kk >> 2) & 1)); }
inline uint kl_v8_tab(uint g, uint kk, uint t) {
  const uint2 fj = kl_v8_fj(kk);
  return (g * 32 + kl_v8_lane_of(fj.y, t & 6u)) * 16 + 2 * fj.x + (t & 1u);
}
inline constexpr uint kl_e_word(uint m, uint s) { return 2 * (s / 2) + m / 4; }
inline constexpr uint kl_e_nib(uint m, uint s) { return (m % 4) + 4 * (s % 2); }
inline constexpr uint kl_e_k(uint m, uint s) { return 8 * kl_e_word(m, s) + kl_e_nib(m, s); }
"""
MOE8E_GU_SRC = r"""
  constexpr uint RT = 2, NSG = 8, CPS = 64 / NSG, P = CPS / 2, LRT = 1;
  constexpr uint gpr = 32, wpr = 256;
  threadgroup half2 xa[32 * 32], xb[32 * 32];
  threadgroup float dots[NSG * 2 * 2 * RT * 32];
  threadgroup float hv[64 * RT];
  threadgroup float red[8 * kV8];
  threadgroup float tsx[RT * 32];
  const uint2 tg = uint2(threadgroup_position_in_grid.x, threadgroup_position_in_grid.y);
  const uint tid = thread_index_in_threadgroup, sg = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
  const uint N = prm[1], w_stride = prm[2], s_stride = prm[3];
  const uint e = sm[tg.y];
  uint t[RT];
  uint nr = 0;
  for (uint i = 0; i < RT; ++i) {
    t[i] = sm[kMT + RT * tg.y + i];
    nr += t[i] != 0xffffffffu ? 1u : 0u;
  }
  for (uint i = 1; i < RT; ++i) t[i] = i < nr ? t[i] : t[0];
  if (tid < 32)
    for (uint r = 0; r < RT; ++r) tsx[r * 32 + tid] = sums[tid * kV8 + t[r]];
  for (uint i = tid; i < 32 * 32; i += 32 * NSG) {
    const uint sp = i / 32, g = i % 32, m = sp / 4, j = sp % 4, ka = kl_e_k(m, 2 * j), kb = kl_e_k(m, 2 * j + 1);
    half2 a, b;
    for (uint r = 0; r < RT; ++r) {
      a[r] = table[kl_v8_tab(g, ka, t[r])];
      b[r] = table[kl_v8_tab(g, kb, t[r])];
    }
    xa[i] = a;
    xb[i] = b;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const bool shr = e == 0xffffffffu;
  const device uint32_t* G = shr ? SWg : Wg + e * w_stride;
  const device uint32_t* U = shr ? SWu : Wu + e * w_stride;
  const device ushort *Gs = shr ? SSg : Sg + e * s_stride, *Gb = shr ? SBg : Bg + e * s_stride;
  const device ushort *Us = shr ? SSu : Su + e * s_stride, *Ub = shr ? SBu : Bu + e * s_stride;
  threadgroup float* dt = dots + sg * (2 * 2 * RT * 32);
  const uint eslice = lane & 1, etrow = (lane >> 1) & (RT - 1), emat = (lane >> (1 + LRT)) & 1,
             ecol = (lane >> (2 + LRT)) & 1;
  for (uint pp = 0; pp < P; ++pp) {
    const uint c0 = tg.x * 64 + sg * CPS + pp * 2;
    uint4 w[4][2];
    for (int r = 0; r < 4; ++r) {
      const device uint4* src = (const device uint4*)((r < 2 ? G : U) + (c0 + (r & 1)) * wpr + lane * 8);
      w[r][0] = src[0];
      w[r][1] = src[1];
    }
    const uint en = c0 + ecol;
    const device uint4* esp = (const device uint4*)((emat ? Us : Gs) + en * gpr + eslice * 16);
    const device uint4* ebp = (const device uint4*)((emat ? Ub : Gb) + en * gpr + eslice * 16);
    const uint4 es0 = esp[0], es1 = esp[1], eb0 = ebp[0], eb1 = ebp[1];
    float d[4][RT];
    for (int r = 0; r < 4; ++r)
      for (uint q = 0; q < RT; ++q) d[r][q] = 0.0f;
    for (uint mh = 0; mh < 2; ++mh) {
      uint4 wv[4];
      for (int r = 0; r < 4; ++r)
        wv[r] = mh ? uint4(w[r][0].y, w[r][0].w, w[r][1].y, w[r][1].w) : uint4(w[r][0].x, w[r][0].z, w[r][1].x, w[r][1].z);
      for (uint m4 = 0; m4 < 4; ++m4) {
        const uint m = mh * 4 + m4, sh = 4 * m4;
        for (uint j = 0; j < 4; ++j) {
          const half2 x0 = xa[(m * 4 + j) * 32 + lane], x1 = xb[(m * 4 + j) * 32 + lane];
          for (int r = 0; r < 4; ++r) {
            const half2 c = as_type<half2>((wv[r][j] >> sh) & 0x000F000Fu);
            for (uint q = 0; q < RT; ++q) d[r][q] = fma(float(c.x), float(x0[q]), d[r][q]);
            for (uint q = 0; q < RT; ++q) d[r][q] = fma(float(c.y), float(x1[q]), d[r][q]);
          }
        }
      }
    }
    for (int r = 0; r < 4; ++r)
      for (uint q = 0; q < RT; ++q) dt[(((r & 1) * 2 + (r >> 1)) * RT + q) * 32 + lane] = d[r][q];
    simdgroup_barrier(mem_flags::mem_threadgroup);
    {
      const threadgroup float* dd = dt + ((ecol * 2 + emat) * RT + etrow) * 32 + eslice * 16;
      const threadgroup float* sx = tsx + etrow * 32 + eslice * 16;
      float acc = 0.0f;
      for (uint i = 0; i < 16; ++i) {
        const uint sw = (i < 8 ? es0 : es1)[(i % 8) / 2], bw = (i < 8 ? eb0 : eb1)[(i % 8) / 2];
        const ushort sb = ushort(i & 1 ? sw >> 16 : sw & 0xffffu), bb = ushort(i & 1 ? bw >> 16 : bw & 0xffffu);
        acc = fma(dd[i], kl_bf(sb) * kV8Sub, acc);
        acc = fma(sx[i], kl_bf(bb), acc);
      }
      acc += simd_shuffle_xor(acc, 1u);
      const float o = simd_shuffle_xor(acc, 1u << (1 + LRT));
      const half h = half(kl_silu(emat ? o : acc) * (emat ? acc : o));
      if (lane < 8 * RT && emat == 0 && eslice == 0) {
        hv[(sg * CPS + pp * 2 + ecol) * RT + etrow] = float(h);
        if (etrow < nr) hs[(RT * tg.y + etrow) * N + en] = float(h);
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  {
    const uint2 rc = kl_v8_lane(lane);
    const uint fm = rc.x, cl = rc.y >> 1;
    constexpr uint NC = CPS / 8 * RT;
    for (uint c0 = 0; c0 < NC; c0 += 4) {
      const uint c = c0 + cl, ci = c / RT, trow = c % RT;
      float ps = c < NC ? hv[(sg * CPS + ci * 8 + fm) * RT + trow] : 0.0f;
      ps += simd_shuffle_xor(ps, 2u);
      ps += simd_shuffle_xor(ps, 4u);
      ps += simd_shuffle_xor(ps, 16u);
      if (fm == 0 && c < NC) red[(sg * (CPS / 8) + ci) * kV8 + trow] = ps;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0 && lane < RT) {
    float s = red[lane];
    for (uint i = 1; i < 8; ++i) s += red[i * kV8 + lane];
    if (lane < nr) hsum[(RT * tg.y + lane) * (N / 64) + tg.x] = s;
  }
"""
MOE8E_DOWN_SRC = r"""
  constexpr uint RT = 2, gpr = 8, wpr = 64, NSG = 4;
  threadgroup half2 xa[32 * 8], xb[32 * 8];
  threadgroup float dots[4 * 16 * RT * 8];
  threadgroup float tsx[RT * 8];
  const uint2 tg = uint2(threadgroup_position_in_grid.x, threadgroup_position_in_grid.y);
  const uint tid = thread_index_in_threadgroup, sg = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
  const uint K = prm[0], N = prm[1], w_stride = prm[2], s_stride = prm[3], k = prm[4];
  const uint e = sm[tg.y];
  uint t[RT], jj[RT];
  uint nr = 0;
  for (uint i = 0; i < RT; ++i) {
    t[i] = sm[kMT + RT * tg.y + i];
    nr += t[i] != 0xffffffffu ? 1u : 0u;
  }
  for (uint i = 1; i < RT; ++i) t[i] = i < nr ? t[i] : t[0];
  for (uint i = 0; i < RT; ++i) jj[i] = sm[kMT + kMT * RT + RT * tg.y + i];
  if (tid < 8)
    for (uint r = 0; r < RT; ++r) tsx[r * 8 + tid] = hsum[(RT * tg.y + (r < nr ? r : 0)) * gpr + tid];
  for (uint i = tid; i < 32 * 8; i += 32 * NSG) {
    const uint sp = i / 8, g = i % 8, m = sp / 4, j = sp % 4;
    const uint ka = 64 * g + kl_e_k(m, 2 * j), kb = 64 * g + kl_e_k(m, 2 * j + 1);
    half2 a, b;
    for (uint r = 0; r < RT; ++r) {
      const device float* hr = hs + (RT * tg.y + (r < nr ? r : 0)) * K;
      a[r] = half(hr[ka]);
      b[r] = half(hr[kb]);
    }
    xa[i] = a;
    xb[i] = b;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const bool shr = e == 0xffffffffu;
  const device uint32_t* Wb = shr ? SW : W + e * w_stride;
  const device ushort* Sb = shr ? SS : S + e * s_stride;
  const device ushort* Bb = shr ? SB : B + e * s_stride;
  const uint c4 = lane >> 3, g = lane & 7;
  threadgroup float* dt = dots + sg * (16 * RT * 8);
  for (uint pp = 0; pp < 2; ++pp) {
    const uint n0 = tg.x * 128 + sg * 32 + pp * 16;
    uint4 w[4][2];
    for (int c = 0; c < 4; ++c) {
      const device uint4* src = (const device uint4*)(Wb + (n0 + 4 * c + c4) * wpr + g * 8);
      w[c][0] = src[0];
      w[c][1] = src[1];
    }
    const uint en = n0 + (lane >> 1);
    const uint4 es = *(const device uint4*)(Sb + en * gpr), eb = *(const device uint4*)(Bb + en * gpr);
    float d[4][RT];
    for (int c = 0; c < 4; ++c)
      for (uint q = 0; q < RT; ++q) d[c][q] = 0.0f;
    for (uint mh = 0; mh < 2; ++mh) {
      uint4 wv[4];
      for (int c = 0; c < 4; ++c)
        wv[c] = mh ? uint4(w[c][0].y, w[c][0].w, w[c][1].y, w[c][1].w) : uint4(w[c][0].x, w[c][0].z, w[c][1].x, w[c][1].z);
      for (uint m4 = 0; m4 < 4; ++m4) {
        const uint m = mh * 4 + m4, sh = 4 * m4;
        for (uint j = 0; j < 4; ++j) {
          const half2 x0 = xa[(m * 4 + j) * 8 + g], x1 = xb[(m * 4 + j) * 8 + g];
          for (int c = 0; c < 4; ++c) {
            const half2 a = as_type<half2>(((wv[c][j] >> sh) & 0x000F000Fu) | 0x64006400u);
            for (uint q = 0; q < RT; ++q) d[c][q] = fma(float(a.x), float(x0[q]), d[c][q]);
            for (uint q = 0; q < RT; ++q) d[c][q] = fma(float(a.y), float(x1[q]), d[c][q]);
          }
        }
      }
    }
    for (int c = 0; c < 4; ++c)
      for (uint q = 0; q < RT; ++q) dt[((4 * c + c4) * RT + q) * 8 + g] = d[c][q];
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint u = 0; u < RT / 2; ++u) {
      const uint col = lane >> 1, trow = (lane & 1) + 2 * u;
      const threadgroup float* dd = dt + (col * RT + trow) * 8;
      const threadgroup float* hsr = tsx + trow * 8;
      float acc = 0.0f;
      for (uint gi = 0; gi < gpr; ++gi) {
        const uint sw = es[gi / 2], bw = eb[gi / 2];
        const float s = kl_bf(ushort(gi & 1 ? sw >> 16 : sw & 0xffffu)), bi = kl_bf(ushort(gi & 1 ? bw >> 16 : bw & 0xffffu));
        acc = fma(dd[gi], s, acc);
        acc = fma(hsr[gi], fma(-1024.0f, s, bi), acc);
      }
      const uint tr = (lane & 1) ? t[1 + 2 * u] : t[2 * u], js = (lane & 1) ? jj[1 + 2 * u] : jj[2 * u];
      if (trow < nr) {
        if (shr) sh_out[tr * N + en] = acc;
        else out[(tr * k + js) * N + en] = acc;
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }
"""
COMBINE_SRC = r"""
  const uint t = threadgroup_position_in_grid.y, i = threadgroup_position_in_grid.x * 64 + thread_position_in_threadgroup.x;
  const uint N = prm[0], k = prm[1], n4 = N / 4;
  if (i >= n4) return;
  const device float4* o4 = (const device float4*)out;
  const device float4* s4 = (const device float4*)sh_out;
  float4 acc = float4(0.0f);
  for (uint j = 0; j < k; ++j) acc += wts[t * (k + 1) + j] * o4[(t * k + j) * n4 + i];
  acc += wts[t * (k + 1) + k] * s4[t * n4 + i];
  ((device float4*)y)[t * n4 + i] = ((const device float4*)x)[t * n4 + i] + acc;
"""

# ---- adapted from Kapitsa src/kernels/tensor/verify8t.metal: qmm8t_impl<NT, PF, 8>, one segment, one 8-row tile ----
QMM8T_HEADER = r"""
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
"""
QMM8T_SRC = r"""
  constexpr uint MAXKS = 8;
  threadgroup float red[(MAXKS - 1) * (NT / 2) * 32];
  const uint tg = threadgroup_position_in_grid.x;
  const uint sg = simdgroup_index_in_threadgroup;
  const uint lane = thread_index_in_simdgroup;
  const uint K = prm[0], R = prm[1], ks = prm[2], N = prm[3];
  const uint n0 = tg * NT, gpr = K / 64, wpr = K / 8;
  const uint gper = (gpr + ks - 1) / ks, g0 = min(sg * gper, gpr), g1 = min(g0 + gper, gpr);
  const uint c = (lane & 1) | (((lane >> 3) & 1) << 1), r = ((lane >> 1) & 3) | (((lane >> 4) & 1) << 2);
  constexpr auto desc = matmul2d_descriptor(NT, 8, 64, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroup> op;
  auto Lw = op.template get_left_input_cooperative_tensor<uint8_t, half, float>();
  auto tX = tensor<device half, dextents<int32_t, 2>, tensor_inline>((device half*)T, dextents<int32_t, 2>(K, 8));
  auto Rx0 = tX.template slice<64, 8>(0, 0);
  auto acc = op.template get_destination_cooperative_tensor<decltype(Lw), decltype(Rx0), float>();
  for (uint16_t i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;
  constexpr int NJ = NT / 8;
  const device uint* wp[NJ];
  const device ushort* sp[NJ];
  const device ushort* bp[NJ];
  for (int j = 0; j < NJ; ++j) {
    const uint row = min(n0 + r + 8 * j, N - 1);
    wp[j] = w + row * wpr + 2 * c;
    sp[j] = s + row * gpr;
    bp[j] = b + row * gpr;
  }
  const uint t0 = 4 * (lane & 1);
  uint2 wn[NJ];
  ushort sn[NJ], bn[NJ];
  auto load = [&](uint g) {
    for (int j = 0; j < NJ; ++j) {
      wn[j] = *(const device uint2*)(wp[j] + g * 8);
      sn[j] = sp[j][g];
      bn[j] = bp[j][g];
    }
  };
  if (PF && g0 < g1) load(g0);
  for (uint g = g0; g < g1; ++g) {
    if (!PF) load(g);
    uint2 wq[NJ];
    ushort sc[NJ], bc[NJ];
    for (int j = 0; j < NJ; ++j) wq[j] = wn[j], sc[j] = sn[j], bc[j] = bn[j];
    if (PF && g + 1 < g1) load(g + 1);
    for (int j = 0; j < NJ; ++j)
      for (int jb = 0; jb < 4; ++jb) {
        const uint u = ((jb >> 1 ? wq[j].y : wq[j].x) >> (4 * (jb & 1))) & 0x0F0F0F0Fu;
        const int i = (j >> 1) * 32 + jb * 8 + (j & 1) * 4;
        Lw[i] = uchar(u);
        Lw[i + 1] = uchar(u >> 8);
        Lw[i + 2] = uchar(u >> 16);
        Lw[i + 3] = uchar(u >> 24);
      }
    auto Rx = tX.template slice<64, 8>(g * 64, 0);
    auto d = op.template get_destination_cooperative_tensor<decltype(Lw), decltype(Rx), float>();
    op.run(Lw, Rx, d);
    const float4 sx = *(const device float4*)(sums + g * 8 + t0);
    for (int j = 0; j < NJ; ++j) {
      const float sj = as_type<float>(uint(sc[j]) << 16), bj = as_type<float>(uint(bc[j]) << 16);
      for (int kk = 0; kk < 4; ++kk) acc[j * 4 + kk] = fma(sj, d[j * 4 + kk], fma(bj, sx[kk], acc[j * 4 + kk]));
    }
  }
  if (ks > 1) {
    constexpr uint cap = NT / 2;
    if (sg)
      for (uint i = 0; i < cap; ++i) red[((sg - 1) * cap + i) * 32 + lane] = acc[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg) return;
    for (uint j = 1; j < ks; ++j)
      for (uint i = 0; i < cap; ++i) acc[i] += red[((j - 1) * cap + i) * 32 + lane];
  }
  if ((lane >> 3) & 1) return;
  for (int j = 0; j < NJ; ++j) {
    const uint n = n0 + r + 8 * j;
    if (n >= N) continue;
    for (int kk = 0; kk < 4; ++kk) {
      const uint tt = t0 + kk;
      if (tt < R) y[tt * N + n] = ADD ? resid[tt * N + n] + acc[j * 4 + kk] : acc[j * 4 + kk];
    }
  }
"""

_QMVF = mx.fast.metal_kernel(name="kl_qmvf", input_names=["w", "s", "b", "x", "nw", "prm", "epsv"],
                             output_names=["y", "xn"], source=QMVF_BODY, header=QMVF_HEADER, compile_options=MATH)
_GU1 = mx.fast.metal_kernel(name="kl_moe_gu3", input_names=["gw", "gsc", "gbi", "uw", "usc", "ubi", "sgw", "sgs", "sgb",
                                                            "suw", "sus", "sub", "xn", "idx", "prm"],
                            output_names=["h"], source=MOE_GU_R1, header=MOE_R1_HEADER, compile_options=MATH)
_DN1 = mx.fast.metal_kernel(name="kl_moe_down2", input_names=["dw", "dsc", "dbi", "sdw", "sds", "sdb", "h", "idx", "wts",
                                                              "x", "prm"],
                            output_names=["y"], source=MOE_DOWN_R1, header=MOE_R1_HEADER, compile_options=MATH)
_QMV2A = mx.fast.metal_kernel(name="kl_qmv2_add", input_names=["w", "s", "b", "x", "resid", "prm"], output_names=["y"],
                              source=QMV2_ADD_SRC, header=MOE_R1_HEADER, compile_options=MATH)
_QMV2AN = mx.fast.metal_kernel(name="kl_qmv2_add_nt", input_names=["w", "s", "b", "x", "resid", "prm"],
                               output_names=["y"], source=QMV2_ADD_NT_SRC, header=COMMON, compile_options=MATH)
_PREP = mx.fast.metal_kernel(name="kl_x8_prep2", input_names=["x", "nw", "prm", "epsv"],
                             output_names=["table", "sums", "table_t"], source=PREP_SRC, compile_options=MATH)
_GUE = mx.fast.metal_kernel(name="kl_moe8e_gu", input_names=["Wg", "Sg", "Bg", "Wu", "Su", "Bu", "table", "sums", "sm",
                                                             "SWg", "SSg", "SBg", "SWu", "SSu", "SBu", "prm"],
                            output_names=["hs", "hsum"], source=MOE8E_GU_SRC, header=MOE8E_HEADER, compile_options=MATH)
_DNE = mx.fast.metal_kernel(name="kl_moe8e_down", input_names=["W", "S", "B", "hs", "hsum", "sm", "SW", "SS", "SB", "prm"],
                            output_names=["out", "sh_out"], source=MOE8E_DOWN_SRC, header=MOE8E_HEADER,
                            compile_options=MATH)
_CMB = mx.fast.metal_kernel(name="kl_moe8_combine", input_names=["out", "sh_out", "wts", "x", "prm"],
                            output_names=["y"], source=COMBINE_SRC, compile_options=MATH)
_QMM8T = mx.fast.metal_kernel(name="kl_qmm8t", input_names=["w", "s", "b", "T", "sums", "resid", "prm"],
                              output_names=["y"],
                              source=QMM8T_SRC, header=QMM8T_HEADER, compile_options=MATH)
_EPS = mx.array([EPS], dtype=mx.float32)


def _prm(*v):
    return mx.array(list(v), dtype=mx.uint32)


# ------------------------------------------------------------------------------------------------ arms
def qmvf(x, L, bits, nt=1, rows=4, xn=False):
    """qmvf source: nt token rows per thread, rows weight rows per simdgroup, 2 simdgroups; R1 = (1, 4)."""
    R, N = x.shape[0], L["N"]
    out = _QMVF(inputs=[L["w"], L["s"], L["b"], x, L["nw"], _prm(D, R, N), _EPS],
                template=[("BITS", bits), ("NT", nt), ("ROWS", rows), ("XN", 1 if xn else 0)],
                grid=(64 * (-(-R // nt)), -(-N // (2 * rows)), 1), threadgroup=(64, 1, 1),
                output_shapes=[(R, N), (R, D) if xn else (1,)], output_dtypes=[mx.float32, mx.float32])
    return out if xn else out[0]


def prep(x, nw, norm=True):
    R, K = x.shape
    xp = x if R == 8 else mx.concatenate([x, mx.zeros((8 - R, K), mx.float32)], 0)
    return _PREP(inputs=[xp, nw, _prm(K, R), _EPS], template=[("NORM", 1 if norm else 0)],
                 grid=(256 * (K // 256), 1, 1), threadgroup=(256, 1, 1),
                 output_shapes=[(K * 8,), (K // 64 * 8,), (8 * K,)],
                 output_dtypes=[mx.float16, mx.float32, mx.float16])


def head_r1(x, L):
    return qmvf(x, L, 4)


def head_k(x, L):
    return qmvf(x, L, 4) if x.shape[0] <= 3 else qmvf(x, L, 4, 2, 8)


def qmm8t(x, L, tt, sums, resid=None):
    """qmm8t with Kapitsa's v8t_plan for N x K (one segment, one 8-row tile)."""
    R, K = x.shape
    N = L["N"]
    nt = 32 if (N + 15) // 16 > 8192 else 16
    gpr = K // 64
    ks = 4 if 32 < gpr <= 64 else 8
    pf = 1 if (nt == 32 or gpr > 32) else 0
    while ks > 1 and gpr < ks:
        ks //= 2
    return _QMM8T(inputs=[L["w"], L["s"], L["b"], tt, sums, resid if resid is not None else sums, _prm(K, R, ks, N)],
                  template=[("NT", nt), ("PF", pf), ("ADD", 0 if resid is None else 1)],
                  grid=(32 * ks * (-(-N // nt)), 1, 1), threadgroup=(32 * ks, 1, 1),
                  output_shapes=[(R, N)], output_dtypes=[mx.float32])[0]


def head_r2(x, L):
    """x8_prep2 (the norm folded into the half table) + qmm8t; LM head: qmm8tp_b4_nt32, ks 8."""
    _, sums, tt = prep(x, L["nw"])
    return qmm8t(x, L, tt, sums)


def oproj_r1(x, L, resid):
    """qmv2_b4_g64_r2_add, token rows on grid y (one row: R1 alone)."""
    R, K = x.shape
    N = L["N"]
    return _QMV2A(inputs=[L["w"], L["s"], L["b"], x, resid, _prm(K, N)], grid=(256 * (N // 16), R, 1),
                  threadgroup=(256, 1, 1), output_shapes=[(R, N)], output_dtypes=[mx.float32])[0]


def oproj_k(x, L, resid, nt=2, rows=4, nsg=4):
    """K for o_proj / GDN out: R1 on the grid for M <= 3; 2 token rows per thread, 4 weight rows x 4 simdgroups."""
    R, K = x.shape
    N = L["N"]
    if R <= 3:
        return oproj_r1(x, L, resid)
    return _QMV2AN(inputs=[L["w"], L["s"], L["b"], x, resid, _prm(K, N, R)],
                   template=[("NT", nt), ("ROWS", rows), ("NSG", nsg)],
                   grid=(32 * nsg * (-(-N // (rows * nsg))), -(-R // nt), 1), threadgroup=(32 * nsg, 1, 1),
                   output_shapes=[(R, N)], output_dtypes=[mx.float32])[0]


def oproj_r2(x, L, resid):
    """x8_prep2 without norm (the attention o_proj's table; GDN out's comes from gdn_gnorm_x8t) + qmm8tp_b4_nt16 ks 4
    with the residual add."""
    _, sums, tt = prep(x, L["nw"], norm=False)
    return qmm8t(x, L, tt, sums, resid)


def router_r1(x, L):
    return qmvf(x, L, 8, xn=True)


def router_k(x, L):
    """K for the router: R1's source on an M-row grid (2 token rows per thread with 8 weight rows per simdgroup is
    also bitwise equal but slower on this tiny matrix: arm k_nt2 of the bench)."""
    return qmvf(x, L, 8, xn=True)


def router_k_nt2(x, L):
    return qmvf(x, L, 8, 2, 8, xn=True)


def moe_r1(xn, x, idx, wts, E):
    """Kapitsa's step: moe_gate_up3_r2 then moe_down2, token rows on grid z / y (one row: R1 alone)."""
    M = x.shape[0]
    p = _prm(D, MI, MI, TOPK)
    h = _GU1(inputs=[E["gw"], E["gs"], E["gb"], E["uw"], E["us"], E["ub"], E["sgw"], E["sgs"], E["sgb"], E["suw"],
                     E["sus"], E["sub"], xn, idx, p],
             grid=(256 * (MI // 16), TOPK + 1, M), threadgroup=(256, 1, 1),
             output_shapes=[(M, TOPK + 1, MI)], output_dtypes=[mx.float32])[0]
    return _DN1(inputs=[E["dw"], E["ds"], E["db"], E["sdw"], E["sds"], E["sdb"], h, idx, wts, x, p],
                grid=(256 * (D // 16), M, 1), threadgroup=(256, 1, 1),
                output_shapes=[(M, D)], output_dtypes=[mx.float32])[0]


def tiles_of(idx_np):
    """moe8_plan with emu 2, split 8, shs 1 on an M5: every routed (expert, rows) as 2-row tiles, the shared expert as
    ceil(M / 2) 2-row tiles. Returns the sm list (experts, rows, slots; kMT = 72) and the tile count."""
    M = idx_np.shape[0]
    by_e = {}
    for t in range(M):
        for j in range(TOPK):
            by_e.setdefault(int(idx_np[t, j]), []).append((t, j))
    tiles = []
    for e in sorted(by_e):
        rs = by_e[e]
        for i in range(0, len(rs), 2):
            tiles.append((e, rs[i:i + 2]))
    sh = [(t, 0) for t in range(M)]
    for i in range(0, M, 2):
        tiles.append((0xFFFFFFFF, sh[i:i + 2]))
    S = len(tiles)
    assert S <= MT
    sm = np.full(5 * MT, 0xFFFFFFFF, dtype=np.uint32)
    for s, (e, rs) in enumerate(tiles):
        sm[s] = e
        for i, (t, j) in enumerate(rs):
            sm[MT + 2 * s + i] = t
            sm[3 * MT + 2 * s + i] = j
    return mx.array(sm), S


def moe_r2(x, nw, sm, S, wts, E):
    """Today's verify on an M5: x8_prep2 (post norm, half table) + moe8e_gu_r2 + moe8e_down_r2 + moe8_combine."""
    M = x.shape[0]
    table, sums, _ = prep(x, nw)
    wst, sst = MI * D // 8, MI * (D // 64)
    hs, hsum = _GUE(inputs=[E["gw"], E["gs"], E["gb"], E["uw"], E["us"], E["ub"], table, sums, sm, E["sgw"], E["sgs"],
                            E["sgb"], E["suw"], E["sus"], E["sub"], _prm(D, MI, wst, sst, TOPK)],
                    grid=(256 * (MI // 64), S, 1), threadgroup=(256, 1, 1),
                    output_shapes=[(2 * S, MI), (2 * S, MI // 64)], output_dtypes=[mx.float32, mx.float32])
    dws, dss = D * MI // 8, D * (MI // 64)
    out, sh_out = _DNE(inputs=[E["dw"], E["ds"], E["db"], hs, hsum, sm, E["sdw"], E["sds"], E["sdb"],
                               _prm(MI, D, dws, dss, TOPK)],
                       grid=(128 * (D // 128), S, 1), threadgroup=(128, 1, 1),
                       output_shapes=[(M * TOPK, D), (M, D)], output_dtypes=[mx.float32, mx.float32])
    return _CMB(inputs=[out, sh_out, wts, x, _prm(D, TOPK)], grid=(64 * (D // 256), M, 1), threadgroup=(64, 1, 1),
                output_shapes=[(M, D)], output_dtypes=[mx.float32])[0]


# ------------------------------------------------------------------------------------------------ data
def _bf16_bits(a):
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32).astype(np.uint64)
    u = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    return u.astype(np.uint16)


def _bf16_val(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


def make_q(rng, n, k, bits, lead=()):
    """MLX affine pack: uint32 words (value j of a word at bits BITS*j), bf16 scale and bias per 64-group."""
    w = rng.integers(0, 2**32, size=lead + (n, k * bits // 32), dtype=np.uint64).astype(np.uint32)
    top = 2**bits - 1
    sc = rng.uniform(0.004, 0.03, size=lead + (n, k // 64)).astype(np.float32) * (15.0 / top)
    bi = (-(top / 2) * sc * rng.uniform(0.9, 1.1, size=sc.shape)).astype(np.float32)
    return w, _bf16_bits(sc), _bf16_bits(bi)


def make_dense(seed, n, bits, k=D):
    rng = np.random.default_rng(seed)
    w, s, b = make_q(rng, n, k, bits)
    nw = rng.uniform(0.2, 1.8, size=k).astype(np.float32)
    L = dict(w=mx.array(w), s=mx.array(s), b=mx.array(b), nw=mx.array(_bf16_bits(nw)), N=n, bits=bits, K=k)
    mx.eval(L["w"], L["s"], L["b"], L["nw"])
    return L


def make_experts(seed):
    rng = np.random.default_rng(seed)
    E = {}
    for p, n, k, lead in (("g", MI, D, (NE,)), ("u", MI, D, (NE,)), ("d", D, MI, (NE,)),
                          ("sg", MI, D, ()), ("su", MI, D, ()), ("sd", D, MI, ())):
        w, s, b = make_q(rng, n, k, 4, lead)
        E[p + "w"], E[p + "s"], E[p + "b"] = mx.array(w), mx.array(s), mx.array(b)
    E["nw"] = mx.array(_bf16_bits(rng.uniform(0.2, 1.8, size=D).astype(np.float32)))
    mx.eval(*E.values())
    return E


def make_x(M, seed, dist, D=D):
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal((M, D)) * rng.uniform(0.01, 2.0, size=(M, 1))).astype(np.float32)
    if dist == "edge":
        for m in range(M):
            kind = (seed + m) % 6
            if kind == 0:
                x[m, rng.integers(0, D, 3)] = np.nan
            elif kind == 1:
                x[m, :] = 0.0
            elif kind == 2:
                x[m, ::7] = 3.0e38
            elif kind == 3:
                x[m, :] = rng.standard_normal(D).astype(np.float32) * 1e-41
            elif kind == 4:
                x[m, ::5] = 1.0e18
            else:
                x[m, ::3] = -1e-39
    return mx.array(x)


def make_route(M, seed, mode="uniform"):
    """Synthetic routing, 8 distinct experts per row. "uniform": drawn uniformly (8 rows: ~57 distinct experts);
    "overlap": each slot reuses an expert of the earlier rows with probability 0.4 (8 rows: ~40 distinct, near the
    "51 tiles of 39 experts" an 8-row verify of the 35B gave in Kapitsa's own BENCH_REPLAY note, verify8.metal).
    Softmax-normalized weights, sigmoid shared-expert weight."""
    rng = np.random.default_rng(seed)
    if mode == "uniform":
        idx = np.stack([rng.choice(NE, TOPK, replace=False) for _ in range(M)]).astype(np.uint32)
    else:
        rows, pool = [], []
        for _ in range(M):
            s = []
            while len(s) < TOPK:
                e = int(rng.choice(pool)) if pool and rng.random() < 0.4 else int(rng.integers(NE))
                if e not in s:
                    s.append(e)
            rows.append(s)
            pool = sorted(set(pool) | set(s))
        idx = np.array(rows, dtype=np.uint32)
    lg = rng.standard_normal((M, TOPK)).astype(np.float32)
    w = np.exp(lg) / np.exp(lg).sum(1, keepdims=True)
    sw = 1.0 / (1.0 + np.exp(-rng.standard_normal((M, 1))))
    wts = np.concatenate([w, sw], 1).astype(np.float32)
    return idx, wts


# ------------------------------------------------------------------------------------------------ float64 references
def _deq(w, s, b, bits):
    per = 32 // bits
    q = np.stack([(w >> (bits * j)) & (2**bits - 1) for j in range(per)], -1).reshape(w.shape[:-1] + (-1,))
    return q.astype(np.float64) * np.repeat(_bf16_val(s), 64, -1) + np.repeat(_bf16_val(b), 64, -1)


def norm64(x, nw):
    xs = np.array(x).astype(np.float64)
    inv = 1.0 / np.sqrt((xs * xs).mean(1, keepdims=True) + EPS)
    return xs * inv * _bf16_val(np.array(nw)).astype(np.float64)


def dense_ref64(x, L, norm=True, resid=None):
    W = _deq(np.array(L["w"]), np.array(L["s"]), np.array(L["b"]), L["bits"])
    y = (norm64(x, L["nw"]) if norm else np.array(x).astype(np.float64)) @ W.T
    return y if resid is None else y + np.array(resid).astype(np.float64)


def moe_ref64(x, idx, wts, E):
    xn = norm64(x, E["nw"])
    x64 = np.array(x).astype(np.float64)
    out = x64.copy()
    g = lambda n: np.array(E[n])
    for t in range(x64.shape[0]):
        for j in range(TOPK + 1):
            if j < TOPK:
                e = int(idx[t, j])
                Wg, Wu, Wd = (_deq(g(p + "w")[e], g(p + "s")[e], g(p + "b")[e], 4) for p in ("g", "u", "d"))
            else:
                Wg, Wu, Wd = (_deq(g(p + "w"), g(p + "s"), g(p + "b"), 4) for p in ("sg", "su", "sd"))
            a, u = Wg @ xn[t], Wu @ xn[t]
            out[t] += wts[t, j] * (Wd @ (a / (1 + np.exp(-a)) * u))
    return out


def _bits(a):
    return np.array(a).view(np.uint32)


# ------------------------------------------------------------------------------------------------ check
def _tally(tot, name, y, ref):
    eq = _bits(y) == _bits(ref)
    t = tot.setdefault(name, [0, 0, 0, 0.0])
    t[0] += int(eq.all(1).sum())
    t[1] += eq.shape[0]
    t[2] += int((~eq).sum())
    a, r = np.array(y), np.array(ref)
    fin = np.isfinite(a) & np.isfinite(r)
    if fin.any():
        t[3] = max(t[3], float(np.abs(a - r)[fin].max()))
    return [int(eq.all(1).sum()), eq.shape[0], int((~eq).sum())]


# class -> (N, K, bits, norm folded, residual add)
DENSE = dict(lm_head=(VOCAB, D, 4, True, False), gdn_in=(12352, D, 4, True, False), attn_qkv=(9216, D, 4, True, False),
             o_proj=(D, 4096, 4, False, True), router=(NE + 1, D, 8, True, False))


def dense_arms(cls):
    """R1 (one row), K, the other exact arms, R2 (None where not transcribed) of a dense class; f(x, L, resid)."""
    if cls == "router":
        return dict(r1=lambda x, L, r: router_r1(x, L)[0], kernl=lambda x, L, r: router_k(x, L)[0],
                    k_nt2=lambda x, L, r: router_k_nt2(x, L)[0]), None
    if cls == "o_proj":
        return dict(r1=oproj_r1, kernl=oproj_k, r1_grid=oproj_r1), oproj_r2
    return dict(r1=lambda x, L, r: head_r1(x, L), kernl=lambda x, L, r: head_k(x, L),
                r1_grid=lambda x, L, r: head_r1(x, L)), (lambda x, L, r: head_r2(x, L))


def check_dense(cls, seeds, rows_list):
    """Every row of K (and of R2) against R1 run on that row alone, float32 bit patterns."""
    N, K, bits, norm, add = DENSE[cls]
    exact, r2 = dense_arms(cls)
    tot, tot_rand, lines, acc = {}, {}, [], []
    for ws in range(2):
        L = make_dense(7000 + 10 * ws + len(cls), N, bits, K)
        for seed in range(seeds):
            for dist in ("randn", "edge"):
                for M in rows_list:
                    x = make_x(M, 1000 * ws + 100 * seed + M + (7 if dist == "edge" else 0), dist, K)
                    resid = make_x(M, 5000 + 100 * seed + M, "randn", N) if add else None
                    ref = mx.concatenate([exact["r1"](x[i:i + 1], L, None if resid is None else resid[i:i + 1])
                                          for i in range(M)], 0)
                    arms = {n: f(x, L, resid) for n, f in exact.items() if n != "r1"}
                    if r2 is not None:
                        arms["r2"] = r2(x, L, resid)
                    extra = {}
                    if cls == "router":   # the normalized row the step hands to the experts
                        extra["kernl_xn"] = (router_k(x, L)[1],
                                             mx.concatenate([router_r1(x[i:i + 1], L)[1] for i in range(M)], 0))
                    mx.eval(ref, *arms.values(), *[v[0] for v in extra.values()], *[v[1] for v in extra.values()])
                    line = dict(ws=ws, seed=seed, dist=dist, M=M)
                    for n, y in list(arms.items()) + [(n, v[0]) for n, v in extra.items()]:
                        rr = extra[n][1] if n in extra else ref
                        line[n] = _tally(tot, n, y, rr)
                        if dist == "randn":
                            _tally(tot_rand, n, y, rr)
                    lines.append(line)
                    if dist == "randn" and seed == 0:
                        y64 = dense_ref64(x, L, norm, resid)
                        a = dict(ws=ws, seed=seed, M=M, max_abs_ref=float(np.abs(y64).max()))
                        for n, y in [("r1", ref), ("kernl", arms["kernl"])] + ([("r2", arms["r2"])] if "r2" in arms else []):
                            a[n] = float(np.abs(np.array(y).astype(np.float64) - y64).max())
                        acc.append(a)
        del L
        mx.clear_cache()
    return _summ(cls, tot, tot_rand, lines, acc)


def check_moe(seeds, rows_list):
    """MoE block: K (R1's kernels over a row grid) and R2 against R1 run on the row alone (input: the row, its
    routing; R1 / K start from the normalized row R1's router kernel writes, as Kapitsa's step does)."""
    tot, tot_rand, lines, acc = {}, {}, [], []
    E = make_experts(9000)
    LR = make_dense(9001, NE + 1, 8)
    LR["nw"] = E["nw"]
    for seed in range(seeds):
        for dist, mode in (("randn", "uniform"), ("randn", "overlap"), ("edge", "uniform")):
            for M in rows_list:
                x = make_x(M, 100 * seed + M + (7 if dist == "edge" else 0) + (3 if mode == "overlap" else 0), dist)
                idx_np, wts_np = make_route(M, 50 * seed + M, mode)
                idx, wts = mx.array(idx_np), mx.array(wts_np)
                xn = mx.concatenate([router_r1(x[i:i + 1], LR)[1] for i in range(M)], 0)
                ref = mx.concatenate([moe_r1(xn[i:i + 1], x[i:i + 1], idx[i:i + 1], wts[i:i + 1], E)
                                      for i in range(M)], 0)
                sm, S = tiles_of(idx_np)
                arms = dict(kernl=moe_r1(xn, x, idx, wts, E), r2=moe_r2(x, E["nw"], sm, S, wts, E))
                mx.eval(ref, *arms.values())
                line = dict(seed=seed, dist=dist, routing=mode, M=M, tiles=S, distinct_experts=len(set(idx_np.ravel().tolist())))
                for n, y in arms.items():
                    line[n] = _tally(tot, n, y, ref)
                    if dist == "randn":
                        _tally(tot_rand, n, y, ref)
                lines.append(line)
                if dist == "randn" and mode == "uniform" and seed < 2:
                    y64 = moe_ref64(x, idx_np, wts_np, E)
                    x64 = np.array(x).astype(np.float64)
                    a = dict(seed=seed, M=M, max_abs_block=float(np.abs(y64 - x64).max()))
                    for n, y in (("r1", ref), ("kernl", arms["kernl"]), ("r2", arms["r2"])):
                        a[n] = float(np.abs(np.array(y).astype(np.float64) - y64).max())
                    acc.append(a)
    del E
    mx.clear_cache()
    return _summ("moe", tot, tot_rand, lines, acc)


def _summ(cls, tot, tot_rand, lines, acc):
    out = dict(lines=lines, accuracy=acc, totals={}, totals_random_inputs={})
    for key, T in (("totals", tot), ("totals_random_inputs", tot_rand)):
        for n, v in T.items():
            out[key][n] = dict(rows_equal=v[0], rows=v[1], elements_differing=v[2], max_abs_diff_finite=v[3])
    print(f"check {cls}: rows bitwise equal to R1 run on the row alone")
    for n, v in out["totals"].items():
        r = out["totals_random_inputs"].get(n, {})
        print(f"  {n:11s} all inputs {v['rows_equal']}/{v['rows']} (elements differing {v['elements_differing']}); "
              f"random inputs {r.get('rows_equal')}/{r.get('rows')}, max |diff| {r.get('max_abs_diff_finite', 0):.3g}",
              flush=True)
    if acc:
        keys = [k for k in ("r1", "kernl", "r2") if k in acc[0]]
        print("  max abs error vs float64: " + ", ".join(f"{k} {max(a[k] for a in acc):.3g}" for k in keys))
    return out


# ------------------------------------------------------------------------------------------------ bench
def _chip():
    try:
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    except Exception:
        return "?"


def _power():
    try:
        return subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout.strip().replace("\n", " | ")
    except Exception:
        return "?"


def _interleave(runs, rounds, seed, base="kernl"):
    for r in runs.values():
        r(); r(); r()
    ts = {n: [] for n in runs}
    rr = random.Random(seed)
    for _ in range(rounds):
        order = list(runs)
        rr.shuffle(order)
        for n in order:
            ts[n].append(runs[n]())
    med = {n: statistics.median(v) for n, v in ts.items()}
    ratio = {n: statistics.median([a / b for a, b in zip(ts[n], ts[base])]) for n in runs if n != base}
    return med, ratio, ts


def _timed(chain, step):
    def run():
        y = None
        for i in range(chain):
            y = step(i, y)
        t0 = time.perf_counter_ns()
        mx.eval(y)
        return (time.perf_counter_ns() - t0) / chain / 1e3
    return run


def bench_dense(cls, rounds, rows_list):
    N, K, bits, norm, add = DENSE[cls]
    exact, r2 = dense_arms(cls)
    lbytes = N * K * bits // 8 + 4 * N * (K // 64)
    nd = int(max(2, min(40, 1.0 * 2**30 // lbytes)))
    chain = max(4, min(40, nd))
    sets = [make_dense(7500 + i, N, bits, K) for i in range(nd)]
    arms = {n: f for n, f in exact.items() if n != "r1"}
    r1 = exact["r1"]
    arms["r1_calls"] = lambda x, L, r: mx.concatenate([r1(x[i:i + 1], L, None if r is None else r[i:i + 1])
                                                       for i in range(x.shape[0])], 0)
    if r2 is not None:
        arms["r2"] = r2
    res = []
    print(f"bench {cls} ({N} x {K}, {bits}-bit): {rounds} rounds, chain {chain} over {nd} weight sets of "
          f"{lbytes / 2**20:.2f} MiB; median us per product; ratio = median over rounds of (arm / kernl)")
    for M in rows_list:
        x0 = make_x(M, 4242 + M, "randn", K)
        r0 = make_x(M, 4343 + M, "randn", N) if add else None

        def mk(f):
            def step(i, y):
                if add:      # the residual carries the chain
                    return f(x0, sets[i % nd], r0 if y is None else y)
                xin = x0 if y is None else y[:, :K]
                return f(xin, sets[i % nd], None)
            return _timed(chain, step)
        med, ratio, ts = _interleave({n: mk(f) for n, f in arms.items()}, rounds, M)
        res.append(dict(M=M, median_us=med, ratio_arm_over_kernl=ratio, samples_us=ts))
        print(f"  M={M} " + "  ".join(f"{n} {v:8.1f}" for n, v in med.items()) + " | " +
              "  ".join(f"{n}/kernl {v:.2f}" for n, v in ratio.items()), flush=True)
    del sets
    mx.clear_cache()
    return res


def bench_moe(rounds, rows_list, chain=16):
    E = make_experts(9500)
    res = []
    for mode in ("uniform", "overlap"):
        res += _bench_moe_mode(E, mode, rounds, rows_list, chain)
    del E
    mx.clear_cache()
    return res


def _bench_moe_mode(E, mode, rounds, rows_list, chain):
    res = []
    print(f"bench moe ({mode} routing): {rounds} rounds, chain {chain} (fresh routing per product); median us per block; "
          f"ratio = median over rounds of (arm / kernl)")
    for M in rows_list:
        x0 = make_x(M, 4343 + M, "randn")
        routes = []
        for i in range(chain):
            idx_np, wts_np = make_route(M, 777 * M + i, mode)
            sm, S = tiles_of(idx_np)
            routes.append((mx.array(idx_np), mx.array(wts_np), sm, S, len(set(idx_np.ravel().tolist()))))
        mx.eval(*[r[0] for r in routes], *[r[1] for r in routes], *[r[2] for r in routes])

        def k_step(i, y):
            x = x0 if y is None else y
            idx, wts = routes[i][0], routes[i][1]
            return moe_r1(x, x, idx, wts, E)

        def calls_step(i, y):
            x = x0 if y is None else y
            idx, wts = routes[i][0], routes[i][1]
            return mx.concatenate([moe_r1(x[j:j + 1], x[j:j + 1], idx[j:j + 1], wts[j:j + 1], E) for j in range(M)], 0)

        def r2_step(i, y):
            x = x0 if y is None else y
            idx, wts, sm, S, _ = routes[i]
            return moe_r2(x, E["nw"], sm, S, wts, E)
        runs = dict(kernl=_timed(chain, k_step), r1_calls=_timed(chain, calls_step), r2=_timed(chain, r2_step))
        med, ratio, ts = _interleave(runs, rounds, M)
        tiles = statistics.mean(r[3] for r in routes)
        dist = statistics.mean(r[4] for r in routes)
        res.append(dict(M=M, routing=mode, median_us=med, ratio_arm_over_kernl=ratio, samples_us=ts, mean_tiles=tiles,
                        mean_distinct_experts=dist))
        print(f"  M={M} (distinct experts {dist:.1f}, tiles {tiles:.1f}) " +
              "  ".join(f"{n} {v:8.1f}" for n, v in med.items()) + " | " +
              "  ".join(f"{n}/kernl {v:.2f}" for n, v in ratio.items()), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(description="One Kapitsa verify cycle, op by op: R1 / R2 / K check and A/B")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--classes", default="gdn_in,attn_qkv,o_proj,router,moe,lm_head")
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--check-rows", default="1,2,3,4,5,6,7,8")
    ap.add_argument("--rows", default="1,2,4,8")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    if not (a.check or a.bench):
        a.check = a.bench = True
    classes = a.classes.split(",")
    rounds, seeds = (5, 1) if a.quick else (a.rounds, a.seeds)
    info = dict(chip=_chip(), macos=platform.mac_ver()[0], mlx=mx.__version__, date=time.strftime("%Y-%m-%d %H:%M:%S"),
                power_start=_power(), rounds=rounds, seeds=seeds, classes=classes,
                kapitsa_commit_read="ea7625f5a41992980c3adb5f6a4542cee7ff7416")
    print(f"{info['chip']}, macOS {info['macos']}, MLX {info['mlx']}; {info['power_start']}", flush=True)
    res = dict(info=info, check={}, bench={})
    crows = [int(v) for v in a.check_rows.split(",")]
    brows = [int(v) for v in a.rows.split(",")]
    for c in classes:
        if a.check:
            res["check"][c] = check_moe(seeds, crows) if c == "moe" else check_dense(c, seeds, crows)
        if a.bench:
            res["bench"][c] = bench_moe(rounds, brows) if c == "moe" else bench_dense(c, rounds, brows)
        if a.json:
            json.dump(res, open(a.json, "w"), indent=1)
    res["info"]["power_end"] = _power()
    res["info"]["date_end"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
