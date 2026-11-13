# Copyright 2026 Yohann Sidot (harness); portions adapted from Kapitsa, Copyright 2026 Nace AI.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this
# file except in compliance with the License. The full text is in LICENSE-APACHE-2.0.txt, shipped with this file, and
# at http://www.apache.org/licenses/LICENSE-2.0. Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# either express or implied.
# This file was written by AI models (Claude) under the author's direction.
#
# Portions adapted from Kapitsa (https://github.com/nace-ai/kapitsa, Apache License 2.0), read as text at commit
# ea7625f5a41992980c3adb5f6a4542cee7ff7416, are marked "adapted from Kapitsa". The three Kapitsa source files carry
# no copyright header at that commit; Kapitsa's LICENSE file states "Copyright 2026 Nace AI", retained here and at
# the end of LICENSE-APACHE-2.0.txt. Kapitsa's credit to MLX (MIT License) is reproduced in THIRD_PARTY_NOTICES.txt.
# Changes made to them: transcribed into mx.fast.metal_kernel bodies (MLX's
# generated signatures instead of buffer indices), parameters passed as small arrays, one segment covering the four
# concatenated matrices instead of four segment pointers, nsg a constant (2) instead of
# [[simdgroups_per_threadgroup]], row tiles, the X8 "lo" table, xn output and the prefetching variant removed,
# bfloat read as its 16 raw bits.
#   QMVF_HEADER, QMVF_BODY  <- src/kernels/qmvf.metal (qmvf_core, qmvf_multi; qmvf_b4_t1_r4_norm). Used for R1 and K.
#   R2_PREP_SRC             <- src/kernels/verify8.metal (x8_prep2_impl<false>, v8_lane, v8t_input_of_slot)
#   R2_QMM8T_SRC            <- src/kernels/tensor/verify8t.metal (qmm8t_impl; qmm8t_b4_nt16)
"""Kernl, an 8-row verify whose rows carry the bytes of Kapitsa's one-row step (context: nace-ai/kapitsa #1234).

What it measures.  Qwen3.6-35B-A3B, layer 0, linear-attention input projections qkv | z | b | a
(2048 -> 8192 | 4096 | 32 | 32, MLX affine 4-bit, group 64, bf16 scales and biases), RMSNorm folded in, on 1 to 8
token rows, synthetic weights and inputs.  Arms, all transcribed by us into mx.fast.metal_kernel and compiled with
math_mode "fast" (Kapitsa compiles its libraries with MTLMathModeFast):
  R1        our transcription of Kapitsa's plain step, qmvf_b4_t1_r4_norm (fp32 activations, norm folded in);
  r1_calls  M one-row R1 dispatches (the trivial exact path);  r1_grid: R1 over an M-row grid (one dispatch);
  R2        our transcription of Kapitsa's current 8-row verify on an M5: x8_prep2 (norm, inputs rounded to half,
            group sums, T table) then qmm8t_b4_nt16 (Metal 4 tensor_ops matmul2d, fp32 epilogue per group);
  K         the same qmvf source with 2 token rows per thread and 8 weight rows per simdgroup, one dispatch (M >= 4;
            for M <= 3, r1_grid itself):
            every row bitwise equal to R1 run on that row alone (checked, not proven; see the README).
Modes (MLX and NumPy only):
  --check   rows of K and of r1_grid vs R1 on the row alone, bit for bit; R2 vs R1 (the drift); fp64 error
  --bench   30 interleaved rounds: K vs r1_calls, vs r1_grid, vs R2
  --quick   1 seed, 5 rounds
"""
import argparse
import json
import math
import platform
import random
import statistics
import subprocess
import time

import numpy as np
import mlx.core as mx

KDIM = 2048
SEGS = [("qkv", 8192), ("z", 4096), ("b", 32), ("a", 32)]
NTOT = sum(n for _, n in SEGS)
EPS = 1e-6
MATH = {"math_mode": "fast"}

# ----------------------------------------------------------------------------------------------------------------
# R1: adapted from Kapitsa src/kernels/qmvf.metal (Apache-2.0), qmvf_core / qmvf_multi with BITS 4, NT 1, ROWS 4,
# NORM true, GS 64, 2 simdgroups per threadgroup (Qwen35::enc_qmvf: nt 1, rows 4, nsg 2).
QMVF_HEADER = r"""
template <int NT, int ROWS>
inline __attribute__((always_inline)) void kl_qmvf_core(const device uint32_t* W, const device ushort* S,
    const device ushort* B, const device float* x, uint x_ld, uint t0, uint R, uint K, uint N,
    const device ushort* nw, uint row0, uint lane, thread float (&acc)[ROWS][NT], thread float (&ss)[NT]) {
  constexpr int BITS = 4, GS = 64;
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
      nv[i] = float4(as_type<float>(uint(nw[k]) << 16), as_type<float>(uint(nw[k + 1]) << 16),
                     as_type<float>(uint(nw[k + 2]) << 16), as_type<float>(uint(nw[k + 3]) << 16));
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
      uint2 v = *(const device uint2*)wp;
      wv[0] = v.x; wv[1] = v.y;
      float d[NT];
      for (int t = 0; t < NT; ++t) d[t] = 0.0f;
      for (int i = 0; i < WORDS; ++i)
        for (int j = 0; j < PER_WORD; ++j) {
          const float wf = float(wv[i] & (MASK << (BITS * j)));
          for (int t = 0; t < NT; ++t) d[t] += wf * xv[t][i * PER_WORD + j];
        }
      const float sc = as_type<float>(uint(S[rows[r] * gpr + g]) << 16);
      const float bi = as_type<float>(uint(B[rows[r] * gpr + g]) << 16);
      for (int t = 0; t < NT; ++t) acc[r][t] += sc * d[t] + bi * xs[t];
    }
  }
}
"""

# body shared by R1 (NT = 1) and the "their template at NT = M" variant (KT); ROWS rows per simdgroup, 2 simdgroups
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
  kl_qmvf_core<NT, ROWS>(w, s, b, x, K, t0, R, K, N, nw, row0, lane, acc, ss);
  for (int t = 0; t < NT; ++t) {
    float inv = metal::precise::rsqrt(simd_sum(ss[t]) / float(K) + eps);
    for (int r = 0; r < ROWS; ++r) {
      float v = simd_sum(acc[r][t]);
      v *= inv;
      const uint row = row0 + r;
      if (lane == 0 && row < N && t0 + uint(t) < R) y[(t0 + t) * N + row] = v;
    }
  }
"""

# ----------------------------------------------------------------------------------------------------------------
# R2, step 1: adapted from Kapitsa src/kernels/verify8.metal (Apache-2.0), x8_prep2_impl<false> with norm 1,
# unroll 1 (V8Knobs::prepu), write_lo 0, write_xn 0, write_t 1; grid K / 256 threadgroups x 256 threads.
R2_PREP_SRC = r"""
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
  {
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
  {
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
  // the X8 table of the simdgroup-MMA GEMMs (written by Kapitsa's x8_prep2 in every case; unused by qmm8t)
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
  {  // T table of qmm8t
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

# R2, step 2: adapted from Kapitsa src/kernels/tensor/verify8t.metal (Apache-2.0), qmm8t_impl<16, 0, 8>
# (v8t_plan for 12352 x 2048: nt 16, ks 8, no prefetch); one 8-row tile; grid Ntot / 16 threadgroups x 32 ks threads.
R2_QMM8T_HEADER = r"""
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
"""
R2_QMM8T_SRC = r"""
  constexpr int NT = 16;
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
  for (uint g = g0; g < g1; ++g) {
    uint2 wq[NJ];
    ushort sc[NJ], bc[NJ];
    for (int j = 0; j < NJ; ++j) {
      wq[j] = *(const device uint2*)(wp[j] + g * 8);
      sc[j] = sp[j][g];
      bc[j] = bp[j][g];
    }
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
      if (tt < R) y[tt * N + n] = acc[j * 4 + kk];
    }
  }
"""

_R1 = mx.fast.metal_kernel(name="kl_r1_qmvf", input_names=["w", "s", "b", "x", "nw", "prm", "epsv"],
                           output_names=["y"], source=QMVF_BODY, header=QMVF_HEADER, compile_options=MATH)
_PREP = mx.fast.metal_kernel(name="kl_r2_x8_prep2", input_names=["x", "nw", "prm", "epsv"],
                             output_names=["table", "sums", "table_t"], source=R2_PREP_SRC, compile_options=MATH)
_QMM8T = mx.fast.metal_kernel(name="kl_r2_qmm8t", input_names=["w", "s", "b", "T", "sums", "prm"],
                              output_names=["y"], source=R2_QMM8T_SRC, header=R2_QMM8T_HEADER, compile_options=MATH)

_EPS = mx.array([EPS], dtype=mx.float32)
K_NT, K_ROWS = 2, 8  # K: token rows per thread, weight rows per simdgroup (2 simdgroups per threadgroup)


def _prm(*v):
    return mx.array(list(v), dtype=mx.uint32)


def r1(x, L):
    """R1 over an R-row grid (grid x = R, one token row per threadgroup: per row the same work as R = 1)."""
    R = x.shape[0]
    return _R1(inputs=[L["w"], L["s"], L["b"], x, L["nw"], _prm(KDIM, R, NTOT), _EPS],
               template=[("NT", 1), ("ROWS", 4)], grid=(64 * R, NTOT // 8, 1), threadgroup=(64, 1, 1),
               output_shapes=[(R, NTOT)], output_dtypes=[mx.float32])[0]


def r1_calls(x, L):
    """The trivial exact path: one R1 dispatch per row."""
    return mx.concatenate([r1(x[i:i + 1], L) for i in range(x.shape[0])], 0)


def kernl(x, L, nt=None, rows=None):
    """K: for M >= 4, the same qmvf source as R1 instantiated with NT = 2 token rows per thread and ROWS = 8 weight
    rows per simdgroup, one dispatch for the M rows (ceil(M / 2) token-row groups); for M <= 3, R1 over the M-row grid. Every row bitwise equal to R1 on that row
    alone on our machine (checked by --check; fast-math compiles, so the equality is a property of the compiled
    kernel, not of the source: other NT / ROWS give other bits, see the README)."""
    R = x.shape[0]
    if R <= 3 and nt is None:
        return r1(x, L)        # 1 to 3 rows: Kapitsa's own grid (NT = 1, ROWS = 4) is the fastest exact form we found
    nt = nt or K_NT
    rows = rows or K_ROWS
    return _R1(inputs=[L["w"], L["s"], L["b"], x, L["nw"], _prm(KDIM, R, NTOT), _EPS],
               template=[("NT", nt), ("ROWS", rows)], grid=(64 * (-(-R // nt)), -(-NTOT // (2 * rows)), 1),
               threadgroup=(64, 1, 1), output_shapes=[(R, NTOT)], output_dtypes=[mx.float32])[0]


def r2(x, L):
    """R2: x8_prep2 (norm, half inputs, group sums, T table) then qmm8t on the matrix units; rows padded to 8."""
    R = x.shape[0]
    xp = x if R == 8 else mx.concatenate([x, mx.zeros((8 - R, KDIM), mx.float32)], 0)
    table, sums, tt = _PREP(inputs=[xp, L["nw"], _prm(KDIM, R), _EPS], grid=(256 * (KDIM // 256), 1, 1),
                            threadgroup=(256, 1, 1),
                            output_shapes=[(KDIM * 8,), (KDIM // 64 * 8,), (8 * KDIM,)],
                            output_dtypes=[mx.float16, mx.float32, mx.float16])
    gpr = KDIM // 64
    ks = 4 if 32 < gpr <= 64 else 8
    while ks > 1 and gpr < ks:
        ks //= 2
    return _QMM8T(inputs=[L["w"], L["s"], L["b"], tt, sums, _prm(KDIM, R, ks, NTOT)],
                  grid=(32 * ks * (NTOT // 16), 1, 1), threadgroup=(32 * ks, 1, 1),
                  output_shapes=[(R, NTOT)], output_dtypes=[mx.float32])[0]


# ----------------------------------------------------------------------------------------------------------------
def _bf16_bits(a):
    """float32 array -> bf16 bit patterns (round to nearest even), as uint16."""
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32).astype(np.uint64)
    u = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    return u.astype(np.uint16)


def _bf16_val(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


def make_layer(seed):
    """Synthetic weights stored as in the pack: uint32 words (value j of a word at bits 4j), bf16 scales and biases
    per group of 64 (biases ~ -8 x scale, as MLX's affine quantization gives), bf16 norm weight."""
    rng = np.random.default_rng(1000 + seed)
    w = rng.integers(0, 2**32, size=(NTOT, KDIM // 8), dtype=np.uint64).astype(np.uint32)
    sc = rng.uniform(0.004, 0.03, size=(NTOT, KDIM // 64)).astype(np.float32)
    bi = (-8.0 * sc * rng.uniform(0.9, 1.1, size=sc.shape)).astype(np.float32)
    nw = rng.uniform(0.2, 1.8, size=KDIM).astype(np.float32)
    L = dict(w=mx.array(w), s=mx.array(_bf16_bits(sc)), b=mx.array(_bf16_bits(bi)), nw=mx.array(_bf16_bits(nw)))
    mx.eval(*L.values())
    return L


def make_x(M, seed, dist):
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal((M, KDIM)) * rng.uniform(0.01, 2.0, size=(M, 1))).astype(np.float32)
    if dist == "edge":
        for m in range(M):
            kind = (seed + m) % 6
            if kind == 0:
                x[m, rng.integers(0, KDIM, 3)] = np.nan
            elif kind == 1:
                x[m, :] = 0.0
            elif kind == 2:
                x[m, ::7] = 3.0e38       # large finite (sum of squares overflows)
            elif kind == 3:
                x[m, :] = rng.standard_normal(KDIM).astype(np.float32) * 1e-41   # denormals
            elif kind == 4:
                x[m, ::5] = 1.0e18
            else:
                x[m, ::3] = -1e-39       # denormal subset in a normal row
    return mx.array(x)


def ref64(x, L):
    """NumPy float64 reference of the op (RMSNorm with the bf16 weight, then the affine 4-bit matmul)."""
    xs = np.array(x).astype(np.float64)
    w = np.array(L["w"])
    q = np.stack([(w >> (4 * j)) & 15 for j in range(8)], -1).reshape(NTOT, KDIM).astype(np.float64)
    s = _bf16_val(np.array(L["s"])).astype(np.float64)
    b = _bf16_val(np.array(L["b"])).astype(np.float64)
    W = q * np.repeat(s, 64, 1) + np.repeat(b, 64, 1)
    nw = _bf16_val(np.array(L["nw"])).astype(np.float64)
    inv = 1.0 / np.sqrt((xs * xs).mean(1, keepdims=True) + EPS)
    return (xs * inv * nw) @ W.T


def _bits(a):
    return np.array(a).view(np.uint32)


def seg_slices():
    o, out = 0, []
    for name, n in SEGS:
        out.append((name, slice(o, o + n)))
        o += n
    return out


def check(seeds, rows_list):
    res = dict(rows=[], drift=[], accuracy=[])
    tot = dict(kernl=[0, 0, 0], r1_grid=[0, 0, 0], r2=[0, 0, 0])
    print("check: every row vs R1 run on that row alone, float32 bit patterns (rows equal / rows; elements differing)")
    for seed in range(seeds):
        L = make_layer(seed)
        for dist in ("randn", "edge"):
            for M in rows_list:
                x = make_x(M, 100 * seed + M + (7 if dist == "edge" else 0), dist)
                ref = r1_calls(x, L)
                arms = dict(kernl=kernl(x, L), r1_grid=r1(x, L), r2=r2(x, L))
                mx.eval(ref, *arms.values())
                rb = _bits(ref)
                line = dict(seed=seed, dist=dist, M=M)
                for n, y in arms.items():
                    eq = _bits(y) == rb
                    tot[n][0] += int(eq.all(1).sum())
                    tot[n][1] += M
                    tot[n][2] += int((~eq).sum())
                    line[n] = [int(eq.all(1).sum()), M, int((~eq).sum())]
                    if n == "r2":
                        a, r = np.array(y), np.array(ref)
                        for sname, sl in seg_slices():
                            e = eq[:, sl]
                            fin = np.isfinite(a[:, sl]) & np.isfinite(r[:, sl])
                            d = np.abs(a[:, sl] - r[:, sl])[fin]
                            res["drift"].append(dict(seed=seed, dist=dist, M=M, proj=sname,
                                                     rows_equal=int(e.all(1).sum()), rows=M,
                                                     elements_differing=int((~e).sum()), elements=int(e.size),
                                                     max_abs_diff=float(d.max()) if d.size else None,
                                                     max_abs_value=float(np.abs(r[:, sl])[fin].max()) if d.size else None))
                res["rows"].append(line)
                if dist == "randn":
                    y64 = ref64(x, L)
                    acc = {}
                    for n, y in (("r1", ref), ("kernl", arms["kernl"]), ("r2", arms["r2"])):
                        e = np.abs(np.array(y).astype(np.float64) - y64)
                        acc[n] = dict(max_abs_err=float(e.max()), mean_abs_err=float(e.mean()))
                    acc["max_abs_ref"] = float(np.abs(y64).max())
                    res["accuracy"].append(dict(seed=seed, M=M, **acc))
        del L
        mx.clear_cache()
    res["totals"] = {n: dict(rows_equal=v[0], rows=v[1], elements_differing=v[2]) for n, v in tot.items()}
    for n, v in tot.items():
        print(f"  {n:8s} rows bitwise equal to R1-alone: {v[0]}/{v[1]}  (elements differing: {v[2]})")
    return res


def drift_summary(res):
    print("drift R2 vs R1 (random inputs only), per projection: rows equal / rows, max |diff|, max |value|")
    out = {}
    for sname, _ in seg_slices():
        rs = [d for d in res["drift"] if d["proj"] == sname and d["dist"] == "randn"]
        re_ = sum(d["rows_equal"] for d in rs)
        rr = sum(d["rows"] for d in rs)
        ed = sum(d["elements_differing"] for d in rs)
        el = sum(d["elements"] for d in rs)
        md = max(d["max_abs_diff"] for d in rs)
        mv = max(d["max_abs_value"] for d in rs)
        out[sname] = dict(rows_equal=re_, rows=rr, elements_differing=ed, elements=el, max_abs_diff=md,
                          max_abs_value=mv)
        print(f"  {sname:4s} rows equal {re_}/{rr}, elements differing {ed}/{el}, max |diff| {md:.3g}, "
              f"max |value| {mv:.3g}")
    return out


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


def bench(rounds, rows_list, chain=16):
    """Interleaved A/B: per round, every arm once in shuffled order; one timed eval = a chain of `chain` products
    cycling over distinct weight sets (next input = first 2048 outputs of the previous product); time per product
    in microseconds (includes MLX's per-dispatch overhead, the same for every arm per dispatch)."""
    lbytes = NTOT * KDIM // 2 + 2 * 2 * NTOT * (KDIM // 64)
    nd = int(max(2, min(16, 1.0 * 2**30 // lbytes)))
    layers = [make_layer(100 + i) for i in range(nd)]
    arms = {"kernl": kernl, "r1_calls": r1_calls, "r1_grid": r1, "r2": r2}
    out = []
    print(f"bench: {rounds} interleaved rounds, {nd} weight sets of {lbytes / 2**20:.1f} MiB, chain {chain}; "
          f"median us per product; ratio = median over rounds of (arm time / kernl time)")
    for M in rows_list:
        x0 = make_x(M, 4242 + M, "randn")

        def runner(f):
            def run():
                x = x0
                for i in range(chain):
                    x = f(x, layers[i % nd])[:, :KDIM]
                t0 = time.perf_counter_ns()
                mx.eval(x)
                return (time.perf_counter_ns() - t0) / chain / 1e3
            return run
        runs = {n: runner(f) for n, f in arms.items()}
        for r in runs.values():
            r(); r(); r()
        ts = {n: [] for n in runs}
        rr = random.Random(M)
        for _ in range(rounds):
            order = list(runs)
            rr.shuffle(order)
            for n in order:
                ts[n].append(runs[n]())
        med = {n: statistics.median(v) for n, v in ts.items()}
        ratio = {n: statistics.median([a / b for a, b in zip(ts[n], ts["kernl"])]) for n in runs if n != "kernl"}
        row = dict(M=M, median_us=med, ratio_arm_over_kernl=ratio, samples_us=ts)
        out.append(row)
        print(f"  M={M}  kernl {med['kernl']:7.1f}  r1_calls {med['r1_calls']:7.1f}  r1_grid {med['r1_grid']:7.1f}  "
              + f"r2 {med['r2']:7.1f} | kernl speedup vs r1_calls {ratio['r1_calls']:.2f}x, vs r1_grid "
              f"{ratio['r1_grid']:.2f}x, vs r2 {ratio['r2']:.2f}x", flush=True)
    del layers
    mx.clear_cache()
    return out


def main():
    global K_NT, K_ROWS
    ap = argparse.ArgumentParser(description="Row-exact 8-row verify for Kapitsa's GDN input projections: check, A/B")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--rows", default="1,2,3,4,5,6,7,8")
    ap.add_argument("--k-nt", type=int, default=K_NT)
    ap.add_argument("--k-rows", type=int, default=K_ROWS)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    K_NT, K_ROWS = a.k_nt, a.k_rows
    if not (a.check or a.bench):
        a.check = a.bench = True
    rows_list = [int(v) for v in a.rows.split(",")]
    rounds, seeds = (5, 1) if a.quick else (a.rounds, a.seeds)
    info = dict(chip=_chip(), macos=platform.mac_ver()[0], mlx=mx.__version__, date=time.strftime("%Y-%m-%d %H:%M:%S"),
                power_start=_power(), rounds=rounds, seeds=seeds, rows=rows_list, k_nt=K_NT, k_rows=K_ROWS,
                shapes=dict(K=KDIM, segments=SEGS), kapitsa_commit_read="ea7625f5a41992980c3adb5f6a4542cee7ff7416")
    print(f"{info['chip']}, macOS {info['macos']}, MLX {info['mlx']}; {info['power_start']}")
    res = dict(info=info)
    if a.check:
        c = check(seeds, rows_list)
        c["drift_summary"] = drift_summary(c)
        res["check"] = c
    if a.bench:
        res["bench"] = bench(rounds, rows_list)
    res["info"]["power_end"] = _power()
    res["info"]["date_end"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
