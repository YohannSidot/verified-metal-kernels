import math
import functools
import mlx.core as mx

HEADER = """
#include <metal_simdgroup_matrix>
using namespace metal;
#define UNR _Pragma("clang loop unroll(full)")
"""

HEADER_MPP = """
#include <metal_simdgroup_matrix>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
#define UNR _Pragma("clang loop unroll(full)")
"""

# ---------------------------------------------------------------- v1 fallback (simdgroup)
PARTIAL_SRC = """
const uint tid = thread_position_in_threadgroup.x;
const uint sg = simdgroup_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const int split = threadgroup_position_in_grid.x;
const int h = threadgroup_position_in_grid.y;
const int zz = threadgroup_position_in_grid.z;
const int b = zz / NRG;
const int rg = zz % NRG;
constexpr int NT = NSG * 32;
constexpr int KS = 136;
constexpr int R = KD * G;
threadgroup half Ks[32 * KS];
threadgroup half Vs[32 * KS];

const ulong hb = ((ulong)(b * HKV + h)) * (ulong)L * 128ul;
const device half* kp = k + hb;
const device half* vp = v + hb;

const int fm = ((lane >> 4) & 1) * 4 + ((lane >> 1) & 3);
const int fn = ((lane >> 3) & 1) * 4 + (lane & 1) * 2;
const int r = rg * NSG * 8 + (int)sg * 8 + fm;

const bool rvalid = r < R;
const int rr = rvalid ? r : (R - 1);
const int qi = rr / G;
const int qg = rr % G;
const uint mybits = rvalid ? (uint)bits[b * KD + qi] : 0xFFFFFFFFu;
const float sc = rvalid ? 0.12751743074602467f : 0.0f;

simdgroup_float8x8 Qm[16];
{
  const device half* qp = q + (((ulong)(b * KD + qi)) * HQ + h * G + qg) * 128ul + fn;
  UNR for (int db = 0; db < 16; ++db) {
    thread float2& e = reinterpret_cast<thread float2&>(Qm[db].thread_elements());
    e[0] = sc * float(qp[db * 8]);
    e[1] = sc * float(qp[db * 8 + 1]);
  }
}

simdgroup_float8x8 O[16];
UNR for (int nd = 0; nd < 16; ++nd) O[nd] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
float m = -1.0e30f;
float l = 0.0f;

const int blk0 = split * BPS;
const int blk1 = min(blk0 + BPS, NB);

uint4 rk[PER];
uint4 rv[PER];

{
  const int t0 = blk0 * 32;
  UNR for (int p = 0; p < PER; ++p) {
    const int idx = (int)tid + p * NT;
    rk[p] = uint4(0); rv[p] = uint4(0);
    if (idx < 512) {
      const int t = t0 + (idx >> 4);
      const int c = idx & 15;
      if (t < L) {
        rk[p] = ((const device uint4*)(kp + (ulong)t * 128ul))[c];
        rv[p] = ((const device uint4*)(vp + (ulong)t * 128ul))[c];
      }
    }
  }
}

for (int blk = blk0; blk < blk1; ++blk) {
  threadgroup_barrier(mem_flags::mem_threadgroup);
  UNR for (int p = 0; p < PER; ++p) {
    const int idx = (int)tid + p * NT;
    if (idx < 512) {
      const int key = idx >> 4;
      const int c = idx & 15;
      *((threadgroup uint4*)(Ks + key * KS + c * 8)) = rk[p];
      *((threadgroup uint4*)(Vs + key * KS + c * 8)) = rv[p];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (blk + 1 < blk1) {
    const int tn = (blk + 1) * 32;
    UNR for (int p = 0; p < PER; ++p) {
      const int idx = (int)tid + p * NT;
      rk[p] = uint4(0); rv[p] = uint4(0);
      if (idx < 512) {
        const int t = tn + (idx >> 4);
        const int c = idx & 15;
        if (t < L) {
          rk[p] = ((const device uint4*)(kp + (ulong)t * 128ul))[c];
          rv[p] = ((const device uint4*)(vp + (ulong)t * 128ul))[c];
        }
      }
    }
  }

  const int t0 = blk * 32;
  simdgroup_float8x8 S[4];
  UNR for (int n = 0; n < 4; ++n) S[n] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  UNR for (int db = 0; db < 16; ++db) {
    UNR for (int n = 0; n < 4; ++n) {
      simdgroup_half8x8 kt;
      simdgroup_load(kt, Ks + n * 8 * KS + db * 8, KS, ulong2(0, 0), true);
      simdgroup_multiply_accumulate(S[n], Qm[db], kt, S[n]);
    }
  }

  float sv[8];
  UNR for (int n = 0; n < 4; ++n) {
    thread float2& e = reinterpret_cast<thread float2&>(S[n].thread_elements());
    sv[2 * n] = e[0];
    sv[2 * n + 1] = e[1];
  }
  if (t0 + 32 > L - KD) {
    UNR for (int e = 0; e < 8; ++e) {
      const int t = t0 + (e >> 1) * 8 + fn + (e & 1);
      bool ok = true;
      if (t >= L) ok = false;
      else if (t >= L - KD) ok = ((mybits >> (uint)(t - (L - KD))) & 1u) != 0u;
      if (!ok) sv[e] = -3.0e38f;
    }
  }
  float mx = sv[0];
  UNR for (int e = 1; e < 8; ++e) mx = max(mx, sv[e]);
  mx = max(mx, simd_shuffle_xor(mx, 1));
  mx = max(mx, simd_shuffle_xor(mx, 8));
  const float mn = max(m, mx);
  const float fac = exp2(m - mn);
  float rs = 0.0f;
  UNR for (int e = 0; e < 8; ++e) { sv[e] = exp2(sv[e] - mn); rs += sv[e]; }
  rs += simd_shuffle_xor(rs, 1);
  rs += simd_shuffle_xor(rs, 8);
  l = l * fac + rs;
  m = mn;
  UNR for (int n = 0; n < 4; ++n) {
    thread float2& e = reinterpret_cast<thread float2&>(S[n].thread_elements());
    e[0] = sv[2 * n];
    e[1] = sv[2 * n + 1];
  }
  UNR for (int nd = 0; nd < 16; ++nd) {
    thread float2& e = reinterpret_cast<thread float2&>(O[nd].thread_elements());
    e[0] *= fac;
    e[1] *= fac;
  }
  UNR for (int kb = 0; kb < 4; ++kb) {
    UNR for (int nd = 0; nd < 16; ++nd) {
      simdgroup_half8x8 vt;
      simdgroup_load(vt, Vs + kb * 8 * KS + nd * 8, KS);
      simdgroup_multiply_accumulate(O[nd], S[kb], vt, O[nd]);
    }
  }
}

const ulong prow = ((ulong)((b * HKV + h) * NSPLIT + split)) * (ulong)RP + (ulong)r;
device float* op = Opart + prow * 128ul + fn;
UNR for (int nd = 0; nd < 16; ++nd) {
  thread float2& e = reinterpret_cast<thread float2&>(O[nd].thread_elements());
  *((device float2*)(op + nd * 8)) = float2(e[0], e[1]);
}
if ((lane & 9u) == 0u) {
  ML[prow * 2ul] = m;
  ML[prow * 2ul + 1ul] = l;
}
"""

# ---------------------------------------------------------------- q permutation pre-pass
QPERM_SRC = """
const uint c = thread_position_in_grid.x;
const int r = thread_position_in_grid.y;
const int bh = thread_position_in_grid.z;
const int b = bh / HKV;
const int h = bh % HKV;
uint4 val = uint4(0);
if (r < KD * G) {
  const int i = r / G;
  const int g = r % G;
  val = ((const device uint4*)(q + (((ulong)(b * KD + i)) * HQ + h * G + g) * 128ul))[c];
}
((device uint4*)qp)[((ulong)bh * RP + r) * 16ul + c] = val;
"""

# ---------------------------------------------------------------- row map of the PV destination layout
ROWMAP_SRC = """
const uint tid = thread_position_in_threadgroup.x;
constexpr int NT = NSG * 32;
threadgroup half Ps[RB * BK];
for (int idx = (int)tid; idx < RB * BK; idx += NT) {
  const int r = idx / BK;
  const int c = idx % BK;
  Ps[idx] = (c == 0) ? half(r) : half(0.0f);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
using DT = tensor<device half, dextents<int32_t, 2>, tensor_inline>;
auto tP = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(Ps, dextents<int32_t, 2>(BK, RB));
constexpr auto dPV = mpp::tensor_ops::matmul2d_descriptor(RB, 128, BK, false, false, false,
    mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
mpp::tensor_ops::matmul2d<dPV, execution_simdgroups<NSG>> pv;
DT tOnes((device half*)ones, dextents<int32_t, 2>(128, BK));
auto cR = pv.template get_destination_cooperative_tensor<decltype(tP), DT, float>();
for (int i = 0; i < (int)cR.get_capacity(); ++i) cR[i] = 0.0f;
pv.run(tP, tOnes, cR);
for (int i = 0; i < (int)cR.get_capacity() && i < 64; ++i) {
  int rv_ = int(cR[i] + 0.5f);
  rmap[tid * 64 + i] = clamp(rv_, 0, RB - 1);
}
"""

# ---------------------------------------------------------------- MPP tensor-op kernel
MPP_SRC = """
const uint tid = thread_position_in_threadgroup.x;
const int split = threadgroup_position_in_grid.x;
const int h = threadgroup_position_in_grid.y;
const int zz = threadgroup_position_in_grid.z;
const int b = zz / NRG;
const int rg = zz % NRG;
constexpr int NT = NSG * 32;
constexpr int TPR = NT / RB;
constexpr int R = KD * G;
constexpr int CPT = BK / TPR;
const int rbase = rg * RB;

threadgroup float Ss[RB * BK];
threadgroup half Ps[RB * BK];
threadgroup float Fs[RB];
threadgroup int Fl[2];

const ulong hb = ((ulong)(b * HKV + h)) * (ulong)L * 128ul;
device half* kpm = ((device half*)k) + hb;
device half* vpm = ((device half*)v) + hb;
device half* qpm = ((device half*)qp) + ((ulong)(b * HKV + h) * RP + rbase) * 128ul;

if (tid == 0) { Fl[0] = 0; Fl[1] = 0; }

using DT = tensor<device half, dextents<int32_t, 2>, tensor_inline>;
DT tQ(qpm, dextents<int32_t, 2>(128, RB));
auto tS = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(Ss, dextents<int32_t, 2>(BK, RB));
auto tP = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(Ps, dextents<int32_t, 2>(BK, RB));

constexpr auto dQK = mpp::tensor_ops::matmul2d_descriptor(RB, BK, 128, false, true, false,
    mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
constexpr auto dPV = mpp::tensor_ops::matmul2d_descriptor(RB, 128, BK, false, false, false,
    mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
mpp::tensor_ops::matmul2d<dQK, execution_simdgroups<NSG>> qk;
mpp::tensor_ops::matmul2d<dPV, execution_simdgroups<NSG>> pv;

auto cO = pv.template get_destination_cooperative_tensor<decltype(tP), DT, float>();
ushort rowi[64];
for (int i = 0; i < (int)cO.get_capacity(); ++i) {
  cO[i] = 0.0f;
  rowi[i] = (ushort)rmap[tid * 64 + i];
}
threadgroup_barrier(mem_flags::mem_threadgroup);

const int tr = (int)tid / TPR;
const int sub = (int)tid % TPR;
const int grow = rbase + tr;
const uint mybits = (grow < R) ? (uint)bits[b * KD + grow / G] : 0xFFFFFFFFu;
const float qs = 0.12751743074602467f;
float m = -1.0e30f;
float l = 0.0f;
const threadgroup float* srow = Ss + tr * BK;
threadgroup half* prow_ = Ps + tr * BK;

const int blk0 = split * BPS;
const int blk1 = min(blk0 + BPS, NB);
for (int blk = blk0; blk < blk1; ++blk) {
  const int tlo = blk * BK;
  const int t0 = min(tlo, L - BK);
  DT tK(kpm + (ulong)t0 * 128ul, dextents<int32_t, 2>(128, BK));
  DT tV(vpm + (ulong)t0 * 128ul, dextents<int32_t, 2>(128, BK));
  auto cS = qk.template get_destination_cooperative_tensor<DT, DT, float>();
  qk.run(tQ, tK, cS);
  cS.store(tS);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) Fl[(blk + 1) & 1] = 0;

  const bool need_mask = (tlo > t0) || (t0 + BK > L - KD);
  float sv[CPT];
  if (need_mask) {
    UNR for (int j = 0; j < CPT; ++j) {
      const int c = j * TPR + sub;
      const int t = t0 + c;
      bool ok = t >= tlo;
      if (ok && t >= L - KD) ok = ((mybits >> (uint)(t - (L - KD))) & 1u) != 0u;
      sv[j] = ok ? srow[c] * qs : -3.0e38f;
    }
  } else {
    UNR for (int j = 0; j < CPT; ++j) sv[j] = srow[j * TPR + sub] * qs;
  }
  float mx = sv[0];
  UNR for (int j = 1; j < CPT; ++j) mx = max(mx, sv[j]);
  UNR for (int o = 1; o < TPR; o <<= 1) mx = max(mx, simd_shuffle_xor(mx, (ushort)o));
  const float mn = max(m, mx);
  const float fac = exp2(m - mn);
  float rs = 0.0f;
  UNR for (int j = 0; j < CPT; ++j) {
    const float p = exp2(sv[j] - mn);
    rs += p;
    prow_[j * TPR + sub] = half(p);
  }
  UNR for (int o = 1; o < TPR; o <<= 1) rs += simd_shuffle_xor(rs, (ushort)o);
  l = l * fac + rs;
  m = mn;
  if (sub == 0) {
    Fs[tr] = fac;
    if (fac != 1.0f) Fl[blk & 1] = 1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (Fl[blk & 1] != 0) {
    for (int i = 0; i < (int)cO.get_capacity(); ++i) cO[i] *= Fs[rowi[i]];
  }
  pv.run(tP, tV, cO);
}

const ulong prow0 = ((ulong)((b * HKV + h) * NSPLIT + split)) * (ulong)RP + (ulong)rbase;
tensor<device float, dextents<int32_t, 2>, tensor_inline> tOut(Opart + prow0 * 128ul, dextents<int32_t, 2>(128, RB));
cO.store(tOut);
if (sub == 0) {
  ML[(prow0 + tr) * 2ul] = m;
  ML[(prow0 + tr) * 2ul + 1ul] = l;
}
"""

COMBINE_SRC = """
const uint d = thread_position_in_grid.x;
const int r = thread_position_in_grid.y;
const int bh = thread_position_in_grid.z;
const int b = bh / HKV;
const int h = bh % HKV;
const ulong base = (ulong)bh * NSPLIT;
float M = -3.0e38f;
for (int s = 0; s < NSPLIT; ++s) {
  M = max(M, ML[((base + s) * RP + r) * 2ul]);
}
float num = 0.0f, den = 0.0f;
for (int s = 0; s < NSPLIT; ++s) {
  const ulong idx = (base + s) * RP + r;
  const float w = exp2(ML[idx * 2ul] - M);
  num += w * Opart[idx * 128ul + d];
  den += w * ML[idx * 2ul + 1ul];
}
const int i = r / G;
const int g = r % G;
out[(((ulong)b * KD + i) * HQ + h * G + g) * 128ul + d] = half(num / den);
"""

PARTIAL_K = mx.fast.metal_kernel(
    name="spec_verify_partial_v1",
    input_names=["q", "k", "v", "bits"],
    output_names=["Opart", "ML"],
    source=PARTIAL_SRC,
    header=HEADER,
)

QPERM_K = mx.fast.metal_kernel(
    name="spec_verify_qperm_v4",
    input_names=["q"],
    output_names=["qp"],
    source=QPERM_SRC,
    header=HEADER,
)

ROWMAP_K = mx.fast.metal_kernel(
    name="spec_verify_rowmap_v6",
    input_names=["ones"],
    output_names=["rmap"],
    source=ROWMAP_SRC,
    header=HEADER_MPP,
)

MPP_K = mx.fast.metal_kernel(
    name="spec_verify_partial_mpp_v6",
    input_names=["qp", "k", "v", "bits", "rmap"],
    output_names=["Opart", "ML"],
    source=MPP_SRC,
    header=HEADER_MPP,
)

COMBINE_K = mx.fast.metal_kernel(
    name="spec_verify_combine_v1",
    input_names=["Opart", "ML"],
    output_names=["out"],
    source=COMBINE_SRC,
    header=HEADER,
)


@functools.lru_cache(maxsize=None)
def ones_block(BK):
    a = mx.concatenate([mx.ones((128,), dtype=mx.float16),
                        mx.zeros(((BK - 1) * 128,), dtype=mx.float16)])
    return a


@functools.lru_cache(maxsize=None)
def row_map(RB, BK, NSG):
    NT = NSG * 32
    return ROWMAP_K(
        inputs=[ones_block(BK)],
        template=[("RB", RB), ("BK", BK), ("NSG", NSG)],
        grid=(NT, 1, 1),
        threadgroup=(NT, 1, 1),
        output_shapes=[(NT * 64,)],
        output_dtypes=[mx.int32],
        init_value=0,
    )[0]


def combine(part, ml, B, kd, Hq, Hkv, G, R, RP, nsplit):
    return COMBINE_K(
        inputs=[part, ml],
        template=[("HKV", Hkv), ("NSPLIT", nsplit), ("RP", RP), ("KD", kd),
                  ("HQ", Hq), ("G", G)],
        grid=(128, R, B * Hkv),
        threadgroup=(128, 1, 1),
        output_shapes=[(B, kd, Hq, 128)],
        output_dtypes=[mx.float16],
    )[0]


def run_fallback(q, k, v, bits, B, kd, Hq, Hkv, L, G, R):
    nsg_tot = (R + 7) // 8
    nrg = (nsg_tot + 7) // 8
    NSG = (nsg_tot + nrg - 1) // nrg
    RP = NSG * 8 * nrg
    NT = NSG * 32
    PER = (512 + NT - 1) // NT
    NB = (L + 31) // 32
    bh = B * Hkv
    target_tg = max(1, 32768 // NT)
    nsplit = max(1, min((target_tg + bh * nrg - 1) // (bh * nrg), (NB + 15) // 16))
    bps = (NB + nsplit - 1) // nsplit
    nsplit = (NB + bps - 1) // bps
    part, ml = PARTIAL_K(
        inputs=[q, k, v, bits],
        template=[("L", L), ("KD", kd), ("HQ", Hq), ("HKV", Hkv), ("G", G),
                  ("NSG", NSG), ("NRG", nrg), ("NSPLIT", nsplit), ("BPS", bps),
                  ("NB", NB), ("PER", PER), ("RP", RP)],
        grid=(nsplit * NT, Hkv, B * nrg),
        threadgroup=(NT, 1, 1),
        output_shapes=[(bh * nsplit * RP * 128,), (bh * nsplit * RP * 2,)],
        output_dtypes=[mx.float32, mx.float32],
    )
    return combine(part, ml, B, kd, Hq, Hkv, G, R, RP, nsplit)


def kernel(q, k, v, bits):
    B, kd, Hq, D = q.shape
    Hkv, L = k.shape[1], k.shape[2]
    G = Hq // Hkv
    R = kd * G
    if bits.dtype != mx.int32:
        bits = bits.astype(mx.int32)

    if R <= 16:
        RB = 16
    elif R <= 32:
        RB = 32
    else:
        RB = 64
    BK = 4096 // RB
    if L < BK:
        return run_fallback(q, k, v, bits, B, kd, Hq, Hkv, L, G, R)

    NSG = 8
    nrg = (R + RB - 1) // RB
    RP = nrg * RB
    NT = NSG * 32
    NB = (L + BK - 1) // BK
    bh = B * Hkv
    units = bh * nrg
    lo = (128 + units - 1) // units
    hi = (256 + units - 1) // units
    cap_r = int(0.025 * L / RP)
    nsplit = max(lo, min(hi, cap_r))
    min_keys = 4096 if RB == 16 else 512
    min_blocks = max(1, min_keys // BK)
    nsplit = max(1, min(nsplit, (NB + min_blocks - 1) // min_blocks))
    bps = (NB + nsplit - 1) // nsplit
    nsplit = (NB + bps - 1) // bps

    qperm = QPERM_K(
        inputs=[q],
        template=[("KD", kd), ("HQ", Hq), ("HKV", Hkv), ("G", G), ("RP", RP)],
        grid=(16, RP, bh),
        threadgroup=(16, 1, 1),
        output_shapes=[(bh * RP * 128,)],
        output_dtypes=[mx.float16],
    )[0]

    part, ml = MPP_K(
        inputs=[qperm, k, v, bits, row_map(RB, BK, NSG)],
        template=[("L", L), ("KD", kd), ("HQ", Hq), ("HKV", Hkv), ("G", G),
                  ("RB", RB), ("BK", BK), ("NSG", NSG), ("NRG", nrg),
                  ("NSPLIT", nsplit), ("BPS", bps), ("NB", NB), ("RP", RP)],
        grid=(nsplit * NT, Hkv, B * nrg),
        threadgroup=(NT, 1, 1),
        output_shapes=[(bh * nsplit * RP * 128,), (bh * nsplit * RP * 2,)],
        output_dtypes=[mx.float32, mx.float32],
    )
    return combine(part, ml, B, kd, Hq, Hkv, G, R, RP, nsplit)
