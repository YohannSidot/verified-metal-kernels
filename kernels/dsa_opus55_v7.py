import mlx.core as mx

TOPK = 2048

COMMON = """
inline uint fkey(float f) {
  uint u = as_type<uint>(f);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
inline void hist_add(device atomic_uint* h, uint bin) {
  bool pend = true;
  while (pend) {
    uint lb = simd_broadcast_first(bin);
    bool m = (bin == lb);
    uint c1 = simd_sum(m ? 1u : 0u);
    if (simd_is_first()) atomic_fetch_add_explicit(&h[lb], c1, memory_order_relaxed);
    if (m) pend = false;
  }
}
"""

HEADER = """
#include <metal_simdgroup_matrix>
using namespace metal;
""" + COMMON

INDEX_SRC = """
uint lane = thread_index_in_simdgroup;
uint sg = simdgroup_index_in_threadgroup;
uint tid = thread_position_in_threadgroup.x;
uint tile = threadgroup_position_in_grid.x;
uint b = threadgroup_position_in_grid.y;
threadgroup half iqT[128 * 72];
const device half* iqb = iq + (size_t)b * 8192;
for (uint i = tid; i < 8192; i += 256) {
  uint h = i >> 7; uint d = i & 127;
  iqT[d * 72 + h] = iqb[i];
}
threadgroup_barrier(mem_flags::mem_threadgroup);
uint qid = lane >> 2;
uint fm = (qid & 4) + ((lane >> 1) & 3);
uint fn = (qid & 2) * 2 + (lane & 1) * 2;
uint l0 = tile * 256 + sg * 32;
const device half* ikb = ik + (size_t)b * L * 128;
const device half* rp[4];
#pragma clang loop unroll(full)
for (int i = 0; i < 4; ++i) {
  uint p = min(l0 + 8 * i + fm, (uint)(L - 1));
  rp[i] = ikb + (size_t)p * 128 + fn;
}
simdgroup_float8x8 acc[4][8];
#pragma clang loop unroll(full)
for (int i = 0; i < 4; ++i) {
  #pragma clang loop unroll(full)
  for (int j = 0; j < 8; ++j)
    acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
}
for (uint k0 = 0; k0 < 128; k0 += 8) {
  simdgroup_float8x8 a[4];
  simdgroup_float8x8 bm[8];
  #pragma clang loop unroll(full)
  for (int i = 0; i < 4; ++i) {
    half2 v = *((const device half2*)(rp[i] + k0));
    reinterpret_cast<thread float2&>(a[i].thread_elements()) = float2(v);
  }
  #pragma clang loop unroll(full)
  for (int j = 0; j < 8; ++j) {
    half2 v = *((const threadgroup half2*)(iqT + (k0 + fm) * 72 + 8 * j + fn));
    reinterpret_cast<thread float2&>(bm[j].thread_elements()) = float2(v);
  }
  #pragma clang loop unroll(full)
  for (int i = 0; i < 4; ++i) {
    #pragma clang loop unroll(full)
    for (int j = 0; j < 8; ++j)
      simdgroup_multiply_accumulate(acc[i][j], a[i], bm[j], acc[i][j]);
  }
}
const device float* wb = w + b * 64;
float wv[16];
#pragma clang loop unroll(full)
for (int j = 0; j < 8; ++j) { wv[2 * j] = wb[8 * j + fn]; wv[2 * j + 1] = wb[8 * j + fn + 1]; }
#pragma clang loop unroll(full)
for (int i = 0; i < 4; ++i) {
  float part = 0.0f;
  #pragma clang loop unroll(full)
  for (int j = 0; j < 8; ++j) {
    float2 cv = reinterpret_cast<thread float2&>(acc[i][j].thread_elements());
    part += wv[2 * j] * max(cv.x, 0.0f) + wv[2 * j + 1] * max(cv.y, 0.0f);
  }
  part += simd_shuffle_xor(part, (ushort)1);
  part += simd_shuffle_xor(part, (ushort)8);
  uint p = l0 + 8 * i + fm;
  if ((lane & 9) == 0 && p < (uint)L) {
    atomic_store_explicit(&scores[(size_t)b * L + p], part, memory_order_relaxed);
    hist_add(hist + (size_t)b * 8192, fkey(part) >> 19);
  }
}
"""

SEL_SRC = """
uint b = threadgroup_position_in_grid.x;
uint tid = thread_position_in_threadgroup.x;
uint lane = thread_index_in_simdgroup;
uint sg = simdgroup_index_in_threadgroup;
threadgroup atomic_uint thist[2048];
threadgroup uint ckey[2048];
threadgroup int cidx[2048];
threadgroup uint ssum[64];
threadgroup uint ssel[2];
threadgroup atomic_uint cg[4];
int nl = HL ? (int)lens[b] : L;
int n = nl < 0 ? 0 : (nl > L ? L : nl);
device int* ob = idx + (size_t)b * TOPK;
if (n <= TOPK) {
  for (int i = (int)tid; i < n; i += 1024) ob[i] = i;
  if (tid == 0) cnt[b] = n;
  return;
}
if (tid == 0) cnt[b] = TOPK;
const device float* s = scores + (size_t)b * L;
if (tid < 4) atomic_store_explicit(&cg[tid], 0u, memory_order_relaxed);
bool fast = false;
uint bstar = 0u, above = 0u, ncand = 0u;
if (n == L) {
  const device uint* hb = hist + (size_t)b * 8192;
  uint c8[8];
  uint x = 0u;
  #pragma clang loop unroll(full)
  for (int k = 0; k < 8; ++k) { c8[k] = hb[8191u - 8u * tid - (uint)k]; x += c8[k]; }
  uint inc = simd_prefix_inclusive_sum(x);
  if (lane == 31) ssum[sg] = inc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    uint v = ssum[lane];
    uint e = simd_prefix_exclusive_sum(v);
    ssum[lane] = e;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint run = ssum[sg] + inc - x;
  #pragma clang loop unroll(full)
  for (int k = 0; k < 8; ++k) {
    if (run < (uint)TOPK && (uint)TOPK <= run + c8[k]) { ssel[0] = 8191u - 8u * tid - (uint)k; ssel[1] = run; }
    run += c8[k];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bstar = ssel[0];
  above = ssel[1];
  ncand = hb[bstar];
  fast = ncand <= 2048u;
  threadgroup_barrier(mem_flags::mem_threadgroup);
} else {
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
if (fast) {
  uint kk = (uint)TOPK - above;
  for (int base = 0; base < n; base += 8192) {
    uint keys[8];
    #pragma clang loop unroll(full)
    for (int k = 0; k < 8; ++k) {
      int i = base + k * 1024 + (int)tid;
      keys[k] = (i < n) ? fkey(s[i]) : 0u;
    }
    #pragma clang loop unroll(full)
    for (int k = 0; k < 8; ++k) {
      int i = base + k * 1024 + (int)tid;
      bool valid = i < n;
      uint key = keys[k];
      uint bin = key >> 19;
      bool g = valid && bin > bstar;
      bool e = valid && bin == bstar;
      uint gi = g ? 1u : 0u;
      uint ei = e ? 1u : 0u;
      uint gp = simd_prefix_exclusive_sum(gi);
      uint gtot = simd_sum(gi);
      uint ep = simd_prefix_exclusive_sum(ei);
      uint etot = simd_sum(ei);
      uint gb = 0u, eb = 0u;
      if (lane == 0) {
        if (gtot > 0u) gb = atomic_fetch_add_explicit(&cg[0], gtot, memory_order_relaxed);
        if (etot > 0u) eb = atomic_fetch_add_explicit(&cg[1], etot, memory_order_relaxed);
      }
      gb = simd_broadcast(gb, (ushort)0);
      eb = simd_broadcast(eb, (ushort)0);
      if (g) ob[gb + gp] = i;
      if (e) { uint pos = eb + ep; ckey[pos] = key; cidx[pos] = i; }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint prefix = bstar << 19;
  for (int bit = 18; bit >= 0; --bit) {
    uint t = prefix | (1u << bit);
    uint cl = 0u;
    for (uint j = tid; j < ncand; j += 1024) cl += (ckey[j] >= t) ? 1u : 0u;
    cl = simd_sum(cl);
    uint par = (uint)(bit & 1) * 32u;
    if (lane == 0) ssum[par + sg] = cl;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint tot = simd_sum(ssum[par + lane]);
    if (tot >= kk) prefix = t;
  }
  uint T = prefix;
  uint ngt2;
  {
    uint cl = 0u;
    for (uint j = tid; j < ncand; j += 1024) cl += (ckey[j] > T) ? 1u : 0u;
    cl = simd_sum(cl);
    if (lane == 0) ssum[32 + sg] = cl;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ngt2 = simd_sum(ssum[32 + lane]);
  }
  uint need = kk - ngt2;
  for (uint base = 0; base < ncand; base += 1024) {
    uint j = base + tid;
    bool valid = j < ncand;
    uint key = valid ? ckey[j] : 0u;
    bool g = valid && key > T;
    bool e = valid && key == T;
    uint gi = g ? 1u : 0u;
    uint ei = e ? 1u : 0u;
    uint gp = simd_prefix_exclusive_sum(gi);
    uint gtot = simd_sum(gi);
    uint ep = simd_prefix_exclusive_sum(ei);
    uint etot = simd_sum(ei);
    uint gb = 0u, eb = 0u;
    if (lane == 0) {
      if (gtot > 0u) gb = atomic_fetch_add_explicit(&cg[2], gtot, memory_order_relaxed);
      if (etot > 0u) eb = atomic_fetch_add_explicit(&cg[3], etot, memory_order_relaxed);
    }
    gb = simd_broadcast(gb, (ushort)0);
    eb = simd_broadcast(eb, (ushort)0);
    if (g) ob[above + gb + gp] = cidx[j];
    if (e) { uint pos = eb + ep; if (pos < need) ob[above + ngt2 + pos] = cidx[j]; }
  }
  return;
}
uint prefix = 0u;
uint kk = (uint)TOPK;
for (int pass = 0; pass < 3; ++pass) {
  uint shift = pass == 0 ? 21u : (pass == 1 ? 10u : 0u);
  uint hish = pass == 1 ? 21u : 10u;
  uint dmask = pass == 2 ? 1023u : 2047u;
  for (uint i = tid; i < 2048; i += 1024) atomic_store_explicit(&thist[i], 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = (int)tid; i < n; i += 1024) {
    uint key = fkey(s[i]);
    bool pend = (pass == 0) || ((key >> hish) == prefix);
    uint bin = (key >> shift) & dmask;
    while (pend) {
      uint lb = simd_broadcast_first(bin);
      bool m = (bin == lb);
      uint c1 = simd_sum(m ? 1u : 0u);
      if (simd_is_first()) atomic_fetch_add_explicit(&thist[lb], c1, memory_order_relaxed);
      if (m) pend = false;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint bhi = 2047u - 2u * tid;
  uint blo = 2046u - 2u * tid;
  uint chi = atomic_load_explicit(&thist[bhi], memory_order_relaxed);
  uint clo = atomic_load_explicit(&thist[blo], memory_order_relaxed);
  uint x = chi + clo;
  uint inc = simd_prefix_inclusive_sum(x);
  if (lane == 31) ssum[sg] = inc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    uint v = ssum[lane];
    uint e = simd_prefix_exclusive_sum(v);
    ssum[lane] = e;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint E = ssum[sg] + inc - x;
  if (E < kk && kk <= E + chi) { ssel[0] = bhi; ssel[1] = E; }
  else if (E + chi < kk && kk <= E + x) { ssel[0] = blo; ssel[1] = E + chi; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint bin = ssel[0];
  uint abv = ssel[1];
  if (pass == 0) prefix = bin;
  else if (pass == 1) prefix = (prefix << 11) | bin;
  else prefix = (prefix << 10) | bin;
  kk -= abv;
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
uint T = prefix;
uint ngt = (uint)TOPK - kk;
for (int base = 0; base < n; base += 1024) {
  int i = base + (int)tid;
  bool valid = i < n;
  uint key = valid ? fkey(s[i]) : 0u;
  bool g = valid && key > T;
  bool e = valid && key == T;
  uint gi = g ? 1u : 0u;
  uint ei = e ? 1u : 0u;
  uint gp = simd_prefix_exclusive_sum(gi);
  uint gtot = simd_sum(gi);
  uint ep = simd_prefix_exclusive_sum(ei);
  uint etot = simd_sum(ei);
  uint gb = 0u, eb = 0u;
  if (lane == 0) {
    if (gtot > 0u) gb = atomic_fetch_add_explicit(&cg[0], gtot, memory_order_relaxed);
    if (etot > 0u) eb = atomic_fetch_add_explicit(&cg[1], etot, memory_order_relaxed);
  }
  gb = simd_broadcast(gb, (ushort)0);
  eb = simd_broadcast(eb, (ushort)0);
  if (g) ob[gb + gp] = i;
  if (e) { uint pos = eb + ep; if (pos < kk) ob[ngt + pos] = i; }
}
"""

QK_SRC = """
uint tid = thread_position_in_threadgroup.x;
uint lane = thread_index_in_simdgroup;
uint sg = simdgroup_index_in_threadgroup;
uint nt = threadgroup_position_in_grid.x;
uint mt = threadgroup_position_in_grid.y;
uint b = threadgroup_position_in_grid.z;
threadgroup half As[64 * 40];
threadgroup half Bs[64 * 40];
threadgroup float red[128];
threadgroup float red2[128];
int c = cnt[b];
int n0 = (int)nt * 64;
int m0 = (int)mt * 64;
device float* Sb = S + ((size_t)b * 128 + m0) * NS;
device float* mb = mst + ((size_t)b * 128 + m0) * NT;
device float* lb = lst + ((size_t)b * 128 + m0) * NT;
if (n0 >= c) {
  if (tid < 64) { mb[tid * NT + nt] = -INFINITY; lb[tid * NT + nt] = 0.0f; }
  return;
}
uint r = tid >> 1;
uint cc = (tid & 1) * 16;
int jr = n0 + (int)r;
bool kvalid = jr < c;
int krow = kvalid ? idx[(size_t)b * TOPK + jr] : 0;
const device half* kp = kv + ((size_t)b * L + krow) * 576 + cc;
const device half* qp = q + ((size_t)b * 128 + m0 + r) * 576 + cc;
threadgroup half4* a4 = (threadgroup half4*)(As + r * 40 + cc);
threadgroup half4* b4 = (threadgroup half4*)(Bs + r * 40 + cc);
uint qid = lane >> 2;
uint fm = (qid & 4) + ((lane >> 1) & 3);
uint fn = (qid & 2) * 2 + (lane & 1) * 2;
uint sm = sg >> 1, sn = sg & 1;
simdgroup_float8x8 acc[4][4];
#pragma clang loop unroll(full)
for (int i = 0; i < 4; ++i) {
  #pragma clang loop unroll(full)
  for (int j = 0; j < 4; ++j)
    acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
}
for (int k0 = 0; k0 < 576; k0 += 32) {
  const device half4* q4 = (const device half4*)(qp + k0);
  const device half4* k4 = (const device half4*)(kp + k0);
  #pragma clang loop unroll(full)
  for (int t = 0; t < 4; ++t) {
    a4[t] = q4[t];
    b4[t] = kvalid ? k4[t] : half4(0.0h);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  #pragma clang loop unroll(full)
  for (int kk = 0; kk < 32; kk += 8) {
    simdgroup_float8x8 a[4];
    simdgroup_float8x8 bm[4];
    #pragma clang loop unroll(full)
    for (int i = 0; i < 4; ++i) {
      half2 v = *((const threadgroup half2*)(As + (32 * sm + 8 * i + fm) * 40 + kk + fn));
      reinterpret_cast<thread float2&>(a[i].thread_elements()) = float2(v);
    }
    #pragma clang loop unroll(full)
    for (int j = 0; j < 4; ++j) {
      half x0 = Bs[(32 * sn + 8 * j + fn) * 40 + kk + fm];
      half x1 = Bs[(32 * sn + 8 * j + fn + 1) * 40 + kk + fm];
      reinterpret_cast<thread float2&>(bm[j].thread_elements()) = float2((float)x0, (float)x1);
    }
    #pragma clang loop unroll(full)
    for (int i = 0; i < 4; ++i) {
      #pragma clang loop unroll(full)
      for (int j = 0; j < 4; ++j)
        simdgroup_multiply_accumulate(acc[i][j], a[i], bm[j], acc[i][j]);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
const float SC = 0.07216878364870323f;
float2 v[4][4];
#pragma clang loop unroll(full)
for (int i = 0; i < 4; ++i) {
  #pragma clang loop unroll(full)
  for (int j = 0; j < 4; ++j) {
    v[i][j] = reinterpret_cast<thread float2&>(acc[i][j].thread_elements()) * SC;
    uint row = 32 * sm + 8 * i + fm;
    uint col = (uint)n0 + 32 * sn + 8 * j + fn;
    *((device float2*)(Sb + (size_t)row * NS + col)) = v[i][j];
  }
}
#pragma clang loop unroll(full)
for (int i = 0; i < 4; ++i) {
  float mx = -INFINITY;
  #pragma clang loop unroll(full)
  for (int j = 0; j < 4; ++j) {
    int col = n0 + (int)(32 * sn + 8 * j + fn);
    if (col < c) mx = max(mx, v[i][j].x);
    if (col + 1 < c) mx = max(mx, v[i][j].y);
  }
  mx = max(mx, simd_shuffle_xor(mx, (ushort)1));
  mx = max(mx, simd_shuffle_xor(mx, (ushort)8));
  if ((lane & 9) == 0) red[sn * 64 + 32 * sm + 8 * i + fm] = mx;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma clang loop unroll(full)
for (int i = 0; i < 4; ++i) {
  uint row = 32 * sm + 8 * i + fm;
  float M = max(red[row], red[64 + row]);
  float sumv = 0.0f;
  #pragma clang loop unroll(full)
  for (int j = 0; j < 4; ++j) {
    int col = n0 + (int)(32 * sn + 8 * j + fn);
    if (col < c) sumv += exp(v[i][j].x - M);
    if (col + 1 < c) sumv += exp(v[i][j].y - M);
  }
  sumv += simd_shuffle_xor(sumv, (ushort)1);
  sumv += simd_shuffle_xor(sumv, (ushort)8);
  if ((lane & 9) == 0) red2[sn * 64 + row] = sumv;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (tid < 64) {
  mb[tid * NT + nt] = max(red[tid], red[64 + tid]);
  lb[tid * NT + nt] = red2[tid] + red2[64 + tid];
}
"""

PV_SRC = """
uint tid = thread_position_in_threadgroup.x;
uint lane = thread_index_in_simdgroup;
uint sg = simdgroup_index_in_threadgroup;
uint ntl = threadgroup_position_in_grid.x;
uint mtl = threadgroup_position_in_grid.y;
uint z = threadgroup_position_in_grid.z;
uint b = z / KS;
uint ks = z % KS;
threadgroup float sM[32];
threadgroup float sI[32];
threadgroup float As[32 * 36];
threadgroup half Bs[32 * 72];
int c = cnt[b];
int n0 = (int)ntl * 64;
int m0 = (int)mtl * 32;
if (tid < 32) {
  const device float* mp = mst + ((size_t)b * 128 + m0 + tid) * NT;
  const device float* lp = lst + ((size_t)b * 128 + m0 + tid) * NT;
  float M = -INFINITY;
  for (int t = 0; t < NT; ++t) M = max(M, mp[t]);
  float Ls = 0.0f;
  for (int t = 0; t < NT; ++t) { float lv = lp[t]; if (lv != 0.0f || isnan(lv)) Ls += lv * exp(mp[t] - M); }
  sM[tid] = M;
  sI[tid] = 1.0f / Ls;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
int js = (int)ks * KC;
int je = min(js + KC, c);
uint qid = lane >> 2;
uint fm = (qid & 4) + ((lane >> 1) & 3);
uint fn = (qid & 2) * 2 + (lane & 1) * 2;
uint sm = sg >> 1, sn = sg & 1;
simdgroup_float8x8 acc[2][4];
#pragma clang loop unroll(full)
for (int i = 0; i < 2; ++i) {
  #pragma clang loop unroll(full)
  for (int j = 0; j < 4; ++j)
    acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
}
if (js < je) {
  uint ar = tid >> 2;
  uint ac = (tid & 3) * 8;
  float Mr = sM[ar];
  float Ir = sI[ar];
  const device float* sp = S + ((size_t)b * 128 + m0 + ar) * NS;
  uint br = tid >> 2;
  uint bc = (tid & 3) * 16;
  for (int jb = js; jb < je; jb += 32) {
    #pragma clang loop unroll(full)
    for (int t = 0; t < 8; ++t) {
      int j = jb + (int)ac + t;
      float pv = 0.0f;
      if (j < je) pv = exp(sp[j] - Mr) * Ir;
      As[ar * 36 + ac + t] = pv;
    }
    int jj = jb + (int)br;
    bool vv = jj < je;
    int row = vv ? idx[(size_t)b * TOPK + jj] : 0;
    const device half4* vp = (const device half4*)(kv + ((size_t)b * L + row) * 576 + n0 + bc);
    threadgroup half4* bd = (threadgroup half4*)(Bs + br * 72 + bc);
    #pragma clang loop unroll(full)
    for (int t = 0; t < 4; ++t) bd[t] = vv ? vp[t] : half4(0.0h);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (int kk = 0; kk < 32; kk += 8) {
      simdgroup_float8x8 a[2];
      simdgroup_float8x8 bm[4];
      #pragma clang loop unroll(full)
      for (int i = 0; i < 2; ++i) {
        float2 v = *((const threadgroup float2*)(As + (16 * sm + 8 * i + fm) * 36 + kk + fn));
        reinterpret_cast<thread float2&>(a[i].thread_elements()) = v;
      }
      #pragma clang loop unroll(full)
      for (int j = 0; j < 4; ++j) {
        half2 v = *((const threadgroup half2*)(Bs + (kk + fm) * 72 + 32 * sn + 8 * j + fn));
        reinterpret_cast<thread float2&>(bm[j].thread_elements()) = float2(v);
      }
      #pragma clang loop unroll(full)
      for (int i = 0; i < 2; ++i) {
        #pragma clang loop unroll(full)
        for (int j = 0; j < 4; ++j)
          simdgroup_multiply_accumulate(acc[i][j], a[i], bm[j], acc[i][j]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}
device float* op = Op + (((size_t)ks * NB + b) * 128 + m0) * 512 + n0;
#pragma clang loop unroll(full)
for (int i = 0; i < 2; ++i) {
  #pragma clang loop unroll(full)
  for (int j = 0; j < 4; ++j) {
    float2 v = reinterpret_cast<thread float2&>(acc[i][j].thread_elements());
    *((device float2*)(op + (16 * sm + 8 * i + fm) * 512 + 32 * sn + 8 * j + fn)) = v;
  }
}
"""

RED_SRC = """
uint i = thread_position_in_grid.x;
float s = 0.0f;
for (int k = 0; k < KS; ++k) s += Op[(size_t)k * TOT + i];
out[i] = (half)s;
"""

GATHER_SRC = """
uint gid = thread_position_in_grid.x;
uint per = (uint)KP * 72u;
uint b = gid / per;
uint rem = gid - b * per;
uint j = rem / 72u;
uint t = rem - j * 72u;
int c = cnt[b];
uint4 v = uint4(0u);
if ((int)j < c) {
  int row = idx[(size_t)b * TOPK + j];
  v = ((const device uint4*)(kv + ((size_t)b * L + row) * 576))[t];
}
((device uint4*)(Kg + ((size_t)b * KP + j) * 576))[t] = v;
"""

SOFT_SRC = """
uint tid = thread_position_in_threadgroup.x;
uint sg = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
uint row = threadgroup_position_in_grid.x;
uint b = row >> 7;
int c = cnt[b];
threadgroup float rm[8];
threadgroup float rs[8];
const device float* sr = S + (size_t)row * KP;
const float SC = 0.07216878364870323f;
float v[EPT];
float m = -INFINITY;
#pragma clang loop unroll(full)
for (int k = 0; k < EPT; ++k) {
  int j = (int)tid + k * 256;
  float x = -INFINITY;
  if (j < c) x = sr[j] * SC;
  v[k] = x;
  m = max(m, x);
}
m = simd_max(m);
if (lane == 0) rm[sg] = m;
threadgroup_barrier(mem_flags::mem_threadgroup);
m = rm[0];
#pragma clang loop unroll(full)
for (int k = 1; k < 8; ++k) m = max(m, rm[k]);
float s = 0.0f;
device half* pr = P + (size_t)row * KP;
#pragma clang loop unroll(full)
for (int k = 0; k < EPT; ++k) {
  int j = (int)tid + k * 256;
  float e = 0.0f;
  if (j < c) e = exp(v[k] - m);
  s += e;
  pr[j] = (half)e;
}
s = simd_sum(s);
if (lane == 0) rs[sg] = s;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (tid == 0) {
  float t = 0.0f;
  for (int k = 0; k < 8; ++k) t += rs[k];
  invl[row] = 1.0f / t;
}
"""

RED2_SRC = """
uint i = thread_position_in_grid.x;
float s = 0.0f;
for (int k = 0; k < KS; ++k) s += Op[(size_t)k * TOT + i];
out[i] = (half)(s * invl[i >> 9]);
"""

MPP_BASE = """
#include <metal_simdgroup_matrix>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
""" + COMMON

MKT_VARIANTS = [
    "#define MKT(T, ptr, d0, d1, ld) tensor<T, dextents<int32_t, 2>, tensor_inline>((T*)(ptr), dextents<int32_t, 2>(d0, d1))\n",
    "#define MKT(T, ptr, d0, d1, ld) tensor<T, dextents<int32_t, 2>, tensor_inline>((T*)(ptr), dextents<int32_t, 2>(d0, d1), array<int32_t, 2>{1, (int32_t)(ld)})\n",
    "#define MKT(T, ptr, d0, d1, ld) tensor((T*)(ptr), dextents<int32_t, 2>(d0, d1))\n",
]

# (fused_gather, relaxed_pv, QM, QN, PM, PN), tried in order
ATT_CFGS = [
    (1, 1, 128, 64, 128, 64),
    (1, 0, 128, 64, 128, 64),
    (0, 0, 128, 64, 128, 64),
    (0, 0, 64, 64, 64, 64),
]

IDXM_SRC = """
uint tid = thread_position_in_threadgroup.x;
uint tile = threadgroup_position_in_grid.x;
uint b = threadgroup_position_in_grid.y;
threadgroup float TT[64 * 64];
threadgroup float sw[64];
if (tid < 64) sw[tid] = w[b * 64 + tid];
int l0 = min((int)tile * 64, L - 64);
{
  constexpr auto desc = matmul2d_descriptor(64, 64, 128, false, true, false);
  matmul2d<desc, execution_simdgroups<4>> op;
  auto tik = MKT(device half, ik, 128, NB * L, 128);
  auto tiq = MKT(device half, iq, 128, NB * 64, 128);
  auto tt = MKT(threadgroup float, TT, 64, 64, 64);
  auto mA = tik.slice(0, (int)b * L + l0);
  auto mB = tiq.slice(0, (int)b * 64);
  op.run(mA, mB, tt);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
uint p = tid >> 1;
uint hh = (tid & 1) * 32;
float part = 0.0f;
for (uint t = 0; t < 32; ++t) {
  uint h = hh + ((t + p) & 31);
  part += sw[h] * max(TT[p * 64 + h], 0.0f);
}
part += simd_shuffle_xor(part, (ushort)1);
int pos = l0 + (int)p;
if ((tid & 1) == 0) {
  atomic_store_explicit(&scores[(size_t)b * L + pos], part, memory_order_relaxed);
  if (pos >= (int)tile * 64) hist_add(hist + (size_t)b * 8192, fkey(part) >> 19);
}
"""

QKM_SRC = """
uint nt = threadgroup_position_in_grid.x;
uint mt = threadgroup_position_in_grid.y;
uint b = threadgroup_position_in_grid.z;
int c = cnt[b];
if ((int)nt * QN >= c) return;
constexpr auto desc = matmul2d_descriptor(QM, QN, 576, false, true, false);
matmul2d<desc, execution_simdgroups<4>> op;
auto tq = MKT(device half, q, 576, NB * 128, 576);
auto tk = MKT(device half, Kg, 576, NB * KP, 576);
auto ts = MKT(device float, S, KP, NB * 128, KP);
auto mA = tq.slice(0, (int)(b * 128 + mt * QM));
auto mB = tk.slice(0, (int)(b * KP + nt * QN));
auto mC = ts.slice((int)(nt * QN), (int)(b * 128 + mt * QM));
op.run(mA, mB, mC);
"""

QKF_SRC = """
uint tid = thread_position_in_threadgroup.x;
uint nt = threadgroup_position_in_grid.x;
uint mt = threadgroup_position_in_grid.y;
uint b = threadgroup_position_in_grid.z;
int c = cnt[b];
int n0 = (int)nt * QN;
device half* kgb = Kg + ((size_t)b * KP + n0) * 576;
for (uint e = tid; e < (uint)(QN * 72); e += 128) {
  uint r = e / 72u;
  uint t = e - r * 72u;
  int j = n0 + (int)r;
  uint4 v = uint4(0u);
  if (j < c) {
    int row = idx[(size_t)b * TOPK + j];
    v = ((const device uint4*)(kv + ((size_t)b * L + row) * 576))[t];
  }
  ((device uint4*)(kgb + (size_t)r * 576))[t] = v;
}
if (n0 >= c) return;
threadgroup_barrier(mem_flags::mem_device);
constexpr auto desc = matmul2d_descriptor(QM, QN, 576, false, true, false);
matmul2d<desc, execution_simdgroups<4>> op;
auto tq = MKT(device half, q, 576, NB * 128, 576);
auto tk = MKT(device half, Kg, 576, NB * KP, 576);
auto ts = MKT(device float, S, KP, NB * 128, KP);
auto mA = tq.slice(0, (int)(b * 128 + mt * QM));
auto mB = tk.slice(0, (int)(b * KP + nt * QN));
auto mC = ts.slice((int)(nt * QN), (int)(b * 128 + mt * QM));
op.run(mA, mB, mC);
"""

PVM_SRC = """
uint nt = threadgroup_position_in_grid.x;
uint mt = threadgroup_position_in_grid.y;
uint z = threadgroup_position_in_grid.z;
uint b = z / KS;
uint ks = z % KS;
constexpr auto desc = matmul2d_descriptor(PM, PN, 512, false, false, PREL);
matmul2d<desc, execution_simdgroups<4>> op;
auto tp = MKT(device half, P, KP, NB * 128, KP);
auto tv = MKT(device half, Kg, 576, NB * KP, 576);
auto to = MKT(device float, Op, 512, KS * NB * 128, 512);
auto mA = tp.slice((int)(ks * 512), (int)(b * 128 + mt * PM));
auto mB = tv.slice((int)(nt * PN), (int)(b * KP + ks * 512));
auto mC = to.slice((int)(nt * PN), (int)((ks * NB + b) * 128 + mt * PM));
op.run(mA, mB, mC);
"""

K_INDEX = mx.fast.metal_kernel(
    name="dsa_indexer_v7", input_names=["iq", "ik", "w"], output_names=["scores", "hist"],
    source=INDEX_SRC, header=HEADER, atomic_outputs=True)
K_SEL = mx.fast.metal_kernel(
    name="dsa_select_v7", input_names=["scores", "hist", "lens"], output_names=["idx", "cnt"],
    source=SEL_SRC, header=HEADER)
K_QK = mx.fast.metal_kernel(
    name="dsa_qk_v7", input_names=["q", "kv", "idx", "cnt"], output_names=["S", "mst", "lst"],
    source=QK_SRC, header=HEADER)
K_PV = mx.fast.metal_kernel(
    name="dsa_pv_v7", input_names=["S", "mst", "lst", "kv", "idx", "cnt"], output_names=["Op"],
    source=PV_SRC, header=HEADER)
K_RED = mx.fast.metal_kernel(
    name="dsa_red_v7", input_names=["Op"], output_names=["out"],
    source=RED_SRC, header=HEADER)
K_GATHER = mx.fast.metal_kernel(
    name="dsa_gather_v7", input_names=["kv", "idx", "cnt"], output_names=["Kg"],
    source=GATHER_SRC, header=HEADER)
K_SOFT = mx.fast.metal_kernel(
    name="dsa_soft_v7", input_names=["S", "cnt"], output_names=["P", "invl"],
    source=SOFT_SRC, header=HEADER)
K_RED2 = mx.fast.metal_kernel(
    name="dsa_red2_v7", input_names=["Op", "invl"], output_names=["out"],
    source=RED2_SRC, header=HEADER)

IDX_SETS = []
ATT_SETS = []
for vi, mk in enumerate(MKT_VARIANTS):
    hdr = MPP_BASE + mk
    IDX_SETS.append(mx.fast.metal_kernel(
        name="dsa_idxm_v7_%d" % vi, input_names=["iq", "ik", "w"],
        output_names=["scores", "hist"], source=IDXM_SRC, header=hdr, atomic_outputs=True))
for ci, cfg in enumerate(ATT_CFGS):
    fused, rel, QM, QN, PM, PN = cfg
    for vi, mk in enumerate(MKT_VARIANTS):
        hdr = (MPP_BASE + mk + "#define QM %d\n#define QN %d\n#define PM %d\n#define PN %d\n" % (QM, QN, PM, PN)
               + "#define PREL %s\n" % ("true" if rel else "false"))
        if fused:
            qk = mx.fast.metal_kernel(
                name="dsa_qkf_v7_%d_%d" % (ci, vi), input_names=["q", "kv", "idx", "cnt"],
                output_names=["S", "Kg"], source=QKF_SRC, header=hdr)
        else:
            qk = mx.fast.metal_kernel(
                name="dsa_qkm_v7_%d_%d" % (ci, vi), input_names=["q", "Kg", "cnt"],
                output_names=["S"], source=QKM_SRC, header=hdr)
        ATT_SETS.append({
            "cfg": cfg,
            "qk": qk,
            "pv": mx.fast.metal_kernel(
                name="dsa_pvm_v7_%d_%d" % (ci, vi), input_names=["P", "Kg"],
                output_names=["Op"], source=PVM_SRC, header=hdr),
        })


def ident(x):
    return x


def f16(x):
    return x if x.dtype == mx.float16 else x.astype(mx.float16)


def indexer_float(iq, ik, w, B, L):
    ntg = (L + 255) // 256
    r = K_INDEX(inputs=[iq, ik, w], template=[("L", L)],
                grid=(ntg * 256, B, 1), threadgroup=(256, 1, 1),
                output_shapes=[(B, L), (B, 8192)], output_dtypes=[mx.float32, mx.uint32],
                init_value=0)
    return r[0], r[1]


def indexer_mpp(kern, iq, ik, w, B, L):
    ntg = (L + 63) // 64
    r = kern(inputs=[iq, ik, w], template=[("L", L), ("NB", B)],
             grid=(ntg * 128, B, 1), threadgroup=(128, 1, 1),
             output_shapes=[(B, L), (B, 8192)], output_dtypes=[mx.float32, mx.uint32],
             init_value=0)
    return r[0], r[1]


def select(scores, hist, lens_in, has_lens, B, L):
    return K_SEL(inputs=[scores, hist, lens_in],
                 template=[("L", L), ("TOPK", TOPK), ("HL", has_lens)],
                 grid=(1024 * B, 1, 1), threadgroup=(1024, 1, 1),
                 output_shapes=[(B, TOPK), (B,)],
                 output_dtypes=[mx.int32, mx.int32])


def attn_float(q, kv, idx, cnt, B, L):
    KMAX = min(L, TOPK)
    NT = (KMAX + 63) // 64
    NS = NT * 64
    KC = 512
    KS = (KMAX + KC - 1) // KC
    S, mst, lst = K_QK(inputs=[q, kv, idx, cnt],
                       template=[("L", L), ("NT", NT), ("NS", NS), ("TOPK", TOPK)],
                       grid=(NT * 128, 2, B), threadgroup=(128, 1, 1),
                       output_shapes=[(B, 128, NS), (B, 128, NT), (B, 128, NT)],
                       output_dtypes=[mx.float32, mx.float32, mx.float32])
    Op = K_PV(inputs=[S, mst, lst, kv, idx, cnt],
              template=[("L", L), ("NT", NT), ("NS", NS), ("TOPK", TOPK),
                        ("KC", KC), ("KS", KS), ("NB", B)],
              grid=(8 * 128, 4, B * KS), threadgroup=(128, 1, 1),
              output_shapes=[(KS, B, 128, 512)], output_dtypes=[mx.float32])[0]
    TOT = B * 128 * 512
    return K_RED(inputs=[Op], template=[("KS", KS), ("TOT", TOT)],
                 grid=(TOT, 1, 1), threadgroup=(256, 1, 1),
                 output_shapes=[(B, 128, 512)], output_dtypes=[mx.float16])[0]


def attn_mpp(aset, q, kv, idx, cnt, B, L, step):
    fused, rel, QM, QN, PM, PN = aset["cfg"]
    KMAX = min(L, TOPK)
    KS = (KMAX + 511) // 512
    KP = KS * 512
    if fused:
        S, Kg = aset["qk"](inputs=[q, kv, idx, cnt],
                           template=[("NB", B), ("KP", KP), ("L", L), ("TOPK", TOPK)],
                           grid=((KP // QN) * 128, 128 // QM, B), threadgroup=(128, 1, 1),
                           output_shapes=[(B, 128, KP), (B, KP, 576)],
                           output_dtypes=[mx.float32, mx.float16])
        S = step(S)
        Kg = step(Kg)
    else:
        Kg = step(K_GATHER(inputs=[kv, idx, cnt], template=[("L", L), ("KP", KP), ("TOPK", TOPK)],
                           grid=(B * KP * 72, 1, 1), threadgroup=(256, 1, 1),
                           output_shapes=[(B, KP, 576)], output_dtypes=[mx.float16])[0])
        S = step(aset["qk"](inputs=[q, Kg, cnt], template=[("NB", B), ("KP", KP)],
                            grid=((KP // QN) * 128, 128 // QM, B), threadgroup=(128, 1, 1),
                            output_shapes=[(B, 128, KP)], output_dtypes=[mx.float32])[0])
    P, invl = K_SOFT(inputs=[S, cnt], template=[("KP", KP), ("EPT", KP // 256)],
                     grid=(B * 128 * 256, 1, 1), threadgroup=(256, 1, 1),
                     output_shapes=[(B, 128, KP), (B * 128,)],
                     output_dtypes=[mx.float16, mx.float32])
    P = step(P)
    invl = step(invl)
    Op = step(aset["pv"](inputs=[P, Kg], template=[("NB", B), ("KP", KP), ("KS", KS)],
                         grid=((512 // PN) * 128, 128 // PM, B * KS), threadgroup=(128, 1, 1),
                         output_shapes=[(KS, B, 128, 512)], output_dtypes=[mx.float32])[0])
    TOT = B * 128 * 512
    return K_RED2(inputs=[Op, invl], template=[("KS", KS), ("TOT", TOT)],
                  grid=(TOT, 1, 1), threadgroup=(256, 1, 1),
                  output_shapes=[(B, 128, 512)], output_dtypes=[mx.float16])[0]


def force(x):
    mx.sum(x.astype(mx.float32)).item()
    return x


def pseudo(shape, seed, scale, dtype):
    n = 1
    for d in shape:
        n *= d
    x = mx.arange(n, dtype=mx.float32)
    v = mx.sin(x * 0.7071 + seed * 1.37) * 43758.5453
    v = v - mx.floor(v)
    return force(((v * 2.0 - 1.0) * scale).reshape(shape).astype(dtype))


def probe():
    res_idx = None
    res_att = None
    try:
        B, L = 2, 3000
        iq_t = pseudo((B, 64, 128), 1, 2.0, mx.float16)
        ik_t = pseudo((B, L, 128), 2, 2.0, mx.float16)
        w_t = pseudo((B, 64), 3, 0.02, mx.float32)
        q_t = pseudo((B, 128, 576), 4, 1.0, mx.float16)
        kv_t = pseudo((B, L, 576), 5, 2.0, mx.float16)
        q_l = pseudo((B, 128, 576), 6, 6.0, mx.float16)
        kv_l = pseudo((B, L, 576), 7, 6.0, mx.float16)
        s_ref, h_ref = indexer_float(iq_t, ik_t, w_t, B, L)
        s_ref = force(s_ref)
        h_ref = force(h_ref)
        idx, cnt = select(s_ref, h_ref, w_t, False, B, L)
        idx = force(idx)
        cnt = force(cnt)
        o_ref = force(attn_float(q_t, kv_t, idx, cnt, B, L)).astype(mx.float32)
        o_refl = force(attn_float(q_l, kv_l, idx, cnt, B, L)).astype(mx.float32)
        sr = mx.max(mx.abs(s_ref)).item()
        orf = mx.max(mx.abs(o_ref)).item()
        orfl = mx.max(mx.abs(o_refl)).item()
    except Exception:
        return None, None
    for kern in IDX_SETS:
        try:
            s2, h2 = indexer_mpp(kern, iq_t, ik_t, w_t, B, L)
            e = mx.max(mx.abs(s2 - s_ref)).item()
            tot = mx.sum(h2.astype(mx.float32)).item()
            if e <= 1e-4 * max(sr, 1e-3) and tot == float(B * L):
                res_idx = kern
                break
        except Exception:
            pass
    for aset in ATT_SETS:
        try:
            rel = aset["cfg"][1]
            tol = 5e-4 if rel else 4e-3
            o2 = attn_mpp(aset, q_t, kv_t, idx, cnt, B, L, force)
            e = mx.max(mx.abs(o2.astype(mx.float32) - o_ref)).item()
            ok = orf > 0 and e <= tol * max(orf, 1.0)
            if ok:
                o3 = attn_mpp(aset, q_l, kv_l, idx, cnt, B, L, force)
                e3 = mx.max(mx.abs(o3.astype(mx.float32) - o_refl)).item()
                ok = orfl > 0 and e3 <= tol * max(orfl, 1.0)
            if ok:
                res_att = aset
                break
        except Exception:
            pass
    return res_idx, res_att


IDX_SET, ATT_SET = probe()


def kernel(q, kv, iq, ik, w, lens):
    B = kv.shape[0]
    L = kv.shape[1]
    q = f16(q)
    kv = f16(kv)
    has_lens = lens is not None
    if has_lens:
        lens_in = lens if lens.dtype == mx.int32 else lens.astype(mx.int32)
    else:
        lens_in = w

    if L > TOPK:
        iq16 = f16(iq)
        ik16 = f16(ik)
        w32 = w if w.dtype == mx.float32 else w.astype(mx.float32)
        if not has_lens:
            lens_in = w32
        if IDX_SET is not None:
            scores, hist = indexer_mpp(IDX_SET, iq16, ik16, w32, B, L)
        else:
            scores, hist = indexer_float(iq16, ik16, w32, B, L)
        idx, cnt = select(scores, hist, lens_in, has_lens, B, L)
    else:
        idx = mx.broadcast_to(mx.arange(TOPK, dtype=mx.int32)[None, :], (B, TOPK))
        if has_lens:
            cnt = mx.clip(lens_in, 0, L).astype(mx.int32)
        else:
            cnt = mx.full((B,), L, dtype=mx.int32)

    if ATT_SET is not None:
        return attn_mpp(ATT_SET, q, kv, idx, cnt, B, L, ident)
    return attn_float(q, kv, idx, cnt, B, L)
