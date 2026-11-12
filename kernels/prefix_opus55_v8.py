import mlx.core as mx
import math

SCALE = (1.0 / math.sqrt(128.0)) * 1.4426950408889634
SCALE_S = "%.10ef" % SCALE

MAIN_HDR = """
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
"""

MAIN_SRC = """
    constexpr int D = 128;
    constexpr int NT = NSG * 32;
    constexpr int TPR = NT / BQ;
    constexpr int CPT = BK / TPR;
    constexpr int R = NREQ * HQ;
    threadgroup float Ss[BQ * BK];
    threadgroup float facs[BQ];
    threadgroup int rflag[2];

    const uint b = threadgroup_position_in_grid.x;
    const uint tid = thread_position_in_threadgroup.x;
    const float sc = SCALE_F;
    const float NEG = -1e30f;

    if (b < (uint)NSUF) {
        const uint sgi = simdgroup_index_in_threadgroup;
        const uint w = b * NSG + sgi;
        if (w < (uint)(NREQ * HKV * CSN)) {
            const uint lane = thread_index_in_simdgroup;
            const uint hs = lane / 16, l16 = lane % 16;
            const uint c = w % CSN;
            const uint nh = w / CSN;
            const uint n = nh / HKV, h = nh % HKV;
            const int start = (int)c * CS;
            const int end = min(start + CS, S);

            float4 q0[G], q1[G], o0[G], o1[G];
            float m[G], l[G];
            for (int g = 0; g < G; ++g) {
                const device half4* qp4 = (const device half4*)(q + (size_t)(n * HQ + h * G + g) * D + l16 * 8);
                q0[g] = float4(qp4[0]) * sc;
                q1[g] = float4(qp4[1]) * sc;
                o0[g] = float4(0.f); o1[g] = float4(0.f);
                m[g] = NEG; l[g] = 0.f;
            }
            const device half* kb = ks + (size_t)nh * S * D + l16 * 8;
            const device half* vb = vs + (size_t)nh * S * D + l16 * 8;
            for (int t = start + (int)hs; t < end; t += 4) {
                const bool two = (t + 2 < end);
                const device half4* k4 = (const device half4*)(kb + (size_t)t * D);
                const device half4* v4 = (const device half4*)(vb + (size_t)t * D);
                float4 ka0 = float4(k4[0]), ka1 = float4(k4[1]);
                float4 va0 = float4(v4[0]), va1 = float4(v4[1]);
                float4 kb0 = float4(0.f), kb1 = float4(0.f), vb0 = float4(0.f), vb1 = float4(0.f);
                if (two) {
                    const device half4* k4b = (const device half4*)(kb + (size_t)(t + 2) * D);
                    const device half4* v4b = (const device half4*)(vb + (size_t)(t + 2) * D);
                    kb0 = float4(k4b[0]); kb1 = float4(k4b[1]);
                    vb0 = float4(v4b[0]); vb1 = float4(v4b[1]);
                }
                for (int g = 0; g < G; ++g) {
                    float sa = dot(q0[g], ka0) + dot(q1[g], ka1);
                    float sb = dot(q0[g], kb0) + dot(q1[g], kb1);
                    sa += simd_shuffle_xor(sa, 1);
                    sb += simd_shuffle_xor(sb, 1);
                    sa += simd_shuffle_xor(sa, 2);
                    sb += simd_shuffle_xor(sb, 2);
                    sa += simd_shuffle_xor(sa, 4);
                    sb += simd_shuffle_xor(sb, 4);
                    sa += simd_shuffle_xor(sa, 8);
                    sb += simd_shuffle_xor(sb, 8);
                    if (!two) sb = NEG;
                    float mn = max(m[g], max(sa, sb));
                    float fac = exp2(m[g] - mn);
                    float pa = exp2(sa - mn);
                    float pb = two ? exp2(sb - mn) : 0.f;
                    l[g] = l[g] * fac + pa + pb;
                    o0[g] = o0[g] * fac + pa * va0 + pb * vb0;
                    o1[g] = o1[g] * fac + pa * va1 + pb * vb1;
                    m[g] = mn;
                }
            }
            for (int g = 0; g < G; ++g) {
                float mo = simd_shuffle_xor(m[g], 16);
                float lo = simd_shuffle_xor(l[g], 16);
                float4 oo0 = simd_shuffle_xor(o0[g], 16);
                float4 oo1 = simd_shuffle_xor(o1[g], 16);
                float mn = max(m[g], mo);
                float w1 = exp2(m[g] - mn);
                float w2 = exp2(mo - mn);
                float lt = l[g] * w1 + lo * w2;
                float4 r0 = o0[g] * w1 + oo0 * w2;
                float4 r1 = o1[g] * w1 + oo1 * w2;
                if (hs == 0) {
                    uint grow = n * HQ + h * G + g;
                    device float* op = Os + ((size_t)c * R + grow) * D + l16 * 8;
                    *((device float4*)op) = r0;
                    *((device float4*)(op + 4)) = r1;
                    if (l16 == 0) {
                        Ms[(size_t)c * R + grow] = mn;
                        Ls[(size_t)c * R + grow] = lt;
                    }
                }
            }
        }
    } else {
        threadgroup half* Ps = (threadgroup half*)Ss;
        using TQ = metal::tensor<device half, metal::dextents<int32_t, 2>, metal::tensor_inline>;
        using TS = metal::tensor<threadgroup float, metal::dextents<int32_t, 2>, metal::tensor_inline>;
        using TP = metal::tensor<threadgroup half, metal::dextents<int32_t, 2>, metal::tensor_inline>;
        using TO = metal::tensor<device float, metal::dextents<int32_t, 2>, metal::tensor_inline>;

        constexpr auto qk_desc = matmul2d_descriptor(BQ, BK, D, false, true, false,
                                                     matmul2d_descriptor::mode::multiply);
        constexpr auto pv_desc = matmul2d_descriptor(BQ, D, BK, false, false, false,
                                                     matmul2d_descriptor::mode::multiply_accumulate);
        matmul2d<qk_desc, metal::execution_simdgroups<NSG>> qk_op;
        matmul2d<pv_desc, metal::execution_simdgroups<NSG>> pv_op;

        const uint bp = b - NSUF;
        const uint qb = bp % QB;
        const uint c = (bp / QB) % CPN;
        const uint h = bp / (QB * CPN);

        if (tid < 2) rflag[tid] = 0;

        const int cstart = (int)c * CH;
        const int cend = min(cstart + CH, PP);

        device half* qbase = (device half*)qa + ((size_t)h * MP + qb * BQ) * D;
        TQ tQ(qbase, metal::dextents<int32_t, 2>(D, BQ));
        TS tS(Ss, metal::dextents<int32_t, 2>(BK, BQ));
        TP tP(Ps, metal::dextents<int32_t, 2>(BK, BQ));
        device float* obase = Op + (((size_t)c * HKV + h) * MP + qb * BQ) * D;
        TO tO(obase, metal::dextents<int32_t, 2>(D, BQ));

        auto cO = pv_op.template get_destination_cooperative_tensor<TP, TQ, float>();
        for (uint16_t i = 0; i < cO.get_capacity(); ++i) {
            if (cO.is_valid_element(i)) cO[i] = 0.0f;
        }

        const uint srow = tid / TPR, spart = tid % TPR;
        float m = NEG, l = 0.f;
        device half* kbase = (device half*)kp + (size_t)h * PP * D;
        device half* vbase = (device half*)vp + (size_t)h * PP * D;

        int it = 0;
        for (int t0 = cstart; t0 < cend; t0 += BK, ++it) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            TQ tK(kbase + (size_t)t0 * D, metal::dextents<int32_t, 2>(D, BK));
            qk_op.run(tQ, tK, tS);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const int par = it & 1;
            {
                threadgroup float* sp = Ss + srow * BK + spart * CPT;
                float vv[CPT];
                float rmax = NEG;
                for (int u = 0; u < CPT; ++u) {
                    float s = sp[u] * sc;
                    if (t0 + (int)spart * CPT + u >= PREAL) s = NEG;
                    vv[u] = s;
                    rmax = max(rmax, s);
                }
                for (int o = 1; o < TPR; o <<= 1) rmax = max(rmax, simd_shuffle_xor(rmax, (ushort)o));
                float fac = 1.f;
                bool resc = rmax > m + 8.f;
                if (resc) {
                    fac = exp2(m - rmax);
                    m = rmax;
                    l *= fac;
                }
                float rsum = 0.f;
                for (int u = 0; u < CPT; ++u) {
                    float p = exp2(vv[u] - m);
                    rsum += p;
                    vv[u] = p;
                }
                for (int o = 1; o < TPR; o <<= 1) rsum += simd_shuffle_xor(rsum, (ushort)o);
                l += rsum;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                threadgroup half* pp = Ps + srow * BK + spart * CPT;
                for (int u = 0; u < CPT; ++u) pp[u] = half(vv[u]);
                if (spart == 0) facs[srow] = fac;
                if (resc && t0 > cstart) rflag[par] = 1;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0) rflag[par ^ 1] = 0;
            if (rflag[par] != 0) {
                cO.store(tO);
                threadgroup_barrier(mem_flags::mem_device);
                for (uint i = tid; i < (uint)(BQ * D); i += NT) obase[i] *= facs[i / D];
                threadgroup_barrier(mem_flags::mem_device);
                cO.load(tO);
            }
            TQ tV(vbase + (size_t)t0 * D, metal::dextents<int32_t, 2>(D, BK));
            pv_op.run(tP, tV, cO);
        }
        cO.store(tO);
        if (spart == 0) {
            size_t idx = ((size_t)c * HKV + h) * MP + qb * BQ + srow;
            Mp[idx] = m;
            Lp[idx] = l;
        }
    }
""".replace("SCALE_F", SCALE_S)

COMBINE_SRC = """
    constexpr int D = 128;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint r = threadgroup_position_in_grid.y * 4 + sg;
    if (r < (uint)R) {
        const uint n = r / HQ, hq = r % HQ;
        const uint h = hq / G, g = hq % G;
        const uint prow = n * G + g;
        float Mx = -1e30f;
        for (int i = (int)lane; i < CP; i += 32) Mx = max(Mx, Mp[((size_t)i * HKV + h) * MP + prow]);
        for (int i = (int)lane; i < CSN; i += 32) Mx = max(Mx, Ms[(size_t)i * R + r]);
        Mx = simd_max(Mx);
        float L = 0.f;
        float4 acc = float4(0.f);
        for (int i0 = 0; i0 < CP; i0 += 32) {
            int i = i0 + (int)lane;
            float w = 0.f, lv = 0.f;
            if (i < CP) {
                size_t idx = ((size_t)i * HKV + h) * MP + prow;
                w = exp2(Mp[idx] - Mx);
                lv = Lp[idx];
            }
            L += simd_sum(w * lv);
            int cnt = min(32, CP - i0);
            for (int j = 0; j < cnt; ++j) {
                float wj = simd_shuffle(w, (ushort)j);
                size_t idx = ((size_t)(i0 + j) * HKV + h) * MP + prow;
                float4 ov = *((const device float4*)(Op + idx * D + lane * 4));
                acc += wj * ov;
            }
        }
        for (int i0 = 0; i0 < CSN; i0 += 32) {
            int i = i0 + (int)lane;
            float w = 0.f, lv = 0.f;
            if (i < CSN) {
                w = exp2(Ms[(size_t)i * R + r] - Mx);
                lv = Ls[(size_t)i * R + r];
            }
            L += simd_sum(w * lv);
            int cnt = min(32, CSN - i0);
            for (int j = 0; j < cnt; ++j) {
                float wj = simd_shuffle(w, (ushort)j);
                float4 ov = *((const device float4*)(Os + ((size_t)(i0 + j) * R + r) * D + lane * 4));
                acc += wj * ov;
            }
        }
        float4 res = acc / L;
        *((device half4*)(out + (size_t)r * D + lane * 4)) = half4(res);
    }
"""

main_k = mx.fast.metal_kernel(
    name="shpfx_main_v8",
    input_names=["qa", "kp", "vp", "q", "ks", "vs"],
    output_names=["Op", "Mp", "Lp", "Os", "Ms", "Ls"],
    source=MAIN_SRC,
    header=MAIN_HDR,
)
combine_k = mx.fast.metal_kernel(
    name="shpfx_combine_v8",
    input_names=["Op", "Mp", "Lp", "Os", "Ms", "Ls"],
    output_names=["out"],
    source=COMBINE_SRC,
)


def cdiv(a, b):
    return -(-a // b)


def kernel(q, kp, vp, ks, vs):
    if q.dtype != mx.float16:
        q = q.astype(mx.float16)
    if kp.dtype != mx.float16:
        kp = kp.astype(mx.float16)
    if vp.dtype != mx.float16:
        vp = vp.astype(mx.float16)
    if ks.dtype != mx.float16:
        ks = ks.astype(mx.float16)
    if vs.dtype != mx.float16:
        vs = vs.astype(mx.float16)
    N, Hq, Dd = q.shape
    Hkv, P = kp.shape[0], kp.shape[1]
    S = ks.shape[2]
    G = Hq // Hkv
    R = N * Hq
    M = N * G

    if M <= 32:
        BQ, nsg, BK = 32, 4, 128
    else:
        BQ, nsg, BK = 64, 8, 64
    QB = cdiv(M, BQ)
    MP = QB * BQ
    NT = nsg * 32

    qa = q.reshape(N, Hkv, G, 128).transpose(1, 0, 2, 3).reshape(Hkv, M, 128)
    if MP > M:
        qa = mx.concatenate([qa, mx.zeros((Hkv, MP - M, 128), dtype=mx.float16)], axis=1)

    PP = cdiv(P, BK) * BK
    if PP > P:
        z = mx.zeros((Hkv, PP - P, 128), dtype=mx.float16)
        kpp = mx.concatenate([kp, z], axis=1)
        vpp = mx.concatenate([vp, z], axis=1)
    else:
        kpp, vpp = kp, vp

    target = 192
    cp0 = max(1, cdiv(target, Hkv * QB))
    ch = cdiv(PP, cp0)
    ch = cdiv(ch, BK) * BK
    cp = cdiv(PP, ch)

    cs0 = max(1, cdiv(1024, N * Hkv))
    csl = max(64, cdiv(S, cs0))
    csn = cdiv(S, csl)

    nsuf = cdiv(N * Hkv * csn, nsg)
    npre = QB * cp * Hkv

    Op, Mp, Lp, Os, Ms, Ls = main_k(
        inputs=[qa, kpp, vpp, q, ks, vs],
        template=[("NSG", nsg), ("BQ", BQ), ("BK", BK), ("HKV", Hkv), ("MP", MP),
                  ("PP", PP), ("PREAL", P), ("CH", ch), ("QB", QB), ("CPN", cp),
                  ("G", G), ("HQ", Hq), ("NREQ", N), ("S", S), ("CS", csl),
                  ("CSN", csn), ("NSUF", nsuf)],
        grid=((nsuf + npre) * NT, 1, 1),
        threadgroup=(NT, 1, 1),
        output_shapes=[(cp, Hkv, MP, 128), (cp, Hkv, MP), (cp, Hkv, MP),
                       (csn, R, 128), (csn, R), (csn, R)],
        output_dtypes=[mx.float32] * 6,
    )

    out = combine_k(
        inputs=[Op, Mp, Lp, Os, Ms, Ls],
        template=[("CP", cp), ("CSN", csn), ("R", R), ("HQ", Hq), ("G", G),
                  ("HKV", Hkv), ("MP", MP)],
        grid=(128, cdiv(R, 4), 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(N, Hq, 128)],
        output_dtypes=[mx.float16],
    )[0]
    return out
