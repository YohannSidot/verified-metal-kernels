"""Agent op (b): shared prefix. N requests (e.g. N agents / N samples of one prompt) share a P-token prefix and each
has its own S-token suffix; one decode token per request.  q:(N,Hq,D)  kp,vp:(Hkv,P,D)  ks,vs:(N,Hkv,S,D).
Reference engines store either a full per-request cache (kf,vf:(N,Hkv,P+S,D), no sharing) or a unified cache
(ku,vu:(1,Hkv,P+N*S,D)) with a per-request mask (llama.cpp's unified KV)."""
import math
import numpy as np
import mlx.core as mx

NAME = "prefix"
HQ, HKV, D = 32, 8, 128
L1_SHAPE = dict(N=4, P=2048, S=256)
BENCH_SHAPES = [dict(N=8, P=8192, S=512), dict(N=16, P=16384, S=1024), dict(N=32, P=4096, S=256)]


def make_inputs(shape, seed=0, dist="randn", with_ref_layouts=True):
    rng = np.random.default_rng(seed)
    N, P, S = shape["N"], shape["P"], shape["S"]
    Hq, Hkv, d = shape.get("Hq", HQ), shape.get("Hkv", HKV), shape.get("D", D)
    q = rng.standard_normal((N, Hq, d), dtype=np.float32)
    kp = rng.standard_normal((Hkv, P, d), dtype=np.float32)
    vp = rng.standard_normal((Hkv, P, d), dtype=np.float32)
    ks = rng.standard_normal((N, Hkv, S, d), dtype=np.float32)
    vs = rng.standard_normal((N, Hkv, S, d), dtype=np.float32)
    G = Hq // Hkv
    if dist == "large":
        q, kp, ks = q * 8, kp * 8, ks * 8
    elif dist == "needle_prefix":
        for h in range(Hkv):
            kp[h, rng.integers(0, P)] = q[:, h * G:(h + 1) * G].mean((0, 1)) * 6.0
    elif dist == "needle_suffix":
        for n in range(N):
            for h in range(Hkv):
                ks[n, h, rng.integers(0, S)] = q[n, h * G:(h + 1) * G].mean(0) * 6.0
                vs[n, h] += 3.0
    elif dist == "nan_inf":
        ks[N - 1, 0, S - 1, 0] = np.nan
        vp[1, P // 2, 7] = np.inf
    elif dist == "tiny":
        q, kp, vp, ks, vs = q * 1e-4, kp * 1e-4, vp * 1e-6, ks * 1e-4, vs * 1e-6
    f = lambda a: mx.array(a.astype(np.float16))
    out = dict(q=f(q), kp=f(kp), vp=f(vp), ks=f(ks), vs=f(vs))
    if with_ref_layouts:
        out.update(build_layouts(out["kp"], out["vp"], out["ks"], out["vs"]))
    return out


def build_layouts(kp, vp, ks, vs):
    N = ks.shape[0]
    kf = mx.concatenate([mx.broadcast_to(kp[None], (N,) + kp.shape), ks], axis=2)
    vf = mx.concatenate([mx.broadcast_to(vp[None], (N,) + vp.shape), vs], axis=2)
    ku = mx.concatenate([kp] + [ks[n] for n in range(N)], axis=1)[None]
    vu = mx.concatenate([vp] + [vs[n] for n in range(N)], axis=1)[None]
    mx.eval(kf, vf, ku, vu)
    return dict(kf=kf, vf=vf, ku=ku, vu=vu)


def ref64(q, kp, vp, ks, vs, **_):
    q, kp, vp, ks, vs = (np.array(x.astype(mx.float32)).astype(np.float64) for x in (q, kp, vp, ks, vs))
    N, Hq, d = q.shape
    Hkv = kp.shape[0]
    G = Hq // Hkv
    k = np.concatenate([np.broadcast_to(kp[None], (N,) + kp.shape), ks], axis=2)
    v = np.concatenate([np.broadcast_to(vp[None], (N,) + vp.shape), vs], axis=2)
    s = np.einsum("nhgd,nhld->nhgl", q.reshape(N, Hkv, G, d), k) / math.sqrt(d)
    with np.errstate(invalid="ignore", over="ignore"):
        s = s - s.max(-1, keepdims=True)
        p = np.exp(s)
        p /= p.sum(-1, keepdims=True)
        o = np.einsum("nhgl,nhld->nhgd", p, v)
    return o.reshape(N, Hq, d)


def _umask(N, G, P, S):
    r = mx.arange(N * G)[:, None] // G  # request of each row
    c = mx.arange(P + N * S)[None, :]
    return (c < P) | ((c >= P + r * S) & (c < P + (r + 1) * S))


# ---------------------------------------------------------------- MLX references
def mlx_naive(q, kf, vf, **_):  # non-sharing engine: every request reads the whole prefix
    o = mx.fast.scaled_dot_product_attention(q[:, :, None, :], kf, vf, scale=1 / math.sqrt(q.shape[-1]))
    return o[:, :, 0, :]


def mlx_unified(q, ku, vu, ks, **_):  # unified cache + mask: prefix read once per KV head, rows = N*G
    N, Hq, d = q.shape
    Hkv = ku.shape[1]
    G = Hq // Hkv
    S = ks.shape[2]
    P = ku.shape[2] - N * S
    qf = q.reshape(N, Hkv, G, d).transpose(1, 0, 2, 3).reshape(1, Hkv, N * G, d)
    o = mx.fast.scaled_dot_product_attention(qf, ku, vu, scale=1 / math.sqrt(d), mask=_umask(N, G, P, S))
    return o.reshape(Hkv, N, G, d).transpose(1, 0, 2, 3).reshape(N, Hq, d)


def mlx_cascade(q, kp, vp, ks, vs, **_):  # cascade written for this project with MLX ops (prefix once + suffix, LSE merge)
    N, Hq, d = q.shape
    Hkv = kp.shape[0]
    G = Hq // Hkv
    sc = 1 / math.sqrt(d)
    qp = q.reshape(N, Hkv, G, d).transpose(1, 0, 2, 3).reshape(Hkv, N * G, d)
    s1 = (qp @ kp.swapaxes(-1, -2)).astype(mx.float32) * sc  # (Hkv, NG, P)
    m1 = s1.max(-1, keepdims=True)
    p1 = mx.exp(s1 - m1)
    l1 = p1.sum(-1, keepdims=True)
    o1 = (p1.astype(mx.float16) @ vp).astype(mx.float32)  # (Hkv, NG, d)
    o1 = o1.reshape(Hkv, N, G, d).transpose(1, 0, 2, 3)
    m1 = m1.reshape(Hkv, N, G, 1).transpose(1, 0, 2, 3)
    l1 = l1.reshape(Hkv, N, G, 1).transpose(1, 0, 2, 3)
    s2 = (q.reshape(N, Hkv, G, d) @ ks.swapaxes(-1, -2)).astype(mx.float32) * sc  # (N,Hkv,G,S)
    m2 = s2.max(-1, keepdims=True)
    p2 = mx.exp(s2 - m2)
    l2 = p2.sum(-1, keepdims=True)
    o2 = (p2.astype(mx.float16) @ vs).astype(mx.float32)
    m = mx.maximum(m1, m2)
    a, b = mx.exp(m1 - m), mx.exp(m2 - m)
    return ((o1 * a + o2 * b) / (l1 * a + l2 * b)).astype(mx.float16).reshape(N, Hq, d)


# ---------------------------------------------------------------- PyTorch MPS references
def torch_naive(q, kf, vf, **_):
    import torch.nn.functional as F
    return F.scaled_dot_product_attention(q[:, :, None, :], kf, vf, enable_gqa=True)[:, :, 0, :]


def torch_unified(q, ku, vu, ks, **_):
    import torch, torch.nn.functional as F
    N, Hq, d = q.shape
    Hkv = ku.shape[1]
    G = Hq // Hkv
    S = ks.shape[2]
    P = ku.shape[2] - N * S
    r = torch.arange(N * G, device=q.device)[:, None] // G
    c = torch.arange(P + N * S, device=q.device)[None, :]
    m = (c < P) | ((c >= P + r * S) & (c < P + (r + 1) * S))
    qf = q.reshape(N, Hkv, G, d).permute(1, 0, 2, 3).reshape(1, Hkv, N * G, d)
    o = F.scaled_dot_product_attention(qf, ku, vu, attn_mask=m)
    return o.reshape(Hkv, N, G, d).permute(1, 0, 2, 3).reshape(N, Hq, d)


MLX_REFS = dict(mlx_naive=mlx_naive, mlx_unified=mlx_unified, mlx_cascade=mlx_cascade)
TORCH_REFS = dict(torch_naive=torch_naive, torch_unified=torch_unified)


L1_REF = "mlx_naive"
L3_REF = "mlx_naive"
FUZZ = [(dict(N=1, P=1, S=1), "randn"), (dict(N=3, P=1000, S=77), "randn"), (dict(N=5, P=4097, S=33), "randn"),
        (dict(N=2, P=65536, S=4096), "randn"), (dict(N=33, P=300, S=10), "randn"), (dict(N=64, P=2048, S=64), "randn"),
        (dict(N=4, P=3000, S=200), "large"), (dict(N=4, P=3000, S=200), "needle_prefix"),
        (dict(N=4, P=3000, S=200), "needle_suffix"), (dict(N=4, P=3000, S=200), "nan_inf"),
        (dict(N=4, P=3000, S=200), "tiny"), (dict(N=7, P=8, S=1), "randn")]
L3_CASES = [(dict(N=8, P=1024, S=128), "randn"), (dict(N=8, P=32768, S=1024), "randn"), (dict(N=4, P=8192, S=512), "large"),
            (dict(N=4, P=8192, S=512), "needle_prefix"), (dict(N=4, P=8192, S=512), "needle_suffix")]


