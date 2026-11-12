"""Agent op (c): speculative verification. kd draft tokens (a chain or a tree) verified in one pass against a cache of
L tokens whose last kd positions are the drafts themselves.  q:(B,kd,Hq,D)  k,v:(B,Hkv,L,D)  bits:(B,kd) int32,
bit j of bits[b,i] = draft i may attend draft j (ancestors + itself).  Llama-3-8B GQA, fp16."""
import math
import numpy as np
import mlx.core as mx

NAME = "spec"
HQ, HKV, D = 32, 8, 128
L1_SHAPE = dict(B=2, L=4096, k=8)
BENCH_SHAPES = [dict(B=1, L=8192, k=8), dict(B=1, L=32768, k=8), dict(B=4, L=16384, k=4), dict(B=1, L=32768, k=16)]


def chain_bits(kd):
    return np.array([(1 << (i + 1)) - 1 for i in range(kd)], dtype=np.int64)


def tree_bits(kd, rng):
    par = [-1] + [int(rng.integers(0, i)) for i in range(1, kd)]
    bits = np.zeros(kd, dtype=np.int64)
    for i in range(kd):
        bits[i] = 1 << i
        if par[i] >= 0:
            bits[i] |= bits[par[i]]
    return bits


def make_inputs(shape, seed=0, dist="randn"):
    rng = np.random.default_rng(seed)
    B, L, kd = shape["B"], shape["L"], shape["k"]
    Hq, Hkv, d = shape.get("Hq", HQ), shape.get("Hkv", HKV), shape.get("D", D)
    q = rng.standard_normal((B, kd, Hq, d), dtype=np.float32)
    k = rng.standard_normal((B, Hkv, L, d), dtype=np.float32)
    v = rng.standard_normal((B, Hkv, L, d), dtype=np.float32)
    bits = np.stack([tree_bits(kd, rng) if dist != "chain" else chain_bits(kd) for _ in range(B)])
    if dist == "large":
        q, k = q * 8.0, k * 8.0
    elif dist == "needle_masked":  # dominant key planted on a draft that row 0 must NOT see
        G = Hq // Hkv
        for b in range(B):
            j = kd - 1
            if (bits[b, 0] >> j) & 1 == 0:
                for h in range(Hkv):
                    k[b, h, L - kd + j] = q[b, 0, h * G:(h + 1) * G].mean(0) * 8.0
                    v[b, h, L - kd + j] += 50.0
    elif dist == "needle_boundary":  # dominant key just before the draft boundary
        G = Hq // Hkv
        for b in range(B):
            for h in range(Hkv):
                k[b, h, L - kd - 1] = q[b, :, h * G:(h + 1) * G].mean((0, 1)) * 6.0
    elif dist == "nan_inf":
        k[0, 0, L // 2, 3] = np.nan
    elif dist == "tiny":
        q, k, v = q * 1e-4, k * 1e-4, v * 1e-6
    f = lambda a: mx.array(a.astype(np.float16))
    return dict(q=f(q), k=f(k), v=f(v), bits=mx.array(bits.astype(np.int32)))


def allowed(bits, L, kd):
    """(B, kd, L) bool."""
    bits = np.array(bits).astype(np.int64)
    B = bits.shape[0]
    a = np.ones((B, kd, L), dtype=bool)
    j = np.arange(kd)
    a[:, :, L - kd:] = ((bits[:, :, None] >> j[None, None, :]) & 1).astype(bool)
    return a


def ref64(q, k, v, bits):
    q, k, v = (np.array(x.astype(mx.float32)).astype(np.float64) for x in (q, k, v))
    B, kd, Hq, d = q.shape
    Hkv, L = k.shape[1], k.shape[2]
    G = Hq // Hkv
    s = np.einsum("bihgd,bhld->bihgl", q.reshape(B, kd, Hkv, G, d), k) / math.sqrt(d)
    a = allowed(bits, L, kd)
    s = np.where(a[:, :, None, None, :], s, -np.inf)
    with np.errstate(invalid="ignore", over="ignore"):
        s = s - s.max(-1, keepdims=True)
        p = np.exp(s)
        p /= p.sum(-1, keepdims=True)
        o = np.einsum("bihgl,bhld->bihgd", p, v)
    return o.reshape(B, kd, Hq, d)


def _mask_mx(bits, L, kd):
    j = mx.arange(kd)
    tail = ((bits[:, :, None] >> j[None, None, :]) & 1).astype(mx.bool_)  # (B,kd,kd)
    head = mx.ones((bits.shape[0], kd, L - kd), mx.bool_)
    return mx.concatenate([head, tail], axis=-1)  # (B,kd,L)


# ---------------------------------------------------------------- MLX references
def mlx_sdpa(q, k, v, bits):  # q_len = kd per head, boolean mask (B,1,kd,L)
    B, kd, Hq, d = q.shape
    m = _mask_mx(bits, k.shape[2], kd)[:, None]
    o = mx.fast.scaled_dot_product_attention(q.transpose(0, 2, 1, 3), k, v, scale=1 / math.sqrt(d), mask=m)
    return o.transpose(0, 2, 1, 3)


def mlx_sdpa_fold(q, k, v, bits):  # GQA folded: kd*G rows per KV head
    B, kd, Hq, d = q.shape
    Hkv, L = k.shape[1], k.shape[2]
    G = Hq // Hkv
    qf = q.reshape(B, kd, Hkv, G, d).transpose(0, 2, 1, 3, 4).reshape(B, Hkv, kd * G, d)
    m = mx.repeat(_mask_mx(bits, L, kd), G, axis=1)[:, None]  # rows (i,g)
    o = mx.fast.scaled_dot_product_attention(qf, k, v, scale=1 / math.sqrt(d), mask=m)
    return o.reshape(B, Hkv, kd, G, d).transpose(0, 2, 1, 3, 4).reshape(B, kd, Hq, d)


def mlx_ops(q, k, v, bits):
    B, kd, Hq, d = q.shape
    Hkv, L = k.shape[1], k.shape[2]
    G = Hq // Hkv
    qf = q.reshape(B, kd, Hkv, G, d).transpose(0, 2, 1, 3, 4).reshape(B, Hkv, kd * G, d)
    s = (qf @ k.swapaxes(-1, -2)).astype(mx.float32) * (1 / math.sqrt(d))
    m = mx.repeat(_mask_mx(bits, L, kd), G, axis=1)[:, None]
    p = mx.softmax(mx.where(m, s, -mx.inf), axis=-1).astype(q.dtype)
    return (p @ v).reshape(B, Hkv, kd, G, d).transpose(0, 2, 1, 3, 4).reshape(B, kd, Hq, d)


# ---------------------------------------------------------------- PyTorch MPS references
def _mask_t(bits, L, kd):
    import torch
    j = torch.arange(kd, device=bits.device)
    tail = ((bits[:, :, None].long() >> j[None, None, :]) & 1).bool()
    head = torch.ones(bits.shape[0], kd, L - kd, dtype=torch.bool, device=bits.device)
    return torch.cat([head, tail], -1)


def torch_sdpa(q, k, v, bits):
    import torch.nn.functional as F
    m = _mask_t(bits, k.shape[2], q.shape[1])[:, None]
    return F.scaled_dot_product_attention(q.transpose(1, 2), k, v, attn_mask=m, enable_gqa=True).transpose(1, 2)


def torch_sdpa_fold(q, k, v, bits):
    import torch.nn.functional as F
    B, kd, Hq, d = q.shape
    Hkv, L = k.shape[1], k.shape[2]
    G = Hq // Hkv
    qf = q.reshape(B, kd, Hkv, G, d).permute(0, 2, 1, 3, 4).reshape(B, Hkv, kd * G, d)
    m = _mask_t(bits, L, kd).repeat_interleave(G, dim=1)[:, None]
    o = F.scaled_dot_product_attention(qf, k, v, attn_mask=m)
    return o.reshape(B, Hkv, kd, G, d).permute(0, 2, 1, 3, 4).reshape(B, kd, Hq, d)


MLX_REFS = dict(mlx_sdpa=mlx_sdpa, mlx_sdpa_fold=mlx_sdpa_fold, mlx_ops=mlx_ops)
TORCH_REFS = dict(torch_sdpa=torch_sdpa, torch_sdpa_fold=torch_sdpa_fold)


L1_REF = "mlx_sdpa"
L3_REF = "mlx_sdpa"
FUZZ = [(dict(B=1, L=8, k=8), "randn"), (dict(B=1, L=9, k=1), "randn"), (dict(B=2, L=4097, k=8), "randn"),
        (dict(B=1, L=100000, k=16), "randn"), (dict(B=3, L=777, k=5), "randn"), (dict(B=1, L=4096, k=31), "randn"),
        (dict(B=2, L=3000, k=8), "chain"), (dict(B=2, L=3000, k=8), "large"), (dict(B=2, L=3000, k=8), "needle_masked"),
        (dict(B=2, L=3000, k=8), "needle_boundary"), (dict(B=2, L=3000, k=8), "nan_inf"), (dict(B=2, L=3000, k=8), "tiny"),
        (dict(B=8, L=2049, k=4), "randn"), (dict(B=1, L=31, k=31), "needle_masked")]
L3_CASES = [(dict(B=1, L=1024, k=8), "randn"), (dict(B=1, L=32768, k=8), "randn"), (dict(B=1, L=65536, k=16), "randn"),
            (dict(B=2, L=8192, k=8), "large"), (dict(B=2, L=8192, k=8), "needle_boundary")]


