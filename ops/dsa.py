"""Recent op (2025-09): DeepSeek Sparse Attention (DSA) decode step, DeepSeek-V3.2 shapes (also GLM-5 family).
Per request b (one new token):
  indexer:  score[l] = sum_h w[h] * relu(iq[h] . ik[l])      (Hi=64 heads x 128, ik = indexer key cache)
  select :  idx = top-2048(score) over valid positions (all if L <= 2048)
  attend :  MLA in absorbed (MQA) form: 128 query heads, latent cache rows of 576 = 512 latent + 64 rope;
            out = softmax(q . kv[idx]^T * scale) @ kv[idx][:, :512]     scale = 1/sqrt(192) (q_head_dim 128+64)
Shapes: q (B,128,576) fp16, kv (B,L,576) fp16, iq (B,64,128) fp16, ik (B,L,128) fp16, w (B,64) fp32 (pre-scaled),
lens (B,) int32 or None.  out (B,128,512) fp16.  The projections producing q/iq/w are outside the op (as in llama.cpp's
LIGHTNING_INDEXER + TOP_K + FLASH_ATTN_EXT decomposition)."""
import math
import numpy as np
import mlx.core as mx

NAME = "dsa"
H, DC, DR, HI, DI, TOPK = 128, 512, 64, 64, 128, 2048
DK = DC + DR
SCALE = 1.0 / math.sqrt(128 + 64)
L1_SHAPE = dict(B=2, L=8192)
BENCH_SHAPES = [dict(B=1, L=8192), dict(B=1, L=32768), dict(B=1, L=65536), dict(B=4, L=16384)]


def make_inputs(shape, seed=0, dist="randn"):
    rng = np.random.default_rng(seed)
    B, L = shape["B"], shape["L"]
    q = rng.standard_normal((B, H, DK), dtype=np.float32)
    kv = rng.standard_normal((B, L, DK), dtype=np.float32)
    iq = rng.standard_normal((B, HI, DI), dtype=np.float32)
    ik = rng.standard_normal((B, L, DI), dtype=np.float32)
    w = rng.standard_normal((B, HI), dtype=np.float32) * (HI ** -0.5 * DI ** -0.5)
    lens = None
    if dist == "large":
        q, kv = q * 6.0, kv * 6.0
    elif dist == "needle":  # one strongly-indexed position per request, also dominant in attention
        for b in range(B):
            p = rng.integers(0, L)
            ik[b, p] = np.sign(w[b] @ np.ones(1) if False else 1.0) * iq[b].mean(0) * 0  # neutral
            ik[b, p] = (iq[b] * np.sign(w[b])[:, None]).mean(0) * 8.0
            kv[b, p, :] = q[b].mean(0) * 4.0
            kv[b, p, :DC] += 5.0
    elif dist == "ragged":
        lens = rng.integers(1, L + 1, (B,)).astype(np.int32)
        lens[0] = min(L, 1000)
        if B > 1:
            lens[1] = L
    elif dist == "nan_inf":
        kv[0, L // 2, 3] = np.nan  # may or may not be selected -> pattern must match the fp64 reference
    elif dist == "tiny":
        q, kv = q * 1e-3, kv * 1e-3
    f = lambda a: mx.array(a.astype(np.float16))
    return dict(q=f(q), kv=f(kv), iq=f(iq), ik=f(ik), w=mx.array(w), lens=None if lens is None else mx.array(lens))


def ref_select64(iq, ik, w, lens=None):
    iq, ik = (np.array(x.astype(mx.float32)).astype(np.float64) for x in (iq, ik))
    w = np.array(w).astype(np.float64)
    B, L = ik.shape[0], ik.shape[1]
    s = np.einsum("bhd,bld->bhl", iq, ik)
    s = (np.maximum(s, 0) * w[:, :, None]).sum(1)  # (B, L)
    sel = []
    for b in range(B):
        n = L if lens is None else int(np.array(lens)[b])
        if n <= TOPK:
            sel.append(np.arange(n))
        else:
            sel.append(np.argpartition(-s[b, :n], TOPK - 1)[:TOPK])
    return s, sel


def ref64(q, kv, iq, ik, w, lens=None, sel=None):
    if sel is None:
        _, sel = ref_select64(iq, ik, w, lens)
    q, kv = (np.array(x.astype(mx.float32)).astype(np.float64) for x in (q, kv))
    B = q.shape[0]
    out = np.zeros((B, H, DC))
    for b in range(B):
        ks = kv[b, sel[b]]  # (k, 576)
        with np.errstate(invalid="ignore", over="ignore"):
            s = q[b] @ ks.T * SCALE
            s = s - s.max(-1, keepdims=True)
            p = np.exp(s)
            p /= p.sum(-1, keepdims=True)
            out[b] = p @ ks[:, :DC]
    return out


# ---------------------------------------------------------------- MLX-LM reference (deepseek_v32.py decode path)
def mlx_lm(q, kv, iq, ik, w, lens=None):
    """Faithful to mlx-lm 0.32.2 DeepseekV32Attention / Indexer decode path (L == 1), absorbed MLA form."""
    B, L = kv.shape[0], kv.shape[1]
    scores = iq[:, :, None, :] @ ik[:, None].swapaxes(-1, -2)  # (B,Hi,1,L)
    scores = mx.maximum(scores, 0)
    scores = scores * w.astype(scores.dtype)[:, :, None, None]
    scores = scores.sum(axis=1, keepdims=True)  # (B,1,1,L)
    mask = None
    if lens is not None:
        mask = (mx.arange(L)[None, :] < lens[:, None])[:, None, None, :]
        scores = mx.where(mask, scores, -float("inf"))
    kv4 = kv[:, None]  # (B,1,L,576)
    if L > TOPK:
        idx = mx.argpartition(scores, kth=-TOPK, axis=-1)[..., -TOPK:]  # (B,1,1,TOPK)
        i4 = idx[:, :, 0, :, None]
        kv4 = mx.take_along_axis(kv4, mx.broadcast_to(i4, i4.shape[:-1] + (DK,)), axis=2)
        if mask is not None:
            mask = mx.take_along_axis(mask, idx, axis=-1)
    latent, k_pe = kv4[..., :DC], kv4[..., DC:]
    q4 = q[:, :, None, :]  # (B,H,1,576)
    q_nope, q_pe = q4[..., :DC], q4[..., DC:]
    pe_scores = (q_pe * SCALE) @ k_pe.swapaxes(-1, -2)  # (B,H,1,k)
    if mask is not None:
        pe_scores = mx.where(mask, pe_scores, mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype))
    o = mx.fast.scaled_dot_product_attention(q_nope, latent, latent, scale=SCALE, mask=pe_scores)
    return o[:, :, 0, :]


# ---------------------------------------------------------------- PyTorch MPS reference (same algorithm)
def torch_ref(q, kv, iq, ik, w, lens=None):
    import torch, torch.nn.functional as F
    B, L = kv.shape[0], kv.shape[1]
    s = torch.relu(torch.matmul(iq, ik.transpose(1, 2)))  # (B,Hi,L)
    s = (s.float() * w[:, :, None]).sum(1)  # (B,L)
    if lens is not None:
        s = s.masked_fill(torch.arange(L, device=s.device)[None, :] >= lens[:, None], float("-inf"))
    m = None
    if L > TOPK:
        idx = torch.topk(s, TOPK, dim=-1).indices  # (B,TOPK)
        ks = torch.gather(kv, 1, idx[:, :, None].expand(-1, -1, DK))
        if lens is not None:
            m = (idx < lens[:, None])[:, None, None, :]
    else:
        ks = kv
        if lens is not None:
            m = (torch.arange(L, device=s.device)[None, :] < lens[:, None])[:, None, None, :]
    return F.scaled_dot_product_attention(q[:, :, None, :], ks[:, None], ks[:, None, :, :DC], attn_mask=m,
                                          scale=SCALE)[:, :, 0, :]


MLX_REFS = dict(mlx_lm=mlx_lm)
TORCH_REFS = dict(torch_ref=torch_ref)


def chain_mlx(out, kw):  # next layer's absorbed query = [this layer's output (512) | its rope part (64)]
    return mx.concatenate([out, kw["q"][..., DC:]], axis=-1)


def chain_torch(out, kw):
    import torch
    return torch.cat([out, kw["q"][..., DC:]], dim=-1)


def mlx_fp32_ref(q, kv, iq, ik, w, lens=None):
    """Same algorithm as mlx_lm, computed in fp32 (the L1 'reference module' output)."""
    f = lambda a: a.astype(mx.float32)
    return mlx_lm(f(q), f(kv), f(iq), f(ik), w, lens).astype(mx.float16)


L1_REF = "mlx_fp32_ref"
L3_REF = "mlx_lm"
FUZZ = [(dict(B=1, L=1), "randn"), (dict(B=1, L=2048), "randn"), (dict(B=1, L=2049), "randn"), (dict(B=2, L=5003), "randn"),
        (dict(B=1, L=131072), "randn"), (dict(B=3, L=9000), "ragged"), (dict(B=1, L=30), "randn"),
        (dict(B=2, L=6000), "large"), (dict(B=2, L=6000), "needle"), (dict(B=2, L=6000), "nan_inf"),
        (dict(B=2, L=6000), "tiny"), (dict(B=4, L=3000), "ragged")]
L3_CASES = [(dict(B=1, L=1500), "randn"), (dict(B=1, L=8192), "randn"), (dict(B=1, L=65536), "randn"),
            (dict(B=2, L=8192), "large"), (dict(B=2, L=8192), "needle")]


def mlx_smart(q, kv, iq, ik, w, lens=None):
    """Strongest MLX composition written for this project: indexer as ONE GEMM per request (ik @ iq^T, uses MLX's M5
    neural-accelerator matmul), fp32 head-weighted sum via a matvec, argpartition, gather, then attention with
    mx.fast SDPA on the latent part and the rope part as an additive mask (mlx-lm's trick)."""
    B, L = kv.shape[0], kv.shape[1]
    if L > TOPK:
        s = mx.maximum(ik @ iq.swapaxes(-1, -2), 0)  # (B, L, Hi) fp16, one GEMM per request
        scores = (s.astype(mx.float32) @ w[:, :, None])[..., 0]  # (B, L) fp32
        if lens is not None:
            scores = mx.where(mx.arange(L)[None, :] < lens[:, None], scores, -float("inf"))
        idx = mx.argpartition(scores, kth=-TOPK, axis=-1)[:, -TOPK:]
        kvs = mx.take_along_axis(kv, idx[:, :, None], axis=1)  # (B, TOPK, 576)
        mask = None
        if lens is not None:
            mask = (idx < lens[:, None])[:, None, None, :]
    else:
        kvs = kv
        mask = None if lens is None else (mx.arange(L)[None, :] < lens[:, None])[:, None, None, :]
    latent, k_pe = kvs[:, None, :, :DC], kvs[:, None, :, DC:]
    q4 = q[:, :, None, :]
    pe = (q4[..., DC:] * SCALE) @ k_pe.swapaxes(-1, -2)
    if mask is not None:
        pe = mx.where(mask, pe, mx.array(mx.finfo(pe.dtype).min, pe.dtype))
    return mx.fast.scaled_dot_product_attention(q4[..., :DC], latent, latent, scale=SCALE, mask=pe)[:, :, 0, :]


MLX_REFS = dict(mlx_lm=mlx_lm, mlx_smart=mlx_smart)


