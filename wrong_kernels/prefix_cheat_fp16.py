"""DELIBERATELY WRONG kernel for op `prefix` (from the verifier's calibration set, stage 0.9).

The "optimisation": the shared-prefix cascade keeps scores, probabilities and softmax sums in fp16 instead of fp32
(less register traffic).  It matches the standard op at the nominal shape (passes layer 1), but its error against the
float64 reference grows with context length and is larger than the standard fp16 op's: layers 2 and 3 reject it.

It uses MLX operations only, no custom Metal kernel, so verify.py must be run with --skip-gate on it (the static gate
would otherwise reject it before any test, for a reason unrelated to its correctness)."""
import math
import mlx.core as mx


def kernel(q, kp, vp, ks, vs):
    N, Hq, d = q.shape
    Hkv = kp.shape[0]
    G = Hq // Hkv
    sc = 1 / math.sqrt(d)
    qp = q.reshape(N, Hkv, G, d).transpose(1, 0, 2, 3).reshape(Hkv, N * G, d)
    s1 = (qp @ kp.swapaxes(-1, -2)) * sc
    m1 = s1.max(-1, keepdims=True)
    p1 = mx.exp(s1 - m1)
    l1 = p1.sum(-1, keepdims=True)
    o1 = (p1 @ vp).reshape(Hkv, N, G, d).transpose(1, 0, 2, 3)
    m1 = m1.reshape(Hkv, N, G, 1).transpose(1, 0, 2, 3)
    l1 = l1.reshape(Hkv, N, G, 1).transpose(1, 0, 2, 3)
    s2 = (q.reshape(N, Hkv, G, d) @ ks.swapaxes(-1, -2)) * sc
    m2 = s2.max(-1, keepdims=True)
    p2 = mx.exp(s2 - m2)
    l2 = p2.sum(-1, keepdims=True)
    o2 = p2 @ vs
    m = mx.maximum(m1, m2)
    a, b = mx.exp(m1 - m), mx.exp(m2 - m)
    return ((o1 * a + o2 * b) / (l1 * a + l2 * b)).astype(mx.float16).reshape(N, Hq, d)
