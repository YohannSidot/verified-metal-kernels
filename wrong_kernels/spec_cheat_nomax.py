"""DELIBERATELY WRONG kernel for op `spec` (from the verifier's calibration set, stage 0.9).

The "optimisation": softmax without subtracting the running maximum (one reduction less), exponentials in fp32.
On ordinary inputs the result matches (passes layer 1 and layer 3); with large scores exp() overflows and the output
is NaN where the float64 reference is finite: the fuzzing layer (layer 2) rejects it.

It uses MLX operations only, no custom Metal kernel, so verify.py must be run with --skip-gate on it."""
import math
import mlx.core as mx


def _mask(bits, L, kd):
    j = mx.arange(kd)
    tail = ((bits[:, :, None] >> j[None, None, :]) & 1).astype(mx.bool_)
    head = mx.ones((bits.shape[0], kd, L - kd), mx.bool_)
    return mx.concatenate([head, tail], axis=-1)


def kernel(q, k, v, bits):
    B, kd, Hq, d = q.shape
    Hkv, L = k.shape[1], k.shape[2]
    G = Hq // Hkv
    qf = q.reshape(B, kd, Hkv, G, d).transpose(0, 2, 1, 3, 4).reshape(B, Hkv, kd * G, d)
    s = (qf @ k.swapaxes(-1, -2)).astype(mx.float32) * (1 / math.sqrt(d))
    m = mx.repeat(_mask(bits, L, kd), G, axis=1)[:, None]
    e = mx.where(m, mx.exp(s), 0.0)
    p = (e / e.sum(-1, keepdims=True)).astype(q.dtype)
    return (p @ v).reshape(B, Hkv, kd, G, d).transpose(0, 2, 1, 3, 4).reshape(B, kd, Hq, d)
