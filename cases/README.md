# Measurement cases

Op-level measurements I posted on public threads of Apple Silicon inference projects, each with its limits. They are
independent work: not affiliated with or endorsed by the projects named. Every number comes from one base Apple M5
(10-core GPU, 24 GB), from rewrites or transcriptions of the project's code path with synthetic weights, never from
the project's engine. The linked thread or folder states the full limits; read them before using a number.

| Thread | Question | Result on a base M5 | Artifact |
|---|---|---|---|
| [nace-ai/kapitsa #1234](https://github.com/nace-ai/kapitsa/issues/1234) | What does an 8-row verify cost if every row must carry the bits of the one-row step (layer-0 input projections)? | Kapitsa's own `qmvf` template set to 2 token rows per thread and 8 weight rows per simdgroup: at 4 to 8 rows, 480 of 480 rows bitwise equal to our transcription of the one-row step, compiled by MLX (not checked against Kapitsa's own binary; under fast math the bits depend on the compiled kernel). At 8 rows, about 2.45x the time of our transcription of today's M5 verify, and 2.55x faster than 8 one-row steps. | [`kapitsa-1234/`](kapitsa-1234/) (Apache-2.0) |
| [Augustas11/macprovider #2335](https://github.com/Augustas11/macprovider/issues/2335) | Does the `qmv_mma` route cost one-row decode speed, and is the stock one-row path plus a row-exact kernel faster from 1 to 16 rows? | On an M5 (with NAX), not the M1 of the issue, so the ratios do not carry over: faster at 1 and 2 rows (1.38x to 1.52x) and 4 rows (1.14x to 1.19x), slower at 8 and 16 rows (0.59x to 0.85x), against our transcription of `qmv_mma` (never compared bit for bit with the fork); `qmv_mma1` not measured. | numbers in the thread |
| [Layr-Labs/d-inference #2068](https://github.com/Layr-Labs/d-inference/issues/2068) | Can 2-bit prompt rows be row-exact and still read the weights once for many rows? | 1.83x to 2.21x the faster of our two rewrites of the per-row path, from 128 rows (geometric mean 2.02x); 5 to 12% slower at one row; its bytes are not those of MLX's one-row kernel. A second kernel that keeps MLX's one-row bytes: 1.21x from 128 rows. Both far behind the batched, non-row-exact path (0.21x to 0.26x and 0.12x to 0.16x). | write-up in the thread |
| [lablup/mlxcel #2979](https://github.com/lablup/mlxcel/issues/2979) | Shared-prefix decode on a paged KV cache | 1.58x to 4.18x our MLX rewrite of mlxcel's flat paged launch (not its code or binary; geometric mean 2.53x, fp16, op in isolation); never compiled or run on M1 to M4. | write-up in the thread |
| [Neywa-Labs/mlx-vlm #2714](https://github.com/Neywa-Labs/mlx-vlm/issues/2714) | Where does the MoE decode time of point 4 go? | On one GLM-5.3-Flash MoE layer rebuilt from MLX ops (not mlx-vlm's code), the fp16 workaround described in the issue gives 8.8x at one row (3.6x at 4 rows); the time goes to a per-call conversion of all experts' scales and biases (10.4 of 11.7 ms). | numbers in the thread |

Each line says where the approach loses, when it does.

Licenses: `kapitsa-1234/` is under the Apache License 2.0; the rest of this repository is under its evaluation license (see the top-level README).

The code and write-ups were produced by Claude models under my direction; I answer for them. None of these
measurements was commissioned or paid for, and none of these projects is a customer. To commission a kernel or a
measurement like these, see [Commissioning a kernel](../README.md#commissioning-a-kernel).
