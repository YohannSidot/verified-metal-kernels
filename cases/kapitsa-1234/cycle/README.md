# Kapitsa #1234: one verify pass, op by op (text of the issue comment)

The text below was posted on https://github.com/nace-ai/kapitsa/issues/1234 on 2026-11-13. Files in this folder: the script (`kernl_kapitsa_cycle.py`), the sum (`cycle_sum.py`) and the run records, under the Apache License 2.0.

---

Understood on the licence: I am not sending any files. Here is the rest of the verify pass, as text. Short version:

- **By reading the code at `ea7625f5`:** on an M5, none of the nine quantized product classes of the 8-row verify pass does the plain step's arithmetic. Every one takes half-rounded activations (the router takes two half terms), where the step takes fp32.
- **Measured at op level, with my transcriptions:** exact forms of the classes I could transcribe sum to 1.32x to 1.46x the time of today's verify products at 8 rows, and 0.89x to 0.92x at 1 and 2 rows, where the exact form is the plain step's own kernel over a grid of rows. My R2 times include an `x8_prep2` that Kapitsa fuses (0.7 to 1.2 ms per pass), so these ratios understate the cost at 8 rows and overstate the gain at 1 and 2 rows. They are 1.72x to 1.95x faster than the products of 8 plain steps. Sums of median times of the products, not a throughput; the router's verify kernel is not transcribed and is outside these sums.
- **Where the cost is:** GDN in_proj and the LM head carry 11.0 of the 15.8 ms difference at 8 rows (overlapping routing). The MoE experts cost little: per block, today's path takes 0.96 to 1.09 times the exact form's time (1.1 of the 15.8 ms), consistent with today's `moe8e` passes already being scalar fp32 FMA chains (on half inputs).

Source read at `ea7625f5` (`src/model/qwen35.cpp`, `src/model/qwen35_v8.cpp`, `src/kernels/{qmvf,qmv,moe,verify8}.metal`, `src/kernels/quant.h`, `src/kernels/tensor/verify8t.metal`). On `main` since then (3 commits), `qwen35.cpp` changed one comment line (c8915928) and the other files did not change. Pack: `nace-ai/Qwen3.6-35B-A3B-MLX-KQuant` @ 37e804f9, `config.json`: 30 linear-attention and 10 full-attention layers, 40 MoE blocks (256 experts, top 8, intermediate 512, shared expert 512), hidden 2048, vocab 248320; affine 4-bit group 64 except `mlp.gate` and `mlp.shared_expert_gate` at 8 bits.

## 1. The map: plain step vs 8-row verify on an M5 (by reading)

| Op | N x K, bits | per pass | Plain step | 8-row verify on M5 (defaults) | Same arithmetic? |
|---|---|---|---|---|---|
| GDN in_proj qkv\|z\|b\|a | 12352 x 2048, 4 | 30 | `qmvf_b4_t1_r4_norm`, fp32 inputs, norm folded (qwen35.cpp:2378, 2334) | X8/T table (`x8_prep2`, or written by the previous layer's `moe8_combine_x8`) + `qmm8t_b4_nt16` (qwen35_v8.cpp:1332-1334) | No: inputs rounded to half (verify8.metal:236), matrix-unit order inside a group (verify8t.metal:14-18) |
| GDN out_proj | 2048 x 4096, 4 | 30 | `qmv2_b4_g64_r2_add` on the fp32 output of `gdn_step` (qwen35.cpp:2393, 2184-2196) | `gdn_gnorm_x8t` writes a half T table, then `qmm8tp_b4_nt16`, ks 4, residual add (qwen35_v8.cpp:1379-1395, 613-622) | No: half inputs, other order |
| Attention q\|k\|v (q with its gate) | 9216 x 2048, 4 | 10 | `qmvf_b4_t1_r4_norm` (qwen35.cpp:2400) | table + `qmm8t_b4_nt16` (qwen35_v8.cpp:803-804) | No: half inputs, other order |
| Attention o_proj | 2048 x 4096, 4 | 10 | `qmv2_b4_g64_r2_add` (qwen35.cpp:2435) | `x8_prep2` without norm + `qmm8tp_b4_nt16` (qwen35_v8.cpp:924-925) | No: half inputs, other order |
| Router + shared-expert gate | 257 x 2048, 8 | 40 | `qmvf_b8_t1_r4_norm`; also writes the normalized fp32 row the experts read (qwen35.cpp:2452) | `x8_prep2` with `write_lo` (inputs as hi + lo, two halves, ~22 bits) + `qmm8x2_b8_nf1`, simdgroup MMA (qwen35_v8.cpp:1032-1033, 1061-1062; verify8.metal:312-315) | No: two-half inputs instead of fp32, MMA order |
| Routed experts gate\|up + SwiGLU | 8 x (2 x 512 x 2048), 4 | 40 | `moe_gate_up3_r2` (`qmv2_rows` on the fp32 normalized row; h in fp32) (qwen35.cpp:2447, 2462-2471; moe.metal:420-462) | `moe8e_gu_r2` (default on Apple10: `NI_V8_MOE_EMU` 2 once `mma_chain_check` passes and with the default gate/up kernel) on the half table; h rounded to half (qwen35_v8.cpp:986-1006, 1046-1048, 1139-1140; verify8.metal:1292-1293, 1368) | No: half inputs, half h, the MMA tile's k order |
| Routed experts down | 8 x (2048 x 512), 4 | 40 | `moe_down2`: fp32 h, weighted sum over slots then `x +=` (qwen35.cpp:2472-2477; moe.metal:491-518) | `moe8e_down_r2` on half h, codes as 1024 + q, then `moe8_combine` / `moe8_combine_x8` (verify8.metal:1425-1522, 1085-1095) | No: half inputs, other order |
| Shared expert gate\|up, down | 512 x 2048, 2048 x 512, 4 | 40 | same kernels, slot k | `moe8e` 2-row tiles (split 8, shs 1) (qwen35_v8.cpp:1050-1059) | No (as the routed experts) |
| LM head | 248320 x 2048, 4 | 1 | `qmvf_b4_t1_r4_norm` with the final norm (qwen35.cpp:2541, 2178) | the last `moe8_combine_x8` writes the final norm's half table, then `qmm8tp_b4_nt32` (qwen35_v8.cpp:1165-1174, 1408-1413; `v8t_plan`: nt 32 above 8192 tiles) | No: half inputs, other order |

Every quantized product class of the verify pass on an M5 takes different inputs than the plain step: half-rounded activations everywhere, and for the router two half terms. The `moe8e` passes are row-independent inside a verify (their comment: bitwise what the 8-row MMA tiles give); they do not reproduce the plain step's `moe_gate_up3` / `moe_down2`.

Other ops, by reading only: the embedding is the same function at R rows (`enc_embed`, qwen35_v8.cpp:1291, qwen35.cpp:2516). The router top-k runs inside `moe8_plan`; its comment says "the same topk256_row code" (qwen35_v8.cpp:1065); not checked. The combine sums the slots in the same order as `moe_down2` on paper (verify8.metal:1080), on inputs that already differ. GDN conv/prep, the GDN state step and gated norm, attention over the cache, and sampling were not read for this map: not determined.

## 2. Exact forms, measured at op level

Method, same as the layer-0 table: my transcriptions of the kernels named above into `mx.fast.metal_kernel` bodies, compiled with `math_mode="fast"`; synthetic weights in the pack's layout; nothing from Kapitsa is run. For each class: R1 = the plain step's kernel; R2 = today's M5 verify path; K = a multi-row form whose every row is meant to carry R1's bits:

- GDN in_proj, q|k|v, LM head: the `qmvf` source at 2 token rows per thread and 8 weight rows per simdgroup for M >= 4, R1 on an M-row grid for M <= 3 (as in the layer-0 table).
- GDN out / attention o: `qmv2_rows`' arithmetic with x read from device memory and 2 token rows per thread (4 weight rows x 4 simdgroups) for M >= 4, R1 on a grid for M <= 3.
- Router: R1 on an M-row grid. R2 (`qmm8x2_b8_nf1`) is not transcribed: no R2 column.
- MoE experts: R1's two kernels (`moe_gate_up3_r2`, `moe_down2`) with a grid index over the rows. Routing is given (the same indices and weights for every arm), either uniform (8 rows: ~58 distinct experts) or overlapping (each slot reuses an earlier row's expert with probability 0.4; 8 rows: ~40 distinct, near the "51 tiles of 39 experts" in the `moe8e` comment). R2 = `x8_prep2` + `moe8e_gu_r2` + `moe8e_down_r2` + `moe8_combine` over the 2-row tile list `moe8_plan` would build.

<details>
<summary>Equality table: every row of K against R1 run on that row alone</summary>

Equality: every row of K against R1 run on that row alone, float32 bit patterns. 2 weight sets x 4 seeds x M = 1..8 (MoE: 4 seeds; random inputs with uniform and with overlapping routing, edge inputs with uniform routing), random inputs and edge inputs (NaN, 3e38 and 1e18 entries, zero rows, subnormal rows and entries).

| Class | K: rows equal to R1 alone (random / all inputs) | R2: rows equal to R1 (random / all) | R2 max abs diff vs R1 (random) | max abs error vs float64: R1 = K / R2 |
|---|---|---|---|---|
| GDN in_proj 12352 x 2048 | 288/288 / 576/576 | 0/288 / 139/576 | 0.0049 | 5e-06 / 0.0048 |
| attention q\|k\|v 9216 x 2048 | 288/288 / 576/576 | 0/288 / 139/576 | 0.005 | 4.7e-06 / 0.005 |
| GDN out / attention o 2048 x 4096 | 288/288 / 576/576 | 0/288 / 193/576 | 0.01 | 8.4e-06 / 0.0088 |
| router + shared gate 257 x 2048 (8-bit) | 288/288 / 576/576 | not transcribed | - | 3.8e-06 / - |
| MoE experts (8 + shared), block output | 288/288 / 432/432 | 0/288 / 123/432 | 0.041 | 5e-05 / 0.038 |
| LM head 248320 x 2048 | 288/288 / 576/576 | 0/288 / 139/576 | 0.0054 | 5.4e-06 / 0.005 |

The router's K also writes the normalized row bitwise equal to R1's (576/576). Output magnitudes: up to ~25 for the dense classes, ~40 for o, MoE block contributions up to ~100 (synthetic data: not comparable with the values in this issue).

</details>

<details>
<summary>Times per class, 1 to 8 rows</summary>

Times: 30 interleaved rounds, median microseconds per product (chains of products over distinct weight sets; LM head: 4 products over 3 weight sets of 273 MiB; MoE: a fresh routing per product). Ratios: median over rounds of the per-round ratio; above 1, K is faster.

| Class | M | K us | R2 us | M x R1 us | R2 / K | M x R1 / K |
|---|---|---|---|---|---|---|
| GDN in_proj | 1 | 120.6 | 129.6 | 120.9 | 1.08 | 1.00 |
| | 2 | 120.3 | 130.3 | 224.0 | 1.08 | 1.87 |
| | 4 | 173.5 | 132.3 | 425.1 | 0.76 | 2.46 |
| | 8 | 343.6 | 136.1 | 837.6 | 0.40 | 2.43 |
| attention q\|k\|v | 1 | 91.4 | 99.8 | 91.6 | 1.09 | 1.00 |
| | 2 | 94.9 | 101.9 | 160.5 | 1.07 | 1.69 |
| | 4 | 139.3 | 103.6 | 295.6 | 0.74 | 2.12 |
| | 8 | 269.6 | 106.4 | 562.9 | 0.40 | 2.09 |
| GDN out / attention o | 1 | 45.5 | 59.0 | 45.4 | 1.30 | 1.00 |
| | 2 | 48.1 | 59.5 | 74.5 | 1.24 | 1.55 |
| | 4 | 61.3 | 59.7 | 133.3 | 0.98 | 2.18 |
| | 8 | 110.4 | 59.1 | 248.8 | 0.53 | 2.26 |
| router + shared gate | 1 | 15.0 | - | 15.1 | - | 1.01 |
| | 2 | 15.1 | - | 18.3 | - | 1.21 |
| | 4 | 19.9 | - | 21.4 | - | 1.09 |
| | 8 | 23.6 | - | 36.9 | - | 1.55 |
| LM head | 1 | 2306.3 | 2597.7 | 2306.3 | 1.13 | 1.00 |
| | 2 | 2484.6 | 2663.1 | 4462.0 | 1.08 | 1.81 |
| | 4 | 3779.8 | 2692.7 | 8875.4 | 0.72 | 2.36 |
| | 8 | 7659.6 | 2874.7 | 19049.0 | 0.37 | 2.46 |
| MoE, uniform routing | 1 | 141.8 | 154.1 | 141.1 | 1.09 | 1.00 |
| | 2 | 248.6 | 264.2 | 266.0 | 1.06 | 1.07 |
| | 4 | 449.7 | 462.5 | 504.5 | 1.03 | 1.12 |
| | 8 | 812.6 | 831.8 | 978.6 | 1.02 | 1.20 |
| MoE, overlapping routing | 1 | 141.6 | 154.5 | 141.8 | 1.09 | 1.00 |
| | 2 | 208.1 | 222.0 | 252.3 | 1.07 | 1.21 |
| | 4 | 335.7 | 341.5 | 477.2 | 1.02 | 1.42 |
| | 8 | 621.2 | 593.6 | 939.9 | 0.96 | 1.51 |

</details>

A second harness (float64 reference, fuzzed sizes from K = 256 to 4096, bitwise batch-invariance probes) accepted the GDN out / o form (L1 5/5, L2 15/15, L3 4/4, invariance 7/7; run on the matrix alone, with the residual set to zero) and timed it at 130.2 us against 77.3 us for R2 and 159.3 us for R1 on a grid at 8 rows (median of per-round ratios against R1 on a grid: 1.23 at 8 rows, 1.00 to 1.23 over 2 to 8 rows). The `qmvf` form used for GDN in_proj, q|k|v and the LM head is the one that harness accepted for the layer-0 table (31/31). GDN in_proj was re-run here: 343.6 against 136.1 us at 8 rows in this run, 334.0 against 134.1 in the first full run, and 336.6 against 137.2 in the layer-0 table above.

The MoE line is the one that differs from the dense ones: today's `moe8e` passes are already scalar fp32 FMA chains over 2-row tiles, so the plain step's own kernels over a row grid run in about the same time here (0.96 to 1.09). The dense products are where exactness costs: at 8 rows K takes 2.5 to 2.7 times R2 on GDN in_proj, q|k|v and the LM head, 1.9 times on GDN out / o.

## 3. One verify pass, summed

Sum over the products of one pass of (count x median time): 30 GDN in_proj, 40 GDN out / attention o, 10 q|k|v, 40 MoE expert blocks, 1 LM head. A ratio of sums of medians, not a throughput. In this table K / R2 above 1 means the exact form is slower.

| Routing | M | today's verify (R2), ms | exact (K), ms | K / R2 | M plain steps' products (M x R1), ms | (M x R1) / K |
|---|---|---|---|---|---|---|
| uniform | 1 | 16.0 | 14.3 | 0.90 | 14.3 | 1.00 |
| uniform | 2 | 20.5 | 18.9 | 0.92 | 26.4 | 1.40 |
| uniform | 4 | 28.6 | 30.8 | 1.08 | 50.1 | 1.63 |
| uniform | 8 | 43.7 | 57.6 | 1.32 | 98.9 | 1.72 |
| overlapping | 1 | 16.0 | 14.3 | 0.89 | 14.3 | 1.00 |
| overlapping | 2 | 18.9 | 17.3 | 0.92 | 25.9 | 1.50 |
| overlapping | 4 | 23.7 | 26.3 | 1.11 | 49.0 | 1.87 |
| overlapping | 8 | 34.1 | 49.9 | 1.46 | 97.4 | 1.95 |

At 8 rows with overlapping routing, the 15.8 ms difference splits as: GDN in_proj 6.2, LM head 4.8, GDN out / o 2.1, q|k|v 1.6, MoE 1.1. Router (not in the sums): K 0.94 ms per pass at 8 rows; its R2 is not measured.

What is in the sums: the quantized products and, on the R2 side, an `x8_prep2` before each dense product and before each MoE block. What is not: the router (R2 missing), attention over the cache, the GDN conv/prep and state update, norms outside the products, top-k and `moe8_plan`, sampling, the drafter, dispatch gaps inside a real pass.

<details>
<summary>Configurations tried</summary>

Configurations tried (all bitwise checks on my compiled kernels; development runs). LM head `qmvf` at (token rows per thread, weight rows per simdgroup): (2,8), the one kept, and (1,4), (2,4), (2,16), (8,4), (8,2) were bitwise equal to R1 on 24 of 24 rows, (4,8) on 10 of 24; at 8 rows they ran 6.4, 8.4, 7.1, 9.7, 48.6, 47.5 and 8.8 ms against 2.4 to 2.6 ms for R2. GDN out / o at (token rows per thread, weight rows, simdgroups): (2,4,4), the one kept, (2,2,8) and (1,2,8) equal on 96 of 96 rows, 110.4, 126.3 and 135.3 us at 8 rows; (8,2,8) and (8,1,8) equal on 96 of 96, 868.8 and 1013.2 us; (4,2,8) and (4,4,2) equal on 41 of 96, 145.5 and 153.0 us; R1 on a grid 139.6 to 142.8 us and R2 58.9 to 60.7 us in the same runs. Router: 2 token rows per thread, 8 weight rows: equal on 576 of 576 rows, 1.17 to 1.48 times slower than the grid (1.16 to 1.53 in the first full run).

</details>

Runs: Apple M5 (base, 10-core GPU), macOS 26.6.2, MLX 0.32.4, on mains power, lid open, 2026-11-13; no model loaded (synthetic weights in the pack's layout). Two full runs; tables from the second. The first used R1 on a grid as K for GDN out / o; its sums gave 1.29x to 1.46x at 8 rows, 0.91x to 0.93x at 1 and 2 rows, and 1.64x to 1.98x against 8 plain steps.

## Not verified

- These are my transcriptions compiled by MLX, not your binaries. Under fast math the bits depend on the compiled kernel: "equal to R1" means equal to my R1. The check has to be run inside Kapitsa.
- Synthetic weights, inputs and routing; the two routing models are mine.
- One machine, a base M5. Only the M5 verify paths are transcribed (`qmm8t*`, `moe8e`); other chips take `qmm8snb`-family kernels and, without `NI_V8_MOE_EMU`, MMA tiles for the experts. I did not run `mma_chain_check`.
- R2 times include an `x8_prep2` that Kapitsa fuses into the previous kernel for GDN in_proj, q|k|v, GDN out and the LM head (`moe8_combine_x8`, `gdn_gnorm_x8t`): about 10 to 17 us per call here (MLX dispatch included), 0.7 to 1.2 ms over the ~70 tables of a pass that are fused in Kapitsa. The R2 column is an upper bound for those classes.
- The MoE R2 time includes the norm; the MoE K time does not (in the step, the router kernel writes the normalized row).
- The MoE and router K forms (R1's kernels with a grid index) were checked by the bitwise test above only, not by the second harness.
- Wide tree verifies (16 / 32 rows, `qmm8t2`) and sampled verifies (`set_verify_moe_exact`) are not covered.
- Whether exact products make the T=0 text identical is not shown: attention, the GDN state, top-k and sampling are not checked.

Disclosure: I do this kind of work commercially. The map, the transcriptions, the harness and this write-up were produced by Claude (Anthropic's models) under my direction; I answer for them.
