# What an exact 8-row verify costs: Kapitsa's GDN input projections (context: nace-ai/kapitsa #1234)

## What this does not show

- **These are our transcriptions, not Kapitsa's engine.** R1 and R2 below are Kapitsa kernels re-typed by us into
  `mx.fast.metal_kernel` bodies (MLX 0.32.4), from the source read as text at commit
  `ea7625f5a41992980c3adb5f6a4542cee7ff7416`. Nothing from Kapitsa was cloned, built or run. Nothing here was ever
  executed inside Kapitsa.
- **"Bitwise equal" means equal to our R1**, compiled by MLX with `math_mode="fast"`. Kapitsa's libraries are compiled
  with `MTLMathModeFast` too, but under fast math the bits depend on the compiled kernel, not only on the source (we
  saw it: with 3 or 4 token rows per thread instead of 2, 0 of 240 random rows at 4 to 8 rows equal R1; records
  `check_nt3.json` and `check_nt4.json`). Whether Kapitsa's own `qmvf_b4_t1_r4_norm`
  binary produces the same bits as our R1 is not known; the check has to be re-run inside Kapitsa.
- **Synthetic weights and activations** (uniform 4-bit codes, bf16 scales and biases, random activations), not the
  Qwen3.6-35B-A3B pack. The drift sizes below are not comparable to the ones in #1234.
- **One machine**: a base Apple M5 (10-core GPU), macOS 26.6.2. Not run on M4 Pro, M5 Pro, M1 Max or M5 Max. The
  verify path transcribed is the M5 one (qmm8t on the matrix units); the simdgroup-MMA path of other chips
  (`qmm8snb` etc.) is not transcribed.
- **One op, one layer**: layer 0's linear-attention input projections only. No routing, no GDN state, no end-to-end
  decode speed.

## What was measured

Op: `qkv | z | b | a` = 2048 -> 8192 | 4096 | 32 | 32 (12352 outputs), MLX affine 4-bit, group 64, bf16 scales and
biases, input RMSNorm (bf16 weight, eps 1e-6) folded in, fp32 out, 1 to 8 token rows.

| Arm | What it is |
|---|---|
| R1 | `qmvf_b4_t1_r4_norm` (`src/kernels/qmvf.metal`, as dispatched by `Qwen35::enc_qmvf`: 1 token row per thread, 4 weight rows per simdgroup, 2 simdgroups), fp32 activations, norm folded into the dot |
| r1_calls | M one-row R1 dispatches: the trivial exact path |
| r1_grid | R1 over an M-row grid, one dispatch (per row the same work as one row alone) |
| R2 | `x8_prep2` (norm, inputs rounded to half, group sums, T table) then `qmm8t_b4_nt16` (Metal 4 `matmul2d` on the matrix units, fp32 epilogue per 64-group, 8 k-slices), as `v8t_plan` picks for 12352 x 2048 on an M5 |
| K | the same `qmvf` source with **2 token rows per thread and 8 weight rows per simdgroup**, one dispatch, for M >= 4; for M <= 3, r1_grid itself |

K is Kapitsa's own template with other parameters (the header of its `qmvf.metal` already describes threads that hold NT
token rows). What we add is the measurement at 4 to 8 rows and the bit-for-bit check.

**Faithful in R1 and R2**: every float operation, its order, the lane split of K, `simd_sum`, `precise::rsqrt`, the
half rounding and group sums of `x8_prep2` (with `prepu` = 1), the T-table permutation, the `matmul2d` descriptor
(16 x 8 x 64, uint8 codes x half inputs), the epilogue `fma(s, d, fma(b, sum, acc))` and the fixed-order k-slice sum.
**Simplified**: the four segments are one stacked matrix (per-row arithmetic unchanged; N's are multiples of 16);
parameters come from small device buffers instead of a `constant` struct; `nsg` is a constant 2 instead of
`[[simdgroups_per_threadgroup]]`; bfloat is read as its raw 16 bits (same value); no row tiles (wide tree verifies),
no prefetch variant, no `lo` table. R2's time includes `x8_prep2` writing the X8 table that `qmm8t` does not read,
as Kapitsa's does. At layers where its fused passes write the tables, R2 would be `qmm8t` alone.

## Results (Apple M5 base, MLX 0.32.4, 2026-11-13 14:00, on mains power, lid open, under `caffeinate -d -i`)

Equality (`--check`, 8 seeds, random and edge inputs: NaN, 3e38 and 1e18 entries, zero rows, subnormal rows and
entries): every row of K and of r1_grid against R1 run on that row alone, float32 bit patterns.

| | rows bitwise equal to R1 alone | elements differing |
|---|---|---|
| K (M = 1..8) | 576 / 576 (480 of them at M = 4..8, where K is not r1_grid) | 0 |
| r1_grid | 576 / 576 | 0 |
| R2 (today's verify) | 139 / 576 (all of them edge-input rows; 0 of the 288 random rows) | 5 370 066 |

Drift of R2 against R1, random inputs only (288 rows): qkv 0 / 288 rows equal, max |diff| 0.0047 (max |value| 26);
z 0 / 288, 0.0048; b 0 / 288, 0.0042; a 0 / 288, 0.0036. Error against a float64 reference at M = 8: R1 and K
5.0e-06 max abs, R2 4.7e-03 (outputs up to 24). That is the half-activation rounding the issue describes, reproduced
at op level on synthetic data.

Speed (`--bench`, 30 interleaved rounds, chains of 16 products over 16 weight sets of 13.6 MiB). Times: median
microseconds per product. Ratios: **median over rounds of the per-round ratio** (arm time / K time), so > 1 means K
is faster.

| M (live rows) | K us | R2 us | r1_grid us | r1_calls us | K vs R2 | K vs r1_grid | K vs r1_calls |
|---|---|---|---|---|---|---|---|
| 1 | 123.9 | 131.7 | 124.2 | 124.1 | 1.06 | 1.00 | 1.00 |
| 2 | 125.4 | 131.8 | 126.1 | 231.8 | 1.05 | 1.00 | 1.84 |
| 3 | 174.4 | 135.8 | 175.0 | 339.9 | 0.77 | 1.00 | 1.94 |
| 4 | 179.7 | 133.8 | 224.2 | 437.3 | 0.74 | 1.25 | 2.43 |
| 5 | 255.1 | 136.5 | 328.2 | 545.0 | 0.54 | 1.29 | 2.14 |
| 6 | 255.5 | 135.8 | 330.8 | 647.1 | 0.53 | 1.30 | 2.53 |
| 7 | 332.2 | 136.5 | 382.3 | 760.4 | 0.41 | 1.14 | 2.28 |
| 8 | 336.6 | 137.2 | 430.2 | 854.5 | 0.41 | 1.28 | 2.55 |

In one line: **at 8 live rows, an exact verify of these projections costs about 2.45 times today's verify (336.6
against 137.2 us), and runs 2.55 times faster than 8 plain steps.** At 1 and 2 live rows it is about as fast (1.05
to 1.06 here; 0.98 at 2 rows in the second harness). Today's verify time barely moves with the live rows (131.7 to
137.2 us); the exact path
pays one fp32 FMA chain per row and per weight, in the order of the one-row kernel, which the matrix units cannot
reproduce.

A second, independent harness (our judge: float64 reference, fuzzed sizes from K = 256 to 4096 and N = 32 to 12352,
bit-for-bit batch-invariance probes) accepted K (L1 5/5, L2 15/15, L3 4/4, invariance 7/7) and timed it the same
way: K against r1_grid 0.99 to 1.28 (geometric mean 1.16 over M = 2..8), against R2 0.98 at 2 rows and 0.42 at 8.

## What we tried that did not help

- Kapitsa's template with 3 or 4 token rows per thread (8 weight rows per simdgroup): under fast math it gives other bits
  than R1. At 4 to 8 rows, 0 of 240 random rows are bitwise equal to R1 (203 of 480 counting the edge-input rows),
  against 480 of 480 with 2 token rows per thread. Records: `check_nt3.json`, `check_nt4.json`
  (`--check --k-nt 3`, `--check --k-nt 4`; the two give the same counts case by case).
- From development runs whose records are not in this package: 8 token rows per thread matched R1 but was several
  times slower at 8 rows; our own reordering (weight words decoded once for all 8 rows, one token row at a time) was
  bitwise equal to R1 only when both were compiled with safe math, and no faster than r1_grid at 8 rows. Decoding is
  not the cost; the per-row FMA chains are.

## Run it

```
python kernl_kapitsa_verify_exact.py --check      # bit-for-bit equality and the R2 drift (8 seeds)
python kernl_kapitsa_verify_exact.py --bench      # 30 interleaved rounds, M = 1..8
python kernl_kapitsa_verify_exact.py --quick      # 1 seed, 5 rounds
python kernl_kapitsa_verify_exact.py --json out.json
```
MLX and NumPy only; Apple M5 (the R2 arm uses Metal 4 tensor operations, macOS 26).

Files: `kernl_kapitsa_verify_exact.py` (script), `run_m5.json` and `run_m5.txt` (the run above, every sample),
`check_nt3.json` and `check_nt4.json` (the equality check with 3 and 4 token rows per thread),
`verify_judge.json`, `bench_judge.json` and `calibration_judge.json` (the second harness),
`LICENSE-APACHE-2.0.txt` (the license of everything in this folder, copied from Kapitsa's LICENSE file),
`THIRD_PARTY_NOTICES.txt` (Kapitsa's credit to MLX and MLX's MIT notice).

## License

Everything here is under the Apache License 2.0 (`LICENSE-APACHE-2.0.txt`), the license of Kapitsa. The kernel
sources (R1, R2, and K, which is R1's source with other template parameters) are adapted from Kapitsa
(https://github.com/nace-ai/kapitsa, commit `ea7625f5a41992980c3adb5f6a4542cee7ff7416`); the list of changes is in
the script's header. The three Kapitsa source files carry no copyright header at that commit; Kapitsa's LICENSE
file states "Copyright 2026 Nace AI", retained here and at the end of `LICENSE-APACHE-2.0.txt`. Kapitsa credits MLX
(MIT License, Copyright © 2023 Apple Inc.) for matrix-fragment layouts and affine quantization formats; that credit
and notice are reproduced in `THIRD_PARTY_NOTICES.txt`. The transcriptions of the Kapitsa kernels and the harness
(synthetic data, float64 reference, check, benchmark) were written by Claude models under the direction of Yohann
Sidot. This is an independent measurement: it is not affiliated with or endorsed by the Kapitsa project.
