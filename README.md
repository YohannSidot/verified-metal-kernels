# Metal attention kernels for Apple M5, with their verifier and benchmark

Three GPU kernels for attention-style operations on Apple Silicon, the three-layer verifier that accepted them, the
interleaved benchmark that measured them, and two deliberately wrong kernels the verifier rejects. Everything needed
to re-check the per-op numbers below on your own M5.

**Commissions.** I write kernels like these for a given engine and operation, paid on result: see [Commissioning a kernel](#commissioning-a-kernel).

**Cases.** Measurements posted on public threads of other projects, with their limits and where our approach lost: see [`cases/`](cases/).

| Op | Kernel | What it computes |
|---|---|---|
| `prefix` | `kernels/prefix_opus55_v8.py` | Shared prefix: N requests share one prefix and have their own suffix, one decode token each |
| `spec` | `kernels/spec_opus55_v7.py` | Speculative verification: k draft rows (chain or tree mask) over a long cache |
| `dsa` | `kernels/dsa_opus55_v7.py` | DeepSeek Sparse Attention decode: indexer, top-2048, sparse MLA attention |

Each op is defined in `ops/<op>.py`: a NumPy float64 reference (the ground truth), the input generator, the
benchmark shapes, the fuzz and accuracy cases, and the references to beat (MLX and PyTorch MPS implementations).

## Who wrote the code

All code here was written by Claude models, in work directed by Yohann Sidot between 9 and 11 November 2026:

- the verifier, the benchmark, the op definitions and the references: Claude Opus 5.5, in an interactive session;
- the three kernels in `kernels/`: Claude Opus 5.5 inside an automated loop, with no tools and no file access, from
  the op definition and the harness reports on its earlier versions. No human edited the kernels;
- the two kernels in `wrong_kernels/`: written on purpose to cheat, as calibration of the verifier, by the same
  model in the same interactive session, so this calibration is weaker than an outsider's would be.

The same model family wrote the verifier and the kernels, so a blind spot could sit on both sides of the test. No one
outside the project has reviewed the verifier. The loop that generates kernels is not part of this package.

## Requirements

Tested on exactly one configuration:

- MacBook with **Apple M5** (base, 10-core GPU, 24 GB), **macOS 26** (26.6.2).
- Python 3.13, `mlx==0.32.4`, `torch==2.15.0`, `numpy==2.5.4`.
- Optional: llama.cpp Metal as an additional reference, see `llama_cpp/README.md`.

The kernels use Metal 4 tensor operations and have only been run on an M5 with macOS 26. They have not been tried
on M1 to M4 or on any other macOS version, and may not compile or may be slow there.

```bash
python3.13 -m venv .venv
.venv/bin/pip install mlx==0.32.4 torch==2.15.0 numpy==2.5.4
```

## Three commands

```bash
# 1. Verify a kernel (three layers; op inferred from the file name)
.venv/bin/python verify.py kernels/prefix_opus55_v8.py     # also kernels/spec_opus55_v7.py, kernels/dsa_opus55_v7.py

# 2. Benchmark the three kernels against every available reference, interleaved (add --rounds 10 for a quick run)
.venv/bin/python bench.py

# 3. See the verifier reject a wrong kernel
.venv/bin/python verify.py wrong_kernels/spec_cheat_nomax.py --skip-gate    # also wrong_kernels/prefix_cheat_fp16.py
```

`--skip-gate` is needed only for `wrong_kernels/`: they use MLX operations and no custom Metal kernel, so the static
gate (which requires one) would reject them before any test, for a reason unrelated to correctness. Run command 3
without the flag to see that rejection too.

## What you should get

**1. Verify.** Each shipped kernel prints `=> CORRECT`, with every case passing: prefix L1 5/5, L2 12/12, L3 5/5;
spec 5/5, 14/14, 5/5; dsa 5/5, 12/12, 5/5. The layers are:

- L1: 5 seeds at the nominal shape, `allclose(atol=rtol=1e-2)` against the platform's standard op (MLX);
- L2: fuzzing of shapes and values (odd and edge shapes, extreme magnitudes, NaN/Inf, ragged lengths) against the
  float64 reference: same NaN/Inf pattern, normalised max error <= 1e-2;
- L3: max and mean error against float64, each of which must stay within 2x that of the standard fp16 op (with a
  floor of 2^-11).

An exception on any case is a failure. The kernel is imported and run only in a child process; the verdict is
computed in the parent, which never imports it.

**2. Benchmark.** The published numbers come from the project's own benchmark harness (30 interleaved rounds);
`bench.py` is a standalone version of it. Run from a clean copy of this package on the same M5 at 10 rounds, with
llama.cpp, it gave 2.49x (2.12–3.10) for shared prefix, 2.26x (1.23–3.23) for speculative verification and 2.30x
(1.43–4.04) for DSA decode; without llama.cpp, 2.49x, 2.19x and 2.52x. That was one 10-round run on the same machine
as the table below.

| Op | Speedup against the strongest reference per shape | Other loop runs |
|---|---|---|
| Shared prefix | 2.59x (2.15–3.38) | 2.55x, 2.45x |
| Speculative verification | 2.34x (1.32–3.44) | 1.75x, 2.03x |
| DSA decode | 2.31x (1.42–4.23) | 1.83x, 2.58x |

Conditions: one MacBook with an Apple M5 (10-core GPU, 24 GB), macOS 26, MLX 0.32.4, fp16 inputs, each op in
isolation, 30 interleaved rounds. A shape's speedup is the median over rounds of (reference time / kernel time)
against the fastest reference at that shape: MLX built-ins and compositions, PyTorch MPS, and llama.cpp Metal.
Geometric mean over the benchmark shapes, range in parentheses. "Other loop runs" are the best kernels of two
earlier, independent runs of the loop, not shipped here, measured before llama.cpp was added to the loop's
internal benchmark (their final numbers are also against the strongest reference): about ±30% between runs of the
same model.

Without the llama.cpp binary, `bench.py` says so at the top and bottom of its output, and measures against MLX and
PyTorch MPS only. In the published 30-round benchmark, llama.cpp was the strongest reference at two shapes (prefix N=32; dsa B=4); in
the 10-round package test it also tied MLX at prefix N=8, so without it those
shapes can read higher than published; for dsa B=4 the difference is large and lifts the dsa mean above the table.
Absolute times depend on the machine and its thermal state; compare the ratios. Results go to `results/bench_<op>.json`
with every per-round sample.

**3. Rejection.** `spec_cheat_nomax` (softmax without subtracting the maximum) passes L1 and L3 and fails L2 on the
`large` fuzz case, where it returns NaN and the float64 reference does not: `=> REJECTED`, L1 5/5, L2 13/14, L3 5/5.
`prefix_cheat_fp16` (scores and sums kept in fp16) passes L1 and fails L2 on one case and L3 on three: L1 5/5,
L2 11/12, L3 2/5. Both pass the standard test at the nominal shape; a check limited to that test would keep them.

## Files

| Path | Content |
|---|---|
| `verify.py` | Static gate + three-layer verifier, one file |
| `bench.py` | Interleaved A/B benchmark, one file (optional llama.cpp) |
| `ops/` | Op definitions: float64 reference, generator, shapes, fuzz cases, references to beat |
| `kernels/` | The three kernels, unmodified from the loop's output (SHA-256 of each file checked against the stored run output on 2026-11-11: identical.) |
| `wrong_kernels/` | Two deliberately wrong kernels from the verifier's calibration |
| `llama_cpp/` | Patch and build steps for the optional llama.cpp reference |

`verify.py` also takes your own kernel: a module with a top-level `def kernel(...)` taking the op's inputs as
keyword arguments (`python verify.py my_kernel.py --op dsa`). Reports of a wrong kernel that passes all three layers
are the most useful feedback this package can get.

## Limits

1. One machine, fp16. These are isolated-op numbers, not tokens per second, and nothing here was measured in a
   serving stack.
2. About ±30% between two loop runs of the same model (the "Other loop runs" column).
3. Some of the strongest references (`mlx_sdpa_fold`, `mlx_smart`, `mlx_cascade`) are compositions of MLX functions
   written for this project, not library entry points. The ops and the references were chosen for this project; a
   stronger reference that was not found would lower every ratio.
4. The kernels use Metal 4 tensor operations: M5 and macOS 26 only, untested anywhere else.
5. llama.cpp is timed in its own harness, warm, repeating the op on the same tensors (see `llama_cpp/README.md`).
6. "No Metal kernel found" for an op does not mean none exists.
7. The kernels read fp16 only; on bf16 inputs they produce invalid values. There is no bf16 variant.
8. A paged-cache variant of the shared-prefix kernel is not included: its strongest reference is an MLX rewrite of
   another engine's decode path, not that engine's code, and does not belong in a standalone sample.
9. This package itself was tested once, from a clean copy, on the same M5, with 10 benchmark rounds.

The verifier accepting a kernel only means it found nothing wrong. Nothing in this package says anything about
non-Apple hardware.

## Commissioning a kernel

I take paid commissions for kernels like these, for hardware that expert libraries serve poorly. Today that means
Apple Silicon (M5); a ROCm port of the loop exists but has not run on an AMD GPU yet.

- You name one operation, its shapes and dtype, and the code path you run today.
- I first tell you, with numbers, whether there is anything to gain. If there is not, I say so.
- If there is, within 5 business days you get the kernel source, a verification report (randomized tests, fuzzing of
  shapes and values, numerical error against a float64 reference) and an A/B script to run on your own machine.
- You pay only if, on your own measurement, the kernel is at least 1.2x faster than your current path (geometric mean
  over the agreed shapes, none below 0.95x): EUR 1,500 per operation for the first three customers, with a perpetual,
  non-exclusive licence to modify and ship it. Otherwise you owe nothing.

To start, open an issue on this repository with the operation and the shapes.

## License

Source-available for evaluation and non-commercial research only; see [LICENSE](LICENSE). Not open source.
Exception: [`cases/kapitsa-1234/`](cases/kapitsa-1234/) is under the Apache License 2.0 (its own `LICENSE-APACHE-2.0.txt`); its kernel sources are adapted from Kapitsa, an Apache-2.0 project.
