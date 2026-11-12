# An LLM that writes GPU kernels is only as good as its verifier: three days on one Apple M5


Between 9 and 11 November 2026, I ran a loop in which a language model writes Metal kernels for attention-style operations, and a harness decides which ones to keep. One thing first: I did not write this code. All of it, the kernels but also the harness, the verifier, the op definitions and the references, was written by Claude models in sessions I directed. That is why this post spends more time on the verifier than on the speedups, and why it is meant to be checked rather than believed.

The short version: on one MacBook with an Apple M5, the loop found real speedups on three ops, tied on one, lost on two, and the gain that survives into tokens per second is smaller than the per-op number.

## Who wrote what, and what keeps it honest

In an interactive session I directed, Claude Opus 5.5 wrote the harness, the verifier, the op definitions, the references and a first set of kernels, which the project's reports call "manual" (I keep the word; no human typed them). Inside the loop, the same model writes kernels and nothing else. What makes the loop's output worth reading is the separation:

- The session that wrote the harness is not the one that writes the kernels.
- The loop's model has no tools, no file access and no context beyond a generic system prompt, the op definition and the harness reports on its earlier versions. It never sees the verifier or benchmark code.
- Model-written code is only imported in a child process, and the verdict is computed in yet another process, from saved outputs.
- The harness files are read-only during a run and SHA-256 fingerprinted before and after each evaluation; the loop stops if one changes.

What this does not remove: the same model wrote the harness and the kernels, so a blind spot it has could sit on both sides of the test. And no one outside the project has reviewed the verifier yet.

## The setup

Each op starts as a Python definition file, written outside the loop: a NumPy float64 reference (the ground truth), an input generator, benchmark shapes, fuzz and accuracy cases. There is no starting kernel.

The loop's model (Claude Opus 5.5) is called headless, with an empty working directory, and returns one Python module containing at least one custom Metal kernel (`mx.fast.metal_kernel`). It never sees the earlier manual kernels. No one edits the code it writes.

Each candidate then goes through:

1. **A static gate**: an AST whitelist checked before import. A few allowed imports, no file or OS access, no reflection.
2. **A three-layer verdict**: random tests against the platform's standard op; fuzzing of shapes and values against the float64 reference; numerical error against float64, compared with that of the standard op.
3. **An interleaved benchmark** against every reference available: native MLX and compositions of MLX functions, PyTorch MPS, and llama.cpp Metal where it implements the op. A shape's score is the speedup against the *strongest* reference for that shape.

One full run over five ops would have cost $16.46 at public API list prices (it ran on a subscription): 31 versions in 48.6 minutes.

## What the loop gains, and what it does not

Conditions: one MacBook with an Apple M5 (10-core GPU, 24 GB), macOS 26, MLX 0.32.4, fp16 inputs, each op in isolation, final benchmark of 30 interleaved rounds. Speedup against the strongest reference per shape, geometric mean over shapes, range in parentheses. "Other runs" are two earlier, independent runs of the loop, before llama.cpp was added to its internal benchmark; their final numbers are also against the strongest reference.

| Op | Speedup | Other runs |
|---|---|---|
| Shared prefix (N queries, one prefix) | 2.59× (2.15–3.38) | 2.55×, 2.45× |
| Speculative verification (k query rows over a long cache) | 2.34× (1.32–3.44) | 1.75×, 2.03× |
| DSA decode (DeepSeek sparse attention: indexer + top-k) | 2.31× (1.42–4.23) | 1.83×, 2.58× |
| Single-token decode | 1.02× (0.94–1.06) | |
| KDA (Kimi Delta Attention) | 0.75× (0.34–1.00) | |

The winning kernels use Metal 4 tensor operations and have only been run on an M5 with macOS 26. An earlier manual attempt with these operations had been slower at small tiles and was dropped; the model went back to them on its own, with larger tiles, and made them pay off.

Now the other half. **Single-token decode is at parity**: it is limited by memory bandwidth, nothing is left to win on this machine. **KDA is a loss**: behind llama.cpp's dedicated kernel, at parity with mlx-lm's. Part of the gap is method: llama.cpp is timed in its own benchmark, with a warm cache.

Variance between two runs of the same model is around ±30%; the "Other runs" column shows it. And in an earlier experiment, before this loop, kernels for these same ops reached 0.77 to 0.92× of FlashInfer on an RTX 2080 Ti. On well-served hardware, they do not beat the existing libraries. That is the reason to look elsewhere.

## The verifier is the actual product

When a model writes the kernels, the verifier is what stands between a fast number and a wrong one. Five wrong versions passed the standard test and were rejected by fuzzing or the numerical layer: 3 out of 59 in one stage, 2 out of 16 in later runs (the day-zero op below and KDA). Two examples:

- **A one-element buffer.** When an input dimension is one (a single position in DSA, a single head in KDA), MLX passes the array as a `constant` buffer, not a `device` one. A kernel that declares it `device` fails in that case only. The standard test, at the nominal shape, never reaches it; the fuzzer does. The trap came back in an independent session, and again in a KDA version that had passed both the standard test and the numerical layer.
- **A missing `template` keyword** in the shared-prefix kernel. The tile size depended on the shape, so compilation broke only for some shapes. It passed the standard test; fuzzing and the numerical layer caught it, and the model fixed it in the next version.

A loop that only checks the standard op at the nominal shape would have kept both.

The verifier was tested before being trusted. Five kernels written on purpose to cheat (for example, ignoring the sliding window, or truncating the context) were all rejected, and 3 of them passed the standard test. Four correct kernels from the earlier manual session were all accepted. No accepted kernel has been found wrong since, which only means the verifier has not found anything else. The cheaters came from the same model as the verifier, so this calibration is weaker than an outsider's would be.

The three winning kernels, the verifier and the benchmark are in a public sample package https://github.com/YohannSidot/verified-metal-kernels, tested once from a clean copy on the same M5 (10 rounds: 2.49x, 2.26x, 2.30x). The loop that generates kernels is not public.

## From per-op speedup to tokens per second

Two kernels were plugged into mlx-lm. Conditions: MacBook M5, MLX 0.32.4, mlx-lm 0.32.2, Qwen3-4B 4-bit computed in fp16 (the model ships in bf16), 10 interleaved rounds, greedy decoding, generated tokens identical to stock mlx-lm.

**Shared prefix holds up.** 1.27× (4 requests) and 1.31× (8 requests) in tokens/s against a cascade written with MLX functions; 1.66× and 2.00× against mlx-lm's per-request layout. Prefix of 8,192 tokens, suffixes of 64 tokens, 32 decode steps. Caveats: mlx-lm has no shared-prefix cache, so the one used here was written for this project, as was the cascade reference; all suffixes have the same length; there is no serving stack. The drop from 2.59× per op is expected in part: attention is 61 to 63% of a decode step in this setting, measured by removing it, and the reference here is not the same as in the per-op table.

**Speculative verification accelerates a mode that is still slower than plain decoding.** Against the strongest speculative configuration, the loop's kernel gives 1.11× to 1.28× in tokens/s (1.14× to 1.32× against mlx-lm as shipped), at 8k and 32k context, with 4 and 8 draft tokens and a Qwen3-0.6B draft. But plain decoding without a draft is faster in every configuration measured: this arm runs at 0.43 to 0.83× of plain decoding. On a second task (8k, 8 draft tokens) the gain falls to 1.08×. One model, one draft, batch size 1. The loop made a slow mode less slow; that is not a reason to use it. Attention is 42 to 81% of a step here, depending on context.

One defect found on the way: the kernels only read fp16. On bf16, the format mlx-community models compute in, they produce invalid values.

## What did not work

**A day-zero op.** The next target was the attention step of NVIDIA Dory 1.0 4B, released on 2026-10-16, for which the search (also run in a Claude session) found no Metal kernel. The first verified kernel came 10.5 minutes after launch; 7 of 8 versions were correct. But the best is at 0.83× the strongest reference (0.43–1.15): 0.96× against MLX and PyTorch alone, 1.22× against the path of the mlx-lm pull request. It got to correct quickly; it did not get to fast.

**An MLX issue I could not reproduce.** An MLX issue reports slow `gather_qmm` at a single token, on the author's M3 Ultra. On the base M5, chained over 42 layers, `gather_qmm` reads at 127 GB/s against 128 GB/s for a plain read (one machine, no 30-round benchmark). No headroom, so no loop run. This issue, at least, seems to need a Max or Ultra chip, which I do not have.

## Limits

- All code was written by Claude models in three days; the same model wrote harness and kernels; no outsider has reviewed the verifier.
- One machine, fp16. The per-op numbers are for isolated ops. The only tokens/s numbers are the two above, in mlx-lm, on one model; none in a serving stack.
- About ±30% between two runs of the same model.
- Some of the strongest references (`mlx_sdpa_fold`, `mlx_smart`, `mlx_cascade`) are compositions of MLX functions written for this project, not library entry points. I chose the ops and the references.
- The winning kernels use Metal 4 tensor operations: M5 and macOS 26 only, untested anywhere else.
- llama.cpp is timed in its own benchmark, with a warm cache.
- "No Metal kernel found" does not mean none exists.
- The kernels read fp16 only; there is no bf16 variant yet.
- **Nothing has run on AMD hardware.** Nothing in this post says anything about AMD.

## What I am looking for

Ops that are slow or missing on hardware other than Nvidia, in a model you actually run: the op, the shapes, the model, and what you use today. A reproducible case is the most useful thing you can send. I can only measure on the M5 today, and would rather hear that an op is already fast than spend a run finding out. And if you see a class of wrong kernel the three layers above would let through, I want to know.

The simplest way to reach me is an issue on the sample repository.
