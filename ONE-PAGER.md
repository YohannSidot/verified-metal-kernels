# A calibrated judge for GPU kernels on Apple Silicon

Inference engines for Apple Silicon ship their own Metal kernels, and models can now write such kernels. Two
questions decide whether one can be trusted: **is it right** (bit for bit, when a batch row must match the same row
alone) and **is it really faster**. I direct Claude models that build a judge for both questions, and I publish what
it finds, including where the kernels lose and where the judge itself was wrong.

## What the judge is

Four layers, then a benchmark:

1. randomized tests against the task's standard implementation;
2. fuzzing against a float64 oracle: odd shapes, extreme magnitudes, NaN and Inf;
3. accuracy against that oracle;
4. where the operation calls for it, bit-for-bit invariance: a row computed in a batch against the same row alone;
5. an interleaved benchmark against references we chose or wrote, one 30-round run per published number.

It is calibrated on kernels that are wrong on purpose: **80 of them across 18 tasks. 47 pass the randomized tests of
layer 1, 11 fall only at the invariance layer, and the judge accepts none.** It also rejects a library function when
it does not hold the property a task asks for: MLX's `quantized_matmul` is rejected on 4 tasks, 3 of them because a
row does not get the same bits in a batch as alone, a property MLX does not promise.

What weakens it, stated plainly: the same model family wrote the judge, the wrong kernels and the candidates, and
nobody outside the project has reviewed it. The public `verify.py` in this repository has the first three layers,
not the invariance layer. One wrong kernel written by a model once passed three layers of an earlier version; the
benchmark's output check stopped it, the judge was fixed and now rejects it. The 175 loop candidates have not been
re-judged with the current judge, and the count of accepted kernels below still includes that one.

## What it has measured

- **196 verdicts** on kernels written by Claude models: 179 accepted, 17 rejected, 11 of those after passing the
  randomized tests of layer 1.
- **20 tasks**, 16 drawn from code paths or issues of 10 public projects (MLX, mlx-lm and other public inference
  engines); one of the 20, for AMD hardware, has never run. The best accepted kernel beats the strongest reference
  on 13 of them (2x or more on 4), ties on 3 and loses on 3. On 8 of those 13, at least one shape is a tie or a
  loss. For the tasks that require exact rows, "strongest reference" means the strongest exact one: against an
  engine's faster, non-exact path, these kernels lose.
- **Five measurements posted on other projects' public threads**, with their limits: see [`cases/`](cases/).
  One of them maps, by reading the code, which quantized products of an engine's 8-row verify do not do the
  arithmetic of its plain step, and what exact forms of those products cost at op level (our transcriptions,
  synthetic weights; not a throughput).

All of it comes from one machine, a base Apple M5 with macOS 26, each operation in isolation (no tokens per second),
on operations we defined from public code paths and issues or designed ourselves, with synthetic inputs and
references chosen and partly written by us.

## What you can ask for

- **An exactness map of your engine's speculative-decode verify path**: which operations do not reproduce the plain
  step, where in the code, and what exact forms cost. EUR 1,500 excl. VAT per path, invoiced only if you confirm in
  writing that at least one finding was new to you.
- **A kernel for one operation**, paid only if it is at least 1.2x faster than your current path on your own
  measurement; or correctness work (a row-exact variant of a kernel you have, with the check), on quote. For an
  open-source project, delivered under your licence.

Terms and details: [Commissioning a kernel](README.md#commissioning-a-kernel). To start, open an issue on this
repository.

The judge, the kernels and this page were produced by Claude (Anthropic's models) under the direction of Yohann
Sidot, who answers for them. None of the projects named in this repository is a customer or has endorsed this work.
