"""Interleaved A/B benchmark of the kernels against every available reference (standalone; Apple Silicon + MLX).

    python bench.py                          # the 3 shipped kernels, 30 rounds (about 10-15 minutes on an M5)
    python bench.py --rounds 10              # quicker, noisier
    python bench.py --ops dsa --kernel my_dsa.py
    python bench.py --llama /path/to/test-backend-ops   # add llama.cpp Metal (see llama_cpp/README.md)

References per op (all defined in ops/<op>.py):
  prefix: mlx_naive, mlx_unified, mlx_cascade (MLX); torch_naive, torch_unified (PyTorch MPS); llama.cpp (optional)
  spec:   mlx_sdpa, mlx_sdpa_fold, mlx_ops (MLX); torch_sdpa, torch_sdpa_fold (PyTorch MPS); llama.cpp (optional)
  dsa:    mlx_lm, mlx_smart (MLX); torch_ref (PyTorch MPS); llama.cpp (optional, sum of its 4 ggml ops)
mlx_cascade, mlx_sdpa_fold, mlx_ops and mlx_smart are compositions of MLX functions written for this project, not
library entry points.

Method (same as the project's final benchmark):
  - each timed call is a chain of >= 16 "layers" on distinct input buffers, each layer's query being the previous
    layer's output (a real data dependency), after a 192 MB write that flushes the GPU caches;
  - all contestants are timed in rounds, in a shuffled order each round (A/B interleaving), median over rounds;
  - the kernel's output is checked against the standard op on the benchmark inputs before timing, and the inputs are
    checked for modification;
  - speedup of a shape = median over rounds of (reference time / kernel time) against the reference that is fastest
    at that shape, i.e. the STRONGEST reference; llama.cpp, when present, runs in its own process in 3 slots that
    alternate with ours, and counts if it is stronger;
  - the op's number is the geometric mean of the per-shape speedups.
Absolute times depend on the machine and its thermal state; only the interleaved ratios are meant to be compared."""
import argparse, importlib, importlib.util, json, math, os, random, re, statistics, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np
import mlx.core as mx

from verify import KEYS, cand_inputs, gate, import_kernel, load_op, np32, as_tuple

DEFAULT_KERNELS = {"prefix": "kernels/prefix_opus55_v8.py", "spec": "kernels/spec_opus55_v7.py",
                   "dsa": "kernels/dsa_opus55_v7.py"}
LLAMA = {"prefix": dict(perf="prefix", n_sub=1), "spec": dict(perf="spec", n_sub=1), "dsa": dict(perf="dsa", n_sub=4)}
# Published numbers (Apple M5 base, 30 rounds, strongest reference INCLUDING llama.cpp): geomean (min-max)
PUBLISHED = {"prefix": "2.59x (2.15-3.38)", "spec": "2.34x (1.32-3.44)", "dsa": "2.31x (1.42-4.23)"}


# ------------------------------------------------------------------------------------------------ timing
_flushbuf = None


def flush_slc(mb=192):
    global _flushbuf
    _flushbuf = mx.full((mb * 2**18,), 1.0, mx.float32) * 1.0001
    mx.eval(_flushbuf)


def warm_gpu(seconds=1.0):
    a = mx.random.normal((2048, 2048)).astype(mx.float16)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        mx.eval(a @ a)


def mlx_chain_runner(fn, layers, q_key="q", chain=None):
    chain = chain or (lambda out, kw: out)

    def run():
        flush_slc()
        q = layers[0][q_key]
        for kw in layers:  # lazy graph construction (MLX executes nothing before eval)
            kw = dict(kw)
            kw[q_key] = q
            out = fn(**kw)
            q = chain(out, kw)
        t0 = time.perf_counter_ns()
        mx.eval(q)  # encode + GPU execution + completion wait
        return (time.perf_counter_ns() - t0) * 1e-9 / len(layers)

    return run


def torch_chain_runner(fn, layers, q_key="q", chain=None):
    import torch
    chain = chain or (lambda out, kw: out)

    def run():
        flush_slc()
        torch.mps.synchronize()
        t0 = time.perf_counter_ns()
        q = layers[0][q_key]
        for kw in layers:
            kw = dict(kw)
            kw[q_key] = q
            out = fn(**kw)
            q = chain(out, kw)
        torch.mps.synchronize()
        return (time.perf_counter_ns() - t0) * 1e-9 / len(layers)

    return run


def interleaved(runners, rounds, warmup=3, seed=0):
    rng = random.Random(seed)
    names = list(runners)
    for n in names:
        for _ in range(warmup):
            runners[n]()
    times = {n: [] for n in names}
    for _ in range(rounds):
        order = names[:]
        rng.shuffle(order)
        for n in order:
            times[n].append(runners[n]())
    return {n: dict(samples_us=[x * 1e6 for x in times[n]]) for n in names}


def merge(parts):
    res = {}
    for n in parts[0]:
        smp = [x for p in parts for x in p[n]["samples_us"]]
        t = sorted(smp)
        res[n] = dict(median_us=statistics.median(t), p10_us=t[int(0.1 * len(t))], p90_us=t[int(0.9 * len(t)) - 1],
                      n=len(t), samples_us=smp)
    return res


def ratio_median(res, a, b):
    """median over rounds of time(b)/time(a): > 1 means a is faster than b."""
    return statistics.median([y / x for x, y in zip(res[a]["samples_us"], res[b]["samples_us"])])


def mx_to_torch(a):
    import torch
    if a is None:
        return None
    if a.dtype == mx.float16:
        return torch.from_numpy(np.array(a.astype(mx.float32))).half().to("mps")
    return torch.from_numpy(np.array(a)).to("mps")


def geomean(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


# ------------------------------------------------------------------------------------------------ llama.cpp
def llama_measure(binary, perf_key, case, n_sub, min_iter, timeout=600):
    """One slot: time of one op call in seconds (sum of the n_sub ggml sub-op medians), measured by the patched
    test-backend-ops in its own process (warm, repeating the op on the same tensors)."""
    env = dict(os.environ, KMAC_PERF=perf_key, KMAC_CASE=str(case), KMAC_MIN_ITER=str(min_iter))
    p = subprocess.run([binary, "perf", "-b", "MTL0"], env=env, capture_output=True, text=True, timeout=timeout)
    med = [float(x) for x in re.findall(r"KMAC n_iter=\d+ n_runs_per_iter=\d+ median_us=([0-9.]+)", p.stdout + p.stderr)]
    if len(med) != n_sub:
        raise RuntimeError(f"llama.cpp {perf_key}#{case}: expected {n_sub} timings, got {med}; "
                           f"tail: {(p.stdout + p.stderr)[-400:]}")
    return sum(med) * 1e-6


# ------------------------------------------------------------------------------------------------ one op
def bench_op(opname, kernel_path, rounds, llama_bin=None, use_torch=True):
    op = load_op(opname)
    errs = gate(open(kernel_path).read())
    if errs:
        raise SystemExit(f"{kernel_path}: static gate rejected the kernel: {errs}")
    fn = import_kernel(kernel_path)
    wrapped = lambda **kw: fn(**cand_inputs(opname, kw))
    chain_m, chain_t = getattr(op, "chain_mlx", None), getattr(op, "chain_torch", None)
    std = getattr(op, op.L1_REF)
    ll = LLAMA[opname] if llama_bin else None
    warm_gpu(1.0)
    out = dict(op=opname, kernel=os.path.relpath(kernel_path, HERE), rounds=rounds, llama_cpp=bool(ll),
               torch=use_torch, shapes=[])
    for si, shape in enumerate(op.BENCH_SHAPES):
        l0 = op.make_inputs(shape, seed=0)
        lbytes = sum(int(v.nbytes) for v in l0.values() if isinstance(v, mx.array))
        n_distinct = int(max(1, min(8, 1.5 * 2**30 // lbytes)))
        n_steps = max(16, n_distinct)
        distinct = [l0] + [op.make_inputs(shape, seed=i) for i in range(1, n_distinct)]
        for l in distinct:
            l.pop("lens", None)
        layers = [distinct[i % n_distinct] for i in range(n_steps)]
        row = dict(shape=shape, n_distinct=n_distinct, n_chain=n_steps)
        # output check on the benchmark inputs themselves, and input-mutation check
        snap = {k: np32(v) for k, v in l0.items() if isinstance(v, mx.array)}
        o = as_tuple(wrapped(**l0))
        mx.eval(*o)
        mutated = [k for k, v in snap.items() if not np.array_equal(np32(l0[k]), v, equal_nan=True)]
        r = as_tuple(std(**l0))
        mx.eval(*r)
        ok = not mutated and len(o) == len(r) and all(
            np.allclose(np32(a), np32(b), atol=1e-2, rtol=1e-2, equal_nan=True) for a, b in zip(o, r))
        row["output_ok"] = bool(ok)
        if not ok:
            raise SystemExit(f"{opname} {shape}: kernel output differs from the standard op on the benchmark inputs"
                             + (f" (inputs modified: {mutated})" if mutated else ""))
        cands = {"kernel": mlx_chain_runner(wrapped, layers, chain=chain_m)}
        for n, f in op.MLX_REFS.items():
            cands[n] = mlx_chain_runner(f, layers, chain=chain_m)
        if use_torch:
            import torch
            tl = [{k: mx_to_torch(v) if isinstance(v, mx.array) else v for k, v in l.items()} for l in distinct]
            tlayers = [tl[i % n_distinct] for i in range(n_steps)]
            for n, f in op.TORCH_REFS.items():
                try:
                    f(**tlayers[0])
                    torch.mps.synchronize()
                    cands[n] = torch_chain_runner(f, tlayers, chain=chain_t)
                except Exception as e:
                    row.setdefault("skipped", {})[n] = f"{type(e).__name__}: {str(e)[:100]}"
        segs = [rounds // 3 + (1 if k < rounds % 3 else 0) for k in range(3)] if rounds >= 3 else [rounds]
        parts, ll_s = [], []
        for k, seg in enumerate(segs):  # slot k: llama.cpp (own process), then `seg` shuffled rounds of the rest
            if ll:
                ll_s.append(llama_measure(llama_bin, ll["perf"], si, ll["n_sub"], 30 if rounds >= 30 else 10))
            parts.append(interleaved(cands, rounds=seg, warmup=3 if k == 0 else 1, seed=k))
        res = merge(parts)
        refs = [n for n in res if n != "kernel"]
        best = min(refs, key=lambda n: res[n]["median_us"])
        vs = {n: ratio_median(res, "kernel", n) for n in refs}
        row.update(median_us={n: round(res[n]["median_us"], 1) for n in res}, speedup_vs_each_ref=vs,
                   best_inproc_ref=best, speedup=vs[best], strongest_ref=best)
        if ll:
            r_ll = statistics.median([ll_s[k] * 1e6 / statistics.median(parts[k]["kernel"]["samples_us"])
                                      for k in range(len(segs))])
            row["median_us"]["llama_cpp"] = round(statistics.median(ll_s) * 1e6, 1)
            row["speedup_vs_each_ref"]["llama_cpp"] = r_ll
            if r_ll < row["speedup"]:
                row["speedup"], row["strongest_ref"] = r_ll, "llama_cpp"
        row["samples_us"] = {n: res[n]["samples_us"] for n in res}
        out["shapes"].append(row)
        print(f"  {opname:6s} {json.dumps(shape):32s} kernel {res['kernel']['median_us']:8.0f} us   "
              f"{row['speedup']:.2f}x vs {row['strongest_ref']} "
              f"({row['median_us'][row['strongest_ref']]:.0f} us)", flush=True)
        del distinct, layers, cands
        mx.clear_cache()
    out["geomean"] = geomean([r["speedup"] for r in out["shapes"]])
    out["min"] = min(r["speedup"] for r in out["shapes"])
    out["max"] = max(r["speedup"] for r in out["shapes"])
    return out


def main():
    ap = argparse.ArgumentParser(description="Interleaved A/B benchmark (see the module docstring).")
    ap.add_argument("--ops", default="prefix,spec,dsa")
    ap.add_argument("--kernel", help="kernel file (only with a single op in --ops; default: the shipped kernel)")
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--llama", default=os.environ.get("LLAMA_TEST_BACKEND_OPS"),
                    help="path to the patched llama.cpp test-backend-ops binary (or env LLAMA_TEST_BACKEND_OPS)")
    ap.add_argument("--no-torch", action="store_true")
    ap.add_argument("--out", default="results", help="output directory for bench_<op>.json")
    a = ap.parse_args()
    ops = [o for o in a.ops.split(",") if o]
    if a.kernel and len(ops) != 1:
        raise SystemExit("--kernel needs exactly one op in --ops")
    use_torch = not a.no_torch
    torch_note = "PyTorch MPS"
    if use_torch:
        try:
            import torch
            assert torch.backends.mps.is_available(), "MPS not available"
            torch_note = f"PyTorch {torch.__version__} MPS"
        except Exception as e:
            print(f"!! PyTorch MPS unavailable ({e}); benchmarking without it")
            use_torch = False
    if a.llama and not os.access(a.llama, os.X_OK):
        raise SystemExit(f"--llama: {a.llama} is not an executable file")
    refs = ["MLX " + mx.__version__] + ([torch_note] if use_torch else []) + (["llama.cpp Metal"] if a.llama else [])
    print("=" * 100)
    print(f"References in this run: {', '.join(refs)}.   Rounds: {a.rounds}.")
    if not a.llama:
        print("!! llama.cpp is NOT in this run (no --llama binary). Speedups below are against MLX"
              + (" and PyTorch MPS" if use_torch else "") + " only.")
        print("!! In the published numbers llama.cpp was the strongest reference at two shapes (prefix N=32 P=4096,")
        print("!! dsa B=4 L=16384). Without it these shapes can read higher than published; for dsa B=4 the gap is")
        print("!! large (about 6x against MLX instead of about 4x against llama.cpp), which lifts the dsa geometric mean.")
    if not use_torch:
        print("!! PyTorch MPS is NOT in this run.")
    print("=" * 100)
    os.makedirs(a.out, exist_ok=True)
    summary = {}
    for opname in ops:
        kpath = os.path.abspath(a.kernel or os.path.join(HERE, DEFAULT_KERNELS[opname]))
        print(f"{opname}: {os.path.relpath(kpath)}", flush=True)
        r = bench_op(opname, kpath, a.rounds, llama_bin=a.llama, use_torch=use_torch)
        r.update(mlx=mx.__version__, references=refs)
        json.dump(r, open(os.path.join(a.out, f"bench_{opname}.json"), "w"), indent=1, default=str)
        summary[opname] = r
    print("=" * 100)
    print(f"{'op':8s} {'this run: geomean (min-max) vs strongest reference':54s} published (with llama.cpp, 30 rounds)")
    for opname, r in summary.items():
        print(f"{opname:8s} {r['geomean']:.2f}x ({r['min']:.2f}-{r['max']:.2f}){'':37s} {PUBLISHED.get(opname, '-')}")
    print("Published numbers vary by about +-30% between two LOOP runs of the same model (different kernels);")
    print("this benchmark re-times one fixed kernel. Only interleaved ratios are comparable across machines and runs.")
    print(f"References: {', '.join(refs)}." + ("" if a.llama else "  (llama.cpp NOT included)"))


if __name__ == "__main__":
    main()
