"""Three-layer verifier for one kernel file (standalone; Apple Silicon + MLX).

    python verify.py kernels/prefix_opus55_v8.py            # op inferred from the file name (prefix_, spec_, dsa_)
    python verify.py wrong_kernels/spec_cheat_nomax.py --skip-gate
    python verify.py my_kernel.py --op dsa --json out.json

A kernel file is a Python module with a top-level `def kernel(...)` taking the op's inputs as keyword arguments
(see ops/<op>.py for the inputs and the float64 reference).

Steps:
  0. Static gate (AST whitelist, applied before the file is imported): imports limited to mlx.core, math, functools;
     no file/OS access, no reflection, no attribute stores, no mx.fast op other than metal_kernel, and at least one
     custom Metal kernel (mx.fast.metal_kernel).  --skip-gate disables it (needed for the wrong_kernels/, which use
     MLX operations only).
  1. The kernel is imported and run on every verification case in a CHILD process, which saves its raw outputs.
  2. The verdict is computed in THIS process, which never imports the kernel, by regenerating the same inputs:
     L1: 5 seeds at the nominal shape, allclose(atol=rtol=1e-2) against the platform's standard op (MLX);
     L2: fuzzing of shapes and values (odd/edge/huge shapes, extreme magnitudes, NaN/Inf, ragged lengths) against the
         float64 NumPy reference: NaN/Inf pattern equal to the reference rounded to fp16 (for NaN/Inf inputs: equal to
         the standard op's pattern) and normalised max error <= 1e-2;
     L3: error against float64 compared with the standard op: max and mean error <= 2 * max(standard op error, 2^-11).
  An exception on any case is a failure.  Verdict: CORRECT only if every case of every layer passes.

Exit code: 0 = CORRECT, 1 = REJECTED, 2 = usage error."""
import argparse, ast, importlib, importlib.util, json, os, re, subprocess, sys, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np
import mlx.core as mx

# Inputs each op's kernel receives (keyword arguments; optional ones absent from the inputs are passed as None).
KEYS = {"prefix": ["q", "kp", "vp", "ks", "vs"],
        "spec": ["q", "k", "v", "bits"],
        "dsa": ["q", "kv", "iq", "ik", "w", "lens"]}


def load_op(opname):
    return importlib.import_module(f"ops.{opname}")


def cand_inputs(opname, inp):
    return {k: inp.get(k) for k in KEYS[opname]}


def cases(op):
    """Verification cases, identical in the child (kernel outputs) and the parent (judgement)."""
    out = [("L1", i, op.L1_SHAPE, "randn", 100 + i) for i in range(5)]
    out += [("L2", i, s, d, 7 + i) for i, (s, d) in enumerate(op.FUZZ)]
    out += [("L3", i, s, d, 3) for i, (s, d) in enumerate(op.L3_CASES)]
    return out


# ------------------------------------------------------------------------------------------------ static gate
ALLOWED_IMPORTS = {"mlx.core", "math", "functools"}
BANNED_NAMES = {"open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars", "getattr", "setattr",
                "delattr", "input", "breakpoint", "help", "exit", "quit", "memoryview", "super", "classmethod",
                "staticmethod", "property", "type", "object", "dir", "id", "hash", "print"}
BANNED_ATTRS = {"save", "load", "savez", "savez_compressed", "save_safetensors", "save_gguf", "export_function",
                "import_function", "export_to_dot", "set_default_device", "set_default_stream", "set_memory_limit",
                "set_wired_limit", "set_cache_limit", "start_capture", "stop_capture", "distributed", "random",
                "scaled_dot_product_attention", "rms_norm", "layer_norm", "rope", "cuda_kernel"}
MAX_BYTES = 200_000


def gate(src):
    """List of violations (empty = accepted)."""
    errs = []
    if len(src.encode()) > MAX_BYTES:
        return [f"module larger than {MAX_BYTES} bytes"]
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [f"SyntaxError: {e}"]
    if not any(isinstance(n, ast.FunctionDef) and n.name == "kernel" for n in tree.body):
        errs.append("no top-level `def kernel(...)`")
    uses_metal = False
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name not in ALLOWED_IMPORTS:
                    errs.append(f"line {n.lineno}: import {a.name} not allowed")
        elif isinstance(n, ast.ImportFrom):
            if n.module not in ALLOWED_IMPORTS or n.level:
                errs.append(f"line {n.lineno}: from {n.module} import ... not allowed")
        elif isinstance(n, ast.Name):
            if n.id in BANNED_NAMES or n.id.startswith("__"):
                errs.append(f"line {n.lineno}: name `{n.id}` not allowed")
        elif isinstance(n, ast.Attribute):
            if isinstance(n.ctx, (ast.Store, ast.Del)):
                errs.append(f"line {n.lineno}: assigning/deleting attributes is not allowed")
            if n.attr.startswith("_"):
                errs.append(f"line {n.lineno}: attribute `.{n.attr}` not allowed")
            if n.attr in BANNED_ATTRS:
                errs.append(f"line {n.lineno}: `.{n.attr}` not allowed")
            if n.attr == "metal_kernel":
                uses_metal = True
            if isinstance(n.value, ast.Attribute) and n.value.attr == "fast" and n.attr != "metal_kernel":
                errs.append(f"line {n.lineno}: mx.fast.{n.attr} not allowed (only mx.fast.metal_kernel)")
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            errs.append(f"line {n.lineno}: global/nonlocal not allowed")
        elif isinstance(n, (ast.AsyncFunctionDef, ast.Await, ast.Yield, ast.YieldFrom)):
            errs.append(f"line {n.lineno}: async/generators not allowed")
    if not uses_metal:
        errs.append("no mx.fast.metal_kernel found: the hot path must use at least one custom Metal kernel")
    return errs


def import_kernel(path):
    s = importlib.util.spec_from_file_location("candidate_kernel", path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m.kernel


# ------------------------------------------------------------------------------------------------ child side
def np32(a):
    if a.dtype == mx.bfloat16:
        a = a.astype(mx.float32)
    return np.array(a)


def as_tuple(o):
    return tuple(o) if isinstance(o, (tuple, list)) else (o,)


def child_outputs(opname, kernel_path, out):
    """The only function that imports the kernel.  Saves raw outputs (.npz) and per-case status (.json)."""
    fn = import_kernel(kernel_path)
    op = load_op(opname)
    arrays, meta = {}, {}
    for layer, i, shape, dist, seed in cases(op):
        key = f"{layer}_{i}"
        inp = cand_inputs(opname, op.make_inputs(shape, seed=seed, dist=dist))
        t0 = time.time()
        try:
            o = as_tuple(fn(**inp))
            mx.eval(*o)
            for j, a in enumerate(o):
                arrays[f"{key}__{j}"] = np32(a)
            meta[key] = dict(ok=True, n_out=len(o), dtypes=[str(a.dtype) for a in o], s=time.time() - t0)
        except Exception as e:
            meta[key] = dict(ok=False, error=f"{type(e).__name__}: {str(e)[:1500]}")
        mx.clear_cache()
    np.savez(out + ".npz", **arrays)
    json.dump(meta, open(out + ".json", "w"))


# ------------------------------------------------------------------------------------------------ parent side
def to64(a):
    if isinstance(a, mx.array):
        if a.dtype == mx.bfloat16:
            a = a.astype(mx.float32)
        a = np.array(a)
    return np.asarray(a).astype(np.float64)


def nerr(o, r):
    fin = np.isfinite(r) & np.isfinite(o)
    if not fin.any():
        return 0.0, 0.0
    scale = max(1.0, float(np.abs(r[fin]).max()))
    d = np.abs(o[fin] - r[fin]) / scale
    return float(d.max()), float(d.mean())


def pattern(a, half=True):
    a16 = a.astype(np.float16 if half else np.float32)
    return np.isnan(a16), np.isinf(a16)


def judge(opname, outpath):
    op = load_op(opname)
    meta = json.load(open(outpath + ".json"))
    arrs = np.load(outpath + ".npz")
    std1, std3 = getattr(op, op.L1_REF), getattr(op, op.L3_REF)
    rep = dict(L1=[], L2=[], L3=[])
    for layer, i, shape, dist, seed in cases(op):
        key = f"{layer}_{i}"
        case = dict(case=key, shape=shape, dist=dist)
        m = meta.get(key, dict(ok=False, error="missing (child crashed or timed out before this case)"))
        if not m["ok"]:
            case.update(ok=False, error=m["error"])
            rep[layer].append(case)
            continue
        inp = op.make_inputs(shape, seed=seed, dist=dist)
        outs = [arrs[f"{key}__{j}"].astype(np.float64) for j in range(m["n_out"])]
        if layer == "L1":
            s = [to64(x) for x in as_tuple(std1(**inp))]
            if len(s) != len(outs) or any(a.shape != b.shape for a, b in zip(outs, s)):
                case.update(ok=False, error=f"output shapes {[a.shape for a in outs]} != expected {[b.shape for b in s]}")
            else:
                ok = all(np.allclose(a, b, atol=1e-2, rtol=1e-2, equal_nan=True) for a, b in zip(outs, s))
                case.update(ok=bool(ok), maxdiff=max(float(np.nanmax(np.abs(a - b))) if a.size else 0.0
                                                     for a, b in zip(outs, s)))
            rep[layer].append(case)
            continue
        r = [to64(x) for x in as_tuple(op.ref64(**inp))]
        if len(r) != len(outs) or any(a.shape != b.shape for a, b in zip(outs, r)):
            case.update(ok=False, error=f"output shapes {[a.shape for a in outs]} != expected {[b.shape for b in r]}")
            rep[layer].append(case)
            continue
        if layer == "L2":
            pat_ok, emax, emean, s = True, 0.0, 0.0, None
            for j, (o, rr) in enumerate(zip(outs, r)):
                half = "float32" not in m["dtypes"][j]
                pn, pi = pattern(o, half)
                rn, ri = pattern(rr, half)
                pj = bool((pn == rn).all() and (pi == ri).all())
                if not pj and dist == "nan_inf":
                    s = s or [to64(x) for x in as_tuple(std3(**inp))]
                    sn, si = pattern(s[j], half)
                    pj = bool((pn == sn).all() and (pi == si).all())
                    case["pattern_vs"] = "std_op"
                if not pj:
                    case.setdefault("pattern_detail", []).append(
                        dict(output=j, nan_got=int(pn.sum()), nan_exp=int(rn.sum()), inf_got=int(pi.sum()),
                             inf_exp=int(ri.sum())))
                em, ea = nerr(o, rr)
                pat_ok, emax, emean = pat_ok and pj, max(emax, em), max(emean, ea)
            case.update(ok=bool(pat_ok and emax <= 1e-2), pattern_ok=pat_ok, err_max=emax, err_mean=emean)
        else:
            s = [to64(x) for x in as_tuple(std3(**inp))]
            ok, worst = True, {}
            for j, (o, rr, ss) in enumerate(zip(outs, r, s)):
                em, ea = nerr(o, rr)
                sm, sa = nerr(ss, rr)
                tol = 2.0 ** -11
                okj = em <= 2 * max(sm, tol) and ea <= 2 * max(sa, tol)
                ok = ok and okj
                worst[f"out{j}"] = dict(err_max=em, err_mean=ea, std_err_max=sm, std_err_mean=sa)
            case.update(ok=bool(ok), errors=worst)
        rep[layer].append(case)
        mx.clear_cache()
    for layer in ("L1", "L2", "L3"):
        rep[f"{layer}_pass"] = sum(c["ok"] for c in rep[layer])
        rep[f"{layer}_n"] = len(rep[layer])
    rep["verdict"] = "CORRECT" if all(rep[f"{l}_pass"] == rep[f"{l}_n"] for l in ("L1", "L2", "L3")) else "REJECTED"
    return rep


def summary(rep, max_fail=6):
    lines = [f"L1 (standard op, nominal shape)  {rep['L1_pass']}/{rep['L1_n']}",
             f"L2 (fuzzing vs float64)          {rep['L2_pass']}/{rep['L2_n']}",
             f"L3 (numerical error vs float64)  {rep['L3_pass']}/{rep['L3_n']}",
             f"=> {rep['verdict']}"]
    fails = [c for l in ("L1", "L2", "L3") for c in rep[l] if not c["ok"]]
    for c in fails[:max_fail]:
        d = {k: v for k, v in c.items() if k != "ok"}
        if "error" in d:
            d["error"] = d["error"][:600]
        lines.append("  FAIL " + json.dumps(d, default=str))
    if len(fails) > max_fail:
        lines.append(f"  ... and {len(fails) - max_fail} more failing cases")
    return "\n".join(lines)


def infer_op(path):
    base = os.path.basename(path)
    for op in KEYS:
        if base.startswith(op + "_"):
            return op
    return None


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--child":   # internal: python verify.py --child OP KERNEL OUT
        child_outputs(sys.argv[2], sys.argv[3], sys.argv[4])
        return 0
    ap = argparse.ArgumentParser(description="Three-layer verifier (see the module docstring).")
    ap.add_argument("kernel", help="path to the kernel file (a module with `def kernel(...)`)")
    ap.add_argument("--op", choices=sorted(KEYS), help="op name (default: inferred from the file name)")
    ap.add_argument("--skip-gate", action="store_true", help="do not apply the static gate (wrong_kernels/ only)")
    ap.add_argument("--json", help="write the full report to this file")
    ap.add_argument("--timeout", type=int, default=1800, help="child process timeout in seconds")
    a = ap.parse_args()
    opname = a.op or infer_op(a.kernel)
    if opname is None:
        print("cannot infer the op from the file name; pass --op", file=sys.stderr)
        return 2
    kpath = os.path.abspath(a.kernel)
    print(f"kernel: {a.kernel}   op: {opname}   MLX {mx.__version__}   device: {mx.default_device()}")
    if a.skip_gate:
        print("static gate: SKIPPED (--skip-gate)")
    else:
        errs = gate(open(kpath).read())
        if errs:
            print("static gate: REJECTED\n  " + "\n  ".join(errs))
            print("=> REJECTED (static gate; the kernel was not imported)")
            return 1
        print("static gate: passed")
    t0 = time.time()
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "outputs")
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "--child", opname, kpath, out],
                           timeout=a.timeout)
        if p.returncode != 0 or not os.path.exists(out + ".json"):
            print(f"child process failed (exit code {p.returncode}) before saving its outputs\n=> REJECTED")
            return 1
        rep = judge(opname, out)
    rep.update(kernel=a.kernel, op=opname, mlx=mx.__version__, seconds=round(time.time() - t0, 1))
    print(summary(rep))
    print(f"({rep['seconds']} s)")
    if a.json:
        json.dump(rep, open(a.json, "w"), indent=1, default=str)
    return 0 if rep["verdict"] == "CORRECT" else 1


if __name__ == "__main__":
    sys.exit(main())
