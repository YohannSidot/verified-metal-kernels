# Copyright 2026 Yohann Sidot. Contains no code from Kapitsa.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
# with the License. The full text is in LICENSE-APACHE-2.0.txt, shipped with this file, and at
# http://www.apache.org/licenses/LICENSE-2.0. Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# either express or implied.
# This file was written by AI models (Claude) under the author's direction.
"""Op-level estimate of one 8-row verify cycle of Qwen3.6-35B-A3B from run_m5.json: the sum, over the quantized
products of one verify pass, of the median time of each product times its count per pass (30 linear-attention
layers, 10 full-attention layers, 40 MoE blocks, 1 LM head; config.json of the pack). A ratio of sums of medians, not a
throughput: attention over the cache, the GDN state update, norms outside the products, top-k, sampling, the drafter
and every dispatch gap are not in it. The router is listed apart: its current verify kernel is not transcribed."""
import json
import sys

COUNTS = dict(gdn_in=30, o_proj=40, attn_qkv=10, moe=40, lm_head=1, router=40)  # o_proj: 30 GDN out + 10 attention o


def main(path, out):
    r = json.load(open(path))["bench"]
    rows = sorted({b["M"] for b in r["lm_head"]})
    res = {}
    for routing in ("uniform", "overlap"):
        for M in rows:
            cur = exact = calls = 0.0
            parts = {}
            for c, n in COUNTS.items():
                if c == "router":
                    continue
                bs = [b for b in r[c] if b["M"] == M and (c != "moe" or b.get("routing") == routing)]
                med = bs[0]["median_us"]
                parts[c] = dict(count=n, kernl_us=med["kernl"], r2_us=med["r2"], r1_calls_us=med["r1_calls"])
                cur += n * med["r2"]
                exact += n * med["kernl"]
                calls += n * med["r1_calls"]
            rb = [b for b in r["router"] if b["M"] == M][0]["median_us"]
            res[f"{routing}_M{M}"] = dict(
                routing=routing, M=M, parts=parts, sum_r2_ms=cur / 1e3, sum_kernl_ms=exact / 1e3,
                sum_r1_calls_ms=calls / 1e3, ratio_kernl_over_r2=exact / cur, ratio_r1_calls_over_kernl=calls / exact,
                router_kernl_ms=40 * rb["kernl"] / 1e3, router_r1_calls_ms=40 * rb["r1_calls"] / 1e3)
            print(f"{routing:8s} M={M}: current verify (R2) {cur / 1e3:6.2f} ms, exact (K) {exact / 1e3:6.2f} ms "
                  f"({exact / cur:.2f}x), M plain steps' products {calls / 1e3:6.2f} ms; router K {40 * rb['kernl'] / 1e3:.2f} ms")
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "run_m5.json", sys.argv[2] if len(sys.argv) > 2 else "cycle_m5.json")
