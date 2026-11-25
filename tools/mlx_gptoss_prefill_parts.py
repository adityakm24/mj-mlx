"""gpt-oss-20b prefill by part, on the engine's own prefill chunks (the suite's config, real weights, real routing).

Each part's inputs are evaluated first, then the part alone is run and evaluated, so its time is GPU-bound time for
that part at the chunk's shapes (plus one host sync, ~0.1-0.2 ms). Parts per layer: attn_qkv / attn_o projections,
attn_core (rope, KV write, K/V gather, SDPA), router, experts; "other" = chunk time - parts (norms, residuals,
embedding, syncs). The un-instrumented chunk is timed too, so the instrumented total can be compared with it.

    MJ_MLOCK=1 .venv/bin/python tools/mlx_gptoss_prefill_parts.py --out results/gptoss_prefill_parts/a8.json --attn-bits 8
"""

import argparse
import collections
import glob
import json
import os
os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "150")
import sys
import time
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attn-bits", type=int, default=None)
    ap.add_argument("--n", type=int, default=2048, help="prompt tokens (chunks of --chunk)")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import mlx.core as mx

    import mlx_bench_worker as W
    from mj_mlx import gpt_oss as G

    path = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/*"))[0]
    wa = SimpleNamespace(stream_experts=None, map_experts="~/.cache/millijoule/gptoss20b-experts-mapped",
                         attn_bits=a.attn_bits, attn_pack=a.attn_bits is None, lean=True, map_head=None,
                         wired_mb=3000, chunk=a.chunk, slots=128)
    run, mx_ = W.make_mj(path, wa)
    mx.set_cache_limit(1 << 30)

    acc = collections.defaultdict(float)
    on = {"v": False}

    def arrays(xs):
        out = []
        for x in xs:
            if isinstance(x, mx.array):
                out.append(x)
            elif isinstance(x, (list, tuple)):
                out += arrays(x)
        return out

    def wrap(name_fn, fn):
        def w(*args, **kw):
            if not on["v"]:
                return fn(*args, **kw)
            mx.eval(arrays(args))
            t0 = time.perf_counter()
            r = fn(*args, **kw)
            mx.eval(arrays([r]) + arrays(args))  # args: in-place KV writes are part of the op
            acc[name_fn(args)] += time.perf_counter() - t0
            return r
        return w

    M = G.GptOss
    op, oa, om, oc = M._proj, M._attention, M._moe, G.Experts.__call__
    M._proj = wrap(lambda a_: "attn_" + a_[2], op)
    M._attention = wrap(lambda a_: "attention", oa)
    M._moe = wrap(lambda a_: "moe", om)
    G.Experts.__call__ = wrap(lambda a_: "experts", oc)

    res = dict(args=vars(a), plain=[], parts=[])
    seed = 50
    for r in range(a.reps + 1):  # first = warm-up
        ids = W.ids_for(a.n, seed + r)
        t = {}
        on["v"] = False
        t0 = time.perf_counter()
        run(ids, 1, lambda: t.__setitem__("f", time.perf_counter()), lambda: None)
        plain = t["f"] - t0
        mx.clear_cache()
        time.sleep(2)
        acc.clear()
        on["v"] = True
        ids = W.ids_for(a.n, seed + 100 + r)
        t0 = time.perf_counter()
        run(ids, 1, lambda: t.__setitem__("f", time.perf_counter()), lambda: None)
        inst = t["f"] - t0
        on["v"] = False
        mx.clear_cache()
        p = dict(acc)
        p["attn_core"] = p["attention"] - p.get("attn_qkv", 0) - p.get("attn_o", 0)
        p["router"] = p["moe"] - p["experts"]
        p["other"] = inst - p["attention"] - p["moe"]
        p["total_instrumented"] = inst
        if r:
            res["plain"].append(plain)
            res["parts"].append(p)
        print(f"rep {r}{' (warm-up)' if not r else ''}: plain {plain:.3f} s ({a.n / plain:.1f} tok/s) | instrumented "
              f"{inst:.3f} s | " + " ".join(f"{k} {1e3 * p[k]:.0f}" for k in
                                             ("attn_qkv", "attn_core", "attn_o", "router", "experts", "other")) +
              " ms", flush=True)
        time.sleep(2)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
