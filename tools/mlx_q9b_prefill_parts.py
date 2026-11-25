"""Qwen3.5-9B prefill by part, ours vs mlx-lm, on each engine's own prefill (same prompt length, real weights).

Each wrapped call evaluates its inputs, runs, and evaluates its output (GPU-bound time of that part plus one host
sync), so the instrumented total is above the plain one; the same wrapping is applied to both engines. Parts:
full_attn (the 8 gated-attention layers), linear_attn (the 24 DeltaNet layers, total), of which proj (their
quantized matmuls) and gdn (the gated-delta recurrence kernel); mlp (all 32 MLPs). "other" = total - full_attn -
linear_attn - mlp (norms, residuals, embedding, head, syncs).

    .venv/bin/python tools/mlx_q9b_prefill_parts.py --engine mj --out results/q9b_prefill_parts/mj.json
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["mj", "mlx-lm"], required=True)
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import mlx.core as mx

    import mlx_bench_worker as W

    path = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-MLX-4bit/snapshots/*"))[0]
    wa = SimpleNamespace(stream_experts=None, map_experts=None, attn_bits=None, attn_pack=False, lean=False,
                         map_head=None, wired_mb=None, chunk=2048, slots=128)
    run, _ = (W.make_mj if a.engine == "mj" else W.make_mlxlm)(path, wa)
    mx.set_cache_limit(1 << 30)
    acc = collections.defaultdict(float)
    depth = {"v": 0}
    on = {"v": False}

    def arrays(xs):
        out = []
        for x in xs:
            if isinstance(x, mx.array):
                out.append(x)
            elif isinstance(x, (list, tuple)):
                out += arrays(x)
            elif isinstance(x, dict):
                out += arrays(list(x.values()))
        return out

    def wrap(name, fn):
        def w(*args, **kw):
            if not on["v"]:
                return fn(*args, **kw)
            mx.eval(arrays(args) + arrays(list(kw.values())))
            t0 = time.perf_counter()
            r = fn(*args, **kw)
            mx.eval(arrays([r]))
            acc[name] += time.perf_counter() - t0
            return r
        return w

    if a.engine == "mj":
        from mj_mlx import gated_delta as GD
        from mj_mlx import qwen3_5_moe as Q

        Q.Qwen35Moe._full_attention = wrap("full_attn", Q.Qwen35Moe._full_attention)
        Q.Qwen35Moe._linear_attention = wrap("linear_attn", Q.Qwen35Moe._linear_attention)
        Q.Qwen35Moe._moe = wrap("mlp", Q.Qwen35Moe._moe)
        orig_mm = Q.mm

        def mm(x, lin, _w=wrap("proj_all", orig_mm)):
            return _w(x, lin)
        Q.mm = mm
        GD_f = GD.gated_delta_pool_packed
        Q_gd = wrap("gdn", GD_f)
        GD.gated_delta_pool_packed = Q_gd
    else:
        import mlx.nn as nn
        from mlx_lm.models import qwen3_5 as L

        L.Attention.__call__ = wrap("full_attn", L.Attention.__call__)
        L.GatedDeltaNet.__call__ = wrap("linear_attn", L.GatedDeltaNet.__call__)
        L.MLP.__call__ = wrap("mlp", L.MLP.__call__)
        L.gated_delta_update = wrap("gdn", L.gated_delta_update)
        nn.QuantizedLinear.__call__ = wrap("proj_all", nn.QuantizedLinear.__call__)

    res = dict(args=vars(a), plain=[], parts=[])
    for r in range(a.reps + 1):
        t = {}
        on["v"] = False
        ids = W.ids_for(a.n, 70 + r)
        t0 = time.perf_counter()
        run(ids, 1, lambda: t.__setitem__("f", time.perf_counter()), lambda: None)
        plain = t["f"] - t0
        mx.clear_cache()
        time.sleep(2)
        acc.clear()
        on["v"] = True
        ids = W.ids_for(a.n, 170 + r)
        t0 = time.perf_counter()
        run(ids, 1, lambda: t.__setitem__("f", time.perf_counter()), lambda: None)
        inst = t["f"] - t0
        on["v"] = False
        mx.clear_cache()
        p = dict(acc)
        p["total_instrumented"] = inst
        p["other"] = inst - p.get("full_attn", 0) - p.get("linear_attn", 0) - p.get("mlp", 0)
        if r:
            res["plain"].append(plain)
            res["parts"].append(p)
        print(f"{a.engine} rep {r}{' (warm-up)' if not r else ''}: plain {plain:.3f} s ({a.n / plain:.1f} tok/s) | "
              f"instrumented {inst:.3f} s | " + " ".join(f"{k} {1e3 * v:.0f}" for k, v in sorted(p.items())) + " ms",
              flush=True)
        time.sleep(2)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
