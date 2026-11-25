"""gpt-oss-20b decode step by part: each part run for all 24 layers as one token would (batch 1, decode shapes),
repeated, timed as GPU-bound throughput (lazy graph of N tokens' worth, one eval). With the bytes each part reads,
that gives achieved GB/s per part, to compare with the measured whole step (tools/mlx_bench_suite.py tg128).

Parts: attn_qkv / attn_o (the layer's projection path: bf16_pack.matvec, or 8-bit affine with --attn-bits 8),
router (addmm + top-4 + softmax), experts (4 routed MXFP4 experts: gate_up, swiglu, down, weighted sum), sdpa (paged
decode attention at --kv tokens, sliding and full layers as the model has them), head (4-bit candidates, top-16,
float32 rescore from the on-disk bf16 rows: greedy_from).

    .venv/bin/python tools/mlx_gptoss_decode_parts.py --out results/gptoss_decode_parts/r1.json
"""

import argparse
import glob
import json
import os
os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "150")
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def nbytes(a):
    if isinstance(a, dict):
        return sum(nbytes(v) for v in a.values())
    if isinstance(a, (list, tuple)):
        return sum(nbytes(v) for v in a)
    return getattr(a, "nbytes", 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attn-bits", type=int, default=None)
    ap.add_argument("--kv", type=int, default=128)
    ap.add_argument("--map-head", default=None)
    ap.add_argument("--tokens", type=int, default=16, help="tokens' worth of each part per timed eval")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import mlx.core as mx

    from mj_mlx import load_model
    from mj_mlx.paged_attention import paged_decode_attention

    mx.set_cache_limit(256 << 20)
    HF = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/*"))[0]
    MAP = os.path.expanduser("~/.cache/millijoule/gptoss20b-experts-mapped")
    kw = dict(attn_bits=a.attn_bits) if a.attn_bits else dict(attn_pack=True)
    if a.map_head:
        kw["map_head"] = os.path.expanduser(a.map_head)
    m = load_model(HF, map_experts=MAP, lean=True, **kw)
    mx.clear_cache()
    c = m.cfg
    H, nq, nkv, D = c.hidden_size, c.num_attention_heads, c.num_key_value_heads, c.head_dim
    rs = np.random.RandomState(0)
    xs = [mx.array(rs.randn(1, H).astype(np.float32) * 0.1).astype(m.dtype) for _ in range(a.tokens)]
    os_ = [mx.array(rs.randn(1, nq * D).astype(np.float32) * 0.1).astype(m.dtype) for _ in range(a.tokens)]
    x32 = [x.astype(mx.float32) for x in xs]
    # paged KV for the sdpa part
    bs = 16
    nblk = (a.kv + bs - 1) // bs + 1
    kp = mx.array(rs.randn(nblk * bs, nkv, D).astype(np.float32)).astype(m.dtype)
    vp = mx.array(rs.randn(nblk * bs, nkv, D).astype(np.float32)).astype(m.dtype)
    bt = mx.array(np.arange(nblk, dtype=np.int32)[None])
    kl = mx.array(np.array([a.kv], np.int32))
    qs = [mx.array(rs.randn(1, nq, D).astype(np.float32)).astype(m.dtype) for _ in range(a.tokens)]
    idxs = [[mx.array(rs.choice(c.num_local_experts, 4, replace=False)[None].astype(np.int32))
             for _ in range(len(m.layers))] for _ in range(a.tokens)]
    wts = mx.array(np.full((1, 4), 0.25, np.float32)).astype(m.dtype)
    mx.eval(xs, os_, x32, kp, vp, qs, idxs)

    def L(i):
        return m.layers[i]

    parts = {
        "attn_qkv": lambda t: [m._proj(L(i), "qkv", xs[t]) for i in range(len(m.layers))],
        "attn_o": lambda t: [m._proj(L(i), "o", os_[t]) for i in range(len(m.layers))],
        "router": lambda t: [mx.softmax(mx.take_along_axis(g, mx.argpartition(g, kth=-4, axis=-1)[..., -4:], axis=-1),
                                        axis=-1, precise=True)
                             for g in [mx.addmm(L(i)["router_b"], x32[t], L(i)["router_wt"])
                                       for i in range(len(m.layers))]],
        "experts": lambda t: [L(i)["experts"](xs[t], idxs[t][i], wts) for i in range(len(m.layers))],
        "sdpa": lambda t: [paged_decode_attention(qs[t], kp, vp, bt, kl, L(i)["sinks"], m.attn_scale, bs,
                                                  c.sliding_window if L(i)["sliding"] else 0, a.kv)
                           for i in range(len(m.layers))],
        "head": lambda t: [m.greedy_from(x32[t])],
    }
    # bytes read per token, per part (weights; KV for sdpa)
    def lay_bytes(name):
        tot = 0
        for i in range(len(m.layers)):
            Li = L(i)
            for k in (f"{name}_p", f"{name}_q", f"{name}_wt"):
                if k in Li:
                    tot += nbytes(Li[k])
        return tot
    e0 = L(0)["experts"]
    per_expert = (e0.gu_w.nbytes + e0.gu_s.nbytes + e0.dn_w.nbytes + e0.dn_s.nbytes) / e0.num_experts
    span = lambda i: min(a.kv, c.sliding_window) if L(i)["sliding"] else a.kv
    bytes_ = dict(
        attn_qkv=lay_bytes("qkv"), attn_o=lay_bytes("o"),
        router=sum(L(i)["router_wt"].nbytes for i in range(len(m.layers))),
        experts=4 * per_expert * len(m.layers),
        sdpa=sum(2 * span(i) * nkv * D * 2 for i in range(len(m.layers))),
        head=nbytes(m.lm_head_qs[4]) if hasattr(m, "lm_head_qs") and 4 in m.lm_head_qs else 0)
    res = {}
    for name, f in parts.items():
        mx.eval([f(t) for t in range(a.tokens)])  # warm-up
        ts = []
        for _ in range(a.reps):
            t0 = time.perf_counter()
            mx.eval([f(t) for t in range(a.tokens)])
            ts.append((time.perf_counter() - t0) / a.tokens * 1e3)
        ms = float(np.median(ts))
        res[name] = dict(ms_per_token=ms, ms_all=ts, bytes_per_token=bytes_[name],
                         gb_s=bytes_[name] / (ms * 1e-3) / 1e9 if bytes_[name] else None)
        print(f"{name:9s} {ms:6.2f} ms/token  {bytes_[name] / 1e6:7.1f} MB  "
              + (f"{res[name]['gb_s']:6.1f} GB/s" if res[name]["gb_s"] else ""), flush=True)
    tot = sum(r["ms_per_token"] for r in res.values())
    print(f"sum of parts {tot:.2f} ms/token ({1e3 / tot:.1f} tok/s if nothing else)")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(dict(args=vars(a), parts=res, sum_ms=tot, notes=__doc__), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
