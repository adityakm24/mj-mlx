"""Miss rate of the low-bit lm_head candidate set: how often the full bf16 head's argmax (computed in float32 from
the same final hidden state) is NOT among the low-bit head's top-k. A miss is the only way the candidate+rescore
path can change a greedy token. Teacher-forced over random-token prompts.

    .venv/bin/python tools/mlx_lm_head_miss.py --model <dir> --bits 8 --k 16 --out results/lm_head_miss.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mj_mlx import GptOss  # noqa: E402
from mj_mlx.gpt_oss import build_meta  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--bits", type=int, nargs="+", default=[8, 4])
    ap.add_argument("--k", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--len", type=int, default=512)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    m = GptOss(a.model, dtype=mx.bfloat16, lm_head_bits=None)
    rs = np.random.RandomState(0)
    xs = []
    for _ in range(a.prompts):
        ids = rs.randint(0, 199998, a.len)
        T = len(ids)
        pool = m.new_kv_pool(((T + 15) // 16) * 16)
        meta = build_meta(ids, np.arange(T), np.array([0]), np.array([T]), np.arange((T + 15) // 16)[None],
                          np.arange(T), 16, m.cfg.sliding_window)
        h = m.hidden(meta, pool)
        x32 = mx.fast.rms_norm(h.astype(mx.float32), m.norm.astype(mx.float32), m.cfg.rms_norm_eps)
        mx.eval(x32)
        xs.append(x32)
    W = m.lm_head
    res = {}
    for bits in a.bits:
        wq = mx.quantize(W, 64, bits)
        miss = {k: 0 for k in a.k}
        n = 0
        for x32 in xs:
            for i in range(0, x32.shape[0], 128):
                xc = x32[i:i + 128]
                # exact argmax: bf16 weights, float32 math, in vocab chunks to bound memory
                best_v = mx.full((xc.shape[0],), -mx.inf)
                best_i = mx.zeros((xc.shape[0],), mx.int32)
                for v0 in range(0, W.shape[0], 16384):
                    lg = xc @ W[v0:v0 + 16384].astype(mx.float32).T
                    vi = mx.argmax(lg, axis=-1)
                    vv = mx.take_along_axis(lg, vi[:, None], axis=-1)[:, 0]
                    upd = vv > best_v
                    best_i = mx.where(upd, vi.astype(mx.int32) + v0, best_i)
                    best_v = mx.where(upd, vv, best_v)
                lq = mx.quantized_matmul(xc.astype(mx.bfloat16), *wq, transpose=True, group_size=64, bits=bits)
                order = mx.argsort(-lq.astype(mx.float32), axis=-1)[:, : max(a.k)]
                for k in a.k:
                    hit = (order[:, :k] == best_i[:, None]).any(axis=-1)
                    miss[k] += int((~hit).sum().item())
                n += xc.shape[0]
        res[f"{bits}-bit"] = {f"top{k}_miss_rate": miss[k] / n for k in a.k} | {"rows": n}
        print(bits, res[f"{bits}-bit"], flush=True)
    out = dict(model=a.model, prompts=a.prompts, prompt_len=a.len, results=res,
               note="miss = exact (bf16 weights, fp32 math) argmax not in the low-bit head's top-k; random-token "
                    "prompts on random weights, teacher-forced")
    json.dump(out, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
