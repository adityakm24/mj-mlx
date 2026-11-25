# Copyright 2026 the MilliJoule authors
# SPDX-License-Identifier: Apache-2.0
"""Same-weights check for a MoE checkpoint that does not fit in memory: our model (experts streamed or mapped, as
given) against mlx-lm's own model code run one decoder layer at a time, teacher-forced on the same token sequences.

mlx-lm loads the checkpoint lazily (nothing read until evaluated); its layers are then evaluated one after another
for every sequence and dropped, so one layer's weights are in memory at a time. Reports, over every position: the
share where both pick the same top token, mean / p99 KL(mlx-lm || ours), and the largest log-probability difference
on mlx-lm's top 20.

    .venv/bin/python tools/mlx_layer_stream_ref.py --model <mlx dir> --mj-kw '{"stream_experts": ..., "expert_slots": 16}' \\
        --out results/x.json [--chat] [--random 3x64]
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import os
import sys

os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "150")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)


def ours_logprobs(path, kw, seqs, dtype=None):
    import mlx.core as mx
    import numpy as np

    from mj_mlx import load_model
    from mj_mlx.gpt_oss import build_meta

    m = load_model(path, **({"dtype": getattr(mx, dtype)} if dtype else {}), **kw)
    out = []
    for ids in seqs:
        T, bs = len(ids), 16
        pool = m.new_kv_pool(T + 64)
        meta = build_meta(np.array(ids), np.arange(T), np.array([0]), np.array([T]),
                          np.arange((T + bs - 1) // bs)[None], np.arange(T), bs, getattr(m.cfg, "sliding_window", 0))
        lg = m.logits(meta, pool).astype(mx.float32)
        out.append(np.array(lg - mx.logsumexp(lg, axis=-1, keepdims=True)))
    del m
    gc.collect()
    mx.clear_cache()
    return out


def mlxlm_logprobs(path, seqs):
    """mlx-lm's model, lazily loaded, run layer by layer over all sequences; each layer dropped after use."""
    from pathlib import Path

    import mlx.core as mx
    import numpy as np
    from mlx_lm.utils import load_model as lm_load

    model, _ = lm_load(Path(path), lazy=True)
    inner = model.model if hasattr(model, "model") else model.language_model.model
    hs = [inner.embed_tokens(mx.array(ids)[None]) for ids in seqs]
    mx.eval(hs)
    for i in range(len(inner.layers)):
        layer = inner.layers[i]
        hs = [layer(h, "causal" if h.shape[1] > 1 else None, cache=None) for h in hs]
        mx.eval(hs)
        inner.layers[i] = None
        del layer
        gc.collect()
        mx.clear_cache()
    head = model.lm_head if hasattr(model, "lm_head") else None
    out = []
    for h in hs:
        x = inner.norm(h)
        lg = (head(x) if head is not None else inner.embed_tokens.as_linear(x))[0].astype(mx.float32)
        out.append(np.array(lg - mx.logsumexp(lg, axis=-1, keepdims=True)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--mj-kw", default="{}")
    ap.add_argument("--chat", action="store_true", help="the 12 chat-check prompts through the chat template")
    ap.add_argument("--random", default="", help="NxT random-token sequences, e.g. 3x64")
    ap.add_argument("--dtype", default=None, help="our model's dtype (default: as load_model picks it, the scales' "
                    "dtype for quantized checkpoints, as mlx-lm); float32 for unquantized float32 checkpoints")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from swap_guard import guard_self

    guard_self()
    import numpy as np

    path = glob.glob(os.path.expanduser(a.model))[0]
    seqs = []
    if a.chat:
        from transformers import AutoTokenizer

        from mlx_chat_check import PROMPTS, chat_ids

        tok = AutoTokenizer.from_pretrained(path)
        seqs += [chat_ids(tok, p) for p in PROMPTS]
    if a.random:
        n, t = (int(x) for x in a.random.split("x"))
        V = json.load(open(os.path.join(path, "config.json")))["vocab_size"]
        rs = np.random.RandomState(0)
        seqs += [[int(x) for x in rs.randint(0, V, t)] for _ in range(n)]
    ours = ours_logprobs(path, json.loads(a.mj_kw), seqs, a.dtype)
    ref = mlxlm_logprobs(path, seqs)
    same, kls, dmax, rows = 0, [], [], []
    for i, (o, r) in enumerate(zip(ours, ref)):
        o, r = o.astype(np.float64), r.astype(np.float64)
        s = int((o.argmax(-1) == r.argmax(-1)).sum())
        kl = (np.exp(r) * (r - o)).sum(-1)
        top = np.argsort(-r, axis=-1)[:, :20]
        d = np.abs(np.take_along_axis(o, top, -1) - np.take_along_axis(r, top, -1)).max(-1)
        same += s
        kls.append(kl)
        dmax.append(d)
        rows.append(dict(seq=i, positions=len(o), same_top1=s, mean_kl=float(kl.mean()), max_dlogprob_top20=float(d.max())))
        print(f"seq {i}: {s}/{len(o)} same top token, mean KL {kl.mean():.4f}, max |dlogprob| top 20 {d.max():.3f}",
              flush=True)
    kl, d = np.concatenate(kls), np.concatenate(dmax)
    n = len(kl)
    res = dict(model=path, mj_kw=a.mj_kw, sequences=len(seqs), positions=n, same_top1=same / n, mean_kl=float(kl.mean()),
               p99_kl=float(np.quantile(kl, 0.99)), max_kl=float(kl.max()), median_max_dlogprob_top20=float(np.median(d)),
               rows=rows, notes=__doc__)
    print(json.dumps({k: v for k, v in res.items() if k not in ("rows", "notes")}), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
