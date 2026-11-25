"""Teacher-forced accuracy of bf16 configurations against a float32 reference, on one gpt-oss checkpoint.

Reference: our float32 forward. It is checked equal to transformers float32 (max |dlogit| ~6e-8) by
tests/test_gpt_oss.py on the tiny model; with --hf the transformers float32 forward is the reference instead.
Reported per configuration: top-1 agreement with the reference, mean KL(ref || cfg), max |dlogit|.

    .venv/bin/python tools/mlx_accuracy.py --model <dir> [--hf] --out results/accuracy.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests", "mlx"))


def ours_logits(model, ids, rescore=None):
    from mj_mlx.gpt_oss import build_meta

    T, bs = len(ids), 16
    pool = model.new_kv_pool(((T + bs - 1) // bs) * bs)
    meta = build_meta(np.array(ids), np.arange(T), np.array([0]), np.array([T]),
                      np.arange((T + bs - 1) // bs)[None], np.arange(T), bs, model.cfg.sliding_window)
    out = model.logits(meta, pool, rescore=rescore).astype(mx.float32)
    mx.eval(out)
    return np.array(out)


def mlx_lm_logits(model, ids):
    out = model(mx.array([ids], dtype=mx.int32))[0].astype(mx.float32)
    mx.eval(out)
    return np.array(out)


def stats(ref, got, chunk=128):
    """Chunked so the [positions, vocab] float64 temporaries stay small."""
    kls, top1, mx_d = [], 0, 0.0
    for i in range(0, ref.shape[0], chunk):
        r, g = ref[i:i + chunk].astype(np.float64), got[i:i + chunk].astype(np.float64)
        r -= r.max(-1, keepdims=True)
        g -= g.max(-1, keepdims=True)
        r -= np.log(np.exp(r).sum(-1, keepdims=True))
        g -= np.log(np.exp(g).sum(-1, keepdims=True))
        kls.append((np.exp(r) * (r - g)).sum(-1))
        top1 += int((ref[i:i + chunk].argmax(-1) == got[i:i + chunk].argmax(-1)).sum())
        mx_d = max(mx_d, float(np.abs(ref[i:i + chunk] - got[i:i + chunk]).max()))
    kl = np.concatenate(kls)
    return dict(top1=top1 / ref.shape[0], mean_kl=float(kl.mean()), max_kl=float(kl.max()),
                max_abs_dlogit=mx_d, positions=int(ref.shape[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--len", type=int, default=256)
    ap.add_argument("--hf", action="store_true", help="transformers float32 as the reference (small models)")
    ap.add_argument("--skip-mlx-lm", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rs = np.random.RandomState(0)
    prompts = [[int(t) for t in rs.randint(0, 199998, a.len)] for _ in range(a.prompts)]

    from mj_mlx import GptOss

    if a.hf:
        import torch
        from hf_ref import load_hf_gpt_oss

        hf = load_hf_gpt_oss(a.model)
        with torch.no_grad():
            ref = [hf(torch.tensor([p])).logits[0].numpy() for p in prompts]
        ref_name = "transformers float32"
        del hf
    else:
        m = GptOss(a.model, dtype=mx.float32)
        ref = [ours_logits(m, p) for p in prompts]
        ref_name = "millijoule float32"
        del m
        mx.clear_cache()
    ref = np.concatenate(ref)

    rows = {}
    for name, kw, rs_k in [("millijoule bf16 precise + fp32 top-8 rescore (default)", dict(precise=True), 8),
                           ("millijoule bf16 precise, no rescore", dict(precise=True), 0),
                           ("millijoule bf16 plain, no rescore", dict(precise=False), 0),
                           ("millijoule float32", dict(dtype=mx.float32), 0)]:
        if name == "millijoule float32" and not a.hf:
            continue
        m = GptOss(a.model, **{"dtype": mx.bfloat16, **kw})
        rows[name] = stats(ref, np.concatenate([ours_logits(m, p, rs_k) for p in prompts]))
        print(name, rows[name], flush=True)
        del m
        mx.clear_cache()
    if not a.skip_mlx_lm:
        import mlx_lm
        from pathlib import Path

        from mlx_lm.utils import load_model

        m, _ = load_model(Path(a.model))
        name = f"mlx-lm {mlx_lm.__version__} bf16 (default load)"
        rows[name] = stats(ref, np.concatenate([mlx_lm_logits(m, p) for p in prompts]))
        print(name, rows[name], flush=True)
        if a.hf:
            m.set_dtype(mx.float32)
            name = f"mlx-lm {mlx_lm.__version__} float32"
            rows[name] = stats(ref, np.concatenate([mlx_lm_logits(m, p) for p in prompts]))
            print(name, rows[name], flush=True)
    out = dict(model=a.model, reference=ref_name, prompts=a.prompts, prompt_len=a.len, rows=rows,
               note="random token prompts, teacher-forced logits at every position")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
