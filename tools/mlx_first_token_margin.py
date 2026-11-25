"""Is a difference between two engines' greedy replies a near-tie? For each chat prompt of tools/mlx_chat_check.py,
the top-5 next-token log-probabilities after the chat template, on our engine and on mlx-lm (same weights), and the
margin between the top two. A first-token difference is a rounding effect when the margin is about the size of the
two engines' logit disagreement; it is a bug when the margin is large.

    .venv/bin/python tools/mlx_first_token_margin.py --model <mlx dir> --out results/chat_check_v1/x_margins.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))


def top(lp, tok, k=5):
    import numpy as np

    lp = np.asarray(lp, dtype=np.float64)
    idx = np.argsort(-lp)[:k]
    return [dict(id=int(i), text=tok.decode([int(i)]), logprob=float(lp[i])) for i in idx]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--mj-kw", default="{}")
    ap.add_argument("--placement", default="auto")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from swap_guard import guard_self

    guard_self()
    import mlx.core as mx
    import numpy as np
    from transformers import AutoTokenizer

    from mlx_chat_check import PROMPTS, chat_ids

    path = glob.glob(os.path.expanduser(a.model))[0]
    tok = AutoTokenizer.from_pretrained(path)
    prompts = [chat_ids(tok, p) for p in PROMPTS]
    res = dict(model=path, prompts=PROMPTS, mj={}, mlx_lm={})

    from mj_mlx import load_model
    from mj_mlx.gpt_oss import build_meta

    kw = json.loads(a.mj_kw)
    if a.placement != "none":
        kw["placement"] = a.placement
    model = load_model(path, **kw)
    mj_lp = []
    for ids in prompts:
        T = len(ids)
        pool = model.new_kv_pool(T + 64)
        meta = build_meta(np.array(ids), np.arange(T), np.array([0]), np.array([T]),
                          np.arange((T + 15) // 16)[None], np.array([T - 1]), 16, model.cfg.sliding_window)
        lg = mx.array(model.full_logits(model.final_hidden(meta, pool)))[0].astype(mx.float32)
        mj_lp.append(np.array(lg - mx.logsumexp(lg)))
    del model
    mx.clear_cache()

    from pathlib import Path

    from mlx_lm.utils import load_model as lm_load

    lm, _ = lm_load(Path(path))
    lm_lp = []
    for ids in prompts:
        lg = lm(mx.array(ids)[None])[0, -1].astype(mx.float32)
        lm_lp.append(np.array(lg - mx.logsumexp(lg)))

    rows = []
    for i, (p, q) in enumerate(zip(mj_lp, lm_lp)):
        tp, tq = top(p, tok), top(q, tok)
        rows.append(dict(prompt=i, mj_top=tp, mlxlm_top=tq, same_top1=tp[0]["id"] == tq[0]["id"],
                         mj_margin=tp[0]["logprob"] - tp[1]["logprob"], mlxlm_margin=tq[0]["logprob"] - tq[1]["logprob"],
                         max_abs_logprob_diff_top20=float(np.max(np.abs(p - q)[np.argsort(-q)[:20]]))))
        print(f"prompt {i}: top1 {'same' if rows[-1]['same_top1'] else 'DIFFERS'}; margin mj "
              f"{rows[-1]['mj_margin']:.3f} mlx-lm {rows[-1]['mlxlm_margin']:.3f}; max |dlogprob| over mlx-lm top 20 "
              f"{rows[-1]['max_abs_logprob_diff_top20']:.3f}", flush=True)
    res["rows"] = rows
    res["notes"] = __doc__
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
