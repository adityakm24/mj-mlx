# Copyright 2026 the MilliJoule authors
# SPDX-License-Identifier: Apache-2.0
"""Is a wrong GSM8K answer token a near-tie? For each item of a chat-format GSM8K run (tools/mlx_lm_eval.py --chat),
take an engine's own reply up to and including "#### ", and give the top-5 next-token log-probabilities there, and
the margin between the top two. RESULTS 1zz: gpt-oss sometimes works the answer out and then writes "#### 1"; if the
right number is a close second, the ending is a greedy near-tie and which way it falls is rounding.

    .venv/bin/python tools/mlx_answer_margin.py --engine mj --model <mlx dir> --mj-kw JSON --run <eval json> --out x.json
    .venv/bin/python tools/mlx_answer_margin.py --engine llama-server --model <mlx dir> --run <eval json> --out x.json

--run names the eval whose replies are the prefixes (run with --full-text: by default replies are cut to their last
600 characters); every engine can be given the same run, so the prefix tokens are identical. llama-server is asked through /completion with token ids and n_probs (its log-probabilities before
sampling).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import urllib.request

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "150")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--engine", choices=["mj", "mlx-lm", "llama-server"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--mj-kw", default="{}")
    ap.add_argument("--run", required=True, help="eval json whose gsm8k replies give the prefixes")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--api-url", default="http://127.0.0.1:8092")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from swap_guard import guard_self

    guard_self()
    import numpy as np
    from transformers import AutoTokenizer

    from mlx_lm_eval import CHAT_GSM, chat_enc

    path = glob.glob(os.path.expanduser(a.model))[0]
    tok = AutoTokenizer.from_pretrained(path)
    run = json.load(open(a.run))
    items = {}
    for line in open(os.path.join(ROOT, "results", "accuracy", "gsm8k_subset.jsonl")):
        it = json.loads(line)
        items[it["idx"]] = it["q"]["question"]
    rows_in = run["tasks"]["gsm8k"]["rows"]

    prefixes = []
    for r in rows_in:
        k = r["text"].rfind("####")
        if k < 0 or len(r["text"]) == 600:  # no answer line, or a reply cut to 600 characters (run without --full-text)
            continue
        q = items[r["idx"]]
        ids = chat_enc(tok, CHAT_GSM.format(q=q), a.effort) + tok.encode(r["text"][:k + 4] + " ", add_special_tokens=False)
        prefixes.append((r, ids))
    print(f"{len(prefixes)} replies with '####'", flush=True)

    def top(lp, k=5):
        lp = np.asarray(lp, dtype=np.float64)
        idx = np.argsort(-lp)[:k]
        return [dict(id=int(i), text=tok.decode([int(i)]), logprob=float(lp[i])) for i in idx]

    if a.engine == "mj":
        import mlx.core as mx

        from mj_mlx import load_model
        from mj_mlx.gpt_oss import build_meta

        model = load_model(path, **json.loads(a.mj_kw))

        def next_top(ids):
            T = len(ids)
            pool = model.new_kv_pool(T + 64)
            meta = build_meta(np.array(ids), np.arange(T), np.array([0]), np.array([T]),
                              np.arange((T + 15) // 16)[None], np.array([T - 1]), 16, model.cfg.sliding_window)
            lg = mx.array(model.full_logits(model.final_hidden(meta, pool)))[0].astype(mx.float32)
            return top(np.array(lg - mx.logsumexp(lg)))
    elif a.engine == "mlx-lm":
        from pathlib import Path

        import mlx.core as mx
        from mlx_lm.utils import load_model as lm_load

        lm, _ = lm_load(Path(path))

        def next_top(ids):
            lg = lm(mx.array(ids)[None])[0, -1].astype(mx.float32)
            return top(np.array(lg - mx.logsumexp(lg)))
    else:
        def next_top(ids):
            body = json.dumps(dict(prompt=ids, n_predict=1, n_probs=5, temperature=0, cache_prompt=False)).encode()
            req = urllib.request.Request(a.api_url + "/completion", body, {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as f:
                d = json.load(f)
            cp = d["completion_probabilities"][0]["top_logprobs"]
            return [dict(id=int(t["id"]), text=t.get("token", ""), logprob=float(t["logprob"])) for t in cp]

    out = []
    for r, ids in prefixes:
        t = next_top(ids)
        gold = r["gold"]
        gold_s = str(int(gold)) if gold == int(gold) else str(gold)
        gold_rank = next((j for j, x in enumerate(t) if x["text"].strip() == gold_s), None)
        out.append(dict(idx=r["idx"], gold=gold, pred=r["pred"], ok=r["ok"], n_prefix=len(ids), top=t,
                        margin=t[0]["logprob"] - t[1]["logprob"], gold_rank_in_top5=gold_rank))
        print(f"idx {r['idx']}: top {t[0]['text']!r} ({t[0]['logprob']:.3f}), 2nd {t[1]['text']!r} "
              f"({t[1]['logprob']:.3f}), margin {out[-1]['margin']:.3f}; gold {gold_s} rank {gold_rank}", flush=True)
    json.dump(dict(engine=a.engine, model=path, run=a.run, mj_kw=a.mj_kw, rows=out, notes=__doc__),
              open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
