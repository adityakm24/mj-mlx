# Copyright 2026 the MilliJoule authors
# SPDX-License-Identifier: Apache-2.0
"""How unevenly does gpt-oss-20b use its experts (checklist 0.11b: pin only the most-used ones)? Runs chat prompts
through our engine (greedy), records every routed expert per layer (prefill and decode counted apart), and reports,
per layer and overall, the share of decode routings that the top 8 / 16 / 24 of the 32 experts take, ranked on a
first half of the prompts and measured on the second half (so the ranking is not fit to the data it is scored on).

    .venv/bin/python tools/mlx_expert_usage.py --mj-kw JSON --new 256 --out results/expert_usage_v1/gptoss.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "150")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)


class Recorder:
    """Wraps a layer's experts: records the routed ids of every call, delegates everything else."""

    def __init__(self, inner, sink):
        self.__dict__["_inner"], self.__dict__["_sink"] = inner, sink

    def __call__(self, x, idx, wts):
        import numpy as np

        self._sink.append(np.array(idx).reshape(-1, idx.shape[-1]))
        return self._inner(x, idx, wts)

    def __getattr__(self, k):
        return getattr(self._inner, k)

    def __setattr__(self, k, v):
        setattr(self._inner, k, v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="~/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/*")
    ap.add_argument("--mj-kw", required=True)
    ap.add_argument("--new", type=int, default=256)
    ap.add_argument("--gsm", type=int, default=12, help="GSM8K questions added to the chat-check prompts")
    ap.add_argument("--out", required=True)
    ap.add_argument("--write-rank", default=None, help="expert store dir: write expert_rank.json (all prompts, decode "
                    "routings) for MJ_MLOCK_TOP")
    a = ap.parse_args()
    from swap_guard import guard_self

    guard_self()
    import numpy as np
    from transformers import AutoTokenizer

    from mj_mlx import Engine, load_model
    from mlx_chat_check import PROMPTS, chat_ids

    path = glob.glob(os.path.expanduser(a.model))[0]
    tok = AutoTokenizer.from_pretrained(path)
    kw = json.loads(a.mj_kw)
    kw.pop("wired_mb", None)
    model = load_model(path, **kw)
    sinks = [[] for _ in model.layers]
    for L, s in zip(model.layers, sinks):
        L["experts"] = Recorder(L["experts"], s)
    E = model.cfg.num_local_experts if hasattr(model.cfg, "num_local_experts") else 32
    prompts = list(PROMPTS)
    for line in list(open(os.path.join(HERE, "..", "results", "accuracy", "gsm8k_subset.jsonl")))[:a.gsm]:
        prompts.append(json.loads(line)["q"]["question"])
    per_prompt = []  # [prompt][layer] -> (prefill counts [E], decode counts [E])
    for p in prompts:
        ids = chat_ids(tok, p)
        for s in sinks:
            s.clear()
        eng = Engine(model=model, num_blocks=(len(ids) + a.new) // 16 + 16, max_running=1, overlap=False)
        eng.add_request(ids, a.new)
        while eng.has_work():
            eng.step()
        rows = []
        for s in sinks:
            pre = np.zeros(E, np.int64)
            dec = np.zeros(E, np.int64)
            for r in s:
                np.add.at(pre if r.shape[0] > 1 else dec, r.reshape(-1), 1)  # one row = a decode step
            rows.append((pre, dec))
        per_prompt.append(rows)
        print(f"prompt {len(per_prompt)}/{len(prompts)}: {len(ids)} tokens in", flush=True)
    rank_on = per_prompt[0::2]  # every other prompt ranks, the rest are scored
    score_on = per_prompt[1::2]
    res = dict(model=path, mj_kw=a.mj_kw, new=a.new, prompts=len(prompts), experts=E, layers=[])
    tot = {k: [0, 0] for k in (8, 16, 24)}
    for li in range(len(model.layers)):
        rank = sum(r[li][1] + r[li][0] for r in rank_on)
        dec = sum(r[li][1] for r in score_on)
        order = np.argsort(-rank)
        d = dict(layer=li, decode_routings_scored=int(dec.sum()))
        for k in (8, 16, 24):
            share = float(dec[order[:k]].sum() / max(dec.sum(), 1))
            d[f"top{k}_share"] = share
            tot[k][0] += int(dec[order[:k]].sum())
            tot[k][1] += int(dec.sum())
        d["decode_counts_all"] = [int(x) for x in sum(r[li][1] for r in per_prompt)]
        res["layers"].append(d)
    res["overall"] = {f"top{k}_share": v[0] / max(v[1], 1) for k, v in tot.items()}
    res["uniform"] = {f"top{k}_share": k / E for k in (8, 16, 24)}
    print("overall decode share held by the top-k experts (ranked on other prompts):", json.dumps(res["overall"]),
          "uniform:", json.dumps(res["uniform"]), flush=True)
    if a.write_rank:
        rank = {str(li): [int(e) for e in np.argsort(-sum(r[li][1] for r in per_prompt), kind="stable")]
                for li in range(len(model.layers))}
        json.dump(dict(source=a.out, prompts=len(prompts), new=a.new, layers=rank),
                  open(os.path.join(os.path.expanduser(a.write_rank), "expert_rank.json"), "w"), indent=1)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
