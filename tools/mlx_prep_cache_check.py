# Copyright 2026 the MilliJoule authors
# SPDX-License-Identifier: Apache-2.0
"""MJ_PREP_CACHE on the real gpt-oss-20b (RESULTS 1zzc): greedy tokens from a model loaded without the cache and from
one loaded from it, same prompts, must be identical; prints and writes both.

    .venv/bin/python tools/mlx_prep_cache_check.py --mj-kw JSON --cache ~/.cache/millijoule/prep --out x.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "150")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
PROMPTS = ["What is 17 times 23? Show the steps.", "Write a haiku about a mountain lake in winter.",
           "Explain what a hash table is to someone who has never programmed."]


def once(a):
    from transformers import AutoTokenizer

    from mj_mlx import Engine, load_model
    from mlx_chat_check import chat_ids

    path = glob.glob(os.path.expanduser(a.model))[0]
    tok = AutoTokenizer.from_pretrained(path)
    kw = json.loads(a.mj_kw)
    kw.pop("wired_mb", None)  # an Engine setting in the suite's kw (MLX's wired limit), not a model argument
    model = load_model(path, **kw)
    outs = []
    for p in PROMPTS:
        ids = chat_ids(tok, p)
        eng = Engine(model=model, num_blocks=(len(ids) + a.new) // 16 + 16, max_running=1, overlap=True)
        eng.add_request(ids, a.new)
        out = []
        while eng.has_work():
            for f in eng.step():
                out = list(f[2])
        outs.append([int(t) for t in out])
    print(json.dumps(outs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="~/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/*")
    ap.add_argument("--mj-kw", required=True)
    ap.add_argument("--cache", default="~/Library/Caches/mj-mlx/prep")
    ap.add_argument("--new", type=int, default=96)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.once:
        return once(a)
    res = {}
    for lab, env in (("no_cache", {"MJ_PREP_CACHE": "0"}), ("cache", {"MJ_PREP_CACHE": a.cache})):
        e = {k: v for k, v in os.environ.items() if k != "MJ_PREP_CACHE"}
        e.update(env, MJ_MLOCK="0")
        r = subprocess.run([sys.executable, __file__, "--once", "--model", a.model, "--mj-kw", a.mj_kw, "--new",
                            str(a.new)], env=e, capture_output=True, text=True)
        if r.returncode:
            print(r.stderr[-2000:])
            sys.exit(r.returncode)
        res[lab] = json.loads(r.stdout.strip().splitlines()[-1])
    same = res["no_cache"] == res["cache"]
    print("identical" if same else "DIFFERENT", [len(x) for x in res["cache"]])
    json.dump(dict(mj_kw=a.mj_kw, same=same, **res), open(a.out, "w"), indent=1)
    sys.exit(0 if same else 1)


if __name__ == "__main__":
    main()
