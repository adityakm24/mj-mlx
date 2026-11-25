"""Token-0 divergences, ours vs mlx-lm (bench_suite v1, Qwen3.5-9B): mlx-lm's last-position logits in bf16 and in
float32 for the two decode prompts whose first generated token differed; which token each engine chose.

    .venv/bin/python results/bench_suite/v1/tie_check.py   # writes tie_check.json next to this file
"""
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import mlx.core as mx  # noqa: E402
from mlx_bench_worker import ids_for  # noqa: E402
from mlx_lm.models.cache import make_prompt_cache  # noqa: E402
from mlx_lm.utils import load_model  # noqa: E402

M9 = glob.glob(str(Path.home() / ".cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-MLX-4bit/snapshots/*"))[0]
rows = [json.loads(l) for f in glob.glob(os.path.join(HERE, "blocks", "q9b_*.jsonl")) for l in open(f)]
CASES = [("tg128@d0", 101, 1), ("tg128@d2048", 201, 2048)]


def first(eng, test, seed):
    return next(r["output_ids"][0] for r in rows if r["engine"] == eng and r["test"] == test and r["seed"] == seed)


out = []
for dname in ("bfloat16", "float32"):
    model, _ = load_model(Path(M9))
    if dname == "float32":
        model.set_dtype(mx.float32)
    for test, seed, n in CASES:
        ids = ids_for(n, seed)
        lg = model(mx.array(ids)[None], cache=make_prompt_cache(model))[0, -1].astype(mx.float32)
        l = np.array(lg)
        top = [int(t) for t in np.argsort(-l)[:3]]
        a, b = first("mj", test, seed), first("mlx-lm", test, seed)
        out.append(dict(mlx_lm_dtype=dname, test=test, seed=seed, prompt_len=n, top3=top,
                        top3_logits=[float(l[t]) for t in top], ours_token=a, ours_logit=float(l[a]),
                        mlx_lm_token=b, mlx_lm_logit=float(l[b]), gap=float(abs(l[a] - l[b]))))
        print(out[-1])
    del model
json.dump(dict(model=M9, cases=out, notes=__doc__), open(os.path.join(HERE, "tie_check.json"), "w"), indent=1)
