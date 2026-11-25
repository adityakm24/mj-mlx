"""Real-chat check (checklist 0.12): fixed chat prompts through each model's chat template, greedy, on our engine and
on mlx-lm with the same weights, then where the two first differ. Tiny checkpoints are already checked token for token
against transformers (tests/); on the real checkpoints, which transformers cannot hold in 16 GB, the reference is
mlx-lm on the same weights. llama-server (its GGUF: other weights) can be added for a read-through, not a
token comparison. All replies are saved for reading.

    .venv/bin/python tools/mlx_chat_check.py --model <mlx dir> [--mj-kw JSON] [--placement none] --new 128 \
        --engines mj,mlx-lm --out results/chat_check/olmoe_v1.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

PROMPTS = [
    "What is 17 times 23? Show the steps.",
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
    "Explain in two sentences why the sky is blue.",
    "Translate into French: 'The train leaves at seven in the morning.'",
    "List three differences between a list and a tuple in Python.",
    "A farmer has 12 sheep and buys 5 more, then sells 7. How many sheep are left?",
    "Summarize the plot of a heist story in exactly three sentences. Make the story up.",
    "What is the capital of Australia, and why is it not Sydney?",
    "Give a JSON object with keys name, age and city for a fictional person.",
    "Write a haiku about a mountain lake in winter.",
    "If a rectangle is 8 cm by 5 cm, what are its area and perimeter?",
    "Explain what a hash table is to someone who has never programmed.",
]


def chat_ids(tok, prompt):
    msgs = [{"role": "user", "content": prompt}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
    ids = enc["input_ids"] if hasattr(enc, "keys") else enc  # transformers 5 returns a BatchEncoding
    if len(ids) and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return [int(t) for t in ids]


def run_mj(a, path, tok, prompts):
    from mj_mlx import load_model
    from mj_mlx.runner import Engine

    kw = json.loads(a.mj_kw)
    if a.placement != "none":
        kw["placement"] = a.placement
    model = load_model(path, **kw)
    outs = []
    for p in prompts:
        ids = list(chat_ids(tok, p))
        eng = Engine(model=model, num_blocks=(len(ids) + a.new) // 16 + 16, max_running=1, max_batch_tokens=2048,
                     overlap=True)
        eng.add_request(ids, a.new)
        out = []
        while eng.has_work():
            for f in eng.step():
                out = list(f[2])
        outs.append([int(t) for t in out])
    return outs


def run_mlxlm(a, path, tok, prompts):
    from pathlib import Path

    import mlx.core as mx
    from mlx_lm.generate import generate_step
    from mlx_lm.utils import load_model as lm_load

    lm, _ = lm_load(Path(path))
    eos = tok.eos_token_id
    eos = set([eos] if isinstance(eos, int) else (eos or []))
    outs = []
    for p in prompts:
        out = []
        for t, _ in generate_step(mx.array(list(chat_ids(tok, p))), lm, max_tokens=a.new):
            t = int(t)
            out.append(t)
            if t in eos:
                break
        outs.append(out)
    return outs


def run_llama_server(a, path, tok, prompts):
    import urllib.request

    outs = []
    for p in prompts:
        body = dict(model="bench", messages=[{"role": "user", "content": p}], max_tokens=a.new, temperature=0)
        req = urllib.request.Request(a.api_url + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3600) as r:
            text = json.load(r)["choices"][0]["message"].get("content") or ""
        outs.append(text)
    return outs


def first_diff(x, y):
    for i, (u, v) in enumerate(zip(x, y)):
        if u != v:
            return i
    return None if len(x) == len(y) else min(len(x), len(y))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--mj-kw", default="{}")
    ap.add_argument("--placement", default="auto")
    ap.add_argument("--engines", default="mj,mlx-lm")
    ap.add_argument("--api-url", default="http://127.0.0.1:8092")
    ap.add_argument("--new", type=int, default=128)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from swap_guard import guard_self

    guard_self()  # stop past MJ_SWAP_ABORT_GB of swap-outs (exit 3); no partial results
    if os.path.exists(a.out):
        sys.exit(f"{a.out} exists; results are never overwritten")
    os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "150")
    from transformers import AutoTokenizer

    path = glob.glob(os.path.expanduser(a.model))[0]
    tok = AutoTokenizer.from_pretrained(path)
    res = dict(model=path, mj_kw=json.loads(a.mj_kw), new=a.new, prompts=PROMPTS, engines={})
    for eng in a.engines.split(","):
        t0 = time.perf_counter()
        outs = {"mj": run_mj, "mlx-lm": run_mlxlm, "llama-server": run_llama_server}[eng](a, path, tok, PROMPTS)
        res["engines"][eng] = dict(seconds=time.perf_counter() - t0, ids=None if eng == "llama-server" else outs,
                                   text=[o if isinstance(o, str) else tok.decode(o) for o in outs])
        print(f"{eng}: {time.perf_counter() - t0:.0f} s", flush=True)
    if "mj" in res["engines"] and "mlx-lm" in res["engines"]:
        cmp = []
        for x, y in zip(res["engines"]["mj"]["ids"], res["engines"]["mlx-lm"]["ids"]):
            d = first_diff(x, y)
            cmp.append(dict(identical=d is None, first_diff=d, len_mj=len(x), len_mlxlm=len(y)))
        res["mj_vs_mlxlm"] = cmp
        same = sum(c["identical"] for c in cmp)
        fd = [c["first_diff"] for c in cmp if c["first_diff"] is not None]
        print(f"ours vs mlx-lm: {same}/{len(cmp)} replies token-identical"
              + (f"; first difference at token {min(fd)}-{max(fd)} in the others" if fd else ""))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
