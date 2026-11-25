"""Long-context retrieval through the chat template (checklist 0.15 and 0.2): does each engine find a fact hidden at a
known depth of a long document?

Raw-text perplexity past a model's training length (RESULTS 1zq) says little about chat use. This is the usual
needle-in-a-haystack form instead (Kamradt 2023; the retrieval tasks of RULER, Hsieh et al. 2024), kept small enough to
run on 16 GB:
- A document of WikiText-2 test paragraphs (results/accuracy/wiki.test.raw, committed) is filled to the target length
  in the model's own tokens.
- Eight invented facts ("The access code for the Velmora archive is 48213.") are placed at evenly spread depths; the
  question asks for one of them, a different depth per document, so every engine sees the same documents and questions.
- Greedy, through each engine's chat template; up to --new tokens (reasoning models think first).
- Scored: correct when the last five-digit number in the reply is the asked code; "distractor" when it is another code
  from the document; "none" when the reply has no five-digit number.
Seeded: the same seed gives the same documents on every engine and every run.

    .venv/bin/python tools/mlx_longctx_qa.py --model <mlx dir> --engine mj --ctx 4096,8192,16384 --docs 12 \
        --out results/longctx_qa_v1/gptoss_mj.json
    .venv/bin/python tools/mlx_longctx_qa.py --model <mlx dir> --engine llama-server --api-url http://127.0.0.1:8092 ...
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
WIKI = os.path.join(ROOT, "results", "accuracy", "wiki.test.raw")
NAMES = ["Velmora", "Quindar", "Ostrel", "Brenwick", "Tavisand", "Kelloran", "Mirovast", "Dunshelm", "Pellagrine",
         "Sorvane", "Ashquill", "Thrennic", "Galvoreth", "Ulmadine", "Corviless", "Fenwarrow", "Istravel", "Norrocan",
         "Hollibrand", "Zenthiar", "Wyrmesk", "Laurivale", "Opheran", "Raskeleth"]
FACTS = 8


def paragraphs():
    text = open(WIKI, encoding="utf-8").read()
    return [p.strip() for p in text.split("\n") if len(p.strip()) > 200 and not p.strip().startswith("=")]


def build(tok, ctx, doc, seed, paras):
    """(user message, asked name, asked code, all codes, depth of the asked fact)."""
    rs = np.random.RandomState(seed * 1000 + doc)
    names = list(rs.choice(NAMES, FACTS, replace=False))
    codes = [int(c) for c in rs.choice(np.arange(10000, 100000), FACTS, replace=False)]
    facts = [f"The access code for the {n} archive is {c}." for n, c in zip(names, codes)]
    ask = doc % FACTS  # the asked fact's depth slot rotates over the documents
    q = (f"\n\nQuestion: What is the access code for the {names[ask]} archive? Answer with the five-digit code only.")
    budget = ctx - len(tok.encode(q)) - sum(len(tok.encode(" " + f)) for f in facts) - 96  # chat template room
    start = rs.randint(0, len(paras))
    body, n_tok, i = [], 0, start
    while n_tok < budget:
        p = paras[i % len(paras)]
        body.append(p)
        n_tok += len(tok.encode(" " + p))
        i += 1
    # trim the last paragraph to the budget, in tokens
    text = " ".join(body)
    ids = tok.encode(text)[:budget]
    text = tok.decode(ids)
    sents = re.split(r"(?<=[.!?]) ", text)
    slots = [int(len(sents) * (k + 0.5) / FACTS) for k in range(FACTS)]  # evenly spread depths
    for k in reversed(range(FACTS)):
        sents.insert(slots[k], facts[k])
    return " ".join(sents) + q, names[ask], codes[ask], codes, (ask + 0.5) / FACTS


def chat_ids(tok, msg):
    enc = tok.apply_chat_template([{"role": "user", "content": msg}], add_generation_prompt=True, tokenize=True)
    ids = enc["input_ids"] if hasattr(enc, "keys") else enc
    if len(ids) and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return [int(t) for t in ids]


def score(reply, code, codes):
    nums = [int(x) for x in re.findall(r"(?<!\d)\d{5}(?!\d)", reply)]
    if not nums:
        return "none"
    if nums[-1] == code:
        return "correct"
    return "distractor" if nums[-1] in codes else "wrong"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--engine", choices=["mj", "mlx-lm", "llama-server"], required=True)
    ap.add_argument("--mj-kw", default="{}")
    ap.add_argument("--placement", default="auto")
    ap.add_argument("--api-url", default="http://127.0.0.1:8092")
    ap.add_argument("--ctx", default="4096,8192,16384")
    ap.add_argument("--docs", type=int, default=12)
    ap.add_argument("--new", type=int, default=384)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import glob

    from swap_guard import guard_self

    from transformers import AutoTokenizer

    path = glob.glob(os.path.expanduser(a.model))[0]
    tok = AutoTokenizer.from_pretrained(path)
    paras = paragraphs()
    eos = tok.eos_token_id
    eos = set([eos] if isinstance(eos, int) else (eos or []))

    if a.engine == "mj":
        import mlx.core as mx

        from mj_mlx import load_model
        from mj_mlx.runner import Engine

        kw = json.loads(a.mj_kw)
        if a.placement != "none":
            kw["placement"] = a.placement
        model = load_model(path, **kw)

        def gen(msg):
            ids = chat_ids(tok, msg)
            eng = Engine(model=model, num_blocks=(len(ids) + a.new) // 16 + 16, max_running=1,
                         max_batch_tokens=2048, overlap=True, enable_prefix_cache=False)
            eng.add_request(ids, a.new)
            out = []
            while eng.has_work():
                for f in eng.step():
                    out = list(f[2])
            mx.clear_cache()
            return len(ids), tok.decode([int(t) for t in out if int(t) not in eos])
    elif a.engine == "mlx-lm":
        from pathlib import Path

        import mlx.core as mx
        from mlx_lm.generate import generate_step
        from mlx_lm.utils import load_model as lm_load

        lm, _ = lm_load(Path(path))

        def gen(msg):
            ids = chat_ids(tok, msg)
            out = []
            for t, _ in generate_step(mx.array(ids), lm, max_tokens=a.new):
                t = int(t)
                if t in eos:
                    break
                out.append(t)
            mx.clear_cache()
            return len(ids), tok.decode(out)
    else:
        import urllib.request

        def gen(msg):
            body = dict(model="bench", messages=[{"role": "user", "content": msg}], max_tokens=a.new, temperature=0)
            req = urllib.request.Request(a.api_url + "/v1/chat/completions", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=7200) as r:
                d = json.load(r)
            m = d["choices"][0]["message"]
            text = (m.get("reasoning_content") or "") + "\n" + (m.get("content") or "")
            return (d.get("usage") or {}).get("prompt_tokens"), text

    rows = []

    def save(aborted=None):
        summ = {}
        for ctx in sorted({r["ctx"] for r in rows}):
            sel = [r for r in rows if r["ctx"] == ctx]
            summ[str(ctx)] = {k: sum(r["result"] == k for r in sel) for k in ("correct", "distractor", "wrong", "none",
                                                                             "error")} | dict(n=len(sel))
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(dict(model=path, engine=a.engine, mj_kw=a.mj_kw, placement=a.placement, new=a.new, seed=a.seed,
                       docs=a.docs, summary=summ, rows=rows, aborted=aborted, notes=__doc__), open(a.out, "w"), indent=1)
        return summ

    guard_self(lambda gb: save(f"swap-outs {gb:.1f} GB passed the limit"))
    for ctx in [int(c) for c in a.ctx.split(",")]:
        for doc in range(a.docs):
            msg, name, code, codes, depth = build(tok, ctx, doc, a.seed, paras)
            t0 = time.perf_counter()
            try:
                n_in, reply = gen(msg)
                err = None
            except Exception as e:  # out of memory, server error: recorded, not hidden
                n_in, reply, err = None, "", f"{type(e).__name__}: {e}"[:300]
            res = "error" if err else score(reply, code, codes)
            rows.append(dict(ctx=ctx, doc=doc, depth=depth, name=name, code=code, n_in=n_in, result=res,
                             seconds=time.perf_counter() - t0, reply=reply[-600:], error=err))
            print(f"ctx {ctx} doc {doc} depth {depth:.2f}: {res} ({n_in} tokens, {rows[-1]['seconds']:.1f} s)",
                  flush=True)
            save()
    print(json.dumps(save()))


if __name__ == "__main__":
    main()
