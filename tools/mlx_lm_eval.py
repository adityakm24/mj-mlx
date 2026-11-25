"""MMLU and GSM8K subsets on our engine, mlx-lm and llama.cpp (checklist 0.2).

The same prompts and scoring on every engine, so the numbers compare engines and weight formats, not prompt styles:
  mmlu   5-shot, Hendrycks format ("The following are multiple choice questions (with answers) about <subject>."; the
         subject's first 5 dev questions as shots; "Answer:"), scored by the highest log-probability among the four
         answer letters at the next token (llama.cpp: among its top-20 next-token log-probabilities). A stratified
         subset: --per-subject questions from each of the 57 subjects (seeded).
  gsm8k  5-shot ("Question: ...\\nAnswer: <worked answer> #### <number>"; the first 5 train problems as shots), greedy,
         up to 256 new tokens, cut at the next "Question:"; correct when the number after "####" (else the last
         number) equals the reference. A seeded subset of the test set.
--chat (for chat / reasoning models such as gpt-oss, whose raw few-shot numbers do not compare across engines):
  zero-shot through each engine's chat template, greedy, up to --chat-new tokens (reasoning first), reasoning effort
  --effort where the template takes it. mmlu: "Answer: X" at the end, scored by the last such letter (else the last
  standalone A-D); gsm8k: the number after "####" (else the last number). --every N keeps every N-th item (an even
  sample across all subjects).
The subsets are written once from the Hugging Face parquet files by --prepare (needs pyarrow) to
results/accuracy/{mmlu,gsm8k}_subset.jsonl, which every later run reads.

    python tools/mlx_lm_eval.py --prepare                      # once; any Python with pyarrow + huggingface_hub
    .venv/bin/python tools/mlx_lm_eval.py --engine mj --model <mlx dir> [--mj-kw JSON] --tasks mmlu,gsm8k --out x.json
    .venv/bin/python tools/mlx_lm_eval.py --engine mlx-lm --model <mlx dir> --out y.json
    .venv/bin/python tools/mlx_lm_eval.py --engine llama-server --api-url http://127.0.0.1:8092 --model <mlx dir> --out z.json
(--model is still needed for llama-server: its tokenizer finds the answer letters' token strings.)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ACC = os.path.join(ROOT, "results", "accuracy")
LETTERS = "ABCD"
SHOTS = 5


# ---- data -----------------------------------------------------------------------------------------------------

def prepare(per_subject=20, gsm_n=250, seed=0):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    rng = np.random.RandomState(seed)
    rows = lambda repo, f: pq.read_table(hf_hub_download(repo, f, repo_type="dataset")).to_pylist()
    test, dev = rows("cais/mmlu", "all/test-00000-of-00001.parquet"), rows("cais/mmlu", "all/dev-00000-of-00001.parquet")
    out = []
    for subj in sorted({r["subject"] for r in test}):
        qs = [r for r in test if r["subject"] == subj]
        shots = [r for r in dev if r["subject"] == subj][:SHOTS]
        for i in sorted(rng.choice(len(qs), min(per_subject, len(qs)), replace=False)):
            out.append(dict(subject=subj, q=qs[i], shots=shots))
    with open(os.path.join(ACC, "mmlu_subset.jsonl"), "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    gtest = rows("openai/gsm8k", "main/test-00000-of-00001.parquet")
    gtrain = rows("openai/gsm8k", "main/train-00000-of-00001.parquet")
    pick = sorted(rng.choice(len(gtest), gsm_n, replace=False))
    with open(os.path.join(ACC, "gsm8k_subset.jsonl"), "w") as f:
        for i in pick:
            f.write(json.dumps(dict(idx=int(i), q=gtest[i], shots=gtrain[:SHOTS])) + "\n")
    print(f"mmlu {len(out)} questions, gsm8k {len(pick)} problems -> {ACC}")


def load(task, limit=None):
    with open(os.path.join(ACC, f"{task}_subset.jsonl")) as f:
        rows = [json.loads(l) for l in f]
    return rows[:limit] if limit else rows


def mmlu_q(r, answer=True):
    s = r["question"].strip() + "\n" + "".join(f"{L}. {c}\n" for L, c in zip(LETTERS, r["choices"])) + "Answer:"
    return s + (f" {LETTERS[r['answer']]}\n\n" if answer else "")


def mmlu_prompt(item):
    head = f"The following are multiple choice questions (with answers) about {item['subject'].replace('_', ' ')}.\n\n"
    return head + "".join(mmlu_q(s) for s in item["shots"]) + mmlu_q(item["q"], answer=False)


def gsm_prompt(item):
    shots = "".join(f"Question: {s['question']}\nAnswer: {s['answer']}\n\n" for s in item["shots"])
    return shots + f"Question: {item['q']['question']}\nAnswer:"


NUM = re.compile(r"-?[\d,]*\.?\d+")


CHAT_MMLU = ("Answer the following multiple choice question about {subject}.\n\n{q}\n\n"
             "End your reply with a line of the form 'Answer: X', where X is A, B, C or D.")
CHAT_GSM = "Solve this problem. End your reply with '#### <number>'.\n\n{q}"


def chat_letter(text):
    m = re.findall(r"Answer:\s*\**\(?([ABCD])\b", text)
    if m:
        return LETTERS.index(m[-1])
    m = re.findall(r"(?<![A-Za-z])([ABCD])(?![A-Za-z])", text)
    return LETTERS.index(m[-1]) if m else None


def gsm_number(text):
    text = text.split("Question:")[0]
    m = re.search(r"####\s*(-?[\d,]*\.?\d+)", text)
    s = m.group(1) if m else (NUM.findall(text) or [None])[-1]
    if s is None:
        return None
    try:
        return float(s.replace(",", "").rstrip("."))
    except ValueError:
        return None


# ---- engines --------------------------------------------------------------------------------------------------

def chat_enc(tok, text, effort):
    kw = {"reasoning_effort": effort} if effort else {}
    enc = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True, **kw)
    ids = enc["input_ids"] if hasattr(enc, "keys") else enc
    return [int(t) for t in (ids[0] if len(ids) and isinstance(ids[0], (list, tuple)) else ids)]


def make_engine(a, tok):
    """(letter_logprobs(prompt) -> [4], generate(prompt, max_new) -> text). With --chat, generate takes the user
    message and runs it through the chat template."""
    letter_ids = [tok(" " + L, add_special_tokens=False)["input_ids"][-1] for L in LETTERS]
    enc = ((lambda text: chat_enc(tok, text, a.effort)) if a.chat
           else (lambda text: tok(text, add_special_tokens=True)["input_ids"]))
    if a.engine == "llama-server" and a.chat:
        import urllib.request

        def generate(msg, new):
            body = dict(model="bench", messages=[{"role": "user", "content": msg}], max_tokens=new, temperature=0,
                        chat_template_kwargs={"reasoning_effort": a.effort} if a.effort else {})
            req = urllib.request.Request(a.api_url + "/v1/chat/completions", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=3600) as r:
                m = json.load(r)["choices"][0]["message"]
            if a.full_text and m.get("reasoning_content") is not None:  # gpt-oss's own format, for teacher forcing
                return ("<|channel|>analysis<|message|>" + m["reasoning_content"] + "<|end|><|start|>assistant"
                        "<|channel|>final<|message|>" + (m.get("content") or ""))
            return (m.get("reasoning_content") or "") + "\n" + (m.get("content") or "")

        return None, generate
    if a.engine == "llama-server":
        import urllib.request

        def post(body):
            req = urllib.request.Request(a.api_url + "/completion", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=3600) as r:
                return json.load(r)

        def letters(prompt):
            d = post(dict(prompt=prompt, n_predict=1, n_probs=20, temperature=0, cache_prompt=False))
            cp = (d.get("completion_probabilities") or [{}])[0]
            top = cp.get("top_logprobs") or cp.get("top_probs") or cp.get("probs") or []
            best = {}
            for t in top:
                s = (t.get("token") or t.get("tok_str") or "").strip()
                lp = t.get("logprob", math.log(max(t.get("prob", 0.0), 1e-30)))
                if s in LETTERS and lp > best.get(s, -1e30):
                    best[s] = lp
            return [best.get(L, -1e30) for L in LETTERS]

        def generate(prompt, new):
            d = post(dict(prompt=prompt, n_predict=new, temperature=0, stop=["Question:"], cache_prompt=False))
            return d.get("content", "")

        return letters, generate

    import mlx.core as mx

    if a.engine == "mlx-lm":
        from pathlib import Path

        from mlx_lm.generate import generate_step
        from mlx_lm.utils import load_model as lm_load

        lm, _ = lm_load(Path(a.model_path))

        def letters(prompt):
            lg = lm(mx.array(enc(prompt))[None])[0, -1].astype(mx.float32)
            lp = lg - mx.logsumexp(lg)
            return np.array(lp[mx.array(letter_ids)]).tolist()

        def generate(prompt, new):
            out = []
            for t, _ in generate_step(mx.array(enc(prompt)), lm, max_tokens=new):
                t = int(t)
                if t in eos:
                    break
                out.append(t)
                if not a.chat and "Question:" in tok.decode(out[-8:]):
                    break
            return tok.decode(out)

        eos = set([tok.eos_token_id] if isinstance(tok.eos_token_id, int) else (tok.eos_token_id or []))
        return letters, generate

    from mj_mlx import load_model
    from mj_mlx.gpt_oss import build_meta
    from mj_mlx.runner import Engine

    kw = json.loads(a.mj_kw)
    if a.placement != "none":
        kw["placement"] = a.placement
    model = load_model(a.model_path, **kw)
    pool_cap = [0, None]

    def letters(prompt):
        ids = enc(prompt)
        T = len(ids)
        if pool_cap[0] < T + 16:  # one KV pool, regrown for longer prompts
            pool_cap[:] = [T + 256, model.new_kv_pool(T + 256)]
        meta = build_meta(np.array(ids), np.arange(T), np.array([0]), np.array([T]),
                          np.arange((T + 15) // 16)[None], np.array([T - 1]), 16, model.cfg.sliding_window)
        x32 = model.final_hidden(meta, pool_cap[1])  # [1, hidden] (the last position)
        if getattr(model, "lm_head", 1) is None and getattr(model, "head_rows", None) is not None:
            # lean gpt-oss: full log-sum-exp over vocab blocks of the bf16 rows, plus the four letters' logits
            rows = model.head_rows.rows
            xb = x32.astype(mx.bfloat16)
            lse = None
            for b in range(0, rows.shape[0], 32768):
                lg = (xb @ mx.array(np.ascontiguousarray(rows[b:b + 32768])).view(mx.bfloat16).T).astype(mx.float32)
                blk = mx.logsumexp(lg, axis=-1)
                lse = blk if lse is None else mx.logaddexp(lse, blk)
            sel = (xb @ mx.array(np.ascontiguousarray(rows[letter_ids])).view(mx.bfloat16).T).astype(mx.float32)
            return np.array(sel[0] - lse[0]).tolist()
        lg = mx.array(model.full_logits(x32))[0].astype(mx.float32)
        return np.array(lg[mx.array(letter_ids)] - mx.logsumexp(lg)).tolist()

    def generate(prompt, new):
        ids = enc(prompt)
        eng = Engine(model=model, num_blocks=(len(ids) + new) // 16 + 16, max_running=1, max_batch_tokens=2048,
                     overlap=True)
        eng.add_request(ids, new)
        out = []
        while eng.has_work():
            for f in eng.step():
                out = list(f[2])
        eos = tok.eos_token_id
        eos = set([eos] if isinstance(eos, int) else (eos or []))
        return tok.decode([t for t in out if t not in eos] if a.chat else out)

    return letters, generate


# ---- run ------------------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--per-subject", type=int, default=20)
    ap.add_argument("--gsm-n", type=int, default=250)
    ap.add_argument("--engine", choices=["mj", "mlx-lm", "llama-server"])
    ap.add_argument("--model", help="mlx checkpoint dir (tokenizer; and the weights for mj / mlx-lm)")
    ap.add_argument("--mj-kw", default="{}")
    ap.add_argument("--placement", default="auto", help="MoeLM families; 'none' for gpt-oss / Qwen3.5 (use --mj-kw)")
    ap.add_argument("--api-url", default="http://127.0.0.1:8092")
    ap.add_argument("--tasks", default="mmlu,gsm8k")
    ap.add_argument("--limit", type=int, default=None, help="first N items per task (smoke tests)")
    ap.add_argument("--every", type=int, default=1, help="keep every N-th item (an even sample over subjects)")
    ap.add_argument("--chat", action="store_true", help="zero-shot through the chat template (reasoning models)")
    ap.add_argument("--chat-new", type=int, default=1024)
    ap.add_argument("--effort", default="low", help="reasoning effort for templates that take one ('' = none)")
    ap.add_argument("--idx", default="", help="gsm8k: only these item idx values (comma list), after --every")
    ap.add_argument("--full-text", action="store_true", help="keep whole replies (default: the last 600 characters)")
    ap.add_argument("--out")
    a = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from swap_guard import guard_self

    guard_self()  # stop past MJ_SWAP_ABORT_GB of swap-outs (exit 3); no partial results
    if a.prepare:
        return prepare(a.per_subject, a.gsm_n)
    if not (a.engine and a.model and a.out):
        ap.error("--engine, --model and --out are needed (or --prepare)")
    os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "150")
    import glob

    from transformers import AutoTokenizer

    a.model_path = glob.glob(os.path.expanduser(a.model))[0]
    tok = AutoTokenizer.from_pretrained(a.model_path)
    t0 = time.perf_counter()
    letters, generate = make_engine(a, tok)
    res = dict(engine=a.engine, model=a.model_path, mj_kw=json.loads(a.mj_kw), load_s=time.perf_counter() - t0, tasks={},
               chat=a.chat, chat_new=a.chat_new if a.chat else None, effort=a.effort if a.chat else None, every=a.every)
    for task in a.tasks.split(","):
        items, rows, t1 = load(task, a.limit)[::a.every], [], time.perf_counter()
        if a.idx and task == "gsm8k":
            keep = {int(x) for x in a.idx.split(",")}
            items = [it for it in items if it["idx"] in keep]
        cut = (lambda t: t) if a.full_text else (lambda t: t[-600:])
        for i, it in enumerate(items):
            if a.chat and task == "mmlu":
                q = it["q"]
                text = generate(CHAT_MMLU.format(subject=it["subject"].replace("_", " "),
                                                 q=mmlu_q(q, answer=False)[:-len("Answer:")].strip()), a.chat_new)
                pred = chat_letter(text.split("<|end|>")[-1] if "<|end|>" in text else text)
                rows.append(dict(subject=it["subject"], pred=pred, gold=q["answer"], ok=pred == q["answer"],
                                 text=cut(text)))
            elif a.chat:
                text = generate(CHAT_GSM.format(q=it["q"]["question"]), a.chat_new)
                gold, pred = gsm_number(it["q"]["answer"]), gsm_number(text)
                rows.append(dict(idx=it["idx"], pred=pred, gold=gold, ok=pred is not None and abs(pred - gold) < 1e-6,
                                 text=cut(text)))
            elif task == "mmlu":
                lp = letters(mmlu_prompt(it))
                pred = int(np.argmax(lp))
                rows.append(dict(subject=it["subject"], pred=pred, gold=it["q"]["answer"], ok=pred == it["q"]["answer"],
                                 logprobs=lp))
            else:
                text = generate(gsm_prompt(it), 256)
                gold = gsm_number(it["q"]["answer"])
                pred = gsm_number(text)
                rows.append(dict(idx=it["idx"], pred=pred, gold=gold, ok=pred is not None and abs(pred - gold) < 1e-6,
                                 text=text[:600]))
            if (i + 1) % 50 == 0:
                acc = np.mean([r["ok"] for r in rows])
                print(f"{task} {i + 1}/{len(items)}: acc {acc:.4f} ({time.perf_counter() - t1:.0f} s)", flush=True)
        acc = float(np.mean([r["ok"] for r in rows]))
        res["tasks"][task] = dict(n=len(rows), acc=acc, se=math.sqrt(acc * (1 - acc) / len(rows)),
                                  seconds=time.perf_counter() - t1, rows=rows)
        print(f"{task}: {acc:.4f} +- {res['tasks'][task]['se']:.4f} (n {len(rows)})", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    if os.path.exists(a.out):
        sys.exit(f"{a.out} exists; results are never overwritten")
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
