"""Real-checkpoint check of the general MoE model (MoeLM) against mlx-lm on the same weights.

For each chat prompt: our greedy tokens (Engine, overlap on, placement as chosen) vs mlx-lm's greedy tokens, the
first position where they differ, and both texts; then teacher-forced on mlx-lm's tokens: top-1 agreement and mean
KL(ours || mlx-lm) of the next-token distributions. Both run the checkpoint's own quantized weights.

    .venv/bin/python tools/mlx_moe_check.py --model ~/.cache/huggingface/hub/models--mlx-community--OLMoE-1B-7B-0125-Instruct-4bit/snapshots/* \
        --out results/moe_lm/olmoe_check.json
"""

import argparse
import glob
import json
import os
os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "150")
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

PROMPTS = [
    "Explain how a refrigerator works, step by step, for a curious ten-year-old.",
    "Write a short Python function that returns the n-th Fibonacci number, then explain its running time.",
    "What are three differences between a comet and an asteroid?",
]


def logsoftmax(a):
    a = a - a.max(-1, keepdims=True)
    return a - np.log(np.exp(a).sum(-1, keepdims=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--new", type=int, default=64)
    ap.add_argument("--placement", default="auto")
    ap.add_argument("--skip-mlxlm", action="store_true", help="ours only (e.g. a model mlx-lm cannot hold in memory)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import mlx.core as mx
    from transformers import AutoTokenizer

    from mj_mlx import Engine, load_model
    from mj_mlx.gpt_oss import build_meta

    path = glob.glob(os.path.expanduser(a.model))[0]
    tok = AutoTokenizer.from_pretrained(path)
    seqs = []
    for p in PROMPTS:
        ids = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=True)
        seqs.append([int(t) for t in (ids["input_ids"] if hasattr(ids, "keys") else ids)])
    t0 = time.perf_counter()
    from mj_mlx import placement as PL

    where = PL.choose(path) if a.placement == "auto" else a.placement
    model = load_model(path, placement=where)
    load_s = time.perf_counter() - t0
    eng_kw = dict(num_blocks=256, max_running=1, max_batch_tokens=512, overlap=True, enable_prefix_cache=False)
    res = dict(model=path, placement=where, load_s=load_s, new=a.new, args=vars(a), prompts=[])
    ours_all = []
    for s in seqs:
        eng = Engine(model=model, **eng_kw)
        t = time.perf_counter()
        out = eng.generate([s], a.new, ignore_eos=True)[0]
        ours_all.append((out, time.perf_counter() - t))
        del eng
    if not a.skip_mlxlm:
        del model
        mx.clear_cache()
        from mlx_lm.generate import generate_step
        from mlx_lm.utils import load_model as lm_load
        from pathlib import Path

        lm, _ = lm_load(Path(path))
        lm_out = []
        for s in seqs:
            o = []
            t = time.perf_counter()
            for i, (tk, _) in enumerate(generate_step(mx.array(s), lm, max_tokens=a.new)):
                o.append(int(tk.item() if hasattr(tk, "item") else tk))
            lm_out.append((o, time.perf_counter() - t))
        # teacher-forced on mlx-lm's tokens: mlx-lm logits now, ours after reloading
        lm_lg = []
        for s, (o, _) in zip(seqs, lm_out):
            full = s + o
            lg = lm(mx.array(full)[None])[0, len(s) - 1: len(full) - 1].astype(mx.float32)
            lm_lg.append(np.array(lg))
        del lm
        mx.clear_cache()
        model = load_model(path, placement=where)
        for i, (s, (o, ts_lm), (ours, ts)) in enumerate(zip(seqs, lm_out, ours_all)):
            full = s + o
            T = len(full)
            pool = model.new_kv_pool(((T + 15) // 16 + 1) * 16)
            meta = build_meta(np.array(full), np.arange(T), np.array([0]), np.array([T]),
                              np.arange((T + 15) // 16)[None], np.arange(len(s) - 1, T - 1), 16, model.cfg.sliding_window)
            x32 = model.final_hidden(meta, pool)
            our_lg = np.array(model.full_logits(x32))
            lp, lq = logsoftmax(our_lg.astype(np.float64)), logsoftmax(lm_lg[i].astype(np.float64))
            kl = (np.exp(lp) * (lp - lq)).sum(-1)
            div = next((j for j in range(min(len(ours), len(o))) if ours[j] != o[j]), None)
            res["prompts"].append(dict(prompt=PROMPTS[i], first_divergence=div, new_tokens=len(ours), ours_text=tok.decode(ours),
                                       mlxlm_text=tok.decode(o), top1_agree=float((our_lg.argmax(-1) == lm_lg[i].argmax(-1)).mean()),
                                       mean_kl=float(kl.mean()), max_kl=float(kl.max()), ours_s=ts, mlxlm_s=ts_lm))
            print(f"prompt {i}: first divergence {div} | teacher-forced top-1 agree {res['prompts'][-1]['top1_agree']:.3f}, "
                  f"mean KL {kl.mean():.4f} | ours {a.new / ts:.1f} tok/s (incl. prefill), mlx-lm {a.new / ts_lm:.1f}",
                  flush=True)
    else:
        for i, (ours, ts) in enumerate(ours_all):
            res["prompts"].append(dict(prompt=PROMPTS[i], ours_text=tok.decode(ours), ours_s=ts, new_tokens=len(ours),
                                       ours_tok_s=len(ours) / ts))
            print(f"prompt {i}: ours {a.new / ts:.1f} tok/s (incl. prefill)", flush=True)
    print("ours, prompt 0:", repr(res["prompts"][0]["ours_text"][:300]))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
