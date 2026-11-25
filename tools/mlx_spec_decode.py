"""MTP speculative decoding vs plain greedy decoding, one request (spec_decode.py).

Per prompt: plain greedy tokens from Engine (overlap), then SpecGenerator with k drafts; reports token agreement with
plain greedy (first divergence), draft acceptance and decode ms/token (first to last token, prefill excluded).

    python tools/mlx_spec_decode.py --model <mlx 9B dir> --mtp ~/.cache/millijoule/Qwen3.5-9B-mtp/mtp.safetensors \
        --k 1,2,3 --new 128 --out results/spec_9b/check.json
"""

import argparse
import json
import os
os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "150")  # before mlx is imported: RESULTS 1p
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

PROMPTS = [
    "Explain how a refrigerator works, step by step, for a curious ten-year-old.",
    "Write a Python function that returns the n-th Fibonacci number using memoization, and explain it.",
    "Summarize the causes and consequences of the French Revolution in about 200 words.",
    "List ten practical tips for writing clear technical documentation.",
]


def prompt_sets(tok_path, which):
    from transformers import AutoTokenizer

    out = []
    if "chat" in which:
        tok = AutoTokenizer.from_pretrained(tok_path)
        for p in PROMPTS:
            ids = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True,
                                          enable_thinking=False, tokenize=True, return_dict=False)
            ids = ids["input_ids"] if isinstance(ids, dict) or hasattr(ids, "keys") else ids
            out.append(("chat:" + p[:40], [int(t) for t in ids]))
    if "random" in which:  # tools/mlx_engine_compare.py's prompt
        out.append(("random256", [int(t) for t in np.random.RandomState(0).randint(0, 200000, 256)]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mtp", required=True)
    ap.add_argument("--k", default="1,2", help="comma list of draft counts and/or auto-time, auto-energy")
    ap.add_argument("--k-max", type=int, default=3)
    ap.add_argument("--cost-file", default="~/.cache/millijoule/Qwen3.5-9B-mtp/spec_cost.json")
    ap.add_argument("--new", type=int, default=128)
    ap.add_argument("--prompts", default="chat,random")
    ap.add_argument("--mtp-bits", type=int, default=4)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--no-plain", action="store_true")
    ap.add_argument("--draft-vocab", default="", help="comma list of .npy files (or 'full')")
    ap.add_argument("--out")
    a = ap.parse_args()
    import mlx.core as mx

    from mj_mlx import Engine, load_model
    from mj_mlx.spec_decode import SpecGenerator
    from mlx_engine_compare import make_policy

    mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    mx.set_cache_limit(1 << 30)
    model = load_model(a.model)
    rows = []
    vocabs = [v for v in a.draft_vocab.split(",") if v] or ["full"]
    gens = {}
    for v in vocabs:
        dv = None if v == "full" else np.load(os.path.expanduser(v))
        for k in a.k.split(","):
            pol = make_policy(k, a.k_max, a.cost_file)
            g = SpecGenerator(model, os.path.expanduser(a.mtp), k=a.k_max if pol else int(k), max_len=1024,
                              mtp_bits=a.mtp_bits or None, draft_vocab=dv)
            g.new_policy = pol
            gens[(k, os.path.basename(v))] = g

    def timed(fn):
        times = []
        toks = fn(lambda n: times.extend([time.perf_counter()] * n))
        return toks, 1e3 * (times[-1] - times[0]) / (len(times) - 1)

    for name, ids in prompt_sets(a.model, a.prompts):
        ref, ref_ms = None, None
        if not a.no_plain:
            def plain(cb):
                eng = Engine(model=model, num_blocks=80, max_running=1, max_batch_tokens=2048, overlap=True,
                             num_state_slots=1)
                eng.add_request(ids, a.new, ignore_eos=True)
                out = []
                while eng.has_work():
                    before = eng.tokens_out
                    for f in eng.step():
                        out = list(f[2])
                    if eng.tokens_out > before:
                        cb(eng.tokens_out - before)
                return out
            plain(lambda n: None)  # warm-up
            ms = []
            for _ in range(a.reps):
                ref, t = timed(plain)
                ms.append(t)
            ref_ms = float(np.median(ms))
            print(f"{name:48s} plain   {ref_ms:6.2f} ms/token", flush=True)
        for (k, vname), g in gens.items():
            if g.new_policy:
                g.policy = g.new_policy()
            g.generate(ids, 16)  # warm-up (compiles the k+1-row shapes)
            ms, st = [], None
            for _ in range(a.reps):
                if g.new_policy:
                    g.policy = g.new_policy()  # a fresh policy per request
                g.reset_stats()
                toks, t = timed(lambda cb: g.generate(ids, a.new, cb))
                ms.append(t)
                st = dict(g.stats)
            first_diff = next((i for i, (x, y) in enumerate(zip(toks, ref)) if x != y), None) if ref else None
            r = dict(prompt=name, k=k, draft_vocab=vname, mtp_bits=a.mtp_bits, ms_per_token=float(np.median(ms)), ms_reps=ms,
                     plain_ms_per_token=ref_ms, speedup=ref_ms / float(np.median(ms)) if ref_ms else None,
                     accept_rate=st["accepted"] / max(st["drafted"], 1), k_hist=st["k_hist"],
                     tokens_per_iter=(st["accepted"] + st["iters"]) / max(st["iters"], 1),
                     same_as_plain=None if ref is None else toks == ref, first_diff=first_diff, tokens=toks,
                     plain_tokens=ref)
            rows.append(r)
            print(f"{name:40s} k={k} {vname:16s} {r['ms_per_token']:6.2f} ms/token  x{r['speedup'] or 0:.2f}  accept "
                  f"{r['accept_rate']:.2f}  tok/iter {r['tokens_per_iter']:.2f}  same {r['same_as_plain']} "
                  f"(first diff {first_diff})", flush=True)
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(dict(model=a.model, mtp=a.mtp, new=a.new, rows=rows, time=time.strftime("%Y-%m-%dT%H:%M:%S")),
                  open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
