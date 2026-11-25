"""Batched MTP speculative decoding (spec_decode.SpecBatch) vs the plain engine, B requests decoding together.

Per level B: B chat prompts (PROMPTS cycled; different lengths), 128 new tokens each, EOS ignored, greedy.
- plain: Engine(overlap=True, max_running=B), all B requests added at once.
- mtp-k0 .. mtp-k3: SpecBatch with a fixed k. mtp-auto: a per-request DraftPolicy in energy mode, with the cost
  table measured per batch size (--batch-cost, written by --calibrate from the mtp-k0..k3 arms: GPU joules per
  request-iteration in the decode window), or else the one-request calibration scaled by batch_energy_cost.
Measured over the decode window only (from the step after which every request has its first token to the last
token): tok/s = tokens in the window / window seconds; GPU mJ/token over the same window (IOReport GPU rail).
Each arm: one warm-up, then --reps runs; medians. Token agreement: each arm's output vs the plain engine's, first
differing position per request.

    python tools/mlx_spec_batch.py --model <mlx 9B dir> --levels 1,2,4,8 --out results/spec_batch_9b/r1.json
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
    "Describe the water cycle and why it matters for agriculture.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Compare TCP and UDP, with one use case for each.",
    "What are the main differences between a list and a tuple in Python? Give examples.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mtp", default="~/.cache/millijoule/Qwen3.5-9B-mtp/mtp.safetensors")
    ap.add_argument("--draft-vocab", default="~/.cache/millijoule/Qwen3.5-9B-mtp/vocab65536.npy")
    ap.add_argument("--cost-file", default="~/.cache/millijoule/Qwen3.5-9B-mtp/spec_cost.json")
    ap.add_argument("--levels", default="1,2,4,8")
    ap.add_argument("--arms", default="plain,mtp-k2,mtp-auto")
    ap.add_argument("--new", type=int, default=128)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--batch-cost", default="", help="json {batch: [J per request-iteration, k = 0..3]}")
    ap.add_argument("--calibrate", default="", help="write the mtp-k0..k3 arms' J per request-iteration here")
    ap.add_argument("--policy", choices=["energy", "edp"], default="edp", help="mtp-auto's objective")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import mlx.core as mx
    from transformers import AutoTokenizer

    from mj_mlx import Engine, load_model
    from mj_mlx.energy import Meter
    from mj_mlx.spec_decode import DraftPolicy, SpecBatch

    mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    mx.set_cache_limit(1 << 30)
    model = load_model(a.model)
    tok = AutoTokenizer.from_pretrained(a.model)

    def ids_of(p):
        r = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True,
                                    enable_thinking=False, tokenize=True, return_dict=False)
        return [int(t) for t in (r["input_ids"] if hasattr(r, "keys") else r)]

    all_ids = [ids_of(p) for p in PROMPTS]
    dv = np.load(os.path.expanduser(a.draft_vocab))
    cost = json.load(open(os.path.expanduser(a.cost_file)))["j_per_iter"]
    if a.batch_cost:
        bc = json.load(open(os.path.expanduser(a.batch_cost)))
        key = "edp_per_iter" if a.policy == "edp" else "j_per_iter"
        cost = {int(b): v for b, v in bc[key].items()}
    levels = [int(x) for x in a.levels.split(",")]
    def make_gen(arm, B):  # one at a time, sized for this level (each holds its own state pools and MTP head)
        if arm.startswith("mtp-k"):
            return SpecBatch(model, os.path.expanduser(a.mtp), B, k=int(arm[5:]), max_len=512, draft_vocab=dv)
        return SpecBatch(model, os.path.expanduser(a.mtp), B, max_len=512, draft_vocab=dv, cost=cost,
                         policy=lambda: DraftPolicy(3, mode=a.policy, cost=cost if isinstance(cost, list) else None))

    def save():
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(dict(model=a.model, round=a.round, new=a.new, rows=rows, notes=__doc__,
                       time=time.strftime("%Y-%m-%dT%H:%M:%S")), open(a.out, "w"), indent=1)

    def run_plain(prompts):
        eng = Engine(model=model, num_blocks=64 * len(prompts) + 64, max_running=len(prompts), max_batch_tokens=2048,
                     overlap=True, num_state_slots=len(prompts))
        rids = [eng.add_request(p, a.new, ignore_eos=True) for p in prompts]
        seen, out, n, t0, m = set(), {}, 0, None, Meter()
        steps0 = None
        while eng.has_work():
            before = eng.tokens_out
            for f in eng.step():
                out[f[0]] = list(f[2])
            got = eng.tokens_out - before
            if t0 is None:
                seen.update(int(r) for r in eng.last_sampled)
                if len(seen) == len(prompts):
                    t0 = time.perf_counter()
                    m.__enter__()
            else:
                n += got
        dt = time.perf_counter() - t0
        m.__exit__(None, None, None)
        return [out[r] for r in rids], n, dt, m.window.gpu_j, n

    def run_spec(g, prompts):
        st = dict(n=0, t0=None)
        m = Meter()

        def cb(n, decoding):
            if not decoding:
                st["t0"] = time.perf_counter()
                m.__enter__()
            else:
                st["n"] += n

        g.stats = dict(steps=0, iters=0, drafted=0, accepted=0, rows=0)
        outs = g.generate(prompts, a.new, cb)
        dt = time.perf_counter() - st["t0"]
        m.__exit__(None, None, None)
        return outs, st["n"], dt, m.window.gpu_j, g.stats["iters"]

    rows = []
    for B in levels:
        prompts = [all_ids[i % len(all_ids)] for i in range(B)]
        ref = None
        for arm in a.arms.split(","):
            gen = None if arm == "plain" else make_gen(arm, B)
            fn = (lambda: run_plain(prompts)) if gen is None else (lambda g=gen: run_spec(g, prompts))
            try:
                fn()  # warm-up
            except RuntimeError as e:  # e.g. Metal out of memory for many rows' rollback capture
                print(f"B={B} {arm}: failed ({str(e)[:80]})", flush=True)
                rows.append(dict(level=B, arm=arm, error=str(e)[:200]))
                del gen, fn
                mx.clear_cache()
                continue
            res = []
            for _ in range(a.reps):
                outs, n, dt, j, iters = fn()
                res.append(dict(tok_s=n / dt, mj_per_token=1e3 * j / max(n, 1), tokens=n, seconds=dt,
                                j_per_iter=j / max(iters, 1), s_per_iter=dt / max(iters, 1)))
            if arm == "plain":
                ref = outs
            first_diff = [next((t for t, (x, y) in enumerate(zip(o, r)) if x != y), None)
                          for o, r in zip(outs, ref)] if ref else None
            st = dict(gen.stats) if gen is not None else None
            row = dict(level=B, arm=arm, tok_s=float(np.median([r["tok_s"] for r in res])),
                       mj_per_token=float(np.median([r["mj_per_token"] for r in res])), reps=res,
                       j_per_iter=float(np.median([r["j_per_iter"] for r in res])),
                       s_per_iter=float(np.median([r["s_per_iter"] for r in res])),
                       accept_rate=(st["accepted"] / max(st["drafted"], 1)) if st else None,
                       rows_per_step=(st["rows"] / max(st["steps"], 1)) if st else None,
                       first_diff_vs_plain=first_diff)
            rows.append(row)
            print(f"B={B} {arm:9s} {row['tok_s']:7.1f} tok/s {row['mj_per_token']:6.0f} mJ/token"
                  + (f" accept {row['accept_rate']:.2f} rows/step {row['rows_per_step']:.1f}" if st else "")
                  + f" first diff vs plain {first_diff}", flush=True)
            del gen, fn
            mx.clear_cache()
            save()
    if a.calibrate:
        tab, edp = {}, {}
        for B in levels:
            r = {x["arm"]: x for x in rows if x["level"] == B and "j_per_iter" in x}
            if all(f"mtp-k{k}" in r for k in range(2)):  # a k that failed (memory) gets an infinite cost
                tab[B] = [r[f"mtp-k{k}"]["j_per_iter"] if f"mtp-k{k}" in r else float("inf") for k in range(4)]
                edp[B] = [r[f"mtp-k{k}"]["j_per_iter"] * r[f"mtp-k{k}"]["s_per_iter"] if f"mtp-k{k}" in r
                          else float("inf") for k in range(4)]
        json.dump(dict(j_per_iter=tab, edp_per_iter=edp, notes="GPU J (and J x s) per request-iteration in the "
                       "decode window, fixed k = 0..3, from tools/mlx_spec_batch.py", source=a.out), open(os.path.expanduser(a.calibrate), "w"),
                  indent=1)
        print("batch cost table", {b: [round(1e3 * x) if x != float("inf") else None for x in v]
                                   for b, v in tab.items()}, "mJ")
    save()


if __name__ == "__main__":
    main()
