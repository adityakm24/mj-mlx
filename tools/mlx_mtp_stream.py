"""MTP speculative decoding on a streamed MoE model (Qwen3.6-35B-A3B: bigger than RAM, experts from the SSD).

Verify steps of k+1 tokens load each routed expert once for all of them, so the SSD reads and the per-layer routing
round trips of a streamed model are shared across the accepted tokens. One process per arm; each arm runs the same
SpecGenerator loop (k = 0 is plain greedy decoding through that loop, the reference), one warm-up request, then
--reps requests. Greedy tokens must equal the k = 0 arm's (checked by the caller from the saved tokens).

    MJ_GPU_STREAM=1 .venv/bin/python tools/mlx_mtp_stream.py --k 2 --prompt chat --out results/mtp_35b/r1_chat_k2.json
"""

import argparse
import glob
import json
import os
os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "150")  # before mlx is imported: RESULTS 1p
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

CHAT = "Write a Python function that returns the n-th Fibonacci number using memoization, and explain it."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="~/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/*")
    ap.add_argument("--experts", default="~/.cache/millijoule/qwen36-35b-experts-stream")
    ap.add_argument("--mtp", default="~/.cache/millijoule/Qwen3.6-35B-mtp/mtp.safetensors")
    ap.add_argument("--draft-vocab", default="")
    ap.add_argument("--slots", type=int, default=96)
    ap.add_argument("--k", default="2", help="drafts per iteration, or auto (DraftPolicy, time mode, k_max 3)")
    ap.add_argument("--prompt", choices=["chat", "random"], default="chat")
    ap.add_argument("--new", type=int, default=128)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import mlx.core as mx
    from transformers import AutoTokenizer

    from mj_mlx import load_model
    from mj_mlx.spec_decode import DraftPolicy, SpecGenerator

    path = glob.glob(os.path.expanduser(a.model))[0]
    mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    model = load_model(path, stream_experts=os.path.expanduser(a.experts), expert_slots=a.slots, lean=True,
                       io_threads=8, io_nocache=True, prefetch=True)
    tok = AutoTokenizer.from_pretrained(path)
    if a.prompt == "chat":
        r = tok.apply_chat_template([{"role": "user", "content": CHAT}], add_generation_prompt=True,
                                    enable_thinking=False, tokenize=True, return_dict=False)
        ids = [int(t) for t in (r["input_ids"] if hasattr(r, "keys") else r)]
    else:
        ids = [int(t) for t in np.random.RandomState(0).randint(1000, 100000, 256)]
    if a.k == "engine":  # reference: the plain engine path (overlap mode), same prompt and decode window
        from mj_mlx import Engine

        runs = []
        for rep in range(a.reps + 1):
            eng = Engine(model=model, num_blocks=(len(ids) + a.new + 64) // 16 + 8, max_running=1,
                         max_batch_tokens=2048, overlap=True)
            eng.add_request(ids, a.new, ignore_eos=True)
            times, out, t0 = [], [], time.perf_counter()
            while eng.has_work():
                before = eng.tokens_out
                for f in eng.step():
                    out = list(f[2])
                if eng.tokens_out > before:
                    times.append((time.perf_counter(), eng.tokens_out - before))
            tt = [t for t, _ in times]
            ntok = sum(n for _, n in times[1:])
            ms = 1e3 * (tt[-1] - tt[0]) / max(ntok, 1)
            print(f"rep {rep}: {ms:.1f} ms/token ({1e3 / ms:.2f} tok/s) engine overlap", flush=True)
            if rep > 0:
                runs.append(dict(rep=rep, decode_ms_per_token=ms, total_s=time.perf_counter() - t0,
                                 tokens=[int(t) for t in out], expert_stats=model.expert_stats()))
            del eng
            mx.clear_cache()
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(dict(model=path, slots=a.slots, k=a.k, prompt=a.prompt, prompt_len=len(ids), new=a.new,
                       gpu_stream=os.environ.get("MJ_GPU_STREAM") == "1", runs=runs, notes=__doc__,
                       time=time.strftime("%Y-%m-%dT%H:%M:%S")), open(a.out, "w"), indent=1)
        return
    dv = np.load(os.path.expanduser(a.draft_vocab)) if a.draft_vocab else None
    auto = a.k == "auto"
    pol = DraftPolicy(3, mode="time") if auto else None
    gen = SpecGenerator(model, os.path.expanduser(a.mtp), k=0 if auto else int(a.k), max_len=len(ids) + a.new + 64,
                        draft_vocab=dv, policy=pol)
    runs = []
    for rep in range(a.reps + 1):
        if auto:
            gen.policy = DraftPolicy(3, mode="time")
        gen.reset_stats()
        times = []

        def on_tokens(n, _t=times):
            _t.append((time.perf_counter(), n))

        t0 = time.perf_counter()
        out = gen.generate(ids, a.new, on_tokens)
        dt = time.perf_counter() - t0
        # decode window: first emitted token to last
        tt = [t for t, _ in times]
        ntok = sum(n for _, n in times[1:])
        ms = 1e3 * (tt[-1] - tt[0]) / max(ntok, 1) if len(tt) > 1 else None
        st = dict(gen.stats)
        row = dict(rep=rep, decode_ms_per_token=ms, total_s=dt, tokens=[int(t) for t in out],
                   accept_rate=st.get("accepted", 0) / max(st.get("drafted", 0), 1), stats=st,
                   expert_stats=model.expert_stats())
        print(f"rep {rep}: {ms:.1f} ms/token ({1e3 / ms:.2f} tok/s), accept {row['accept_rate']:.2f}, "
              f"total {dt:.1f} s", flush=True)
        if rep > 0:
            runs.append(row)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(dict(model=path, slots=a.slots, k=a.k, prompt=a.prompt, prompt_len=len(ids), new=a.new,
                   gpu_stream=os.environ.get("MJ_GPU_STREAM") == "1", runs=runs, notes=__doc__,
                   time=time.strftime("%Y-%m-%dT%H:%M:%S")), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
