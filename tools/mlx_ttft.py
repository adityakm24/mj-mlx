"""Time to first token for one request (warm), MilliJoule MLX runner vs mlx-lm, any supported model.

TTFT here = prompt processing plus the first sampled token, measured after a warm-up request of the same length,
one request at a time (no queueing). Prefill throughput = prompt tokens / TTFT.

    .venv/bin/python tools/mlx_ttft.py --engine mj --model <dir> --lens 128,512,2048 --out results/ttft/x.json
"""

from __future__ import annotations

import argparse
import json
import os
os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "150")  # before mlx is imported: RESULTS 1p
import sys
import time

import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["mj", "mlx-lm"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--lens", default="128,512,2048")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-batch-tokens", type=int, default=2048)
    ap.add_argument("--experts", default=None, help="repacked experts dir: stream experts from the SSD (mj only)")
    ap.add_argument("--slots", type=int, default=24, help="resident expert slots per layer when streaming")
    ap.add_argument("--prefetch", action=argparse.BooleanOptionalAction, default=True)  # --no-prefetch: off
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    mx.set_cache_limit(1 << 30)
    cfg = json.load(open(os.path.join(a.model, "config.json")))
    V = cfg.get("text_config", cfg)["vocab_size"]
    if a.engine == "mlx-lm":
        from pathlib import Path

        import mlx_lm
        from mlx_lm.generate import generate_step
        from mlx_lm.utils import load_model

        model, _ = load_model(Path(a.model))
        mx.eval(model.parameters())
        name = f"mlx-lm {mlx_lm.__version__}"

        def ttft(ids):
            t = time.perf_counter()
            for tok, _ in generate_step(mx.array(ids), model, max_tokens=1, prefill_step_size=a.max_batch_tokens):
                mx.eval(tok)
                break
            return time.perf_counter() - t
    else:
        from mj_mlx import Engine, load_model

        if a.experts:
            model = load_model(a.model, stream_experts=a.experts, expert_slots=a.slots, lean=True,
                               prefetch=a.prefetch)
            name = f"millijoule mlx runner, SSD expert streaming, {a.slots} slots/layer"
        else:
            model = load_model(a.model)
            name = "millijoule mlx runner"

        def ttft(ids):
            eng = Engine(model=model, num_blocks=(len(ids) + 64) // 16 + 8, max_running=1,
                         max_batch_tokens=a.max_batch_tokens)
            eng.add_request(ids, 1, ignore_eos=True)
            t = time.perf_counter()
            while eng.has_work():
                eng.step()
            return time.perf_counter() - t
    rows = []
    for L in [int(x) for x in a.lens.split(",")]:
        ids = [int(t) for t in np.random.RandomState(L).randint(0, min(V, 199998), L)]
        ttft(ids)  # warm-up (kernels for this shape)
        ts = sorted(ttft(ids) for _ in range(a.reps))
        rows.append(dict(prompt_tokens=L, ttft_s=ts[len(ts) // 2], ttft_min_s=ts[0], ttft_max_s=ts[-1],
                         prefill_tok_s=L / ts[len(ts) // 2]))
        print(f"{name}: {L:5d} tokens  TTFT {rows[-1]['ttft_s'] * 1e3:8.1f} ms  ({rows[-1]['prefill_tok_s']:7.0f} tok/s)",
              flush=True)
        mx.clear_cache()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(dict(engine=name, model=a.model, experts=a.experts, slots=a.slots if a.experts else None, model_note=cfg.get("bench_note", ""), reps=a.reps,
                   max_batch_tokens=a.max_batch_tokens, rows=rows), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
