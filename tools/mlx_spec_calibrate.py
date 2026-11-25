"""Cost table for spec_decode.DraftPolicy(mode="energy"): GPU joules and seconds per speculative iteration for each
fixed draft count k = 0 .. k_max, on this Mac.

Each k runs tools/mlx_engine_compare.py's mj-mtp path (one warm-up, then --reps runs of 128 tokens) on each prompt
(chat and random: the iteration cost barely depends on acceptance, both are kept so it is checked). j_per_iter is
the median over runs of (GPU-rail energy first token to last token) / iterations; s_per_iter likewise.

    python tools/mlx_spec_calibrate.py --model <mlx 9B dir> --out ~/.cache/millijoule/Qwen3.5-9B-mtp/spec_cost.json
"""

import argparse
import glob
import json
import os
os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "150")  # before mlx is imported: RESULTS 1p
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import mlx_engine_compare as ec  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mtp", default="~/.cache/millijoule/Qwen3.5-9B-mtp/mtp.safetensors")
    ap.add_argument("--draft-vocab", default="~/.cache/millijoule/Qwen3.5-9B-mtp/vocab65536.npy")
    ap.add_argument("--k-max", type=int, default=3)
    ap.add_argument("--prompts", default="chat,random")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    path = glob.glob(os.path.expanduser(a.model))[0]
    ec._TOK_PATH = path
    rows = []
    for prompt in a.prompts.split(","):
        ec.PROMPT = prompt
        for k in range(a.k_max + 1):
            one = ec.run_inproc("mj-mtp", path, a.mtp, str(k), a.draft_vocab, a.k_max)
            one()
            for i in range(a.reps):
                r = one()
                rows.append(dict(prompt=prompt, k=k, rep=i, j_per_iter=r["j_per_iter"], s_per_iter=r["s_per_iter"],
                                 ms_per_token=r["decode_ms_per_token"], mj_per_token=r["gpu_mj_per_token"],
                                 accept_rate=r["accept_rate"], gpu_w=r["gpu_w"]))
                print(f"{prompt:6s} k={k} rep {i}: {1e3 * r['j_per_iter']:6.1f} mJ/iter {1e3 * r['s_per_iter']:6.2f} "
                      f"ms/iter  {r['decode_ms_per_token']:6.2f} ms/token {r['gpu_mj_per_token']:6.1f} mJ/token "
                      f"accept {r['accept_rate']:.2f}", flush=True)
            del one
            import mlx.core as mx

            mx.clear_cache()
    j = [float(np.median([r["j_per_iter"] for r in rows if r["k"] == k])) for k in range(a.k_max + 1)]
    s = [float(np.median([r["s_per_iter"] for r in rows if r["k"] == k])) for k in range(a.k_max + 1)]
    out = dict(j_per_iter=j, s_per_iter=s, model=path, mtp=a.mtp, draft_vocab=a.draft_vocab, rows=rows,
               notes=__doc__, time=time.strftime("%Y-%m-%dT%H:%M:%S"))
    os.makedirs(os.path.dirname(os.path.expanduser(a.out)) or ".", exist_ok=True)
    json.dump(out, open(os.path.expanduser(a.out), "w"), indent=1)
    print("j_per_iter", [round(1e3 * x, 1) for x in j], "mJ;  s_per_iter", [round(1e3 * x, 2) for x in s], "ms")


if __name__ == "__main__":
    main()
