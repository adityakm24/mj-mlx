"""gpt-oss-20b lean decode: where the host time of one token goes (Engine, overlap mode, batch 1).

Timed by wrapping, per decode step (no repo code changed):
  resolve    runner._resolve_inputs: lean mode reads the previous step's token on the host (waits for that step's
             GPU work to finish), then substitutes it
  embed      embed_rows.take: the embedding row read on the CPU (page cache) and copied into a new mx.array
  build      the rest of model.hidden: building the 24-layer lazy graph in Python (no GPU work)
  head_wait  greedy_from up to the candidate ids on the host: np.array(cand) evaluates the whole step
             (layers + 4-bit head + top-16) and waits for it
  head_rows  head_rows.take of the 16 rescore rows (CPU) and the rescore graph
  submit     the final async_eval
  out_wait   Engine's np.array(toks) for the step (the rescore + argmax finishing)
Per-token wall time is printed next to the sum.

    .venv/bin/python tools/mlx_gptoss_step_trace.py --out results/gptoss_decode_parts/step_trace.json
"""

import argparse
import glob
import json
import os
os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "150")
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))


def pageins():
    import re
    import subprocess
    s = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    return int(re.search(r"Pageins:\s+(\d+)", s).group(1)) * 16384


def swap_mb():
    import re
    import subprocess
    s = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    return float(re.search(r"used = ([\d.]+)M", s).group(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attn-bits", type=int, default=None)
    ap.add_argument("--new", type=int, default=129)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--wired-mb", type=int, default=3000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import mlx.core as mx

    from mj_mlx import Engine, load_model
    from mj_mlx import runner as R
    from mlx_bench_worker import ids_for

    HF = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/*"))[0]
    MAP = os.path.expanduser("~/.cache/millijoule/gptoss20b-experts-mapped")
    for f in sorted(glob.glob(MAP + "/*.bin")):
        with open(f, "rb") as fh:
            while fh.read(64 << 20):
                pass
    kw = dict(attn_bits=a.attn_bits) if a.attn_bits else dict(attn_pack=True)
    m = load_model(HF, map_experts=MAP, lean=True, **kw)
    mx.set_cache_limit(1 << 30)
    T = defaultdict(float)
    on = [False]

    def wrap(obj, name, key, sub=None):
        f = getattr(obj, name)

        def g(*args, **kwargs):
            t = time.perf_counter()
            try:
                return f(*args, **kwargs)
            finally:
                if on[0]:
                    T[key] += time.perf_counter() - t
                    if sub:
                        T[sub] -= time.perf_counter() - t
        setattr(obj, name, g)

    wrap(R.MlxRunner, "_resolve_inputs", "resolve")
    wrap(m.embed_rows, "take", "embed", sub="build_incl")  # embed is called inside hidden: subtract from build
    wrap(m, "hidden", "build_incl")

    # greedy_from, split at the host read of the candidates (same math as GptOss.greedy_from)
    from mj_mlx.gpt_oss import topk_indices

    def greedy_from(x32):
        t0 = time.perf_counter()
        lg = m._candidate_logits(x32)
        cand = topk_indices(lg, m.rescore_k)
        c = np.array(cand)
        t1 = time.perf_counter()
        rows = m.head_rows.take(c.reshape(-1), m.dtype).reshape(*cand.shape, -1)
        exact = (rows.astype(mx.float32) * x32[:, None, :]).sum(axis=-1)
        out = mx.take_along_axis(cand, mx.argmax(exact, axis=-1)[:, None], axis=-1)[:, 0]
        t2 = time.perf_counter()
        if on[0]:
            T["head_wait"] += t1 - t0
            T["head_rows"] += t2 - t1
        return out
    m.greedy_from = greedy_from
    real_async = mx.async_eval

    def async_eval(*args):
        t = time.perf_counter()
        real_async(*args)
        if on[0]:
            T["submit"] += time.perf_counter() - t
    R.mx.async_eval = async_eval  # runner's module-level mx is mlx.core: patch the attribute it calls
    real_np_array = np.array

    res = []
    for rep in range(a.reps + 1):
        eng = Engine(model=m, num_blocks=(1 + a.new) // 16 + 16, max_running=1, max_batch_tokens=2048,
                     overlap=True, wired_limit_bytes=a.wired_mb << 20)
        eng.add_request(ids_for(1, 700 + rep), a.new, ignore_eos=True)
        T.clear()
        times = []
        pi0 = pageins()
        while eng.has_work():
            before = eng.tokens_out
            on[0] = len(times) >= 1  # time decode steps only (after the first token)
            t = time.perf_counter()
            eng.step()
            T["engine_step"] += time.perf_counter() - t if on[0] else 0
            if eng.tokens_out > before:
                times.append(time.perf_counter())
        on[0] = False
        pi = (pageins() - pi0) / 1e6
        n = len(times) - 1
        wall = (times[-1] - times[0]) / n * 1e3
        per = {k: v / n * 1e3 for k, v in T.items()}
        per["build"] = per.pop("build_incl", 0.0)
        per["out_wait"] = per["engine_step"] - sum(per.get(k, 0) for k in
                                                   ("resolve", "embed", "build", "head_wait", "head_rows", "submit"))
        row = dict(rep=rep, warmup=rep == 0, ms_per_token=wall, tok_s=1e3 / wall, pagein_mb=pi, swap_mb=swap_mb(),
                   parts_ms={k: round(v, 3) for k, v in per.items()})
        res.append(row)
        print(f"rep {rep}{' (warm-up)' if rep == 0 else ''}: {wall:.2f} ms/token ({1e3 / wall:.2f} tok/s) page-ins {pi:.0f} MB "
              + "  ".join(f"{k} {v:.2f}" for k, v in per.items()), flush=True)
        del eng
        mx.clear_cache()
        time.sleep(3)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(dict(args=vars(a), reps=res, notes=__doc__), open(a.out, "w"), indent=1)
    del real_np_array


if __name__ == "__main__":
    main()
