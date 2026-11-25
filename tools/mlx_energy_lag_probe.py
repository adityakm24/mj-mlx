"""How the IOReport GPU energy counter updates during one request (prefill, then decode).

Suite v4 (Qwen3.5-9B, our engine, tg128@d2048) gave two kinds of decode window: ~10 W (as at depth 0 and as
mlx-lm) and ~38-41 W (above the prefill's own power, so not real). gpt-oss and the 35B showed the same once, and
two windows read 0 J at 99.9% GPU active. Hypothesis: the counter is updated in lumps, and a prefill's energy
can land after the decode window opened. This samples the cumulative GPU rail every --dt seconds in a thread
through whole requests and writes (t, joules) with the first-token and last-token times, so the step pattern can
be read off.

    .venv/bin/python tools/mlx_energy_lag_probe.py --engine mj --out results/energy_lag/mj_d2048.json
"""

import argparse
import json
import os
os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "150")
import sys
import threading
import time
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from mj_mlx.energy import EnergyCounter  # noqa: E402
from mj_mlx.energy import _cf  # noqa: E402


def gpu_j(c, a, b):
    return c.between(a, b).get("GPU Energy", 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="mj", choices=["mj", "mlx-lm"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--depth", type=int, default=2048)
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--settle", type=float, default=3.0, help="idle seconds sampled after each request")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import glob

    import mlx_bench_worker as W
    path = a.model or glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-MLX-4bit/snapshots/*"))[0]
    wa = SimpleNamespace(stream_experts=None, map_experts=None, attn_bits=None, attn_pack=False, lean=False,
                         map_head=None, wired_mb=None, chunk=2048, slots=128)
    run, mx = (W.make_mj if a.engine == "mj" else W.make_mlxlm)(path, wa)
    c = EnergyCounter()
    t00 = time.perf_counter()
    base = c.snapshot()
    reps = []
    for r in range(a.reps):
        samples, stop = [], threading.Event()

        def sampler():
            while not stop.is_set():
                s = c.snapshot()
                samples.append((time.perf_counter() - t00, gpu_j(c, base, s)))
                _cf.CFRelease(s)
                time.sleep(a.dt)

        th = threading.Thread(target=sampler, daemon=True)
        marks = {}
        th.start()
        time.sleep(1.0)
        marks["start"] = time.perf_counter() - t00
        times = []
        W_first = lambda: (times.append(time.perf_counter() - t00))  # noqa: E731
        run(W.ids_for(a.depth, 100 + r), W.NEW, W_first, W_first)
        marks["first"], marks["last"] = times[0], times[-1]
        time.sleep(a.settle)
        stop.set()
        th.join()
        # energy by phase as the counter shows it
        def at(t):
            best = None
            for ts, j in samples:
                if ts <= t:
                    best = j
            return best
        e = {k: at(v) for k, v in marks.items()}
        e["end"] = samples[-1][1]
        prefill = e["first"] - e["start"]
        decode = e["last"] - e["first"]
        tail = e["end"] - e["last"]
        dec_s = marks["last"] - marks["first"]
        pre_s = marks["first"] - marks["start"]
        print(f"rep {r}: prefill {pre_s:.2f} s {prefill:.1f} J ({prefill / pre_s:.1f} W) | decode {dec_s:.2f} s "
              f"{decode:.1f} J ({decode / dec_s:.1f} W) | +{a.settle:.0f} s idle tail {tail:.1f} J", flush=True)
        reps.append(dict(marks=marks, joules=e, samples=samples))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(dict(args=vars(a), model=path, reps=reps, notes=__doc__), open(a.out, "w"))


if __name__ == "__main__":
    main()
