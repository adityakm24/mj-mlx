"""Where a cold start's time goes (checklist 0.11): one fresh process per call, wall-clock seconds for each phase.

Phases: import (mlx + mj_mlx), load_model, Engine construction, first token of a 128-token prompt, then the next 32
tokens (to see whether background work such as MJ_MLOCK=bg slows the first steps). Weights are assumed to be in the
page cache, as in tools/mlx_cold_start.py; one JSON line per call is appended to --out.

    MJ_MLOCK=1  .venv/bin/python tools/mlx_load_profile.py --model <gpt-oss dir> --mj-kw '{...}' --label mlock --out x.jsonl
    MJ_MLOCK=bg .venv/bin/python tools/mlx_load_profile.py ... --label mlock-bg --out x.jsonl
"""

from __future__ import annotations

import time

T0 = time.perf_counter()

import argparse  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402

os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "150")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mj-kw", default="{}")
    ap.add_argument("--placement", default="none")
    ap.add_argument("--label", default="")
    ap.add_argument("--materialize", action="store_true",
                    help="evaluate every model array right after load_model and time it as its own phase (is the "
                         "Engine's construction time lazy weight preparation, such as attention quantizing?)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t = {}
    import mlx.core as mx
    import numpy as np

    from mj_mlx import load_model
    from mj_mlx.runner import Engine

    t["import"] = time.perf_counter() - T0
    path = glob.glob(os.path.expanduser(a.model))[0]
    kw = json.loads(a.mj_kw)
    if a.placement != "none":
        kw["placement"] = a.placement
    if kw.get("wired_mb"):
        mx.set_wired_limit(int(kw.pop("wired_mb")) << 20)
    s = time.perf_counter()
    model = load_model(path, **kw)
    t["load_model"] = time.perf_counter() - s
    if a.materialize:
        from mlx.utils import tree_flatten

        s = time.perf_counter()
        arrs = [v for _, v in tree_flatten(model.parameters()) if isinstance(v, mx.array)]
        mx.eval(*arrs)
        t["materialize"] = time.perf_counter() - s
    s = time.perf_counter()
    eng = Engine(model=model, num_blocks=64, max_running=1, max_batch_tokens=512, overlap=True)
    t["engine"] = time.perf_counter() - s
    ids = [int(x) for x in np.random.RandomState(0).randint(1000, 30000, 128)]
    s = time.perf_counter()
    eng.add_request(ids, 33)
    stamps, n = [], 0
    while eng.has_work():
        for f in eng.step():
            if len(f[2]) > n:
                n = len(f[2])
                stamps.append(time.perf_counter())
    t["first_token"] = stamps[0] - s
    t["next_32_tokens"] = stamps[-1] - stamps[0] if len(stamps) > 1 else None
    t["total_to_first_token"] = stamps[0] - T0
    m = getattr(model, "map", None)
    row = dict(label=a.label, materialize=a.materialize, model=path, mj_kw=a.mj_kw, env={k: v for k, v in os.environ.items()
                                                              if k.startswith(("MJ_", "MLX_"))},
               seconds=t, locked_gb=(m.locked / 1e9) if m is not None else None,
               lock_thread_alive=bool(m is not None and m.lock_thread is not None and m.lock_thread.is_alive()),
               load_stages=getattr(model, "load_times", None) or None,  # MJ_LOAD_TIMING=1 (gpt-oss)
               time=time.strftime("%Y-%m-%dT%H:%M:%S"))
    with open(a.out, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row["seconds"]), row["locked_gb"])
    if row["load_stages"]:
        print("load stages:", json.dumps({k: round(v, 3) for k, v in row["load_stages"].items()}))


if __name__ == "__main__":
    main()
