"""Where a cold start's time goes, on exactly the path tools/mlx_cold_start.py times (checklist 0.11): a fresh process
builds the suite variant's model through mlx_bench_worker.make_mj (same arguments, same environment) and serves one
128-token prompt for one token. The variant's prewarm files are read first, outside the timer, as the cold-start
tool does. Phases are wall-clock seconds from the process start; with --cprofile, the 40 functions with the most
cumulative time in the load and in the first request are written next to --out.

    .venv/bin/python tools/mlx_cold_profile.py --variant gptoss:mj-a8w --label a8w --out results/x.jsonl [--cprofile]
"""

from __future__ import annotations

import time

T0 = time.perf_counter()

import argparse  # noqa: E402
import glob  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "150")
sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, help="suite model:engine, e.g. gptoss:mj-a8w")
    ap.add_argument("--label", default="")
    ap.add_argument("--cprofile", action="store_true")
    ap.add_argument("--no-prewarm", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from mlx_bench_suite import MODELS

    key, eng = a.variant.split(":")
    m = MODELS[key]
    args = m.get("variant_args", {}).get(eng, m["mj_args"])
    env = m.get("variant_env", {}).get(eng, m.get("mj_env", {}))
    for k, v in env.items():  # the variant's environment, unless the caller set it (A/B runs)
        os.environ.setdefault(k, v)
    path = glob.glob(os.path.expanduser(m["mlx"]))[0]
    t = {}
    if not a.no_prewarm:
        s = time.perf_counter()
        for pat in m.get("prewarm", {}).get(eng, []):
            for f in sorted(glob.glob(os.path.expanduser(pat))):
                with open(f, "rb") as fh:
                    while fh.read(64 << 20):
                        pass
        t["prewarm_untimed"] = time.perf_counter() - s
    s = time.perf_counter()
    import mlx_bench_worker as w  # mlx, mj_mlx

    t["import"] = time.perf_counter() - s
    wa = w.parser().parse_args(["--engine", "mj", "--model", path, "--out", os.devnull, *args])
    wa.slots_set = "--slots" in args
    prof = {}

    def timed(name, fn):
        if not a.cprofile:
            s = time.perf_counter()
            r = fn()
            t[name] = time.perf_counter() - s
            return r
        import cProfile
        import pstats

        pr = cProfile.Profile()
        s = time.perf_counter()
        r = pr.runcall(fn)
        t[name] = time.perf_counter() - s
        buf = io.StringIO()
        pstats.Stats(pr, stream=buf).sort_stats("cumulative").print_stats(40)
        prof[name] = buf.getvalue()
        return r

    run, mx = timed("load", lambda: w.make_mj(path, wa))
    first = {}
    timed("first_request", lambda: run(w.ids_for(128, 12345), 1, lambda: first.setdefault("t", time.perf_counter()),
                                       lambda: None))
    t["to_first_token_from_import"] = first.get("t", time.perf_counter()) - (T0 + t.get("prewarm_untimed", 0))
    row = dict(label=a.label, variant=a.variant, args=args, env={k: v for k, v in os.environ.items()
                                                                 if k.startswith(("MJ_", "MLX_"))},
               seconds=t, locked_gb=getattr(wa, "locked_gb", None), time=time.strftime("%Y-%m-%dT%H:%M:%S"))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(row) + "\n")
    if prof:
        base = os.path.splitext(a.out)[0] + f"_{a.label}_{time.strftime('%H%M%S')}"
        for k, v in prof.items():
            open(f"{base}_{k}.prof.txt", "w").write(v)
    print(json.dumps(t), row["locked_gb"])


if __name__ == "__main__":
    main()
