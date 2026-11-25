"""Summarize quiet window 5: median of 3 runs per file (ms/token, GPU mJ/token), output identity vs ours plain."""
import glob
import json
import os
import sys

import numpy as np

Q = os.path.dirname(os.path.abspath(__file__))


def load(sub):
    rows = {}
    for f in sorted(glob.glob(f"{Q}/{sub}/r[0-9]_*.json")):
        n = os.path.basename(f)[:-5]
        rnd, prompt, arm = n.split("_", 2)
        d = json.load(open(f))
        runs = d["runs"]
        rows[(prompt, arm, rnd)] = dict(
            ms=float(np.median([r["decode_ms_per_token"] for r in runs])),
            mj=float(np.median([r["gpu_mj_per_token"] for r in runs])),
            ttft=float(np.median([r["ttft_s"] for r in runs])),
            acc=runs[0].get("accept_rate"), out=[r.get("output_ids") for r in runs],
            pagein=float(np.median([r.get("pageins_mb", 0) for r in runs])))
    return rows


def table(sub, prompts, arms):
    rows = load(sub)
    for p in prompts:
        base = rows.get((p, "plain", "r1"), {}).get("out", [None])[0]
        for a in arms:
            cells = []
            for rnd in ("r1", "r2"):
                v = rows.get((p, a, rnd))
                if not v:
                    cells.append("-")
                    continue
                same = None if base is None or v["out"][0] is None else all(o == base for o in v["out"])
                cells.append(f"{v['ms']:6.2f} ms {v['mj']:4.0f} mJ ttft {v['ttft']:.2f}s pg {v['pagein']:4.0f}MB"
                             + (f" acc {v['acc']:.2f}" if v["acc"] is not None else "")
                             + (f" same {same}" if a != "plain" and same is not None else ""))
            print(f"{sub:7s} {p:6s} {a:12s} " + " || ".join(cells))


if __name__ == "__main__":
    table("gptoss", ["chat", "random"], ["plain", "eagle-auto", "eagle-k2", "llama-server"])
    table("qwen9b", ["chat"], ["mlx-lm", "plain", "mtp"])
    for f in sorted(glob.glob(f"{Q}/machine_state_*.json")):
        s = json.load(open(f))
        print(os.path.basename(f), s["time"], s["loadavg"], s["swap"].split("used = ")[1].split()[0], s["top_cpu"][:3])
