"""Summarize ABAB benchmark JSONs (tools/mlx_bench.py) into a markdown table.

    .venv/bin/python tools/mlx_report.py results/abab_r*_a_*.json -- results/abab_r*_b_*.json
Arms before `--` are A (baseline), after are B. Per level: mean of rounds, and the A/A and B/B spread
(max-min over rounds, as % of the mean), so a delta can be judged against run-to-run noise.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict


def load(paths):
    by_level = defaultdict(list)
    meta = None
    for p in paths:
        d = json.load(open(p))
        meta = meta or d
        for r in d["rows"]:
            by_level[r["level"]].append(r)
    return by_level, meta


def agg(rows, key):
    v = [r[key] for r in rows]
    m = sum(v) / len(v)
    return m, (max(v) - min(v)) / m * 100 if m else 0.0


def main():
    i = sys.argv.index("--")
    a, am = load(sys.argv[1:i])
    b, bm = load(sys.argv[i + 1:])
    print(f"A = {am['engine']}, B = {bm['engine']} ({bm.get('label', '')}); {am['model_note']}")
    print(f"Energy: {am['energy_boundary']}. Rounds: A {len(sys.argv[1:i])}, B {len(sys.argv[i + 1:])}.\n")
    print("| level | A tok/s (spread) | B tok/s (spread) | speed B/A | A mJ/token | B mJ/token | energy B/A | "
          "A W | B W |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for lv in sorted(a):
        at, asp = agg(a[lv], "tok_s")
        bt, bsp = agg(b[lv], "tok_s")
        aj, ajs = agg(a[lv], "gpu_j_per_token")
        bj, bjs = agg(b[lv], "gpu_j_per_token")
        aw, _ = agg(a[lv], "gpu_w")
        bw, _ = agg(b[lv], "gpu_w")
        print(f"| {lv} | {at:.1f} ({asp:.0f}%) | {bt:.1f} ({bsp:.0f}%) | {bt / at:.2f}x | {aj * 1e3:.1f} ({ajs:.0f}%) | "
              f"{bj * 1e3:.1f} ({bjs:.0f}%) | {bj / aj:.2f}x | {aw:.1f} | {bw:.1f} |")


if __name__ == "__main__":
    main()
