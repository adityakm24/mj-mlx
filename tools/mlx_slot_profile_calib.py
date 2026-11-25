# Copyright 2026 the MilliJoule authors
# SPDX-License-Identifier: Apache-2.0
"""Can uneven expert slots come from a short calibration at load (checklist 6.2b)? Offline, on a recorded routing
trace (tools/mlx_router_lookahead.py collect; Qwen3.6-35B: 8 documents, 8,192 tokens): per-layer LRU miss curves
from the first N tokens only, the greedy allocation they give (placement.allocate_slots, the same total as S slots
in every layer), and the LRU misses of that allocation on held-out tokens (the second half of the trace), against
96 per layer and against the allocation from the whole first half (what the live A/B of RESULTS 1zy used, roughly).

    .venv/bin/python tools/mlx_slot_profile_calib.py --trace ~/.cache/millijoule/qwen36-route-trace \\
        --store ~/.cache/millijoule/qwen36-35b-experts-stream --out results/slot_profile_calib_v1.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mj_mlx import placement  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--slots", type=int, default=96)
    ap.add_argument("--calib", default="64,128,256,512,1024,2048,4096")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    trace, store = os.path.expanduser(a.trace), os.path.expanduser(a.store)
    idx = json.load(open(os.path.join(store, "index.json")))
    E = idx["num_experts"]
    moe = set(idx.get("moe_layers", range(idx["num_layers"])))
    seqs = {}
    for f in sorted(glob.glob(os.path.join(trace, "r_*.npy"))):
        li = int(os.path.basename(f)[2:-4])
        if li in moe:
            seqs[li] = np.load(f).astype(np.int64)
    T = next(iter(seqs.values())).shape[0]
    half = T // 2
    # held-out misses: stack distances over the whole trace (the cache persists), counted on the second half only
    held = {}
    for li, s in seqs.items():
        d = placement.lru_stack_distances(s)
        n_first = sum(len(dict.fromkeys(int(x) for x in row)) for row in s[:half])
        held[li] = placement.miss_curve(d[n_first:], E)

    def held_misses(alloc):
        return int(sum(held[li][alloc[li] if isinstance(alloc, dict) else alloc] for li in held))

    budget = a.slots * len(seqs)
    res = dict(trace=trace, tokens=T, held_out_tokens=T - half, slots=a.slots, layers=len(seqs),
               uniform_misses=held_misses(a.slots), rows=[])
    for n in [int(x) for x in a.calib.split(",")]:
        n = min(n, half)
        curves = {li: placement.miss_curve(placement.lru_stack_distances(s[:n]), E) for li, s in seqs.items()}
        al = placement.allocate_slots(curves, budget)
        m = held_misses(al)
        res["rows"].append(dict(calib_tokens=n, misses=m, vs_uniform=m / res["uniform_misses"] - 1,
                                slot_range=[min(al.values()), max(al.values())], total=sum(al.values())))
        print(f"calibrate on {n:5d} tokens: held-out misses {m} ({100 * (m / res['uniform_misses'] - 1):+.1f}% vs "
              f"{a.slots} per layer), slots {min(al.values())}-{max(al.values())}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
