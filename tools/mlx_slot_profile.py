"""Write a slot profile next to an expert store: per-layer LRU miss curves from a recorded routing trace.

With a profile, `placement.uneven_slots(store, S)` (and load_model(..., placement=..., MJ_SLOT_PROFILE=1), or the bench
worker's --slot-profile) gives each MoE layer its own slot count with the same total as S in every layer, so layers
whose routing is spread out get more slots (RESULTS 1zt: about 7% fewer misses at 96 average slots on Qwen3.6,
offline). The trace is a directory of r_XX.npy files, layer XX's routed expert ids [T, k] in token order, as written
by tools/mlx_router_lookahead.py.

    .venv/bin/python tools/mlx_slot_profile.py --trace ~/.cache/millijoule/qwen36-route-trace \\
        --store ~/.cache/millijoule/qwen36-35b-experts-stream
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
    a = ap.parse_args()
    trace, store = os.path.expanduser(a.trace), os.path.expanduser(a.store)
    idx = json.load(open(os.path.join(store, "index.json")))
    E = idx["num_experts"]
    moe = set(idx.get("moe_layers", range(idx["num_layers"])))
    curves, T = {}, None
    for f in sorted(glob.glob(os.path.join(trace, "r_*.npy"))):
        li = int(os.path.basename(f)[2:-4])
        if li not in moe:
            continue
        seq = np.load(f).astype(np.int64)
        T = seq.shape[0]
        curves[li] = placement.miss_curve(placement.lru_stack_distances(seq), E).tolist()
    missing = sorted(moe - set(curves))
    assert not missing, f"trace has no routing for MoE layers {missing}"
    meta = {}
    mf = os.path.join(trace, "meta.json")
    if os.path.exists(mf):
        meta = json.load(open(mf))
    out = dict(trace=trace, trace_meta=meta, tokens=T, num_experts=E, source=idx.get("source"), curves=curves)
    json.dump(out, open(os.path.join(store, placement.PROFILE), "w"))
    for S in (64, 96, 128):
        al = placement.uneven_slots(store, S)
        print(f"avg {S}: per-layer {min(al.values())}-{max(al.values())}, total {sum(al.values())} of {S * len(al)}")
    print("wrote", os.path.join(store, placement.PROFILE))


if __name__ == "__main__":
    main()
