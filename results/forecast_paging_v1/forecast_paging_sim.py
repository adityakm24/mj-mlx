"""Forecast paging, offline study part 1 (checklist 6.1): how far ahead must a routing forecast see to help eviction?

On the recorded routing trace (per layer: [T, k] expert ids per token), a cache of S expert slots per layer, demand
misses only (as tools/mlx_expert_cache_sim.py). Policies:
  lru          the streamer today
  opt          Belady's optimum (knows all future routing)
  hor:<h>      horizon-limited oracle: the pager knows the true routing of the next h tokens in this layer. On a miss it
               evicts the least recently used expert that is NOT needed within the horizon; if every resident expert is
               needed within the horizon, the one needed furthest away. h = 0 is LRU; h = infinity is opt.
  noisy:<h>:<recall>:<false>   the same with an imperfect forecast: each truly needed expert in the horizon is seen with
               probability <recall>; per token, round(<false> * k) experts that are not needed are wrongly protected.
A forecast of h tokens is what a draft of h tokens verified in the same step gives (their routing is computed with the
step), or a predictor run on draft tokens. Layer-ahead prediction does not change eviction (each layer has its own
cache); it changes when a read can be issued, which part 2 measures on the live streamer.

    python forecast_paging_sim.py --trace ~/.cache/millijoule/qwen36-route-trace --slots 96 --out out.json
"""

import argparse
import glob
import json
import os
from collections import OrderedDict

import numpy as np


def next_use_table(rows, E):
    """after[t]: dict expert -> next token index (> t) at which it is routed, for experts routed at t; and
    nxt[t][e] for any e via a per-expert sorted list of use times."""
    uses = [[] for _ in range(E)]
    for t, r in enumerate(rows):
        for e in r:
            uses[int(e)].append(t)
    return [np.asarray(u, np.int64) for u in uses]


def sim_horizon(seq, S, E, h, recall=1.0, false=0.0, rng=None):
    rows = [np.unique(r) for r in seq]
    T = len(rows)
    uses = next_use_table(rows, E)
    ptr = np.zeros(E, np.int64)  # index into uses[e] of the first use > current t
    lru = OrderedDict()  # resident experts, least recently used first
    miss = 0
    k = seq.shape[1]
    for t in range(T):
        cur = set(int(e) for e in rows[t])
        for e in cur:
            u = uses[e]
            while ptr[e] < len(u) and u[ptr[e]] <= t:
                ptr[e] += 1
        # forecast for this token's evictions: experts needed in (t, t + h]
        seen = {}
        if h > 0:
            for e in lru:
                if e in cur:
                    continue
                u = uses[e]
                p = ptr[e]
                while p < len(u) and u[p] <= t:
                    p += 1
                ptr[e] = p
                if p < len(u) and u[p] <= t + h:
                    if recall >= 1.0 or rng.random() < recall:
                        seen[e] = int(u[p])
            if false > 0:
                cand = [e for e in lru if e not in cur and e not in seen]
                nf = min(len(cand), int(round(false * k)))
                if nf:
                    for e in rng.choice(cand, nf, replace=False):
                        seen[int(e)] = t + 1 + int(rng.integers(0, h))
        for e in cur:
            if e in lru:
                lru.move_to_end(e)
        for e in cur:
            if e in lru:
                continue
            miss += 1
            if len(lru) >= S:
                victim = None
                for x in lru:  # oldest first
                    if x not in cur and x not in seen:
                        victim = x
                        break
                if victim is None:  # everything is forecast as needed: drop the one needed last
                    victim = max((x for x in lru if x not in cur), key=lambda x: seen.get(x, 1 << 60))
                del lru[victim]
                seen.pop(victim, None)
            lru[e] = None
    return miss


def sim_opt(seq, S, E):
    return sim_horizon(seq, S, E, h=1 << 40)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default="~/.cache/millijoule/qwen36-route-trace")
    ap.add_argument("--slots", default="96")
    ap.add_argument("--experts", type=int, default=256)
    ap.add_argument("--policies", default="hor:0,hor:1,hor:2,hor:3,hor:4,hor:8,hor:16,hor:64,hor:256,opt,"
                                          "noisy:2:0.9:0,noisy:2:0.8:0,noisy:4:0.9:0,noisy:4:0.8:0,noisy:4:0.9:0.25,"
                                          "noisy:8:0.9:0,noisy:8:0.8:0.25")
    ap.add_argument("--layers", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    d = os.path.expanduser(a.trace)
    files = sorted(glob.glob(os.path.join(d, "r_*.npy")))
    if a.layers:
        keep = {int(x) for x in a.layers.split(",")}
        files = [f for f in files if int(os.path.basename(f)[2:4]) in keep]
    seqs = [np.load(f).astype(np.int64) for f in files]
    res = {}
    for S in [int(x) for x in a.slots.split(",")]:
        for pol in a.policies.split(","):
            tot = acc = 0
            for i, seq in enumerate(seqs):
                acc += seq.size
                rng = np.random.default_rng(1000 + i)
                if pol == "opt":
                    tot += sim_opt(seq, S, a.experts)
                elif pol.startswith("hor:"):
                    tot += sim_horizon(seq, S, a.experts, int(pol[4:]), rng=rng)
                elif pol.startswith("noisy:"):
                    _, h, r, f = pol.split(":")
                    tot += sim_horizon(seq, S, a.experts, int(h), float(r), float(f), rng)
            k = seqs[0].shape[1]
            res[f"{S}/{pol}"] = dict(slots=S, policy=pol, misses=tot, accesses=acc, miss_rate=tot / acc,
                                     misses_per_token=tot / (acc / k))
            print(f"slots {S:4d} {pol:22s} miss rate {tot / acc:.4f}  misses/token {tot / (acc / k):.2f}", flush=True)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(dict(trace=d, layers=len(files), tokens=int(seqs[0].shape[0]), results=res, notes=__doc__),
                  open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
