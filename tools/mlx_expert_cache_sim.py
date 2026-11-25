"""Offline expert-cache policy simulation on a recorded routing trace (tools/mlx_router_lookahead.py collect).

Per layer, a cache of S expert slots; each token routes to k experts, all needed at once (the token's own experts
are never evicted to make room for each other). Counts demand misses only (no prefetch), so it measures the
policy's headroom, not end-to-end speed. Policies:
  lru            the streamer's policy today
  lfu:<decay>    evict the lowest exponentially decayed use count (score *= decay per token; 1.0 = pure LFU)
  opt            Belady's optimum (evict the expert used furthest in the future): the bound for any policy

    .venv/bin/python tools/mlx_expert_cache_sim.py --trace ~/.cache/millijoule/qwen36-route-trace --slots 96,128
"""

import argparse
import glob
import heapq
import json
import os

import numpy as np


def sim_lru_fast(seq, S):
    from collections import OrderedDict
    od, miss = OrderedDict(), 0
    for row in seq:
        rs = set(int(x) for x in row)
        for e in rs:
            if e in od:
                od.move_to_end(e)
        for e in rs:
            if e not in od:
                miss += 1
                while len(od) >= S:
                    for x in od:  # oldest not in this token
                        if x not in rs:
                            del od[x]
                            break
                od[e] = None
    return miss


def sim_lfu(seq, S, decay, E):
    score = np.zeros(E)
    res = np.zeros(E, bool)
    n, miss = 0, 0
    for row in seq:
        score *= decay
        rs = np.unique(row)
        score[rs] += 1.0
        new = rs[~res[rs]]
        miss += len(new)
        need = n + len(new) - S
        if need > 0:
            sc = np.where(res, score, np.inf)
            sc[rs] = np.inf
            ev = np.argpartition(sc, need - 1)[:need]
            res[ev] = False
            n -= need
        res[new] = True
        n += len(new)
    return miss


def sim_opt(seq, S, E):
    """Belady: on a miss, evict the resident expert (not routed by this token) whose next use is furthest away."""
    T = len(seq)
    rows = [np.unique(r) for r in seq]
    after = [None] * T  # after[t][e]: next time e is used after t
    nexts = np.full(E, T + 1, np.int64)
    for t in range(T - 1, -1, -1):
        after[t] = {int(e): int(nexts[e]) for e in rows[t]}
        nexts[rows[t]] = t
    res, heap, miss = {}, [], 0  # res: expert -> its next use; heap: (-next use, expert), lazily invalidated
    for t in range(T):
        cur = after[t]
        for e in cur:
            if e in res:
                continue
            miss += 1
            if len(res) >= S:
                keep = []
                while True:
                    nu, x = heapq.heappop(heap)
                    if res.get(x) != -nu:
                        continue  # stale entry
                    if x in cur:
                        keep.append((nu, x))
                        continue
                    del res[x]
                    break
                for k in keep:
                    heapq.heappush(heap, k)
            res[e] = -1  # placeholder until updated below
        for e, nu in cur.items():
            res[e] = nu
            heapq.heappush(heap, (-nu, e))
    return miss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default="~/.cache/millijoule/qwen36-route-trace")
    ap.add_argument("--slots", default="96,128")
    ap.add_argument("--experts", type=int, default=256)
    ap.add_argument("--policies", default="lru,lfu:1.0,lfu:0.999,lfu:0.995,lfu:0.99,lfu:0.98,opt")
    ap.add_argument("--layers", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    d = os.path.expanduser(a.trace)
    files = sorted(glob.glob(os.path.join(d, "r_*.npy")))
    if a.layers:
        keep = {int(x) for x in a.layers.split(",")}
        files = [f for f in files if int(os.path.basename(f)[2:4]) in keep]
    res = {}
    for S in [int(x) for x in a.slots.split(",")]:
        for pol in a.policies.split(","):
            tot = acc = 0
            for f in files:
                seq = np.load(f).astype(np.int64)
                acc += seq.size
                if pol == "lru":
                    m = sim_lru_fast(seq, S)
                elif pol.startswith("lfu:"):
                    m = sim_lfu(seq, S, float(pol[4:]), a.experts)
                elif pol == "opt":
                    m = sim_opt(seq, S, a.experts)
                tot += m
            res[f"{S}/{pol}"] = dict(slots=S, policy=pol, misses=tot, accesses=acc, miss_rate=tot / acc,
                                     misses_per_token=tot / (acc / 8))
            print(f"slots {S:4d} {pol:12s} miss rate {tot / acc:.4f}  misses/token {tot / (acc / 8):.2f}", flush=True)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(dict(trace=d, layers=len(files), results=res, notes=__doc__), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
