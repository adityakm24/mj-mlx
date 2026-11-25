"""Uneven expert slots per layer, offline (checklist follow-up from 2026-11-22): same total memory, fewer misses?

The streamer gives every MoE layer the same slot count S (placement.stream_slots). Layers differ in how concentrated
their routing is, so moving slots from predictable layers to spread-out ones could lower total misses at the same
memory (every Qwen3.6 layer's expert has the same byte size, so slots are the memory unit).

Method: on the recorded routing trace (per layer [T, k] expert ids; 8 documents of 1,024 tokens), LRU misses for every
cache size at once from LRU stack distances (Mattson et al. 1970: LRU has the inclusion property, so an access misses
in a cache of S slots iff its stack distance > S). Within a token the k experts are touched in order; this matches the
streamer for S >= k (it protects a token's own experts, which only matters below k slots). Demand misses only, no
prefetch, as tools/mlx_expert_cache_sim.py and RESULTS 1zn.

Allocation: start every layer at a floor, then hand out slots in steps of `step` to the layer whose miss count falls
most per slot (greedy on marginal gain), until the total equals the uniform budget. Fitted on one half of the
documents and scored on the other half (both ways), so the gain is not read off the data it was fitted to.

    python slot_allocation_sim.py --trace ~/.cache/millijoule/qwen36-route-trace --slots 96 --out out.json
"""

import argparse
import glob
import json
import os

import numpy as np


def stack_distances(seq: np.ndarray) -> np.ndarray:
    """LRU stack distance (1 = most recent) of every access in token order; 10**9 for a first access."""
    order = []  # most recent first
    out = []
    for row in seq:
        for e in dict.fromkeys(int(x) for x in row):  # unique experts of this token, in routing order
            try:
                d = order.index(e)
                out.append(d + 1)
                del order[d]
            except ValueError:
                out.append(10 ** 9)
            order.insert(0, e)
    return np.asarray(out, np.int64)


def miss_curve(dist: np.ndarray, E: int) -> np.ndarray:
    """misses[S] for S = 0 .. E: accesses with stack distance > S."""
    h = np.bincount(np.minimum(dist, E + 1), minlength=E + 2)  # E + 1 bucket = cold / never within E
    ge = np.cumsum(h[::-1])[::-1]  # ge[d] = count with distance >= d
    return np.array([ge[S + 1] if S + 1 < len(ge) else 0 for S in range(E + 1)], np.int64)


def allocate(curves: list[np.ndarray], budget: int, floor: int, step: int, cap: int) -> list[int]:
    alloc = [floor] * len(curves)
    left = budget - floor * len(curves)
    while left >= step:
        gains = [(c[a] - c[min(cap, a + step)]) if a + step <= cap else -1 for c, a in zip(curves, alloc)]
        i = int(np.argmax(gains))
        if gains[i] <= 0:
            break
        alloc[i] += step
        left -= step
    return alloc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default="~/.cache/millijoule/qwen36-route-trace")
    ap.add_argument("--slots", default="64,96,128")
    ap.add_argument("--experts", type=int, default=256)
    ap.add_argument("--docs", type=int, default=8)
    ap.add_argument("--floor", type=int, default=16)
    ap.add_argument("--step", type=int, default=4)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    d = os.path.expanduser(a.trace)
    files = sorted(glob.glob(os.path.join(d, "r_*.npy")))
    seqs = [np.load(f).astype(np.int64) for f in files]
    T, k = seqs[0].shape
    per = T // a.docs
    halves = {"A": range(0, a.docs // 2), "B": range(a.docs // 2, a.docs)}
    # one LRU per layer runs over the whole trace (the cache is not emptied between documents, as in the live engine);
    # each access is credited to its document's half
    curves = {h: [] for h in halves}
    whole = []
    for seq in seqs:
        dist = stack_distances(seq)
        doc_of = np.concatenate([np.full(len(dict.fromkeys(int(x) for x in row)), t // per)
                                 for t, row in enumerate(seq)])
        whole.append(miss_curve(dist, a.experts))
        for h, docs in halves.items():
            curves[h].append(miss_curve(dist[np.isin(doc_of, list(docs))], a.experts))
    acc = {h: sum(int(np.isin(np.arange(T) // per, list(docs)).sum()) * k for _ in seqs) for h, docs in halves.items()}
    res = {}
    for S in [int(x) for x in a.slots.split(",")]:
        budget = S * len(seqs)
        row = dict(slots=S, layers=len(seqs), budget=budget)
        uni_all = sum(int(c[S]) for c in whole)
        row["uniform_miss_rate_all"] = uni_all / (T * k * len(seqs))
        for fit, score in (("A", "B"), ("B", "A")):
            alloc = allocate(curves[fit], budget, a.floor, a.step, a.experts)
            uni = sum(int(c[S]) for c in curves[score])
            opt = sum(int(c[s]) for c, s in zip(curves[score], alloc))
            row[f"fit_{fit}_score_{score}"] = dict(uniform_misses=uni, uneven_misses=opt,
                                                   uniform_miss_rate=uni / acc[score], uneven_miss_rate=opt / acc[score],
                                                   reduction=1 - opt / uni, alloc=alloc,
                                                   alloc_min=min(alloc), alloc_max=max(alloc))
        alloc = allocate(whole, budget, a.floor, a.step, a.experts)  # in-sample, for reference only
        row["in_sample"] = dict(uneven_miss_rate=sum(int(c[s]) for c, s in zip(whole, alloc)) / (T * k * len(seqs)),
                                alloc=alloc)
        res[str(S)] = row
        f1, f2 = row["fit_A_score_B"], row["fit_B_score_A"]
        print(f"S={S}: uniform {row['uniform_miss_rate_all']:.4f} (all) | fit A->B {f1['uniform_miss_rate']:.4f} -> "
              f"{f1['uneven_miss_rate']:.4f} ({f1['reduction']:+.1%} fewer), alloc {f1['alloc_min']}-{f1['alloc_max']} | "
              f"fit B->A {f2['uniform_miss_rate']:.4f} -> {f2['uneven_miss_rate']:.4f} ({f2['reduction']:+.1%}) | "
              f"in-sample {row['in_sample']['uneven_miss_rate']:.4f}", flush=True)
    if a.out:
        json.dump(dict(trace=d, layers=len(seqs), tokens=T, k=k, docs=a.docs, floor=a.floor, step=a.step,
                       results=res, notes=__doc__), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
