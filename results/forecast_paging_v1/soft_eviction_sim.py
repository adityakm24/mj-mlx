"""Router-score eviction, offline (checklist 6.1 follow-up): can the router's own scores for experts it did NOT pick
predict what will be needed soon, better than recency or use counts?

Per layer, a cache of S expert slots, demand misses only (as tools/mlx_expert_cache_sim.py). Every policy here is causal
(uses only tokens already seen), exact (routing unchanged) and free at run time (the router logits are computed anyway).
Score per expert, decayed per token (score *= decay), plus a per-token contribution; evict the lowest score among
residents the current token does not route to.
  count          1 for each routed expert (= LFU with decay, the earlier study's best: 10.3-10.7% at 96 slots)
  prob:<temp>    softmax(logits / temp) over all experts
  near:<n>:<w>   1 for routed experts, w for the next n - k experts by logit (near misses)
  both:<temp>:<a>   routed count + a * softmax(logits / temp)
Reference points at 96 slots: LRU 11.08%, horizon oracle 16 / 64 tokens 8.70% / 6.23%, optimum 5.22%.

    VECLIB_MAXIMUM_THREADS=1 python soft_eviction_sim.py --slots 96 --out out.json
"""
import argparse, glob, json, os
import numpy as np


def sim(seq, contrib, S, decay, E):
    score = np.zeros(E)
    res = np.zeros(E, bool)
    n = miss = 0
    for t in range(len(seq)):
        score *= decay
        score += contrib[t]
        rs = np.unique(seq[t])
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


def contribution(kind, seq, logits, E):
    T, k = seq.shape
    c = np.zeros((T, E), np.float32)
    rows = np.arange(T)[:, None]
    p = kind.split(":")
    if p[0] in ("count", "near", "both"):
        c[rows, seq] = 1.0
    if p[0] == "near":
        n, w = int(p[1]), float(p[2])
        order = np.argsort(-logits, axis=1)[:, k:n]
        c[rows, order] = np.maximum(c[rows, order], w)
    if p[0] in ("prob", "both"):
        temp = float(p[1])
        z = logits / temp
        z = z - z.max(1, keepdims=True)
        pr = np.exp(z)
        pr /= pr.sum(1, keepdims=True)
        c = pr if p[0] == "prob" else c + float(p[2]) * pr
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default="~/.cache/millijoule/qwen36-route-trace")
    ap.add_argument("--slots", default="96")
    ap.add_argument("--experts", type=int, default=256)
    ap.add_argument("--decays", default="0.98,0.99")
    ap.add_argument("--kinds", default="count,prob:1,prob:2,prob:4,near:16:0.5,near:32:0.5,near:32:0.25,near:64:0.25,both:1:4,both:2:8,both:4:8")
    ap.add_argument("--layers", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    d = os.path.expanduser(a.trace)
    R = np.load(os.path.join(d, "routers.npz"))
    layers = sorted(int(os.path.basename(f)[2:4]) for f in glob.glob(os.path.join(d, "r_*.npy")))
    if a.layers:
        layers = [l for l in layers if l in {int(x) for x in a.layers.split(",")}]
    data = []
    for l in layers:
        seq = np.load(os.path.join(d, f"r_{l:02d}.npy")).astype(np.int64)
        x = np.load(os.path.join(d, f"x_{l:02d}.npy")).astype(np.float32)
        data.append((seq, x @ R[str(l)].astype(np.float32)))
    res = {}
    for S in [int(s) for s in a.slots.split(",")]:
        for kind in a.kinds.split(","):
            cs = [contribution(kind, seq, lg, a.experts) for seq, lg in data]
            for decay in [float(x) for x in a.decays.split(",")]:
                tot = sum(sim(seq, c, S, decay, a.experts) for (seq, _), c in zip(data, cs))
                acc = sum(seq.size for seq, _ in data)
                res[f"{S}/{kind}/{decay}"] = dict(slots=S, kind=kind, decay=decay, misses=tot, miss_rate=tot / acc)
                print(f"slots {S} {kind:14s} decay {decay}: miss rate {tot / acc:.4f}", flush=True)
    if a.out:
        json.dump(dict(trace=d, layers=len(layers), results=res, notes=__doc__), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
