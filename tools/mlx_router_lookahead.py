"""Learned router lookahead for streamed MoE prefetch (Qwen3.6-35B-A3B).

collect: run text through the streamed model (prefill, teacher-forced: each token routes as decode would for that
text) and save, per layer, the MoE input (float16) and the routed experts.
train: for each layer j and distance d, fit W [H, E] (initialised from layer j's router, i.e. today's predictor)
so that layer (j-d)'s MoE input predicts layer j's routing: softmax cross-entropy against the k routed experts,
Adam, a few epochs. Report recall@k (share of routed experts in the predicted top-k) on held-out tokens, against
the baseline (layer j's own router on layer j-d's input).

    .venv/bin/python tools/mlx_router_lookahead.py collect --out ~/.cache/millijoule/qwen36-route-trace --tokens 8192
    .venv/bin/python tools/mlx_router_lookahead.py train --data ~/.cache/millijoule/qwen36-route-trace --d 1,2
"""

import argparse
import glob
import json
import os
os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "150")
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
GLOBS = [os.path.join(sys.prefix, "lib/python3*/site-packages/**/*.md"),
         os.path.join(sys.prefix, "lib/python3*/site-packages/**/*.py"), "/usr/share/man/man1/*.1"]


def corpus(tok, n_tokens, seed):
    files = sorted(f for g in GLOBS for f in glob.glob(g, recursive=True) if os.path.getsize(f) > 2000)
    rs = np.random.RandomState(seed)
    rs.shuffle(files)
    out = []
    for f in files:
        try:
            t = open(f, errors="ignore").read()[:6000]
        except OSError:
            continue
        out.append([int(x) for x in tok(t, add_special_tokens=False)["input_ids"]][:1024])
        if sum(len(x) for x in out) >= n_tokens:
            break
    return out


def collect(a):
    import mlx.core as mx
    from transformers import AutoTokenizer

    from mj_mlx import Engine, load_model

    path = glob.glob(os.path.expanduser(a.model))[0]
    mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    model = load_model(path, stream_experts=os.path.expanduser(a.experts), expert_slots=a.slots, lean=True,
                       io_threads=8, io_nocache=True, prefetch=True)
    tok = AutoTokenizer.from_pretrained(path)
    docs = corpus(tok, a.tokens, a.seed)
    out = os.path.expanduser(a.out)
    os.makedirs(out, exist_ok=True)
    nl = len(model.layers)
    X = [[] for _ in range(nl)]
    R = [[] for _ in range(nl)]

    def cb(li, xr, idx):
        X[li].append(np.array(xr.astype(mx.float16)))
        R[li].append(np.array(idx).astype(np.int16))

    model.route_trace = cb
    t = time.perf_counter()
    for i, d in enumerate(docs):
        eng = Engine(model=model, num_blocks=len(d) // 16 + 8, max_running=1, max_batch_tokens=512)
        eng.add_request(d, 1, ignore_eos=True)
        while eng.has_work():
            eng.step()
        print(f"doc {i}: {len(d)} tokens, {time.perf_counter() - t:.0f} s", flush=True)
    model.route_trace = None
    for li in range(nl):
        np.save(f"{out}/x_{li:02d}.npy", np.concatenate(X[li]))
        np.save(f"{out}/r_{li:02d}.npy", np.concatenate(R[li]))
    routers = {li: np.array(model.layers[li]["router_wt"].w_t.astype(mx.float32)) if model.layers[li]["router_wt"].q
               is None else np.array(mx.dequantize(*model.layers[li]["router_wt"].q, group_size=model.layers[li][
                   "router_wt"].gs, bits=model.layers[li]["router_wt"].bits).astype(mx.float32)).T for li in range(nl)}
    np.savez(f"{out}/routers.npz", **{str(k): v for k, v in routers.items()})
    json.dump(dict(model=path, docs=len(docs), tokens=int(sum(len(d) for d in docs)), seed=a.seed),
              open(f"{out}/meta.json", "w"))


def train(a):
    import mlx.core as mx
    import mlx.optimizers as optim

    d_ = os.path.expanduser(a.data)
    rt = np.load(f"{d_}/routers.npz")
    nl = len(rt.files)
    k = np.load(f"{d_}/r_00.npy").shape[1]
    res = {}
    for d in [int(x) for x in a.d.split(",")]:
        base_acc, new_acc, Ws = [], [], {}
        for j in range(d, nl):
            x = np.load(f"{d_}/x_{j - d:02d}.npy").astype(np.float32)
            r = np.load(f"{d_}/r_{j:02d}.npy").astype(np.int64)
            n = len(x)
            ntr = int(n * 0.75)  # first 3/4 of the tokens to train, the rest (later documents) to test
            E = rt[str(j)].shape[1]
            tgt = np.zeros((n, E), np.float32)
            np.put_along_axis(tgt, r, 1.0 / k, axis=1)
            Xtr, Ttr, Xte, Rte = mx.array(x[:ntr]), mx.array(tgt[:ntr]), mx.array(x[ntr:]), r[ntr:]
            W0 = mx.array(rt[str(j)])

            def recall(W):
                top = np.array(mx.argpartition(-(Xte @ W), kth=k - 1, axis=-1)[:, :k])
                return float(np.mean([len(set(a_) & set(b_)) / k for a_, b_ in zip(top, Rte)]))

            params = {"W": W0, "b": mx.zeros((E,))}
            opt = optim.Adam(learning_rate=a.lr)

            def loss_fn(p, X, T):
                lg = X @ p["W"] + p["b"]
                return -(T * (lg - mx.logsumexp(lg, axis=-1, keepdims=True))).sum(axis=-1).mean() + \
                    a.l2 * ((p["W"] - W0) ** 2).sum()

            g = mx.value_and_grad(loss_fn)
            bs = 512
            for ep in range(a.epochs):
                perm = np.random.RandomState(ep).permutation(ntr)
                for s in range(0, ntr, bs):
                    ix = mx.array(perm[s:s + bs])
                    l, gr = g(params, Xtr[ix], Ttr[ix])
                    opt.update(params, gr)
                    mx.eval(params, opt.state)
            b0 = recall(W0)
            top = np.array(mx.argpartition(-(Xte @ params["W"] + params["b"]), kth=k - 1, axis=-1)[:, :k])
            b1 = float(np.mean([len(set(a_) & set(b_)) / k for a_, b_ in zip(top, Rte)]))
            base_acc.append(b0)
            new_acc.append(b1)
            Ws[j] = (np.array(params["W"]).astype(np.float16), np.array(params["b"]).astype(np.float32))
            print(f"d={d} layer {j:2d}: recall@{k} router {b0:.3f} -> learned {b1:.3f}", flush=True)
        res[d] = dict(router=float(np.mean(base_acc)), learned=float(np.mean(new_acc)))
        print(f"d={d}: mean recall@{k} router {res[d]['router']:.3f} -> learned {res[d]['learned']:.3f}", flush=True)
        np.savez(f"{d_}/lookahead_d{d}.npz", **{f"W{j}": w for j, (w, _) in Ws.items()},
                 **{f"b{j}": b for j, (_, b) in Ws.items()})
    json.dump(res, open(f"{d_}/lookahead_result.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--model", default="~/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/*")
    c.add_argument("--experts", default="~/.cache/millijoule/qwen36-35b-experts-stream")
    c.add_argument("--slots", type=int, default=96)
    c.add_argument("--tokens", type=int, default=8192)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--out", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--d", default="1,2")
    t.add_argument("--epochs", type=int, default=4)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--l2", type=float, default=0.0)
    a = ap.parse_args()
    collect(a) if a.cmd == "collect" else train(a)


if __name__ == "__main__":
    main()
