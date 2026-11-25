"""Candidate-head miss rate on the REAL gpt-oss-20b weights, on the model's own greedy decode.

Lean decode picks the greedy token among the top-k of a low-bit copy of lm_head, rescored exactly from the bf16 rows
(GptOss.rescore_host). A miss (the exact argmax over the full bf16 head is not among the candidates) is the only way
that changes a greedy token. This runs real decodes (Engine, lean, overlap), records every final hidden state, and
for each one computes:
  - the exact argmax: float32 logits of all 201088 bf16 lm_head rows (host, chunked, from the safetensors memmap);
  - the top-k of 2-, 3- and 4-bit (group 64) copies of lm_head (GPU), for several k.
Also times one candidate matmul per width (1 row, GPU), so the trade can be read off.

    .venv/bin/python tools/mlx_gptoss_head_miss_real.py --out results/gptoss_head_miss_real.json
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
sys.path.insert(0, os.path.dirname(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=int, default=16)
    ap.add_argument("--new", type=int, default=128)
    ap.add_argument("--bits", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--k", type=int, nargs="+", default=[1, 4, 16, 32, 64])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import mlx.core as mx

    from mj_mlx import Engine, load_model
    from mj_mlx.expert_stream import SafetensorsRows
    from mj_mlx.gpt_oss import topk_indices
    from mlx_bench_worker import ids_for

    HF = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/*"))[0]
    MAP = os.path.expanduser("~/.cache/millijoule/gptoss20b-experts-mapped")
    m = load_model(HF, map_experts=MAP, attn_pack=True, lean=True)
    mx.set_cache_limit(512 << 20)
    xs = []
    real = m.rescore_host

    def rec(cand, x32):
        xs.append(np.array(x32, np.float32))
        return real(cand, x32)
    m.rescore_host = rec
    t0 = time.time()
    for i in range(a.prompts):
        # half random-token prompts (as the suite), half from earlier decodes (natural text)
        eng = Engine(model=m, num_blocks=64, max_running=1, max_batch_tokens=512, overlap=True,
                     wired_limit_bytes=3000 << 20)
        eng.add_request(ids_for(1 + 31 * (i % 2), 900 + i), a.new, ignore_eos=True)
        while eng.has_work():
            eng.step()
        del eng
    X = np.concatenate(xs, axis=0)  # [N, H]
    print(f"{len(X)} hidden states in {time.time() - t0:.0f} s", flush=True)
    rows = SafetensorsRows(HF, "lm_head.weight")
    V = rows.rows.shape[0]
    # exact argmax, host float32, chunked over the vocabulary
    best = np.full(len(X), -np.inf, np.float64)
    arg = np.zeros(len(X), np.int64)
    step = 16384
    for lo in range(0, V, step):
        w = (np.asarray(rows.rows[lo:lo + step]).astype(np.uint32) << 16).view(np.float32)
        lg = X @ w.T  # [N, step]
        j = lg.argmax(axis=1)
        v = lg[np.arange(len(X)), j]
        upd = v > best
        best[upd] = v[upd]
        arg[upd] = lo + j[upd]
    print("exact argmax done", flush=True)
    Xg = mx.array(X).astype(mx.bfloat16)
    res = {}
    for bits in a.bits:
        miss = {k: 0 for k in a.k}
        q = None
        # quantize from the bf16 head in vocabulary chunks (keeps the bf16 temporary small)
        parts = []
        for lo in range(0, V, 32768):
            wb = mx.array(np.asarray(rows.rows[lo:lo + 32768])).view(mx.bfloat16)
            parts.append(mx.quantize(wb, 64, bits))
            mx.eval(parts[-1])
        wq = mx.concatenate([p[0] for p in parts]); sc = mx.concatenate([p[1] for p in parts])
        bi = mx.concatenate([p[2] for p in parts])
        mx.eval(wq, sc, bi)
        del parts
        for lo in range(0, len(X), 256):
            lg = mx.quantized_matmul(Xg[lo:lo + 256], wq, sc, bi, transpose=True, group_size=64, bits=bits)
            for k in a.k:
                c = np.array(topk_indices(lg, k))
                miss[k] += int((c != arg[lo:lo + 256, None]).all(axis=1).sum())
        x1 = Xg[:1]
        for _ in range(5):
            mx.eval(mx.quantized_matmul(x1, wq, sc, bi, transpose=True, group_size=64, bits=bits))
        ts = []
        for _ in range(20):
            t = time.perf_counter()
            mx.eval(mx.quantized_matmul(x1, wq, sc, bi, transpose=True, group_size=64, bits=bits))
            ts.append((time.perf_counter() - t) * 1e3)
        nbytes = wq.nbytes + sc.nbytes + bi.nbytes
        res[bits] = dict(miss={str(k): miss[k] for k in a.k}, n=len(X), head_bytes=nbytes,
                         matmul_ms_median=float(np.median(ts)), matmul_ms_min=float(min(ts)))
        print(f"{bits}-bit: {nbytes / 1e6:.0f} MB, 1-row matmul {np.median(ts):.2f} ms (min {min(ts):.2f}); misses "
              + ", ".join(f"top-{k} {miss[k]}/{len(X)}" for k in a.k), flush=True)
        del wq, sc, bi
        mx.clear_cache()
    json.dump(dict(args=vars(a), results=res, notes=__doc__), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
