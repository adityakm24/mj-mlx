"""Correctness gate for checkpoints with no transformers reference that fits on this Mac: teacher-forced logits of
the MilliJoule MLX model vs mlx-lm's own implementation, on the same (quantized) weights, any model both support.

Configurations (logits for every position of each sequence, one sequence at a time):
- ours float32 activations (weights as stored)   vs  mlx-lm float32 activations (`set_dtype(float32)`):
  implementation check; quantized weights are dequantized identically by both, so these should agree to ~1e-6.
- ours bf16 (the engine default, precise mode) and mlx-lm bf16: each scored against mlx-lm float32.
- --chunks 5,8,16: ours bf16 again, the same sequences fed C tokens per step (a C-row step, as a decode batch of C
  requests runs its matmuls), with the small-tile kernel (qmm_small.py, the default) and with mx.quantized_matmul.
Data: chat-formatted texts written for this test, model's own chat template (date pinned where it has one).

    .venv/bin/python tools/mlx_vs_mlxlm.py --model <dir> --out results/check_<name>.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "150")  # before mlx is imported: RESULTS 1p
import re
import sys

import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mj_mlx.runner import StepArrays, load_model  # noqa: E402

PINNED_DATE = "2026-11-16"
TEXTS = [
    [("user", "Why does ice float on water?"),
     ("assistant", "Ice floats because it is less dense than liquid water. When water freezes, its molecules lock into "
      "a hexagonal lattice held by hydrogen bonds, which keeps them farther apart than in the liquid.")],
    [("user", "Write a haiku about a train at night."),
     ("assistant", "Lit windows sliding\nacross the sleeping wheat fields\none long breath of steel")],
    [("user", "What is 17 times 23?"),
     ("assistant", "17 times 23 is 391: 17 x 20 = 340 and 17 x 3 = 51, and 340 + 51 = 391.")],
]


def pin_date(text: str) -> str:
    return re.sub(r"Current date: \d{4}-\d{2}-\d{2}", f"Current date: {PINNED_DATE}", text)


def ours_logits(model, ids, chunk=None) -> np.ndarray:
    """Teacher-forced logits for every position: one step over the whole sequence, or `chunk` tokens per step."""
    n = len(ids)
    bs = 16
    nb = (n + bs - 1) // bs
    pools = model.new_pools((nb + 4) * bs, 2)
    out = []
    for a in range(0, n, chunk or n):
        b = min(n, a + (chunk or n))
        s = StepArrays(np.array(ids[a:b], np.uint32), np.arange(a, b, dtype=np.int32), np.zeros(1, np.uint64),
                       np.array([a], np.int32), np.array([b - a], np.int32), np.array([True]),
                       np.arange(1, 1 + nb)[None].astype(np.int32), np.arange(b - a, dtype=np.int32),
                       np.zeros(1, np.uint64), [], None, state_slots=np.array([1], np.int32))
        x = model.hidden(model.prepare(s, bs, True), pools)
        x32 = mx.fast.rms_norm(x.astype(mx.float32), model.norm.astype(mx.float32), model.cfg.rms_norm_eps)
        lg = model.full_logits(x32)
        mx.eval(lg, model.pool_arrays(pools))
        out.append(np.array(lg, dtype=np.float64))
    return np.concatenate(out)


def mlxlm_logits(model, ids) -> np.ndarray:
    lg = model(mx.array([ids], dtype=mx.int32))[0].astype(mx.float32)
    mx.eval(lg)
    return np.array(lg, dtype=np.float64)


def logsoftmax(a):
    a = a - a.max(-1, keepdims=True)
    return a - np.log(np.exp(a).sum(-1, keepdims=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunks", default="", help="e.g. 5,8,16: also score bf16 run C tokens per step")
    a = ap.parse_args()
    mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    mx.set_cache_limit(1 << 30)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    seqs = []
    for conv in TEXTS:
        text = pin_date(tok.apply_chat_template([{"role": r, "content": c} for r, c in conv], tokenize=False))
        seqs.append([int(t) for t in tok(text, add_special_tokens=False)["input_ids"]])
    print("sequences:", [len(s) for s in seqs], flush=True)

    logits = {}
    from mlx_lm import load

    m, _ = load(a.model)
    logits["mlxlm_bf16"] = [mlxlm_logits(m, s) for s in seqs]
    m.set_dtype(mx.float32)
    logits["mlxlm_fp32"] = [mlxlm_logits(m, s) for s in seqs]
    del m
    gc.collect()
    mx.clear_cache()
    from mj_mlx import qmm_small

    for name, dt in (("mj_fp32", mx.float32), ("mj_bf16", mx.bfloat16)):
        m = load_model(a.model, dtype=dt)
        logits[name] = [ours_logits(m, s) for s in seqs]
        if dt == mx.bfloat16:
            for c in [int(c) for c in a.chunks.split(",") if c]:
                for kern in (True, False):
                    qmm_small.ENABLED = kern
                    logits[f"mj_bf16_chunk{c}_{'qmm_small' if kern else 'mx_qmm'}"] = [ours_logits(m, s, c)
                                                                                       for s in seqs]
                qmm_small.ENABLED = True
        del m
        gc.collect()
        mx.clear_cache()

    ref = logits["mlxlm_fp32"]
    res = {}
    for k in ["mj_fp32", "mj_bf16", "mlxlm_bf16"] + [k for k in logits if "chunk" in k]:
        d = np.concatenate([x - r for x, r in zip(logits[k], ref)])
        r = np.concatenate(ref)
        kl = np.concatenate([(np.exp(logsoftmax(rr)) * (logsoftmax(rr) - logsoftmax(x))).sum(-1)
                             for x, rr in zip(logits[k], ref)])
        top1 = np.mean(np.concatenate([x.argmax(-1) == rr.argmax(-1) for x, rr in zip(logits[k], ref)]))
        res[k] = dict(rel_rms=float(np.sqrt((d ** 2).mean() / (r ** 2).mean())), max_abs=float(np.abs(d).max()),
                      top1_vs_mlxlm_fp32=float(top1), mean_kl=float(kl.mean()), max_kl=float(kl.max()),
                      positions=int(len(kl)))
        print(k, res[k], flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(dict(model=a.model, reference="mlx-lm float32 activations, same weights", prompt_date=PINNED_DATE,
                   sequences=[len(s) for s in seqs], results=res), open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
