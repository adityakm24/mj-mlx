"""Cost of storing the Gated DeltaNet recurrent state in bf16 instead of float32 (math stays float32).

Teacher-forced logits at every position, the rest of the model in float32 so only the state storage differs. The
state is rounded each time it is written back: after every token when fed one token per step (decode-like,
worst case), after every chunk when fed in chunks. Reports top-1 agreement, mean/max KL and the logit error
relative to the logits' RMS, against the float32-state run.

    .venv/bin/python tools/mlx_state_dtype.py --model <qwen3.6 dir> --out results/qwen_state_dtype_tiny.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mj_mlx import Qwen35Moe  # noqa: E402
from mj_mlx.runner import StepArrays  # noqa: E402


def logits_for(m, ids, chunk):
    pools = m.new_pools(((len(ids) + 15) // 16 + 1) * 16, 1)
    out, start = [], 0
    while start < len(ids):
        n = min(chunk, len(ids) - start)
        bt = np.arange((start + n + 15) // 16)[None].astype(np.int32)
        s = StepArrays(np.array(ids[start:start + n], np.uint32), np.arange(start, start + n, dtype=np.int32),
                       np.zeros(1, np.uint64), np.array([start], np.int32), np.array([n], np.int32), np.array([True]),
                       bt, np.arange(n, dtype=np.int32), np.zeros(1, np.uint64), [], None,
                       state_slots=np.array([0], np.int32))
        x = m.hidden(m.prepare(s, 16, True), pools)
        lg = mx.fast.rms_norm(x.astype(mx.float32), m.norm, m.cfg.rms_norm_eps) @ m.lm_head.astype(mx.float32).T
        mx.eval(lg, m.pool_arrays(pools))
        out.append(np.array(lg))
        start += n
    return np.concatenate(out)


def compare(ref, got):
    def ls(a):
        a = a.astype(np.float64)
        a = a - a.max(-1, keepdims=True)
        return a - np.log(np.exp(a).sum(-1, keepdims=True))

    lr, lg = ls(ref), ls(got)
    kl = (np.exp(lr) * (lr - lg)).sum(-1)
    return dict(top1=float((ref.argmax(-1) == got.argmax(-1)).mean()), mean_kl=float(kl.mean()),
                max_kl=float(kl.max()),
                rel_logit_rms_err=float(np.sqrt(((ref - got) ** 2).mean() / (ref ** 2).mean())),
                positions=int(ref.shape[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--prompts", type=int, default=4)
    ap.add_argument("--len", type=int, default=400)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    path = a.model or glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--trl-internal-testing--tiny-Qwen3_5MoeForConditionalGeneration-3.6/"
        "snapshots/*"))[0]
    rs = np.random.RandomState(0)
    prompts = [[int(t) for t in rs.randint(0, 248000, a.len)] for _ in range(a.prompts)]
    m32 = Qwen35Moe(path, dtype=mx.float32, state_dtype=mx.float32)
    m16 = Qwen35Moe(path, dtype=mx.float32, state_dtype=mx.bfloat16)
    res = {}
    for chunk in (1, 64):
        ref = np.concatenate([logits_for(m32, p, chunk) for p in prompts])
        got = np.concatenate([logits_for(m16, p, chunk) for p in prompts])
        res[f"chunk{chunk}"] = compare(ref, got)
        print(f"state bf16 vs fp32, {chunk}-token steps:", res[f"chunk{chunk}"], flush=True)
    out = dict(model=path, note="float32 model; only the stored recurrent state differs; teacher-forced random tokens; "
               "tiny random checkpoint (real state shapes Hv 32, Dk = Dv = 128; hidden 16, flat logits)",
               state_bytes_per_slot={"float32": m32.state_bytes_per_slot(), "bfloat16": m16.state_bytes_per_slot()},
               results=res)
    json.dump(out, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
