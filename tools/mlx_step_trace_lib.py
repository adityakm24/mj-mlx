"""The same decode-step trace as tools/mlx_step_trace.py, for the existing library (mlx-lm), so the two can be drawn side
by side (Bench Board, per-layer timeline).

What is the same as ours (mlx_step_trace.py; its outer(), match(), gpu_intervals() and segments() are reused as is):
- one request, greedy, the same chat prompt (or --prompt-len random tokens, seed 0), the same warm / normal / split
  step counts, wired limit = max recommended working set, cache limit 1 GB;
- normal: the library's own decode loop (mlx_lm.generate.generate_step, which pipelines one step ahead with
  mx.async_eval), untouched, at the library's MLX defaults (no MLX_MAX_* overrides; ours runs at its own default,
  MLX_MAX_MB_PER_BUFFER 150), once untraced and once under xctrace;
- split: mx.async_eval at every matmul's input and output (every nn.Linear / nn.QuantizedLinear call, at run time;
  mlx-lm's files are not changed) plus the one-element marker op, with MLX_MAX_OPS_PER_BUFFER / MB unlimited, as ours.
  Groups use ours' categories (in_proj, attention, conv+gated-delta, out_proj, mlp_up_gate_act, mlp_down, router,
  expert_gather, shared_expert, norms+misc, lm_head+sampling; unmatched).
What differs, because the code differs: mlx-lm runs separate q / k / v (and DeltaNet qkv / z / b / a, MLP gate / up)
matmuls where ours fuses some, so its split step has more op groups.

    .venv/bin/python tools/mlx_step_trace_lib.py --model <mlx 9B dir> --out results/step_trace_qwen35_9b/lib_mlxlm_r1.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mlx_step_trace as ST  # noqa: E402  (sets MLX_MAX_MB_PER_BUFFER in this outer process only; see run_inner)

# mlx-lm module path (relative to a decoder layer) -> (group before the matmul, the matmul's own group)
ROLE = {
    "self_attn.q_proj": ("norms+misc", "in_proj"), "self_attn.k_proj": ("norms+misc", "in_proj"),
    "self_attn.v_proj": ("norms+misc", "in_proj"), "self_attn.o_proj": ("attention", "out_proj"),
    "linear_attn.in_proj_qkv": ("norms+misc", "in_proj"), "linear_attn.in_proj_z": ("norms+misc", "in_proj"),
    "linear_attn.in_proj_b": ("norms+misc", "in_proj"), "linear_attn.in_proj_a": ("norms+misc", "in_proj"),
    "linear_attn.in_proj_qkvz": ("norms+misc", "in_proj"), "linear_attn.in_proj_ba": ("norms+misc", "in_proj"),
    "linear_attn.out_proj": ("conv+gated-delta", "out_proj"),
    "mlp.gate_proj": ("norms+misc", "mlp_up_gate_act"), "mlp.up_proj": ("norms+misc", "mlp_up_gate_act"),
    "mlp.down_proj": ("mlp_up_gate_act", "mlp_down"),
    "mlp.gate": ("norms+misc", "router"),
    "mlp.switch_mlp.gate_proj": ("router", "expert_gather"), "mlp.switch_mlp.up_proj": ("router", "expert_gather"),
    "mlp.switch_mlp.down_proj": ("expert_gather", "expert_gather"),
    "mlp.shared_expert.gate_proj": ("expert_gather", "shared_expert"),
    "mlp.shared_expert.up_proj": ("expert_gather", "shared_expert"),
    "mlp.shared_expert.down_proj": ("shared_expert", "shared_expert"),
    "mlp.shared_expert_gate": ("shared_expert", "shared_expert"),
}


def inner(a):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.generate import generate_step
    from mlx_lm.utils import load as lm_load
    from transformers import AutoTokenizer

    mx.set_wired_limit((a.wired_mb << 20) if a.wired_mb else mx.device_info()["max_recommended_working_set_size"])
    mx.set_cache_limit(1 << 30)
    model, _ = lm_load(a.model)
    if a.prompt_len:
        ids = [int(t) for t in np.random.RandomState(0).randint(1000, 100000, a.prompt_len)]
    else:
        tok = AutoTokenizer.from_pretrained(a.model)
        r = tok.apply_chat_template([{"role": "user", "content": ST.PROMPT}], add_generation_prompt=True,
                                    enable_thinking=False, tokenize=True, return_dict=False)
        ids = [int(t) for t in (r["input_ids"] if hasattr(r, "keys") else r)]
    layers = list(model.layers)
    names, wbytes = {}, {}
    for li, L in enumerate(layers):
        for path, m in L.named_modules():
            if path in ROLE:
                names[id(m)] = (li, path)
    head = getattr(model, "lm_head", None)
    if head is None and hasattr(model, "language_model"):
        head = getattr(model.language_model, "lm_head", None)
    if head is not None:
        names[id(head)] = (len(layers), "lm_head")
    lin_types = tuple(t for t in (nn.Linear, nn.QuantizedLinear, getattr(nn, "QQLinear", None)) if t is not None)
    try:
        from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwitchLinear
        lin_types += (SwitchLinear, QuantizedSwitchLinear)
    except ImportError:
        pass
    total_wb = 0
    for _, m in model.named_modules():
        if isinstance(m, lin_types):
            from mlx.utils import tree_flatten
            wbytes[id(m)] = sum(int(v.nbytes) for _, v in tree_flatten(m.parameters()))
            total_wb += wbytes[id(m)]
    gen = generate_step(mx.array(ids), model, max_tokens=a.warm + a.normal + a.split + 8)
    for _ in range(1 + a.warm):  # prefill (first next()), then warm decode steps
        next(gen)
    log = dict(normal=[], split=[], prompt_tokens=len(ids), split_mode=a.split_mode, engine="mlx-lm",
               env={k: os.environ.get(k) for k in ("MLX_MAX_OPS_PER_BUFFER", "MLX_MAX_MB_PER_BUFFER")})
    time.sleep(1.0)
    for _ in range(a.normal if a.phase == "normal" else 0):
        t0 = time.perf_counter_ns()
        next(gen)
        log["normal"].append([t0, time.perf_counter_ns()])
    time.sleep(1.0)
    real_eval, real_async = mx.eval, mx.async_eval
    state = dict(done={}, evals=None, one=mx.ones((1,), mx.float32))
    mx.eval(state["one"])

    def flush(arr, cat, layer, name, wb=0):  # as ST.inner's flush
        if id(arr) in state["done"]:
            return
        state["done"][id(arr)] = arr
        t0 = time.perf_counter_ns()
        marker = state["one"] + len(state["evals"])
        (real_async if a.split_mode == "async" else real_eval)(arr, marker)
        state["evals"].append(dict(category=cat, layer=layer, name=name, weight_bytes=wb, h0=t0,
                                   h1=time.perf_counter_ns()))

    origs = {}

    def wrap(cls):
        orig = cls.__call__
        origs[cls] = orig

        def call(self, x, *args, **kw):
            if state["evals"] is None or id(self) not in names:
                return orig(self, x, *args, **kw)
            li, path = names[id(self)]
            before, own = ROLE.get(path, ("norms+misc", "lm_head+sampling" if path == "lm_head" else "norms+misc"))
            flush(x, before, li, f"before {path}")
            y = orig(self, x, *args, **kw)
            flush(y, own, li, path, wbytes.get(id(self), 0))
            return y
        cls.__call__ = call

    for cls in lin_types:
        if cls not in origs and "__call__" in cls.__dict__:
            wrap(cls)
    try:
        for _ in range(a.split if a.phase == "split" else 0):
            time.sleep(0.5)  # separates the steps in the trace
            state["evals"], state["done"] = [], {}
            t0 = time.perf_counter_ns()
            next(gen)
            t1 = time.perf_counter_ns()
            last = state["evals"][-1]["h1"] if state["evals"] else t0
            state["evals"].append(dict(category="lm_head+sampling", layer=len(layers), name="tail (head, "
                                       "sampling, state writes)", weight_bytes=0, h0=last, h1=t1))
            log["split"].append(dict(h0=t0, h1=t1, evals=state["evals"]))
            state["evals"] = None
    finally:
        for cls, orig in origs.items():
            cls.__call__ = orig
    log["weight_bytes_total"] = total_wb
    json.dump(log, open(a.host_log, "w"))


def run_inner(a, phase, host_log, trace=None, env=None):
    """ST.run_inner for this file: the library's normal steps run at MLX defaults (no MLX_MAX_* from ours)."""
    cmd = [sys.executable, __file__, "--inner", "--phase", phase, "--split-mode", a.split_mode, "--host-log", host_log,
           "--model", a.model, "--warm", str(a.warm), "--normal", str(a.normal), "--split", str(a.split),
           *(["--wired-mb", str(a.wired_mb)] if a.wired_mb else []),
           *(["--prompt-len", str(a.prompt_len)] if a.prompt_len else [])]
    e = {k: v for k, v in os.environ.items() if not k.startswith("MLX_MAX_")}
    e.update(env or {})
    if trace:
        subprocess.run(["rm", "-rf", trace])
        cmd = ["xcrun", "xctrace", "record", "--template", "Metal System Trace", "--output", trace, "--launch",
               "--", *cmd]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, env=e)
    return json.load(open(host_log))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--trace-dir", default=None)
    ap.add_argument("--split-mode", choices=["async", "sync"], default="async")
    ap.add_argument("--normal-only", action="store_true")
    ap.add_argument("--wired-mb", type=int, default=None)
    ap.add_argument("--prompt-len", type=int, default=0)
    ap.add_argument("--phase", choices=["normal", "split"], default="normal")
    ap.add_argument("--warm", type=int, default=8)
    ap.add_argument("--normal", type=int, default=5)
    ap.add_argument("--split", type=int, default=3)
    ap.add_argument("--inner", action="store_true")
    ap.add_argument("--host-log", default=None)
    a = ap.parse_args()
    if a.inner:
        return inner(a)
    a.overlap = None  # the library has no overlap switch; its loop pipelines one step ahead
    ST.run_inner = run_inner
    ST.outer(a)
    out = json.load(open(a.out))
    out.update(engine="mlx-lm", library_version=__import__("importlib.metadata").metadata.version("mlx-lm"),
               method=out["method"].replace("the engine's decode step", "mlx-lm's generate_step decode step"),
               notes=__doc__)
    out["options"] = vars(a)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
