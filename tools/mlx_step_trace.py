"""Where one decode step's GPU time goes, op group by op group, and where the GPU waits (one request, greedy).

Two kinds of step, each recorded in its own process under Instruments' Metal System Trace (xctrace, ships with
Xcode):
- normal: the engine's own decode step, untouched, at MLX's defaults (engine overlap on unless --no-overlap). Its GPU
  command buffers (each covers several ops; Metal does not report per-dispatch times) and the idle gaps between
  them are the real timeline (normal_segment, normal_steps).
- split (default --split-mode async): the same step with mx.async_eval at every matmul's input and output, plus
  a one-element marker op, and MLX_MAX_OPS_PER_BUFFER / MLX_MAX_MB_PER_BUFFER unlimited. Each op group is then its
  own command buffer (MLX still merges a few back-to-back ones), submitted without waiting, so the GPU stays fed.
  Groups: norms+misc, in_proj, attention (full layers) or conv+gated-delta (DeltaNet layers), out_proj,
  mlp_up_gate_act, mlp_down, lm_head+sampling (after the last layer's MLP: head, sampling, state writes); MoE
  layers add router, expert_gather and shared_expert. Buffers are matched to groups by a sequence alignment that
  allows merges (match(); fit cost per buffer reported). Gaps in split steps are real waits where the step itself
  syncs (e.g. routing readback in streamed MoE), plus the split's own small per-buffer cost.
  --split-mode sync (mx.eval per group) is kept for comparison only: the GPU drops its clocks between syncs and
  durations come out about 3x (RESULTS 1t).
Also runs the same normal steps once without xctrace (untraced wall time), so tracing distortion is visible.

    .venv/bin/python tools/mlx_step_trace.py --model <mlx 9B dir> --out results/step_trace_qwen35_9b/r1.json
"""

from __future__ import annotations

import argparse
import json
import os
os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "150")  # before mlx is imported: RESULTS 1p
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

PROMPT = "Explain how a refrigerator works, step by step, for a curious ten-year-old."
BEFORE = {"q_wt": "norms+misc", "in_wt": "norms+misc", "mlp_gu": "norms+misc", "mlp_g": "norms+misc",
          "head": "norms+misc", "mlp_d": "mlp_up_gate_act", "router_wt": "norms+misc", "sh_d": "shared_expert"}  # o_wt / out_wt: the layer's mixer (set per layer)
OWN = {"q_wt": "in_proj", "kv_wt": "in_proj", "in_wt": "in_proj", "ba_wt": "in_proj", "o_wt": "out_proj",
       "out_wt": "out_proj", "mlp_gu": "mlp_up_gate_act", "mlp_g": "mlp_up_gate_act", "mlp_u": "mlp_up_gate_act",
       "mlp_d": "mlp_down", "head": "lm_head+sampling", "router_wt": "router", "sh_g": "shared_expert",
       "sh_u": "shared_expert", "sh_d": "shared_expert", "sh_gate": "shared_expert"}


def inner(a):
    import mlx.core as mx
    from transformers import AutoTokenizer

    from mj_mlx import Engine, load_model
    from mj_mlx import qwen3_5_moe as q35

    mx.set_wired_limit((a.wired_mb << 20) if a.wired_mb else mx.device_info()["max_recommended_working_set_size"])
    mx.set_cache_limit(1 << 30)
    kw = {}
    if a.map_experts:
        kw["map_experts"] = os.path.expanduser(a.map_experts)
    if a.attn_bits:
        kw["attn_bits"] = a.attn_bits
    elif a.attn_pack:
        kw["attn_pack"] = True
    if a.lean:
        kw["lean"] = True
    if a.stream_experts:
        kw.update(stream_experts=os.path.expanduser(a.stream_experts), expert_slots=a.slots, lean=True,
                  io_threads=8, io_nocache=True, prefetch=True)
    model = load_model(a.model, **kw)
    if a.prompt_len:  # random-token prompt (nothing for a draft model to predict), seed 0
        ids = [int(t) for t in np.random.RandomState(0).randint(1000, 100000, a.prompt_len)]
    else:
        tok = AutoTokenizer.from_pretrained(a.model)
        r = tok.apply_chat_template([{"role": "user", "content": PROMPT}], add_generation_prompt=True,
                                    enable_thinking=False, tokenize=True, return_dict=False)
        ids = [int(t) for t in (r["input_ids"] if hasattr(r, "keys") else r)]
    names = {}
    for li, L in enumerate(model.layers):
        for k, v in L.items():
            if isinstance(v, q35.Lin):
                names[id(v)] = (li, k, L.get("type"))
    if getattr(model, "head_lin", None) is not None:
        names[id(model.head_lin)] = (len(model.layers), "head", None)
    eng = Engine(model=model, num_blocks=256 if not a.prompt_len else a.prompt_len // 16 + 64, max_running=1, max_batch_tokens=2048,
                 overlap=a.overlap and a.phase == "normal")
    eng.add_request(ids, a.warm + a.normal + a.split + 8, ignore_eos=True)
    for _ in range(1 + a.warm):  # prefill, then warm decode steps
        eng.step()
    log = dict(normal=[], split=[], prompt_tokens=len(ids), split_mode=a.split_mode,
               env={k: os.environ.get(k) for k in ("MLX_MAX_OPS_PER_BUFFER", "MLX_MAX_MB_PER_BUFFER")})
    time.sleep(1.0)
    for _ in range(a.normal if a.phase == "normal" else 0):
        t0 = time.perf_counter_ns()
        eng.step()
        log["normal"].append([t0, time.perf_counter_ns()])
    time.sleep(1.0)
    orig = q35.mm
    state = dict(done={}, evals=None, one=mx.ones((1,), mx.float32))
    mx.eval(state["one"])

    def flush(arr, cat, layer, name, wbytes=0):
        if id(arr) in state["done"]:  # already evaluated in this step: no GPU work, no command buffer
            return
        state["done"][id(arr)] = arr  # keeps it alive, so the id stays unique
        t0 = time.perf_counter_ns()
        # a one-element GPU add rides along, so a group with no other GPU work still makes exactly one buffer
        marker = state["one"] + len(state["evals"])
        (real_async if a.split_mode == "async" else real_eval)(arr, marker)
        state["evals"].append(dict(category=cat, layer=layer, name=name, weight_bytes=wbytes, h0=t0,
                                   h1=time.perf_counter_ns()))

    def traced(x, lin):
        li, name, typ = names.get(id(lin), (None, "?", None))
        if name == "router_wt" and state.get("cur") is not None and li == state["cur"] + 1:
            li, name = state["cur"], "router_wt (next layer: prefetch prediction)"
        elif li is not None:
            state["cur"] = li
        before = BEFORE.get(name.split(" ")[0])
        if name in ("o_wt", "out_wt"):
            before = "attention" if typ == "full_attention" else "conv+gated-delta"
        flush(x, before or "norms+misc", li, f"before {name}")
        y = orig(x, lin)
        wb = sum(int(t.nbytes) for t in lin.arrays()) if isinstance(lin, q35.Lin) else 0
        flush(y, OWN.get(name.split(" ")[0], "norms+misc"), li, name, wb)
        return y

    # streamed MoE: the expert computation (slab pair function) and run_moe's own evals get their own groups
    real_eval, real_async = mx.eval, mx.async_eval

    def logged(fn, cat):
        def f(*arrs):
            if sys._getframe(1).f_code.co_name != "run_moe":
                return fn(*arrs)
            t0 = time.perf_counter_ns()
            fn(*arrs, state["one"] + len(state["evals"]))
            state["evals"].append(dict(category=cat, layer=state.get("cur"), name=f"run_moe {cat}", weight_bytes=0,
                                       h0=t0, h1=time.perf_counter_ns()))
            for x in arrs:
                state["done"][id(x)] = x
        return f

    orig_pair_fn = getattr(model, "_slab_pair_fn", None)

    def pair_fn(fmt):
        pair = orig_pair_fn(fmt)

        def p(slab, x, tok, eid, slots, sorted_):
            y = pair(slab, x, tok, eid, slots, sorted_)
            n = len(np.unique(np.array(eid)))
            flush(y, "expert_gather", state.get("cur"), f"experts ({n} unique)", n * slab.store.stride)
            return y
        return p

    q35.mm = traced
    if a.phase == "split":
        mx.eval = logged(real_eval, "router")  # run_moe's routing readback (host sync)
        mx.async_eval = logged(real_async, "expert_gather")
        if orig_pair_fn is not None:
            model._slab_pair_fn = pair_fn
    try:
        for _ in range(a.split if a.phase == "split" else 0):
            time.sleep(0.5)  # separates the steps in the trace
            state["evals"], state["done"] = [], {}
            t0 = time.perf_counter_ns()
            eng.step()
            t1 = time.perf_counter_ns()
            last = state["evals"][-1]["h1"] if state["evals"] else t0
            state["evals"].append(dict(category="lm_head+sampling", layer=len(model.layers), name="tail (head, "
                                       "sampling, state writes)", weight_bytes=0, h0=last, h1=t1))
            log["split"].append(dict(h0=t0, h1=t1, evals=state["evals"]))
    finally:
        q35.mm = orig
        mx.eval, mx.async_eval = real_eval, real_async
    time.sleep(1.0)
    log["weight_bytes_total"] = int(sum(sum(int(t.nbytes) for t in v.arrays()) for L in model.layers
                                        for v in L.values() if isinstance(v, q35.Lin))
                                    + (sum(int(t.nbytes) for t in model.head_lin.arrays())
                                       if getattr(model, "head_lin", None) is not None else 0))
    json.dump(log, open(a.host_log, "w"))


EXPECT_US = {"norms+misc": 20, "conv+gated-delta": 140, "attention": 150, "mlp_up_gate_act": 10,
             "lm_head+sampling": 300, "router": 20, "shared_expert": 20, "expert_gather": 100}
MATMUL_GBS = 55.0  # rough bandwidth of a matmul's own command buffer, only to shape the alignment cost


def match(ivs, evs, gap_pen=2.0):
    """Align GPU intervals (in order) to op groups (in submission order): each interval covers 1-3 consecutive
    groups (MLX sometimes merges back-to-back submissions), a group may have no interval, an interval may match
    nothing; cost |ln((d + 5) / (expected + 5))|. Returns ([(interval, [group indices])], mean cost per interval)."""
    exp = [max(5.0, e["weight_bytes"] / (MATMUL_GBS * 1e3)) if e["weight_bytes"] else
           EXPECT_US.get(e["category"], 20) for e in evs]
    d = [(f - s) / 1e3 for s, f in ivs]
    n, m = len(d), len(exp)
    INF = float("inf")
    dp = np.full((n + 1, m + 1), INF)
    back = {}
    dp[0, 0] = 0
    for i in range(n + 1):
        for j in range(m + 1):
            c = dp[i, j]
            if c == INF:
                continue
            if j < m and c + gap_pen < dp[i, j + 1]:
                dp[i, j + 1] = c + gap_pen
                back[(i, j + 1)] = (i, j, 0)
            if i < n and c + gap_pen < dp[i + 1, j]:
                dp[i + 1, j] = c + gap_pen
                back[(i + 1, j)] = (i, j, -1)
            for k in (1, 2, 3):
                if i < n and j + k <= m:
                    v = c + abs(np.log((d[i] + 5) / (sum(exp[j:j + k]) + 5))) + 0.3 * (k - 1)
                    if v < dp[i + 1, j + k]:
                        dp[i + 1, j + k] = v
                        back[(i + 1, j + k)] = (i, j, k)
    pairs, i, j = [], n, m
    while (i, j) != (0, 0):
        pi, pj, k = back[(i, j)]
        if k == -1:
            pairs.append((ivs[pi], []))
        elif k > 0:
            pairs.append((ivs[pi], list(range(pj, pj + k))))
        i, j = pi, pj
    pairs.reverse()
    return pairs, float(dp[n, m]) / max(n, 1)


def gpu_intervals(trace):
    import mlx_gpu_timeline as tl

    ivs = []
    for r in tl.rows(tl.export(trace, "metal-gpu-intervals")):
        if "python" not in r.get("process", ("", ""))[1].lower():
            continue
        if "compute" not in r.get("channel-name", ("", ""))[1].lower():
            continue
        if r.get("event-depth", ("0",))[0] not in ("0", None):
            continue
        s = int(r["start"][0])
        ivs.append((s, s + int(r["duration"][0])))
    ivs.sort()
    return ivs


def segments(ivs, gap_ns):
    cuts = [i + 1 for i in range(len(ivs) - 1) if ivs[i + 1][0] - ivs[i][1] > gap_ns]
    return [ivs[i:j] for i, j in zip([0] + cuts, cuts + [len(ivs)])]


def run_inner(a, phase, host_log, trace=None, env=None):
    py = sys.executable
    extra = [*(["--stream-experts", a.stream_experts, "--slots", str(a.slots)] if a.stream_experts else []),
             *(["--map-experts", a.map_experts] if a.map_experts else []), *(["--attn-pack"] if a.attn_pack else []),
             *(["--attn-bits", str(a.attn_bits)] if a.attn_bits else []),
             *(["--lean"] if a.lean else []), *(["--wired-mb", str(a.wired_mb)] if a.wired_mb else []),
             *(["--prompt-len", str(a.prompt_len)] if a.prompt_len else [])]
    cmd = [py, __file__, "--inner", "--phase", phase, *([] if a.overlap else ["--no-overlap"]), *extra, "--split-mode", a.split_mode, "--host-log", host_log,
           "--model", a.model, "--warm", str(a.warm), "--normal", str(a.normal), "--split", str(a.split)]
    e = dict(os.environ, **(env or {}))
    if trace:
        subprocess.run(["rm", "-rf", trace])
        cmd = ["xcrun", "xctrace", "record", "--template", "Metal System Trace", "--output", trace, "--launch",
               "--", *cmd]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, env=e)
    return json.load(open(host_log))


def outer(a):
    base = os.path.splitext(a.out)[0]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tdir = a.trace_dir or os.path.dirname(os.path.abspath(a.out))
    os.makedirs(tdir, exist_ok=True)
    # 1. normal steps without Instruments (untraced wall time)
    un = run_inner(a, "normal", base + "_untraced_host.json")
    untraced_us = [(t1 - t0) / 1e3 for t0, t1 in un["normal"]]
    # 2. normal steps traced: the real timeline (MLX defaults)
    hn = run_inner(a, "normal", base + "_normal_host.json", os.path.join(tdir, "normal.trace"))
    seg_normal = [g for g in segments(gpu_intervals(os.path.join(tdir, "normal.trace")), 400e6) if len(g) > 5][-1]
    # steps of the normal segment: the host step windows have no gap info in the trace, so split by host spacing:
    # each step's buffers start after the previous step's last one; use the host step durations as a guide
    busy_n = sum(f - s for s, f in seg_normal)
    span_n = seg_normal[-1][1] - seg_normal[0][0]
    gaps_n = [seg_normal[i + 1][0] - seg_normal[i][1] for i in range(len(seg_normal) - 1)]
    normal_segment = dict(steps=len(hn["normal"]), gpu_span_us=span_n / 1e3, gpu_busy_us=busy_n / 1e3,
                          gpu_busy_pct=100 * busy_n / span_n, idle_us=sum(gaps_n) / 1e3,
                          idle_gaps_over_50us=sum(g > 5e4 for g in gaps_n), command_buffers=len(seg_normal),
                          host_wall_us=(hn["normal"][-1][1] - hn["normal"][0][0]) / 1e3,
                          intervals=[[(s - seg_normal[0][0]) / 1e3, (f - seg_normal[0][0]) / 1e3]
                                     for s, f in seg_normal])
    normal_steps = []
    starts = [t0 for t0, _ in hn["normal"]]
    off = seg_normal[0][0] - starts[0]  # first buffer ~ first step start (host dispatch lead is small, reported)
    for t0, t1 in hn["normal"]:
        mine = [x for x in seg_normal if t0 + off - 2e5 <= x[0] < t1 + off - 2e5]
        if not mine:
            continue
        g0 = mine[0][0]
        busy = sum(f - s for s, f in mine)
        normal_steps.append(dict(wall_us=(t1 - t0) / 1e3, gpu_span_us=(mine[-1][1] - g0) / 1e3,
                                 gpu_busy_us=busy / 1e3, gpu_busy_pct=100 * busy / (mine[-1][1] - g0),
                                 command_buffers=len(mine),
                                 intervals=[[(s - g0) / 1e3, (f - g0) / 1e3] for s, f in mine]))
    if a.normal_only:
        out = dict(model=a.model, prompt_tokens=hn["prompt_tokens"], overlap=a.overlap, options=vars(a),
                   method="xctrace Metal System Trace, metal-gpu-intervals; normal steps only",
                   untraced_normal_step_us=untraced_us, traced_normal_step_us=[s["wall_us"] for s in normal_steps],
                   normal_segment=normal_segment, normal_steps=normal_steps, traces=tdir,
                   time=time.strftime("%Y-%m-%dT%H:%M:%S"))
        json.dump(out, open(a.out, "w"), indent=1)
        ns_ = normal_segment
        print("untraced step ms", [round(x / 1e3, 2) for x in untraced_us])
        print("normal segment: %d steps, GPU span %.0f us, busy %.1f%%, idle %.0f us in %d gaps > 50 us, %d buffers"
              % (ns_["steps"], ns_["gpu_span_us"], ns_["gpu_busy_pct"], ns_["idle_us"], ns_["idle_gaps_over_50us"],
                 ns_["command_buffers"]))
        return
    # 3. split steps traced: one command buffer per op group (async: the GPU stays fed, no per-op host sync)
    env = {"MLX_MAX_OPS_PER_BUFFER": "1000000", "MLX_MAX_MB_PER_BUFFER": "1000000"} if a.split_mode == "async" \
        else None
    hs = run_inner(a, "split", base + "_split_host.json", os.path.join(tdir, "split.trace"), env)
    segs = [g for g in segments(gpu_intervals(os.path.join(tdir, "split.trace")), 300e6) if len(g) > 5][-a.split:]
    split_steps = []
    for st, mine in zip(hs["split"], segs):
        evs, t_step = st["evals"], mine[0][0]
        pairs, cost = match(mine, evs)
        kernels = []
        for (s, f), grp in pairs:
            e = evs[grp[0]] if grp else dict(layer=None, category="unmatched", name="unmatched", weight_bytes=0)
            kernels.append(dict(layer=e["layer"], category=e["category"],
                                name=" + ".join(evs[g]["name"] for g in grp) if grp else "unmatched",
                                merged_groups=len(grp), gpu_start_us=(s - t_step) / 1e3, gpu_end_us=(f - t_step) / 1e3,
                                weight_bytes=sum(evs[g]["weight_bytes"] for g in grp)))
        ok = cost
        busy = sum(f - s for s, f in mine)
        split_steps.append(dict(step_total_us=(mine[-1][1] - t_step) / 1e3, wall_us=(st["h1"] - st["h0"]) / 1e3,
                                gpu_busy_us=busy / 1e3, command_buffers=len(mine), op_groups=len(evs),
                                mapping_cost_per_buffer=ok, merged_buffers=sum(k["merged_groups"] > 1 for k in kernels),
                                unmatched_buffers=sum(k["merged_groups"] == 0 for k in kernels), kernels=kernels))
    by_cat = {}
    for st in split_steps:
        for k in st["kernels"]:
            by_cat.setdefault(k["category"], []).append(k["gpu_end_us"] - k["gpu_start_us"])
    wb_cat = {}
    for k in split_steps[0]["kernels"]:
        wb_cat[k["category"]] = wb_cat.get(k["category"], 0) + k["weight_bytes"]
    ns = len(split_steps)
    out = dict(model=a.model, prompt=PROMPT, prompt_tokens=hn["prompt_tokens"], split_mode=a.split_mode,
               method="xctrace Metal System Trace, metal-gpu-intervals (GPU start / end per command buffer; Metal "
                      "gives no per-dispatch times). normal: the engine's decode step at MLX defaults. split: "
                      "mx.async_eval at every matmul's input and output, MLX_MAX_OPS_PER_BUFFER / MB unlimited, so "
                      "each op group is exactly one command buffer, mapped to its group in submission order",
               untraced_normal_step_us=untraced_us, traced_normal_step_us=[s["wall_us"] for s in normal_steps],
               split_step_wall_us=[s["wall_us"] for s in split_steps],
               gpu_us_per_step_by_category={k: sum(v) / ns for k, v in sorted(by_cat.items())},
               weight_bytes_by_category=wb_cat, resident_weight_bytes=hn["weight_bytes_total"],
               overlap=a.overlap, normal_segment=normal_segment,
               normal_steps=normal_steps, split_steps=split_steps, traces=tdir,
               time=time.strftime("%Y-%m-%dT%H:%M:%S"), notes=__doc__)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("normal_steps", "split_steps", "notes", "normal_segment")}, indent=1))
    ns_ = normal_segment
    print("normal segment: %d steps, GPU span %.0f us, busy %.1f%%, idle %.0f us in %d gaps > 50 us, %d buffers"
          % (ns_["steps"], ns_["gpu_span_us"], ns_["gpu_busy_pct"], ns_["idle_us"], ns_["idle_gaps_over_50us"],
             ns_["command_buffers"]))
    for s in normal_steps:
        print("normal step: wall %.0f us, GPU span %.0f us, busy %.0f us (%.0f%%), %d command buffers"
              % (s["wall_us"], s["gpu_span_us"], s["gpu_busy_us"], s["gpu_busy_pct"], s["command_buffers"]))
    for s in split_steps:
        print("split step: wall %.0f us, GPU span %.0f us, busy %.0f us, %d buffers for %d op groups, "
              "fit cost/buffer %.2f, merged %d, unmatched %d"
              % (s["wall_us"], s["step_total_us"], s["gpu_busy_us"], s["command_buffers"], s["op_groups"],
                 s["mapping_cost_per_buffer"], s["merged_buffers"], s["unmatched_buffers"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--trace-dir", default=None, help="where to keep the .trace bundles (default: next to --out)")
    ap.add_argument("--split-mode", choices=["async", "sync"], default="async")
    ap.add_argument("--no-overlap", dest="overlap", action="store_false",
                    help="normal steps without the engine's overlap mode (host work then shows as GPU idle)")
    ap.add_argument("--normal-only", action="store_true", help="skip the split steps (models other than Qwen3.5)")
    ap.add_argument("--map-experts", default=None)
    ap.add_argument("--stream-experts", default=None, help="SSD expert streaming from this store (prefetch on)")
    ap.add_argument("--slots", type=int, default=96)
    ap.add_argument("--attn-pack", action="store_true")
    ap.add_argument("--attn-bits", type=int, default=None, help="8: the mj-a8 arm's 8-bit attention weights")
    ap.add_argument("--lean", action="store_true")
    ap.add_argument("--wired-mb", type=int, default=None)
    ap.add_argument("--prompt-len", type=int, default=0, help="random-token prompt of this length (else chat)")
    ap.add_argument("--phase", choices=["normal", "split"], default="normal")
    ap.add_argument("--warm", type=int, default=8)
    ap.add_argument("--normal", type=int, default=5)
    ap.add_argument("--split", type=int, default=3)
    ap.add_argument("--inner", action="store_true")
    ap.add_argument("--host-log", default=None)
    a = ap.parse_args()
    inner(a) if a.inner else outer(a)


if __name__ == "__main__":
    main()
