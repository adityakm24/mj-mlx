"""Closed-loop throughput and energy benchmark on Apple silicon: mlx-lm (baseline) vs MilliJoule's MLX runner.

Rules (a Mac-sized copy of plan.md's harness rules):
- Same prompts (seeded), same output lengths (EOS ignored), same concurrency levels for both engines.
- Per level, 4x as many requests as the level; the engine keeps at most `level` in flight. The measurement
  window opens when `level` requests are generating and closes when fewer than `level` remain.
- Energy: IOReport "GPU Energy" rail (whole-chip SoC energy model, idle floor included). Mac-only; never compared
  with NVML/B200 numbers. Idle GPU power with the model loaded is recorded per run.
- Tokens: generated tokens counted at the engine.
- Per level also: TTFT p50/p99 (all of a level's requests are submitted at t0, so TTFT includes queueing behind
  the running set), GPU active residency over the window (IOReport GPUPH, = not power-gated, includes other apps;
  idle reference recorded), process CPU % over the window (process CPU time / wall, all threads, 100 = one core),
  MLX peak memory per level; process peak RSS at the top level.

    .venv/bin/python tools/mlx_bench.py --engine mlx-lm --model <dir> --levels 1,8,32,64 --out results/x.json
"""

from __future__ import annotations

import argparse
import json
import os
os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "150")  # before mlx is imported: RESULTS 1p
import platform
import resource
import sys
import time

import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mj_mlx.energy import Meter, _cf  # noqa: E402


def workload(n, vocab, lo, hi, seed):
    rs = np.random.RandomState(seed)
    lens = rs.randint(lo, hi + 1, n)
    return [[int(t) for t in rs.randint(0, min(vocab, 199998), L)] for L in lens]


def vocab_of(cfg):
    return cfg.get("text_config", cfg)["vocab_size"]


class Window:
    """Opens/closes one energy window around the steady-state part of a run; also tracks TTFT per request."""

    def __init__(self, level, n):
        self.level, self.n = level, n
        self.t0 = self.e0 = None
        self.tokens = 0
        self.done = False
        self.result = None
        self._meter = Meter()
        self.t_submit = time.perf_counter()
        self.first = {}
        self.max_active = 0

    def first_tokens(self, ids):
        now = time.perf_counter()
        for i in ids:
            self.first.setdefault(int(i), now - self.t_submit)

    def tick(self, active, finished, new_tokens):
        self.max_active = max(self.max_active, active)
        if self.done:
            return
        if self.t0 is None:
            if active >= self.level:
                mx.synchronize()
                self._meter.__enter__()
                self.t0 = time.perf_counter()
                self.cpu0 = time.process_time()
            return
        self.tokens += new_tokens
        if self.n - finished < self.level:
            mx.synchronize()
            self._meter.__exit__(None, None, None)
            w = self._meter.window
            self.result = dict(seconds=w.seconds, gpu_j=w.gpu_j, tokens=self.tokens, gpu_active_pct=w.gpu_active_pct,
                               cpu_pct=100.0 * (time.process_time() - self.cpu0) / w.seconds)
            self.done = True

    def ttft(self):
        v = np.array(sorted(self.first.values()))
        return (float(np.percentile(v, 50)), float(np.percentile(v, 99))) if len(v) else (None, None)


def idle_power(seconds=2.0):
    mx.synchronize()
    time.sleep(0.5)
    with Meter() as m:
        time.sleep(seconds)
    return m.window.gpu_j / m.window.seconds, m.window.gpu_active_pct


def run_mlx_lm(model, prompts, out_len, level, args):
    from mlx_lm.generate import BatchGenerator

    gen = BatchGenerator(model, max_tokens=out_len, completion_batch_size=level,
                         prefill_batch_size=min(level, args.prefill_batch), prefill_step_size=args.prefill_step)
    gen.insert(prompts, [out_len] * len(prompts))
    win = Window(level, len(prompts))
    started, finished, total = set(), 0, 0
    t = time.perf_counter()
    while True:
        resps = gen.next_generated()
        if not resps:
            break
        win.first_tokens(r.uid for r in resps)
        for r in resps:
            started.add(r.uid)
            if r.finish_reason is not None:
                finished += 1
        total += len(resps)
        win.tick(len(started) - finished, finished, len(resps))
    wall = time.perf_counter() - t
    gen.close()
    return win, total, wall


def run_mj(engine, prompts, out_len, level, args):
    win = Window(level, len(prompts))
    for p in prompts:
        engine.add_request(p, out_len, ignore_eos=True)
    finished, total = 0, 0
    t = time.perf_counter()
    while engine.has_work():
        before = engine.tokens_out
        finished += len(engine.step())
        win.first_tokens(engine.last_sampled.tolist())
        new = engine.tokens_out - before
        total += new
        win.tick(engine.sched.num_running, finished, new)
    wall = time.perf_counter() - t
    return win, total, wall


def run_llama_server(url, prompts, out_len, level, args):
    """llama.cpp's llama-server over HTTP, the same closed loop: `level` client threads, each streaming one /completion
    at a time with the exact token ids (ignore_eos, no prompt cache). A request counts as generating from its first
    token. The server must be started with -np >= level (one slot per in-flight request)."""
    import threading
    import urllib.request

    win = Window(level, len(prompts))
    lock = threading.Lock()
    state = dict(next=0, active=0, finished=0, total=0)

    def worker():
        while True:
            with lock:
                i = state["next"]
                if i >= len(prompts):
                    return
                state["next"] += 1
            body = json.dumps(dict(prompt=prompts[i], n_predict=out_len, ignore_eos=True, stream=True,
                                   temperature=0.0, cache_prompt=False)).encode()
            req = urllib.request.Request(url + "/completion", body, {"Content-Type": "application/json"})
            first = True
            with urllib.request.urlopen(req, timeout=3600) as r:
                for line in r:
                    line = line.strip()
                    if not line.startswith(b"data:"):
                        continue
                    ev = json.loads(line[5:])
                    n = len(ev.get("tokens") or []) or (1 if ev.get("content") else 0)
                    done = bool(ev.get("stop"))
                    with lock:
                        if n and first:
                            first = False
                            state["active"] += 1
                            win.first_tokens([i])
                        state["total"] += n
                        if done:
                            state["active"] -= 1
                            state["finished"] += 1
                        win.tick(state["active"], state["finished"], n)
                    if done:
                        break

    t = time.perf_counter()
    th = [threading.Thread(target=worker) for _ in range(level)]
    for x in th:
        x.start()
    for x in th:
        x.join()
    return win, state["total"], time.perf_counter() - t


def run_text_server(kind, url, model_id, texts, out_len, level):
    """vllm-mlx (kind "openai": /v1/completions) or Ollama (kind "ollama": /api/generate raw) in the same closed
    loop. These servers take text, so each prompt is the checkpoint tokenizer's decode of the same ids, and they
    cannot ignore EOS: a request may stop before out_len (the window counts the tokens that arrived). One streamed
    chunk counts as one token (both stream one token per chunk at their defaults)."""
    import threading
    import urllib.request

    win = Window(level, len(texts))
    lock = threading.Lock()
    state = dict(next=0, active=0, finished=0, total=0)

    def events(i):
        if kind == "openai":
            body = dict(model=model_id, prompt=texts[i], max_tokens=out_len, temperature=0.0, stream=True)
            req = urllib.request.Request(url + "/v1/completions", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=3600) as r:
                for line in r:
                    line = line.strip()
                    if not line.startswith(b"data:") or line[5:].strip() == b"[DONE]":
                        continue
                    ev = json.loads(line[5:])
                    ch = ev.get("choices") or []
                    if ch and ch[0].get("text"):
                        yield 1, False
                    if ch and ch[0].get("finish_reason"):
                        yield 0, True
                        return
            yield 0, True
        else:
            body = dict(model=model_id, prompt=texts[i], raw=True, stream=True, think=False,
                        options=dict(temperature=0, num_predict=out_len))
            req = urllib.request.Request(url + "/api/generate", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=3600) as r:
                for line in r:
                    ev = json.loads(line)
                    yield (1 if ev.get("response") else 0), bool(ev.get("done"))
                    if ev.get("done"):
                        return

    def worker():
        while True:
            with lock:
                i = state["next"]
                if i >= len(texts):
                    return
                state["next"] += 1
            first = True
            for n, done in events(i):
                with lock:
                    if n and first:
                        first = False
                        state["active"] += 1
                        win.first_tokens([i])
                    state["total"] += n
                    if done:
                        state["active"] -= 1 if not first else 0
                        state["finished"] += 1
                    win.tick(state["active"], state["finished"], n)
                if done:
                    break

    t = time.perf_counter()
    th = [threading.Thread(target=worker) for _ in range(level)]
    for x in th:
        x.start()
    for x in th:
        x.join()
    return win, state["total"], time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["mlx-lm", "mj", "llama-server", "openai", "ollama"], required=True)
    ap.add_argument("--model-id", default=None, help="openai / ollama: the server's model name")
    ap.add_argument("--model", required=True, help="MLX checkpoint dir (llama-server: only for the vocab size)")
    ap.add_argument("--url", default="http://127.0.0.1:8080", help="llama-server: server URL (started with -np >= "
                                                                  "the level)")
    ap.add_argument("--weights", default="", help="llama-server: recorded, which GGUF the server runs")
    ap.add_argument("--levels", default="1,8,32,64")
    ap.add_argument("--prompt-min", type=int, default=64)
    ap.add_argument("--prompt-max", type=int, default=320)
    ap.add_argument("--out-len", type=int, default=128)
    ap.add_argument("--per-level", type=int, default=4, help="requests = per_level * level")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prefill-step", type=int, default=2048)
    ap.add_argument("--prefill-batch", type=int, default=8)
    ap.add_argument("--max-batch-tokens", type=int, default=2048)
    ap.add_argument("--no-paged", action="store_true", help="mj: gather + SDPA instead of the paged kernel")
    ap.add_argument("--lm-head-bits", type=int, default=-1, help="mj: lm_head candidate width (-1 auto by rows, 0 bf16)")
    ap.add_argument("--plain", action="store_true", help="mj: bf16 residual/router instead of precise")
    ap.add_argument("--state-layout", default=None, choices=["per_slot", "pool"],
                    help="mj, Qwen models: recurrent-state layout (default per_slot; pool = the old whole-pool path)")
    ap.add_argument("--max-running", type=int, default=None,
                    help="mj: allocated capacity (KV blocks, and state slots for recurrent models) for N requests at "
                         "every level, as a server sized for N; `level` requests stay in flight; default = the level")
    ap.add_argument("--overlap", action="store_true",
                    help="mj: overlap scheduling (plan and queue step N+1 before step N's tokens reach the host)")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    levels = [int(x) for x in args.levels.split(",")]

    cfg = json.load(open(os.path.join(args.model, "config.json")))
    t = time.perf_counter()
    model = None
    http = args.engine in ("llama-server", "openai", "ollama")
    if args.engine == "llama-server":
        info = json.load(__import__("urllib.request").request.urlopen(args.url + "/props", timeout=30))
        version = f"llama-server {info.get('build_info', '')}".strip()
    elif args.engine in ("openai", "ollama"):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        version = f"{args.engine} server {args.url} model {args.model_id}"
    elif args.engine == "mlx-lm":
        import mlx_lm
        from mlx_lm.utils import load_model

        model, _ = load_model(__import__("pathlib").Path(args.model))
        mx.eval(model.parameters())
        version = f"mlx-lm {mlx_lm.__version__}"
    else:
        from mj_mlx import load_model

        rec = mx.device_info().get("max_recommended_working_set_size")
        mx.set_wired_limit(rec)
        mx.set_cache_limit(1 << 30)
        model = load_model(args.model, dtype=mx.bfloat16, lm_head_bits=args.lm_head_bits or None,
                           precise=not args.plain,
                           **({"state_layout": args.state_layout} if args.state_layout else {}))
        version = "millijoule mlx runner" + (f", state_layout={args.state_layout}" if args.state_layout else "") + (
            ", overlap" if args.overlap else "")
    load_s = time.perf_counter() - t
    idle_w, idle_act = idle_power()
    print(f"{version}: loaded in {load_s:.1f}s, idle GPU {idle_w:.2f} W", flush=True)

    def run(prompts, level):
        if args.engine == "llama-server":
            return run_llama_server(args.url, prompts, args.out_len, level, args)
        if args.engine in ("openai", "ollama"):
            return run_text_server(args.engine, args.url, args.model_id, [tokenizer.decode(p) for p in prompts],
                                   args.out_len, level)
        if args.engine == "mlx-lm":
            return run_mlx_lm(model, prompts, args.out_len, level, args)
        from mj_mlx import Engine

        cap = max(args.max_running or level, level)  # allocated capacity; `level` requests stay in flight
        need = cap * (args.prompt_max + args.out_len) // 16
        # Recurrent models: one state slot per in-flight request by default, the same concurrency the baseline gets
        # (mlx-lm allocates its float32 states for the whole batch too); --max-running N allocates N, as a server
        # sized for N that is serving `level` requests.
        eng = Engine(model=model, max_running=level, max_batch_tokens=args.max_batch_tokens,
                     enable_prefix_cache=True, num_blocks=int(need * 1.5) + 64,
                     paged_decode=not args.no_paged,
                     num_state_slots=cap if getattr(model, "recurrent", False) else None, overlap=args.overlap)
        run.last_eng = eng
        return run_mj(eng, prompts, args.out_len, level, args)

    run(workload(8, vocab_of(cfg), args.prompt_min, args.prompt_max, 999), 4)  # warm-up, not measured
    rows = []
    for level in levels:
        n = args.per_level * level
        mx.reset_peak_memory()
        with Meter() as whole:
            win, total, wall = run(workload(n, vocab_of(cfg), args.prompt_min, args.prompt_max, args.seed + level),
                                   level)
        r = win.result
        never_full = r is None
        if never_full:  # the engine never had `level` requests in flight (e.g. it serves one at a time): whole run
            w = whole.window
            r = dict(seconds=w.seconds, gpu_j=w.gpu_j, tokens=total, gpu_active_pct=w.gpu_active_pct, cpu_pct=None)
        p50, p99 = win.ttft()
        row = dict(level=level, requests=n, total_tokens=total, wall_s=wall, window_s=r["seconds"],
                   window_tokens=r["tokens"], tok_s=r["tokens"] / r["seconds"], gpu_w=r["gpu_j"] / r["seconds"],
                   gpu_j_per_token=r["gpu_j"] / r["tokens"],
                   gpu_j_per_token_above_idle=(r["gpu_j"] - idle_w * r["seconds"]) / r["tokens"],
                   ttft_p50_s=p50, ttft_p99_s=p99, gpu_util_pct=r["gpu_active_pct"], cpu_util_pct=r["cpu_pct"],
                   mem_peak_gb=mx.get_peak_memory() / 1e9, max_in_flight=win.max_active,
                   window="whole run (never reached level in flight)" if never_full else "all level in flight")
        if http:  # the server's own CPU and memory are not this process's
            row.update(cpu_util_pct=None, mem_peak_gb=None)
        if args.engine == "mj":
            st = run.last_eng.sched.stats()
            row.update(overlapped_inputs=st.get("overlapped_inputs"), discarded_samples=st.get("discarded_samples"))
        rows.append(row)
        print(f"  level {level:4d}: {row['tok_s']:8.1f} tok/s  {row['gpu_w']:5.1f} W  "
              f"{row['gpu_j_per_token'] * 1e3:7.2f} mJ/token (GPU rail)  TTFT p50 {p50:.2f}s p99 {p99:.2f}s  "
              f"GPU act {row['gpu_util_pct'] or 0:.0f}%  CPU {row['cpu_util_pct'] or 0:.0f}%  "
              f"mem {row['mem_peak_gb'] or 0:.1f} GB  wall {wall:.1f}s", flush=True)
        mx.clear_cache()
    if args.engine == "llama-server":
        note = f"GGUF on llama-server: {args.weights or 'unrecorded'} (prompt ids from --model's vocab)"
    elif args.engine in ("openai", "ollama"):
        note = (f"{args.engine} server, weights: {args.weights or 'unrecorded'}; text prompts = tokenizer decode of "
                f"the ids; EOS not ignorable (tokens counted as they arrive)")
    elif cfg.get("bench_note"):
        note = cfg["bench_note"]
    elif cfg.get("model_type") == "gpt_oss":  # tools/mlx_make_bench_model.py checkpoints before bench_note existed
        note = f"random weights, gpt-oss-20b dims, {cfg['num_hidden_layers']} of 24 layers, MXFP4 experts, bf16 rest"
    else:  # a released checkpoint
        note = f"released checkpoint ({cfg.get('model_type')}): {args.model}"
    out = dict(
        engine=version, label=args.label, model=args.model, model_note=note,
        weights=args.weights or None, url=args.url if http else None,
        idle_gpu_active_pct=idle_act,
        process_peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9,
        metric_notes=dict(
            ttft="seconds from submit (all of a level's requests at t0) to the first token; includes queueing",
            gpu_util_pct="IOReport GPUPH active residency (not power-gated) over the window; includes other "
                         "apps; compare with idle_gpu_active_pct; not a busy-cycle counter (needs root)",
            cpu_util_pct="process CPU time / wall over the window, all threads (100 = one core)",
            mem_peak_gb="mx.get_peak_memory(), reset per level",
            process_peak_rss_gb="resource.getrusage ru_maxrss (bytes on macOS), whole run"),
        machine=platform.platform(), chip="Apple M4 16 GB", mlx=mx.__version__,
        energy_boundary="IOReport 'GPU Energy' rail (SoC energy model), whole GPU incl. idle; Mac-only",
        state_layout=getattr(model, "state_layout", None), max_running=args.max_running,
        overlap=bool(args.overlap) if args.engine == "mj" else None,
        idle_gpu_w=idle_w, idle_window_s=2.0,
        idle_note="measured over idle_window_s with the model loaded; 0.0 W with 0% active = GPU power-gated (the "
                  "IOReport counter did not advance), not a missing reading",
        load_s=load_s, args=vars(args), rows=rows,
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
