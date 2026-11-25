"""One engine, one model, the llama-bench-shaped micro grid (results/BENCH_PROTOCOL.md B.1), timed by our clock.

Tests (random token ids in [1000, 100000), a different seed per rep so no prefix cache can hit; EOS ignored):
  ppN       prefill N tokens: request start to the first token (the first token's lm_head + argmax included,
            as llama-bench includes the logits of the last position). tok/s = N / seconds.
  tg128@dD  context of D tokens (D = 0 means a 1-token prompt), then 129 new tokens; decode tok/s = 128 /
            (time of token 129 - time of token 1), i.e. 1 / TPOT with TPOT = (E2E - TTFT) / (N_out - 1).
Energy per test window (IOReport, "estimated, SoC rails"): GPU, CPU, ANE, DRAM joules; J/token over the window
(prefill: / N_in; decode: / 128). Each test: one untimed warm-up, then --reps timed reps, 5 s apart.
Output: JSON lines, one per rep, plus the greedy token ids of each decode rep (correctness across engines).

Engines: mj (MilliJoule Engine, overlap mode), mlx-lm (mlx_lm.generate.generate_step).

    .venv/bin/python tools/mlx_bench_worker.py --engine mj --model <dir> --tests pp512,tg128@d0 --reps 5 --out x.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "150")  # before mlx is imported: RESULTS 1p
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mj_mlx.energy import Meter  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlx_proc_mem import swap_pages  # noqa: E402  (swap-ins / -outs at the window's edges, ~1 us)

NEW = 129  # 1 (first token) + 128 timed decode tokens
STEADY_DROP_S = 3.0  # steady-state energy (checklist 0.16): drop this much of decode after the first token


def new_tokens(test):
    """1 + N for a decode test "tgN@dD" (tg128@d0 -> 129)."""
    return 1 + int(test.split("@")[0][2:])


def ids_for(n, seed, kind=0):
    """Random ids, distinct per (seed, length, test kind): with one seed for every test, the pp2048 prompt began with
    the pp512 prompt and the d2048 prompt was the pp2048 prompt, which servers with a prompt cache (Ollama, LM Studio)
    reused (suite v10: Ollama's d2048 prompts ran at ~9,700 tok/s). Our engine and mlx-lm start each request cold."""
    return [int(t) for t in np.random.RandomState([seed, n, kind]).randint(1000, 100000, max(n, 1))]


def _power_sampler():
    """Whole-machine watts from the SMC (mj_mlx.smc_power), started; None when the SMC is not readable."""
    try:
        from mj_mlx.smc_power import PowerSampler

        return PowerSampler().__enter__()
    except Exception:
        return None


class Win:
    """A Meter window opened and closed by hand (first token .. last token)."""

    def __init__(self):
        self.m = None

    def open(self):
        self.steady = None
        self.p = _power_sampler()
        self.m = Meter()
        self.m.__enter__()
        self.sw0 = swap_pages()
        self.t0 = time.perf_counter()

    def close(self):
        self.m.__exit__(None, None, None)
        t1 = time.perf_counter()
        sw1 = swap_pages()
        sys_j = None
        if self.p:  # PSTR trails the load: keep sampling past the window, integrate over the shifted window
            time.sleep(3.0)
            self.p.__exit__(None, None, None)
            sys_j = self.p.joules_between(self.t0, t1)
            if t1 - self.t0 >= STEADY_DROP_S + 5.0:  # long enough for a steady-state figure (decode only)
                self.steady = dict(t0=self.t0 + STEADY_DROP_S, t1=t1,
                                   sys_j=self.p.joules_between(self.t0 + STEADY_DROP_S, t1))
        w = self.m.window
        e = dict(seconds=w.seconds, gpu_j=w.gpu_j, cpu_j=w.rail("CPU Energy"), ane_j=w.rail("ANE"),
                 dram_j=w.rail("DRAM"), gpu_active_pct=w.gpu_active_pct)
        # CPU / DRAM rails occasionally return a bogus delta (v1: 1087 J CPU + 1232 J DRAM over 2.5 s, ~900 W):
        # such windows get no SoC figure (the GPU rail was unaffected in every case seen)
        e["soc_valid"] = e["cpu_j"] / max(w.seconds, 1e-9) < 60 and e["dram_j"] / max(w.seconds, 1e-9) < 30
        if self.sw0 and sw1:  # machine-wide paging inside the timed window (SSD writes count in the energy figure)
            e.update(swapins_mb=(sw1[0] - self.sw0[0]) * 16384 / 1e6, swapouts_mb=(sw1[1] - self.sw0[1]) * 16384 / 1e6)
        if sys_j is not None:  # whole machine (SMC PSTR), over the window shifted by its lag
            e.update(sys_j=sys_j, sys_w=sys_j / max(t1 - self.t0, 1e-9), sys_samples=len(self.p.samples),
                     sys_window_s=t1 - self.t0)
        return e


def make_mj(path, a):
    import mlx.core as mx

    from mj_mlx import Engine, load_model

    kw = {}
    if a.stream_experts:
        slots = a.slots
        if a.slot_profile:  # per-layer slots, same total, from the store's slot profile (RESULTS 1zt)
            from mj_mlx import placement

            slots = placement.uneven_slots(os.path.expanduser(a.stream_experts), a.slots)
            assert isinstance(slots, dict), "no slot_profile.json in the expert store (tools/mlx_slot_profile.py)"
        kw = dict(stream_experts=os.path.expanduser(a.stream_experts), expert_slots=slots, lean=True, io_threads=8,
                  io_nocache=True, prefetch=True)
    elif a.map_experts:
        kw = dict(map_experts=os.path.expanduser(a.map_experts))
    if a.attn_bits:  # opt-in lossy attention weights (matched-quantization rows vs llama.cpp's Q8_0)
        kw["attn_bits"] = a.attn_bits
    elif a.attn_pack:
        kw["attn_pack"] = True
    if a.lean:
        kw["lean"] = True
        if a.map_head:  # zero-copy embedding / lm_head maps: rows gathered on the GPU, decode steps overlap
            kw["map_head"] = os.path.expanduser(a.map_head)
    if getattr(a, "placement", None):  # MoeLM families: experts placed by memory (placement.py)
        kw.update(placement=a.placement, prefetch=True)  # slots: from memory (placement.stream_slots)
        if a.slots_set:
            kw["expert_slots"] = a.slots
    model = load_model(path, **kw)
    a.locked_gb = model.map.locked / 1e9 if getattr(model, "map", None) else None  # MJ_MLOCK=1: bytes pinned
    wired = (a.wired_mb << 20) if a.wired_mb else mx.device_info()["max_recommended_working_set_size"]

    def run(ids, new, on_first, on_token):
        eng = Engine(model=model, num_blocks=(len(ids) + new) // 16 + 16, max_running=1, max_batch_tokens=a.chunk,
                     overlap=True, wired_limit_bytes=wired, prefill_area_cap=a.area_cap,
                     max_seq_len=max(32768, len(ids) + new + 16))  # 32k prompts: the default limit is 32768 in all
        eng.add_request(ids, new, ignore_eos=True)
        out, first = [], True
        stats = getattr(model, "expert_stats", None) if getattr(model, "store", None) is not None else None
        snap = [stats() if stats else None, None]  # streamed experts: counters at the start and at the first token
        while eng.has_work():
            before = eng.tokens_out
            for f in eng.step():
                out = list(f[2])
            if eng.tokens_out > before:
                for _ in range(eng.tokens_out - before):
                    if first:
                        on_first()
                        first = False
                        if stats:
                            snap[1] = stats()
                    else:
                        on_token()
        if stats and snap[1] is not None:  # expert hits / misses / SSD bytes: prefill (to the first token), decode
            end = stats()

            def delta(a, b):
                d = {k: b[k] - a[k] for k in ("hits", "misses", "bytes_read") if k in a and k in b}
                n = d.get("hits", 0) + d.get("misses", 0)
                d["hit_rate"] = d["hits"] / n if n else None
                return d

            g0, g1 = (snap[1].get("gpu_stream") or {}), (end.get("gpu_stream") or {})  # the C++ streamer (decode)
            gd = {k: g1[k] - g0.get(k, 0) for k in g1 if isinstance(g1[k], (int, float))}
            run.expert = dict(prefill=delta(snap[0], snap[1]), decode=delta(snap[1], end), gpu_stream_decode=gd)
        del eng
        return out

    return run, mx


def make_mlxlm(path, a):
    from pathlib import Path

    import mlx.core as mx
    from mlx_lm.generate import generate_step
    from mlx_lm.utils import load_model

    model, _ = load_model(Path(path))
    mx.eval(model.parameters())
    if a.wired_mb:
        mx.set_wired_limit(a.wired_mb << 20)

    def run(ids, new, on_first, on_token):
        out = []
        for i, (tok, _) in enumerate(generate_step(mx.array(ids), model, max_tokens=new)):
            out.append(int(tok.item() if hasattr(tok, "item") else tok))
            (on_first if i == 0 else on_token)()
        return out

    return run, mx


def stable_text(tok, ids):
    """Text that re-tokenizes to len(ids) tokens: decoded random ids merge into fewer tokens when a server
    re-encodes them (a 2048-id prompt arrived as ~1,100), so re-encode, cut to length and repeat."""
    n, cur = len(ids), list(ids)
    for _ in range(6):
        text = tok.decode(cur)
        enc = tok(text, add_special_tokens=False)["input_ids"]
        if len(enc) == n:
            return text
        extra = [int(t) for t in np.random.RandomState(len(enc)).randint(1000, 100000, max(n - len(enc), 0))]
        cur = (enc + extra)[:n]
    return tok.decode(cur)


OLLAMA_OPENER = "\n\nOnce upon a time, in a small village by the sea,"


def make_ollama(path, a):
    """Ollama's native /api/generate, raw prompt = the checkpoint tokenizer's decode of the test ids, greedy,
    streamed (one event per token), timed by our clock like the other engines. EOS cannot be ignored (a short reply is
    flagged in the row); num_ctx is fixed so the model is never reloaded between tests. Server timings are kept."""
    import urllib.request

    import mlx.core as mx
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)

    def run(ids, new, on_first, on_token):
        # Ollama cannot ignore EOS and random-token prompts end at once: decode tests (new > 1) get a short
        # narrative opener appended (its prompt_eval_count is recorded); prefill tests send the ids' text as is
        text = stable_text(tok, ids) + (OLLAMA_OPENER if new > 1 else "")
        body = json.dumps(dict(model=a.ollama_model, prompt=text, raw=True, stream=True, think=False,
                               keep_alive="30m", options=dict(temperature=0, num_predict=new, num_ctx=8192))).encode()
        req = urllib.request.Request(a.ollama_url + "/api/generate", body, {"Content-Type": "application/json"})
        first, n, final = True, 0, None
        with urllib.request.urlopen(req, timeout=3600) as r:
            for line in r:
                ev = json.loads(line)
                if ev.get("done"):
                    final = ev
                    continue
                if first:
                    on_first()
                    first = False
                else:
                    on_token()
                n += 1
        if first and final is not None:  # one-token replies come only in the final event
            on_first()
        run.server = {k: final.get(k) for k in ("prompt_eval_count", "prompt_eval_duration", "eval_count",
                                                "eval_duration", "load_duration")} if final else None
        run.server_tokens = n
        return []

    return run, mx


def make_openai(path, a):
    """An OpenAI-compatible server (LM Studio, vllm-mlx): streamed /v1/completions, greedy, the same re-tokenizing
    text prompt and decode opener as Ollama (`ignore_eos` is sent too, honoured by vLLM-style servers). Timed by our
    clock; when a server packs several tokens into one stream chunk, decode tok/s is recomputed from its reported
    completion_tokens (kept in the row with the chunk count)."""
    import urllib.request

    import mlx.core as mx
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)

    def run(ids, new, on_first, on_token):
        text = stable_text(tok, ids) + (OLLAMA_OPENER if new > 1 else "")
        body = json.dumps(dict(model=a.api_model, prompt=text, max_tokens=new, temperature=0, stream=True,
                               ignore_eos=True, stream_options={"include_usage": True})).encode()
        req = urllib.request.Request(a.api_url + "/v1/completions", body, {"Content-Type": "application/json"})
        first, chunks, usage, last, err = True, 0, None, None, None
        with urllib.request.urlopen(req, timeout=3600) as r:
            for line in r:
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                ev = json.loads(data)
                last = ev
                if ev.get("error"):  # an error event inside a 200 stream (llama-server on DeepSeek-V2-Lite, v15)
                    err = json.dumps(ev["error"])[:500]
                    break
                if ev.get("usage"):
                    usage = ev["usage"]
                ch = ev.get("choices") or []
                if not ch or not ch[0].get("text"):
                    continue
                if first:
                    on_first()
                    first = False
                else:
                    on_token()
                chunks += 1
        if first:  # nothing streamed with text (a one-token reply may come empty): the end of the reply
            on_first()
        run.server = dict(usage=usage, chunks=chunks, stats=(last or {}).get("stats"), error=err)
        return []

    return run, mx


def one(run, mx, test, seed):
    if test.startswith("pp"):
        n = int(test[2:])
        ids = ids_for(n, seed)
        t = {}
        w = Win()
        w.open()
        t0 = time.perf_counter()
        run(ids, 1, lambda: t.__setitem__("first", time.perf_counter()), lambda: None)
        e = w.close()
        s = t["first"] - t0
        return dict(n_in=n, seconds=s, tok_s=n / s, j_per_token_gpu=e["gpu_j"] / n,
                    j_per_token_soc=(e["gpu_j"] + e["cpu_j"] + e["ane_j"]) / n if e["soc_valid"] else None, energy=e)
    d = int(test.split("@d")[1])
    ids = ids_for(d, seed, 1)
    w = Win()
    times = []

    def first():
        times.append(time.perf_counter())
        w.open()

    def tok():
        times.append(time.perf_counter())

    t0 = time.perf_counter()
    out = run(ids, new_tokens(test), first, tok)
    e = w.close()
    n = len(times) - 1
    if n < 1:  # the reply had at most one token: no rate (tok_s None); row['short'] marks any reply under the requested length
        return dict(n_in=len(ids), n_out=len(times), ttft_s=(times[0] - t0) if times else None, seconds=None,
                    tok_s=None, tpot_ms=None, j_per_token_gpu=None, j_per_token_soc=None, energy=e, output_ids=out)
    s = times[-1] - times[0]
    gaps = np.diff(times) * 1e3
    st = getattr(w, "steady", None)
    if st and st["sys_j"] is not None:  # tokens finished inside the steady window, and its whole-machine energy
        k = sum(1 for x in times[1:] if x > st["t0"])
        if k:
            e["steady"] = dict(seconds=st["t1"] - st["t0"], sys_j=st["sys_j"], tokens=k,
                               tok_s=k / (st["t1"] - st["t0"]), j_per_token_sys=st["sys_j"] / k)
    ex = getattr(run, "expert", None)
    if ex is not None:  # per decoded token (the first token's step counts as prefill)
        dec, gd = ex["decode"], ex.get("gpu_stream_decode") or {}
        h, m = dec.get("hits", 0) + gd.get("hits", 0), dec.get("misses", 0) + gd.get("misses", 0)
        ex["decode_bytes_per_token"] = (dec.get("bytes_read", 0) + gd.get("bytes_read", 0)) / n
        ex["decode_hit_rate"] = h / (h + m) if h + m else None
        run.expert = None
    return dict(n_in=len(ids), n_out=len(times), ttft_s=times[0] - t0, seconds=s, tok_s=n / s, tpot_ms=1e3 * s / n,
                expert=ex,
                itl_p50_ms=float(np.median(gaps)), itl_p90_ms=float(np.percentile(gaps, 90)),
                itl_p99_ms=float(np.percentile(gaps, 99)), j_per_token_gpu=e["gpu_j"] / n,
                j_per_token_soc=(e["gpu_j"] + e["cpu_j"] + e["ane_j"]) / n if e["soc_valid"] else None, energy=e,
                output_ids=out)


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=["mj", "mlx-lm", "ollama", "openai"])
    ap.add_argument("--api-url", default="http://127.0.0.1:1234", help="--engine openai: server base URL")
    ap.add_argument("--api-model", default=None, help="--engine openai: model name in requests")
    ap.add_argument("--ollama-model", default=None)
    ap.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tests", default="pp512,pp2048,tg128@d0,tg128@d2048")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=0, help="rep r uses seed seed0 + r (vary per round)")
    ap.add_argument("--stream-experts", default=None)
    ap.add_argument("--slots", type=int, default=128)
    ap.add_argument("--slot-profile", action="store_true",
                    help="per-layer slot counts with the same total as --slots, from the store's slot_profile.json")
    ap.add_argument("--map-experts", default=None)
    ap.add_argument("--attn-pack", action="store_true")
    ap.add_argument("--attn-bits", type=int, default=None)
    ap.add_argument("--lean", action="store_true")
    ap.add_argument("--map-head", default=None, help="with --lean: dir of page-aligned embedding/lm_head copies")
    ap.add_argument("--wired-mb", type=int, default=None)
    ap.add_argument("--placement", default=None, help="MoeLM families: auto / resident / mapped / stream")
    ap.add_argument("--gap-s", type=float, default=5.0)
    ap.add_argument("--chunk", type=int, default=2048, help="prefill chunk (Engine max_batch_tokens; llama.cpp ubatch)")
    ap.add_argument("--area-cap", type=int, default=None,
                    help="prefill attention area cap, chunk x context (Engine prefill_area_cap; default: the model's)")
    ap.add_argument("--label", default=None, help="engine name in the rows (suite variants such as mj-a8)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", required=True)
    return ap


def provenance():
    """Which code ran (checklist 0.17): git commit, whether src / tools had local changes, the MJ_ / MLX_ settings."""
    import subprocess

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    git = lambda *c: subprocess.run(["git", "-C", root, *c], capture_output=True, text=True).stdout.strip()
    return dict(git=git("rev-parse", "--short=10", "HEAD"), dirty=bool(git("status", "--porcelain", "src", "tools")),
                env={k: v for k, v in sorted(os.environ.items()) if k.startswith(("MJ_", "MLX_"))})


def idle_power(seconds=10.0):
    """Watts with the engine loaded and idle (checklist 0.16), subtracted from each window's energy: the GPU rail
    (IOReport) and the whole machine (SMC PSTR; IOReport's CPU / DRAM rails read 0 on this Mac)."""
    time.sleep(3.0)  # PSTR still shows the warm-up for ~2 s
    p = _power_sampler()
    with Meter() as m:
        time.sleep(seconds)
    if p:
        p.__exit__(None, None, None)
    w = m.window
    return dict(seconds=w.seconds, gpu_w=w.gpu_j / w.seconds, sys_w=p.mean_w if p else None)


def net_energy(row, idle):
    """J per token above idle: GPU rail, and the whole machine (SMC)."""
    e = row.get("energy")
    n = row.get("n_in") if "n_out" not in row else (row["n_out"] - 1 if row.get("n_out") else None)
    if not e or not n or n < 1 or not idle:
        return
    s = e["seconds"]
    row["j_per_token_gpu_net"] = (e["gpu_j"] - idle["gpu_w"] * s) / n
    if e.get("sys_j") is not None and idle.get("sys_w") is not None:
        row["j_per_token_sys"] = e["sys_j"] / n
        row["j_per_token_sys_net"] = (e["sys_j"] - idle["sys_w"] * s) / n
    st = e.get("steady")
    if st and idle.get("sys_w") is not None:  # steady state: decode only, the first STEADY_DROP_S seconds dropped
        st["j_per_token_sys_net"] = (st["sys_j"] - idle["sys_w"] * st["seconds"]) / st["tokens"]
        row["j_per_token_sys_steady_net"] = st["j_per_token_sys_net"]


def main():
    a = parser().parse_args()
    a.slots_set = "--slots" in sys.argv
    import glob

    path = glob.glob(os.path.expanduser(a.model))[0]
    t = time.perf_counter()
    run, mx = {"mj": make_mj, "mlx-lm": make_mlxlm, "ollama": make_ollama, "openai": make_openai}[a.engine](path, a)
    load_s = time.perf_counter() - t
    mx.set_cache_limit(1 << 30)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    f = open(a.out, "a")
    prov = provenance()
    for test in a.tests.split(","):
        one(run, mx, test, 10_000 + a.seed0)  # warm-up, untimed
        mx.clear_cache()
        idle = idle_power()  # the loaded engine (or server) at rest, just before this test's timed reps
        for r in range(a.reps):
            time.sleep(a.gap_s)
            row = one(run, mx, test, a.seed0 + r)
            net_energy(row, idle)
            row.update(prov=prov, idle_w=idle)
            row.update(engine=a.label or a.engine, tag=a.tag, model=path, test=test, rep=r, seed=a.seed0 + r, load_s=load_s,
                       mlx_peak_gb=mx.get_peak_memory() / 1e9, time=time.strftime("%Y-%m-%dT%H:%M:%S"),
                       config=dict(stream_experts=a.stream_experts, slots=a.slots if a.stream_experts else None,
                                   slot_profile=a.slot_profile,
                                   map_experts=a.map_experts, attn_pack=a.attn_pack, attn_bits=a.attn_bits, lean=a.lean,
                                   map_head=a.map_head,
                                   wired_mb=a.wired_mb, chunk=a.chunk, gpu_stream=os.environ.get("MJ_GPU_STREAM"),
                                   mlock=os.environ.get("MJ_MLOCK"), locked_gb=getattr(a, "locked_gb", None),
                                   rows_parallel=os.environ.get("MJ_ROWS_PARALLEL", "1"),
                                   fast_synch=os.environ.get("MLX_METAL_FAST_SYNCH")))
            if a.engine in ("ollama", "openai"):
                row["server"] = getattr(run, "server", None)
                sv = row["server"] or {}
                if sv.get("error") or (a.engine == "openai" and not sv.get("usage") and not sv.get("chunks")):
                    # the server failed (an error event, or no usage and no text at all): no rate, never a number
                    row.update(invalid=sv.get("error") or "no usage and no streamed text", tok_s=None,
                               j_per_token_gpu=None)
                row["short"] = "n_out" in row and row["n_out"] < new_tokens(test)
            pt = (((row.get("server") or {}).get("usage") or {}).get("prompt_tokens")
                  if a.engine == "openai" else None)
            if pt and "n_out" not in row and pt != row["n_in"]:  # prefill: the tokens the server actually encoded
                row.update(n_in_sent=row["n_in"], n_in=pt, tok_s=pt / row["seconds"],
                           j_per_token_gpu=row["energy"]["gpu_j"] / pt)
            if a.engine == "openai" and "n_out" in row:
                ct = ((row["server"] or {}).get("usage") or {}).get("completion_tokens")
                if ct and ct != row["n_out"] and ct > 1 and row["seconds"]:  # several tokens per stream chunk: count the server's tokens
                    row.update(tok_s=(ct - 1) / row["seconds"], tpot_ms=1e3 * row["seconds"] / (ct - 1),
                               n_out=ct, short=ct < new_tokens(test))
            f.write(json.dumps(row) + "\n")
            f.flush()
            if row["tok_s"] is None:
                err = (row.get("server") or {}).get("error")  # a server error reply has no n_out
                print(f"{a.tag or a.engine} {test} rep {r}: reply of {row.get('n_out', 0)} token(s), no rate"
                      + (f" (error: {str(err)[:120]})" if err else ""), flush=True)
            else:
                print(f"{a.tag or a.engine} {test} rep {r}: {row['tok_s']:.2f} tok/s"
                      + (f", ttft {row['ttft_s']:.2f} s" if row.get("ttft_s") is not None else "")
                      + f", {1e3 * row['j_per_token_gpu']:.0f} mJ/token GPU", flush=True)
            mx.clear_cache()


if __name__ == "__main__":
    main()
