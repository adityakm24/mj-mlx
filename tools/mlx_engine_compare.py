"""Single-request decode, any engine on this Mac, measured the same way: one 256-token prompt, 128 new tokens.

Per run, after one unmeasured warm-up run:
- decode_ms_per_token: (time of last token - time of first token) / (tokens - 1). The prompt's prefill is excluded.
- gpu_mj_per_token: IOReport GPU-rail energy over the same first-to-last-token window / (tokens - 1). This is
  whole-GPU energy including idle, so other apps count too; Mac-only.
- decode_gap_median_ms / _max_ms: median and max time between token arrivals (a chunk of several tokens counts once).
- --diff (openai only): decode ms and mJ per token = (128-token request - 32-token request) / token difference,
  unstreamed. Used for LM Studio, whose MLX stream holds back early tokens: streaming timing understated its
  ms/token (42-44 streamed vs 54.4 by difference, 2026-11-17).
- ttft_s: request start to first token (includes the prefill of the 256 prompt tokens).

Engines:
- mj, mj-overlap, mlx-lm: in-process, on the MLX checkpoint, with the exact token ids.
- mj-eagle: in-process gpt-oss with an EAGLE-3 draft head (eagle3.py; --k drafts or --k auto = one draft or none
  by measured acceptance and step time; head in EAGLE_HEAD); greedy output equals mj's.
- mj-mtp: in-process, speculative decoding with the checkpoint's MTP head (spec_decode.py; --mtp, --k,
  --draft-vocab). Greedy output is bit-identical to mj's (checked by tools/mlx_spec_decode.py). --k auto-time /
  auto-energy: drafts per iteration chosen by spec_decode.DraftPolicy (0 .. --k-max), a fresh policy per request;
  auto-energy needs --cost-file (tools/mlx_spec_calibrate.py). Runs record k_hist (iterations per draft count) and
  j_per_iter (window energy / iterations).

Prompts (--prompt): random = 256 ids from RandomState(0) (the default; hard for any draft model: the context is
noise), chat = the checkpoint's chat template around CHAT_PROMPT (thinking off), the same ids for every engine.
- llama-server: llama.cpp's HTTP server, /completion with the exact token ids, ignore_eos, streamed.
- openai: any OpenAI-compatible server (LM Studio) on /v1/completions. It takes text, so the prompt is the
  checkpoint tokenizer's decode of the same ids. EOS cannot be ignored, so a run that stops early is recorded
  with its real token count and flagged short. Tokens are the server's usage count (a streamed chunk may carry
  several tokens); the window still runs first chunk to last chunk, so a multi-token first chunk slightly
  understates ms/token. LM Studio reuses a cached prompt across requests, so its ttft_s is not a prefill time.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "150")  # before mlx is imported: RESULTS 1p
import platform
import sys
import time
import urllib.request

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mj_mlx.energy import Meter  # noqa: E402

PROMPT_LEN, NEW = 256, 128
PROMPT = "random"
CHAT_PROMPT = "Write a Python function that returns the n-th Fibonacci number using memoization, and explain it."
_TOK_PATH = None
_MODELS = {}
EAGLE_HEAD = "~/.cache/millijoule/eagle3/nebius-eagle3.safetensors"
EAGLE_COST = (0.0323, 0.0466)  # s per iteration, 0 / 1 draft, gpt-oss-20b on M4 (RESULTS 1p: verify 1 / 2 tokens + draft)


def prompt_ids():
    if PROMPT == "chat":
        from transformers import AutoTokenizer

        ids = AutoTokenizer.from_pretrained(_TOK_PATH).apply_chat_template(
            [{"role": "user", "content": CHAT_PROMPT}], add_generation_prompt=True, enable_thinking=False,
            tokenize=True, return_dict=False)
        ids = ids["input_ids"] if hasattr(ids, "keys") else ids
        return [int(t) for t in ids]
    return [int(t) for t in np.random.RandomState(0).randint(0, 200000, PROMPT_LEN)]


class Clock:
    """Token arrival times, with the energy window opened at the first token and closed at the last."""

    def __init__(self):
        self.t_start = time.perf_counter()
        self.times = []
        self.m = Meter()

    def token(self, n=1):
        now = time.perf_counter()
        if not self.times:
            self.m.__enter__()
        self.times += [now] * n

    def close(self):
        self.m.__exit__(None, None, None)
        n = len(self.times)
        dt = self.times[-1] - self.times[0]
        w = self.m.window
        gaps = np.diff(np.unique(self.times)) if n > 1 else np.zeros(1)
        return dict(tokens=n, decode_ms_per_token=1e3 * dt / (n - 1), gpu_mj_per_token=1e3 * w.gpu_j / (n - 1),
                    decode_gap_median_ms=1e3 * float(np.median(gaps)), decode_gap_max_ms=1e3 * float(gaps.max()),
                    gpu_w=w.gpu_j / w.seconds, gpu_active_pct=w.gpu_active_pct, ttft_s=self.times[0] - self.t_start,
                    window_s=dt, gpu_j=w.gpu_j, chunks=n)


def make_policy(k, k_max, cost_file):
    """None for a fixed k (int), else a fresh DraftPolicy factory."""
    if k.isdigit():
        return None
    from mj_mlx.spec_decode import DraftPolicy

    mode = {"auto-time": "time", "auto-energy": "energy", "auto-edp": "edp"}[k]
    cost = None
    if mode == "energy":
        cost = json.load(open(os.path.expanduser(cost_file)))["j_per_iter"][:k_max + 1]
    elif mode == "edp":  # joules x seconds per iteration, from the same calibration
        c = json.load(open(os.path.expanduser(cost_file)))
        cost = [j * t for j, t in zip(c["j_per_iter"], c["s_per_iter"])][:k_max + 1]
    elif cost_file and os.path.exists(os.path.expanduser(cost_file)):  # seed the online timing with the calibration
        cost = json.load(open(os.path.expanduser(cost_file)))["s_per_iter"][:k_max + 1]
    return lambda: DraftPolicy(k_max, mode=mode, cost=cost)


def pageins_mb():
    import subprocess

    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    return int([l for l in out.splitlines() if l.startswith("Pageins")][0].split()[-1].rstrip(".")) * 16384 / 2**20


def run_inproc(which, path, mtp=None, k="2", draft_vocab=None, k_max=3, cost_file=None, load_kw=None,
               wired_mb=None):
    import mlx.core as mx

    load_kw = load_kw or {}
    wired = (wired_mb << 20) if wired_mb is not None else mx.device_info()["max_recommended_working_set_size"]
    mx.set_wired_limit(wired)
    mx.set_cache_limit(1 << 30)
    ids = prompt_ids()
    if which == "mlx-lm":
        from pathlib import Path

        from mlx_lm.generate import generate_step
        from mlx_lm.utils import load_model

        model, _ = load_model(Path(path))
        mx.eval(model.parameters())

        def one():
            c = Clock()
            for tok, _ in generate_step(mx.array(ids), model, max_tokens=NEW):
                mx.eval(tok)
                c.token()
            return c.close()
    elif which == "mj-eagle":  # gpt-oss + EAGLE-3 head (eagle3.py), k drafts (or auto-time with --cost-file s)
        from mj_mlx import load_model
        from mj_mlx.eagle3 import Eagle3Generator

        if path not in _MODELS:
            _MODELS[path] = load_model(path, **load_kw)
        model = _MODELS[path]
        auto = k == "auto"  # one draft or none, by measured first-draft acceptance (shadow drafts) and step time
        if auto:
            from mj_mlx.spec_decode import DraftPolicy

            mkpol = lambda: DraftPolicy(1, mode="time", cost=list(EAGLE_COST))
        gen = Eagle3Generator(model, os.path.expanduser(EAGLE_HEAD), k=1 if auto else int(k),
                              max_len=len(ids) + NEW + 64, policy=mkpol() if auto else None)

        def one():
            if auto:
                gen.policy = mkpol()
            p0 = pageins_mb()
            c = Clock()
            out = gen.generate(ids, NEW, c.token)
            r = c.close()
            st = gen.stats
            r.update(accept_rate=st["accepted"] / max(st["drafted"], 1), spec_iters=st["iters"],
                     pageins_mb=pageins_mb() - p0, output_ids=out)
            gen.stats = dict(iters=0, drafted=0, accepted=0)
            return r
    elif which == "mj-mtp":
        from mj_mlx import load_model
        from mj_mlx.spec_decode import SpecGenerator

        if path not in _MODELS:  # tools/mlx_spec_calibrate.py calls this once per k
            _MODELS[path] = load_model(path)
        model = _MODELS[path]
        dv = np.load(os.path.expanduser(draft_vocab)) if draft_vocab else None
        pol = make_policy(k, k_max, cost_file)
        gen = SpecGenerator(model, os.path.expanduser(mtp), k=k_max if pol else int(k), max_len=len(ids) + NEW + 64,
                            draft_vocab=dv)

        def one():
            if pol:
                gen.policy = pol()
            c = Clock()
            gen.generate(ids, NEW, c.token)
            r = c.close()
            st = gen.stats
            r.update(accept_rate=st["accepted"] / max(st["drafted"], 1), spec_iters=st["iters"], k_hist=st["k_hist"],
                     depth_tried=st["depth_tried"], depth_acc=st["depth_acc"],
                     j_per_iter=r["gpu_j"] / max(st["iters"], 1), s_per_iter=r["window_s"] / max(st["iters"], 1))
            gen.reset_stats()
            return r
    else:
        from mj_mlx import Engine, load_model

        model = load_model(path, **load_kw)

        def one():
            eng = Engine(model=model, num_blocks=64, max_running=1, max_batch_tokens=2048,
                         overlap=which == "mj-overlap", wired_limit_bytes=wired)
            eng.add_request(ids, NEW, ignore_eos=True)
            p0 = pageins_mb()
            c = Clock()
            out = []
            while eng.has_work():
                before = eng.tokens_out
                for f in eng.step():
                    out = list(f[2])
                if eng.tokens_out > before:
                    c.token(eng.tokens_out - before)
            r = c.close()
            r.update(pageins_mb=pageins_mb() - p0, output_ids=out)
            return r
    return one


def sse(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            line = line.strip()
            if line.startswith(b"data:"):
                data = line[5:].strip()
                if data == b"[DONE]":
                    return
                yield json.loads(data)


def run_llama_server(url):
    ids = prompt_ids()

    def one():
        c = Clock()
        timings = None
        p0 = pageins_mb()
        out = []
        for ev in sse(url + "/completion", dict(prompt=ids, n_predict=NEW, ignore_eos=True, stream=True,
                                               temperature=0.0, cache_prompt=False, return_tokens=True)):
            if ev.get("content") or ev.get("tokens"):
                c.token(max(1, len(ev.get("tokens") or [])))
                out += list(ev.get("tokens") or [])
            if ev.get("stop"):
                timings = ev.get("timings")
        r = c.close()
        r.update(pageins_mb=pageins_mb() - p0, output_ids=out)
        if timings:
            r["server_predicted_ms_per_token"] = timings.get("predicted_per_token_ms")
            r["server_prompt_n"] = timings.get("prompt_n")
            r["server_predicted_n"] = timings.get("predicted_n")
        return r
    return one


def run_openai(url, model_id, tok_path):
    from transformers import AutoTokenizer

    text = AutoTokenizer.from_pretrained(tok_path).decode(prompt_ids())

    def one():
        c = Clock()
        usage = None
        for ev in sse(url + "/v1/completions", dict(model=model_id, prompt=text, max_tokens=NEW, temperature=0.0,
                                                   stream=True, stream_options={"include_usage": True})):
            ch = ev.get("choices") or []
            if ch and ch[0].get("text"):
                c.token()
            if ev.get("usage"):
                usage = ev["usage"]
        r = c.close()
        if usage:  # a chunk can carry several tokens (e.g. speculative decoding): count the server's tokens
            n = int(usage["completion_tokens"])
            r["server_prompt_tokens"] = usage.get("prompt_tokens")
            r.update(tokens=n, decode_ms_per_token=1e3 * r["window_s"] / (n - 1),
                     gpu_mj_per_token=1e3 * r["gpu_j"] / (n - 1))
        r["short"] = r["tokens"] < NEW
        return r
    return one


def run_ollama(url, model_id, tok_path):
    """Ollama's native /api/generate, raw (no template: the prompt is the checkpoint tokenizer's decode of the same
    ids, chat markers included), greedy, streamed one token per line. Its OpenAI endpoint routes Qwen3.5's output
    into a reasoning field and streams empty text, so it is not used. EOS cannot be ignored (flagged short)."""
    from transformers import AutoTokenizer

    text = AutoTokenizer.from_pretrained(tok_path).decode(prompt_ids())

    def one():
        c = Clock()
        final = None
        body = json.dumps(dict(model=model_id, prompt=text, raw=True, stream=True, think=False,
                               options=dict(temperature=0, num_predict=NEW))).encode()
        req = urllib.request.Request(url + "/api/generate", body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            for line in r:
                ev = json.loads(line)
                if ev.get("response"):
                    c.token()
                if ev.get("done"):
                    final = ev
        res = c.close()
        if final:
            res["server_eval_count"] = final.get("eval_count")
            res["server_ms_per_token"] = 1e-6 * final["eval_duration"] / max(final["eval_count"], 1)
            res["server_prompt_eval_count"] = final.get("prompt_eval_count")
        res["short"] = res["tokens"] < NEW
        return res
    return one


def run_openai_diff(url, model_id, tok_path, short_n=32):
    """Decode cost by request-length difference: (128-token request - 32-token request) / token difference, for
    time and GPU energy. Needs no streaming, so it cannot be biased by how a server batches tokens into chunks."""
    from transformers import AutoTokenizer

    text = AutoTokenizer.from_pretrained(tok_path).decode(prompt_ids())

    def req(n):
        body = json.dumps(dict(model=model_id, prompt=text, max_tokens=n, temperature=0.0)).encode()
        m = Meter()
        t = time.perf_counter()
        with m:
            r = json.load(urllib.request.urlopen(
                urllib.request.Request(url + "/v1/completions", body, {"Content-Type": "application/json"}),
                timeout=600))
        return time.perf_counter() - t, m.window.gpu_j, int(r["usage"]["completion_tokens"]), m.window.seconds

    def one():
        ta, ja, na, _ = req(short_n)
        tb, jb, nb, sb = req(NEW)
        d = nb - na
        return dict(tokens=nb, short_tokens=na, decode_ms_per_token=1e3 * (tb - ta) / d,
                    gpu_mj_per_token=1e3 * (jb - ja) / d, gpu_w=jb / sb, decode_gap_median_ms=float("nan"),
                    decode_gap_max_ms=float("nan"), ttft_s=float("nan"), method="length-difference",
                    wall_short_s=ta, wall_long_s=tb, short=nb < NEW)
    return one


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=["mj", "mj-overlap", "mj-mtp", "mj-eagle", "mlx-lm", "llama-server", "openai", "ollama"])
    ap.add_argument("--prompt", choices=["random", "chat"], default="random")
    ap.add_argument("--mtp", default="~/.cache/millijoule/Qwen3.5-9B-mtp/mtp.safetensors")
    ap.add_argument("--k", default="2", help="drafts per iteration, or auto-time / auto-energy / auto-edp")
    ap.add_argument("--k-max", type=int, default=3)
    ap.add_argument("--cost-file", default="~/.cache/millijoule/Qwen3.5-9B-mtp/spec_cost.json")
    ap.add_argument("--draft-vocab", default="~/.cache/millijoule/Qwen3.5-9B-mtp/vocab65536.npy")
    ap.add_argument("--model", default=None, help="MLX checkpoint dir (in-process engines; tokenizer for openai)")
    ap.add_argument("--map-experts", default=None, help="mj engines: memory-mapped experts dir (gpt-oss, Qwen MoE)")
    ap.add_argument("--attn-pack", action="store_true", help="mj engines, gpt-oss: lossless packed attention")
    ap.add_argument("--lean", action="store_true", help="mj engines, gpt-oss: embedding and bf16 head on disk")
    ap.add_argument("--wired-mb", type=int, default=None, help="mj engines: MLX wired limit")
    ap.add_argument("--prompt-len", type=int, default=256, help="random prompt length")
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--model-id", default=None, help="openai: the server's model identifier")
    ap.add_argument("--arm", default=None, help="name recorded in the output (default: --engine)")
    ap.add_argument("--weights", default="", help="recorded: which weights the engine ran")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--diff", action="store_true", help="openai: time by request-length difference, not streaming")
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    path = glob.glob(os.path.expanduser(args.model))[0] if args.model else None
    global PROMPT, _TOK_PATH, PROMPT_LEN
    PROMPT, _TOK_PATH, PROMPT_LEN = args.prompt, path, args.prompt_len
    if args.engine == "llama-server":
        one = run_llama_server(args.url)
    elif args.engine == "ollama":
        one = run_ollama(args.url, args.model_id, path)
    elif args.engine == "openai":
        one = (run_openai_diff if args.diff else run_openai)(args.url, args.model_id, path)
    else:
        kw = {}
        if args.map_experts:
            kw["map_experts"] = os.path.expanduser(args.map_experts)
        if args.attn_pack:
            kw["attn_pack"] = True
        if args.lean:
            kw["lean"] = True
        one = run_inproc(args.engine, path, args.mtp, args.k, args.draft_vocab, args.k_max, args.cost_file, kw,
                         args.wired_mb)
    one()  # warm-up, not recorded
    runs = []
    for i in range(args.reps):
        r = one()
        r.update(rep=i)
        runs.append(r)
        print(f"{args.arm or args.engine:14s} rep {i}: {r['decode_ms_per_token']:6.2f} ms/token  "
              f"(median gap {r['decode_gap_median_ms']:6.2f}, max {r['decode_gap_max_ms']:6.1f})  {r['gpu_mj_per_token']:6.1f} mJ/token  {r['gpu_w']:4.1f} W  tokens {r['tokens']}  "
              f"ttft {r['ttft_s']:.2f}s  page-ins {r.get('pageins_mb', 0):.0f} MB", flush=True)
    out = dict(arm=args.arm or args.engine, engine=args.engine, round=args.round, weights=args.weights,
               model=path, url=args.url if args.engine in ("llama-server", "openai", "ollama") else None,
               model_id=args.model_id,
               prompt=(f"chat template around: {CHAT_PROMPT}" if PROMPT == "chat" else
                       f"{PROMPT_LEN} ids from RandomState(0).randint(0, 200000)"),
               prompt_len=len(prompt_ids()), spec=dict(k=args.k, k_max=args.k_max, mtp=args.mtp if args.engine == "mj-mtp" else EAGLE_HEAD,
                                                   draft_vocab=args.draft_vocab,
                                                   cost_file=args.cost_file if args.k in ("auto-energy", "auto-edp", "auto-time") else None)
               if args.engine in ("mj-mtp", "mj-eagle") else None, new_tokens=NEW, metric_notes=__doc__, machine=platform.platform(),
               chip="Apple M4 16 GB", runs=runs)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
