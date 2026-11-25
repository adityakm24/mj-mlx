"""Cold start per engine (checklist 0.11): load time and time to first token from a new process or an unloaded model.

For each engine, `--rounds` ABAB rounds of `--reps` starts. One start:
  mj / mj-<variant> / mlx-lm  a new Python process (tools/mlx_cold_start.py --once) imports, loads the model with the
                     suite's options for that engine, and runs one request (a 128-token prompt, 1 new token)
  llama.cpp          a new `llama-server` (-fa on, all layers on the GPU, the suite's llama_args, -c 8192): ready on
                     /health, then one streamed completion
  ollama-gguf        `ollama serve` stays up (a daemon); the model is unloaded (`ollama stop`), then one streamed
                     /api/generate loads it and answers (Ollama's own load_duration is kept)
  lmstudio-gguf / lmstudio-mlx  its server stays up; the model is unloaded, `lms load` (timed), then one request
  vllm-mlx           a new `vllm-mlx serve`: ready on /v1/models, then one streamed completion
Times, from the start of the process or of the load: `ready_s` (model loaded / server answering), `first_token_s`
(first token received). The page cache is warm for every engine: each start reads the engine's weight files first
(a disk-cold start needs `sudo purge`, which this tool does not run).

    .venv/bin/python tools/mlx_cold_start.py --models olmoe --engines mj,mlx-lm,llama.cpp,ollama-gguf --reps 3 \
        --rounds 2 --out results/cold_start/v1
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = os.path.join(ROOT, ".venv", "bin", "python")
PROMPT_LEN = 128
LLAMA_PORT = 8091


def once(argv):
    """In a fresh process: import, load, one request. Prints one JSON line (times from the parent's spawn)."""
    t_spawn = float(argv[0])
    t_py = time.time()
    os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "150")
    sys.path.insert(0, HERE)
    import mlx_bench_worker as w  # imports mlx and mj_mlx

    t_import = time.time()
    a = w.parser().parse_args(argv[1:])
    a.slots_set = "--slots" in argv
    path = glob.glob(os.path.expanduser(a.model))[0]
    run, mx = {"mj": w.make_mj, "mlx-lm": w.make_mlxlm}[a.engine](path, a)
    t_load = time.time()
    first = {}
    run(w.ids_for(PROMPT_LEN, 12345), 1, lambda: first.setdefault("t", time.time()), lambda: None)
    print(json.dumps(dict(python_s=t_py - t_spawn, import_s=t_import - t_spawn, ready_s=t_load - t_spawn,
                          first_token_s=first.get("t", time.time()) - t_spawn, mlx_peak_gb=mx.get_peak_memory() / 1e9)))


def prewarm(files):
    for f in files:
        with open(f, "rb") as fh:
            while fh.read(64 << 20):
                pass


def stream_first(url, body, timeout=900):
    """POST, return (seconds to the first streamed chunk with text, last JSON object seen)."""
    t0 = time.time()
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    first, last = None, None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode(errors="replace").strip()
            if line.startswith("data:"):
                line = line[5:].strip()
            if not line or line == "[DONE]":
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            last = obj
            text = obj.get("response") or obj.get("content") or \
                ((obj.get("choices") or [{}])[0].get("text") or ((obj.get("choices") or [{}])[0].get("delta") or {}).get("content"))
            if first is None and text:
                first = time.time() - t0
    return first, last


def wait_ok(url, tries=1200, gap=0.25):
    for _ in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(gap)
    return False


def start_once(S, key, eng, m, mlx_dir, gguf, text):
    """One cold start of `eng`. Returns a result dict."""
    if eng.startswith("mj") or eng == "mlx-lm":
        base = "mj" if eng.startswith("mj") else "mlx-lm"
        args = m.get("variant_args", {}).get(eng, m["mj_args"]) if base == "mj" else []
        env = dict(os.environ, **(m.get("variant_env", {}).get(eng, m.get("mj_env", {})) if base == "mj" else {}))
        if env.get("MJ_PREP_CACHE_FRESH") == "1":  # the first-ever load: an empty prepared-weights cache every start
            import shutil

            shutil.rmtree(os.path.expanduser(env["MJ_PREP_CACHE"]), ignore_errors=True)
        t0 = time.time()
        p = subprocess.run([PY, os.path.abspath(__file__), "--once", str(t0), "--engine", base, "--model", mlx_dir,
                            "--out", os.devnull, *args], env=env, capture_output=True, text=True)
        if p.returncode:
            return dict(error=p.stderr[-800:])
        return json.loads(p.stdout.strip().splitlines()[-1])
    if eng == "llama.cpp":
        t0 = time.time()
        srv = subprocess.Popen(["llama-server", "-m", gguf, "-fa", "on", "-ngl", "99", "-c", "8192", "--port",
                                str(LLAMA_PORT), *m["llama_args"]], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            ok = wait_ok(f"http://127.0.0.1:{LLAMA_PORT}/health")
            ready = time.time() - t0
            if not ok:
                return dict(error="llama-server did not come up")
            first, _ = stream_first(f"http://127.0.0.1:{LLAMA_PORT}/completion",
                                    dict(prompt=text, n_predict=1, stream=True, temperature=0, cache_prompt=False))
            return dict(ready_s=ready, first_token_s=ready + first if first is not None else None)
        finally:
            srv.terminate()
            srv.wait(60)
    if eng.startswith("ollama"):
        name = S.ollama_prepare(key, eng, gguf if eng == "ollama-gguf" else mlx_dir)
        subprocess.run(["ollama", "stop", name], capture_output=True)
        time.sleep(2)
        first, last = stream_first(S.OLLAMA + "/api/generate",
                                   dict(model=name, prompt=text, stream=True, raw=True, keep_alive=0,
                                        options=dict(temperature=0, num_predict=1, num_ctx=8192)))
        load = (last or {}).get("load_duration")
        subprocess.run(["ollama", "stop", name], capture_output=True)
        return dict(ready_s=load / 1e9 if load else None, first_token_s=first, server_load_s=load / 1e9 if load else None)
    if eng.startswith("lmstudio"):
        name = S.lmstudio_prepare(key, eng, gguf if eng == "lmstudio-gguf" else mlx_dir)  # loads it
        subprocess.run([S.LMS, "unload", name], capture_output=True)
        listed = json.loads(subprocess.run([S.LMS, "ls", "--json"], capture_output=True, text=True).stdout or "[]")
        mk = next((x["modelKey"] for x in listed if x.get("path", "").startswith(f"mj-bench/{name}")), name)
        t0 = time.time()
        subprocess.run([S.LMS, "load", mk, "--gpu", "max", "-c", "8192", "--parallel", "1", "--identifier", name, "-y"],
                       capture_output=True, text=True)
        ready = time.time() - t0
        first, _ = stream_first(S.LMS_URL + "/v1/completions",
                                dict(model=name, prompt=text, max_tokens=1, stream=True, temperature=0))
        subprocess.run([S.LMS, "unload", name], capture_output=True)
        return dict(ready_s=ready, first_token_s=ready + first if first is not None else None)
    if eng == "vllm-mlx":
        t0 = time.time()
        srv = S.vllm_mlx_start(mlx_dir)
        try:
            ready = time.time() - t0
            first, _ = stream_first(S.VLLM_URL + "/v1/completions",
                                    dict(model="bench", prompt=text, max_tokens=1, stream=True, temperature=0))
            return dict(ready_s=ready, first_token_s=ready + first if first is not None else None)
        finally:
            srv.terminate()
            srv.wait(60)
    raise SystemExit(f"unknown engine {eng}")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--once":
        return once(sys.argv[2:])
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", required=True)
    ap.add_argument("--engines", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--gap-s", type=float, default=10.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit(f"{a.out} exists; results are never overwritten")
    os.makedirs(a.out)
    sys.path.insert(0, HERE)
    import mlx_bench_suite as S
    from transformers import AutoTokenizer

    sys.path.insert(0, HERE)
    import mlx_bench_worker as w

    rows = []
    for key in a.models.split(","):
        m = S.MODELS[key]
        mlx_dir = sorted(glob.glob(os.path.expanduser(m["mlx"])))[0]
        gguf = (sorted(glob.glob(os.path.expanduser(m["gguf"]))) or [None])[0]
        tok = AutoTokenizer.from_pretrained(mlx_dir)
        text = w.stable_text(tok, w.ids_for(PROMPT_LEN, 12345))
        engines = a.engines.split(",")
        for rnd in range(1, a.rounds + 1):
            for eng in (engines if rnd % 2 else engines[::-1]):
                pw = [gguf if x == "GGUF" else x for x in m.get("prewarm", {}).get(eng, [])]
                files = [f for x in pw for f in glob.glob(os.path.expanduser(x))]
                if not files:  # every engine starts with its weights in the page cache
                    src = gguf if eng in ("llama.cpp", "ollama-gguf", "lmstudio-gguf") else mlx_dir
                    files = [src] if os.path.isfile(src) else sorted(glob.glob(os.path.join(src, "*.safetensors")))
                for r in range(a.reps):
                    prewarm(files)
                    time.sleep(a.gap_s)
                    res = start_once(S, key, eng, m, mlx_dir, gguf, text)
                    res.update(model=key, engine=eng, round=rnd, rep=r, time=time.strftime("%Y-%m-%dT%H:%M:%S"),
                               swap_mb=S.swap_mb())
                    rows.append(res)
                    print(json.dumps(res), flush=True)
                    with open(os.path.join(a.out, "rows.jsonl"), "a") as f:
                        f.write(json.dumps(res) + "\n")
    summarize(rows, a.out)


def summarize(rows, out):
    import numpy as np

    lines = ["| model | engine | ready s (median, min-max) | first token s (median, min-max) | n |", "| --- | --- | --- | --- | --- |"]
    for key in sorted({r["model"] for r in rows}):
        for eng in dict.fromkeys(r["engine"] for r in rows if r["model"] == key):
            sel = [r for r in rows if r["model"] == key and r["engine"] == eng and "error" not in r]
            def cell(k):
                xs = [r[k] for r in sel if r.get(k) is not None]
                return f"{np.median(xs):.2f} ({min(xs):.2f}-{max(xs):.2f})" if xs else "-"
            lines.append(f"| {key} | {eng} | {cell('ready_s')} | {cell('first_token_s')} | {len(sel)} |")
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
