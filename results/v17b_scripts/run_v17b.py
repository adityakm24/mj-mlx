"""Campaign v17b (2026-11-23, user's standing OK until ~16:00 IST), after v17. Same rules as run_v17.py (steps in
order, a log per step, servers always stopped, nothing starts after STOP_AT, new output names). Steps:

- client: llama.cpp both ways on gpt-oss-20b, llama-bench and llama-server through our client, in one ABAB run: the
  client-overhead check (docs/benchmarks.md), on a model fast enough (~37 tok/s) for a per-token cost to show.
- dsv2srv: why llama-server's DeepSeek-V2-Lite rows were invalid in v15 (one-token replies, no usage): the server run
  directly with its log kept, our worker against it (it now records error events), plain requests besides.
- margins: DeepSeek-V2-Lite's chat check differed from mlx-lm at token 0 on one prompt (a haiku): the top-5
  first-token log-probabilities on both engines, to see whether it is a near-tie.
- uneven2: three more rounds of uneven vs uniform expert slots on Qwen3.6-35B (decode only).

    .venv/bin/python results/v17b_scripts/run_v17b.py [--only a,b] [--skip a,b]
"""

import argparse
import json
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "v17_scripts"))
import run_v17 as V  # noqa: E402  (helpers: sh, Server, suite, model paths)

ROOT, PY = V.ROOT, V.PY
LOG = os.path.join(ROOT, "results", "v17b_scripts", "log")
STOP_AT = V.STOP_AT


def step_client(log):
    return V.suite(log, "results/bench_suite/v17b_gptoss_client", "gptoss", "llama.cpp,llama-server",
                   "pp512,tg128@d0,tg128@d2048", 5, 2)


def post(path, body, timeout=600):
    req = urllib.request.Request(f"http://127.0.0.1:{V.PORT}{path}", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return dict(status=r.status, body=r.read().decode(errors="replace")[:20000])
    except urllib.error.HTTPError as e:
        return dict(status=e.code, body=e.read().decode(errors="replace")[:20000])
    except Exception as e:
        return dict(status=None, error=f"{type(e).__name__}: {e}")


def wait_free(log, pct=35, timeout=300):
    """Wait until macOS reports at least `pct`% of memory free (memory_pressure), up to `timeout` s: v17's gpt-oss
    llama-server, started 2 s after a large process exited, failed its warm-up with Metal "Insufficient Memory"."""
    import re
    import subprocess

    t0 = time.time()
    while time.time() - t0 < timeout:
        out = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout
        m = re.search(r"free percentage: (\d+)%", out)
        if m and int(m.group(1)) >= pct:
            break
        time.sleep(10)
    with open(log, "a") as f:
        f.write(f"\n[wait_free] {time.time() - t0:.0f} s, {m.group(0) if m else '?'}\n")


class Server8k(V.Server):
    """V.Server with a context sized to the task (the suite's 8k by default; v17's 20k for every task left gpt-oss's
    server without room for its warm-up), started once memory is free."""

    def __init__(self, gguf, log, ctx=8192):
        import subprocess

        wait_free(log)
        self.p = subprocess.Popen(["llama-server", "-m", gguf, "-fa", "on", "-ngl", "99", "-c", str(ctx), "-np", "1",
                                   "--no-cache-prompt", "--port", V.PORT, "--alias", "bench"],
                                  stdout=open(log, "a"), stderr=subprocess.STDOUT)
        for _ in range(600):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{V.PORT}/health", timeout=3)
                return
            except Exception:
                time.sleep(1)
        self.stop()
        raise RuntimeError("llama-server did not come up")


def step_dsv2srv(log):
    d = os.path.join(ROOT, "results", "dsv2_llama_server_debug_v1")
    os.makedirs(d, exist_ok=True)
    s = Server8k(V.GGUF["dsv2lite"], os.path.join(d, "server.log"))
    try:
        out = {}
        out["completion_short"] = post("/v1/completions", dict(model="bench", prompt="The capital of France is",
                                                               max_tokens=16, temperature=0))
        out["completion_stream_short"] = post("/v1/completions", dict(model="bench", prompt="The capital of France is",
                                                                      max_tokens=16, temperature=0, stream=True,
                                                                      stream_options={"include_usage": True}))
        out["chat_short"] = post("/v1/chat/completions", dict(model="bench", max_tokens=32, temperature=0, messages=[
            {"role": "user", "content": "What is the capital of France?"}]))
        out["completion_ignore_eos"] = post("/v1/completions", dict(model="bench", prompt="Once upon a time,",
                                                                    max_tokens=64, temperature=0, ignore_eos=True))
        json.dump(out, open(os.path.join(d, "requests.json"), "w"), indent=1)
        rc = V.sh([PY, "tools/mlx_bench_worker.py", "--engine", "openai", "--label", "llama-server", "--model",
                   V.MLX["dsv2lite"], "--tests", "pp512,tg128@d0", "--reps", "2", "--tag", "dbg", "--api-url",
                   f"http://127.0.0.1:{V.PORT}", "--api-model", "bench", "--out", os.path.join(d, "worker.jsonl")],
                  log)
        return rc
    finally:
        s.stop()


def step_margins(log):
    return V.sh([PY, "tools/mlx_first_token_margin.py", "--model", V.MLX["dsv2lite"], "--out",
                 "results/chat_check_v1/dsv2lite_first_token_margins.json"], log)


def step_uneven2(log):
    """Three more ABAB rounds of uneven vs uniform slots on Qwen3.6 (v17's three were 6 of 6 paired wins, +4.8-10%,
    with overlapping intervals): rounds 4-6, new seeds, a new folder; pooled with v17 in the write-up."""
    rc = V.sh([PY, "tools/mlx_bench_suite.py", "--models", "q35b", "--engines", "mj,mj-u96", "--tests",
               "tg128@d0,tg128@d2048", "--reps", "5", "--rounds", "3", "--round0", "4", "--out",
               "results/bench_suite/v17b_q35b_uneven_r4"], log)
    V.sh([PY, "tools/mlx_bench_suite.py", "--summary-only", "--models", "q35b", "--out",
          "results/bench_suite/v17b_q35b_uneven_r4"], log)
    return rc


def step_loadprof2(log):
    """The v17 load profile's 2-3 s of Engine construction: with every array evaluated after load_model (its own
    phase), and with the wired-limit order fix (default) against the old order (MJ_WIRED_RAISE=1)."""
    kw = json.dumps(dict(map_experts=f"{V.C}/gptoss20b-experts-mapped", attn_bits=8, lean=True, wired_mb=1000))
    rc = 0
    for r in range(2):
        for lab, mlock, raise_ in (("mlock-bg", "bg", "0"), ("no-mlock", "0", "0"), ("mlock-bg-oldwire", "bg", "1")):
            rc |= V.sh([PY, "tools/mlx_load_profile.py", "--model", V.MLX["gptoss"], "--mj-kw", kw, "--label", lab,
                        "--materialize", "--out", "results/load_profile_v1/gptoss_a8w_materialize.jsonl"], log,
                       env={"MJ_MLOCK": mlock, "MJ_WIRED_RAISE": raise_})
    return rc


def step_coldprof(log):
    """gpt-oss cold start on the cold-start tool's own path (worker make_mj, suite variant args, prewarmed files):
    phase times for mj-a8w (experts pinned before serving) and mj-a8wb (pinned in the background), then one cProfile
    run each."""
    out = "results/cold_profile_v1/gptoss.jsonl"
    rc = 0
    for r in range(2):
        for v in ("mj-a8w", "mj-a8wb"):
            rc |= V.sh([PY, "tools/mlx_cold_profile.py", "--variant", f"gptoss:{v}", "--label", v, "--out", out], log)
    for v in ("mj-a8w", "mj-a8wb"):
        rc |= V.sh([PY, "tools/mlx_cold_profile.py", "--variant", f"gptoss:{v}", "--label", v + "-prof", "--cprofile",
                    "--out", out], log)
    return rc


def step_energy2(log):
    """gpt-oss steady-state energy again, now with swap-outs counted inside every timed window (288745b): v17's
    8-bit arm blocks swapped out 1.6 / 2.2 GB each (load included), llama-server 0.01 / 0.95 GB."""
    return V.suite(log, "results/bench_suite/v17b_energy_gptoss", "gptoss", "mj-a8w,llama-server", "tg1024@d0", 3, 2)


# v17's gpt-oss chat eval, on the wired-order fix, still swapped out ~640 MB a minute: 13.3 GB of the 16 GB was wired
# (10 GB of mlocked experts, MLX's 1000 MB, the kernel), too little for the rest of a long-context or long-reply run.
# Accuracy runs do not need the pin: unpinned, the experts are file-backed pages the system can drop and re-read
# (SSD reads, not swap writes).
NO_MLOCK = {"MJ_MLOCK": "0"}


def step_chat_eval_mj(log):
    """gpt-oss chat-format MMLU / GSM8K, our exact arm, experts not pinned (v17's pinned run was stopped: swap)."""
    return V.sh([PY, "tools/mlx_lm_eval.py", "--engine", "mj", "--model", V.MLX["gptoss"], "--mj-kw", V.GPTOSS_X,
                 "--placement", "none", "--tasks", "mmlu,gsm8k", "--chat", "--every", "4", "--effort", "low", "--out",
                 "results/lm_eval_chat_v1/gptoss_mj_x_nomlock.json"], log, env=NO_MLOCK)


def step_qa_gptoss(log):
    """gpt-oss long-context retrieval again. In v17 the exact arm stopped itself at 8k (20.4 GB of swap-outs, the
    swap guard) and llama-server then ran out of Metal memory on every request (it started on a machine still full
    of swap). Since then an Engine built without wired_limit_bytes, as this tool builds one per document, no longer
    raises MLX's wired limit before dropping it for mapped experts (5e5f9e4), the suspected cause. Waits for swap
    to settle before the server."""
    d = "results/longctx_qa_v1"
    rc = V.qa(log, "gptoss", "mj", f"{d}/gptoss_mj_x_r2.json", "4096,8192,16384", V.GPTOSS_X, NO_MLOCK, "none")
    srv = Server8k(V.GGUF["gptoss"], log + ".server", ctx=17152)  # 16k documents + the template + 384 new tokens
    try:
        rc |= V.qa(log, "gptoss", "llama-server", f"{d}/gptoss_llama_r2.json", "4096,8192,16384")
    finally:
        srv.stop()
    return rc


def step_chat_llama(log):
    """gpt-oss chat-format MMLU / GSM8K and the chat check on llama-server, 8k context (v17's 20k server failed)."""
    srv = Server8k(V.GGUF["gptoss"], log + ".server")
    try:
        rc = V.sh([PY, "tools/mlx_lm_eval.py", "--engine", "llama-server", "--model", V.MLX["gptoss"], "--tasks",
                   "mmlu,gsm8k", "--chat", "--every", "4", "--effort", "low", "--out",
                   "results/lm_eval_chat_v1/gptoss_llama_r2.json"], log)
        rc |= V.sh([PY, "tools/mlx_chat_check.py", "--model", V.MLX["gptoss"], "--engines", "llama-server", "--new",
                    "256", "--out", "results/chat_check_v1/gptoss_llama_r2.json"], log)
    finally:
        srv.stop()
    return rc


STEPS = [("energy2", step_energy2), ("qa_gptoss", step_qa_gptoss), ("chat_eval_mj", step_chat_eval_mj), ("chat_llama", step_chat_llama), ("coldprof", step_coldprof), ("loadprof2", step_loadprof2), ("client", step_client), ("dsv2srv", step_dsv2srv), ("margins", step_margins), ("uneven2", step_uneven2)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--skip", default="")
    a = ap.parse_args()
    os.makedirs(LOG, exist_ok=True)
    only, skip = set(filter(None, a.only.split(","))), set(filter(None, a.skip.split(",")))
    status = {}
    for name, fn in STEPS:
        if (only and name not in only) or name in skip:
            continue
        if time.strftime("%H:%M") >= STOP_AT and time.strftime("%H:%M") < "23:00":
            status[name] = "not started (past STOP_AT)"
            continue
        t = time.time()
        log = os.path.join(LOG, f"{name}.log")
        print(f"{time.strftime('%H:%M:%S')} {name} start", flush=True)
        try:
            rc = fn(log)
            status[name] = f"rc {rc}, {time.time() - t:.0f} s"
        except Exception as e:
            status[name] = f"error {type(e).__name__}: {e}"[:300]
        print(f"{time.strftime('%H:%M:%S')} {name}: {status[name]}", flush=True)
        json.dump(status, open(os.path.join(LOG, "status.json"), "w"), indent=1)
    print("V17B_DONE", json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
