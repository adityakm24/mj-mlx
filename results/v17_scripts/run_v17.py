"""Campaign v17 (2026-11-23, user's standing OK until ~16:00 IST): Phase 0 runs after suite v16.

Steps run one after another; each is logged to results/v17_scripts/log/<step>.log, a failed step is recorded and the
next one starts, a server a step started is always stopped, and no step starts after STOP_AT. Output names are new
(results are never overwritten). Run from the repository root:

    .venv/bin/python results/v17_scripts/run_v17.py [--only a,b] [--skip a,b]
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PY = os.path.join(ROOT, ".venv", "bin", "python")
LOG = os.path.join(ROOT, "results", "v17_scripts", "log")
STOP_AT = "15:30"
sys.path.insert(0, os.path.join(ROOT, "tools"))
from mlx_bench_suite import MODELS  # noqa: E402

H = os.path.expanduser("~/.cache/huggingface/hub")
C = os.path.expanduser("~/.cache/millijoule")


def g(p):
    m = glob.glob(os.path.expanduser(p))
    return m[0] if m else None


MLX = {k: g(MODELS[k]["mlx"]) for k in ("olmoe", "dsv2lite", "gptoss", "q35b")}
GGUF = {k: g(MODELS[k]["gguf"]) for k in ("olmoe", "dsv2lite", "gptoss")}
GPTOSS_X = json.dumps(dict(map_experts=f"{C}/gptoss20b-experts-mapped", attn_pack=True, lean=True))  # the exact arm
GPTOSS_ENV = {"MJ_MLOCK": "1"}
PORT = "8092"


SWAP_STOP = {"log": None}  # set when a tool stops itself for swap (exit 3): later tools logging to that step skip
# Engine+model configs that tripped a swap stop, kept across steps and scripts. Only the user re-enables one (by
# deleting its entry). Config = the tool's --engine and the base name of its --model.
TRIPPED = os.path.join(ROOT, "results", "swap_tripped.json")
RUN_SWAP_BUDGET_GB = 30.0  # past this many GB of machine swap-outs since run_steps began: no new step, and the
RUN_S0 = {"gb": None}      # running tool is killed (sh checks every 5 s); RUN_S0 holds the counter at the start


def _cfg(cmd):
    def arg(name):
        return cmd[cmd.index(name) + 1] if name in cmd and cmd.index(name) + 1 < len(cmd) else None

    eng, model = arg("--engine"), arg("--model")
    if eng is None and "--engines" in cmd:
        eng = arg("--engines")
    if model is None and "--models" in cmd:
        model = arg("--models")
    if not (eng and model):
        return None
    hub = [part for part in model.split(os.sep) if part.startswith("models--")]  # HF cache: the repo, not the hash
    return f"{eng}:{hub[0][len('models--'):].replace('--', '/') if hub else os.path.basename(os.path.normpath(model))}"


def tripped():
    try:
        return json.load(open(TRIPPED))
    except (OSError, ValueError):
        return {}


def sh(cmd, log, env=None, timeout=None):
    """Run one tool. Exit 3 is the swap guard's stop (tools/swap_guard.py): every later call for the same step log
    is skipped, so one step cannot pass the 20 GB swap-out limit once per tool (v17d dsv2srv2 swapped 2 x 20 GB), and
    the tool's engine+model config is recorded in results/swap_tripped.json and skipped in every later step and run."""
    cfg = _cfg(cmd)
    with open(log, "a") as f:
        if SWAP_STOP["log"] == log:
            f.write(f"\n[skipped after a swap stop in this step] {' '.join(cmd)}\n")
            return 3
        if cfg and cfg in tripped():
            f.write(f"\n[skipped: {cfg} tripped a swap stop before ({tripped()[cfg]}); only the user re-enables it] "
                    f"{' '.join(cmd)}\n")
            return 3
        f.write(f"\n$ {' '.join(cmd)}\n")
        f.flush()
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=dict(os.environ, **(env or {})))
        t0 = time.time()
        while p.poll() is None:  # the run-wide swap budget, checked every 5 s inside the step as well as between steps
            if RUN_S0["gb"] is not None and (swapouts_gb() or 0) - RUN_S0["gb"] > RUN_SWAP_BUDGET_GB:
                p.kill()
                p.wait()
                f.write(f"\n[killed: run swap-outs passed {RUN_SWAP_BUDGET_GB} GB]\n")
                break
            if timeout and time.time() - t0 > timeout:
                p.kill()
                p.wait()
                raise subprocess.TimeoutExpired(cmd, timeout)
            time.sleep(5)
        rc = 3 if p.returncode and p.returncode < 0 and RUN_S0["gb"] is not None and \
            (swapouts_gb() or 0) - RUN_S0["gb"] > RUN_SWAP_BUDGET_GB else p.returncode
    if rc == 3:
        SWAP_STOP["log"] = log
        if cfg:
            t = tripped()
            t[cfg] = f"{time.strftime('%Y-%m-%dT%H:%M')} {os.path.basename(log)}"
            json.dump(t, open(TRIPPED, "w"), indent=1)
    return rc


def swapouts_gb():
    from mlx_proc_mem import swap_pages

    p = swap_pages()
    return p[1] * 16384 / 1e9 if p else None


DAILY = os.path.join(ROOT, "results", "swap_daily.json")
DAILY_SWAP_BUDGET_GB = 40.0  # no run starts once the day's tally passes this (only the user lifts it)


def daily_total(day=None):
    try:
        d = json.load(open(DAILY))
    except (OSError, ValueError):
        d = {}
    return sum(r["gb"] for r in d.get(day or time.strftime("%Y-%m-%d"), [])), d


def run_steps(steps, log_dir, done_tag):
    """The run loop every script from v17f on uses: STOP_AT; the run-wide swap budget (RUN_SWAP_BUDGET_GB of
    swap-outs since this call, machine-wide counter: no new step past it, the running tool killed, the run ended);
    and the daily budget (results/swap_daily.json, every run's swap-outs added under its date: no run starts once
    the day passes DAILY_SWAP_BUDGET_GB)."""
    os.makedirs(log_dir, exist_ok=True)
    day = time.strftime("%Y-%m-%d")
    used_today, _ = daily_total(day)
    lifted = os.environ.get("MJ_SWAP_DAILY_OK") == "1"  # the user lifted the daily cap for this run (their word)
    if used_today > DAILY_SWAP_BUDGET_GB and not lifted:
        print(f"{done_tag} not run: {used_today:.1f} GB of swap-outs today > {DAILY_SWAP_BUDGET_GB} GB daily budget "
              f"({DAILY}); only the user lifts it", flush=True)
        return {"_refused": f"daily swap {used_today:.1f} GB"}
    status, s0 = {}, swapouts_gb()
    RUN_S0["gb"] = s0
    os.environ["MJ_RUN_STEPS"] = "1"  # tools it starts (the suite) leave the daily accounting to this loop
    for name, fn in steps:
        used = (swapouts_gb() - s0) if s0 is not None else 0.0
        if time.strftime("%H:%M") >= STOP_AT and time.strftime("%H:%M") < "23:00":
            status[name] = "not started (past STOP_AT)"
            continue
        if used > RUN_SWAP_BUDGET_GB or (used_today + used > DAILY_SWAP_BUDGET_GB and not lifted):
            status[name] = (f"not started (run swap-outs {used:.1f} GB; budgets {RUN_SWAP_BUDGET_GB} GB per run, "
                            f"{DAILY_SWAP_BUDGET_GB} GB per day with {used_today:.1f} before this run)")
            print(f"{time.strftime('%H:%M:%S')} {name}: {status[name]}", flush=True)
            continue
        t = time.time()
        print(f"{time.strftime('%H:%M:%S')} {name} start (run swap-outs so far {used:.1f} GB)", flush=True)
        try:
            rc = fn(os.path.join(log_dir, f"{name}.log"))
            status[name] = f"rc {rc}, {time.time() - t:.0f} s"
        except Exception as e:
            status[name] = f"error {type(e).__name__}: {e}"[:300]
        print(f"{time.strftime('%H:%M:%S')} {name}: {status[name]}", flush=True)
    used = (swapouts_gb() - s0) if s0 is not None else None
    status["_run_swapouts_gb"] = round(used, 2) if used is not None else None
    if used is not None:
        _, d = daily_total(day)
        d.setdefault(day, []).append(dict(run=os.path.basename(os.path.dirname(log_dir)) + "/" +
                                          os.path.basename(log_dir), gb=round(used, 2),
                                          end=time.strftime("%H:%M")))
        json.dump(d, open(DAILY, "w"), indent=1)
    json.dump(status, open(os.path.join(log_dir, "status.json"), "w"), indent=1)
    print(done_tag, json.dumps(status), flush=True)
    return status


class Server:
    def __init__(self, gguf, log, extra=()):
        self.p = subprocess.Popen(["llama-server", "-m", gguf, "-fa", "on", "-ngl", "99", "-c", "20480", "-np", "1",
                                   "--no-cache-prompt", "--port", PORT, "--alias", "bench", *extra],
                                  stdout=open(log + ".server", "a"), stderr=subprocess.STDOUT)
        for _ in range(600):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3)
                return
            except Exception:
                time.sleep(1)
        raise RuntimeError("llama-server did not come up")

    def stop(self):
        self.p.terminate()
        try:
            self.p.wait(30)
        except subprocess.TimeoutExpired:
            self.p.kill()


def suite(log, out, models, engines, tests, reps, rounds):
    rc = sh([PY, "tools/mlx_bench_suite.py", "--models", models, "--engines", engines, "--tests", tests, "--reps",
             str(reps), "--rounds", str(rounds), "--out", out], log)
    sh([PY, "tools/mlx_bench_suite.py", "--summary-only", "--models", models, "--out", out], log)
    return rc


def step_uneven(log):
    sh([PY, "tools/mlx_slot_profile.py", "--trace", f"{C}/qwen36-route-trace", "--store",
        f"{C}/qwen36-35b-experts-stream"], log)
    return suite(log, "results/bench_suite/v17_q35b_uneven", "q35b", "mj,mj-u96", "pp2048,tg128@d0,tg128@d2048", 5, 3)


def step_energy(log):
    rc = suite(log, "results/bench_suite/v17_energy_olmoe", "olmoe", "mj,mlx-lm,llama-server", "tg1024@d0", 3, 2)
    return rc | suite(log, "results/bench_suite/v17_energy_gptoss", "gptoss", "mj-a8w,mj-x,llama-server", "tg1024@d0",
                      3, 2)


def step_cold(log):
    return sh([PY, "tools/mlx_cold_start.py", "--models", "gptoss", "--engines", "mj-a8w,mj-a8wb,llama.cpp",
               "--reps", "3", "--rounds", "2", "--out", "results/cold_start_v2"], log)


def step_loadprof(log):
    kw = json.dumps(dict(map_experts=f"{C}/gptoss20b-experts-mapped", attn_bits=8, lean=True, wired_mb=1000))
    rc = 0
    for r in range(2):
        for lab, mlock in (("mlock", "1"), ("mlock-bg", "bg"), ("no-mlock", "0")):
            rc |= sh([PY, "tools/mlx_load_profile.py", "--model", MLX["gptoss"], "--mj-kw", kw, "--label", lab,
                      "--out", "results/load_profile_v1/gptoss_a8w.jsonl"], log, env={"MJ_MLOCK": mlock})
    return rc


def qa(log, key, engine, out, ctx, kw=None, env=None, placement="auto"):
    cmd = [PY, "tools/mlx_longctx_qa.py", "--model", MLX[key], "--engine", engine, "--ctx", ctx, "--docs", "12",
           "--out", out]
    if kw:
        cmd += ["--mj-kw", kw]
    cmd += ["--placement", placement]
    return sh(cmd, log, env=env)


def with_server(key, log, fn):
    s = Server(GGUF[key], log)
    try:
        return fn()
    finally:
        s.stop()


def step_qa(log):
    d = "results/longctx_qa_v1"
    rc = qa(log, "gptoss", "mj", f"{d}/gptoss_mj_x.json", "4096,8192,16384", GPTOSS_X, GPTOSS_ENV, "none")
    rc |= with_server("gptoss", log, lambda: qa(log, "gptoss", "llama-server", f"{d}/gptoss_llama.json",
                                                "4096,8192,16384"))
    rc |= qa(log, "dsv2lite", "mj", f"{d}/dsv2_mj.json", "4096,8192,16384")
    rc |= qa(log, "dsv2lite", "mlx-lm", f"{d}/dsv2_mlxlm.json", "4096,8192,16384")
    rc |= qa(log, "olmoe", "mj", f"{d}/olmoe_mj.json", "2048,3584")
    rc |= qa(log, "olmoe", "mlx-lm", f"{d}/olmoe_mlxlm.json", "2048,3584")
    rc |= with_server("olmoe", log, lambda: qa(log, "olmoe", "llama-server", f"{d}/olmoe_llama.json", "2048,3584"))
    return rc


def step_chat_eval(log):
    d = "results/lm_eval_chat_v1"
    base = ["--tasks", "mmlu,gsm8k", "--chat", "--every", "4", "--effort", "low"]
    rc = sh([PY, "tools/mlx_lm_eval.py", "--engine", "mj", "--model", MLX["gptoss"], "--mj-kw", GPTOSS_X,
             "--placement", "none", *base, "--out", f"{d}/gptoss_mj_x.json"], log, env=GPTOSS_ENV)
    rc |= with_server("gptoss", log, lambda: sh([PY, "tools/mlx_lm_eval.py", "--engine", "llama-server", "--model",
                                                 MLX["gptoss"], *base, "--out", f"{d}/gptoss_llama.json"], log))
    return rc


def step_chat_check(log):
    rc = sh([PY, "tools/mlx_chat_check.py", "--model", MLX["gptoss"], "--mj-kw", GPTOSS_X, "--placement", "none",
             "--engines", "mj", "--new", "256", "--out", "results/chat_check_v1/gptoss_mj_x.json"], log, env=GPTOSS_ENV)
    rc |= with_server("gptoss", log, lambda: sh([PY, "tools/mlx_chat_check.py", "--model", MLX["gptoss"],
                                                 "--engines", "llama-server", "--new", "256", "--out",
                                                 "results/chat_check_v1/gptoss_llama.json"], log))
    return rc


STEPS = [("uneven", step_uneven), ("energy", step_energy), ("cold", step_cold), ("loadprof", step_loadprof),
         ("qa", step_qa), ("chat_eval", step_chat_eval), ("chat_check", step_chat_check)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--skip", default="")
    a = ap.parse_args()
    os.makedirs(LOG, exist_ok=True)
    for d in ("results/load_profile_v1", "results/longctx_qa_v1", "results/lm_eval_chat_v1"):
        os.makedirs(os.path.join(ROOT, d), exist_ok=True)
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
    print("V17_DONE", json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
