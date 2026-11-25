"""Campaign v17e (2026-11-23, user's standing OK until ~16:00 IST), after v17d. Same rules as run_v17.py.

- qa_llama128: v17c's llama-server at ubatch 256 answered 12 / 12 at 4k and 8k, but at 16k ran out of Metal memory
  on the 3rd document and failed every request after it (the backend stays in an error state). The capacity rule's
  next step: batch = ubatch 128, same precision, 16k only.
- loadstages: gpt-oss load split into stages (MJ_LOAD_TIMING=1, branch load-timing at ../mj-mlx-wt-load: its own
  tools and src), 2 runs each of the 8-bit arm pinned before serving, pinned in the background, and the exact arm
  unpinned.
- uneven3: the fresh A/B that 1zy needs (BENCH_PROTOCOL D, paired rule; fixed in RESULTS 1zy before this run):
  Qwen3.6-35B, mj vs mj-u96, tg128@d0 and tg128@d2048, 5 timed reps, 6 new ABAB rounds (7-12, new seeds). Per
  test, a lead when the paired 95% CI over the 6 round ratios excludes 0 and at least 5 of 6 rounds go the same way.

    .venv/bin/python results/v17e_scripts/run_v17e.py
"""

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "v17_scripts"))
sys.path.insert(0, os.path.join(HERE, "..", "v17b_scripts"))
sys.path.insert(0, os.path.join(HERE, "..", "v17c_scripts"))
import run_v17 as V  # noqa: E402
import run_v17c as Cc  # noqa: E402

ROOT, PY = V.ROOT, V.PY
LOG = os.path.join(ROOT, "results", "v17e_scripts", "log")
WT = os.path.join(ROOT, "..", "mj-mlx-wt-load")


def step_qa_llama128(log):
    s = Cc.ServerUb(V.GGUF["gptoss"], log + ".server", 17152, 128)
    try:
        ok = Cc.warm_ok()
        with open(log, "a") as f:
            f.write(f"\n[ubatch 128] short request {'ok' if ok else 'failed'}\n")
        if not ok:
            return 1
        return V.qa(log, "gptoss", "llama-server", "results/longctx_qa_v1/gptoss_llama_ub128_16k.json", "16384")
    finally:
        s.stop()
        time.sleep(20)


def step_loadstages(log):
    a8 = json.dumps(dict(map_experts=f"{V.C}/gptoss20b-experts-mapped", attn_bits=8, lean=True, wired_mb=1000))
    out = os.path.join(ROOT, "results", "load_profile_v1", "gptoss_stages.jsonl")
    rc = 0
    for r in range(2):
        for lab, kw, mlock in (("a8w", a8, "1"), ("a8wb", a8, "bg"), ("x-nomlock", V.GPTOSS_X, "0")):
            rc |= V.sh([PY, os.path.join(WT, "tools", "mlx_load_profile.py"), "--model", V.MLX["gptoss"], "--mj-kw",
                        kw, "--label", lab, "--materialize", "--out", out], log,
                       env={"MJ_MLOCK": mlock, "MJ_LOAD_TIMING": "1"})
    return rc


def step_uneven3(log):
    out = "results/bench_suite/v17e_q35b_uneven_r7"
    rc = V.sh([PY, "tools/mlx_bench_suite.py", "--models", "q35b", "--engines", "mj,mj-u96", "--tests",
               "tg128@d0,tg128@d2048", "--reps", "5", "--rounds", "6", "--round0", "7", "--out", out], log)
    V.sh([PY, "tools/mlx_bench_suite.py", "--summary-only", "--models", "q35b", "--out", out], log)
    return rc


STEPS = [("qa_llama128", step_qa_llama128), ("loadstages", step_loadstages), ("uneven3", step_uneven3)]


def main():
    os.makedirs(LOG, exist_ok=True)
    status = {}
    for name, fn in STEPS:
        if time.strftime("%H:%M") >= V.STOP_AT and time.strftime("%H:%M") < "23:00":
            status[name] = "not started (past STOP_AT)"
            continue
        t = time.time()
        print(f"{time.strftime('%H:%M:%S')} {name} start", flush=True)
        try:
            rc = fn(os.path.join(LOG, f"{name}.log"))
            status[name] = f"rc {rc}, {time.time() - t:.0f} s"
        except Exception as e:
            status[name] = f"error {type(e).__name__}: {e}"[:300]
        print(f"{time.strftime('%H:%M:%S')} {name}: {status[name]}", flush=True)
    json.dump(status, open(os.path.join(LOG, "status.json"), "w"), indent=1)
    print("V17E_DONE", json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
