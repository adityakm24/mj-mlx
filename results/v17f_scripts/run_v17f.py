"""Campaign v17f (2026-11-23, user's standing OK until ~16:00 IST), after v17e. run_v17.run_steps: STOP_AT, the 30 GB
run swap budget, tripped configs skipped.

The gpt-oss load stages (v17e, RESULTS 1zzc) put 0.9-1.0 s of every load in quantizing the candidate head and 0.7 /
3.4 s in 8-bit / packed attention: arrays derived from the checkpoint the same way every time. Branch load-timing
(../mj-mlx-wt-load, its own tools and src) keeps them in a cache (MJ_PREP_CACHE):
- prep_write: one load per arm with the cache, which writes it (8-bit arm, exact arm).
- prep_tokens: real-model check, 96 greedy tokens on 3 chat prompts, no cache vs cache (reading), per arm: identical?
- loadprof: load profile, 3 ABAB rounds, no cache vs cache, 8-bit arm pinned in the background and exact arm unpinned.
- coldstart: tools/mlx_cold_start.py, mj-a8wb vs mj-a8wbc (the cache) vs llama.cpp, 3 starts x 2 rounds, new folder.

    .venv/bin/python results/v17f_scripts/run_v17f.py
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "v17_scripts"))
import run_v17 as V  # noqa: E402

ROOT, PY = V.ROOT, V.PY
LOG = os.path.join(ROOT, "results", "v17f_scripts", "log")
WT = os.path.abspath(os.path.join(ROOT, "..", "mj-mlx-wt-load"))
CACHE = "~/.cache/millijoule/prep"
A8 = json.dumps(dict(map_experts=f"{V.C}/gptoss20b-experts-mapped", attn_bits=8, lean=True, wired_mb=1000))
ARMS = (("a8wb", A8, "bg"), ("x-nomlock", V.GPTOSS_X, "0"))
LP = os.path.join(ROOT, "results", "load_profile_v1", "gptoss_prepcache.jsonl")


def lp(log, lab, kw, mlock, cache):
    env = {"MJ_MLOCK": mlock}
    if cache:
        env["MJ_PREP_CACHE"] = CACHE
    return V.sh([PY, os.path.join(WT, "tools", "mlx_load_profile.py"), "--model", V.MLX["gptoss"], "--mj-kw", kw,
                 "--label", lab, "--materialize", "--out", LP], log, env=env)


def step_prep_write(log):
    rc = 0
    for lab, kw, mlock in ARMS:
        rc |= lp(log, f"{lab}-cache-write", kw, mlock, True)
    return rc


def step_prep_tokens(log):
    rc = 0
    for lab, kw, _ in ARMS:
        rc |= V.sh([PY, os.path.join(WT, "tools", "mlx_prep_cache_check.py"), "--model", V.MLX["gptoss"], "--mj-kw",
                    kw, "--cache", CACHE, "--out", f"results/load_profile_v1/gptoss_prepcache_tokens_{lab}.json"], log)
    return rc


def step_loadprof(log):
    rc = 0
    for r in range(3):
        for lab, kw, mlock in ARMS:
            order = (False, True) if r % 2 == 0 else (True, False)
            for cache in order:
                rc |= lp(log, f"{lab}-{'cache' if cache else 'nocache'}", kw, mlock, cache)
    return rc


def step_coldstart(log):
    return V.sh([PY, os.path.join(WT, "tools", "mlx_cold_start.py"), "--models", "gptoss", "--engines",
                 "mj-a8wb,mj-a8wbc,llama.cpp", "--reps", "3", "--rounds", "2", "--out", "results/cold_start_v3"], log)


STEPS = [("prep_write", step_prep_write), ("prep_tokens", step_prep_tokens), ("loadprof", step_loadprof),
         ("coldstart", step_coldstart)]

if __name__ == "__main__":
    V.run_steps(STEPS, LOG, "V17F_DONE")
