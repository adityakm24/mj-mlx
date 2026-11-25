"""Campaign v17g (2026-11-23, user's standing OK until ~16:00 IST). run_v17.run_steps (STOP_AT, 30 GB run swap
budget, tripped configs skipped).

Checklist 0.11b: pinning all 10 GB of gpt-oss's experts swaps out ~2.2 GB at every load and leaves too little memory
for long contexts. MJ_MLOCK_TOP=k pins each layer's k most-used experts (expert_rank.json from
tools/mlx_expert_usage.py: top 24 of 32 took 87-96% of decode routings on held-out chat prompts, top 16 66-87%).
- warm: one load per arm so the prepared-weights cache matches this code before the timed blocks.
- pin_ab: suite, mj-a8wb (all pinned, background) vs mj-a8wb24 vs mj-a8wb16, pp512 / tg128@d0 / tg128@d2048, 5 reps,
  3 rounds. Swap per block (load included), decode and prefill, energy columns. The suite's prompts are random
  tokens, not chat: the ranking was made on chat, so this is the harder case for partial pinning.
- qa16: long-context retrieval at 16k on the exact arm with its top 16 experts pinned (MJ_MLOCK=1, MJ_MLOCK_TOP=16):
  fully pinned it swapped past 20 GB at 8k (v17); unpinned it got 12 of 12.

    .venv/bin/python results/v17g_scripts/run_v17g.py
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "v17_scripts"))
import run_v17 as V  # noqa: E402

ROOT, PY = V.ROOT, V.PY
LOG = os.path.join(ROOT, "results", "v17g_scripts", "log")
A8 = json.dumps(dict(map_experts=f"{V.C}/gptoss20b-experts-mapped", attn_bits=8, lean=True, wired_mb=1000))


def step_warm(log):
    rc = 0
    for kw, mlock in ((A8, "bg"), (V.GPTOSS_X, "0")):
        rc |= V.sh([PY, "tools/mlx_load_profile.py", "--model", V.MLX["gptoss"], "--mj-kw", kw, "--label", "warm",
                    "--out", "results/load_profile_v1/gptoss_warm_v17g.jsonl"], log, env={"MJ_MLOCK": mlock})
    return rc


def step_pin_ab(log):
    out = "results/bench_suite/v17g_gptoss_pin_top"
    rc = V.sh([PY, "tools/mlx_bench_suite.py", "--models", "gptoss", "--engines", "mj-a8wb,mj-a8wb24,mj-a8wb16",
               "--tests", "pp512,tg128@d0,tg128@d2048", "--reps", "5", "--rounds", "3", "--out", out], log)
    V.sh([PY, "tools/mlx_bench_suite.py", "--summary-only", "--models", "gptoss", "--out", out], log)
    return rc


def step_qa16(log):
    return V.qa(log, "gptoss", "mj", "results/longctx_qa_v1/gptoss_mj_x_pin16_16k.json", "16384", V.GPTOSS_X,
                {"MJ_MLOCK": "1", "MJ_MLOCK_TOP": "16"}, "none")


STEPS = [("warm", step_warm), ("pin_ab", step_pin_ab), ("qa16", step_qa16)]

if __name__ == "__main__":
    V.run_steps(STEPS, LOG, "V17G_DONE")
