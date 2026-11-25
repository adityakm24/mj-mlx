# Copyright 2026 the MilliJoule authors
# SPDX-License-Identifier: Apache-2.0
"""Campaign v18: GLM-4.5-Air (106B, mlx-community 4-bit, 65.4 GB) on the 16 GB M4, ours only (user, 2026-11-23:
GLM-4.5-Air priority, speed and long context without swap; daily swap cap lifted "for GLM today", the per-tool
20 GB and per-run 30 GB stops kept; no rival on GLM without a separate OK).

    MJ_SWAP_DAILY_OK=1 .venv/bin/python results/v18_scripts/run_v18.py prep     # before the user's restart
    MJ_SWAP_DAILY_OK=1 .venv/bin/python results/v18_scripts/run_v18.py measure  # after it

prep: repack the routed experts into a stream store (~56 GB written once, SSD writes not swap).
measure: (1) same-weights check first: ours streamed vs mlx-lm's own GLM code layer by layer, teacher-forced, the
12 chat-check prompts and 2 random 128-token sequences; (2) a slot profile from placement.CALIB_TEXT (once);
(3) suite: mj (slots per layer from memory) vs mj-u (the same total, split by the profile), pp512 / tg128@d0 /
tg128@d2048, 3 reps, 2 ABAB rounds, expert hit rate and SSD bytes per token in every row; (4) long-context
retrieval at 4k and 8k, 4 documents each.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "v17_scripts"))
import run_v17 as V  # noqa: E402

ROOT, PY = V.ROOT, V.PY
GLM = V.g(V.MODELS["glm_air"]["mlx"])
STORE = os.path.expanduser("~/.cache/mj-mlx/mlx-community--GLM-4.5-Air-4bit-experts")
D = "results/glm_air_v1"


def step_repack(log):
    if os.path.exists(os.path.join(STORE, "index.json")):
        return 0
    return V.sh([PY, "tools/mlx_repack_experts.py", "--model", GLM, "--out", STORE], log)


def step_refcheck(log):
    kw = json.dumps(dict(placement="auto"))
    return V.sh([PY, "tools/mlx_layer_stream_ref.py", "--model", GLM, "--mj-kw", kw, "--chat", "--random", "2x128",
                 "--out", f"{D}/refcheck.json"], log, env={"MJ_GPU_STREAM": "1"})


def step_calib(log):
    code = ("import os; from mj_mlx import load_model; "
            f"load_model({GLM!r}, placement='auto'); print('profile', os.path.exists({STORE!r} + '/slot_profile.json'))")
    return V.sh([PY, "-c", code], log, env={"MJ_SLOT_PROFILE_AUTO": "1"})


def step_speed(log):
    out = "results/bench_suite/v18_glm_air"
    rc = V.sh([PY, "tools/mlx_bench_suite.py", "--models", "glm_air", "--engines", "mj,mj-u", "--tests",
               "pp512,tg128@d0,tg128@d2048", "--reps", "3", "--rounds", "2", "--out", out], log)
    V.sh([PY, "tools/mlx_bench_suite.py", "--summary-only", "--models", "glm_air", "--out", out], log)
    return rc


def step_qa(log):
    return V.sh([PY, "tools/mlx_longctx_qa.py", "--model", GLM, "--engine", "mj", "--placement", "auto", "--ctx",
                 "4096,8192", "--docs", "4", "--out", "results/longctx_qa_v1/glm_air_mj_4k8k.json"], log,
                env={"MJ_GPU_STREAM": "1"})


if __name__ == "__main__":
    which = sys.argv[1]
    steps = {"prep": [("repack", step_repack)],
             "measure": [("refcheck", step_refcheck), ("calib", step_calib), ("speed", step_speed),
                         ("qa", step_qa)]}[which]
    V.run_steps(steps, os.path.join(ROOT, "results", "v18_scripts", f"log_{which}"), f"V18_{which.upper()}_DONE")
