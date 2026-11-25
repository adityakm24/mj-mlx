"""Repack a MoE checkpoint's routed experts into one file per layer for SSD streaming (MilliJoule MLX backend).

The format and the supported sources are described in mj_mlx.expert_repack (gpt-oss, Qwen3.5 / 3.6) and
mj_mlx.placement.repack (the general MoE families). `load_model(path, placement="auto")` does this on first use.

    .venv/bin/python tools/mlx_repack_experts.py --model <checkpoint dir> --out <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mj_mlx import expert_repack, placement  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    mt = json.load(open(os.path.join(a.model, "config.json"))).get("model_type")
    log = lambda *x: print(*x, flush=True)
    (expert_repack.repack if mt in expert_repack.FAMILIES else placement.repack)(a.model, a.out, log=log)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
