# Copyright 2026 the MilliJoule authors
# SPDX-License-Identifier: Apache-2.0
"""Packaging checks: every source file carries a license header, and the owned scheduler core imports."""

import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_every_source_file_has_a_license_header():
    missing = []
    for top in ("src/mj_mlx", "crates"):
        for dirpath, dirnames, files in os.walk(os.path.join(ROOT, top)):
            dirnames[:] = [d for d in dirnames if d not in ("build", "__pycache__", "target")]
            for f in files:
                if f.endswith((".py", ".cpp", ".rs")):
                    with open(os.path.join(dirpath, f), encoding="utf-8") as fh:
                        if "Apache" not in fh.read(400):
                            missing.append(os.path.relpath(os.path.join(dirpath, f), ROOT))
    assert not missing, f"no Apache header: {missing}"


def test_scheduler_core_is_owned():
    import mj_mlx
    from mj_mlx import scheduler

    assert scheduler.Scheduler.__module__ == "mj_mlx._sched"
    assert mj_mlx.runner.scheduler is scheduler
    s = scheduler.Scheduler(block_size=4, num_blocks=8, max_running=2, max_batch_tokens=16)
    assert not s.has_work()
