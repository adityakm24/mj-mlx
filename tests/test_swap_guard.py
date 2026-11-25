"""The suite's swap stop rule (after suite v16, RESULTS 1zu): a block whose swap-outs pass the limit is killed."""

import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import mlx_bench_suite as S  # noqa: E402


def test_guard_kills_past_the_limit():
    p = subprocess.Popen(["sleep", "60"])
    with S.SwapGuard(lambda: [p.pid], limit_gb=-1, every=0.1) as g:  # any reading passes -1 GB
        p.wait(10)
    assert g.tripped is not None and p.returncode == -9


def test_guard_leaves_a_normal_block_alone():
    p = subprocess.Popen(["sleep", "0.3"])
    with S.SwapGuard(lambda: [p.pid], limit_gb=1e6, every=0.1) as g:
        p.wait(10)
    assert g.tripped is None and p.returncode == 0
    assert S.SWAP_ABORT_GB == float(os.environ.get("MJ_SWAP_ABORT_GB", "20"))


def test_tripped_configs_are_recorded_and_keyed_by_repo(tmp_path, monkeypatch):
    import swap_guard as G

    monkeypatch.setattr(G, "TRIPPED", str(tmp_path / "swap_tripped.json"))
    path = "/x/.cache/huggingface/hub/models--mlx-community--DeepSeek-V2-Lite-Chat-4bit-mlx/snapshots/a33a8ad"
    key = G.cfg_key("llama-server", path)
    assert key == "llama-server:mlx-community/DeepSeek-V2-Lite-Chat-4bit-mlx"
    assert G.tripped() == {}
    G.add_tripped(key, "test")
    assert key in G.tripped() and G.cfg_key("mj", path) not in G.tripped()


def test_seeded_trips_match_the_run_scripts_keys():
    """results/swap_tripped.json and run_v17's tool keys agree, so a trip recorded by either side is seen by both."""
    import swap_guard as G

    sys.path.insert(0, os.path.join(ROOT, "results", "v17_scripts"))
    import run_v17 as V

    path = "/x/.cache/huggingface/hub/models--mlx-community--DeepSeek-V2-Lite-Chat-4bit-mlx/snapshots/a33a8ad"
    assert V._cfg(["py", "t.py", "--engine", "llama-server", "--model", path]) == G.cfg_key("llama-server", path)
