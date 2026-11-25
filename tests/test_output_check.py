"""Checklist 4.6: the lean greedy path raises on a garbage step instead of returning tokens from it."""

import os
import sys

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from mj_mlx.gpt_oss import GarbageOutputError, check_output  # noqa: E402


def rows(seed=1):
    rs = np.random.RandomState(seed)
    return rs.randn(2, 64).astype(np.float32), rs.randn(2, 16)


def test_good_rows_pass():
    check_output(*rows(0))


def test_non_finite_hidden_raises():
    x, c = rows()
    x[0, 3] = np.nan
    with pytest.raises(GarbageOutputError, match="non-finite final hidden"):
        check_output(x, c)


def test_zero_hidden_raises():
    x, c = rows()
    x[1] = 0.0
    with pytest.raises(GarbageOutputError, match="all-zero") as e:
        check_output(x, c)
    assert e.value.rows == [1]  # only the bad row is named


def test_top2_tie_does_not_raise():
    x, c = rows()
    c[0, 1] = c[0].max()  # an exact tie for first place: legitimate (RESULTS 1zz)
    c[0, 0] = c[0].max()
    check_output(x, c)


def test_non_finite_logits_raise():
    x, c = rows()
    c[0, 2] = np.inf
    with pytest.raises(GarbageOutputError, match="non-finite logits"):
        check_output(x, c)


def test_constant_logits_raise():
    x, c = rows()
    c[1] = 0.5
    with pytest.raises(GarbageOutputError, match="constant logits"):
        check_output(x, c)


def test_switch_off(monkeypatch):
    monkeypatch.setenv("MJ_CHECK_OUTPUT", "0")
    check_output(np.full((1, 8), np.nan, np.float32), np.zeros((1, 4)))


def test_lean_engine_raises_on_a_nan_step():
    """End to end on the tiny gpt-oss checkpoint: a lean model whose final hidden state is NaN stops the request
    with GarbageOutputError instead of returning tokens."""
    sys.path.insert(0, os.path.dirname(__file__))
    from hf_ref import snapshot

    from mj_mlx import Engine, GptOss

    path = snapshot()
    if path is None:
        pytest.skip("tiny gpt-oss checkpoint not downloaded")
    m = GptOss(path, dtype=mx.bfloat16, lean=True)
    if m.lm_head is not None:
        pytest.skip("tiny model too narrow for the lean head")
    eng = Engine(model=m, num_blocks=64, max_running=1)
    assert eng.generate([[1, 5, 9, 2]], 4, ignore_eos=True)  # sane model: tokens come back
    real = m.final_hidden
    m.final_hidden = lambda *a, **k: real(*a, **k) * mx.array(float("nan"))
    eng = Engine(model=m, num_blocks=64, max_running=1)
    with pytest.raises(GarbageOutputError):
        eng.generate([[1, 5, 9, 2]], 4, ignore_eos=True)
