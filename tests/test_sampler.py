"""On-device sampler: determinism from (seed, position), filters, distribution, preemption reproducibility."""

import sys
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
sys.path.insert(0, str(Path(__file__).parent))
from hf_ref import snapshot  # noqa: E402

from mj_mlx import Engine  # noqa: E402
from mj_mlx.sampler import sample, uniform  # noqa: E402


def P(S, **kw):
    d = dict(temperature=np.full(S, 1.0, np.float32), top_p=np.ones(S, np.float32), top_k=np.zeros(S, np.int32),
             seed=np.arange(S, dtype=np.uint64), pos=np.arange(S, dtype=np.int32))
    d.update({k: np.asarray(v, dtype=d[k].dtype) if not np.isscalar(v) else np.full(S, v, d[k].dtype)
              for k, v in kw.items()})
    return d


def test_uniform_is_a_function_of_seed_and_position():
    a = uniform(mx.array([1, 1, 2], mx.uint32), mx.array([5, 6, 5]), 1000)
    b = uniform(mx.array([1, 1, 2], mx.uint32), mx.array([5, 6, 5]), 1000)
    a, b = np.array(a), np.array(b)
    assert (a == b).all() and (a > 0).all() and (a < 1).all()
    assert not np.allclose(a[0], a[1]) and not np.allclose(a[0], a[2])
    assert abs(a.mean() - 0.5) < 0.02


def test_greedy_and_topk1_equal_argmax():
    lg = mx.random.normal((16, 5000), key=mx.random.key(0))
    am = np.array(mx.argmax(lg, -1))
    assert (np.array(sample(lg, **P(16, temperature=0.0))) == am).all()
    assert (np.array(sample(lg, **P(16, top_k=1))) == am).all()


def test_top_p_stays_in_nucleus():
    rs = np.random.RandomState(0)
    lg = mx.array(rs.randn(64, 50).astype(np.float32) * 3)
    toks = np.array(sample(lg, **P(64, top_p=0.5, pos=rs.randint(0, 1 << 20, 64))))
    p = np.exp(np.array(lg) - np.array(lg).max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    for i, t in enumerate(toks):
        order = np.argsort(-p[i])
        before = np.cumsum(p[i][order]) - p[i][order]
        assert t in set(order[before < 0.5])


def test_frequencies_match_softmax():
    V, N, T = 6, 20000, 0.7
    base = np.array([0.5, -0.3, 1.2, 0.0, -1.0, 0.8], np.float32)
    lg = mx.array(np.tile(base, (N, 1)))
    toks = np.array(sample(lg, **P(N, temperature=T, seed=np.full(N, 7), pos=np.arange(N))))
    freq = np.bincount(toks, minlength=V) / N
    p = np.exp(base / T) / np.exp(base / T).sum()
    assert np.abs(freq - p).max() < 0.015, (freq, p)


PATH = snapshot()


@pytest.mark.skipif(PATH is None, reason="tiny gpt-oss checkpoint not downloaded")
def test_sampled_outputs_survive_preemption_and_are_seeded():
    rs = np.random.RandomState(9)
    prompts = [[int(t) for t in rs.randint(0, 200000, n)] for n in [30, 140, 7, 90, 200, 64, 15, 120]]
    kw = dict(temperature=0.9, top_p=0.95, top_k=50)

    def run(num_blocks, seed_offset=0):
        eng = Engine(PATH, dtype=mx.float32, num_blocks=num_blocks, max_running=4, max_batch_tokens=64)
        rids = [eng.add_request(p, 25, ignore_eos=True, seed=1000 + i + seed_offset, **kw)
                for i, p in enumerate(prompts)]
        out = {}
        while eng.has_work():
            for f in eng.step():
                out[f[0]] = list(f[2])
            eng.sched.check_invariants()
        return [out[r] for r in rids], eng.sched.stats()["preemptions"]

    roomy, _ = run(512)
    tight, pre = run(30)
    assert pre > 0
    assert roomy == tight
    other, _ = run(512, seed_offset=1)
    assert other != roomy
