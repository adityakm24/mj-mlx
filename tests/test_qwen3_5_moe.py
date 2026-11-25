"""Qwen3.6 hybrid MoE (Gated DeltaNet + gated attention + shared expert) on the tiny random checkpoint, float32,
vs transformers. Same tolerance as the gpt-oss gate (agreed with the user): token-identical, a divergence excused
only at a true near-tie (transformers top-1 vs top-2 gap < 1e-5)."""

import sys
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).parent))
from hf_ref_qwen import hf_greedy, hf_logits, load_hf_qwen, qwen_snapshot  # noqa: E402

from mj_mlx import Engine, Qwen35Moe  # noqa: E402
from mj_mlx.runner import StepArrays  # noqa: E402

PATH = qwen_snapshot()
pytestmark = pytest.mark.skipif(PATH is None, reason="tiny Qwen3.6 checkpoint not downloaded")
NEAR_TIE = 1e-5


@pytest.fixture(scope="module")
def hf():
    return load_hf_qwen(PATH)


def prompts(seed, lens):
    rs = np.random.RandomState(seed)
    return [[int(t) for t in rs.randint(0, 248000, n)] for n in lens]


@pytest.mark.parametrize("chunks", [[150], [64, 50, 36], [1] * 20 + [130]], ids=["whole", "chunked", "steps"])
def test_prefill_logits_match(hf, chunks):
    ids = prompts(0, [150])[0]
    ref = hf_logits(hf, ids)
    m = Qwen35Moe(PATH, dtype=mx.float32)
    pools = m.new_pools(64 * 16, 2)
    out, start = [], 0
    for n in chunks:
        toks = np.array(ids[start:start + n])
        bt = np.arange(3, 3 + (start + n + 15) // 16)[None].astype(np.int32)
        s = StepArrays(toks.astype(np.uint32), np.arange(start, start + n, dtype=np.int32), np.zeros(1, np.uint64),
                       np.array([start], np.int32), np.array([n], np.int32), np.array([True]), bt,
                       np.arange(n, dtype=np.int32), np.zeros(1, np.uint64), [], None, state_slots=np.array([1], np.int32))
        x = m.hidden(m.prepare(s, 16, True), pools)
        lg = mx.fast.rms_norm(x.astype(mx.float32), m.norm, m.cfg.rms_norm_eps) @ m.lm_head.T
        mx.eval(lg, m.pool_arrays(pools))
        out.append(np.array(lg))
        start += n
    ours = np.concatenate(out)
    assert np.abs(ours - ref).max() < 1e-6
    assert (ours.argmax(-1) == ref.argmax(-1)).all()


def test_batch1_greedy(hf):
    ps = prompts(1, [5, 40, 129, 200])
    ref = [hf_greedy(hf, p, 30)[0] for p in ps]
    eng = Engine(PATH, dtype=mx.float32, num_blocks=256, max_batch_tokens=64, max_running=1)
    assert eng.generate(ps, 30, ignore_eos=True, check=True) == ref


def test_batched_tight_equals_roomy_and_matches_hf(hf):
    lens = [3, 17, 60, 130, 7, 200, 33, 1, 90, 150, 12, 64]
    ps = prompts(3, lens)
    new = [15 + 3 * i for i in range(len(ps))]
    ref = [hf_greedy(hf, p, n) for p, n in zip(ps, new)]

    def run(num_blocks):
        eng = Engine(PATH, dtype=mx.float32, num_blocks=num_blocks, max_running=6, max_batch_tokens=96)
        rids = [eng.add_request(p, n, ignore_eos=True) for p, n in zip(ps, new)]
        out = {}
        while eng.has_work():
            for f in eng.step():
                out[f[0]] = list(f[2])
            eng.sched.check_invariants()
        return [out[r] for r in rids], eng.sched.stats()

    roomy, rs_ = run(1024)
    tight, ts_ = run(30)
    print("roomy", rs_, "\ntight", ts_)
    assert ts_["preemptions"] > 0
    assert roomy == tight
    excused = 0
    for i, (o, (r, lg)) in enumerate(zip(roomy, ref)):
        div = next((j for j in range(len(r)) if o[j] != r[j]), None)
        if div is None:
            continue
        top2 = np.sort(lg[div])[-2:]
        assert top2[1] - top2[0] < NEAR_TIE, f"seq {i} diverges at {div}"
        excused += 1
    print("near-tie excused:", excused)


def test_sampling_survives_preemption():
    ps = prompts(9, [30, 140, 7, 90, 64, 15])
    kw = dict(temperature=0.9, top_p=0.95, top_k=50)

    def run(num_blocks):
        eng = Engine(PATH, dtype=mx.float32, num_blocks=num_blocks, max_running=4, max_batch_tokens=64)
        rids = [eng.add_request(p, 20, ignore_eos=True, seed=500 + i, **kw) for i, p in enumerate(ps)]
        out = {}
        while eng.has_work():
            for f in eng.step():
                out[f[0]] = list(f[2])
        return [out[r] for r in rids], eng.sched.stats()["preemptions"]

    roomy, _ = run(512)
    tight, pre = run(12)
    assert pre > 0 and roomy == tight


def test_kernel_path_equals_reference_path():
    """Metal gated-delta kernel + vectorized conv == per-token MLX-ops path, mixed decode and prefill items."""
    m = Qwen35Moe(PATH, dtype=mx.float32)
    outs = []
    for use_kernel in (False, True):
        Qwen35Moe.use_kernel = use_kernel
        eng = Engine(model=m, num_blocks=128, max_running=4, max_batch_tokens=48)
        outs.append(eng.generate(prompts(11, [3, 70, 1, 45, 120]), 12, ignore_eos=True))
    Qwen35Moe.use_kernel = True
    assert outs[0] == outs[1]
    from mj_mlx.gated_delta import gated_delta_pool

    rs = np.random.RandomState(0)
    T, H, D = 9, 4, 128
    q, k = (mx.array(rs.randn(T, H, D).astype(np.float32) * 0.1) for _ in range(2))
    v = mx.array(rs.randn(T, H, D).astype(np.float32))
    g = mx.array(-np.abs(rs.randn(T, H)).astype(np.float32))
    b = mx.array(rs.rand(T, H).astype(np.float32))
    pool = rs.randn(4, H, D, D).astype(np.float32) * 0.1  # [slots, H, Dv, Dk]
    item_of_slot = np.array([-1, 1, 0, -1], np.int32)  # slot 2 -> item 0 (fresh), slot 1 -> item 1
    fresh = np.array([1, 0], np.uint8)
    cu = np.array([0, 4, 9], np.int32)
    out, nxt = gated_delta_pool(q, k, v, g, b, mx.array(pool), mx.array(item_of_slot), mx.array(fresh), mx.array(cu))
    nxt = np.array(nxt)
    for slot in (0, 3):  # idle slots are carried over
        np.testing.assert_array_equal(nxt[slot], pool[slot])
    for n, slot in ((0, 2), (1, 1)):
        S = np.zeros((H, D, D), np.float32) if fresh[n] else pool[slot].swapaxes(-1, -2).copy()  # [H, Dk, Dv]
        for t in range(cu[n], cu[n + 1]):
            S = S * np.exp(np.array(g[t]))[:, None, None]
            kt, vt, qt = np.array(k[t]), np.array(v[t]), np.array(q[t])
            delta = (vt - np.einsum("hkv,hk->hv", S, kt)) * np.array(b[t])[:, None]
            S = S + kt[:, :, None] * delta[:, None, :]
            np.testing.assert_allclose(np.array(out[t]), np.einsum("hkv,hk->hv", S, qt), rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(nxt[slot], S.swapaxes(-1, -2), rtol=1e-4, atol=1e-5)


def test_state_slots_sized_from_memory():
    from mj_mlx.runner import MlxRunner

    m = Qwen35Moe(PATH, dtype=mx.float32)
    free = MlxRunner.free_bytes(m)
    per = m.state_bytes_per_slot()
    assert MlxRunner.state_capacity(m, 4) == 4  # small max_running: not the binding limit
    big = MlxRunner.state_capacity(m, 10 ** 9)
    assert big == int(free * 0.5) // per  # memory-bound: half the free budget
    real_like = type("M", (), {"state_bytes_per_slot": lambda self: 63 << 20, "parameters": lambda self: []})()
    assert MlxRunner.state_capacity(real_like, 512) < 512  # real Qwen3.6 fp32 state cannot hold 512 on this Mac
    eng = Engine(model=m, max_running=8)
    assert eng.sched.num_free_state_slots == 8


def test_bf16_state_storage_runs_and_stays_close():
    ps = prompts(13, [50, 9, 120])
    outs = []
    for sd in (mx.float32, mx.bfloat16):
        eng = Engine(model=Qwen35Moe(PATH, dtype=mx.float32, state_dtype=sd), num_blocks=128, max_running=3)
        outs.append(eng.generate(ps, 20, ignore_eos=True))
    agree = np.mean([a == b for o32, o16 in zip(*outs) for a, b in zip(o32, o16)])
    assert agree > 0.9, agree
