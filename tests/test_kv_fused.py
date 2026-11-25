"""Fused KV write: on decode-only steps the paged decode kernel stores the step's new K/V row into the pool itself
(instead of `kp[slots] = k`, which copies the whole layer pool). Same attention output and same pool, bit for bit;
same tokens end to end."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parent))
mx = pytest.importorskip("mlx.core")

from mj_mlx import gpt_oss as G  # noqa: E402
from mj_mlx.paged_attention import PS, paged_decode_attention  # noqa: E402


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("window", [0, 128])
@pytest.mark.parametrize("shape", [(64, 8, 64), (16, 4, 256)])
def test_fused_write_matches_scatter_then_attend(dtype, window, shape):
    nq, nkv, D = shape
    dt = getattr(mx, dtype)
    rs = np.random.RandomState(3)
    kv_lens = [1, 17, PS, PS + 1, 700, 2100]  # single-pass and split-K rows
    B, bs = len(kv_lens), 16
    mb = (max(kv_lens) + bs - 1) // bs
    nblocks = mb * B + 8
    # row-exclusive blocks, as the scheduler's tables are for the blocks a decode row writes (see build_meta)
    bt = rs.permutation(nblocks)[:B * mb].reshape(B, mb).astype(np.int32)
    kp0 = rs.randn(nblocks * bs, nkv, D).astype(np.float32)
    vp0 = rs.randn(nblocks * bs, nkv, D).astype(np.float32)
    q = mx.array(rs.randn(B, nq, D).astype(np.float32)).astype(dt)
    kn = mx.array(rs.randn(B, nkv, D).astype(np.float32)).astype(dt)
    vn = mx.array(rs.randn(B, nkv, D).astype(np.float32)).astype(dt)
    sinks = mx.array(rs.randn(nq).astype(np.float32))
    pos = np.array(kv_lens) - 1
    slots = mx.array((bt[np.arange(B), pos // bs] * bs + pos % bs).astype(np.int32))
    btm, kl = mx.array(bt), mx.array(np.array(kv_lens, np.int32))
    # reference: scatter, then attend
    kr, vr = mx.array(kp0).astype(dt), mx.array(vp0).astype(dt)
    kr[slots] = kn
    vr[slots] = vn
    ref = paged_decode_attention(q, kr, vr, btm, kl, sinks, 0.125, bs, window, max(kv_lens))
    mx.eval(ref, kr, vr)
    # fused: the kernel writes
    kf, vf = mx.array(kp0).astype(dt), mx.array(vp0).astype(dt)
    mx.eval(kf, vf)
    got = paged_decode_attention(q, kf, vf, btm, kl, sinks, 0.125, bs, window, max(kv_lens), kn, vn)
    mx.eval(got)
    assert np.array_equal(np.array(got.astype(mx.float32)), np.array(ref.astype(mx.float32)))
    assert np.array_equal(np.array(kf.astype(mx.float32)), np.array(kr.astype(mx.float32)))
    assert np.array_equal(np.array(vf.astype(mx.float32)), np.array(vr.astype(mx.float32)))


def _run_engine(make, ps, new):
    eng = make()
    rids = [eng.add_request(p, n, ignore_eos=True) for p, n in zip(ps, new)]
    out = {}
    while eng.has_work():
        for f in eng.step():
            out[f[0]] = list(f[2])
    return [out[r] for r in rids], eng


@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("family", ["gpt_oss", "gpt_oss_lean", "qwen"])
def test_fused_kv_same_tokens_and_pool(family, overlap, monkeypatch):
    from hf_ref import snapshot
    from hf_ref_qwen import qwen_snapshot

    from mj_mlx import Engine, GptOss

    path = snapshot() if family.startswith("gpt_oss") else qwen_snapshot()
    if path is None:
        pytest.skip("tiny checkpoint not downloaded")
    rs = np.random.RandomState(5)
    vocab = 200000 if family.startswith("gpt_oss") else 248000
    ps = [[int(t) for t in rs.randint(0, vocab, n)] for n in (30, 7, 90, 3)]
    new = [14, 20, 9, 25]

    def make():
        if family == "qwen":
            return Engine(path, dtype=mx.float32, num_blocks=64, max_running=4, max_batch_tokens=64,
                          overlap=overlap)
        lean = family == "gpt_oss_lean"
        m = GptOss(path, dtype=mx.bfloat16 if lean else mx.float32, lean=lean)
        return Engine(model=m, num_blocks=64, max_running=4, max_batch_tokens=64, overlap=overlap)

    monkeypatch.setattr(G, "FUSED_KV", True)
    calls = {"n": 0}
    real = G.build_meta

    def counting(*a, **k):
        m = real(*a, **k)
        calls["n"] += m.fused_kv
        return m
    monkeypatch.setattr(G, "build_meta", counting)
    import mj_mlx.qwen3_5_moe as Q
    import mj_mlx.runner as R
    monkeypatch.setattr(R, "build_meta", counting)
    monkeypatch.setattr(Q, "build_meta", counting)
    got, eng_f = _run_engine(make, ps, new)
    assert calls["n"] > 0  # decode-only steps took the fused path
    monkeypatch.setattr(G, "FUSED_KV", False)
    ref, eng_r = _run_engine(make, ps, new)
    assert got == ref
    pf, pr = eng_f.runner._pool_arrays(), eng_r.runner._pool_arrays()
    flat = lambda p: [a for x in p for a in (x if isinstance(x, (list, tuple)) else [x])]
    for a, b in zip(flat(pf), flat(pr)):
        if a.ndim == 3 and a.shape == b.shape:  # KV pools (recurrent state slots may be reused differently)
            assert np.array_equal(np.array(a.astype(mx.float32)), np.array(b.astype(mx.float32)))


@pytest.mark.parametrize("lean", [False, True])
def test_fused_glue_same_hidden_states(lean, monkeypatch):
    """MJ_FUSE_GLUE (precomputed f32 norm weights, compiled residual add, bias + SwiGLU, bias * routing weight):
    the same hidden states bit for bit, prefill and decode."""
    from hf_ref import snapshot

    from mj_mlx import Engine, GptOss

    path = snapshot()
    if path is None:
        pytest.skip("tiny gpt-oss not downloaded")
    rs = np.random.RandomState(8)
    ps = [[int(t) for t in rs.randint(0, 200000, n)] for n in (40, 5)]
    outs = []
    for glue in (True, False):
        monkeypatch.setattr(G, "FUSE_GLUE", glue)
        m = GptOss(path, dtype=mx.bfloat16 if lean else mx.float32, lean=lean)
        eng = Engine(model=m, num_blocks=64, max_running=2, max_batch_tokens=64)
        x32 = []
        real = m.final_hidden

        def grab(meta, pool, real=real, x32=x32):
            r = real(meta, pool)
            x32.append(np.array(r))
            return r
        m.final_hidden = grab
        rids = [eng.add_request(p, 12, ignore_eos=True) for p in ps]
        toks = {}
        while eng.has_work():
            for f in eng.step():
                toks[f[0]] = list(f[2])
        outs.append(([toks[r] for r in rids], x32))
    assert outs[0][0] == outs[1][0]
    assert len(outs[0][1]) == len(outs[1][1]) and all(np.array_equal(a, b) for a, b in zip(outs[0][1], outs[1][1]))


def test_fused_kv_off_when_a_write_block_is_shared():
    bs = 16
    # row 0 writes position 20 (block 1 of its table = physical 7); row 1 reads physical 7 too
    bt = np.array([[3, 7], [7, 9]], np.int32)
    meta = G.build_meta(np.zeros(2, np.uint32), np.array([20, 25], np.int32), np.array([20, 25], np.int32),
                        np.array([1, 1], np.int32), bt, np.array([0, 1], np.int32), bs, 128, None, True)
    assert not meta.fused_kv
    bt2 = np.array([[3, 7], [5, 9]], np.int32)
    meta2 = G.build_meta(np.zeros(2, np.uint32), np.array([20, 25], np.int32), np.array([20, 25], np.int32),
                         np.array([1, 1], np.int32), bt2, np.array([0, 1], np.int32), bs, 128, None, True)
    assert meta2.fused_kv == G.FUSED_KV
