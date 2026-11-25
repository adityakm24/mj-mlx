"""Split-K (flash-decoding) paged decode attention: matches SDPA, and a row's numbers depend only on its own kv_len
(bitwise), so speculative verify rows and plain decode agree at every context length."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
mx = pytest.importorskip("mlx.core")

from mj_mlx.paged_attention import PS, paged_decode_attention  # noqa: E402


def setup(seed, B, nq, nkv, D, kv_lens, dt, bs=16):
    rs = np.random.RandomState(seed)
    mb = (max(kv_lens) + bs - 1) // bs
    nblocks = mb * B + 8
    bt = np.stack([rs.permutation(nblocks)[:mb] for _ in range(B)]).astype(np.int32)
    kp = mx.array(rs.randn(nblocks * bs, nkv, D).astype(np.float32)).astype(dt)
    vp = mx.array(rs.randn(nblocks * bs, nkv, D).astype(np.float32)).astype(dt)
    q = mx.array(rs.randn(B, nq, D).astype(np.float32)).astype(dt)
    return bt, kp, vp, q, rs


SHAPES = [(64, 8, 64), (16, 4, 256)]  # (nq, nkv, D): gpt-oss-like, Qwen3.5-9B-like


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("window", [0, 128, 300])
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("sinks_on", [True, False])
def test_splitk_matches_sdpa(dtype, window, shape, sinks_on):
    nq, nkv, D = shape
    dt = getattr(mx, dtype)
    kv_lens = [1, 17, PS, PS + 1, 700, 2100]
    B, bs = len(kv_lens), 16
    bt, kp, vp, q, rs = setup(1, B, nq, nkv, D, kv_lens, dt, bs)
    sinks = mx.array(rs.randn(nq).astype(np.float32)) if sinks_on else mx.full((nq,), -mx.inf)
    scale = 1 / np.sqrt(D)
    out = paged_decode_attention(q, kp, vp, mx.array(bt), mx.array(np.array(kv_lens, np.int32)), sinks, scale, bs,
                                 window, max(kv_lens))
    for b in range(B):
        lo = max(0, kv_lens[b] - window) if window else 0
        p = np.arange(lo, kv_lens[b])
        slots = mx.array((bt[b, p // bs] * bs + p % bs).astype(np.int32))
        kw = dict(sinks=sinks.astype(dt)) if sinks_on else {}
        ref = mx.fast.scaled_dot_product_attention(q[b][None, :, None], kp[slots].transpose(1, 0, 2)[None],
                                                   vp[slots].transpose(1, 0, 2)[None], scale=scale, **kw)[0, :, 0]
        tol = 2e-5 if dtype == "float32" else 2e-2
        np.testing.assert_allclose(np.array(out[b].astype(mx.float32)), np.array(ref.astype(mx.float32)),
                                   atol=tol, rtol=tol)


@pytest.mark.parametrize("shape", SHAPES)
def test_row_numbers_depend_only_on_own_length(shape):
    """Each row alone (max_kv = its own length) == the same row inside a batch with a much longer row (bitwise);
    rows within PS positions also == the single-pass kernel (max_kv unknown) bitwise."""
    nq, nkv, D = shape
    dt = mx.bfloat16
    kv_lens = [5, PS, PS + 1, 513, 1500, 4000]
    B, bs = len(kv_lens), 16
    bt, kp, vp, q, rs = setup(2, B, nq, nkv, D, kv_lens, dt, bs)
    sinks = mx.array(rs.randn(nq).astype(np.float32))
    kl = mx.array(np.array(kv_lens, np.int32))
    full = paged_decode_attention(q, kp, vp, mx.array(bt), kl, sinks, 0.1, bs, 0, max(kv_lens))
    single = paged_decode_attention(q, kp, vp, mx.array(bt), kl, sinks, 0.1, bs, 0, 0)
    for b in range(B):
        alone = paged_decode_attention(q[b:b + 1], kp, vp, mx.array(bt[b:b + 1]), kl[b:b + 1], sinks, 0.1, bs, 0,
                                       kv_lens[b])
        np.testing.assert_array_equal(np.array(alone[0].astype(mx.float32)), np.array(full[b].astype(mx.float32)))
        if kv_lens[b] <= PS:
            np.testing.assert_array_equal(np.array(single[b].astype(mx.float32)),
                                          np.array(full[b].astype(mx.float32)))
