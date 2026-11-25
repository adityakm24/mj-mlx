# Copyright 2026 the MilliJoule authors
# SPDX-License-Identifier: Apache-2.0
"""mla_decode_attention equals SDPA over the gathered latent rows (the path it replaces), with and without the fused
cache write, across rows of different lengths in scattered blocks."""
import mlx.core as mx
import numpy as np
import pytest

from mj_mlx.mla_decode import mla_decode_attention


def _setup(B, H, DK, kv_lens, bs=16, seed=0, dtype=mx.bfloat16):
    rng = np.random.RandomState(seed)
    nblk = [(n + bs - 1) // bs for n in kv_lens]
    total = sum(nblk) + 3
    perm = rng.permutation(total)  # scattered blocks
    max_blocks = max(nblk)
    bt = np.full((B, max_blocks), -1, np.int32)
    k = 0
    for b, n in enumerate(nblk):
        bt[b, :n] = perm[k:k + n]
        k += n
    pool = mx.array(rng.randn(total * bs, 1, DK).astype(np.float32)).astype(dtype)
    q = mx.array(rng.randn(B, H, DK).astype(np.float32)).astype(dtype)
    return q, pool, mx.array(bt), mx.array(np.array(kv_lens, np.int32))


def _ref(q, pool, bt, kv_lens, scale, bs, dv):
    """SDPA over the gathered rows, in float32 (the bf16 SDPA path rounds its scores: 2-4x the kernel's error)."""
    q, pool = q.astype(mx.float32), pool.astype(mx.float32)
    B = q.shape[0]
    K = int(np.array(kv_lens).max())
    pos = mx.arange(K)
    slots = mx.take(bt, pos // bs, axis=1) * bs + pos % bs
    kk = pool[mx.maximum(slots, 0)].transpose(0, 2, 1, 3)  # [B, 1, K, DK]
    mask = (pos[None, :] < kv_lens[:, None])[:, None, None, :]
    o = mx.fast.scaled_dot_product_attention(q[:, :, None, :], kk, kk, scale=scale, mask=mask)[:, :, 0, :]
    return o[..., :dv]


def _close(a, b, tol=1.5e-2):
    a, b = np.array(a.astype(mx.float32)), np.array(b.astype(mx.float32))
    assert np.abs(a - b).max() < tol, np.abs(a - b).max()


@pytest.mark.parametrize("kv_lens", [[1], [257], [1, 300, 1000], [5000, 16, 2049]])
def test_matches_sdpa_over_gathered_rows(kv_lens):
    B, H, DK, DV, bs = len(kv_lens), 16, 576, 512, 16
    q, pool, bt, kl = _setup(B, H, DK, kv_lens)
    scale = 1.0 / np.sqrt(192)
    out = mla_decode_attention(q, pool, bt, kl, scale, bs, max(kv_lens), DV)
    _close(out, _ref(q, pool, bt, kl, scale, bs, DV))


def test_other_head_counts_and_dims():
    for H, DK, DV in [(8, 576, 512), (128, 576, 512), (4, 96, 64)]:
        q, pool, bt, kl = _setup(2, H, DK, [700, 33], seed=H)
        out = mla_decode_attention(q, pool, bt, kl, 0.1, 16, 700, DV)
        _close(out, _ref(q, pool, bt, kl, 0.1, 16, DV))


def test_fused_write_equals_writing_first():
    kv_lens, bs, DK, DV = [300, 1, 1025], 16, 576, 512
    q, pool, bt, kl = _setup(3, 16, DK, kv_lens, seed=3)
    rng = np.random.RandomState(9)
    new = mx.array(rng.randn(3, DK).astype(np.float32)).astype(mx.bfloat16)
    btn = np.array(bt)
    slots = [int(btn[b, (n - 1) // bs]) * bs + (n - 1) % bs for b, n in enumerate(kv_lens)]
    written = mx.array(np.array(pool.astype(mx.float32))).astype(mx.bfloat16)  # a copy, new rows stored first
    written[mx.array(slots)] = new[:, None, :]
    ref = _ref(q, written, bt, kl, 0.08, bs, DV)
    out = mla_decode_attention(q, pool, bt, kl, 0.08, bs, max(kv_lens), DV, new_rows=new)
    mx.eval(out)
    _close(out, ref)
    _close(pool[mx.array(slots)][:, 0, :], new, tol=1e-6)  # the pool now holds them
