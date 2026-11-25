# Copyright 2026 the MilliJoule authors
# SPDX-License-Identifier: Apache-2.0
"""sdpa_prefill: same result as one SDPA call, with smaller score buffers for unfused head dims."""
import mlx.core as mx
import numpy as np

from mj_mlx.prefill_attention import query_block, sdpa_prefill


def _qkv(B, H, Hk, L, K, dq, dv, seed=0):
    mx.random.seed(seed)
    q = mx.random.normal((B, H, L, dq)).astype(mx.bfloat16)
    k = mx.random.normal((B, Hk, K, dq)).astype(mx.bfloat16)
    v = mx.random.normal((B, Hk, K, dv)).astype(mx.bfloat16)
    return q, k, v


def _close(a, b):
    a, b = np.array(a.astype(mx.float32)), np.array(b.astype(mx.float32))
    assert np.abs(a - b).max() < 2e-2, np.abs(a - b).max()


def test_block_rows():
    assert query_block(1, 16, 2048, 16384, 128, 128) == 2048  # fused dims: never split
    n = query_block(1, 16, 2048, 16384, 192, 128)  # MLA: 64 MB of scores per block (the default)
    assert n == 128 and n % 32 == 0
    assert query_block(1, 16, 2048, 16384, 192, 128, budget_mb=256) == 512
    assert query_block(1, 16, 2048, 16384, 192, 128, budget_mb=0) == 2048


def test_causal_chunk_after_cache():
    # a 300-row chunk at the end of 1000 keys (earlier chunks cached), blocks of 32-64 rows
    q, k, v = _qkv(1, 4, 1, 300, 1000, 192, 128)
    ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.07, mask="causal")
    for mb in (1, 2):
        assert query_block(1, 4, 300, 1000, 192, 128, budget_mb=mb) < 300
        _close(sdpa_prefill(q, k, v, scale=0.07, mask="causal", budget_mb=mb), ref)


def test_explicit_mask_rows():
    B, H, L, K = 2, 4, 96, 160
    q, k, v = _qkv(B, H, H, L, K, 256, 256, seed=1)
    lens = np.array([160, 120])
    pos_q = lens[:, None] - L + np.arange(L)[None, :]
    m = (np.arange(K)[None, None, :] <= pos_q[:, :, None]) & (np.arange(K)[None, None, :] < lens[:, None, None])
    mask = mx.array(m[:, None])  # [B, 1, L, K] bool
    ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.06, mask=mask)
    assert query_block(B, H, L, K, 256, 256, budget_mb=0.1) == 32
    out = sdpa_prefill(q, k, v, scale=0.06, mask=mask, budget_mb=0.1)
    _close(out, ref)


def test_blocks_cover_every_row_once():
    # 2048 x 16384 at the default 64 MB budget: 128-row blocks, 16 SDPA calls, the same result as one call
    calls = []
    orig = mx.fast.scaled_dot_product_attention

    def spy(q, k, v, **kw):
        calls.append((q.shape[2], k.shape[2]))
        return orig(q, k, v, **kw)

    q, k, v = _qkv(1, 16, 1, 2048, 16384, 192, 128, seed=2)
    mx.fast.scaled_dot_product_attention = spy
    try:
        out = sdpa_prefill(q, k, v, scale=0.07, mask="causal")
    finally:
        mx.fast.scaled_dot_product_attention = orig
    assert len(calls) == 16 and all(n == 128 for n, _ in calls)
    assert [kv for _, kv in calls] == [16384 - 2048 + 128 * (i + 1) for i in range(16)]
    _close(out, orig(q, k, v, scale=0.07, mask="causal"))
