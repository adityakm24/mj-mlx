# Copyright 2026 the MilliJoule authors
# SPDX-License-Identifier: Apache-2.0
"""Prefill attention whose memory does not grow with chunk x context.

MLX's fused attention kernel handles query/key head dims 64, 80, 96 and 128 (with the value dim equal). For other dims
(DeepSeek-V2 MLA: 192 / 128; Qwen3.5 / 3.6: 256) `mx.fast.scaled_dot_product_attention` falls back to an unfused
form that builds the whole [B, H, L, K] score matrix. At a 2048-token chunk over 16k keys that is ~1 GB per layer
(measured: peak +1052 MB for H 16, dims 192 / 128), and Metal keeps the buffers of the ~10 command buffers in flight,
so DeepSeek-V2-Lite ran out of Metal memory at 16k (RESULTS 1zi) unless the whole step chunk was cut to 512.

`sdpa_prefill` splits the queries into blocks so each block's scores stay under a budget, and leaves the step chunk
(and the expert matmuls it batches) alone. With a "causal" mask the block of queries [a, b) of the L query rows sees
keys 0 .. K - L + b - 1; MLX's "causal" mode is bottom-right aligned, so the block runs with "causal" over those keys.
An explicit mask is sliced by rows. The result is the same attention, computed in pieces.

Budget: MJ_ATTN_SCORES_MB (default 64 MB; 0 turns splitting off). At 256 MB DeepSeek-V2-Lite's 16k prefill still logged
hundreds of Metal "required > max" lines and macOS killed it twice; at 64 MB none of that, and prefill was 1-5% faster
(RESULTS 1zm). Qwen3.5-9B: the same within 0.6%.
"""
from __future__ import annotations

import os

import mlx.core as mx

FUSED_DIMS = (64, 80, 96, 128)  # mlx 0.32.4: dims the fused prefill kernel takes (measured; others materialize scores)
SCORES_MB = int(os.environ.get("MJ_ATTN_SCORES_MB", "64"))  # 64: RESULTS 1zm (256 still hit the Metal limit at 16k)


def fused(dq: int, dv: int) -> bool:
    return dq == dv and dq in FUSED_DIMS


def query_block(B: int, H: int, L: int, K: int, dq: int, dv: int, budget_mb: float | None = None) -> int:
    """Rows per query block (L when no split is needed). Scores are counted at 2 bytes, as measured."""
    budget = SCORES_MB if budget_mb is None else budget_mb
    if budget <= 0 or fused(dq, dv):
        return L
    rows = int(budget * (1 << 20)) // max(1, B * H * K * 2)
    if rows >= L:
        return L
    return max(32, rows // 32 * 32)


def sdpa_prefill(q: mx.array, k: mx.array, v: mx.array, *, scale: float, mask=None, sinks=None,
                 budget_mb: float | None = None) -> mx.array:
    """q [B, H, L, dq], k [B, Hk, K, dq], v [B, Hk, K, dv]; mask None, "causal" or an array broadcastable to
    [B, H, L, K]. Same result as mx.fast.scaled_dot_product_attention."""
    B, H, L, dq = q.shape
    K, dv = k.shape[2], v.shape[3]
    extra = {} if sinks is None else {"sinks": sinks}
    n = query_block(B, H, L, K, dq, dv, budget_mb)
    if n >= L:
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask, **extra)
    outs = []
    for a in range(0, L, n):
        b = min(L, a + n)
        qb = q[:, :, a:b]
        if isinstance(mask, str):  # "causal": these queries see keys up to K - L + b
            kv = K - L + b
            o = mx.fast.scaled_dot_product_attention(qb, k[:, :, :kv], v[:, :, :kv], scale=scale, mask="causal",
                                                     **extra)
        else:
            mb = None if mask is None else (mask if mask.shape[-2] == 1 else mask[..., a:b, :])
            o = mx.fast.scaled_dot_product_attention(qb, k, v, scale=scale, mask=mb, **extra)
        outs.append(o)
    return mx.concatenate(outs, axis=2)
