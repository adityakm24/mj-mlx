# Copyright 2026 the MilliJoule authors
# SPDX-License-Identifier: Apache-2.0
"""On-device sampling: temperature, top-k, top-p, with randomness from (seed, position).

Randomness is a counter-based hash of (seed, position of the new token, vocab id), turned into Gumbel noise:
token = argmax(logits / T + G) over the kept set. The same (seed, position) always gives the same draw, so a request
that is preempted and recomputed samples the same tokens again (the scheduler's tight-pool-equals-roomy-pool rule
holds with sampling on). Greedy rows (temperature 0) take the argmax.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np


def _mix32(x: mx.array) -> mx.array:
    """murmur3 finalizer on uint32 (wrapping multiply)."""
    x = x ^ (x >> 16)
    x = x * mx.array(0x7FEB352D, mx.uint32)
    x = x ^ (x >> 15)
    x = x * mx.array(0x846CA68B, mx.uint32)
    return x ^ (x >> 16)


def uniform(seed32: mx.array, pos: mx.array, V: int) -> mx.array:
    """[S, V] uniforms in (0, 1), a pure function of (seed, position, vocab id)."""
    row = _mix32(seed32 ^ _mix32(pos.astype(mx.uint32) * mx.array(0x9E3779B1, mx.uint32)))
    col = mx.arange(V, dtype=mx.uint32) * mx.array(0x85EBCA77, mx.uint32)
    h = _mix32(row[:, None] ^ col[None, :])
    return ((h >> 8).astype(mx.float32) + 0.5) * (1.0 / (1 << 24))


def sample(logits: mx.array, temperature: np.ndarray, top_p: np.ndarray, top_k: np.ndarray, seed: np.ndarray,
           pos: np.ndarray) -> mx.array:
    """logits [S, V] float32 -> tokens [S] (uint32). Per-row parameters are host arrays."""
    S, V = logits.shape
    greedy = mx.argmax(logits, axis=-1)
    if not (temperature > 0).any():
        return greedy.astype(mx.uint32)
    t = mx.array(np.where(temperature > 0, temperature, 1.0).astype(np.float32))[:, None]
    lg = logits / t
    need_k = (top_k > 0) & (top_k < V)
    need_p = top_p < 1.0
    if need_k.any() or need_p.any():
        srt = mx.sort(lg, axis=-1)[:, ::-1]  # descending
        cut = mx.full((S,), -mx.inf)
        if need_k.any():
            kk = mx.array(np.where(need_k, top_k - 1, V - 1).astype(np.int32))
            kth = mx.take_along_axis(srt, kk[:, None], axis=-1)[:, 0]
            cut = mx.maximum(cut, mx.where(mx.array(need_k), kth, -mx.inf))
        if need_p.any():
            p = mx.softmax(srt, axis=-1)
            # keep the smallest prefix whose mass reaches top_p: entry j kept iff mass before it < top_p
            before = mx.cumsum(p, axis=-1) - p
            keep = before < mx.array(top_p.astype(np.float32))[:, None]
            nkeep = mx.maximum(keep.sum(axis=-1), 1)
            pth = mx.take_along_axis(srt, (nkeep - 1)[:, None], axis=-1)[:, 0]
            cut = mx.maximum(cut, mx.where(mx.array(need_p), pth, -mx.inf))
        lg = mx.where(lg >= cut[:, None], lg, -mx.inf)
    seed32 = mx.array(((seed ^ (seed >> np.uint64(32))) & np.uint64(0xFFFFFFFF)).astype(np.uint32))
    u = uniform(seed32, mx.array(pos.astype(np.int32)), V)
    g = -mx.log(-mx.log(u))
    drawn = mx.argmax(lg + g, axis=-1)
    return mx.where(mx.array(temperature > 0), drawn, greedy).astype(mx.uint32)
