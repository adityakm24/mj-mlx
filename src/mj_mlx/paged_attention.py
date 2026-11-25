# Copyright 2026 the MilliJoule authors
# SPDX-License-Identifier: Apache-2.0
"""Paged decode attention for MLX as a custom Metal kernel.

One query token per sequence. Reads K/V straight from the paged pool through the block table (no gather copy),
one threadgroup per (sequence, kv head) serving all query heads of that GQA group, so every K/V byte is read
once. Supports attention sinks (pass -inf sinks for models without them) and a sliding window. Online softmax in
float32.
"""

from __future__ import annotations

import mlx.core as mx

_SRC = r"""
    constexpr int NSG = 8;
    constexpr int E = D / 32;
    const uint tg = threadgroup_position_in_grid.y;
    const uint lid = thread_position_in_threadgroup.x;
    const int sg = lid / 32;
    const int lane = lid % 32;
    const int b = tg / NKV;
    const int h = tg % NKV;
    const int max_blocks = params[0];
    const int bs = params[1];
    const int window = params[2];
    const float scale = fparams[0];
    const int kv_len = kv_lens[b];
    const int start = window > 0 ? max(0, kv_len - window) : 0;

    if (FUSED) {  // decode-only step: write this row's new K/V into the pool here (replaces a whole-pool scatter copy)
        const int pw = kv_len - 1;
        if (pw >= start && pw < kv_len) {  // uniform per threadgroup: exactly one threadgroup per (b, h) holds pw
            const int wslot = bt[b * max_blocks + pw / bs] * bs + pw % bs;
            const size_t wb = (size_t(wslot) * NKV + h) * D;
            device T* kw = (device T*)k;
            device T* vw = (device T*)v;
            for (int d = lid; d < D; d += NSG * 32) {
                kw[wb + d] = kn[(b * NKV + h) * D + d];
                vw[wb + d] = vn[(b * NKV + h) * D + d];
            }
            threadgroup_barrier(mem_flags::mem_device);
        }
    }

    float qr[G][E];
    for (int g = 0; g < G; ++g)
        for (int e = 0; e < E; ++e)
            qr[g][e] = float(q[((b * NKV + h) * G + g) * D + lane * E + e]) * scale;
    float m[G], l[G], o[G][E];
    for (int g = 0; g < G; ++g) {
        m[g] = -INFINITY; l[g] = 0.0f;
        for (int e = 0; e < E; ++e) o[g][e] = 0.0f;
    }
    for (int p = start + sg; p < kv_len; p += NSG) {
        const int slot = bt[b * max_blocks + p / bs] * bs + p % bs;
        const size_t base = (size_t(slot) * NKV + h) * D + lane * E;
        float kr[E], vr[E];
        for (int e = 0; e < E; ++e) { kr[e] = float(k[base + e]); vr[e] = float(v[base + e]); }
        for (int g = 0; g < G; ++g) {
            float s = 0.0f;
            for (int e = 0; e < E; ++e) s += qr[g][e] * kr[e];
            s = simd_sum(s);
            const float nm = max(m[g], s);
            const float f = fast::exp(m[g] - nm);
            const float w = fast::exp(s - nm);
            l[g] = l[g] * f + w;
            for (int e = 0; e < E; ++e) o[g][e] = o[g][e] * f + w * vr[e];
            m[g] = nm;
        }
    }
    // Combine the NSG partial softmaxes, one query head at a time, through an [NSG][D] buffer (fits 32 KB of
    // threadgroup memory for any G and D <= 512).
    threadgroup float tm[NSG][G];
    threadgroup float tl[NSG][G];
    threadgroup float to[NSG][D];
    if (lane == 0)
        for (int g = 0; g < G; ++g) { tm[sg][g] = m[g]; tl[sg][g] = l[g]; }
    for (int g = 0; g < G; ++g) {
        for (int e = 0; e < E; ++e) to[sg][lane * E + e] = o[g][e];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float sink = float(sinks[h * G + g]);
        float M = sink;
        for (int s = 0; s < NSG; ++s) M = max(M, tm[s][g]);
        float L = fast::exp(sink - M);
        for (int s = 0; s < NSG; ++s) L += tl[s][g] * fast::exp(tm[s][g] - M);
        for (int d = lid; d < D; d += NSG * 32) {
            float acc = 0.0f;
            for (int s = 0; s < NSG; ++s) acc += to[s][d] * fast::exp(tm[s][g] - M);
            out[((b * NKV + h) * G + g) * D + d] = T(acc / L);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
"""

_kernel = mx.fast.metal_kernel(
    name="mj_paged_decode_attention",
    input_names=["q", "k", "v", "bt", "kv_lens", "sinks", "params", "fparams"],
    output_names=["out"],
    source=_SRC.replace("if (FUSED) {", "if (false) {", 1).replace("kn[", "q[").replace("vn[", "q["),
)
# Fused KV write (decode-only steps, MJ_KV_FUSED): the kernel also stores the step's new K/V row (kn, vn) into the
# pool in place, before reading it. MLX sees the pool only as an input, so ordering relies on the step's data
# dependencies: nothing else reads this layer's pool within the step, and the next step's attention depends on this
# step's output. RESULTS 2026-11-19: replaces `kp[slots] = k`, which copied the whole layer pool every step.
_kernel_f = mx.fast.metal_kernel(
    name="mj_paged_decode_attention_fkv",
    input_names=["q", "k", "v", "bt", "kv_lens", "sinks", "params", "fparams", "kn", "vn"],
    output_names=["out"],
    source=_SRC.replace("if (FUSED) {", "if (true) {", 1),
)


# Split-K (flash-decoding) for long contexts: one threadgroup per (sequence, kv head, PS-position partition) instead
# of one per (sequence, kv head). B = 1 decode on a 4-kv-head model ran 4 threadgroups on a 10-core GPU, walking the
# whole context serially (Qwen3.5-9B at 2048 tokens: 2.1 ms/token slower than at 0, vs 0.3 for mlx-lm).
# Exactness: a row's numbers depend only on its own kv_len (never on the batch's maximum or the partition count).
# Rows spanning <= PS positions are computed by the single-pass math above, bit for bit (pass 1 writes them
# directly); longer rows always use the partitioned math, partitions aligned to the row's start. Empty partitions
# contribute exact zeros. So speculative verify rows match plain decode at every length.
PS = 256  # partition size: 64-512 within noise on Qwen3.5-9B at 2048 and 8192 (RESULTS 1x)

_SRC_PART = r"""
    constexpr int NSG = 8;
    constexpr int E = D / 32;
    const uint tg = threadgroup_position_in_grid.y;
    const uint lid = thread_position_in_threadgroup.x;
    const int sg = lid / 32;
    const int lane = lid % 32;
    const int P = params[3];
    const int b = tg / (NKV * P);
    const int h = (tg / P) % NKV;
    const int pi = tg % P;
    const int max_blocks = params[0];
    const int bs = params[1];
    const int window = params[2];
    const float scale = fparams[0];
    const int kv_len = kv_lens[b];
    const int start = window > 0 ? max(0, kv_len - window) : 0;
    const bool single = kv_len - start <= PS;
    if (single && pi > 0) return;
    const int lo = single ? start : start + pi * PS;
    const int hi = single ? kv_len : min(kv_len, lo + PS);

    if (FUSED) {  // decode-only step: write this row's new K/V into the pool here (replaces a whole-pool scatter copy)
        const int pw = kv_len - 1;
        if (pw >= lo && pw < hi) {  // uniform per threadgroup: exactly one threadgroup per (b, h) holds pw
            const int wslot = bt[b * max_blocks + pw / bs] * bs + pw % bs;
            const size_t wb = (size_t(wslot) * NKV + h) * D;
            device T* kw = (device T*)k;
            device T* vw = (device T*)v;
            for (int d = lid; d < D; d += NSG * 32) {
                kw[wb + d] = kn[(b * NKV + h) * D + d];
                vw[wb + d] = vn[(b * NKV + h) * D + d];
            }
            threadgroup_barrier(mem_flags::mem_device);
        }
    }

    float qr[G][E];
    for (int g = 0; g < G; ++g)
        for (int e = 0; e < E; ++e)
            qr[g][e] = float(q[((b * NKV + h) * G + g) * D + lane * E + e]) * scale;
    float m[G], l[G], o[G][E];
    for (int g = 0; g < G; ++g) {
        m[g] = -INFINITY; l[g] = 0.0f;
        for (int e = 0; e < E; ++e) o[g][e] = 0.0f;
    }
    for (int p = lo + sg; p < hi; p += NSG) {
        const int slot = bt[b * max_blocks + p / bs] * bs + p % bs;
        const size_t base = (size_t(slot) * NKV + h) * D + lane * E;
        float kr[E], vr[E];
        for (int e = 0; e < E; ++e) { kr[e] = float(k[base + e]); vr[e] = float(v[base + e]); }
        for (int g = 0; g < G; ++g) {
            float s = 0.0f;
            for (int e = 0; e < E; ++e) s += qr[g][e] * kr[e];
            s = simd_sum(s);
            const float nm = max(m[g], s);
            const float f = fast::exp(m[g] - nm);
            const float w = fast::exp(s - nm);
            l[g] = l[g] * f + w;
            for (int e = 0; e < E; ++e) o[g][e] = o[g][e] * f + w * vr[e];
            m[g] = nm;
        }
    }
    threadgroup float tm[NSG][G];
    threadgroup float tl[NSG][G];
    threadgroup float to[NSG][D];
    if (lane == 0)
        for (int g = 0; g < G; ++g) { tm[sg][g] = m[g]; tl[sg][g] = l[g]; }
    for (int g = 0; g < G; ++g) {
        for (int e = 0; e < E; ++e) to[sg][lane * E + e] = o[g][e];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (single) {  // the single-pass kernel's combine, verbatim
            const float sink = float(sinks[h * G + g]);
            float M = sink;
            for (int s = 0; s < NSG; ++s) M = max(M, tm[s][g]);
            float L = fast::exp(sink - M);
            for (int s = 0; s < NSG; ++s) L += tl[s][g] * fast::exp(tm[s][g] - M);
            for (int d = lid; d < D; d += NSG * 32) {
                float acc = 0.0f;
                for (int s = 0; s < NSG; ++s) acc += to[s][d] * fast::exp(tm[s][g] - M);
                out1[((b * NKV + h) * G + g) * D + d] = T(acc / L);
            }
        } else {  // this partition's unnormalized partial (no sink)
            float M = -INFINITY;
            for (int s = 0; s < NSG; ++s) M = max(M, tm[s][g]);
            const size_t pb = ((size_t(b) * NKV + h) * P + pi) * G + g;
            if (M == -INFINITY) {  // empty partition
                if (lid == 0) { pm[pb] = -INFINITY; pl[pb] = 0.0f; }
                for (int d = lid; d < D; d += NSG * 32) po[pb * D + d] = 0.0f;
            } else {
                float L = 0.0f;
                for (int s = 0; s < NSG; ++s) L += tl[s][g] * fast::exp(tm[s][g] - M);
                if (lid == 0) { pm[pb] = M; pl[pb] = L; }
                for (int d = lid; d < D; d += NSG * 32) {
                    float acc = 0.0f;
                    for (int s = 0; s < NSG; ++s) acc += to[s][d] * fast::exp(tm[s][g] - M);
                    po[pb * D + d] = acc;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
"""

_SRC_REDUCE = r"""
    const uint row = threadgroup_position_in_grid.y;  // (b, h, g)
    const uint lid = thread_position_in_threadgroup.x;
    const int P = params[3];
    const int window = params[2];
    const int g = row % G;
    const int h = (row / G) % NKV;
    const int b = row / (G * NKV);
    const int kv_len = kv_lens[b];
    const int start = window > 0 ? max(0, kv_len - window) : 0;
    const size_t orow = ((size_t(b) * NKV + h) * G + g) * D;
    if (kv_len - start <= PS) {
        for (int d = lid; d < D; d += 256) out[orow + d] = out1[orow + d];
        return;
    }
    const float sink = float(sinks[h * G + g]);
    float M = sink;
    for (int p = 0; p < P; ++p) M = max(M, pm[((size_t(b) * NKV + h) * P + p) * G + g]);
    float L = fast::exp(sink - M);
    for (int p = 0; p < P; ++p) {
        const size_t pb = ((size_t(b) * NKV + h) * P + p) * G + g;
        L += pl[pb] * fast::exp(pm[pb] - M);
    }
    for (int d = lid; d < D; d += 256) {
        float acc = 0.0f;
        for (int p = 0; p < P; ++p) {
            const size_t pb = ((size_t(b) * NKV + h) * P + p) * G + g;
            acc += po[pb * D + d] * fast::exp(pm[pb] - M);
        }
        out[orow + d] = T(acc / L);
    }
"""

_kpart = mx.fast.metal_kernel(
    name="mj_paged_decode_attention_part",
    input_names=["q", "k", "v", "bt", "kv_lens", "sinks", "params", "fparams"],
    output_names=["out1", "pm", "pl", "po"],
    source=_SRC_PART.replace("if (FUSED) {", "if (false) {", 1).replace("kn[", "q[").replace("vn[", "q["),
)
_kpart_f = mx.fast.metal_kernel(
    name="mj_paged_decode_attention_part_fkv",
    input_names=["q", "k", "v", "bt", "kv_lens", "sinks", "params", "fparams", "kn", "vn"],
    output_names=["out1", "pm", "pl", "po"],
    source=_SRC_PART.replace("if (FUSED) {", "if (true) {", 1),
)
_kreduce = mx.fast.metal_kernel(
    name="mj_paged_decode_attention_reduce",
    input_names=["out1", "pm", "pl", "po", "kv_lens", "sinks", "params"],
    output_names=["out"],
    source=_SRC_REDUCE,
)


def paged_decode_attention(q: mx.array, k_pool: mx.array, v_pool: mx.array, block_tables: mx.array,
                           kv_lens: mx.array, sinks: mx.array, scale: float, block_size: int,
                           window: int = 0, max_kv: int = 0, k_new: mx.array | None = None,
                           v_new: mx.array | None = None) -> mx.array:
    """q [B, nq, D]; pools [slots, nkv, D]; block_tables [B, max_blocks] int32; kv_lens [B] int32 -> [B, nq, D].
    max_kv: the host-known largest kv_len in the batch; above PS positions (after the window) the split-K kernels
    run (0 = unknown: single pass, which is exact only while every row spans <= PS positions or none is split).
    k_new / v_new [B, nkv, D]: decode-only step; the kernel writes row b's new K/V at position kv_lens[b] - 1 of the
    pool in place (the caller skips its scatter). Same numbers as scattering first."""
    B, nq, D = q.shape
    nkv = k_pool.shape[1]
    assert D % 32 == 0 and nq % nkv == 0
    if k_new is not None:
        assert k_pool.dtype == v_pool.dtype == q.dtype == k_new.dtype == v_new.dtype
        assert k_new.shape == v_new.shape == (B, nkv, D)
    span = min(max_kv, window) if window > 0 else max_kv
    if span > PS:
        P = (span + PS - 1) // PS
        params = mx.array([block_tables.shape[1], block_size, window, P], dtype=mx.int32)
        tmpl = [("T", q.dtype), ("D", D), ("G", nq // nkv), ("NKV", nkv), ("PS", PS)]
        extra = [] if k_new is None else [k_new, v_new]
        out1, pm, pl, po = (_kpart if k_new is None else _kpart_f)(
            inputs=[q, k_pool, v_pool, block_tables, kv_lens, sinks, params, mx.array([scale], mx.float32)] + extra,
            template=tmpl, grid=(256, B * nkv * P, 1), threadgroup=(256, 1, 1),
            output_shapes=[(B, nq, D), (B, nkv, P, nq // nkv), (B, nkv, P, nq // nkv), (B, nkv, P, nq // nkv, D)],
            output_dtypes=[q.dtype, mx.float32, mx.float32, mx.float32])
        return _kreduce(inputs=[out1, pm, pl, po, kv_lens, sinks, params], template=tmpl,
                        grid=(256, B * nq, 1), threadgroup=(256, 1, 1), output_shapes=[(B, nq, D)],
                        output_dtypes=[q.dtype])[0]
    params = mx.array([block_tables.shape[1], block_size, window], dtype=mx.int32)
    extra = [] if k_new is None else [k_new, v_new]
    return (_kernel if k_new is None else _kernel_f)(
        inputs=[q, k_pool, v_pool, block_tables, kv_lens, sinks, params, mx.array([scale], mx.float32)] + extra,
        template=[("T", q.dtype), ("D", D), ("G", nq // nkv), ("NKV", nkv)],
        grid=(256, B * nkv, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(B, nq, D)],
        output_dtypes=[q.dtype],
    )[0]
