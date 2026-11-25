# Copyright 2026 the MilliJoule authors
# SPDX-License-Identifier: Apache-2.0
"""MLA decode attention (absorbed form) over the paged latent cache, as a custom Metal kernel (checklist 0.7c).

DeepSeek-V2 / V3 cache one row per token and layer, [latent c (r) | rope key (dr)], shared by every query head. A
decode row attends in the absorbed form: q = [q_nope W_uk | q_rope] (DK = r + dr) against the rows, and the output is
the softmax-weighted sum of the rows' first DV = r dims. Before this kernel, decode went through
`mx.fast.scaled_dot_product_attention` over a gather of every cached row (`kp[slots]`): a copy of the layer's whole
context per token, and SDPA's unfused path at DK = 576. DeepSeek-V2-Lite decode fell from 52.5 tok/s at 2k to
20.1-21.1 at 16k (v14 probe). The GQA paged kernel is no answer: with one KV head it keeps all 16 heads' queries
and outputs per thread (2 x 16 x 18 floats) and ran at 1.7 tok/s at 16k.

Here each token's row is read from the pool once per step, through the block table, with no gather:
  pass 1: one threadgroup per (sequence, PS-position partition). Its 8 simdgroups load tiles of TILE rows into
          threadgroup memory together; each simdgroup then owns H / 8 heads (2 for H = 16: 36 query and 32 output
          floats per lane), with an online softmax per head. Writes the partition's (max, sum, unnormalized output).
  pass 2: one threadgroup per (sequence, head) merges the partitions.
float32 accumulation. Same math as SDPA over the gathered rows; the test compares against that.
"""

from __future__ import annotations

import mlx.core as mx

PS = 256  # positions per partition
TILE = 16  # rows staged in threadgroup memory at a time (16 x 576 bf16 = 18 KB)

_PASS1 = r"""
    constexpr int NSG = 8;
    constexpr int HG = (H + NSG - 1) / NSG;     // heads per simdgroup
    constexpr int EK = (DK + 31) / 32;          // q / k dims per lane
    constexpr int EV = (DV + 31) / 32;          // output dims per lane
    const uint lid = thread_position_in_threadgroup.x;
    const int sg = lid / 32;
    const int lane = lid % 32;
    const int P = params[2];
    const int b = threadgroup_position_in_grid.y / P;
    const int part = threadgroup_position_in_grid.y % P;
    const int max_blocks = params[0];
    const int bs = params[1];
    const float scale = fparams[0];
    const int kv_len = kv_lens[b];
    const int p0 = part * PS;
    const int p1 = min(kv_len, p0 + PS);

    threadgroup T tk[TILE * DK];

    if (FUSED) {  // decode-only step: store this row's new cache row (position kv_len - 1) before reading it
        const int pw = kv_len - 1;
        if (pw >= p0 && pw < p1) {  // exactly one partition holds it; uniform per threadgroup
            const int wslot = bt[b * max_blocks + pw / bs] * bs + pw % bs;
            device T* w = (device T*)pool;
            for (int d = lid; d < DK; d += NSG * 32) w[size_t(wslot) * DK + d] = kn[b * DK + d];
        }
        threadgroup_barrier(mem_flags::mem_device);
    }

    float qr[HG][EK], o[HG][EV], m[HG], l[HG];
    for (int j = 0; j < HG; ++j) {
        const int h = sg * HG + j;
        for (int e = 0; e < EK; ++e) {
            const int d = lane * EK + e;
            qr[j][e] = (h < H && d < DK) ? float(q[(b * H + h) * DK + d]) * scale : 0.0f;
        }
        for (int e = 0; e < EV; ++e) o[j][e] = 0.0f;
        m[j] = -INFINITY;
        l[j] = 0.0f;
    }
    for (int t0 = p0; t0 < p1; t0 += TILE) {
        const int nt = min(TILE, p1 - t0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = lid; i < nt * DK; i += NSG * 32) {  // stage nt rows (each row once from device memory)
            const int r = i / DK, d = i % DK;
            const int p = t0 + r;
            const int slot = bt[b * max_blocks + p / bs] * bs + p % bs;
            tk[r * DK + d] = pool[size_t(slot) * DK + d];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int r = 0; r < nt; ++r) {
            float kr[EK], vr[EV];
            for (int e = 0; e < EK; ++e) {
                const int d = lane * EK + e;
                kr[e] = d < DK ? float(tk[r * DK + d]) : 0.0f;
            }
            for (int e = 0; e < EV; ++e) {
                const int d = lane * EV + e;
                vr[e] = d < DV ? float(tk[r * DK + d]) : 0.0f;
            }
            for (int j = 0; j < HG; ++j) {
                float s = 0.0f;
                for (int e = 0; e < EK; ++e) s += qr[j][e] * kr[e];
                s = simd_sum(s);
                const float nm = max(m[j], s);
                const float f = fast::exp(m[j] - nm);
                const float w = fast::exp(s - nm);
                l[j] = l[j] * f + w;
                for (int e = 0; e < EV; ++e) o[j][e] = o[j][e] * f + w * vr[e];
                m[j] = nm;
            }
        }
    }
    for (int j = 0; j < HG; ++j) {
        const int h = sg * HG + j;
        if (h >= H) continue;
        const int idx = (b * P + part) * H + h;
        if (lane == 0) { pm[idx] = m[j]; pl[idx] = l[j]; }
        for (int e = 0; e < EV; ++e) {
            const int d = lane * EV + e;
            if (d < DV) po[size_t(idx) * DV + d] = o[j][e];
        }
    }
"""

_PASS2 = r"""
    const uint lid = thread_position_in_threadgroup.x;
    const int P = params[2];
    const int b = threadgroup_position_in_grid.y / H;
    const int h = threadgroup_position_in_grid.y % H;
    float M = -INFINITY;
    for (int p = 0; p < P; ++p) M = max(M, pm[(b * P + p) * H + h]);
    float L = 0.0f;
    for (int p = 0; p < P; ++p) {
        const float mp = pm[(b * P + p) * H + h];
        L += mp == -INFINITY ? 0.0f : pl[(b * P + p) * H + h] * fast::exp(mp - M);
    }
    for (int d = lid; d < DV; d += 256) {
        float acc = 0.0f;
        for (int p = 0; p < P; ++p) {
            const float mp = pm[(b * P + p) * H + h];
            if (mp != -INFINITY) acc += po[size_t((b * P + p) * H + h) * DV + d] * fast::exp(mp - M);
        }
        out[(b * H + h) * DV + d] = T(acc / L);
    }
"""

_k1 = mx.fast.metal_kernel(name="mj_mla_decode_part", input_names=["q", "pool", "bt", "kv_lens", "params", "fparams"],
                           output_names=["pm", "pl", "po"],
                           source=_PASS1.replace("if (FUSED) {", "if (false) {", 1).replace("kn[", "q["))
# Fused cache write (decode-only steps): MLX sees the pool only as an input; ordering relies on the step's data
# dependencies, as for the GQA paged kernel (paged_attention.py): nothing else reads this layer's pool in the step.
_k1f = mx.fast.metal_kernel(name="mj_mla_decode_part_fkv",
                            input_names=["q", "pool", "bt", "kv_lens", "params", "fparams", "kn"],
                            output_names=["pm", "pl", "po"], source=_PASS1.replace("if (FUSED) {", "if (true) {", 1))
_k2 = mx.fast.metal_kernel(name="mj_mla_decode_merge", input_names=["pm", "pl", "po", "params"],
                           output_names=["out"], source=_PASS2)


def mla_decode_attention(q: mx.array, pool: mx.array, block_tables: mx.array, kv_lens: mx.array, scale: float,
                         block_size: int, max_kv: int, dv: int, new_rows: mx.array | None = None) -> mx.array:
    """q [B, H, DK] (absorbed queries), pool [slots, 1, DK] or [slots, DK] (latent | rope rows), block_tables
    [B, max_blocks] int32, kv_lens [B] int32, max_kv: the largest kv_len (host-known). -> [B, H, dv]: the softmax
    over each row's first kv_len positions of the rows' first dv dims. new_rows [B, DK] (decode-only step): written
    at each row's position kv_len - 1 inside the kernel (the caller skips its scatter); same result as writing first."""
    B, H, DK = q.shape
    assert pool.dtype == q.dtype and pool.shape[-1] == DK and dv <= DK
    P = max(1, (int(max_kv) + PS - 1) // PS)
    params = mx.array([block_tables.shape[1], block_size, P], dtype=mx.int32)
    tmpl = [("T", q.dtype), ("H", H), ("DK", DK), ("DV", dv), ("PS", PS), ("TILE", TILE)]
    extra = [] if new_rows is None else [new_rows.reshape(B, DK)]
    pm, pl, po = (_k1 if new_rows is None else _k1f)(
                     inputs=[q, pool.reshape(-1, DK), block_tables, kv_lens, params, mx.array([scale], mx.float32)] + extra,
                     template=tmpl, grid=(256, B * P, 1), threadgroup=(256, 1, 1),
                     output_shapes=[(B, P, H), (B, P, H), (B, P, H, dv)],
                     output_dtypes=[mx.float32, mx.float32, mx.float32])
    return _k2(inputs=[pm, pl, po, params], template=tmpl, grid=(256, B * H, 1), threadgroup=(256, 1, 1),
               output_shapes=[(B, H, dv)], output_dtypes=[q.dtype])[0]
