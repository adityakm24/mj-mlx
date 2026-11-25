# Copyright 2026 the MilliJoule authors
# SPDX-License-Identifier: Apache-2.0
"""Grouped MXFP4 expert GEMV kernels for decode-sized MoE steps (Metal, via mx.fast.metal_kernel).

Token/expert pairs are sorted by expert. One threadgroup owns one tile of output rows of one expert, reads that
tile's MXFP4 weights once, and applies them to every token routed to the expert (in chunks of TM tokens).
The gate/up kernel computes the gate row and the matching up row together and writes the clamped SwiGLU directly,
so the [pairs, 2I] intermediate never exists. The down kernel applies the expert bias and the routing weight.

Weights are the checkpoint's MXFP4: uint32 words of 8 E2M1 nibbles (low nibble first) and one E8M0 scale per
32 values. Accumulation is float32.
"""

from __future__ import annotations

import mlx.core as mx

_HEADER = r"""
constant float MJ_FP4[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                             -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
"""

# NR = rows of weights per simdgroup (gate+up pairs count twice), TM = tokens per chunk.
_SRC = r"""
    constexpr int G32 = K / 32;
    constexpr int WPR = K / 8;                       // uint32 words per weight row
    constexpr int NR = FUSE ? 2 * ROWS : ROWS;
    const int e = threadgroup_position_in_grid.y;
    const int tile = threadgroup_position_in_grid.x;
    const int sgi = simdgroup_index_in_threadgroup;
    const int lane = thread_index_in_simdgroup;
    const int row0 = (tile * SG + sgi) * ROWS;
    if (row0 >= NOUT) return;
    const int beg = offsets[e];
    const int end = offsets[e + 1];
    if (beg == end) return;
    constexpr int WROWS = FUSE ? 2 * NOUT : NOUT;
    const device uint32_t* wb = w + size_t(e) * WROWS * WPR;
    const device uint8_t* sb = s + size_t(e) * WROWS * G32;

    for (int t0 = beg; t0 < end; t0 += TM) {
        const int tn = min(TM, end - t0);
        float acc[NR][TM];
        for (int r = 0; r < NR; ++r)
            for (int t = 0; t < TM; ++t) acc[r][t] = 0.0f;
        for (int g = lane; g < G32; g += 32) {
            for (int r = 0; r < NR; ++r) {
                const int row = (r < ROWS) ? row0 + r : NOUT + row0 + (r - ROWS);
                const device uint32_t* wp = wb + size_t(row) * WPR + g * 4;
                const float sc = metal::exp2(float(sb[size_t(row) * G32 + g]) - 127.0f);
                float wf[32];
                for (int q = 0; q < 4; ++q) {
                    const uint32_t u = wp[q];
                    for (int j = 0; j < 8; ++j) wf[q * 8 + j] = MJ_FP4[(u >> (4 * j)) & 0xF];
                }
                for (int t = 0; t < TM; ++t) {
                    if (t < tn) {
                        const device T* xp = x + size_t(t0 + t) * K + g * 32;
                        float d = 0.0f;
                        for (int j = 0; j < 32; ++j) d += float(xp[j]) * wf[j];
                        acc[r][t] += d * sc;
                    }
                }
            }
        }
        for (int r = 0; r < NR; ++r)
            for (int t = 0; t < TM; ++t) acc[r][t] = simd_sum(acc[r][t]);
        if (lane == 0) {
            for (int t = 0; t < tn; ++t) {
                for (int r = 0; r < ROWS; ++r) {
                    const int row = row0 + r;
                    float y;
                    if (FUSE) {
                        float gate = acc[r][t] + float(bias[size_t(e) * WROWS + row]);
                        float up = acc[ROWS + r][t] + float(bias[size_t(e) * WROWS + NOUT + row]);
                        gate = metal::min(gate, limit[0]);
                        up = metal::clamp(up, -limit[0], limit[0]);
                        y = (up + 1.0f) * (gate / (1.0f + metal::exp(-limit[1] * gate)));
                    } else {
                        y = (acc[r][t] + float(bias[size_t(e) * WROWS + row])) * float(rw[t0 + t]);
                    }
                    out[size_t(t0 + t) * NOUT + row] = T(y);
                }
            }
        }
    }
"""

_kernel = mx.fast.metal_kernel(
    name="mj_moe_mxfp4_grouped_gemv",
    input_names=["x", "w", "s", "bias", "offsets", "rw", "limit"],
    output_names=["out"],
    header=_HEADER,
    source=_SRC,
)

ROWS, SG, TM = 4, 2, 8


def grouped_mxfp4(x, w, s, bias, offsets, nout, fuse_glu, rw=None, limit=7.0, alpha=1.702):
    """x [P, K] sorted by expert; w [E, rows, K/8] u32; s [E, rows, K/32] u8; offsets [E+1] -> [P, nout]."""
    P, K = x.shape
    E = w.shape[0]
    if rw is None:
        rw = mx.ones((P,), x.dtype)
    tiles = (nout + ROWS * SG - 1) // (ROWS * SG)
    return _kernel(
        inputs=[x, w, s, bias, offsets, rw, mx.array([limit, alpha], mx.float32)],
        template=[("T", x.dtype), ("K", K), ("NOUT", nout), ("ROWS", ROWS), ("SG", SG), ("TM", TM),
                  ("FUSE", bool(fuse_glu))],
        grid=(32 * SG * tiles, E, 1),
        threadgroup=(32 * SG, 1, 1),
        output_shapes=[(P, nout)],
        output_dtypes=[x.dtype],
    )[0]


# ---- Prefill glue around MLX's sorted gather_qmm (pairs sorted by expert) --------------------------------------
# The separate ops materialize the gathered biases ([P, 2I] and [P, H]), the biased gate/up, the un-permuted expert
# outputs and their weighted copies; at a 512-token gpt-oss chunk that glue was 2.5 ms of a 38.8 ms MoE layer.
# These two kernels do it in two passes, in float32 (one rounding at the end instead of one per op), so they are not
# bitwise equal to the op sequence: used only for prefill-sized steps (gpt_oss.Experts.FUSED_GLUE_MIN pairs), never
# for decode or speculative verify, which must match decode exactly.

_BIAS_SWIGLU_SRC = r"""
    const uint j = thread_position_in_grid.x;
    const uint r = thread_position_in_grid.y;
    if (j >= I) return;
    const size_t e = size_t(si[r]);
    float g = float(h[size_t(r) * 2 * I + j]) + float(bias[e * 2 * I + j]);
    float u = float(h[size_t(r) * 2 * I + I + j]) + float(bias[e * 2 * I + I + j]);
    g = metal::min(g, prm[0]);
    u = metal::clamp(u, -prm[0], prm[0]);
    out[size_t(r) * I + j] = T((u + 1.0f) * (g / (1.0f + metal::exp(-prm[1] * g))));
"""

_COMBINE_SRC = r"""
    const uint c = thread_position_in_grid.x;
    const uint t = thread_position_in_grid.y;
    if (c >= H) return;
    float acc = 0.0f;
    for (int kk = 0; kk < K; ++kk) {
        const size_t q = size_t(t) * K + kk;
        const size_t p = size_t(inv[q]);
        const size_t e = size_t(eid[q]);
        acc += float(w[q]) * (float(y[p * H + c]) + float(bias[e * H + c]));
    }
    out[size_t(t) * H + c] = OT(acc);
"""

_bias_swiglu_k = mx.fast.metal_kernel(name="mj_moe_bias_swiglu", input_names=["h", "bias", "si", "prm"],
                                      output_names=["out"], source=_BIAS_SWIGLU_SRC)
_combine_k = mx.fast.metal_kernel(name="mj_moe_combine", input_names=["y", "bias", "inv", "eid", "w"],
                                  output_names=["out"], source=_COMBINE_SRC)


def bias_swiglu_sorted(h, bias, si, limit=7.0, alpha=1.702):
    """h [P, 2I] (gate | up, sorted pairs), bias [E, 2I], si [P] expert of each pair -> [P, I] clamped SwiGLU."""
    P, I2 = h.shape
    I = I2 // 2
    return _bias_swiglu_k(inputs=[h, bias, si, mx.array([limit, alpha], mx.float32)],
                          template=[("T", h.dtype), ("I", I)], grid=(I, P, 1), threadgroup=(256, 1, 1),
                          output_shapes=[(P, I)], output_dtypes=[h.dtype])[0]


def combine_sorted(y, bias, inv, eid, w, out_dtype):
    """y [P, H] expert outputs in sorted order; inv [T*k] sorted row of each (token, slot); eid [T*k] its expert;
    w [T, k] routing weights -> [T, H] = sum_k w * (y[inv] + bias[eid])."""
    P, H = y.shape
    T, k = w.shape
    return _combine_k(inputs=[y, bias, inv, eid, w.reshape(-1)],
                      template=[("OT", out_dtype), ("H", H), ("K", k)], grid=(H, T, 1), threadgroup=(256, 1, 1),
                      output_shapes=[(T, H)], output_dtypes=[out_dtype])[0]
