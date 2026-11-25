# Copyright 2026 the MilliJoule authors
# SPDX-License-Identifier: Apache-2.0
"""Gated delta rule (Gated DeltaNet recurrence) as a Metal kernel that reads and writes the state pool itself.

Per item n, value head h and token t of the item (state S [Dk, Dv], float32 math):
    S <- S * exp(g[t, h]);  delta = (v[t, h] - S^T k[t, h]) * beta[t, h];  S <- S + k[t, h] delta^T;  o = S^T q[t, h]

Pool layout [slots, Hv, Dv, Dk] (column-major per head, as mlx-lm stores it), so each lane's Dk/32 values of one
column are contiguous and a simdgroup's loads coalesce. One simdgroup owns COLS value columns of one (slot, head)
and reuses each token's q / k registers across them (COLS = 4 took a B=64 decode update from 4.9 to 3.5 ms on M4,
the speed of a plain copy of the pool); the two reductions over Dk are simd_sums; the token loop of a prefill
chunk runs inside the kernel.

The kernel covers every slot of the pool and returns the next pool: slots used by an item this step are updated
(starting from zero when the item is fresh, i.e. start == 0), idle slots are copied. State is read once and
written once per step, with no separate gather / scatter of the B states.
"""

from __future__ import annotations

import mlx.core as mx

_SRC = r"""
    constexpr int E = DK / 32;
    const int sh = threadgroup_position_in_grid.y;        // slot * HV + head
    const int slot = sh / HV;
    const int h = sh % HV;
    const int sg = simdgroup_index_in_threadgroup;
    const int lane = thread_index_in_simdgroup;
    const int col0 = (threadgroup_position_in_grid.x * NSG + sg) * COLS;  // first of COLS value columns
    if (col0 >= DV) return;
    const int n = item_of_slot[slot];
    if (n < 0) {  // idle slot: carry the state over unchanged
        for (int c = 0; c < COLS; ++c) {
            const size_t base = (size_t(sh) * DV + col0 + c) * DK + lane * E;
            for (int e = 0; e < E; ++e) pool_out[base + e] = pool[base + e];
        }
        return;
    }
    // q / k are loaded once per token and reused for COLS columns.
    float s[COLS][E];
    const bool fresh_item = fresh[n] != 0;
    for (int c = 0; c < COLS; ++c) {
        const size_t base = (size_t(sh) * DV + col0 + c) * DK + lane * E;
        for (int e = 0; e < E; ++e) s[c][e] = fresh_item ? 0.0f : float(pool[base + e]);
    }
    const int t0 = cu[n], t1 = cu[n + 1];
    for (int t = t0; t < t1; ++t) {
        const size_t th = size_t(t) * HV + h;
        const float decay = metal::exp(g[th]);
        const float bt = beta[th];
        float kr[E], qr[E];
        for (int e = 0; e < E; ++e) { kr[e] = k[th * DK + lane * E + e]; qr[e] = q[th * DK + lane * E + e]; }
        for (int c = 0; c < COLS; ++c) {
            float kv = 0.0f;
            for (int e = 0; e < E; ++e) { s[c][e] *= decay; kv += s[c][e] * kr[e]; }
            kv = simd_sum(kv);
            const float delta = (v[th * DV + col0 + c] - kv) * bt;
            float o = 0.0f;
            for (int e = 0; e < E; ++e) { s[c][e] += kr[e] * delta; o += s[c][e] * qr[e]; }
            o = simd_sum(o);
            if (lane == 0) out[th * DV + col0 + c] = o;
        }
    }
    for (int c = 0; c < COLS; ++c) {
        const size_t base = (size_t(sh) * DV + col0 + c) * DK + lane * E;
        for (int e = 0; e < E; ++e) pool_out[base + e] = ST(s[c][e]);
    }
"""

_kernel = mx.fast.metal_kernel(
    name="mj_gated_delta_pool",
    input_names=["q", "k", "v", "g", "beta", "pool", "item_of_slot", "fresh", "cu"],
    output_names=["out", "pool_out"],
    source=_SRC,
)

NSG = 8
COLS = 4


def gated_delta_pool(q, k, v, g, beta, pool, item_of_slot, fresh, cu, cols: int | None = None):
    """q, k [T, Hv, Dk]; v [T, Hv, Dv]; g, beta [T, Hv] (float32); pool [slots, Hv, Dv, Dk] (float32 or bf16);
    item_of_slot [slots] int32 (-1 idle); fresh [N] uint8; cu [N+1] int32 -> (out [T, Hv, Dv] f32, next pool)."""
    T, Hv, Dk = k.shape
    Dv = v.shape[-1]
    S = pool.shape[0]
    if cols is None:  # 4 columns per simdgroup sharing q/k registers: fastest measured for decode and prefill
        cols = COLS
    assert Dk % 32 == 0 and Dv % cols == 0 and pool.shape[1:] == (Hv, Dv, Dk)
    return _kernel(
        inputs=[q, k, v, g, beta, pool, item_of_slot, fresh, cu],
        template=[("DK", Dk), ("DV", Dv), ("HV", Hv), ("NSG", NSG), ("COLS", cols), ("ST", pool.dtype)],
        grid=(32 * NSG * ((Dv + NSG * cols - 1) // (NSG * cols)), S * Hv, 1),
        threadgroup=(32 * NSG, 1, 1),
        output_shapes=[(T, Hv, Dv), pool.shape],
        output_dtypes=[mx.float32, pool.dtype],
    )


# ---- packed variant: 4 lanes per value column, 8 columns per simdgroup ---------------------------------------------
# Same math and pool semantics as above. Work split idea from mlx-lm's packed gated-delta kernel (MIT; idea only, code
# written here): with 4 lanes per column each lane holds Dk/4 contiguous state values, so the two reductions per token
# are 2 shuffles inside the 4-lane group instead of a 32-lane simd_sum. q/k are read per key head (no repeat copy).
_SRC_PACKED = r"""
    constexpr int LPC = 4;                 // lanes per value column
    constexpr int CPS = 32 / LPC;          // columns per simdgroup
    constexpr int E = DK / LPC;            // state values per lane
    const int sh = threadgroup_position_in_grid.y;        // slot * HV + head
    const int slot = sh / HV;
    const int h = sh % HV;
    const int hk = h / (HV / HK);
    const int sg = simdgroup_index_in_threadgroup;
    const int lane = thread_index_in_simdgroup;
    const int col = (threadgroup_position_in_grid.x * NSG + sg) * CPS + lane / LPC;
    const int sub = lane % LPC;
    if (col >= DV) return;
    const size_t base = (size_t(sh) * DV + col) * DK + sub * E;
    const int n = item_of_slot[slot];
    if (n < 0) {
        for (int e = 0; e < E; ++e) pool_out[base + e] = pool[base + e];
        return;
    }
    float s[E];
    const bool fresh_item = fresh[n] != 0;
    for (int e = 0; e < E; ++e) s[e] = fresh_item ? 0.0f : float(pool[base + e]);
    const int t0 = cu[n], t1 = cu[n + 1];
    for (int t = t0; t < t1; ++t) {
        const size_t th = size_t(t) * HV + h;
        const size_t tk = (size_t(t) * HK + hk) * DK + sub * E;
        const float decay = metal::exp(g[th]);
        float kf[E], qf[E];
        for (int e = 0; e < E; ++e) { kf[e] = float(k[tk + e]); qf[e] = float(q[tk + e]); }
        if (NORM) {  // L2-normalize q and k over Dk in float32 (x * rsqrt(sum x^2 + 1e-6)), then q /= sqrt(Dk)
            float sk = 0.0f, sq = 0.0f;
            for (int e = 0; e < E; ++e) { sk += kf[e] * kf[e]; sq += qf[e] * qf[e]; }
            sk += simd_shuffle_xor(sk, 1); sk += simd_shuffle_xor(sk, 2);
            sq += simd_shuffle_xor(sq, 1); sq += simd_shuffle_xor(sq, 2);
            const float ik = metal::rsqrt(sk + 1e-6f), iq = metal::rsqrt(sq + 1e-6f) * metal::rsqrt(float(DK));
            for (int e = 0; e < E; ++e) { kf[e] *= ik; qf[e] *= iq; }
        }
        float kv = 0.0f;
        for (int e = 0; e < E; ++e) { s[e] *= decay; kv += s[e] * kf[e]; }
        kv += simd_shuffle_xor(kv, 1);
        kv += simd_shuffle_xor(kv, 2);
        const float delta = (float(v[th * DV + col]) - kv) * beta[th];
        float o = 0.0f;
        for (int e = 0; e < E; ++e) { s[e] += kf[e] * delta; o += s[e] * qf[e]; }
        o += simd_shuffle_xor(o, 1);
        o += simd_shuffle_xor(o, 2);
        if (sub == 0) out[th * DV + col] = o;
        /*KEEP*/
    }
    for (int e = 0; e < E; ++e) pool_out[base + e] = ST(s[e]);
"""

_kernel_packed = mx.fast.metal_kernel(
    name="mj_gated_delta_pool_packed",
    input_names=["q", "k", "v", "g", "beta", "pool", "item_of_slot", "fresh", "cu"],
    output_names=["out", "pool_out"],
    source=_SRC_PACKED,
)

# Same kernel that also writes the state after every token (steps [T, Hv, Dv, Dk]): speculative decoding verifies
# several tokens in one step and keeps the state after the last accepted one.
_kernel_packed_keep = mx.fast.metal_kernel(
    name="mj_gated_delta_pool_packed_keep",
    input_names=["q", "k", "v", "g", "beta", "pool", "item_of_slot", "fresh", "cu"],
    output_names=["out", "pool_out", "steps"],
    source=_SRC_PACKED.replace(
        "/*KEEP*/",
        "{ const size_t sb = ((size_t(t) * HV + h) * DV + col) * DK + sub * E;"
        " for (int e = 0; e < E; ++e) steps[sb + e] = ST(s[e]); }"),
)


def gated_delta_pool_packed(q, k, v, g, beta, pool, item_of_slot, fresh, cu, nsg: int = 1, normalize: bool = False,
                            keep_steps: bool = False):
    """As gated_delta_pool, but q, k are [T, Hk, Dk] (not repeated to Hv) in any float dtype (bf16 halves the bytes:
    2.3x faster at a 2048-token prefill on M4). normalize=True does the L2 norm of q and k and the 1/sqrt(Dk) scale
    inside, in float32, so bf16 inputs keep transformers' float32 normalization. keep_steps=True also returns the
    state after each token, [T, Hv, Dv, Dk] in the pool dtype (the same numbers the final state is made of)."""
    T, Hk, Dk = k.shape
    Hv, Dv = v.shape[1], v.shape[2]
    S = pool.shape[0]
    assert Dk % 4 == 0 and Hv % Hk == 0 and pool.shape[1:] == (Hv, Dv, Dk)
    cols_per_tg = nsg * 8
    kern = _kernel_packed_keep if keep_steps else _kernel_packed
    shapes, dtypes = [(T, Hv, Dv), pool.shape], [mx.float32, pool.dtype]
    if keep_steps:
        shapes.append((T, Hv, Dv, Dk))
        dtypes.append(pool.dtype)
    return kern(
        inputs=[q, k, v, g, beta, pool, item_of_slot, fresh, cu],
        template=[("DK", Dk), ("DV", Dv), ("HV", Hv), ("HK", Hk), ("NSG", nsg), ("NORM", bool(normalize)),
                  ("ST", pool.dtype)],
        grid=(32 * nsg * ((Dv + cols_per_tg - 1) // cols_per_tg), S * Hv, 1),
        threadgroup=(32 * nsg, 1, 1),
        output_shapes=shapes,
        output_dtypes=dtypes,
    )
