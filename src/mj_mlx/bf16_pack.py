# Copyright 2026 the MilliJoule authors
# SPDX-License-Identifier: Apache-2.0
"""Lossless 12.25-bit storage for bf16 weight matrices, and a matvec that reads it directly.

Trained bf16 weights use few distinct exponents within a small group. A bf16 is [sign | 8-bit exponent | 7-bit
mantissa] = [hi byte | lo byte], with hi = sign + exponent >> 1 and lo = exponent & 1 + mantissa. Per weight matrix
W [N, K] (rows contiguous along K):
- lo   uint8 [N, K]        the lo byte, unchanged;
- nib  uint8 [N, K / 2]    per weight 4 bits: sign << 3 | off, off = base - (exponent >> 1) in 0..6; 7 = escape;
- base uint8 [N, K / 32]   per group of 32 weights, the largest exponent >> 1;
- escapes, CSR by row: weights whose exponent is more than 13-14 below the group's largest (tiny ones; gpt-oss-20b
  attention: ~2e-4 of weights). They are 0 in the planes and added back exactly by the kernel.
12.25 bits per weight plus ~6 bytes per escape, vs 16. Every weight is reproduced bit for bit (unpack(pack(W)) == W).

Use: decode reads weights once per token, so it is memory-bound; 23% fewer bytes for the bf16 matrices. matvec()
handles 1-4 rows (decode and short verify steps) and accumulates in float32 (not MLX's bf16 gemv order, so the
last bit of an output can differ from x @ W.T; the weights are exact). unpack() rebuilds the bf16 matrix for longer
steps (prefill), which then use MLX's own matmul: identical to the uncompressed model there.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

GROUP = 32
ESC = 7


def pack(w) -> dict:
    """w: bf16 [N, K] (mx.array or uint16 numpy view). K % 32 == 0."""
    u = np.asarray(w.view(mx.uint16)) if isinstance(w, mx.array) else np.asarray(w, np.uint16)
    N, K = u.shape
    assert K % GROUP == 0
    hi = (u >> 8).astype(np.uint8)
    lo = (u & 0xFF).astype(np.uint8)
    s = hi >> 7
    e7 = (hi & 0x7F).astype(np.int16)
    base = e7.reshape(N, K // GROUP, GROUP).max(axis=-1)
    off = np.repeat(base, GROUP, axis=1) - e7
    esc = off >= ESC
    off = np.where(esc, ESC, off).astype(np.uint8)
    nb = (s << 3 | off).astype(np.uint8)
    nib = (nb[:, 0::2] | (nb[:, 1::2] << 4)).astype(np.uint8)
    rows, cols = np.nonzero(esc)
    ptr = np.zeros(N + 1, np.int32)
    np.add.at(ptr, rows + 1, 1)
    ptr = np.cumsum(ptr).astype(np.int32)
    return dict(lo=mx.array(lo), nib=mx.array(nib), base=mx.array(base.astype(np.uint8)), esc_ptr=mx.array(ptr),
                esc_k=mx.array(cols.astype(np.int32)), esc_v=mx.array(u[rows, cols]).view(mx.bfloat16),
                shape=(N, K))


def nbytes(p: dict) -> int:
    return sum(p[k].nbytes for k in ("lo", "nib", "base", "esc_ptr", "esc_k", "esc_v"))


def arrays(p: dict) -> list[mx.array]:
    return [p[k] for k in ("lo", "nib", "base", "esc_ptr", "esc_k", "esc_v")]


_UNPACK = """
  const uint n = thread_position_in_grid.y, c = thread_position_in_grid.x;  // c: chunk of 8 weights
  if (c >= K / 8) return;
  const uint2 l = *(const device uint2*)(lo + size_t(n) * K + c * 8);
  const uint q = *(const device uint*)(nib + size_t(n) * (K / 2) + c * 4);
  const uint b = base[size_t(n) * (K / 32) + c / 4];
  for (int j = 0; j < 8; ++j) {
    const uint nb = (q >> (4 * j)) & 0xF, off = nb & 7;
    const uint lb = ((j < 4 ? l.x : l.y) >> (8 * (j & 3))) & 0xFF;
    const ushort u = off == 7 ? ushort(0) : ushort((((nb >> 3) << 7 | (b - off)) << 8) | lb);
    out[size_t(n) * K + c * 8 + j] = u;
  }
  for (int e = esc_ptr[n]; e < esc_ptr[n + 1]; ++e)  // this chunk's escapes (rows have ~1)
    if (uint(esc_k[e]) / 8 == c) out[size_t(n) * K + esc_k[e]] = esc_u[e];
"""

_kern: dict = {}


def unpack(p: dict) -> mx.array:
    """bf16 [N, K], bit-identical to the packed matrix."""
    N, K = p["shape"]
    if "unpack" not in _kern:
        _kern["unpack"] = mx.fast.metal_kernel(name="mj_bf16_unpack", input_names=["lo", "nib", "base", "esc_ptr",
                                               "esc_k", "esc_u"], output_names=["out"], source=_UNPACK)
    c = K // 8
    tx = min(256, c)
    out = _kern["unpack"](inputs=[p["lo"], p["nib"], p["base"], p["esc_ptr"], p["esc_k"], p["esc_v"].view(mx.uint16)],
                          template=[("K", K)], grid=(((c + tx - 1) // tx) * tx, N, 1), threadgroup=(tx, 1, 1),
                          output_shapes=[(N, K)], output_dtypes=[mx.uint16], init_value=0)[0]
    return out.view(mx.bfloat16)


def _gen_matvec(rows: int, rps: int, bias: bool) -> str:
    L = []
    a = L.append
    a("  const uint sg = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;")
    a(f"  const uint row0 = threadgroup_position_in_grid.y * (NSG * {rps}) + sg * {rps};")
    for m in range(rows):
        for r in range(rps):
            a(f"  float acc_{m}_{r} = 0;")
    a("  for (uint c = lane; c < K / 8; c += 32) {")
    for m in range(rows):
        a(f"    const device T* xm{m} = x + {m} * K + c * 8;")
        a(f"    float x{m}[8]; for (int j = 0; j < 8; ++j) x{m}[j] = float(xm{m}[j]);")
    for r in range(rps):
        a("    {")
        a(f"      const size_t n = row0 + {r};")
        a("      const uint2 l = *(const device uint2*)(lo + n * K + c * 8);")
        a("      const uint q = *(const device uint*)(nib + n * (K / 2) + c * 4);")
        a("      const uint b = base[n * (K / 32) + c / 4];")
        a("      float w[8];")
        a("      for (int j = 0; j < 8; ++j) {")
        a("        const uint nb = (q >> (4 * j)) & 0xF, off = nb & 7;")
        a("        const uint lb = ((j < 4 ? l.x : l.y) >> (8 * (j & 3))) & 0xFF;")
        a("        const ushort u = ushort((((nb >> 3) << 7 | (b - off)) << 8) | lb);")
        a("        w[j] = off == 7 ? 0.0f : float(as_type<bfloat>(u));")
        a("      }")
        for m in range(rows):
            a(f"      for (int j = 0; j < 8; ++j) acc_{m}_{r} = fma(x{m}[j], w[j], acc_{m}_{r});")
        a("    }")
    a("  }")
    for r in range(rps):  # escapes of this simdgroup's rows, spread over the lanes
        a(f"  for (int e = esc_ptr[row0 + {r}] + lane; e < esc_ptr[row0 + {r} + 1]; e += 32) {{")
        a("    const float v = float(esc_v[e]); const int k = esc_k[e];")
        for m in range(rows):
            a(f"    acc_{m}_{r} = fma(float(x[{m} * K + k]), v, acc_{m}_{r});")
        a("  }")
    for m in range(rows):
        for r in range(rps):
            b = f" + float(bias[row0 + {r}])" if bias else ""
            a(f"  {{ const float v = simd_sum(acc_{m}_{r}); if (lane == 0) y[{m} * N + row0 + {r}] = T(v{b}); }}")
    return "\n".join(L)


ROWWISE = False  # long steps through matvec in chunks of MAX_ROWS too (only to measure its accuracy)
RPS, NSG = 1, 4  # fastest on M4 for gpt-oss-20b qkv (5120 x 2880) and o (2880 x 4096)
MAX_ROWS = 4


def matvec(x: mx.array, p: dict, bias: mx.array | None = None) -> mx.array:
    """y [R, N] = x [R, K] @ W.T (+ bias), R <= MAX_ROWS, float32 accumulation, rounded once to x.dtype."""
    R, K = x.shape
    N = p["shape"][0]
    assert K == p["shape"][1] and R <= MAX_ROWS and N % (RPS * NSG) == 0 and K % 8 == 0
    key = (R, bias is not None)
    if key not in _kern:
        names = ["x", "lo", "nib", "base", "esc_ptr", "esc_k", "esc_v"] + (["bias"] if bias is not None else [])
        _kern[key] = mx.fast.metal_kernel(name=f"mj_bf16p_mv_{R}{'_b' if bias is not None else ''}",
                                          input_names=names, output_names=["y"],
                                          source=_gen_matvec(R, RPS, bias is not None))
    ins = [x, p["lo"], p["nib"], p["base"], p["esc_ptr"], p["esc_k"], p["esc_v"]] + ([bias] if bias is not None else [])
    return _kern[key](inputs=ins, template=[("T", x.dtype), ("K", K), ("N", N), ("NSG", NSG)],
                      grid=(32, NSG * (N // (RPS * NSG)), 1), threadgroup=(32, NSG, 1), output_shapes=[(R, N)],
                      output_dtypes=[x.dtype])[0]
