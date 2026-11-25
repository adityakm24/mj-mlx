# Copyright 2026 the MilliJoule authors
# SPDX-License-Identifier: Apache-2.0
"""4-bit quantized matmul for decode batches of 5-16 rows: MLX's own tiled kernel pieces (steel BlockMMA, BlockLoader,
QuantizedBlockLoader from the installed mlx headers, MIT licence, Apple) with 8- or 16-row tiles instead of 32.

Why: `mx.quantized_matmul` runs a matrix-vector kernel per row up to a few rows and a 32-row tiled kernel beyond,
so on an M4 (4-bit gs64, [12288 x 4096]) 8 rows take 2.4x one row (665 vs 281 us) and 16 rows 3.4x (957 us),
although reading the weights once takes ~280 us. A decode step of 5-16 requests pays that in every layer: both
MLX engines' 9B decode step doubles from 4 to 8 requests (60 -> 123 ms).
Measured here: 8-row tile, 6-8 rows 349-354 us; 16-row tile, 12-16 rows 536-538 us (1.8-1.9x faster).

Same math as MLX's tiled kernel (dequantize a weight tile into threadgroup memory, simdgroup 8x8 multiply-accumulate);
a row's result does not depend on the other rows in its tile. Error vs float64 equals mx.quantized_matmul's.
Not bitwise equal to mx.quantized_matmul (a different kernel; MLX's own results already differ between batch sizes).
Anything else (other bits / group sizes, float32, shapes, 1-4 or > 16 rows) goes to mx.quantized_matmul, and so does
everything if the kernel cannot be built from the installed headers.
"""

from __future__ import annotations

import os
import re

import mlx.core as mx

ENABLED = True
MIN_ROWS, MAX_ROWS = 5, 16
# (row tile, K tile, N tile, simdgroups along M, along N), from a sweep on an M4 (scratch qmm_small.py)
CONFIGS = {8: (8, 64, 32, 1, 4), 16: (16, 32, 64, 1, 4)}

_SKIP = {f"mlx/backend/metal/kernels/{h}.h" for h in ("utils", "bf16", "bf16_math", "complex", "defines", "logging")}

_IMPL = r"""
template <typename T, int group_size, int bits, int BM, int BK, int BN, int WM, int WN>
METAL_FUNC void mj_qmm_t_small(const device uint32_t* w, const device T* scales, const device T* biases,
                               const device T* x, device T* y, threadgroup T* Xs, threadgroup T* Ws,
                               const int K, const int N, uint3 tid, uint simd_gid, uint simd_lid) {
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  using mma_t = mlx::steel::BlockMMA<T, T, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded>;
  using loader_x_t = mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;
  using loader_w_t = QuantizedBlockLoader<T, BN, BK, BK_padded, 1, WM * WN * SIMD_SIZE, group_size, bits>;
  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;
  auto wl = (const device uint8_t*)w;
  x += y_row * static_cast<int64_t>(K);
  wl += y_col * K_w;
  scales += y_col * K_g;
  biases += y_col * K_g;
  y += y_row * static_cast<int64_t>(N) + y_col;
  loader_x_t loader_x(x, K, Xs, simd_gid, simd_lid);
  loader_w_t loader_w(wl, scales, biases, K, Ws, simd_gid, simd_lid);
  mma_t mma_op(simd_gid, simd_lid);
  for (int k = 0; k < K; k += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    loader_x.load_unsafe();  // rows are padded to a whole tile by the caller
    loader_w.load_unsafe();
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma_op.mma(Xs, Ws);
    loader_x.next();
    loader_w.next();
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  mma_op.store_result(y, N);
}
"""

_SOURCE = """
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];
  mj_qmm_t_small<T, GS, 4, BM, BK, BN, WM, WN>(w, scales, biases, x, y, Xs, Ws, K, N,
      threadgroup_position_in_grid, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
"""

_kernel = None
_pads = {}  # (rows, K, dtype) -> evaluated zero rows that fill the tile


def _inline(inc, path, seen):
    if path in seen or path in _SKIP:  # utils.h and what it pulls in are already in every metal_kernel's prelude
        return ""
    seen.add(path)
    out = []
    for line in open(os.path.join(inc, path)):
        m = re.match(r'\s*#include\s+"(mlx/[^"]+)"', line)
        if m:
            out.append(_inline(inc, m.group(1), seen))
        elif line.strip() != "#pragma once":
            out.append(line)
    return "".join(out)


def _get():
    global _kernel, ENABLED
    if _kernel is None:
        try:
            inc = os.path.join(os.path.dirname(mx.__file__), "include")
            seen = set()
            header = (_inline(inc, "mlx/backend/metal/kernels/steel/gemm/gemm.h", seen)
                      + _inline(inc, "mlx/backend/metal/kernels/quantized.h", seen) + _IMPL)
            _kernel = mx.fast.metal_kernel(name="mj_qmm_small", input_names=["x", "w", "scales", "biases"],
                                           output_names=["y"], source=_SOURCE, header=header)
        except OSError:  # headers missing in this mlx install: always fall back
            ENABLED = False
    return _kernel


def eligible(x, w, scales, biases, group_size, bits) -> bool:
    if (not ENABLED or bits != 4 or group_size != 64 or x.ndim != 2 or x.dtype not in (mx.bfloat16, mx.float16)
            or x.dtype != scales.dtype or biases is None):  # float32 runs (tests, references) stay on MLX's
        return False
    M, K = x.shape
    if not (MIN_ROWS <= M <= MAX_ROWS) or K % 64 or w.shape[1] * 8 != K:
        return False
    BN = CONFIGS[8 if M <= 8 else 16][2]
    return w.shape[0] % BN == 0


def quantized_matmul(x, w, scales, biases, group_size=64, bits=4):
    """x @ dequantize(w).T (transposed weights, mlx affine layout); the small-tile kernel when eligible."""
    if not eligible(x, w, scales, biases, group_size, bits):
        return mx.quantized_matmul(x, w, scales, biases, transpose=True, group_size=group_size, bits=bits)
    k = _get()
    if k is None:
        return mx.quantized_matmul(x, w, scales, biases, transpose=True, group_size=group_size, bits=bits)
    M, K = x.shape
    N = w.shape[0]
    BM, BK, BN, WM, WN = CONFIGS[8 if M <= 8 else 16]
    pad = BM - M
    if pad:  # whole tiles only (the bounds-checked tile load costs ~25%): zero rows in, their outputs dropped
        z = _pads.get((pad, K, x.dtype))
        if z is None:
            z = _pads[(pad, K, x.dtype)] = mx.zeros((pad, K), x.dtype)
            mx.eval(z)
        x = mx.concatenate([x, z])
    nt = 32 * WM * WN
    y = k(inputs=[x, w, scales, biases],
          template=[("T", x.dtype), ("K", K), ("N", N), ("GS", group_size), ("BM", BM), ("BK", BK), ("BN", BN),
                    ("WM", WM), ("WN", WN)],
          grid=((N // BN) * nt, 1, 1), threadgroup=(nt, 1, 1), output_shapes=[(BM, N)], output_dtypes=[x.dtype])[0]
    return y[:M] if pad else y
