# Copyright 2026 the MilliJoule authors
# SPDX-License-Identifier: Apache-2.0
"""4-bit matvec for 2-4 rows with each row's float ops exactly those of MLX's one-row kernel (qmv_fast), so row r of
the result is bit-identical to mx.quantized_matmul(x[r:r+1], ...). The weights are read once for all rows.

Why: speculative decoding verifies k+1 = 2-4 tokens in one step. MLX's own 2-4 row path rounds differently from its
one-row path, and on Qwen3.5-9B it is measurably less accurate against float32 (2-token steps: mean KL 2.7e-4 vs
2.0e-4 with every row through the one-row kernel; tools/mlx_vs_mlxlm.py). With this kernel a verify step computes
the same numbers as plain decoding, so speculative greedy output equals plain greedy output.

The source is generated per row count (explicit scalars, no thread arrays). It mirrors qmv_fast for bits=4, gs=64:
per simdgroup `rps` output rows, each lane 16 inputs per 512-wide block, x pre-scaled by 1, 1/16, 1/256, 1/4096
against the masked nibbles, accum = s * dot + sum(x) * b, simd_sum at the end.
"""

from __future__ import annotations

import mlx.core as mx

ENABLED = True
MIN_ROWS, MAX_ROWS = 2, 4
RPS, NSG = 2, 4  # default: output rows per simdgroup, simdgroups per threadgroup
CONFIGS = {2: (2, 2), 3: (2, 4), 4: (2, 2)}  # per row count, fastest on M4 over the 9B shapes and its lm_head


def _gen(rows: int, rps: int) -> str:
    L = []
    a = L.append
    a("  constexpr int block_size = 512; constexpr int scale_step = GS / 16;")
    a("  const uint simd_gid = simdgroup_index_in_threadgroup; const uint simd_lid = thread_index_in_simdgroup;")
    a(f"  const int out_row = threadgroup_position_in_grid.y * (NSG * {rps}) + simd_gid * {rps};")
    a("  const int in_w = K / 2; const int in_g = K / GS;")
    a("  const device uint8_t* ws = (const device uint8_t*)w + size_t(out_row) * in_w + simd_lid * 8;")
    a("  const device T* sc = scales + size_t(out_row) * in_g + simd_lid / scale_step;")
    a("  const device T* bi = biases + size_t(out_row) * in_g + simd_lid / scale_step;")
    a("  const device T* xp = x + simd_lid * 16;")
    for m in range(rows):
        for r in range(rps):
            a(f"  float acc_{m}_{r} = 0;")
    a("  for (int k = 0; k < K; k += block_size) {")
    for r in range(rps):
        a(f"    const device uint16_t* wl{r} = (const device uint16_t*)(ws + {r} * in_w);")
        for i in range(4):
            a(f"    const uint16_t q{r}_{i} = wl{r}[{i}];")
        for i in range(4):
            for j, mask in enumerate(("0x000f", "0x00f0", "0x0f00", "0xf000")):
                a(f"    const float w{r}_{4 * i + j} = q{r}_{i} & {mask};")
        a(f"    const float s{r} = sc[{r} * in_g]; const float b{r} = bi[{r} * in_g];")
    for m in range(rows):
        a("    {")
        a(f"      const device T* xm = xp + {m} * K + k;")
        a("      float sum = 0;")
        xs = []
        for i in range(0, 16, 4):
            a(f"      sum += float(xm[{i}]) + float(xm[{i + 1}]) + float(xm[{i + 2}]) + float(xm[{i + 3}]);")
            a(f"      const float x{i} = xm[{i}]; const float x{i + 1} = xm[{i + 1}] / 16.0f;"
              f" const float x{i + 2} = xm[{i + 2}] / 256.0f; const float x{i + 3} = xm[{i + 3}] / 4096.0f;")
            xs += [f"x{i}", f"x{i + 1}", f"x{i + 2}", f"x{i + 3}"]
        for r in range(rps):
            a("      { float accum = 0;")
            for i in range(4):
                t = " + ".join(f"{xs[4 * i + j]} * w{r}_{4 * i + j}" for j in range(4))
                a(f"        accum += ({t});")
            a(f"        acc_{m}_{r} += s{r} * accum + sum * b{r}; }}")
        a("    }")
    a("    ws += block_size / 2; sc += block_size / GS; bi += block_size / GS;")
    a("  }")
    for m in range(rows):
        for r in range(rps):
            a(f"  {{ float v = simd_sum(acc_{m}_{r}); if (simd_lid == 0) y[{m} * N + out_row + {r}] ="
              f" static_cast<T>(v); }}")
    return "\n".join(L)


_kernels: dict = {}


def eligible(x, w, scales, biases, group_size, bits) -> bool:
    if not ENABLED or bits != 4 or group_size != 64 or biases is None or x.ndim != 2:
        return False
    M, K = x.shape
    N = w.shape[0]
    rps, nsg = CONFIGS.get(M, (RPS, NSG))
    return (MIN_ROWS <= M <= MAX_ROWS and K % 512 == 0 and N % (rps * nsg) == 0 and x.dtype == scales.dtype
            and x.dtype in (mx.bfloat16, mx.float16))


def quantized_matmul(x, w, scales, biases, group_size: int = 64, bits: int = 4):
    """y = x @ dequantize(w).T for x [M, K], 2 <= M <= 4; anything else goes to mx.quantized_matmul."""
    if not eligible(x, w, scales, biases, group_size, bits):
        return mx.quantized_matmul(x, w, scales, biases, transpose=True, group_size=group_size, bits=bits)
    M, K = x.shape
    N = w.shape[0]
    rps, nsg = CONFIGS.get(M, (RPS, NSG))
    key = (M, rps)
    if key not in _kernels:
        _kernels[key] = mx.fast.metal_kernel(name=f"mj_qmv_rows_{M}_{rps}", input_names=["x", "w", "scales", "biases"],
                                             output_names=["y"], source=_gen(M, rps))
    return _kernels[key](inputs=[x, w, scales, biases],
                         template=[("T", x.dtype), ("K", K), ("N", N), ("GS", 64), ("NSG", nsg)],
                         grid=(32, nsg * (N // (rps * nsg)), 1), threadgroup=(32, nsg, 1), output_shapes=[(M, N)],
                         output_dtypes=[x.dtype])[0]
