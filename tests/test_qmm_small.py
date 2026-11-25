"""Small-tile 4-bit quantized matmul (qmm_small.py): accuracy vs a float64 reference no worse than
mx.quantized_matmul's, a row's result independent of the other rows in its tile, and exact fallbacks."""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from mj_mlx import qmm_small  # noqa: E402

SHAPES = [(1024, 512), (512, 2048), (2080, 1024)]  # (N, K); 2080 = 65 x 32: last N tile of a 64-wide config is short


def weights(N, K, seed=0):
    mx.random.seed(seed)
    w = mx.random.normal((N, K)).astype(mx.bfloat16)
    q = mx.quantize(w, 64, 4)
    mx.eval(q)
    deq = np.array(mx.dequantize(*q, group_size=64, bits=4).astype(mx.float32)).astype(np.float64)
    return q, deq


def mlx_qmm(x, q):
    return mx.quantized_matmul(x, *q, transpose=True, group_size=64, bits=4)


@pytest.mark.parametrize("N,K", SHAPES)
@pytest.mark.parametrize("M", [5, 7, 8, 9, 13, 16])
def test_error_no_worse_than_mlx(N, K, M):
    q, deq = weights(N, K)
    x = (mx.random.normal((M, K)) * 2).astype(mx.bfloat16)
    if not qmm_small.eligible(x, *q, 64, 4):
        assert N % qmm_small.CONFIGS[8 if M <= 8 else 16][2]  # only a short N tile makes a shape ineligible
        assert bool((qmm_small.quantized_matmul(x, *q) == mlx_qmm(x, q)).all())
        return
    ref = np.array(x.astype(mx.float32)).astype(np.float64) @ deq.T
    err = lambda y: np.abs(np.array(y.astype(mx.float32)) - ref).mean()
    assert err(qmm_small.quantized_matmul(x, *q)) <= 1.05 * err(mlx_qmm(x, q))


@pytest.mark.parametrize("M", [6, 8, 12, 16])
def test_rows_independent_of_tile_mates(M):
    q, _ = weights(1024, 1024)
    x = mx.random.normal((M, 1024)).astype(mx.bfloat16)
    y = qmm_small.quantized_matmul(x, *q)
    other = mx.random.normal((M, 1024)).astype(mx.bfloat16)
    for i in range(M):  # row i with different tile-mates (and a different position in the tile)
        mixed = mx.concatenate([other[: M - 1 - i], x[i:i + 1], other[M - 1 - i: M - 1]])
        assert bool((qmm_small.quantized_matmul(mixed, *q)[M - 1 - i] == y[i]).all())


@pytest.mark.parametrize("case", ["rows4", "rows17", "bits8", "float32", "disabled"])
def test_fallbacks_are_mlx_exactly(case):
    M = {"rows4": 4, "rows17": 17}.get(case, 8)
    w = mx.random.normal((512, 512))
    dt = mx.float32 if case == "float32" else mx.bfloat16
    bits = 8 if case == "bits8" else 4
    q = mx.quantize(w.astype(dt), 64, bits)
    x = mx.random.normal((M, 512)).astype(dt)
    assert not (case != "disabled" and qmm_small.eligible(x, *q, 64, bits))
    qmm_small.ENABLED = case != "disabled"
    try:
        got = qmm_small.quantized_matmul(x, *q, group_size=64, bits=bits)
    finally:
        qmm_small.ENABLED = True
    ref = mx.quantized_matmul(x, *q, transpose=True, group_size=64, bits=bits)
    assert bool((got == ref).all())
