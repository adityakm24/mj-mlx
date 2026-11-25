"""2-4 row 4-bit matvec (qmv_rows.py): each row bit-identical to MLX's one-row quantized matmul; exact fallbacks."""

import pytest

mx = pytest.importorskip("mlx.core")

from mj_mlx import qmv_rows  # noqa: E402


@pytest.mark.parametrize("N,K", [(1024, 512), (512, 2048), (2056, 1024)])
@pytest.mark.parametrize("M", [2, 3, 4])
@pytest.mark.parametrize("dt", ["bfloat16", "float16"])
def test_rows_bit_identical_to_one_row(N, K, M, dt):
    dt = getattr(mx, dt)
    mx.random.seed(M)
    q = mx.quantize(mx.random.normal((N, K)).astype(dt), 64, 4)
    x = (mx.random.normal((M, K)) * 3).astype(dt)
    one = mx.concatenate([mx.quantized_matmul(x[i:i + 1], *q, transpose=True, group_size=64, bits=4)
                          for i in range(M)])
    assert qmv_rows.eligible(x, *q, 64, 4)
    assert bool((qmv_rows.quantized_matmul(x, *q) == one).all())


@pytest.mark.parametrize("case", ["rows1", "rows5", "bits8", "float32", "k_not_512"])
def test_fallbacks_are_mlx(case):
    M = {"rows1": 1, "rows5": 5}.get(case, 3)
    K = 576 if case == "k_not_512" else 512
    dt = mx.float32 if case == "float32" else mx.bfloat16
    bits = 8 if case == "bits8" else 4
    q = mx.quantize(mx.random.normal((256, K)).astype(dt), 64, bits)
    x = mx.random.normal((M, K)).astype(dt)
    assert not qmv_rows.eligible(x, *q, 64, bits)
    got = qmv_rows.quantized_matmul(x, *q, group_size=64, bits=bits)
    assert bool((got == mx.quantized_matmul(x, *q, transpose=True, group_size=64, bits=bits)).all())
