"""bf16_pack: lossless 12.25-bit bf16 weights. Round trip is bit-identical (escapes, zeros, signs, extremes); the
matvec matches float32 math as closely as MLX's own bf16 matmul; gpt-oss with packed attention computes the same
prefill numbers as without and decodes the same tokens on the tiny checkpoint."""

import sys
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mj_mlx import bf16_pack as B  # noqa: E402


def rand_bf16(N, K, seed):
    rng = np.random.RandomState(seed)
    w = (rng.randn(N, K) * 0.05).astype(np.float32)
    w[rng.rand(N, K) < 0.01] *= 1e-6  # tiny weights: escapes
    w[rng.rand(N, K) < 0.005] = 0.0
    w[0, :3] = [-0.0, 3.0e38, -1.0e-38]
    return mx.array(w).astype(mx.bfloat16)


@pytest.mark.parametrize("N,K", [(64, 256), (520, 2880), (36, 4096)])
def test_round_trip_bit_identical(N, K):
    w = rand_bf16(N, K, N + K)
    p = B.pack(w)
    assert p["esc_k"].size > 0
    assert mx.array_equal(B.unpack(p).view(mx.uint16), w.view(mx.uint16)).item()
    assert B.nbytes(p) < 0.85 * w.nbytes  # ~1.5% escapes here; real gpt-oss attention: 0.768 (2e-4 escapes)


def test_size_without_tiny_weights():
    w = mx.array((np.random.RandomState(0).randn(512, 2880) * 0.05).astype(np.float32)).astype(mx.bfloat16)
    assert B.nbytes(B.pack(w)) < 0.775 * w.nbytes


@pytest.mark.parametrize("R", [1, 2, 3, 4])
@pytest.mark.parametrize("bias", [False, True])
def test_matvec_accuracy(R, bias):
    N, K = 520, 2880
    w = rand_bf16(N, K, 7)
    p = B.pack(w)
    x = (mx.random.normal((R, K), key=mx.random.key(R)) * 0.5).astype(mx.bfloat16)
    b = (mx.random.normal((N,), key=mx.random.key(9)) * 0.1).astype(mx.bfloat16) if bias else None
    ref = x.astype(mx.float32) @ w.astype(mx.float32).T + (b.astype(mx.float32) if bias else 0)
    y = B.matvec(x, p, b)
    ym = mx.addmm(b, x, w.T) if bias else x @ w.T
    err = float(mx.abs(y.astype(mx.float32) - ref).max())
    err_mlx = float(mx.abs(ym.astype(mx.float32) - ref).max())
    assert y.dtype == mx.bfloat16 and err <= 1.5 * err_mlx + 1e-6


def test_matvec_float32_includes_escapes_exactly():
    """float32 inputs and output: the result is the float32 dot product, escapes (tiny weights) included."""
    N, K = 64, 512
    w = rand_bf16(N, K, 3)
    p = B.pack(w)
    x = mx.random.normal((2, K), key=mx.random.key(0))
    ref = x @ w.astype(mx.float32).T
    y = B.matvec(x, p)
    assert float(mx.abs(y - ref).max()) < 1e-5 * float(mx.abs(ref).max())
    zero_esc = dict(p, esc_v=mx.zeros_like(p["esc_v"]))  # without the escapes the result must change
    assert float(mx.abs(B.matvec(x, zero_esc) - ref).max()) > 0


def test_gpt_oss_attn_pack_tiny():
    from hf_ref import snapshot

    from mj_mlx import Engine, GptOss

    path = snapshot()
    if path is None:
        pytest.skip("tiny gpt-oss checkpoint not downloaded")
    plain = GptOss(path, dtype=mx.bfloat16)
    packed = GptOss(path, dtype=mx.bfloat16, attn_pack=True)
    assert "qkv_p" in packed.layers[0] and packed.layers[0]["qkv_wt"] is None
    prompts = [[int(t) for t in np.random.RandomState(s).randint(0, 1000, n)] for s, n in ((0, 7), (1, 40))]
    outs = []
    for m in (plain, packed):
        eng = Engine(model=m, num_blocks=32, max_running=2, max_batch_tokens=256)
        outs.append(eng.generate(prompts, 16, ignore_eos=True))
    assert outs[0] == outs[1]
