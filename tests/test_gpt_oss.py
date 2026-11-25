"""MLX gpt-oss runner gates (plan P2-1 / P2-2, MoE target) on the tiny random checkpoint, float32, vs transformers.

Batched-gate tolerance (agreed with the user, 2026-11-16): every sequence must be token-identical to
transformers' batch-1 greedy output. A divergence is excused only where transformers' top-1 vs top-2 logit gap
is < 1e-5 (a true near-tie); the rest of that sequence is then not compared. The max logit difference is recorded.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).parent))
from hf_ref import hf_greedy, load_hf_gpt_oss, snapshot  # noqa: E402

from mj_mlx import Engine, GptOss  # noqa: E402
from mj_mlx.gpt_oss import build_meta, yarn_inv_freq  # noqa: E402

PATH = snapshot()
pytestmark = pytest.mark.skipif(PATH is None, reason="tiny gpt-oss checkpoint not downloaded")
NEAR_TIE = 1e-5


@pytest.fixture(scope="module")
def hf():
    return load_hf_gpt_oss(PATH)


def rand_prompts(seed, lens, prefix=()):
    rs = np.random.RandomState(seed)
    return [list(prefix) + [int(t) for t in rs.randint(0, 200000, n)] for n in lens]


def compare(ours, ref, ref_logits):
    """Token-identical, or a near-tie at the first divergence. Returns number of excused sequences."""
    excused = 0
    for i, (o, r) in enumerate(zip(ours, ref)):
        assert len(o) == len(r), f"seq {i}: length {len(o)} vs {len(r)}"
        div = next((j for j in range(len(r)) if o[j] != r[j]), None)
        if div is None:
            continue
        top2 = np.sort(ref_logits[i][div])[-2:]
        gap = float(top2[1] - top2[0])
        assert gap < NEAR_TIE, f"seq {i}: diverges at {div} with top-2 gap {gap:.3g} (not a near-tie)"
        excused += 1
    return excused


def test_yarn_matches_transformers():
    import torch
    from transformers import AutoConfig
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    cfg = AutoConfig.from_pretrained(PATH)
    inv, att = ROPE_INIT_FUNCTIONS["yarn"](cfg, torch.device("cpu"))
    model_cfg = GptOss(PATH, dtype=mx.float32).cfg
    ours, mscale = yarn_inv_freq(model_cfg)
    np.testing.assert_allclose(ours, inv.numpy(), rtol=1e-6, atol=0)  # float32 ulp level
    assert abs(mscale - att) < 1e-9


@pytest.mark.parametrize("dense", [True, False], ids=["dense", "mxfp4"])
def test_prefill_logits_match(hf, dense):
    import torch

    ids = rand_prompts(0, [300])[0]
    with torch.no_grad():
        ref = hf(torch.tensor([ids])).logits[0].numpy()
    m = GptOss(PATH, dtype=mx.float32, dense_experts=dense)
    bs, T = 16, len(ids)
    pool = m.new_kv_pool(64 * bs)
    bt = np.arange(7, 7 + (T + bs - 1) // bs)[None]  # non-contiguous-from-zero block ids
    meta = build_meta(np.array(ids), np.arange(T), np.array([0]), np.array([T]), bt, np.arange(T), bs,
                      m.cfg.sliding_window)
    ours = np.array(m.logits(meta, pool))
    diff = np.abs(ours - ref).max()
    print(f"max |logit diff| = {diff:.3g} (max |logit| {np.abs(ref).max():.3g})")
    assert diff < 1e-6
    assert (ours.argmax(-1) == ref.argmax(-1)).all()


@pytest.mark.parametrize("dense", [True, False], ids=["dense", "mxfp4"])
def test_batch1_greedy(hf, dense):
    prompts = rand_prompts(1, [5, 40, 129, 300])
    ref = [hf_greedy(hf, p, 40) for p in prompts]
    eng = Engine(PATH, dtype=mx.float32, dense_experts=dense, num_blocks=256, max_batch_tokens=64, max_running=1)
    ours = eng.generate(prompts, 40, ignore_eos=True, check=True)
    assert ours == [r[0] for r in ref]


def test_batched_tight_equals_roomy_and_matches_hf(hf):
    system = rand_prompts(2, [64])[0]
    lens = [3, 17, 60, 130, 7, 200, 33, 1, 90, 150, 12, 64, 5, 41, 129, 77]
    prompts = rand_prompts(3, lens, prefix=system)
    new = [20 + 3 * i for i in range(len(prompts))]
    ref = [hf_greedy(hf, p, n) for p, n in zip(prompts, new)]

    def run(num_blocks):
        eng = Engine(PATH, dtype=mx.float32, num_blocks=num_blocks, max_running=8, max_batch_tokens=96)
        rids = [eng.add_request(p, n, ignore_eos=True) for p, n in zip(prompts, new)]
        out, max_items = {}, 0
        while eng.has_work():
            for f in eng.step():
                out[f[0]] = list(f[2])
            eng.sched.check_invariants()
        return [out[r] for r in rids], eng.sched.stats()

    roomy, roomy_stats = run(1024)
    tight, tight_stats = run(40)
    print("roomy", roomy_stats, "\ntight", tight_stats)
    assert tight_stats["preemptions"] > 0
    assert roomy_stats["cached_prompt_tokens"] > 0
    assert roomy == tight
    excused = compare(roomy, [r[0] for r in ref], [r[1] for r in ref])
    print("near-tie excused sequences:", excused)


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("window", [0, 128])
def test_paged_decode_kernel_matches_sdpa(dtype, window):
    from mj_mlx.paged_attention import paged_decode_attention

    dt = getattr(mx, dtype)
    rs = np.random.RandomState(4)
    B, nq, nkv, D, bs, nblocks = 5, 64, 8, 64, 16, 200
    kv_lens = np.array([1, 17, 128, 129, 700])
    mb = (kv_lens.max() + bs - 1) // bs
    bt = np.stack([rs.permutation(nblocks)[:mb] for _ in range(B)]).astype(np.int32)
    kp = mx.array(rs.randn(nblocks * bs, nkv, D).astype(np.float32)).astype(dt)
    vp = mx.array(rs.randn(nblocks * bs, nkv, D).astype(np.float32)).astype(dt)
    q = mx.array(rs.randn(B, nq, D).astype(np.float32)).astype(dt)
    sinks = mx.array(rs.randn(nq).astype(np.float32))
    scale = 0.125
    out = paged_decode_attention(q, kp, vp, mx.array(bt), mx.array(kv_lens.astype(np.int32)), sinks, scale, bs,
                                 window)
    for b in range(B):
        lo = max(0, kv_lens[b] - window) if window else 0
        p = np.arange(lo, kv_lens[b])
        slots = mx.array((bt[b, p // bs] * bs + p % bs).astype(np.int32))
        k = kp[slots].transpose(1, 0, 2)[None]
        v = vp[slots].transpose(1, 0, 2)[None]
        ref = mx.fast.scaled_dot_product_attention(q[b][None, :, None], k, v, scale=scale,
                                                   sinks=sinks.astype(dt))[0, :, 0]
        tol = 1e-5 if dtype == "float32" else 2e-2
        np.testing.assert_allclose(np.array(out[b].astype(mx.float32)), np.array(ref.astype(mx.float32)),
                                   atol=tol, rtol=tol)


@pytest.mark.parametrize("T", [1, 3, 8, 64, 128])
def test_grouped_mxfp4_kernel_matches_gather_qmm(T):
    """Decode-size MoE through the grouped kernel equals the gather_qmm path (same MXFP4 weights)."""
    m = GptOss(PATH, dtype=mx.float32)
    e = m.layers[0]["experts"]
    k = m.cfg.num_experts_per_tok
    rs = np.random.RandomState(T)
    x = mx.array(rs.randn(T, m.cfg.hidden_size).astype(np.float32))
    logits = mx.array(rs.randn(T, e.num_experts).astype(np.float32))
    idx = mx.argpartition(logits, kth=-k, axis=-1)[..., -k:]
    wts = mx.softmax(mx.take_along_axis(logits, idx, axis=-1), axis=-1)
    e.kernel_max_pairs = 0
    ref = e(x, idx, wts)
    e.kernel_max_pairs = 1 << 30
    out = e(x, idx, wts)
    np.testing.assert_allclose(np.array(out), np.array(ref), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("bits", [8, 4])
def test_low_bit_lm_head_candidates_keep_greedy(bits):
    """bf16 model: picking candidates with a low-bit lm_head, then rescoring them in float32, gives the same
    greedy tokens as rescoring candidates from the full bf16 head."""
    prompts = rand_prompts(5, [40, 129, 300])
    outs = []
    for b in (None, bits):
        eng = Engine(model=GptOss(PATH, dtype=mx.bfloat16, lm_head_bits=b), num_blocks=256, max_running=4)
        outs.append(eng.generate(prompts, 30, ignore_eos=True))
    assert outs[0] == outs[1]


@pytest.mark.parametrize("V", [201088, 1000, 130 * 128 + 77])
def test_topk_indices_exact(V):
    from mj_mlx.gpt_oss import topk_indices

    lg = mx.random.normal((7, V), key=mx.random.key(V))
    got = np.sort(np.take_along_axis(np.array(lg), np.array(topk_indices(lg, 16)), -1), -1)
    ref = np.sort(np.array(lg), -1)[:, -16:]
    np.testing.assert_array_equal(got, ref)


def test_paged_decode_kernel_qwen_shape_no_sinks():
    """head_dim 256, 8 query heads per kv head, no sinks (Qwen3.6 full-attention layers)."""
    from mj_mlx.paged_attention import paged_decode_attention

    rs = np.random.RandomState(6)
    B, nq, nkv, D, bs, nblocks = 3, 16, 2, 256, 16, 64
    kv_lens = np.array([1, 50, 300])
    mb = (kv_lens.max() + bs - 1) // bs
    bt = np.stack([rs.permutation(nblocks)[:mb] for _ in range(B)]).astype(np.int32)
    kp = mx.array(rs.randn(nblocks * bs, nkv, D).astype(np.float32))
    vp = mx.array(rs.randn(nblocks * bs, nkv, D).astype(np.float32))
    q = mx.array(rs.randn(B, nq, D).astype(np.float32))
    out = paged_decode_attention(q, kp, vp, mx.array(bt), mx.array(kv_lens.astype(np.int32)),
                                 mx.full((nq,), -mx.inf), 1 / 16, bs, 0)
    for b in range(B):
        p = np.arange(kv_lens[b])
        slots = mx.array((bt[b, p // bs] * bs + p % bs).astype(np.int32))
        ref = mx.fast.scaled_dot_product_attention(q[b][None, :, None], kp[slots].transpose(1, 0, 2)[None],
                                                   vp[slots].transpose(1, 0, 2)[None], scale=1 / 16)[0, :, 0]
        np.testing.assert_allclose(np.array(out[b]), np.array(ref), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("T", [128, 300])
def test_fused_prefill_moe_glue_matches_ops(T, dtype):
    """Prefill-sized MoE steps (Experts._fused_sorted: bias + SwiGLU and bias + un-permute + weighted sum fused, float32
    inside) agree with the separate ops; float32 to rounding, bf16 within the ops' own bf16 rounding."""
    dt = getattr(mx, dtype)
    m = GptOss(PATH, dtype=dt)
    e = m.layers[0]["experts"]
    k = m.cfg.num_experts_per_tok
    rs = np.random.RandomState(T)
    x = mx.array(rs.randn(T, m.cfg.hidden_size).astype(np.float32)).astype(dt)
    logits = mx.array(rs.randn(T, e.num_experts).astype(np.float32))
    idx = mx.argpartition(logits, kth=-k, axis=-1)[..., -k:]
    wts = mx.softmax(mx.take_along_axis(logits, idx, axis=-1), axis=-1)
    assert idx.size >= e.FUSED_GLUE_MIN
    ref = np.array((e._unweighted(x, idx) * wts[..., None]).sum(axis=1).astype(mx.float32))
    out = np.array(e(x, idx, wts).astype(mx.float32))
    if dtype == "float32":
        np.testing.assert_allclose(out, ref, atol=1e-5, rtol=1e-5)
    else:  # one bf16 rounding of values of this size (the ops round after every step)
        assert np.abs(out - ref).max() <= 4 * np.abs(ref).max() * 2.0 ** -8


def test_short_steps_keep_the_op_sequence():
    """Decode and verify steps (< FUSED_GLUE_MIN pairs) must stay bitwise equal to decode: no fused glue there."""
    m = GptOss(PATH, dtype=mx.bfloat16)
    e = m.layers[0]["experts"]
    k = m.cfg.num_experts_per_tok
    rs = np.random.RandomState(5)
    for T in (1, 2, 4, 16):
        x = mx.array(rs.randn(T, m.cfg.hidden_size).astype(np.float32)).astype(mx.bfloat16)
        logits = mx.array(rs.randn(T, e.num_experts).astype(np.float32))
        idx = mx.argpartition(logits, kth=-k, axis=-1)[..., -k:]
        wts = mx.softmax(mx.take_along_axis(logits, idx, axis=-1), axis=-1)
        assert idx.size < e.FUSED_GLUE_MIN
        ref = (e._unweighted(x, idx) * wts[..., None]).sum(axis=1)
        assert mx.array_equal(e(x, idx, wts), ref).item()
