"""MTP speculative decoding (spec_decode.py) on a tiny random dense Qwen3.5, float32: greedy output equals plain
greedy decoding whatever the drafts are. Random MTP weights (drafts mostly rejected: the rollback path), oracle
drafts (all accepted) and half-wrong oracle drafts (partial acceptance) are all checked."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

mx = pytest.importorskip("mlx.core")
torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from mj_mlx import Engine  # noqa: E402
from mj_mlx.runner import load_model  # noqa: E402
from mj_mlx.spec_decode import DraftPolicy, SpecGenerator  # noqa: E402


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    torch.manual_seed(0)
    cfg = Qwen3_5TextConfig(
        vocab_size=512, hidden_size=64, intermediate_size=96, num_hidden_layers=4,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, linear_conv_kernel_dim=4,
        linear_key_head_dim=16, linear_value_head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4,
        rms_norm_eps=1e-6, tie_word_embeddings=False,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25},
    )
    m = Qwen3_5ForCausalLM(cfg).to(torch.float32).eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "norm" in n or n.endswith(("A_log", "dt_bias")):
                p.add_(0.3 * torch.randn_like(p))
    out = tmp_path_factory.mktemp("tiny_qwen35_mtp")
    m.save_pretrained(out, safe_serialization=True)
    H, I, nq, nkv, D = 64, 96, 4, 2, 16
    rng = np.random.RandomState(1)
    shapes = {"fc.weight": (H, 2 * H), "layers.0.self_attn.q_proj.weight": (2 * nq * D, H),
              "layers.0.self_attn.k_proj.weight": (nkv * D, H), "layers.0.self_attn.v_proj.weight": (nkv * D, H),
              "layers.0.self_attn.o_proj.weight": (H, nq * D), "layers.0.mlp.gate_proj.weight": (I, H),
              "layers.0.mlp.up_proj.weight": (I, H), "layers.0.mlp.down_proj.weight": (H, I)}
    w = {f"mtp.{k}": mx.array((0.2 * rng.randn(*s)).astype(np.float32)) for k, s in shapes.items()}
    for k, n in (("pre_fc_norm_embedding", H), ("pre_fc_norm_hidden", H), ("norm", H),
                 ("layers.0.input_layernorm", H), ("layers.0.post_attention_layernorm", H),
                 ("layers.0.self_attn.q_norm", D), ("layers.0.self_attn.k_norm", D)):
        w[f"mtp.{k}.weight"] = mx.array((0.1 * rng.randn(n)).astype(np.float32))
    mtp = str(out / "mtp.safetensors")
    mx.save_safetensors(mtp, w)
    return str(out), mtp


def plain(model, ids, n):
    eng = Engine(model=model, num_blocks=32, max_running=1, max_batch_tokens=256, num_state_slots=1)
    return eng.generate([ids], n, ignore_eos=True)[0]


PROMPTS = [[int(t) for t in np.random.RandomState(s).randint(0, 512, n)] for s, n in ((0, 9), (1, 23), (2, 40))]


@pytest.mark.parametrize("k", [1, 2, 3])
def test_random_drafts_equal_plain(tiny, k):
    path, mtp = tiny
    model = load_model(path, dtype=mx.float32)
    g = SpecGenerator(model, mtp, k=k, max_len=256, mtp_bits=None, chunk=16)
    for ids in PROMPTS:
        assert g.generate(ids, 30) == plain(model, ids, 30)
    assert g.stats["iters"] > 0


@pytest.mark.parametrize("pipeline", ["defer", "eager", "off"])
@pytest.mark.parametrize("k,wrong_every", [(2, 0), (3, 0), (3, 2), (2, 3)])
def test_oracle_drafts_equal_plain(tiny, k, wrong_every, pipeline):
    """Drafts are the true next tokens (all accepted), or every `wrong_every`-th draft is corrupted. With the
    pipeline, the next verify step is queued assuming acceptance, and a rejection discards it."""
    path, mtp = tiny
    model = load_model(path, dtype=mx.float32)
    g = SpecGenerator(model, mtp, k=k, max_len=256, mtp_bits=None, pipeline=pipeline)
    for ids in PROMPTS:
        ref = plain(model, ids, 30)
        full = ids + ref + [0] * (k + 2)
        calls = [0]

        def oracle(hid32, next_ids, start, kk):
            p = start + hid32.shape[0] + 1  # position of the first draft
            d = list(full[p:p + kk])
            calls[0] += 1
            if wrong_every and calls[0] % wrong_every == 0:
                d[calls[0] % kk] = (d[calls[0] % kk] + 1) % 512
            return mx.array(np.array(d, np.int32))

        g._drafts = oracle
        g.reset_stats()
        assert g.generate(ids, 30) == ref
        if not wrong_every:
            assert g.stats["accepted"] == g.stats["drafted"] or g.stats["iters"] * (k + 1) >= 29
        if pipeline != "off":
            assert g.stats["ahead_used"] > 0
            if wrong_every:
                assert g.stats["ahead_used"] < g.stats["ahead_queued"]  # some were discarded
        else:
            assert g.stats["ahead_queued"] == 0


def test_zero_drafts_equal_plain(tiny):
    """k = 0: plain decoding through the speculative loop (the MTP layer never runs after prefill)."""
    path, mtp = tiny
    model = load_model(path, dtype=mx.float32)
    g = SpecGenerator(model, mtp, k=0, max_len=256, mtp_bits=None, chunk=16)
    for ids in PROMPTS:
        assert g.generate(ids, 30) == plain(model, ids, 30)
    assert g.stats["drafted"] == 0 and g.stats["k_hist"][0] == g.stats["iters"]


@pytest.mark.parametrize("mode", ["time", "energy"])
def test_adaptive_random_drafts_equal_plain(tiny, mode):
    """Random MTP weights: acceptance is low, the energy policy drops to k = 0 with probes (MTP KV backfilled when
    drafting resumes); the output is plain greedy either way."""
    path, mtp = tiny
    model = load_model(path, dtype=mx.float32)
    cost = [1.0, 1.6, 2.0, 2.4] if mode == "energy" else None
    pol = DraftPolicy(3, mode=mode, cost=cost, probe=4)
    g = SpecGenerator(model, mtp, max_len=256, mtp_bits=None, chunk=16, policy=pol)
    for ids in PROMPTS:
        assert g.generate(ids, 40) == plain(model, ids, 40)
    if mode == "energy":
        assert g.stats["k_hist"][0] > 0 and sum(g.stats["k_hist"][1:]) > 0


def test_adaptive_oracle_mixed(tiny):
    """Oracle drafts that turn wrong for a stretch, then right again: the policy goes 3 -> 0 -> back up, and the
    output (including MTP KV backfill after k = 0 runs) stays plain greedy."""
    path, mtp = tiny
    model = load_model(path, dtype=mx.float32)
    pol = DraftPolicy(3, mode="energy", cost=[1.0, 1.5, 1.8, 2.1], probe=3, alpha=0.4, drift=0.2)
    g = SpecGenerator(model, mtp, max_len=256, mtp_bits=None, policy=pol)
    real = g._drafts
    for ids in PROMPTS:
        ref = plain(model, ids, 60)
        full = ids + ref + [0] * 8
        seen = []

        def oracle(hid32, next_ids, start, kk):
            real(hid32, next_ids, start, kk)  # keep the MTP layer's KV path exercised
            p = start + hid32.shape[0] + 1
            seen.append((p, kk))
            d = list(full[p:p + kk])
            if len(ids) + 15 <= p < len(ids) + 35:
                d = [(t + 1) % 512 for t in d]
            return mx.array(np.array(d, np.int32))

        g._drafts = oracle
        assert g.generate(ids, 60) == ref
        ks = [kk for _, kk in seen]
        assert 3 in ks and max(ks[-3:]) >= 2


def test_policy_choice():
    pol = DraftPolicy(3, mode="energy", cost=[1.0, 1.3, 1.5, 1.7], p0=0.9)
    assert pol.choose() == 3
    for _ in range(40):
        pol.update(pol.choose(), 0, None)  # every draft rejected
    # it stops drafting once one draft no longer pays: (1 + p1) / 1.3 < 1, i.e. p1 < 0.3
    assert pol.p[1] < 0.3 and pol.choose() == 0
    # probes: with k = 0 chosen repeatedly, one draft is tried every `probe` iterations
    seq = []
    for _ in range(16):
        k = pol.choose()
        seq.append(k)
        pol.update(k, 0, None)
    assert seq.count(1) == 2 and set(seq) <= {0, 1}
    assert abs(DraftPolicy(2, p0=0.5).expected(2) - (1 + 0.5 + 0.5 * 0.5 * DraftPolicy.DEPTH_DECAY)) < 1e-9


@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_batch_equals_plain(tiny, k):
    """SpecBatch: three requests of different lengths decoding together, fixed k, random MTP weights."""
    from mj_mlx.spec_decode import SpecBatch

    path, mtp = tiny
    model = load_model(path, dtype=mx.float32)
    g = SpecBatch(model, mtp, max_batch=4, k=k, max_len=256, mtp_bits=None, chunk=16)
    out = g.generate(PROMPTS, 30)
    assert out == [plain(model, ids, 30) for ids in PROMPTS]


def test_batch_adaptive_and_oracle(tiny):
    """Per-request policies (energy mode, batch-scaled cost) and oracle drafts for one request, wrong for another."""
    from mj_mlx.spec_decode import SpecBatch

    path, mtp = tiny
    model = load_model(path, dtype=mx.float32)
    refs = [plain(model, ids, 40) for ids in PROMPTS]
    g = SpecBatch(model, mtp, max_batch=4, max_len=256, mtp_bits=None, cost=[1.0, 1.4, 1.7, 2.0],
                  policy=lambda: DraftPolicy(3, mode="energy", probe=3))
    assert g.generate(PROMPTS, 40) == refs
    full = [ids + r + [0] * 8 for ids, r in zip(PROMPTS, refs)]
    real = g._drafts

    def oracle(h_rows, id_rows, starts, lens, slots, kmax):
        real(h_rows, id_rows, starts, lens, slots, kmax)
        rows = []
        for st, ln, sl in zip(starts, lens, slots):
            p = st + ln + 1
            d = full[sl][p:p + kmax]
            if sl == 1:
                d = [(t + 1) % 512 for t in d]  # request 1: always wrong
            rows.append(d)
        return mx.array(np.array(rows, np.int32))

    g = SpecBatch(model, mtp, max_batch=4, k=3, max_len=256, mtp_bits=None)
    g._drafts = oracle
    assert g.generate(PROMPTS, 40) == refs
    assert g.stats["accepted"] > 0


def test_policy_edp_prefers_faster_when_energy_ties():
    """Energy per token equal for k = 0 and 2 (one request, 9B-like table), k = 2 twice as fast: edp picks 2."""
    pol = DraftPolicy(3, mode="edp", cost=[0.43 * 0.052, 0.76 * 0.057, 1.03 * 0.059, 1.17 * 0.075], p0=0.7)
    assert pol.choose() == 2
    assert DraftPolicy(3, mode="energy", cost=[0.43, 0.76, 1.03, 1.17], p0=0.7).choose() == 0


def test_policy_seeded_time_cost_survives_one_slow_sample():
    """A calibrated time table is blended, not replaced: one cold 90 ms sample for k = 1 must not lock it out."""
    pol = DraftPolicy(1, mode="time", cost=[0.032, 0.047], p0=0.74)
    assert pol.choose() == 1
    pol.update(1, 1, 0.090)  # one slow first sample
    for _ in range(3):
        pol.update(0, 0, 0.032)
        pol.update(1, 1, None)  # shadow drafts: depth 1 keeps being accepted
    assert pol.choose() == 1
