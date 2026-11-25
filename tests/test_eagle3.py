"""EAGLE-3 speculative decoding for gpt-oss (eagle3.py) on the tiny random gpt-oss checkpoint, float32, with a random
EAGLE-3 head: greedy output equals plain greedy for 1-3 drafts, for oracle drafts (all accepted) and half-wrong ones."""

import sys
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
sys.path.insert(0, str(Path(__file__).parent))
from hf_ref import snapshot  # noqa: E402

from mj_mlx import Engine, GptOss  # noqa: E402
from mj_mlx.eagle3 import Eagle3Generator  # noqa: E402

PATH = snapshot()
pytestmark = pytest.mark.skipif(PATH is None, reason="tiny gpt-oss checkpoint not downloaded")
AUX = (0, 1, 1)  # the tiny model has 2 layers; a real head uses (2, L/2, L-3)


@pytest.fixture(scope="module")
def head(tmp_path_factory):
    H, nq, nkv, D, I, Vd = 32, 4, 2, 64, 48, 512
    rng = np.random.RandomState(0)
    r = lambda *s: mx.array((0.2 * rng.randn(*s)).astype(np.float32))
    w = {"fc.weight": r(H, 3 * H), "layers.0.self_attn.q_proj.weight": r(nq * D, 2 * H),
         "layers.0.self_attn.k_proj.weight": r(nkv * D, 2 * H), "layers.0.self_attn.v_proj.weight": r(nkv * D, 2 * H),
         "layers.0.self_attn.o_proj.weight": r(H, nq * D), "layers.0.mlp.gate_proj.weight": r(I, H),
         "layers.0.mlp.up_proj.weight": r(I, H), "layers.0.mlp.down_proj.weight": r(H, I), "lm_head.weight": r(Vd, H)}
    for n in ("layers.0.input_layernorm", "layers.0.hidden_norm", "layers.0.post_attention_layernorm", "norm"):
        w[f"{n}.weight"] = mx.array((1 + 0.1 * rng.randn(H)).astype(np.float32))
    w["d2t"] = mx.array(np.zeros(Vd, np.int32))  # draft id i -> target id i
    p = str(tmp_path_factory.mktemp("eagle3") / "head.safetensors")
    mx.save_safetensors(p, w)
    return p


def plain(model, ids, n):
    return Engine(model=model, num_blocks=32, max_running=1, max_batch_tokens=256).generate([ids], n, ignore_eos=True)[0]


PROMPTS = [[int(t) for t in np.random.RandomState(s).randint(0, 512, n)] for s, n in ((0, 9), (1, 30))]


@pytest.mark.parametrize("k", [1, 2, 3])
def test_random_head_equals_plain(head, k):
    model = GptOss(PATH, dtype=mx.float32)
    g = Eagle3Generator(model, head, k=k, max_len=256, bits=None, emb_bits=None, aux_layers=AUX, chunk=16)
    for ids in PROMPTS:
        assert g.generate(ids, 24) == plain(model, ids, 24)


@pytest.mark.parametrize("wrong_every", [0, 2])
def test_oracle_drafts(head, wrong_every):
    model = GptOss(PATH, dtype=mx.float32)
    g = Eagle3Generator(model, head, k=3, max_len=256, bits=None, emb_bits=None, aux_layers=AUX)
    for ids in PROMPTS:
        ref = plain(model, ids, 24)
        full = ids + ref + [0] * 8
        calls = [0]
        real = g._drafts

        def oracle(aux, next_ids, start, k):
            real(aux, next_ids, start, k)
            p = start + aux.shape[0] + 1
            d = list(full[p:p + k])
            calls[0] += 1
            if wrong_every and calls[0] % wrong_every == 0:
                d[0] = (d[0] + 1) % 512
            return mx.array(np.array(d, np.int32))

        g._drafts = oracle
        g.stats = dict(iters=0, drafted=0, accepted=0)
        assert g.generate(ids, 24) == ref
        assert g.stats["accepted"] > 0
        g._drafts = real


def test_adaptive_zero_draft_backfill(head):
    """A policy that drafts rarely: rows queue up at k = 0 and the head catches up every SHADOW_EVERY iterations;
    output stays plain greedy."""
    from mj_mlx.spec_decode import DraftPolicy

    model = GptOss(PATH, dtype=mx.float32)
    g = Eagle3Generator(model, head, max_len=256, bits=None, emb_bits=None, aux_layers=AUX,
                        policy=DraftPolicy(1, mode="time", cost=[1.0, 1.9], p0=0.5))
    for ids in PROMPTS:
        g.policy = DraftPolicy(1, mode="time", cost=[1.0, 1.9], p0=0.5)
        assert g.generate(ids, 40) == plain(model, ids, 40)
    assert g.stats["k_hist"][0] > 0
