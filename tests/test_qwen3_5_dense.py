"""Dense Qwen3.5 (hybrid DeltaNet + gated attention, plain SwiGLU MLP) vs transformers float32.

No checkpoint download: a tiny random Qwen3_5ForCausalLM is built from a config here (seeded, norms and DeltaNet
parameters perturbed so zero-init shortcuts cannot hide bugs), saved with save_pretrained, and loaded by both."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import mlx.core as mx  # noqa: E402

from mj_mlx import Engine, Qwen35Moe  # noqa: E402
from mj_mlx.runner import StepArrays, load_model  # noqa: E402

torch = pytest.importorskip("torch")
tf = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def tiny_dense(tmp_path_factory):
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
    cfg._attn_implementation = "eager"
    m = Qwen3_5ForCausalLM(cfg).to(torch.float32).eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "norm" in n or n.endswith(("A_log", "dt_bias")):
                p.add_(0.3 * torch.randn_like(p))
    out = tmp_path_factory.mktemp("tiny_qwen35_dense")
    m.save_pretrained(out, safe_serialization=True)
    return m, str(out)


def hf_logits(m, ids):
    with torch.no_grad():
        return m(input_ids=torch.tensor([ids])).logits[0].numpy()


def ours_logits(model, ids):
    n = len(ids)
    nb = (n + 15) // 16
    pools = model.new_pools((nb + 4) * 16, 2)
    s = StepArrays(np.array(ids, np.uint32), np.arange(n, dtype=np.int32), np.zeros(1, np.uint64),
                   np.array([0], np.int32), np.array([n], np.int32), np.array([True]),
                   np.arange(1, 1 + nb)[None].astype(np.int32), np.arange(n, dtype=np.int32), np.zeros(1, np.uint64),
                   [], None, state_slots=np.array([1], np.int32))
    x = model.hidden(model.prepare(s, 16, True), pools)
    lg = mx.fast.rms_norm(x.astype(mx.float32), model.norm.astype(mx.float32), model.cfg.rms_norm_eps) @ \
        model.lm_head.astype(mx.float32).T
    mx.eval(lg)
    return np.array(lg)


def test_dense_prefill_logits_match_transformers(tiny_dense):
    hf, path = tiny_dense
    model = load_model(path, dtype=mx.float32)
    assert isinstance(model, Qwen35Moe) and model.cfg.num_experts == 0
    ids = [int(t) for t in np.random.RandomState(3).randint(0, 512, 150)]
    ref, ours = hf_logits(hf, ids), ours_logits(model, ids)
    assert np.abs(ours - ref).max() < 1e-5 * max(1.0, np.abs(ref).max())
    assert (ours.argmax(-1) == ref.argmax(-1)).all()


def test_dense_batched_greedy_matches_transformers(tiny_dense):
    hf, path = tiny_dense
    rs = np.random.RandomState(9)
    ps = [[int(t) for t in rs.randint(0, 512, n)] for n in (5, 40, 77)]
    ref = []
    for p in ps:
        with torch.no_grad():
            out = hf.generate(input_ids=torch.tensor([p]), max_new_tokens=16, do_sample=False, eos_token_id=None,
                              pad_token_id=0)
        ref.append(out[0, len(p):].tolist())
    ours = Engine(path, dtype=mx.float32, num_blocks=64, max_running=3, max_batch_tokens=64).generate(
        ps, 16, ignore_eos=True)
    assert ours == ref
