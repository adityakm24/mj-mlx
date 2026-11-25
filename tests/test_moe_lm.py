"""MoeLM (moe_lm.py) against transformers float32, one tiny random checkpoint per MoE family, built locally.

Gates (as for gpt-oss and Qwen3.5): prefill logits of every position within 1e-4 of transformers, and greedy
tokens from the Engine (chunked prefill, paged decode, several requests at once) identical to transformers'
batch-1 greedy, except at a true near-tie (top-2 gap < 1e-5).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).parent))
from hf_ref import hf_greedy  # noqa: E402

from mj_mlx import Engine  # noqa: E402
from mj_mlx.gpt_oss import build_meta  # noqa: E402
from mj_mlx.moe_lm import MoeLM, MoeLMConfig, rope_inv_freq  # noqa: E402

V = 512
BASE = dict(vocab_size=V, hidden_size=64, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
            num_hidden_layers=3, initializer_range=0.25, max_position_embeddings=512, rms_norm_eps=1e-6,
            tie_word_embeddings=False)
YARN = dict(rope_type="yarn", rope_theta=10000.0, factor=4.0, original_max_position_embeddings=64, mscale=0.707,
            mscale_all_dim=0.707, beta_fast=32, beta_slow=1)
MLA = dict(num_key_value_heads=4, kv_lora_rank=32, qk_rope_head_dim=32, qk_nope_head_dim=32, v_head_dim=32, head_dim=32)
FAMILY_CFG = {
    "mixtral": dict(num_local_experts=8, num_experts_per_tok=2, intermediate_size=48, sliding_window=8),
    "qwen2_moe": dict(num_experts=8, num_experts_per_tok=2, moe_intermediate_size=32, intermediate_size=48,
                      shared_expert_intermediate_size=40, mlp_only_layers=[1], norm_topk_prob=False),
    "qwen3_moe": dict(num_experts=8, num_experts_per_tok=2, moe_intermediate_size=32, intermediate_size=48,
                      norm_topk_prob=True),
    "olmoe": dict(num_attention_heads=2, num_key_value_heads=2, num_experts=8, num_experts_per_tok=2, intermediate_size=32, norm_topk_prob=False),
    "deepseek_v2": dict(MLA, n_routed_experts=8, num_experts_per_tok=2, moe_intermediate_size=32, intermediate_size=48,
                        n_shared_experts=1, first_k_dense_replace=1, topk_method="group_limited_greedy", n_group=4,
                        topk_group=2, routed_scaling_factor=2.5, q_lora_rank=None),
    "deepseek_v2_yarn": dict(MLA, n_routed_experts=8, num_experts_per_tok=2, moe_intermediate_size=32,
                             intermediate_size=48, n_shared_experts=1, first_k_dense_replace=1, q_lora_rank=48,
                             rope_parameters=YARN),
    "deepseek_v3": dict(MLA, n_routed_experts=8, num_experts_per_tok=2, moe_intermediate_size=32, intermediate_size=48,
                        n_shared_experts=1, first_k_dense_replace=1, n_group=4, topk_group=2,
                        routed_scaling_factor=2.5, q_lora_rank=48, norm_topk_prob=True),
    "glm4_moe": dict(n_routed_experts=8, num_experts_per_tok=2, moe_intermediate_size=32, intermediate_size=48,
                     n_shared_experts=1, first_k_dense_replace=1, n_group=2, topk_group=1, routed_scaling_factor=1.8,
                     use_qk_norm=True, attention_bias=True,
                     rope_parameters=dict(rope_type="default", rope_theta=10000.0, partial_rotary_factor=0.5)),
}


def _mt(name):
    return "deepseek_v2" if name == "deepseek_v2_yarn" else name


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """{family: (dir, transformers model)} built once per test module."""
    from transformers import AutoConfig, AutoModelForCausalLM

    out = {}

    def make(name):
        if name in out:
            return out[name]
        torch.manual_seed(0)
        cfg = AutoConfig.for_model(_mt(name), **{**BASE, **FAMILY_CFG[name]})
        cfg._attn_implementation = "eager"
        m = AutoModelForCausalLM.from_config(cfg).to(torch.float32).eval()
        with torch.no_grad():
            for n, p in m.named_parameters():
                if n.endswith("e_score_correction_bias") or "norm" in n:
                    p.add_(torch.randn_like(p) * 0.3)
            for n, b in m.named_buffers():
                if n.endswith("e_score_correction_bias"):
                    b.copy_(torch.randn_like(b) * 0.3)
        d = tmp_path_factory.mktemp(name)
        m.save_pretrained(d, safe_serialization=True)
        out[name] = (str(d), m)
        return out[name]

    return make


def rand_prompts(seed, lens):
    rs = np.random.RandomState(seed)
    return [[int(t) for t in rs.randint(0, V, n)] for n in lens]


@pytest.mark.parametrize("name", list(FAMILY_CFG))
def test_prefill_logits_match(tiny, name):
    path, hf = tiny(name)
    ids = rand_prompts(1, [150])[0]  # 150 tokens x top-2 = 300 pairs: the fused prefill combine (>= 256)
    with torch.no_grad():
        ref = hf(torch.tensor([ids])).logits[0].numpy()
    m = MoeLM(path, dtype=mx.float32)
    bs, T = 16, len(ids)
    pool = m.new_kv_pool(16 * bs)
    bt = np.arange(3, 3 + (T + bs - 1) // bs)[None]
    meta = build_meta(np.array(ids), np.arange(T), np.array([0]), np.array([T]), bt, np.arange(T), bs,
                      m.cfg.sliding_window)
    ours = np.array(m.logits(meta, pool))
    diff = np.abs(ours - ref).max()
    print(f"{name}: max |logit diff| {diff:.3g} (max |logit| {np.abs(ref).max():.3g})")
    assert diff < 1e-4


@pytest.mark.parametrize("name", list(FAMILY_CFG))
def test_engine_greedy_matches_hf(tiny, name):
    path, hf = tiny(name)
    prompts = rand_prompts(2, [5, 37, 70])
    ref = [hf_greedy(hf, p, 20) for p in prompts]
    eng = Engine(model=MoeLM(path, dtype=mx.float32), num_blocks=64, max_batch_tokens=32, max_running=4,
                 enable_prefix_cache=False)
    ours = eng.generate(prompts, 20, ignore_eos=True)
    for i, (o, (r, lg)) in enumerate(zip(ours, ref)):
        div = next((j for j in range(len(r)) if o[j] != r[j]), None)
        if div is not None:
            top2 = np.sort(lg[div])[-2:]
            assert top2[1] - top2[0] < 1e-5, f"{name} seq {i}: diverges at {div}, top-2 gap {top2[1] - top2[0]:.3g}"


@pytest.mark.parametrize("name", ["qwen3_moe", "deepseek_v3"])
def test_bf16_overlap_same_tokens_as_plain(tiny, name):
    """bf16 with the low-bit candidate head, overlap on and off: same tokens (the engine paths, not accuracy)."""
    path, _ = tiny(name)
    prompts = rand_prompts(3, [9, 50, 23])
    m = MoeLM(path, dtype=mx.bfloat16)
    a = Engine(model=m, num_blocks=64, max_batch_tokens=32, max_running=4).generate(prompts, 16, ignore_eos=True)
    b = Engine(model=m, num_blocks=64, max_batch_tokens=32, max_running=4, overlap=True).generate(
        prompts, 16, ignore_eos=True)
    assert a == b


@pytest.mark.parametrize("rs", [None, dict(rope_type="linear", factor=2.0), YARN,
                                dict(rope_type="llama3", factor=8.0, low_freq_factor=1.0, high_freq_factor=4.0,
                                     original_max_position_embeddings=64)])
def test_rope_inv_freq_matches_transformers(rs):
    from transformers import AutoConfig
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    rp = {"rope_theta": 10000.0, **(rs or {"rope_type": "default"})}
    cfg = AutoConfig.for_model("llama", hidden_size=128, num_attention_heads=2, head_dim=64,
                               max_position_embeddings=512, rope_parameters=rp)
    t = rp["rope_type"]
    if t == "default":
        ref = 1.0 / (10000.0 ** (torch.arange(0, 64, 2).float() / 64))
        ref_af = 1.0
    else:
        ref, ref_af = ROPE_INIT_FUNCTIONS[t](cfg, "cpu")
    ours, af = rope_inv_freq(64, 10000.0, rp, 512)
    np.testing.assert_allclose(ours, ref.numpy(), rtol=1e-6)
    assert abs(af - ref_af) < 1e-9


def test_config_rejects_unknown_family():
    with pytest.raises(ValueError):
        MoeLMConfig.from_hf(dict(model_type="llama"))


@pytest.mark.parametrize("name", ["qwen3_moe", "deepseek_v3", "mixtral"])
def test_mapped_and_streamed_experts_same_tokens(tiny, tmp_path, name):
    """placement.repack -> SSD-streamed (3 of 8 slots per layer, so experts are evicted and re-read) and
    memory-mapped experts give the resident model's greedy tokens (dense first layers have no expert file)."""
    from mj_mlx import placement
    from mj_mlx.expert_map import repack_part_major

    path, _ = tiny(name)
    store = placement.repack(path, str(tmp_path / "store"), log=lambda *_: None)
    mapped = str(tmp_path / "mapped")
    repack_part_major(store, mapped)
    prompts = rand_prompts(4, [6, 45, 19])

    def gen(**kw):
        m = MoeLM(path, dtype=mx.float32, **kw)
        return Engine(model=m, num_blocks=64, max_batch_tokens=32, max_running=3,
                      enable_prefix_cache=False).generate(prompts, 14, ignore_eos=True)

    ref = gen()
    assert gen(stream_experts=store, expert_slots=4, prefetch=True) == ref
    assert gen(stream_experts=store, expert_slots=4, prefetch=False) == ref
    assert gen(map_experts=mapped) == ref
    assert MoeLM(path, dtype=mx.float32, stream_experts=store, expert_slots=4).prefetch  # streaming: on by default
    assert not MoeLM(path, dtype=mx.float32).prefetch  # nothing to prefetch without a store


def test_choose_placement_by_memory(tiny):
    from mj_mlx import placement

    path, _ = tiny("qwen3_moe")
    total, experts = placement.checkpoint_bytes(path)
    assert 0 < experts < total
    gb = 1 << 30
    assert placement.choose(path, mem=dict(working_set=16 * gb, ram=24 * gb)) == "resident"
    assert placement.choose(path, reserve=0, mem=dict(working_set=total - 1, ram=total + 4 * gb)) == "mapped"
    assert placement.choose(path, reserve=0, mem=dict(working_set=total - 1, ram=total)) == "stream"


@pytest.mark.parametrize("name", ["qwen3_moe", "deepseek_v3"])
def test_gpu_streamed_experts_same_tokens(tiny, tmp_path, name, monkeypatch):
    """MJ_GPU_STREAM: decode steps route through the C++ Streamer (dense first layers have no slab)."""
    from mj_mlx import expert_stream as ES
    from mj_mlx import placement

    try:
        ES.load_gpu_streamer()
    except ImportError:
        pytest.skip("C++ extension not built")
    path, _ = tiny(name)
    store = placement.repack(path, str(tmp_path / "store"), log=lambda *_: None)
    prompts = rand_prompts(6, [6])

    def gen(**kw):
        m = MoeLM(path, dtype=mx.float32, **kw)
        return Engine(model=m, num_blocks=64, max_batch_tokens=32, max_running=2,
                      enable_prefix_cache=False).generate(prompts, 14, ignore_eos=True)

    ref = gen()
    monkeypatch.setenv("MJ_GPU_STREAM", "1")
    assert gen(stream_experts=store, expert_slots=8) == ref  # the C++ path: slots >= 4 x tokens x top-k
    monkeypatch.setenv("MJ_GPU_STREAM", "0")
    assert gen(stream_experts=store, expert_slots=3) == ref


def test_stream_slots_from_memory(tiny, tmp_path):
    from mj_mlx import placement

    path, _ = tiny("qwen3_moe")
    store = placement.repack(path, str(tmp_path / "s"), log=lambda *_: None)
    total, experts = placement.checkpoint_bytes(path)
    gb = 1 << 30
    assert placement.stream_slots(path, store, reserve=0, mem=dict(working_set=total + gb, ram=0)) == 8  # all
    assert placement.stream_slots(path, store, reserve=total, mem=dict(working_set=total, ram=0)) == 2  # floor
