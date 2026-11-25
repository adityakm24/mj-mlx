"""Write a random gpt-oss checkpoint at real gpt-oss-20b dimensions, in the transformers/HF layout.

The real gpt-oss-20b (24 layers, ~13 GB) does not fit this 16 GB Mac's GPU working set, so speed and energy runs
use these random weights with fewer layers. Per-layer shapes, dtypes and the MXFP4 expert format are exactly the
real model's; routing is close to uniform random (the plan's roofline assumption). Results from it are labelled
"random weights, N of 24 layers".

    .venv/bin/python tools/mlx_make_bench_model.py --layers 8 --out <dir>
"""

from __future__ import annotations

import argparse
import json
import os

import mlx.core as mx

REAL_20B = dict(
    architectures=["GptOssForCausalLM"], model_type="gpt_oss", attention_bias=True, hidden_size=2880,
    intermediate_size=2880, head_dim=64, num_attention_heads=64, num_key_value_heads=8, num_local_experts=32,
    num_experts_per_tok=4, experts_per_token=4, vocab_size=201088, rms_norm_eps=1e-5, sliding_window=128,
    rope_theta=150000.0, swiglu_limit=7.0, max_position_embeddings=131072, initial_context_length=4096,
    rope_scaling=dict(beta_fast=32.0, beta_slow=1.0, factor=32.0, original_max_position_embeddings=4096,
                      rope_type="yarn", truncate=False),
    tie_word_embeddings=False, eos_token_id=200002, pad_token_id=199999, dtype="bfloat16",
    quantization_config=dict(quant_method="mxfp4", modules_to_not_convert=[
        "model.layers.*.self_attn", "model.layers.*.mlp.router", "model.embed_tokens", "lm_head"]),
)


def make_qwen36(a) -> None:
    """Random Qwen3.6-35B-A3B-shaped checkpoint (dims from the real config.json, fetched with the user's OK).

    Layout: transformers' Qwen3_5MoeForConditionalGeneration keys for everything (zero-centered norms, conv1d
    [C, 1, K]), except the routed experts, which are stored in mlx-lm's native 4-bit form
    (`mlp.switch_mlp.{gate,up,down}_proj.{weight,scales,biases}`, affine, group 64, bf16 scales) with
    config["quantization"] = {group_size 64, bits 4}. mlx-lm's own loader then quantizes exactly those modules;
    MilliJoule's loader reads the same tensors. Everything else is bf16."""
    src = json.load(open(a.config))
    t = dict(src.get("text_config", src))
    n = a.layers
    pattern = t["layer_types"][:n]
    assert pattern.count("full_attention") * 3 == pattern.count("linear_attention"), pattern
    t.update(num_hidden_layers=n, layer_types=pattern, mtp_num_hidden_layers=0)
    cfg = {"architectures": ["Qwen3_5MoeForConditionalGeneration"], "model_type": "qwen3_5_moe",
           "text_config": t, "tie_word_embeddings": False, "quantization": {"group_size": 64, "bits": 4},
           "bench_note": f"random weights, Qwen3.6-35B-A3B dims, {n} of {src.get('text_config', src)['num_hidden_layers']} "
                         "layers (3:1 linear:full), 4-bit affine gs64 routed experts, bf16 rest"}
    os.makedirs(a.out, exist_ok=True)
    mx.random.seed(a.seed)
    bf = mx.bfloat16
    H, V, E, I, SI = t["hidden_size"], t["vocab_size"], t["num_experts"], t["moe_intermediate_size"], \
        t["shared_expert_intermediate_size"]
    nq, nkv, D = t["num_attention_heads"], t["num_key_value_heads"], t["head_dim"]
    Hk, Hv, Dk, Dv, K = (t["linear_num_key_heads"], t["linear_num_value_heads"], t["linear_key_head_dim"],
                         t["linear_value_head_dim"], t["linear_conv_kernel_dim"])
    C = 2 * Hk * Dk + Hv * Dv

    def normal(shape, std):
        return (mx.random.normal(shape) * std).astype(bf)

    def q4(shape, std):
        w, sc, bi = mx.quantize(normal(shape, std), 64, 4)
        return w, sc.astype(bf), bi.astype(bf)

    pre = "model.language_model"
    mx.save_safetensors(os.path.join(a.out, "model-embed.safetensors"), {
        f"{pre}.embed_tokens.weight": normal((V, H), 0.02), "lm_head.weight": normal((V, H), 0.02),
        f"{pre}.norm.weight": mx.zeros((H,), bf)})
    for i, lt in enumerate(pattern):
        p = f"{pre}.layers.{i}"
        w = {f"{p}.input_layernorm.weight": mx.zeros((H,), bf), f"{p}.post_attention_layernorm.weight": mx.zeros((H,), bf)}
        if lt == "linear_attention":
            a_ = f"{p}.linear_attn"
            w.update({
                f"{a_}.in_proj_qkv.weight": normal((C, H), 0.02), f"{a_}.in_proj_z.weight": normal((Hv * Dv, H), 0.02),
                f"{a_}.in_proj_b.weight": normal((Hv, H), 0.02), f"{a_}.in_proj_a.weight": normal((Hv, H), 0.02),
                f"{a_}.conv1d.weight": normal((C, 1, K), 0.3), f"{a_}.dt_bias": mx.ones((Hv,), bf),
                f"{a_}.A_log": mx.log(mx.random.uniform(1, 16, (Hv,))).astype(bf),
                f"{a_}.norm.weight": mx.ones((Dv,), bf), f"{a_}.out_proj.weight": normal((H, Hv * Dv), 0.02)})
        else:
            a_ = f"{p}.self_attn"
            w.update({
                f"{a_}.q_proj.weight": normal((nq * D * 2, H), 0.02), f"{a_}.k_proj.weight": normal((nkv * D, H), 0.02),
                f"{a_}.v_proj.weight": normal((nkv * D, H), 0.02), f"{a_}.o_proj.weight": normal((H, nq * D), 0.02),
                f"{a_}.q_norm.weight": mx.zeros((D,), bf), f"{a_}.k_norm.weight": mx.zeros((D,), bf)})
        m = f"{p}.mlp"
        w.update({f"{m}.gate.weight": normal((E, H), 1.0 / H ** 0.5),
                  f"{m}.shared_expert.gate_proj.weight": normal((SI, H), 0.02),
                  f"{m}.shared_expert.up_proj.weight": normal((SI, H), 0.02),
                  f"{m}.shared_expert.down_proj.weight": normal((H, SI), 0.02),
                  f"{m}.shared_expert_gate.weight": normal((1, H), 0.02)})
        for name, shape in (("gate_proj", (E, I, H)), ("up_proj", (E, I, H)), ("down_proj", (E, H, I))):
            qw, qs, qb = q4(shape, 0.02)
            w.update({f"{m}.switch_mlp.{name}.weight": qw, f"{m}.switch_mlp.{name}.scales": qs,
                      f"{m}.switch_mlp.{name}.biases": qb})
        mx.save_safetensors(os.path.join(a.out, f"model-layer{i:02d}.safetensors"), w)
        del w
    with open(os.path.join(a.out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    print("wrote", a.out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["gpt_oss", "qwen36"], default="gpt_oss")
    ap.add_argument("--config", help="qwen36: path to the real Qwen3.6-35B-A3B config.json")
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.arch == "qwen36":
        return make_qwen36(a)
    cfg = dict(REAL_20B, num_hidden_layers=a.layers,
               layer_types=["sliding_attention", "full_attention"] * (a.layers // 2))
    os.makedirs(a.out, exist_ok=True)
    mx.random.seed(a.seed)
    H, I, D = cfg["hidden_size"], cfg["intermediate_size"], cfg["head_dim"]
    nq, nkv, E, V = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["num_local_experts"], cfg["vocab_size"]
    bf = mx.bfloat16

    def normal(shape, std):
        return (mx.random.normal(shape) * std).astype(bf)

    def mxfp4(rows, cols):
        # Random E2M1 nibbles; E8M0 scale 2^-7 gives entries with std ~0.02 (about 1/sqrt(2880)).
        blocks = mx.random.randint(0, 256, (E, rows, cols // 32, 16)).astype(mx.uint8)
        scales = mx.full((E, rows, cols // 32), 120, mx.uint8)
        return blocks, scales

    shard = {"model.embed_tokens.weight": normal((V, H), 0.02), "lm_head.weight": normal((V, H), 0.02),
             "model.norm.weight": mx.ones((H,), bf)}
    mx.save_safetensors(os.path.join(a.out, "model-embed.safetensors"), shard)
    del shard
    for i in range(a.layers):
        p, s = f"model.layers.{i}", f"model.layers.{i}.self_attn"
        w = {
            f"{p}.input_layernorm.weight": mx.ones((H,), bf),
            f"{p}.post_attention_layernorm.weight": mx.ones((H,), bf),
            f"{s}.q_proj.weight": normal((nq * D, H), 0.02), f"{s}.q_proj.bias": mx.zeros((nq * D,), bf),
            f"{s}.k_proj.weight": normal((nkv * D, H), 0.02), f"{s}.k_proj.bias": mx.zeros((nkv * D,), bf),
            f"{s}.v_proj.weight": normal((nkv * D, H), 0.02), f"{s}.v_proj.bias": mx.zeros((nkv * D,), bf),
            f"{s}.o_proj.weight": normal((H, nq * D), 0.02), f"{s}.o_proj.bias": mx.zeros((H,), bf),
            f"{s}.sinks": normal((nq,), 1.0),
            f"{p}.mlp.router.weight": normal((E, H), 1.0 / H**0.5), f"{p}.mlp.router.bias": mx.zeros((E,), bf),
            f"{p}.mlp.experts.gate_up_proj_bias": mx.zeros((E, 2 * I), bf),
            f"{p}.mlp.experts.down_proj_bias": mx.zeros((E, H), bf),
        }
        w[f"{p}.mlp.experts.gate_up_proj_blocks"], w[f"{p}.mlp.experts.gate_up_proj_scales"] = mxfp4(2 * I, H)
        w[f"{p}.mlp.experts.down_proj_blocks"], w[f"{p}.mlp.experts.down_proj_scales"] = mxfp4(H, I)
        mx.save_safetensors(os.path.join(a.out, f"model-layer{i:02d}.safetensors"), w)
        del w
    with open(os.path.join(a.out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
