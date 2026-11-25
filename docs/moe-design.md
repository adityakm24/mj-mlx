# General MoE model for the MLX backend (moe_lm.py)

Goal: one model class that runs any common decoder-only MoE checkpoint (transformers or mlx-community layout) on
the existing engine (Rust scheduler, paged KV, overlap, lean head, expert placement), with the tuned per-model paths
(gpt_oss.py, qwen3_5_moe.py) kept for the models that have them.

## What differs between MoE families (and nothing else)

| piece | variants | families |
| --- | --- | --- |
| attention | GQA with optional bias, per-head q/k RMSNorm, whole-projection q/k RMSNorm, partial rotary, sliding window | Mixtral, Qwen2/3-MoE, OLMoE, GLM-4-MoE |
| | MLA: latent KV (kv_lora_rank) + shared rope key, optional q LoRA | DeepSeek-V2/V3 |
| rope | default, linear, llama3, yarn (with attention factor / mscale), interleaved or half-split pairs | all |
| router | softmax over all experts, top-k, optional renormalisation | Mixtral (always renorm), Qwen2/3, OLMoE |
| | softmax, greedy or group-limited top-k, x routed_scaling_factor | DeepSeek-V2 |
| | sigmoid + e_score_correction_bias for the choice, group top-k (sum of top-2 per group), renorm, x scaling | DeepSeek-V3, GLM-4-MoE |
| shared experts | none / plain (n_shared x moe_intermediate) / sigmoid-gated (shared_expert_gate) | DeepSeek, GLM / Qwen2-MoE |
| dense layers | first_k_dense_replace, decoder_sparse_step, mlp_only_layers | DeepSeek, GLM / Qwen |
| weights | bf16/f32 per-expert or fused [E, 2I, H]; mlx affine quantized (switch_mlp); MXFP4 (gpt-oss only) | |

`MoeLMConfig.from_hf(config.json)` turns each family's config into these fields. Checkpoint keys are normalised
once (`block_sparse_moe` -> `mlp`, `w1/w3/w2` -> `gate/up/down_proj`, `switch_mlp`, `language_model.` prefixes).

## Design rules

- Every piece is a small function with its own test against the matching transformers module on a tiny random
  checkpoint (tests/mlx/test_moe_lm.py builds them locally with transformers; no downloads). End to end: prefill
  logits and greedy tokens equal transformers float32.
- No new engine code paths: the model implements what the runner already calls (hidden / final_hidden /
  greedy_from / full_logits / new_kv_pool / kv_bytes_per_token / parameters, and GptOss's head utilities by
  inheritance), so overlap, the low-bit candidate head, lean mode and the host gate apply unchanged.
- MLA uses the existing paged decode kernel as single-head attention over the cache rows [latent | rope key]
  (D = kv_lora_rank + qk_rope_head_dim, K and V the same pool): q_nope is absorbed through kv_b's key half, the
  output's first kv_lora_rank dims go through kv_b's value half. ~1.2 KB per token per layer for DeepSeek-V2-Lite
  instead of ~10 KB expanded.
- Expert placement (resident / memory-mapped + mlock / SSD streamed) is chosen by size against this Mac's memory;
  the mapped and streamed stores are the existing expert_map / expert_stream (affine experts).

## Order of work

1. Config + key normalisation + GQA attention + softmax routers + resident experts: Mixtral, Qwen2-MoE, Qwen3-MoE,
   OLMoE (tiny models vs transformers).
2. MLA + DeepSeek routers + shared experts + dense first layers: DeepSeek-V2, DeepSeek-V3, GLM-4-MoE.
3. Placement by memory; real checkpoints (Qwen3-30B-A3B, DeepSeek-V2-Lite, gpt-oss-120b streamed).
4. Suite rows against llama.cpp / mlx-lm / Ollama.
