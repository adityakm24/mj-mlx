# mj-mlx internals

_Moved here from `src/mj_mlx/README.md` (2026-11-21). Paths are relative to
`src/mj_mlx/`; tools and tests are at the repository root._

The MoE runner for the gpt-oss architecture on MLX, driven by the Rust scheduler (`millijoule.Scheduler`) through
the flat `StepBatch`. It is the dev and test backend of the plan. **Its energy numbers are Mac whole-chip GPU-rail
estimates and are never put next to an NVML / B200 number.**

## Files

| File | What |
| --- | --- |
| `gpt_oss.py` | gpt-oss model (adapted from mlx-lm, MIT): flat varlen step, paged KV, YaRN with `truncate`, fused q/k/v, MXFP4 experts, precise mode, low-bit lm_head candidates + float32 rescore |
| `paged_attention.py` | Metal kernel: paged decode attention (sinks, sliding window, GQA group per threadgroup) |
| `moe_kernels.py` | Metal kernel: grouped MXFP4 expert GEMV with fused clamped SwiGLU (correct; off by default, slower than `gather_qmm` on M4) |
| `qwen3_5_moe.py` | Qwen3.5/3.6 MoE text model (written from transformers, Apache-2.0): Gated DeltaNet layers with per-request recurrent-state slots, gated full attention on paged KV, softmax router + shared expert |
| `gated_delta.py` | Metal kernel: gated delta rule over the whole state pool (reads/writes it in place of a gather + scatter), state layout `[slots, Hv, Dv, Dk]`, 4 value columns per simdgroup sharing q/k, token loop inside the kernel (decode and prefill); runs at plain-copy speed on M4 |
| `expert_stream.py` | SSD expert streaming: `ExpertStore` (per-layer repacked files, reader pool, `F_NOCACHE`), `StreamingExperts` (resident slots, zero-copy `preadv` into MLX arrays, LRU, next-layer prefetch), `SafetensorsRows` (memory-mapped embedding / lm_head rows) |
| `expert_repack.py` | Repack of gpt-oss (MXFP4) and Qwen3.5 / 3.6 routed experts into per-layer `ExpertStore` files; `placement.repack` calls it for those families and repacks the general MoE families itself |
| `sampler.py` | On-device sampling: temperature / top-k / top-p, Gumbel-max with a hash of (seed, position, vocab id) |
| `runner.py` | `MlxRunner` (owns the KV pool) and `Engine` (scheduler + runner) |
| `energy.py` | IOReport energy counter (no sudo); only the "GPU Energy" rail updates without root |

Tools: `tools/mlx_make_bench_model.py` (random checkpoint at real gpt-oss-20b dims), `tools/mlx_bench.py`
(closed-loop throughput + energy, mlx-lm or ours), `tools/mlx_accuracy.py` (teacher-forced KL / top-1 vs a float32
reference), `tools/mlx_report.py` (ABAB table). Tests: `tests/mlx/` (needs the tiny checkpoint).

```bash
.venv/bin/python -m pytest tests/mlx -q
.venv/bin/python tools/mlx_make_bench_model.py --layers 8 --out <dir>
.venv/bin/python tools/mlx_bench.py --engine mlx-lm --model <dir> --levels 1,8,32,64 --out results/a.json
.venv/bin/python tools/mlx_bench.py --engine mj     --model <dir> --levels 1,8,32,64 --out results/b.json
.venv/bin/python tools/mlx_report.py results/a.json -- results/b.json
```

## Design

- **One step = one flat batch.** All items (prefill chunks and decodes of any length) are concatenated. Token-wise
  ops (projections, norms, router, experts) run once over all tokens; only attention is grouped. Single-token items
  go to the paged decode kernel in one call; longer items (prefill chunks, resumed multi-token decodes) each use
  a gather of their KV + `mx.fast.scaled_dot_product_attention` with a causal/window mask. For head sizes MLX's
  fused kernel does not take (it takes 64, 80, 96 and 128 with equal K / V sizes), MLX builds the whole score
  matrix, so `prefill_attention.sdpa_prefill` splits the queries into blocks whose scores fit a budget (256 MB by
  default); with the causal mask, the query block [a, b) of a chunk attends the first K - L + b keys.
- **KV pool** per layer `[num_blocks * block_size, n_kv, head_dim]`, written at `slot_mapping`, read through
  `block_tables`. The scheduler owns allocation; the runner never frees anything (`release` is a no-op for
  gpt-oss).
- **Logits only for sampled rows.** Greedy on device, one host sync per step.
- **Precise mode (default for bf16):** residual stream, RMSNorm statistics and the router in float32. Weights and
  KV stay bf16 / MXFP4. MoE routing is where bf16 rounding hurts most (a flipped expert changes the whole FFN).
- **lm_head candidates (greedy rows only):** a low-bit copy of lm_head picks the top-16 per row (exact two-stage
  top-k: chunk maxima, then top-k inside the best chunks; a full-vocab argpartition cost 26 ms at 64 rows), and
  those 16 are recomputed in float32 from the bf16 rows. Width by rows, measured on M4: 4-bit at <= 4 rows, 8-bit
  up to 48, bf16 above (quantized matmul turns compute-bound). Greedy output equals the full bf16 head unless
  the true top-1 is outside the low-bit top-16 (0 of 4096 rows for 8-bit and 4-bit, `tools/mlx_lm_head_miss.py`).
  Subject to the real-weight accuracy gate (PM decision).
- **Sampling:** rows with temperature > 0 use the full bf16 head (float32 out), then temperature, top-k and top-p,
  then Gumbel-max with noise that is a pure function of (seed, position, vocab id). Preempted-and-recomputed
  requests sample the same tokens again (tested: tight pool equals roomy pool with sampling on).
- **Memory:** weights and KV are wired (as mlx-lm does), MLX's buffer cache is capped at 1 GB and cleared after
  prefill steps. Without that, a 16 GB Mac swaps and throughput halves.

## Correctness (float32, tiny-GptOssForCausalLM vs transformers 5.23)

- Prefill logits: max |dlogit| 5.8e-8 (dense and MXFP4 expert paths).
- Batch-1 greedy: token-identical on 4 prompts (5-300 tokens, prefill chunked at 64, across the 128 window).
- Batched: 16 sequences with a shared 64-token prefix, tight pool (9 preemptions, 73 evictions) equals roomy pool,
  and both equal transformers' batch-1 greedy with 0 near-tie excuses (tolerance agreed with the user).
- transformers' reference uses its own `convert_moe_packed_tensors` for MXFP4 (`tests/mlx/hf_ref.py`), because its
  MXFP4 loader requires `accelerate`.

## Qwen3.6 (hybrid linear attention)

`load_model(path)` picks `Qwen35Moe` for `model_type` `qwen3_5_moe`. The engine then runs the scheduler with
`enable_prefix_cache=False` and `num_state_slots` sized from memory like the KV pool (`MlxRunner.state_capacity`:
at most half of the free budget after weights, never more than `max_running`; pass `num_state_slots=` to
override). Admission stops when no slot is free, so this also caps the running set: real Qwen3.6 state is roughly
31 MB per request in bf16 and 63 MB in float32 (plan estimate), so 512 requests would need 16-32 GB before any KV.
Each linear-attention layer keeps a conv state
`[slots, K-1, conv_dim]` (model dtype) and a delta-rule state `[slots, Hv, Dv, Dk]` (float32) per request slot; an
item with `start == 0` zeroes its slot; preemption is a full recompute. `Qwen35Moe(state_dtype=mx.bfloat16)` stores
the delta-rule state in bf16 between steps (math stays float32): half the state bytes, which on CUDA moves the
J/token floor (plan estimate 0.009 vs 0.015 J at 512 requests). Its accuracy cost on the tiny model is in
`results/RESULTS.md` 2b; it is a lower bound, real weights are needed for the gate. Full-attention layers (1 in 4) use the
same paged KV and paged decode kernel as gpt-oss (no sinks; head_dim 256 supported).

Correctness on `trl-internal-testing/tiny-Qwen3_5MoeForConditionalGeneration-3.6`, float32 vs transformers 5.23
(`tests/test_qwen3_5_moe.py`, 7 tests): prefill logits max |dlogit| < 3e-7 whole, chunked (64/50/36) and
token-by-token then chunk; batch-1 greedy token-identical; 12 sequences batched, tight pool (15 preemptions)
equals roomy pool and both equal transformers with 0 near-tie excuses; sampling reproducible under preemption;
kernel path equals the per-token reference path. Speed (tiny model, not meaningful for energy): the kernel and
vectorized conv take 4 x (1000-token prompt + 32 tokens) from 2.33 s to 0.31 s.

Experts: bf16 (transformers layout, fused or per-expert keys) or mlx-lm-native quantized `mlp.switch_mlp.*`
(`weight`/`scales`/`biases`, e.g. 4-bit affine group 64), in which case gate and up are fused row-wise into one
`gather_qmm`. `tools/mlx_make_bench_model.py --arch qwen36 --config <real config.json>` writes a random checkpoint at
the real Qwen3.6-35B-A3B dims in that mixed layout, which mlx-lm's own loader also reads (used for the benchmark).
Construction evaluates and frees one layer at a time, so peak load memory equals the final weights.

Not done for Qwen3.6: real weights (Qwen3.6-35B-A3B is ~70 GB in bf16; needs its own download OK and does not
fit this Mac; the NVFP4 checkpoint would be the Mac-sized option for a layer-streamed check), quantized experts,
radix-cache support (state snapshots at block boundaries), a chunked (WY / UT) delta-rule kernel for long
prefills on many heads, the MTP head.

## Models bigger than memory (SSD expert streaming)

```bash
.venv/bin/python tools/mlx_repack_experts.py --model <gpt-oss dir> --out <experts dir>       # once
.venv/bin/python tools/mlx_stream_run.py --experts <experts dir> --slots 8,16,24 --out results/x.json
```

`load_model(path, stream_experts=<dir>, expert_slots=S, lean=True)` works for every supported MoE
(gpt-oss, Qwen3.6): S experts per layer resident, the rest read from the SSD on demand, with the next layers' experts
predicted and read early (`prefetch=True`, the default since 2026-11-22; `prefetch=False` or the tools' `--no-prefetch`
turns it off); `lean` keeps the embedding and
bf16 lm_head on disk (greedy only). One core (`ExpertStore` / `ExpertSlab` / `run_moe`); a model adds only its
per-pair expert math. Real gpt-oss-20b on this 16 GB M4: 4.7 / 7.2 / 10.4 tok/s at S = 8 / 16 / 24 (4.2 / 6.7 /
9.2 GB resident); Qwen3.6 at real dims (8 layers) in 1.3-2.7 GB. See `results/RESULTS.md` 1d and 1d.1.

## Real weights (openai/gpt-oss-20b)

`tools/mlx_real_layer_check.py` (one real layer vs transformers float32: rel. RMS 5.6e-7 ours, 2.8e-3 mlx-lm) and
`tools/mlx_real_eval.py` (all 24 layers streamed one at a time; bf16 vs float32 reference: greedy top-1 97.2% ours vs
95.4% mlx-lm, KL 0.0157 vs 0.0221; one 1296-position teacher-forced sample, not the plan's lm-eval gate). Results: `results/RESULTS.md` section 2a. The full model does not fit
this Mac's GPU working set for serving.

## Known gaps / next

1. Qwen3.6: see its section above for what is not done yet.
2. Attention compute in float32 would cut bf16 KL by another ~23% but needs a mixed-precision GEMM (MLX promotes
   bf16 weights to fp32, doubling bytes).
3. A grouped MXFP4 GEMV kernel that beats `gather_qmm` at B >= 32 (MoE is the largest share of a B=64 step).
4. Done: overlap scheduling (plan step N+1 during step N) is in the scheduler and the engine (`overlap=True`).
