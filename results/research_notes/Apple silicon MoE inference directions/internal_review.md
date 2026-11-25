# Internal review of mj-mlx: where time, energy and memory go, what was rejected, what is unexplained (state 2026-11-21)

Scope and method: local files only, read with the Read tool; nothing was run. Hardware in every number: one M4 MacBook
Pro (10-core GPU), 16 GB, MLX 0.32.4. "RESULTS 1xx" is the section label inside `results/RESULTS.md`. The file is not
in label order: sections run 1, 1b, 1c, 1d ... 1u, then 1zm down to 1za (newest first), then 1z, 1y, 1x, 1w, 1v, 1t, 1e,
then 2-6. All of RESULTS.md (2,905 lines) was read. Source links are relative to this notes file.

Sources: [RESULTS](../../RESULTS.md), [LAUNCH_TODO](../../LAUNCH_TODO.md), [BENCH_PROTOCOL](../../BENCH_PROTOCOL.md),
[README](../../../README.md), [CHANGELOG](../../../CHANGELOG.md), [internals](../../../docs/internals.md),
[moe-design](../../../docs/moe-design.md), [models](../../../docs/models.md), [usage](../../../docs/usage.md),
[energy](../../../docs/energy.md), [neural-engine](../../../docs/neural-engine.md),
[benchmarks](../../../docs/benchmarks.md), module docstrings under [src/mj_mlx](../../../src/mj_mlx/) and
[scheduler.rs](../../../crates/mjx-core/src/scheduler.rs).

## 1. Plan: what the phases in LAUNCH_TODO.md contain, what is done, what is open

### Takeaway
Phase 0 (defensible claims) is mostly done or half done; Phases 1-5 (server / CLI, more families, speed gaps, hardware
and soak, Neural Engine) are almost entirely open. The speed gaps the project itself lists as open are Phase 3 (3.1-3.6)
plus 0.7c (DeepSeek-V2-Lite decode at long context).

### Cited Findings
- Order agreed with the user on 2026-11-19: Phase 0 first, then Phase 1, then the rest — [LAUNCH_TODO](../../LAUNCH_TODO.md)
- **Phase 0 "claims we can defend", done [x]:** 0.1 accuracy suite (WikiText-2 perplexity, KL vs bf16, on ours / mlx-lm /
  llama.cpp, for OLMoE, DeepSeek-V2-Lite, Qwen3-30B, gpt-oss); 0.1b gpt-oss vs a float32 reference (reference 180.29;
  ours exact +0.5%, 8-bit -1.8%, llama.cpp -4.1%); 0.7b long prompts without a flag (prefill attention in 64 MB score
  blocks); 0.13 harness fixes; 0.14 "where we lose, say so" — [LAUNCH_TODO](../../LAUNCH_TODO.md)
- **Phase 0 partly done [~], with the "Left:" item:** 0.2 MMLU / GSM8K subset (tool written; left: download datasets,
  run); 0.3 quality-matched comparisons (OLMoE done; DeepSeek-V2-Lite, Qwen3-30B next; llama.cpp KL vs the f32 reference
  not measured); 0.4 Ollama / LM Studio / vllm-mlx in the suite; 0.5 quiet re-runs (left: mj-a8 at the 1000 MB wired
  limit re-run; the paging rule, since every engine swaps out 0.6-2.8 GB during gpt-oss timed reps); 0.6 llama.cpp
  energy per token via llama-server (written, not run); 0.7 long context (left: 32k, the exact arm at 16k, Ollama / LM
  Studio past 8k, vllm-mlx at 8k+ invalid); 0.8 one-command reproduction (left: add v13 rows as claims); 0.9 rivals at
  their best settings (left: versions in meta.json, a speed sweep of each rival's main knob); 0.10 memory as a claim
  (left: a run); 0.11 cold-start load time and TTFT (tool written, not run) — [LAUNCH_TODO](../../LAUNCH_TODO.md)
- **Phase 0 open [ ]:** 0.7c DeepSeek-V2-Lite decode falls with context (52.24 at 2k, 33.82 at 8k, 22.17 at 16k): "MLA
  decode at long context"; 0.12 real-chat quality check vs transformers; 0.15 long-context teacher-forced accuracy
  (lesson cited from Anemll-Forge's notes: an FP8 attention build matched on short KL traces and failed at 64K), needed
  before 32k and for every change on the long-context path; 0.16 whole-machine energy above idle next to the GPU rail,
  with an idle baseline and a temperature gate ("a hot machine cost 11% at 16K" in their notes); 0.17 record git commit,
  dirty flag and MJ_* / MLX_* environment per worker row — [LAUNCH_TODO](../../LAUNCH_TODO.md)
- **Phase 1 "usable":** done: own repo (1.0a), package (1.0). Open: 1.0b one `placement=` for every family (GptOss /
  Qwen35Moe reject it today); 1.1 OpenAI-compatible server; 1.1b tool calls, JSON output, prefix cache across requests;
  1.2 sampling in every mode (gpt-oss lean is greedy only); 1.3 CLI; 1.4 wheel; 1.5 expert-repack disk use; 1.7 clear
  errors; 1.9 launch material. Partly: 1.6 docs, 1.8 licenses — [LAUNCH_TODO](../../LAUNCH_TODO.md)
- **Phase 2 "breadth" (all open):** 2.1 fallback tier for dense / unknown models; 2.2 Gemma-4, GLM-4.5-Air,
  gpt-oss-120b, Qwen3-Next; 2.3 mlx 3/6/8-bit, MXFP4 in the general model, GGUF loading — [LAUNCH_TODO](../../LAUNCH_TODO.md)
- **Phase 3 "speed gaps" (all open):** 3.1 DeepSeek-V2-Lite short-context decode (11% behind mlx-lm): MLA decode kernel;
  3.2 OLMoE prefill (94-96% of the best rival), gpt-oss pp512 (94% of llama.cpp); 3.3 KV-cache quantization for long
  context; 3.4 speculative decoding beyond gpt-oss and chat-style speculative decoding in the suite; 3.5 small-model
  decode (OLMoE ~3% to vllm-mlx / mlx-lm); 3.6 batch throughput against rivals' servers — [LAUNCH_TODO](../../LAUNCH_TODO.md)
- **Phase 4 "hardware and reliability" (all open):** 4.1 results beyond one M4 16 GB; 4.2 soak tests and the unexplained
  silent exit on the exact gpt-oss gated path (candidate cause: macOS memorystatus kill at 75% of wired memory; a
  65-minute soak at the 1000 MB wired limit ran clean; left: reproduce the 1zc exit on the old setting, soak the other
  models); 4.3 C++ streamer errors as Python exceptions; 4.4 CI; 4.5 sustained throughput (30-minute runs, thermal
  throttling, battery vs AC) — [LAUNCH_TODO](../../LAUNCH_TODO.md)
- **Phase 5 "Neural Engine" (planned, all open):** 5.1 a one-day spike (a dense FFN block through Core ML at prefill
  chunk sizes, call overhead, memory of the compiled copy, whole-machine energy, GPU + ANE overlap); 5.2 a `device` per
  component in placement, dense parts on the ANE during prefill; 5.3 a dense draft model on the ANE judged on energy
  per accepted token — [LAUNCH_TODO](../../LAUNCH_TODO.md)
- First full reproduction (2026-11-21): every claim ran; the one sentence that did not hold was "OLMoE tied with
  vllm-mlx at d2048" (ours 102.42 vs 99.53, intervals apart, i.e. better than claimed); Qwen3.6-35B multiples moved to
  5.16x (d0), 4.26x (d2048), 10.37x (pp512), 20.00x (pp2048) — [RESULTS 1zl](../../RESULTS.md)

### Inferences
- The project's own priority list for speed work is: MLA decode (0.7c, 3.1), small-model prefill and gpt-oss pp512
  (3.2), KV quantization (3.3), speculative decoding on more models (3.4), batch throughput (3.6). Literature ideas
  that land on those items match the plan directly.
- Any idea touching the long-context path is gated by 0.15 (teacher-forced long-context accuracy), which does not exist
  yet.

### Gaps
- LAUNCH_TODO does not list an item for streamed-model decode (Qwen3.6-35B / Qwen3-30B SSD limits) although RESULTS
  shows large remaining idle time there (question 3). Whether that is deliberate was not stated.

## 2. Architecture: the main mechanisms

### Takeaway
A Rust continuous-batching scheduler hands the MLX runner one flat token batch per step (prefill chunks and decodes
mixed) over a paged KV pool; custom Metal kernels cover decode attention (paged, split-K, fused KV write), the Gated
DeltaNet recurrence, small-row quantized matmuls and MLA decode; experts live in one of three placements (resident,
memory-mapped page cache, SSD-streamed slots); decode is pipelined against the host by "overlap" plus a Metal
shared-event gate; speculative decoding uses the model's own MTP head (Qwen) or an EAGLE-3 head (gpt-oss).

### Cited Findings
- **Scheduler:** continuous batching; decodes and chunked prefills share one token budget per step; one item per
  request per step; the plan is flat arrays (`cu_seqlens`, `block_tables` padded with -1, per-item sampling params,
  `state_slots` for recurrent state); with `EngineConfig::overlap`, `schedule()` may run once ahead (two steps in
  flight); preempted requests are recomputed from their start (mostly a prefix-cache hit); radix prefix cache — [scheduler.rs](../../../crates/mjx-core/src/scheduler.rs)
- **Runner / step = flat batch:** the runner owns one `[num_blocks * block_size, n_kv, head_dim]` K and V array per
  layer, addressed only through block tables; token-wise ops (projections, norms, router, experts) run once over all
  tokens, only attention is grouped; single-token items go to the paged decode kernel in one call, longer items use a
  KV gather plus `mx.fast.scaled_dot_product_attention`; logits only for sampled rows; one host sync per step — [runner.py](../../../src/mj_mlx/runner.py), [internals](../../../docs/internals.md)
- **Engine defaults:** `overlap` False by default (benchmarks use True), `max_running` 256, `max_batch_tokens` 4096,
  `max_seq_len` 32768, 16-token KV blocks, prefix cache on (off for Qwen3.5 / 3.6), `cache_limit_bytes` 1 GiB, wired
  limit = MLX's recommended working set, 1000 MB with mapped experts — [usage](../../../docs/usage.md)
- **Overlap scheduling:** step N+1 is planned and queued on the GPU before step N's tokens reach the host; decode
  inputs are read from step N's lazy tokens on device; 1.14-1.15x the plain loop at batch 1 on Qwen3.5-9B — [RESULTS 1g](../../RESULTS.md)
- **Deferred lean step + host gate (gpt-oss lean):** step N+1 is built on a page-aligned host embedding buffer
  (`mx.from_dlpack`, zero copy) while N runs; the greedy choice among 16 candidates is made on the host in float64;
  with the gate, N+1 is submitted behind a Metal shared-event wait (C++ `HostGate`) and the host signals after writing
  the embedding row. Deferred: +9.4% decode; gate: +5.6% / +7.5% (8-bit attention) and +11.5% / +17.6% (exact) — [RESULTS 1z, 1zc](../../RESULTS.md), [runner.py](../../../src/mj_mlx/runner.py)
- **MLX eval throttle:** MLX's eval waits while about 10 command buffers are in flight; at `MLX_MAX_MB_PER_BUFFER=150`
  a gpt-oss step is ~50 buffers (each 400 MB mapped expert layer forces one), so a gated step must be one command
  buffer (the extension switches MLX's per-buffer limits per step through private fields) — [RESULTS 1zc](../../RESULTS.md)
- **Paged decode attention kernel:** one query token per sequence, K/V read through the block table with no gather,
  one threadgroup per (sequence, kv head) serving all GQA query heads, sinks and sliding window, online softmax in
  float32 — [paged_attention.py](../../../src/mj_mlx/paged_attention.py)
- **Split-K (flash-decoding):** when the longest row spans more than 256 positions, one threadgroup per (sequence, kv
  head, 256-position partition) then a reduce kernel; a row's numbers depend only on its own length. Qwen3.5-9B batch 1:
  +1.9% at 2048, +11.8% at 8192 — [RESULTS 1x](../../RESULTS.md)
- **KV write inside the decode kernel:** replaces a whole-pool scatter copy on decode-only steps; Qwen3.5-9B +1.4% /
  +2.3% / +5.3% at depth 0 / 2048 / 8192; gpt-oss lean within noise — [RESULTS 1za](../../RESULTS.md)
- **MLA (DeepSeek-V2/V3):** the cache row is `[latent c | rope key]` (one head), ~1.2 KB per token per layer vs ~10 KB
  expanded (mlx-lm caches expanded K/V); decode is absorbed-form attention, by default SDPA over a gather of the cached
  rows (`kp[slots]`, a copy of the layer's whole context per token, unfused at DK = 576); prefill expands K/V from the
  latents and attends per head. A dedicated kernel (`mla_decode.py`: tiles of 16 rows in threadgroup memory, 2 heads
  per simdgroup, split-K with 256-position partitions, fused cache-row write) exists behind `MJ_MLA_DECODE=kernel`;
  default is still "sdpa" — [moe_lm.py](../../../src/mj_mlx/moe_lm.py), [mla_decode.py](../../../src/mj_mlx/mla_decode.py), [RESULTS 1zd, 1zm](../../RESULTS.md)
- **Prefill attention in query blocks:** MLX 0.32.4's fused attention kernel takes head sizes 64, 80, 96, 128 with equal
  K and V sizes; other sizes (DeepSeek-V2 MLA 192 / 128, Qwen3.5 / 3.6 256) build the whole score matrix (~1 GB per layer
  at 2048 x 16k), so `sdpa_prefill` splits queries into blocks under a score budget (`MJ_ATTN_SCORES_MB`, default 64) — [prefill_attention.py](../../../src/mj_mlx/prefill_attention.py), [RESULTS 1zj, 1zm](../../RESULTS.md)
- **Expert placement:** `resident` (checkpoint + 2 GiB fits the GPU working set), `mapped` (checkpoint + 3 GiB fits RAM;
  part-major, 16 KB-aligned files, zero-copy `mx.from_dlpack` views of the page cache, optional mlock), `stream`
  (per-layer files, S slots per layer, LRU, next-layer prefetch, `F_NOCACHE`); quantized checkpoints run in their
  scales' dtype — [placement.py](../../../src/mj_mlx/placement.py), [models](../../../docs/models.md)
- **Mapped experts:** shared mapping (`"r+"`) so pages stay file cache; `MJ_MLOCK=1` pins them; MLX's wired limit must
  stay small (1000 MB default) — [expert_map.py](../../../src/mj_mlx/expert_map.py), [RESULTS 1p, 1y, 1zk](../../RESULTS.md)
- **Streamed experts:** `ExpertStore` / `ExpertSlab` / `run_moe`; misses are `preadv`'d straight into the slots' MLX
  memory (8 reader threads); the next layer's router applied to this layer's MoE input predicts what to prefetch
  (depth 2 by default); speculative all-resident dispatch queues the expert computation before the routing reaches the
  host and discards it on a miss; a C++ GPU-driven streamer (`MJ_GPU_STREAM=1`) reads the routing from unified memory
  on a Metal shared-event signal so Python never waits mid-step; a GPU keep-alive spin kernel covers SSD waits — [expert_stream.py](../../../src/mj_mlx/expert_stream.py), [RESULTS 1d, 1d.3, 1s, 1u](../../RESULTS.md)
- **Lean trunk (gpt-oss):** embedding table and bf16 lm_head stay on disk as memory-mapped rows; a low-bit candidate
  head picks top-16 and the 16 rows are rescored exactly (greedy only) — [RESULTS 1d, 1zb](../../RESULTS.md)
- **lm_head candidates:** exact two-stage top-k (chunk maxima, then top-k inside the best chunks; a full-vocab
  argpartition cost 26 ms at 64 rows); width by rows: 4-bit at <= 4 rows, 8-bit up to 48, bf16 above; lean gpt-oss now
  uses a 3-bit head (0 misses of 2048 at top-16; 2.72 ms vs 3.36 ms for 4-bit; +1.0-3.7% decode) — [internals](../../../docs/internals.md), [RESULTS 1zb](../../RESULTS.md)
- **Precise mode (default for bf16):** residual stream, RMSNorm statistics, router logits and top-k softmax, the
  attention softmax and the candidate rescore in float32; weights, KV, q/k/v and expert activations stay bf16 — [RESULTS 6](../../RESULTS.md)
- **gpt-oss model:** fused q/k/v projection, fused gate/up MXFP4 projection, YaRN honouring `truncate: false`; sorted
  `gather_qmm` for experts; fused prefill MoE glue (two Metal kernels for steps of >= 256 pairs) — [gpt_oss.py](../../../src/mj_mlx/gpt_oss.py), [RESULTS 1zc](../../RESULTS.md)
- **Lossless packed bf16 (12.25 bits per weight):** low byte unchanged, 4 bits of sign + exponent offset, one base byte
  per group of 32, escapes in CSR; decode matvec for 1-4 rows reads it directly; 0.768 of bf16's bytes on gpt-oss
  attention; +6% to +18% decode — [bf16_pack.py](../../../src/mj_mlx/bf16_pack.py), [RESULTS 1m, 1o](../../RESULTS.md)
- **Qwen3.5 / 3.6:** Gated DeltaNet layers (depthwise causal conv, gated delta rule on a per-request float32 state
  `[Hv, Dv, Dk]`) with per-request state slots, 1 in 4 layers gated full attention on paged KV, softmax router plus a
  gated shared expert; the gated-delta Metal kernel reads and writes the state pool itself at plain-copy speed;
  `SlotStates` keeps states per slot — [qwen3_5_moe.py](../../../src/mj_mlx/qwen3_5_moe.py), [gated_delta.py](../../../src/mj_mlx/gated_delta.py), [RESULTS 1c, 1f](../../RESULTS.md)
- **Small-row kernels:** `qmm_small` (8- and 16-row tiles built from MLX's own steel pieces, used for 5-16 rows) took
  an 8-request Qwen3.5-9B decode step from 123.8 to 73.1 ms; `qmv_rows` gives 2-4 row steps the exact float ops of
  MLX's one-row kernel so speculative verify equals plain decode — [RESULTS 1i, 1j](../../RESULTS.md), [qmv_rows.py](../../../src/mj_mlx/qmv_rows.py)
- **Grouped MXFP4 GEMV kernel:** correct, off by default, slower than `gather_qmm` on M4 — [moe_kernels.py](../../../src/mj_mlx/moe_kernels.py), [internals](../../../docs/internals.md)
- **Sampler:** on-device temperature / top-k / top-p with Gumbel noise that is a pure function of (seed, position,
  vocab id), so preempted requests resample the same tokens — [sampler.py](../../../src/mj_mlx/sampler.py)
- **Import-time settings:** `MLX_MAX_MB_PER_BUFFER=150` (mapped-expert models decode ~1.6x slower at MLX's default) and
  `MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS=200` (first prefill after idle ~10% slower at MLX's default) — [usage](../../../docs/usage.md)
- **Neural Engine:** not used. Stated obstacles: no MLX path; fixed shapes vs routing (about 0.2 ms overhead per ANE
  call, reported by Anemll-Forge); a second copy of the weights; no extra memory bandwidth. Possible uses: prefill of
  dense fixed-shape parts, energy, a dense draft model — [neural-engine](../../../docs/neural-engine.md)

### Inferences
- The engine already has a place for most per-step ideas: per-row exact kernels, a pipelined host, and three placements.
  What it lacks structurally (stated in the files) is multi-token decode items and rollback inside the Rust scheduler
  (speculative decoding lives outside the serving engine), a prefix cache for recurrent-state models, and a uniform
  `placement=` for the tuned classes.

### Gaps
- Only module docstrings were read, not the code; mechanisms are described as the docstrings and RESULTS state them.
- The qwen3_5_moe.py docstring says "The MTP head is not used" while spec_decode.py uses it; the docstring looks stale.
- docs/internals.md "Known gaps" item 4 (overlap needs a scheduler change) is stale: overlap exists (RESULTS 1g).

## 3. Step traces and profiles per headline model

### Takeaway
In-memory decode is at the memory-bandwidth floor (Qwen3.5-9B: 98% of a 45.3 ms floor, GPU 99.9% busy), so plain
single-request decode can only improve through fewer bytes per token. gpt-oss-20b decode is ~26-28 ms of GPU-bound
parts (experts ~half) plus host / throttle gaps that the gate mostly removed; its prefill is expert-matmul-bound and,
on 16 GB, page-cache-bound. Streamed models are limited by SSD misses and one GPU-CPU-GPU round trip per layer: the GPU
is busy only 50-66% of a streamed Qwen3.6-35B decode step. OLMoE, DeepSeek-V2-Lite and Qwen3-30B have no per-part
breakdown in the files.

### Cited Findings

**Machine constants measured here**
- A chained 4-bit matvec reads at 94-99 GB/s on this M4; BENCH_PROTOCOL discloses the machine as 120 GB/s — [RESULTS 1i](../../RESULTS.md), [BENCH_PROTOCOL](../../BENCH_PROTOCOL.md)
- Compute ceilings: dense bf16 matmul 3.64-3.68 TFLOPS; dense MXFP4 and 8-bit `quantized_matmul` 3.31-3.38; sorted
  `gather_qmm` 2.81 (512 tokens) to 3.06 (2048 tokens) TFLOPS — [RESULTS 1zc](../../RESULTS.md)
- SSD ceiling 3.17-3.3 GB/s; one 13.2 MB gpt-oss expert = 4.2 ms whatever the chunking — [RESULTS 1d.3](../../RESULTS.md)
- The GPU drops its clocks within milliseconds of idling: the same 14 MB 4-bit matvec takes 366 µs with no idle before
  it, 418 µs after 0.5 ms, 1094 µs after 2 ms, 1543 µs after 5 ms — [RESULTS 1u](../../RESULTS.md)
- A real GPU-to-CPU-to-GPU Metal shared-event round trip costs about 95-105 µs with a spinning CPU and about 160 µs
  blocking; a satisfied wait is free; CPU writes become visible to the GPU at command-buffer boundaries only — [RESULTS 1v](../../RESULTS.md)
- `mx.quantized_matmul` (4-bit gs64, [12288 x 4096]) by rows: 1 row 281 µs, 2 rows 274, 4 rows 339, 8 rows 664, 16 and
  32 rows 955, 64 rows 1890 (per-row matvec up to a few rows, a 32-row tile beyond) — [RESULTS 1i](../../RESULTS.md)
- Isolated per-op timings mislead on MLX (a sync per call ~0.3-0.5 ms): one layer alone showed ~31 GB/s for experts at
  B=1 while the whole step ran at ~91 GB/s effective — [RESULTS 3](../../RESULTS.md)

**Qwen3.5-9B (dense, 4-bit, in memory)**
- Decode step by op group (async split, mean of 3 steps): MLP up/gate + activation 19.7 ms (1.81 GB, 92 GB/s); in_proj
  10.0 ms (0.87 GB); MLP down 9.7 ms (0.91 GB, 94 GB/s); lm_head + sampling 5.6 ms (0.57 GB); out_proj 3.8 ms; DeltaNet
  conv + recurrence 1.5 ms; full attention 0.55 ms; norms and small ops 0.55 ms — [RESULTS 1t](../../RESULTS.md)
- GPU busy 99.9% with overlap on (47.7-47.9 ms untraced); 92.3% with overlap off (about 4 ms host gap per step,
  54.6-57.7 ms) — [RESULTS 1t](../../RESULTS.md)
- Floor: the step reads 4.47 GB of weights, so 45.3 ms/token; the step runs at 46.3 ms, 98% of the floor. "Faster than
  this at one request needs fewer bytes per token (speculative decoding); it is not available from any engine's
  kernels" — [RESULTS 1i](../../RESULTS.md)
- Graph size: 2,088 ops per step vs mlx-lm's 1,121; fusing with shapeless `mx.compile` left 1,465 — [RESULTS 1i](../../RESULTS.md)
- Depth cost: decode costs about 1.07 ms/token more at depth 2048 than at depth 0, against mlx-lm's 0.68; KV writes
  (whole-pool copies) were the measured part and were moved into the kernel — [RESULTS 1x, 1za](../../RESULTS.md)
- Prefill by part, 2048 tokens (instrumented, ours vs mlx-lm): MLP 5.95-6.28 vs 6.48-6.58 s, DeltaNet layers 2.37-2.48
  vs 2.57-2.65 s (gated-delta kernel 0.23 vs 0.19 s), full attention 0.72-0.76 vs 0.89-0.94 s, rest 0.21 vs 1.48-1.53 s;
  syncing cost mlx-lm 2.1 s and ours 0.1-0.3 s, "so in plain runs its kernels overlap more than ours"; both are bound by
  the 4-bit matmuls (~3.3 TFLOPS) — [RESULTS 1zc](../../RESULTS.md)
- Batched: 8 requests 123.8 -> 73.1 ms per step with `qmm_small`; 1.67x mlx-lm at 8 requests, 1.49x at 16, 1.18x at 32 — [RESULTS 1i](../../RESULTS.md)

**gpt-oss-20b (13.7 GB; 10.1 GB of MXFP4 experts; 24 layers, 32 experts top-4; mapped experts)**
- Decode by part, all 24 layers at decode shapes, GPU-bound, exact (packed) attention: qkv 5.57, o 4.41, router 0.22,
  experts 13.91, sdpa 0.17, head 3.79 ms; sum 28.07 ms. 8-bit attention: sum 23.81 ms. The whole step ran 36-37 ms in
  that session, so 9-12 ms were outside the parts — [RESULTS 1y](../../RESULTS.md)
- Re-timed later in the "slow state": the same parts 25.77 ms per token, so the kernels were not slower — [RESULTS 1zc](../../RESULTS.md)
- Metal trace of one decode step: 24 back-to-back per-layer command buffers (~26 ms), the 4-bit head + top-16
  (~3.3-3.7 ms), an idle gap while the host reads the candidates and 16 rescore rows, the GPU rescore (~20-35 µs), a
  second idle gap while the host reads the token, its embedding row and builds the next step. Idle in the GPU span:
  median 5.7 ms (serial row reads), 2.7 ms (parallel reads), 2.4 ms (plus fast synch) — [RESULTS 1z](../../RESULTS.md)
- Host timings per step before the gate: wait for the candidates 11.2 ms, rescore 0.4, build 1.0, "arm" 17 ms; the
  17 ms was MLX's eval throttle waiting on the GPU, not CPU encoding; with the gate, encoding a step took ~3 ms — [RESULTS 1zc](../../RESULTS.md)
- Command-buffer splitting: attention alone 9.6-10.0 ms, MoE alone 12.9-13.2, head 3.5-3.8, but all together
  48.9-50.8 ms at MLX's default buffer limit; 29.4-30.5 ms with `MLX_MAX_MB_PER_BUFFER` raised (each ~400 MB mapped
  expert array forced its own command buffer) — [RESULTS 1p](../../RESULTS.md)
- Page-ins decide decode under memory pressure: 302 MB of page-ins per 128 tokens -> 45.5 ms/token, 796 MB -> 52.3,
  2422 MB -> 59.2 — [RESULTS 1y](../../RESULTS.md)
- Multi-token verify step: 1 / 2 / 3 / 4 tokens 32.3 / 43.9 / 56.2 / 70.3 ms (each extra token routes to ~4 more
  experts per layer); a draft step ~2.7 ms — [RESULTS 1p](../../RESULTS.md)
- Prefill by part (8-bit attention, 2048 tokens in 512-token chunks, each part synced; instrumented total 6.1 s vs
  4.93 s plain): experts 4.31-4.35 s, attention projections 1.01-1.03 s, attention core 0.28 s, router 0.04-0.05 s,
  other 0.43 s — [RESULTS 1zc](../../RESULTS.md)
- Experts, one layer, real mapped weights: at 512 tokens 38.78 ms, of which the two sorted `gather_qmm` calls 36.24 ms
  and glue 2.54 ms; at 2048 tokens 144.0 ms with glue 10.7 ms. Fused glue: 38.16 -> 35.63 ms and 146.18 -> 136.50 ms
  (-6.6%); suite +5.0% pp512, +6.2% pp2048 — [RESULTS 1zc](../../RESULTS.md)
- Steady prefill ~420 tok/s back to back, so the pp512 gap is per-token, not a fixed per-request cost — [RESULTS 1zc](../../RESULTS.md)
- Prefill chunk: pp2048 409.0-415.0 tok/s at chunk 512 vs 224.9-401.6 at 2048 and 323.5-399.7 at 1024; MLX peak 13.09
  vs 14.08 / 14.26 GB — [RESULTS 1za](../../RESULTS.md)
- Prefill on 16 GB is memory-limited: 0.78-0.94 GB paged in per 512-token prefill (a prompt touches nearly every
  expert); with mlock pp512 went from 52-103 to 326-390 tok/s — [RESULTS 1m, 1y](../../RESULTS.md)
- Streamed gpt-oss (24 slots), earlier state: 82.9 ms/token, of which 2.25 ms/layer host waiting for routing and
  1.36 ms/layer host work with the GPU idle; a no-sync replay ran 56.7 ms/token. Of the remaining ~58 ms: large matmuls
  at 74-93 GB/s ~32 ms; candidate head 4.4 ms; SSD 3.3 expert reads/token (44 MB), ~9.7 ms on the critical path; an
  empty decode step takes 8.3 ms at 24 slots vs 4.3 ms at 4 slots — [RESULTS 1d.3](../../RESULTS.md)
- In-memory bench model: MoE is ~70% of a B=64 step and "at the limit of MLX's built-in ops; beating it needs a better
  grouped MXFP4 GEMM kernel" — [RESULTS 5](../../RESULTS.md)

**Qwen3.6-35B-A3B (20.4 GB, 40 layers, 256 experts top-8, hidden 2048; SSD-streamed, 96 slots = 8.2 GB resident)**
- Host-side split of a 57.6 ms decode step (Python path): waiting for the per-layer routing sync 26.1 ms (GPU busy);
  waiting on SSD reads 14.2 ms (GPU idle); Python in `run_moe` 11.4 ms (idle); Python outside it 5.8 ms (idle) — [RESULTS 1u](../../RESULTS.md)
- GPU busy only 50-66% of the decode window; expert hit rate 95.4%; 83% of prefetched experts used; each layer is about
  4 command buffers and ~0.9 ms of GPU work; between layers the GPU waits 1-5 ms; 448 of 505 ms of idle time is in gaps
  longer than 0.5 ms — [RESULTS 1s](../../RESULTS.md)
- One miss layer as recorded: speculative 8-expert pass 0.36 ms (discarded), routing readback at 1.82 ms, 3 resident
  experts computed at 2.78 ms, GPU idles ~4 ms for the SSD reads of the other 5, which then compute in 0.44 ms; the
  layer's other work ~1.5 ms — [RESULTS 1t](../../RESULTS.md)
- C++ GPU-driven streamer: GPU busy 57% (about 32 ms busy, 24 ms idle per step); 12.3 SSD misses per token, about 14 ms
  of IO-thread read wait per token; 9.1 misses per token at 128 slots; with keep-alive, 63.5% busy and about 5.8 ms per
  step idle in ~36 gaps of 100-300 µs (the per-layer event round trip, near its ~145 µs floor; "at most about 2 ms per
  step is left there") — [RESULTS 1u, 1v](../../RESULTS.md)
- Keep-alive: +16% (17.63 / 18.04 off vs 20.78 / 20.75 on) because kernels after an SSD wait run at full clocks — [RESULTS 1u](../../RESULTS.md)
- Routing prediction (layer li+d's router on layer li's MoE input): d = 1 80-83%, d = 2 74-77%, d = 3 70-73% of routed
  experts in decode; of 1386 discarded speculative passes, 760 (55%) had the prediction right but the read not landed — [RESULTS 1s](../../RESULTS.md)
- Cache policy, offline on 8,192 tokens: miss rate per expert access LRU 11.1% (96 slots) / 6.8% (128); LFU with decay
  10.3-10.7% / 6.4-6.5%; Belady optimum 5.2% / 3.0%. "The 2x down to the optimum comes only from knowing future
  routing" — [RESULTS 1v](../../RESULTS.md)
- No-sync replay floor: 49.6 ms/token vs ~80-86 ms then; SSD bytes per decode step 0.116 GB at 96 slots, 0.312 GB at
  32 slots — [RESULTS 1d.3](../../RESULTS.md)
- Slots: 96 -> 128 slots is 11-12% faster (hit rate 96.0% -> 97.4%) but 128 slots is not safe on 16 GB (3 of 4 suite
  blocks failed with Metal "Insufficient Memory" in pp2048) — [RESULTS 1s, 1w](../../RESULTS.md)
- Best streamed decode: 96 slots 20.5-20.8 tok/s, 128 slots 23.5-23.9; suite v4: tg128@d0 24.06 +- 5.18 (median 21.34,
  range 19.76-32.20), tg128@d2048 21.65; "streamed decode depends on the prompt (expert reuse): one seed gave 32.20
  tok/s, most 20-23" — [RESULTS 1u, 1zb](../../RESULTS.md)
- Prefill: a 512-token random prompt moves 11.6 GB from the SSD with cold slots (5.8 GB warm) at 1.7-2.2 GB/s effective
  against the 3.2 GB/s ceiling; filling the next layer's slots during this layer's compute: cold 4.85 / 4.77 s vs
  5.22 / 5.23 s — [RESULTS 1s](../../RESULTS.md)
- MTP loop on the streamed 35B: the IO thread waits on SSD reads 4.0 s of each 6.4 s, 128-token run; about 17 demand
  misses per token; "SSD-latency-bound, not host-bound" — [RESULTS 1v](../../RESULTS.md)
- In-memory 8-layer bench model (random weights), B=1 decode step 11.7 ms: experts 21%, recurrent state 7%, rest of
  DeltaNet 36%, full attention 13%, lm_head 26%; extrapolated to 40 layers (not a run): experts 26%, state 9%, rest of
  DeltaNet 46%, attention 16%, head 7%. At B=64 the recurrent state is ~21% and its cost is bandwidth (the update runs
  at plain-copy speed; bf16 storage halves it) — [RESULTS 1c](../../RESULTS.md)

**Qwen3-30B-A3B (17.2 GB, streamed, 64 slots)**
- Suite v9: tg128@d0 27.71 +- 4.73, tg128@d2048 26.24 +- 4.01, pp512 188.13, pp2048 291.75 vs llama.cpp (`-ncmoe 48`)
  2.71 / 3.57 / 15.57 / 13.35; llama.cpp's best pp512 with partial offload 26.81, so the pp512 lead is ~7x — [RESULTS 1zd](../../RESULTS.md)

**DeepSeek-V2-Lite (4-bit, resident, MLA)**
- Decode 61.00 (d0), 52.24 (d2048), 33.82 (d8192), 22.17 (d16384) tok/s; prefill pp512 588, pp2048 594.54, pp8192
  481.39, pp16384 368.06 (64 MB score blocks) — [RESULTS 1zh, 1zi, 1zm](../../RESULTS.md)
- Only step-level facts given: graph build 1.43 ms of a 17.9 ms synced step at d0; SDPA at D = 576 has no fused kernel;
  decode gathers every cached row per token — [RESULTS 1zd](../../RESULTS.md), [mla_decode.py](../../../src/mj_mlx/mla_decode.py)

**OLMoE-1B-7B (4-bit, resident)**
- Decode 137.45 (d0) and 99.79 (d2048) tok/s; prefill ~1200 (pp512) and ~1260 (pp2048); GPU energy at d0 67 mJ/token,
  the same as mlx-lm and vllm-mlx — [RESULTS 1zh, 1zg](../../RESULTS.md)

**Mixtral-8x7B 4-bit (26.3 GB, streamed, 3 of 8 slots):** 0.59-0.60 tok/s including prefill; SSD-bound by construction
(each token reads two ~100 MB experts in each of 32 layers) — [RESULTS 1zd](../../RESULTS.md)

### Inferences
- Share by resource, as far as the files support it: (a) in-memory decode at batch 1 is memory-bandwidth-bound (weight
  bytes per token); (b) prefill is compute-bound (2.8-3.7 TFLOPS matmuls) except gpt-oss on 16 GB, where page-ins
  dominate without mlock; (c) streamed decode is I/O-and-sync-bound: roughly 40-50% of a Qwen3.6-35B step is GPU idle,
  split between SSD misses (~12-14 ms) and per-layer round trips (~6 ms), plus clock-drop slowdown of the kernels that
  follow a wait; (d) host / Python time is no longer on the critical path for in-memory models with overlap, and is
  ~17 ms per step on the Python streamed path (removed by the C++ streamer).
- For streamed models the largest stated lever is knowing future routing earlier (Belady is 2x better than LRU; 55% of
  discarded speculative passes were right but late), i.e. better or earlier prediction and fewer unique experts per
  token, not eviction policy.
- Decode after ~96% hit rate still pays one round trip per layer because Metal has no conditional GPU wait; a design
  that removes the per-layer host decision for all-hit layers would need a different mechanism than those tried.

### Gaps
- No per-part decode or prefill breakdown exists for OLMoE, DeepSeek-V2-Lite or Qwen3-30B (only totals).
- No bytes-read-per-token or hit-rate figures for Qwen3-30B in RESULTS.
- The Qwen3.6-35B percentages for a full in-memory step are an extrapolation from 8 random-weight layers, as the file
  says; the real 35B has only streamed traces, one taken during a 72 GB download ("preliminary").
- No CPU-side energy or time split for the reader threads; the GPU rail does not see them (question 6).

## 4. Where it loses or only ties, and which explanations were measured

### Takeaway
Losses are small on in-memory models (1-8% on short-context decode and small-model prefill) and are mostly not
explained to a single cause; the two structural ones are DeepSeek-V2-Lite decode at long context (a per-token gather of
the whole latent cache; a kernel exists but its speed is unmeasured) and gpt-oss-20b on 16 GB (memory / paging and
exact-weight bytes).

### Cited Findings
- **README "Where it is behind":** short-context decode (DeepSeek-V2-Lite 61.00 vs vllm-mlx 66.02; OLMoE 137.45 vs
  139.29); prefill on small models a few percent behind mlx-lm and llama.cpp; gpt-oss pp512 (LM Studio 1.04-1.05x ahead
  of both arms); DeepSeek-V2-Lite decode falling with context; paging on 16 GB — [README](../../../README.md)
- **OLMoE:** d0: vllm-mlx 1.01x ahead (CIs apart), tie with mlx-lm; d2048: tie with vllm-mlx (later reproduction: ours
  1.03x ahead); prefill: llama.cpp leads pp512 (ours 0.95x), mlx-lm leads pp2048 (ours 0.95x; 1343.35 vs 1260.10 in
  v11). No cause given. Our own run-to-run spread at d0 (132.23-138.39 across runs) was itself flagged as a finding — [RESULTS 1zh, 1zg, 1zl](../../RESULTS.md)
- **DeepSeek-V2-Lite short context:** vllm-mlx 1.08x ahead at d0; mlx-lm's flagged reps 67.41 (1.11x ours; 58.40 vs
  65.67 in v8). Measured: graph build is 1.43 ms of a 17.9 ms synced step; precise mode and `MLX_MAX_MB_PER_BUFFER` 1000
  made no difference (61.1-61.7 in all three arms); "SDPA at D = 576 has no fused kernel". Planned fix: MLA decode
  kernel (checklist 3.1) — [RESULTS 1zd, 1zh](../../RESULTS.md)
- **DeepSeek-V2-Lite decode vs context:** 52.24 (2k), 33.82 (8k), 22.17 (16k). Probe: SDPA over gathered latent rows
  52.5-52.7 / 33.8-34.0 / 20.1-22.6 tok/s at d2048 / d8192 / d16384; the GQA paged kernel 20.6-21.6 / 3.8 / 1.7 (it
  holds all 16 heads' 576-wide queries and outputs per thread). The stated mechanism is a copy of the layer's whole
  context per token plus SDPA's unfused path. The dedicated kernel is in, more accurate against float32 (error
  0.006-0.008 vs 0.015-0.030 for bf16 SDPA), "Its speed: next" — [RESULTS 1zm](../../RESULTS.md), [mla_decode.py](../../../src/mj_mlx/mla_decode.py)
- **DeepSeek-V2-Lite prefill:** llama.cpp leads pp512 (ours 0.97x); ours leads pp2048 (1.03x vllm-mlx, 1.07x
  llama.cpp); absorbed attention at D = 576 does ~1.8x the attention math of the expanded form, so prefill expands K/V — [RESULTS 1zh, 1zd](../../RESULTS.md)
- **Qwen3.5-9B plain decode:** ties mlx-lm at d0 (21.53 vs 21.56; 21.89 vs 21.86), 1.9% behind at d2048 in v5 (21.09
  vs 21.49, CIs apart), 2% slower in the direct harness (48.13 / 47.68 vs 47.22 / 46.67 ms); both within 3-6% of the
  45.3 ms floor. Measured causes fixed on the way: host work while the GPU waits (overlap), 2,088 vs 1,121 graph ops
  (compile), whole-pool KV copies (fused KV write) — [RESULTS 1i, 1zb, 1x, 1za](../../RESULTS.md)
- **Qwen3.5-9B prefill:** 1-4% behind mlx-lm (pp512 217.47 vs 227.01, pp2048 220.16 vs 228.70 in v5). Measured: matmuls
  tie; our gated-delta kernel is faster in one probe (8.5 vs 9.1 ms) and +40 ms slower over a run in another; host step
  building 6 ms; linear-attention layers 95.5 vs 91.7 ms and full attention 88.1 vs 86.1 ms account for ~107 ms of the
  342 ms gap; bf16 residual would recover ~1%; "The rest is spread over many small ops with no single cause found. Not
  fixed"; mlx-lm's kernels overlap more in plain runs; ours falls rep to rep (226.9 / 218.4 / 213.2) — [RESULTS 1j, 1zc, 1zb](../../RESULTS.md)
- **gpt-oss-20b pp512:** llama.cpp 428.84 vs ours 406-408 (1.06x) in v14; LM Studio 421.96 vs 403-406 in v13; tied at
  pp2048. Measured: experts are 4.3 of 6.1 s of an instrumented 2048 prefill at 2.81-3.06 TFLOPS against a dense
  ceiling of 3.3-3.7; glue fusion gave +5-6%; a larger chunk does not help; the gap is per-token — [RESULTS 1zk, 1zi, 1zc, 1zf](../../RESULTS.md)
- **gpt-oss-20b exact arm:** tied with llama.cpp in decode (36.91 vs 35.83, intervals overlap), while the 8-bit arm
  leads by 1.08-1.10x; the exact arm reads more attention bytes (bf16 or 12.25-bit packed vs Q8_0) and holds more
  memory; llama.cpp's attention is lossy Q8_0 (perplexity -4.1% from the f32 reference vs ours +0.5%) — [RESULTS 1zi, 1zk, 1ze](../../RESULTS.md)
- **gpt-oss-20b, nothing to draft:** on a random prompt llama-server was 1.08-1.19x faster than ours plain (earlier
  state, before the gate and 3-bit head) while ours used 26-27% less GPU energy — [RESULTS 1q](../../RESULTS.md)
- **Random / undraftable text with speculation:** adaptive EAGLE-3 with no accepted drafts was 3-5 ms/token slower
  than plain before `SHADOW_EVERY`; after, within noise to 1.13x slower — [RESULTS 1q](../../RESULTS.md)
- **Batch of 8+ with MTP:** plain decoding is faster (108-110 vs at best 92.8 tok/s); adaptive now reaches parity
  (107.9 vs 108.9) — [RESULTS 1n, 1r](../../RESULTS.md)
- **Weight-format quality:** llama.cpp's Q4_K_M is the better 4-bit file (KL 0.081 vs 0.159 for mlx-community 4-bit g64
  on OLMoE, each against its own reference; DeepSeek-V2-Lite perplexity 8.89 vs 9.08); at matched quality (5-bit g64,
  perplexity 11.593 vs 12.000) our OLMoE decode drops to 114.6-115.1 tok/s (still ~12% ahead of llama.cpp's 102.41)
  and prefill is ~84-91% of llama.cpp — [RESULTS 1ze](../../RESULTS.md)
- **Robustness losses:** vllm-mlx comparison invalid at 8k+; our mj-x arm noisy at 16k (323.03 [234.97-411.09] /
  25.57 [16.82-34.33]); the exact arm's v14 block paged 13 GB in and out — [RESULTS 1zi, 1zk](../../RESULTS.md)

### Inferences
- Measured explanations: host / GPU serialisation (fixed by overlap, defer, gate), command-buffer splitting (fixed),
  whole-pool KV copies (fixed), paging on 16 GB (mitigated by mlock and the wired limit), MLA gather per token
  (mechanism identified, fix unmeasured).
- Still guesses or unexplained: the small-model prefill gaps (OLMoE 5%, Qwen3.5-9B 1-4%; "many small ops", mlx-lm's
  better kernel overlap), the short-context decode gaps to vllm-mlx (OLMoE 1%, DeepSeek-V2-Lite 8%), and why
  `gather_qmm` prefill sits at 2.8-3.1 TFLOPS against 3.3-3.7 for dense matmuls.
- vllm-mlx runs mlx-lm 0.31.3 underneath; its d0 lead over both ours and mlx-lm 0.32.2 on DeepSeek-V2-Lite and OLMoE
  was not investigated in the files.

### Gaps
- No trace of an OLMoE or DeepSeek-V2-Lite step exists, so the d0 gaps cannot be attributed.
- The MLA decode kernel's speed at 2k / 8k / 16k is not in RESULTS.
- Rivals were not swept for their best knobs (checklist 0.9), so "ties" could move either way.

## 5. Memory: the 16 GB story

### Takeaway
Metal's wired ceiling on this Mac is 12.71 GB; crossing it produces kernel "required > max" lines and, at ~75-77% of
system wired memory, a macOS memorystatus kill. Three fixes worked: a 1000 MB MLX wired limit for mapped experts,
mlock for expert pages, and 64 MB score blocks for unfused prefill attention. What remains: every engine pages during
gpt-oss runs on 16 GB, the silent-exit link is unproven, and several page-in sources are untraced.

### Cited Findings
- maxWired is 12.71 GB; `vm.user_wire_limit` 13.06 GB — [RESULTS 1zi, 1y](../../RESULTS.md)
- **The kill:** kernel log `memorystatus: python3.11 [16381] has 75% of wired memory (11409 MiB of 15208 MiB total
  wired), killing`; before it IOGPUFamily logged `required > max` (descriptorBytes about 13.45 GB vs 12.71 GB). The
  exact arm had MLX peak 13.9 GB (experts mlocked plus packed bf16 attention) — [RESULTS 1zh](../../RESULTS.md)
- **Cause is MLX's wired limit, not mlock or the chunk:** "required > max" lines per probe: 992-1010 at 3000 MB (mlock
  on or off, chunk 512 or 256), 4 at 1500 MB, 0 at 1000 MB; speed the same — [RESULTS 1zi](../../RESULTS.md)
- 8-bit arm at 1000 MB: of 3,074 over-limit lines, 3,072 fell in the 3000 MB blocks, none in the 1000 MB blocks; speeds
  tie; default moved to 1000 MB — [RESULTS 1zk](../../RESULTS.md)
- Soak at 1000 MB: 65 minutes, 450 timed reps, no kill, three over-limit lines, peak RSS 10.6 GB, decode within 2% — [RESULTS 1zi](../../RESULTS.md)
- **Paging during gpt-oss runs:** swap-outs between first and last timed rep 0.6-0.7 GB for mj-x, 0-2.8 GB mj-a8,
  0.7-1.8 GB Ollama, 0.7-1.1 GB LM Studio; "no gpt-oss number here is paging-free" — [RESULTS 1zi](../../RESULTS.md)
- One mj-x block paged 13.0 GB in and 13.8 GB out (it followed a 3000 MB mj-a8 block) without tripping the net-swap
  rule; its reps fell to 98-210 tok/s prefill and 25.65 decode; proposed rule: flag a block whose swap-ins exceed 1 GB — [RESULTS 1zk](../../RESULTS.md)
- **mmap pitfalls:** a private copy-on-write mapping became anonymous memory once Metal used it (swap +2 GB per
  process, 41.6-43.4 vs 33.3-35.2 ms/token); a read-only mapping made Metal thrash (no progress in 10 min); shared
  `"r+"` is the default — [RESULTS 1p](../../RESULTS.md)
- **Metal wires a whole no-copy buffer while a command buffer uses it:** one GPU gather from a 1.16 GB mapped table
  raised wired memory by 1.21 GB (so `map_head` adds ~2.3 GB wired and thrashed at 0.07 tok/s) — [RESULTS 1y](../../RESULTS.md)
- **mlock:** pins 10.15 GB; system wired right after load 14.08-14.11 GB with mlock vs 6.56-6.97 GB without; added
  ~1.1-1.5 GB of swap; prefill pp512 3.8-7.4x faster, decode unchanged — [RESULTS 1y](../../RESULTS.md)
- **Wired-limit extremes:** at MLX's recommended working set the first command buffer fails with Metal "Insufficient
  Memory" with mapped experts; 5500 / 8000 MB gave no gain; 11000 / 12000 MB run out of Metal memory — [RESULTS 1j, 1p, 1y](../../RESULTS.md)
- **Models bigger than RAM cannot be mapped:** 17 GB of mapped Qwen3.6 experts fails at any setting, as does llama.cpp
  with every layer on the GPU — [RESULTS 1j, 1s](../../RESULTS.md)
- **`MLX_MAX_MB_PER_BUFFER`:** 4000 kept temporaries alive and a 2048-token Qwen3.5-9B prefill ran out of Metal memory
  at 400, 1000 and 4000 (works at 40-250); default 150 — [RESULTS 1p](../../RESULTS.md)
- **The 64 MB score-block finding:** unfused SDPA builds a ~1 GB score matrix per layer at 2048 x 16k and Metal keeps
  the buffers of in-flight command buffers. At 256 MB blocks DeepSeek-V2-Lite 16k logged 274-376 over-limit lines and
  macOS killed a worker ("77% of wired memory"); at 64 MB: 0-2 lines, no kill, pp2048 +3.1%, pp8192 +4.6%. MLX peak
  12.28 GB (64 MB) vs 12.51 GB (256 MB). An MLX wired limit of 10,000 MB also gave 1 line (364 tok/s) — [RESULTS 1zj, 1zm](../../RESULTS.md)
- **Capacity vs rivals:** llama.cpp ran out of Metal memory on gpt-oss 16k at defaults and ubatch 256 (fits at 128), and
  on DeepSeek-V2-Lite at 8k and 16k at every ubatch tried (f16 KV; q8_0 KV not tried); mlx-lm cannot load gpt-oss-20b
  (same `required > max`, 12.73-13.10 GB needed) — [RESULTS 1zi, 1zh](../../RESULTS.md)
- **MLX buffer cache:** without a 1 GB cap and a clear after prefill steps, freed prefill temporaries stayed wired, the
  process grew to 9.3 GB (5.7 GB weights) and level-32 throughput fell from ~190 to 78-118 tok/s; a 4 GiB KV pool on
  top of 5.7 GB of weights made a decode step 15x slower from paging — [RESULTS 3](../../RESULTS.md)
- **Resident set costs time by itself:** an empty decode step takes 8.3 ms at 24 gpt-oss slots vs 4.3 ms at 4 slots
  with the machine 8-9 GB into swap — [RESULTS 1d.3](../../RESULTS.md)
- **Recurrent state memory:** real Qwen3.6 state is roughly 31 MB per request in bf16 and 63 MB in float32 (plan
  estimate); MTP rollback at 3 drafts x 8 requests keeps state after every row (~1.5 GB at 32 rows) and ran out of GPU
  memory once — [internals](../../../docs/internals.md), [RESULTS 1n](../../RESULTS.md)
- **KV bytes:** MLA latent cache ~1.2 KB per token per layer vs ~10 KB expanded; mlx-lm was left out at 16k on
  DeepSeek because expanded K/V is about 4.5 GB more — [RESULTS 1zd, 1zi](../../RESULTS.md)

### Inferences
- Unresolved, per the files: (1) the paging rule for gpt-oss tables; (2) proof that the 3000 MB wired limit caused the
  1zc silent exit; (3) the 456-595 MB of page-ins per pp2048 rep that remain with mlock; (4) 32k context as a capacity
  check; (5) KV-cache quantization (planned 3.3, nothing measured); (6) slots chosen from free memory rather than a
  fixed 96; (7) long-context accuracy gate (0.15) for the blocked attention and the MLA kernel.
- The memory ceiling, not compute, sets several speed results: gpt-oss prefill, the exact arm's noise, 128-slot
  streaming (+11-12% but unsafe), 128 slots + MTP (out of memory).

### Gaps
- Single-op readings of MLX's peak-memory counter were withdrawn as not repeatable (1zm); only model-level measures
  (over-limit lines, kills, the run's MLX peak) are evidence.
- No measurement of memory compression or of what macOS evicts first; no result on a Mac with more RAM (1m says the
  gpt-oss prefill gap "does not appear on a Mac with room for the whole model in the page cache (not tested)").

## 6. Energy: what is measured and what it shows

### Takeaway
Every published energy number is the IOReport "GPU Energy" rail (an SoC model estimate, idle floor included, no sudo).
No whole-machine (SMC) or wall number exists in RESULTS as of this reading, although one doc says the benchmarks
report it. On this GPU, energy follows arithmetic (rows computed), not bytes read, so speculative verify rows and
unpacking cost energy even when time is flat.

### Cited Findings
- Published mJ/token = GPU rail over a test window / its tokens; CPU, DRAM and ANE rails are recorded but occasionally
  return impossible deltas and are not used; display, SSD and fans are never included; not comparable with NVML or wall
  meters — [energy](../../../docs/energy.md), [RESULTS header](../../RESULTS.md)
- Without sudo only the GPU rail updates on this M4 (CPU, DRAM and the rest read 0); `energy.py` contains IOReport
  counters and GPU residency only — [energy.py](../../../src/mj_mlx/energy.py), [RESULTS 3](../../RESULTS.md)
- **Whole-machine energy is planned, not present in RESULTS:** checklist 0.16 is open ("Whole-machine energy above idle
  next to the GPU rail ... The GPU rail misses CPU, DRAM and SSD, which matter for streamed experts"); BENCH_PROTOCOL
  says of a wall meter "We do not have one yet" — [LAUNCH_TODO](../../LAUNCH_TODO.md), [BENCH_PROTOCOL](../../BENCH_PROTOCOL.md); contradicted by [neural-engine](../../../docs/neural-engine.md), which says "The benchmarks report energy above idle for the whole machine (SMC), not only the GPU rail ... (RESULTS 1zm onward)"
- **Streaming rule:** the GPU rail does not see SSD reads or the CPU's preadv reader threads, "which at batch 1 are a
  real share of the joules", so a GPU-rail J/token would undercount streaming; streaming energy is left out or must be
  whole-system — [RESULTS 1d](../../RESULTS.md)
- **Meter pitfalls:** a stale first IOReport sample after a long gap put a whole prefill's energy in the next window
  (fixed by sampling twice); windows that read 0 J; SoC CPU / DRAM rails returned bogus deltas in 7 of 207 rows — [RESULTS 1zb, 1w](../../RESULTS.md)
- **Machine performance states:** idle GPU 0.9-1.3 W in a slow state, 0.00 W in a fast one; both engines move together;
  run-to-run spread on a shared laptop 20-45% — [RESULTS 1, 3](../../RESULTS.md)
- **Energy follows arithmetic:** the main model's forward for 1 / 2 / 3 / 4 tokens takes 40.3 / 40.9 / 42.1 / 53.3 ms but
  417 / 713 / 933 / 1050 mJ; MLX's own 4-bit matmul on one matrix: 1 / 2 / 3 rows 0.56 / 0.54 / 0.54 ms but 5.1 / 8.4 /
  11.8 mJ. "One draft pays for itself in energy only when its acceptance is above about 0.58" — [RESULTS 1l](../../RESULTS.md)
- Packed bf16 matvec: faster per call (216 vs 297 µs) but 17-19% more GPU energy per call at one row — [RESULTS 1m](../../RESULTS.md)
- **Headline energy results (GPU rail):** gpt-oss-20b decode 262-299 mJ/token vs LM Studio 366-400 and Ollama 363-388
  (16-28% less); Qwen3.5-9B batched 0.54-0.83x mlx-lm; MTP chat 339-406 mJ/token vs plain 455-506; OLMoE d0 67 mJ for
  ours, mlx-lm and vllm-mlx, LM Studio 61; gpt-oss prefill 46 / 49 mJ/token and decode 198 / 271 (v5) — [RESULTS 1zi, 1zk, 1i, 1l, 1zg, 1zb](../../RESULTS.md)
- Qwen3.5-9B plain decode energy is level with mlx-lm (481 vs 480 at d0, 468 vs 456 at d2048) — [RESULTS 1zb](../../RESULTS.md)
- Where an equal-speed energy gain came from on Qwen3.6 bench: lower GPU power for the same tokens (11.7 W vs 14.9 W at
  level 8); "We do not have a per-kernel energy split on the Mac" — [RESULTS 1c](../../RESULTS.md)
- **Trade-offs:** rejected drafts cost verify work (random prompt, fixed k = 2: 1.2-1.5x plain's energy); overlap keeps
  the GPU powered between steps (+13-16% in 30 s windows, ~0 in 2 min windows, "Not resolved"); speculative expert
  dispatch raised gpt-oss GPU-rail J/token (0.48 -> 0.59) but lowered Qwen3.6's (0.243 -> 0.220), cause not
  established; the keep-alive spin's energy cost is "Not measured yet" — [RESULTS 1l, 1o, 1h, 1d.3, 1u](../../RESULTS.md)
- Energy-aware draft policy: per-depth acceptance, cost table in joules or seconds per iteration, `edp` (energy x
  delay) objective — [RESULTS 1l, 1n](../../RESULTS.md)
- llama.cpp has no energy window from llama-bench (its window includes model load); a `llama-server` engine was
  written to fix that, not run — [RESULTS 1w](../../RESULTS.md), [LAUNCH_TODO](../../LAUNCH_TODO.md)

### Inferences
- Because the rail tracks arithmetic, ideas that trade extra compute for fewer bytes (unpacking, speculative rows,
  keep-alive spinning, wider prefetch) can win on time and lose on GPU-rail energy; ideas that cut arithmetic (fewer
  experts, fewer rows, early exit) should show in energy even at flat time.
- No energy claim for streamed models can be made until whole-machine measurement exists.

### Gaps
- No whole-machine or wall-power number was found in RESULTS.md; whether an SMC reader exists in `tools/` was not
  checked (only the listed files were read).
- No idle-subtracted (net) J/token, no temperature gate, no battery-vs-AC or sustained-throughput run (0.16, 4.5).
- No per-kernel or per-component energy split.

## 7. Speculative decoding: what exists, acceptance, speedups, costs, rejections

### Takeaway
Two exact (token-identical to plain greedy) speculative paths exist outside the serving scheduler: the model's own MTP
head for Qwen3.5 / 3.6 (about 2x on chat text at one request for the in-memory 9B, only +7-14% on the streamed 35B)
and an EAGLE-3 head for gpt-oss-20b (about 1.15-1.2x over our plain decode on chat). Gains vanish at 8+ concurrent
requests and on undraftable text, and on MoE each verified token pulls its own experts.

### Cited Findings
- **MTP (Qwen3.5-9B):** one MTP layer fetched separately (487 MB of the 19 GB checkpoint), quantized to 4-bit; the
  draft head scores only the 65,536 most frequent token ids (full 248k head 5.9 ms per draft vs 1.9 ms for the whole MTP
  layer); bf16 or 8-bit head gave about the same acceptance; one host sync per iteration; DeltaNet and conv state roll
  back via the `keep_steps` kernel variant — [RESULTS 1j](../../RESULTS.md), [spec_decode.py](../../../src/mj_mlx/spec_decode.py)
- mlx-lm cannot speculate on this family (recurrent cache not trimmable; strips `mtp.*`) — [RESULTS 1j](../../RESULTS.md)
- **Exactness work:** multi-token steps first changed outputs; fixed by float32 gather conv for steps <= 16 tokens,
  `qmv_rows` for 2-4 rows, row-by-row paged attention for 2-4 token items, per-row router logits for gpt-oss, per-row
  8-bit calls for the 35B's router; result: 1-4 token steps give identical numbers (mean KL 1.841e-4) — [RESULTS 1j, 1p, 1u](../../RESULTS.md)
- **Qwen3.5-9B, one request, chat:** MTP k = 2: 22.08 / 22.80 ms/token vs plain 46.46 / 46.66 and mlx-lm 46.49 / 45.91
  (2.0-2.1x), acceptance 0.82, 406 / 350 mJ/token; next engine Ollama with MTP 33.67 ms (acceptance 0.76-0.79);
  vllm-mlx with MTP 41.65 ms — [RESULTS 1k](../../RESULTS.md)
- **Random prompt:** acceptance 0.32; still 1.23-1.25x faster than plain but more energy (543 / 536 vs 456 / 435 mJ) — [RESULTS 1k](../../RESULTS.md)
- **Cost per iteration (calibration):** k = 0 / 1 / 2 / 3: 463 / 738 / 912 / 1203 mJ and 47.6 / 57.0 / 61.3 / 76.9 ms; the
  4-token step is slow because the exact 4-row kernel is 1.3x MLX's 4-row time (0.86 vs 0.66 ms) — [RESULTS 1l](../../RESULTS.md)
- **Adaptive policy:** energy mode picks 2 drafts on chat (358 / 339 mJ, 21-33% below plain) and falls back to plain on
  hard stretches; `edp` used for batches; `PROBE_GAIN` and `DEPTH_DECAY` added — [RESULTS 1l, 1n, 1r](../../RESULTS.md)
- **Batched MTP (`SpecBatch`, outside the scheduler):** vs plain: 1 request 1.80-1.89x; 2 requests 1.41-1.42x with
  30-43% less energy; 4 requests 1.25-1.30x (fixed k = 1 or adaptive with decay) with 44-47% less; 8 requests plain is
  fastest, adaptive at parity (107.9 vs 108.9 tok/s). "Rule for now: no speculation at 8+ concurrent requests on this
  Mac" — [RESULTS 1n, 1o, 1r](../../RESULTS.md)
- **MTP on the streamed Qwen3.6-35B:** MoE MTP head (1.7 GB bf16; 256 experts quantized to 4-bit, 0.45 GB resident);
  k = 0 / 1 / 2: 16.97 / 18.78 / 19.41 tok/s, acceptance 0.96-0.97; "+14% and not 2x: each verified token routes to
  its own 8 experts, so a 3-token step loads about 2.5x the unique experts of a single step"; later same-prompt check:
  +7% over the engine path (19.01 / 18.86 vs 17.74); 128 slots + MTP: Metal out of memory — [RESULTS 1u, 1v](../../RESULTS.md)
- **EAGLE-3 (gpt-oss-20b):** public head nebius/EAGLE3-gpt-oss-20b (701 MB bf16, 4-bit at load), `fc` over the residual
  entering layers 2, 12, 21, one decoder layer, 64k draft vocabulary; no rollback needed (no recurrent state) — [RESULTS 1p](../../RESULTS.md), [eagle3.py](../../../src/mj_mlx/eagle3.py)
- **EAGLE-3 results, chat, same ids:** k = 2 26.21 / 26.18 ms/token, k = 1 26.87 / 26.33, plain 30.72 / 30.23,
  llama-server 28.07 / 27.18; acceptance 0.74 (k = 1), 0.64 (k = 2); 277-286 mJ vs llama.cpp 361-368; output identical
  to plain in every run. Quiet window: 1.06-1.08x llama.cpp with 22-29% less GPU energy — [RESULTS 1p, 1q](../../RESULTS.md)
- llama.cpp with the same head: draft on the GPU ran out of Metal memory; draft on the CPU accepted 0.54 but was
  slower than its plain decode (33.7-34.6 vs 29.4-30.3 ms/token) — [RESULTS 1p](../../RESULTS.md)
- **Rejected / neutral within speculation:** time-mode policy (did not beat fixed k = 2); energy-only policy at batch
  (drifted to 0 drafts); eager pipelined verify (19-43% slower on the 9B: a rejected step costs a whole verify step);
  deferred pipelined verify (no gain: the Python graph build is not on the critical path); speculation at 8+ requests — [RESULTS 1l, 1n, 1v](../../RESULTS.md)
- **Structural limits stated:** the Rust scheduler has no multi-token decode items or rollback, so speculation is not
  in the serving engine; the MTP loop cannot use overlap; EAGLE-3 and MTP are greedy, one request (or a fixed batch) — [RESULTS 1n, 1u](../../RESULTS.md), [usage](../../../docs/usage.md)
- The cache-policy study names literature candidates the project noted but did not build: prefetch the verify step's
  experts from the draft tokens (SP-MoE, MoE-SpeQ) and draft lengths chosen by the new experts they pull in
  (DraftExpert; "exact, 1.45x reported") — [RESULTS 1v](../../RESULTS.md)

### Inferences
- On bandwidth-bound dense decode, verify rows are nearly free in time up to 3 rows, so acceptance converts almost
  directly to speed; on MoE (gpt-oss, Qwen3.6) each extra row costs ~4-8 more experts per layer (gpt-oss verify 32.3 ->
  43.9 -> 56.2 ms), which caps the gain, and on streamed models it also raises SSD misses.
- No speculative path exists for OLMoE, DeepSeek-V2-Lite or Qwen3-30B (checklist 3.4), and none is in the benchmark
  suite's standard grid (random-token prompts give nothing to draft).

### Gaps
- No speculative numbers after the host gate / 3-bit head changes for gpt-oss (EAGLE figures are from 1p / 1q).
- No sampling-mode (non-greedy) speculative decoding; no acceptance data on long documents or code.

## 8. Experiments tried and rejected or found neutral (with the stated reason)

### Takeaway
About fifty experiments were closed. The recurring reasons: the SSD, not the GPU or policy, sets the pace on streamed
models; MLX's built-in `gather_qmm` / matmuls are already near bandwidth or compute limits; anything that wires or
keeps more memory alive fails on 16 GB; and near-ties in bf16 forbid kernels that change rounding on decode paths.

### Cited Findings

**Expert streaming and caching**
- Keep resident experts' rows on a speculative miss and recompute only absent ones: no gain (12.48 -> 12.79, 12.45 ->
  11.55 tok/s); the GPU is mostly idle and SSD reads set the pace. Code removed — [RESULTS 1d.3](../../RESULTS.md)
- Prefetch the next layer's top-(k + N) predicted experts: more hits but +48-108% SSD bytes and fewer kept speculative
  passes; 13.5 -> 11.2 -> 8.7-9.5 tok/s. Code removed — [RESULTS 1d.3](../../RESULTS.md)
- LFU or 2-ahead prefetch for gpt-oss at 24 slots: save <= 0.5 ms; even Belady stalls 6.9 ms vs 9.7. "Not worth
  pursuing at this budget" — [RESULTS 1d.3](../../RESULTS.md)
- Eviction policy for Qwen3.6 (LFU, LFU with decay): at most ~7% fewer misses than LRU; pure LFU is worse (17.6% vs
  11.1%). LRU stays — [RESULTS 1v](../../RESULTS.md)
- `WAIT_PENDING` (wait for in-flight reads, then queue the pass): discarded passes 1386 -> 626 but no time gain. Off — [RESULTS 1s](../../RESULTS.md)
- Prefetch depth 3: keeps more passes, reads 25% more, no gain over depth 2 (depth 2 is +4-8% over 1 and is the
  default) — [RESULTS 1s](../../RESULTS.md)
- Learned linear router lookahead: overfits at ~6k training tokens; at best +0.7 points recall@8 over the next layer's
  own router (0.715 vs 0.708 at d = 2). Not deployed — [RESULTS 1u](../../RESULTS.md)
- GPU spin-wait on a shared flag instead of the per-layer Metal event: CPU writes are not visible to a running Metal
  kernel (any encoder / barrier / memory type tried). Code removed — [RESULTS 1u, 1v](../../RESULTS.md)
- Wider keep-alive spinners (256, 2048 threads): worse than a 32-thread spin (1092-1169 vs 594 µs) — [RESULTS 1u](../../RESULTS.md)
- Committing MLX's command buffer from the C++ primitive: Metal assertion. Blocking `waitUntilSignaledValue`: ~5 ms
  per step wake latency (1 ms spin used instead). C++ path without engine overlap: slower (14.6 tok/s) — [RESULTS 1u](../../RESULTS.md)
- Prefill prefetch without the keep-resident rule: cold prefill 8-10% faster but warm prefill 14% slower — [RESULTS 1s](../../RESULTS.md)
- Prefetch together with the OS page cache at 8 slots: worse (3.8 tok/s) — [RESULTS 1d](../../RESULTS.md)
- Streaming a model that fits in memory: costs speed (45 -> 13-18 tok/s) — [RESULTS 1d.1](../../RESULTS.md)
- 128 slots as a default on 16 GB: 3 of 4 suite blocks failed in pp2048 — [RESULTS 1w](../../RESULTS.md)
- `MLX_MAX_MB_PER_BUFFER` 40 vs 150 for streaming: no consistent effect — [RESULTS 1s](../../RESULTS.md)
- Memory-mapping experts of a model bigger than RAM: Metal "Insufficient Memory" at any setting — [RESULTS 1j, 1s](../../RESULTS.md)

**Weight formats and compression**
- Hadamard-rotated 3-bit experts: rotation changes nothing (0.191 vs 0.190 relative error; experts are near-Gaussian
  with no outliers); 3-bit doubles the error for 22% fewer bytes. Not made a mode — [RESULTS 1s](../../RESULTS.md)
- Lossless coding of the 4-bit experts: codes have 3.72 bits of entropy, an ideal coder saves about 12% — [RESULTS 1s](../../RESULTS.md)
- 8-bit attention weights for gpt-oss as a default: clearly lossy (top-1 93.8%, KL 0.050 vs 96.9%, 0.016); kept as an
  opt-in (it is the `mj-a8` benchmark arm matched to llama.cpp's GGUF) — [RESULTS 1d.4](../../RESULTS.md)
- 2-bit candidate head: misses even at top-64 — [RESULTS 1zb](../../RESULTS.md)
- float32 attention compute: KL -23% but needs a mixed-precision GEMM (doubles weight bytes otherwise). Not adopted — [RESULTS 2, 6](../../RESULTS.md)
- bf16 residual stream for prefill speed: recovers ~1%; float32 kept for accuracy — [RESULTS 1j](../../RESULTS.md)
- bf16 recurrent-state storage: halves state bytes; accuracy only measured on a tiny model (a lower bound) — [RESULTS 2b](../../RESULTS.md)

**Kernels**
- Grouped MXFP4 GEMV kernel: correct but 1.7 -> 3.5 ms at B=1 and 9.0 -> 24.2 ms at B=64 vs `gather_qmm`; faster only
  near B=8 — [RESULTS 5](../../RESULTS.md)
- Padded [E, C, H] capacity layout and a per-expert quantized-matmul loop: both slower than sorted `gather_qmm` at
  B = 8-128 — [RESULTS 5](../../RESULTS.md)
- Batched copy of MLX's matvec with per-row-identical float ops: 1.06-1.15x slower than MLX at 4-8 rows (ALU-bound);
  dequantize-then-FMA matvec: only 5% faster at 8 rows — [RESULTS 1i](../../RESULTS.md)
- Staging input blocks in threadgroup memory for the 4-row exact kernel: slower at every row count. Reverted — [RESULTS 1l](../../RESULTS.md)
- Compiled `logaddexp`: rounds differently (1 ulp on ~1/3 of values) and flipped a near-tie token; left outside the
  compiled kernels — [RESULTS 1i](../../RESULTS.md)
- `mx.compile` of the gpt-oss MoE block: no change (14.23-14.37 vs 14.25-14.34 ms); it already reads at ~89 GB/s — [RESULTS 1s](../../RESULTS.md)
- Fused decode glue (`MJ_FUSE_GLUE`): bitwise identical but slower (28.45 vs 29.27, 27.37 vs 29.52 tok/s). "Cause not
  traced" — [RESULTS 1za](../../RESULTS.md)
- Split-K partition size 64 / 128 / 256 / 512: all within noise — [RESULTS 1x](../../RESULTS.md)
- GQA paged kernel for MLA decode: 21 tok/s at d2048 and 1.7 at 16k (one KV head = one threadgroup holding all 16
  heads) — [RESULTS 1zd, 1zm](../../RESULTS.md)
- Absorbed MLA attention in prefill: ~1.8x the attention math; replaced by expanded K/V — [RESULTS 1zd](../../RESULTS.md)
- Per-slot recurrent state re-stacked every step: lost the batched win (L32 0.97x) — [RESULTS 1f](../../RESULTS.md)
- Per-group synced tracing: inflated GPU time to 150 ms per step vs 50 (clock drops) — [RESULTS 1t](../../RESULTS.md)

**Memory and OS**
- `F_RDADVISE` read-ahead of the next layers' expert files: 7.1-7.8 s per prefill, ~10.5 GB read each time. Reverted — [RESULTS 1m](../../RESULTS.md)
- madvise hints (random, willneed) and wired limits of 5500 / 8000 MB: no help; 11000 MB runs out of Metal memory — [RESULTS 1p](../../RESULTS.md)
- Copy-on-write and read-only mappings of experts — [RESULTS 1p](../../RESULTS.md)
- `MLX_MAX_MB_PER_BUFFER` 4000 (out of memory in long prefills), unlimited (out of memory in a 512 chunk), 2600 global
  (cost prefill: pp512 278.63 vs 397.58) — [RESULTS 1p, 1zc](../../RESULTS.md)
- `map_head` (GPU gathers from mapped embedding / lm_head): thrashed, Metal wires whole buffers. Opt-in only — [RESULTS 1y](../../RESULTS.md)
- gpt-oss with lean off: does not fit (0.07-0.08 tok/s, MLX peak 15.7 GB) — [RESULTS 1y](../../RESULTS.md)
- Larger gpt-oss prefill chunk (1024, 2048): level or slower, higher peak — [RESULTS 1za, 1zf](../../RESULTS.md)
- Chunk 1024 / 512 as the DeepSeek-V2-Lite 16k fix: works but costs 6-15% of prefill; 64 MB score blocks chosen — [RESULTS 1zm](../../RESULTS.md)
- 256 MB score blocks: over-limit lines and two kills — [RESULTS 1zm](../../RESULTS.md)

**Host / scheduling**
- `MLX_METAL_FAST_SYNCH=1`: no clear gain on either path. Not enabled — [RESULTS 1z](../../RESULTS.md)
- Main-thread QoS `USER_INTERACTIVE`: inconclusive (the window stayed in the fast state) — [RESULTS 1zb](../../RESULTS.md)
- Removing remaining lean host syncs (~1-1.5 ms): not pursued — [RESULTS 1y](../../RESULTS.md)
- mlx-lm prefill knobs (prefill batch / step): none beats its defaults — [RESULTS 4](../../RESULTS.md)

**Speculative decoding**
- Time-mode draft policy; energy-only policy for batches; eager and deferred pipelined verify; MTP at 8+ requests;
  full-vocabulary draft head; bf16 / 8-bit MTP head (same acceptance, slower) — [RESULTS 1j, 1l, 1n, 1v](../../RESULTS.md)

**Not attempted by decision**
- The Neural Engine (fixed shapes vs routing, second weight copy, shared bandwidth) — [neural-engine](../../../docs/neural-engine.md)
- GGUF loading; q8_0 KV for rivals — [models](../../../docs/models.md), [RESULTS 1zi](../../RESULTS.md)

### Inferences
- Ideas likely to be re-proposals of closed work: smarter eviction (LFU / frequency), wider or deeper prefetch,
  learned routing predictors of the simple linear kind, rotation-based low-bit expert quantization, entropy coding of
  4-bit expert codes, OS read-ahead / madvise, GPU-side polling of CPU state, a custom grouped expert GEMV, and
  pipelining the speculative verify step.
- Ideas the files leave open in the same areas: prefetch driven by draft tokens, draft lengths chosen by expert
  overlap, routing-aware expert reuse across tokens, KV quantization, a mixed-precision GEMM, a conditional path that
  skips the per-layer round trip, non-linear or trained-at-scale routing predictors (the linear one overfit on 6k
  tokens).

### Gaps
- Most rejections rest on one or two paired rounds on a noisy 16 GB laptop, several "during a 72 GB download"; the
  files say so. A rejection there means "no gain at this budget on this machine".

## 9. Open questions the project itself flags

### Takeaway
The unexplained items cluster in four places: small per-token gaps on in-memory models, the gpt-oss exact path's
state dependence and silent exit, residual page-ins, and the streamed path's idle time and energy.

### Cited Findings

**Speed, unexplained**
- gpt-oss 8-layer bench prefill: per-layer MoE and attention match mlx-lm (185 vs 180 ms, 51 vs 49 ms at 2048) "yet
  the full prefill is still 10-15% slower; not explained yet" — [RESULTS 1e](../../RESULTS.md)
- Qwen3.5-9B prefill gap (3.5% at 2048): "spread over many small ops with no single cause found. Not fixed"; "not a
  single slow part; not pursued further" — [RESULTS 1j, 1zc](../../RESULTS.md)
- Long-prompt prefill gap "not investigated yet" (4-6% behind mlx-lm at 512-2048) — [RESULTS 1i](../../RESULTS.md)
- gpt-oss decode: "the GPU step (35 ms) is 7 ms over its parts (28 ms) even with few page-ins ... Not yet explained"
  (later partly attributed to host gaps and MLX's eval throttle; "Interpretation (not proven)") — [RESULTS 1y, 1zc](../../RESULTS.md)
- "Unexplained state dependence of the exact path": the same command gave 33.16-33.99 tok/s and, half an hour later,
  27.86-30.13; not the GPU clock, not slower kernels, no thermal note; QoS test inconclusive — [RESULTS 1za, 1zb, 1zc](../../RESULTS.md)
- Fused decode glue slower although bitwise identical: "Cause not traced" — [RESULTS 1za](../../RESULTS.md)
- OLMoE with float16 scales in bf16 (48 tok/s; "consistent with gather_qmm reading every expert; not traced further") — [RESULTS 1zd](../../RESULTS.md)
- KV in-place update failing under overlap: "Likely cause, not verified" — [RESULTS 1x](../../RESULTS.md)
- Streamed trace: a 14 MB in_proj at 24-34 GB/s against ~90 GB/s in the 9B; clock drop suspected, "not separated yet" — [RESULTS 1t](../../RESULTS.md)
- Deferred pipelined verify: round-2 slowdown on the SSD side (IO read wait 6.3 s vs 4.0 s, misses identical). "Not
  explained" — [RESULTS 1v](../../RESULTS.md)
- Prefill prefetch's effect on the following decode (16.46 vs 17.30 tok/s, one run each): "not settled" — [RESULTS 1s](../../RESULTS.md)
- Streamed GPU idle: "Not yet split" between SSD misses and routing readback at the time; later split by the C++ trace — [RESULTS 1s, 1u](../../RESULTS.md)
- DeepSeek-V2-Lite decode vs context: "a speed gap to work on"; MLA kernel "Its speed: next" — [RESULTS 1zi, 1zm](../../RESULTS.md)
- OLMoE d0 gap: within our own run-to-run spread, "the spread is itself a finding" — [RESULTS 1zg](../../RESULTS.md)
- At 4 requests the adaptive draft mix chains the MTP layer to the largest k: "Not tuned yet" (later improved by
  DEPTH_DECAY) — [RESULTS 1n, 1r](../../RESULTS.md)

**Memory and reliability, unexplained**
- pp2048 with mlock still pages in 456-595 MB per rep: "Not yet traced (candidates: non-expert pages, the bf16 rescore
  rows, prefill temporaries pushing out other file cache)" — [RESULTS 1y](../../RESULTS.md)
- The silent exit of the exact gated process: "Cause unknown"; candidate (memorystatus kill at 3000 MB wired) "not
  reproduced on the old code to prove the link" — [RESULTS 1zc, 1zh, 1zi](../../RESULTS.md)
- System-wide stalls (5.55 tok/s, 16.49 s TTFT) also hit ungated runs — [RESULTS 1zc](../../RESULTS.md)
- Ollama's DeepSeek-V2-Lite swap growth (4852 / 8923 MB) and slow, noisy decode: "cause not established", "Not
  investigated" — [RESULTS 1zg, 1zf](../../RESULTS.md)
- LM Studio one-token replies on DeepSeek-V2-Lite: "cause of the one-token replies not established" — [RESULTS 1zh](../../RESULTS.md)
- vllm-mlx at 8k / 16k streams no tokens: "Cause not established" — [RESULTS 1zi](../../RESULTS.md)
- llama.cpp tg128@d2048 failing with Metal out of memory in some rounds on gpt-oss — [RESULTS 1zc, 1zh](../../RESULTS.md)
- The paging rule for gpt-oss tables (net-swap rule misses heavy paging; a timed-rep rule would flag nearly every
  block of every engine) — [RESULTS 1zi, 1zk](../../RESULTS.md)

**Energy, unexplained**
- Why speculative dispatch raised gpt-oss's GPU-rail J/token and lowered Qwen3.6's: "not established (not
  investigated)" — [RESULTS 1d.3](../../RESULTS.md)
- Overlap's energy cost at one request: "Not resolved" — [RESULTS 1h](../../RESULTS.md)
- Keep-alive's energy cost: "Not measured yet" — [RESULTS 1u](../../RESULTS.md)
- Streamed-model energy: no valid number (GPU rail blind to SSD and CPU readers) — [RESULTS 1d](../../RESULTS.md)

**Accuracy, open**
- llama.cpp's gpt-oss KL against the float32 reference: not measured — [RESULTS 1ze](../../RESULTS.md), [LAUNCH_TODO](../../LAUNCH_TODO.md)
- llama.cpp F16 perplexity 11.532 vs 11.492 from saved logits: "The cause is not established" — [RESULTS 1ze](../../RESULTS.md)
- bf16 recurrent state on real Qwen3.6 weights: not measured — [RESULTS 2b](../../RESULTS.md)
- Greedy divergences from mlx-lm after the first token "were not checked one by one" — [RESULTS 1w](../../RESULTS.md)
- Long-context teacher-forced accuracy (0.15), real-chat check (0.12), MMLU / GSM8K (0.2): not run — [LAUNCH_TODO](../../LAUNCH_TODO.md)
- Slot count changes bf16 accumulation order, so greedy output depends on the slot setting at exact ties — [RESULTS 1s](../../RESULTS.md)

**"Left:" items and stated next steps**
- Checklist "Left:" items are listed under question 1 — [LAUNCH_TODO](../../LAUNCH_TODO.md)
- Not done for Qwen3.6: radix-cache support (state snapshots at block boundaries), a chunked (WY / UT) delta-rule
  kernel for long prefills on many heads — [internals](../../../docs/internals.md)
- A grouped MXFP4 GEMM kernel that beats `gather_qmm` at B >= 32; float32 attention compute via a mixed-precision GEMM — [internals](../../../docs/internals.md)
- Expert-major prefill and keeping prompt-hot experts as "the next levers" for streamed TTFT — [RESULTS 1d](../../RESULTS.md)
- "With exact outputs, SSD misses (about 12 ms per step) are the bigger lever" than the per-layer round trip — [RESULTS 1v](../../RESULTS.md)
- Engine should pick the prefill chunk from the context length; `prefill_area_cap` exists but is off — [RESULTS 1zi](../../RESULTS.md), [CHANGELOG](../../../CHANGELOG.md)
- Known limits: C++ streamer needs >= 4 x tokens x top-k slots per layer; no prefix cache for Qwen3.5 / 3.6; MLA keeps
  one KV head — [models](../../../docs/models.md)

### Inferences
- The questions most likely to respond to outside ideas: (1) how to cut unique experts or SSD bytes per token on
  streamed models without changing outputs; (2) how to remove a per-layer host round trip when routing is
  data-dependent; (3) an MLA decode path that scales with context; (4) why `gather_qmm`-based prefill is 10-20% under
  the dense matmul rate; (5) memory-safe long context on 16 GB (KV quantization, score budgets) with an accuracy gate.
- The questions that are measurement problems rather than design problems: the exact path's state dependence, the
  paging rule, whole-machine energy.

### Gaps
- Document inconsistencies found while reading (so a later report does not cite the stale one): README says
  DeepSeek-V2-Lite past 8k needs `--chunk 512` (1zj) while 1zm and usage.md say the 64 MB default removed that need;
  docs/usage.md says score blocks stay "near 256 MB per layer, then in blocks of 64 MB" (mixed wording);
  docs/neural-engine.md cites whole-machine SMC energy "RESULTS 1zm onward", which RESULTS 1zm does not contain;
  docs/internals.md still lists overlap scheduling as a known gap; benchmarks.md mentions suite "v15", for which
  RESULTS has no section yet (latest is v14 / 1zm).
- A timed benchmark was running during this review, so newer results may exist that are not yet written into
  RESULTS.md.
