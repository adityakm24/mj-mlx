# MLX engine: launch checklist

Kept by the MLX Agent; updated as items land (commit hashes and RESULTS sections in brackets). Order agreed with
the user on 2026-11-19: Phase 0 first, then Phase 1, then the rest.

## Phase 0: claims we can defend
- [x] 0.1 Standard accuracy suite (RESULTS 1ze: OLMoE, DeepSeek-V2-Lite, Qwen3-30B, gpt-oss-20b; gpt-oss vs a
      full-precision reference in 0.1b): WikiText-2 perplexity (llama.cpp convention: 512-token chunks, last half
      scored), KL divergence vs the bf16 model, on ours / mlx-lm / llama.cpp. Models: OLMoE, DeepSeek-V2-Lite,
      Qwen3-30B-A3B, gpt-oss-20b.
- [x] 0.1b gpt-oss-20b WikiText vs a full-precision reference: reference 180.29; ours exact +0.5%, 8-bit -1.8%,
      llama.cpp -4.1% (RESULTS 1ze).
- [~] 0.2 MMLU / GSM8K subsets (RESULTS 1zr): ours equals mlx-lm on the same weights for OLMoE (53.3 / 65.2 vs 53.4 /
      65.2) and DeepSeek-V2-Lite (53.5 / 48.8 vs 53.8 / 49.2), level with llama.cpp on OLMoE. Left: llama.cpp on
      DeepSeek-V2-Lite (its server errors on this model); gpt-oss needs a chat-format evaluation (its raw few-shot
      numbers are not comparable across engines).
      2026-11-23 (RESULTS 1zz): gpt-oss chat format, every 4th item, ours exact (experts unpinned) vs llama-server:
      MMLU 85.6 vs 85.3% (10 / 9 discordant), GSM8K 85.7 vs 92.1% (0 / 4 discordant, sign test p = 0.125). 3 of the 4
      are a "#### 1" ending after a correct derivation, which llama-server's replies also show (3 items); a near-tie
      is a hypothesis. No significant difference at n = 285 / 63. Same-weights check done (v17d, 1zz): teacher-forced
      through our float32 reference and mlx-lm's own layers, the reference picks what our engine wrote on 6 of 7,
      and 1167 is a 0.026-nat tie both bf16 paths round to a tie; our exact arm is closer to the reference than
      mlx-lm's bf16 (97.6 vs 96.8% same top token). The "exact" label stands. Left: llama.cpp on DeepSeek-V2-Lite
      (0.6).
- [~] 0.3 Quality-matched comparisons (quality of each engine's 4-bit format next to its speed). OLMoE done (5-bit
      g64: lower perplexity than Q4_K_M, decode still ~12% ahead, RESULTS 1ze). gpt-oss: the 8-bit arm leads decode;
      its perplexity is closer to the f32 reference than llama.cpp's (-1.8% vs -4.1%); llama.cpp's KL against that
      reference is not measured; the exact arm ties llama.cpp in decode; prefill gap grows. DeepSeek-V2-Lite and
      Qwen3-30B next.
- [~] 0.4 Ollama, LM Studio, vllm-mlx in the suite. v10 (Ollama, RESULTS 1zf): decode ahead of Ollama on all 3
      models; its prefill rows withdrawn (prompt-cache reuse from a shared seed, fixed in 70903ec). LM Studio (MLX
      and llama.cpp engines) and vllm-mlx wired (70903ec). Suite v11 with all six engines done (RESULTS 1zg): OLMoE
      clean; DeepSeek-V2-Lite and gpt-oss mostly flagged (swap 4676 MB at start, 13 of 34 blocks flagged), so their
      server rows wait for the quiet re-run (0.5); LM Studio DeepSeek-V2-Lite decode needs 0.13.
- [~] 0.5 Quiet-machine re-runs of the headline tables (no flagged blocks). v12 after a restart (RESULTS 1zh): OLMoE
      clean for every engine; DeepSeek-V2-Lite clean except mlx-lm and LM Studio (their own memory swaps); gpt-oss
      8-bit arm clean, exact arm never clean (its own wired memory; killed by macOS once). v13 (RESULTS 1zi): exact
      arm fixed (mj-x, MLX wired limit 1000 MB): clean, ahead of LM Studio / Ollama, tied with llama.cpp. Left: mj-a8
      to the 1000 MB limit (all v13 Metal over-limit lines were its blocks), then a re-run; the paging rule: every
      engine swaps out 0.6-2.8 GB during gpt-oss timed reps on 16 GB, so the timed-rep rule would flag nearly all of
      them (say so on any gpt-oss table, or pick the rule with the PM).
      v14 (RESULTS 1zk): the 8-bit arm at the 1000 MB limit is clean and ties 3000 MB on every test; 1000 MB is the engine
      default since e40c051. The net-swap rule missed a block with 13 GB of swap-ins (1zk) and Ollama blocks with 7.8 /
      14.8 GB (v15): add a swap-in rule (to agree with the PM). v15 first pass (2026-11-21/22) ran partly with another
      agent session active (5 of 12 gpt-oss blocks flagged); a fresh gpt-oss and DeepSeek-V2-Lite pass on the quiet
      machine is running (`bench_suite/v15q_*`). OLMoE v15: every block clean.
      v16 (RESULTS 1zu, quiet machine, 2026-11-23): Qwen3.6-35B ours clean; llama.cpp can never be clean there (the
      model is bigger than RAM: every block swaps). gpt-oss: one clean round for every engine except LM Studio (both
      rounds flagged, 0.3-0.9 GB swap growth). Swap stop rule added: a block is killed past 20 GB of swap-outs.
- [~] 0.6 llama.cpp energy per token (llama-server window, not llama-bench). Suite engine `llama-server` merged and run
      in v15: OLMoE valid (decode 100.54 / 88.28 tok/s; whole machine above idle 162 / 313 mJ per token vs ours 71 /
      249); gpt-oss first pass noisy (750 / 850 vs ours 8-bit 500 / 553), clean pass running; DeepSeek-V2-Lite invalid
      (the server returned one-token replies and an impossible prefill rate: cause to find). Left: the clean gpt-oss
      numbers, the DeepSeek-V2-Lite fix, RESULTS write-up.
      2026-11-23: those DeepSeek rows had no chunks and no usage, so the server most likely sent an error event that
      our client ignored; the client now records `error` events and marks such rows invalid (e2ff236). The cause on
      the server side is still to find (run the server directly, v17b). llama-server's steady-state energy measured on
      OLMoE and gpt-oss (RESULTS 1zw).
      2026-11-23 (RESULTS 1zzb): run directly, the server fails warm-up and every request with Metal out of memory on
      84% free memory; its log: "failed to fit params ... n_gpu_layers already set by user to 99". At -ngl 99 it does
      not fit at 8k (1zi: at any batch size). v17d (1zzb): -ub 256 / 128 fail too; at the default -ngl (partial
      offload) it serves (tg128 43-48 tok/s, QA 7 of 7 at 4k) but swaps past 20 GB in QA and in MMLU. Closed for this
      Mac: llama.cpp does not fit DeepSeek-V2-Lite at 8k under macOS's default GPU working-set limit (no sudo), and
      swaps when partially offloaded; its energy row stays invalid.
      Client check on gpt-oss: decode through our client within 2.2% of llama-bench, server prefill 10.6% lower.
- [~] 0.7 Long context (16k / 32k). Scope set 2026-11-21 (the user left the call to us): batch throughput (many
      requests at once) and chat-style speculative decoding moved to Phase 3 (3.6, 3.4): they are speed work, not the
      defence of a published claim. 32k stays here as a capacity check (with 0.10). v13 (RESULTS 1zi): 16k gpt-oss, mj-a8
      1.29x prefill / 1.12x decode vs llama.cpp at its best fitting setting; DeepSeek-V2-Lite 8k / 16k where llama.cpp
      does not fit. Left: 32k; the exact arm at 16k; Ollama / LM Studio past 8k (harness fixes num_ctx 8192); vllm-mlx at
      8k+ invalid (no tokens streamed, cause unknown).
      32k (RESULTS 1zp): DeepSeek-V2-Lite runs it on 16 GB (~240 tok/s prefill, 13.6 decode; capacity result);
      gpt-oss does not complete 32k on either engine at defaults (ours prefills, then runs out of Metal memory).
- [x] 0.7b Long prompts without a flag (RESULTS 1zj, 1zm): prefill attention in query blocks of 64 MB of scores for head
      sizes MLX's fused kernel does not take. DeepSeek-V2-Lite 16k at the default chunk: no Metal over-limit lines, no
      kills (256 MB blocks: hundreds of lines, two kills), prefill +1-5%. Scheduler `prefill_area_cap` available (off).
- [~] 0.7c DeepSeek-V2-Lite decode falls with context (52.24 at 2k, 33.82 at 8k, 22.17 at 16k). Measured 2026-11-21:
      the GQA paged kernel is far worse for MLA (21 / 3.8 / 1.7 tok/s at 2k / 8k / 16k, RESULTS 1zm); a dedicated MLA
      decode kernel (`mla_decode.py`, opt-in `MJ_MLA_DECODE=kernel`) is 2-4x more accurate than the bf16 SDPA path and
      token-identical in 11 of 12 greedy replies, but slower at 2k (44.1 vs 52.0) and 8k (30.5 vs 32.8) and tied at 16k
      (22.5 vs 22.3) (`bench_suite/v15_mla_kernel/`, one round each clean). So the gather copy is not the cost. Left:
      profile with the GPU clock / DRAM counters to find what is; write up v15_mla_kernel in RESULTS.
- [x] 0.8 One command that reproduces every published number: `tools/reproduce.py` + `results/published.json` (RESULTS
      1zj). First full run done (RESULTS 1zl): every claim ran, every judged sentence holds, accuracy identical to every
      digit. Follow-up (not blocking): add the v13 README rows (gpt-oss exact arm, 16k context) as claims.
- [~] 0.9 Rivals at their best settings, written down per engine, versions pinned. Written down (docs/benchmarks.md, Rival
      settings) with the capacity rule; every rival's version is now in each run's meta.json. Left: the llama.cpp
      threads x ubatch sweep (queued tonight); Ollama num_batch, LM Studio batch, vllm-mlx prefill step; and the new
      threat from the 2026-11-22 report: llama.cpp's expert cache `--moe-cache-mib` against our Qwen3.6 claim (6.0).
      Done 2026-11-22 (RESULTS 1zp): llama.cpp -t x -ub on OLMoE and gpt-oss; its defaults are best or within 3%.
      Done 2026-11-23 (RESULTS 1zs): `--moe-cache-mib` cannot run on Qwen3.6-35B on 16 GB (mmap disables it; without
      mmap the experts do not fit), so `-ncmoe 40` stays llama.cpp's best working setting there.
- [~] 0.10 Memory as a claim: peak RAM / GPU memory per engine per test, and the largest model / longest context each
      engine runs on 16 GB. In the suite since v15: per block, peak footprint (macOS's figure, GPU buffers in, mapped
      files out) and peak resident (mapped weight files in), plus swap-ins / -outs. First numbers: DeepSeek-V2-Lite
      ours 11.76 GB footprint, vllm-mlx 12.6, Ollama 11.2, llama.cpp 1.35 footprint / 11.3-11.6 resident. Longest
      context: 1zi / 1zj / 1zm (gpt-oss 16k all engines; DeepSeek-V2-Lite 8k / 16k ours only). Left: 32k (queued
      tonight), the clean gpt-oss numbers, a RESULTS table.
      2026-11-23: RESULTS 1zzf compiles it (what runs, longest verified context, peak memory per engine with the
      comparability caveat). Left: Qwen3.6 long context on ours, gpt-oss 32k decode, Ollama / LM Studio long
      context and process memory.
- [~] 0.11 Load time and time to first token from a cold start, per engine. First run (RESULTS 1zp): OLMoE ours 0.76 s,
      2.9x faster than mlx-lm, llama.cpp 0.68 s; **gpt-oss ours 6.4 s vs 2.3-2.9 s for LM Studio / llama.cpp (a loss:**
      we pin 12 GB of experts and pack attention before the first request). Left: rerun gpt-oss on a clean machine;
      capture LM Studio's first token when its one-token reply is empty; a disk-cold run needs `sudo purge` (not run).
      Fix to try: serve the first request while experts are pinned in the background (6.8).
      2026-11-23 (`cold_start_v2/`, 2 rounds x 3): first token, median, mj-a8w 7.61 s, mj-a8wb (MJ_MLOCK=bg: experts
      pinned in a background thread) 5.86 s, llama.cpp 2.65 s. Ready drops from 6.27 to 3.03 s with the background
      pin, but the first request then waits about 2.8 s more. Still a loss. A phase profile on the same path is queued
      (v17b `coldprof`). An Engine built without wired_limit_bytes was raising MLX's wired limit before dropping it for
      mapped experts; fixed in 5e5f9e4. The suite never took that path.
      2026-11-23 (RESULTS 1zza): phases on the cold-start path: a8w 5.0-5.4 s, a8wb 3.8-4.5 s (one cold-file run 10.3
      s). Load: mlock 1.19 s, the model constructor 1.13 s, layer building 0.81 s, mmap 0.29 s; first request: arming
      the first step 0.71 s, the wired-limit call 0.19-0.45 s. The wired-order fix cut Engine construction from 1.25 s
      to 0.07 s. Next: the constructor / layer building and the first step's arming.
      2026-11-23 (RESULTS 1zzc): load stages: ~1 s of every load re-quantizes the head, 0.7 / 3.4 s the attention.
      MJ_PREP_CACHE keeps them on disk (same tokens on the real model): load to first token 1.8 s (8-bit) / 2.6 s
      (exact) from 3.7-5.2 / 7.5-7.7; cold start median first token 2.52 s vs llama.cpp 3.14 s (level: ranges overlap,
      one round each way). Left: cache on by default; rerun on a quiet machine.
- [ ] 0.11b gpt-oss load swaps out about 2.2 GB every time (RESULTS 1zw: 1.6-2.2 GB per block, all at load, at most 18 MB
      inside any timed window). Pinning 10 GB of experts pushes other memory out: SSD wear, and memory pressure for
      whatever runs next. A usability and cold-start item, not steady-state energy. To try: pin only the most-used
      experts (the exact arm leaves 13.3 GB of 16 GB wired, too little for long contexts or long replies), or pin
      lazily as experts are first used.
      2026-11-23 (RESULTS 1zzd): MJ_MLOCK_TOP=k pins each layer's k most-used experts (top 16 of 32 take 76% of
      decode routings, top 24 93%). Decode unchanged, prefill slower and noisy (414 -> 381 / 354 tok/s, CV 18-25%),
      resident 10.3 -> 7.8 / 5.2 GB; exact arm with top 16 pinned: 12 / 12 at 16k. Load swap not clearly lower
      (round 3: 4.8-4.9 GB for every arm). Option, not default. Left: lazy pinning; load swap on a quiet machine.
- [~] 0.12 Real-chat quality check on each headline model. `tools/mlx_chat_check.py` merged: 12 fixed chat prompts through
      the chat template, greedy, ours vs mlx-lm on the same weights (first differing token), replies saved. Left: the
      run (queued tonight; the tool has not run yet).
      Ran 2026-11-22: ours vs mlx-lm, 128 tokens, token-identical in 11 of 12 replies on OLMoE (first difference at
      token 81) and 9 of 12 on DeepSeek-V2-Lite (first differences at tokens 0-57: a difference at token 0 needs a
      look); gpt-oss 8-bit arm ours only (`chat_check_v1/`). Left: gpt-oss exact arm vs llama-server (v17), a RESULTS
      write-up.
      2026-11-23: gpt-oss exact arm, 12 prompts, 256 tokens (`chat_check_v1/gptoss_mj_x.json`): every reply read
      through is coherent and correct where checkable (17 x 23 steps, 10 sheep, area 40 / perimeter 26, the French,
      valid JSON); several hit the 256-token limit while reasoning. llama-server's run failed (Metal out of memory at
      warm-up, a 20k context on a machine just freed); rerun in v17b with an 8k context.
      2026-11-23 (RESULTS 1zz): llama-server at 8k ran; its 12 replies read correct. DeepSeek-V2-Lite's token-0
      difference is a bf16 tie (our margin 0.000 nat, mlx-lm's 0.094); the top token and margin agree on the other 11.
- [x] 0.13 Harness fixes: a reply of at most one token is recorded with no rate (tok_s null), not a crash;
      summaries count replies under the requested length. The v11 LM Studio DeepSeek-V2-Lite failure did not reproduce after the run (RESULTS 1zg).
- [x] 0.14 Where we lose, say so: the README's "Where it is behind" list carries every loss found so far (short-context
      decode vs vllm-mlx, small-model prefill, gpt-oss pp512, DeepSeek-V2-Lite decode at long context, paging on 16 GB),
      and suite tables keep every row. `results/published.json` checks the loss sentences on every reproduction.
- [~] 0.15 Long-context accuracy, teacher-forced (RESULTS 1zo, 1zp, 1zq). DeepSeek-V2-Lite: verified end to end at 16k
      (one forward 6.609, the Engine's 2,048-token pieces 6.604; blocked attention identical across block sizes; mlx-lm
      8.1158 vs ours 8.1157 at 4k). gpt-oss: our paths agree with each other and with our float32 math (+0.27% at 2k),
      and our YaRN parameters equal transformers'; but ours, mlx-lm (float32) and llama.cpp give 1,403 / 1,282 / 1,184 at
      2k and the gap to llama.cpp grows to 1.9x at 16k, on raw text where the model itself collapses. Left: a chat-format
      long-context task (retrieval / QA at 8k-16k) per engine; the tiny-model test against transformers past position
      4,096; the constant-logits guard (4.6). Use as the gate for every long-context change.
      2026-11-23 chat-format retrieval (RESULTS 1zx): gpt-oss ours 36 of 36 at 4k / 8k / 16k (experts unpinned);
      llama-server 12 / 12 at 4k and 8k (ubatch 256) and 11 / 12 at 16k (ubatch 128, its only fitting setting).
      DeepSeek-V2-Lite ours 10 / 10 / 9; OLMoE ours = mlx-lm token for token.
- [~] 0.16 Whole-machine energy above idle next to the GPU rail. In the harness since v15: `mj_mlx.smc_power` reads the
      SMC's system total power (IOReport's CPU / DRAM rails read 0 on this Mac, so the old "SoC" column was GPU only);
      10 s idle baseline per test with the engine loaded; the SMC reading trails the load by ~1.75 s, so windows are
      integrated shifted. v15 (RESULTS 1zo): recorded but not usable: decode windows of 0.9-3.5 s are shorter than the
      meter's lag and at d2048 sit in the tail of the prefill before them. Left: an energy test with 10 s+ of decode
      and the first seconds dropped (steady-state power over tokens per second); a temperature gate before each block;
      no wall meter to check against.
      Done 2026-11-23 (RESULTS 1zw): tg1024@d0 with the first 3 s dropped. Steady state above idle, per token:
      OLMoE ours 171 mJ vs llama-server 246 (-30%), mlx-lm 169; gpt-oss ours 8-bit 601 vs llama-server 783 (-23%).
      Steady and whole-window agree within 2-8% at this length. Left: a temperature gate; a wall-meter check.
- [x] 0.17 Prove which code ran: every worker row records the git commit, a dirty flag for src / tools, and the MJ_ /
      MLX_ environment (`prov` in each row, since v15).

## Phase 1: usable
- [x] 1.0a Own project and repository (user, 2026-11-21): mj-mlx is separate from MilliJoule, same goals for the Mac.
      Imported from MilliJoule 519af45; the scheduler is forked into crates/ (mjx-core, `mj_mlx._sched`); tests,
      tools and results moved in; 252 Python tests and 29 Rust tests pass in this repo's own venv.
- [x] 1.0 mj-mlx package (2026-11-21): `packages/mj-mlx` (`import mj_mlx`, setuptools, version 0.1.0.dev0); the MLX
      code moved there with history; `mj_mlx` is an alias to the same module objects, so tools and
      tests run unchanged (253 passed). Wheel and sdist build offline (source, LICENSE, NOTICE, docs; no build dir).
- [x] 1.0b One `placement=` for every family (done 2026-11-23: load_model handles it for GptOss and Qwen35Moe too;
      gpt-oss / Qwen repack moved into mj_mlx.expert_repack; existing stores found by source; bf16 Qwen experts stream).
      Was: `GptOss` / `Qwen35Moe` reject it today (TypeError) and take
      map_experts / stream_experts directories instead (docs/usage.md says so).
- [ ] 1.1 OpenAI-compatible HTTP server (streaming, chat templates, stop strings, sampling).
- [ ] 1.1b Server extras users expect: tool calling, JSON / structured output, prefix cache across requests,
      cancellation, usage stats; checked with the OpenAI SDK and a common UI (Open WebUI).
- [ ] 1.2 Sampling in every mode (gpt-oss lean is greedy only).
- [ ] 1.3 CLI: pull / run / serve / bench.
- [ ] 1.4 macOS arm64 wheel with the Rust core and the C++ extension prebuilt.
- [~] 1.5 Expert-repack disk use (in-place option, warning, cleanup command). 2026-11-23: `python -m mj_mlx cache
      list / rm [--orphans] [--force]` (src/mj_mlx/cache.py; 122.6 GB on this Mac today); placement.prepare refuses a
      repack that would not fit the free disk (+2 GB), saying what to do. Left: an in-place option.
- [~] 1.6 Docs; README positioning (PM-owned file); license decision and NOTICE for the transformers-derived code.
      License: Apache-2.0 for the whole repo (user, 2026-11-21); LICENSE, NOTICE, pyproject, Cargo and source headers
      done. Package docs done (README, install, usage, models, benchmarks, energy, internals, moe-design). Left:
      the root README positioning (PM) and HANDOFF's "Closed source" row (PM).
- [~] 1.7 Clear errors for unsupported models and for models too big for this Mac (what to do instead).
      2026-11-23: an unsupported model_type lists the supported families and points to mlx-lm; a repack short of
      disk says so (1.5); placement.choose raises a MemoryError saying what to do for a dense model above the GPU
      working set and for a MoE whose non-expert weights alone do not fit.
- [~] 1.8 Third-party licenses checked (dependencies, kernels adapted from others), attribution in NOTICE.
      NOTICE covers mlx-lm (gpt_oss.py), MLX (qmv_rows.py mirrors qmv_fast; qmm_small.py builds against installed
      headers) and transformers (qwen3_5_moe.py, gated_delta.py, moe_lm.py). Left: runtime dependency licenses
      (mlx, numpy: both permissive) written down, and a pass over tools/ before they ship.
- [ ] 1.9 Launch material: methodology page, charts from the suite JSON, a short "how we benchmark" (with
      the PM; the Bench Board is the user's decision).

## Phase 2: breadth
- [ ] 2.1 Fallback tier: mlx-lm model files inside the engine (dense / unknown models).
- [ ] 2.2 More MoE families: Gemma-4, GLM-4.5-Air, gpt-oss-120b, Qwen3-Next.
- [ ] 2.3 Weight formats: mlx 3/6/8-bit, MXFP4 in the general model, GGUF loading.

## Phase 3: speed gaps
- [ ] 3.1 DeepSeek-V2-Lite short-context decode (11% behind mlx-lm): MLA decode kernel.
- [ ] 3.2 OLMoE prefill (94-96% of the best rival); gpt-oss pp512 (94% of llama.cpp).
- [ ] 3.3 KV-cache quantization for long context.
- [ ] 3.4 Speculative decoding beyond gpt-oss, and chat-style speculative decoding in the suite (moved from 0.7).
- [ ] 3.5 Small-model decode (OLMoE): close the ~3% to vllm-mlx / mlx-lm at short context if the quiet
      re-run confirms it.
- [ ] 3.6 Batch throughput: many requests at once against the rivals' servers (moved from 0.7).

## Phase 4: hardware and reliability
- [ ] 4.1 Results beyond one M4 16 GB (M1-M4 Pro / Max / Ultra).
- [ ] 4.2 Soak tests; the unexplained silent exit on the exact gpt-oss gated path (RESULTS 1zc). Candidate cause
      (v12, RESULTS 1zh, not yet checked against 1zc): macOS memorystatus killed the worker at 75% of wired memory;
      Metal buffers needed 13.45 GB vs 12.71 GB maxWired. Fix: wire less on 16 GB, then soak. v13 (RESULTS 1zi): MLX's
      wired limit drives the over-limit buffers (about 1,000 per probe at 3000 MB, 0 at 1000 MB); at 1000 MB a 65-minute
      soak (450 timed reps) ran with no kill and decode within 2%. Left: reproduce the 1zc exit on the old setting to
      confirm the cause; soak the other headline models.
      2026-11-21: macOS memorystatus killed DeepSeek-V2-Lite 16k workers three times at 74-77% of wired memory (MLX
      wires ~12.7 GB by default for resident models). A 9 GB wired-limit variant (`mj-w9`) is queued against the default
      at 16k / 32k; if it holds, lower the default wired limit on 16 GB machines (6.5).
- [ ] 4.3 C++ streamer errors as Python exceptions, not process aborts.
- [ ] 4.4 CI on a macOS runner.
- [ ] 4.5 Sustained throughput: 30-minute runs, thermal throttling, on battery vs on AC.
- [ ] 4.6 Never return garbage silently: one 16k run returned constant logits (perplexity = vocabulary size) with no
      error, between two runs that failed with Metal out-of-memory (RESULTS 1zo, 1zp). Add a cheap check on sampled
      rows (constant or non-finite logits raise), and a start-up self-check of the quantized matmul kernels (an Xcode
      release once mis-compiled MLX's, per the research notes).
      2026-11-23: the lean greedy path (every model's host rescore, gpt_oss.check_output) now raises
      GarbageOutputError on a non-finite or all-zero final hidden state, non-finite logits, or candidates with equal
      logits (MJ_CHECK_OUTPUT=0 off); free, the rows are on the host already (tests/test_output_check.py, incl. an
      end-to-end NaN step). Left: the GPU-argmax paths (non-lean), the kernel self-check at start-up.
- [ ] 4.6b One bad row must end only its own sequence: GarbageOutputError names the bad sample rows (`.rows`), but
      the Engine step raises for the whole batch today. With the server (1.1), finish that sequence with an error
      and keep the others running.
- [ ] 4.7 Default wired limit on 16 GB for resident models: 9000 MB removed ~3,150 Metal over-limit lines at 32k at no
      speed cost (RESULTS 1zp), and the default was killed three times at 16k (1zm, 1zo). Before changing it: ABAB at
      short context on the headline tests (OLMoE, DeepSeek-V2-Lite, Qwen3.5-9B). Part of 6.12.

## Phase 5: the Neural Engine (planned; user, 2026-11-21: keep a way to exploit it later)
- [ ] 5.1 One-day spike on the M4 16 GB (docs/neural-engine.md): a dense FFN block through Core ML at prefill chunk sizes,
      call overhead, memory of the compiled copy, whole-machine energy against our Metal kernel, and GPU + ANE overlap.
      Go / no-go from the numbers.
- [ ] 5.2 If go: a `device` per component in placement (GPU / ANE), dense parts (shared experts, dense layers,
      attention projections) on the ANE during prefill, pipelined with the GPU's experts; fixed compiled chunk sizes.
- [ ] 5.3 A dense draft model on the ANE for speculative decoding, judged on energy per accepted token.

## Phase 6: the plan to be state of the art (user, 2026-11-22)
Three headlines, all required: **the fastest without quality loss**, **models that don't fit in memory**, **the lowest
energy per token**. Positioning: the fastest engine on the Mac at equal or better quality, for models that fit and for
models bigger than the Mac's memory, at the lowest whole-machine energy per token, every claim reproducible.
Research behind this phase: `results/reports/Apple silicon MoE inference directions.md` (17 ranked directions, 13
threats; notes in `results/research_notes/`). "Dn" below is the report's direction number.

The speed rule: a speed claim counts only with exact output (token-identical to our reference path) and at equal or
better measured quality than the rival's weights (perplexity, KL, task accuracy: 0.2, 0.3). No routing changes, expert
skipping or lossy caches in a headline number; lossy modes are opt-in and labelled. Energy claims for streamed models
are whole machine (the GPU rail does not see the SSD or the CPU). "Fastest" is scoped to the hardware we measured.

Order of work: 6.0 (threats) and 6.1 (offline study) first, both cheap; then 6.2-6.4, the three ideas that can be ours;
6.5-6.7 make the claims safe; 6.8-6.12 are catch-ups; 6.13-6.16 are hardware probes.

**A. Settle what could overturn today's claims (report: thirteen threats)**
- [ ] 6.0 Benchmark each threat on this M4, interleaved, at the rival's best quality-preserving setting. Downloads and
      new installs need the user's OK.
      - [x] llama.cpp expert cache `--moe-cache-mib` (merged Oct 2026): our Qwen3.6 5.2x and Qwen3-30B multiples are
            against `-ncmoe`. Checked 2026-11-23 (RESULTS 1zs): it needs the experts in host memory, so with mmap it
            disables itself and without mmap it swaps 12.7 GB before loading; `-ncmoe` stays llama.cpp's best working
            setting on 16 GB. Re-check on a Mac with more RAM (4.1). Tweet 1's comparison stands on this point.
      - [ ] llama.cpp disk-offload fork (slot pool, exact; 13 tok/s for Qwen3-30B on a 16 GB M1 Pro, unmerged).
      - [ ] Expert-streaming engines on Apple silicon: AiiStream (15.0 tok/s for Qwen3.6-35B on a 16 GB M5 Air, ~1.7 GB,
            self-reported), Edge0 (changes routing; quality 79.2 vs 83.2), TurboFieldfare, flash-moe, ds4, MLX-Flash.
            Compare speed per resident GB and exactness.
      - [ ] Colibrì (pure-C MoE engine; expert streaming with LRU, learned pinned hot set, next-layer router lookahead,
            O_DIRECT; Metal tier; no Apple-silicon numbers published) and AirLLM (whole layer from disk per token):
            same-Mac run on Qwen3.6-35B, interleaved. Downloads need the user's OK.
      - [ ] BaseRT (MoE engine; small-model rows are the ones at risk) and vllm-metal / Locals (batch throughput, 3.6).
      - [ ] Native MTP in rivals (mlx-lm PR, MTPLX, llama.cpp `draft-mtp`): our speculative margins.
      - [ ] Rivals' best settings at long context: q8_0 KV in llama.cpp, stated as a precision change (0.9).
      - [ ] Upstream MLX 0.32.2+ (gated-delta kernels, read-once decode kernel, fused 256-wide attention on M5): re-run
            the mlx-lm rows per MLX release; add a start-up self-check of quantized matmul outputs.
      - Not runnable here: Splash (needs 36 GB; 210 tok/s on Qwen3.6-35B on an M5 Pro with a draft model, vendor). 4.1.

**B. The ideas that can be ours (report: candidates for a general systems idea)**
- [~] 6.1 (D1) Forecast paging, offline study. Part 1 done (RESULTS 1zn): **the eviction half is closed.** A one-token
      oracle (10.84% misses) does not beat use counts with decay (10.33%); halving misses needs ~64 tokens of true
      future routing; 1-8 imperfect tokens are worth 2-9%; router scores for unpicked experts add nothing (10.14-10.29%).
      Left, part 2 (needs the GPU): collect a 65,000+ token trace from the streamed model and measure predictor recall
      at depths 1-4 (raw router, residual-corrected router, pre-attention linear predictor); ours reached 80-83% at
      depth 1 on ~6,000 tokens, papers report 93-97%. Kill if depth 2-3 recall does not pass 80-83%.
- [ ] 6.2 (D1) Forecast paging in the streamer, now prefetch only: issue reads from a depth 2-4 forecast so the GPU does
      not stall on them (about 14 ms of a ~48 ms step is read wait; ceiling about +40%); the real router always decides
      what runs (exact). Reads per token do not fall (1zn), so the energy gain is from a shorter step, not fewer reads.
      Gate: 6.1 part 2.
- [ ] 6.3 (D8) Optimistic expert indirection: the expert-to-slot table in a GPU buffer, no host round trip on all-hit
      layers, replay on a miss. At most 8-12% of a streamed step; only after 6.2 lowers misses. First: re-run the
      no-sync replay harness for the ceiling; drop if under ~2 ms.
- [ ] 6.4 (D3) Priced speculation in the scheduler: multi-token items with rollback (and recurrent-state checkpoints);
      draft depth chosen per step from a cost table in joules that counts the extra expert rows and SSD misses a draft
      pulls in; a model-free drafter (suffix / prompt lookup) for models with no draft head; exact for greedy, sampling
      to verify by seed. Measured outside the engine today: 1.8-2.1x on Qwen3.5-9B chat. First: offline test of
      expert-aware depth rules on logged MTP runs; real chat prompts, not random tokens.

**C. Make the claims safe (report: measurement leads)**
- [ ] 6.5 (D2) Energy leaderboard: whole-machine joules per token above idle, interleaved rounds, 95% intervals, every
      engine, every headline model (harness in place since suite v15; 0.16); SSD read energy for streamed models (D16).
      No Mac engine publishes this.
- [ ] 6.6 (D6) Real prompts and a fidelity column: a benchmark grid on real chat / code / agent prompts next to the
      random-token grid (streamed decode varies 19.76-32.20 tok/s by seed; random tokens distort routing and inflate
      speculative gains by ~23%), with output agreement against the reference in every row.
- [ ] 6.7 (D11) Attribution: per-part step traces for OLMoE, DeepSeek-V2-Lite and Qwen3-30B (none exist), and three
      variance tests, including a sweep of MLX's ops-per-command-buffer limit (40 on base chips), a candidate cause of
      the unexplained 1-8% small-model gaps.

**D. Fastest, exact, everywhere (report: catch-ups)**
- [ ] 6.8 Close the measured losses: short-context decode vs vllm-mlx on DeepSeek-V2-Lite (1.07-1.08x behind), gpt-oss
      pp512 (llama.cpp 1.06x ahead), small-model prefill (1-5% behind). (D9) Run-length-aware expert tiles in prefill:
      about +9-10% gpt-oss prefill (estimate); output must be checked bitwise. Uses 6.7's traces. With 3.1, 3.2, 3.5.
- [ ] 6.9 (D4) State-checkpoint prefix cache for recurrent / hybrid models (Qwen3.5 / 3.6 have none; 512-token cold
      streamed prefill 4.85 s): seconds of time to first token per repeated prefix. Also needed by 6.4 and the server
      (1.1b).
- [ ] 6.10 (D7) Phase-aware streamed prefill: 11.6 GB read per cold 512-token prefill at 1.7-2.2 of 3.2 GB/s; up to
      1.5-1.9x on the SSD-bound part (estimate).
- [~] 6.11 (D5) MLA decode at long context: the kernel is written and timed (0.7c): slower at 2k / 8k, tied at 16k. Next:
      profile with GPU clock and DRAM counters.
- [ ] 6.2b Uneven expert slots per layer (RESULTS 1zt): offline, about 7% fewer misses at 96 average slots, the same
      memory; layers 0-4 want ~130-165 slots, middle layers 72-100; stable across halves of the trace (r = 0.97). Needs
      a per-layer slot list in the slabs and the C++ streamer, a profile stored with the expert files, then a paired
      decode A/B at equal memory. Combine with decayed-LFU eviction (1zn) after.
      2026-11-23: live (RESULTS 1zv, 1zy), 6 ABAB rounds paired: +7.3% decode at d0 (95% CI +5.1 to +9.5, 6 of 6),
      +4.6% at d2048 (+1.0 to +8.2, 5 of 6), output unchanged. Ahead by a paired test whose rule (BENCH_PROTOCOL D)
      was set after the data. Fresh A/B (v17e, rounds 7-12, rule fixed beforehand): d2048 confirmed (+7.1%, CI
      +3.1 to +11.1, 5 of 5 valid rounds; also non-overlapping pooled intervals); d0 confirmed with round 12's
      flagged block excluded (+6.6%, +1.7 to +11.6), not if it is kept. Offline (1zze): a profile from the first
      512 tokens of in-distribution routing gives -6.4% held-out misses (the full half -6.9%; 64 tokens +10.6%);
      calibration on CALIB_TEXT not yet measured (one model load). Left: record it
      during the first 512+ prefill tokens of a store without a profile, so it can be the default.
- [ ] 6.12 One wired-memory budget (ours; supports "never killed, never swaps"): weights slots, KV and temporaries under
      one limit sized from the machine (three 16k workers killed at 74-77% of wired memory on 2026-11-21; rivals
      swapping 8-17 GB per block). With 4.2.

**E. Opt-in modes that change quality (labelled, never the headline)**
- [ ] 6.13 (D12) 8-bit KV with attention computed in 8-bit: 2x KV capacity; needs the long-context accuracy gate (0.15).
      With 3.3.
- [ ] 6.14 (D17) Quality-per-byte modes: llama.cpp's 4-bit file is better than the mlx-community one (KL 0.081 vs
      0.159, RESULTS 1ze); mixed precision per expert, 3-bit-class experts, bit-sliced storage. With 0.3, 2.3.

**F. Hardware probes (report: hardware levers)**
- [ ] 6.15 (D13) Paced decode and Low Power Mode energy probe: one cheap run (hours).
- [ ] 6.16 (D10) M5 matrix units through Metal 4 tensor ops: blocked on an M5 (4.1). (D14) SME2 co-execution in
      prefill: 1.18-1.31x reported on dense models; numerics change. (D15) Neural Engine spike: Phase 5. (D16) SSD read
      path and read-energy accounting: feeds 6.5 and 6.10.

**What the research says not to do** (report section; do not re-propose without new evidence)
- Speculative decoding as a default for MoE at one request: verification costs 2-3x on MoE, two Apple-silicon reports
  found it break-even or worse; ours pays only above ~0.58 acceptance in energy (RESULTS 1l). Hence 6.4's pricing.
- Memory-mapped expert buffers on Metal (three outside sources agree with RESULTS 1j / our mmap findings).
- Our own closed experiments (about fifty, listed in `research_notes/.../internal_review.md` section 8): wider top-(k+N)
  prefetch, depth-3 raw-router lookahead, eviction from history alone, pipelined verify, Hadamard-rotated 3-bit experts,
  entropy-coded 4-bit codes, OS read-ahead / madvise, GPU polling of CPU state, a grouped MXFP4 GEMV kernel.
- Routing changes or expert skipping in any headline number (Edge0, CECAR, cache-biased routing): they trade quality.
