# Changelog

## 0.1.0.dev0 (unreleased)

- Public beta prep (2026-11-25): README rewritten for new users (beta status, quick start, latest measured results,
  known limitations), ROADMAP.md, CONTRIBUTING.md and GitHub issue templates; local paths removed from results.
- mj-mlx becomes its own project and repository, with the same goals as MilliJoule but for the Mac. It was imported
  from MilliJoule at commit 519af45 (2026-11-21); the history before that stays in the MilliJoule repository.
- The step scheduler (Rust) is forked from MilliJoule's engine core and now lives here as `mjx-core`, with Python
  bindings `mj_mlx._sched` and the wrapper `mj_mlx.scheduler`. mj-mlx no longer depends on the `millijoule` package.
- Tests, benchmark tools and benchmark results moved in (`tests/`, `tools/`, `results/`).
- Licensed under Apache-2.0; NOTICE lists the code adapted from mlx-lm, MLX and transformers.
- Docs: install, usage, models, benchmarks, energy, internals, MoE design.
- Prefill attention in query blocks for head sizes MLX's fused kernel does not take (DeepSeek-V2 MLA, Qwen3.5 / 3.6):
  the score matrix no longer grows with chunk x context (`MJ_ATTN_SCORES_MB`, default 64 MB). DeepSeek-V2-Lite runs
  16k prompts at the default chunk on 16 GB (RESULTS 1zj, 1zm).
- Scheduler `prefill_area_cap` (Engine `prefill_area_cap`, `MJ_PREFILL_AREA_CAP`): the prefill chunk shrinks as the
  context grows. Off by default.
- Memory-mapped experts wire 1000 MB by default (was 3000; RESULTS 1zk).
- MLA decode kernel (`MJ_MLA_DECODE=kernel`): reads the paged latent rows in place instead of a gather per token.
- Tools: cold start (`mlx_cold_start.py`), MMLU / GSM8K subsets (`mlx_lm_eval.py`), real-chat check
  (`mlx_chat_check.py`), per-engine peak memory and swap per block, a `llama-server` engine and rival versions in the
  suite.
- `tools/reproduce.py` and `results/published.json`: one command re-runs every published number and reports which
  still hold.
- `tools/mlx_accuracy_ppl.py` reads the committed `results/accuracy/wiki.test.raw` (pyarrow no longer needed).
