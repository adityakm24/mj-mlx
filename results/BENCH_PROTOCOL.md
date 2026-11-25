# MLX benchmark protocol (adopted 2026-11-18)

Built from the industry conventions: llama-bench / llama.cpp discussion #4167 for micro tests; vLLM `bench serve`
and MLPerf LoadGen for per-request latency; MLPerf Power for energy; MLPerf Client v2.0 for the Mac chat categories.
Where they disagree, the formula is printed next to the name. Results that do not follow this file say so.

## A. Metrics (formula next to every name)

| name | definition |
| --- | --- |
| TTFT | `generate()` call (or request send) to the first output token available to the caller: tokenization + prefill included |
| TPOT | (E2E - TTFT) / (N_out - 1), per request (vLLM, MLPerf); first token excluded |
| decode tok/s | 1 / TPOT (per stream) |
| ITL | distribution of gaps between consecutive emitted tokens; a k-token chunk (speculative decoding) counts as k gaps of gap/k |
| E2E | call to last token |
| prefill tok/s | N_in / prefill-only time (micro, llama-bench `pp`); N_in / TTFT for served runs, labeled |
| output throughput | sum N_out / (last response - first request) (batched or served runs) |
| goodput | requests meeting all SLOs / window. SLOs: interactive TTFT <= 500 ms, TPOT <= 30 ms; conversational 2000 / 100 ms |

- llama-bench `tg128` is 128 decodes from an empty context, not decode after a prompt. Compare it only with our
  depth-0 decode, or use `-d <depth>` / `-pg` on both sides.
- mlx-lm's `generation_tps` counts (n + 1) tokens over the time after the first token, about 1/N high (unverified
  detail). Quote engine-native numbers as such, or re-time all engines with one client.

## B. Workload grid

1. Micro, batch 1, random tokens (fixed seed), EOS ignored: pp512, pp2048, tg128, tg128 at depth 2048, pg512+128.
   - llama.cpp: `llama-bench -p 512,2048 -n 128 -d 0,2048 -r 10`.
   - mlx-lm: `mlx_lm.benchmark -p ... -g 128` (its default is `-g 1024`).
2. Chat set: fixed prompts and a fixed output length with EOS ignored, plus MLPerf Client v2.0 categories 128->256,
   512->512 and 1856->128 tokens.
3. Batched or served: concurrency 1 / 2 / 4 / 8, at least 200 requests per point, with each server's parallel slots
   set explicitly.
4. Models bigger than RAM (streamed MoE): the same grid; report the expert hit rate and bytes read per token alongside.

## C. Run protocol

- Untimed warm-up after load. Then 10 timed reps for micro tests (5 is the llama-bench default; this machine is
  noisy).
- Interleave engines or arms A/B/A/B; never one block per engine. Leave 30 s cool-down between tests (`--delay 30`,
  the #4167 norm).
- Clear KV / prefix caches between reps, unless caching is the thing being tested.
- `tools/mlx_machine_state.py` before and after each block: swap, memory pressure, thermal state, top processes.
  - A rep with swap growth or non-nominal thermal pressure is invalid; flag it or drop it.
  - **Swap rules (fixed 2026-11-23, agreed with the PM before the first run that uses them, v17b `energy2`):**
    - Block: flagged when the swap file grows by more than 256 MB over the block (as before), or when its swap-outs pass
      20 GB, which also stops the block (SwapGuard).
    - Timed window: every rep records the machine's swap-ins and swap-outs inside its own timed window. A rep whose
      window swaps out more than **64 MB** is flagged and left out of the tables, with its value kept in the flagged
      table. The threshold is the same for every engine and model.
    - A block with more than half its reps flagged by the window rule is flagged as a whole.
    - The per-block swap-in / swap-out table stays in every summary, so paging during load stays visible.
    - llama-bench reps have no window counters (its own timer), so only the block rules apply to them; llama.cpp's
      server (`llama-server`) is timed through our client and gets the window rule.
- Machine: on AC power, battery full, Spotlight and Time Machine idle, only the benchmark plus the sampler running,
  `caffeinate -i`. State the Energy Mode.
- Sustained throughput is reported separately: a 10-minute tg128 loop.

## D. Statistics

- Micro: mean +- sample sd (llama-bench compatible), plus median, 95% CI (t or bootstrap), CV and n.
  - CV > 5%: re-run.
- Per-request metrics: mean, median, p90, p99 (p99 only with >= 200 samples).
- A difference is claimed only when the 95% CIs do not overlap. This rule holds for every engine-vs-engine
  comparison.
- Exception, written 2026-11-23 (after the six uneven-slot rounds of RESULTS 1zy, so those rounds do not confirm
  it): two settings of our own engine, run as ABAB rounds on the same code and machine state. Pair the blocks by
  round (mean of each block's timed reps) and take the t-interval over the per-round ratios. A lead is claimed when
  that paired 95% CI excludes 0 and at least 5 of 6 rounds (or the same share of more rounds) go the same way. A
  result first found with this rule needs a fresh A/B (new rounds, rule fixed beforehand) to be called confirmed.
  A flagged block (swap or failure) drops its whole round from the paired test; the result then reports how many
  rounds were left, the result with the round kept, and whether the flagged arm swaps more than the other overall.
- Never report a best-of-N run.
- Publish every per-rep sample (JSON/JSONL next to the summary).

## E. Energy

- Primary (when available): wall AC meter, integrated over the window, with idle measured in the same session.
  Disclose the meter model and its accuracy. We do not have one yet.
- Today: powermetrics / IOReport GPU (and CPU + ANE) rail energy, labeled **"estimated, SoC rails"**.
  - Relative comparisons on this machine only. It excludes DRAM, SSD and conversion losses; Apple's man page warns
    against cross-device comparison.
- J/token reported as three numbers: prefill J / N_in, decode J / N_out, request J / N_out.
  - Each is gross, and also net of idle, labeled.

## F. Correctness

- Our optimizations (MTP / EAGLE, streaming, overlap, kernels): greedy output must be token-identical to our
  unoptimized path on every grid prompt. Report the identical-sequence rate.
- Across engines at the same weights: first-divergence position, top-1 agreement, KL vs a bf16 reference, and
  wikitext-2 perplexity at a fixed context.
- Any lossy change must reach >= 99% of the bf16 reference score on a small task set (MLPerf's 99% rule).

## G. Disclosure

- Hardware: M4, 10-core GPU, 16 GB, 120 GB/s, form factor, macOS build, Energy Mode.
- Software: pinned engine versions / commits.
- Model files with hashes, and bits per weight per engine. Q4_K_M (~4.8 bpw) vs MLX 4-bit g64 (~4.5 bpw) is
  disclosed, or a matched-bpw row is added.
- Settings: context size, batch / ubatch, flash attention, threads, `-ngl` / `-ncmoe`, sampler, seed.

## H. Rejected by a reviewer (do not do)

- Mismatched quantization compared as equal.
- Mixed tok/s definitions, e.g. one engine's tg against another's decode after a prompt.
- Single runs, no warm-up, best-of-N, no spread.
- Early EOS giving different lengths; cache hits across reps.
- Competitors run on untuned defaults.
- Swap or throttling during a run; unpinned versions.
- powermetrics presented as wall energy.
- Speed claims without correctness evidence; no raw data.

## Status of existing results

Most results in `RESULTS.md` before this date use 1-3 reps, one or two A/B rounds, decode = first to last token,
and GPU-rail energy. They follow sections A, F and most of C, but not D: too few reps, no CIs. The headline
comparisons are to be re-run under this protocol (`tools/mlx_bench_suite.py`, to be written) before being quoted
externally.
