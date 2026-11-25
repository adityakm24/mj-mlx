# How the numbers are measured

Every published number comes from a run on this repository's tools, and the run's raw files are kept under
`results/`. The full protocol is [`results/BENCH_PROTOCOL.md`](../results/BENCH_PROTOCOL.md), and
every run is written up in [`results/RESULTS.md`](../results/RESULTS.md).

## Measurement method: what is standard and what is ours

Results are only as good as the tools behind them. Rule since 2026-11-22: every part of a measurement follows a named
industry method where one exists, every tool is checked against an independent measurement before its numbers are
trusted, and every departure from the standard is written next to the number it affects. Where nothing standard
exists, the method is labelled **our own**.

| Part | Standard followed (source) | What we do differently today | Checked against an independent source? |
| --- | --- | --- | --- |
| Speed tests | llama.cpp's `llama-bench` definitions of `pp` (prefill) and `tg` (decode) tests, with random-token prompts | Nothing in the definitions. Real prompts and serving metrics (time to first token and time per output token, measured through an OpenAI-compatible client as vLLM's serving benchmark does) are not in the grid yet (checklist 6.6) | No |
| Client | One client and one set of prompts for every engine (MLPerf Inference's LoadGen; vLLM's serving benchmark) | **Departure.** Three timing paths: mj-mlx and mlx-lm in-process (`tools/mlx_bench_worker.py`); llama.cpp by `llama-bench`'s own timer; vllm-mlx, Ollama and LM Studio through their HTTP servers. HTTP adds per-token overhead to those rivals only, which can favour us. One client for all engines needs our own server (checklist 1.1) | Not yet. Timed both ways on Qwen3.6-35B in suite v16 (RESULTS 1zu), but that check has no power: every block of both is flagged for swap, the spread is about 1 tok/s at about 4 tok/s, and a per-token HTTP cost would not show at that rate. On gpt-oss-20b (RESULTS 1zzb, about 36 tok/s, fits in memory, 2 ABAB rounds): decode through llama-server and our client is within 2.2% of `llama-bench` at d0 and equal at d2048, so the HTTP path costs the rivals at most about 2% in decode; prefill through the server is 10.6% lower (request, tokenizing and scheduling inside the window), which is why llama.cpp's prefill rows come from `llama-bench`. Checked on one model and one rival server only |
| Warm-up and steady state | Untimed warm-up; long enough runs to reach steady state (MLPerf Inference runs for at least 10 minutes by default) | One untimed warm-up per test, then 5 reps of 128 decode tokens. **No steady-state or thermal check** yet: no sustained runs (checklist 4.5) | No |
| Order and spread | Interleaved runs so drift falls on every engine | ABAB rounds with a fresh process per block; mean with a 95% confidence interval; a lead only when intervals do not overlap; blocks with more than 256 MB of swap growth, and reps whose timed window swaps out more than 64 MB, are left out. The thresholds are **our own** | n/a |
| Rival settings | Each rival at its own recommended settings, plus a separately labelled tuned row | Recommended or default settings, listed below. **No tuned row yet**: the sweeps in checklist 0.9 are not done, so a speed claim holds at the listed settings only | n/a |
| Energy | An external power analyser at the wall (MLPerf Power, SPECpower) | **Departure.** IOReport "GPU Energy": the chip's own power model, GPU rail only, no sudo (`docs/energy.md`). Whole-machine power is published since RESULTS 1zw: the SMC's own estimate of system power (PSTR), idle-subtracted (a 10 s baseline with the model loaded), steady state over 1,024-token decodes with the first 3 s dropped. It is an estimate by the machine itself, not a wall reading | No. Neither the GPU rail nor PSTR has been checked against an independent meter. Planned: the GPU rail against `powermetrics` (Apple's tool; needs sudo, so the user runs it) on the same window, and PSTR against a wall meter if one is available |
| Memory | `phys_footprint`, the figure macOS itself uses (Activity Monitor's "Memory", the jetsam limit) | Peak over the block, sampled from outside the process (`tools/mlx_proc_mem.py`), plus swap-ins and swap-outs over the block and, since 288745b, inside each timed window. "Peak resident" is not comparable across engines: a memory-mapped file read by the GPU through a no-copy buffer (llama.cpp's Metal backend) counts only where the CPU touched it (RESULTS 1zw) | No. Planned: compare with Apple's `footprint` tool on one run |
| Accuracy | WikiText-2 perplexity and KL divergence against bf16, the `llama-perplexity` way (llama.cpp); MMLU 5-shot and GSM8K 5-shot as in EleutherAI's lm-evaluation-harness | Perplexity and KL as in llama.cpp. MMLU and GSM8K run on **our own harness** (`tools/mlx_lm_eval.py`) with the same prompts on every engine, on **seeded subsets**, not the full sets; llama.cpp's MMLU answers are read from its top-20 next-token log-probabilities | Partly: our MMLU / GSM8K scores match mlx-lm's on the same weights (RESULTS 1zr). Not yet compared with lm-evaluation-harness itself |
| Profiling | Xcode Instruments (Metal System Trace) for GPU timelines | **Our own** step traces: `mx.async_eval` at each matmul's input and output, the same wrapping for our engine and mlx-lm (`tools/mlx_step_trace_lib.py`). Forcing evaluation points can change the timing it measures | No. Planned: one step traced both ways, Instruments against ours |

Before a new campaign, the plan names the standard each part follows, and runs the planned cross-checks above for any
tool whose numbers the campaign will publish. A number that rests on an unchecked tool says so where it is published.

## Tests

These follow llama-bench:

| Test | What |
| --- | --- |
| `pp512`, `pp2048` | Prefill 512 or 2048 tokens. tok/s = prompt tokens / time to the first token. |
| `tg128@d0` | Decode 128 tokens after a 1-token prompt. tok/s = 128 / (time of token 129 - time of token 1). |
| `tg128@d2048` | The same after a 2048-token prompt. |

Prompts are random token ids, different for each rep and each test, so no engine's prompt cache can hit. EOS is
ignored. Each test runs one untimed warm-up, then 5 timed reps 5 s apart.

## Engines

| Engine | How it runs |
| --- | --- |
| mj-mlx | `tools/mlx_bench_worker.py --engine mj` (overlap mode) |
| mlx-lm | `tools/mlx_bench_worker.py --engine mlx-lm` (`generate_step`) |
| vllm-mlx | its OpenAI server on the mlx checkpoint, one request at a time, prefix cache off |
| llama.cpp | `llama-bench` for speed (its own timing; `-fa` on, all layers on the GPU, `-ncmoe` for models bigger than memory); `llama-server` for chat and energy |
| Ollama, LM Studio | their servers, same GGUF or MLX weights where possible |

Versions and settings for each run are in the run's `meta.json`.

## Rival settings

What each rival ran with in the runs since RESULTS 1zg (2026-11-20 and 21), and what was left at its default. Versions
were read from the installed engines on 2026-11-21; the suite's `meta.json` records llama.cpp and MLX, and from v15 on
it records the others too.

| Engine | Version | Set by the suite | Left at the default |
| --- | --- | --- | --- |
| llama.cpp (`llama-bench`) | ggml 1.0.0, build 12359 (b1b770d90), Homebrew | `-fa 1` (flash attention on), `-ngl 99` (all layers on the GPU), `-ncmoe N` for models bigger than memory (Qwen3-30B 48, Qwen3.6-35B 40), `-r` reps | `-b 2048 -ub 512`, `-t 4` threads, f16 K / V cache, mmap on |
| Ollama | 0.40.16 | `num_ctx 8192` (so the model never reloads between tests), `temperature 0`, `keep_alive 30m`, its llama.cpp runner on the same GGUF | `num_batch`, `num_thread`, flash attention and K / V cache type (Ollama's own defaults) |
| LM Studio | app 0.4.25+1; llama.cpp engine 2.80.1, MLX engine 1.14.0 | `--gpu max`, `-c 8192`, `--parallel 1` | evaluation batch size, flash attention, K / V cache type (the engine's defaults) |
| vllm-mlx | 0.6.0 (its own venv: mlx 0.32.4, mlx-lm 0.31.3) | `--disable-prefix-cache`, `--stream-interval 1`, `--max-num-seqs 1` | prefill step size, everything else |
| mlx-lm | 0.32.2 (mlx 0.32.4) | `generate_step` called directly | `prefill_step_size` 2048, no K / V quantization |

Capacity: when a rival runs out of memory at its defaults, llama.cpp is re-run with batch = ubatch 256, then 128
(the same f16 K / V cache, the same weights; only speed and memory change), stopping at the first setting that fits.
Settings that change output quality, such as a q8_0 K / V cache, are not tried, and the claim is worded "at every
quality-preserving setting tried" (RESULTS 1zi).

llama.cpp's GPU expert cache (`--moe-cache-mib`, for experts kept on the CPU) was tried for Qwen3.6-35B on the 16 GB M4
(RESULTS 1zs). It only works with the experts in host memory: with the default memory-mapped load it switches itself
off, and with `--load-mode none` the 18 GB of experts do not fit (12.7 GB of swap before the server was up). So
`-ncmoe 40` is llama.cpp's best working setting for that model on 16 GB; on a Mac with enough RAM the cache should be
compared.

Not yet tuned for speed (checklist 0.9): llama.cpp threads and ubatch at short context, Ollama `num_batch`, LM Studio's
batch size, vllm-mlx's prefill step. A speed claim against a rival holds at the settings above. It is not a claim
against that rival's best possible setting until these are swept.

## Controls

- **Which code ran.** Every result row records the git commit, whether `src/` or `tools/` had local changes, and the
  `MJ_` / `MLX_` environment. Every run's `meta.json` records the version of each engine.
- **Memory.** Each block records the peak memory of the process holding the model (footprint and resident size) and
  the swap-ins and swap-outs during the block.

- **Order:** ABAB rounds, with a fresh worker process for each block, so drift and thermal state are spread across
  engines.
- **Swap:** a block is flagged when swap grows by more than 256 MB, and only unflagged blocks enter the tables. Since 2026-11-23, also: a rep whose timed window swaps out more than 64 MB is flagged (same threshold for every engine), and a block with more than half its reps flagged that way is flagged whole (results/BENCH_PROTOCOL.md, swap rules). llama-bench reps have no window counters, so only the block rules apply to them. The
  machine state of each run (top processes, swap-ins and swap-outs every 5 s) is kept next to it.
- **Swap stop rule:** a block is killed once its swap-outs pass 20 GB (`MJ_SWAP_ABORT_GB`) and marked aborted. Such a block would be flagged anyway; past that point it only wears the SSD. Suite v16 showed why: llama-server on Qwen3.6-35B (bigger than RAM) wrote about 900 GB of swap per block (RESULTS 1zu). Engines whose memory is above RAM and that swap are not run without the user's OK.
- **Statistics:** each table cell is a mean with a 95% confidence interval. A lead is only claimed when the
  intervals do not overlap.
- **Losses stay in:** rows where mj-mlx is slower stay in the tables.
- **Quality:** a speed comparison names each engine's weight format, with WikiText-2 perplexity and KL divergence
  measured the llama.cpp `llama-perplexity` way (`tools/mlx_accuracy_ppl.py`).

## Reproduce

Benchmarks need a quiet machine: one model at a time and no other servers. On a 16 GB Mac, close other heavy apps.

Every number the README publishes is listed in [`results/published.json`](../results/published.json), with the run
behind it. One command runs them all again into a new folder (`results/repro/<date>`) and writes `report.md`: each
published number next to the new one, and whether each sentence the README makes from them ("faster than", "tied
with") still holds:

```bash
.venv/bin/python tools/reproduce.py --dry-run
```

```bash
.venv/bin/python tools/reproduce.py
```

`--list` shows the claims, `--only <id>,<id>` runs some of them, and `--quick` (1 rep, 1 round, 3 perplexity chunks)
checks that the commands run without producing a result. The full run takes several hours. KL divergence needs the
bf16 reference log-probabilities in `~/.cache/millijoule/accuracy_refs` (about 2 GB, not in git); the OLMoE bf16
run writes them when they are missing.

To run the suite by hand:

```bash
.venv/bin/python tools/mlx_bench_suite.py --models olmoe,dsv2lite,gptoss --reps 5 --rounds 2 --out results/bench_suite/mine
```

```bash
.venv/bin/python tools/mlx_bench_suite.py --models olmoe,dsv2lite,gptoss --out results/bench_suite/mine --summary-only
```

The suite downloads nothing. Model keys (`q9b`, `gptoss`, `olmoe`, `dsv2lite`, `q30b`), paths and GGUF files are configured at the top of
`tools/mlx_bench_suite.py`.
