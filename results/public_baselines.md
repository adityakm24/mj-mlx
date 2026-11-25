# Public Apple-silicon baselines and the benchmark protocol (2026-11-16)

Collected by web research (sources below); nothing here was measured by us. All numbers are the publishers' own and
are batch 1 unless stated. Several sites publish computed (not measured) "M4 16 GB" figures; those are excluded.

| model | engine | quant | hardware | prompt tok/s | generation tok/s | source |
| --- | --- | --- | --- | --- | --- | --- |
| Llama-2 7B | llama.cpp (llama-bench pp512 / tg128, commit 8e672ef) | Q4_0 | **M4 base, 10-core GPU, 120 GB/s** | 221.3 | 24.1 | github.com/ggml-org/llama.cpp/discussions/4167 |
| gpt-oss-20b | llama.cpp, `--n-cpu-moe 12 -fa on --no-mmap` (anecdotal, not llama-bench) | MXFP4 GGUF | Mac mini M4 **16 GB** | - | ~26 | github.com/ml-explore/mlx-lm/issues/644 |
| gpt-oss-20b | mlx-lm | MXFP4-Q8 | Mac mini M4 **16 GB** | - | crashes (Metal OOM: needs 11.5 GB > 10.9 GB recommended) | same issue |
| gpt-oss-20b | oMLX v0.3.8 (MLX), 1k / 4k context | MXFP4-Q8 | M4 base 24 GB | 379 / 403 | 35.2 / 33.2 | omlx.ai/benchmarks/fcud9ll1 |
| gpt-oss-20b | oMLX v0.2.6 | MXFP4-Q8 | M4 base 32 GB | 292 / 318 | 34.5 / 32.1 | omlx.ai/benchmarks/3j6kqnsx |
| gpt-oss-20b | LM Studio 0.4.7 (MLX) | MXFP4 | Mac mini M4 24 GB | - | ~30.5 | metrillm.dev/result/eabdff3a-1142-4eb6-b98b-9668e945603b |
| gpt-oss-20b | llama-bench (`-fa 1 -b 2048 -ub 2048`) | MXFP4 | M4 Pro 64 GB (273 GB/s) | 701 (pp2048) | 63.3 | github.com/ggml-org/llama.cpp/discussions/15396 |
| Qwen3-4B | oMLX v0.2.13, 1k / 4k | MLX 4-bit | **M4 base 16 GB** | 382 / 347 | 37.7 / 30.7 | omlx.ai/benchmarks/6bw07j97 |
| Qwen3.5-35B-A3B | llama-server, CPU only (`-ngl 0 --mmap`) | UD-IQ3_XXS (~13 GB) | Mac mini M4 16 GB | - | 17.3 (second-hand) | resume-website-ds0.pages.dev/blog/running-qwen-locally-mac-mini-m4 |

No measured Qwen3.6-35B-A3B number on a 16 GB Mac was found. No measured J/token on a base M4 was found; the one
public energy study (GreenBench) reports implausible package power, and an Apple forum thread shows powermetrics'
GPU figure covering ~36% of the SMC system power rise on an M4 Max, so GPU-rail energy undercounts badly.

## Measured on this Mac (M4, 16 GB), real gpt-oss-20b, 2026-11-16

`llama-bench -m gpt-oss-20b-MXFP4.gguf -p 512 -n 128 -r 3 -fa 1 -ncmoe N` (Homebrew llama.cpp 1.0.0, build 12359,
commit b1b770d90; ggml-org/gpt-oss-20b-GGUF, 12.1 GB; files in `results/llamacpp/`), and ours with the same
prompt / generation lengths (`tools/mlx_stream_run.py --prompt-tokens 512 --new 128 --prefetch`). Swap was 8-11 GB in
use during all runs (other apps).

| engine | placement | pp512 tok/s | tg128 tok/s |
| --- | --- | --- | --- |
| llama.cpp, `-ncmoe 0` | whole GGUF memory-mapped and used by the GPU in place (page-cache backed, not wired) | 304 +- 97 | **26.7 +- 0.1** |
| llama.cpp, `-ncmoe 6` | 6 layers' experts on the CPU | 57 +- 2 | 9.6 +- 3.1 |
| llama.cpp, `-ncmoe 12` | 12 layers' experts on the CPU (the public 16 GB recipe) | 39 +- 2 | 7.4 +- 0.8 |
| MilliJoule MLX, 24 of 32 expert slots | SSD streaming, 9.2 GB resident | 132 cold (first request, TTFT 3.9 s); **288 warm** (pp2048 285) | 14.1 (17.2 with speculative dispatch, RESULTS 1d.3) |
| MilliJoule MLX, 16 slots | 6.7 GB resident | 129 | 8.2 |
| mlx-lm 0.32.2 | everything allocated | - | does not run (TTFT 105 s, then swap; public reports: Metal OOM) |
| MilliJoule MLX, all 32 slots resident, no wiring | 11.8 GB allocated | - | stalled in swap, killed at 420 s |

Not the same weights: the GGUF (read from its header) keeps the experts in MXFP4 like ours but stores attention
q/k/v/o, the output head and the embedding as **Q8_0** (8-bit); only norms, biases, sinks and the router are F32. Ours
runs attention in the checkpoint's bf16 (2x the attention bytes per token, +0.6 GB resident) and picks tokens from a
4-bit candidate head with a float32 rescore of the original bf16 rows. llama.cpp's output is therefore not the
reference model's output; ours matches transformers (section 2a of RESULTS.md).

Prompt processing, like for like (warm, as llama-bench reports): ours 288 tok/s at 512 tokens via
`tools/mlx_ttft.py --experts ... --slots 24` (median of 3, `results/ttft/gptoss20b_stream24_warm.json`), measured
later the same night with the GPU in a slower state (2.7 W vs 6-7 W at equal settings); llama.cpp 304 +- 97.

Reading: on a model that fits in RAM but not in the GPU working set, llama.cpp is ~1.9x faster in generation and ~2.3x
in prompt processing than our streaming. Its edge is memory-mapped, page-cache-backed GPU buffers, which the OS can
page without wiring; MLX allocates its own buffers (no API to wrap an mmap'd file without a copy), so we stream
instead. The public "--n-cpu-moe 12, ~26 tok/s" anecdote did not reproduce here (7.4 tok/s with that flag); the
all-GPU mmap run gave the ~26.7.

## Same-session re-measurement, 2026-11-17 03:53 IST (GPU in its normal state)

llama-bench before and after ours (`results/llamacpp_v2/`), same flags as above (`-ncmoe 0`); ours with
speculative dispatch (RESULTS 1d.3), 24 of 32 expert slots, 9.24 GB resident, 512-token random prompt:

| engine | pp512 tok/s | tg128 tok/s |
| --- | --- | --- |
| llama.cpp, run A | 394.5 +- 29.9 | 34.8 +- 1.0 |
| MilliJoule MLX (exact attention) | 362 (warm TTFT 1.41 s, median of 3) | 18.9 |
| llama.cpp, run B | 424.9 +- 6.4 | 35.0 +- 0.5 |

Prompt processing is within 8-15% of llama.cpp; generation is 1.85x behind (llama.cpp's attention and head are
Q8_0, ours bf16 / fp32-rescored; its whole 12.1 GB file is GPU-visible through mmap, ours streams 8 of 32 experts
per layer from the SSD). Swap rose from 8.0 to 10.6 GB over the session (other apps + llama.cpp's mapping).

## What this means for our results

- gpt-oss-20b on this 16 GB Mac (measured above): llama.cpp 26.7 tok/s vs our 17.2 (was 14.1). The model fits in RAM; only the
  GPU working set is short, and mmap-backed buffers get around that. MLX's CPU path is no help (an MXFP4 MoE layer on
  MLX's CPU stream: 262 ms vs 0.96 ms on the GPU). Options: the user raises `iogpu.wired_limit_mb` (a system setting,
  theirs to change), or MLX gains file-backed no-copy arrays (a change to MLX itself).
- SSD streaming is the right tool for models bigger than RAM (Qwen3.6-35B-A3B, 20.4 GB: 4.5-7.4 tok/s in 3.7-8.2 GB).
- MLX on base M4 with enough memory runs gpt-oss-20b at ~32-35 tok/s generation, ~300-400 tok/s prompt.

## Protocol for our comparisons

- Same model weights or the closest format per engine (state it: MXFP4 GGUF vs MLX MXFP4 / 4-bit).
- Single-stream: prompt 512 and 2048 tokens (TTFT / prompt tok/s, warm, median of 3-5), generation 128 tokens
  (tok/s), as llama-bench pp512 / tg128 and `mlx_lm.benchmark -p 512 -g 128`.
- Concurrency (ours vs mlx-lm only, llama.cpp server optional): tools/mlx_bench.py levels 1 / 8 / 32 / 64, ABAB rounds.
- Same session: AC power, Low Power Mode off, other apps closed, swap recorded (`sysctl vm.swapusage`), idle GPU power
  recorded per run (fast/slow machine state), no rebuilds during a run, engine versions recorded.
- Energy: GPU rail only, labelled; whole-system power needs root (user's call).

Sources: github.com/ggml-org/llama.cpp/discussions/4167, github.com/ggml-org/llama.cpp/discussions/15396,
github.com/ml-explore/mlx-lm/issues/644, omlx.ai benchmark pages cited above, metrillm.dev result cited above,
gist.github.com/awni/c1790e4c3a39be6e8f1c4afd42423d2d (mlx_lm.benchmark usage), arxiv.org/html/2608.28667v1
(GreenBench), Apple Developer Forums thread 819927.
