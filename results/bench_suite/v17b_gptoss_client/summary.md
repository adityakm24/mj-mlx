# Benchmark suite summary

Protocol: results/BENCH_PROTOCOL.md. tok/s: mean +- sd [95% CI], CV, n (timed reps, flagged blocks excluded). Energy: IOReport GPU rail over the test window, estimated (SoC rails); none for llama-bench (its window includes model load).

## gpt-oss-20b

Weights: mj: openai/gpt-oss-20b exact (MXFP4 experts memory-mapped, packed bf16 attention, experts mlocked, 512-token prefill chunks as llama.cpp's ubatch); mj-a8: the same with 8-bit affine attention weights (lossy, matched to the GGUF); llama.cpp / Ollama / LM Studio (its llama.cpp engine): ggml-org MXFP4 GGUF (Q8_0 attention)

| engine | test | tok/s | 95% CI | CV | n | GPU mJ/token | machine mJ/token above idle | machine, steady state | swap-out MB in window (mean / max) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama-server | pp512 | 394.02 +- 8.86 | 387.20-400.84 | 2.2% | 9 | 41 | 51 | - | 4 / 36 |
| llama.cpp | pp512 | 440.66 +- 10.84 | 432.91-448.40 | 2.5% | 10 | - | - | - | block swap rule only (no window counters) |
| llama-server | tg128@d0 | 35.96 +- 0.74 | 35.39-36.53 | 2.1% | 9 | 335 | 693 | - | 0 / 0 |
| llama.cpp | tg128@d0 | 36.78 +- 0.62 | 36.33-37.22 | 1.7% | 10 | - | - | - | block swap rule only (no window counters) |
| llama-server | tg128@d2048 | 35.98 +- 0.12 | 35.89-36.07 | 0.3% | 9 | 362 | 804 | - | 8 / 63 |
| llama.cpp | tg128@d2048 | 35.75 +- 0.68 | 35.27-36.24 | 1.9% | 10 | - | - | - | block swap rule only (no window counters) |

Window swap flags (swap-out > 64 MB in the timed window): gptoss_r1_llama-server.jsonl: 2 of 15 reps; gptoss_r2_llama-server.jsonl: 1 of 15 reps

Flagged reps and blocks (excluded above; block reasons in blocks/*.flagged; a rep is also flagged when its timed window swapped out more than 64 MB, and a block when more than half its reps are):

| engine | test | tok/s (flagged reps) | n |
| --- | --- | --- | --- |
| llama-server | pp512 | 395.38 +- 0.00 (395.38-395.38) | 1 |
| llama-server | tg128@d0 | 34.49 +- 0.00 (34.49-34.49) | 1 |
| llama-server | tg128@d2048 | 35.75 +- 0.00 (35.75-35.75) | 1 |

Peak memory per block, GB (footprint: macOS's figure, GPU buffers included, mapped files not; resident: pages of the process in RAM that its page tables map, which counts a mapped weight file only where the CPU touched it: a file the GPU reads through a no-copy buffer, as llama.cpp's Metal backend does, can read near 0 or near its full size depending on how it was paged in, so resident is not comparable across engines); swap-ins / -outs over the whole block (load included):

| block | engine | peak footprint | peak resident | swap-in GB | swap-out GB |
| --- | --- | --- | --- | --- | --- |
| r1_llama-server | llama-server | 0.91 | 10.78 | 0.60 | 0.71 |
| r2_llama-server | llama-server | 0.91 | 11.96 | 0.65 | 0.76 |
| r1_llama.cpp | llama.cpp | 0.35 | 11.97 | - | - |
| r2_llama.cpp | llama.cpp | 0.35 | 12.01 | - | - |
