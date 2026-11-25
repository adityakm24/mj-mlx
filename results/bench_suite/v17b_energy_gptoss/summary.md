# Benchmark suite summary

Protocol: results/BENCH_PROTOCOL.md. tok/s: mean +- sd [95% CI], CV, n (timed reps, flagged blocks excluded). Energy: IOReport GPU rail over the test window, estimated (SoC rails); none for llama-bench (its window includes model load).

## gpt-oss-20b

Weights: mj: openai/gpt-oss-20b exact (MXFP4 experts memory-mapped, packed bf16 attention, experts mlocked, 512-token prefill chunks as llama.cpp's ubatch); mj-a8: the same with 8-bit affine attention weights (lossy, matched to the GGUF); llama.cpp / Ollama / LM Studio (its llama.cpp engine): ggml-org MXFP4 GGUF (Q8_0 attention)

| engine | test | tok/s | 95% CI | CV | n | GPU mJ/token | machine mJ/token above idle | machine, steady state | swap-out MB in window (mean / max) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama-server | tg1024@d0 | 36.58 +- 0.07 | 35.94-37.23 | 0.2% | 2 | 361 | 766 | 772 +- 1 | 30 / 60 |
| mj-a8w | tg1024@d0 | 39.82 +- 0.33 | 39.47-40.17 | 0.8% | 6 | 262 | 554 | 561 +- 41 | 5 / 18 |

Window swap flags (swap-out > 64 MB in the timed window): gptoss_r1_llama-server.jsonl: 1 of 3 reps; gptoss_r2_llama-server.jsonl: 1 of 3 reps

Flagged reps and blocks (excluded above; block reasons in blocks/*.flagged; a rep is also flagged when its timed window swapped out more than 64 MB, and a block when more than half its reps are):

| engine | test | tok/s (flagged reps) | n |
| --- | --- | --- | --- |
| llama-server | tg1024@d0 | 36.30 +- 0.24 (35.98-36.57) | 4 |

Peak memory per block, GB (footprint: macOS's figure, GPU buffers included, mapped files not; resident: pages of the process in RAM that its page tables map, which counts a mapped weight file only where the CPU touched it: a file the GPU reads through a no-copy buffer, as llama.cpp's Metal backend does, can read near 0 or near its full size depending on how it was paged in, so resident is not comparable across engines); swap-ins / -outs over the whole block (load included):

| block | engine | peak footprint | peak resident | swap-in GB | swap-out GB |
| --- | --- | --- | --- | --- | --- |
| r1_llama-server | llama-server | 0.50 | 1.00 | 0.71 | 0.66 |
| r1_mj-a8w | mj-a8w | 3.28 | 10.30 | 1.57 | 2.16 |
| r2_llama-server | llama-server | 0.50 | 11.47 | 0.64 | 0.65 |
| r2_mj-a8w | mj-a8w | 3.28 | 10.34 | 1.43 | 2.17 |
