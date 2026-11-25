# Benchmark suite summary

Protocol: results/BENCH_PROTOCOL.md. tok/s: mean +- sd [95% CI], CV, n (timed reps, flagged blocks excluded). Energy: IOReport GPU rail over the test window, estimated (SoC rails); none for llama-bench (its window includes model load).

## gpt-oss-20b

Weights: mj: openai/gpt-oss-20b exact (MXFP4 experts memory-mapped, packed bf16 attention, experts mlocked, 512-token prefill chunks as llama.cpp's ubatch); mj-a8: the same with 8-bit affine attention weights (lossy, matched to the GGUF); llama.cpp / Ollama / LM Studio (its llama.cpp engine): ggml-org MXFP4 GGUF (Q8_0 attention)

| engine | test | tok/s | 95% CI | CV | n | GPU mJ/token | machine mJ/token above idle | machine, steady state | swap-out MB in window (mean / max) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mj-a8wb | pp512 | 413.93 +- 4.33 | 408.55-419.31 | 1.0% | 5 | 45 | 55 | - | 0 / 0 |
| mj-a8wb16 | pp512 | 354.23 +- 86.98 | 292.07-416.40 | 24.6% | 10 | 45 | 64 | - | 16 / 53 |
| mj-a8wb24 | pp512 | 380.51 +- 68.85 | 327.49-433.52 | 18.1% | 9 | 45 | 59 | - | 8 / 44 |
| mj-a8wb | tg128@d0 | 39.84 +- 0.52 | 39.19-40.49 | 1.3% | 5 | 245 | 497 | - | 0 / 0 |
| mj-a8wb16 | tg128@d0 | 40.08 +- 0.46 | 39.80-40.35 | 1.2% | 13 | 245 | 495 | - | 0 / 0 |
| mj-a8wb24 | tg128@d0 | 40.11 +- 0.42 | 39.80-40.41 | 1.1% | 10 | 245 | 507 | - | 0 / 0 |
| mj-a8wb | tg128@d2048 | 37.75 +- 2.01 | 34.54-40.95 | 5.3% | 4 | 273 | 696 | - | 0 / 0 |
| mj-a8wb16 | tg128@d2048 | 37.42 +- 4.51 | 34.93-39.92 | 12.1% | 15 | 269 | 646 | - | 1 / 12 |
| mj-a8wb24 | tg128@d2048 | 38.64 +- 0.17 | 38.51-38.76 | 0.4% | 10 | 269 | 677 | - | 0 / 0 |

Window swap flags (swap-out > 64 MB in the timed window): gptoss_r1_mj-a8wb.jsonl: 4 of 15 reps; gptoss_r2_mj-a8wb16.jsonl: 3 of 15 reps; gptoss_r2_mj-a8wb24.jsonl: 1 of 15 reps; gptoss_r3_mj-a8wb.jsonl: 1 of 15 reps; gptoss_r3_mj-a8wb16.jsonl: 4 of 15 reps; gptoss_r3_mj-a8wb24.jsonl: 4 of 15 reps

Flagged reps and blocks (excluded above; block reasons in blocks/*.flagged; a rep is also flagged when its timed window swapped out more than 64 MB, and a block when more than half its reps are):

| engine | test | tok/s (flagged reps) | n |
| --- | --- | --- | --- |
| mj-a8wb | pp512 | 410.75 +- 11.75 (385.04-420.55) | 10 |
| mj-a8wb16 | pp512 | 123.73 +- 43.70 (82.09-192.24) | 5 |
| mj-a8wb24 | pp512 | 294.50 +- 136.93 (103.62-411.14) | 6 |
| mj-a8wb | tg128@d0 | 40.01 +- 0.57 (38.64-40.40) | 10 |
| mj-a8wb16 | tg128@d0 | 22.58 +- 6.44 (18.02-27.13) | 2 |
| mj-a8wb24 | tg128@d0 | 32.60 +- 7.08 (24.76-39.11) | 5 |
| mj-a8wb | tg128@d2048 | 38.16 +- 1.58 (33.54-38.89) | 11 |
| mj-a8wb24 | tg128@d2048 | 36.60 +- 3.80 (29.91-38.76) | 5 |

Peak memory per block, GB (footprint: macOS's figure, GPU buffers included, mapped files not; resident: pages of the process in RAM that its page tables map, which counts a mapped weight file only where the CPU touched it: a file the GPU reads through a no-copy buffer, as llama.cpp's Metal backend does, can read near 0 or near its full size depending on how it was paged in, so resident is not comparable across engines); swap-ins / -outs over the whole block (load included):

| block | engine | peak footprint | peak resident | swap-in GB | swap-out GB |
| --- | --- | --- | --- | --- | --- |
| r1_mj-a8wb16 | mj-a8wb16 | 2.38 | 5.24 | 0.79 | 0.71 |
| r1_mj-a8wb24 | mj-a8wb24 | 2.38 | 7.78 | 0.75 | 0.84 |
| r1_mj-a8wb | mj-a8wb | 2.38 | 10.34 | 2.14 | 2.40 |
| r2_mj-a8wb16 | mj-a8wb16 | 2.38 | 5.49 | 0.57 | 0.53 |
| r2_mj-a8wb24 | mj-a8wb24 | 2.39 | 7.77 | 1.34 | 1.17 |
| r2_mj-a8wb | mj-a8wb | 2.36 | 10.33 | 1.00 | 1.17 |
| r3_mj-a8wb16 | mj-a8wb16 | 2.36 | 5.22 | 5.04 | 4.93 |
| r3_mj-a8wb24 | mj-a8wb24 | 2.38 | 7.93 | 4.52 | 4.77 |
| r3_mj-a8wb | mj-a8wb | 2.38 | 10.31 | 4.66 | 4.93 |
