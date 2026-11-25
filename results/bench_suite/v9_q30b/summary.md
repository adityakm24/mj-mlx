# Benchmark suite summary

Protocol: results/BENCH_PROTOCOL.md. tok/s: mean +- sd [95% CI], CV, n (timed reps, flagged blocks excluded). Energy: IOReport GPU rail over the test window, estimated (SoC rails); none for llama-bench (its window includes model load).

## Qwen3-30B-A3B (17.2 GB 4-bit, bigger than RAM; general MoE model)

Weights: ours: mlx-community 4-bit g64, experts placed by memory (streamed from the SSD, 64 slots/layer); llama.cpp: Qwen Q4_K_M, experts on the CPU (-ncmoe 48)

| engine | test | tok/s | 95% CI | CV | n | GPU mJ/token |
| --- | --- | --- | --- | --- | --- | --- |
| llama.cpp | pp512 | 15.57 +- 0.40 | 14.58-16.57 | 2.6% | 3 | - |
| mj | pp512 | 188.13 +- 6.68 | 179.83-196.43 | 3.5% | 5 | 18 |
| llama.cpp | pp2048 | 13.35 +- 0.31 | 12.56-14.13 | 2.4% | 3 | - |
| mj | pp2048 | 291.75 +- 3.52 | 287.38-296.13 | 1.2% | 5 | 39 |
| llama.cpp | tg128@d0 | 2.71 +- 0.45 | 1.60-3.81 | 16.5% | 3 | - |
| mj | tg128@d0 | 27.71 +- 4.73 | 21.82-33.60 | 17.1% | 5 | 224 |
| llama.cpp | tg128@d2048 | 3.57 +- 0.59 | 2.11-5.03 | 16.4% | 3 | - |
| mj | tg128@d2048 | 26.24 +- 4.01 | 21.25-31.23 | 15.3% | 5 | 314 |

Flagged blocks (excluded above; reason in blocks/*.flagged):

| engine | test | tok/s (flagged reps) | n |
| --- | --- | --- | --- |
| llama.cpp | pp512 | 14.41 +- 1.02 (13.24-15.10) | 3 |
| mj | pp512 | 196.63 +- 11.08 (178.68-207.99) | 5 |
| llama.cpp | pp2048 | 13.21 +- 0.33 (12.87-13.53) | 3 |
| mj | pp2048 | 290.15 +- 3.77 (285.99-293.92) | 5 |
| llama.cpp | tg128@d0 | 2.66 +- 0.26 (2.44-2.95) | 3 |
| mj | tg128@d0 | 26.21 +- 4.92 (22.76-34.71) | 5 |
| llama.cpp | tg128@d2048 | 3.69 +- 0.65 (2.94-4.09) | 3 |
| mj | tg128@d2048 | 28.55 +- 0.11 (28.43-28.67) | 5 |
