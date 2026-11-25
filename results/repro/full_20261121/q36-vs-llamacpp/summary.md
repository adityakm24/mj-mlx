# Benchmark suite summary

Protocol: results/BENCH_PROTOCOL.md. tok/s: mean +- sd [95% CI], CV, n (timed reps, flagged blocks excluded). Energy: IOReport GPU rail over the test window, estimated (SoC rails); none for llama-bench (its window includes model load).

## Qwen3.6-35B-A3B (20.4 GB, bigger than RAM)

Weights: ours: mlx-community 4-bit g64, experts streamed from the SSD (96 slots/layer, C++ streamer; 128 ran out of Metal memory in v1); llama.cpp: Q4_K_M, experts on the CPU (-ncmoe 40)

| engine | test | tok/s | 95% CI | CV | n | GPU mJ/token |
| --- | --- | --- | --- | --- | --- | --- |
| llama.cpp | pp512 | 16.36 +- 0.35 | 15.49-17.23 | 2.1% | 3 | - |
| mj | pp512 | 169.73 +- 3.84 | 166.98-172.47 | 2.3% | 10 | 31 |
| llama.cpp | pp2048 | 13.92 +- 1.40 | 10.44-17.40 | 10.1% | 3 | - |
| mj | pp2048 | 278.41 +- 2.23 | 276.81-280.00 | 0.8% | 10 | 36 |
| llama.cpp | tg128@d0 | 4.42 +- 0.15 | 4.04-4.80 | 3.5% | 3 | - |
| mj | tg128@d0 | 22.84 +- 2.68 | 20.93-24.75 | 11.7% | 10 | 222 |
| llama.cpp | tg128@d2048 | 4.77 +- 1.34 | 1.43-8.11 | 28.2% | 3 | - |
| mj | tg128@d2048 | 20.30 +- 1.34 | 19.34-21.26 | 6.6% | 10 | 251 |

Flagged blocks (excluded above; reason in blocks/*.flagged):

| engine | test | tok/s (flagged reps) | n |
| --- | --- | --- | --- |
| llama.cpp | pp512 | 17.00 +- 0.46 (16.46-17.29) | 3 |
| llama.cpp | pp2048 | 13.66 +- 1.13 (12.60-14.85) | 3 |
| llama.cpp | tg128@d0 | 4.13 +- 0.65 (3.40-4.61) | 3 |
| llama.cpp | tg128@d2048 | 3.93 +- 1.43 (2.29-4.93) | 3 |
