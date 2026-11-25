# Benchmark suite summary

Protocol: results/BENCH_PROTOCOL.md. tok/s: mean +- sd [95% CI], CV, n (timed reps, flagged blocks excluded). Energy: IOReport GPU rail over the test window, estimated (SoC rails); none for llama-bench (its window includes model load).

## Qwen3.6-35B-A3B (20.4 GB, bigger than RAM)

Weights: ours: mlx-community 4-bit g64, experts streamed from the SSD (96 slots/layer, C++ streamer; 128 ran out of Metal memory in v1); llama.cpp: Q4_K_M, experts on the CPU (-ncmoe 40)

| engine | test | tok/s | 95% CI | CV | n | GPU mJ/token |
| --- | --- | --- | --- | --- | --- | --- |
| llama.cpp | pp512 | 16.51 +- 0.43 | 15.44-17.57 | 2.6% | 3 | - |
| mj | pp512 | 161.69 +- 4.64 | 155.92-167.46 | 2.9% | 5 | 18 |
| llama.cpp | pp2048 | 12.95 +- 0.30 | 12.21-13.69 | 2.3% | 3 | - |
| mj | pp2048 | 277.36 +- 4.32 | 271.99-282.73 | 1.6% | 5 | 33 |
| llama.cpp | tg128@d0 | 3.78 +- 0.80 | 1.79-5.76 | 21.2% | 3 | - |
| mj | tg128@d0 | 24.06 +- 5.18 | 17.62-30.49 | 21.5% | 5 | 209 |
| llama.cpp | tg128@d2048 | 3.20 +- 0.60 | 1.70-4.69 | 18.8% | 3 | - |
| mj | tg128@d2048 | 21.65 +- 0.91 | 20.52-22.78 | 4.2% | 5 | 232 |

Flagged blocks (excluded above; reason in blocks/*.flagged):

| engine | test | tok/s (flagged reps) | n |
| --- | --- | --- | --- |
| llama.cpp | pp512 | 16.61 +- 0.15 (16.45-16.74) | 3 |
| mj | pp512 | 162.91 +- 4.26 (158.85-167.73) | 5 |
| llama.cpp | pp2048 | 14.42 +- 1.47 (12.74-15.48) | 3 |
| mj | pp2048 | 280.23 +- 4.28 (274.54-286.20) | 5 |
| llama.cpp | tg128@d0 | 3.96 +- 0.57 (3.56-4.61) | 3 |
| mj | tg128@d0 | 21.02 +- 2.29 (17.24-23.25) | 5 |
| llama.cpp | tg128@d2048 | 4.21 +- 1.27 (2.77-5.14) | 3 |
| mj | tg128@d2048 | 21.62 +- 0.62 (20.68-22.31) | 5 |
