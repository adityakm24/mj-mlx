# Benchmark suite summary

Protocol: results/BENCH_PROTOCOL.md. tok/s: mean +- sd [95% CI], CV, n (timed reps, flagged blocks excluded). Energy: IOReport GPU rail over the test window, estimated (SoC rails); none for llama-bench (its window includes model load).

## Qwen3.6-35B-A3B (20.4 GB, bigger than RAM)

Weights: ours: mlx-community 4-bit g64, experts streamed from the SSD (96 slots/layer, C++ streamer; 128 ran out of Metal memory in v1); llama.cpp: Q4_K_M, experts on the CPU (-ncmoe 40)

| engine | test | tok/s | 95% CI | CV | n | GPU mJ/token | machine mJ/token above idle |
| --- | --- | --- | --- | --- | --- | --- | --- |
| llama-server | pp512 | every block flagged (see below) | | | 0 | - | - |
| llama.cpp | pp512 | every block flagged (see below) | | | 0 | - | - |
| mj | pp512 | 171.15 +- 3.03 | 168.99-173.32 | 1.8% | 10 | 31 | 71 |
| llama-server | pp2048 | every block flagged (see below) | | | 0 | - | - |
| llama.cpp | pp2048 | every block flagged (see below) | | | 0 | - | - |
| mj | pp2048 | 275.60 +- 2.18 | 274.05-277.16 | 0.8% | 10 | 37 | 76 |
| llama-server | tg128@d0 | every block flagged (see below) | | | 0 | - | - |
| llama.cpp | tg128@d0 | every block flagged (see below) | | | 0 | - | - |
| mj | tg128@d0 | 22.63 +- 2.79 | 20.63-24.63 | 12.3% | 10 | 208 | 772 |
| llama-server | tg128@d2048 | every block flagged (see below) | | | 0 | - | - |
| llama.cpp | tg128@d2048 | every block flagged (see below) | | | 0 | - | - |
| mj | tg128@d2048 | 20.62 +- 1.26 | 19.72-21.51 | 6.1% | 10 | 232 | 941 |

Flagged blocks (excluded above; reason in blocks/*.flagged):

| engine | test | tok/s (flagged reps) | n |
| --- | --- | --- | --- |
| llama-server | pp512 | 15.87 +- 0.46 (14.83-16.62) | 10 |
| llama.cpp | pp512 | 16.31 +- 0.86 (14.76-17.07) | 6 |
| llama-server | pp2048 | 14.49 +- 1.34 (11.71-16.17) | 10 |
| llama.cpp | pp2048 | 13.17 +- 1.09 (12.41-15.22) | 6 |
| llama-server | tg128@d0 | 4.15 +- 1.19 (2.22-5.66) | 10 |
| llama.cpp | tg128@d0 | 4.16 +- 0.87 (3.45-5.86) | 6 |
| llama-server | tg128@d2048 | 3.06 +- 0.52 (2.33-3.86) | 10 |
| llama.cpp | tg128@d2048 | 4.02 +- 1.09 (2.87-5.38) | 6 |

Peak memory per block, GB (footprint: macOS's figure, GPU buffers included, mapped files not; resident: pages in RAM, mapped weight files included); swap-ins / -outs over the block:

| block | engine | peak footprint | peak resident | swap-in GB | swap-out GB |
| --- | --- | --- | --- | --- | --- |
| r1_llama-server | llama-server | 24.50 | 4.96 | 977.61 | 983.69 |
| r1_mj | mj | 11.52 | 7.87 | 1.58 | 1.19 |
| r2_llama-server | llama-server | 24.50 | 4.71 | 882.98 | 887.08 |
| r2_mj | mj | 11.51 | 7.87 | 0.76 | 0.09 |
| r1_llama.cpp | llama.cpp | 20.04 | 12.07 | - | - |
| r2_llama.cpp | llama.cpp | 20.03 | 11.22 | - | - |
