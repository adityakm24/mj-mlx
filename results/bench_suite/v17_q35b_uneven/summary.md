# Benchmark suite summary

Protocol: results/BENCH_PROTOCOL.md. tok/s: mean +- sd [95% CI], CV, n (timed reps, flagged blocks excluded). Energy: IOReport GPU rail over the test window, estimated (SoC rails); none for llama-bench (its window includes model load).

## Qwen3.6-35B-A3B (20.4 GB, bigger than RAM)

Weights: ours: mlx-community 4-bit g64, experts streamed from the SSD (96 slots/layer, C++ streamer; 128 ran out of Metal memory in v1); llama.cpp: Q4_K_M, experts on the CPU (-ncmoe 40)

| engine | test | tok/s | 95% CI | CV | n | GPU mJ/token | machine mJ/token above idle | machine, steady state |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mj | pp2048 | 276.36 +- 3.04 | 274.68-278.03 | 1.1% | 15 | 36 | 72 | - |
| mj-u96 | pp2048 | 275.62 +- 4.34 | 273.22-278.02 | 1.6% | 15 | 36 | 75 | - |
| mj | tg128@d0 | 22.90 +- 2.64 | 21.44-24.35 | 11.5% | 15 | 203 | 818 | - |
| mj-u96 | tg128@d0 | 24.81 +- 2.24 | 23.58-26.05 | 9.0% | 15 | 202 | 798 | - |
| mj | tg128@d2048 | 20.57 +- 1.02 | 20.01-21.13 | 5.0% | 15 | 228 | 966 | - |
| mj-u96 | tg128@d2048 | 21.83 +- 1.68 | 20.90-22.76 | 7.7% | 15 | 225 | 935 | - |

Peak memory per block, GB (footprint: macOS's figure, GPU buffers included, mapped files not; resident: pages in RAM, mapped weight files included); swap-ins / -outs over the block:

| block | engine | peak footprint | peak resident | swap-in GB | swap-out GB |
| --- | --- | --- | --- | --- | --- |
| r1_mj-u96 | mj-u96 | 11.60 | 7.87 | 0.33 | 0.45 |
| r1_mj | mj | 11.53 | 6.91 | 0.39 | 0.51 |
| r2_mj-u96 | mj-u96 | 11.45 | 7.88 | 0.27 | 0.47 |
| r2_mj | mj | 11.42 | 7.87 | 0.21 | 0.17 |
| r3_mj-u96 | mj-u96 | 11.55 | 7.88 | 0.10 | 0.01 |
| r3_mj | mj | 11.58 | 7.87 | 0.34 | 0.48 |
