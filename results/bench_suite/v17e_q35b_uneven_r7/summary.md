# Benchmark suite summary

Protocol: results/BENCH_PROTOCOL.md. tok/s: mean +- sd [95% CI], CV, n (timed reps, flagged blocks excluded). Energy: IOReport GPU rail over the test window, estimated (SoC rails); none for llama-bench (its window includes model load).

## Qwen3.6-35B-A3B (20.4 GB, bigger than RAM)

Weights: ours: mlx-community 4-bit g64, experts streamed from the SSD (96 slots/layer, C++ streamer; 128 ran out of Metal memory in v1); llama.cpp: Q4_K_M, experts on the CPU (-ncmoe 40)

| engine | test | tok/s | 95% CI | CV | n | GPU mJ/token | machine mJ/token above idle | machine, steady state | swap-out MB in window (mean / max) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mj | tg128@d0 | 22.73 +- 3.29 | 21.50-23.96 | 14.5% | 30 | 204 | 817 | - | 0 / 0 |
| mj-u96 | tg128@d0 | 24.19 +- 2.92 | 22.98-25.39 | 12.1% | 25 | 204 | 819 | - | 0 / 0 |
| mj | tg128@d2048 | 20.65 +- 1.37 | 20.13-21.16 | 6.7% | 30 | 227 | 932 | - | 0 / 0 |
| mj-u96 | tg128@d2048 | 22.29 +- 1.37 | 21.73-22.86 | 6.1% | 25 | 227 | 922 | - | 0 / 0 |

Flagged reps and blocks (excluded above; block reasons in blocks/*.flagged; a rep is also flagged when its timed window swapped out more than 64 MB, and a block when more than half its reps are):

| engine | test | tok/s (flagged reps) | n |
| --- | --- | --- | --- |
| mj-u96 | tg128@d0 | 21.70 +- 1.21 (20.14-23.11) | 5 |
| mj-u96 | tg128@d2048 | 20.76 +- 2.56 (18.46-24.24) | 5 |

Peak memory per block, GB (footprint: macOS's figure, GPU buffers included, mapped files not; resident: pages of the process in RAM that its page tables map, which counts a mapped weight file only where the CPU touched it: a file the GPU reads through a no-copy buffer, as llama.cpp's Metal backend does, can read near 0 or near its full size depending on how it was paged in, so resident is not comparable across engines); swap-ins / -outs over the whole block (load included):

| block | engine | peak footprint | peak resident | swap-in GB | swap-out GB |
| --- | --- | --- | --- | --- | --- |
| r10_mj-u96 | mj-u96 | 11.39 | 7.87 | 0.17 | 0.07 |
| r10_mj | mj | 11.63 | 7.88 | 0.04 | 0.09 |
| r11_mj-u96 | mj-u96 | 11.51 | 7.86 | 0.34 | 0.48 |
| r11_mj | mj | 11.49 | 7.88 | 0.16 | 0.19 |
| r12_mj-u96 | mj-u96 | 11.53 | 7.86 | 1.71 | 2.56 |
| r12_mj | mj | 11.54 | 7.88 | 0.31 | 0.00 |
| r7_mj-u96 | mj-u96 | 11.51 | 7.87 | 0.17 | 0.27 |
| r7_mj | mj | 11.54 | 6.90 | 1.01 | 0.41 |
| r8_mj-u96 | mj-u96 | 11.52 | 7.87 | 0.30 | 0.46 |
| r8_mj | mj | 11.51 | 7.88 | 0.19 | 0.10 |
| r9_mj-u96 | mj-u96 | 11.42 | 7.88 | 0.38 | 0.46 |
| r9_mj | mj | 11.49 | 7.88 | 0.15 | 0.19 |
