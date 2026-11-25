# Benchmark suite summary

Protocol: results/BENCH_PROTOCOL.md. tok/s: mean +- sd [95% CI], CV, n (timed reps, flagged blocks excluded). Energy: IOReport GPU rail over the test window, estimated (SoC rails); none for llama-bench (its window includes model load).

## Qwen3.6-35B-A3B (20.4 GB, bigger than RAM)

Weights: ours: mlx-community 4-bit g64, experts streamed from the SSD (96 slots/layer, C++ streamer; 128 ran out of Metal memory in v1); llama.cpp: Q4_K_M, experts on the CPU (-ncmoe 40)

| engine | test | tok/s | 95% CI | CV | n | GPU mJ/token | machine mJ/token above idle | machine, steady state | swap-out MB in window (mean / max) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mj | tg128@d0 | 21.89 +- 3.77 | 19.80-23.97 | 17.2% | 15 | 204 | 863 | - | 0 / 0 |
| mj-u96 | tg128@d0 | 23.21 +- 3.76 | 21.13-25.29 | 16.2% | 15 | 204 | 842 | - | 0 / 0 |
| mj | tg128@d2048 | 21.39 +- 1.53 | 20.54-22.23 | 7.1% | 15 | 226 | 919 | - | 0 / 0 |
| mj-u96 | tg128@d2048 | 22.03 +- 1.72 | 21.09-22.98 | 7.8% | 15 | 234 | 977 | - | 0 / 0 |

Peak memory per block, GB (footprint: macOS's figure, GPU buffers included, mapped files not; resident: pages of the process in RAM that its page tables map, which counts a mapped weight file only where the CPU touched it: a file the GPU reads through a no-copy buffer, as llama.cpp's Metal backend does, can read near 0 or near its full size depending on how it was paged in, so resident is not comparable across engines); swap-ins / -outs over the whole block (load included):

| block | engine | peak footprint | peak resident | swap-in GB | swap-out GB |
| --- | --- | --- | --- | --- | --- |
| r4_mj-u96 | mj-u96 | 11.45 | 7.88 | 0.58 | 0.53 |
| r4_mj | mj | 11.42 | 7.88 | 0.17 | 0.00 |
| r5_mj-u96 | mj-u96 | 11.56 | 7.89 | 0.03 | 0.00 |
| r5_mj | mj | 11.40 | 7.89 | 0.21 | 0.00 |
| r6_mj-u96 | mj-u96 | 11.51 | 7.87 | 0.13 | 0.07 |
| r6_mj | mj | 11.51 | 7.88 | 0.29 | 0.23 |
