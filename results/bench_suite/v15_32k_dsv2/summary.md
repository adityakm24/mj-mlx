# Benchmark suite summary

Protocol: results/BENCH_PROTOCOL.md. tok/s: mean +- sd [95% CI], CV, n (timed reps, flagged blocks excluded). Energy: IOReport GPU rail over the test window, estimated (SoC rails); none for llama-bench (its window includes model load).

## DeepSeek-V2-Lite-Chat (MLA, general MoE model)

Weights: ours / mlx-lm: mlx-community 4-bit g64; ours caches MLA latents, mlx-lm expanded K/V; llama.cpp / Ollama: mradermacher Q4_K_M; LM Studio (its MLX engine) and vllm-mlx: the mlx-community 4-bit

| engine | test | tok/s | 95% CI | CV | n | GPU mJ/token | machine mJ/token above idle |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mj | pp16384 | 375.31 +- 3.80 | 341.18-409.45 | 1.0% | 2 | 47 | 97 |
| mj | pp32768 | 240.70 +- 3.07 | 213.15-268.26 | 1.3% | 2 | 56 | 121 |
| mj | tg128@d16384 | 22.69 +- 0.04 | 22.33-23.05 | 0.2% | 2 | 529 | 1176 |
| mj | tg128@d32768 | 13.66 +- 0.05 | 13.19-14.13 | 0.4% | 2 | 834 | 1808 |
| mj-w9 | pp16384 | every block flagged (see below) | | | 0 | - | - |
| mj-w9 | pp32768 | every block flagged (see below) | | | 0 | - | - |
| mj-w9 | tg128@d16384 | every block flagged (see below) | | | 0 | - | - |
| mj-w9 | tg128@d32768 | every block flagged (see below) | | | 0 | - | - |

Flagged blocks (excluded above; reason in blocks/*.flagged):

| engine | test | tok/s (flagged reps) | n |
| --- | --- | --- | --- |
| mj | pp16384 | 364.88 +- 1.76 (363.64-366.13) | 2 |
| mj | tg128@d16384 | 22.33 +- 0.01 (22.32-22.34) | 2 |
| mj-w9 | pp16384 | 373.06 +- 4.34 (369.18-377.57) | 4 |
| mj-w9 | pp32768 | 242.96 +- 7.61 (237.58-248.34) | 2 |
| mj-w9 | tg128@d16384 | 22.51 +- 0.21 (22.20-22.65) | 4 |
| mj-w9 | tg128@d32768 | 13.40 +- 0.33 (13.17-13.63) | 2 |

Peak memory per block, GB (footprint: macOS's figure, GPU buffers included, mapped files not; resident: pages in RAM, mapped weight files included); swap-ins / -outs over the block:

| block | engine | peak footprint | peak resident | swap-in GB | swap-out GB |
| --- | --- | --- | --- | --- | --- |
| r1_mj-w9 | mj-w9 | 12.75 | 2.53 | 0.74 | 0.65 |
| r1_mj | mj | 12.71 | 0.36 | 2.00 | 1.91 |
| r2_mj-w9 | mj-w9 | 14.06 | 2.67 | 3.37 | 4.12 |
| r2_mj | mj | 13.48 | 2.91 | 2.68 | 2.47 |
