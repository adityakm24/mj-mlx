# Benchmark suite summary

Protocol: results/BENCH_PROTOCOL.md. tok/s: mean +- sd [95% CI], CV, n (timed reps, flagged blocks excluded). Energy: IOReport GPU rail over the test window, estimated (SoC rails); none for llama-bench (its window includes model load).

## DeepSeek-V2-Lite-Chat (MLA, general MoE model)

Weights: ours / mlx-lm: mlx-community 4-bit g64; ours caches MLA latents, mlx-lm expanded K/V; llama.cpp / Ollama: mradermacher Q4_K_M; LM Studio (its MLX engine) and vllm-mlx: the mlx-community 4-bit

| engine | test | tok/s | 95% CI | CV | n | GPU mJ/token | machine mJ/token above idle |
| --- | --- | --- | --- | --- | --- | --- | --- |
| llama-server | pp512 | 7911.17 +- 1071.28 | 6579.29-9243.04 | 13.5% | 5 | 0 | -0 |
| llama.cpp | pp512 | 590.73 +- 0.91 | 590.08-591.38 | 0.2% | 10 | - | - |
| mj | pp512 | 584.17 +- 3.06 | 580.36-587.97 | 0.5% | 5 | 34 | 39 |
| ollama-gguf | pp512 | 374.77 +- 21.18 | 348.44-401.10 | 5.7% | 5 | 29 | 60 |
| vllm-mlx | pp512 | 485.47 +- 2.07 | 482.90-488.05 | 0.4% | 5 | 33 | 40 |
| llama-server | pp2048 | 28294.49 +- 2984.29 | 24584.27-32004.72 | 10.5% | 5 | 0 | 0 |
| llama.cpp | pp2048 | 549.88 +- 16.68 | 537.96-561.80 | 3.0% | 10 | - | - |
| mj | pp2048 | 594.89 +- 0.52 | 594.24-595.53 | 0.1% | 5 | 34 | 59 |
| ollama-gguf | pp2048 | 358.58 +- 19.79 | 333.98-383.18 | 5.5% | 5 | 32 | 88 |
| vllm-mlx | pp2048 | 477.45 +- 186.09 | 246.10-708.81 | 39.0% | 5 | 34 | 60 |
| llama-server | tg128@d0 | no rate: 5 replies of at most one token | | | 0 | - | - |
| llama.cpp | tg128@d0 | 49.97 +- 5.07 | 46.34-53.59 | 10.1% | 10 | - | - |
| mj | tg128@d0 | 62.24 +- 0.05 | 62.17-62.30 | 0.1% | 5 | 152 | 303 |
| ollama-gguf | tg128@d0 | 40.40 +- 5.14 | 34.01-46.80 | 12.7% | 5 | 212 | 624 (5 replies under the requested length; tok/s over the tokens produced) |
| vllm-mlx | tg128@d0 | 66.34 +- 0.80 | 65.35-67.34 | 1.2% | 5 | 150 | 294 |
| llama-server | tg128@d2048 | no rate: 5 replies of at most one token | | | 0 | - | - |
| llama.cpp | tg128@d2048 | 39.78 +- 2.33 | 38.11-41.44 | 5.9% | 10 | - | - |
| mj | tg128@d2048 | 52.46 +- 0.21 | 52.21-52.72 | 0.4% | 5 | 197 | 478 |
| ollama-gguf | tg128@d2048 | 24.69 +- 5.90 | 17.36-32.02 | 23.9% | 5 | 257 | 1137 (1 replies under the requested length; tok/s over the tokens produced) |
| vllm-mlx | tg128@d2048 | 49.29 +- 0.44 | 48.74-49.84 | 0.9% | 5 | 189 | 500 |

Flagged blocks (excluded above; reason in blocks/*.flagged):

| engine | test | tok/s (flagged reps) | n |
| --- | --- | --- | --- |
| llama-server | pp512 | 7706.87 +- 846.75 (6231.95-8262.78) | 5 |
| mj | pp512 | 585.60 +- 2.07 (581.96-586.88) | 5 |
| ollama-gguf | pp512 | 371.25 +- 15.83 (344.60-383.76) | 5 |
| vllm-mlx | pp512 | 485.38 +- 4.32 (479.74-490.82) | 5 |
| llama-server | pp2048 | 29110.99 +- 1787.88 (26075.77-30806.99) | 5 |
| mj | pp2048 | 595.16 +- 0.86 (594.24-596.21) | 5 |
| ollama-gguf | pp2048 | 387.71 +- 8.45 (379.04-396.78) | 5 |
| vllm-mlx | pp2048 | 561.73 +- 1.26 (560.50-563.80) | 5 |
| mj | tg128@d0 | 62.06 +- 0.10 (61.90-62.13) | 5 |
| ollama-gguf | tg128@d0 | 46.34 +- 0.69 (45.24-46.99) | 5 |
| vllm-mlx | tg128@d0 | 64.68 +- 3.38 (58.75-66.83) | 5 |
| mj | tg128@d2048 | 51.09 +- 2.59 (46.47-52.50) | 5 |
| ollama-gguf | tg128@d2048 | 33.03 +- 5.58 (24.25-37.24) | 5 |
| vllm-mlx | tg128@d2048 | 49.64 +- 0.18 (49.36-49.82) | 5 |

Peak memory per block, GB (footprint: macOS's figure, GPU buffers included, mapped files not; resident: pages in RAM, mapped weight files included); swap-ins / -outs over the block:

| block | engine | peak footprint | peak resident | swap-in GB | swap-out GB |
| --- | --- | --- | --- | --- | --- |
| r1_llama-server | llama-server | 2.36 | 0.98 | 0.09 | 0.49 |
| r1_mj | mj | 11.76 | 1.44 | 0.13 | 1.08 |
| r1_ollama-gguf | ollama-gguf | 11.18 | 12.58 | 7.83 | 15.53 |
| r1_vllm-mlx | vllm-mlx | 12.63 | 2.27 | 4.58 | 4.58 |
| r2_llama-server | llama-server | 2.36 | 10.71 | 0.46 | 0.00 |
| r2_mj | mj | 11.76 | 2.49 | 0.20 | 0.00 |
| r2_ollama-gguf | ollama-gguf | 11.18 | 12.52 | 14.83 | 22.99 |
| r2_vllm-mlx | vllm-mlx | 12.56 | 2.62 | 0.79 | 1.27 |
| r1_llama.cpp | llama.cpp | 1.35 | 11.57 | - | - |
| r2_llama.cpp | llama.cpp | 1.35 | 11.31 | - | - |
