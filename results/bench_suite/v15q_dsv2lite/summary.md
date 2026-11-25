# Benchmark suite summary

Protocol: results/BENCH_PROTOCOL.md. tok/s: mean +- sd [95% CI], CV, n (timed reps, flagged blocks excluded). Energy: IOReport GPU rail over the test window, estimated (SoC rails); none for llama-bench (its window includes model load).

## DeepSeek-V2-Lite-Chat (MLA, general MoE model)

Weights: ours / mlx-lm: mlx-community 4-bit g64; ours caches MLA latents, mlx-lm expanded K/V; llama.cpp / Ollama: mradermacher Q4_K_M; LM Studio (its MLX engine) and vllm-mlx: the mlx-community 4-bit

| engine | test | tok/s | 95% CI | CV | n | GPU mJ/token | machine mJ/token above idle |
| --- | --- | --- | --- | --- | --- | --- | --- |
| llama.cpp | pp512 | 590.20 +- 0.87 | 589.12-591.28 | 0.1% | 5 | - | - |
| mj | pp512 | 585.68 +- 2.77 | 583.71-587.66 | 0.5% | 10 | 34 | 38 |
| ollama-gguf | pp512 | 369.08 +- 31.25 | 330.22-407.93 | 8.5% | 5 | 29 | 58 |
| vllm-mlx | pp512 | 493.20 +- 7.28 | 484.15-502.25 | 1.5% | 5 | 33 | 35 |
| llama.cpp | pp2048 | 543.32 +- 16.84 | 522.38-564.25 | 3.1% | 5 | - | - |
| mj | pp2048 | 594.68 +- 1.45 | 593.64-595.71 | 0.2% | 10 | 34 | 60 |
| ollama-gguf | pp2048 | 378.87 +- 14.70 | 360.59-397.14 | 3.9% | 5 | 32 | 85 |
| vllm-mlx | pp2048 | 560.25 +- 3.56 | 555.81-564.68 | 0.6% | 5 | 33 | 61 |
| llama.cpp | tg128@d0 | 54.84 +- 0.46 | 54.27-55.42 | 0.8% | 5 | - | - |
| mj | tg128@d0 | 62.09 +- 0.36 | 61.83-62.35 | 0.6% | 10 | 152 | 292 |
| ollama-gguf | tg128@d0 | 40.55 +- 6.05 | 33.02-48.07 | 14.9% | 5 | 214 | 642 (5 replies under the requested length; tok/s over the tokens produced) |
| vllm-mlx | tg128@d0 | 66.88 +- 0.20 | 66.64-67.13 | 0.3% | 5 | 150 | 299 |
| llama.cpp | tg128@d2048 | 41.77 +- 0.05 | 41.71-41.83 | 0.1% | 5 | - | - |
| mj | tg128@d2048 | 52.37 +- 0.27 | 52.18-52.56 | 0.5% | 10 | 198 | 488 |
| ollama-gguf | tg128@d2048 | 29.71 +- 3.77 | 25.03-34.39 | 12.7% | 5 | 262 | 883 (1 replies under the requested length; tok/s over the tokens produced) |
| vllm-mlx | tg128@d2048 | 49.52 +- 0.50 | 48.90-50.15 | 1.0% | 5 | 190 | 487 |

Flagged blocks (excluded above; reason in blocks/*.flagged):

| engine | test | tok/s (flagged reps) | n |
| --- | --- | --- | --- |
| llama.cpp | pp512 | 590.39 +- 1.08 (588.77-591.79) | 5 |
| ollama-gguf | pp512 | 388.87 +- 5.52 (382.60-394.77) | 5 |
| vllm-mlx | pp512 | 479.72 +- 3.92 (476.11-484.33) | 5 |
| llama.cpp | pp2048 | 550.46 +- 14.02 (529.48-561.52) | 5 |
| ollama-gguf | pp2048 | 359.34 +- 26.80 (312.88-379.77) | 5 |
| vllm-mlx | pp2048 | 558.10 +- 1.72 (555.60-560.22) | 5 |
| llama.cpp | tg128@d0 | 55.13 +- 0.10 (54.97-55.22) | 5 |
| ollama-gguf | tg128@d0 | 41.64 +- 5.99 (31.19-45.65) | 5 |
| vllm-mlx | tg128@d0 | 66.80 +- 0.08 (66.69-66.88) | 5 |
| llama.cpp | tg128@d2048 | 40.90 +- 0.61 (39.89-41.40) | 5 |
| ollama-gguf | tg128@d2048 | 27.50 +- 3.27 (24.40-32.68) | 5 |
| vllm-mlx | tg128@d2048 | 48.88 +- 1.08 (47.25-49.69) | 5 |

Peak memory per block, GB (footprint: macOS's figure, GPU buffers included, mapped files not; resident: pages in RAM, mapped weight files included); swap-ins / -outs over the block:

| block | engine | peak footprint | peak resident | swap-in GB | swap-out GB |
| --- | --- | --- | --- | --- | --- |
| r1_mj | mj | 11.74 | 0.31 | 1.36 | 1.67 |
| r1_ollama-gguf | ollama-gguf | 11.18 | 12.51 | 14.83 | 23.32 |
| r1_vllm-mlx | vllm-mlx | 12.63 | 1.21 | 1.69 | 2.36 |
| r2_mj | mj | 11.76 | 2.40 | 0.22 | 0.03 |
| r2_ollama-gguf | ollama-gguf | 11.18 | 12.52 | 10.76 | 18.73 |
| r2_vllm-mlx | vllm-mlx | 12.56 | 2.76 | 0.26 | 0.34 |
| r1_llama.cpp | llama.cpp | 1.35 | 11.56 | - | - |
| r2_llama.cpp | llama.cpp | 1.35 | 11.64 | - | - |
