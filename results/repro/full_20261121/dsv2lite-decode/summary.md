# Benchmark suite summary

Protocol: results/BENCH_PROTOCOL.md. tok/s: mean +- sd [95% CI], CV, n (timed reps, flagged blocks excluded). Energy: IOReport GPU rail over the test window, estimated (SoC rails); none for llama-bench (its window includes model load).

## DeepSeek-V2-Lite-Chat (MLA, general MoE model)

Weights: ours / mlx-lm: mlx-community 4-bit g64; ours caches MLA latents, mlx-lm expanded K/V; llama.cpp / Ollama: mradermacher Q4_K_M; LM Studio (its MLX engine) and vllm-mlx: the mlx-community 4-bit

| engine | test | tok/s | 95% CI | CV | n | GPU mJ/token |
| --- | --- | --- | --- | --- | --- | --- |
| llama.cpp | pp512 | 590.96 +- 1.26 | 590.06-591.87 | 0.2% | 10 | - |
| mj | pp512 | 585.75 +- 2.45 | 584.00-587.50 | 0.4% | 10 | 34 |
| ollama-gguf | pp512 | every block flagged (see below) | | | 0 | - |
| vllm-mlx | pp512 | 485.38 +- 12.26 | 476.62-494.14 | 2.5% | 10 | 33 |
| llama.cpp | pp2048 | 543.20 +- 15.57 | 532.07-554.33 | 2.9% | 10 | - |
| mj | pp2048 | 577.06 +- 1.33 | 576.10-578.01 | 0.2% | 10 | 35 |
| ollama-gguf | pp2048 | every block flagged (see below) | | | 0 | - |
| vllm-mlx | pp2048 | 563.00 +- 2.65 | 561.10-564.89 | 0.5% | 10 | 34 |
| llama.cpp | tg128@d0 | 54.98 +- 0.21 | 54.83-55.14 | 0.4% | 10 | - |
| mj | tg128@d0 | 62.29 +- 0.26 | 62.10-62.48 | 0.4% | 10 | 156 |
| ollama-gguf | tg128@d0 | every block flagged (see below) | | | 0 | - |
| vllm-mlx | tg128@d0 | 67.20 +- 0.22 | 67.04-67.36 | 0.3% | 10 | 155 |
| llama.cpp | tg128@d2048 | 41.53 +- 0.55 | 41.14-41.93 | 1.3% | 10 | - |
| mj | tg128@d2048 | 52.52 +- 0.19 | 52.38-52.65 | 0.4% | 10 | 204 |
| ollama-gguf | tg128@d2048 | every block flagged (see below) | | | 0 | - |
| vllm-mlx | tg128@d2048 | 49.73 +- 0.14 | 49.63-49.83 | 0.3% | 10 | 198 |

Flagged blocks (excluded above; reason in blocks/*.flagged):

| engine | test | tok/s (flagged reps) | n |
| --- | --- | --- | --- |
| ollama-gguf | pp512 | 386.22 +- 12.12 (363.16-404.55) | 10 |
| ollama-gguf | pp2048 | 378.08 +- 18.03 (331.49-395.62) | 10 |
| ollama-gguf | tg128@d0 | 45.71 +- 1.98 (40.26-46.86) | 10 |
| ollama-gguf | tg128@d2048 | 35.31 +- 3.18 (29.62-37.80) | 10 |
