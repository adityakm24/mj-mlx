# Benchmark suite summary

Protocol: results/BENCH_PROTOCOL.md. tok/s: mean +- sd [95% CI], CV, n (timed reps, flagged blocks excluded). Energy: IOReport GPU rail over the test window, estimated (SoC rails); none for llama-bench (its window includes model load).

## DeepSeek-V2-Lite-Chat (MLA, general MoE model)

Weights: ours / mlx-lm: mlx-community 4-bit g64; ours caches MLA latents, mlx-lm expanded K/V; llama.cpp / Ollama: mradermacher Q4_K_M; LM Studio (its MLX engine) and vllm-mlx: the mlx-community 4-bit

| engine | test | tok/s | 95% CI | CV | n | GPU mJ/token |
| --- | --- | --- | --- | --- | --- | --- |
| mj | pp8192 | 457.17 +- 5.13 | 444.42-469.91 | 1.1% | 3 | 39 |
| mj | tg128@d8192 | 33.82 +- 0.15 | 33.45-34.19 | 0.4% | 3 | 330 |
| mj-c512 | pp8192 | 417.16 +- 6.42 | 410.42-423.89 | 1.5% | 6 | 45 |
| mj-c512 | tg128@d8192 | 33.81 +- 0.25 | 33.55-34.07 | 0.7% | 6 | 326 |
| vllm-mlx | pp8192 | every block flagged (see below) | | | 0 | - |
| vllm-mlx | tg128@d8192 | every block flagged (see below) | | | 0 | - |

Flagged blocks (excluded above; reason in blocks/*.flagged):

| engine | test | tok/s (flagged reps) | n |
| --- | --- | --- | --- |
| mj | pp8192 | 459.81 +- 0.58 (459.20-460.37) | 3 |
| mj | tg128@d8192 | 33.59 +- 0.32 (33.26-33.89) | 3 |
| vllm-mlx | pp8192 | 564.74 +- 95.77 (384.88-646.46) | 6 |
