# Benchmark suite summary

Protocol: results/BENCH_PROTOCOL.md. tok/s: mean +- sd [95% CI], CV, n (timed reps, flagged blocks excluded). Energy: IOReport GPU rail over the test window, estimated (SoC rails); none for llama-bench (its window includes model load).

## DeepSeek-V2-Lite-Chat (MLA, general MoE model)

Weights: ours / mlx-lm: mlx-community 4-bit g64; ours caches MLA latents, mlx-lm expanded K/V; llama.cpp / Ollama: mradermacher Q4_K_M; LM Studio (its MLX engine) and vllm-mlx: the mlx-community 4-bit

| engine | test | tok/s | 95% CI | CV | n | GPU mJ/token |
| --- | --- | --- | --- | --- | --- | --- |
| mj | pp16384 | every block flagged (see below) | | | 0 | - |
| mj | pp8192 | every block flagged (see below) | | | 0 | - |
| mj | tg128@d16384 | every block flagged (see below) | | | 0 | - |
| mj | tg128@d8192 | every block flagged (see below) | | | 0 | - |
| mj-c512 | pp16384 | 311.65 +- 0.38 | 311.25-312.05 | 0.1% | 6 | 54 |
| mj-c512 | pp8192 | 420.36 +- 4.62 | 415.51-425.21 | 1.1% | 6 | 45 |
| mj-c512 | tg128@d16384 | 22.61 +- 0.19 | 22.41-22.80 | 0.8% | 6 | 525 |
| mj-c512 | tg128@d8192 | 33.98 +- 0.12 | 33.86-34.11 | 0.3% | 6 | 345 |

Flagged blocks (excluded above; reason in blocks/*.flagged):

| engine | test | tok/s (flagged reps) | n |
| --- | --- | --- | --- |
| mj | pp16384 | 363.92 +- 2.19 (361.46-366.74) | 6 |
| mj | pp8192 | 469.66 +- 8.90 (455.14-478.12) | 6 |
| mj | tg128@d16384 | 22.48 +- 0.07 (22.39-22.56) | 4 |
| mj | tg128@d8192 | 33.41 +- 0.37 (32.97-33.91) | 6 |
