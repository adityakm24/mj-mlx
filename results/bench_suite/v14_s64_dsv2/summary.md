# Benchmark suite summary

Protocol: results/BENCH_PROTOCOL.md. tok/s: mean +- sd [95% CI], CV, n (timed reps, flagged blocks excluded). Energy: IOReport GPU rail over the test window, estimated (SoC rails); none for llama-bench (its window includes model load).

## DeepSeek-V2-Lite-Chat (MLA, general MoE model)

Weights: ours / mlx-lm: mlx-community 4-bit g64; ours caches MLA latents, mlx-lm expanded K/V; llama.cpp / Ollama: mradermacher Q4_K_M; LM Studio (its MLX engine) and vllm-mlx: the mlx-community 4-bit

| engine | test | tok/s | 95% CI | CV | n | GPU mJ/token |
| --- | --- | --- | --- | --- | --- | --- |
| mj | pp512 | 586.91 +- 2.56 | 580.55-593.27 | 0.4% | 3 | 33 |
| mj-s64 | pp512 | 588.46 +- 1.81 | 586.55-590.36 | 0.3% | 6 | 34 |
| mj | pp2048 | 576.71 +- 1.65 | 572.61-580.81 | 0.3% | 3 | 34 |
| mj-s64 | pp2048 | 594.54 +- 1.67 | 592.78-596.29 | 0.3% | 6 | 34 |
| mj | tg128@d2048 | 52.62 +- 0.06 | 52.46-52.77 | 0.1% | 3 | 198 |
| mj-s64 | tg128@d2048 | 52.71 +- 0.07 | 52.64-52.79 | 0.1% | 6 | 199 |
| mj | pp16384 | 363.26 +- 1.14 | 360.43-366.09 | 0.3% | 3 | 45 |
| mj | pp8192 | 460.13 +- 3.99 | 450.22-470.04 | 0.9% | 3 | 38 |
| mj-s64 | pp16384 | 368.06 +- 0.89 | 367.13-368.99 | 0.2% | 6 | 45 |
| mj-s64 | pp8192 | 481.39 +- 2.77 | 478.48-484.30 | 0.6% | 6 | 39 |

Flagged blocks (excluded above; reason in blocks/*.flagged):

| engine | test | tok/s (flagged reps) | n |
| --- | --- | --- | --- |
| mj | pp512 | 587.27 +- 2.83 (585.04-590.45) | 3 |
| mj | pp2048 | 576.78 +- 0.26 (576.50-577.02) | 3 |
| mj | tg128@d2048 | 52.78 +- 0.04 (52.75-52.82) | 3 |
| mj | pp8192 | 470.14 +- 2.82 (467.68-473.21) | 3 |
