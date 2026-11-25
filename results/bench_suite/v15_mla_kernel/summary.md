# Benchmark suite summary

Protocol: results/BENCH_PROTOCOL.md. tok/s: mean +- sd [95% CI], CV, n (timed reps, flagged blocks excluded). Energy: IOReport GPU rail over the test window, estimated (SoC rails); none for llama-bench (its window includes model load).

## DeepSeek-V2-Lite-Chat (MLA, general MoE model)

Weights: ours / mlx-lm: mlx-community 4-bit g64; ours caches MLA latents, mlx-lm expanded K/V; llama.cpp / Ollama: mradermacher Q4_K_M; LM Studio (its MLX engine) and vllm-mlx: the mlx-community 4-bit

| engine | test | tok/s | 95% CI | CV | n | GPU mJ/token |
| --- | --- | --- | --- | --- | --- | --- |
| mj | tg128@d0 | every block flagged (see below) | | | 0 | - |
| mj-mlak | tg128@d0 | 58.78 +- 0.03 | 58.71-58.86 | 0.1% | 3 | 153 |
| mj | tg128@d2048 | every block flagged (see below) | | | 0 | - |
| mj-mlak | tg128@d2048 | 44.07 +- 0.02 | 44.03-44.12 | 0.0% | 3 | 213 |
| mj | tg128@d16384 | every block flagged (see below) | | | 0 | - |
| mj | tg128@d8192 | every block flagged (see below) | | | 0 | - |
| mj-mlak | tg128@d16384 | 22.53 +- 0.15 | 22.15-22.92 | 0.7% | 3 | 551 |
| mj-mlak | tg128@d8192 | 30.38 +- 0.27 | 29.71-31.06 | 0.9% | 3 | 379 |

Flagged blocks (excluded above; reason in blocks/*.flagged):

| engine | test | tok/s (flagged reps) | n |
| --- | --- | --- | --- |
| mj | tg128@d0 | 61.55 +- 0.85 (60.13-62.43) | 6 |
| mj-mlak | tg128@d0 | 57.75 +- 0.95 (56.69-58.52) | 3 |
| mj | tg128@d2048 | 51.79 +- 0.60 (50.73-52.32) | 6 |
| mj-mlak | tg128@d2048 | 43.68 +- 0.65 (42.92-44.06) | 3 |
| mj | tg128@d16384 | 22.26 +- 0.16 (22.13-22.44) | 3 |
| mj | tg128@d8192 | 33.29 +- 0.59 (32.36-33.91) | 6 |
| mj-mlak | tg128@d8192 | 30.30 +- 0.47 (29.76-30.58) | 3 |

Peak memory per block (phys_footprint of the process holding the model; swap-ins / -outs over the block, GB):

| block | engine | peak GB | swap-in GB | swap-out GB |
| --- | --- | --- | --- | --- |
| r1_mj-mlak | mj-mlak | 12.70 | 0.07 | 0.00 |
| r1_mj | mj | 12.84 | 0.33 | 1.46 |
| r2_mj-mlak | mj-mlak | 12.65 | 0.62 | 1.87 |
| r2_mj | mj | 12.31 | 1.48 | 2.42 |
