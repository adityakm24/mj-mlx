# Benchmark suite summary

Protocol: results/BENCH_PROTOCOL.md. tok/s: mean +- sd [95% CI], CV, n (timed reps, flagged blocks excluded). Energy: IOReport GPU rail over the test window, estimated (SoC rails); none for llama-bench (its window includes model load).

## DeepSeek-V2-Lite-Chat (MLA, general MoE model)

Weights: ours / mlx-lm: mlx-community 4-bit g64; ours caches MLA latents, mlx-lm expanded K/V; llama.cpp / Ollama: mradermacher Q4_K_M; LM Studio (its MLX engine) and vllm-mlx: the mlx-community 4-bit

| engine | test | tok/s | 95% CI | CV | n | GPU mJ/token |
| --- | --- | --- | --- | --- | --- | --- |
| mj-c512 | pp16384 | 309.48 +- 0.50 | 308.25-310.71 | 0.2% | 3 | 53 |
| mj-c512 | tg128@d16384 | 22.17 +- 0.17 | 21.75-22.60 | 0.8% | 3 | 500 |
