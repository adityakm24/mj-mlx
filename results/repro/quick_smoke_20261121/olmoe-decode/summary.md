# Benchmark suite summary

Protocol: results/BENCH_PROTOCOL.md. tok/s: mean +- sd [95% CI], CV, n (timed reps, flagged blocks excluded). Energy: IOReport GPU rail over the test window, estimated (SoC rails); none for llama-bench (its window includes model load).

## OLMoE-1B-7B-0125-Instruct (general MoE model)

Weights: ours / mlx-lm: mlx-community 4-bit g64 (float16 scales; run in float16); llama.cpp / Ollama (its GGUF runner; its MLX runner rejects OlmoeForCausalLM): allenai Q4_K_M; LM Studio (its MLX engine) and vllm-mlx: the mlx-community 4-bit

| engine | test | tok/s | 95% CI | CV | n | GPU mJ/token |
| --- | --- | --- | --- | --- | --- | --- |
| llama.cpp | pp512 | 1267.80 +- 0.00 | nan-nan | 0.0% | 1 | - |
| mj | pp512 | 1253.03 +- 0.00 | nan-nan | 0.0% | 1 | 14 |
| mlx-lm | pp512 | 1272.71 +- 0.00 | nan-nan | 0.0% | 1 | 13 |
| vllm-mlx | pp512 | 1033.43 +- 0.00 | nan-nan | 0.0% | 1 | 15 |
| llama.cpp | pp2048 | 1206.35 +- 0.00 | nan-nan | 0.0% | 1 | - |
| mj | pp2048 | 1274.01 +- 0.00 | nan-nan | 0.0% | 1 | 15 |
| mlx-lm | pp2048 | 1354.90 +- 0.00 | nan-nan | 0.0% | 1 | 14 |
| vllm-mlx | pp2048 | 1263.62 +- 0.00 | nan-nan | 0.0% | 1 | 15 |
| llama.cpp | tg128@d0 | 115.79 +- 0.00 | nan-nan | 0.0% | 1 | - |
| mj | tg128@d0 | 122.19 +- 0.00 | nan-nan | 0.0% | 1 | 71 |
| mlx-lm | tg128@d0 | 138.92 +- 0.00 | nan-nan | 0.0% | 1 | 66 |
| vllm-mlx | tg128@d0 | 139.49 +- 0.00 | nan-nan | 0.0% | 1 | 68 |
| llama.cpp | tg128@d2048 | 89.55 +- 0.00 | nan-nan | 0.0% | 1 | - |
| mj | tg128@d2048 | 97.82 +- 0.00 | nan-nan | 0.0% | 1 | 90 |
| mlx-lm | tg128@d2048 | 94.72 +- 0.00 | nan-nan | 0.0% | 1 | 88 |
| vllm-mlx | tg128@d2048 | 99.82 +- 0.00 | nan-nan | 0.0% | 1 | 86 |

Greedy decode ids, ours vs mlx-lm (same seeds): 2/2 identical
