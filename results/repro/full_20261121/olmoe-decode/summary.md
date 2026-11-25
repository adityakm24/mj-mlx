# Benchmark suite summary

Protocol: results/BENCH_PROTOCOL.md. tok/s: mean +- sd [95% CI], CV, n (timed reps, flagged blocks excluded). Energy: IOReport GPU rail over the test window, estimated (SoC rails); none for llama-bench (its window includes model load).

## OLMoE-1B-7B-0125-Instruct (general MoE model)

Weights: ours / mlx-lm: mlx-community 4-bit g64 (float16 scales; run in float16); llama.cpp / Ollama (its GGUF runner; its MLX runner rejects OlmoeForCausalLM): allenai Q4_K_M; LM Studio (its MLX engine) and vllm-mlx: the mlx-community 4-bit

| engine | test | tok/s | 95% CI | CV | n | GPU mJ/token |
| --- | --- | --- | --- | --- | --- | --- |
| llama.cpp | pp512 | 1267.22 +- 2.62 | 1265.35-1269.09 | 0.2% | 10 | - |
| mj | pp512 | 1241.78 +- 26.64 | 1222.74-1260.81 | 2.1% | 10 | 14 |
| mlx-lm | pp512 | 1307.84 +- 6.57 | 1303.14-1312.53 | 0.5% | 10 | 14 |
| vllm-mlx | pp512 | 1058.89 +- 22.28 | 1042.97-1074.82 | 2.1% | 10 | 15 |
| llama.cpp | pp2048 | 1208.09 +- 1.54 | 1206.99-1209.19 | 0.1% | 10 | - |
| mj | pp2048 | 1271.98 +- 7.65 | 1266.51-1277.45 | 0.6% | 10 | 15 |
| mlx-lm | pp2048 | 1361.00 +- 9.41 | 1354.28-1367.73 | 0.7% | 10 | 15 |
| vllm-mlx | pp2048 | 1269.39 +- 5.94 | 1265.14-1273.63 | 0.5% | 10 | 15 |
| llama.cpp | tg128@d0 | 116.50 +- 0.19 | 116.36-116.64 | 0.2% | 10 | - |
| mj | tg128@d0 | 139.59 +- 0.24 | 139.42-139.77 | 0.2% | 10 | 67 |
| mlx-lm | tg128@d0 | 138.05 +- 1.94 | 136.66-139.44 | 1.4% | 10 | 67 |
| vllm-mlx | tg128@d0 | 140.05 +- 0.55 | 139.66-140.45 | 0.4% | 10 | 69 |
| llama.cpp | tg128@d2048 | 89.22 +- 0.33 | 88.99-89.45 | 0.4% | 10 | - |
| mj | tg128@d2048 | 102.42 +- 0.17 | 102.30-102.54 | 0.2% | 10 | 89 |
| mlx-lm | tg128@d2048 | 95.75 +- 0.58 | 95.34-96.16 | 0.6% | 10 | 89 |
| vllm-mlx | tg128@d2048 | 99.53 +- 1.72 | 98.30-100.76 | 1.7% | 10 | 89 |

Greedy decode ids, ours vs mlx-lm (same seeds): 16/20 identical; first divergence at token 8-25 otherwise
