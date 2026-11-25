# Benchmark suite summary

Protocol: results/BENCH_PROTOCOL.md. tok/s: mean +- sd [95% CI], CV, n (timed reps, flagged blocks excluded). Energy: IOReport GPU rail over the test window, estimated (SoC rails); none for llama-bench (its window includes model load).

## OLMoE-1B-7B-0125-Instruct (general MoE model)

Weights: ours / mlx-lm: mlx-community 4-bit g64 (float16 scales; run in float16); llama.cpp / Ollama (its GGUF runner; its MLX runner rejects OlmoeForCausalLM): allenai Q4_K_M; LM Studio (its MLX engine) and vllm-mlx: the mlx-community 4-bit

| engine | test | tok/s | 95% CI | CV | n | GPU mJ/token | machine mJ/token above idle |
| --- | --- | --- | --- | --- | --- | --- | --- |
| llama-server | pp512 | 1147.44 +- 7.18 | 1142.30-1152.57 | 0.6% | 10 | 14 | 9 |
| llama.cpp | pp512 | 1267.31 +- 2.62 | 1265.44-1269.19 | 0.2% | 10 | - | - |
| lmstudio-mlx | pp512 | 1114.03 +- 22.00 | 1098.31-1129.76 | 2.0% | 10 | 15 | 10 |
| mj | pp512 | 1251.76 +- 13.76 | 1241.92-1261.59 | 1.1% | 10 | 14 | 7 |
| mlx-lm | pp512 | 1290.02 +- 21.10 | 1274.95-1305.10 | 1.6% | 10 | 14 | 6 |
| ollama-gguf | pp512 | 1151.71 +- 15.58 | 1140.58-1162.85 | 1.4% | 10 | 14 | 10 |
| vllm-mlx | pp512 | 1057.18 +- 15.18 | 1046.33-1068.03 | 1.4% | 10 | 15 | 10 |
| llama-server | pp2048 | 1157.12 +- 7.36 | 1151.86-1162.38 | 0.6% | 10 | 16 | 24 |
| llama.cpp | pp2048 | 1208.10 +- 1.80 | 1206.81-1209.38 | 0.1% | 10 | - | - |
| lmstudio-mlx | pp2048 | 1285.08 +- 5.85 | 1280.90-1289.26 | 0.5% | 10 | 15 | 23 |
| mj | pp2048 | 1272.09 +- 7.94 | 1266.42-1277.76 | 0.6% | 10 | 15 | 19 |
| mlx-lm | pp2048 | 1358.91 +- 1.63 | 1357.74-1360.08 | 0.1% | 10 | 14 | 18 |
| ollama-gguf | pp2048 | 1159.79 +- 7.41 | 1154.49-1165.08 | 0.6% | 10 | 16 | 24 |
| vllm-mlx | pp2048 | 1268.24 +- 8.60 | 1262.09-1274.39 | 0.7% | 10 | 15 | 22 |
| llama-server | tg128@d0 | 100.54 +- 0.35 | 100.29-100.79 | 0.3% | 10 | 101 | 162 |
| llama.cpp | tg128@d0 | 115.65 +- 2.24 | 114.05-117.25 | 1.9% | 10 | - | - |
| lmstudio-mlx | tg128@d0 | 127.05 +- 0.29 | 126.84-127.25 | 0.2% | 10 | 65 | 113 (10 replies under the requested length; tok/s over the tokens produced) |
| mj | tg128@d0 | 139.09 +- 0.42 | 138.79-139.39 | 0.3% | 10 | 66 | 71 |
| mlx-lm | tg128@d0 | 137.19 +- 5.80 | 133.05-141.33 | 4.2% | 10 | 66 | 85 |
| ollama-gguf | tg128@d0 | 117.40 +- 0.24 | 117.23-117.56 | 0.2% | 10 | 102 | 180 |
| vllm-mlx | tg128@d0 | 139.96 +- 0.25 | 139.78-140.14 | 0.2% | 10 | 67 | 92 |
| llama-server | tg128@d2048 | 88.28 +- 0.09 | 88.21-88.35 | 0.1% | 10 | 122 | 313 |
| llama.cpp | tg128@d2048 | 88.94 +- 0.55 | 88.55-89.33 | 0.6% | 10 | - | - |
| lmstudio-mlx | tg128@d2048 | 97.86 +- 0.18 | 97.73-97.99 | 0.2% | 10 | 81 | 226 (10 replies under the requested length; tok/s over the tokens produced) |
| mj | tg128@d2048 | 100.76 +- 2.50 | 98.98-102.55 | 2.5% | 10 | 87 | 249 |
| mlx-lm | tg128@d2048 | 94.74 +- 0.69 | 94.24-95.23 | 0.7% | 10 | 87 | 242 |
| ollama-gguf | tg128@d2048 | 89.46 +- 0.52 | 89.09-89.82 | 0.6% | 10 | 122 | 342 |
| vllm-mlx | tg128@d2048 | 98.92 +- 1.46 | 97.87-99.97 | 1.5% | 10 | 86 | 269 |

Greedy decode ids, ours vs mlx-lm (same seeds): 16/20 identical; first divergence at token 8-25 otherwise

Peak memory per block, GB (footprint: macOS's figure, GPU buffers included, mapped files not; resident: pages in RAM, mapped weight files included); swap-ins / -outs over the block:

| block | engine | peak footprint | peak resident | swap-in GB | swap-out GB |
| --- | --- | --- | --- | --- | --- |
| r1_llama-server | llama-server | 4.75 | 8.58 | 0.00 | 0.00 |
| r1_lmstudio-mlx | lmstudio-mlx | 0.00 | 0.00 | 0.01 | 0.00 |
| r1_mj | mj | 7.16 | 0.21 | 0.12 | 0.08 |
| r1_mlx-lm | mlx-lm | 6.58 | 0.82 | 0.02 | 0.02 |
| r1_ollama-gguf | ollama-gguf | 0.00 | 0.00 | 0.01 | 0.00 |
| r1_vllm-mlx | vllm-mlx | 7.34 | 1.02 | 0.00 | 0.01 |
| r2_llama-server | llama-server | 4.67 | 8.29 | 0.02 | 0.00 |
| r2_lmstudio-mlx | lmstudio-mlx | 0.00 | 0.00 | 0.00 | 0.00 |
| r2_mj | mj | 7.16 | 0.35 | 0.01 | 0.00 |
| r2_mlx-lm | mlx-lm | 6.57 | 0.69 | 0.00 | 0.00 |
| r2_ollama-gguf | ollama-gguf | 0.00 | 0.00 | 0.01 | 0.00 |
| r2_vllm-mlx | vllm-mlx | 7.38 | 1.02 | 0.00 | 0.00 |
| r1_llama.cpp | llama.cpp | 0.70 | 4.91 | - | - |
| r2_llama.cpp | llama.cpp | 0.70 | 4.91 | - | - |
