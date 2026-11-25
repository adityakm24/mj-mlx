# Benchmark suite summary

Protocol: results/BENCH_PROTOCOL.md. tok/s: mean +- sd [95% CI], CV, n (timed reps, flagged blocks excluded). Energy: IOReport GPU rail over the test window, estimated (SoC rails); none for llama-bench (its window includes model load).

## OLMoE-1B-7B-0125-Instruct (general MoE model)

Weights: ours / mlx-lm: mlx-community 4-bit g64 (float16 scales; run in float16); llama.cpp / Ollama (its GGUF runner; its MLX runner rejects OlmoeForCausalLM): allenai Q4_K_M; LM Studio (its MLX engine) and vllm-mlx: the mlx-community 4-bit

| engine | test | tok/s | 95% CI | CV | n | GPU mJ/token |
| --- | --- | --- | --- | --- | --- | --- |
| llama.cpp | pp512 | 1267.18 +- 2.40 | 1265.46-1268.89 | 0.2% | 10 | - |
| lmstudio-mlx | pp512 | 1060.23 +- 14.41 | 1049.93-1070.52 | 1.4% | 10 | 15 |
| mj | pp512 | 1200.64 +- 21.71 | 1185.12-1216.16 | 1.8% | 10 | 14 |
| mlx-lm | pp512 | 1228.24 +- 24.23 | 1210.93-1245.56 | 2.0% | 10 | 13 |
| ollama-gguf | pp512 | 1107.13 +- 41.54 | 1077.44-1136.82 | 3.8% | 10 | 14 |
| vllm-mlx | pp512 | 992.56 +- 53.54 | 954.29-1030.82 | 5.4% | 10 | 14 |
| llama.cpp | pp2048 | 1207.86 +- 1.65 | 1206.68-1209.04 | 0.1% | 10 | - |
| lmstudio-mlx | pp2048 | 1273.73 +- 11.35 | 1265.62-1281.84 | 0.9% | 10 | 15 |
| mj | pp2048 | 1260.10 +- 7.35 | 1254.85-1265.36 | 0.6% | 10 | 15 |
| mlx-lm | pp2048 | 1343.35 +- 9.43 | 1336.62-1350.09 | 0.7% | 10 | 14 |
| ollama-gguf | pp2048 | 1135.33 +- 22.64 | 1119.15-1151.51 | 2.0% | 10 | 15 |
| vllm-mlx | pp2048 | 1252.84 +- 22.12 | 1237.03-1268.65 | 1.8% | 10 | 15 |
| llama.cpp | tg128@d0 | 116.11 +- 0.32 | 115.88-116.34 | 0.3% | 10 | - |
| lmstudio-mlx | tg128@d0 | 126.25 +- 0.48 | 125.91-126.59 | 0.4% | 10 | 61 (10 replies under the requested length; tok/s over the tokens produced) |
| mj | tg128@d0 | 133.77 +- 3.15 | 131.52-136.03 | 2.4% | 10 | 67 |
| mlx-lm | tg128@d0 | 137.63 +- 0.70 | 137.13-138.13 | 0.5% | 10 | 65 |
| ollama-gguf | tg128@d0 | 116.59 +- 0.37 | 116.33-116.86 | 0.3% | 10 | 101 |
| vllm-mlx | tg128@d0 | 138.87 +- 0.59 | 138.45-139.28 | 0.4% | 10 | 67 |
| llama.cpp | tg128@d2048 | 88.51 +- 1.71 | 87.30-89.73 | 1.9% | 10 | - |
| lmstudio-mlx | tg128@d2048 | 96.82 +- 0.11 | 96.75-96.90 | 0.1% | 10 | 82 (10 replies under the requested length; tok/s over the tokens produced) |
| mj | tg128@d2048 | 98.91 +- 3.33 | 96.53-101.30 | 3.4% | 10 | 88 |
| mlx-lm | tg128@d2048 | 95.15 +- 1.28 | 94.24-96.06 | 1.3% | 10 | 87 |
| ollama-gguf | tg128@d2048 | 89.63 +- 0.15 | 89.52-89.74 | 0.2% | 10 | 122 |
| vllm-mlx | tg128@d2048 | 98.03 +- 3.14 | 95.78-100.27 | 3.2% | 10 | 87 |

Greedy decode ids, ours vs mlx-lm (same seeds): 16/20 identical; first divergence at token 8-25 otherwise

## DeepSeek-V2-Lite-Chat (MLA, general MoE model)

Weights: ours / mlx-lm: mlx-community 4-bit g64; ours caches MLA latents, mlx-lm expanded K/V; llama.cpp / Ollama: mradermacher Q4_K_M; LM Studio (its MLX engine) and vllm-mlx: the mlx-community 4-bit

| engine | test | tok/s | 95% CI | CV | n | GPU mJ/token |
| --- | --- | --- | --- | --- | --- | --- |
| llama.cpp | pp512 | 590.97 +- 0.88 | 589.88-592.06 | 0.1% | 5 | - |
| lmstudio-mlx | pp512 | every block flagged (see below) | | | 0 | - |
| mj | pp512 | 574.67 +- 1.51 | 572.79-576.54 | 0.3% | 5 | 33 |
| mlx-lm | pp512 | every block flagged (see below) | | | 0 | - |
| ollama-gguf | pp512 | every block flagged (see below) | | | 0 | - |
| vllm-mlx | pp512 | 412.89 +- 117.46 | 328.94-496.83 | 28.4% | 10 | 33 |
| llama.cpp | pp2048 | 555.18 +- 9.23 | 543.70-566.66 | 1.7% | 5 | - |
| lmstudio-mlx | pp2048 | every block flagged (see below) | | | 0 | - |
| mj | pp2048 | 574.19 +- 1.48 | 572.35-576.02 | 0.3% | 5 | 35 |
| mlx-lm | pp2048 | every block flagged (see below) | | | 0 | - |
| ollama-gguf | pp2048 | every block flagged (see below) | | | 0 | - |
| vllm-mlx | pp2048 | 545.39 +- 4.80 | 541.96-548.83 | 0.9% | 10 | 34 |
| llama.cpp | tg128@d0 | 54.99 +- 0.03 | 54.96-55.03 | 0.1% | 5 | - |
| mj | tg128@d0 | 62.16 +- 0.05 | 62.10-62.22 | 0.1% | 5 | 155 |
| mlx-lm | tg128@d0 | every block flagged (see below) | | | 0 | - |
| ollama-gguf | tg128@d0 | every block flagged (see below) | | | 0 | - |
| vllm-mlx | tg128@d0 | 61.93 +- 1.17 | 61.10-62.77 | 1.9% | 10 | 166 |
| llama.cpp | tg128@d2048 | 41.18 +- 0.29 | 40.83-41.54 | 0.7% | 5 | - |
| mj | tg128@d2048 | 47.25 +- 3.69 | 42.65-51.84 | 7.8% | 5 | 219 |
| mlx-lm | tg128@d2048 | every block flagged (see below) | | | 0 | - |
| ollama-gguf | tg128@d2048 | every block flagged (see below) | | | 0 | - |
| vllm-mlx | tg128@d2048 | 44.78 +- 1.87 | 43.44-46.12 | 4.2% | 10 | 219 |

Flagged blocks (excluded above; reason in blocks/*.flagged):

| engine | test | tok/s (flagged reps) | n |
| --- | --- | --- | --- |
| llama.cpp | pp512 | 539.82 +- 5.83 (533.87-546.93) | 5 |
| lmstudio-mlx | pp512 | 293.94 +- 185.07 (11.85-526.37) | 10 |
| mj | pp512 | 573.39 +- 5.46 (568.43-582.06) | 5 |
| mlx-lm | pp512 | 491.96 +- 122.81 (243.65-561.97) | 10 |
| ollama-gguf | pp512 | 187.28 +- 64.00 (86.62-294.04) | 10 |
| llama.cpp | pp2048 | 513.35 +- 19.15 (495.10-540.17) | 5 |
| lmstudio-mlx | pp2048 | 327.94 +- 191.98 (30.03-557.79) | 10 |
| mj | pp2048 | 574.34 +- 1.51 (571.96-575.98) | 5 |
| mlx-lm | pp2048 | 399.48 +- 185.82 (53.09-579.48) | 10 |
| ollama-gguf | pp2048 | 210.63 +- 41.55 (111.01-250.45) | 10 |
| llama.cpp | tg128@d0 | 50.24 +- 1.15 (48.36-51.21) | 5 |
| mj | tg128@d0 | 61.92 +- 0.13 (61.77-62.06) | 5 |
| mlx-lm | tg128@d0 | 65.75 +- 2.00 (62.20-67.37) | 10 |
| ollama-gguf | tg128@d0 | 14.74 +- 5.97 (6.36-25.09) | 10 |
| llama.cpp | tg128@d2048 | 37.25 +- 1.76 (34.91-39.20) | 5 |
| mj | tg128@d2048 | 52.08 +- 0.41 (51.38-52.35) | 5 |
| mlx-lm | tg128@d2048 | 49.20 +- 1.03 (47.13-49.90) | 10 |
| ollama-gguf | tg128@d2048 | 14.80 +- 7.81 (2.35-26.73) | 10 |

Greedy decode ids, ours vs mlx-lm (same seeds): 13/20 identical; first divergence at token 1-71 otherwise

## gpt-oss-20b

Weights: mj: openai/gpt-oss-20b exact (MXFP4 experts memory-mapped, packed bf16 attention, experts mlocked, 512-token prefill chunks as llama.cpp's ubatch); mj-a8: the same with 8-bit affine attention weights (lossy, matched to the GGUF); llama.cpp / Ollama / LM Studio (its llama.cpp engine): ggml-org MXFP4 GGUF (Q8_0 attention)

| engine | test | tok/s | 95% CI | CV | n | GPU mJ/token |
| --- | --- | --- | --- | --- | --- | --- |
| llama.cpp | pp512 | 426.49 +- 23.96 | 409.36-443.61 | 5.6% | 10 | - |
| lmstudio-gguf | pp512 | every block flagged (see below) | | | 0 | - |
| mj | pp512 | every block flagged (see below) | | | 0 | - |
| mj-a8 | pp512 | 399.80 +- 1.49 | 397.96-401.65 | 0.4% | 5 | 46 |
| ollama-gguf | pp512 | 379.26 +- 19.30 | 365.47-393.05 | 5.1% | 10 | 42 |
| llama.cpp | pp2048 | 433.03 +- 20.41 | 418.44-447.62 | 4.7% | 10 | - |
| lmstudio-gguf | pp2048 | every block flagged (see below) | | | 0 | - |
| mj | pp2048 | every block flagged (see below) | | | 0 | - |
| mj-a8 | pp2048 | 435.98 +- 0.63 | 435.20-436.76 | 0.1% | 5 | 48 |
| ollama-gguf | pp2048 | 404.56 +- 4.93 | 401.03-408.08 | 1.2% | 10 | 43 |
| llama.cpp | tg128@d0 | 36.45 +- 0.48 | 36.11-36.79 | 1.3% | 10 | - |
| lmstudio-gguf | tg128@d0 | every block flagged (see below) | | | 0 | - |
| mj | tg128@d0 | every block flagged (see below) | | | 0 | - |
| mj-a8 | tg128@d0 | 38.08 +- 1.48 | 36.24-39.92 | 3.9% | 5 | 259 |
| ollama-gguf | tg128@d0 | 34.59 +- 1.05 | 33.84-35.34 | 3.0% | 10 | 358 |
| llama.cpp | tg128@d2048 | 34.90 +- 0.40 | 34.61-35.19 | 1.2% | 10 | - |
| lmstudio-gguf | tg128@d2048 | every block flagged (see below) | | | 0 | - |
| mj | tg128@d2048 | every block flagged (see below) | | | 0 | - |
| mj-a8 | tg128@d2048 | 37.82 +- 0.24 | 37.52-38.11 | 0.6% | 5 | 283 |
| ollama-gguf | tg128@d2048 | 34.25 +- 0.24 | 34.07-34.42 | 0.7% | 10 | 383 (2 replies under the requested length; tok/s over the tokens produced) |

Flagged blocks (excluded above; reason in blocks/*.flagged):

| engine | test | tok/s (flagged reps) | n |
| --- | --- | --- | --- |
| lmstudio-gguf | pp512 | 377.18 +- 12.38 (362.52-392.58) | 10 |
| mj | pp512 | 385.18 +- 33.16 (294.76-407.46) | 10 |
| mj-a8 | pp512 | 399.25 +- 4.41 (391.86-402.44) | 5 |
| lmstudio-gguf | pp2048 | 402.70 +- 48.65 (264.97-425.49) | 10 |
| mj | pp2048 | 426.23 +- 1.70 (423.17-429.33) | 10 |
| mj-a8 | pp2048 | 429.22 +- 3.03 (424.77-432.60) | 5 |
| lmstudio-gguf | tg128@d0 | 34.98 +- 0.86 (33.27-36.01) | 10 |
| mj | tg128@d0 | 35.82 +- 0.69 (33.96-36.35) | 10 |
| mj-a8 | tg128@d0 | 39.49 +- 0.05 (39.45-39.57) | 5 |
| lmstudio-gguf | tg128@d2048 | 33.09 +- 1.01 (30.42-33.96) | 10 |
| mj | tg128@d2048 | 34.24 +- 1.26 (30.87-35.13) | 10 |
| mj-a8 | tg128@d2048 | 37.61 +- 0.62 (36.56-38.13) | 5 |
