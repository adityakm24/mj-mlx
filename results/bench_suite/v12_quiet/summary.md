# Benchmark suite summary

Protocol: results/BENCH_PROTOCOL.md. tok/s: mean +- sd [95% CI], CV, n (timed reps, flagged blocks excluded). Energy: IOReport GPU rail over the test window, estimated (SoC rails); none for llama-bench (its window includes model load).

## OLMoE-1B-7B-0125-Instruct (general MoE model)

Weights: ours / mlx-lm: mlx-community 4-bit g64 (float16 scales; run in float16); llama.cpp / Ollama (its GGUF runner; its MLX runner rejects OlmoeForCausalLM): allenai Q4_K_M; LM Studio (its MLX engine) and vllm-mlx: the mlx-community 4-bit

| engine | test | tok/s | 95% CI | CV | n | GPU mJ/token |
| --- | --- | --- | --- | --- | --- | --- |
| llama.cpp | pp512 | 1266.19 +- 2.40 | 1264.47-1267.90 | 0.2% | 10 | - |
| lmstudio-mlx | pp512 | 1066.02 +- 51.60 | 1029.14-1102.90 | 4.8% | 10 | 15 |
| mj | pp512 | 1203.85 +- 20.02 | 1189.54-1218.16 | 1.7% | 10 | 14 |
| mlx-lm | pp512 | 1244.24 +- 24.26 | 1226.90-1261.57 | 1.9% | 10 | 14 |
| ollama-gguf | pp512 | 1112.52 +- 37.36 | 1085.82-1139.22 | 3.4% | 10 | 14 |
| vllm-mlx | pp512 | 988.90 +- 55.72 | 949.08-1028.72 | 5.6% | 10 | 14 |
| llama.cpp | pp2048 | 1205.73 +- 1.52 | 1204.65-1206.82 | 0.1% | 10 | - |
| lmstudio-mlx | pp2048 | 1255.61 +- 29.81 | 1234.31-1276.91 | 2.4% | 10 | 15 |
| mj | pp2048 | 1248.75 +- 18.54 | 1235.50-1262.00 | 1.5% | 10 | 16 |
| mlx-lm | pp2048 | 1318.75 +- 24.08 | 1301.54-1335.95 | 1.8% | 10 | 15 |
| ollama-gguf | pp2048 | 1145.66 +- 12.99 | 1136.38-1154.95 | 1.1% | 10 | 16 |
| vllm-mlx | pp2048 | 1247.32 +- 20.58 | 1232.61-1262.03 | 1.7% | 10 | 15 |
| llama.cpp | tg128@d0 | 115.55 +- 0.13 | 115.46-115.64 | 0.1% | 10 | - |
| lmstudio-mlx | tg128@d0 | 115.98 +- 6.24 | 111.52-120.44 | 5.4% | 10 | 50 (10 replies under the requested length; tok/s over the tokens produced) |
| mj | tg128@d0 | 137.45 +- 1.30 | 136.53-138.38 | 0.9% | 10 | 67 |
| mlx-lm | tg128@d0 | 135.55 +- 2.14 | 134.02-137.08 | 1.6% | 10 | 67 |
| ollama-gguf | tg128@d0 | 112.59 +- 4.85 | 109.13-116.06 | 4.3% | 10 | 104 |
| vllm-mlx | tg128@d0 | 139.29 +- 0.33 | 139.06-139.52 | 0.2% | 10 | 67 |
| llama.cpp | tg128@d2048 | 88.15 +- 0.98 | 87.45-88.86 | 1.1% | 10 | - |
| lmstudio-mlx | tg128@d2048 | 92.08 +- 3.63 | 89.49-94.68 | 3.9% | 10 | 69 (10 replies under the requested length; tok/s over the tokens produced) |
| mj | tg128@d2048 | 99.79 +- 3.37 | 97.38-102.20 | 3.4% | 10 | 90 |
| mlx-lm | tg128@d2048 | 92.53 +- 2.85 | 90.49-94.57 | 3.1% | 10 | 91 |
| ollama-gguf | tg128@d2048 | 87.64 +- 2.81 | 85.63-89.65 | 3.2% | 10 | 128 |
| vllm-mlx | tg128@d2048 | 98.91 +- 1.72 | 97.68-100.15 | 1.7% | 10 | 88 |

Flagged blocks (excluded above; reason in blocks/*.flagged):

| engine | test | tok/s (flagged reps) | n |
| --- | --- | --- | --- |
| lmstudio-mlx | pp512 | 1090.69 +- 16.46 (1066.28-1103.63) | 5 |
| mj | pp512 | 1158.04 +- 20.94 (1137.04-1189.65) | 5 |
| ollama-gguf | pp512 | 1082.12 +- 66.19 (968.54-1135.71) | 5 |
| lmstudio-mlx | pp2048 | 1252.68 +- 17.84 (1231.36-1270.42) | 5 |
| mj | pp2048 | 1260.72 +- 6.56 (1254.08-1268.50) | 5 |
| ollama-gguf | pp2048 | 1128.07 +- 14.44 (1107.62-1145.50) | 5 |
| lmstudio-mlx | tg128@d0 | 102.46 +- 12.08 (90.01-118.70) | 5 |
| mj | tg128@d0 | 130.28 +- 9.56 (115.26-137.17) | 5 |
| ollama-gguf | tg128@d0 | 115.79 +- 0.63 (114.71-116.30) | 5 |
| lmstudio-mlx | tg128@d2048 | 96.04 +- 2.84 (91.04-97.63) | 5 |
| mj | tg128@d2048 | 100.36 +- 0.40 (99.65-100.59) | 5 |
| ollama-gguf | tg128@d2048 | 86.51 +- 4.10 (79.19-88.61) | 5 |

Greedy decode ids, ours vs mlx-lm (same seeds): 16/20 identical; first divergence at token 8-25 otherwise

## DeepSeek-V2-Lite-Chat (MLA, general MoE model)

Weights: ours / mlx-lm: mlx-community 4-bit g64; ours caches MLA latents, mlx-lm expanded K/V; llama.cpp / Ollama: mradermacher Q4_K_M; LM Studio (its MLX engine) and vllm-mlx: the mlx-community 4-bit

| engine | test | tok/s | 95% CI | CV | n | GPU mJ/token |
| --- | --- | --- | --- | --- | --- | --- |
| llama.cpp | pp512 | 591.48 +- 1.12 | 590.68-592.28 | 0.2% | 10 | - |
| lmstudio-mlx | pp512 | every block flagged (see below) | | | 0 | - |
| mj | pp512 | 572.75 +- 4.94 | 569.22-576.28 | 0.9% | 10 | 33 |
| mlx-lm | pp512 | every block flagged (see below) | | | 0 | - |
| ollama-gguf | pp512 | 381.16 +- 29.01 | 360.42-401.89 | 7.6% | 10 | 30 |
| vllm-mlx | pp512 | 463.25 +- 30.31 | 441.59-484.91 | 6.5% | 10 | 32 |
| llama.cpp | pp2048 | 538.21 +- 15.16 | 527.37-549.04 | 2.8% | 10 | - |
| lmstudio-mlx | pp2048 | every block flagged (see below) | | | 0 | - |
| mj | pp2048 | 574.90 +- 2.56 | 573.08-576.73 | 0.4% | 10 | 35 |
| mlx-lm | pp2048 | every block flagged (see below) | | | 0 | - |
| ollama-gguf | pp2048 | 384.24 +- 8.95 | 377.84-390.64 | 2.3% | 10 | 32 |
| vllm-mlx | pp2048 | 557.05 +- 6.35 | 552.50-561.59 | 1.1% | 10 | 34 |
| llama.cpp | tg128@d0 | 53.38 +- 1.97 | 51.98-54.79 | 3.7% | 10 | - |
| lmstudio-mlx | tg128@d0 | every block flagged (see below) | | | 0 | - |
| mj | tg128@d0 | 61.00 +- 1.34 | 60.04-61.95 | 2.2% | 10 | 160 |
| mlx-lm | tg128@d0 | every block flagged (see below) | | | 0 | - |
| ollama-gguf | tg128@d0 | 45.46 +- 1.79 | 44.17-46.74 | 3.9% | 10 | 177 (9 replies under the requested length; tok/s over the tokens produced) |
| vllm-mlx | tg128@d0 | 66.02 +- 1.49 | 64.96-67.08 | 2.3% | 10 | 159 |
| llama.cpp | tg128@d2048 | 41.02 +- 1.23 | 40.14-41.90 | 3.0% | 10 | - |
| lmstudio-mlx | tg128@d2048 | every block flagged (see below) | | | 0 | - |
| mj | tg128@d2048 | 52.24 +- 0.84 | 51.64-52.84 | 1.6% | 10 | 208 |
| mlx-lm | tg128@d2048 | every block flagged (see below) | | | 0 | - |
| ollama-gguf | tg128@d2048 | 35.01 +- 1.69 | 33.81-36.22 | 4.8% | 10 | 257 (3 replies under the requested length; tok/s over the tokens produced) |
| vllm-mlx | tg128@d2048 | 49.64 +- 0.61 | 49.20-50.08 | 1.2% | 10 | 201 |

Flagged blocks (excluded above; reason in blocks/*.flagged):

| engine | test | tok/s (flagged reps) | n |
| --- | --- | --- | --- |
| lmstudio-mlx | pp512 | 526.59 +- 5.88 (519.68-536.49) | 10 |
| mj | pp512 | 573.69 +- 4.45 (569.77-579.93) | 5 |
| mlx-lm | pp512 | 565.09 +- 2.46 (561.92-570.64) | 10 |
| lmstudio-mlx | pp2048 | 517.82 +- 29.04 (450.35-552.94) | 10 |
| mj | pp2048 | 573.48 +- 1.56 (571.03-575.07) | 5 |
| mlx-lm | pp2048 | 564.65 +- 24.72 (509.03-580.75) | 10 |
| lmstudio-mlx | tg128@d0 | 58.17 +- 0.46 (57.56-58.61) | 8 |
| mj | tg128@d0 | 62.00 +- 0.36 (61.59-62.35) | 5 |
| mlx-lm | tg128@d0 | 67.41 +- 0.10 (67.24-67.54) | 10 |
| lmstudio-mlx | tg128@d2048 | 44.00 +- 0.70 (43.00-44.63) | 6 |
| mj | tg128@d2048 | 52.44 +- 0.05 (52.38-52.50) | 5 |
| mlx-lm | tg128@d2048 | 49.09 +- 1.17 (46.29-49.99) | 10 |

Greedy decode ids, ours vs mlx-lm (same seeds): 13/20 identical; first divergence at token 1-71 otherwise

## gpt-oss-20b

Weights: mj: openai/gpt-oss-20b exact (MXFP4 experts memory-mapped, packed bf16 attention, experts mlocked, 512-token prefill chunks as llama.cpp's ubatch); mj-a8: the same with 8-bit affine attention weights (lossy, matched to the GGUF); llama.cpp / Ollama / LM Studio (its llama.cpp engine): ggml-org MXFP4 GGUF (Q8_0 attention)

| engine | test | tok/s | 95% CI | CV | n | GPU mJ/token |
| --- | --- | --- | --- | --- | --- | --- |
| llama.cpp | pp512 | 429.17 +- 5.87 | 424.98-433.36 | 1.4% | 10 | - |
| lmstudio-gguf | pp512 | 400.82 +- 14.53 | 390.44-411.21 | 3.6% | 10 | 42 |
| mj | pp512 | every block flagged (see below) | | | 0 | - |
| mj-a8 | pp512 | 411.85 +- 3.46 | 409.38-414.32 | 0.8% | 10 | 46 |
| ollama-gguf | pp512 | 394.39 +- 21.48 | 379.04-409.74 | 5.4% | 10 | 42 |
| llama.cpp | pp2048 | 438.76 +- 12.40 | 429.89-447.62 | 2.8% | 10 | - |
| lmstudio-gguf | pp2048 | 436.62 +- 2.67 | 434.71-438.53 | 0.6% | 10 | 44 |
| mj | pp2048 | every block flagged (see below) | | | 0 | - |
| mj-a8 | pp2048 | 434.69 +- 5.47 | 430.78-438.60 | 1.3% | 10 | 49 |
| ollama-gguf | pp2048 | 412.91 +- 14.17 | 402.78-423.04 | 3.4% | 10 | 44 |
| llama.cpp | tg128@d0 | 36.82 +- 0.66 | 36.34-37.29 | 1.8% | 10 | - |
| lmstudio-gguf | tg128@d0 | 35.67 +- 0.50 | 35.31-36.03 | 1.4% | 10 | 360 |
| mj | tg128@d0 | every block flagged (see below) | | | 0 | - |
| mj-a8 | tg128@d0 | 39.67 +- 0.51 | 39.30-40.04 | 1.3% | 10 | 265 |
| ollama-gguf | tg128@d0 | 35.07 +- 0.37 | 34.80-35.33 | 1.1% | 10 | 364 |
| llama.cpp | tg128@d2048 | 35.06 +- 2.07 | 32.49-37.64 | 5.9% | 5 | - |
| lmstudio-gguf | tg128@d2048 | 33.33 +- 0.56 | 32.93-33.73 | 1.7% | 10 | 374 (3 replies under the requested length; tok/s over the tokens produced) |
| mj | tg128@d2048 | every block flagged (see below) | | | 0 | - |
| mj-a8 | tg128@d2048 | 38.33 +- 0.49 | 37.97-38.68 | 1.3% | 10 | 285 |
| ollama-gguf | tg128@d2048 | 34.28 +- 0.56 | 33.88-34.68 | 1.6% | 10 | 377 (2 replies under the requested length; tok/s over the tokens produced) |

Flagged blocks (excluded above; reason in blocks/*.flagged):

| engine | test | tok/s (flagged reps) | n |
| --- | --- | --- | --- |
| llama.cpp | pp512 | 435.13 +- 1.64 (433.30-437.63) | 5 |
| mj | pp512 | 407.23 +- 3.19 (402.38-412.16) | 15 |
| llama.cpp | pp2048 | 431.91 +- 13.43 (417.38-448.80) | 5 |
| mj | pp2048 | 429.11 +- 3.51 (419.96-433.15) | 15 |
| llama.cpp | tg128@d0 | 34.94 +- 2.26 (31.10-36.39) | 5 |
| mj | tg128@d0 | 35.34 +- 2.18 (29.48-36.72) | 15 |
| llama.cpp | tg128@d2048 | 35.04 +- 0.16 (34.85-35.29) | 5 |
| mj | tg128@d2048 | 35.03 +- 0.31 (34.28-35.46) | 10 |
