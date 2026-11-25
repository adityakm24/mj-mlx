# Benchmark suite summary

Protocol: results/BENCH_PROTOCOL.md. tok/s: mean +- sd [95% CI], CV, n (timed reps, flagged blocks excluded). Energy: IOReport GPU rail over the test window, estimated (SoC rails); none for llama-bench (its window includes model load).

## OLMoE-1B-7B-0125-Instruct (general MoE model)

Weights: ours / mlx-lm: mlx-community 4-bit g64 (float16 scales; run in float16); llama.cpp / Ollama (its GGUF runner; its MLX runner rejects OlmoeForCausalLM): allenai Q4_K_M

| engine | test | tok/s | 95% CI | CV | n | GPU mJ/token |
| --- | --- | --- | --- | --- | --- | --- |
| llama.cpp | pp512 | 1267.20 +- 2.84 | 1265.17-1269.23 | 0.2% | 10 | - |
| mj | pp512 | 1202.86 +- 22.12 | 1175.36-1230.35 | 1.8% | 5 | 14 |
| mlx-lm | pp512 | 1227.15 +- 18.71 | 1213.78-1240.52 | 1.5% | 10 | 13 |
| ollama-gguf | pp512 | 1083.72 +- 50.02 | 1047.98-1119.47 | 4.6% | 10 | 14 |
| llama.cpp | pp2048 | 1206.51 +- 2.83 | 1204.49-1208.54 | 0.2% | 10 | - |
| mj | pp2048 | 1266.80 +- 6.13 | 1259.18-1274.41 | 0.5% | 5 | 15 |
| mlx-lm | pp2048 | 1325.51 +- 23.22 | 1308.91-1342.10 | 1.8% | 10 | 14 |
| ollama-gguf | pp2048 | 1263.99 +- 25.08 | 1246.07-1281.92 | 2.0% | 10 | 14 |
| llama.cpp | tg128@d0 | 109.40 +- 6.51 | 104.75-114.05 | 5.9% | 10 | - |
| mj | tg128@d0 | 138.39 +- 0.60 | 137.64-139.13 | 0.4% | 5 | 66 |
| mlx-lm | tg128@d0 | 137.09 +- 1.27 | 136.19-138.00 | 0.9% | 10 | 65 |
| ollama-gguf | tg128@d0 | 116.71 +- 0.46 | 116.38-117.04 | 0.4% | 10 | 103 |
| llama.cpp | tg128@d2048 | 84.87 +- 2.33 | 83.20-86.54 | 2.8% | 10 | - |
| mj | tg128@d2048 | 101.54 +- 0.19 | 101.30-101.77 | 0.2% | 5 | 87 |
| mlx-lm | tg128@d2048 | 94.63 +- 2.04 | 93.17-96.09 | 2.2% | 10 | 88 |
| ollama-gguf | tg128@d2048 | 88.82 +- 0.24 | 88.65-89.00 | 0.3% | 10 | 120 |

Flagged blocks (excluded above; reason in blocks/*.flagged):

| engine | test | tok/s (flagged reps) | n |
| --- | --- | --- | --- |
| mj | pp512 | 1143.25 +- 123.97 (928.71-1237.45) | 5 |
| mj | pp2048 | 1257.02 +- 4.07 (1250.05-1260.05) | 5 |
| mj | tg128@d0 | 137.23 +- 1.28 (135.00-138.20) | 5 |
| mj | tg128@d2048 | 101.14 +- 0.43 (100.39-101.47) | 5 |

Greedy decode ids, ours vs mlx-lm (same seeds): 16/20 identical; first divergence at token 9-119 otherwise

## DeepSeek-V2-Lite-Chat (MLA, general MoE model)

Weights: ours / mlx-lm: mlx-community 4-bit g64; ours caches MLA latents, mlx-lm expanded K/V; llama.cpp / Ollama: mradermacher Q4_K_M

| engine | test | tok/s | 95% CI | CV | n | GPU mJ/token |
| --- | --- | --- | --- | --- | --- | --- |
| llama.cpp | pp512 | 591.12 +- 0.98 | 590.42-591.82 | 0.2% | 10 | - |
| mj | pp512 | 576.92 +- 5.02 | 570.68-583.17 | 0.9% | 5 | 34 |
| mlx-lm | pp512 | every block flagged (see below) | | | 0 | - |
| ollama-gguf | pp512 | 309.17 +- 64.91 | 228.46-389.87 | 21.0% | 5 | 29 |
| llama.cpp | pp2048 | 553.05 +- 11.29 | 544.99-561.12 | 2.0% | 10 | - |
| mj | pp2048 | 574.88 +- 0.74 | 573.96-575.80 | 0.1% | 5 | 35 |
| mlx-lm | pp2048 | every block flagged (see below) | | | 0 | - |
| ollama-gguf | pp2048 | 295.19 +- 22.63 | 267.04-323.33 | 7.7% | 5 | 24 |
| llama.cpp | tg128@d0 | 51.67 +- 1.70 | 50.45-52.89 | 3.3% | 10 | - |
| mj | tg128@d0 | 60.87 +- 0.13 | 60.70-61.03 | 0.2% | 5 | 158 |
| mlx-lm | tg128@d0 | every block flagged (see below) | | | 0 | - |
| ollama-gguf | tg128@d0 | 37.05 +- 6.34 | 29.17-44.93 | 17.1% | 5 | 120 |
| llama.cpp | tg128@d2048 | 41.20 +- 0.42 | 40.89-41.50 | 1.0% | 10 | - |
| mj | tg128@d2048 | 51.62 +- 0.15 | 51.44-51.80 | 0.3% | 5 | 206 |
| mlx-lm | tg128@d2048 | every block flagged (see below) | | | 0 | - |
| ollama-gguf | tg128@d2048 | 23.43 +- 1.46 | 21.61-25.24 | 6.2% | 5 | 191 |

Flagged blocks (excluded above; reason in blocks/*.flagged):

| engine | test | tok/s (flagged reps) | n |
| --- | --- | --- | --- |
| mj | pp512 | 578.70 +- 3.92 (574.31-584.00) | 5 |
| mlx-lm | pp512 | 558.58 +- 3.68 (552.71-562.82) | 10 |
| ollama-gguf | pp512 | 343.39 +- 20.91 (310.14-362.58) | 5 |
| mj | pp2048 | 574.69 +- 1.80 (572.31-577.36) | 5 |
| mlx-lm | pp2048 | 469.34 +- 120.60 (145.63-570.97) | 10 |
| ollama-gguf | pp2048 | 325.48 +- 12.50 (313.02-341.53) | 5 |
| mj | tg128@d0 | 61.09 +- 1.38 (58.64-61.87) | 5 |
| mlx-lm | tg128@d0 | 65.68 +- 0.89 (64.26-67.12) | 10 |
| ollama-gguf | tg128@d0 | 39.75 +- 4.53 (32.33-43.44) | 5 |
| mj | tg128@d2048 | 52.13 +- 0.23 (51.80-52.31) | 5 |
| mlx-lm | tg128@d2048 | 48.40 +- 0.21 (47.93-48.57) | 10 |
| ollama-gguf | tg128@d2048 | 21.59 +- 1.93 (19.63-24.30) | 5 |

Greedy decode ids, ours vs mlx-lm (same seeds): 16/20 identical; first divergence at token 1-88 otherwise

## gpt-oss-20b

Weights: mj: openai/gpt-oss-20b exact (MXFP4 experts memory-mapped, packed bf16 attention, experts mlocked, 512-token prefill chunks as llama.cpp's ubatch); mj-a8: the same with 8-bit affine attention weights (lossy, matched to the GGUF); llama.cpp: ggml-org MXFP4 GGUF (Q8_0 attention)

| engine | test | tok/s | 95% CI | CV | n | GPU mJ/token |
| --- | --- | --- | --- | --- | --- | --- |
| llama.cpp | pp512 | 431.57 +- 5.36 | 427.74-435.40 | 1.2% | 10 | - |
| mj | pp512 | every block flagged (see below) | | | 0 | - |
| mj-a8 | pp512 | 403.94 +- 1.64 | 401.90-405.98 | 0.4% | 5 | 46 |
| ollama-gguf | pp512 | 396.31 +- 30.00 | 359.01-433.61 | 7.6% | 5 | 42 |
| llama.cpp | pp2048 | 437.12 +- 13.20 | 427.69-446.56 | 3.0% | 10 | - |
| mj | pp2048 | every block flagged (see below) | | | 0 | - |
| mj-a8 | pp2048 | 433.34 +- 2.23 | 430.57-436.12 | 0.5% | 5 | 49 |
| ollama-gguf | pp2048 | 515.80 +- 32.28 | 475.67-555.94 | 6.3% | 5 | 33 |
| llama.cpp | tg128@d0 | 35.63 +- 0.49 | 35.28-35.98 | 1.4% | 10 | - |
| mj | tg128@d0 | every block flagged (see below) | | | 0 | - |
| mj-a8 | tg128@d0 | 38.07 +- 0.86 | 37.01-39.14 | 2.2% | 5 | 265 |
| ollama-gguf | tg128@d0 | 34.72 +- 0.41 | 34.20-35.23 | 1.2% | 5 | 360 |
| llama.cpp | tg128@d2048 | 34.25 +- 1.64 | 33.08-35.42 | 4.8% | 10 | - |
| mj | tg128@d2048 | every block flagged (see below) | | | 0 | - |
| mj-a8 | tg128@d2048 | 35.63 +- 1.03 | 34.35-36.91 | 2.9% | 5 | 275 |
| ollama-gguf | tg128@d2048 | 34.12 +- 0.32 | 33.71-34.52 | 0.9% | 5 | 374 |

Flagged blocks (excluded above; reason in blocks/*.flagged):

| engine | test | tok/s (flagged reps) | n |
| --- | --- | --- | --- |
| mj | pp512 | 402.48 +- 9.42 (377.60-409.25) | 10 |
| mj-a8 | pp512 | 410.60 +- 0.76 (409.70-411.62) | 5 |
| ollama-gguf | pp512 | 386.04 +- 15.51 (360.75-399.92) | 5 |
| mj | pp2048 | 428.14 +- 2.99 (422.21-430.90) | 10 |
| mj-a8 | pp2048 | 438.56 +- 2.49 (436.06-442.75) | 5 |
| ollama-gguf | pp2048 | 524.03 +- 10.62 (508.35-535.04) | 5 |
| mj | tg128@d0 | 35.46 +- 1.04 (32.65-36.43) | 10 |
| mj-a8 | tg128@d0 | 39.47 +- 0.26 (39.15-39.78) | 5 |
| ollama-gguf | tg128@d0 | 34.68 +- 0.87 (33.14-35.20) | 5 |
| mj | tg128@d2048 | 34.11 +- 0.54 (33.37-34.98) | 10 |
| mj-a8 | tg128@d2048 | 37.73 +- 0.27 (37.37-38.01) | 5 |
| ollama-gguf | tg128@d2048 | 33.21 +- 2.29 (29.16-34.59) | 5 |
