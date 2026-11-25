# Benchmark suite summary

Protocol: results/BENCH_PROTOCOL.md. tok/s: mean +- sd [95% CI], CV, n (timed reps, flagged blocks excluded). Energy: IOReport GPU rail over the test window, estimated (SoC rails); none for llama-bench (its window includes model load).

## DeepSeek-V2-Lite-Chat (MLA, general MoE model)

Weights: ours / mlx-lm: mlx-community 4-bit g64; ours caches MLA latents, mlx-lm expanded K/V; llama.cpp: mradermacher Q4_K_M

| engine | test | tok/s | 95% CI | CV | n | GPU mJ/token |
| --- | --- | --- | --- | --- | --- | --- |
| llama.cpp | pp512 | every block flagged (see below) | | | 0 | - |
| mj | pp512 | 572.71 +- 5.62 | 565.72-579.71 | 1.0% | 5 | 33 |
| mlx-lm | pp512 | 513.31 +- 127.47 | 422.21-604.41 | 24.8% | 10 | 33 |
| llama.cpp | pp2048 | every block flagged (see below) | | | 0 | - |
| mj | pp2048 | 570.16 +- 2.36 | 567.23-573.09 | 0.4% | 5 | 35 |
| mlx-lm | pp2048 | 340.96 +- 200.49 | 197.67-484.25 | 58.8% | 10 | 33 |
| llama.cpp | tg128@d0 | every block flagged (see below) | | | 0 | - |
| mj | tg128@d0 | 58.40 +- 2.67 | 55.09-61.72 | 4.6% | 5 | 165 |
| mlx-lm | tg128@d0 | 65.67 +- 1.37 | 64.69-66.65 | 2.1% | 10 | 154 |
| llama.cpp | tg128@d2048 | every block flagged (see below) | | | 0 | - |
| mj | tg128@d2048 | 51.08 +- 0.91 | 49.94-52.21 | 1.8% | 5 | 211 |
| mlx-lm | tg128@d2048 | 47.15 +- 2.46 | 45.39-48.91 | 5.2% | 10 | 195 |

Flagged blocks (excluded above; reason in blocks/*.flagged):

| engine | test | tok/s (flagged reps) | n |
| --- | --- | --- | --- |
| llama.cpp | pp512 | 509.39 +- 56.23 (397.05-565.01) | 10 |
| mj | pp512 | 571.11 +- 3.52 (567.79-576.74) | 5 |
| llama.cpp | pp2048 | 524.64 +- 28.98 (457.06-553.04) | 10 |
| mj | pp2048 | 573.12 +- 1.52 (571.45-574.58) | 5 |
| llama.cpp | tg128@d0 | 51.06 +- 0.95 (48.99-52.67) | 10 |
| mj | tg128@d0 | 61.78 +- 0.07 (61.69-61.87) | 5 |
| llama.cpp | tg128@d2048 | 38.27 +- 1.41 (35.37-40.01) | 10 |
| mj | tg128@d2048 | 51.89 +- 0.35 (51.34-52.24) | 5 |

Greedy decode ids, ours vs mlx-lm (same seeds): 16/20 identical; first divergence at token 1-88 otherwise

## OLMoE-1B-7B-0125-Instruct (general MoE model)

Weights: ours / mlx-lm: mlx-community 4-bit g64 (float16 scales; run in float16); llama.cpp: allenai Q4_K_M

| engine | test | tok/s | 95% CI | CV | n | GPU mJ/token |
| --- | --- | --- | --- | --- | --- | --- |
| llama.cpp | pp512 | 1264.78 +- 8.01 | 1259.05-1270.50 | 0.6% | 10 | - |
| mj | pp512 | 1189.54 +- 11.06 | 1175.79-1203.30 | 0.9% | 5 | 14 |
| mlx-lm | pp512 | 1221.17 +- 14.27 | 1210.98-1231.37 | 1.2% | 10 | 14 |
| llama.cpp | pp2048 | 1206.59 +- 3.47 | 1204.11-1209.07 | 0.3% | 10 | - |
| mj | pp2048 | 1263.35 +- 8.34 | 1252.98-1273.72 | 0.7% | 5 | 15 |
| mlx-lm | pp2048 | 1318.79 +- 11.82 | 1310.35-1327.24 | 0.9% | 10 | 14 |
| llama.cpp | tg128@d0 | 102.41 +- 3.08 | 100.21-104.61 | 3.0% | 10 | - |
| mj | tg128@d0 | 138.01 +- 0.60 | 137.25-138.76 | 0.4% | 5 | 67 |
| mlx-lm | tg128@d0 | 136.02 +- 1.56 | 134.91-137.13 | 1.1% | 10 | 66 |
| llama.cpp | tg128@d2048 | 84.06 +- 2.09 | 82.57-85.56 | 2.5% | 10 | - |
| mj | tg128@d2048 | 99.59 +- 4.28 | 94.27-104.91 | 4.3% | 5 | 89 |
| mlx-lm | tg128@d2048 | 95.49 +- 0.37 | 95.22-95.75 | 0.4% | 10 | 88 |

Flagged blocks (excluded above; reason in blocks/*.flagged):

| engine | test | tok/s (flagged reps) | n |
| --- | --- | --- | --- |
| mj | pp512 | 1176.04 +- 4.35 (1172.50-1182.47) | 5 |
| mj | pp2048 | 1246.31 +- 5.25 (1238.06-1251.44) | 5 |
| mj | tg128@d0 | 106.76 +- 3.18 (101.31-109.49) | 5 |
| mj | tg128@d2048 | 90.62 +- 9.38 (78.70-100.50) | 5 |

Greedy decode ids, ours vs mlx-lm (same seeds): 16/20 identical; first divergence at token 9-119 otherwise
