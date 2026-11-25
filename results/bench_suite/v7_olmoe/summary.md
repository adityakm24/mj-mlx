# Benchmark suite summary

Protocol: results/BENCH_PROTOCOL.md. tok/s: mean +- sd [95% CI], CV, n (timed reps, flagged blocks excluded). Energy: IOReport GPU rail over the test window, estimated (SoC rails); none for llama-bench (its window includes model load).

## OLMoE-1B-7B-0125-Instruct (general MoE model)

Weights: ours / mlx-lm: mlx-community 4-bit g64 (float16 scales; run in float16); llama.cpp: allenai Q4_K_M

| engine | test | tok/s | 95% CI | CV | n | GPU mJ/token |
| --- | --- | --- | --- | --- | --- | --- |
| llama.cpp | pp512 | 1260.29 +- 8.67 | 1254.09-1266.49 | 0.7% | 10 | - |
| mj | pp512 | 1032.83 +- 15.48 | 1021.77-1043.89 | 1.5% | 10 | 14 |
| mlx-lm | pp512 | 1207.92 +- 8.37 | 1201.94-1213.90 | 0.7% | 10 | 13 |
| llama.cpp | pp2048 | 1207.92 +- 2.21 | 1206.34-1209.50 | 0.2% | 10 | - |
| mj | pp2048 | 1184.31 +- 3.56 | 1181.77-1186.86 | 0.3% | 10 | 15 |
| mlx-lm | pp2048 | 1322.39 +- 10.39 | 1314.97-1329.82 | 0.8% | 10 | 14 |
| llama.cpp | tg128@d0 | 101.68 +- 2.40 | 99.97-103.40 | 2.4% | 10 | - |
| mj | tg128@d0 | 136.78 +- 0.45 | 136.46-137.10 | 0.3% | 10 | 66 |
| mlx-lm | tg128@d0 | 136.33 +- 1.57 | 135.21-137.45 | 1.2% | 10 | 66 |
| llama.cpp | tg128@d2048 | 83.72 +- 1.68 | 82.52-84.92 | 2.0% | 10 | - |
| mj | tg128@d2048 | 100.00 +- 1.08 | 99.23-100.77 | 1.1% | 10 | 86 |
| mlx-lm | tg128@d2048 | 94.79 +- 2.03 | 93.34-96.24 | 2.1% | 10 | 87 |

Greedy decode ids, ours vs mlx-lm (same seeds): 16/20 identical; first divergence at token 9-119 otherwise
