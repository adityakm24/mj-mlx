# Benchmark suite summary

Protocol: results/BENCH_PROTOCOL.md. tok/s: mean +- sd [95% CI], CV, n (timed reps, flagged blocks excluded). Energy: IOReport GPU rail over the test window, estimated (SoC rails); none for llama-bench (its window includes model load).

## Qwen3.5-9B

Weights: ours / mlx-lm: mlx-community 4-bit g64 (~4.5 bpw); llama.cpp: lmstudio-community Q4_K_M (~4.8 bpw)

| engine | test | tok/s | 95% CI | CV | n | GPU mJ/token |
| --- | --- | --- | --- | --- | --- | --- |
| llama.cpp | pp512 | 216.47 +- 3.69 | 213.83-219.11 | 1.7% | 10 | - |
| mj | pp512 | 219.87 +- 1.13 | 218.46-221.28 | 0.5% | 5 | 86 |
| mlx-lm | pp512 | 227.19 +- 1.96 | 225.79-228.59 | 0.9% | 10 | 85 |
| llama.cpp | pp2048 | 200.21 +- 2.34 | 198.54-201.88 | 1.2% | 10 | - |
| mj | pp2048 | 224.45 +- 1.98 | 221.98-226.91 | 0.9% | 5 | 91 |
| mlx-lm | pp2048 | 226.27 +- 5.37 | 222.43-230.11 | 2.4% | 10 | 84 |
| llama.cpp | tg128@d0 | 18.82 +- 0.03 | 18.80-18.84 | 0.1% | 10 | - |
| mj | tg128@d0 | 21.89 +- 0.04 | 21.84-21.94 | 0.2% | 5 | 471 |
| mlx-lm | tg128@d0 | 21.86 +- 0.17 | 21.74-21.98 | 0.8% | 10 | 430 |
| llama.cpp | tg128@d2048 | 18.36 +- 0.11 | 18.28-18.43 | 0.6% | 10 | - |
| mj | tg128@d2048 | 21.41 +- 0.03 | 21.36-21.45 | 0.2% | 5 | 1034 |
| mlx-lm | tg128@d2048 | 21.46 +- 0.23 | 21.29-21.63 | 1.1% | 10 | 454 |

Flagged blocks (excluded above; reason in blocks/*.flagged):

| engine | test | tok/s (flagged reps) | n |
| --- | --- | --- | --- |
| mj | pp512 | 219.52 +- 0.71 (218.37-220.13) | 5 |
| mj | pp2048 | 218.88 +- 3.80 (215.45-224.73) | 5 |
| mj | tg128@d0 | 21.81 +- 0.02 (21.78-21.83) | 5 |
| mj | tg128@d2048 | 21.37 +- 0.05 (21.31-21.44) | 5 |

Greedy decode ids, ours vs mlx-lm (same seeds): 9/20 identical; first divergence at token 0-120 otherwise
