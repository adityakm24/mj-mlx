# Benchmark suite summary

Protocol: results/BENCH_PROTOCOL.md. tok/s: mean +- sd [95% CI], CV, n (timed reps, flagged blocks excluded). Energy: IOReport GPU rail over the test window, estimated (SoC rails); none for llama-bench (its window includes model load).

## Qwen3.5-9B

Weights: ours / mlx-lm: mlx-community 4-bit g64 (~4.5 bpw); llama.cpp: lmstudio-community Q4_K_M (~4.8 bpw)

| engine | test | tok/s | 95% CI | CV | n | GPU mJ/token |
| --- | --- | --- | --- | --- | --- | --- |
| llama.cpp | pp512 | 216.49 +- 3.63 | 213.89-219.09 | 1.7% | 10 | - |
| mj | pp512 | 217.47 +- 1.84 | 216.15-218.79 | 0.8% | 10 | 88 |
| mlx-lm | pp512 | 227.01 +- 0.84 | 226.41-227.61 | 0.4% | 10 | 86 |
| llama.cpp | pp2048 | 200.14 +- 2.58 | 198.30-201.98 | 1.3% | 10 | - |
| mj | pp2048 | 220.16 +- 3.33 | 217.78-222.54 | 1.5% | 10 | 88 |
| mlx-lm | pp2048 | 228.70 +- 3.79 | 225.99-231.41 | 1.7% | 10 | 85 |
| llama.cpp | tg128@d0 | 18.75 +- 0.16 | 18.64-18.87 | 0.8% | 10 | - |
| mj | tg128@d0 | 21.53 +- 0.21 | 21.38-21.68 | 1.0% | 10 | 481 |
| mlx-lm | tg128@d0 | 21.56 +- 0.50 | 21.21-21.92 | 2.3% | 10 | 480 |
| llama.cpp | tg128@d2048 | 18.36 +- 0.01 | 18.36-18.37 | 0.1% | 10 | - |
| mj | tg128@d2048 | 21.09 +- 0.22 | 20.93-21.25 | 1.1% | 10 | 468 |
| mlx-lm | tg128@d2048 | 21.49 +- 0.08 | 21.44-21.55 | 0.4% | 10 | 456 |

Greedy decode ids, ours vs mlx-lm (same seeds): 9/20 identical; first divergence at token 0-120 otherwise
