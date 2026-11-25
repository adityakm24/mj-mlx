# Benchmark suite summary

Protocol: results/BENCH_PROTOCOL.md. tok/s: mean +- sd [95% CI], CV, n (timed reps, flagged blocks excluded). Energy: IOReport GPU rail over the test window, estimated (SoC rails); none for llama-bench (its window includes model load).

## Qwen3.5-9B

Weights: ours / mlx-lm: mlx-community 4-bit g64 (~4.5 bpw); llama.cpp: lmstudio-community Q4_K_M (~4.8 bpw)

| engine | test | tok/s | 95% CI | CV | n | GPU mJ/token |
| --- | --- | --- | --- | --- | --- | --- |
| mj | pp512 | 226.53 +- 0.28 | 226.24-226.82 | 0.1% | 6 | 87 |
| mj-s64 | pp512 | 226.43 +- 0.38 | 226.03-226.82 | 0.2% | 6 | 87 |
| mj | pp2048 | 227.52 +- 0.07 | 227.44-227.60 | 0.0% | 6 | 90 |
| mj-s64 | pp2048 | 226.21 +- 0.14 | 226.06-226.35 | 0.1% | 6 | 91 |
| mj | tg128@d2048 | 21.49 +- 0.05 | 21.44-21.55 | 0.2% | 6 | 490 |
| mj-s64 | tg128@d2048 | 21.10 +- 0.49 | 20.59-21.62 | 2.3% | 6 | 500 |
| mj | pp8192 | 210.84 +- 1.41 | 209.35-212.32 | 0.7% | 6 | 88 |
| mj-s64 | pp8192 | 211.25 +- 1.27 | 209.91-212.58 | 0.6% | 6 | 88 |
