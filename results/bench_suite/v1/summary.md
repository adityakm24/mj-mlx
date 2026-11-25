# Benchmark suite summary

Protocol: results/BENCH_PROTOCOL.md. tok/s: mean +- sd [95% CI], CV, n (timed reps, flagged blocks excluded). Energy: IOReport GPU rail over the test window, estimated (SoC rails); none for llama-bench (its window includes model load).

## Qwen3.5-9B

Weights: ours / mlx-lm: mlx-community 4-bit g64 (~4.5 bpw); llama.cpp: lmstudio-community Q4_K_M (~4.8 bpw)

| engine | test | tok/s | 95% CI | CV | n | GPU mJ/token |
| --- | --- | --- | --- | --- | --- | --- |
| llama.cpp | pp512 | 217.53 +- 2.35 | 215.85-219.21 | 1.1% | 10 | - |
| mj | pp512 | 219.11 +- 1.10 | 218.33-219.90 | 0.5% | 10 | 87 |
| mlx-lm | pp512 | 226.88 +- 0.64 | 226.42-227.33 | 0.3% | 10 | 85 |
| llama.cpp | pp2048 | 201.25 +- 2.07 | 199.77-202.73 | 1.0% | 10 | - |
| mj | pp2048 | 222.74 +- 3.74 | 220.07-225.42 | 1.7% | 10 | 89 |
| mlx-lm | pp2048 | 230.72 +- 3.43 | 228.27-233.17 | 1.5% | 10 | 87 |
| llama.cpp | tg128@d0 | 18.85 +- 0.01 | 18.85-18.86 | 0.0% | 10 | - |
| mj | tg128@d0 | 21.64 +- 0.04 | 21.61-21.67 | 0.2% | 10 | 474 |
| mlx-lm | tg128@d0 | 21.83 +- 0.07 | 21.78-21.88 | 0.3% | 10 | 472 |
| llama.cpp | tg128@d2048 | 18.38 +- 0.01 | 18.37-18.39 | 0.1% | 10 | - |
| mj | tg128@d2048 | 20.36 +- 0.46 | 20.03-20.69 | 2.3% | 10 | 502 |
| mlx-lm | tg128@d2048 | 21.51 +- 0.02 | 21.50-21.53 | 0.1% | 10 | 476 |

Greedy decode ids, ours vs mlx-lm (same seeds): 8/20 identical; first divergence at token 0-125 otherwise

## gpt-oss-20b

Weights: ours: openai/gpt-oss-20b exact (MXFP4 experts memory-mapped, packed bf16 attention); llama.cpp: ggml-org MXFP4 GGUF (Q8_0 attention)

| engine | test | tok/s | 95% CI | CV | n | GPU mJ/token |
| --- | --- | --- | --- | --- | --- | --- |
| llama.cpp | pp512 | 431.34 +- 8.92 | 424.96-437.72 | 2.1% | 10 | - |
| mj | pp512 | 69.71 +- 9.72 | 57.63-81.79 | 13.9% | 5 | 37 |
| llama.cpp | pp2048 | 444.40 +- 7.57 | 438.99-449.81 | 1.7% | 10 | - |
| mj | pp2048 | 300.50 +- 27.12 | 266.78-334.22 | 9.0% | 5 | 46 |
| llama.cpp | tg128@d0 | 36.61 +- 0.52 | 36.24-36.98 | 1.4% | 10 | - |
| mj | tg128@d0 | 27.27 +- 1.01 | 26.02-28.52 | 3.7% | 5 | 276 |
| llama.cpp | tg128@d2048 | 35.01 +- 1.28 | 34.10-35.92 | 3.6% | 10 | - |
| mj | tg128@d2048 | 26.38 +- 0.68 | 25.53-27.23 | 2.6% | 5 | 296 |

## Qwen3.6-35B-A3B (20.4 GB, bigger than RAM)

Weights: ours: mlx-community 4-bit g64, experts streamed from the SSD (128 slots/layer, C++ streamer); llama.cpp: Q4_K_M, experts on the CPU (-ncmoe 40)

| engine | test | tok/s | 95% CI | CV | n | GPU mJ/token |
| --- | --- | --- | --- | --- | --- | --- |
| llama.cpp | pp512 | 16.14 +- 0.37 | 15.75-16.52 | 2.3% | 6 | - |
| mj | pp512 | 195.66 +- 5.68 | 188.60-202.72 | 2.9% | 5 | 21 |
| llama.cpp | pp2048 | 13.04 +- 1.10 | 11.89-14.19 | 8.4% | 6 | - |
| mj | pp2048 | 295.52 +- 3.39 | 291.31-299.74 | 1.1% | 5 | 34 |
| llama.cpp | tg128@d0 | 3.85 +- 0.62 | 3.20-4.51 | 16.2% | 6 | - |
| mj | tg128@d0 | 25.95 +- 2.09 | 23.35-28.54 | 8.0% | 5 | 151 |
| llama.cpp | tg128@d2048 | 3.49 +- 0.47 | 2.99-3.98 | 13.5% | 6 | - |
| mj | tg128@d2048 | 21.97 +- 0.59 | 21.23-22.71 | 2.7% | 5 | 233 |
