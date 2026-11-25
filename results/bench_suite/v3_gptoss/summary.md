# Benchmark suite summary

Protocol: results/BENCH_PROTOCOL.md. tok/s: mean +- sd [95% CI], CV, n (timed reps, flagged blocks excluded). Energy: IOReport GPU rail over the test window, estimated (SoC rails); none for llama-bench (its window includes model load).

## gpt-oss-20b

Weights: mj: openai/gpt-oss-20b exact (MXFP4 experts memory-mapped, packed bf16 attention, experts mlocked, 512-token prefill chunks as llama.cpp's ubatch); mj-a8: the same with 8-bit affine attention weights (lossy, matched to the GGUF); llama.cpp: ggml-org MXFP4 GGUF (Q8_0 attention)

| engine | test | tok/s | 95% CI | CV | n | GPU mJ/token |
| --- | --- | --- | --- | --- | --- | --- |
| llama.cpp | pp512 | 432.32 +- 6.21 | 427.88-436.76 | 1.4% | 10 | - |
| mj | pp512 | 372.10 +- 5.07 | 365.80-378.40 | 1.4% | 5 | 46 |
| mj-a8 | pp512 | 383.23 +- 3.65 | 378.70-387.77 | 1.0% | 5 | 46 |
| llama.cpp | pp2048 | 445.73 +- 7.04 | 440.70-450.75 | 1.6% | 10 | - |
| mj | pp2048 | 403.16 +- 2.45 | 400.12-406.20 | 0.6% | 5 | 48 |
| mj-a8 | pp2048 | 416.94 +- 1.01 | 415.68-418.20 | 0.2% | 5 | 48 |
| llama.cpp | tg128@d0 | 36.81 +- 0.50 | 36.46-37.17 | 1.4% | 10 | - |
| mj | tg128@d0 | 29.60 +- 0.42 | 29.08-30.11 | 1.4% | 5 | 293 |
| mj-a8 | tg128@d0 | 36.27 +- 0.15 | 36.08-36.46 | 0.4% | 5 | 192 |
| llama.cpp | tg128@d2048 | 35.87 +- 0.96 | 35.18-36.56 | 2.7% | 10 | - |
| mj | tg128@d2048 | 28.91 +- 0.74 | 27.99-29.83 | 2.6% | 5 | 332 |
| mj-a8 | tg128@d2048 | 34.40 +- 0.53 | 33.74-35.07 | 1.6% | 5 | 217 |

Flagged blocks (excluded above; reason in blocks/*.flagged):

| engine | test | tok/s (flagged reps) | n |
| --- | --- | --- | --- |
| mj | pp512 | 329.98 +- 87.06 (106.95-374.24) | 10 |
| mj-a8 | pp512 | 376.26 +- 2.91 (371.53-379.03) | 5 |
| mj | pp2048 | 403.24 +- 1.96 (399.67-405.89) | 10 |
| mj-a8 | pp2048 | 415.13 +- 2.08 (411.94-416.85) | 5 |
| mj | tg128@d0 | 31.25 +- 2.44 (28.70-33.77) | 10 |
| mj-a8 | tg128@d0 | 36.24 +- 0.19 (36.08-36.56) | 5 |
| mj | tg128@d2048 | 30.80 +- 1.91 (28.81-33.64) | 10 |
| mj-a8 | tg128@d2048 | 34.35 +- 0.29 (34.05-34.82) | 5 |
