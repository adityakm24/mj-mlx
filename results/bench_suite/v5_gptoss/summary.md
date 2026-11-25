# Benchmark suite summary

Protocol: results/BENCH_PROTOCOL.md. tok/s: mean +- sd [95% CI], CV, n (timed reps, flagged blocks excluded). Energy: IOReport GPU rail over the test window, estimated (SoC rails); none for llama-bench (its window includes model load).

## gpt-oss-20b

Weights: mj: openai/gpt-oss-20b exact (MXFP4 experts memory-mapped, packed bf16 attention, experts mlocked, 512-token prefill chunks as llama.cpp's ubatch); mj-a8: the same with 8-bit affine attention weights (lossy, matched to the GGUF); llama.cpp: ggml-org MXFP4 GGUF (Q8_0 attention)

| engine | test | tok/s | 95% CI | CV | n | GPU mJ/token |
| --- | --- | --- | --- | --- | --- | --- |
| llama.cpp | pp512 | 434.95 +- 1.61 | 432.95-436.96 | 0.4% | 5 | - |
| mj | pp512 | every block flagged (see below) | | | 0 | - |
| mj-a8 | pp512 | 377.45 +- 3.34 | 375.06-379.83 | 0.9% | 10 | 46 |
| llama.cpp | pp2048 | 439.70 +- 6.95 | 431.07-448.34 | 1.6% | 5 | - |
| mj | pp2048 | every block flagged (see below) | | | 0 | - |
| mj-a8 | pp2048 | 411.38 +- 2.29 | 409.74-413.01 | 0.6% | 10 | 49 |
| llama.cpp | tg128@d0 | 36.90 +- 0.12 | 36.75-37.04 | 0.3% | 5 | - |
| mj | tg128@d0 | every block flagged (see below) | | | 0 | - |
| mj-a8 | tg128@d0 | 37.35 +- 0.80 | 36.78-37.92 | 2.1% | 10 | 198 |
| llama.cpp | tg128@d2048 | 33.85 +- 1.91 | 31.48-36.22 | 5.6% | 5 | - |
| mj | tg128@d2048 | every block flagged (see below) | | | 0 | - |
| mj-a8 | tg128@d2048 | 34.39 +- 1.39 | 33.39-35.38 | 4.0% | 10 | 271 |

Flagged blocks (excluded above; reason in blocks/*.flagged):

| engine | test | tok/s (flagged reps) | n |
| --- | --- | --- | --- |
| llama.cpp | pp512 | 432.36 +- 6.48 (421.63-438.66) | 5 |
| mj | pp512 | 367.78 +- 10.52 (345.81-378.75) | 10 |
| llama.cpp | pp2048 | 431.35 +- 8.60 (420.36-443.72) | 5 |
| mj | pp2048 | 404.04 +- 0.54 (403.17-404.70) | 10 |
| llama.cpp | tg128@d0 | 36.34 +- 0.37 (35.80-36.81) | 5 |
| mj | tg128@d0 | 32.00 +- 2.98 (28.18-35.07) | 10 |
| llama.cpp | tg128@d2048 | 34.84 +- 0.65 (33.87-35.66) | 5 |
| mj | tg128@d2048 | 31.29 +- 2.51 (28.64-33.75) | 10 |
