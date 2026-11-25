# Benchmark suite summary

Protocol: results/BENCH_PROTOCOL.md. tok/s: mean +- sd [95% CI], CV, n (timed reps, flagged blocks excluded). Energy: IOReport GPU rail over the test window, estimated (SoC rails); none for llama-bench (its window includes model load).

## gpt-oss-20b

Weights: mj: openai/gpt-oss-20b exact (MXFP4 experts memory-mapped, packed bf16 attention, experts mlocked, 512-token prefill chunks as llama.cpp's ubatch); mj-a8: the same with 8-bit affine attention weights (lossy, matched to the GGUF); llama.cpp: ggml-org MXFP4 GGUF (Q8_0 attention)

| engine | test | tok/s | 95% CI | CV | n | GPU mJ/token |
| --- | --- | --- | --- | --- | --- | --- |
| llama.cpp | pp512 | 437.42 +- 2.36 | 434.48-440.36 | 0.5% | 5 | - |
| mj | pp512 | 372.01 +- 5.56 | 365.10-378.92 | 1.5% | 5 | 45 |
| mj-a8 | pp512 | 370.40 +- 2.44 | 367.37-373.43 | 0.7% | 5 | 46 |
| llama.cpp | pp2048 | 442.32 +- 7.25 | 433.31-451.34 | 1.6% | 5 | - |
| mj | pp2048 | 401.61 +- 3.70 | 397.01-406.21 | 0.9% | 5 | 47 |
| mj-a8 | pp2048 | 416.42 +- 2.06 | 413.87-418.98 | 0.5% | 5 | 49 |
| llama.cpp | tg128@d0 | 37.20 +- 0.06 | 37.13-37.28 | 0.2% | 5 | - |
| mj | tg128@d0 | 28.80 +- 0.46 | 28.23-29.37 | 1.6% | 5 | 295 |
| mj-a8 | tg128@d0 | 36.01 +- 0.71 | 35.12-36.89 | 2.0% | 5 | 199 |
| llama.cpp | tg128@d2048 | 35.74 +- 0.84 | 34.69-36.78 | 2.4% | 5 | - |
| mj | tg128@d2048 | 27.89 +- 0.45 | 27.33-28.45 | 1.6% | 5 | 332 |
| mj-a8 | tg128@d2048 | 28.02 +- 0.35 | 27.59-28.45 | 1.2% | 5 | 301 |

Flagged blocks (excluded above; reason in blocks/*.flagged):

| engine | test | tok/s (flagged reps) | n |
| --- | --- | --- | --- |
| llama.cpp | pp512 | 427.85 +- 8.30 (416.13-436.62) | 5 |
| mj | pp512 | 366.38 +- 2.13 (362.61-367.79) | 5 |
| mj-a8 | pp512 | 359.59 +- 15.79 (345.24-376.68) | 5 |
| llama.cpp | pp2048 | 447.33 +- 3.70 (441.95-452.31) | 5 |
| mj | pp2048 | 400.02 +- 1.68 (398.17-402.37) | 5 |
| mj-a8 | pp2048 | 404.23 +- 5.86 (397.70-410.61) | 5 |
| llama.cpp | tg128@d0 | 36.71 +- 0.39 (36.02-36.98) | 5 |
| mj | tg128@d0 | 28.09 +- 0.37 (27.73-28.64) | 5 |
| mj-a8 | tg128@d0 | 34.45 +- 0.69 (33.31-35.11) | 5 |
| llama.cpp | tg128@d2048 | 35.07 +- 2.44 (30.72-36.51) | 5 |
| mj | tg128@d2048 | 27.58 +- 0.58 (26.67-28.13) | 5 |
| mj-a8 | tg128@d2048 | 32.94 +- 0.52 (32.16-33.36) | 5 |
