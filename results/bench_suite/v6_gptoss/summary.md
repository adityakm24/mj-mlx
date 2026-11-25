# Benchmark suite summary

Protocol: results/BENCH_PROTOCOL.md. tok/s: mean +- sd [95% CI], CV, n (timed reps, flagged blocks excluded). Energy: IOReport GPU rail over the test window, estimated (SoC rails); none for llama-bench (its window includes model load).

## gpt-oss-20b

Weights: mj: openai/gpt-oss-20b exact (MXFP4 experts memory-mapped, packed bf16 attention, experts mlocked, 512-token prefill chunks as llama.cpp's ubatch); mj-a8: the same with 8-bit affine attention weights (lossy, matched to the GGUF); llama.cpp: ggml-org MXFP4 GGUF (Q8_0 attention)

| engine | test | tok/s | 95% CI | CV | n | GPU mJ/token |
| --- | --- | --- | --- | --- | --- | --- |
| llama.cpp | pp512 | 428.57 +- 10.19 | 415.90-441.25 | 2.4% | 5 | - |
| mj | pp512 | every block flagged (see below) | | | 0 | - |
| mj-a8 | pp512 | 400.80 +- 1.71 | 398.67-402.92 | 0.4% | 5 | 46 |
| llama.cpp | pp2048 | 441.17 +- 5.82 | 433.93-448.41 | 1.3% | 5 | - |
| mj | pp2048 | every block flagged (see below) | | | 0 | - |
| mj-a8 | pp2048 | 435.65 +- 3.09 | 431.81-439.49 | 0.7% | 5 | 48 |
| llama.cpp | tg128@d0 | 35.38 +- 0.14 | 35.20-35.56 | 0.4% | 5 | - |
| mj | tg128@d0 | every block flagged (see below) | | | 0 | - |
| mj-a8 | tg128@d0 | 39.46 +- 0.33 | 39.05-39.88 | 0.8% | 5 | 263 |
| llama.cpp | tg128@d2048 | 35.46 +- 0.41 | 34.95-35.97 | 1.2% | 5 | - |
| mj | tg128@d2048 | every block flagged (see below) | | | 0 | - |
| mj-a8 | tg128@d2048 | 38.09 +- 0.07 | 38.01-38.17 | 0.2% | 5 | 292 |

Flagged blocks (excluded above; reason in blocks/*.flagged):

| engine | test | tok/s (flagged reps) | n |
| --- | --- | --- | --- |
| llama.cpp | pp512 | 432.42 +- 4.79 (426.14-439.25) | 5 |
| mj | pp512 | 386.62 +- 10.04 (370.57-402.06) | 10 |
| mj-a8 | pp512 | 398.22 +- 1.98 (396.25-401.47) | 5 |
| llama.cpp | pp2048 | 434.31 +- 13.50 (419.38-446.74) | 5 |
| mj | pp2048 | 423.93 +- 4.68 (415.20-427.99) | 10 |
| mj-a8 | pp2048 | 429.95 +- 2.87 (425.12-432.11) | 5 |
| llama.cpp | tg128@d0 | 35.98 +- 0.60 (35.09-36.59) | 5 |
| mj | tg128@d0 | 35.91 +- 0.85 (33.71-36.71) | 10 |
| mj-a8 | tg128@d0 | 39.68 +- 0.06 (39.62-39.77) | 5 |
| mj | tg128@d2048 | 34.56 +- 0.77 (32.95-35.17) | 10 |
| mj-a8 | tg128@d2048 | 38.04 +- 0.21 (37.67-38.16) | 5 |
