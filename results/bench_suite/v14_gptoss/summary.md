# Benchmark suite summary

Protocol: results/BENCH_PROTOCOL.md. tok/s: mean +- sd [95% CI], CV, n (timed reps, flagged blocks excluded). Energy: IOReport GPU rail over the test window, estimated (SoC rails); none for llama-bench (its window includes model load).

## gpt-oss-20b

Weights: mj: openai/gpt-oss-20b exact (MXFP4 experts memory-mapped, packed bf16 attention, experts mlocked, 512-token prefill chunks as llama.cpp's ubatch); mj-a8: the same with 8-bit affine attention weights (lossy, matched to the GGUF); llama.cpp / Ollama / LM Studio (its llama.cpp engine): ggml-org MXFP4 GGUF (Q8_0 attention)

| engine | test | tok/s | 95% CI | CV | n | GPU mJ/token |
| --- | --- | --- | --- | --- | --- | --- |
| llama.cpp | pp512 | 428.84 +- 4.15 | 425.87-431.81 | 1.0% | 10 | - |
| lmstudio-gguf | pp512 | 406.59 +- 21.63 | 379.70-433.48 | 5.3% | 5 | 43 |
| mj-a8 | pp512 | 408.14 +- 3.36 | 405.74-410.54 | 0.8% | 10 | 46 |
| mj-a8w | pp512 | 406.31 +- 3.26 | 402.26-410.35 | 0.8% | 5 | 46 |
| mj-x | pp512 | 352.19 +- 108.19 | 274.87-429.50 | 30.7% | 10 | 46 |
| ollama-gguf | pp512 | 394.72 +- 5.78 | 387.54-401.91 | 1.5% | 5 | 43 |
| llama.cpp | pp2048 | 433.50 +- 17.60 | 420.93-446.08 | 4.1% | 10 | - |
| lmstudio-gguf | pp2048 | 434.42 +- 5.68 | 427.36-441.47 | 1.3% | 5 | 44 |
| mj-a8 | pp2048 | 436.82 +- 1.11 | 436.02-437.61 | 0.3% | 10 | 48 |
| mj-a8w | pp2048 | 437.40 +- 1.35 | 435.72-439.09 | 0.3% | 5 | 49 |
| mj-x | pp2048 | 400.75 +- 68.94 | 351.49-450.02 | 17.2% | 10 | 48 |
| ollama-gguf | pp2048 | 410.30 +- 7.21 | 401.34-419.26 | 1.8% | 5 | 44 |
| llama.cpp | tg128@d0 | 36.95 +- 0.12 | 36.86-37.03 | 0.3% | 10 | - |
| lmstudio-gguf | tg128@d0 | 35.76 +- 0.33 | 35.35-36.17 | 0.9% | 5 | 373 |
| mj-a8 | tg128@d0 | 39.41 +- 0.68 | 38.93-39.90 | 1.7% | 10 | 266 |
| mj-a8w | tg128@d0 | 39.80 +- 0.24 | 39.50-40.11 | 0.6% | 5 | 267 |
| mj-x | tg128@d0 | 34.98 +- 3.62 | 32.39-37.56 | 10.3% | 10 | 299 |
| ollama-gguf | tg128@d0 | 35.50 +- 0.15 | 35.31-35.69 | 0.4% | 5 | 368 |
| llama.cpp | tg128@d2048 | 35.09 +- 2.00 | 33.66-36.52 | 5.7% | 10 | - |
| lmstudio-gguf | tg128@d2048 | 33.48 +- 0.19 | 33.24-33.71 | 0.6% | 5 | 400 (2 replies under the requested length; tok/s over the tokens produced) |
| mj-a8 | tg128@d2048 | 38.79 +- 0.28 | 38.59-38.99 | 0.7% | 10 | 296 |
| mj-a8w | tg128@d2048 | 38.60 +- 0.08 | 38.50-38.71 | 0.2% | 5 | 296 |
| mj-x | tg128@d2048 | 35.49 +- 0.21 | 35.34-35.64 | 0.6% | 10 | 323 |
| ollama-gguf | tg128@d2048 | 34.24 +- 0.22 | 33.96-34.51 | 0.7% | 5 | 378 (1 replies under the requested length; tok/s over the tokens produced) |

Flagged blocks (excluded above; reason in blocks/*.flagged):

| engine | test | tok/s (flagged reps) | n |
| --- | --- | --- | --- |
| lmstudio-gguf | pp512 | 414.76 +- 4.63 (408.25-419.19) | 5 |
| mj-a8w | pp512 | 407.76 +- 3.82 (402.79-411.63) | 5 |
| ollama-gguf | pp512 | 399.43 +- 6.22 (392.58-406.31) | 5 |
| lmstudio-gguf | pp2048 | 437.93 +- 1.43 (436.61-439.68) | 5 |
| mj-a8w | pp2048 | 438.16 +- 1.23 (436.05-439.08) | 5 |
| ollama-gguf | pp2048 | 415.77 +- 4.92 (407.61-420.48) | 5 |
| lmstudio-gguf | tg128@d0 | 35.41 +- 0.82 (33.96-35.96) | 5 |
| mj-a8w | tg128@d0 | 39.54 +- 0.51 (39.11-40.11) | 5 |
| ollama-gguf | tg128@d0 | 35.31 +- 0.14 (35.12-35.48) | 5 |
| lmstudio-gguf | tg128@d2048 | 33.39 +- 0.36 (33.04-33.91) | 5 |
| mj-a8w | tg128@d2048 | 38.72 +- 0.13 (38.54-38.87) | 5 |
| ollama-gguf | tg128@d2048 | 32.22 +- 1.61 (30.71-34.33) | 5 |
