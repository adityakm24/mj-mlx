# Benchmark suite summary

Protocol: results/BENCH_PROTOCOL.md. tok/s: mean +- sd [95% CI], CV, n (timed reps, flagged blocks excluded). Energy: IOReport GPU rail over the test window, estimated (SoC rails); none for llama-bench (its window includes model load).

## gpt-oss-20b

Weights: mj: openai/gpt-oss-20b exact (MXFP4 experts memory-mapped, packed bf16 attention, experts mlocked, 512-token prefill chunks as llama.cpp's ubatch); mj-a8: the same with 8-bit affine attention weights (lossy, matched to the GGUF); llama.cpp / Ollama / LM Studio (its llama.cpp engine): ggml-org MXFP4 GGUF (Q8_0 attention)

| engine | test | tok/s | 95% CI | CV | n | GPU mJ/token |
| --- | --- | --- | --- | --- | --- | --- |
| llama.cpp | pp512 | 418.69 +- 35.63 | 393.23-444.16 | 8.5% | 10 | - |
| lmstudio-gguf | pp512 | 421.96 +- 3.11 | 418.09-425.82 | 0.7% | 5 | 43 |
| mj-a8 | pp512 | 403.39 +- 1.68 | 401.31-405.48 | 0.4% | 5 | 46 |
| mj-x | pp512 | 405.64 +- 7.88 | 395.85-415.43 | 1.9% | 5 | 45 |
| ollama-gguf | pp512 | 409.51 +- 17.11 | 397.28-421.73 | 4.2% | 10 | 43 |
| llama.cpp | pp2048 | 437.73 +- 12.66 | 428.68-446.78 | 2.9% | 10 | - |
| lmstudio-gguf | pp2048 | 435.10 +- 2.98 | 431.39-438.81 | 0.7% | 5 | 44 |
| mj-a8 | pp2048 | 438.35 +- 1.09 | 437.01-439.70 | 0.2% | 5 | 48 |
| mj-x | pp2048 | 429.09 +- 1.38 | 427.38-430.80 | 0.3% | 5 | 47 |
| ollama-gguf | pp2048 | 426.70 +- 12.11 | 418.05-435.36 | 2.8% | 10 | 44 |
| llama.cpp | tg128@d0 | 35.83 +- 1.20 | 34.97-36.69 | 3.4% | 10 | - |
| lmstudio-gguf | tg128@d0 | 36.00 +- 0.07 | 35.92-36.08 | 0.2% | 5 | 366 |
| mj-a8 | tg128@d0 | 39.57 +- 0.72 | 38.68-40.47 | 1.8% | 5 | 262 |
| mj-x | tg128@d0 | 36.91 +- 0.51 | 36.28-37.55 | 1.4% | 5 | 290 |
| ollama-gguf | tg128@d0 | 35.20 +- 0.25 | 35.02-35.38 | 0.7% | 10 | 363 |
| llama.cpp | tg128@d2048 | 35.10 +- 1.82 | 33.80-36.40 | 5.2% | 10 | - |
| lmstudio-gguf | tg128@d2048 | 33.21 +- 0.85 | 32.15-34.26 | 2.6% | 5 | 394 (1 replies under the requested length; tok/s over the tokens produced) |
| mj-a8 | tg128@d2048 | 36.64 +- 1.91 | 34.27-39.02 | 5.2% | 5 | 308 |
| mj-x | tg128@d2048 | 35.82 +- 0.03 | 35.79-35.85 | 0.1% | 5 | 324 |
| ollama-gguf | tg128@d2048 | 33.37 +- 1.42 | 32.35-34.38 | 4.3% | 10 | 388 (2 replies under the requested length; tok/s over the tokens produced) |

Flagged blocks (excluded above; reason in blocks/*.flagged):

| engine | test | tok/s (flagged reps) | n |
| --- | --- | --- | --- |
| lmstudio-gguf | pp512 | 422.58 +- 3.98 (417.88-427.73) | 5 |
| mj-a8 | pp512 | 404.90 +- 3.84 (400.99-409.79) | 5 |
| mj-x | pp512 | 401.19 +- 3.87 (396.18-405.75) | 5 |
| lmstudio-gguf | pp2048 | 439.84 +- 1.08 (438.36-440.85) | 5 |
| mj-a8 | pp2048 | 438.09 +- 0.49 (437.46-438.84) | 5 |
| mj-x | pp2048 | 428.49 +- 1.58 (426.78-430.84) | 5 |
| lmstudio-gguf | tg128@d0 | 35.46 +- 0.72 (34.19-35.94) | 5 |
| mj-a8 | tg128@d0 | 40.01 +- 0.37 (39.69-40.62) | 5 |
| mj-x | tg128@d0 | 37.16 +- 0.48 (36.33-37.57) | 5 |
| lmstudio-gguf | tg128@d2048 | 33.72 +- 0.29 (33.27-34.00) | 5 |
| mj-a8 | tg128@d2048 | 38.68 +- 0.67 (37.58-39.16) | 5 |
| mj-x | tg128@d2048 | 35.24 +- 1.44 (32.67-35.98) | 5 |
