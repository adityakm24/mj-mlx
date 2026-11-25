# Benchmark suite summary

Protocol: results/BENCH_PROTOCOL.md. tok/s: mean +- sd [95% CI], CV, n (timed reps, flagged blocks excluded). Energy: IOReport GPU rail over the test window, estimated (SoC rails); none for llama-bench (its window includes model load).

## gpt-oss-20b

Weights: mj: openai/gpt-oss-20b exact (MXFP4 experts memory-mapped, packed bf16 attention, experts mlocked, 512-token prefill chunks as llama.cpp's ubatch); mj-a8: the same with 8-bit affine attention weights (lossy, matched to the GGUF); llama.cpp / Ollama / LM Studio (its llama.cpp engine): ggml-org MXFP4 GGUF (Q8_0 attention)

| engine | test | tok/s | 95% CI | CV | n | GPU mJ/token |
| --- | --- | --- | --- | --- | --- | --- |
| llama.cpp | pp512 | 446.74 +- 18.85 | 433.27-460.22 | 4.2% | 10 | - |
| lmstudio-gguf | pp512 | 420.01 +- 2.64 | 416.72-423.30 | 0.6% | 5 | 42 |
| mj-a8 | pp512 | 407.00 +- 2.80 | 403.52-410.49 | 0.7% | 5 | 46 |
| ollama-gguf | pp512 | 415.07 +- 26.54 | 382.07-448.06 | 6.4% | 5 | 42 |
| llama.cpp | pp2048 | 444.21 +- 10.78 | 436.51-451.91 | 2.4% | 10 | - |
| lmstudio-gguf | pp2048 | 439.94 +- 0.38 | 439.46-440.42 | 0.1% | 5 | 44 |
| mj-a8 | pp2048 | 439.12 +- 0.97 | 437.91-440.33 | 0.2% | 5 | 48 |
| ollama-gguf | pp2048 | 435.57 +- 2.49 | 432.48-438.67 | 0.6% | 5 | 44 |
| llama.cpp | tg128@d0 | 37.51 +- 0.22 | 37.36-37.67 | 0.6% | 10 | - |
| lmstudio-gguf | tg128@d0 | 35.68 +- 0.71 | 34.80-36.57 | 2.0% | 5 | 359 |
| mj-a8 | tg128@d0 | 39.99 +- 0.37 | 39.53-40.45 | 0.9% | 5 | 264 |
| ollama-gguf | tg128@d0 | 33.21 +- 3.14 | 29.30-37.11 | 9.5% | 5 | 363 |
| llama.cpp | tg128@d2048 | 35.95 +- 0.51 | 35.58-36.31 | 1.4% | 10 | - |
| lmstudio-gguf | tg128@d2048 | 33.74 +- 0.63 | 32.96-34.52 | 1.9% | 5 | 397 (2 replies under the requested length; tok/s over the tokens produced) |
| mj-a8 | tg128@d2048 | 38.77 +- 0.35 | 38.33-39.20 | 0.9% | 5 | 293 |
| ollama-gguf | tg128@d2048 | 34.33 +- 0.30 | 33.96-34.70 | 0.9% | 5 | 382 (1 replies under the requested length; tok/s over the tokens produced) |

Flagged blocks (excluded above; reason in blocks/*.flagged):

| engine | test | tok/s (flagged reps) | n |
| --- | --- | --- | --- |
| lmstudio-gguf | pp512 | 407.30 +- 14.77 (383.32-419.02) | 5 |
| mj-a8 | pp512 | 409.14 +- 2.21 (406.36-411.58) | 5 |
| ollama-gguf | pp512 | 398.97 +- 7.79 (389.51-409.13) | 5 |
| lmstudio-gguf | pp2048 | 435.08 +- 2.82 (431.14-438.49) | 5 |
| mj-a8 | pp2048 | 438.57 +- 1.26 (436.70-440.13) | 5 |
| ollama-gguf | pp2048 | 409.16 +- 6.86 (403.48-420.78) | 5 |
| lmstudio-gguf | tg128@d0 | 35.39 +- 0.65 (34.31-36.01) | 5 |
| mj-a8 | tg128@d0 | 40.21 +- 0.05 (40.14-40.27) | 5 |
| ollama-gguf | tg128@d0 | 34.15 +- 0.58 (33.25-34.86) | 5 |
| lmstudio-gguf | tg128@d2048 | 33.71 +- 0.31 (33.39-34.06) | 5 |
| mj-a8 | tg128@d2048 | 39.33 +- 0.06 (39.25-39.40) | 5 |
| ollama-gguf | tg128@d2048 | 33.25 +- 1.12 (31.68-34.33) | 5 |
