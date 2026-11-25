# Benchmark suite summary

Protocol: results/BENCH_PROTOCOL.md. tok/s: mean +- sd [95% CI], CV, n (timed reps, flagged blocks excluded). Energy: IOReport GPU rail over the test window, estimated (SoC rails); none for llama-bench (its window includes model load).

## gpt-oss-20b

Weights: mj: openai/gpt-oss-20b exact (MXFP4 experts memory-mapped, packed bf16 attention, experts mlocked, 512-token prefill chunks as llama.cpp's ubatch); mj-a8: the same with 8-bit affine attention weights (lossy, matched to the GGUF); llama.cpp / Ollama / LM Studio (its llama.cpp engine): ggml-org MXFP4 GGUF (Q8_0 attention)

| engine | test | tok/s | 95% CI | CV | n | GPU mJ/token | machine mJ/token above idle |
| --- | --- | --- | --- | --- | --- | --- | --- |
| llama-server | pp512 | 375.00 +- 11.30 | 360.95-389.04 | 3.0% | 5 | 43 | 50 |
| llama.cpp | pp512 | 430.30 +- 3.32 | 427.93-432.67 | 0.8% | 10 | - | - |
| lmstudio-gguf | pp512 | every block flagged (see below) | | | 0 | - | - |
| mj-a8w | pp512 | 419.80 +- 2.08 | 418.32-421.29 | 0.5% | 10 | 46 | 50 |
| mj-x | pp512 | 392.22 +- 6.25 | 384.46-399.99 | 1.6% | 5 | 45 | 59 |
| ollama-gguf | pp512 | 392.90 +- 9.61 | 380.96-404.85 | 2.4% | 5 | 42 | 52 |
| llama-server | pp2048 | 413.50 +- 20.04 | 388.59-438.42 | 4.8% | 5 | 44 | 72 |
| llama.cpp | pp2048 | 433.92 +- 19.63 | 419.89-447.95 | 4.5% | 10 | - | - |
| lmstudio-gguf | pp2048 | every block flagged (see below) | | | 0 | - | - |
| mj-a8w | pp2048 | 437.43 +- 4.10 | 434.50-440.36 | 0.9% | 10 | 48 | 80 |
| mj-x | pp2048 | 417.43 +- 20.80 | 391.58-443.29 | 5.0% | 5 | 46 | 78 |
| ollama-gguf | pp2048 | 380.14 +- 22.20 | 352.54-407.75 | 5.8% | 5 | 44 | 84 |
| llama-server | tg128@d0 | 35.35 +- 0.94 | 34.18-36.51 | 2.7% | 5 | 368 | 750 |
| llama.cpp | tg128@d0 | 36.41 +- 1.56 | 35.30-37.52 | 4.3% | 10 | - | - |
| lmstudio-gguf | tg128@d0 | every block flagged (see below) | | | 0 | - | - |
| mj-a8w | tg128@d0 | 40.86 +- 0.63 | 40.41-41.31 | 1.5% | 10 | 257 | 500 |
| mj-x | tg128@d0 | 36.63 +- 0.45 | 36.07-37.20 | 1.2% | 5 | 285 | 558 |
| ollama-gguf | tg128@d0 | 29.78 +- 3.15 | 25.87-33.70 | 10.6% | 5 | 367 | 805 |
| llama-server | tg128@d2048 | 34.25 +- 0.79 | 33.27-35.24 | 2.3% | 5 | 369 | 850 |
| llama.cpp | tg128@d2048 | 35.19 +- 1.17 | 34.35-36.03 | 3.3% | 10 | - | - |
| lmstudio-gguf | tg128@d2048 | every block flagged (see below) | | | 0 | - | - |
| mj-a8w | tg128@d2048 | 38.78 +- 0.73 | 38.25-39.30 | 1.9% | 10 | 282 | 553 |
| mj-x | tg128@d2048 | 35.30 +- 0.64 | 34.50-36.09 | 1.8% | 5 | 316 | 749 |
| ollama-gguf | tg128@d2048 | 28.97 +- 5.13 | 22.59-35.35 | 17.7% | 5 | 398 | 1110 (1 replies under the requested length; tok/s over the tokens produced) |

Flagged blocks (excluded above; reason in blocks/*.flagged):

| engine | test | tok/s (flagged reps) | n |
| --- | --- | --- | --- |
| llama-server | pp512 | 399.37 +- 6.01 (390.59-404.67) | 5 |
| lmstudio-gguf | pp512 | 366.09 +- 33.12 (299.07-391.48) | 10 |
| mj-x | pp512 | 408.73 +- 2.10 (405.51-411.39) | 5 |
| ollama-gguf | pp512 | 353.82 +- 43.78 (303.52-394.12) | 5 |
| llama-server | pp2048 | 419.47 +- 9.94 (403.06-427.75) | 5 |
| lmstudio-gguf | pp2048 | 398.87 +- 56.90 (288.71-437.40) | 10 |
| mj-x | pp2048 | 431.17 +- 0.58 (430.19-431.62) | 5 |
| ollama-gguf | pp2048 | 403.23 +- 10.48 (391.02-419.93) | 5 |
| llama-server | tg128@d0 | 35.34 +- 1.12 (33.35-35.90) | 5 |
| lmstudio-gguf | tg128@d0 | 33.22 +- 2.80 (27.69-36.05) | 10 |
| mj-x | tg128@d0 | 36.99 +- 0.65 (36.14-37.51) | 5 |
| ollama-gguf | tg128@d0 | 33.16 +- 1.60 (30.95-34.49) | 5 |
| llama-server | tg128@d2048 | 33.03 +- 3.41 (26.93-34.74) | 5 |
| lmstudio-gguf | tg128@d2048 | 32.48 +- 1.01 (30.04-33.31) | 10 |
| mj-x | tg128@d2048 | 35.74 +- 0.52 (34.81-36.07) | 5 |
| ollama-gguf | tg128@d2048 | 32.78 +- 0.61 (31.81-33.25) | 5 |

Peak memory per block, GB (footprint: macOS's figure, GPU buffers included, mapped files not; resident: pages in RAM, mapped weight files included); swap-ins / -outs over the block:

| block | engine | peak footprint | peak resident | swap-in GB | swap-out GB |
| --- | --- | --- | --- | --- | --- |
| r1_llama-server | llama-server | 1.27 | 11.31 | 4.36 | 5.59 |
| r1_lmstudio-gguf | lmstudio-gguf | 1.70 | 3.49 | 16.90 | 18.47 |
| r1_mj-a8w | mj-a8w | 3.53 | 10.50 | 0.86 | 1.57 |
| r1_mj-x | mj-x | 3.80 | 10.47 | 2.63 | 3.94 |
| r1_ollama-gguf | ollama-gguf | 1.59 | 10.93 | 9.97 | 12.15 |
| r2_llama-server | llama-server | 1.27 | 9.65 | 7.84 | 10.09 |
| r2_lmstudio-gguf | lmstudio-gguf | 1.70 | 5.86 | 3.41 | 4.82 |
| r2_mj-a8w | mj-a8w | 3.53 | 10.37 | 2.81 | 4.25 |
| r2_mj-x | mj-x | 3.79 | 10.65 | 4.68 | 5.33 |
| r2_ollama-gguf | ollama-gguf | 13.40 | 11.73 | 6.46 | 8.80 |
| r1_llama.cpp | llama.cpp | 0.35 | 12.24 | - | - |
| r2_llama.cpp | llama.cpp | 0.33 | 11.31 | - | - |
