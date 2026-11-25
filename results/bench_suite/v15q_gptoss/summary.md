# Benchmark suite summary

Protocol: results/BENCH_PROTOCOL.md. tok/s: mean +- sd [95% CI], CV, n (timed reps, flagged blocks excluded). Energy: IOReport GPU rail over the test window, estimated (SoC rails); none for llama-bench (its window includes model load).

## gpt-oss-20b

Weights: mj: openai/gpt-oss-20b exact (MXFP4 experts memory-mapped, packed bf16 attention, experts mlocked, 512-token prefill chunks as llama.cpp's ubatch); mj-a8: the same with 8-bit affine attention weights (lossy, matched to the GGUF); llama.cpp / Ollama / LM Studio (its llama.cpp engine): ggml-org MXFP4 GGUF (Q8_0 attention)

| engine | test | tok/s | 95% CI | CV | n | GPU mJ/token | machine mJ/token above idle |
| --- | --- | --- | --- | --- | --- | --- | --- |
| llama-server | pp512 | 387.50 +- 7.55 | 378.11-396.89 | 1.9% | 5 | 42 | 54 |
| llama.cpp | pp512 | every block flagged (see below) | | | 0 | - | - |
| lmstudio-gguf | pp512 | 403.93 +- 11.69 | 389.39-418.47 | 2.9% | 5 | 42 | 50 |
| mj-a8w | pp512 | 407.06 +- 2.00 | 404.57-409.55 | 0.5% | 5 | 46 | 56 |
| mj-x | pp512 | 398.55 +- 3.34 | 394.40-402.70 | 0.8% | 5 | 45 | 55 |
| ollama-gguf | pp512 | 350.89 +- 12.21 | 335.70-366.07 | 3.5% | 5 | 42 | 63 |
| llama-server | pp2048 | 424.61 +- 4.67 | 418.81-430.41 | 1.1% | 5 | 44 | 83 |
| llama.cpp | pp2048 | every block flagged (see below) | | | 0 | - | - |
| lmstudio-gguf | pp2048 | 431.80 +- 5.54 | 424.91-438.69 | 1.3% | 5 | 43 | 69 |
| mj-a8w | pp2048 | 435.06 +- 5.59 | 428.11-442.01 | 1.3% | 5 | 47 | 80 |
| mj-x | pp2048 | 423.88 +- 3.56 | 419.45-428.30 | 0.8% | 5 | 47 | 89 |
| ollama-gguf | pp2048 | 377.72 +- 20.08 | 352.75-402.69 | 5.3% | 5 | 43 | 52 |
| llama-server | tg128@d0 | 35.91 +- 0.73 | 35.01-36.82 | 2.0% | 5 | 350 | 533 |
| llama.cpp | tg128@d0 | every block flagged (see below) | | | 0 | - | - |
| lmstudio-gguf | tg128@d0 | 35.69 +- 0.65 | 34.89-36.50 | 1.8% | 5 | 348 | 782 |
| mj-a8w | tg128@d0 | 40.20 +- 0.41 | 39.68-40.71 | 1.0% | 5 | 253 | 512 |
| mj-x | tg128@d0 | 34.79 +- 1.56 | 32.85-36.73 | 4.5% | 5 | 291 | 564 |
| ollama-gguf | tg128@d0 | 34.26 +- 2.22 | 31.50-37.02 | 6.5% | 5 | 352 | 845 |
| llama-server | tg128@d2048 | 35.98 +- 0.24 | 35.69-36.28 | 0.7% | 5 | 380 | 865 |
| llama.cpp | tg128@d2048 | every block flagged (see below) | | | 0 | - | - |
| lmstudio-gguf | tg128@d2048 | 33.62 +- 0.28 | 33.28-33.96 | 0.8% | 5 | 385 | 877 (1 replies under the requested length; tok/s over the tokens produced) |
| mj-a8w | tg128@d2048 | 38.77 +- 0.32 | 38.38-39.17 | 0.8% | 5 | 279 | 646 |
| mj-x | tg128@d2048 | 35.44 +- 1.01 | 34.18-36.70 | 2.9% | 5 | 316 | 742 |
| ollama-gguf | tg128@d2048 | 34.48 +- 0.22 | 34.20-34.76 | 0.6% | 5 | 376 | 861 (1 replies under the requested length; tok/s over the tokens produced) |

Flagged blocks (excluded above; reason in blocks/*.flagged):

| engine | test | tok/s (flagged reps) | n |
| --- | --- | --- | --- |
| llama-server | pp512 | 391.20 +- 8.09 (380.99-400.80) | 5 |
| llama.cpp | pp512 | 407.32 +- 34.06 (339.02-435.83) | 10 |
| lmstudio-gguf | pp512 | 389.69 +- 9.54 (381.85-405.97) | 5 |
| mj-a8w | pp512 | 403.37 +- 3.35 (398.40-406.60) | 5 |
| mj-x | pp512 | 389.29 +- 8.82 (375.54-398.75) | 5 |
| ollama-gguf | pp512 | 391.93 +- 9.11 (378.67-400.60) | 5 |
| llama-server | pp2048 | 424.92 +- 2.41 (421.80-427.56) | 5 |
| llama.cpp | pp2048 | 362.65 +- 95.20 (202.42-446.01) | 10 |
| lmstudio-gguf | pp2048 | 436.73 +- 3.08 (432.35-440.48) | 5 |
| mj-a8w | pp2048 | 423.98 +- 7.69 (412.54-431.70) | 5 |
| mj-x | pp2048 | 369.58 +- 93.59 (208.19-427.29) | 5 |
| ollama-gguf | pp2048 | 404.78 +- 7.78 (391.67-411.42) | 5 |
| llama-server | tg128@d0 | 35.95 +- 0.79 (34.87-36.59) | 5 |
| llama.cpp | tg128@d0 | 36.02 +- 1.27 (32.74-37.10) | 10 |
| lmstudio-gguf | tg128@d0 | 35.49 +- 1.03 (33.64-36.02) | 5 |
| mj-a8w | tg128@d0 | 40.91 +- 1.21 (38.75-41.60) | 5 |
| mj-x | tg128@d0 | 31.36 +- 0.17 (31.09-31.52) | 5 |
| ollama-gguf | tg128@d0 | 35.52 +- 0.18 (35.37-35.82) | 5 |
| llama-server | tg128@d2048 | 35.95 +- 0.11 (35.82-36.05) | 5 |
| llama.cpp | tg128@d2048 | 28.29 +- 12.43 (4.20-36.10) | 10 |
| lmstudio-gguf | tg128@d2048 | 33.69 +- 0.45 (33.12-34.31) | 5 |
| mj-a8w | tg128@d2048 | 39.02 +- 0.30 (38.56-39.31) | 5 |
| mj-x | tg128@d2048 | 30.99 +- 1.35 (29.45-32.88) | 5 |
| ollama-gguf | tg128@d2048 | 34.67 +- 0.32 (34.16-34.96) | 5 |

Peak memory per block, GB (footprint: macOS's figure, GPU buffers included, mapped files not; resident: pages in RAM, mapped weight files included); swap-ins / -outs over the block:

| block | engine | peak footprint | peak resident | swap-in GB | swap-out GB |
| --- | --- | --- | --- | --- | --- |
| r1_llama-server | llama-server | 1.27 | 11.90 | 0.99 | 1.55 |
| r1_lmstudio-gguf | lmstudio-gguf | 1.69 | 2.23 | 1.95 | 2.53 |
| r1_mj-a8w | mj-a8w | 3.18 | 10.37 | 2.02 | 2.21 |
| r1_mj-x | mj-x | 3.80 | 10.40 | 4.82 | 5.91 |
| r1_ollama-gguf | ollama-gguf | 1.59 | 10.78 | 1.27 | 2.11 |
| r2_llama-server | llama-server | 1.27 | 12.06 | 1.33 | 2.03 |
| r2_lmstudio-gguf | lmstudio-gguf | 1.70 | 10.57 | 0.91 | 1.47 |
| r2_mj-a8w | mj-a8w | 3.53 | 10.36 | 2.81 | 4.57 |
| r2_mj-x | mj-x | 3.79 | 10.58 | 25.87 | 27.30 |
| r2_ollama-gguf | ollama-gguf | 13.40 | 11.73 | 5.31 | 7.24 |
| r1_llama.cpp | llama.cpp | 0.33 | 11.72 | - | - |
| r2_llama.cpp | llama.cpp | 0.33 | 11.04 | - | - |
