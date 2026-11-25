# Benchmark suite summary

Protocol: results/BENCH_PROTOCOL.md. tok/s: mean +- sd [95% CI], CV, n (timed reps, flagged blocks excluded). Energy: IOReport GPU rail over the test window, estimated (SoC rails); none for llama-bench (its window includes model load).

## gpt-oss-20b

Weights: mj: openai/gpt-oss-20b exact (MXFP4 experts memory-mapped, packed bf16 attention, experts mlocked, 512-token prefill chunks as llama.cpp's ubatch); mj-a8: the same with 8-bit affine attention weights (lossy, matched to the GGUF); llama.cpp / Ollama / LM Studio (its llama.cpp engine): ggml-org MXFP4 GGUF (Q8_0 attention)

| engine | test | tok/s | 95% CI | CV | n | GPU mJ/token | machine mJ/token above idle |
| --- | --- | --- | --- | --- | --- | --- | --- |
| llama.cpp | pp512 | 426.14 +- 4.41 | 420.66-431.63 | 1.0% | 5 | - | - |
| lmstudio-gguf | pp512 | every block flagged (see below) | | | 0 | - | - |
| mj | pp512 | 396.77 +- 2.27 | 393.95-399.59 | 0.6% | 5 | 44 | 56 |
| mj-a8 | pp512 | 407.64 +- 2.18 | 406.08-409.21 | 0.5% | 10 | 45 | 52 |
| mj-a8w | pp512 | 403.79 +- 5.50 | 396.96-410.63 | 1.4% | 5 | 45 | 57 |
| mj-x | pp512 | 401.15 +- 2.38 | 398.19-404.12 | 0.6% | 5 | 44 | 56 |
| ollama-gguf | pp512 | 404.82 +- 21.65 | 377.90-431.74 | 5.3% | 5 | 41 | 48 |
| llama.cpp | pp2048 | 446.94 +- 3.04 | 443.15-450.72 | 0.7% | 5 | - | - |
| lmstudio-gguf | pp2048 | every block flagged (see below) | | | 0 | - | - |
| mj | pp2048 | 426.67 +- 2.61 | 423.42-429.92 | 0.6% | 5 | 46 | 79 |
| mj-a8 | pp2048 | 435.92 +- 2.29 | 434.28-437.56 | 0.5% | 10 | 46 | 80 |
| mj-a8w | pp2048 | 435.20 +- 0.83 | 434.17-436.23 | 0.2% | 5 | 46 | 73 |
| mj-x | pp2048 | 428.99 +- 1.45 | 427.19-430.80 | 0.3% | 5 | 46 | 78 |
| ollama-gguf | pp2048 | 414.87 +- 9.61 | 402.92-426.82 | 2.3% | 5 | 42 | 72 |
| llama.cpp | tg128@d0 | 37.04 +- 0.12 | 36.89-37.19 | 0.3% | 5 | - | - |
| lmstudio-gguf | tg128@d0 | every block flagged (see below) | | | 0 | - | - |
| mj | tg128@d0 | 37.00 +- 0.50 | 36.38-37.63 | 1.4% | 5 | 278 | 573 |
| mj-a8 | tg128@d0 | 39.85 +- 0.41 | 39.56-40.15 | 1.0% | 10 | 243 | 490 |
| mj-a8w | tg128@d0 | 40.16 +- 0.56 | 39.46-40.85 | 1.4% | 5 | 246 | 508 |
| mj-x | tg128@d0 | 37.04 +- 0.44 | 36.49-37.59 | 1.2% | 5 | 274 | 570 |
| ollama-gguf | tg128@d0 | 34.75 +- 0.47 | 34.17-35.33 | 1.3% | 5 | 337 | 751 |
| llama.cpp | tg128@d2048 | 35.29 +- 1.24 | 33.75-36.83 | 3.5% | 5 | - | - |
| lmstudio-gguf | tg128@d2048 | every block flagged (see below) | | | 0 | - | - |
| mj | tg128@d2048 | 35.73 +- 0.22 | 35.46-36.00 | 0.6% | 5 | 301 | 732 |
| mj-a8 | tg128@d2048 | 38.71 +- 0.69 | 38.21-39.20 | 1.8% | 10 | 270 | 670 |
| mj-a8w | tg128@d2048 | 38.57 +- 0.87 | 37.49-39.65 | 2.3% | 5 | 271 | 664 |
| mj-x | tg128@d2048 | 35.88 +- 0.14 | 35.70-36.06 | 0.4% | 5 | 301 | 723 |
| ollama-gguf | tg128@d2048 | 34.00 +- 0.62 | 33.22-34.77 | 1.8% | 5 | 359 | 848 (1 replies under the requested length; tok/s over the tokens produced) |

Flagged blocks (excluded above; reason in blocks/*.flagged):

| engine | test | tok/s (flagged reps) | n |
| --- | --- | --- | --- |
| llama.cpp | pp512 | 432.81 +- 4.69 (424.69-435.92) | 5 |
| lmstudio-gguf | pp512 | 407.85 +- 13.25 (383.51-421.96) | 10 |
| mj | pp512 | 402.91 +- 10.31 (385.53-410.96) | 5 |
| mj-a8w | pp512 | 405.75 +- 3.36 (400.90-410.37) | 5 |
| mj-x | pp512 | 400.54 +- 2.11 (397.71-403.46) | 5 |
| ollama-gguf | pp512 | 391.18 +- 6.80 (381.11-397.60) | 5 |
| llama.cpp | pp2048 | 446.82 +- 2.06 (443.35-448.48) | 5 |
| lmstudio-gguf | pp2048 | 434.91 +- 3.06 (428.52-438.24) | 10 |
| mj | pp2048 | 430.45 +- 1.22 (429.53-432.54) | 5 |
| mj-a8w | pp2048 | 434.98 +- 1.34 (433.23-436.36) | 5 |
| mj-x | pp2048 | 427.17 +- 1.61 (425.21-429.55) | 5 |
| ollama-gguf | pp2048 | 400.06 +- 10.82 (383.59-412.57) | 5 |
| llama.cpp | tg128@d0 | 36.94 +- 0.55 (35.97-37.25) | 5 |
| lmstudio-gguf | tg128@d0 | 35.70 +- 0.22 (35.36-35.99) | 10 |
| mj | tg128@d0 | 36.63 +- 0.61 (36.11-37.31) | 5 |
| mj-a8w | tg128@d0 | 40.57 +- 0.04 (40.52-40.62) | 5 |
| mj-x | tg128@d0 | 36.67 +- 0.53 (36.06-37.36) | 5 |
| ollama-gguf | tg128@d0 | 34.79 +- 0.63 (33.78-35.29) | 5 |
| llama.cpp | tg128@d2048 | 36.34 +- 0.11 (36.19-36.45) | 5 |
| lmstudio-gguf | tg128@d2048 | 33.47 +- 0.55 (32.13-34.01) | 10 |
| mj | tg128@d2048 | 36.00 +- 0.07 (35.95-36.11) | 5 |
| mj-a8w | tg128@d2048 | 38.97 +- 0.08 (38.83-39.02) | 5 |
| mj-x | tg128@d2048 | 35.87 +- 0.17 (35.58-36.01) | 5 |
| ollama-gguf | tg128@d2048 | 34.02 +- 0.45 (33.27-34.49) | 5 |

Peak memory per block, GB (footprint: macOS's figure, GPU buffers included, mapped files not; resident: pages in RAM, mapped weight files included); swap-ins / -outs over the block:

| block | engine | peak footprint | peak resident | swap-in GB | swap-out GB |
| --- | --- | --- | --- | --- | --- |
| r1_lmstudio-gguf | lmstudio-gguf | 1.69 | 1.18 | 1.21 | 1.98 |
| r1_mj-a8 | mj-a8 | 3.53 | 10.50 | 2.25 | 2.48 |
| r1_mj-a8w | mj-a8w | 3.53 | 10.62 | 2.26 | 3.30 |
| r1_mj-x | mj-x | 4.01 | 10.70 | 2.37 | 2.78 |
| r1_mj | mj | 3.79 | 10.62 | 2.95 | 4.44 |
| r1_ollama-gguf | ollama-gguf | 1.59 | 11.03 | 1.37 | 1.96 |
| r2_lmstudio-gguf | lmstudio-gguf | 1.70 | 12.18 | 1.06 | 1.55 |
| r2_mj-a8 | mj-a8 | 3.54 | 10.49 | 1.59 | 2.23 |
| r2_mj-a8w | mj-a8w | 3.53 | 10.45 | 2.36 | 2.97 |
| r2_mj-x | mj-x | 3.79 | 10.48 | 3.37 | 4.20 |
| r2_mj | mj | 3.44 | 10.72 | 3.43 | 3.81 |
| r2_ollama-gguf | ollama-gguf | 13.40 | 11.78 | 3.39 | 5.11 |
| r1_llama.cpp | llama.cpp | 0.33 | 11.92 | - | - |
| r2_llama.cpp | llama.cpp | 0.34 | 11.20 | - | - |
