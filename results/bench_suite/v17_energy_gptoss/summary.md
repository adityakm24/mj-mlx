# Benchmark suite summary

Protocol: results/BENCH_PROTOCOL.md. tok/s: mean +- sd [95% CI], CV, n (timed reps, flagged blocks excluded). Energy: IOReport GPU rail over the test window, estimated (SoC rails); none for llama-bench (its window includes model load).

## gpt-oss-20b

Weights: mj: openai/gpt-oss-20b exact (MXFP4 experts memory-mapped, packed bf16 attention, experts mlocked, 512-token prefill chunks as llama.cpp's ubatch); mj-a8: the same with 8-bit affine attention weights (lossy, matched to the GGUF); llama.cpp / Ollama / LM Studio (its llama.cpp engine): ggml-org MXFP4 GGUF (Q8_0 attention)

| engine | test | tok/s | 95% CI | CV | n | GPU mJ/token | machine mJ/token above idle | machine, steady state |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama-server | tg1024@d0 | 36.01 +- 0.59 | 35.40-36.63 | 1.6% | 6 | 348 | 776 | 783 +- 23 |
| mj-a8w | tg1024@d0 | 38.94 +- 1.50 | 37.37-40.51 | 3.9% | 6 | 260 | 593 | 601 +- 35 |
| mj-x | tg1024@d0 | every block flagged (see below) | | | 0 | - | - | - |

Flagged blocks (excluded above; reason in blocks/*.flagged):

| engine | test | tok/s (flagged reps) | n |
| --- | --- | --- | --- |
| mj-x | tg1024@d0 | 36.91 +- 0.24 (36.49-37.18) | 6 |

Peak memory per block, GB (footprint: macOS's figure, GPU buffers included, mapped files not; resident: pages in RAM, mapped weight files included); swap-ins / -outs over the block:

| block | engine | peak footprint | peak resident | swap-in GB | swap-out GB |
| --- | --- | --- | --- | --- | --- |
| r1_llama-server | llama-server | 0.50 | 0.71 | 0.30 | 0.01 |
| r1_mj-a8w | mj-a8w | 3.18 | 10.37 | 1.28 | 1.61 |
| r1_mj-x | mj-x | 3.75 | 10.36 | 1.70 | 3.24 |
| r2_llama-server | llama-server | 0.50 | 11.50 | 0.91 | 0.95 |
| r2_mj-a8w | mj-a8w | 3.28 | 10.32 | 1.43 | 2.16 |
| r2_mj-x | mj-x | 3.79 | 10.64 | 1.51 | 2.38 |
