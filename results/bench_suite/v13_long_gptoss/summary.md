# Benchmark suite summary

Protocol: results/BENCH_PROTOCOL.md. tok/s: mean +- sd [95% CI], CV, n (timed reps, flagged blocks excluded). Energy: IOReport GPU rail over the test window, estimated (SoC rails); none for llama-bench (its window includes model load).

## gpt-oss-20b

Weights: mj: openai/gpt-oss-20b exact (MXFP4 experts memory-mapped, packed bf16 attention, experts mlocked, 512-token prefill chunks as llama.cpp's ubatch); mj-a8: the same with 8-bit affine attention weights (lossy, matched to the GGUF); llama.cpp / Ollama / LM Studio (its llama.cpp engine): ggml-org MXFP4 GGUF (Q8_0 attention)

| engine | test | tok/s | 95% CI | CV | n | GPU mJ/token |
| --- | --- | --- | --- | --- | --- | --- |
| mj-a8 | pp16384 | 355.65 +- 3.46 | 352.02-359.29 | 1.0% | 6 | 52 |
| mj-a8 | tg128@d16384 | 31.37 +- 0.48 | 30.86-31.87 | 1.5% | 6 | 386 |
| mj-x | pp16384 | 323.03 +- 35.47 | 234.97-411.09 | 11.0% | 3 | 53 |
| mj-x | tg128@d16384 | 25.57 +- 3.53 | 16.82-34.33 | 13.8% | 3 | 445 |

Flagged blocks (excluded above; reason in blocks/*.flagged):

| engine | test | tok/s (flagged reps) | n |
| --- | --- | --- | --- |
| mj-x | pp16384 | 341.62 +- 5.76 (335.46-346.89) | 3 |
| mj-x | tg128@d16384 | 29.51 +- 0.17 (29.30-29.62) | 3 |
