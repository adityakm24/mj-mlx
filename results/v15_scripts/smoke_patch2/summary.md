# Benchmark suite summary

Protocol: results/BENCH_PROTOCOL.md. tok/s: mean +- sd [95% CI], CV, n (timed reps, flagged blocks excluded). Energy: IOReport GPU rail over the test window, estimated (SoC rails); none for llama-bench (its window includes model load).

## OLMoE-1B-7B-0125-Instruct (general MoE model)

Weights: ours / mlx-lm: mlx-community 4-bit g64 (float16 scales; run in float16); llama.cpp / Ollama (its GGUF runner; its MLX runner rejects OlmoeForCausalLM): allenai Q4_K_M; LM Studio (its MLX engine) and vllm-mlx: the mlx-community 4-bit

| engine | test | tok/s | 95% CI | CV | n | GPU mJ/token | machine mJ/token above idle |
| --- | --- | --- | --- | --- | --- | --- | --- |
| llama-server | pp512 | 1133.21 +- 10.45 | 1039.33-1227.09 | 0.9% | 2 | 14 | -3 |
| llama.cpp | pp512 | 1267.38 +- 2.23 | 1247.30-1287.46 | 0.2% | 2 | - | - |
| mj | pp512 | 1251.23 +- 1.82 | 1234.86-1267.60 | 0.1% | 2 | 14 | -2 |
| llama-server | tg128@d0 | 99.65 +- 0.02 | 99.44-99.85 | 0.0% | 2 | 102 | -43 |
| llama.cpp | tg128@d0 | 115.79 +- 0.03 | 115.51-116.07 | 0.0% | 2 | - | - |
| mj | tg128@d0 | 138.90 +- 0.31 | 136.16-141.64 | 0.2% | 2 | 66 | -26 |

Peak memory per block, GB (footprint: macOS's figure, GPU buffers included, mapped files not; resident: pages in RAM, mapped weight files included); swap-ins / -outs over the block:

| block | engine | peak footprint | peak resident | swap-in GB | swap-out GB |
| --- | --- | --- | --- | --- | --- |
| r1_llama-server | llama-server | 1.43 | 5.65 | 0.01 | 0.00 |
| r1_mj | mj | 5.17 | 0.33 | 0.01 | 0.00 |
| r1_llama.cpp | llama.cpp | 0.19 | 4.36 | - | - |
