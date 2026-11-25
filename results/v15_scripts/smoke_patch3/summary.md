# Benchmark suite summary

Protocol: results/BENCH_PROTOCOL.md. tok/s: mean +- sd [95% CI], CV, n (timed reps, flagged blocks excluded). Energy: IOReport GPU rail over the test window, estimated (SoC rails); none for llama-bench (its window includes model load).

## OLMoE-1B-7B-0125-Instruct (general MoE model)

Weights: ours / mlx-lm: mlx-community 4-bit g64 (float16 scales; run in float16); llama.cpp / Ollama (its GGUF runner; its MLX runner rejects OlmoeForCausalLM): allenai Q4_K_M; LM Studio (its MLX engine) and vllm-mlx: the mlx-community 4-bit

| engine | test | tok/s | 95% CI | CV | n | GPU mJ/token | machine mJ/token above idle |
| --- | --- | --- | --- | --- | --- | --- | --- |
| llama-server | pp2048 | 1148.47 +- 1.08 | 1138.79-1158.16 | 0.1% | 2 | 15 | 23 |
| mj | pp2048 | 1267.75 +- 7.54 | 1199.99-1335.51 | 0.6% | 2 | 15 | 19 |
| llama-server | tg128@d0 | 107.35 +- 9.75 | 19.68-195.01 | 9.1% | 2 | 101 | 152 |
| mj | tg128@d0 | 139.14 +- 0.17 | 137.61-140.67 | 0.1% | 2 | 66 | 107 |

Peak memory per block, GB (footprint: macOS's figure, GPU buffers included, mapped files not; resident: pages in RAM, mapped weight files included); swap-ins / -outs over the block:

| block | engine | peak footprint | peak resident | swap-in GB | swap-out GB |
| --- | --- | --- | --- | --- | --- |
| r1_llama-server | llama-server | 2.04 | 6.20 | 0.00 | 0.00 |
| r1_mj | mj | 7.14 | 0.31 | 0.00 | 0.00 |
