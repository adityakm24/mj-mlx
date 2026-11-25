# Benchmark suite summary

Protocol: results/BENCH_PROTOCOL.md. tok/s: mean +- sd [95% CI], CV, n (timed reps, flagged blocks excluded). Energy: IOReport GPU rail over the test window, estimated (SoC rails); none for llama-bench (its window includes model load).

## OLMoE-1B-7B-0125-Instruct (general MoE model)

Weights: ours / mlx-lm: mlx-community 4-bit g64 (float16 scales; run in float16); llama.cpp / Ollama (its GGUF runner; its MLX runner rejects OlmoeForCausalLM): allenai Q4_K_M; LM Studio (its MLX engine) and vllm-mlx: the mlx-community 4-bit

| engine | test | tok/s | 95% CI | CV | n | GPU mJ/token | machine mJ/token above idle | machine, steady state |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama-server | tg1024@d0 | 96.07 +- 2.25 | 93.71-98.42 | 2.3% | 6 | 109 | 237 | 246 +- 5 |
| mj | tg1024@d0 | 127.68 +- 0.44 | 127.21-128.14 | 0.3% | 6 | 71 | 159 | 171 +- 7 |
| mlx-lm | tg1024@d0 | 127.69 +- 0.22 | 127.45-127.93 | 0.2% | 6 | 71 | 157 | 169 +- 6 |

Greedy decode ids, ours vs mlx-lm (same seeds): 2/6 identical; first divergence at token 8-337 otherwise

Peak memory per block, GB (footprint: macOS's figure, GPU buffers included, mapped files not; resident: pages in RAM, mapped weight files included); swap-ins / -outs over the block:

| block | engine | peak footprint | peak resident | swap-in GB | swap-out GB |
| --- | --- | --- | --- | --- | --- |
| r1_llama-server | llama-server | 1.60 | 5.75 | 0.15 | 0.00 |
| r1_mj | mj | 4.19 | 0.38 | 0.22 | 0.00 |
| r1_mlx-lm | mlx-lm | 4.35 | 0.61 | 0.01 | 0.00 |
| r2_llama-server | llama-server | 1.32 | 5.54 | 0.01 | 0.00 |
| r2_mj | mj | 4.19 | 0.29 | 0.00 | 0.00 |
| r2_mlx-lm | mlx-lm | 4.36 | 0.72 | 0.01 | 0.00 |
