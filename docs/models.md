# Models

## Families

| Family | Class | Tested against transformers (tiny checkpoints) | Run on real weights here |
| --- | --- | --- | --- |
| gpt-oss | `GptOss` | yes | gpt-oss-20b (MXFP4 experts; bf16 or 8-bit attention) |
| Qwen3.5 / Qwen3.6 (hybrid Gated DeltaNet) | `Qwen35Moe` | yes | Qwen3.5-9B (mlx-community 4-bit), Qwen3.6-35B-A3B (mlx-community 4-bit, experts streamed) |
| OLMoE | `MoeLM` | yes | OLMoE-1B-7B-0125-Instruct (bf16 and mlx-community 4-bit) |
| DeepSeek-V2 (MLA) | `MoeLM` | yes | DeepSeek-V2-Lite-Chat (mlx-community 4-bit) |
| Qwen3-MoE | `MoeLM` | yes | Qwen3-30B-A3B (mlx-community 4-bit, experts streamed) |
| Mixtral | `MoeLM` | yes | Mixtral-8x7B-Instruct 4-bit (26.3 GB, experts streamed; a coherence check only, not benchmarked) |
| Qwen2-MoE, DeepSeek-V3, GLM-4-MoE | `MoeLM` | yes | not yet |

The tiny-checkpoint tests (`tests/`) compare prefill logits, greedy tokens, and batched runs with preemption
against transformers in float32. The real-weight checks are WikiText-2 perplexity and KL divergence against a
full-precision reference (RESULTS 1ze), plus the cross-engine benchmarks.

## Precision we run in

- Quantized checkpoints run in the dtype of their quantization scales (float16 for most mlx-community files,
  bfloat16 for some), as mlx-lm does. Running a float16-scaled checkpoint in bf16 made decode about 3x slower.
- **Unquantized checkpoints run in bfloat16 by default, whatever dtype they are stored in.** This departs from
  mlx-lm, which keeps a float32 checkpoint in float32. It is deliberate: bf16 halves memory and bandwidth, and the
  real checkpoints here are stored in bf16 anyway. Pass `dtype=mx.float32` to `load_model` for float32 (the
  tests and `tools/mlx_layer_stream_ref.py --dtype float32` do, to compare against float32 references).
- **"Exact"** in our results means the checkpoint's own weights at the checkpoint's own precision: quantized
  weights as stored, no extra quantization of our own (gpt-oss's "exact arm" against its 8-bit-attention arm).
  So a float32 checkpoint run at our bf16 default is not "exact" against that checkpoint.

## Checkpoint formats

- **Hugging Face bf16** (`model.safetensors*` with transformers key names). Experts can be stacked or per-expert.
- **mlx-community quantized** (`mlp.switch_mlp.*` with `weight` / `scales` / `biases`; for example 4-bit affine,
  group size 64). Gate and up projections are fused row-wise into one `gather_qmm`.
- **gpt-oss MXFP4**: the released checkpoint, as is.

GGUF is not read. It is the llama.cpp, Ollama and LM Studio format, and those are the engines we compare against.

## Where the experts live

For every MoE family (gpt-oss, Qwen3.5 / 3.6 and the `MoeLM` families), `load_model(path, placement="auto")` compares the checkpoint size with
this Mac's GPU working set (MLX's recommended limit) and RAM, and picks one of three placements:

| Placement | When | How |
| --- | --- | --- |
| `resident` | checkpoint + 2 GiB (KV, activations) fits the GPU working set | Experts are MLX arrays in memory, wired like the other weights. |
| `mapped` | it does not, but checkpoint + 3 GiB fits in RAM | Experts are repacked once to page-aligned, part-major files (`~/.cache/millijoule/<model>-experts-mapped`). The GPU reads them as zero-copy views of the page cache, and macOS pages them like any file. |
| `stream` | bigger than that | Experts are repacked once into per-layer files (`<model>-experts`). Each layer keeps a fixed number of expert slots in memory (LRU, next-layer prefetch, `F_NOCACHE` reads), sized from free memory. |

The repack writes once per model and needs as much free disk as the experts take. Existing repacked files are found
again by the source checkpoint recorded in their `index.json`, whatever their directory name.

## Known limits

- **Memory-mapped experts need a small wired limit.** At MLX's recommended working set, Metal runs out of memory
  (RESULTS 1j). The engine sets 1000 MB when the model has mapped experts (RESULTS 1zk).
- **MLA models keep one KV head.** Decode uses SDPA over the latent rows and prefill uses expanded K/V. The paged
  decode kernel would get one threadgroup per sequence.
- **C++ streamer slots.** The C++ streamer needs at least 4 x tokens x top-k slots per layer. Below that, the Python
  streamer reads in rounds.
- **No prefix cache for Qwen3.5 / 3.6** yet. The radix cache cannot restore recurrent state.
