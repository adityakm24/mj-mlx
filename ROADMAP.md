# Roadmap

mj-mlx is in beta. This is the plan, roughly in order; dates are not promised. The full working checklist, with the
evidence behind each item, is [`results/LAUNCH_TODO.md`](results/LAUNCH_TODO.md).

The goals do not change: the fastest MoE inference on a Mac **without quality loss**, models **bigger than RAM** run
at their own precision without swapping, and the **lowest whole-machine energy per token**. Every claim is measured
against the other Mac engines on the same machine, and the losses are published with the wins.

## Now: first release (1.0)

- **OpenAI-compatible server**: streaming, chat templates, stop strings, sampling; then tool calls, JSON output and
  a prefix cache across requests.
- **CLI**: `pull`, `run`, `serve`, `bench`.
- **Sampling on every path** (the gpt-oss fast path is greedy only today).
- **A prebuilt macOS wheel** (`pip install mj-mlx`, no Rust toolchain needed).
- **Results from more Macs**: M1-M4, Pro / Max / Ultra, 8 GB to 128 GB+ (help wanted: see
  [CONTRIBUTING.md](CONTRIBUTING.md)).
- **CI** on a macOS runner.

## Next: more models, longer context

- **GLM-4.5-Air** (106B total, 12B active; 65 GB at 4-bit) on a 16 GB Mac, streamed from the SSD: correctness check,
  speed, then long context.
- **8-bit KV cache**: twice the context in the same memory, behind a long-context accuracy check.
- More MoE families: Gemma-4, gpt-oss-120b, Qwen3-Next. More weight formats: mlx 3/6/8-bit, MXFP4 in the general
  model, GGUF loading.
- A fallback for dense and unknown models through mlx-lm's model code.

## Then: speed and energy

- Close the measured gaps: gpt-oss-20b prefill vs llama.cpp (2.5-6.9%), short-context decode on DeepSeek-V2-Lite vs
  vllm-mlx.
- Streaming: read experts ahead from a routing forecast, uneven expert memory per layer by default, faster cold
  prefill for models bigger than RAM.
- Speculative decoding beyond gpt-oss and Qwen3.6, priced by energy as well as speed.
- An energy leaderboard: whole-machine joules per token above idle, interleaved rounds, confidence intervals.
- Batch throughput against the other engines' servers.

## Later

- The Neural Engine for dense parts (shared experts, draft models), if a one-day spike shows an energy win
  ([docs/neural-engine.md](docs/neural-engine.md)).
- M5 matrix units through Metal 4, once tested on an M5.
