# The Neural Engine: a planned second accelerator

Status: not used yet. mj-mlx runs on the GPU (MLX and its own Metal kernels) and the CPU. This note records why, where
the Apple Neural Engine (ANE) could help later, and what the rest of the engine keeps in place so that adding it does not
mean a rewrite.

## What it is and why we do not use it yet

Every Apple-silicon Mac has one (16 cores on M1 to M4, including the base chips). It runs precompiled, fixed-shape
programs through Core ML (Core AI on newer macOS), in FP16 or INT8, and is very efficient per operation. It shares the
unified memory and its bandwidth with the GPU.

What stands in the way for an MoE engine on a 16 GB Mac:

- **No MLX path.** MLX targets the GPU and CPU only. ANE work goes through Core ML programs compiled ahead of time.
- **Fixed shapes vs routing.** MoE routing picks different experts per token. A fixed-shape program would run every
  expert (losing the point of MoE) or need one small call per expert and token; at about 0.2 ms of overhead per ANE call
  (reported by Anemll-Forge), decode would drown in call overhead.
- **A second copy of the weights.** The ANE runs its own compiled weights. Sharing one copy with the GPU has only been shown
  with System Integrity Protection off (Anemll-Forge, research build), which a product cannot ask for. On 16 GB a
  duplicate does not fit for the models we target.
- **No extra bandwidth.** Decode is bound by memory bandwidth, which the ANE and GPU share; run together they slow each
  other through DRAM (Anemll-Forge's measurements on M6).

## Where it could pay off

1. **Prefill of the dense, fixed-shape parts**, pipelined with the GPU: shared experts (Qwen3.5 / 3.6), the dense first
   layers of DeepSeek-V2 / V3, attention projections. Prefill is compute-bound, so a second compute unit can add speed,
   the way Anemll-Forge runs every FFN on the ANE while the GPU mixes the next block.
2. **Energy.** Even where it adds no speed, moving dense work to the ANE can lower joules per token.
3. **A dense draft model** for speculative decoding (fixed shapes, small).

## What the engine keeps in place for it

- **Placement decides where each part runs.** `placement.py` already chooses resident / mapped / streamed per expert
  group; a future `device` per component (GPU, ANE) belongs there, not in the model classes.
- **One step = one flat batch** (internals.md): the shapes of the dense parts in a step are known before it runs, which is
  what ANE programs need (fixed shapes; pad prefill chunks to a few compiled sizes).
- **Whole-machine energy.** The benchmarks report energy above idle for the whole machine (SMC), not only the GPU rail, so an
  ANE path is measured fairly against GPU-only engines (in the harness from suite v15, 2026-11-21; results not
  published yet).
- **Memory accounting** reports footprint and resident memory per engine, so a duplicate weight copy shows up.

## The first step: a one-day spike (checklist 5.1)

On the M4 16 GB, without touching the engine:

1. One dense FFN block (a shared expert) as a Core ML program at prefill chunk sizes 64 / 256 / 1024: time per call,
   throughput, and call overhead, against our Metal kernel on the same shapes.
2. Memory: the compiled copy of just the dense parts next to a loaded model; does it fit on 16 GB?
3. Energy per token of those parts, ANE against GPU, whole machine above idle.
4. Overlap: GPU experts and ANE dense block at the same time; how much each slows the other.

Go if the dense parts run faster or cheaper on the ANE and the overlap leaves a net prefill or energy gain; otherwise,
record the numbers and stop.
