# MLX backend results (2026-11-16)

Summary: on this Mac, against mlx-lm 0.32.2 at its best measured settings, MilliJoule's MLX runner is faster and
uses less GPU energy per token at 1, 8, 32 and 64 requests (3 ABAB rounds, section 1), and on the real
gpt-oss-20b weights its bf16 output is closer to float32 (KL -29%, greedy top-1 97.2% vs 95.4%, section 2a; one 1296-position
teacher-forced sample, not the plan's lm-eval gate). Its
float32 path equals transformers to float32 rounding on real layers; mlx-lm's does not. On Qwen3.6 (random weights at
real dims, 8 of 40 layers, 4-bit experts on both sides) GPU energy per token is lower at levels 1-32 in all 3 ABAB
rounds (-9% to -30%), speed is higher at level 1 (1.28-1.47x) and tied at 8-64 (section 1c). The real gpt-oss-20b,
which does not fit in memory, runs at 4.5-8.5 tok/s in 4.2-9.2 GB resident with SSD expert streaming, where mlx-lm
needs 105 s for the first token and swaps (section 1d).

**Boundary.** Energy is the IOReport "GPU Energy" rail: whole-chip GPU energy from the SoC energy model, idle
floor included. **Mac only, not comparable to NVML or any B200 number.** No J/token claim of the project comes
from this file.

**Model.** Random weights at real gpt-oss-20b dimensions (hidden 2880, 32 experts, top-4, 64 q / 8 kv heads,
head_dim 64, vocab 201088, sliding window 128 alternating with full attention, sinks, MXFP4 experts, bf16 rest),
**8 of 24 layers** so it fits a 16 GB Mac. Built by `tools/mlx_make_bench_model.py --layers 8` (seed 0). Routing
is close to uniform random. Correctness is checked separately on the real-architecture tiny checkpoint.

**Machine.** Apple M4 (10-core GPU), 16 GB, macOS 27.3, MLX 0.32.4. The Mac was under memory pressure from other
apps during all runs (7-18 GB swap in use), which is the main source of run-to-run spread.

**Tree.** Base commit `0908550` (main) plus the untracked MLX files (`python/millijoule/backends/`, `tests/mlx/`,
`tools/mlx_*.py`); backend sources hash `47791f98d215` (`find . -name '*.py' | sort | xargs shasum | shasum` in
`python/millijoule/backends`).

## 1. Throughput and energy, v4, ABAB (3 rounds per arm) — current

Tree: base `0908550` plus the PM's uncommitted state-slot scheduler change (rebuilt 17:13, before our first arm),
plus the untracked MLX backend; backend sources hash `b4ee9fa69b3d`. Same model, workload and commands as the v3
section below (`results/v4_abab/`, script: A then B, three times). v4 = v3 plus the exact two-stage top-k for
lm_head candidates (a full-vocab argpartition had cost 26 ms at 64 rows) and the candidate-head width chosen by
rows (4-bit <= 4, 8-bit <= 48, bf16 above). mlx-lm is at its defaults (its best in the section 4 sweep).

Raw, per round (tok/s / mJ per token, GPU rail):

| arm, round | level 1 | level 8 | level 32 | level 64 |
| --- | --- | --- | --- | --- |
| A mlx-lm, r1 (slow state + rebuild during A) | 33.1 / 333.7 | 93.6 / 152.9 | 164.6 / 93.7 | 252.2 / 66.3 |
| B ours, r1 (slow state) | 50.8 / 239.3 | 106.0 / 131.8 | 204.5 / 81.9 | 374.9 / 44.2 |
| A mlx-lm, r2 | 45.4 / 194.0 | 124.3 / 115.9 | 220.5 / 76.2 | 323.4 / 52.1 |
| B ours, r2 | 72.2 / 156.2 | 147.2 / 89.1 | 263.4 / 65.4 | 384.7 / 44.2 |
| A mlx-lm, r3 | 45.0 / 197.1 | 123.9 / 115.2 | 219.8 / 76.3 | 324.3 / 52.0 |
| B ours, r3 | 68.7 / 135.6 | 146.6 / 88.0 | 259.9 / 62.9 | 382.6 / 43.0 |

**Headline (rounds 2 and 3, the clean fast-state pair):** speed B/A at level 1 / 8 / 32 / 64 = 1.59-1.53x /
1.18x / 1.19-1.18x / 1.19-1.18x; GPU energy per token B/A = 0.81-0.69x / 0.77-0.76x / 0.86-0.83x / 0.85-0.83x.
Round 1 ran in the slow state and the main agent's `maturin develop` compiled during its mlx-lm arm (17:13), which
may have slowed A and inflated round 1's ratios (level 64: 1.49x vs 1.19 / 1.18); it is kept below, flagged.

Paired ratios B/A (same round), speed and GPU energy per token (r1 flagged: slow state + rebuild during A):

| level | speed r1 / r2 / r3 | energy r1 / r2 / r3 |
| --- | --- | --- |
| 1 | 1.53 / 1.59 / 1.53 | 0.72 / 0.81 / 0.69 |
| 8 | 1.13 / 1.18 / 1.18 | 0.86 / 0.77 / 0.76 |
| 32 | 1.24 / 1.19 / 1.18 | 0.87 / 0.86 / 0.83 |
| 64 | 1.49 / 1.19 / 1.18 | 0.67 / 0.85 / 0.83 |

Reading:
- **The Mac has two performance states**, and both engines move together with it. Round 1 ran in a slow state
  (idle GPU 0.9-1.3 W); rounds 2 and 3 in a fast one (idle reads 0.00 W) and agree with each other to ~2% for
  both arms. That, not our runner, is the source of the large cross-round spread seen in v3; paired same-round
  ratios are the meaningful statistic.
- **Established at every level** under the plan's rule (energy down by more than 5% and the repeats agree): in the
  clean pair (r2, r3), and also in r1, B is faster and uses less energy per token. Worst of r2/r3 per level:
  level 1 +53% / -19%, level 8 +18% / -23%, level 32 +18% / -14%, level 64 +18% / -15%. At level 64 B's slowest
  run (374.9 tok/s) is above A's fastest (324.3), and B's spread is 3%.

## 1b. Throughput and energy, v3, ABAB (2 rounds per arm) — superseded by v4

Files moved to `results/v3_abab/`.

Command, per arm and round (`abab.sh`: A then B, twice):

```bash
.venv/bin/python tools/mlx_bench.py --engine mlx-lm --model <8L> --levels 1,8,32,64 --out <dir>/r{N}_a_mlx_lm.json
.venv/bin/python tools/mlx_bench.py --engine mj     --model <8L> --levels 1,8,32,64 --out <dir>/r{N}_b_mj.json
.venv/bin/python tools/mlx_report.py <dir>/r*_a_*.json -- <dir>/r*_b_*.json
```

Workload: closed loop, 4x level requests per level, random-token prompts of 64-320 tokens (seeded, identical for
both engines), 128 output tokens each with EOS ignored. Window: open while `level` requests are generating.
A = mlx-lm 0.32.2 `BatchGenerator` (completion_batch_size = level, prefill_batch_size = min(level, 8),
prefill_step_size = 2048; mlx-lm's own loader on the same checkpoint). B = MilliJoule MLX runner (paged decode
kernel, precise mode, 8-bit lm_head candidates + fp32 top-16 rescore, max_batch_tokens 2048, prefix cache on).

Per-round paired ratios (B/A, same round):

| level | speed B/A, r1 | speed B/A, r2 | energy/token B/A, r1 | energy/token B/A, r2 |
| --- | --- | --- | --- | --- |
| 1 | 1.31x | 1.66x | 0.77x | 0.51x |
| 8 | 1.19x | 1.53x | 0.92x | 0.66x |
| 32 | 1.10x | 1.33x | 0.87x | 0.73x |
| 64 | 1.04x | 1.05x | 0.93x | 0.92x |

Means over rounds (spread = (max - min) / mean over the two rounds):

| level | A tok/s (spread) | B tok/s (spread) | A mJ/token (spread) | B mJ/token (spread) |
| --- | --- | --- | --- | --- |
| 1 | 33.1 (6%) | 49.2 (29%) | 342.3 (5%) | 220.5 (46%) |
| 8 | 91.6 (3%) | 124.6 (27%) | 150.2 (2%) | 118.4 (31%) |
| 32 | 178.6 (5%) | 217.5 (24%) | 95.7 (6%) | 76.8 (23%) |
| 64 | 290.5 (23%) | 303.7 (24%) | 59.6 (24%) | 55.1 (25%) |

Reading: B is faster and uses less GPU energy per token than A at every level in both rounds, and energy moves by
more than the plan's 5% in both rounds at every level. How far that can be trusted, given the noise:
- **Levels 1 and 8: established.** A is very stable there (A/A spread 2-6% over 2 ABAB rounds plus earlier runs:
  level 1 32.1-34.0 tok/s, level 8 89.6-92.8), and B's worst run is still +19% / -8% (level 8) and +31% / -23%
  (level 1). The size is uncertain within roughly 1.2-1.7x.
- **Level 32: likely, not established.** A also ran 215.9 tok/s / 82.6 mJ in the tuning run (section 4), above
  B's round-1 result (191.6 / 85.7).
- **Level 64: not established.** ~5% speed / ~7% energy, inside both arms' 23-25% spread.

## 1c. Qwen3.6 throughput and energy, ABAB (3 rounds per arm), and the decode-step split

**Model.** Random weights at the real Qwen3.6-35B-A3B dimensions (from its config.json, fetched with the user's
OK): hidden 2048, 256 experts top-8 (intermediate 512) + shared expert (512), 16 q / 2 kv heads of 256 with partial
rotary, Gated DeltaNet 16 key / 32 value heads of 128, vocab 248320. **8 of 40 layers** in the real 3:1 pattern
(linear x3, full x1, twice). **Routed experts 4-bit affine, group 64, bf16 scales, on both sides** (mlx-lm-native
`switch_mlp` tensors read by both loaders); everything else bf16. Recurrent state float32 in both engines. Built by
`tools/mlx_make_bench_model.py --arch qwen36 --layers 8` (seed 0). Before timing, the two engines were checked to
compute the same model: float32 prefill logits ours vs mlx-lm on these weights, rel. RMS diff 7.2e-6, argmax 100%
over 200 positions.

**Tree.** Base `98bd401` plus the MLX backend (sources hash `e643f25bc704`, `results/qwen36_v1_abab/backend_sources_sha.txt`).
mlx-lm 0.32.2 `BatchGenerator` at its defaults (completion_batch_size = level, prefill_batch_size = min(level, 8),
prefill_step_size 2048). Ours: precise mode, auto lm_head candidates, paged decode kernel, the gated-delta pool
kernel, `num_state_slots = level` (the same concurrency the baseline holds states for). Same workload as section
1: closed loop, 4x level requests, prompts 64-320 random tokens (seeded), 128 output tokens, EOS ignored. No venv
rebuild during the campaign (confirmed with the main agent).

```bash
.venv/bin/python tools/mlx_bench.py --engine mlx-lm --model <qwen36-8L> --levels 1,8,32,64 --out results/qwen36_v1_abab/r{N}_a_mlx_lm.json
.venv/bin/python tools/mlx_bench.py --engine mj     --model <qwen36-8L> --levels 1,8,32,64 --out results/qwen36_v1_abab/r{N}_b_mj.json
```

Raw per round, tok/s / mJ per token (GPU rail, Mac whole-GPU energy, not comparable to NVML):

| arm, round | level 1 | level 8 | level 32 | level 64 |
| --- | --- | --- | --- | --- |
| A mlx-lm, r1 (idle 0.00 W) | 53.7 / 136.5 | 243.1 / 61.2 | 365.6 / 38.9 | 448.9 / 31.5 |
| B ours, r1 (idle 0.08 W) | 78.9 / 111.7 | 244.6 / 43.4 | 382.0 / 29.7 | 454.8 / 27.0 |
| A mlx-lm, r2 (idle 0.00 W) | 53.8 / 147.7 | 242.6 / 59.0 | 364.6 / 37.7 | 367.0 / 35.3 |
| B ours, r2 (idle 0.00 W) | 68.6 / 128.1 | 240.6 / 45.3 | 393.4 / 34.2 | 393.0 / 34.4 |
| A mlx-lm, r3 (idle 1.23 W) | 40.5 / 257.8 | 183.3 / 85.2 | 278.7 / 51.2 | 343.1 / 41.9 |
| B ours, r3 (idle 1.29 W) | 58.0 / 181.6 | 192.3 / 70.2 | 293.8 / 44.0 | 344.8 / 38.3 |

Paired same-round ratios B/A:

| level | speed r1 / r2 / r3 | GPU energy per token r1 / r2 / r3 | reading |
| --- | --- | --- | --- |
| 1 | 1.47 / 1.28 / 1.43 | 0.82 / 0.87 / 0.70 | faster and lower energy in every round: established |
| 8 | 1.01 / 0.99 / 1.05 | 0.71 / 0.77 / 0.82 | speed tie; energy -18% to -29% in every round: established |
| 32 | 1.04 / 1.08 / 1.05 | 0.76 / 0.91 / 0.86 | energy -9% to -24% in every round: established; speed +4-8% (beyond 5% in 2 of 3) |
| 64 | 1.01 / 1.07 / 1.00 | 0.86 / 0.97 / 0.91 | energy lower in every round but r2 is inside 5%: not established; speed tie |

Rounds 1-2 ran in the fast machine state (idle reads ~0 W), round 3 in the slow one (idle 1.2-1.3 W); both arms of
each round were in the same state.

Where the energy gain comes from at equal speed: our GPU draws less power for the same tokens (level 8: 11.7 W vs
14.9 W mean). We do not have a per-kernel energy split on the Mac; the profile below is time.

TTFT, mean of 3 rounds, p50 / p99 seconds (closed loop with **all of a level's requests submitted at t0**, so TTFT
includes queueing behind the running set; this is not the plan's server workload and the 2 s limit does not apply):

| arm | level 1 | level 8 | level 32 | level 64 |
| --- | --- | --- | --- | --- |
| A mlx-lm | 4.2 / 8.1 | 8.1 / 15.5 | 22.8 / 44.1 | 39.4 / 79.8 |
| B ours | 3.0 / 5.8 | 7.9 / 15.1 | 20.7 / 40.7 | 35.2 / 72.8 |

Resources, mean of 3 rounds: **GPU active residency, %** (IOReport GPUPH, time the GPU is powered on, not busy
cycles; ~83% at desktop idle because other apps keep it on) / process CPU % (CPU time / wall, 100 = one core) /
MLX peak memory GB (max over rounds):

| arm | level 1 | level 8 | level 32 | level 64 |
| --- | --- | --- | --- | --- |
| A mlx-lm | 100% / 43% / 6.5 | 100% / 32% / 7.3 | 99% / 18% / 7.8 | 99% / 16% / 8.6 |
| B ours | 100% / 29% / 7.8 | 97% / 36% / 8.6 | 98% / 20% / 9.0 | 99% / 13% / 9.6 |

Process peak RSS: mlx-lm 4.6-5.1 GB, ours 4.2 GB. Our MLX peak is ~1 GB higher (float32 residual stream and the 4- and
8-bit lm_head candidate copies, ~0.75 GB).

**Decode-step split (ours)**, `results/qwen36_profile.json`: one decode step at context ~250, each component
replaced by zeros (the state kernel by a no-op that leaves the pool unchanged), difference to the full step. Measured
on the 8-layer model, ms (share of step):

| B | step | experts (routed + shared, router) | recurrent state read/write (gated-delta kernel) | rest of DeltaNet (projections, conv, gated norm) | full attention | lm_head |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 11.7 | 2.5 (21%) | 0.9 (7%) | 4.3 (36%) | 1.5 (13%) | 3.0 (26%) |
| 8 | 29.0 | 11.1 (38%) | 3.7 (13%) | 4.8 (16%) | 3.0 (10%) | 7.8 (27%) |
| 32 | 60.4 | 27.8 (46%) | 10.7 (18%) | 8.5 (14%) | 3.2 (5%) | 10.3 (17%) |
| 64 | 101.8 | 44.6 (44%) | 21.5 (21%) | 10.4 (10%) | 4.7 (5%) | 18.7 (18%) |

Estimate for the full 40-layer model (**an extrapolation, not a run**: per-layer parts x5, head once):

| B | est. step ms | experts | recurrent state | rest of DeltaNet | full attention | lm_head |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 46 | 26% | 9% | 46% | 16% | 7% |
| 8 | 114 | 49% | 16% | 21% | 13% | 7% |
| 32 | 261 | 53% | 21% | 16% | 6% | 4% |
| 64 | 434 | 51% | 25% | 12% | 5% | 4% |

For the CUDA plan: at B=64 the recurrent state is ~21% of the step here and the experts ~44%; the state share grows
with batch (0.9 ms of 11.7 at B=1, 21.5 of 101.8 at B=64) because every request's 2 MB per layer is read and written
each step. The gated-delta update runs at the speed of a plain copy of the pool (3.5 ms per layer at B=64, float32;
1.8 ms with bf16 storage), so on this Mac the state cost is bandwidth, and halving its bytes halves it.

What moved our Qwen3.6 numbers before the campaign (single profile runs, B=32 decode step 131 -> 60 ms): the state
pool read/written inside the kernel instead of gather + multiply + scatter (DeltaNet 77.5 -> 24.7 ms), the
`[Hv, Dv, Dk]` state layout for coalesced loads, and 4 value columns per simdgroup sharing q/k (kernel 4.9 -> 3.5 ms
at B=64).

## 1d. Real gpt-oss-20b on this 16 GB Mac with SSD expert streaming

The released `openai/gpt-oss-20b` (13.7 GB: 24 layers x 32 MXFP4 experts = 10.1 GB of experts) does not fit this
Mac's GPU working set. MilliJoule now streams the routed experts from the SSD (`expert_stream.py`, design taken from
reading Colibri, Flash-MoE, streamlx and mlx-moe-streaming: `results/expert_streaming_research.md`):

- experts repacked once into per-layer files, one contiguous 16 KB-aligned record per expert in the in-memory
  layout (`tools/mlx_repack_experts.py`; 9.5 GB, 23 s);
- per layer `S` resident slots; misses `preadv`'d straight into the slots' MLX memory (zero copy, verified), 8 reader
  threads, `F_NOCACHE`; least-recently-used eviction;
- lean trunk: embedding table and bf16 lm_head stay on disk (memory-mapped rows); a 4-bit candidate head plus float32
  rescore of 16 rows from disk is resident (greedy only);
- optional next-layer prefetch: layer L+1's router applied to layer L's MoE input, in the same host sync, reads
  overlapped with layer L's compute.

Correctness (`tests/mlx/test_expert_stream.py`, tiny gpt-oss checkpoint, float32, 2 of 4 slots resident so every
step misses and runs in rounds): greedy output token-identical to transformers and to the in-memory engine, 0
near-tie excuses, with and without the lean trunk; prefetch on and off give identical tokens. Not bitwise against the
in-memory engine in bf16: MLX's `gather_qmm` accumulation depends on the shape of the whole call (measured: a
160-row call and the same rows one at a time differ by one bf16 ulp on 40 of 160 rows), so streaming, like batching,
changes accumulation order. Real-weight sanity: the generated text is coherent (the model's analysis channel answers
the prompt).

Speed, one request, greedy, 97-token chat prompt, 64 new tokens (`tools/mlx_stream_run.py`, single runs, files in
`results/stream_gptoss20b/`):

| expert slots per layer | resident memory | expert cache | decode hit rate (no prefetch / prefetch) | decode tok/s, no prefetch | decode tok/s, prefetch | TTFT |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | 4.2 GB | 2.5 GB | 64% / 90% | 3.1 (4.4 with OS page cache) | **4.5** | 4.0-4.3 s |
| 16 | 6.7 GB | 5.1 GB | 89% / 96% | 5.8 | **6.7** (6.9 with page cache) | 3.5-3.8 s |
| 24 | 9.2 GB | 7.6 GB | 97% / 99% | 8.0 | **8.5** | 3.9-4.4 s |

Prefetch accuracy on the real model: 88-90% of prefetched experts were used (1835 of 2068 at 8 slots). Prefetch
together with the page cache was worse at 8 slots (3.8 tok/s) and equal at 16.

**Baseline:** mlx-lm 0.32.2 loading the same checkpoint (lazy load, `generate_step`) took 105 s to the first token
and had not produced 32 tokens when killed at 420 s; system swap grew from 9.2 to 14.7 GB. On this Mac it does not
run the model usefully; ours runs it at 4.5-8.5 tok/s in 4.2-9.2 GB.

Energy for these runs is not reported as a result: the GPU rail read 0.8-5.4 W across single runs in a way that
tracks the machine's power state more than the configuration (section 1, rule 4). **Rule for any future streaming
energy number:** the IOReport GPU rail does not see SSD reads or the CPU's preadv reader threads, which at batch 1
are a real share of the joules, so a GPU-rail-only J/token would undercount streaming against in-memory engines.
Streaming energy is either left out (as here) or measured as whole-system power (CPU, DRAM and SSD rails need root:
the user's call) and labelled as such. An ABAB energy campaign is a
follow-up. TTFT (~4 s for 97 tokens) is dominated by prefill reading most experts of every layer once (a prompt
routes to nearly all 32 experts per layer); expert-major prefill and keeping prompt-hot experts are the next levers.

### 1d.1 The same streaming core on every MoE model (gpt-oss and Qwen3.6)

Streaming is now model-agnostic (`expert_stream.py`): `ExpertStore` reads a self-describing repack (index.json lists
each expert part's name, dtype and shape), `ExpertSlab` holds one layer's slots for any parts, `run_moe` is the shared
step driver (routing sync with next-layer prediction, rounds, reordering, routing weights). A model supplies only
the math of one expert on (token, expert) pairs. `tools/mlx_repack_experts.py` writes the same format for gpt-oss
MXFP4, Qwen3.6 4-bit affine (mlx-native) and Qwen3.6 bf16 experts; `tools/mlx_stream_run.py` runs any of them.

Same correctness gate for both (`tests/mlx/test_expert_stream.py`, float32, tiny checkpoints, 2 of 4 experts
resident): greedy token-identical to transformers and to each model's in-memory engine, 0 near-tie excuses, with
and without the lean trunk.

Same metrics, one request, 97-token prompt, 64 new tokens, prefetch on (`results/stream_v2/`, single runs):

| model | mode | resident memory | decode tok/s | decode hit rate | TTFT |
| --- | --- | --- | --- | --- | --- |
| gpt-oss-20b (real weights; does not fit in memory) | 8 / 16 / 24 of 32 slots | 4.2 / 6.7 / 9.2 GB | 4.7 / 7.2 / 10.4 | 90 / 96 / 99% | 3.9-4.2 s |
| Qwen3.6 real dims, 8 of 40 layers, random weights, 4-bit experts | in memory | 7.1 GB | 45.2 | - | 0.26 s |
| same | 32 / 64 / 128 of 256 slots | 1.3 / 1.8 / 2.7 GB | 13.4 / 14.6 / 17.7 | 57 / 64 / 76% | 1.3-1.9 s |

gpt-oss after the refactor matches or beats the gpt-oss-only version (4.5 / 6.7 / 8.5 tok/s, same hit rates; single
runs). Qwen3.6 hit rates are pessimistic: random weights route close to uniformly, and real gpt-oss routing was far
more predictable (88-90% of prefetches used vs 69-77% here). Streaming a model that fits in memory costs speed (45 ->
13-18 tok/s here); its use is models that do not fit. A full 40-layer Qwen3.6-35B-A3B at 4-bit (~18 GB of experts) would
need ~3 GB of trunk plus slots: within this Mac's reach, untested without real weights.

### 1d.2 Real Qwen3.6-35B-A3B (20.4 GB, mlx-community 4-bit) on this 16 GB Mac

Downloaded with the user's OK (`mlx-community/Qwen3.6-35B-A3B-4bit`, revision 38740b84). Everything in it is quantized
(experts, attention, DeltaNet, shared expert, embedding and head at 4-bit gs64; router and shared-expert gate at 8-bit)
and stored in mlx-lm's converted layout; the Qwen loader now reads both that and transformers' layout (a `Lin` per
linear layer: bf16 or quantized with bits/group inferred from shapes; mlx conv layout detected so norms are not folded
twice; quantized embedding dequantized in the model dtype; a quantized head is used directly).

Loader check against mlx-lm on a 4-layer cut of the real checkpoint (3 DeltaNet + 1 full attention), float32 both:
logits rel. RMS diff 9.9e-7, max 1.7e-5, argmax 100% over 96 positions (component checks on layer 0: DeltaNet 1.7e-7,
MoE block 3.5e-8). mlx-lm cannot load the full model on this Mac.

Streaming the full 40-layer model, one request, 41-token chat prompt, 64 new tokens, prefetch on, single runs
(`results/stream_v2/qwen36_35b_real*.json`); text coherent ("Here's a thinking process: 1. Analyze User Input..."):

| experts resident per layer (of 256) | resident memory | decode hit rate | decode tok/s (first run / with fast path + overlap) | TTFT |
| --- | --- | --- | --- | --- |
| 32 | 3.7 GB | 86% | 4.27 / 4.53 | 3.7-3.8 s |
| 64 | 5.9 GB | 90% | 4.87 / - | 3.4 s |
| 96 | 8.2 GB | 94% | 5.35 / 7.35 | 3.0-3.1 s |

Where the time went: with all experts resident (8-layer bench model, 99.6% hit) streaming decoded at 18 tok/s vs 61
in memory: a host sync per layer for the routing (3.4 ms wait + 1.0 ms Python per layer). A fast path (all experts of
the layer resident: one call, `mx.async_eval`, prefetch issued after the GPU is busy) took that to 27-33 tok/s; when a
layer has misses, the resident experts' pairs now run while the misses are read. At 32 slots the remaining limit is the
SSD: ~0.31 GB of misses per token at ~3 GB/s.

Public reference (not comparable: different quantization and placement): llama-server CPU-only Qwen3.5-35B-A3B
UD-IQ3_XXS (~13 GB, fits in RAM) on a Mac mini M4 16 GB, ~17 tok/s second-hand (`results/public_baselines.md`).

### 1d.3 Speculative expert dispatch (2026-11-16): +42-53% streamed decode, bit-identical

Profile of real gpt-oss-20b streaming at 24 slots (one request, decode): 82.9 ms/token, of which 2.25 ms/layer was
the host waiting for the routing and 1.36 ms/layer host work after it with the GPU idle. A replay of the same
routing with no per-layer sync ran 56.7 ms/token with the same tokens, so the sync, not the GPU, was the overhead.

Change (`expert_stream.run_moe`, shared by every model): for small batches (tokens x k < 64) the all-resident expert
computation is queued *before* the routing reaches the host, with slots looked up on the GPU from a snapshot of the
residency map (experts whose reads are still in flight count as absent). When the host sees the routing, if every
routed expert was resident in the snapshot the queued result is used (same calls and shapes as the fast path, so bit
for bit the same); otherwise it is discarded and the existing exact path runs. `expert_stream.SPECULATE = False`
turns it off. Tests: `test_speculative_dispatch_is_exact` (on/off identical tokens), all 44 MLX tests pass.

ABAB on this Mac, same session (`results/spec_dispatch/`), greedy, prefetch on:

| model, setting | off tok/s | on tok/s | paired ratio | layer calls kept / discarded | tokens on vs off |
| --- | --- | --- | --- | --- | --- |
| gpt-oss-20b, 24 slots, 512-token prompt, 128 new, round 1 | 11.82 | 17.15 | 1.45 | 2707 / 341 | - |
| gpt-oss-20b, same, round 2 | 12.17 | 17.33 | 1.42 | 2707 / 341 | - |
| gpt-oss-20b, 24 slots, chat prompt, 64 new | 11.40 | 16.70 | 1.47 | - | identical (64/64) |
| Qwen3.6-35B-A3B 4-bit, 96 slots, chat prompt, 64 new, round 1 | 6.16 | 9.42 | 1.53 | 916 / 1604 | identical (64/64) |
| Qwen3.6-35B-A3B, same, round 2 | 6.41 | 9.53 | 1.49 | 916 / 1604 | - |
| Qwen3.6-35B-A3B, 32 slots (3.65 GB resident), single round | 5.99 | 7.29 | 1.22 | 127 / 2393 | identical (64/64) |

GPU-rail energy per token (whole run incl. prefill, `gpu_j_per_token`) moved differently per model:
- gpt-oss-20b: up, 0.48 -> 0.59, 0.38 -> 0.50 and 0.32 -> 0.41 J/token (GPU power 3.5-4.2 -> 5.9-6.9 W);
- Qwen3.6-35B-A3B: down, 0.243 -> 0.220 and 0.285 -> 0.224 at 96 slots, 0.180 -> 0.093 at 32 slots.
Why the two differ is not established (not investigated). The rail does not see CPU/SSD/system power, which run for
30-35% less time per token in every case; whole-system energy needs root (user's call). (Energy split checked by the
PM session against the run files.)

Tried and dropped (2026-11-17): on a speculative miss, keep the resident experts' rows and recompute only the
absent experts' rows (Qwen discards 59% of its speculative dispatches at 96 slots). Real Qwen3.6, chat prompt, 64
new, ABAB (`results/spec_patch/`, GPU slow state): 96 slots 12.48 -> 12.79 and 12.45 -> 11.55 tok/s, 32 slots
7.48 -> 7.27; tokens identical on vs off (64/64 at both). No gain: the GPU is mostly idle there (0.5-0.7 W) and the
SSD reads, not the recomputed rows, set the pace. Code removed.

Tried and dropped (2026-11-17): prefetching the next layer's top-(k + N) predicted experts. Real Qwen3.6, chat prompt,
64 new, paired rounds (`results/stream_pfwide/`), tokens identical in every arm:

| slots | extra predicted | tok/s (r1 / r2) | SSD GB per decode step (new tokens - 1) | decode hit rate |
| --- | --- | --- | --- | --- |
| 96 | 0 (current) | 13.51 / 13.33 | 0.116 | 0.937 |
| 96 | 4 | 11.17 / 11.48 | 0.172 | 0.956 |
| 96 | 8 | 8.72 / 9.52 | 0.241 | 0.967 |
| 32 | 0 (current) | 7.52 | 0.312 | 0.857 |
| 32 | 4 | 6.16 | 0.440 | 0.912 |

More hits but +48-108% SSD bytes on an SSD already near its 3.3 GB/s ceiling, and more in-flight reads invalidate
speculative dispatches (kept 916 -> 788 -> 616). Code removed. Qwen's own floor: replaying the same routing with no
per-layer sync runs 49.6 ms/token vs ~80-86 ms normally; the difference is layers whose misses were not predicted
(host waits for routing, then for the SSD, GPU idle). Offline, LRU vs LFU differ by <= 0.2 ms at 96 slots on a
573-token Qwen trace.

Where the remaining ~58 ms/token (gpt-oss, 24 slots) goes (profile and ablation runs, single runs):
- large matmuls (bf16 qkv / o, MXFP4 experts) at 74-93 GB/s: ~32 ms/token by microbenchmark;
- 4-bit candidate head: 4.4 ms;
- SSD: 3.3 expert reads/token (44 MB, 13.9 ms of SSD at the measured 3.17 GB/s ceiling, one 13.2 MB expert = 4.2 ms
  whatever the chunking), ~9.7 ms/token of it on the critical path;
- the 9.2 GB resident set itself: an empty decode step (no attention / MoE) takes 8.3 ms at 24 slots vs 4.3 ms at 4
  slots, with the machine 8-9 GB into swap from other apps.
Cache-policy headroom, simulated offline on a recorded 765-token routing trace (3 prompts): at 24 slots LRU +
1-ahead prefetch (current) stalls 9.7 ms/token; LFU or 2-ahead prefetch save <= 0.5 ms; even the offline optimum
(Belady, no prefetch) stalls 6.9 ms. Not worth pursuing at this budget.

### 1d.4 Opt-in 8-bit attention weights (llama.cpp's choice), measured: faster but clearly lossy

llama.cpp's gpt-oss GGUF stores attention q/k/v/o, the head and the embedding as Q8_0 (`public_baselines.md`).
`GptOss(attn_bits=8)` (`--attn-bits 8` in the run tools) quantizes the attention weights the same way (affine 8-bit,
group 32); default stays the checkpoint's bf16.

Accuracy on the real weights (`tools/mlx_real_eval.py`, 1296 teacher-forced positions, pinned data; vs the float32
reference): bf16 precise top-1 96.9%, mean KL 0.016; **8-bit attention 93.8%, mean KL 0.050** (3.2x), worse than
mlx-lm bf16 (95.4%, 0.022). Free-running greedy output first differs from bf16 at token 67-111 of 128.
Speed (streaming, 24 slots, `results/stream_attn8/`, GPU in its slow state, pairs within a round): 20.72 -> 25.68
and 20.01 -> 20.36 tok/s (1.24x and 1.02x: inconclusive); resident 9.24 -> 8.68 GB; GPU-rail J/token 0.26 -> 0.23 and
0.27 -> 0.24. Not recommended as a default; kept as an opt-in for users who accept llama.cpp-like accuracy.

Measurement note: gpt-oss's chat template stamps today's date ("Current date: ..."). An eval run after midnight
therefore scores different token sequences; this looked like a regression (precise KL 0.016 -> 0.039, mlx-lm bf16
0.022 -> 0.044) until the date was pinned (`PINNED_DATE` in `tools/mlx_real_eval.py` and `tools/mlx_stream_run.py`),
after which the earlier numbers reproduced to every digit.

### 1f. Qwen3.5-9B (dense, 4-bit, fits in memory) vs mlx-lm, and the recurrent-state layout (2026-11-17)

ABAB, same session, `tools/mlx_bench.py` levels 1 / 8 / 32 (`results/qwen35_9b_abab_v3/`, two rounds, each
round: mlx-lm, ours with the old whole-pool state, ours with per-slot state). Ratios ours / mlx-lm within a round:

| level | tok/s, pool (r1 / r2) | tok/s, per_slot (r1 / r2) | GPU-rail J/token, both layouts |
| --- | --- | --- | --- |
| 1 | 0.90 / 0.88 | 0.89 / 0.87 | 1.04-1.05 (r1; r2's mlx-lm L1 window read 0 W: counter did not advance, not used) |
| 8 | 1.10 / 1.10 | 1.10 / 1.09 | 0.84-0.85 |
| 32 | 1.15 / 1.15 | 1.14 / 1.14 | 0.82-0.83 |

Batch 1 is slower than mlx-lm (decode 53.6-55.8 vs 45.9 ms/token, direct timing): ~4-5 ms of host work per step
runs while the GPU waits (mlx-lm builds step N+1 while step N runs; a scheduler-free prototype that does the same is
48.3-48.6 ms/token, same tokens; the real fix is a lookahead plan in the Rust scheduler, held for the user's OK),
and our DeltaNet layers are ~2.6 ms/token over their bandwidth floor (component ablation; mlx-lm is at it).

Recurrent-state layout. The DeltaNet state used to be one [slots, ...] array per layer passed whole through the
kernel, which writes every slot: a step cost O(allocated slots), not O(running requests). 9B decode, 1 request
running, 32 state slots allocated: 86 ms/token vs 48 with 2 allocated (prototype loop). `SlotStates` keeps states
per slot and reuses the previous step's stacked block when the same slots run again (steady decoding: only the
kernel's own read + write), re-stacking when the set changes; released requests' states are dropped
(`MlxRunner.release`). Result: 48.4 ms/token at 32 slots allocated (= 2 allocated); batched throughput equal to the
pool layout (table above); peak memory at level 32 9.2 GB (pool 9.4; 10.8 before releases dropped states). In
`mlx_bench` the engine is created with `max_running = level`, so slots ~ level and the pool layout's cost never
shows; it matters when many slots are allocated and few requests run (a server sized for 32 serving one user).
A first per-slot version that re-stacked every step (`qwen35_9b_abab_v2/`, superseded) lost the batched win
(L32 0.97x): ~4N state transfers per layer vs the pool kernel's 2S.

Level 1 with a long window and a server-sized allocation (`results/qwen35_9b_l1_long/`, 16 requests x 128
tokens per run, ~2 min windows, two rounds; `--max-running 32` allocates KV and state slots for 32 requests while
one runs). mlx-lm's own L1 energy repeated within 3.0% (580 / 563 mJ/token), so the window is long enough:

| arm (1 request in flight) | tok/s (r1 / r2) | GPU-rail mJ/token (r1 / r2) | vs mlx-lm J/token | MLX peak |
| --- | --- | --- | --- | --- |
| mlx-lm | 18.0 / 18.0 | 580 / 563 | 1 | 5.7 GB |
| ours, per_slot, capacity 1 | 16.1 / 16.1 | 538 / 561 | 0.93 / 1.00 | 6.0 GB |
| ours, per_slot, capacity 32 | 16.0 / 16.0 | 548 / 566 | 0.95 / 1.01 | 6.7 GB |
| ours, old pool, capacity 32 | 9.7 / 9.9 | 892 / 861 | 1.54 / 1.53 | 8.3 GB |

So per_slot makes allocated capacity free (capacity 32 = capacity 1), where the old layout ran 1.6x slower and used
57% more energy per token. At batch 1 we are 0.89x mlx-lm's speed at 0.93-1.01x its GPU energy per token.

### 1g. Overlap scheduling at batch 1, and every Mac engine on Qwen3.5-9B (2026-11-17)

**Overlap scheduling** (`Engine(overlap=True)`, the Rust scheduler's lookahead plan): step N+1 is planned and queued
on the GPU before step N's tokens reach the host; its decode inputs are read from step N's lazy tokens on the
device. Same tokens as the plain loop (`tests/mlx/test_overlap.py`: gpt-oss, gpt-oss lean, Qwen3.6; preemption,
stop tokens, sampling). A lean model (embedding on disk) falls back to a host resolve: correct, not overlapped.
Direct timing, quiet machine, 256-token prompt, 128 new tokens, mean ms/token of 3 runs per arm
(`results/qwen35_9b_b1_overlap/decode_b1.json`):

| arm | r1 | r2 |
| --- | --- | --- |
| mlx-lm | 48.04 | 47.42 |
| ours, plain loop | 54.30 | 54.74 |
| ours, overlap | 47.79 | 47.69 |
| hand-pipelined prototype (no scheduler) | 47.83 | 47.65 |

Overlap: 1.14-1.15x the plain loop, matches the prototype, at parity with mlx-lm.

**Every engine on this Mac, one request** (`tools/mlx_engine_compare.py`, `results/engines_9b_b1_v2/`, two
rounds, r2 reversed). Same 256-token prompt (exact ids for the MLX engines and llama-server, the tokenizer's text
of them for LM Studio, which counts 262 tokens), 128 new tokens. Decode ms/token excludes the prefill; GPU-rail mJ
per token over the decode window. LM Studio is timed by request-length difference ((128 - 32 tokens) / 96): its MLX
stream holds back early tokens, and the streamed timing understated it (42-44 ms/token streamed vs 54.4 by
difference in the first pass, `engines_9b_b1/`, kept for the record). Weights: mlx-community 4-bit for the MLX
engines; lmstudio-community Q4_K_M GGUF (5.63 GB) for llama.cpp, so the quantizations differ.

| engine | ms/token r1 / r2 | GPU mJ/token r1 / r2 |
| --- | --- | --- |
| mlx-lm 0.32.2 | 50.98 / 47.02 | 528 / 501 |
| ours, overlap | 51.50 / 53.88 | 499 / 479 |
| LM Studio, MLX engine (mlx-llm 1.11.0) | 50.29 / 50.87 | 549 / 479 |
| ours, plain loop | 56.05 / 57.49 | 469 / 440 |
| llama.cpp llama-server (Homebrew build 12359), `-fa on -ngl 99` | 58.02 / 61.40 | 678 / 728 |
| LM Studio, llama.cpp engine (2.51.0) | 63.57 / 61.17 | 664 / 690 |

Prefill, 512 tokens, warm, from the first pass (`engines_9b_b1/`): ours 190 / 194 tok/s, mlx-lm 173 / 172,
llama-bench pp512 177.5 / 158.5 (depth 0) and 178.8 / 180.4 (depth 256). llama-bench tg128: 16.4-16.9 tok/s
(59-61 ms/token), the same as llama-server.

Reading, with a caveat: the machine was busy for both passes. macOS's call service held ~66% CPU and swap was
8.8-10 GB in use, and the morning's quiet-machine overlap number (47.7) did not repeat (51.5-53.9). Within these
rounds:
- the three MLX engines are within ~10% of each other at batch 1;
- ours uses the least GPU energy per token of them (0.94-0.96x mlx-lm with overlap, 0.88-0.89x plain);
- llama.cpp on the Q4_K_M GGUF is 1.1-1.3x slower than the MLX engines and uses 1.3-1.45x the energy per token;
- ours prefills fastest (1.1x mlx-lm and llama.cpp).
Batched, ours is 1.10-1.15x mlx-lm (1f). Re-run the batch-1 rows on a quiet machine before quoting a ranking.

### 1h. Quiet-machine re-run (2026-11-17): overlap vs plain energy, and every engine again

Quiet window: every other session on the Mac paused, LM Studio quit, a stuck MacInCallService (~70% CPU) restarted;
start/end machine state (top CPU, swap, memory pressure) saved in each dir. Two rounds, r2 reversed.

**`tools/mlx_bench.py`, Qwen3.5-9B, levels 1 / 8 / 32** (`results/qwen35_9b_overlap_abab/`; 4x level requests,
prompts 64-320, 128 tokens, EOS ignored, so `discarded_samples` = 0). Ratios vs mlx-lm in the same round, r1 / r2:

| level | mlx-lm tok/s, mJ/token | plain: speed, energy | overlap: speed, energy | overlapped_inputs |
| --- | --- | --- | --- | --- |
| 1 | 18.6 / 18.7, 624 / 618 | 0.89 / 0.89, 0.91 / 0.91 | 0.93 / 0.90, 1.03 / 1.06 | 508 |
| 8 | 44.1 / 44.2, 419 / 414 | 1.10 / 1.09, 0.85 / 0.85 | 1.12 / 1.12, 0.87 / 0.88 | 4064 |
| 32 | 67.9 / 68.2, 238 / 237 | 1.14 / 1.14, 0.83 / 0.82 | 1.15 / 1.15, 0.84 / 0.83 | 16256 |

**Long level-1 window** (`results/qwen35_9b_overlap_l1_long/`, 16 requests per run, ~2 min windows):

| arm | tok/s r1 / r2 | mJ/token r1 / r2 |
| --- | --- | --- |
| mlx-lm | 18.4 / 17.9 | 650 / 565 |
| ours, plain | 16.3 / 16.2 | 568 / 566 |
| ours, overlap | 17.8 / 17.9 | 565 / 575 |

**Every engine, one request** (`results/engines_9b_b1_v3/`, same method as 1g). Round 1 is the clean one: the
last two r2 arms ran slower and the end snapshot shows Apple's on-device inference service at 22% CPU.

| engine | ms/token r1 / r2 | GPU mJ/token r1 / r2 |
| --- | --- | --- |
| mlx-lm | 46.31 / 53.48 | 489 / 538 |
| LM Studio, MLX engine (length difference) | 46.61 / 46.73 | 505 / 507 |
| ours, overlap | 48.62 / 47.69 | 464 / 420 |
| ours, plain | 53.09 / 60.98 | 437 / 466 |
| llama.cpp llama-server, Q4_K_M | 54.41 / 54.42 | 662 / 663 |
| LM Studio, llama.cpp engine (length difference) | 59.84 / 61.43 | 686 / 636 |

Reading:
- **Batched (8-32 requests): ours is 1.09-1.15x mlx-lm's speed at 0.82-0.88x its GPU energy per token**, repeated
  in both rounds and matching 1f. Overlap adds 1-3% speed there at 1-4% more energy than plain.
- **One request: overlap is 1.10x plain** in the 2 min windows (17.8-17.9 vs 16.2-16.3 tok/s) and the direct
  runs (47.7-48.6 vs 53.1 ms). It is only 1.05x / 1.01x in the 30 s windows, where 4 requests' prefills weigh more.
  It is within 0-5% of mlx-lm and LM Studio's MLX engine.
- **Overlap's energy cost at one request is inside the noise.** It was +13-16% vs plain in the 30 s windows, ~0 in
  the 2 min windows, and -10 to +6% in the direct runs. mlx-lm's own long-window energy moved 15% between rounds.
  Overlap keeps the GPU powered between steps (higher W) but finishes sooner. Not resolved; it stays opt-in.
- **llama.cpp** on the Q4_K_M GGUF is 1.17x (server) to 1.29x (LM Studio's build) slower than mlx-lm at one request
  and uses 1.35-1.4x the energy per token. The quantizations differ (Q4_K_M vs MLX affine 4-bit).
- Of the engines measured, ours used the least GPU energy per token in most rows. The exceptions are all at level 1:
  the 30 s windows with overlap on (1.03-1.06x mlx-lm), and long-window round 2, where all three arms were within
  2% (mlx-lm 565, plain 566, overlap 575 mJ/token).

### 1i. Small-batch matmul kernel, fused elementwise ops, and llama.cpp batched (2026-11-17, quiet window 2)

**Where single-request time goes.** At one request the 9B decode step reads 4.47 GB of weights (5.04 GB of
parameters minus the 0.57 GB embedding table, which is only gathered). A chained 4-bit matvec reads at 94-99 GB/s
on this M4, so the floor is **45.3 ms/token**. Our step ran 2,088 graph ops against mlx-lm's 1,121: 450 dtype casts
and many small elementwise kernels. Fusing them with shapeless `mx.compile` (`qwen3_5_moe.py`) left 1,465 ops. The
other changes: load-time float32 copies of the conv and norm weights, one gate|up matmul, and no conv-state reset on
pure decode steps. Same-harness profile: 47.6 -> 46.3 ms/token, now 98% of the floor. Greedy tokens are bit-identical
to before (1 request, 4 requests, 4 requests with overlap). One op is left out: MLX's compiled `logaddexp` rounds
differently from the op (1 ulp on ~1/3 of values) and flipped a near-tie token, so the DeltaNet `g` keeps it outside
the compiled kernels. Faster than this at one request needs fewer bytes per token (speculative decoding); it is not
available from any engine's kernels.

**Why batched decode was slow.** `mx.quantized_matmul` (4-bit gs64, [12288 x 4096]) by rows: 1 row 281 us, 2 rows
274, 4 rows 339, 8 rows 664, 16 and 32 rows 955, 64 rows 1890. It runs a per-row matvec up to a few rows and a
32-row tile beyond. So both MLX engines' 9B decode step doubled from 4 to 8 requests (60 -> 123 ms) while the weight
read stays ~280 us. `qmm_small.py` builds MLX's own steel pieces (BlockMMA, BlockLoader, QuantizedBlockLoader,
inlined from the installed headers) with 8- and 16-row tiles. Rows are zero-padded to a whole tile, because the
bounds-checked tile load cost ~25%. It is used for 5-16 rows; everything else stays on MLX.

| shape [N x K] | 5 rows | 8 rows | 12 rows | 16 rows |
| --- | --- | --- | --- | --- |
| [12352 x 4096] (DeltaNet in_proj), ours / MLX us | 361 / 420 | 358 / 674 | 555 / 1003 | 551 / 972 |
| [4096 x 12288] (MLP down) | 392 / 415 | 366 / 651 | 581 / 960 | 569 / 1025 |
| [248320 x 4096] (lm_head) | 6688 / 8205 | 6663 / 13358 | 10547 / 20146 | 10536 / 19141 |

Pure decode step (all prompts prefilled first, median of 64 steps, overlap on): 8 requests 123.8 -> 73.1 ms, 16
requests 191 -> 122 ms. Accuracy on the real weights (`tools/mlx_vs_mlxlm.py --chunks`, the 3 texts teacher-forced
C tokens per step, against mlx-lm float32; `results/check_qwen35_9b_qmm_small.json`):

| run | mean KL | max KL | top-1 |
| --- | --- | --- | --- |
| whole-sequence prefill (MLX's 32-row tile) | 2.6e-4 | 4.4e-3 | 99.4% |
| 8 tokens per step, qmm_small | 2.0e-4 | 2.4e-3 | 98.3% |
| 8 tokens per step, mx.quantized_matmul | 6.9e-4 | 2.8e-2 | 98.3% |
| 16 tokens per step, qmm_small | 2.1e-4 | 2.5e-3 | 99.4% |
| 16 tokens per step, mx.quantized_matmul | 3.4e-4 | 7.8e-3 | 100% |

A row's result does not depend on the other rows in its tile (`tests/mlx/test_qmm_small.py`); results are not bitwise
those of `mx.quantized_matmul`, whose own batched output already differs from its per-row output (2-17 values per
shape of 25k-99k at 2-8 rows). Tried first and dropped: a batched copy of MLX's matvec with the same per-row float ops (bit-identical
to one row, but 1.06-1.15x slower than MLX at 4-8 rows in its best layout, ALU-bound by that formulation), and a dequantize-then-FMA
matvec (batch-invariant, 284 vs 338 us at 4 rows but only 5% faster at 8).

**Quiet window 2.** Every other session paused, LM Studio not running. Spotlight indexing (`spotlightknowledged`)
was at ~85% CPU from the end of the batched runs on (not in the start snapshot), and 8-10 GB of swap were in use from before; machine state is saved in each dir.
Two rounds, r2 reversed. `tools/mlx_bench.py` (closed loop, 4x level requests, prompts 64-320, 128 tokens, EOS
ignored). llama-server takes the same token ids over HTTP (`--engine llama-server`, started per level with
`-np <level> -c <level*512> -fa on`; Q4_K_M GGUF). Files: `results/qwen35_9b_batched_v1/`.

| level | mlx-lm tok/s, mJ/token | ours plain | ours overlap | llama-server |
| --- | --- | --- | --- | --- |
| 1 | 18.7 / 18.5, 629 / 614 | 16.8 / 17.1, 556 / 568 | 18.6 / 18.6, 605 / 633 | 16.0 / 16.2, 765 / 780 |
| 8 | 44.2 / 44.2, - / 413 | 69.1 / 68.9, 209 / 207 | **73.6 / 73.6**, 224 / 223 | 58.3 / 58.5, 236 / 235 |
| 16 | 50.3 / 50.4, 322 / 324 | 73.1 / 72.8, 197 / 198 | **75.1 / 75.1**, 203 / 203 | 58.0 / 57.9, 252 / 253 |
| 32 | 68.0 / 68.1, 238 / 238 | 80.3 / 80.2, 195 / 195 | **80.4 / 80.7**, 197 / 196 | 59.6 / 59.3, 231 / 234 |

(r1 / r2. mlx-lm's r1 level-8 energy read 0 J: the IOReport counter did not advance; excluded.)

- **8 requests: ours is 1.67x mlx-lm and 1.26x llama-server**, at 0.54x mlx-lm's and 0.95x llama-server's GPU
  energy per token (overlap; plain uses the least energy of all: 207-209 mJ). Before the kernel, this level was
  49.6 tok/s at 364 mJ (1h).
- **16 requests: 1.49x mlx-lm, 1.29-1.30x llama-server**, at 0.63x and 0.80-0.81x their energy. **32 requests:
  1.18x mlx-lm, 1.35-1.36x llama-server**, at 0.83x and 0.84-0.85x their energy. 32 rows take MLX's own tile; the gain there is
  from the rest of the step (1f).
- **1 request: overlap ties mlx-lm** (18.6 vs 18.5-18.7 tok/s; energy 0.96x / 1.03x) and is 1.15-1.16x llama-server at
  0.79-0.81x its energy.
- TTFT p50 under load (requests queue behind the running set): ours 29.9 vs 47.9 (mlx-lm) vs 36.5 s (llama-server)
  at level 8; 109.7 vs 137.0 vs 137.6-139.5 s at level 32.

**One request, direct** (`tools/mlx_engine_compare.py`, `results/engines_9b_b1_v4/`, median of 3):
ours (overlap) 48.13 / 47.68 ms/token at 505 / 503 mJ vs mlx-lm 47.22 / 46.67 ms at 497 / 494 mJ: **2% slower and
2% more energy** in this harness (round 1 / round 2). Both are within 3-6% of the 45.3 ms floor. By mean, round 1
reverses (ours 48.07 vs 48.48 ms) only because one mlx-lm rep ran 52.28 ms (others 45.94, 47.22); round 2, whose
reps are all within 0.4 ms, is ours 47.83 vs 46.67 ms mean: 2.5% slower. The dashboard plots means.

**Prefill, one request, warm** (`tools/mlx_ttft.py` and `llama-bench -n 0`, `results/prefill_9b_quiet/`, prompt
tok/s, r1 / r2):

| prompt | ours | mlx-lm | llama-bench (Q4_K_M) |
| --- | --- | --- | --- |
| 128 | **208 / 211** | 198 / 199 | 197 / 199 |
| 512 | 210 / 212 | **222 / 223** | 190 / 203 |
| 2048 | 209 / 213 | **220 / 223** | 203 / 203 |

Ours is 5-6% ahead of both at 128 tokens and 4-6% behind mlx-lm at 512-2048 (still ahead of llama.cpp). Prefill
steps are 32+ rows, so the new kernel is not involved; the gap is in long-prompt prefill (not investigated yet).

### 1j. MTP speculative decoding, exact multi-token steps, memory-mapped experts (2026-11-17)

Quiet-window numbers are in 1k. This section covers what was built and the checks made before that window.

**MTP speculative decoding** (`spec_decode.py`, `tools/mlx_spec_decode.py`, `tests/mlx/test_spec_decode.py`).
- Qwen3.5 checkpoints carry one MTP layer. The mlx-community conversion drops it, so `tools/mlx_fetch_mtp.py` fetches
  only the `mtp.*` tensors of Qwen/Qwen3.5-9B with HTTP range requests: 487 MB of the 19 GB checkpoint, bf16.
  The head is quantized to 4-bit (group 64) at load.
- One iteration:
  - The main model verifies [last token, d1..dk] as one (k+1)-token item.
  - The longest prefix of drafts that matches the main model's argmax is accepted.
  - The DeltaNet state and conv state roll back to the last accepted token. The gated-delta kernel's `keep_steps`
    variant writes the state after every token.
  - The MTP layer drafts the next k tokens, chaining on its own output.
- One host sync per iteration.
- The draft head scores only the 65,536 most frequent token ids (`tools/mlx_draft_vocab.py`, counted over local
  Python and man-page text, plus every special token). The full 248k-row head cost 5.9 ms per draft against 1.9 ms
  for the whole MTP layer.
- A bf16 or 8-bit MTP head gave about the same acceptance as 4-bit on the same prompts, and 4-bit was fastest.
- mlx-lm cannot speculate on this model family: its recurrent cache is not trimmable (`can_trim_prompt_cache` is
  False) and it strips `mtp.*` at load.

**Exact multi-token steps.**
- At first, verify steps changed outputs: 2 of 4 chat prompts diverged from plain greedy at token 19 and token 40,
  for every k.
- Bisected on the real 9B with `tools/mlx_vs_mlxlm.py`: steps of C tokens per forward, mean KL against mlx-lm float32.
  1. MLX's bf16 depthwise `conv1d` on multi-token items: 2-token steps 6.9e-4 vs decode 1.8e-4. Steps of 16 tokens or
     fewer now use decode's float32 gather conv (2.7e-4).
  2. MLX's 2-4 row quantized matmul rounds differently from its one-row kernel. New `qmv_rows.py` gives each row
     exactly the one-row kernel's float ops (bit-identical, `tests/mlx/test_qmv_rows.py`) at MLX's own 2-4 row
     speed (±5%, 9B shapes and its lm_head).
  3. SDPA on a short item vs the paged decode kernel (2.0e-4 vs 1.8e-4). Items of 2-4 tokens now attend row by row
     through the paged decode kernel (`build_meta`, `PAGED_ROWS_MAX`).
- Result: 1-, 2-, 3- and 4-token steps give identical numbers: mean KL 1.841e-4, max 1.505e-3, rel. RMS 6.24e-3,
  top-1 98.85% against mlx-lm float32. Whole-sequence prefill: 2.6e-4. Files:
  `results/check_qwen35_9b_spec_chunks_v2.json` (after the fixes) and `check_qwen35_9b_spec_chunks.json`
  (before the last two).
- Speculative greedy output is now bit-identical to plain greedy on all 5 test prompts (4 chat, 1 random) at
  k = 2 and 3. The tiny-model tests check equality with random, oracle and half-wrong oracle drafts.

**Memory-mapped experts** (`expert_map.py`).
- `mx.from_dlpack` of a page-aligned NumPy array is zero-copy. Checked: a write through NumPy shows in the MLX
  array. `mx.array()` copies.
- Experts are repacked part-major: one [E, *shape] block per part, 16 KB aligned. gpt-oss-20b is 9.5 GB, Qwen3.6-35B
  17 GB. Every routed expert is then a view of the page cache, which macOS pages like any file. This is llama.cpp's
  approach.
- Mapped arrays equal the stored bytes, and gather_qmm on them is bit-identical to the same call on copies.
- MLX's wired limit must stay below the mapped size. At the recommended working set, the first command buffer
  failed with Metal "Insufficient Memory".
- A model bigger than RAM (Qwen3.6-35B, 17 GB of experts) fails the same way at any setting. So does llama.cpp
  with every layer on the GPU (its 21 GB Q4_K_M GGUF). SSD streaming stays our path for those models.

**Competitor setup (for 1k).**
- vllm-mlx 0.6.0 has Qwen3.5 MTP (`--enable-mtp`).
  - Its continuous-batching engine used the MTP head only when the tensors were listed in the checkpoint's own
    `model.safetensors.index.json` (as `language_model.mtp.*`). From the `mtp/weights.safetensors` it also
    supports, acceptance was 0%.
  - Its raw-norm detection adds 1 only when a norm's mean is below 0.5. That misses `q_norm`, `k_norm` and `norm`
    of the real raw file (means 0.76, 0.75, 1.43), so we give it pre-shifted norms with the manifest it reads.
  - Its batched MTP path uses one draft whatever `--mtp-num-draft-tokens` says (stats identical for 1, 2, 3).
- Ollama 0.40.16 has an MLX runner. It imports the mlx-community 4-bit checkpoint directly (`ollama create`).
  Importing that checkpoint with MTP tensors added produced garbage text, so Ollama's MTP is tried from the
  original bf16 checkpoint (quantized by Ollama).
- llama.cpp on Qwen3.6-35B-A3B Q4_K_M (21 GB, bigger than RAM), with experts on the CPU (`-ncmoe` 20 / 30 / 40),
  single noisy runs:
  - prompt 4.2 / 3.9 / 4.9 tok/s;
  - generation 1.4 / 1.7 / 1.8 tok/s.

**Prefill gap.** Ours vs mlx-lm at 2048 tokens, interleaved, with mlx-lm prefilling the way it does in
`generate_step`: 212.6 vs 220.4 tok/s, 3.5% behind; 2.6% behind at 512 tokens.
- Each component at 2048 tokens: matmuls tie (fused and separate in_proj both 60.8 ms; MLP 182.7 vs 182.5 ms).
  Our gated-delta kernel is faster (8.5 vs 9.1 ms including mlx-lm's normalize). Host step building takes 6 ms.
- Linear-attention layers total 95.5 vs 91.7 ms and full attention 88.1 vs 86.1 ms, so about 107 ms of the 342 ms
  gap.
- A bf16 residual instead of float32 recovers about 1%. The float32 residual is kept for accuracy.
- The rest is spread over many small ops with no single cause found. Not fixed.

### 1k. Quiet window 3 (2026-11-17 15:43-16:57): every Mac engine, one request, batched, and models too big for the GPU

**Conditions.** Every other session was paused and LM Studio was quit. Two rounds, with round 2 in reverse order.
- Spotlight indexing (`corespotlightd`, `spotlightknowledged`, `mds`) ran at 40-170% CPU from about 15:56, in
  single-request round 2 and in parts of (2)-(4). Restarting it did not help: it relaunched and kept indexing.
- Root daemons (`mobileassetd`, Apple's on-device model service) spiked briefly.
- Swap was 8 GB at the start and peaked at 15.8 GB after llama.cpp's 35B runs.
- Machine state is saved in each dir. Script: `quiet3.sh`, which runs tools/mlx_engine_compare.py, mlx_bench.py,
  mlx_stream_run.py and llama-bench. Every file carries `engine_name`; MTP runs also carry `mtp_accept_rate`.

**(1) One request, Qwen3.5-9B** (`results/engines_9b_b1_v5/`, 128 new tokens, greedy, median of 3, r1 / r2).
Ours, mlx-lm, vllm-mlx and Ollama's plain arm run the same mlx-community 4-bit weights; llama-server runs the Q4_K_M
GGUF. ollama-mtp runs Ollama's own int4 import of the bf16 checkpoint (8.3 GiB loaded), because Ollama only uses MTP
from that path (1j).

| engine | chat prompt, ms/token | mJ/token | MTP acceptance | random prompt, ms/token |
| --- | --- | --- | --- | --- |
| **ours, MTP k=2** | **22.08 / 22.80** | **406 / 350** | 0.82 | 36.83 / 39.45 (acceptance 0.32) |
| Ollama 0.40.16, MTP | 33.67 / 33.73 | 518 / 504 | 0.76-0.79 | 31.58 / 32.28 (acceptance 0.83-0.91) |
| vllm-mlx 0.6.0, continuous batching + MTP | 41.65 / 42.15 | 477 / 492 | 0.82 | 118.1 / 128.6 (acceptance 0) |
| Ollama MLX runner, plain | 46.07 / 46.75 | 469 / 503 | - | 46.16 / 47.04 |
| mlx-lm 0.32.2 | 46.49 / 45.91 | 464 / 498 | - | 46.02 / 48.44 |
| ours, plain (overlap) | 46.46 / 46.66 | 455 / 505 | - | 46.67 / 49.37 |
| llama-server (Q4_K_M) | 53.77 / 54.31 | 737 / 666 | - | 54.85 / 54.45 |

- **Chat prompt: ours is 1.5x the next engine (Ollama with MTP) and 2.0-2.1x mlx-lm, and uses the least GPU energy
  per token.** Output is bit-identical to plain greedy (1j).
- **Random prompt.** Ollama with MTP looked faster, but it was not the same context:
  - The servers take text, and re-tokenizing the decoded random ids gives 262 different ids.
  - Their greedy continuation is a trivially draftable `' \t\ufffd'` loop, in Ollama and in ours.
  - On those exact ids ours runs at **19.85 ms/token (acceptance 0.97)**, vs Ollama's 31.6-32.3.
  - Source: `retokenized_random_check.json`, a single run after the window, still quiet.
  - On the original ids (raw ids, our harness only) the output is chat-marker noise and our acceptance is 0.32.
    At that acceptance MTP still decodes 1.23-1.25x faster than plain, but uses more GPU energy per token than
    plain decode (543 / 536 vs 456 / 435 mJ): rejected drafts cost verify work. Speed comes first here; an adaptive
    draft count (drop to 0-1 drafts when acceptance is low, as Ollama does) would recover the energy.
- vllm-mlx's MTP accepted 0 of 500 drafts on the random prompt, and ran at 118-129 ms/token.

**(2) gpt-oss-20b, real weights** (13.7 GB: fits RAM, not the GPU working set; `results/gptoss20b_map_v1/`;
512-token random prompt, 128 new).
- In the window rounds (r1, r2) the memory-mapped arms ran with a cold page cache, after other engines had pushed
  the expert files out: 3.9-9.5 tok/s, plus one warm run at 25.5. So two warmed rounds were added after the window:
  each engine's files were read once (`cat > /dev/null`) right before its run, and llama-bench does its own warm-up.

| engine | decode tok/s | prompt tok/s (512) | GPU mJ/token |
| --- | --- | --- | --- |
| llama.cpp llama-bench (MXFP4 experts, Q8_0 attention / head), warm | 36.4 (window rounds, cold: 28.4 / 27.7) | 432 (391 / 344) | - |
| **ours, mapped experts, 8-bit attention (opt-in, lossy as the GGUF)**, warm | **35.27 / 35.61** | 315 / 333 | 422 / 409 |
| ours, mapped experts, bf16 attention (reference weights), warm | 28.07 / 28.32 (wired 3000 MB); 27.30 / 26.20 (wired 0) | 194-241 | 453-464 |
| ours, SSD streaming, 24 of 32 slots | 16.63 / 16.97 | 120 / 133 | 388 / 378 |
| mlx-lm | does not run (swaps; RESULTS 1d) | | |

- Memory-mapped experts give 1.65-1.7x our SSD streaming.
- With the same weight format as llama.cpp's GGUF (8-bit attention), ours is within 2-3% of llama.cpp decode
  (35.3-35.6 vs 36.4 tok/s).
- With the checkpoint's own bf16 attention (exact reference weights) it is 1.3x behind.
- Prompt processing is 1.3x (8-bit: 432 vs 315-333) to 1.8-2.2x (bf16: 432 vs 194-241) behind.

**(3) Qwen3.6-35B-A3B** (20.4 GB: bigger than RAM; `results/qwen36_35b_vs_llamacpp_v1/`, 512-token random
prompt).

| engine | decode tok/s | prompt tok/s | memory |
| --- | --- | --- | --- |
| **ours, SSD expert streaming, 96 of 256 slots** | **14.59 / 14.72** | **85 / 82** | 8.2 GB resident |
| llama.cpp, all layers on the GPU (default) | Metal "Insufficient Memory" | | |
| llama.cpp, all experts on the CPU (`-ncmoe 40`, its best of 20/30/40) | 1.42 / 2.10 | 15.7 / 13.9 | whole 21 GB GGUF mmap'd |
| mlx-lm | does not run | | |

7-10x llama.cpp's decode and 5-6x its prompt speed, on the only setting where llama.cpp runs at all. Streaming hit
rate: 95%. Not the same run length: llama-bench ran a 32-token generation, one rep per round (no spread), and ours
64 new tokens.

**(4) Batched closed loop, Qwen3.5-9B** (`results/qwen35_9b_batched_v2/`, 4x level requests, prompts 64-320,
128 tokens). Run order: ours, vllm-mlx, Ollama, ours (A-B-C-A).
- vllm-mlx and Ollama take text and cannot ignore EOS, so their windows count the tokens that arrived. vllm-mlx
  delivered 95% at level 8 and 90% at level 32.

| level | ours r1 / r2, tok/s, mJ/token | vllm-mlx | Ollama |
| --- | --- | --- | --- |
| 1 | 17.0 / 18.5, 572 / 626 | 14.9, 695 | (single request: above) |
| 8 | **68.6 / 68.4**, 210 / 230 | 44.8, 389 | 18.6, 595 |
| 16 | **74.4 / 75.1**, 206 / 203 | 44.6, 352 | - |
| 32 | **79.8 / 80.9**, 198 / 197 | 69.8, 229 | 18.1, 560 |

- Ours is 1.53x vllm-mlx at 8 requests, 1.67x at 16 and 1.14x at 32, with less GPU energy per token. The second run
  of ours matches the first within 2%, so the Spotlight load did not move these numbers.
- Ollama's MLX runner never had more than one request in flight (`max_in_flight` 1), even with OLLAMA_NUM_PARALLEL=32.
  It serves requests one at a time, about 18 tok/s at any level.
  - Its window-3 run crashed: our harness only measured while `level` requests were in flight.
  - The harness now falls back to whole-run throughput. The rerun was `r1b_ollama.json`, 1 request per level, after
    the window, still quiet.
- vs mlx-lm and llama-server at these levels (1i, window 2): ours 1.18-1.67x and 1.26-1.36x.

**Where this leaves us** (this Mac, these models):
- Fastest at one request (chat prompt), and with the least energy.
- Fastest batched at 8-32 requests, against mlx-lm, llama.cpp, vllm-mlx and Ollama.
- The only engine that runs Qwen3.6-35B at a usable speed.
- Behind llama.cpp on gpt-oss-20b: within 3% in its weight format, 1.3x slower in the exact one.
- Prompt processing: behind llama.cpp on gpt-oss-20b by 1.3-2.2x, and behind mlx-lm by 2.6-3.5% at 512-2048
  tokens (1j).

### 1l. Adaptive MTP draft count, and where a verify step's energy goes (2026-11-17, after a restart)

Run after a Mac restart (swap 0 at boot, 2.4-2.6 GB during the runs), no other model sessions. Spotlight
(`spotlightknowledged`) ran at 87-90% CPU through the A/B runs. Single machine, interleaved arms, two rounds (round 2
in reverse order); treat as preliminary until a quiet-window repeat.

**What was built** (`spec_decode.DraftPolicy`, `SpecGenerator(policy=...)`; tests in `tests/mlx/test_spec_decode.py`,
12 pass).
- Each iteration picks k = 0-3 drafts to maximize expected tokens per unit cost. Acceptance is tracked per depth as a
  conditional probability (moving average); expected tokens E(k) = 1 + sum over j <= k of p_1 ... p_j.
- Cost per draft count: measured joules per iteration (`--k auto-energy`, table from `tools/mlx_spec_calibrate.py`) or
  measured seconds (`--k auto-time`, updated online, seeded with the calibrated table).
- k = 0 is plain decoding. It skips the MTP layer; the MTP layer's KV for the skipped positions is filled in one pass
  when drafting resumes. Every 8th iteration probes with one draft so acceptance keeps being measured.
- k = 0 runs are pipelined like the plain engine (next step queued before the host waits). Before this they were
  1-12% slower than plain decode; now within 1%.
- Output equals plain greedy on the real 9B for k = 0 and the adaptive policy, 4 chat prompts and the random prompt
  (`tools/mlx_spec_decode.py`, token for token).

**Calibration** (`results/spec_9b/calibrate_v1.json`, chat and random prompt, 3 runs each, medians):

| drafts k | 0 | 1 | 2 | 3 |
| --- | --- | --- | --- | --- |
| GPU mJ per iteration | 463 | 738 | 912 | 1203 |
| ms per iteration | 47.6 | 57.0 | 61.3 | 76.9 |

**Why extra drafts cost so much energy.** Measured per call, looped, pipelined (scratch profile, one run):
- The main model's forward for 1 / 2 / 3 / 4 tokens: 40.3 / 40.9 / 42.1 / 53.3 ms but 417 / 713 / 933 / 1050 mJ
  (10.4 / 17.4 / 22.2 / 19.7 W). Time stays flat to 3 tokens, energy grows about 300 mJ per extra row.
- MLX's own 4-bit matmul does the same on one 24576 x 4096 matrix: 1 / 2 / 3 rows 0.56 / 0.54 / 0.54 ms, 5.1 / 8.4 /
  11.8 mJ. So the GPU energy counter mostly reflects arithmetic per row, not weight reads; this is not specific to
  our kernel. One draft pays for itself in energy only when its acceptance is above about 0.58 (738 / 463 - 1).
- The 4-token step is slow (+11 ms over 3 tokens) because the 4-row kernel is 1.3x MLX's 4-row speed
  (0.86 vs 0.66 ms on that matrix; MLX's 4-row path is not bit-identical to its 1-row path). Staging each block of
  inputs in threadgroup memory was tried: still bit-identical, but slower at every row count (e.g. 4 rows 1.14 ms).
  Reverted.
- Waiting for each step vs queueing the next one first made no power difference at k = 0 (10.2-10.6 W both ways).

**A/B** (`results/spec_9b/adaptive_v1/`, speed-first rerun in `adaptive_v2/`; 128 new tokens, median of 3, r1 / r2):

| prompt | arm | ms/token | GPU mJ/token | acceptance | iterations by k (0/1/2/3, 3 runs) |
| --- | --- | --- | --- | --- | --- |
| chat | plain (overlap) | 46.84 / 46.89 | 455 / 506 | - | - |
| chat | k = 1 | 29.47 / 26.40 | 389 / 422 | 0.92 | 0 / 198 / 0 / 0 |
| chat | k = 2 | 22.19 / 22.25 | 389 / 362 | 0.82 | 0 / 0 / 144 / 0 |
| chat | **adaptive, energy** | 22.78 / 22.94 | **358 / 339** | 0.82 | 0 / 0 / 144 / 0 |
| chat | adaptive, time (seeded, v2) | 22.30 / 21.95 | 387 / 382 | 0.82 | 0 / 0 / 144 / 0 |
| random | plain (overlap) | 46.98 / 47.26 | 467 / 437 | - | - |
| random | k = 1 | 40.84 / 46.47 | 486 / 537 | 0.40 | 0 / 273 / 0 / 0 |
| random | k = 2 | 36.92 / 42.04 | 559 / 600 | 0.32 | 0 / 0 / 231 / 0 |
| random | **adaptive, energy** | **38.64 / 38.73** | **448 / 447** | 0.56 | 201 / 30 / 63 / 0 |
| random | adaptive, time (seeded, v2) | 42.53 / 39.91 | 523 / 539 | 0.34 | 135 / 22 / 134 / 0 |

(v2's own k = 2 arm: chat 22.71 / 23.15 ms, 393 / 344 mJ; random 37.24 / 37.37 ms, 543 / 545 mJ.)

- **Energy mode is the setting to use.** On chat it chooses 2 drafts on every iteration, the same as fixed k = 2,
  and its energy (358 / 339) is within the run-to-run spread of k = 2's (389 / 362 in v1, 393 / 344 in v2); 21-33%
  below plain. On the random prompt it falls back
  to plain decoding in the hard stretches: energy at plain's level (448 / 447 vs 467 / 437), still 1.2x faster
  than plain. Fixed k = 2 there uses 1.2-1.4x plain's energy.
- Time mode did not beat fixed k = 2. Acceptance on the random prompt comes in bursts; in the bad stretches the
  policy stops drafting and its probes every 8 iterations miss the good ones. Unseeded (v1) it also picked 3 drafts
  too often (chat 435 / 379 mJ).
- The first run's unseeded time-mode rows are in `adaptive_v1/` (chat 21.91 / 22.95 ms, 435 / 379 mJ; random 40.56 /
  36.37 ms, 616 / 624 mJ).
- Machine state: `adaptive_v1/machine_state_end.json`, `adaptive_v2/machine_state_{start,end}.json` (the
  calibration run's and v1's start snapshots were lost to a wrong flag in the run script).

### 1m. gpt-oss-20b: lossless 12.25-bit attention weights, and why its prefill is slow on 16 GB (2026-11-17)

**Lossless packed bf16** (`bf16_pack.py`, `GptOss(attn_pack=True)`, `--attn-pack` in tools/mlx_stream_run.py;
tests `tests/mlx/test_bf16_pack.py`).
- gpt-oss-20b's bf16 tensors use few exponents per group: exponent entropy 2.6-3.4 bits (scratch analysis of every
  bf16 matrix; entropy bound 10.96 bits per weight overall).
- Layout per matrix: the bf16 low byte unchanged (8 bits), 4 bits of sign plus exponent offset from the group's
  largest exponent, one base byte per group of 32 weights. Weights more than 13-14 binary orders below their group's
  largest are stored aside (CSR by row) and added back in the kernel: 2e-4 of attention weights.
- 0.768 of bf16's bytes on the real attention matrices (qkv 5120 x 2880, o 2880 x 4096). Unpacking gives every
  weight bit for bit (checked on the real matrices and in the tests).
- Steps of 1-4 tokens read the packed matrix directly (a matvec with float32 accumulation and the bias fused).
  Longer steps rebuild the bf16 matrix per call and use MLX's matmul, so prefill computes the same numbers as before.
- Per call on the real layer-0..3 matrices, one row, cycling 4 layers (one run): qkv 216 us vs MLX's bf16 297 us;
  o 194 vs 246 us. GPU energy per call 17-19% higher (unpacking is arithmetic, and this GPU's energy counter tracks
  arithmetic, 1l). Two and four rows: faster and lower energy than MLX's bf16 matmul.

**Real gpt-oss-20b, mapped experts, 512-token random prompt, 128 new tokens** (`results/gptoss20b_pack_v1/`,
wired 3000 MB, expert files read once before each run, A-B then B-A; Spotlight at ~92% CPU throughout):

| arm | decode tok/s (r1 / r2) | time to first token, s | GPU J/token | resident GB |
| --- | --- | --- | --- | --- |
| bf16 attention (default) | 27.02 / 29.67 | 1.43 / 1.49 | 0.388 / 0.454 | 11.78 |
| **packed attention** | **31.41 / 31.32** | 1.33 / 1.36 | 0.396 / 0.399 | 11.48 |

- +6% to +16% decode with the checkpoint's exact weights. The 128 tokens are identical to bf16 in both rounds.
- On the chat prompt (96 new tokens, one run each) greedy output first differs at token 67: the matvec sums in a
  different order than MLX's bf16 matmul, so near-ties can round the other way. Decode there: 27.2 vs 34.6 tok/s.
- Accuracy, real weights, 1296 teacher-forced positions vs the float32 reference (`tools/mlx_real_eval.py`, new config
  `mj_precise_pack` sends every position through the decode matvec in 4-row chunks; `real_eval_pack.json`):
  bf16 precise top-1 96.9%, mean KL 0.0157, p99 0.225; **packed 96.3%, 0.0134, 0.183**. Same accuracy within noise,
  vs 8-bit attention's 93.8% and 0.050 (1d.4).
- So exact weights now reach 31.3-31.4 tok/s vs llama.cpp's 36.4 (lossy Q8_0 attention, 1k): 1.16x behind instead
  of 1.3x. Opt-in for now (`attn_pack=True`).

**Prefill on 16 GB: limited by memory, not compute.**
- Page-ins during each 512-token prefill (scratch run, 4 prefills per process, mapped experts warm): 0.78-0.94 GB
  with bf16 attention and 0.68-1.14 GB with 8-bit attention. Prefill took 2.14-2.70 s and 2.16-2.43 s: no
  difference between the two in that run.
- The 9.5 GB of mapped experts do not all stay in the page cache next to the rest of the process and the other
  apps; a prompt touches nearly every expert, so ~0.9 GB is faulted back in 16 KB pages per prefill.
- Tried: asking macOS to read the next 2 or 4 layers' expert files ahead (fcntl F_RDADVISE) with one sync per layer.
  Worse: 7.1-7.8 s per prefill and ~10.5 GB read each time (reading ahead evicts pages just before they are used).
  Reverted.
- In the A/B above, where the expert files were read right before each run, a 512-token prefill took 1.33-1.49 s
  (344-385 tok/s), vs llama.cpp's 432 tok/s warm (1k): 1.1-1.25x behind, against 1.8-2.2x in the window-3 runs.
- So the prefill gap to llama.cpp on gpt-oss here is mostly a memory-capacity problem; it does not appear on a Mac with
  room for the whole model in the page cache (not tested: no such Mac here).

### 1n. MTP speculative decoding for several requests at once (2026-11-17)

**What was built** (`spec_decode.SpecBatch`, `tools/mlx_spec_batch.py`; tests in `tests/mlx/test_spec_decode.py`).
- A fixed batch of requests decoding together, outside the scheduler: each request has its own state slot and KV
  block range. The Rust scheduler has no multi-token decode items or rollback yet, so this is not in the serving
  engine.
- Each step verifies one item per request ([last token, its k_i drafts]); items of different lengths run in one
  step. Rejected drafts are rolled back per request (DeltaNet state and conv inputs). The MTP layer runs for every
  request every step, so it never needs a backfill.
- Per-request draft counts: fixed, or a DraftPolicy per request with a cost table measured per batch size
  (`--calibrate` from fixed k = 0..3 runs: `results/spec_batch_9b/batch_cost_v1.json`).
- New policy objective `edp` (energy x delay per token). At one request, 0, 1 and 2 drafts cost about the same
  energy per token (429 / 448 / 447 mJ in calibration), so an energy-only policy was indifferent and drifted to 0
  drafts, giving up the 2x speedup. `edp` picks the faster one when energy ties.
- Fix found on the way: steps take decode's float32 conv path when every item is 16 tokens or fewer (before: when
  the whole step was, which a batch of short verify items can exceed).
- Tests: 3 requests of different lengths, fixed k = 0-3, per-request policies, and oracle drafts right for one
  request and always wrong for another: output equals plain greedy per prompt on the tiny model (float32).

**Real Qwen3.5-9B, chat prompts** (8 different prompts cycled; 128 new tokens each, EOS ignored; decode window only:
from the step after which every request has its first token; median of 2 runs; plain = the engine with overlap).
Calibration (`calibrate_r1.json`, fixed k, tok/s and GPU mJ/token):

| requests | k = 0 | k = 1 | k = 2 | k = 3 |
| --- | --- | --- | --- | --- |
| 1 | 19.0, 429 | 30.0, 448 | **38.8**, 447 | 31.0, 488 |
| 2 | 35.6, 323 | 44.7, 327 | **57.0, 244** | 56.5, 252 |
| 4 | 53.6, 286 | **76.8, 161** | 65.5, 196 | 71.0, 187 |
| 8 | **97.1**, 138 | 91.1, **137** | 78.2, 187 | 79.7, 186 |

Against the plain engine (`r1.json`, `r2.json`, `r3_edp.json`, three rounds run one after another):

| requests | plain (r1 / r2 / r3) | best fixed k (r2) | adaptive, edp (r3) |
| --- | --- | --- | --- |
| 1 | 20.7 / 21.6 / 20.8 tok/s, 463 / 490 / 479 mJ | k = 2: 37.9, 405 | **39.3, 429** (1.89x) |
| 2 | 41.6 / 42.0 / 42.2, 407 / 379 / 414 | k = 2: 57.9, 236 | **59.4, 236** (1.41x) |
| 4 | 60.6 / 60.4 / 60.5, 302 / 304 / 307 | **k = 1: 77.4, 161** (1.28x) | 69.6, 182 (1.15x) |
| 8 | 108.1 / 110.3 / 109.8, 148 / 148 / 149 | k = 1: 92.8, 134 | 76.4, 163 |


- **1-4 requests: speculation wins on speed and energy.** 1.9x at one request, 1.4x at two (43% less GPU energy per
  token), 1.15-1.28x at four (40-47% less).
- **8 requests: plain decoding is faster** (108-110 tok/s vs at best 92.8). The batch already fills the GPU; even
  k = 0 in this loop (97 tok/s) is slower than the engine because the loop waits for every step instead of queueing
  the next one. Rule for now: no speculation at 8+ concurrent requests on this Mac.
- The energy-only policy with the measured table (r2: 20.0 / 40.5 / 68.2 / 76.3 tok/s at 1 / 2 / 4 / 8) and with the
  one-request estimate (r1: 24.8 / 37.7 / 55.9 / 94.8) both drafted too little at 1-2 requests. The estimate
  (`batch_energy_cost`) was wrong because verify rows are much cheaper per row in the multi-row matmul (5-16 rows)
  than in the 1-4 row matvec.
- At 4 requests, fixed k = 1 beats the adaptive mix (77.4 vs 69.6 tok/s): mixed draft counts chain the MTP layer to
  the largest k for every request. Not tuned yet.
- Exactness: batched steps are not bit-identical to one request at a time, for plain decoding too. Requests 2 and 3
  of 4 (scratch check, 128 tokens): plain batched first differs from single-request decoding at token 19 / never;
  speculative batched at 25 / 108. Steps of 5-16 rows use `qmm_small` (mean KL 2.0-2.1e-4 vs decode's 1.84e-4, 1i),
  over 16 rows MLX's tiled matmul (prefill's 2.6e-4): near-ties round differently.
- Conditions: no other model sessions; the machine had restarted earlier; swap 2.7-3.6 GB. Machine state in
  `machine_state_r1_start.json` and `machine_state_r2_start.json`. One round per configuration: preliminary.
- The 3-draft, 8-request arm once ran out of GPU memory: rollback keeps the recurrent state after every row (about
  1.5 GB at 32 rows). The tool now builds one loop at a time and records a failed arm instead of stopping.

### 1o. Quiet window 4 (2026-11-17 23:10-23:33): repeat of 1l, 1m and 1n

**Conditions.** Mac restarted earlier the same evening; no other model sessions or servers; the Mac was not used.
Script and log: `results/quiet4/run.sh`, `run.log`; machine state before and after each part.
- Start: load 1.4-2.0, WindowServer 19% CPU, swap 2.6 of 4 GB.
- After part A (23:18): Apple's media analysis and on-device model services were active (`mediaanalysisd` 52%,
  `TGOnDeviceInference` 28%, `aned` 19%). The random-prompt round 2 of part A ran last before that snapshot; its
  plain and mlx-lm arms were also slower than in round 1 (48.9 vs 46.8 and 46.8 vs 45.7 ms).
- After part B (23:31): quiet (top process 5%).
- End (23:33): Spotlight (`corespotlightd`) at 94% CPU; swap 4.4 of 5 GB after the gpt-oss runs.
- Two rounds per part, round 2 in reverse arm order; one-request arms median of 3 runs, batched median of 2.

**(A) One request, Qwen3.5-9B** (`quiet4/b1/`; ms/token, GPU mJ/token, r1 / r2):

| arm | chat | random |
| --- | --- | --- |
| mlx-lm 0.32.2 | 47.31 / 45.53, 445 / 498 | 45.66 / 46.81, 476 / 429 |
| ours, plain | 46.37 / 46.15, 447 / 503 | 46.79 / 48.88, 495 / 438 |
| ours, MTP k = 2 | 22.47 / 23.11, 376 / 339 | 35.64 / 42.92, 620 / 643 |
| ours, adaptive energy | 22.18 / 22.83, 375 / 348 | 38.88 / 46.22, 475 / 459 |
| ours, adaptive edp | 22.16 / 22.92, 384 / 345 | 39.22 / 44.92, 492 / 508 |

- Chat: 1.97-2.1x plain and mlx-lm, with 14-33% less GPU energy per token than plain. All three MTP arms chose 2 drafts on every
  iteration, so they are the same computation; their energy differences are run-to-run spread.
- Random: adaptive energy matches plain's energy (475 / 459 vs 495 / 438) and was 1.2x faster in round 1; in round 2
  it was no faster than plain (46.2 vs 48.9 plain in that round). Fixed k = 2 is the fastest there but uses
  1.3-1.5x plain's energy. edp sits between the two (459-508 mJ).
- Reproduces 1l within its spread.

**(B) Batched, Qwen3.5-9B, chat prompts** (`quiet4/batch/r1.json`, `r2.json`; tok/s, GPU mJ/token, r1 / r2):

| requests | plain | MTP k = 1 | MTP k = 2 | adaptive edp |
| --- | --- | --- | --- | --- |
| 1 | 21.2 / 21.6, 509 / 493 | 29.6 / 29.9, 466 / 445 | 37.3 / 38.4, 389 / 424 | **37.7 / 39.1**, 430 / 440 |
| 2 | 41.4 / 41.9, 360 / 349 | 45.2 / 46.0, 325 / 329 | 57.8 / 58.2, 245 / 237 | **59.0 / 59.7, 243 / 245** |
| 4 | 59.3 / 60.3, 291 / 302 | **77.2 / 78.5, 161 / 168** | 67.5 / 66.5, 219 / 207 | 70.2 / 70.1, 185 / 178 |
| 8 | **109.6 / 110.1**, 147 / 147 | 93.3 / 93.4, **138 / 139** | 79.6 / 78.3, 199 / 188 | 77.6 / 75.8, 175 / 159 |

- 1 request: 1.75-1.81x plain. 2 requests: 1.42x with 30-32% less energy. 4 requests: fixed k = 1 1.30x with 44-45%
  less energy; the adaptive mix 1.16-1.18x. 8 requests: plain is fastest (1.18x MTP k = 1); k = 1 uses 6% less energy.
- Reproduces 1n: every cell within a few percent of the earlier runs.

**(C) gpt-oss-20b, real weights** (`quiet4/gptoss/`; 512-token random prompt, 128 new; ours with mapped experts,
wired 3000 MB, expert files read right before each run; llama-bench `-p 512 -n 128 -fa 1 -r 3` on the MXFP4 GGUF):

| engine | decode tok/s (r1 / r2) | prompt tok/s (r1 / r2) | GPU mJ/token | resident GB |
| --- | --- | --- | --- | --- |
| llama.cpp llama-bench (Q8_0 attention) | 37.3 / 37.1 (sd 0.1) | 435.5 / 458.4 | - | - |
| **ours, packed attention (exact weights)** | **31.67 / 33.63** | 359 / 349 | 422 / 482 | 11.48 |
| ours, bf16 attention (exact weights) | 28.14 / 28.49 | 259 / 318 | 470 / 495 | 11.78 |

- Packed attention: +12-18% decode over bf16 with the same weights. Behind llama.cpp by 1.10-1.18x on decode (llama.cpp
  stores attention in lossy 8-bit; ours are the checkpoint's exact weights) and 1.21-1.31x on prompt processing.
- llama-bench's prompt rate is its own `-p 512` (one batch, warm); ours is the measured time to the first token,
  prefill plus one decode step. Not the same quantity, so the prompt gap is approximate.

### 1p. gpt-oss-20b: same-ids comparison with llama.cpp, two memory/launch fixes, EAGLE-3 (2026-11-18, overnight)

**The fair baseline.** llama-bench's generation test is not a greedy continuation of a real prompt. Ours and
llama-server were run on the same token ids instead (`tools/mlx_engine_compare.py --engine llama-server`, 512
random ids, 128 greedy tokens, `results/gptoss20b_server_v1/`): llama-server 27.2-27.5 ms/token, ~10 MB
system page-ins per request; ours 37.5-43.4 ms/token with 550-1013 MB page-ins. Two causes, both fixed:

1. **Copy-on-write expert mapping.** The expert files were mapped private copy-on-write (`np.memmap(mode="c")`). Once
   Metal used them (no copy), macOS treated the pages as anonymous memory: under pressure they went to swap instead
   of being dropped and re-read. Shared mapping (`"r+"`, the arrays are never written) A/B, 2 runs each, alternating
   (scratch runs): shared 33.3-35.2 ms/token, time to first token 1.27-1.39 s, swap unchanged; copy-on-write 41.6-43.4
   (one run 201 ms), 3.3-28.8 s, swap +2 GB per process. A read-only mapping (`"r"`) made Metal thrash (no progress in
   10 min). Default is now `"r+"` (`expert_map.MAP_MODE`).
2. **MLX command-buffer size.** MLX starts a new Metal command buffer once the arrays referenced by the current one
   exceed `MLX_MAX_MB_PER_BUFFER`. Each mapped expert array is one ~400 MB buffer per layer and part, so nearly every
   MoE op got its own command buffer. Decode step on the real model (scratch profile, pipelined loop): attention
   alone 9.6-10.0 ms, MoE alone 12.9-13.2, head 3.5-3.8, but all together 48.9-50.8 ms at MLX's default; 29.4-30.5 ms
   with `MLX_MAX_MB_PER_BUFFER` = 1000 / 4000 / 100000. It must be set before `mlx.core` is first imported (setting it
   later has no effect, checked); the package and every tool set it. Qwen3.5-9B decode unchanged (plain 46.4-47.0
   ms/token either way, MTP 22.0-22.9).
   - Correction (2026-11-18 02:00): 4000 is too large. It keeps more temporaries alive per command buffer, and a
     2048-token Qwen3.5-9B prefill in our engine ran out of Metal memory at 400, 1000 and 4000 (works at 40, 100, 150,
     250). gpt-oss decode step at 100 / 150 / 250 / 400 / 1000: 29.18 / 29.03 / 28.95 / 29.59 / 28.62 ms. Default is now
     150. The gpt-oss numbers in this section and 1q were taken at 4000 (no long prefills there). Files:
     `results/prefill_9b_mb/`.
   - Qwen3.5-9B prefill at 150, ours vs mlx-lm interleaved (`r1_mb150_*`, `r2_mb150_*`, prompt tok/s r1 / r2): 512
     tokens 229 / 214 vs 217 / 220; 2048 tokens 213 / 213 vs 221 / 221. The 1j gap at 2048 (3.5%) is still there. One
     earlier run at 40 gave 225 at 2048 (`mb40_mj.json`); a later one at 40 gave 214, so that was machine state.
- With both: ours 30.8-32.7 ms/token vs llama-server 27.4-28.6 on the random prompt (`gptoss20b_server_v2/`), with 20-35%
  less GPU energy per token (240-306 vs 372-385 mJ). Wired limits of 5500 / 8000 MB and madvise hints (random,
  willneed) did not help (30.4-32.9, 46-48 ms/token); 11000 MB runs out of Metal memory.

**Exact multi-token verify for gpt-oss.** Router logits of a 2-4 row step are now computed row by row with decode's own
call (the 2-4 row matmul differed by 2.9e-6, which moved MoE weights by 9.8e-4), and the sorted expert gather (rows
routed to the same expert share one read) is used from 8 (token, expert) pairs: it is row-independent and equal to
the per-pair path for one row (checked bitwise). A 4-token step's hidden states now equal 4 decode steps' exactly
(real model, max |diff| 0).

**EAGLE-3** (`eagle3.py`, `tests/mlx/test_eagle3.py`): a public EAGLE-3 head for gpt-oss-20b (nebius/EAGLE3-gpt-oss-20b,
CC-BY-4.0, 701 MB bf16; quantized to 4-bit at load), vLLM's layout: `fc` over the residual stream entering layers 2,
12 and 21, one decoder layer, 64k draft vocabulary. Verify step cost on the real model (scratch, random ids): 1 / 2 /
3 / 4 tokens 32.3 / 43.9 / 56.2 / 70.3 ms (each extra token routes to ~4 more experts per layer); a draft step ~2.7 ms.

llama.cpp with the same head (`--spec-type draft-eagle3`, the head's Q5_K_M GGUF, `results/gptoss20b_eagle_llama_v1/`,
chat prompt): with the draft on the GPU, Metal "Insufficient Memory" at every setting tried (draft max 2 / 3 / 5,
context 4096 and 1024); with the draft on the CPU, acceptance 0.54 (2.59 tokens per step) but 33.7-34.6 ms/token vs its
plain 29.4-30.3. So its best here is plain decoding.

**Chat prompt, same ids, 128 greedy tokens** (`results/gptoss20b_eagle_v1/`, quiet Mac, two rounds, round 2
reversed; median of 3 runs; ms/token, GPU mJ/token):

| engine | r1 | r2 | GPU mJ/token | output |
| --- | --- | --- | --- | --- |
| **ours + EAGLE-3, k = 2** | **26.21** | **26.18** | 277-286 | identical to ours plain (all 6 runs) |
| ours + EAGLE-3, k = 1 | 26.87 | 26.33 | **256-273** | identical to ours plain |
| llama.cpp llama-server (MXFP4 GGUF, Q8_0 attention) | 28.07 | 27.18 | 361-368 | differs from ours from token 45 (other weights) |
| ours, plain (exact weights, packed attention) | 30.72 | 30.23 | 260-297 | - |

- **gpt-oss-20b on this Mac: ours with EAGLE-3 is 1.04-1.07x llama.cpp's best, with 23-30% less GPU energy per token,
  on the checkpoint's exact weights** (llama.cpp's attention is lossy Q8_0). Acceptance 0.74 (k = 1) and 0.64 (k = 2).
- Without speculation ours is still 1.09-1.11x behind on this prompt.
- The comparison holds only where drafts are accepted (chat text). On the random-token prompt (no draftable text) the
  plain numbers above apply.

### 1q. Quiet window 5 (2026-11-18 01:20-01:31, part 2 01:32-01:35): final gpt-oss-20b and Qwen3.5-9B numbers

**Conditions.** Mac unattended, no other model sessions; load 1.1-2.4; swap 4.0 GB at the start, 5.6 GB after the
gpt-oss runs, 5.4 GB at the end of part 2 (other apps' memory pushed out by the 20B model over the night). Script, log,
machine state and summary: `results/quiet5/` (`run.sh`, `run2.sh`, `summarize.py`). Same token ids for every
engine; 128 greedy tokens; median of 3 runs; two rounds, round 2 reversed. Ours: gpt-oss-20b with the checkpoint's
exact weights (packed bf16 attention, mapped MXFP4 experts, lean head); llama.cpp: llama-server on the MXFP4 GGUF
(Q8_0 attention), its best setting here (1p).

**gpt-oss-20b** (ms/token, GPU mJ/token, r1 / r2):

| prompt | ours, plain | ours + EAGLE-3, adaptive | ours + EAGLE-3, k = 2 | llama-server |
| --- | --- | --- | --- | --- |
| chat | 32.04 / 30.34, 254 / 295 | **26.50 / 26.62, 261 / 265** | 26.37 / 27.06, 282 / 287 | 28.29 / 28.62, 364 / 370 |
| random 512 | 30.75 / 32.78, 281 / 277 | 35.56 / 34.10, 317 / 327 (a) | - | **28.46 / 27.52**, 378 / 383 |

- **Chat: ours with EAGLE-3 is 1.06-1.08x llama.cpp's best, with 22-29% less GPU energy per token.** Output identical
  to ours plain in all 12 EAGLE runs (acceptance 0.74-0.75 adaptive, 0.64 at k = 2). llama.cpp's output differs from
  ours from token 45 (other attention weights).
- Random prompt (nothing to draft): llama.cpp 1.08-1.19x faster than ours plain; ours uses 26-27% less GPU energy.
  Time to first token there: ours 1.42 s / 6.22 s (r2: 1.7 GB of page-ins), llama.cpp 1.17 s.
- (a) In this window, adaptive EAGLE-3 with no accepted drafts was 3-5 ms/token slower than plain: at k = 0 it still ran
  the draft layer and a test draft every iteration. Changed: at k = 0 the verified rows are queued and the head catches
  up every 8th iteration (`Eagle3Generator.SHADOW_EVERY`, test `test_adaptive_zero_draft_backfill`). Part 2 (r3 / r4,
  same prompt): adaptive 33.28 / 34.93 vs plain 33.12 / 30.92 ms/token. Within noise in r3, 1.13x slower in r4; its
  test drafts hit often enough on this repetitive continuation (acceptance 0.50-0.56 when it drafted) that it keeps
  trying one draft, at a 2-token verify cost. Chat after the change: 27.72 ms/token (one round).
- Page-ins per request were 10-71 MB for llama.cpp and 14-101 MB for ours on chat, but 0.5-2.6 GB for ours on the
  512-token random prompt: our prompt prefill touches nearly every expert and the page cache cannot keep them all.

- Rechecked at the corrected `MLX_MAX_MB_PER_BUFFER` = 150 (1p; `quiet5/gptoss_mb150/`, one round, 3 runs, ms/token
  median and range): EAGLE-3 k = 2 25.97 (25.81-26.89), adaptive 27.74 (26.52-28.19), plain 30.81 (30.18-31.03). Against
  llama-server's 27.04-28.62 from the same night: k = 2 is 1.05-1.10x faster, adaptive 1.02-1.03x. One k = 2 run's
  energy counter did not advance (0 mJ recorded).

**Qwen3.5-9B anchor** (chat prompt, ms/token, GPU mJ/token, r1 / r2): mlx-lm 46.17 / 45.54, 482 / 497; ours plain
47.74 / 46.50, 492 / 506; **ours MTP (adaptive, edp) 22.40 / 22.76, 372 / 359** (acceptance 0.82). Same as 1o: about 2x
mlx-lm with less energy.

**Where this leaves us** (this M4 16 GB Mac, every engine measured on the same ids):
- Qwen3.5-9B, one request: fastest by 2x (MTP), lowest energy (1o, 1q).
- Qwen3.5-9B, 1-4 requests with batched MTP: 1.3-1.8x plain with up to 45% less energy (1o); adaptive MTP is above
  plain at 1-4 requests and at parity at 8 (107.9 vs 108.9 tok/s, 1r); 8-32 requests: fastest plain engine (1k).
- gpt-oss-20b, chat text: fastest (EAGLE-3: 1.05-1.10x llama.cpp at k = 2, 1.02-1.08x adaptive) with the least energy
  and exact weights (1q).
- gpt-oss-20b, text with nothing to draft: llama.cpp 1.08-1.19x faster; ours uses ~26% less energy (1q).
- Qwen3.6-35B (bigger than RAM): 4.7-6.3x llama.cpp's decode, 4.8-6.8x its prompt speed, the only usable engine
  (1s; GPU busy only 50-66% of decode, so there is headroom).
- Prompt processing, still behind: Qwen3.5-9B 3.5% behind mlx-lm at 2048 tokens (ahead or level at 512; 1p); gpt-oss-20b
  1.2x behind llama.cpp on a 512-token prompt (1.42 vs 1.17 s), limited by the page cache on 16 GB (1m, 1q).

### 1r. Batched MTP: plain stretches pipelined, probes only where drafting can pay (2026-11-18 02:10-02:40)

Changes to `spec_decode.SpecBatch` / `DraftPolicy` (tests in `tests/mlx/test_spec_decode.py`, 25 + EAGLE tests pass):
- When no request in the batch drafts, steps are queued ahead like `Engine(overlap=True)` and the MTP layer is
  skipped; each request's verified rows wait and run through the MTP layer in one pass when drafting resumes (or
  every 64 rows). Before: every step waited for the host and ran the MTP layer.
- A k = 0 policy probes with one draft only if, at 0.95 acceptance per depth, some k would beat plain decoding by
  10% under its cost table (`DraftPolicy.PROBE_GAIN`). At 8 requests no k can, so probes were pure cost: all
  requests probed on the same step and the batch paid a verify-and-rollback step every 8 steps.
- Cost table re-measured with the new loop (`results/spec_batch_9b_pipe/batch_cost_v2.json`; mJ per
  request-iteration at k = 0..3: 1 request 496 / 767 / 938 / 1164, 2: 359 / 584 / 607 / 691, 4: 293 / 287 / 496 /
  517, 8: 146 / 246 / 466 / 507).

Qwen3.5-9B, chat prompts, decode window, tok/s and GPU mJ/token (`spec_batch_9b_pipe/`; r2 / r3 before the probe rule,
r4 after it):

| requests | plain (r2 / r3 / r4) | k = 0 in the loop | MTP k = 1 (r2 / r3) | adaptive edp (r2 / r3 / r4) |
| --- | --- | --- | --- | --- |
| 1 | 21.6 / 21.7 / 21.7 | 20.3 | 30.0 / 29.8 | 38.3 / 38.9 / **39.1** (440-447 mJ) |
| 2 | 41.2 / 41.9 / 41.8 | 41.5 | 45.5 / 46.7 | 59.6 / 59.5 / **59.2** (236-243 mJ) |
| 4 | 60.4 / 60.3 / 59.2 | 59.5 | **77.3 / 77.4** (160 mJ) | 67.6 / 67.6 / 67.3 (191-200 mJ) |
| 8 | 109.1 / 109.1 / 108.9 (147-149 mJ) | 111.5 (calibration) | 94.7 / 92.8 | 83.8 / 83.5 / **107.9** (146 mJ) |

- The zero-draft loop now matches the plain engine (8 requests: 111.5 vs 103.7-109.1; before 97 vs 108-110, 1n).
- Adaptive MTP is now at least plain at every batch size: 1.80x at one request, 1.42x at two (39% less energy), 1.14x
  at four (34% less), parity at eight (2% less energy). At four requests fixed k = 1 was still better (1.28x): the
  adaptive mix sometimes chained 3 drafts (deeper-depth estimates drifted up to the depth above while untried).
- Fixed after: untried depths drift toward 0.85 x the depth above, and start there (`DraftPolicy.DEPTH_DECAY`; measured
  acceptance falls with depth). r5 (`r5_decay.json`): 1 request 39.1 vs plain 21.6 tok/s; 2: 58.5 vs 41.9 (240 vs 416
  mJ); **4: 75.5 vs 60.3 (1.25x, 166 vs 306 mJ: 46% less)**. One request alone, chat / random prompt, unchanged:
  22.06-22.46 / 39.13-39.52 ms/token (`single_decay_*.json`).

### 1s. Qwen3.6-35B-A3B rerun after the 1p fixes (2026-11-18 06:00-06:07)

The weights were still on disk; only the scratch copy of the streaming store was gone. It was rebuilt into
`~/.cache/millijoule/qwen36-35b-experts-stream` (`tools/mlx_repack_experts.py`, 14 s). Same setup as 1k (3):
- 512-token random prompt (seed 0), 64 new tokens.
- Ours: SSD streaming, 96 of 256 slots, prefetch.
- llama.cpp: llama-bench, experts on the CPU (`-ncmoe 40`), `-p 512 -n 64` (1k used `-n 32`).

Two rounds, round 2 reversed (`results/qwen36_35b_v2/`, `run.sh`). Swap was 4.0-4.4 GB throughout.

| engine | decode tok/s, r1 / r2 | prompt tok/s, r1 / r2 | GPU J/token |
| --- | --- | --- | --- |
| **ours, streaming, MLX_MAX_MB_PER_BUFFER 150** | **15.15 / 16.12** | 78 / 91 (512 / time to first token) | 0.185 / 0.217 |
| **ours, streaming, MLX_MAX_MB_PER_BUFFER 40** | **16.88 / 15.07** | 95 / 82 | 0.189 / 0.303 |
| llama.cpp, `-ncmoe 40` | 3.19 / 2.66 | 16.2 / 13.9 | - |

- **Lead over llama.cpp:** ours is 4.7-6.3x its decode and 4.8-6.8x its prompt speed. This replaces 1k's "7-10x":
  llama.cpp's generation test is faster at 64 tokens than at 32 (2.7-3.2 vs 1.4-2.1 tok/s).
- **Outputs:** all four of our runs produce exactly the token ids of 1k's run.
- **Buffer limit:** no consistent effect on streaming. Round 1 favours 40, round 2 favours 150, so the default stays
  150.
- **Memory-mapped experts still do not run this model.** With shared mapping and a 3000 MB wired limit, the first
  command buffer fails with Metal "Insufficient Memory", as in 1j. 17 GB of mapped experts is more than the GPU can
  address here, so streaming stays the path for models bigger than RAM.
- **The GPU is busy only 50-66% of the decode window:**
  - expert hit rate 95.4%;
  - 83% of prefetched experts used (1989 of 2387).

  The rest is waiting: SSD misses, and the per-layer routing readback that streaming needs before it can issue reads.
  Not yet split; next is a GPU timeline per step from a Metal System Trace.
- **Metal System Trace of a short run** (16 new tokens, `tools/mlx_gpu_timeline.py`):
  - each layer is about 4 command buffers, about 0.9 ms of GPU work;
  - between layers the GPU waits 1-5 ms;
  - over the last 1.5 s, 448 of 505 ms of GPU idle time is in gaps longer than 0.5 ms.

  So the idle time is per layer (routing sync, SSD reads), not between kernels.
- **Why the speculative MoE pass is discarded.** Of 1386 discarded passes, 760 (55%) had every routed expert either
  resident or with its prefetch read still in flight. The prediction was right, but the read had not landed.
  Two changes were tried (one run each, during a 72 GB download, tokens identical in every run):

  | change | discarded passes | decode tok/s |
  | --- | --- | --- |
  | none | 1386 | 16.10 |
  | count reads that have already finished as resident (`HARVEST`) | 1349 | 16.13 |
  | wait for the in-flight reads while attention and the router run, then queue the pass (`WAIT_PENDING`) | 626 | 13.74 / 15.78 vs 16.22 / 15.96 off |

  - `HARVEST` is free and kept on.
  - `WAIT_PENDING` cuts discarded passes by more than half but no time: the SSD reads set the pace. It stays off.
  - The prefetch is issued one layer ahead (about 1.5 ms), and a layer's reads take longer than that.

**Rotated 3-bit experts: not worth it on these weights** (real bf16 experts of layer 1, 64 experts, relative
Frobenius error):

| format | plain | Hadamard-rotated | bytes per weight |
| --- | --- | --- | --- |
| 4-bit, group 64 (mlx-community) | 0.091 | 0.091 | 0.5625 |
| 3-bit, group 64 | 0.191 | 0.190 | 0.4375 |
| 3-bit, group 32 | 0.172 | 0.171 | 0.5 |

- The gate/up and down projections give the same numbers. The experts are near-Gaussian with no outliers, so
  rotation has nothing to spread.
- 3-bit doubles the error for 22% fewer bytes. Not made a mode.
- Lossless headroom of the 4-bit experts is also small. The codes have 3.72 bits of entropy, the bf16
  scales/biases about 8 of 16, so an ideal coder saves about 12%.

**Prefetch two layers ahead (`MJ_PREFETCH_DEPTH`, default now 2).**
- Routing prediction, measured with `MJ_PRED_STATS=1` (`stream_prefetch_depth/routing_prediction.json`): layer
  li+d's router applied to layer li's MoE input predicts this share of the routed experts in decode:
  - d = 1: 80-83%;
  - d = 2: 74-77%;
  - d = 3: 70-73%.
- At depth d, each layer also starts reads for its d following layers, nearest first. Reads for layer L+2 then
  get about two layers of time to land.
- Paired A/B, round 2 reversed (`stream_prefetch_depth/`). Same setup as above, during the download; tokens
  identical in all six runs.

| depth | decode tok/s, r1 / r2 | SSD GB per run | speculative passes kept / discarded | GPU J/token |
| --- | --- | --- | --- | --- |
| 1 | 16.00 / 15.49 | 5.87 | 1173 / 1347 | 0.210 / 0.214 |
| **2** | **16.67 / 16.80** | 6.64 | 1457 / 1063 | 0.215 / 0.209 |
| 3 | 16.60 / 16.52 | 7.33 | 1713 / 807 | 0.222 / 0.226 |

Depth 2 is 4-8% faster than depth 1. Depth 3 keeps more speculative passes but reads 25% more and gains nothing
over 2.

**More expert slots: 128 of 256 is 11-12% faster than 96** (prefetch depth 2, paired, round 2 reversed,
`stream_slots_35b/`; during the download):

| slots per layer | resident | decode tok/s, r1 / r2 | hit rate | SSD GB per run | tokens vs reference |
| --- | --- | --- | --- | --- | --- |
| 96 | 8.2 GB | 17.50 / 17.26 | 96.0% | 6.64 | identical |
| 112 | 9.3 GB | 18.74 / 17.77 | 96.6% | 5.79 | identical |
| **128** | **10.5 GB** | **19.58 / 19.14** | 97.4% | 4.74 | differ from step 11, a bf16 tie |

- **The 128-slot difference is an accumulation-order flip.** Its two runs agree with each other. At step 11 its top
  two bf16 logits are exactly equal (tokens 103916 / 104980; 96 slots: 104980 ahead by 0.31). Which experts are
  resident changes the expert calls' shapes, and so the bf16 accumulation order. The logits of the two settings
  differ by 1-2 bf16 ulp at every earlier step too (step 3: 28.125 vs 28.375; step 4: 25.5 vs 25.75;
  `margin_96.json`, `margin_128.json`). So the slot count changes the accumulation order in general. Greedy output
  depends on the slot setting whenever two tokens are this close; the 96 and 112 runs only happened to have no such
  tie. This is the known streaming behaviour (expert_stream docstring), gated in tests by the float32 near-tie rule.
  - The "bit-identical" claims for speculative dispatch (1d.3) and for HARVEST, WAIT_PENDING and prefetch depth
    (above) compare on vs off at a fixed slot count. Across slot counts, outputs are equal only up to such ties.
- **gpt-oss MoE block, compiled (`mx.compile`) vs not** (24 layers, real dimensions, random weights, in memory):
  bit-identical output, 14.23-14.37 vs 14.25-14.34 ms. It already reads its 1.27 GB of expert weights at about
  89 GB/s, so fusing its small elementwise ops does not help.
- **Cost:** swap rose from 3.9 to 6.5 GB over these runs (other apps plus the download). 128 slots (MLX peak
  11.4 GB) is for a Mac with little else running; 96 stays the conservative setting.
- With depth 2 + 128 slots, the 35B decodes at about 19 tok/s, against 14.6-16.9 at the start of the day (1k, 1s).

**Prefill: fill the next layer's slots while this layer computes** (`expert_stream.PREFILL_PREFETCH`, default on;
`stream_prefill_prefetch/`).
- **The problem:**
  - a 512-token random prompt moves 11.6 GB from the SSD with cold slots (5.8 GB when slots are warm from a previous
    request), at only 1.7-2.2 GB/s effective against the 3.2 GB/s ceiling;
  - the next-layer prefetch was skipped whenever its predicted set exceeded the slots, which is always true in
    prefill, so SSD reads and GPU compute took turns layer by layer.
- **The fix:** keep every predicted expert that is already resident, and fill the remaining slots with the most
  often predicted others.
- **First version without the keep rule:** cold prefill 8-10% faster, but warm prefill 14% slower (3.84 vs 3.37 s),
  because it evicted resident experts the prompt still needed (9.3 vs 5.8 GB read).

Prefill plus first token, 96 slots, 512 random tokens (`prefill_io.py`), on / off / on / off:

| slots | prefill prefetch on | off |
| --- | --- | --- |
| cold | 4.85 / 4.77 s (11.7 GB, 2.42-2.46 GB/s) | 5.22 / 5.23 s (11.6 GB, 2.21 GB/s) |
| warm | 3.23 / 3.24 s (6.8 GB) | 3.35 / 3.40 s (5.8 GB) |

- Full run (64 new tokens): tokens identical on vs off, time to first token 5.02 vs 5.36 s.
- Decode in that pair: 16.46 vs 17.30 tok/s, one run each, during the download. The change does not touch the
  decode path, but it leaves different experts resident when decode starts, so this is not settled; it goes in the
  quiet-window rerun.

### 1u. GPU-driven expert streaming in C++ (prototype, 2026-11-18 morning, during the download)

**Why.** Host-side timing of a streamed Qwen3.6-35B decode step, 96 slots (`host_split.py` method), 57.6 ms:

| part | ms | GPU at that time |
| --- | --- | --- |
| waiting for the per-layer routing sync | 26.1 | busy |
| waiting on SSD reads | 14.2 | idle |
| Python in `run_moe` | 11.4 | idle |
| Python outside it | 5.8 | idle |

The host and the GPU take turns.

**What it is** (`src/mj_mlx/ext/mj_stream.cpp`; build: `tools/build_mj_stream.sh`; on with
`MJ_GPU_STREAM=1`). A C++ MLX primitive, nanobind domain "mlx", linked to the venv's libmlx:
1. After each layer's router, it encodes a Metal shared-event signal into MLX's own command buffer, then a wait on a
   second event.
2. A C++ IO thread wakes on the signal (spinning up to 1 ms, then blocking), and reads the routing straight from
   unified memory.
3. It makes the routed experts resident in the same slab arrays (preadv, LRU, waiting for in-flight prefetches), and
   writes each pair's slot id into the primitive's output.
4. It signals ready, then starts prefetch reads for the next two layers' predicted experts.
5. The GPU then runs the same gather_qmm call as the speculative fast path.

Python never waits mid-step, so the engine's overlap mode applies: the next step's graph is built while the GPU runs
this one. Decode only (fewer than 64 pairs); prefill keeps the Python path, and slot ownership is handed over in both
directions.

**Exact:**
- `test_gpu_streaming_is_exact` (tiny Qwen3.6, 8 and 16 slots, 3 requests with prefill/decode handoffs) passes.
- On the real 35B: the same 64 tokens as the Python path at 96 slots.

**Paired A/B on the real 35B** (`gpu_stream_v1/`; 512 random tokens, 64 new, depth 2, ABBA; during the download):

| slots | Python path (overlap off) | C++ GPU-driven (overlap on) |
| --- | --- | --- |
| 96 | 16.47 / 16.15 tok/s | **17.58 / 17.00** (about +6%) |
| 128 | 18.55 / 19.67 tok/s | **19.39 / 21.07** (about +7%) |

What the first version taught:
- **Committing MLX's command buffer from the primitive** trips a Metal assertion: MLX attaches its completion
  handler later. MLX's own commits every few dozen ops are enough.
- **Without the engine's overlap mode** the C++ path is slower (14.6 tok/s): Python builds the whole graph first.
- **A blocking `waitUntilSignaledValue`** costs about 5 ms per step in wake latency. The 1 ms spin fixes it:
  17.3-17.7 vs 16.1 tok/s.

**What is left** (trace in C++ mode, 96 slots: GPU busy 57%, about 32 ms busy and 24 ms idle per step):
- SSD misses: 12.3 per token, about 14 ms of IO-thread read wait per token, gaps over 1 ms.
- One GPU-to-CPU-to-GPU event round trip per layer, even when every expert is resident (gaps of 100-300 µs).
- With 128 slots, misses drop to 9.1 per token.

**Learned router lookahead: no useful gain** (`tools/mlx_router_lookahead.py`, `results/router_lookahead/`).
- Data: 8 local documents (Python package docs and sources, man pages), 8192 tokens, run through the streamed 35B
  as prefill. Each layer's MoE input and routing were saved.
- Predictors: per layer, a linear map W [2048, 256] plus bias from layer j-d's MoE input to layer j's routing.
  Initialised from layer j's router (today's predictor), softmax cross-entropy against the 8 routed experts, Adam.
- Split: first 3/4 of the tokens to train, the last documents to test.

| setting | recall@8, d = 1 | recall@8, d = 2 |
| --- | --- | --- |
| router of layer j (current) | 0.780 | 0.708 |
| learned, lr 1e-3, 4 epochs | 0.647 | 0.627 |
| learned, lr 3e-5, 2 epochs | - | 0.682 |
| learned, lr 1e-5, 2 epochs | - | 0.715 |
| learned, lr 1e-4 + L2 to the router | - | 0.682 |

Overfits at about 6k training tokens. At best +0.7 points over the router. Not deployed.

**GPU keep-alive during expert reads (C++ path, default on, `MJ_KEEPALIVE`).**
- **The finding:** the GPU drops its clocks within milliseconds of idling. The same 14 MB 4-bit matvec takes, with
  the GPU idle before each call:
  - none: 366 µs;
  - 0.5 ms: 418 µs;
  - 2 ms: 1094 µs;
  - 5 ms: 1543 µs.

  Our SSD-miss waits are 1-5 ms.
- **Microbenchmark of a keep-alive:** a 32-thread spin kernel on a second queue, covering a 2 ms gap, brings the
  matvec back to 594 µs. Wider spinners (256 and 2048 threads) do worse (1092-1169 µs); they compete with the real
  kernel.
- **In the C++ streamer:** the IO thread compiles the spin kernel at startup and calibrates it. When a layer has
  misses, it launches a spin sized to the expected read time, at an assumed 2.5 GB/s.
- **Paired, 96 slots, keep-alive off / on / on / off:** 17.63 / **20.78 / 20.75** / 18.04 tok/s, about +16%. The
  IO thread now waits for the GPU's routing signals 29 ms per token instead of 40-42, so the GPU's kernels run
  faster.
- **Spin sizing:** at an assumed 1.5 GB/s 18.30 tok/s, at 4 GB/s 19.88, at 2.5 GB/s 20.53.
- **Not measured yet:** the energy cost (quiet window).

**MTP speculative decoding on the streamed 35B** (`tools/mlx_mtp_stream.py`, `results/mtp_35b/`).
- **The draft head:** the bf16 checkpoint's one-layer MoE MTP head (`mtp.*`, 1.7 GB bf16, from shards 25-26 of
  the download). Its 256 experts are quantized to 4-bit at load and held resident, 0.45 GB. MtpHead now supports a
  MoE MLP through the model's in-memory MoE path.
- **Fix 1, 8-bit row steps:** the 35B's router and shared-expert gate are 8-bit, which the 2-4 row kernel
  (`qmv_rows`) does not cover. Multi-token verify steps fell back to a multi-row matmul that rounds differently.
  `mm` now runs one decode-shaped call per row when `qmv_rows` does not apply.
- **Fix 2, strided routing:** the C++ streamer read the routing as packed `[T*k]`, but the top-k slice of an
  argpartition is strided for T > 1. So multi-token steps used the wrong experts, and prefetch the wrong
  predictions; it showed as 2-3x the misses.
  - Found by recording each layer's MoE input with oracle drafts: decode vs verify rows differed from layer 0 on
    the C++ path, and were bit-identical on the Python path (`bisect_verify.py`).
  - Fixed with `mx::contiguous` in `route()`. The GPU-streaming test now also covers batched decode.
- **After both fixes:** greedy tokens identical to plain decoding for k = 1, 2 and adaptive.

Chat prompt, 96 new tokens, 96 slots, C++ streaming with keep-alive; one warm-up, one measured run each:

| drafts | decode tok/s | acceptance |
| --- | --- | --- |
| 0 | 16.97 | - |
| 1 | 18.78 | 0.96 |
| 2 | **19.41** | 0.97 |
| adaptive (time mode, up to 3) | 19.36 | 0.95 (3 drafts in 24 of 25 iterations) |

- **Why +14% and not 2x:** each verified token routes to its own 8 experts, so a 3-token step loads about 2.5x the
  unique experts of a single step. The SSD and the expert reads dominate a streamed step.
- **128 slots + MTP:** Metal out of memory, the slabs (10.5 GB) plus the resident draft layer. 128 slots plain:
  22.27 / 21.85 tok/s through the same loop.

**GPU spin-wait instead of the per-layer event: not feasible on M4.**
- **The problem:** in C++ mode (96 slots, keep-alive) the GPU is 63.5% busy, and idles about 5.8 ms per step in
  about 36 gaps of 100-300 µs. That is one GPU-to-CPU-to-GPU event round trip per layer, even when every expert is
  resident.
- **What was tried:** a kernel, one thread per (token, k) pair, spinning with atomic loads on a tagged entry that the
  IO thread writes. With a shared buffer, and again with a write-combined one, the running kernel never saw the CPU's
  write: it hit its iteration cap, while the IO thread had finished in about 5 ms.
- **Conclusion:** CPU writes to shared memory are not visible to a running Metal kernel here. Metal events are the
  supported path. Code removed.

**Best streamed Qwen3.6-35B decode so far:** engine overlap mode, C++ GPU-driven streaming, keep-alive, depth-2
prefetch.
- 96 slots: 20.5-20.8 tok/s.
- 128 slots: 23.5-23.9 tok/s.
- At the start of the day: 14.6-16.9 tok/s at 96 slots.
- MTP (above) reaches 19.4 tok/s at 96 slots, but its loop cannot use overlap yet, so it is not the fastest path.

### 1zzf. What runs on a 16 GB M4, per engine: largest model and longest verified context (2026-11-23, compiled)

Checklist 0.10. Compiled from earlier runs (no new run); each cell cites its section. "Runs" means the suite's
tests completed with clean blocks unless marked; "longest context" is the longest that completed and, where
marked, was checked for accuracy.

| model (size) | ours | mlx-lm | llama.cpp (llama-bench / llama-server) | vllm-mlx | Ollama / LM Studio |
| --- | --- | --- | --- | --- | --- |
| OLMoE-1B-7B (4-bit, trained to 4k) | runs; QA to 3.5k, token-identical to mlx-lm (1zx) | runs; QA to 3.5k (1zx) | runs; QA to 3.5k (1zx) | runs (1zo) | run (1zo) |
| DeepSeek-V2-Lite (16B, 4-bit) | **32k** (pp 240, tg 13.6 tok/s; ~3,150 Metal over-limit lines at the default wired limit, 1 at 9000 MB; 1zp); QA 9 / 12 at 16k (1zx) | 4k QA; 8k stopped past 21 GB of swap-outs (1zx) | 2k (1zh); 8k does not fit under the default GPU working-set limit at any batch size, swaps when partially offloaded (1zi, 1zzb) | 2k; 8k / 16k invalid (no tokens streamed, 1zi) | 2k (1zh) |
| gpt-oss-20b (MXFP4) | **16k**: QA 36 / 36 exact arm unpinned, 12 / 12 with its top 16 experts pinned (1zx, 1zzd); 32k prefill ran (267-277 tok/s), decode out of Metal memory (1zp) | does not load (Metal out of memory at `mx.eval(model.parameters())`, 1zh) | 16k only at ubatch 128 (QA 11 / 12, 1zx); 32k out of Metal memory at defaults (1zp) | does not run the MXFP4 checkpoint here (1zg) | 2k (1zu) |
| Qwen3.6-35B-A3B (20.4 GB, above RAM) | **runs, experts streamed from the SSD**: tg 22.6 / 20.6 tok/s at 0 / 2k context, blocks clean (1zu); longer contexts not yet tested | not run (above RAM) | runs only with experts on the CPU (`-ncmoe 40`), every block flagged: llama-server swapped ~900 GB out per block (1zu); its expert cache cannot run it (1zs) | not run | not run |

Peak memory per block, GB (suite memory tables; footprint = macOS's figure with GPU buffers, without mapped files;
resident counts mapped weight files only where touched, so **neither figure compares across engines that map
weights differently**, e.g. llama.cpp's no-copy Metal buffers):

| model | ours footprint / resident | others (footprint / resident) |
| --- | --- | --- |
| OLMoE (v15) | 7.16 / 0.21-0.35 | mlx-lm 6.57 / 0.69-0.82; vllm-mlx 7.34 / 1.02; llama-server 4.67-4.75 / 8.29-8.58 |
| DeepSeek-V2-Lite (v15) | 11.76 / 1.44-2.49 | vllm-mlx 12.56-12.63 / 2.27-2.62; Ollama 11.18 / 12.52-12.58; llama-server 2.36 / 0.98-10.71 |
| gpt-oss-20b (v16) | exact 3.44-3.79 / 10.62-10.72 (experts mapped and pinned); 8-bit 3.53 / 10.45-10.62 | Ollama 1.59-13.40 / 11.03-11.78; LM Studio 1.69 / 1.18-12.18 |
| Qwen3.6-35B (v16) | 11.51-11.52 / 7.87 | llama-server 24.50 / 4.71-4.96 (and ~900 GB of swap-outs per block) |

- **The capability claim, on a 16 GB M4:**
  - DeepSeek-V2-Lite: ours is the only engine among those tested past 4k (Ollama / LM Studio were tested at 2k only;
    mlx-lm stopped at 8k for swap; llama.cpp does not fit at 8k). **QA-checked to 16k (9 / 12); 32k prefill and
    decode ran** at the default wired limit, with about 3,150 Metal "required > max" log lines during the 32k tests
    and no stall, kill or slowdown (the 9000 MB arm, with 1 such line, ran at the same speed); the 32k tokens were
    not checked for accuracy.
  - gpt-oss-20b: ours and llama.cpp both run it at 16k (llama.cpp at ubatch 128, the same precision); mlx-lm does
    not load it. The QA counts (ours 36 / 36 over 4k-16k, llama.cpp 11 / 12 at 16k) are not a lead: other item
    counts and other weights.
  - Qwen3.6-35B: ours is the only engine tried that runs it with clean blocks, at 0-2k context; longer not tested.
- Gaps: Qwen3.6 long context on ours; gpt-oss 32k decode on ours; Ollama / LM Studio long context; whole-process
  memory for Ollama / LM Studio on OLMoE (their worker processes were not sampled: 0.00 in v15).

### 1zze. Uneven expert slots from a short calibration: 512 tokens of in-distribution routing give -6.4% offline (2026-11-23 11:45)

Checklist 6.2b: the uneven-slot profile (1zy) came from an 8,192-token routing trace; to be the default it must come
from something a load can afford. Offline, on the same Qwen3.6-35B trace (`~/.cache/millijoule/qwen36-route-trace`:
8 documents, routing recorded teacher-forced), `tools/mlx_slot_profile_calib.py` builds per-layer LRU miss curves
from the first N tokens only, allocates the same total as 96 slots per layer (`placement.allocate_slots`), and
counts that allocation's LRU misses on the held-out second half (4,096 tokens; the cache state carried over from
the first half). `results/slot_profile_calib_v1.json`; 2 s of CPU, no model loaded.

| calibration tokens | held-out misses vs 96 per layer | slots per layer |
| --- | --- | --- |
| 64 | +10.6% (worse) | 56-144 |
| 128 | -3.0% | 60-136 |
| 256 | -5.7% | 60-160 |
| 512 | **-6.4%** | 72-172 |
| 1,024 | -6.4% | 72-156 |
| 2,048 | -6.5% | 68-172 |
| 4,096 (the whole first half) | -6.9% | 72-160 |

- **512 tokens of in-distribution routing give -6.4% offline** (93% of the full half's -6.9%); 64 tokens are worse
  than uniform slots. The calibration tokens come from the same 8-document trace as the held-out half, so this is
  an upper estimate for calibrating on other text: **calibration on `placement.CALIB_TEXT` is not yet measured**
  (record its routing, score against the same held-out half: one model load). A 512-token prefill on the streamed model takes about 3 s (pp512 171 tok/s, 1zu).
- Caveats: one model, one trace, text of one kind (the documents the trace was recorded on), prefill routing
  scored as an LRU replay, not live decode. The live A/B (1zy) measured +6.6-7.1% decode from the full-trace profile.
- **Built, opt-in** (157fab7): with `MJ_SLOT_PROFILE_AUTO=1`, a streamed load whose store has no `slot_profile.json`
  runs `placement.CALIB_TEXT` (our own mixed text: prose, a list, code, arithmetic, dialogue; 677-701 tokens, the
  first 512 used) through the model, records the routing, and writes the profile once; later loads use it with
  `MJ_SLOT_PROFILE=1`. Tested on the tiny Qwen checkpoint. **Not checked on the real model** (no model loads after
  today's swap budget): whether the calibration text's routing gives the same allocation as the WikiText trace,
  and the live A/B, are next.

### 1zzd. gpt-oss-20b: pin only the most-used experts? Same decode, slower prefill, long context fits (2026-11-23 09:58-11:13)

Checklist 0.11b: the 8-bit arm pins all 10 GB of experts (MJ_MLOCK), which swaps out other memory at every load and
leaves 13.3 GB of 16 GB wired, too little for a 16k context (1zx).

**Expert usage** (`tools/mlx_expert_usage.py`, `expert_usage_v1/gptoss.json`): 12 chat-check prompts and 12 GSM8K
questions through our exact arm, 256 greedy tokens each, every routed expert recorded per layer; experts ranked on
every other prompt and scored on the rest (251,808 decode routings scored).

| top k of 32 experts per layer | share of decode routings, all layers | range over layers | if usage were even |
| --- | --- | --- | --- |
| 8 | 46% | 38-62% | 25% |
| 16 | 76% | 66-87% | 50% |
| 24 | 93% | 87-96% | 75% |

**Partial pinning:** `MJ_MLOCK_TOP=k` pins each layer's k most-used experts (ranking in `expert_rank.json` in the
store, from the same run, decode counts over all 24 prompts); the rest stay file cache (`src/mj_mlx/expert_map.py`,
test in `tests/test_expert_map.py`). Suite A/B (`bench_suite/v17g_gptoss_pin_top/`, 3 ABAB rounds, 5 reps, the
8-bit arm (lossy attention) with the experts pinned in the background; the suite's prompts are random tokens, not
the chat text the ranking came from):

| test | all 32 pinned (mj-a8wb) | top 24 (mj-a8wb24) | top 16 (mj-a8wb16) |
| --- | --- | --- | --- |
| pp512 | 413.9 (408.6-419.3), CV 1.0% | 380.5 (327.5-433.5), CV 18% | 354.2 (292.1-416.4), CV 25% |
| tg128@d0 | 39.84 (39.19-40.49) | 40.11 (39.80-40.41) | 40.08 (39.80-40.35) |
| tg128@d2048 | 37.75 (34.54-40.95) | 38.64 (38.51-38.76) | 37.42 (34.93-39.92) |
| peak resident, GB | 10.31-10.34 | 7.77-7.93 | 5.22-5.49 |
| block swap-out, GB, r1 / r2 / r3 | 2.40 / 1.17 / 4.93 | 0.84 / 1.17 / 4.77 | 0.71 / 0.53 / 4.93 |

- **Decode: no difference** (intervals overlap at both depths).
- **Prefill: slower and much noisier** with partial pinning: a 512-token prompt routes to nearly every expert, and
  unpinned ones are read back from the SSD when macOS has dropped them (the 5 flagged prefill reps of top 16, swapping in
  their windows, average 124 tok/s). So partial pinning trades prefill for memory.
- **Swap at load:** lower with partial pinning in rounds 1 and 2, but in round 3 every arm swapped out 4.8-4.9 GB,
  so the machine state (machine-wide counter) dominates; 3 blocks flagged for swap growth (2 of the all-pinned
  arm, 1 of top 24). Not a clean result either way.
- **Long context:** the exact arm with its top 16 pinned (`MJ_MLOCK=1 MJ_MLOCK_TOP=16`) got **12 of 12 at 16k**,
  median 49.4 s per document (`longctx_qa_v1/gptoss_mj_x_pin16_16k.json`); fully pinned it stopped at 8k past 20 GB
  of swap-outs (v17), unpinned 12 of 12 at 51.3 s.
- **Swap written by this run: 30.35 GB in all** (run-wide counter: 21.5 GB in the suite's 9 blocks, most of the rest
  during the 16k QA). The run budget (30 GB) stops new steps; the last step started at 23.2 GB. Every gpt-oss load
  on this machine, in this state (about 4-5 GB of swap in use all morning), pushes out memory: that is 0.11b itself.
- **Not default.** Kept as an option: `MJ_MLOCK_TOP=16` for long contexts and memory headroom, full pinning for
  prefill-heavy use. Next for 0.11b: pin lazily as experts are first used, and measure load swap on a quiet machine
  (after a reboot; the user's call), where the swap per load is not dominated by what other processes hold.

### 1zzc. gpt-oss-20b cold start: the load split into stages, and a prepared-weights cache that brings it level with llama.cpp (2026-11-23 08:53-09:56)

Checklist 0.11. `MJ_LOAD_TIMING=1` (branch `load-timing`, e11ae34) evaluates at each stage of the gpt-oss constructor
and records its seconds; `tools/mlx_load_profile.py --materialize` writes them (`load_profile_v1/gptoss_stages.jsonl`,
v17e `loadstages`, 2 runs per arm). The barriers change when work runs, not what it computes; totals with them are a
little higher than without.

| stage, s | 8-bit arm, pinned before serving | 8-bit arm, pinned in the background | exact arm, unpinned |
| --- | --- | --- | --- |
| attention weights (8-bit quantize / bf16 pack) | 0.62, 0.64 | 0.68, 0.79 | 3.36, 3.41 |
| experts (map, and pin when not in the background) | 6.54, 1.44 | 0.44, 0.68 | 0.31, 0.23 |
| candidate head (3-bit quantize of the 1.16 GB head) | 0.93, 1.04 | 1.00, 1.01 | 0.94, 0.87 |
| everything else | < 0.05 | < 0.05 | < 0.05 |
| **to first token** | 10.78, 6.00 | **3.85, 4.19** | 7.94, 7.43 |

- **The candidate head and the attention weights are derived from the checkpoint the same way on every load**: about
  1.7 s of the background-pinned arm's 3.9-4.2 s, and 4.3 s of the exact arm's 7.4-7.9 s. Kept on disk once, they
  could be read instead (llama.cpp's 2.65 s first token reads its GGUF as stored).
- Pinning before serving costs 1.4-6.5 s (the 6.5 s run was the first after the 16k QA server).
- `MJ_PREP_CACHE=<dir>` (branch `load-timing`, merged ac9ffc6) keeps those arrays in safetensors files keyed by the
  checkpoint's files and the options; written once on the first load (2.0 GB for both arms), read on later loads.

**v17f (09:46-09:56): the cache on the real model.**
- **Same output:** 96 greedy tokens on 3 chat prompts, a model loaded without the cache vs one loaded from it:
  identical on the exact arm and on the 8-bit arm (`load_profile_v1/gptoss_prepcache_tokens_*.json`). The tiny-model
  tests check every array bit for bit (`tests/test_prep_cache.py`).
- **Load profile**, 3 ABAB rounds (`load_profile_v1/gptoss_prepcache.jsonl`), seconds to the first token:

  | arm | no cache | first-ever load (builds the cache) | cache |
  | --- | --- | --- | --- |
  | 8-bit attention (lossy), pinned in the background | 5.06, 3.70, 5.22 | 7.72 | **1.83, 1.84, 1.90** |
  | exact, unpinned | 7.60, 7.70, 7.46 | 9.71 | **2.54, 2.56, 2.56** |

  The first-ever load computes the arrays and writes them (v17f `prep_write`, one load each): it is slower than no
  cache, once per model and option set. The 8-bit arm's cache holds its lossy 8-bit attention arrays: the cache
  changes its load time, not its accuracy label.

  `load_model` drops from 2.1-2.7 s to 0.46-0.50 s (8-bit) and from 4.6-4.8 s to 0.65-0.67 s (exact). MLX's peak
  memory during the load drops from 13.6 to 11.5 GB (no bf16 head copy to quantize).
- **Cold start against llama.cpp** (`tools/mlx_cold_start.py`, 3 starts x 2 rounds, `cold_start_v3/`; a new process
  per start, every engine's weight files read first, the cache files included for ours):

  | engine | ready, median (range) s | first token, median (range) s | round medians, first token |
  | --- | --- | --- | --- |
  | ours, 8-bit attention (lossy), pinned in the background, no cache (mj-a8wb then) | 3.20 (2.58-4.24) | 6.44 (4.41-8.09) | 8.02 / 4.78 |
  | ours, the same with the cache built (mj-a8wbc) | 0.99 (0.41-1.31) | **2.52 (2.34-4.98)** | 3.31 / 2.40 |
  | llama.cpp (llama-server, -ngl 99) | 2.70 (2.07-22.60) | 3.14 (2.52-23.07) | 2.79 / 3.29 |

  - **From 2.4x behind llama.cpp to level:** median first token 2.52 s vs 3.14 s, but llama.cpp is ahead in round 1
    and behind in round 2 and the ranges overlap, so this is level, not a lead. Ours is ready (model loaded) in
    0.99 s against llama.cpp's 2.70 s.
  - Not a quiet machine: about 6 GB of swap was in use throughout (left from earlier runs), and the run swapped out
    8.8 GB in all (v17f's whole run; swap per start was not recorded). llama.cpp's 23 s first start is in the
    ranges above.
- The first-ever cold start (an empty cache: it builds it) was not in this run; the cold-start tool now has it as
  its own engine (`mj-a8wbf`, an empty cache each start), next to `mj-a8wbn` (cache off) and `mj-a8wbc`. The load
  profile's first-build row above (7.7 s) is the closest figure until then.
- **On by default since (PM review, conditions met):** under `~/Library/Caches/mj-mlx/prep`, `MJ_PREP_CACHE=0`
  turns it off, deleting the folder clears it, the first write prints the folder and the size written. The key adds
  MLX's version and a hash of the deriving code's source; every load re-derives layer 0's attention and the head's
  first 4,096 rows and compares bit for bit (a mismatch deletes the folder and the load computes); no write under
  twice 2.5 GB of free disk; a new key deletes older folders of the same checkpoint and options. Tests:
  `tests/test_prep_cache.py` (9). On the real model with these checks (`load_profile_v1/gptoss_prepcache_default.jsonl`):
  greedy tokens identical on both arms again; to first token 1.93 / 2.14 s (8-bit) and 2.90 / 4.31 s (exact) when
  reading, so the spot check costs about 0.1-0.3 s (layer 0's attention re-derived; the head's rows read from the
  file, not the whole head); a code change rebuilt the cache once (6.2 / 9.6 s) and pruned the old folders. Rerun of the cold start on a quiet machine goes
  with the gpt-oss energy rerun (both need low starting swap; the user's call). Until then: "level, machine not
  quiet (~6 GB swap in use)".

### 1zzb. llama-server through our client on gpt-oss; llama-server on DeepSeek-V2-Lite runs out of Metal memory at -ngl 99 (2026-11-23 07:21-07:35)

**Client check, gpt-oss-20b** (v17b `client`, `bench_suite/v17b_gptoss_client/`): the same GGUF through llama-bench
(llama.cpp's own timer) and through llama-server and our HTTP client (`--engine openai`). 2 ABAB rounds, 5 reps.

| test | llama-bench | llama-server, our client | change |
| --- | --- | --- | --- |
| pp512 | 440.7 (432.9-448.4) | 394.0 (387.2-400.8) | -10.6% |
| tg128@d0 | 36.78 (36.33-37.22) | 35.96 (35.39-36.53) | -2.2% |
| tg128@d2048 | 35.75 (35.27-36.24) | 35.98 (35.89-36.07) | none |

- **Decode through the server and our client is within about 2% of llama-bench.** The d0 intervals touch
  (36.33-36.53), and d2048 has no difference. So server decode rates in our tables are not held back by our client.
- **Prefill through the server is 10.6% lower.** That time covers the HTTP request, tokenizing and the server's
  scheduling as well as the prompt pass. So llama.cpp's prefill rows come from llama-bench, as before.
- 3 of 30 llama-server reps swapped out more than 64 MB in their window and are excluded (2 in r1, 1 in r2). No
  block is flagged.

**DeepSeek-V2-Lite on llama-server** (checklist 0.6, v17b `dsv2srv`, `dsv2_llama_server_debug_v1/`): with 84% of
memory free, an 8k context, `-ngl 99` and default batch sizes, the server fails its own warm-up and every request
(a 5-token completion, streaming or not, and chat) with Metal "Insufficient Memory". Its log says why: "failed to fit
params to free device memory: n_gpu_layers already set by user to 99, abort". So with every layer forced onto the
GPU, the Q4_K_M weights, the 8k cache and the 2048-token compute buffers do not fit the GPU's share of 16 GB.
llama-bench runs it at 2k (1zh), and in 1zi llama.cpp did not fit at 8k or 16k at any batch size tried, so a
smaller ubatch may not be enough. Our worker then crashed on the error reply (a KeyError on `n_out`, now fixed); it
records errors without a rate. **Next (v17d), per the rival-capacity rule:** the server at `-ub 256`, then `128`,
same precision, all layers on the GPU; only if both fail, `-ngl` left at llama.cpp's default (it fits layers to free
memory itself, some on the CPU), labelled as a partial offload: a different, slower configuration.

**v17d `dsv2srv2` (08:22-08:38):** at `-ngl 99`, `-ub 256` and `-ub 128` both fail the first short request (Metal
"Insufficient Memory", the same "failed to fit params" warning). With `-ngl` at llama.cpp's default (**partial
offload**, labelled so; the log does not print the layer split) the server answers:
- worker check, 2 reps each, a debug check, not a benchmark: pp512 179 / 152 tok/s (with 0.8-1.0 GB swapped out in
  each window), tg128 43.3 / 48.0 tok/s (no swap in the window).
- long-context QA at 4k: 7 of 7 correct (12-13 s per document), then the swap guard stopped the tool at 20.4 GB of
  swap-outs. 1zr's MMLU then ran to 400 of 1,140 items before the guard stopped it at 20.0 GB too (no results kept).
- **So llama.cpp does not hold DeepSeek-V2-Lite at an 8k context on this Mac without swapping:** all on the GPU it
  does not fit **under macOS's default GPU working-set limit** (raising that limit needs sudo, which we do not use),
  and partially offloaded it swaps. Read it as "does not fit under the default limit", not "cannot fit on 16 GB". This agrees with 1zi. Ours runs it at 8k and 16k (1zx).
- **The run wrote about 40 GB of swap to the SSD**, not 20: the second tool in the step started with its own guard.
  The run scripts now end a step at the first swap stop (`results/v17_scripts/run_v17.py` `sh`). Since then
  (PM review) also: an engine+model config that trips a swap stop goes into `results/swap_tripped.json` and is
  skipped in every later step and run until the user re-enables it (seeded with this one and v17's mlx-lm 8k run);
  and `run_steps` starts no new step once the machine's swap-outs since the run began pass 30 GB. The suite's own
  per-block guard (20 GB) is unchanged.

### 1zza. gpt-oss-20b cold start: where the seconds go, and the wired-limit order fix (2026-11-23 07:20-07:21)

Checklist 0.11. Our gpt-oss cold start was 7.61 s (8-bit arm, experts pinned before serving) and 5.86 s (pinned in
the background) against llama.cpp's 2.65 s (`tools/mlx_cold_start.py`, 1zu).

**Phases** (`tools/mlx_cold_profile.py`, the cold-start tool's own path: a fresh process, the suite's arguments and
environment, prewarmed files; `cold_profile_v1/gptoss.jsonl`). Seconds from the process start, 3 runs per arm:

| arm | import | load | first request | to first token |
| --- | --- | --- | --- | --- |
| a8w (pinned before serving) | 0.14-0.27 | 3.5-9.0 | 0.9-1.6 | 5.0, 5.4, 10.3 |
| a8wb (pinned in the background) | 0.05-0.15 | 2.3-2.8 | 1.3-1.4 | 3.8, 4.1, 4.5 |

- The 10.3 s run was the first after llama-server's chat eval: the files were read back from the SSD (its untimed
  prewarm took 3.3 s, the others 0.7 s). The other runs are 5.0-5.4 s and 3.8-4.5 s.
- **The load** (cProfile, a8w, 3.5 s): `mlock` of the experts 1.19 s, the model's constructor 1.13 s of its own
  time, building each layer 0.81 s, memory-mapping the files 0.29 s. With the background pin the `mlock` leaves the
  load.
- **The first request** (1.4-1.6 s): arming the first step's GPU work 0.71-0.73 s, setting the wired limit 0.19-0.45
  s, the rest under 0.4 s.
- **The wired-limit order fix** (5e5f9e4; `tools/mlx_load_profile.py --materialize`, 2 rounds,
  `load_profile_v1/gptoss_a8w_materialize.jsonl`): building the Engine took 1.25 and 1.26 s with the old order
  (`MJ_WIRED_RAISE=1`) and 0.06-0.07 s with the fix. Time to the first token: 4.27 / 4.59 s old, 3.47 / 3.69 s
  fixed, background pin both times. Evaluating every array right after the load took 0.0 s, so nothing is left lazy.
- **So the remaining gap to llama.cpp** is the model's own constructor and layer building (about 2 s of Python and
  array work), the first step's arming (0.7 s) and the wired-limit call. These are the next targets; none of them
  is reading weights.

### 1zz. gpt-oss-20b chat-format MMLU and GSM8K: ours vs llama-server, a same-weights check, and DeepSeek-V2-Lite's first token (2026-11-23 06:27-08:21)

Checklist 0.2. Same items on both engines: every 4th item of 1zr's subsets (285 MMLU, 63 GSM8K), the chat template,
reasoning effort low, greedy, up to 1024 new tokens, scored on "Answer: X" / "#### n" in the final reply
(`tools/mlx_lm_eval.py --chat`; `lm_eval_chat_v1/`). Ours is the exact arm with the experts not pinned (`MJ_MLOCK=0`;
pinned, v17's run swapped). llama-server runs the ggml-org MXFP4 GGUF (Q8_0 attention), so these are not the same
weights and not a token comparison.

| task | ours, exact | llama-server | both right | only ours | only llama-server |
| --- | --- | --- | --- | --- | --- |
| MMLU (285) | 85.6% +- 2.1 | 85.3% +- 2.1 | 234 | 10 | 9 |
| GSM8K (63) | 85.7% +- 4.4 | 92.1% +- 3.4 | 54 | 0 | 4 |

- **MMLU: no difference** (10 against 9 items one engine gets and the other misses).
- **GSM8K: 4 items llama-server gets and ours misses, none the other way** (exact two-sided sign test p = 0.125, not
  significant at 63 items). So both engines' accuracy reads as "no significant difference at n = 285 / 63".
- 3 of the 4 have the same ending: the reply works the answer out correctly ("250 minutes left") and then ends with
  "#### 1". llama-server's replies end that way on 3 other items (one of them is right, since its answer is 1).
  **Hypothesis, not yet tested:** a greedy near-tie between "1" and the right number, falling one way on 6 of our
  replies and on 3 of llama-server's. The 4th item is a wrong derivation (40 for 9).
- llama-server runs other weights, so it is not the reference for our exact arm. The same-weights check follows.

**Same-weights check (v17d, 08:13-08:21, `lm_eval_chat_v1/gptoss_items_*`).** The 7 items above plus 5 both engines
got right (1, 8, 25, 47, 84), rerun with whole replies kept: ours reproduces all 12 of its v17b replies (greedy is
deterministic run to run), llama-server all 12 of its answers. mlx-lm cannot load the checkpoint whole on 16 GB, so
the 24 replies (prompt + reply, 8,033 tokens) were **teacher-forced** through `tools/mlx_real_eval.py --seqs-json`,
layer by layer: our float32 reference (1zq; checked against transformers on the tiny model), our bf16 paths, and
mlx-lm's own gpt-oss layers in bf16 and float32. This scores the given replies; it is not free generation by mlx-lm.

Every position, against the float32 reference. That reference is our own tool (`tools/mlx_real_eval.py`), not an
outside one; its independent support is mlx-lm's own float32 layers agreeing with it at 99.0% same top token and
mean KL 0.0021 (last row), and its check against transformers on the architecture-exact tiny model:

| path | same top token | mean KL | p99 KL |
| --- | --- | --- | --- |
| ours, exact arm (bf16, packed attention) | 97.6% | 0.0077 | 0.089 |
| ours, bf16 unpacked | 97.4% | 0.0075 | 0.140 |
| mlx-lm's layers, bf16 | 96.8% | 0.0131 | 0.233 |
| mlx-lm's layers, float32 | 99.0% | 0.0021 | 0.059 |

The token after "####" in our replies (log-probabilities of the top two):

| item | gold | our engine wrote | float32 reference | mlx-lm float32 | ours bf16 (exact arm) | mlx-lm bf16 |
| --- | --- | --- | --- | --- | --- | --- |
| 243 | 250 | 1 | 1 (-0.000), 0 (-11.2) | 1 (-0.000) | 1 (-0.000) | 1 (-0.000) |
| 452 | 50 | 1 | 1 (-0.51), 50 (-1.45) | 1 (-0.50), 50 (-1.51) | 1 (-0.54), 50 (-1.41) | 1 (-0.60), 50 (-1.23) |
| 931 | 50 | 1 | 1 (-0.67), 50 (-0.86) | 1 (-0.66), 50 (-0.85) | 1 (-0.65), 50 (-0.90) | 50 (-0.64), 1 (-0.89) |
| 1167 | 120 | 1 | 120 (-0.739), 1 (-0.765) | 120 (-0.64), 1 (-0.91) | tie, 120 = 1 (-0.754) | tie, 120 = 1 (-0.782) |
| 1182 | 1800 | 1 | 1 (-0.29), 180 (-1.88) | 1 (-0.32), 180 (-1.73) | 1 (-0.28), 180 (-1.91) | 1 (-0.28), 180 (-2.03) |
| 337 | 1 | 1 | 1 (-0.000) | | | |
| 694 | 9 | 40 | 40 (-0.000) | | | |

- **6 of 7: the float32 reference picks what our engine wrote**, and on 243 and 694 with certainty (11 nats). On our
  prefixes, "#### 1" is the model's own choice.
- **1167: the reference prefers 120 by 0.026 nat**, and both bf16 implementations (ours and mlx-lm's) sit on an exact
  tie there. That is below bf16 rounding (mean KL of either bf16 path is 0.008-0.013), so it falls either way;
  our engine took "1". Not a fault of our exact arm. On 931 mlx-lm's bf16 falls the other way.
- **On llama-server's replies the reference picks the right number** on 243, 931 and 1167 with certainty (120 at
  -0.000), and "1" on 452 and 1182 as for ours. So the answer token follows the reply text before it. The two
  engines' replies part earlier, on other weights, at near-ties: under teacher forcing the float32 reference itself
  would leave 9 of our 12 replies and all 12 of llama-server's somewhere before the answer.
- **llama-server given our prefixes** (`tools/mlx_answer_margin.py`, the same token ids, its pre-sampling
  log-probabilities): "1" on 243 with certainty (10.9 nats), "40" on 694 likewise, "1" on 452 and 1182; on 931 and
  1167 it picks the right number by 0.13 and 0.24 nat, near-ties as on our side. On its own replies its margins
  match the float32 reference's picks on all 12.
- **Verdict:** our exact arm is faithful on these items. **It sits closer to float32 than mlx-lm's own bf16 does:
  mean KL 0.0077 vs 0.0131, p99 0.089 vs 0.233, teacher-forced on 24 replies / 8,033 positions.** The GSM8K gap to llama-server is the model plus llama-server's other weights, and is not significant at
  63 items. The "exact" label stands.
- The chat check on llama-server (12 prompts, 256 new tokens, `chat_check_v1/gptoss_llama_r2.json`): replies read
  correct, like ours in v17.

**DeepSeek-V2-Lite's first-token difference** (v17's chat check: ours and mlx-lm on the same weights differed at
token 0 of the haiku prompt; `tools/mlx_first_token_margin.py`, `chat_check_v1/dsv2lite_first_token_margins.json`):
- On 11 of 12 prompts the top token is the same, with the same margin to the second token (within 0.06 nat).
- On the haiku prompt mlx-lm's margin is 0.094 nat and ours is 0.000: an exact tie in bf16, so the difference is a
  rounding near-tie, not a fault.
- The largest log-probability difference over mlx-lm's top 20 is 0.03-0.08 nat on 10 prompts, and 0.37 and 0.64 on
  two (prompts 3 and 4, on lower-ranked tokens; the top token and margin agree).

### 1zy. Uneven expert slots per layer: ahead by a paired test, then confirmed by a fresh A/B (2026-11-23 07:35-09:46)

Three more ABAB rounds of 1zv's test (v17b `uneven2`, rounds 4-6, new seeds; `bench_suite/v17b_q35b_uneven_r4/`;
all blocks at `8d5bc07`, clean tree; no rep swapped out in its window). Both arms run the same code in each round.

| round | d0, 96 per layer | d0, uneven | change | d2048, 96 per layer | d2048, uneven | change |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 24.20 | 25.78 | +6.5% | 20.62 | 22.13 | +7.3% |
| 2 | 21.15 | 23.26 | +10.0% | 20.31 | 21.29 | +4.8% |
| 3 | 23.33 | 25.40 | +8.9% | 20.77 | 22.07 | +6.3% |
| 4 | 21.49 | 23.23 | +8.1% | 21.69 | 21.23 | -2.1% |
| 5 | 24.44 | 25.45 | +4.1% | 21.08 | 22.42 | +6.4% |
| 6 | 19.72 | 20.95 | +6.2% | 21.39 | 22.45 | +4.9% |

- **Paired over the 6 rounds** (mean of 5 reps per block, t-interval over the round ratios): **+7.3% at d0
  (95% CI +5.1 to +9.5%, 6 of 6 rounds), +4.6% at d2048 (+1.0 to +8.2%, 5 of 6).** Both intervals exclude zero.
- Pooled by rep, the intervals still overlap (rounds 4-6 alone: 21.89 vs 23.21 at d0, 21.39 vs 22.03 at d2048),
  because decode on the streamed model drifts 20-25 tok/s from round to round. The paired test removes that drift.
  BENCH_PROTOCOL's lead rule (non-overlapping intervals) is for engine-vs-engine comparisons. For two settings of
  our own engine, run in alternating pairs on the same code, a paired interval is the usual test, and the protocol
  now has that rule (paired 95% CI excludes 0, at least 5 of 6 rounds agree). **It was written after rounds 1-6,
  so this result is "ahead by a paired test (rule set after the data)"** until a fresh A/B with the rule fixed
  beforehand confirms it.
- GPU energy per token: unchanged at d0 (204 vs 204 mJ), slightly higher at d2048 in rounds 4-6 (234 vs 226).
- Output is unchanged (only which experts stay in memory differs). **Next:** build the profile from a short
  calibration at load so it can be the default for streamed models (6.2b).
- **Confirmation test, fixed before it runs (2026-11-23 08:20, v17e `uneven3`):** 6 new ABAB rounds (7-12, new
  seeds), same arms, tests and reps. Per test (tg128@d0, tg128@d2048): a lead when the paired 95% CI over the 6
  round ratios excludes 0 and at least 5 of 6 rounds go the same way. Rounds 1-6 are not pooled into it.
- **Result (v17e, 08:53-09:46, `bench_suite/v17e_q35b_uneven_r7/`, rounds 7-12).** Round 12's uneven block is
  flagged (swap grew 806 MB during it, 2.56 GB swapped out); its other blocks are clean, no timed window swapped out.

  | round | d0, 96 per layer | d0, uneven | change | d2048, 96 per layer | d2048, uneven | change |
  | --- | --- | --- | --- | --- | --- | --- |
  | 7 | 20.53 | 22.70 | +10.6% | 21.38 | 22.23 | +3.9% |
  | 8 | 23.46 | 24.77 | +5.6% | 20.04 | 21.67 | +8.1% |
  | 9 | 25.74 | 26.11 | +1.5% | 21.04 | 21.96 | +4.4% |
  | 10 | 21.83 | 24.17 | +10.7% | 21.18 | 23.69 | +11.9% |
  | 11 | 22.14 | 23.18 | +4.7% | 20.41 | 21.92 | +7.4% |
  | 12 (uneven block flagged) | 22.68 | 21.70 | -4.3% | 19.83 | 20.76 | +4.7% |

  - **d2048: confirmed under either reading.** Flagged block excluded (the protocol's standing rule): 5 of 5,
    +7.1% (95% CI +3.1 to +11.1). Kept: 6 of 6, +6.7% (+3.6 to +9.9). Pooled by rep the intervals do not overlap
    either (20.65, 20.13-21.16 vs 22.29, 21.73-22.86), so it also meets the engine-vs-engine rule.
  - **d0: confirmed only with the flagged block excluded:** 5 of 5, +6.6% (+1.7 to +11.6). Kept, round 12 goes the
    other way and the interval includes zero (5 of 6, +4.8%, -1.2 to +10.8). The rule fixed beforehand did not say
    how to treat a flagged block; the protocol excludes them everywhere else, so the d0 lead stands on that, and is
    labelled so.
  - GPU energy per token: unchanged (204 vs 204 mJ at d0, 227 vs 227 at d2048).
  - **The uneven arm does not cause the swap** (excluding r12 would otherwise hide that): over rounds 1-11 the
    peak footprint is the same (96 per layer 11.40-11.63 GB, uneven 11.39-11.60) and block swap-outs are alike
    (mean 0.21 vs 0.30 GB, sd 0.18 vs 0.22, max 0.51 vs 0.53, 11 blocks each); r12's 2.56 GB is a one-off.
  - So uneven slots are faster at the same memory: a confirmed lead at d2048 (+7%), and at d0 (+6.6%) with one
    flagged round excluded. Making it the default still needs the profile built at load (6.2b).

### 1zx. Long-context retrieval through the chat template (needle in a haystack), part 1 (2026-11-23 05:23-05:50)

Checklist 0.15: does each engine find a fact hidden at a known depth of a long document, asked through the chat
template? Raw-text perplexity past the training length (1zq) says little about chat use.

**Method** (`tools/mlx_longctx_qa.py`; the usual form, after Kamradt 2023 and the retrieval tasks of RULER):
- Documents: WikiText-2 test paragraphs, filled to the target length in the model's own tokens.
- Eight invented facts ("The access code for the Velmora archive is 48213.") at evenly spread depths.
- The question asks for one fact, at a different depth per document.
- 12 documents per length, seeded, the same on every engine. Greedy, up to 384 new tokens.
- Scored on the last five-digit number in the reply: correct, distractor (another code from the document), wrong, or
  none.

**Results** (`longctx_qa_v1/`), correct of 12:

| model | engine | 2k | 3.5k | 4k | 8k | 16k |
| --- | --- | --- | --- | --- | --- | --- |
| OLMoE-1B-7B (trained to 4k) | ours | 10 | 7 | | | |
| OLMoE-1B-7B | mlx-lm | 10 | 7 | | | |
| OLMoE-1B-7B | llama-server (Q4_K_M) | 9 | 9 | | | |
| DeepSeek-V2-Lite | ours | | | 10 | 10 | 9 |
| DeepSeek-V2-Lite | mlx-lm | | | 10 | stopped (swap) | |
| gpt-oss-20b | ours, exact, experts pinned (v17) | | | 12 | 3 of 3, then stopped (swap) | |
| gpt-oss-20b | ours, exact, experts not pinned (v17b) | | | **12** | **12** | **12** |
| gpt-oss-20b | llama-server, defaults | | | errors | errors | errors |
| gpt-oss-20b | llama-server, ubatch 256 (v17c) | | | **12** | **12** | 1, then errors |
| gpt-oss-20b | llama-server, ubatch 128 (v17e) | | | | | **11** |

- **Ours and mlx-lm give token-identical replies on all 24 OLMoE documents.** On DeepSeek-V2-Lite at 4k the 12
  results also agree. So at these lengths our long-context path is the reference library's path.
- **llama.cpp** on its own Q4_K_M weights (other weights, so not a token comparison): 9 and 9 on OLMoE. It is behind
  at 2k and ahead at 3.5k, near the model's trained length. Most misses on every engine are distractors (another
  fact from the document); ours has one "wrong" at 3.5k.
- **DeepSeek-V2-Lite at 16k: 9 of 12 on ours, 51 s per document.** mlx-lm passed 21 GB of swap-outs during its 8k
  documents and stopped itself (the swap guard, tools/swap_guard.py). It does not hold an 8k DeepSeek-V2-Lite chat on
  16 GB, which agrees with 1zm. llama.cpp was not run on DeepSeek-V2-Lite (its server's failures there are 0.6).
- **gpt-oss-20b: no result yet.**
  - Our exact arm passed 20.4 GB of swap-outs at 8k and stopped. The suspected cause: this tool builds an Engine per
    document without a wired limit, and on the code of that run each Engine first raised MLX's wired limit to the
    recommended size (fixed in 5e5f9e4 while the run was going; the run had loaded the old code).
  - llama-server then started on a machine still full of swap. It warned that the model did not fit free GPU memory
    and failed every request with Metal "Insufficient Memory".
  - **Rerun (v17b `qa_gptoss`, 06:09-06:27): ours 36 of 36** at 4k, 8k and 16k, with the experts not pinned
    (`MJ_MLOCK=0`) and no swap stop. Pinned, 13.3 GB of the 16 GB is wired (10 GB of experts, MLX's 1000 MB, the
    kernel), too little for a 16k context; unpinned, the experts are file pages the system can drop and re-read.
    That is the setting for long contexts on 16 GB (checklist 0.11b).
  - llama-server failed again with a 17k context, on a machine with 87% of memory free: Metal "Insufficient Memory"
    in its own warm-up. So it is not the machine's state. As in 1zi, llama.cpp fits gpt-oss at 16k on this Mac only
    with a smaller batch; v17c retries at ubatch 256, then 128 (same precision).
  - **v17c (08:02-08:13), llama-server at batch = ubatch 256, 17k context:** 12 of 12 at 4k (12.6-19.0 s per
    document) and 12 of 12 at 8k (24.8-27.5 s). At 16k: a distractor, a correct answer, then Metal "Insufficient
    Memory" on the 3rd document, and every later request failed (the backend stays in an error state; 10 errors).
    The short warm-up request had succeeded, so the script did not go on to 128. v17e runs 16k at ubatch 128.
  - **v17e (08:38-08:53), ubatch 128, 16k only:** 11 of 12 (the miss is a distractor, on doc 0, where it also gave a
    distractor at ubatch 256), no errors, no Metal errors in its log, median 68.8 s per document. Ours at 16k: 12 of
    12, median 51.3 s. The times are per document (prefill of 16k tokens plus the reply), from runs that were not
    interleaved, so they are an observation, not a benchmark. llama.cpp needs ubatch 128 to run gpt-oss at 16k on
    this Mac (as in 1zi); ours runs at its defaults.
  - At 4k and 8k, both engines get 12 of 12 at about the same time per document (median ours 12.3 / 24.9 s, llama-server
    12.7 / 26.2 s), on different weights (ours exact, llama-server the MXFP4 GGUF with Q8_0 attention).

### 1zw. Steady-state whole-machine energy per token: OLMoE and gpt-oss-20b against llama-server (2026-11-23 04:46-05:17)

1zo found the whole-machine readings unusable for short tests: the SMC's power reading lags the load by about 1.75 s,
and tg128 windows last 1-3.5 s. Checklist 0.16 asked for long decode windows with the first seconds dropped.

- **Test:** tg1024@d0, 1,024 tokens of decode (about 8-28 s here).
- **Steady state:** SMC system power from 3 s into the window to its end, minus the idle baseline (10 s, engine
  loaded), divided by the decode rate. Windows under 8 s get no steady value.
- **Run:** suites `bench_suite/v17_energy_olmoe` and `v17_energy_gptoss`. Git `c2a6c7d` / `2dea435` (`src/` and the
  worker identical), clean tree. 3 timed reps x 2 ABAB rounds. llama-server is timed through our client (window =
  the HTTP stream) on the same GGUF files as before.

| model | engine | tok/s | GPU mJ/token | machine mJ/token above idle, whole window | steady state |
| --- | --- | --- | --- | --- | --- |
| OLMoE-1B-7B | ours | 127.68 (127.21-128.14) | 71 | 159 | 171 +- 7 |
| OLMoE-1B-7B | mlx-lm | 127.69 (127.45-127.93) | 71 | 157 | 169 +- 6 |
| OLMoE-1B-7B | llama-server | 96.07 (93.71-98.42) | 109 | 237 | 246 +- 5 |
| gpt-oss-20b | ours, 8-bit attention | 38.94 (37.37-40.51) | 260 | 593 | 601 +- 35 |
| gpt-oss-20b | ours, exact (flagged both rounds) | 36.91 (flagged reps) | 289 | - | 649 +- 7 (flagged reps) |
| gpt-oss-20b | llama-server | 36.01 (35.40-36.63) | 348 | 776 | 783 +- 23 |

- **Whole-machine energy per decoded token above idle, steady state:**
  - OLMoE: **ours 30% below llama.cpp** (171 vs 246 mJ), equal to mlx-lm.
  - gpt-oss-20b: **23% below** with 8-bit attention, the GGUF's own attention precision (601 vs 783 mJ). The exact
    arm's reps read 649 mJ, 17% below, but its blocks were flagged (swap grew 871 / 277 MB), so that number stays out
    of tables.
- **Swap per block** (load included; whether any of it fell inside the timed windows was not measured in v17):

  | model | engine | swap-out GB, r1 / r2 | swap-in GB, r1 / r2 | flagged (swap file grew > 256 MB) |
  | --- | --- | --- | --- | --- |
  | OLMoE | ours / mlx-lm / llama-server | 0.00 / 0.00 for each | 0.00-0.22 | no |
  | gpt-oss-20b | ours, 8-bit attention | 1.61 / 2.16 | 1.28 / 1.43 | no |
  | gpt-oss-20b | ours, exact | 3.24 / 2.38 | 1.70 / 1.51 | yes, both |
  | gpt-oss-20b | llama-server | 0.01 / 0.95 | 0.30 / 0.91 | no |

  **The OLMoE comparison is free of paging.** On gpt-oss, our blocks wrote 1.6-2.2 GB of swap and llama-server's
  0-1 GB. The flag rule is net growth of the swap file (BENCH_PROTOCOL), which a block can avoid while still swapping
  out (pages written and freed again). Some or most of ours may happen at load, when 10 GB of experts are pinned and
  push other pages out; v17 did not count inside the windows, so that is unknown. Since 288745b every timed window
  records its own swap-ins and swap-outs. The gpt-oss energy run was repeated with that (v17b `energy2`, below);
  v17's own gpt-oss figure (601 vs 783 mJ) is superseded by it. A swap-out flag rule with one threshold for every arm
  is to be agreed with the PM. A block threshold of 1 GB would flag 4 of these 6 gpt-oss blocks (all four of ours), so the window count
  is the better basis.
- **Rerun with the window rule (v17b `energy2`, 06:00-06:09; `bench_suite/v17b_energy_gptoss/`).** Commits
  `88ff6ba` / `41b9a96`: `src/` and the tools are identical, only the checklist changed. The 64 MB window rule was
  fixed in BENCH_PROTOCOL before the run.

  | engine | tok/s (95% CI) | steady state mJ/token above idle | swap-out MB in window, mean / max | n clean (flagged) |
  | --- | --- | --- | --- | --- |
  | ours, 8-bit attention | 39.82 (39.47-40.17) | 561 +- 41 | 5 / 18 | 6 (0) |
  | llama-server | 36.58 (35.94-37.23) | 772 +- 1 | 30 / 60 | 2 (4) |

  - **Our swap-outs happen at load, not while decoding:** 2.16 / 2.17 GB per block, but at most 18 MB inside any timed
    window.
  - **llama-server swaps more inside its windows.** Two of its six reps passed 64 MB (155 and 205 MB). Its round-1
    block was also flagged for 270 MB of swap growth.
  - **Result, provisional: ours 27% below llama.cpp in whole-machine energy per token** (561 vs 772 mJ), 8.9%
    faster. Why it is still provisional:
    - Both clean llama-server reps come from one block (r1 flagged whole, r2 lost a rep to the window rule), so
      that figure has no interleaved rounds behind it and its +- 1 from n = 2 means nothing.
    - Every block started with 4.2-4.6 GB of the 5 GB swap file in use, carried over from earlier runs. The rival's
      clean reps still swapped out 30 MB per window against our 5: memory pressure it did not all cause.
  - **The direction is safe:** every excluded llama-server rep read worse (785-816 mJ). To make the size final: at
    least 2 clean llama-server blocks, each starting with under about 1 GB of swap in use. That probably needs a
    reboot, which is the user's call.
  - Our two rounds differ (about 600 vs 524 mJ): the spread is between rounds, not between reps, hence the
    +- 41.
  - **This replaces v17's gpt-oss figure** (601 vs 783 mJ, which did not count in-window paging); both say ours uses
    about a quarter less, and both are provisional for the reasons above.
  - **Separate from steady-state energy:** our 2.2 GB of swap-outs at every gpt-oss load is SSD wear and memory
    pressure for whatever runs next (checklist 0.11b).
- **Steady state and whole window agree within 2-8%** at this length (steady a little higher: the whole window
  includes the ramp). So the lag problem of 1zo is a short-window problem. At tg1024 either number can be quoted.
- **Decode on OLMoE:** ours and mlx-lm tie, both 33% faster than llama-server.
- **Limits:**
  - This is the Mac's own SMC estimate of system power, not a wall meter (no wall meter here).
  - Idle is subtracted, so it is the energy the run adds, not the total.
  - The idle baseline is taken with each engine's model loaded, so a resident engine's background load counts as idle.
- **Ours and mlx-lm are tied on OLMoE** in speed (127.68 vs 127.69 tok/s) and energy (171 vs 169 mJ). The 30% against
  llama.cpp there is MLX against llama.cpp, not a lead of ours over the existing MLX library.
- **The memory table's "peak resident" is not comparable across engines.** llama-server read 0.71 and 11.50 GB in
  its two rounds on the same GGUF. llama.cpp's Metal backend hands the memory-mapped file to the GPU as a no-copy
  buffer, so the process's resident count includes only pages the CPU touched. That depends on how the file was paged
  in, not on what the server holds. Ours reads about 10 GB because mlock touches every expert page. The summaries now
  say so; footprint is the comparable column.

### 1zv. Uneven expert slots per layer, live on Qwen3.6-35B: ahead in every paired round, not yet a clean lead (2026-11-23 04:10-04:46)

1zt's offline replay said that splitting the same slot memory unevenly across layers misses about 7% less than 96
slots in every layer. This is the live test (checklist 6.2b).

- **Setup:** `tools/mlx_slot_profile.py` turned the recorded routing trace into per-layer miss curves and wrote
  `slot_profile.json` into the expert store. Per-layer slots 72-168, total 3,840 = 40 x 96, so the memory is the same.
- **Arms:** the suite's `mj` (96 slots per layer) vs `mj-u96` (`--slot-profile`; the worker refuses to start without
  the profile). Blocks ran at commits `e2ff236` to `c2a6c7d`, all with a clean tree. Those commits change only the harness's swap guard, docs and new tools: `src/` and the worker are identical. 3 ABAB rounds, 5 timed reps, all 6 blocks clean.
- **Summary:** `bench_suite/v17_q35b_uneven/summary.md`.

| test | 96 per layer | uneven, same memory | change |
| --- | --- | --- | --- |
| pp2048 | 276.4 (274.7-278.0) | 275.6 (273.2-278.0) | none |
| tg128@d0 | 22.90 (21.44-24.35) | 24.81 (23.58-26.05) | +8.3% |
| tg128@d2048 | 20.57 (20.01-21.13) | 21.83 (20.90-22.76) | +6.1% |

- **Uneven wins in all 6 paired rounds:** +6.5 / +10.0 / +8.9% at d0 and +7.3 / +4.8 / +6.3% at d2048. That matches
  the size of the offline miss reduction. Prefill does not change: it is not bound by misses.
- **The intervals overlap**, so by our rule this is not yet a lead. Decode at d0 varies 20-28 tok/s rep to rep (SSD
  reads). Three more rounds (v17b) are pooled with these in 1zy: ahead by a paired test (rule set after the data).
- **GPU energy per token is unchanged or slightly lower** with uneven slots: 202 mJ (uneven) vs 203 (96 per layer) at
  d0, 225 vs 228 at d2048.
- **Not default yet.** The profile is model-specific and needs a routing trace. The plan is to make it the default
  for streamed models once a profile can be built from a short calibration run at load.

### 1zu. Suite v16: Qwen3.6-35B against llama-bench and llama-server, gpt-oss-20b with every engine (2026-11-23 00:10-04:05)

Quiet machine (other sessions stopped), git `e5b5d99` (Qwen3.6) and `0b483d4` (gpt-oss), tree clean. Summaries:
`bench_suite/v16_q35b/summary.md`, `bench_suite/v16_gptoss/summary.md`. ABAB, 2 rounds, 5 timed reps per test.

**Qwen3.6-35B-A3B (20.4 GB on a 16 GB Mac).** Ours streams experts from the SSD (96 slots per layer); llama.cpp runs
Q4_K_M with experts on the CPU (`-ncmoe 40`), its best working setting here (1zs).

| test | mj tok/s (95% CI) | llama-bench (flagged reps) | llama-server (flagged reps) |
| --- | --- | --- | --- |
| pp512 | 171.2 (169.0-173.3) | 16.3 | 15.9 |
| pp2048 | 275.6 (274.1-277.2) | 13.2 | 14.5 |
| tg128@d0 | 22.6 (20.6-24.6) | 4.2 | 4.2 |
| tg128@d2048 | 20.6 (19.7-21.5) | 4.0 | 3.1 |

- **Every llama.cpp block is flagged** for swap growth (0.8-11.9 GB per block): a 20 GB model in CPU memory on a
  16 GB Mac swaps whatever the client. The numbers match suite v4 (1zb: 16.5 / 13.0 / 3.8 / 3.2), so the flag is the
  model not fitting, not a noisy run. Our blocks are clean (1.6 / 0.8 GB swapped in, no growth).
- **llama-server and llama-bench agree within the spread** (tg128@d0 4.15 vs 4.16). This does **not** size our client's overhead: every block of both is flagged for swap, the spread is about 1 tok/s, and a per-token HTTP cost would not show at about 4 tok/s. That check needs a fast model that fits (gpt-oss-20b, planned).
- **llama-server's footprint peaked at 24.5 GB and it swapped about 900 GB in and 900 GB out per block** (the macOS
  swap-in and swap-out page counters over the block, 16 KB pages): about 1.9 TB written to the SSD over the two
  blocks, for blocks that are flagged anyway. Since then the suite kills a block once its swap-outs pass 20 GB, and
  measures swap around llama-bench too (not measured in v16), and no engine above RAM that swaps is run without the
  user's OK.
- So, at this size: **prefill 10-21x and decode about 5x llama.cpp**, with machine energy per token above idle of
  0.77-0.94 J for ours. None was measured for llama.cpp (llama-bench's window includes loading; the llama-server
  window is swap-bound).

**gpt-oss-20b.** Exact arm `mj` (MXFP4 experts mapped and mlocked, packed bf16 attention); `mj-a8`/`mj-a8w`: 8-bit
attention weights, the precision llama.cpp's GGUF uses (lossy, 1d.4); `mj-x` the exact arm with mapped experts and
the lean host path. Rivals on the ggml-org MXFP4 GGUF.

| test | llama.cpp | mj (exact) | mj-a8w (8-bit attention) | Ollama | LM Studio (flagged reps) |
| --- | --- | --- | --- | --- | --- |
| pp512 | 426.1 | 396.8 | 403.8 (a8: 407.6) | 404.8 | 407.9 |
| pp2048 | 446.9 | 426.7 | 435.2 (a8: 435.9) | 414.9 | 434.9 |
| tg128@d0 | 37.04 | 37.00 | 40.16 (a8: 39.85) | 34.75 | 35.70 |
| tg128@d2048 | 35.29 | 35.73 | 38.57 (a8: 38.71) | 34.00 | 33.47 |

- **Decode: the exact arm ties llama.cpp** (37.00 vs 37.04 at d0, CIs overlap; 35.73 vs 35.29 at d2048). At
  llama.cpp's own attention precision we are **8.4% faster at d0 and 9.3% at d2048** (CIs apart).
- **Prefill: llama.cpp leads.** Exact arm -6.9% at pp512 and -4.5% at pp2048; 8-bit arm -4.3% and -2.5%. This is the
  remaining gap on gpt-oss (checklist 0.7 items).
- Machine energy per token above idle, tg128@d0: mj-a8w 0.51 J, mj 0.57 J, Ollama 0.75 J (llama.cpp: none, see above).
- Flagged blocks (excluded): one round each of mj, mj-a8w, mj-x, Ollama, LM Studio (both rounds) and llama.cpp
  round 2, for swap growth of 0.3-0.9 GB. Their reps agree with the clean ones (for example mj-a8w tg128@d0 40.57
  flagged vs 40.16 clean).
- The same order as suite v15q (1zp: mj-a8w 40.20, llama-server 35.91 at tg128@d0).

### 1zt. Uneven expert slots per layer, offline: about 7% fewer misses at the same memory (2026-11-23 00:10)

**Question:** the streamer gives every MoE layer the same number of slots (96 on Qwen3.6). Layers differ in how
spread out their routing is. With the same total memory, would moving slots between layers miss less?

**Method** (`slot_allocation_v1/slot_allocation_sim.py`, CPU only):
- Data: the recorded Qwen3.6 routing trace of 1zn (40 layers, 8 documents x 1,024 tokens).
- LRU misses for every slot count at once, from LRU stack distances (Mattson et al., 1970). Demand misses only, no
  prefetch. Each layer's cache runs over all 8 documents, as in the live engine.
- Check against 1zn's event simulation at 96 slots: 11.20% here vs 11.08% there. The small gap is the order of the k
  experts within a token; the streamer protects a token's own experts, which this method does not model.
- Allocation: every layer starts at 16 slots; steps of 4 go to the layer whose misses fall most per slot, until the
  total equals the uniform budget. Every layer's expert has the same size, so equal slots means equal memory.
- Fitted on documents 1-4 and scored on 5-8, and the reverse, so the gain is not read off the data it was fitted on.

**Result** (`slot_allocation_v1/qwen36_uneven_slots.json`), miss rate on the held-out half:

| average slots | uniform (half B / half A) | uneven on half B (fit on A) | uneven on half A (fit on B) | fewer misses (B / A) |
| --- | --- | --- | --- | --- |
| 64 | 20.6% / 16.8% | 20.1% | 16.2% | 2.5% / 4.0% |
| 96 | 12.6% / 9.8% | 11.7% | 9.2% | 6.9% / 6.6% |
| 128 | 7.5% / 6.1% | 6.7% | 5.6% | 11.4% / 8.6% |

- **The pattern is stable.** The allocations fitted on the two halves correlate at 0.97. Layers 0-4 take about
  130-165 slots and most middle layers 72-100. Early layers route the most spread out.
- **Size of the lever:** at 96 slots about 7% fewer misses, similar to the best eviction rule of 1zn (LFU with decay:
  11.1% -> 10.3%). The two act on different things (how memory is split, which expert leaves) and may add up; not
  tested together.
- **Not yet live.** The C++ streamer and the slabs take one slot count today. A per-layer list, plus a profile stored
  with the expert files (from a short routing trace), is needed; then a paired decode A/B at the same memory.

### 1zs. llama.cpp's expert cache cannot run Qwen3.6-35B on a 16 GB Mac (2026-11-23 00:08-00:12)

The 2026-11-22 report named llama.cpp's new expert cache (`--moe-cache-mib`, a GPU cache for MoE experts kept on the
CPU) as a threat to our Qwen3.6 claim, which is measured against `-ncmoe 40` (checklist 6.0). Installed build: ggml
1.0.0, Homebrew (`llama-server` has the flag; `llama-bench` does not).

- **Default loading (memory-mapped):** `llama-server -ncmoe 40 --moe-cache-mib 4096` starts and answers, but logs
  "no layer has all of its experts in host memory, MoE cache is disabled". It also suggests `--load-mode none`. One
  32-token reply ran at 2.6 tok/s. That is a single smoke request, not a benchmark row.
- **Without memory-mapping (`--load-mode none`):** llama.cpp has to read all 18 GB of experts into RAM on a 16 GB
  Mac. Swap reached 12.7 GB 16 s into loading, before the server was up, and the run was stopped.
- **So on this Mac the cache cannot be switched on for a model bigger than RAM:** it needs the experts in host
  memory, and they do not fit. `-ncmoe 40` stays llama.cpp's best working setting for Qwen3.6-35B on 16 GB. On a Mac
  with enough RAM to hold the experts, the cache would work, and should be compared there (checklist 4.1).
- Logs: `moe_cache_v1/smoke_server.log`, `moe_cache_v1/smoke_server_nommap.log`.

### 1zr. MMLU and GSM8K subsets: no quality loss against mlx-lm on the same weights (2026-11-22 05:00-09:49)

`tools/mlx_lm_eval.py`, first run; `lm_eval_v1/`. MMLU: 1,140 questions (20 per subject, 5-shot, the answer letter with
the highest next-token log-probability). GSM8K: 250 problems (5-shot, greedy, up to 256 new tokens, exact match on the
final number). Same prompts on every engine; +- is the binomial standard error.

| model | engine | MMLU | GSM8K |
| --- | --- | --- | --- |
| OLMoE (4-bit) | ours | 53.33 +- 1.48 | 65.20 +- 3.01 |
| | mlx-lm (same weights) | 53.42 +- 1.48 | 65.20 +- 3.01 |
| | llama.cpp (Q4_K_M) | 54.12 +- 1.48 | 63.60 +- 3.04 |
| DeepSeek-V2-Lite (4-bit) | ours | 53.51 +- 1.48 | 48.80 +- 3.16 |
| | mlx-lm (same weights) | 53.77 +- 1.48 | 49.20 +- 3.16 |
| | llama.cpp | failed (its server returned HTTP 500, as in the v15 speed suite) | - |
| gpt-oss-20b | ours (8-bit attention) | 50.44 +- 1.48 | 27.20 +- 2.81 |
| | llama.cpp (MXFP4 GGUF) | 41.05 +- 1.46 | 30.80 +- 2.92 |

- **Ours equals mlx-lm on the same weights** on both models where both run (every difference well inside one standard
  error), and is level with llama.cpp on OLMoE. The engine adds no task-level error.
- **gpt-oss rows are not a quality comparison.** llama.cpp's MMLU here reads the answer letters from its top-20
  next-token list (the server returns no full distribution), so a letter outside the top 20 scores as absent; gpt-oss, a
  reasoning model, puts much of that mass elsewhere. Our scorer reads all four letters exactly. GSM8K for gpt-oss is low on
  both engines because the plain 5-shot format and the 256-token cap cut its reasoning off; the two engines are within
  1.3 standard errors. A chat-format evaluation is needed for gpt-oss.
- Run on a quiet machine apart from macOS maintenance; timing is not used.

### 1zq. gpt-oss-20b at long context: three implementations disagree, ours follows the reference math (2026-11-22 04:41-04:59)

Trigger: the exact arm scored perplexity 5,789 on 16,384-token WikiText chunks (181 on 512-token chunks, 1ze).
`accuracy_long_v2/`; our runs with `tools/mlx_accuracy_ppl.py` (one forward per chunk, or `--prefill-chunk 512` = pieces
over one KV pool, as the Engine prefills); llama.cpp with `llama-perplexity -c <ctx> -b 512 -ub 128` on the MXFP4 GGUF;
the second half of each chunk scored everywhere.

| chunk length (chunks) | ours exact, one forward | ours exact, 512-token pieces | ours 8-bit attention, pieces | llama.cpp |
| --- | --- | --- | --- | --- |
| 512 (80; 1ze) | 181.16 | - | 176.96 (one forward) | 172.83 |
| 2,048 (4) | 1,406.66 | 1,415.44 | - | 1,183.96 |
| 8,192 (2) | 11,069.32 | 10,930.80 | - | 6,437.73 |
| 16,384 (1) | 8,179.23 | 8,439.86 | 7,985.31 | 4,260.62 |

- **The collapse with context is the model's**, on raw text outside its chat format: llama.cpp shows it too.
- **Our paths agree with each other**: pieces vs one forward 0.6% / 1.3% / 3.2% apart at 2k / 8k / 16k; the 16k figure
  reproduces to every digit (8,179.2313 in two runs with different scoring code), so it is not memory corruption.
- **Ours sits above llama.cpp by a margin that grows with context** (1.05x, 1.19x, 1.72x, 1.92x).
- Third opinions at 2,048-token chunks (`gptoss_f32ref_c2048.json`, `tools/mlx_real_eval.py --wikitext-ctx 2048`, 4,092
  positions): our math in float32 **1,402.92 +- 72.49**; our bf16 engine path 1,406.66 (+0.27%, mean KL 0.0181 against the
  float32 run, same top token 92.6%); mlx-lm's own implementation in float32 **1,282.07 +- 65.71** (mean KL 0.0328 against
  our float32, same top token 89.5%). So three implementations of the same weights give 1,403 / 1,282 / 1,184, and the
  spread is between implementations, not between our fast path and full precision.
- Position encoding checked: `mx.fast.rope` in bf16 has the same error at position 32,000 as at 100 (0.010-0.014, CPU);
  our YaRN inverse frequencies and attention factor equal transformers' `_compute_yarn_parameters` on the real config
  (max relative difference 1e-7; the config says `truncate: false`, which transformers and ours honour and mlx-lm does
  not, per its source; llama.cpp's handling not read).
- **Not established:** which implementation matches the model as trained. Raw-text perplexity cannot say (the model is
  far out of distribution here, and the implementations that differ from the reference math score lower). What would:
  a long-context task in the chat format (retrieval or QA at 8k-16k) on each engine, and our tiny-model test against
  transformers at positions past 4,096 (today's test sequences are short).
- DeepSeek-V2-Lite, same check: 16,384-token chunks in 2,048-token pieces 6.604 against 6.609 in one forward (0.07%).
  Its 16k path is verified end to end.

### 1zp. Suite v15, part 2: DeepSeek-V2-Lite on the quiet machine, llama.cpp's knobs, cold start, 32k context, accuracy at 16k (2026-11-22 02:23-04:37)

Other agent sessions were stopped from 02:23 (`v15_scripts/interference_log.md`); macOS's own night maintenance (XProtect,
Spotlight, Siri bookkeeping, ~1.5-2.4 cores at times) ran during the first blocks. Swap already in use: 1.9-2.9 GB.

**1. DeepSeek-V2-Lite** (`bench_suite/v15q_dsv2lite/`, 02:23-03:03, 5 reps x 2 rounds; ours clean in both rounds, the
other engines one clean block each):

| engine | pp512 | pp2048 | tg128@d0 | tg128@d2048 | GPU mJ/token (d0 / d2048) | peak footprint / resident GB |
| --- | --- | --- | --- | --- | --- | --- |
| mj (n 10) | 585.68 [583.71-587.66] | **594.68 [593.64-595.71]** | 62.09 [61.83-62.35] | **52.37 [52.18-52.56]** | 152 / 198 | 11.75 / 0.3-2.4 |
| vllm-mlx (n 5) | 493.20 [484.15-502.25] | 560.25 [555.81-564.68] | **66.88 [66.64-67.13]** | 49.52 [48.90-50.15] | 150 / 190 | 12.6 / 1.2-2.8 |
| llama.cpp (n 5) | 590.20 [589.12-591.28] | 543.32 [522.38-564.25] | 54.84 [54.27-55.42] | 41.77 [41.71-41.83] | - | 1.35 / 11.6 |
| Ollama (n 5; 10.8-14.8 GB of swap-ins per block) | 369.08 | 378.87 | 40.55 | 29.71 | 214 / 262 | 11.2 / 12.5 |

Ahead in decode at 2048 (1.06x vllm-mlx, 1.25x llama.cpp) and in pp2048 (1.06x, 1.09x); behind vllm-mlx at a 1-token
context (1.08x); llama.cpp 0.8% ahead at pp512. The same picture as 1zh and 1zl. One Ollama block with 10.8 GB of swap-ins
passed the net-swap rule: the swap-in rule proposed in 1zk is needed. llama-server on this model stays invalid
(one-token replies, v15 first pass); not rerun here.

**2. llama.cpp threads x ubatch** (`rival_knobs_v1/`, llama-bench `-t 4,6,10 -ub 256,512,1024`, pp512 and tg128, 3
reps; checklist 0.9):

| model | default (-t 4, -ub 512) pp512 / tg128 | best pp512 | best tg128 |
| --- | --- | --- | --- |
| OLMoE | 1269.5 / 116.7 | 1269.5 (the default) | 117.0 (-t 6 or 10, -ub 256) |
| gpt-oss-20b | 406.1 / 36.9 | 418.6 (-t 6, -ub 256) | 37.2 (-t 6, -ub 256) |

Threads do not matter with every layer on the GPU; the default ubatch is best or within 3% (gpt-oss pp512). Our
comparisons against llama.cpp at its defaults stand on these two knobs: tuned, its gpt-oss decode is 37.2 (ours 8-bit
40.2, 1.08x) and its pp512 418.6 (ours 407, llama.cpp 1.03x ahead).

**3. Cold start** (`cold_start_v1/`, `tools/mlx_cold_start.py`, first run of the tool; seconds from a new process, or
an unloaded model, to the first token of a 128-token prompt; weights in the page cache; median (min-max) of 6):

| engine | OLMoE | gpt-oss-20b |
| --- | --- | --- |
| mj | 0.76 (0.75-0.83) | 6.42 (4.64-8.45) (8-bit arm, wired 1000 MB, experts mlocked) |
| llama.cpp (new llama-server) | 0.68 (0.67-1.02) | 2.89 (1.77-21.58) |
| LM Studio (unload, `lms load`, request) | ready 2.33; first token not captured (empty one-token reply) | 2.27 (1.88-2.64) |
| Ollama (daemon up; `ollama stop`, then a request) | 0.68 (0.67-2.70) | 9.35 (8.83-9.62) |
| mlx-lm (new process) | 2.18 (2.12-2.59) | - |
| vllm-mlx (new server) | 2.93 (2.92-4.24) | - |

OLMoE: ours 2.9x faster to the first token than mlx-lm on the same weights and 3.9x faster than vllm-mlx; llama.cpp 0.08 s
ahead. **gpt-oss: a loss.** 6.4 s against 2.3-2.9 s for LM Studio and llama.cpp (they map the file and start; we pin 12 GB
of experts and pack the attention weights first). Swap was 2.7-4.5 GB during the gpt-oss rows, so the sizes need a rerun;
the direction is clear. Ollama's rows are "model unloaded", not "new process".

**4. 16k and 32k context** (`bench_suite/v15_32k_dsv2/`, `v15_32k_gptoss/`, 03:26-04:33, 2 reps; the worker's sequence
limit was raised for 32k prompts in d4b1cc1, so round 1 has 16k only). DeepSeek-V2-Lite, default wired limit (MLX's
recommended working set) against 9000 MB (`mj-w9`):

| | default wired limit | 9000 MB |
| --- | --- | --- |
| pp16384 / tg128@d16384 | 363.64-378.00 / 22.32-22.72 (4 reps) | 369.18-377.57 / 22.20-22.65 (4 reps) |
| pp32768 / tg128@d32768 | 238.53-242.87 / 13.63-13.70 (2 reps) | 237.58-248.34 / 13.17-13.63 (2 reps; block flagged, swap +700 MB) |
| Metal "required > max" lines | about 3,150, all during the 32k tests | 1 |
| kills | none here (three at 16k on 2026-11-21, 1zm / 1zo) | none |

- **DeepSeek-V2-Lite runs a 32k context on 16 GB**: about 240 tok/s prefill (136 s to the first token) and 13.6 tok/s
  decode. One or two blocks per arm, one flagged: a capacity result, not a speed claim.
- **The 9000 MB wired limit removes the over-limit events at no speed cost.** Candidate default for 16 GB machines; it
  needs an ABAB at short context on the headline tests first (not run).
- gpt-oss-20b at 32k: ours (8-bit arm) prefilled, pp32768 267.13 / 276.50 tok/s, then ran out of Metal memory in the
  decode test; llama.cpp ran out of Metal memory at its defaults in the warm-up. Neither completes 32k at defaults; the
  smaller-ubatch retry for llama.cpp and a lower wired limit for ours were not run. 16k stays the longest gpt-oss
  context with results (1zi).

**5. Accuracy at a 16k context** (`accuracy_long_v1/`, DeepSeek-V2-Lite, WikiText-2, two 16,384-token chunks, 16,382
scored tokens, one forward per chunk): perplexity 6.608904787728725 with 64 MB score blocks and the same to every digit
with 16 MB blocks (4k chunks: 8.1157, 1zo). Blocked prefill attention does not change the output at 16k.
- The run of 2026-11-21 that returned 102,400.0 (constant logits) did not reproduce with the same code and settings.
  It sat between two runs that failed with Metal out-of-memory errors; a memory shortfall that returns zeros without an
  error is the likely cause, not established. A guard against constant logits is on the checklist (4.6).
- Not covered yet: the Engine's chunked 16k prefill (the path the speed tests use) against this single-forward number;
  mlx-lm cannot be the 16k reference on 16 GB.

### 1zo. Suite v15, part 1: OLMoE with seven engines, memory and energy columns, the MLA decode kernel, long-context accuracy (2026-11-21 21:18 to 2026-11-22 02:22)

Harness changes first used here (git 1180222 for the suite runs): per block, peak memory of the process holding the
model (footprint: macOS's figure, GPU buffers in, mapped files out; resident: mapped weight files in) and swap-ins /
-outs; whole-machine power from the SMC key PSTR with a 10 s idle baseline per test (`mj_mlx.smc_power`; IOReport's CPU
and DRAM rails read 0 on this Mac, so the earlier "SoC" energy column was the GPU rail); a `llama-server` engine
(llama.cpp's server on the suite's GGUF, our clock, so it gets an energy window); every rival's version in `meta.json`;
git commit, dirty flag and MJ_ / MLX_ environment in every row. Other load on the machine during these runs:
`v15_scripts/interference_log.md`.

**1. OLMoE-1B-7B, seven engines** (`bench_suite/v15_olmoe/`, 22:06-23:10, 5 reps x 2 rounds, every block clean, n 10):

| engine | pp512 | pp2048 | tg128@d0 | tg128@d2048 | GPU mJ/token (d0 / d2048) | peak footprint / resident GB |
| --- | --- | --- | --- | --- | --- | --- |
| mj | 1251.76 [1241.92-1261.59] | 1272.09 [1266.42-1277.76] | 139.09 [138.79-139.39] | **100.76 [98.98-102.55]** | 66 / 87 | 7.16 / 0.2-0.4 |
| mlx-lm | 1290.02 [1274.95-1305.10] | **1358.91 [1357.74-1360.08]** | 137.19 [133.05-141.33] | 94.74 [94.24-95.23] | 66 / 87 | 6.58 / 0.7-0.8 |
| vllm-mlx | 1057.18 [1046.33-1068.03] | 1268.24 [1262.09-1274.39] | **139.96 [139.78-140.14]** | 98.92 [97.87-99.97] | 67 / 86 | 7.34-7.38 / 1.0 |
| LM Studio (MLX engine) | 1114.03 [1098.31-1129.76] | 1285.08 [1280.90-1289.26] | 127.05 [126.84-127.25] (short replies) | 97.86 [97.73-97.99] (short replies) | 65 / 81 | not captured |
| llama.cpp (llama-bench) | 1267.31 [1265.44-1269.19] | 1208.10 [1206.81-1209.38] | 115.65 [114.05-117.25] | 88.94 [88.55-89.33] | - | 0.70 / 4.91 |
| llama.cpp (llama-server) | 1147.44 [1142.30-1152.57] | 1157.12 [1151.86-1162.38] | 100.54 [100.29-100.79] | 88.28 [88.21-88.35] | 101 / 122 | 4.67-4.75 / 8.3-8.6 |
| Ollama | 1151.71 [1140.58-1162.85] | 1159.79 [1154.49-1165.08] | 117.40 [117.23-117.56] | 89.46 [89.09-89.82] | 102 / 122 | not captured |

- Decode at a 2048-token context: ours ahead of mlx-lm (1.06x), llama.cpp (1.13x), Ollama, LM Studio; tied with vllm-mlx
  (intervals overlap), as published. At a 1-token context vllm-mlx is ahead by 0.6% with the intervals just apart; ours
  and mlx-lm tie.
- Losses: pp2048, mlx-lm 1.07x ahead, LM Studio 1.01x ahead; pp512, mlx-lm 1.03x and llama.cpp 1.01x ahead.
- **First llama.cpp GPU energy on our clock:** llama-server 101 / 122 mJ per token against our 66 / 87 (35% / 29% less).
  llama-server decodes 13% slower than llama-bench at a 1-token context (100.54 vs 115.65), so its figure is not
  llama.cpp's best; Ollama on the same GGUF reads the same energy (102 / 122).
- **Memory: ours is not the smallest.** Footprint 7.16 GB against mlx-lm 6.58 and llama-server 4.7 (8.3-8.6 resident with
  its mapped file). Ollama and LM Studio were not captured in this run (process matching fixed afterwards).
- **Whole-machine energy above idle: recorded, not usable yet.** The SMC reading is a ~1 s average that trails the load
  by ~1.75 s; decode windows here are 0.9-1.4 s, and at d2048 the window sits in the tail of the prefill before it
  (ours read 71 mJ per token at d0 and 249 at d2048, where the GPU rail says 66 and 87). The same flaw explains
  llama-server on gpt-oss reading 750 in one pass and 533 in the next. Needed: decode runs of 10 s or more with the
  first seconds dropped (steady-state power over tokens per second). No whole-machine figure is published from v15.

**2. MLA decode kernel against SDPA over gathered rows** (`bench_suite/v15_mla_kernel/`, DeepSeek-V2-Lite, 21:18-21:40,
3 reps x 2 rounds; round 2 of both arms was killed by macOS at 16k, "74-76% of wired memory", so one block each, the
SDPA block flagged for swap +1076 MB):

| tg128 at context | 1 token | 2048 | 8192 | 16384 |
| --- | --- | --- | --- | --- |
| SDPA over gathered rows (default; flagged block) | 61.05-62.01 | 51.88-52.29 | 32.36-33.13 | 22.13-22.44 |
| MLA kernel (`MJ_MLA_DECODE=kernel`; clean block) | 58.76-58.82 | 44.06-44.09 | 30.07-30.56 | 22.37-22.67 |

The kernel is slower at 2k (-15%) and 8k (-7%) and ties at 16k; greedy output was token-identical in 11 of 12 replies
(one diverges at token 71); against a float32 reference its error is 0.006-0.008 where the bf16 SDPA path's is
0.015-0.030 (1zm). So the per-token gather is not what makes MLA decode fall with context; the kernel stays opt-in.

**3. Long-context accuracy, first runs** (`accuracy_long_v1/`, DeepSeek-V2-Lite, WikiText-2):
- 4,096-token chunks (8 chunks, 16,376 scored tokens): blocked prefill attention (64 MB) and unblocked give the same
  perplexity to every digit, 8.115727808916395; mlx-lm on the same weights 8.115787074416035.
- 16,384-token chunks did not produce a valid number: two runs ran out of Metal memory computing logits for 8,191 rows
  at once (tool fixed: NLL in row blocks), one run returned perplexity 102,400.0 = the vocabulary size, i.e. constant
  logits for every position from a single 16k forward (not explained: engine or tool), and gpt-oss exact at 16k was
  killed by macOS for 90% of wired memory with the experts mlocked. Until the 102,400 is explained, no accuracy claim
  at 16k; the Engine's chunked 16k path (the one the speed tests use) has not been checked for accuracy either.

**4. gpt-oss-20b, two passes, neither clean** (`bench_suite/v15_gptoss/`, 00:01-01:10, another agent session active, 5
of 12 blocks flagged; `bench_suite/v15q_gptoss/`, 01:11-02:22, 7 of 12 flagged). Clean blocks of the second pass
(round 1, one block of 5 reps each): mj-a8w tg128@d0 40.20 [39.68-40.71], d2048 38.77 [38.38-39.17]; llama-server 35.91 /
35.98; LM Studio 35.69 / 33.62; Ollama 34.26 / 34.48; mj-x 34.79 / 35.44; llama-bench: both blocks flagged. The 8-bit arm's
lead is the same as in 1zk and 1zl (four passes now), but neither v15 pass meets the protocol: with 2-4 GB already in
swap, most gpt-oss blocks cross the flag threshold. Peak memory (footprint / resident GB): mj-a8w 3.2-3.5 / 10.4, mj-x 3.8 /
10.4-10.6, llama-server 1.3 / 11.9-12.1, llama-bench 0.3 / 11.0-11.7, Ollama 1.6-13.4 / 10.8-11.7. Swap-ins per block ran
from 0.9 GB to 25.9 GB (mj-x round 2, with other jobs on the machine). A pass after a restart is needed.

### 1zn. Forecast paging, offline study part 1: eviction cannot be the lever (2026-11-22 02:10-02:15)

Checklist 6.1, the first experiment of the research report's top direction. Offline, CPU only, on the recorded routing of
Qwen3.6-35B-A3B (8 documents, 8,192 teacher-forced tokens, all 40 layers; the trace of 1v), 96 slots per layer, demand
misses per expert access (`forecast_paging_v1/`; scripts and JSON there). Run at low priority on one core while the
timed gpt-oss block `v15q_gptoss` r2 mj-a8w (02:09-02:15) was running.

**How far ahead must a forecast see?** A horizon-limited oracle: the pager knows the true routing of the next h tokens
in the layer, evicts the least recently used expert not needed in that horizon (h = 0 is LRU; the new simulator
reproduces the old one: 11.08% vs 11.1%, optimum 5.22% vs 5.2%).

| horizon (tokens of true future routing) | 0 (LRU) | 1 | 2 | 4 | 8 | 16 | 64 | 256 | all (optimum) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| miss rate | 11.08% | 10.84% | 10.63% | 10.26% | 9.64% | 8.70% | 6.23% | 5.26% | 5.22% |

With an imperfect forecast: 2 tokens at 90% / 80% recall 10.69% / 10.75%; 4 tokens 10.43% / 10.56%; 8 tokens at 90%
10.06%; 8 tokens at 80% recall with 2 wrongly protected experts per token 10.34%.

**Do the router's scores for experts it did not pick help?** Causal, exact and free (the logits are computed anyway):
decayed scores from the full router output instead of use counts. Use counts with decay 0.98 (the earlier best):
10.33%. Counting near misses (the next 4-8 experts by logit at weight 0.25-0.5): 10.14-10.29%. Softmax probabilities:
worse (13.0-15.8% on three layers).

- **The eviction half of forecast paging is closed.** The report's kill rule was "drop it if a one-token oracle does not
  beat LFU with decay": it does not (10.84% vs 10.33%). Halving the misses needs about 64 tokens of true future
  routing; what a draft or a predictor can give (1-8 tokens, imperfect) is worth 2-9%, no more than use counts already
  give. The gap between LRU and the optimum is long-range (which experts the next ~100 tokens will use), and neither
  recency, counts, nor the router's own scores see it.
- So the report's "+17% if misses halve" does not hold by this route. Reads per token stay near today's; what is left
  of the direction is **hiding the wait**: issuing each read early enough that the GPU does not stall (about 14 ms of a
  ~48 ms step is read wait, and 55% of discarded speculative passes had the right experts but a late read, 1u / 1v).
  That is the predictor half: recall at depth 1-4 with a properly trained predictor (ours: 80-83% at depth 1 on ~6,000
  tokens; published: 93-97% with 65,000+). It needs a larger trace from the streamed model (GPU), so it waits for a free
  machine.
- Other ways to cut reads at fixed memory stay open and are separate items: more slots under one wired budget (128 slots
  was 11-12% faster but did not fit, 1s / 1w), and smaller experts (lossy, opt-in).

### 1zm. DeepSeek-V2-Lite at 16k: the 64 MB score budget fixes the Metal limit; MLA decode paths (2026-11-21 19:30-21:13)

**Memory probe** (`dsv2_16k_mem_probe_v1/`, one fresh worker per setting, pp16384 x 2 reps, login services stopped;
over-limit = kernel "required > max" lines in the worker's window):

| setting | pp16384 tok/s | over-limit lines | MLX peak GB |
| --- | --- | --- | --- |
| chunk 2048, score blocks 256 MB (then default) | 359.5 / 359.0; repeat 366.7 / 365.0 | 274; repeat 283 | 12.51 |
| **chunk 2048, score blocks 64 MB** | **366.1 / 366.9** | **2** | 12.28 |
| chunk 2048, MLX wired limit 10,000 MB | 364.1 / 363.5 | 1 | 12.56 |
| chunk 1024 | 344.9 / 343.0 | 1 | 12.02 |
| chunk 512 | 312.8 / 311.2 | 1 | 11.99 |
| pp8192: chunk 2048 / chunk 1024 | - | 4 / 1 | - |

**ABAB, 64 MB vs 256 MB** (`bench_suite/v14_s64_dsv2/`, `v14_s64_q9b/`, 3 reps x 2 rounds, 20:24-21:13). DeepSeek-V2-Lite:
the 256 MB arm logged 376 and 140 over-limit lines and **macOS killed its round-2 worker** in pp16384 ("77% of wired
memory", 20:48:39); the 64 MB arm logged none, ran both blocks, and is faster or tied everywhere:

| test | DeepSeek-V2-Lite 256 MB (n 3, one block) | 64 MB (n 6) | Qwen3.5-9B 256 MB | 64 MB |
| --- | --- | --- | --- | --- |
| pp512 | 586.91 [580.55-593.27] | 588.46 [586.55-590.36] | 226.53 [226.24-226.82] | 226.43 [226.03-226.82] |
| pp2048 | 576.71 [572.61-580.81] | **594.54 [592.78-596.29]** | **227.52 [227.44-227.60]** | 226.21 [226.06-226.35] |
| tg128@d2048 | 52.62 [52.46-52.77] | 52.71 [52.64-52.79] | 21.49 [21.44-21.55] | 21.10 [20.59-21.62] |
| pp8192 | 460.13 [450.22-470.04] | **481.39 [478.48-484.30]** | 210.84 [209.35-212.32] | 211.25 [209.91-212.58] |
| pp16384 | 363.26 [360.43-366.09] | **368.06 [367.13-368.99]** | - | - |

The default becomes 64 MB (DeepSeek-V2-Lite +3.1% pp2048, +4.6% pp8192; Qwen3.5-9B -0.6% pp2048, the rest tied). Its 16k
prefill now runs at the default chunk; the docs drop the `--chunk 512` advice.

**MLA decode paths** (same probe, chunk 512, ABBA, 2 reps): SDPA over the gathered latent rows (default) 52.5-52.7 /
33.8-34.0 / 20.1-22.6 tok/s at d2048 / d8192 / d16384; the GQA paged kernel 20.6-21.6 / 3.8 / 1.7 (it holds all 16 heads'
576-wide queries and outputs per thread). A dedicated MLA decode kernel (`mla_decode.py`: rows read in place in tiles,
2 heads per simdgroup, split-K) is now in; against a float32 reference its error is 0.006-0.008 where the bf16 SDPA path's
is 0.015-0.030 (`tests/test_mla_decode.py`). Its speed: next.

### 1zl. First full reproduction of every published number (2026-11-21 16:10-19:28)

`tools/reproduce.py`, one claim per invocation into `repro/full_20261121/` (git cd0db49 for the first claim, 54b639c for
the rest: only docs and the checklist changed between them; `src/` and `tools/` clean), servers stopped between claims,
login services stopped. Report: `repro/full_20261121/report.md`. Every claim ran; every sentence the README makes from
them that could be judged still holds.

| claim (source) | our numbers vs published | README sentences |
| --- | --- | --- |
| gpt-oss-20b decode (1zh) | mj-a8: all 4 same (-1.2% to +1.2%) | 6 of 6 hold: decode 1.07x / 1.08x llama.cpp, 1.12x / 1.15x LM Studio, 1.20x / 1.13x Ollama (d0 / d2048) |
| DeepSeek-V2-Lite decode (1zh) | all 4 same (+0.4% to +2.3%) | d2048 faster than vllm-mlx (1.06x) and llama.cpp (1.26x) hold; behind vllm-mlx at d0 (1.08x) holds; vs Ollama not judged (both its blocks flagged for its own swap growth, 264 / 643 MB) |
| OLMoE decode (1zh) | pp512 +3.2% faster, the rest same | ahead of mlx-lm (1.07x) and llama.cpp (1.15x) at d2048 hold; behind mlx-lm in pp2048 (1.07x) and llama.cpp in pp512 (1.02x) hold; **"tied with vllm-mlx" at d2048 does not hold: ours 102.42 [102.30-102.54] vs 99.53 [98.30-100.76], 1.03x with the intervals apart** |
| Qwen3.6-35B-A3B vs llama.cpp (1zb) | pp2048 / tg128@d0 same, pp512 +5.0%, tg128@d2048 -6.2% (20.30 vs 21.65) | all 4 "faster than llama.cpp" hold, at 5.16x (d0), 4.26x (d2048), 10.37x (pp512), 20.00x (pp2048) vs the published 6.4x / 6.8x / 9.8x / 21.4x; llama.cpp's decode came out 17% faster (4.42 vs 3.78, 3 reps per block, experts on the CPU) |
| accuracy, OLMoE / DeepSeek-V2-Lite / Qwen3-30B / gpt-oss (1ze) | perplexity and KL identical to every printed digit, every engine (ours, mlx-lm, llama.cpp) | - |

- Rivals moved more than we did between the source runs and this one: up to +5.5% prefill for Ollama and LM Studio on
  gpt-oss, +7.1% vllm-mlx pp512 on OLMoE. Same versions as in docs/benchmarks.md (nothing was upgraded).
- The README keeps OLMoE "tied with vllm-mlx" (one reproduction is not a basis to upgrade a claim) and now gives the
  Qwen3.6 multiples as the range over the two runs.

### 1zk. Suite v14, gpt-oss-20b: the 8-bit arm at the 1000 MB wired limit, and a paging gap in the swap rule (2026-11-21 15:21-16:09)

`--models gptoss --engines mj-a8,mj-a8w,mj-x,llama.cpp,ollama-gguf,lmstudio-gguf --reps 5 --rounds 2`
(`bench_suite/v14_gptoss/`, git 091917a; login services stopped). New variant `mj-a8w`: `mj-a8` with `--wired-mb 1000`
instead of 3000. Flagged by the net-swap rule: mj-a8w r1, Ollama r2, LM Studio r1, so those rest on one block (n 5).

| engine | pp512 | pp2048 | tg128@d0 | tg128@d2048 | GPU mJ/token (d0 / d2048) |
| --- | --- | --- | --- | --- | --- |
| mj-a8 (8-bit attention, wired 3000 MB) | 408.14 [405.74-410.54] | 436.82 [436.02-437.61] | 39.41 [38.93-39.90] | 38.79 [38.59-38.99] | 266 / 296 |
| **mj-a8w (8-bit attention, wired 1000 MB)** | 406.31 [402.26-410.35] | 437.40 [435.72-439.09] | **39.80 [39.50-40.11]** | **38.60 [38.50-38.71]** | 267 / 296 |
| mj-x (exact, wired 1000 MB) | 352.19 [274.87-429.50] | 400.75 [351.49-450.02] | 34.98 [32.39-37.56] | 35.49 [35.34-35.64] | 299 / 323 |
| llama.cpp | 428.84 [425.87-431.81] | 433.50 [420.93-446.08] | 36.95 [36.86-37.03] | 35.09 [33.66-36.52] | - |
| LM Studio (llama.cpp engine) | 406.59 [379.70-433.48] | 434.42 [427.36-441.47] | 35.76 [35.35-36.17] | 33.48 [33.24-33.71] | 373 / 400 |
| Ollama | 394.72 [387.54-401.91] | 410.30 [401.34-419.26] | 35.50 [35.31-35.69] | 34.24 [33.96-34.51] | 368 / 378 |

- **The 1000 MB wired limit removes the 8-bit arm's Metal over-limit events at no cost.** Of 3,074 "required > max"
  lines, 3,072 fell in the two mj-a8 blocks (15:27-15:29, 16:02-16:04), none in the mj-a8w blocks, and one in each
  mj-x block (15:23, and 16:09 as the round-2 worker exited, as in v13). mj-a8w and mj-a8
  tie on every test (intervals overlap). The engine's default for mapped experts moves from 3000 to 1000 MB.
- 8-bit arm decode vs the rivals, clean blocks: mj-a8w 1.08x llama.cpp at d0 and 1.10x at d2048 (intervals apart), 1.11x
  / 1.15x LM Studio, 1.12x / 1.13x Ollama; 22-28% less GPU energy per token than LM Studio and Ollama. Prefill: llama.cpp
  ahead at pp512 (1.06x); tied at pp2048.
- **mj-x round 2 was paging hard and the swap rule did not flag it.** Swap-ins and swap-outs between block start and end
  (`vmstat_5s.log`, 16 KB pages): 13.0 GB in and 13.8 GB out in that block (it followed a 3000 MB mj-a8 block), against
  0.2-2.0 GB in every other block; its net swap grew less than 256 MB. Its reps fell to 98-210 tok/s in prefill and
  25.65 in decode, so mj-x's v14 means and intervals are not usable. Round 1 (swap-in 2.0 GB) matches v13: pp512
  404.89-412.80, tg128@d0 35.57-37.66, tg128@d2048 35.42-35.72. Proposed rule for the protocol: also flag a block whose
  swap-ins exceed 1 GB during its timed reps (to agree with the PM before it changes published tables).

### 1zj. Prefill attention in query blocks (checklist 0.7b) and one command for every published number (0.8) (2026-11-21 14:00-14:45)

**Cause of the DeepSeek-V2-Lite 16k out-of-memory (1zi).** MLX 0.32.4's fused attention kernel takes head sizes 64, 80,
96 and 128 with equal K and V sizes. For any other size `mx.fast.scaled_dot_product_attention` builds the whole score
matrix. Extra peak memory of one call, 16 heads, 2048 queries over 16,384 keys, bf16 (`attn_blocks_check_v1/`,
scratch script; peak minus memory before the call):

| q / k head size, v head size | extra peak |
| --- | --- |
| 64 / 64, 80 / 80, 96 / 96, 128 / 128 | none above the inputs (-165 to +8 MB) |
| 192 / 128 (DeepSeek-V2 MLA, expanded for prefill) | 1052 MB |
| 192 / 192, 256 / 256 (Qwen3.5 / 3.6 full attention), 128 / 64 | 1052-1056 MB |

That is about one bf16 score matrix (1024 MB). Metal keeps the buffers of the command buffers still in flight, so
several layers' matrices are live at once. A 512-token chunk makes each matrix 256 MB, which is why `--chunk 512` fitted.

**Change.** `mj_mlx.prefill_attention.sdpa_prefill` splits the queries into blocks whose scores fit a budget
(`MJ_ATTN_SCORES_MB`, default 256, the size that fitted), and only for head sizes the fused kernel does not take. The
step chunk, and with it the expert matmuls, stays at the default. Used by the general MoE model (GQA and MLA prefill) and
Qwen3.5 / 3.6 full attention. gpt-oss (head size 64) does not change.
- Same result: unit tests against one SDPA call (causal chunk after a cache, explicit masks, `tests/test_prefill_attention.py`).
  On the real DeepSeek-V2-Lite, WikiText perplexity over 6 chunks with 64-row blocks forced (`MJ_ATTN_SCORES_MB=1`; 216
  block calls counted) equals the default exactly: 9.186658809047955 both.
- Measured peak at 2048 x 16k, 192 / 128: whole 1052 / 1076 MB, blocked no rise above the memory before the call (-14 MB
  both times). **Withdrawn in 1zm:** these single-op readings of MLX's peak counter are not repeatable (the same whole
  call read 44 MB after a warm-up; blocked calls read 767-998 MB at other budgets), so they are not evidence. The
  model-level measures (Metal over-limit lines, kills, the run's MLX peak) are.
- **DeepSeek-V2-Lite 16k at the default 2048-token chunk now runs** (`attn_blocks_check_v1/dsv2_16k_default/`, 1 rep, 1 block,
  login services running): pp16384 352.36 tok/s, tg128@d16384 21.93 tok/s. The block is flagged (swap grew 2990 MB), so
  these are not results; before the change this setting ran out of Metal memory in both rounds (1zi).
- **Timed ABAB, default chunk vs `--chunk 512`** (`bench_suite/v14_attn_blocks/`, 14:37-15:21, pp8192 / tg128@d8192 /
  pp16384 / tg128@d16384, 3 reps, 2 rounds; login services stopped). Both default-chunk blocks are flagged (swap +859 /
  +333 MB), and **macOS killed the round-1 default worker** (signal 9) in tg128@d16384 rep 1, while system wired memory
  read 15.07 GB. Metal "required > max" lines: 1,315 in the two default blocks, none in the two `--chunk 512` blocks.
  MLX peak: 12.45 GB default vs 11.99 GB at 512 (16k; 12.29 vs 11.46 at 8k), against maxWired 12.71 GB.

  | test | default chunk 2048 (flagged reps, n) | `--chunk 512` (clean, n 6) |
  | --- | --- | --- |
  | pp8192 | 469.66 +- 8.90 (6) | 420.36 [415.51-425.21] |
  | tg128@d8192 | 33.41 +- 0.37 (6) | 33.98 [33.86-34.11] |
  | pp16384 | 363.92 +- 2.19 (6) | 311.65 [311.25-312.05] |
  | tg128@d16384 | 22.48 +- 0.07 (4) | 22.61 [22.41-22.80] |

  The 2048 chunk prefills 12-17% faster, but on 16 GB it sits at the Metal limit at 8k and 16k, so it is not safe there.
  Blocked attention removed the out-of-memory at the first prefill step; it did not remove the over-limit pressure. A probe
  of what does (score budget, chunk 1024, wired limit) is queued (`dsv2_16k_mem_probe_v1/`). Until then the docs say
  `--chunk 512` past 8k.

**One command for every published number (0.8).** `tools/reproduce.py` reads `results/published.json` (8 claims: the
README's four speed rows, its "behind" rows and accuracy paragraph; each with the source run's own output file and the
sentences the README makes from it) and re-runs them into a new `results/repro/<date>/`, then writes `report.md`.
- Checked without a benchmark: a "reproduction" that copies the published runs reports every number "same" and every
  sentence "holds" (`tests/test_reproduce.py`), so the manifest's sentences are true of the published data.
- Smoke run, `--quick` (1 rep, 1 round, 3 perplexity chunks; not a result): `repro/quick_smoke_20261121/` (OLMoE, 4 engines:
  every command ran; with 1 rep no sentence is judged) and `repro/quick_smoke_20261121_acc/` (OLMoE accuracy: our engine,
  mlx-lm, llama-perplexity all ran; the first try failed because the mj-mlx venv has no pyarrow, so
  `tools/mlx_accuracy_ppl.py` now reads the committed `accuracy/wiki.test.raw`, checked identical to the parquet text,
  1,285,622 characters).
- The full reproduction has not been run.

### 1zi. Suite v13: the gpt-oss exact arm fixed, a 1-hour soak, and long context to 16k (2026-11-21 01:50-12:51)

First run from the mj-mlx repository (46a9a83 plus the suite changes committed with this section), its own venv
(mlx 0.32.4, mlx-lm 0.32.2, transformers 5.23.0; llama.cpp, Ollama, LM Studio, vllm-mlx as in 1zg). M4 MacBook Pro
16 GB, AC power, caffeinate. The user's four login services (two local dev services, openshell, a dev backend)
were stopped for the runs (`launchctl bootout`) and started again after each script. No other Claude sessions were
running. The session paused 02:01-09:29 (the network was off); the probes before and the runs after are separate
windows. Scripts and their logs: `v13_scripts/`.

**1. Memory probes, gpt-oss exact arm** (`gptoss_mem_fix_v1/`, `gptoss_mem_fix_v2/`; one worker per setting,
pp2048 / tg128@d0 / tg128@d2048, 3 reps; Metal "required > max" lines counted from the kernel log over each worker's
window). The v12 kill (1zh) was preceded by about a thousand such lines; the cause is MLX's wired limit, not mlock or
the prefill chunk:

| setting (mlock, MLX wired limit, chunk) | "required > max" | max descriptorBytes | tg128@d0 | tg128@d2048 |
| --- | --- | --- | --- | --- |
| mlock off, 3000 MB, 512 | 992 | 14.77 GB | 36.4-37.4 | 35.8-36.0 |
| mlock on, 3000 MB, 512 (v12 arm) | 1010 | 14.58 GB | 34.9-36.1 | 35.2-35.7 |
| mlock on, 3000 MB, 256 | 1007 | 14.24 GB | 37.2-37.4 | 35.8-35.9 |
| mlock on, 1500 MB, 512 | 4 | 13.01 GB | 35.1-37.2 | 34.2-35.7 |
| mlock off, 1500 MB, 512 | 4 | 12.99 GB | 36.2-37.1 | 31.3-35.8 |
| **mlock on, 1000 MB, 512 (mj-x)** | **0** | - | **36.9-37.0** | **35.0-35.6** |
| mlock off, 1000 MB, 512 | 0 | - | 31.5-36.7 | 32.9-35.7 |

maxWired on this Mac is 12.71 GB. No probe was killed. The machine was already swapping from the first probes
(swap 0.4-2.0 GB), so the speeds are only a sanity check; the suite below measures them. New suite variant `mj-x`
(`--wired-mb 1000`, otherwise the v12 exact arm); `variant_env` lets a variant override the model's environment.

**2. Suite v13, gpt-oss-20b** (`bench_suite/v13_gptoss/`, 09:39-10:18, `--engines mj-x,mj-a8,llama.cpp,ollama-gguf,
lmstudio-gguf --reps 5 --rounds 2`). Flagged by the net-swap rule: mj-x r2, mj-a8 r2, LM Studio r2, so those three
rest on one block (n 5). No process was killed. Every Metal "required > max" line (3,072) fell in the two mj-a8
blocks (09:45-09:47, 10:11-10:13; mj-a8 still has the 3000 MB wired limit) plus one at 10:18:40 when the mj-x r2
worker exited.

| engine | pp512 | pp2048 | tg128@d0 | tg128@d2048 | GPU mJ/token (d0 / d2048) |
| --- | --- | --- | --- | --- | --- |
| mj-x (exact) | 405.64 [395.85-415.43] | 429.09 [427.38-430.80] | **36.91 [36.28-37.55]** | **35.82 [35.79-35.85]** | 290 / 324 |
| mj-a8 (8-bit attention) | 403.39 [401.31-405.48] | 438.35 [437.01-439.70] | **39.57 [38.68-40.47]** | 36.64 [34.27-39.02] | 262 / 308 |
| llama.cpp | 418.69 [393.23-444.16] | 437.73 [428.68-446.78] | 35.83 [34.97-36.69] | 35.10 [33.80-36.40] | - |
| LM Studio (llama.cpp engine) | 421.96 [418.09-425.82] | 435.10 [431.39-438.81] | 36.00 [35.92-36.08] | 33.21 [32.15-34.26] | 366 / 394 |
| Ollama | 409.51 [397.28-421.73] | 426.70 [418.05-435.36] | 35.20 [35.02-35.38] | 33.37 [32.35-34.38] | 363 / 388 |

- **The exact arm now runs clean**: decode ahead of LM Studio (1.03x at d0, 1.08x at d2048) and Ollama (1.05x, 1.07x);
  tied with llama.cpp (intervals overlap at both depths); 16-21% less GPU energy per token than LM Studio and Ollama.
- mj-a8: decode 1.10x llama.cpp at d0 (v12: 1.08x); at d2048 the intervals overlap with llama.cpp and touch Ollama's and
  LM Studio's (v12: 1.09x llama.cpp), so no d2048 lead is claimed from v13. 21-28% less GPU energy than LM Studio and
  Ollama.
- Losses: pp512, LM Studio ahead of both our arms (1.04x / 1.05x); pp2048, LM Studio ahead of mj-x (1.01x, intervals
  just apart); llama.cpp and mj-a8 tie at pp2048.
- **Paging during timed reps** (`vmstat_5s.log`, swap-outs between each block's first and last timed rep): 0.6-0.7 GB
  for mj-x, 0-2.8 GB mj-a8, 0.7-1.8 GB Ollama, 0.7-1.1 GB LM Studio (llama.cpp: llama-bench's own timing, no rep
  times). The net-swap rule flagged other blocks than this measure would. Every engine pages on gpt-oss-20b on 16 GB, so
  the comparison is like for like, but no gpt-oss number here is paging-free; the timed-rep rule proposed in 1zh would
  flag nearly every gpt-oss block of every engine.

**3. Soak, mj-x** (`gptoss_soak_v1/`, 10:20-11:25, one worker, pp2048 / tg128@d0 / tg128@d2048 x 150 reps, 3 s gaps):
no kill, three "required > max" lines in 65 minutes, peak RSS 10.6 GB.

| test | n | mean | first third -> last third | min - max |
| --- | --- | --- | --- | --- |
| pp2048 | 150 | 422.48 | 415.05 -> 429.82 | 130.08 - 434.61 |
| tg128@d0 | 150 | 36.63 | 37.18 -> 36.45 | 30.77 - 37.52 |
| tg128@d2048 | 150 | 35.66 | 35.61 -> 35.72 | 33.23 - 36.19 |

Checklist 4.2 (the silent exit on the exact gated path, 1zc): the 3000 MB wired limit is the likely cause, and at
1000 MB the arm runs an hour without it; not reproduced on the old code to prove the link.

**4. Long context.** Tests pp16384 / tg128@d16384 (and pp8192 / tg128@d8192 for DeepSeek-V2-Lite), 3 reps, 2 rounds,
ABAB. Ollama and LM Studio were left out (the harness fixes their context at 8k). mlx-lm was left out on DeepSeek
(expanded K/V at 16k is about 4.5 GB more). Rival check, in line with industry practice for capacity claims: when
llama.cpp ran out of Metal memory at its defaults (`-fa 1 -ngl 99`, batch 2048 / ubatch 512, f16 KV), it was re-run
with batch = ubatch 256 then 128 (speed and memory only; same f16 KV, same weights), stopping at the first fit
(`llama_16k_check_v1/`, `llama_8k_check_v1/`). Quality-changing options (q8_0 KV) were not tried.

| model, context | ours | llama.cpp |
| --- | --- | --- |
| gpt-oss-20b, 16k (`v13_long_gptoss/`) | mj-a8: pp16384 **355.65 [352.02-359.29]**, tg128@d16384 **31.37 [30.86-31.87]** (n 6); mj-x: 323.03 [234.97-411.09] / 25.57 [16.82-34.33] (one clean block, noisy; its flagged block 341.62 / 29.51) | defaults and ubatch 256: out of Metal memory; ubatch 128: pp16384 275.83 +- 1.27, tg128@d16384 27.89 +- 1.10 (n 3) |
| DeepSeek-V2-Lite, 16k (`v13_long_dsv2/`, `v13_long_dsv2_c512/`) | default chunk 2048: out of Metal memory (both rounds); `--chunk 512`: pp16384 **309.48 [308.25-310.71]**, tg128@d16384 **22.17 [21.75-22.60]** (n 3, clean) | out of Metal memory at ubatch 512, 256 and 128 |
| DeepSeek-V2-Lite, 8k (`v13_8k_dsv2/`) | default: pp8192 **457.17 [444.42-469.91]**, tg128@d8192 **33.82 [33.45-34.19]** (n 3); `--chunk 512`: 417.16 [410.42-423.89] / 33.81 [33.55-34.07] (n 6) | out of Metal memory at ubatch 512 (both rounds), 256 and 128 |

- gpt-oss at 16k: our 8-bit arm 1.29x llama.cpp prefill and 1.12x decode against llama.cpp's best fitting
  quality-preserving setting (llama.cpp needs ubatch 128 to fit; our arm runs at its defaults). llama.cpp's runs were
  after ours, not interleaved.
- DeepSeek-V2-Lite: ours runs at 8k (defaults) and 16k (512-token chunks); llama.cpp did not fit at 8k or 16k at any
  batch size tried. At 2k it ran (1zh: 41.02 decode). Our 16k needs a non-default chunk until the engine picks the chunk
  from the context length; 512-token chunks cost 9% of 8k prefill and nothing in decode.
- vllm-mlx at 8k and 16k is not valid: on every prefill rep its server streamed no tokens and reported no usage
  (`server.chunks` 0), so the "rate" (8k: 384-646, 16k: 1,127-1,257 tok/s, above its own 2k rate of 557) is the time to
  the end of a reply with no tokens; decode replies were one token. Cause not established (prompt rejected or cut);
  not compared.
- Decode falls with context on DeepSeek-V2-Lite (52.24 at 2k, 1zh; 33.82 at 8k; 22.17 at 16k): a speed gap to work on.

### 1zh. Suite v12: quiet re-run after a restart (2026-11-20 14:22-16:36)

Same command as 1zg (`tools/mlx_bench_suite.py --models olmoe,dsv2lite,gptoss --tests pp512,pp2048,tg128@d0,tg128@d2048
--reps 5 --rounds 2`), out `bench_suite/v12_quiet/` (git cd395b6), then a round 3 of six flagged blocks (`--round0 3
--rounds 1 --engines ...`, `v12_quiet_rerun.log`). Versions as in 1zg. Machine: restarted at 14:15, AC power, swap 0 MB
at the start, no other Claude sessions running; post-restart Spotlight / Siri work was still running at 14:22 (load
3.7). Four of the user's login services (two local dev services, two openshell VMs, a dev backend; about 3.5 GB
resident) ran until 14:45, then were stopped with the user's OK (`launchctl bootout`; started again after the run).
`machine_top.log` has the top processes each minute; `vmstat_5s.log` has swap-ins / swap-outs every 5 s from 14:58.

**Flagged blocks and where the swap came from.** Rounds 1-2: 11 of 34 blocks flagged (1zg: 13 of 34). After round 3
every engine has at least two clean blocks except mlx-lm and LM Studio on DeepSeek-V2-Lite (none), ours exact on
gpt-oss (none) and llama.cpp gpt-oss tg128@d2048 (one, n 5).
- Background: OLMoE ours r1, Ollama r1 and LM Studio r2 ran while the login services held about 3.5 GB. After 14:45
  every OLMoE block was clean, including the three re-runs.
- The engine's own memory (swap-outs during its timed reps, `vmstat_5s.log`): mlx-lm on DeepSeek-V2-Lite (during its
  pp2048 reps; 2 of 5 reps at 528 / 509 vs about 575 tok/s, the same as in 1zg; it caches expanded K/V, ours the MLA
  latents); LM Studio on DeepSeek-V2-Lite (14.7-16.3 GB swapped out during its reps; a 10.5 s first token; one-token
  replies on 3 of 10 decode reps in both rounds, at the same rep positions with different prompts (seeds 100-104 and
  200-204), cause of the one-token replies not established; the 1zg failure was the same).
- Ours, gpt-oss exact: 1.4-1.7 GB swapped out at load (before the first timed rep) in all three rounds; during the reps
  0 MB (r1), 1.5 GB (r2), 77 MB (r3).

**macOS killed our gpt-oss exact worker (round 2).** Kernel log, 16:16:03: `memorystatus: python3.11 [16381] has 75% of
wired memory (11409 MiB of 15208 MiB total wired), killing`. From 16:15:40 IOGPUFamily logged `required > max` for its
command buffers (descriptorBytes about 13.45 GB, maxWired 12.71 GB). tg128@d0 reps 2-4 fell to 29.5-32.3 tok/s and the
worker exited -9 before tg128@d2048. This is a candidate cause of the unexplained silent exit on the exact gpt-oss path
(1zc, checklist 4.2), not yet checked. The same `required > max` (12.73-13.10 GB needed, 12.71 GB max) was logged when
mlx-lm 0.32.2 failed to load openai/gpt-oss-20b for a step trace (Metal out of memory in `mx.eval(model.parameters())`).
llama.cpp's round-3 tg128@d2048 failed with `kIOGPUCommandBufferCallbackErrorOutOfMemory` and no such kernel line; its
rounds 1 and 2 ran that test.

Decode tok/s, mean [95% CI], unflagged blocks only (n 10 unless marked; all rows, prefill and energy in `summary.md`):

| model | test | ours | vllm-mlx | mlx-lm | LM Studio | llama.cpp | Ollama |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OLMoE | tg128@d0 | 137.45 [136.53-138.38] | **139.29 [139.06-139.52]** | 135.55 [134.02-137.08] | 115.98 [111.52-120.44] | 115.55 [115.46-115.64] | 112.59 [109.13-116.06] |
| OLMoE | tg128@d2048 | **99.79 [97.38-102.20]** | **98.91 [97.68-100.15]** | 92.53 [90.49-94.57] | 92.08 [89.49-94.68] | 88.15 [87.45-88.86] | 87.64 [85.63-89.65] |
| DeepSeek-V2-Lite | tg128@d0 | 61.00 [60.04-61.95] | **66.02 [64.96-67.08]** | flagged (67.41) | flagged | 53.38 [51.98-54.79] | 45.46 [44.17-46.74] |
| DeepSeek-V2-Lite | tg128@d2048 | **52.24 [51.64-52.84]** | 49.64 [49.20-50.08] | flagged (49.09) | flagged | 41.02 [40.14-41.90] | 35.01 [33.81-36.22] |
| gpt-oss-20b (ours: 8-bit attention arm) | tg128@d0 | **39.67 [39.30-40.04]** | - | - | 35.67 [35.31-36.03] | 36.82 [36.34-37.29] | 35.07 [34.80-35.33] |
| gpt-oss-20b (ours: 8-bit attention arm) | tg128@d2048 | **38.33 [37.97-38.68]** | - | - | 33.33 [32.93-33.73] | 35.06 [32.49-37.64] (n 5) | 34.28 [33.88-34.68] |

Findings (claims held to the 95% CIs):
- **OLMoE.** Short-context decode: vllm-mlx 1.01x ahead of ours (CIs apart); ours ties mlx-lm (CIs overlap); v11's
  0.96x behind mlx-lm (1zg) does not reproduce on the quiet machine (ours 137.45 vs 133.77). Long context: ours ties
  vllm-mlx and is ahead of mlx-lm (1.08x), LM Studio (1.08x), llama.cpp (1.13x) and Ollama (1.14x). Prefill: llama.cpp
  leads pp512 (ours 0.95x), mlx-lm leads pp2048 (ours 0.95x). GPU energy at d0: ours, mlx-lm and vllm-mlx all 67 mJ/token.
- **DeepSeek-V2-Lite.** Short-context decode: vllm-mlx 1.08x ahead of ours (CIs apart), so 1zg's "ties vllm-mlx at d0"
  is not confirmed; mlx-lm has no clean block (flagged reps 67.41, 1.11x ours, the gap in checklist 3.1). Long
  context: ours 1.05x vllm-mlx, 1.27x llama.cpp, 1.49x Ollama. Prefill: ours leads pp2048 (1.03x vllm-mlx, 1.07x
  llama.cpp); llama.cpp leads pp512 (ours 0.97x).
- **gpt-oss-20b, 8-bit attention arm.** Decode 1.08x llama.cpp at d0 and 1.09x at d2048 (llama.cpp n 5 there; CIs
  apart in both), 1.11-1.15x LM Studio and Ollama; GPU energy 265 / 285 mJ per token vs LM Studio 360 / 374 and Ollama
  364 / 377 (24-27% less). Prefill: llama.cpp 1.04x ahead at pp512, a tie at pp2048 (CIs overlap).
- **gpt-oss-20b, exact arm:** no clean block (above); not a measurement we stand on. For reference only, its two blocks
  with swap only at load: decode 36.43 [36.28-36.58] (d0) and 35.03 [34.81-35.24] (d2048), n 10. Its memory on 16 GB
  (experts mlocked plus packed bf16 attention, MLX peak 13.9 GB) is the issue to fix, not the rule.

Proposal for the PM (not applied to any number here): flag a block when swap-outs during its timed reps exceed 256 MB,
next to today's whole-block growth rule; swap at load is then recorded but not flagged. Under it ours exact r1 / r3 and
mlx-lm DeepSeek-V2-Lite would be judged by what happened while timing.

### 1zg. Suite v11: every engine, prompts distinct per test (2026-11-19 16:29-18:35)

`tools/mlx_bench_suite.py --models olmoe,dsv2lite,gptoss --tests pp512,pp2048,tg128@d0,tg128@d2048 --reps 5
--rounds 2`, out `bench_suite/v11_servers/` (git 70903ec). Engines and versions: ours; mlx-lm 0.32.2 (mlx 0.32.4);
llama.cpp build 12359 b1b770d90 (llama-bench); Ollama 0.40.16 (its GGUF runner, `num_ctx` 8192); LM Studio 0.4.25
(its MLX engine mlx-llm 1.14.0 for OLMoE / DeepSeek-V2-Lite, its llama.cpp engine 2.73.0 for gpt-oss; loaded with
`-c 8192 --parallel 1`); vllm-mlx 0.6.0 (mlx-lm 0.31.3; `--max-num-seqs 1 --disable-prefix-cache`). Servers are
driven over their streaming APIs and timed by our clock; prefill tok/s uses the server's own prompt-token count. Every
test now has its own prompt (1zf correction), so no server can reuse a cached prefix. Greedy; AC power.

**The machine was not quiet.** Swap was already 4676 MB when the run started (the swap file is 5120 MB), and 13 of
34 blocks were flagged for swap growth (309-8923 MB). OLMoE ran clean (no flagged block). On DeepSeek-V2-Lite and
gpt-oss many rows rest on one round (n 5) or have no unflagged block at all. Ollama's DeepSeek-V2-Lite blocks grew
swap the most (4852 and 8923 MB; cause not established), and they ran just before LM Studio's and vllm-mlx's blocks
in round 1. Tables use unflagged blocks only; flagged means are in `summary.md` and are not a measurement we stand on.
mJ/token is the GPU rail (estimated); none for llama-bench.

| model | test | ours | mlx-lm | vllm-mlx | LM Studio | llama.cpp | Ollama | ours vs best rival |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLMoE-1B-7B | pp512 | 1200.64 +- 21.71 | 1228.24 +- 24.23 | 992.56 +- 53.54 | 1060.23 +- 14.41 | **1267.18 +- 2.40** | 1107.13 +- 41.54 | 0.95x |
| | pp2048 | 1260.10 +- 7.35 | **1343.35 +- 9.43** | 1252.84 +- 22.12 | 1273.73 +- 11.35 | 1207.86 +- 1.65 | 1135.33 +- 22.64 | 0.94x |
| | tg128@d0 | 133.77 +- 3.15 | 137.63 +- 0.70 | **138.87 +- 0.59** | 126.25 +- 0.48 | 116.11 +- 0.32 | 116.59 +- 0.37 | **0.96x** |
| | tg128@d2048 | 98.91 +- 3.33 | 95.15 +- 1.28 | 98.03 +- 3.14 | 96.82 +- 0.11 | 88.51 +- 1.71 | 89.63 +- 0.15 | 1.01x (tie with vllm-mlx and LM Studio) |
| DeepSeek-V2-Lite | pp512 | 574.67 +- 1.51 (n 5) | flagged | 412.89 +- 117.46 (CV 28%) | flagged | **590.97 +- 0.88** (n 5) | flagged | 0.97x |
| | pp2048 | **574.19 +- 1.48** (n 5) | flagged | 545.39 +- 4.80 | flagged | 555.18 +- 9.23 (n 5) | flagged | 1.03x |
| | tg128@d0 | **62.16 +- 0.05** (n 5) | flagged (65.75) | 61.93 +- 1.17 | failed | 54.99 +- 0.03 (n 5) | flagged | 1.00x (tie with vllm-mlx); 0.95x vs flagged mlx-lm |
| | tg128@d2048 | 47.25 +- 3.69 (n 5) | flagged (49.20) | 44.78 +- 1.87 | failed | 41.18 +- 0.29 (n 5) | flagged | 1.06x (tie with vllm-mlx: CIs overlap) |
| gpt-oss-20b (ours: 8-bit attention arm) | pp512 | 399.80 +- 1.49 (n 5) | - | - | flagged | **426.49 +- 23.96** | 379.26 +- 19.30 | 0.94x |
| | pp2048 | **435.98 +- 0.63** (n 5) | - | - | flagged | 433.03 +- 20.41 | 404.56 +- 4.93 | 1.01x (tie) |
| | tg128@d0 | 38.08 +- 1.48 (n 5) | - | - | flagged | 36.45 +- 0.48 | 34.59 +- 1.05 | 1.04x (within llama.cpp's CI) |
| | tg128@d2048 | **37.82 +- 0.24** (n 5) | - | - | flagged | 34.90 +- 0.40 | 34.25 +- 0.24 | 1.08x |

n is 10 (two rounds) unless marked. mlx-lm and vllm-mlx do not run gpt-oss's MXFP4 checkpoint in this suite.
"failed": LM Studio's DeepSeek-V2-Lite decode test crashed our harness in both rounds (its reply to the warm-up was at
most one token, and the worker divided by zero; checklist 0.13). Ours with exact bf16 attention on gpt-oss: every
block flagged (flagged means pp512 385.18, pp2048 426.23, tg128@d0 35.82, d2048 34.24).

Replies under the requested length: LM Studio returns 128 of the 129 requested tokens on every decode request (finish reason "length");
its tok/s is over the tokens it produced. 2 of Ollama's 10 gpt-oss d2048 replies stopped early (EOS). The
DeepSeek-V2-Lite crash did not reproduce at 19:00, after the run: with the same model, settings and prompts, LM
Studio's streamed decode returned 128 tokens at both depths (2 reps each; not a benchmark, swap still 5.4 GB). Cause
not established; the worker now records such a reply with no rate (tok_s null) instead of crashing (checklist 0.13).

Findings:
- **Where we lose (OLMoE, clean run).** Short-context decode trails vllm-mlx by 4% and mlx-lm by 3%. Our round 1
  was 135.32 and round 2 132.23, while mlx-lm (137.52 / 137.74) and vllm-mlx (138.53 / 139.20) held steady; in v10
  ours was 138.39, ahead of mlx-lm. So the gap is within our own run-to-run spread, and the spread is itself a
  finding: a quiet re-run decides it (checklist 0.5, 3.5). Prefill is 94-95% of the best rival (llama.cpp at 512,
  mlx-lm at 2048), as in v10.
- **Where we lead or tie** (by the 95% CIs). OLMoE d2048 decode ties vllm-mlx and LM Studio, and is ahead of mlx-lm
  (94.24-96.06, no overlap), llama.cpp and Ollama. DeepSeek-V2-Lite decode ties vllm-mlx at d0 and at d2048 (our
  d2048 block: 42.65-51.84, CV 7.8%, vs 43.44-46.12; the flagged round 1 gave 52.08), and is ahead of llama.cpp at
  both (d2048 40.83-41.54). gpt-oss decode: d0 is within llama.cpp's CI (1.04x; ours 36.24-39.92, n 5, vs
  36.11-36.79); d2048 1.08x is clear (37.52-38.11 vs 34.61-35.19); 1.10x over Ollama at both depths.
- **Ollama, cold prompts.** With distinct prompts its prefill is below ours on OLMoE and gpt-oss (its DeepSeek-V2-Lite blocks were all flagged; gpt-oss pp2048 404.56, in the
  403-431 range of the 1zf fresh re-measure; OLMoE pp2048 1135.33, v10's cache-biased row was 1263.99). Decode: ours
  is 1.15x / 1.10x on OLMoE and 1.10x / 1.10x on gpt-oss.
- **LM Studio** (MLX engine on OLMoE): 6% behind us at d0, 2% at d2048; its pp2048 (1273.73) is above ours. It uses
  the least GPU energy per decode token on OLMoE (61 mJ vs our 67; mlx-lm 65, vllm-mlx 67, Ollama 101).
- **Energy, gpt-oss decode:** ours 259 / 283 mJ per token vs Ollama 358 / 383.
- Greedy ids, ours vs mlx-lm: OLMoE 16/20 identical, DeepSeek-V2-Lite 13/20 (first divergence at token 1-71
  otherwise, as in earlier runs: float16 scales, different reduction order).

### 1zf. Suite v10: Ollama in the head-to-head (2026-11-19 14:55-16:14)

`tools/mlx_bench_suite.py --models olmoe,dsv2lite,gptoss --reps 5 --rounds 2`, out `bench_suite/v10_ollama/`
(git a24d034; Ollama 0.40.16, its GGUF runner on the same GGUF as llama.cpp, `num_ctx` 8192, greedy, streamed and
timed by our clock; llama.cpp build 12359 b1b770d90; mlx-lm 0.32.2). The machine was busy: 9 of 24 blocks flagged
for swap growth (271-2289 MB), so tables below use unflagged blocks only and some rows rest on one round (n 5).
mJ/token is the GPU rail (estimated); none for llama-bench.

Ollama's decode prompts carry a short narrative opener (14 tokens in gpt-oss's tokenizer; it cannot ignore EOS); 2 of 20 gpt-oss d2048 replies
still stopped early (51 and 62 tokens; tok/s over the tokens produced). Its server-side prompt rates agree with our
clock (gpt-oss pp2048: 506-557 server vs 477-546 client). In tg128@d2048 Ollama reused the cached KV of the pp2048
prompt with the same seed (server prompt rate ~9,700 tok/s), which changes TTFT only, not decode tok/s.

| model | test | ours | mlx-lm | llama.cpp | Ollama | ours vs best rival (Ollama prefill excluded, see correction) |
| --- | --- | --- | --- | --- | --- | --- |
| OLMoE-1B-7B | pp512 | 1202.86 +- 22.12 (n 5) | 1227.15 +- 18.71 | **1267.20 +- 2.84** | 1083.72 +- 50.02 | 0.95x |
| | pp2048 | 1266.80 +- 6.13 (n 5) | **1325.51 +- 23.22** | 1206.51 +- 2.83 | 1263.99 +- 25.08 | 0.96x |
| | tg128@d0 | **138.39 +- 0.60** (n 5) | 137.09 +- 1.27 | 109.40 +- 6.51 | 116.71 +- 0.46 | 1.01x |
| | tg128@d2048 | **101.54 +- 0.19** (n 5) | 94.63 +- 2.04 | 84.87 +- 2.33 | 88.82 +- 0.24 | 1.07x |
| DeepSeek-V2-Lite | pp512 | 576.92 +- 5.02 (n 5) | flagged (558.58) | **591.12 +- 0.98** | 309.17 +- 64.91 (n 5) | 0.98x |
| | pp2048 | **574.88 +- 0.74** (n 5) | flagged (469.34) | 553.05 +- 11.29 | 295.19 +- 22.63 (n 5) | 1.04x |
| | tg128@d0 | 60.87 +- 0.13 (n 5) | flagged (**65.68**) | 51.67 +- 1.70 | 37.05 +- 6.34 (n 5) | 0.93x vs flagged mlx-lm; 1.18x vs llama.cpp |
| | tg128@d2048 | **51.62 +- 0.15** (n 5) | flagged (48.40) | 41.20 +- 0.42 | 23.43 +- 1.46 (n 5) | 1.25x |
| gpt-oss-20b (ours: 8-bit attention arm) | pp512 | 403.94 +- 1.64 (n 5) | - | **431.57 +- 5.36** | 396.31 +- 30.00 (n 5) | 0.94x |
| | pp2048 | 433.34 +- 2.23 (n 5) | - | 437.12 +- 13.20 | **515.80 +- 32.28** (n 5) | **0.84x** |
| | tg128@d0 | **38.07 +- 0.86** (n 5) | - | 35.63 +- 0.49 | 34.72 +- 0.41 (n 5) | 1.07x |
| | tg128@d2048 | **35.63 +- 1.03** (n 5) | - | 34.25 +- 1.64 | 34.12 +- 0.32 (n 5) | 1.04x |

Flagged values (in brackets) are means of flagged blocks, shown because every mlx-lm block on DeepSeek-V2-Lite was
flagged; they are not a measurement we stand on. Ours on gpt-oss with exact bf16 attention: every block flagged
(flagged means pp512 402.48, pp2048 428.14, tg128@d0 35.46, d2048 34.11).

**Correction (16:55): Ollama's prefill figures in this table are invalid.** Every test used the same seed per
rep, so the pp2048 prompt began with the pp512 prompt of the same seed, and the d2048 prompt was the pp2048 prompt.
Ollama keeps a prompt cache across requests and reused those prefixes (visible in d2048: prompt rate ~9,700
tok/s). Our engine, mlx-lm and llama-bench start each request cold, so only Ollama gained. Re-measured with prompts
distinct per test (`ids_for` now seeds on (seed, length, kind); `gptoss_chunk/ollama_fresh.jsonl`, 3 reps, one
round): Ollama gpt-oss pp512 355.20-399.34, **pp2048 403.49-431.13** (v10 table: 515.80), at or below ours (8-bit
attention arm, chunk 512: 428.11-436.46 in the chunk test; ranges overlap; different runs, 3 reps each). Its pp2048 / pp512 ratio fell from 1.30 (a reused 512-token prefix
predicts 1.33) to about 1.1. Decode tok/s is unaffected (timed from the first
token). Ollama's OLMoE and DeepSeek-V2-Lite pp2048 rows carry the same bias; all Ollama rows are re-run in v11.

Findings (decode; prefill rows of Ollama withdrawn):
- Decode: ours is ahead of Ollama on all three models (OLMoE 1.19x at d0 / 1.14x at d2048, DeepSeek-V2-Lite
  1.64x / 2.20x, gpt-oss 1.10x / 1.04x), at less GPU energy per token (OLMoE 66 vs 103 mJ, gpt-oss 265 vs 360 mJ at d0).
- Ollama on DeepSeek-V2-Lite is slow and noisy in decode too (CV 6-17%), on the same GGUF where llama.cpp is steady.
  Not investigated.
- A larger prefill chunk does not raise our gpt-oss pp2048 (`gptoss_chunk/`, 8-bit attention arm, 2 rounds x 3
  reps, 16:20-16:45): chunk 512 428.11-436.46, chunk 1024 404.84-436.37 (level), chunk 2048 329.00-406.44 (slower).
- Unchanged gaps: OLMoE prefill (0.95-0.96x), DeepSeek-V2-Lite short-context decode behind mlx-lm (on flagged data
  here; 11% in v8), gpt-oss pp512 (0.94x of llama.cpp).

### 1ze. Standard accuracy: WikiText-2 perplexity and KL divergence, ours vs mlx-lm vs llama.cpp (2026-11-19)

**Method** (`tools/mlx_accuracy_ppl.py`, the llama.cpp `llama-perplexity` convention): WikiText-2 raw test
(Salesforce/wikitext, rows joined as wiki.test.raw; the same file, `results/accuracy/wiki.test.raw`, is given to
llama-perplexity -f); 512-token chunks, first 80 (20,400 scored tokens), positions 256-510 predict the next token;
BOS replaces a chunk's first token when the tokenizer adds one. Perplexity +- its standard error. KL divergence:
reference log-probabilities saved from the bf16 checkpoint run in our engine (validated against transformers on
the tiny models; the 2 GB of reference logits are in ~/.cache/millijoule/accuracy_refs, not in git), then per-token
KL(ref || run), its 99th percentile and the fraction with the same top token. llama.cpp's figures are its own
`--kl-divergence` of Q4_K_M against its F16 GGUF: a different reference and implementation, so they compare
quantization formats, not engines.

| model / weights | engine | perplexity | mean KL | KL p99 | same top |
| --- | --- | --- | --- | --- | --- |
| OLMoE-1B-7B bf16 (reference) | ours | 11.540 +- 0.251 | - | - | - |
| OLMoE mlx-community 4-bit g64 (4.5 bpw) | ours | 12.004 +- 0.258 | 0.1592 +- 0.0019 | 1.204 | 81.96% |
| same weights | mlx-lm | 11.987 +- 0.257 | 0.1589 +- 0.0019 | 1.197 | 81.99% |
| OLMoE mlx 4-bit g32 (5.0 bpw, converted here) | ours | 11.799 | 0.1258 | 0.974 | 83.96% |
| OLMoE mlx 5-bit g64 (5.5 bpw, converted here) | ours | 11.593 | 0.0518 | 0.426 | 89.56% |
| OLMoE F16 GGUF (llama.cpp's reference) | llama.cpp | 11.532 +- 0.251 | - | - | - |
| OLMoE Q4_K_M GGUF | llama.cpp | 12.000 +- 0.263 | 0.0813 +- 0.0011 (vs its F16) | 0.677 | 87.28% |
| DeepSeek-V2-Lite mlx-community 4-bit | ours | 9.078 +- 0.183 | | | |
| same weights | mlx-lm | 9.077 +- 0.183 | | | |
| DeepSeek-V2-Lite Q4_K_M GGUF | llama.cpp | 8.893 +- 0.181 | | | |

| Qwen3-30B-A3B mlx-community 4-bit, streamed (40 chunks) | ours | 8.123 +- 0.238 | | | |
| Qwen3-30B-A3B Q4_K_M GGUF, -ncmoe 48 (40 chunks) | llama.cpp | 8.159 +- 0.244 | | | |
| gpt-oss-20b, exact bf16 attention, lean | ours | 181.158 +- 3.975 | | | |
| gpt-oss-20b, 8-bit attention, lean | ours | 176.956 +- 3.874 | | | |
| gpt-oss-20b MXFP4 GGUF (ggml-org) | llama.cpp | 172.832 +- 3.776 | | | |

- The two llama.cpp F16 figures differ by 0.04: 11.532 is `olmoe_llamacpp_f16.log` (the run that saved the base
  logits); the `--kl-divergence` run reports its base as Mean PPL(base) 11.492, computed from those saved logits.
  The cause is not established (a possible one: the precision at which llama.cpp stores base logits; the file
  was not kept, so it was not checked). The table keeps the direct run's 11.532.
- **Our engine = mlx-lm on the same weights** (perplexity within 0.02, KL 0.1592 vs 0.1589, same top 81.96 vs
  81.99%). The engine adds no error of its own.
- **llama.cpp's Q4_K_M is the better 4-bit file**: half the KL of mlx-community 4-bit g64 (0.081 vs 0.159) at the
  same perplexity on OLMoE, and a lower perplexity on DeepSeek-V2-Lite (8.89 vs 9.08).
- Qwen3-30B-A3B: equal (8.12 vs 8.16). The 20,400 / 10,200-token jobs took ours 124 s and llama.cpp ~29 min.
- **gpt-oss-20b: settled against a full-precision reference.** Raw-text perplexity of this chat model is high on
  every engine (llama.cpp's first 3 chunks 318.8, ours 358.6 on the same 3). The float32 reference (float32
  activations on the released weights, layer-streamed, `tools/mlx_real_eval.py --wikitext-chunks 80`,
  `accuracy/gptoss_f32ref_wikitext.json`; validated against transformers to ~6e-8 on the tiny model) gives
  **180.29 +- 3.95** on the same 80 chunks. Ours exact 181.16 (+0.5%, mean KL 0.0162, p99 0.305, top-1 93.0%; the
  engine's greedy path 93.6%), ours 8-bit attention 176.96 (-1.8%, KL 0.0306, top-1 89.5%), llama.cpp's MXFP4 GGUF
  172.83 (-4.1%; KL not measured against this reference). llama.cpp's lower perplexity is further from the model's,
  not closer; our exact path is within 0.5% of it. The tool caps MLX's cache and wires the weights as the
  Engine does (without either, a streamed 30B swapped and a lean gpt-oss run thrashed).
- **Quality-matched speed** (OLMoE, ours, `accuracy/olmoe_quant_speed.jsonl`, 3 reps, one run each, not ABAB):
  4-bit g32 tg128@d0 125.7-126.4, pp512 1180.9-1194.0; 5-bit g64 114.6-115.1 and 1067.3-1154.7 tok/s. Quality is
  compared on the same text by perplexity: 5-bit g64 11.593 vs Q4_K_M 12.000 (SE ~0.25 each; bf16 11.540). The KL
  figures (5-bit 0.052 vs Q4_K_M 0.081) are secondary: they are against different references (our bf16 run vs
  llama.cpp's own F16). Against llama.cpp Q4_K_M in suite v8 (102.41 decode, 1264.78 pp512): at a lower perplexity,
  our decode is still ~12% ahead; prefill ~84-91% of llama.cpp.

### 1zd. The general MoE model (MoeLM): any family, any size (2026-11-19, from 08:00)

**What it is** (`src/mj_mlx/moe_lm.py`, design in `MOE_DESIGN.md`): one configurable model for
decoder-only MoE checkpoints (transformers or mlx-community layout) on the existing engine. Families wired:
Mixtral, Qwen2-MoE, Qwen3-MoE, OLMoE, DeepSeek-V2, DeepSeek-V3, GLM-4-MoE. `load_model` dispatches them;
gpt-oss and Qwen3.5/3.6 keep their tuned classes. Placement (`placement.py`): resident / memory-mapped /
SSD-streamed experts, chosen from the checkpoint size against the GPU working set and RAM, repacked once into
~/.cache/millijoule (`load_model(path, placement="auto")`).

**Correctness gates** (`tests/mlx/test_moe_lm.py`, tiny random checkpoints built locally with transformers; 8
configs incl. DeepSeek-V2 with YaRN and group-limited routing, V3 sigmoid + correction bias, GLM partial rope +
qk-norm + bias, Mixtral sliding window, Qwen2-MoE gated shared expert + dense layer): prefill logits within 7.0e-5
of transformers float32 (all 8); Engine greedy (chunked prefill, paged decode, 4 requests) equal to transformers;
bf16 overlap == plain; streamed (3 of 8 slots) and mapped experts == resident tokens; rope inverse frequencies
== transformers'. 242 MLX tests pass (after the streaming commits).

**Real checkpoints vs mlx-lm on the same weights** (`tools/mlx_moe_check.py`: 3 chat prompts, 64 greedy tokens,
then teacher-forced on mlx-lm's tokens):

| model | greedy tokens identical | top-1 agreement | mean KL |
| --- | --- | --- | --- |
| OLMoE-1B-7B-0125-Instruct 4-bit | 3 / 3 prompts (64 tokens each) | 1.000 / 1.000 / 1.000 | 0.0010 / 0.0002 / 0.0006 |
| DeepSeek-V2-Lite-Chat 4-bit (MLA) | 3 / 3 | 1.000 / 1.000 / 1.000 | 0.0002 / 0.0011 / 0.0010 |

**Speed** (`results/moe_lm/{olmoe,dsv2lite}_quick.jsonl`; mlx_bench_worker, 3 reps per test, 2 s gaps; each
engine in its own run, not ABAB, so these are indicative only; [min-max]):

| model | test | ours (first) | ours (now) | mlx-lm |
| --- | --- | --- | --- | --- |
| OLMoE 4-bit | tg128@d0 | 47.84-48.05 | **136.07-136.80** | 129.71-135.82 |
| | pp512 | 954.04-964.31 | 1139.44-1169.21 | 1205.71-1245.64 |
| DeepSeek-V2-Lite 4-bit | tg128@d0 | 48.05-48.13 | 61.43-61.56 | 64.58-66.31 |
| | tg128@d2048 | 21.16-21.24 | **51.77-52.07** | 49.02-49.39 |
| | pp512 | 521.34-529.46 | 557.18-561.38 | 525.02-566.83 |
| | pp2048 | 451.93-452.83 | **524.33-549.57** | 378.40-507.51 |

What moved the numbers:
- **Checkpoint dtype.** OLMoE's mlx checkpoint has float16 scales; bf16 activations against them made decode 48
  tok/s (a synced profile put the experts at 1.6 ms per layer, consistent with gather_qmm reading every expert;
  not traced further). Quantized checkpoints now run in their scales' dtype (`placement.checkpoint_dtype`), as
  mlx-lm does: 136 tok/s.
- **MLA decode.** The paged kernel runs one threadgroup per (sequence, KV head) and MLA has one KV head, so all 16
  query heads shared one threadgroup (d2048: 21 tok/s). Decode rows now use SDPA over the gathered latent rows
  (49.8-50.4), with kv_b kept 4-bit for the absorption (per-head gather_qmm instead of float16 copies: 60-62 at d0).
- **MLA prefill.** Absorbed attention at D = 576 does ~1.8x the attention math of the expanded form; prefill items
  now expand K / V from the cached latents and attend per head (pp2048 396 -> 524-550), decode stays absorbed.
  The cache stays the latent rows: ~1.2 KB per token per layer vs mlx-lm's expanded ~10 KB.
- Remaining at d0 on DeepSeek-V2-Lite: ~6% behind in these quick runs; in suite v8 below 11% (58.40 vs 65.67) (graph build 1.43 ms of a 17.9 ms synced step; precise mode and
  MLX_MAX_MB_PER_BUFFER 1000 made no difference: 61.1-61.7 in all three arms). SDPA at D = 576 has no fused kernel.

**Suites v8 / v9: the general MoE model against mlx-lm and llama.cpp** (`results/bench_suite/v8_moe/summary.md`,
`v9_q30b/summary.md`; 2 rounds x 5 reps, mean of unflagged blocks; flagged = swap grew > 256 MB, listed in each
summary). The residency-refresh default and the fused prefill glue were in; the Qwen3-30B run used 64 slots per
layer (`--slots 64`; the memory-derived default is 76).

| model | test | ours | mlx-lm | llama.cpp |
| --- | --- | --- | --- | --- |
| OLMoE-1B-7B 4-bit | tg128@d0 | **138.01 +- 0.60** | 136.02 +- 1.56 | 102.41 +- 3.08 |
| | tg128@d2048 | **99.59 +- 4.28** | 95.49 +- 0.37 | 84.06 +- 2.09 |
| | pp512 | 1189.54 +- 11.06 | 1221.17 +- 14.27 | **1264.78 +- 8.01** |
| | pp2048 | 1263.35 +- 8.34 | **1318.79 +- 11.82** | 1206.59 +- 3.47 |
| DeepSeek-V2-Lite 4-bit | tg128@d0 | 58.40 +- 2.67 | **65.67 +- 1.37** | 51.06 (flagged) |
| | tg128@d2048 | **51.08 +- 0.91** | 47.15 +- 2.46 | 38.27 (flagged) |
| | pp512 | **572.71 +- 5.62** | 513.31 +- 127.47 | 509.39 (flagged) |
| | pp2048 | **570.16 +- 2.36** | 340.96 +- 200.49 | 524.64 (flagged) |
| Qwen3-30B-A3B 4-bit (17.2 GB > RAM) | tg128@d0 | **27.71 +- 4.73** | cannot hold it | 2.71 +- 0.45 |
| | tg128@d2048 | **26.24 +- 4.01** | | 3.57 +- 0.59 |
| | pp512 | **188.13 +- 6.68** | | 15.57 +- 0.40 |
| | pp2048 | **291.75 +- 3.52** | | 13.35 +- 0.31 |

- OLMoE: ours from one unflagged block (round 2 flagged, +331 MB swap); the d2048 lead over mlx-lm is inside the CI.
- DeepSeek-V2-Lite: every llama.cpp block was flagged (+1143 / +1535 MB swap), its figures are the flagged means;
  ours one unflagged block (round 1 flagged; its tg128@d0 61.78). mlx-lm's prefill swung 198-604 tok/s.
- Qwen3-30B: different rounds, one unflagged block each: ours round 2, llama.cpp round 1 (q30b_r1_mj +641 MB and
  q30b_r2_llama.cpp +706 MB were flagged). llama.cpp with all experts on the CPU (-ncmoe 48). Partial offload,
  single runs (`moe_lm/q30b_llama_ncmoe.txt`): -ncmoe 32 pp512 20.64 / tg128 2.89 tok/s (swap 10.8 GB after);
  -ncmoe 24 26.81 / 1.46 (swap 10.4 GB). Decode no faster beyond noise (2.89 vs 2.71 +- 0.45); prefill faster:
  llama.cpp's best measured pp512 is 26.81, so our pp512 lead is ~7x (188.13 / 26.81), not 12x.
- Greedy ids ours vs mlx-lm: 16/20 identical on both OLMoE and DeepSeek-V2-Lite (random-token prompts).
- **Mixtral-8x7B-Instruct 4-bit (26.3 GB, 1.6x this Mac's RAM)** runs through the same path, placement auto ->
  stream, 3 of 8 expert slots per layer (each expert ~100 MB): 0.59-0.60 tok/s including prefill (run with --new 32: 32
  tokens in 53.2-54.1 s per prompt), coherent output (`moe_lm/mixtral_check.json`; the token count was added to the
  file afterwards from the command line, the tool now records it). SSD-bound by construction (each token reads two ~100 MB experts in each of 32
  layers, most of them misses). With 3 slots the C++ streamer aborted ("no free slot": in-flight prefetches held
  the slots); it is now used only with slots >= 4 x tokens x top-k, else the Python streamer reads in rounds. Not
  compared with another engine (no GGUF downloaded; mlx-lm cannot hold it).

### 1zc. 2026-11-19 morning (06:10-): what limits gpt-oss decode and prefill, fused prefill MoE glue, the host gate

**The exact path's "slow state" is not the GPU clock and not slower kernels** (`tools/mlx_decode_state_probe.py`,
`results/decode_state/p1.jsonl`, 29 reps of tg128@d0, exact config, mlock; stopped early, every rep in one state).
- 30.6-32.1 tok/s throughout. GPU 98-99% active, 95-96% of active time at its top state (GPUPH P15); P cluster at
  mid states (mean ~P13 of 19). ~80 MB free, swap 3.9-4.8 GB, page-ins and swap-ins in every rep.
- The same parts as 1y, timed GPU-bound in this state (`decode_state/parts_pack_now.json`): 25.77 ms per token vs
  28.07 ms in 1y. So the kernels were not slower.
- A tiny kernel on a mapped, mlocked 265 MB expert array costs the same per eval as on an MLX-allocated one
  (0.15-0.25 ms incl. sync): no per-commit residency cost for the no-copy buffers.
- Host timings per step (wrapped runner calls): wait for the candidates 11.2 ms, rescore 0.4, build 1.0, and "arm"
  (embedding row + async_eval of step N+1) 17 ms. The 17 ms turned out to be MLX's eval waiting on the GPU (below),
  not CPU encoding.

**gpt-oss prefill by part** (`tools/mlx_gptoss_prefill_parts.py`, `gptoss_prefill_parts/a8_n2048.json`; 8-bit
attention, 2048 tokens in 512-token chunks, each part synced, so the instrumented total is 6.1 s vs 4.93 s plain):
experts 4.31-4.35 s, attention projections 1.01-1.03 s, attention core 0.28 s, router 0.04-0.05 s, other 0.43 s.
- Experts, one layer, real mapped weights, random routing (scratch probe): at 512 tokens 38.78 ms, of which the two
  sorted `gather_qmm` calls 36.24 ms (2.81 TFLOPS) and glue 2.54 ms (bias gathers + adds, SwiGLU, un-permute,
  weighting, sum); at 2048 tokens 144.0 ms, gather_qmm 3.06 TFLOPS, glue 10.7 ms. For scale on this M4: dense bf16
  matmul 3.64-3.68 TFLOPS, dense MXFP4 and 8-bit `quantized_matmul` 3.31-3.38.
- Steady prefill rate ~420 tok/s in a back-to-back probe (engine init 6-19 ms after the first request), so the
  suite's pp512 gap is per-token, not a fixed per-request cost.

**Fused prefill MoE glue (on by default, `MJ_PREFILL_GLUE=0` turns it off).** For steps of at least 256 (token,
expert) pairs, `Experts._fused_sorted` replaces the glue with two Metal kernels (`moe_kernels.bias_swiglu_sorted`:
bias + clamped SwiGLU from the sorted expert of each row; `combine_sorted`: bias + un-permute + routing weight + sum
over k), float32 inside. Not bitwise equal to the op sequence, so decode and speculative verify steps (< 256 pairs)
keep the old ops (test).
- One layer: 38.16 -> 35.63 ms at 512 tokens, 146.18 -> 136.50 ms at 2048 (-6.6%).
- vs a float32 reference of the same MoE layer (512 tokens): mean abs error 0.0342 -> 0.0256, RMS 0.0853 -> 0.0699.
- Real-weights accuracy (`tools/mlx_real_eval.py`, 1296 teacher-forced positions vs the float32 reference;
  `gptoss_prefill_parts/real_eval_glue{0,1}.json`):

  | config | mean KL | p99 KL | max KL | top-1 |
  | --- | --- | --- | --- | --- |
  | precise (exact), ops | 0.01571 | 0.2246 | 1.073 | 0.969 (greedy path 0.972) |
  | precise (exact), fused | 0.01202 | 0.1517 | 1.043 | 0.959 (greedy path 0.975) |
  | 8-bit attention, ops | 0.05021 | 1.138 | 3.215 | 0.9375 |
  | 8-bit attention, fused | 0.02903 | 0.7577 | 1.014 | 0.9414 |

- A/B (`gptoss_prefill_parts/glue_ab.jsonl`, mj-a8 suite config, ABAB, 3 timed reps per test per round):

  | test | ops: median [min-max] | fused: median [min-max] |
  | --- | --- | --- |
  | pp512 | 379.77 [376.08-381.14] | 398.64 [397.08-402.94] (+5.0%) |
  | pp2048 | 414.59 [411.04-417.15] | 440.28 [437.49-441.91] (+6.2%) |
  | tg128@d2048 | 35.85 [35.65-36.27] | 35.68 [35.49-36.06] |

  Decode ids after the 2048-token random prompt: 3 of 6 seeds identical, 3 diverge at token 11-19 (the KV cache
  differs in the last bits; random-token prompts are full of near-ties). Tests: `tests/mlx/test_gpt_oss.py`
  (fused vs ops at 128 / 300 tokens in float32 and bf16; short steps bitwise unchanged).

**The host gate (on by default; `MJ_LEAN_GATE=0` turns it off).** Lean gpt-oss decode builds step N+1 on a host
embedding buffer while N runs (1z), but only submitted it once N's token was on the host. Now it is submitted right
away behind a Metal shared-event wait (`HostGate` in `ext/mj_stream.cpp`, a primitive whose eval encodes
`encodeWait`), and `arm()` writes the embedding row and signals. Falls back to the ungated path without the extension.
- First attempt deadlocked on the real model (Metal "GPU Timeout"): `sample` showed the main thread in MLX's
  `eval_impl` waiting on a condition variable. MLX's eval waits while about 10 command buffers are in flight, and at
  `MLX_MAX_MB_PER_BUFFER=150` a gpt-oss step is ~50 buffers (each 400 MB mapped expert layer forces one); buffers
  behind a closed gate cannot finish. The tiny test model never reached the limit.
- Global limits do not work: unlimited ran out of Metal memory in a 512-token prefill chunk; MB 2600
  (`decode_state/gate_ab3.jsonl`) cost prefill (pp512 278.63 / pp2048 327.62 vs 397.58 / 431.09 at 150).
- So the extension switches MLX's per-buffer limits per step (`set_buffer_limits`; Device's private fields through
  `#define private public`): a gated decode step is one command buffer, everything else keeps 150.
- Encoding a gated step then took ~3 ms (scratch trace): the old 17 ms "arm" was MLX's throttle waiting on the GPU.
- Tests (`tests/mlx/test_overlap.py`): same tokens as the ungated deferred path (roomy and tight pools, preemption,
  stops), every gate opened, and a gate holds the GPU until signalled. 212 MLX tests pass.
- A/B (`results/decode_state/gate_ab4.jsonl`, 07:08-07:26; suite configs, mlock, ABAB, 3 timed reps per test per
  round, medians [min-max] of 6):

  | test | mj-a8 gate off | mj-a8 gate on | mj (exact) gate off | mj (exact) gate on |
  | --- | --- | --- | --- | --- |
  | pp512 | 392.97 [389.83-412.07] | 397.35 [370.89-403.79] | 396.95 [366.05-401.41] | 399.96 [393.23-403.03] |
  | pp2048 | 432.19 [429.85-433.51] | 430.45 [420.22-439.34] | 416.68 [410.31-425.49] | 422.50 [413.25-427.24] |
  | tg128@d0 | 37.58 [34.80-37.85] | **39.70** [39.63-39.87] | 30.58 [29.95-30.94] | **34.09** [18.20-36.41] |
  | tg128@d2048 | 35.21 [32.66-36.24] | **37.84** [35.75-38.17] | 29.79 [29.62-30.27] | **35.02** [33.85-35.03] (n = 3) |

  - Decode: **+5.6% / +7.5%** (8-bit attention, depth 0 / 2048) and **+11.5% / +17.6%** (exact). Greedy ids
    identical with the gate on and off in all 21 pairs. Prefill unchanged (no gated steps there).
  - The exact gated process of the first round (`--seed0 0`) ended silently after its tg128@d0 reps (the last at
    18.20 tok/s with a 7.06 s TTFT, a system-wide stall); no traceback, no crash report, and `log show` returned no
    entries for that minute. Not seen in the second round. Stalls of that size also hit ungated runs this morning
    (5.55 tok/s, 16.49 s TTFT in `gate_ab.jsonl`). Cause unknown.
  - The exact path's gap to the 8-bit path shrank from ~7 tok/s to ~3-5. Interpretation (not proven): the exact
    path's "slow state" was mostly the serial wait in each step, which the gate overlaps.

**Suite v6, gpt-oss-20b, with both changes on by default** (`results/bench_suite/v6_gptoss/summary.md`,
07:33-07:55; same command as v5: 2 rounds x 5 reps, mlock, chunk 512). Unflagged blocks: llama.cpp round 1, mj-a8
round 2; every mj (exact) block was flagged for swap growth (+1002 / +1669 MB, mlock), as were mj-a8 round 1 (+499)
and llama.cpp round 2 (+541 MB; its tg128@d2048 also failed there with a Metal out-of-memory error).

| test | llama.cpp | mj-a8 (8-bit attention, matched to the GGUF) | mj-a8 / llama.cpp |
| --- | --- | --- | --- |
| pp512 | 428.57 +- 10.19 [415.90-441.25] | 400.80 +- 1.71 [398.67-402.92] | 0.94 |
| pp2048 | 441.17 +- 5.82 [433.93-448.41] | 435.65 +- 3.09 [431.81-439.49] | 0.99 (CIs overlap) |
| tg128@d0 | 35.38 +- 0.14 [35.20-35.56] | **39.46 +- 0.33** [39.05-39.88] | **1.12** |
| tg128@d2048 | 35.46 +- 0.41 [34.95-35.97] | **38.09 +- 0.07** [38.01-38.17] | **1.07** |

- **gpt-oss-20b decode, 8-bit attention (lossy, matched to the GGUF's Q8_0): ahead of llama.cpp at both depths, CIs
  disjoint** (v5: level). One unflagged block of 5 reps each, from different rounds (llama.cpp r1, mj-a8 r2); the
  flagged blocks point the same way (mj-a8 r1 tg128@d0 39.68 vs llama.cpp r2 35.98). GPU energy per decode token
  263 / 292 mJ (llama-bench gives no energy window).
- Not a claim for the exact path: mj (bf16 attention) was flagged in both rounds and its flagged reps tie llama.cpp
  at depth 0 (35.91 vs 35.98).
- Accuracy of the two arms vs the float32 reference (1296 teacher-forced positions, fused glue on,
  `gptoss_prefill_parts/real_eval_glue1.json`): mj-a8 mean KL 0.0290, top-1 0.941; mj (exact) 0.0120, 0.959. No
  matching figure for llama.cpp's Q8_0 GGUF yet.
- llama.cpp r2 has no tg128@d2048: llama-bench stopped with `ggml_metal_synchronize: command buffer 0 failed with
  status 5` / `Insufficient Memory (kIOGPUCommandBufferCallbackErrorOutOfMemory)` (`v6_gptoss.log`), not a timeout
  or kill; that block was also flagged (+541 MB swap).
- Prefill: level at 2048 tokens, 94% at 512.
- mj (exact) in its flagged blocks: tg128@d0 35.91 +- 0.85, @d2048 34.56 +- 0.77, pp2048 423.93 +- 4.68 (n = 10).

**Qwen3.5-9B prefill vs mlx-lm, by part** (`tools/mlx_q9b_prefill_parts.py`, `results/q9b_prefill_parts/`;
2048 tokens, 3 reps each, each wrapped call synced; 07:30). Plain: ours 226.9 / 218.4 / 213.2 tok/s (falling rep to
rep), mlx-lm 220.0 / 220.1 / 219.8. Instrumented, ours vs mlx-lm: MLP 5.95-6.28 vs 6.48-6.58 s, DeltaNet layers
2.37-2.48 vs 2.57-2.65 s (of which the gated-delta kernel 0.23 vs 0.19 s), full attention 0.72-0.76 vs 0.89-0.94 s,
rest 0.21 vs 1.48-1.53 s (mlx-lm computes the head for all 2048 positions). Syncing cost mlx-lm 2.1 s and us
0.1-0.3 s, so in plain runs its kernels overlap more than ours; every part of ours is as fast or faster when timed
alone except the recurrence kernel (+40 ms). The suite's 4% gap (1zb) is not a single slow part; not pursued further
today. Both are bound by the 4-bit matmuls (~3.3 TFLOPS for the MLPs).

### 1zb. Overnight 2026-11-19 (01:31-05:30): suite v4 for Qwen3.6-35B and Qwen3.5-9B, a stale energy counter, the candidate head

**Suite v4, Qwen3.6-35B-A3B** (`results/bench_suite/v4_q35b/summary.md`, 01:31-02:28; 96 expert slots per
layer, C++ streamer; llama.cpp Q4_K_M with experts on the CPU, `-ncmoe 40`, 3 reps). Unflagged blocks (one per
engine; the other was flagged for swap growth, +750 / +438 MB, and is listed in summary.md):

| test | llama.cpp | mj | mj / llama.cpp |
| --- | --- | --- | --- |
| pp512 | 16.51 +- 0.43 | 161.69 +- 4.64 | 9.8x |
| pp2048 | 12.95 +- 0.30 | 277.36 +- 4.32 | 21.4x |
| tg128@d0 | 3.78 +- 0.80 | 24.06 +- 5.18 (median 21.34, 19.76-32.20) | 6.4x |
| tg128@d2048 | 3.20 +- 0.60 | 21.65 +- 0.91 | 6.8x |

- No Metal out-of-memory at 96 slots (v1 at 128: 3 of 4 blocks failed in pp2048).
- Streamed decode depends on the prompt (expert reuse): one seed gave 32.20 tok/s, most 20-23.

**Suite v4, Qwen3.5-9B** (`results/bench_suite/v4_q9b/summary.md`, 02:29-02:58; one of our blocks flagged,
+278 MB swap, and its numbers are within 0.6% of the counted block except pp2048 -2.5%):

| test | llama.cpp | mlx-lm | mj |
| --- | --- | --- | --- |
| pp512 | 216.47 +- 3.69 | **227.19 +- 1.96** | 219.87 +- 1.13 |
| pp2048 | 200.21 +- 2.34 | **226.27 +- 5.37** | 224.45 +- 1.98 |
| tg128@d0 | 18.82 +- 0.03 | 21.86 +- 0.17 | **21.89 +- 0.04** |
| tg128@d2048 | 18.36 +- 0.11 | **21.46 +- 0.23** | 21.41 +- 0.03 |

- Decode ties mlx-lm (both CIs overlap) and is 16-17% ahead of llama.cpp; prefill is 1-3% behind mlx-lm and
  2-12% ahead of llama.cpp. Our spread is the narrowest (CV 0.2% vs 0.8-1.1%).
- Greedy ids vs mlx-lm: 9 of 20 identical (v1: 8 of 20), as before different float paths.

**Energy windows were wrong in some reps: a stale first IOReport sample.** Our tg128@d2048 windows on the 9B read
either ~10 W (as mlx-lm and as depth 0) or 37.5-40.6 W (5 of 10 reps), above the prefill's own ~20 W; gpt-oss had
one 31.3 W window and the 35B and mlx-lm one 0 J window each at 99.9% GPU active. tok/s is not affected (timed on
the host). Cause, measured (`tools/mlx_energy_lag_probe.py`, `results/energy_lag/`):
- Sampling the GPU rail every 50 ms through 6 + 4 requests (ours, mlx-lm) gave consistent phases every time:
  prefill 18.5-20.6 W, decode 9.3-10.9 W, idle tail 0.0-0.9 J over 3 s.
- With only two samples per window (as the worker), one of 8 reps read a first-token sample equal to the one from
  before the 9 s prefill (prefill 0.0 J, decode 228.4 J); a second sample taken right after it was current
  (decode 58.6 J). So the first sample after a long gap can be stale and the whole prefill lands in the next
  window.
- Fix: `EnergyCounter.snapshot()` samples twice and keeps the second. 12 of 12 reps clean afterwards
  (`double_read_check.txt`: prefill 166-186 J, decode 54-65 J).
- Energy in the v1-v4 suite tables before this fix is suspect where a window followed a long prefill (tg@d2048)
  or a long idle; medians are mostly unaffected but means are not. Suite v5 (below) is measured with the fix.

**3-bit candidate head for lean gpt-oss (default now; `MJ_LEAN_HEAD_BITS=4` restores the old one).** Lean decode
takes the top-16 of a low-bit lm_head and picks among them exactly on the host (float64, bf16 rows). The only way
the head width changes a token is a miss (exact argmax outside the top-16).
- Miss rate on the real weights (`tools/mlx_gptoss_head_miss_real.py`, `results/gptoss_head_miss_real.json`):
  2048 final hidden states from 16 real greedy decodes of 128 tokens (half random-token prompts, half one token),
  exact argmax over all 201088 bf16 rows on the host:

  | head | size | 1-row matmul (median of 20) | top-1 | top-4 | top-16 | top-32 | top-64 |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | 2-bit g64 | 181 MB | 2.21 ms | 329 | 62 | 15 | 7 | 2 |
  | 3-bit g64 | 253 MB | 2.72 ms | 148 | 19 | **0** | 0 | 0 |
  | 4-bit g64 (was default) | 326 MB | 3.36 ms | 73 | 1 | **0** | 0 | 0 |

  Misses out of 2048. 0 of 2048 bounds the rate below ~0.15% (95%), for 3 and 4 bits alike; 3-bit's top-4 count
  shows a thinner margin than 4-bit's. 2-bit misses even at top-64, so it is out.
- A/B end to end (`results/gptoss_decode_parts/headbits/`, 03:21-03:35, 2 rounds, order reversed in round 2,
  3 reps per test per arm, mlock, chunk 512; the exact path was in its fast state this window):

  | arm | tg128@d0 4-bit | 3-bit | tg128@d2048 4-bit | 3-bit |
  | --- | --- | --- | --- | --- |
  | mj (exact attention) | 33.54 | **34.77 (+3.7%)** | 32.37 | **33.50 (+3.5%)** |
  | mj-a8 (8-bit attention) | 36.49 | **37.77 (+3.5%)** | 35.34 | **35.68 (+1.0%)** |

  Medians of 6, tok/s. Greedy ids identical in 24 of 24 (arm x test x seed) pairs. Test:
  `test_lean_head_bits_3_same_tokens_as_4` (tiny model). 204 MLX tests pass.

**Suite v5, gpt-oss-20b** (`results/bench_suite/v5_gptoss/summary.md`, 03:37-03:59; 3-bit candidate head,
double-read energy meter, otherwise as v4). Both of our exact-path blocks were flagged (swap +675 / +614 MB; mlock
pins 10 GB, so this engine grows swap itself), as was llama.cpp's round 2 (+717 MB). The summary used to drop an
engine whose blocks were all flagged; it now prints it, and the flagged numbers are given here as such:

| test | llama.cpp (unflagged, n = 5) | mj-a8 (unflagged, n = 10) | mj exact (flagged, n = 10) |
| --- | --- | --- | --- |
| pp512 | 434.95 +- 1.61 | 377.45 +- 3.34 | 367.78 +- 10.52 |
| pp2048 | 439.70 +- 6.95 | 411.38 +- 2.29 | 404.04 +- 0.54 |
| tg128@d0 | 36.90 +- 0.12 | **37.35 +- 0.80** | 32.00 +- 2.98 (r1 34.67-35.07, r2 28.18-29.78) |
| tg128@d2048 | 33.85 +- 1.91 | **34.39 +- 1.39** | 31.29 +- 2.51 (r1 33.59-33.75, r2 28.64-29.18) |

- **Decode with 8-bit attention (llama.cpp's own attention precision) now matches llama.cpp**: means +1.2% at
  depth 0 and +1.6% at depth 2048, 95% CIs overlapping (36.78-37.92 vs 36.75-37.04; 33.39-35.38 vs 31.48-36.22).
  llama.cpp's flagged round-2 block: 36.34 / 34.84.
- Prefill is still behind: 87% (pp512) and 94% (pp2048) of llama.cpp.
- GPU energy (fixed meter): mj-a8 46 / 49 mJ/token prefill, 198 / 271 mJ/token decode at depth 0 / 2048.
- The exact path's slow state came back between rounds (r1 03:39-03:40 fast, r2 03:57-03:58 slow) with prefill
  unchanged (404 tok/s both rounds), swap 3.5-4.2 GB before each block, memory free 83-85%, loadavg 1.3-1.9, no
  thermal note. mj-a8 at depth 2048 dropped less (r1 34.95-36.11, r2 32.55-33.79). Decode-only slowdown points at
  the host side of each step (the deferred step waits on the host for every token).

**Suite v5, Qwen3.5-9B** (`results/bench_suite/v5_q9b/summary.md`, 04:00-04:29; double-read energy meter;
no block flagged, n = 10 each):

| test | llama.cpp | mlx-lm | mj | mj GPU mJ/token (mlx-lm) |
| --- | --- | --- | --- | --- |
| pp512 | 216.49 +- 3.63 | **227.01 +- 0.84** | 217.47 +- 1.84 | 88 (86) |
| pp2048 | 200.14 +- 2.58 | **228.70 +- 3.79** | 220.16 +- 3.33 | 88 (85) |
| tg128@d0 | 18.75 +- 0.16 | 21.56 +- 0.50 | 21.53 +- 0.21 | 481 (480) |
| tg128@d2048 | 18.36 +- 0.01 | **21.49 +- 0.08** | 21.09 +- 0.22 | 468 (456) |

- The energy fix shows: our tg128@d2048 reads 468 mJ/token (v4: a mean of 1034 from stale windows), level with
  mlx-lm.
- vs mlx-lm: decode tied at depth 0, 1.9% behind at depth 2048 (CIs apart; v4 had them tied at 21.41 vs 21.46);
  prefill 4% behind. vs llama.cpp: decode +15% / +15%, prefill +0.5% / +10%.

**Main-thread QoS A/B for the slow state (inconclusive)** (`tools/mlx_qos_run.py`, `gptoss_decode_parts/qos2/`,
04:29-04:39, 3 rounds alternating, exact path, 3 reps per test): default vs `QOS_CLASS_USER_INTERACTIVE` medians
33.48 vs 34.04 (tg128@d0) and 32.69 vs 32.91 (tg128@d2048), n = 9, ranges overlapping. The whole window stayed in
the fast state (30.6-34.6), so it neither confirms nor rules out host-core placement as the slow state's cause.
Default unchanged.

### 1za. Overnight 2026-11-18/19: chunked prefill, suite v4, KV write inside the decode kernel

**gpt-oss prefill chunk** (`tools/mlx_gptoss_mem_probe.py --chunk`, `results/gptoss_mem/chunk_ab.jsonl`; mlock on,
order 2048 / 1024 / 512 twice, 2 timed reps per test): pp2048 409.0-415.0 tok/s at 512 vs 224.9-401.6 at 2048 and
323.5-399.7 at 1024; MLX peak 13.09 GB vs 14.08 (2048) and 14.26 (1024); pp512 386.9-396.9 at every chunk. The
suite's gpt-oss rows now prefill in 512-token chunks (`mlx_bench_worker.py --chunk`), as llama.cpp's default ubatch.

**Suite v3** (`results/bench_suite/v3_gptoss/`, 23:48-00:11): one block per engine was counted (one of ours
flagged per round for swap growth). Our exact path decoded 28.7-30.2 tok/s in both its blocks; the same suite
command for that engine alone at 00:20 (`--round0 3`, `v3_gptoss_r3.log`) gave 33.38-33.77. Kept as run.

**Suite v4, gpt-oss-20b** (`results/bench_suite/v4_gptoss/summary.md`, 00:25-00:48; 2 rounds x 5 reps; mj and
mj-a8 with mlock, deferred lean steps, chunk 512; blocks flagged for swap growth > 256 MB are excluded and listed
separately in summary.md). Unflagged blocks (n = 5 each):

| test | llama.cpp | mj-a8 (8-bit attention, matched to the GGUF) | mj (exact) |
| --- | --- | --- | --- |
| pp512 | 437.42 +- 2.36 | 370.40 +- 2.44 | 372.01 +- 5.56 |
| pp2048 | 442.32 +- 7.25 | 416.42 +- 2.06 | 401.61 +- 3.70 |
| tg128@d0 | 37.20 +- 0.06 | 36.01 +- 0.71 | 28.80 +- 0.46 |
| tg128@d2048 | 35.74 +- 0.84 | 28.02 +- 0.35 | 27.89 +- 0.45 |

- mj-a8's unflagged block is round 2; its flagged round-1 block gave tg128@d2048 32.94 +- 0.52 and tg128@d0
  34.45 +- 0.69 (summary.md). Our flags: mj r2 (swap +787 MB), mj-a8 r1 (+382 MB); llama.cpp r1 (+273 MB).
- vs llama.cpp: prefill 85-94%; 8-bit decode 97% at depth 0. Not yet ahead on gpt-oss.
- **Unexplained state dependence of the exact path:** with the same code and command (tg128@d0, mlock, chunk 512),
  33.16-33.99 tok/s at 00:12-00:20 (`gptoss_decode_parts/suite_vs_ab.jsonl`, 5 runs; seeds 10 / 100 / 200 and
  prefill-first made no difference) and 27.86-30.13 at 00:49-00:53 (`qos_ab.jsonl`, 3 runs, foreground, so not
  background QoS). GPU utilization read 0% while idle (no outside GPU load), no thermal or performance warning in
  `pmset -g therm`, MacBook Pro M4 on AC. The v3 / v4 suite windows fell in the slow state.

**KV write inside the paged decode kernel** (merged adb1547). `kp[slots] = k` copied each layer's whole pool per
step when MLX could not update it in place (1x). On decode-only steps the paged kernel now stores the row's new
K/V itself: the threadgroup holding position kv_len - 1 writes it, issues a device-memory barrier, then reads as
before (`paged_decode_attention(..., k_new, v_new)`, single-pass and split-K). `build_meta` turns it on only when
every item is one decode token through the paged kernel, the write slots equal the block-table slots, and no other
row reads a write block (`_write_blocks_private`; true for the scheduler's tables, since only full blocks are shared
by the prefix cache). Prefill, verify and mixed steps keep the scatter. `MJ_KV_FUSED=0` turns it off.
- Tests (`tests/mlx/test_kv_fused.py`): output and pool bitwise equal to scatter-then-attend (single-pass and
  split-K rows, windows 0 / 128, float32 and bf16, gpt-oss and Qwen shapes); same tokens and same final KV pools
  end to end (gpt-oss float32, gpt-oss lean bf16, Qwen; overlap on and off); falls back when a write block is
  shared. 203 MLX tests pass.
- A/B, Qwen3.5-9B (`results/q9b_kv_fused_ab.jsonl`, 2 rounds x 3 reps per arm): tg128@d0 median 21.53 ->
  21.84, @d2048 20.85 -> 21.33, @d8192 19.30 -> 20.33 tok/s (**+1.4% / +2.3% / +5.3%**), narrower spread; greedy ids
  identical (18 test x seed pairs).
- A/B, gpt-oss-20b lean (`gptoss_decode_parts/kvglue_ab.jsonl`): within noise (d0 29.27 -> 29.62, d2048 29.52 ->
  29.02); ids identical. Its base arm shows no depth-2048 penalty, so the lean deferred path seems to update the
  small suite-sized pool in place already.

**Fused decode glue (tried, off by default):** f32 norm weights precomputed, compiled residual add, bias + SwiGLU
and bias x routing weight in one kernel each (`MJ_FUSE_GLUE=1`). Bitwise-identical hidden states (test), but slower
on gpt-oss-20b: tg128@d0 median 28.45 vs 29.27, @d2048 27.37 vs 29.52 (`kvglue_ab.jsonl`). Cause not traced.

### 1z. gpt-oss-20b decode: GPU idle around the lean host syncs (Metal trace, 2026-11-18 23:00-23:40)

**Method:** `tools/mlx_step_trace.py --normal-only` (xctrace Metal System Trace, GPU start / end per command buffer)
on the engine's own decode steps, `MJ_MLOCK=1`, packed exact attention, lean, overlap on; 84-token chat prompt.
Files: `results/step_trace_gptoss20b/r1_mlock.json`, `r2_mlock_prows.json`, `r3_mlock_prows_fastsynch.json`.
Figures below skip each run's first (warm-up) step. Steps are cut from the trace by host spacing, so a few are
mis-split (per-step busy 0.0 and 10.9 ms appear); medians are used for that reason.

**One decode step** (r1, from the buffer timeline): 24 back-to-back per-layer command buffers (~26 ms), the 4-bit
head + top-16 (~3.3-3.7 ms), an idle gap while the host reads the candidates and the 16 bf16 rescore rows, the GPU
rescore (~20-35 us), and a second idle gap while the host reads the token, reads its embedding row and builds the
next step.

| run | rows | MLX_METAL_FAST_SYNCH | idle in GPU span: median [range], ms | wall - busy: median [range], ms | steps |
| --- | --- | --- | --- | --- | --- |
| r1_mlock | serial | off | 5.7 [3.6-7.2] | 6.1 [4.4-25.4] | 4 |
| r2_mlock_prows | parallel | off | 2.7 [0.0-3.2] | 5.2 [0.6-38.4] | 5 |
| r3_mlock_prows_fastsynch | parallel | on | 2.4 [2.2-6.3] | 2.7 [1.9-9.1] | 7 |

- **Rescore rows:** 16 random lm_head rows read one at a time from the safetensors memmap took 2.58 ms median
  (p90 4.07); with 16 parallel `pread` calls 0.38 ms (p90 2.57), same bytes (`SafetensorsRows.read`; test in
  `tests/mlx/test_expert_map.py`). Rows macOS evicted come from the SSD. With parallel reads the traced idle
  roughly halves (r1 -> r2).
- **Wall time did not move measurably:** untraced steps (the tool's own 4-7 steps per run) were 32.4-33.8 ms
  (r1), 32.0-34.6 (r2), 31.6-35.2 (r3). A 128-token A/B (`results/gptoss_decode_parts/gaps_ab.jsonl`, 6 of 8
  arms, stopped for a reboot; swap 6-7 GB, ~0.1 GB free) is inconclusive: round 1 serial 22.25-25.97, parallel
  23.41-27.59, parallel + fast synch 22.98-27.72, serial + fast synch 26.93-27.44 tok/s; round 2 serial
  25.45-26.65, parallel 26.79-27.22. To be redone after a reboot.
- **The deferred lean step (merged 4d7e950; A/B `defer_ab.jsonl` ran 2026-11-18 23:33-23:40):** under overlap, a lean gpt-oss step N+1 is
  built on a page-aligned host embedding buffer (`runner._host_buffer`, wrapped by `mx.from_dlpack` without a copy)
  while step N runs; once N's token is on the host its embedding row is written into the buffer and N+1 is
  submitted (`MlxRunner.launch_deferred` / `arm` / `finish`, `Engine._step_overlap_deferred`). The greedy choice
  among the 16 candidates is made on the host (`GptOss.rescore_host`: float64 dot products of the bf16 rows with the
  float32 final hidden state), so the GPU rescore round trip is gone. Lean greedy without overlap uses the same host
  rescore. `MJ_LEAN_DEFER=0` restores the old path.
  - Tests: `tests/mlx/test_overlap.py` (deferred == synced overlap == plain, under preemption and stop tokens; the
    host buffer is read at run time, not copied at wrap time). 186 MLX tests pass.
  - A/B (`results/gptoss_decode_parts/defer_ab.jsonl`; tg128@d0, 2 rounds x 4 timed reps per arm, arm order
    old / defer / defer + fast synch / old + fast synch, mlock 10.15 GB, parallel rows in every arm):

    | arm | tok/s min | median | max |
    | --- | --- | --- | --- |
    | old (sync, then build) | 28.93 | 30.54 | 32.67 |
    | deferred | 32.25 | 33.42 | 34.04 |
    | deferred + MLX_METAL_FAST_SYNCH=1 | 32.88 | 33.54 | 34.29 |
    | old + MLX_METAL_FAST_SYNCH=1 | 30.12 | 30.74 | 35.60 |

  - **Deferred: +9.4% median decode** (30.54 -> 33.42 tok/s). Fast synch adds no clear gain on either path; not
    enabled. Greedy token ids identical in all four arms for all 8 seeds (128 tokens each).

### 1y. gpt-oss-20b gap to llama.cpp: page-ins, not host syncs; mlock the experts (2026-11-18 19:30-22:30)

**The question (1w):** on the suite, ours trailed llama.cpp on gpt-oss-20b at every test (one block: pp512 69.7 vs
431, pp2048 300.5 vs 444, tg128 27.3 vs 36.6 tok/s).

**Decode by part** (`tools/mlx_gptoss_decode_parts.py`, `results/gptoss_decode_parts/parts_*.json`): each part
for all 24 layers at decode shapes, timed GPU-bound. Exact (packed) attention: qkv 5.57, o 4.41, router 0.22,
experts 13.91, sdpa 0.17, head 3.79 ms; sum 28.07 ms. 8-bit attention: sum 23.81 ms. The whole step ran 36-37 ms
in the same session, so 9-12 ms were outside the parts.

**Hypothesis 1, host syncs (wrong):** lean mode reads the embedding row and the 16 rescore rows on the CPU, so the
runner cannot overlap decode steps (`_resolve_inputs` waits for the previous token).
- Lean off does not fit: 0.07-0.08 tok/s, MLX peak 15.7 GB, swapping (`lean_ab.jsonl`, 2 reps, stopped).
- Built `map_head` (gpt_oss.py, `expert_map.map_rows`): page-aligned copies of the embedding and bf16 lm_head,
  wrapped as zero-copy GPU arrays, so both row reads are GPU gathers and decode steps overlap. Tokens identical to
  the CPU-row path (`tests/mlx/test_overlap.py`, new cases). On the real model it thrashed (0.07 tok/s,
  `maphead_ab.jsonl`): one GPU gather from the 1.16 GB mapped table raised wired memory by 1.21 GB
  (`map_residency_probe.txt`). **Metal wires a whole no-copy buffer while a command buffer uses it**, so the two
  tables add ~2.3 GB wired. Kept as an opt-in for machines with the headroom; not in the suite.
- Host time per decode token, measured by wrapping the runner (`tools/mlx_gptoss_step_trace.py`,
  `step_trace_run1.txt`): graph build 0.75-0.81 ms, rescore rows 2.1-3.8 ms, token read + embedding row + submit
  + final wait 0.6-0.8 ms; "GPU running the step" 34.8-35.1 ms of 38.5-40.2 ms. Removing the syncs could save
  ~1-1.5 ms; not pursued. **Wrong (1z):** that 34.8-35.1 ms wait is not all GPU work; the graph is only submitted
  when the candidates are read, so host work sits inside it. The Metal trace of the same configuration (serial
  row reads) shows 3.6-7.2 ms of GPU idle per decode step, median 5.7 ms, around the two syncs.

**Hypothesis 2, page-ins (right):** the mapped experts (9.56 GB) are ordinary file cache, and macOS evicts them
under memory pressure; the GPU then waits on SSD reads mid-step. A second trace run in a more pressured state
(`step_trace_pack.json`): 302 MB of page-ins per 128 tokens -> 45.5 ms/token, 796 MB -> 52.3, 2422 MB -> 59.2.
llama.cpp keeps its weights resident (system wired peaked at 13.79 GB in llama-bench, 1w), and MLX's own wired
limit cannot hold ours (11000 / 12000 MB: Metal OOM, 1w).

**The change:** `MJ_MLOCK=1` pins every mapped expert page with `mlock` (`expert_map.py`; counts against
`vm.user_wire_limit`, 13.06 GB here; nothing in the system settings changed). Test:
`tests/mlx/test_expert_map.py`.

**A/B** (`tools/mlx_gptoss_mem_probe.py`, `results/gptoss_mem/mlock_ab.jsonl`, console `mlock_ab.log`): arms nolock / mlock / nolock /
mlock, each a fresh process, 1 warm-up + 2 timed reps per test, packed exact attention, lean, wired limit 3000 MB.
Pinned: 10.15 GB. System wired right after load (`mlock_ab.log`): 14.08-14.11 GB with mlock, 6.56-6.97 GB
without; per rep (jsonl): 13.11-13.91 GB with mlock, 5.97-13.41 GB without (Metal wires the mapped buffers while
command buffers use them).

| test | no mlock: tok/s (page-ins per rep) | mlock: tok/s (page-ins per rep) |
| --- | --- | --- |
| pp512 | 52.1, 57.3, 100.8, 103.4 (2.7-5.9 GB) | 326.1, 383.2, 389.9, 390.0 (40-106 MB) |
| pp2048 | 267.1, 260.0, 307.7, 311.2 (4.2-5.0 GB) | 401.5, 326.0, 382.6, 395.5 (456-595 MB) |
| tg128@d0 | 27.9, 27.7, 30.0, 30.5 (31-109 MB) | 27.9, 28.0, 30.5, 30.2 (23-59 MB) |

- **Prefill: pp512 5.7-7.4x (round 1) and 3.8-3.9x (round 2), pp2048 1.2-1.5x and 1.2-1.3x** (rep vs rep within a
  round). A random-token prefill touches most
  experts; without mlock each one re-read 2.7-5.9 GB from the SSD.
- **Decode: no change**: 27.7-30.5 tok/s across both arms (both moved together between rounds: 27.7-28.0, then
  30.0-30.5). Its timed reps had few
  page-ins in both arms, because decode touches the experts the warm-up just read.
- pp2048 is not fully SSD-free with mlock: 456-595 MB of page-ins per timed rep (1.36 GB in the first warm-up),
  vs 4.2-5.0 GB without. Not yet traced (candidates: non-expert pages, the bf16 rescore rows, prefill temporaries
  pushing out other file cache).
- Cost: other processes are pushed out instead. Swap was high in both arms (4.4-6.4 GB without mlock, 5.8-7.5 GB
  with it), so mlock added ~1.1-1.5 GB of swap, not the whole figure. The same trade llama.cpp makes.
- vs llama.cpp's one 1w block: pp512 326-390 vs 431, pp2048 326-401.5 vs 444 tok/s (different sessions; not a
  matched comparison); decode unchanged at 27.7-30.5 vs 36.6 tok/s. Suite blocks with mlock and the matched 8-bit attention row (`mj-a8`) are next.
- Decode remaining: the GPU step (35 ms) is 7 ms over its parts (28 ms) even with few page-ins; 8-bit attention
  saves ~4 ms of parts. Not yet explained.

### 1x. Split-K (flash-decoding) paged decode attention (2026-11-18 18:00-18:50)

**The problem (1w):** at batch 1 the paged decode kernel ran one threadgroup per (sequence, kv head). On
Qwen3.5-9B (4 kv heads) that is 4 threadgroups on a 10-core GPU, each walking the whole context.

**The change** (`paged_attention.py`):
- When a batch's longest row spans more than 256 positions, one threadgroup per (sequence, kv head, 256-position
  partition), then a reduce kernel.
- **Exactness:** a row's numbers depend only on its own length.
  - Rows within 256 positions take the single-pass math bit for bit.
  - Longer rows always take the partitioned math, with partitions aligned to the row's start; empty partitions add
    exact zeros.
  - So speculative verify rows equal plain decode at every length.
- **Tests** (`tests/mlx/test_paged_attention_splitk.py`): matches SDPA in float32 and bf16, windows 0 / 128 / 300,
  sinks on and off, gpt-oss and Qwen shapes; per-row results are bitwise independent of the batch. 178 MLX tests
  pass.

**Qwen3.5-9B decode, batch 1** (`results/splitk_decode/q9b.jsonl`; old / new alternated, 2 rounds x 3 reps):

| context | old tok/s | new tok/s | change | GPU mJ/token, old / new |
| --- | --- | --- | --- | --- |
| 0 | 21.74 +- 0.03 | 21.64 +- 0.11 | none (same kernel) | 464 / 459 |
| 2048 | 20.62 +- 0.05 | 21.02 +- 0.16 | +1.9% | 512 / 491 |
| 8192 | 17.22 +- 0.50 | 19.26 +- 0.25 | **+11.8%** | 508 / 477 |

**Partition size** (`ps_sweep.jsonl`): 64, 128, 256 and 512 are all within noise (21.0-21.25 tok/s at 2048,
19.3-19.4 at 8192), so it stays 256.

**What is left at depth 2048:** decode costs about 1.07 ms/token more than at depth 0, against mlx-lm's 0.68
(1w: 21.83 -> 21.51 tok/s; not the 0.3 said earlier today).
- **KV writes:** skipping them in a timing-only experiment (outputs wrong) gains about 0.7 ms/token at depth 0 and
  about 1.1 ms at 2048.
- **Why they cost:** each step's `kp[slots] = k` copies the whole layer pool. In a microbenchmark, both scatter and
  slice updates copied a 128 MB array (about 2.9 ms) almost every time; only occasionally was it updated in place
  (0.25 ms).
- **Likely cause, not verified:** with overlap, step i + 1's write is built while step i's attention still holds the
  old buffer, so MLX cannot update in place.
- **Next:** an in-place KV write, e.g. writing the new K / V inside the attention kernel's own output.
- **Caveat:** the suite's 9B rows (1w) predate split-K. Do not set this section's 21.02 against the suite's mlx-lm
  21.51: different sessions, and this one ran about 3% slower at depth 0. A suite rerun of the 9B will settle the
  gap.

### 1w. Benchmark suite v1, to BENCH_PROTOCOL (2026-11-18 10:45-17:35)

Setup: `tools/mlx_bench_suite.py` with `tools/mlx_bench_worker.py`; `results/bench_suite/v1/` (`summary.md`,
raw blocks, machine state per block).
- **Grid:** llama-bench-shaped micro tests. pp512 and pp2048 (prefill tok/s); tg128 at context 0 and at context
  2048 (decode tok/s = 1 / TPOT).
- **Prompts:** random tokens, a different seed per rep; EOS ignored.
- **Runs:** 5 timed reps per block after an untimed warm-up. Two interleaved rounds, the second reversed. 30 s
  cool-downs between blocks.
- **Exclusions:** a block is excluded when swap grew by more than 256 MB or the block failed.
- **llama.cpp:** llama-bench build 12359 (commit b1b770d90), `-fa 1 -ngl 99`.
- **Energy:** IOReport GPU rail, estimated. None for llama-bench, whose window includes model load.
- **Machine:** an openshell VM (2 GB) and the Claude apps were running throughout.

**Qwen3.5-9B** (all 6 blocks clean; n = 10 each). Ours and mlx-lm: mlx-community 4-bit g64; llama.cpp: Q4_K_M.

| test | ours | mlx-lm | llama.cpp |
| --- | --- | --- | --- |
| pp512 | 219.11 +- 1.10 (218.3-219.9) | **226.88 +- 0.64** (226.4-227.3) | 217.53 +- 2.35 |
| pp2048 | 222.74 +- 3.74 (220.1-225.4) | **230.72 +- 3.43** (228.3-233.2) | 201.25 +- 2.07 |
| tg128 | 21.64 +- 0.04 (21.61-21.67) | **21.83 +- 0.07** (21.78-21.88) | 18.85 +- 0.01 |
| tg128 at depth 2048 | 20.36 +- 0.46 (20.0-20.7) | **21.51 +- 0.02** (21.50-21.53) | 18.38 +- 0.01 |

- **Plain decoding (no MTP): ours trails mlx-lm on every test, with no CI overlap.**
  - Prefill: 3.4-3.5% behind.
  - Decode: 0.9% behind at depth 0, 5.3% behind at depth 2048.
  - GPU energy at depth 2048: 502 vs 476 mJ/token.
- **Against llama.cpp:** ahead by 11-15% on decode and 11% on pp2048; level at pp512.
- **Greedy ids, ours vs mlx-lm** on the same random prompts: 8 of 20 identical. Two differ at the first token; both
  are bf16 near-ties, and mlx-lm in float32 picks our token both times:
  - depth 0, seed 101: bf16 logits 12.25 vs 12.1875 (one bf16 step); float32 prefers ours, 314, by 0.087.
  - depth 2048, seed 201: an exact bf16 tie; float32 prefers ours, 248046, by 0.027.
  - Our precise mode rescores the head in float32. The later divergences (tokens 1-125) were not checked one by one.
  - Run file: `bench_suite/v1/tie_check.json` (script `tie_check.py`).
- **Energy field:** the SoC J/token field (`j_per_token_soc`) is invalid in 7 of 207 rows, where the CPU / DRAM rails
  returned bogus deltas (e.g. 1087 J CPU + 1232 J DRAM in 2.5 s). The tables use the GPU rail only. The worker now
  drops SoC figures over 60 W CPU or 30 W DRAM.
- **Diagnosis, depth-2048 decode:** our decode slows 2.1 ms/token from depth 0 to 2048, mlx-lm's only 0.3. The paged
  decode kernel runs one threadgroup per (sequence, kv head): 4 threadgroups on a 10-core GPU at batch 1. A split-K
  (flash-decoding) version is on branch `mlx-splitk-decode`, not yet tested.

**gpt-oss-20b.** Ours: exact weights, MXFP4 experts memory-mapped, `--wired-mb 3000`. llama.cpp: ggml-org MXFP4
GGUF.
- Our engine's weight files were read once before each of its blocks (llama.cpp's GGUF too, before its own).
- Even so, **3 of our 4 blocks grew swap by 1.7-2.6 GB and were excluded**; none of llama.cpp's did.
- The remaining block (r2; **one block, 5 reps, not the protocol's two interleaved rounds**):

| test | ours | llama.cpp (n = 10) |
| --- | --- | --- |
| pp512 | 69.7 +- 9.7 (CV 14%) | **431.3 +- 8.9** |
| pp2048 | 300.5 +- 27.1 | **444.4 +- 7.6** |
| tg128 | 27.3 +- 1.0 | **36.6 +- 0.5** |
| tg128 at depth 2048 | 26.4 +- 0.7 | **35.0 +- 1.3** |

- **Behind on every test:** 6.2x on pp512, 1.5x on pp2048, about 25% on decode.
- 1q measured ours at about 360 tok/s on a 512-token prompt when the page cache was warm and swap was low. Here our
  own memory use drives the machine into swap, and prefill pages experts back in.
- **The largest gap to SOTA in this suite.** It is a memory-footprint problem first.

**Qwen3.6-35B-A3B** (20.4 GB, bigger than RAM). Ours: experts streamed, 128 slots, C++ streamer. llama.cpp: Q4_K_M,
`-ncmoe 40`, 3 reps per block.
- **Robustness:** 3 of our 4 blocks failed with Metal "Insufficient Memory" during pp2048 prefill at 128 slots. The
  10.5 GB slab plus prefill activations is too close to the limit under normal background load. 128 slots is not a
  safe default on 16 GB; 96 (or slots chosen from free memory) is.
- The remaining block (r1; **one block, 5 reps, not two rounds**; at a slot count that failed 3 of 4 times, so a
  96-slot rerun is needed for a figure a user could reproduce):

| test | ours | llama.cpp (n = 6) |
| --- | --- | --- |
| pp512 | **195.7 +- 5.7** | 16.1 +- 0.4 |
| pp2048 | **295.5 +- 3.4** | 13.0 +- 1.1 |
| tg128 | **25.95 +- 2.09** | 3.85 +- 0.62 |
| tg128 at depth 2048 | **21.97 +- 0.59** | 3.49 +- 0.47 |

- Ours is 6.3-23x llama.cpp.

**Standing against SOTA on this Mac, from this suite:**
- Ahead: the model bigger than RAM, by far; 9B decode and long prefill vs llama.cpp.
- Behind: 9B plain decode and prefill vs mlx-lm (1-5%); gpt-oss everything vs llama.cpp (25% to 6x).
- Not covered by this grid: MTP and EAGLE decoding, chat prompts, batching, wall-power energy.

### 1v. MTP loop pipelining, Metal sync probes (2026-11-18 09:40-11:00, quiet machine)

**Same-prompt baseline first** (`results/mtp_overlap/base_k*.json`; chat prompt, 128 new tokens, 96 slots, C++
streaming + keep-alive, one warm-up and two measured runs, decode first to last token):

| path | decode tok/s |
| --- | --- |
| engine, overlap mode | 17.74 / 17.74 |
| MTP loop, k = 0 (already pipelined like overlap mode) | 16.81 / 16.80 |
| MTP loop, k = 2 | **19.01 / 18.86** (acceptance 0.91) |

- All tokens are identical across the three.
- 1u's "MTP is slower than the best path" compared different prompts. The 20.9 tok/s best path is the 512-token
  random prompt. On chat text the engine path is 17.7 and MTP k = 2 already beats it by 7%.

**Pipelined verify** (`SpecGenerator(pipeline=...)`, `MJ_SPEC_PIPELINE`): while the GPU runs verify step i, build step
i + 1 assuming every draft is accepted.
- **eager:** submit step i + 1 at once. A rejection drops its results; the GPU still runs it.
- **defer:** submit it only after the acceptance test. A rejection restores the KV / conv / DeltaNet / MTP pools from
  a snapshot and drops the graph unrun.
  - The pools are written by `__setitem__`, which rebinds the array object in place. So the snapshot keeps a second
    handle per array (`copy.copy`, same graph node, no copy).
  - The C++ streamer numbers its events at encode time, so an unsubmitted graph leaves no hole in the sequence.

Results, two A/B rounds each, tok/s (`r*_k2_*.json`, `q9b_r*_*.json`):

| model | off (original loop) | eager | defer |
| --- | --- | --- | --- |
| Qwen3.6-35B streamed, k = 2 | 19.69-19.90 | 18.33-18.70 | 19.60 / 19.80, then 17.41 / 14.14 |
| Qwen3.5-9B in memory, k = 2, 4 chat prompts (ms/token) | 25.07-29.54 | 29.77-42.26 | 24.91-29.94 |

- **Exactness:** tokens are identical in every arm. 27 spec-decode tests pass in all three modes (including oracle
  drafts with every 2nd or 3rd corrupted).
- **9B eager:** 19-43% slower than off, per prompt and round.
- **Run-to-run drift:** `base_k2` (18.86-19.01) and `r*_k2_off` (19.69-19.90) are the same configuration, taken
  minutes apart, and differ by 4-5%. Compare only arms from the same back-to-back block: MTP k = 2 is +7% over the
  engine path (`base_k*`), not 19.8 against 17.74.
- **Eager loses:** one rejected step costs a whole verify step of GPU time (35B: about 140 ms; 6 of 44 discarded).
  The host time it hides is about 10 ms per step, so eager is net slower.
- **Defer gains nothing:** the Python graph build is not on the critical path. Defer's round-2 slowdown was on the
  SSD side: IO read wait 6.3 s vs 4.0 s per run, with misses identical. Not explained.
- **Default:** `off`. Both modes stay, with tests.

**Where the MTP loop's time goes (35B, 96 slots):** the IO thread waits on SSD reads 4.0 s of each 6.4 s, 128-token
run. That is about 17 demand misses per token, read one layer at a time on the critical path. The loop is
SSD-latency-bound, not host-bound.

**Metal sync probes** (`results/metal_sync_probe/`, M4).

*Visibility of CPU writes* (`vis.mm`):
- Setup: one command buffer of 2,000 short dispatches, each reading a shared flag. The CPU writes the flag about 5 ms
  after commit.
- No dispatch saw the write, with any of: one encoder; one encoder per dispatch; memory barriers between dispatches;
  write-combined memory; volatile or atomic loads.
- With 60 separate command buffers, all committed before the write, the first one to start after the write saw it.
- So CPU writes become visible to the GPU at command-buffer boundaries only. A chain of short polling command buffers
  could stand in for a CPU-to-GPU event; nothing finer works.

*Round-trip cost* (`rt.mm`): 40 layers per command buffer, each with a kernel, an encoder break and optionally a shared
event signal plus wait. µs per layer:

| short kernel (2000 iters) | long kernel (20000 iters) | per-layer pattern |
| --- | --- | --- |
| 45.8 | 259.1 | encoder break only |
| 46.9 | 260.0 | + signal + wait, wait already satisfied |
| 151.8 | 354.4 | + real round trip, CPU spinning on `signaledValue` |
| 204.0 | 433.2 | + real round trip, CPU `waitUntilSignaledValue` |

- A satisfied wait is free.
- A real GPU-to-CPU-to-GPU round trip costs about 95-105 µs even with a spinning CPU, and about 160 µs blocking.
- The streamer's measured idle, 5.8 ms per step over about 40 round trips (about 145 µs each), is near this floor.
  At most about 2 ms per step is left there.
- Skipping the round trip on all-hit layers would need a conditional GPU wait. Metal has none, and running kernels
  cannot see CPU writes. So with exact outputs, SSD misses (about 12 ms per step) are the bigger lever.

**Expert-cache policy headroom** (`tools/mlx_expert_cache_sim.py`, `results/expert_cache_sim/qwen36_trace.json`).
- Setup: offline, on the recorded routing of 8 documents, 8,192 teacher-forced tokens, all 40 layers. Demand misses
  only (no prefetch). Each token's own experts are never evicted for each other.
- Miss rate per expert access:

| policy | 96 slots | 128 slots |
| --- | --- | --- |
| LRU (the streamer today) | 11.1% | 6.8% |
| LFU, pure | 17.6% | 11.2% |
| LFU, decay 0.99 per token | 10.7% | 6.5% |
| LFU, decay 0.98 per token | 10.3% | 6.4% |
| Belady optimum (knows the future) | 5.2% | 3.0% |

- **Eviction policy:** worth at most about 7% fewer misses over LRU. That is not a lever, so LRU stays.
- **Where the headroom is:** the 2x down to the optimum comes only from knowing future routing, i.e. prediction and
  prefetch. Candidates from the literature review: prefetch the verify step's experts from the draft tokens
  (SP-MoE, MoE-SpeQ), and draft lengths chosen by the new experts they pull in (DraftExpert; exact, 1.45x reported).

### 1t. Per-op decode-step trace, Qwen3.5-9B (2026-11-18 06:25, during a 72 GB download)

`tools/mlx_step_trace.py` (method in its docstring; `results/step_trace_qwen35_9b/`):
- **Normal steps:** the engine's own decode step under Metal System Trace (per command buffer; Metal reports no
  per-dispatch times).
- **Split steps:** `mx.async_eval` at every matmul's input and output, so each op group gets its own command buffer.
  Buffers are matched to groups by an alignment that allows MLX's occasional merges (3-7 of about 262 per step).
- **A first version synced after every group.** That added up to 150 ms of GPU time per step vs 50 ms, because the
  GPU drops its clocks between syncs.
- **Distortion of the async split** (overlap on): its GPU span is 51.6-52.4 ms per step.
  - Against the same untouched step under tracing (49.5-51.7 ms), it adds up to about 2-6%.
  - Against the untraced step (47.7-47.9 ms), it adds about 8-10%; part of that is the tracing itself.

| setting | GPU busy over 5 steps | step time untraced / traced |
| --- | --- | --- |
| engine overlap on (serving default) | **99.9%** | 47.7-47.9 / 49.5-51.7 ms (first step 36.0 / 35.5) |
| overlap off | 92.3% (about 4 ms host gap per step) | 54.6-57.7 / 53.6-62.4 ms |

GPU time per step by op group (async split, mean of 3 steps), with weight bytes:
- MLP up/gate + activation: 19.7 ms, 1.81 GB (92 GB/s);
- in_proj: 10.0 ms, 0.87 GB;
- MLP down: 9.7 ms, 0.91 GB (94 GB/s);
- lm_head + sampling: 5.6 ms, 0.57 GB;
- out_proj: 3.8 ms;
- DeltaNet conv + recurrence: 1.5 ms;
- full attention: 0.55 ms;
- norms and other small ops: 0.55 ms.

The 9B decode step has no waiting left: the GPU is busy the whole step, and every matmul runs near the memory
bandwidth. Faster single-request decode here needs fewer bytes per token (MTP, already 2x) or fewer weight bytes. The
waiting that the timeline chart is meant to show is on the streamed 35B (1s), not here.

**Streamed Qwen3.6-35B, labelled** (`results/step_trace_qwen36_35b_stream/r0_during_download.json`).
Preliminary: taken while a 72 GB download used the SSD; a quiet rerun follows.
- Settings: overlap off, as the stream tool runs it; 96 slots; chat prompt.
- MoE groups: router (including the next-layer prediction that drives prefetch), expert_gather (the slab experts),
  shared_expert, and run_moe's own routing readback.
- Mapping: 20-40 of about 640 buffers per step unmatched; fit cost 0.54-0.72 per buffer.
- Normal steps: GPU busy 50% over 5 steps (41-70% per step). Untraced steps 61.8-75.3 ms.
- One miss layer, as recorded (layer 0 of a split step):

  | time | what happens |
  | --- | --- |
  | 1.46 ms | speculative 8-expert pass (0.36 ms), later discarded |
  | 1.82 ms | routing readback |
  | 2.78 ms | the 3 resident experts are computed |
  | 2.96-6.98 ms | the GPU idles about 4 ms waiting for the SSD reads of the other 5 |
  | 6.98 ms | the 5 read experts are computed (0.44 ms) |

  The layer's other work (attention, router, shared expert) is about 1.5 ms.
- In the same trace, a 14 MB in_proj takes 0.41-0.59 ms (24-34 GB/s), against about 90 GB/s in the 9B. That fits
  the GPU lowering its clocks during the waits, as with the sync split, but this is not separated yet.

### 1e. Prefill (time to first token), one request, warm

`tools/mlx_ttft.py`, 8-layer real-dimension bench models, median of 3 (`results/ttft/`):

| model | prompt | mlx-lm 0.32.2 | ours before | ours now |
| --- | --- | --- | --- | --- |
| Qwen3.6 | 128 / 512 / 2048 | 167 / 283 / 904 ms | 142 / 349 / 1276 ms | **134** / 309 / 1066 ms |
| gpt-oss | 128 / 512 / 2048 | 194 / 480 / 1737 ms | 163 / 506 / 2016 ms | **164** / 510 / 1991 ms |

Changes: a packed gated-delta kernel (4 lanes per value column, q/k/v read in the model dtype, L2 norm of q/k in float32
inside the kernel; 6.4 ms vs mlx-lm 7.2 ms per layer at 2048 tokens), MLX's depthwise `conv1d` for prefill chunks, and
MLX's fused `mask="causal"` attention for single-item prefill on full-attention layers (checked equal to the explicit
bottom-right mask; 2x faster at 2048). Per-layer gpt-oss MoE and attention now match mlx-lm (185 vs 180 ms, 51 vs 49
ms at 2048) yet the full prefill is still 10-15% slower; not explained yet.

## 2. Accuracy

`tools/mlx_accuracy.py`, teacher-forced logits at every position of random-token prompts.

Tiny model (`trl-internal-testing/tiny-GptOssForCausalLM`), reference = transformers 5.23 float32, 8 x 256
positions (`results/accuracy_tiny_vs_hf.json`). Logits here are tiny (|max| 0.08), so top-1 mostly
measures near-ties; KL and max |dlogit| are the signal.

| config | mean KL | max abs dlogit | top-1 |
| --- | --- | --- | --- |
| MilliJoule float32 | 1.0e-17 | 7.5e-8 | 100% |
| mlx-lm float32 | 8.8e-10 | 1.1e-3 | 99.5% |
| MilliJoule bf16 (precise, default) | 3.2e-9 | 8.8e-4 | 96.3% |
| MilliJoule bf16 (plain) | 4.2e-9 | 1.4e-3 | 96.6% |
| mlx-lm bf16 | 5.6e-9 | 1.6e-3 | 96.3% |

The mlx-lm float32 gap comes from its YaRN RoPE ignoring gpt-oss's `rope_scaling.truncate: false` (transformers
honours it; `yarn_inv_freq` in `gpt_oss.py` does too).

Real dims, 2 layers (`--layers 2` bench model), reference = MilliJoule float32, 4 x 256 positions
(`results/accuracy_realdims_2L.json`):

| config | mean KL | top-1 |
| --- | --- | --- |
| MilliJoule bf16 precise + fp32 rescore (default) | 0.0115 | 81.5% |
| MilliJoule bf16 plain | 0.0144 | 78.6% |
| mlx-lm bf16 | 0.0193 | 76.2% |

Ablation (same setup): float32 attention compute (KV still bf16) would give KL 0.0088 / top-1 84.7%, but needs a
mixed-precision GEMM to avoid doubling attention weight bytes; not adopted. float32 MoE activations: no change.

### lm_head candidates (greedy only)

`tools/mlx_lm_head_miss.py`, real dims 2L, 8 x 512 = 4096 teacher-forced rows
(`results/lm_head_miss_2L.json`). Miss = the exact argmax (bf16 weights, fp32 math) is not in the low-bit
head's top-k.

| head | top-1 miss | top-4 miss | top-16 miss |
| --- | --- | --- | --- |
| 8-bit (default candidates) | 3.5% | 0% | 0% |
| 4-bit | 22.1% | 1.2% | 0% |

So the fp32 rescore is required (8-bit top-1 alone would change 3.5% of greedy tokens), and top-16 had no miss in
4096 rows. This is a sample on random weights, not a proof: it still goes through the normal accuracy gate on real
weights, and it is greedy-only (sampling rows must use the full head).

## 2a. Accuracy on the REAL openai/gpt-oss-20b weights

Snapshot `6cee5e81ee83917806bbde320786a8fb61efebee` (released MXFP4 experts, bf16 rest; downloaded with the user's
OK). The full model (13.7 GB) does not fit the Mac's GPU working set, so these runs stream it one layer at a time;
teacher-forced only (no free-running generation).

**Reference check, one real layer vs transformers float32** (`tools/mlx_real_layer_check.py`,
`results/real_layer_check.json`): embedding + real layer i + final norm, 276 tokens of chat (longer than the
128 window), transformers 5.23 eager float32 with its own MXFP4 unpacking.

| layer | type | MilliJoule fp32 rel. RMS err | mlx-lm 0.32.2 fp32 rel. RMS err | max abs (ours / mlx-lm), max abs ref |
| --- | --- | --- | --- | --- |
| 0 | sliding | 5.6e-7 | 2.8e-3 | 2.9e-4 / 1.56, 313 |
| 1 | full | 5.7e-7 | 3.7e-3 | 2.9e-4 / 6.79, 350 |

So on real weights our float32 path equals transformers to float32 rounding, and is the reference below.

**Full 24-layer model, bf16 configurations vs that reference** (`tools/mlx_real_eval.py`,
`results/real_gpt_oss_20b.json`): 8 chat conversations written for this test, formatted with the model's own
tokenizer and chat template, 1296 teacher-forced positions.

| config | top-1 = reference | mean KL | p99 KL |
| --- | --- | --- | --- |
| MilliJoule bf16 precise, engine greedy path (low-bit candidates + fp32 rescore) | 97.2% | - | - |
| MilliJoule bf16 precise, full head | 96.9% | 0.0157 | 0.225 |
| mlx-lm 0.32.2 bf16 (its layers, loader quantization, RoPE) | 95.4% | 0.0221 | 0.537 |
| MilliJoule bf16 plain | 94.1% | 0.0266 | 0.930 |
| mlx-lm 0.32.2 float32 | 98.1% | 0.0056 | 0.075 |

Reading: on the released weights our default bf16 engine is closer to the float32 model than mlx-lm's bf16: KL
-29% (0.0157 vs 0.0221), p99 KL -58%, and greedy top-1 97.2% vs 95.4% (+1.8 points; 23 of 1296 positions). The
low-bit-candidate greedy path agrees with the float32 argmax slightly more often than the full bf16 head
(97.2% vs 96.9%), i.e. it did not lose any accuracy here. Precise mode is what makes the difference: our plain
bf16 is worse than mlx-lm's. mlx-lm in float32 still disagrees with the transformers-equivalent float32 model on
1.9% of positions; the one-layer check above shows its float32 layer math itself differs (YaRN `truncate`). This is one
1296-position sample on hand-written chats, not the plan's lm-eval / task-accuracy gate, which still has to run
on the CUDA side.

## 2b. Qwen3.6 hybrid MoE: correctness (tiny checkpoint only)

`trl-internal-testing/tiny-Qwen3_5MoeForConditionalGeneration-3.6`, float32, vs transformers 5.23 float32
(`tests/mlx/test_qwen3_5_moe.py`):

| check | result |
| --- | --- |
| prefill logits, one 150-token item | max abs dlogit 2.2e-7 (max abs logit 0.46), argmax 100% |
| same prompt as chunks 64 / 50 / 36 (state carried across chunks) | 2.6e-7, 100% |
| 20 single-token steps then a 130-token chunk | 2.7e-7, 100% |
| batch-1 greedy, 4 prompts (5-200 tokens, prefill chunked at 64) | token-identical |
| 12 sequences batched, max_running 6, tight pool (15 preemptions) vs roomy pool | identical; both equal transformers, 0 near-tie excuses |
| sampling (T 0.9, top-p 0.95, top-k 50) tight vs roomy (preemptions > 0) | identical |
| Metal gated-delta kernel + vectorized conv vs per-token reference path | identical tokens; kernel vs numpy recurrence rtol 1e-4 |

**Recurrent-state storage dtype** (`tools/mlx_state_dtype.py`, `results/qwen_state_dtype_tiny.json`): float32
model, only the stored delta-rule state differs (bf16 vs float32; math float32), 4 x 400 teacher-forced random
tokens. Per-slot state on the tiny model: 2.1 MB float32 / 1.1 MB bf16 (one linear layer).

| fed as | top-1 agreement | mean KL | max KL | logit error / logit RMS |
| --- | --- | --- | --- | --- |
| 1 token per step (state rounded every token, decode-like) | 100% | 4.3e-14 | 3.2e-13 | 3.7e-6 |
| 64-token chunks (rounded once per chunk) | 100% | 6.2e-16 | 6.8e-14 | 4.3e-7 |

This is a **lower bound**: the state shapes are real (Hv 32, Dk = Dv = 128) but the tiny model's hidden size is 16
and its weights are random, so the linear-attention branch contributes little to the logits. Whether bf16 state
passes the accuracy gate has to be measured on real Qwen3.6 weights (not downloaded; needs the user's OK).

No Qwen3.6 speed, energy or real-weight numbers yet (real weights need their own download OK).

## 2c. Dense Qwen3.5 (Qwen3.5-9B, mlx-community 4-bit): correctness (2026-11-17)

Dense Qwen3.5 uses Qwen3.6's attention and DeltaNet layers with a plain SwiGLU MLP; `Qwen35Moe` treats
`num_experts` absent/0 as that MLP and `load_model` accepts model_type `qwen3_5` / `qwen3_5_text`. The checkpoint was
already in the HF cache (since 2026-09-29).

Against mlx-lm on the same quantized weights (`tools/mlx_vs_mlxlm.py`, 3 chat texts written for the test, 174
teacher-forced positions, reference = mlx-lm with float32 activations; `results/check_qwen35_9b_vs_mlxlm.json`):

| config | logits rel. RMS vs ref | top-1 vs ref | mean KL | max KL |
| --- | --- | --- | --- | --- |
| MilliJoule float32 activations | 2.8e-6 | 100% | 4e-11 | 1.2e-9 |
| MilliJoule bf16 (precise, engine default) | 7.1e-3 | 99.4% | 2.7e-4 | 5.7e-3 |
| mlx-lm bf16 (its default) | 1.15e-2 | 98.9% | 6.2e-4 | 3.2e-2 |

The float32 row says the two implementations compute the same function; the bf16 rows say ours stays ~2.3x closer
(mean KL) to float32 than mlx-lm's bf16. A transformers float32 check of the dense MLP path on a tiny random dense
Qwen3.5 built locally is in `tests/mlx/test_qwen3_5_dense.py`.

## 3. Measurement rules learned here (HANDOFF section 9 style)

1. **MLX buffer cache on a 16 GB Mac.**
   - Without a cap, freed buffers from variable-size prefill temporaries stay cached and wired. The process grew
     to 9.3 GB (5.7 GB of weights), the system hit swap, and level-32 throughput fell from ~190 to 78-118 tok/s
     with the GPU at 6-9 W (idle-ish), steps of 1-9 s.
   - Fix: `mx.set_cache_limit(1 GB)` and `mx.clear_cache()` after every step that ran a prefill chunk (mlx-lm
     clears after every prefill chunk too). Also wire weights + KV with `mx.set_wired_limit` (mlx-lm does).
   - Size the KV pool to the workload, not to "all free memory": 4 GiB of pool on top of 5.7 GB of weights made
     a decode step 15x slower (1.38 s vs 92 ms) from paging.
2. **Isolated per-op timings mislead on MLX.** One layer timed alone includes a sync per call (~0.3-0.5 ms) and
   showed ~31 GB/s for experts at B=1, while the whole B=1 step ran at ~91 GB/s effective. Judge kernels in
   the full step.
3. **IOReport without sudo** exposes only the "GPU Energy" rail (nJ) on M4; CPU, DRAM and SoC rails read 0. Idle
   GPU read between 0.0 and 1.2 W across runs.
4. **Run-to-run spread on a shared laptop is 20-45%**, much more than the plan's 5% rule. Part is memory pressure
   from other apps; most is a machine-wide performance state that moves both engines together (v4: idle GPU
   0.9-1.3 W in the slow state, 0.00 W in the fast one; rounds in the same state agree to ~2%). Only paired,
   same-round ratios are meaningful; record idle power per run to tell the states apart.
5. **Don't rebuild the shared venv during a benchmark.** A `maturin develop` mid-run briefly broke
   `millijoule.as_numpy` for new processes, and its compile load can skew a timing arm.

## 4. Baseline tuning (mlx-lm prefill knobs)

`tools/mlx_bench.py --engine mlx-lm --levels 32,64 --prefill-batch PB --prefill-step PS`, one run each, same
workload (`results/tune_mlx_lm_*.json`). completion_batch_size is the level by construction.

| prefill_batch_size | prefill_step_size | L32 tok/s | L32 mJ/token | L64 tok/s | L64 mJ/token |
| --- | --- | --- | --- | --- | --- |
| 8 (default) | 2048 (default) | 215.9 | 82.6 | 261.7 | 65.9 |
| 32 | 2048 | 180.8 | 95.5 | 264.3 | 65.4 |
| 32 | 512 | 179.9 | 95.9 | 257.1 | 67.3 |
| 4 | 2048 | 171.7 | 98.1 | 252.6 | 65.4 |

No setting beats the defaults beyond noise, so the ABAB baseline (defaults) is mlx-lm's best measured
configuration here. The default's own level-32 result ranged 174.5-215.9 tok/s across runs (A/A spread ~21%).

## 5. What moved the numbers (our side, in order)

1. Paged decode attention kernel: attention per layer at B=64, ctx ~400: 8.1 -> 3.9 ms (full), 4.6 -> 2.9 ms
   (sliding) vs gather + SDPA (single profile runs).
2. Buffer-cache cap + clear after prefill (section 3.1): level 32 from 78-118 to ~190 tok/s.
3. Low-bit lm_head candidates + fp32 rescore: head matmul 21.4 -> 11.0 ms at B=8 (8-bit), 12.7 -> 3.7 ms at 1
   row (4-bit); bf16 kept above 48 rows, where quantized matmul is slower (20.3 vs 22.6 ms at 64 rows).
4. Exact two-stage top-k for the candidates: 26.1 -> 1.25 ms at 64 rows; B=64 decode step 172 -> 113 ms,
   B=32 120 -> 108 ms (profile runs). This is most of the v3 -> v4 gain at levels 32-64.
5. Precise mode is for accuracy, not speed (fp32 residual and router; cost not separately measured).
6. Tried and dropped: grouped MXFP4 GEMV kernel (correct, but 1.7 -> 3.5 ms at B=1 and 9.0 -> 24.2 ms at B=64 vs
   `gather_qmm`; faster only near B=8); a padded [E, C, H] capacity layout and a per-expert quantized-matmul loop
   (both slower than sorted `gather_qmm` at B = 8-128). MoE is now ~70% of a B=64 step and is at the limit of
   MLX's built-in ops; beating it needs a better grouped MXFP4 GEMM kernel.
7. Speculative expert dispatch for streamed MoE (section 1d.3): gpt-oss-20b streamed 12 -> 17.2 tok/s, Qwen3.6-35B
   6.3 -> 9.5 tok/s at 96 slots, bit-identical.
8. Sampling (temperature / top-k / top-p) on device, reproducible under preemption; not part of the greedy
   benchmark numbers.

## 6. For the B200 / CUDA work

- The YaRN `truncate: false` detail matters for gpt-oss: an engine that floors/ceils the correction range drifts
  from the reference (mlx-lm float32: 1e-3 logit error, 0.5% greedy flips on the tiny model).
- A float32 router and residual stream cut bf16 KL by ~40% at no weight-byte cost; cheap to carry to CUDA.
  Full list of what "precise" keeps in float32 on MLX: the residual stream; RMSNorm statistics (all layers and
  the final norm); router logits and the top-k softmax; the attention softmax with the sink logit added in
  float32 (MLX SDPA and our paged kernel both do this; sinks are stored bf16 and upcast); the lm_head candidate
  rescore. Weights, KV, q/k/v and expert activations stay bf16. Measured but not adopted: float32 attention
  compute (q/k/v projection outputs) cut KL a further ~23% on random weights, at the cost of fp32 weight bytes
  without a mixed-precision GEMM; on CUDA a bf16-in / fp32-out GEMM for q/k/v may get that for free.
- Low-bit lm_head candidates + exact rescore of the top-16 is a candidate for the output layer (P2-9 analog), to
  be gated on real weights like any change (PM decision).
- Decode attention reading KV through the block table in one kernel per (sequence, kv head), all GQA heads
  together, is the shape the CUDA decode kernel should have too.
