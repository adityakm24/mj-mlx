# Running MoE models bigger than memory on a 16 GB Mac: what the open-source engines do

2026-11-16. Source read (shallow clones, nothing built or run) of: Colibri (JustVugg/colibri, 0a5504a), Flash-MoE
(danveloper/flash-moe, 3601d41), streamlx (srcterm/streamlx, 9ab2cca), deepseek-v4-flash-mlx (ssd-moe, a374ff9),
mlx-moe-streaming (ryohji, 27c94e6), mlx-flash (szibis, f5cb3e8), AirLLM (lyogavin/airllm, fb1e0b8),
mixtral-offloading (dvmazur, ce54518). All speed and hit-rate numbers below are the authors' own claims unless marked
as measured here.

Licences of the projects read (clones kept outside the repo; ideas only, no code copied into MilliJoule): Colibri
Apache-2.0; AirLLM Apache-2.0; streamlx MIT (c) 2026 Ian Hubbard; mlx-moe-streaming MIT (c) 2026 Ryohji Ikebe;
deepseek-v4-flash-mlx MIT (c) 2026 its contributors; mlx-flash MIT (c) 2024-2026 Slawomir Skowron; mixtral-offloading
MIT (c) 2023 Artyom Eliseev, Denis Mazur; Flash-MoE: no licence file in the repository (nothing taken beyond ideas).

## The one idea that matters: keep the trunk resident, stream only the routed experts

Every engine that is fast on a small Mac does the same thing: attention / DeltaNet, router, shared expert, norms
and the output head stay in memory; the routed experts live on the SSD and are read when the router picks them,
into a bounded per-layer cache. Only the experts a token routes to are read, so bytes per token = active-expert
bytes x miss rate, not model size.

AirLLM is the counter-example: it re-reads every layer from disk for every token. Its Mac (MLX) path also skips
prefetch, quantization and MoE handling, and re-`mx.load`s each layer per token. For a 4-bit 70B model at this
Mac's SSD speed that is ~0.15 tok/s. Not worth porting.

## What each engine contributes

| idea | who | evidence (authors) | fit for us |
| --- | --- | --- | --- |
| Per-layer slot slab `[S, out, in]` + id->slot remap + `gather_qmm` over slots; misses `preadv`'d straight into the evaluated MLX array's memory (writable `memoryview`), only slots no queued kernel reads are rewritten | streamlx | Qwen3.6-35B 4-bit on a 32 GB M4 Air (2.4 GB/s SSD): 7.2 tok/s at 8 GiB expert budget (8.3% miss), 15.9 at 16 GiB; bitwise-identical to stock | **Best match for our gather_qmm design.** MXFP4 = weight + uint8 scales, no biases |
| Bitwise identity: keep S <= E and copy the stock sorted/unsorted kernel choice; per-row outputs are independent, so hit/miss split + inverse permutation is exact | streamlx | `array_equal` logits vs stock with tiny pools | Adopt as the gate |
| Repack experts into per-layer files, fixed stride, gate/up/down/scales contiguous, aligned; one `pread` per expert into a 2 MB-aligned buffer wrapped zero-copy | Flash-MoE, ryohji | Flash-MoE: mmap 5x slower than pread (16 KB page faults); 2 MB alignment 3.6x faster DMA; ryohji: defrag +20-28%, zero-copy +20% | Adopt; small offline tool |
| No custom cache at all, use the OS page cache | Flash-MoE | their Metal LRU at 55% hit: 2.24 tok/s vs 5.74 with page cache only (wired GPU memory shrinks the page cache, compressor churns) | **Decisive on 16 GB**: keep our slabs small, measure cap vs tok/s; in Colibri a smaller cap also beat a bigger one (OLMoE cap 32 74% hit 19.1 tok/s > cap 64 97% hit 16.3) |
| Next-layer router run on the current residual to prefetch experts | Colibri (PILOT, off by default), ryohji, mixtral-offloading | ryohji: recall 76-92% one layer ahead; Flash-MoE: every predictor lost (SSD DMA competes with the GPU for memory bandwidth) | Measure on our models before relying on it |
| Dispatch hit experts first (`async_eval`), read misses meanwhile | streamlx, ryohji | each routing sync ~1.5 ms/layer (ryohji) | Needed: our step is one lazy graph today; streaming forces a host sync per layer to learn routing |
| Pin frequent experts (warm-up or persisted routing history) | Colibri, streamlx, ssd-moe | streamlx warm start: 15.9 -> 19.0 tok/s | Cheap later win |
| Prefill expert-major (sweep each expert once over all tokens) | streamlx, ryohji | | Needed for long prompts |
| Low-bit experts (2-bit) | Flash-MoE, mixtral-offloading | Flash-MoE 2-bit +32% but broke JSON/tool output | Accuracy-gated only |

Avoid: silently skipping or aliasing uncached experts (mlx-flash maps them to slot 0 and its "bit parity" test does
not test streaming), filling slots with `mx` slice-assign or `mx.stack` (copies), large batches when streaming (the
union of routed experts grows with batch, so the miss count grows), fancy eviction policies (ARC / S3-FIFO within
+-1 point of LRU in ryohji's tests).

## Numbers for this Mac

Measured here: uncached SSD reads (`pread`, `F_NOCACHE`, random offsets) on the real gpt-oss-20b shards: 13.2 MB
reads 2.6-3.3 GB/s, 1.8 MB reads 2.0-3.9 GB/s, best at 8 threads. The Mac was swapping 7-18 GB during all our work,
which shares the same SSD.

Estimates (arithmetic, not runs):

| model | experts on disk | resident trunk | expert budget that fits | reads per token at batch 1 | rough decode |
| --- | --- | --- | --- | --- | --- |
| gpt-oss-20b (real, downloaded) | 10.1 GB (24 x 32 x 13.2 MB) | ~2.4 GB (attention bf16 1.3 GB, 4-bit candidate head, embed/head rows from mmap) | ~5-7 GB (50-70% of experts) | 96 experts; at 30% miss ~380 MB | ~5-8 tok/s |
| Qwen3.6-35B-A3B 4-bit (not downloaded) | ~18 GB (40 x 256 x 1.77 MB) | ~2.5 GB | ~5-7 GB (~35%) | 320 experts; streamlx saw 8% miss at 8 GiB | a few tok/s |

## Plan for our engine (gpt-oss-20b first, since the real weights are already here)

1. Repack tool: per-layer expert file, fixed stride, [gate_up blocks | gate_up scales | down blocks | down scales]
   per expert, 2 MB aligned.
2. `ExpertSlab` per layer: evaluated `[S, ...]` MLX arrays for weights and scales, `slot_table` remap, `preadv` into
   the slab's memory, a reader pool (8 threads), slots in use by queued kernels never rewritten.
3. Decode with a host sync per layer for routing; hit experts dispatched first, misses read meanwhile; next-layer
   prefetch from the next router on the current residual, measured on and off.
4. Gate: bitwise-identical logits against the in-memory engine on a model that fits (the 8-layer bench model, with
   S < E to force misses), then real gpt-oss-20b greedy against the layer-streamed float32 reference.
5. Sweep the expert budget against tok/s and against the macOS page cache; record GPU slowdown under concurrent reads.

Untested assumption to check first: writing into an evaluated MLX array through `memoryview` is not safe if a
queued kernel still reads that slot. streamlx guards this with its own in-flight tracking; we would need the same.
