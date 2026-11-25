# Prior art: speculative expert dispatch (2026-11-17)

**Our technique** (`expert_stream.run_moe`, flag `SPECULATE`): in SSD-streamed MoE decode (tokens x k < 64), queue the
layer's expert computation on the GPU *before* the routing reaches the host. Expert id -> slot is looked up on the
GPU from a snapshot of the slot table, with pending reads counted as absent. The host then checks the routing. If every
routed expert was resident in the snapshot, the result is kept, bit-identical to the normal path. Otherwise it is
discarded and the normal miss path runs. Measured: 1.42-1.53x streamed decode, identical tokens (RESULTS 1d.3).

**Verdict: no exact match found.**
- Pieces exist separately: a GPU-side expert->slot indirection, and computing on predicted or resident experts.
- Nothing found combines them the way we do: run the exact expert compute on a slot-table snapshot, verify on the
  host, and discard on a miss. Nothing found does it bit-exactly with SSD-backed misses either.
- This is a search result, not proof. Patents were not searched, and SeqMoE and CommitMoE were read from abstracts only.

## Closest work (links checked to exist; titles from GitHub / arXiv)

| Work | What it does | vs ours |
|---|---|---|
| llama.cpp PR #27861 (open, 2026-08), "GPU-resident LRU cache for host-offloaded MoE expert weights" | Device copy of the slot table; misses go to a dummy slot and the CPU computes them | GPU indirection, exact by splitting the work, not by speculating; needs the weights in host RAM |
| ONNX Runtime PR #33976 (merged 2026-11) | Hybrid GPU/CPU expert offload; the GPU skips cold tokens, the CPU computes them | Same split; no discard |
| xinfer PR #502 (merged 2026-10) | Device LRU plus a gather kernel that fetches misses from pinned host memory, no host sync | Misses resolved on the device; not possible from SSD |
| vLLM RFC #63666 (2026-10) | Tiered expert offload with lookahead scheduling | Its iGPU/SSD path keeps one top-k readback per layer: the gap ours closes |
| llama.cpp issue #20757 | A Metal two-tier cache prototype lost ~half its speed to the per-layer sync (per the subagent's reading of a comment) | Problem statement / motivation |
| streamlx (MLX, SSD streaming) | Resident experts start (`async_eval`) while misses load, *after* a routing sync. Hides the sync across MTP-drafted tokens staggered one layer apart, with exact acceptance | Still syncs on routing per layer; needs an MTP head. Different mechanism, also exact |
| slotstream (MLX + Swift, SSD streaming) | Cuts syncs by draining every few layers with forecasts (per the subagent) | Prefetch-only |
| mlx-moe | Fixed resident expert set, no sync | Lossy below capacity |
| Speculating Experts (arXiv 2603.19289), CommitMoE (AAAI 2026), BuddyMoE (arXiv 2511.10054) | Compute on predicted or substituted experts | Lossy, no verify-and-discard |
| "The Other Half of the Memory Wall" (arXiv 2609.18063) | Serves 35B MoEs from SSD with trained routing prediction | Nearest in setting; changes the routing (lossy) |
| Specter (arXiv 2511.14102), SeqMoE (arXiv 2609.12978) | Speculative / predictive offloading of weights | Prefetch, not speculative compute |

Older work placed from prior knowledge and not re-read: Mixtral-offloading, MoE-Infinity, Pre-gated MoE, ProMoE,
AdapMoE, HOBBIT, Fiddler, MoE-Lightning, SwapMoE, ExpertFlow, EdgeMoE, LLM in a Flash, PowerInfer. Speculate-and-roll-back
on GPUs is an old general pattern; the application here is the new part.

## What makes ours distinct (to state if we write it up)
1. **Exact**: the result is kept only when the snapshot covered every routed expert, so the output is bit-identical.
   The near items that compute on resident or predicted experts are lossy.
2. **SSD-backed misses**, where neither the device nor the CPU can serve a miss, so GPU-indirection designs that
   compute misses elsewhere don't carry over.
3. **Hides the sync instead of removing it**: the host still checks the routing every layer, but the GPU never waits
   for it (MLX lazy graph plus async_eval). No model change, no MTP head.
4. Still to report for a write-up: hit rate (`spec_used` / `spec_wasted` are recorded), the cost of a discard, the
   tokens x k < 64 cutoff, and A/B against the #27861-style split and slotstream-style draining.

Search: done by a research subagent (queries: speculative MoE execution with verification, MLX/Apple SSD expert
streaming, sync-free GPU slot tables, execute-on-predicted-then-verify, "speculative dispatch", mlx-lm per-layer eval
sync, zero-eval MLX, llama.cpp Metal residency, generic speculative launch with rollback). Links were spot-checked by
GitHub API / arXiv API title lookups. The streamlx README was read directly.
