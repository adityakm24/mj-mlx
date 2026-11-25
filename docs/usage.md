# Using mj-mlx

> This is the engine's current Python API. It works on token ids; tokenizing and chat templates are up to the caller
> (for example transformers' `AutoTokenizer`). The first release adds an OpenAI-compatible server and a CLI on top
> of it, and this API may change before then.

## Import order

Import `mj_mlx` before `mlx`. On import it sets two MLX environment variables that MLX reads only once:

- `MLX_MAX_MB_PER_BUFFER=150`. Memory-mapped expert models decode about 1.6x slower at MLX's default.
- `MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS=200`. At MLX's default, the first prefill after an idle period is about
  10% slower.

If `mlx` was imported first and the variables are not set, `mj_mlx` warns.

## Load a model and generate

```python
import mj_mlx

model = mj_mlx.load_model(path, placement="auto")
engine = mj_mlx.Engine(model=model, overlap=True)
out = engine.generate([prompt_ids_1, prompt_ids_2], max_new_tokens=256)   # list of token-id lists
```

`load_model(path)` picks the model class from `config.json`'s `model_type`:

| `model_type` | Class |
| --- | --- |
| `gpt_oss` | `GptOss` |
| `qwen3_5_moe`, `qwen3_5_moe_text`, `qwen3_5`, `qwen3_5_text` | `Qwen35Moe` |
| `mixtral`, `qwen2_moe`, `qwen3_moe`, `olmoe`, `deepseek_v2`, `deepseek_v3`, `glm4_moe` | `MoeLM` (general MoE model) |

For every MoE family (`GptOss`, `Qwen35Moe` and `MoeLM`), `placement` decides where the routed experts live:

- `"resident"`: in memory.
- `"mapped"`: memory-mapped from a repacked file, so macOS pages them like any file.
- `"stream"`: on the SSD, with a fixed number of slots per layer held in memory.
- `"auto"`: chosen from the checkpoint size and free memory.

The repacked files are written once, under `~/.cache/mj-mlx/` (stores made earlier under `~/.cache/millijoule/` are
still used). Files already made from the same checkpoint are
found again by the source path in their `index.json`, under any name, so nothing is repacked twice. See
[models.md](models.md).

On a 16 GB Mac, `"auto"` picks `resident` for OLMoE and DeepSeek-V2-Lite, `mapped` for gpt-oss-20b, and `stream` for
Qwen3-30B-A3B and Qwen3.6-35B-A3B. Qwen3.5 / 3.6 checkpoints with bf16 experts are streamed instead of mapped (only
quantized experts are mapped).

Explicit arguments still work and win over `placement`:

- `map_experts=`: a directory written by `expert_map.repack_part_major`.
- `stream_experts=`: an `ExpertStore` directory, plus `expert_slots=` (more slots: fewer SSD reads, more memory).
- `lean=True`: keeps the embedding table and lm_head on disk.

`tools/mlx_bench_suite.py` shows the arguments used for each benchmarked model. Its Qwen3.6 rows pin 96 slots; the
memory formula gives 117 on a 16 GB M4, inside the range that has run (112 and 128 ran plain; 128 with the MTP draft
layer ran out of memory).

Quantized checkpoints run in the dtype of their scales (`float16` for most mlx-community files), as mlx-lm does.
Running them in bf16 makes decode about 3x slower.

## Engine options

`Engine(model_path=None, *, model=None, ...)`. These are the options that matter most:

| Option | Default | Meaning |
| --- | --- | --- |
| `overlap` | `False` | Schedule step N+1 while step N runs on the GPU. Use `True` (the benchmarks do). |
| `max_running` | 256 | Maximum requests decoding at once. |
| `max_batch_tokens` | 4096 | Tokens per step (prefill chunks plus decodes). |
| `max_prefill_chunk` | `max_batch_tokens` | Longest prefill chunk per request per step. |
| `max_seq_len` | 32768 | Longest request (prompt plus output). |
| `enable_prefix_cache` | `True` | Radix prefix cache. Turned off automatically for Qwen3.5 / 3.6, whose recurrent state it cannot restore yet. |
| `num_blocks` | from memory | KV pool size in 16-token blocks. |
| `wired_limit_bytes` | MLX's recommended working set | Memory macOS may not page out. Memory-mapped experts set 1000 MB (RESULTS 1zk). |
| `cache_limit_bytes` | 1 GiB | Cap on MLX's freed-buffer cache. Without it, a 16 GB Mac swaps during long prefills. |

## Requests, streaming and stopping

`generate` runs requests to completion. For incremental output, drive the engine yourself:

```python
rid = engine.add_request(prompt_ids, max_new_tokens=256, stop_token_ids=None, ignore_eos=False)
while engine.has_work():
    for finished in engine.step():
        # (request_id, reason, output_tokens, cached_prompt_tokens, num_preemptions)
        ...
```

`stop_token_ids=None` uses the model's EOS tokens.

## Sampling

Greedy is the default. To sample, pass `temperature`, `top_p`, `top_k` and `seed` to `add_request` or `generate`.

```python
engine.generate([ids], max_new_tokens=128, temperature=0.7, top_p=0.9, seed=1)
```

Randomness is a pure function of (seed, position, vocab id). A request that is preempted and recomputed samples the
same tokens again.

**Limit:** the gpt-oss lean fast path (`GptOss(..., lean=True)`, `--lean` in the tools) is greedy only today.

## Speculative decoding (one request)

- **Qwen3.5 / Qwen3.6, the model's own MTP head:**
  `mj_mlx.spec_decode.SpecGenerator(model, mtp_path, k=...)`, then `.generate(ids, max_new)`. The mlx-community
  conversions drop the MTP tensors; `tools/mlx_fetch_mtp.py` fetches just those.
- **gpt-oss, an EAGLE-3 head** (for example `nebius/EAGLE3-gpt-oss-20b`):
  `mj_mlx.eagle3.Eagle3Generator(model, head_path, k=...)`, then `.generate(ids, max_new)`.
- **Draft length:** `spec_decode.DraftPolicy(k_max, mode="time", cost=...)` chooses it per step from the measured
  acceptance rate and step cost. `tools/mlx_spec_calibrate.py` measures that cost.

Greedy speculative output is token-identical to plain greedy output. The 2-4 row verify step uses a kernel whose
rows round exactly like the one-row kernel (`qmv_rows.py`).

## Load time: the prepared-weights cache

gpt-oss-20b derives some arrays from the checkpoint at every load: the low-bit candidate head (about 1 s), and
8-bit or packed attention (0.7 s / 3.4 s). The first load writes them to a cache and later loads read them, so a
load goes from 3.7-5.2 s to about 1.9 s to the first token (8-bit attention) and from 7.5-7.7 s to about 2.6 s
(exact), with the same tokens (RESULTS 1zzc).

- Where: `~/Library/Caches/mj-mlx/prep`, one subfolder per checkpoint and option set (its `key.json` says which),
  about 2 GB for gpt-oss-20b's two arms, up to twice that while two code versions are kept (below). The first write
  prints the folder and the size written.
- `MJ_PREP_CACHE=<dir>` puts it elsewhere, `MJ_PREP_CACHE=0` turns it off; deleting the folder clears it.
- Safety: the key covers the checkpoint files, the options, MLX's version and the source of the code that derives
  the arrays; every load also re-derives a sample (layer 0's attention, the head's first rows) and compares it bit
  for bit, and deletes the folder on a mismatch. It does not write when free disk is under about 5 GB. The two
  newest folders per checkpoint and option set are kept, older ones deleted.
- The first-ever load is slower than no cache (it computes and writes: about 7.7 s / 9.7 s).

## What is on disk, and clearing it

`python -m mj_mlx cache list` shows every cache entry (repacked expert stores under `~/.cache/mj-mlx`, and under
`~/.cache/millijoule` for stores written before 2026-11-23, which are still found and used; the
prepared-weights cache under `~/Library/Caches/mj-mlx/prep`) with its size, its source checkpoint, and whether that
checkpoint still exists. `python -m mj_mlx cache rm <name>` deletes entries by the name `list` prints;
`python -m mj_mlx cache rm --orphans` deletes the ones whose checkpoint is gone. Entries mj-mlx did not write are
shown as `other` and kept unless `--force` is given.

## Memory on a 16 GB Mac

- One model at a time. Two engines in one process, or a second model server, will swap.
- Models bigger than about 12 GB stream their experts (`placement="auto"` handles this). Decode speed then depends
  on how often the same experts are reused across tokens.
- **Pinning mapped experts:** `MJ_MLOCK=1` pins every mapped expert page in RAM, so macOS cannot drop them and decode
  does not wait on the SSD (RESULTS 1y); `MJ_MLOCK=bg` pins in a background thread after the load, so the first
  request does not wait; unset (the default), they are file cache. Pinning gpt-oss-20b's 10 GB of experts leaves
  too little memory for a 16k context on 16 GB. `MJ_MLOCK_TOP=k` pins only each layer's k most-used experts, ranked
  by `expert_rank.json` in the expert folder (`tools/mlx_expert_usage.py --write-rank`): with 16 of 32, decode
  speed is unchanged, prefill is slower and less steady (unpinned experts are read back from the SSD), resident
  memory drops from 10.3 to 5.2 GB, and the exact arm answers 12 of 12 long-context questions at 16k (RESULTS 1zzd).
- **Wired limit for mapped experts:** the engine sets 1000 MB. At 3000 MB, gpt-oss-20b peaked at about 13.9 GB, above
  macOS's 12.71 GB Metal limit on this Mac, and macOS once killed it (RESULTS 1zh). At 1000 MB neither the exact nor
  the 8-bit arm logged a Metal over-limit event, at the same speed, and an hour-long soak ran clean (RESULTS 1zi, 1zk).
  Pass `wired_limit_bytes` (`--wired-mb` in the tools) to change it.
- **Long prompts:** prefill attention runs in blocks of queries when a model's head size is one MLX's fused attention
  kernel does not take (DeepSeek-V2 MLA 192 / 128, Qwen3.5 / 3.6 256), so the score matrix stays near 256 MB per
  layer, then in blocks of 64 MB of scores (`MJ_ATTN_SCORES_MB`, 0 turns it off). At 256 MB per block
  DeepSeek-V2-Lite's 16k prefill sat at macOS's 12.71 GB Metal limit and was killed twice; at 64 MB it runs at the
  default 2048-token chunk with no over-limit events, 1-5% faster in prefill (RESULTS 1zj, 1zm). If a model still
  runs out of Metal memory at long context, `Engine(prefill_area_cap=...)` shrinks the chunk as the context grows.
