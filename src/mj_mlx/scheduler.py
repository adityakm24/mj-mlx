# Copyright 2026 the MilliJoule authors
# SPDX-License-Identifier: Apache-2.0
"""The step scheduler (Rust, `mj_mlx._sched`): requests, paged KV blocks, radix prefix cache, flat step batches.

Forked from MilliJoule's engine core on 2026-11-21; mj-mlx owns this copy and tunes it for the Mac.
"""

from mj_mlx._sched import Scheduler, StepBatch, __version__ as core_version

# StepBatch array fields: name -> (numpy dtype, shape kind). "items" arrays have num_items rows, "tokens" arrays
# num_tokens, "sample" arrays num_sample; cu_seqlens has num_items + 1; block_tables is [num_items, max_blocks].
STEP_BATCH_DTYPES = {
    "request_ids": ("uint64", "items"),
    "starts": ("int32", "items"),
    "lens": ("int32", "items"),
    "kv_lens": ("int32", "items"),
    "sample": ("uint8", "items"),
    "is_decode": ("uint8", "items"),
    "temperature": ("float32", "items"),
    "top_p": ("float32", "items"),
    "top_k": ("int32", "items"),
    "seed": ("uint64", "items"),
    "state_slots": ("int32", "items"),
    "cu_seqlens": ("int32", "items+1"),
    "block_tables": ("int32", "items x max_blocks"),
    "token_ids": ("uint32", "tokens"),
    "prev_sample_idx": ("int32", "tokens"),
    "positions": ("int32", "tokens"),
    "slot_mapping": ("int32", "tokens"),
    "sample_rows": ("int32", "sample"),
    "sample_request_ids": ("uint64", "sample"),
}


def as_numpy(batch: StepBatch) -> dict:
    """Every StepBatch array as a read-only numpy view (no copy), with block_tables shaped [items, max_blocks]."""
    import numpy as np

    out = {name: np.frombuffer(getattr(batch, name), dtype=dtype) for name, (dtype, _) in STEP_BATCH_DTYPES.items()}
    out["block_tables"] = out["block_tables"].reshape(batch.num_items, batch.max_blocks)
    return out


__all__ = ["Scheduler", "StepBatch", "STEP_BATCH_DTYPES", "as_numpy", "core_version"]
