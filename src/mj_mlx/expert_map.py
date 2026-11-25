# Copyright 2026 the MilliJoule authors
# SPDX-License-Identifier: Apache-2.0
"""Memory-mapped MoE experts: the GPU reads expert weights straight from the page cache, with no copy and no wiring.

For a model that fits in RAM but not in the GPU working set (gpt-oss-20b on a 16 GB Mac: 13.7 GB, of which 10.1 GB
experts), llama.cpp gets 1.85x our SSD-streaming decode speed by handing Metal no-copy buffers over its mmap'd file
(results/public_baselines.md). MLX can do the same: `mx.from_dlpack` of a page-aligned NumPy array wraps the
memory without copying (checked: a write through NumPy is visible in the MLX array). macOS then pages expert
weights in and out of the unified buffer cache like any file, instead of MLX wiring them.

Layout (part-major, one file per layer, written by `repack_part_major` from an ExpertStore): for each part (gu_w,
gu_s, dn_w, dn_s, ...) the [E, *shape] block of all experts, each block starting at a 16 KB page boundary, so
each part maps to one contiguous [E, *shape] array that gather_qmm reads in place.
"""

from __future__ import annotations

import json
import os

import mlx.core as mx
import numpy as np

PAGE = 16384
# Shared mapping ("r+"): the expert pages stay ordinary file cache, which macOS drops and re-reads under memory
# pressure. A private copy-on-write mapping ("c") was turned into anonymous memory once Metal used it (no copy):
# gpt-oss-20b on 16 GB then swapped (+2 GB per run) and decoded at 41.6-43.4 vs 33.3-35.2 ms/token (RESULTS 1p). A
# read-only mapping ("r") made Metal thrash (free memory ~0, no progress in 10 min). Nothing writes these arrays.
MAP_MODE = os.environ.get("MJ_MAP_MODE", "r+")
MADVISE = os.environ.get("MJ_MADVISE", "")  # experiment: random / willneed / sequential
# MJ_MLOCK=1: pin every mapped expert page in RAM (mlock; counts against vm.user_wire_limit). Without it macOS evicts
# expert pages under memory pressure and the GPU waits on SSD reads mid-step (gpt-oss-20b on 16 GB: 300 MB-2.4 GB of
# page-ins per 128 decode tokens, 45-59 ms/token; RESULTS 1y). llama.cpp's weights stay resident the same way.
# MJ_MLOCK=bg: the same pinning in a background thread once the model is built (start_background_lock), so the first
# request does not wait for it (checklist 0.11: gpt-oss cold start 6.4 s vs 2.3-2.9 s for llama.cpp / LM Studio).
MLOCK = os.environ.get("MJ_MLOCK") == "1"
MLOCK_BG = os.environ.get("MJ_MLOCK") == "bg"
# MJ_MLOCK_TOP=<k> (with MJ_MLOCK=1 or bg): pin only each layer's k most-used experts, ranked by expert_rank.json in
# the store (tools/mlx_expert_usage.py --write-rank); the rest stay file cache (checklist 0.11b: pinning all 10 GB of
# gpt-oss's experts swaps out ~2.2 GB at every load and leaves too little memory for long contexts). On gpt-oss the
# top 24 of 32 took 87-96% of a layer's decode routings on held-out prompts, the top 16 66-87% (RESULTS 1zzd).
MLOCK_TOP = int(os.environ.get("MJ_MLOCK_TOP", "0"))


def _mlock(a: np.ndarray, tries: int = 20, wait_s: float = 0.25) -> bool:
    """Pin `a`'s pages. EAGAIN is transient under memory pressure (seen once at load; the same 10.15 GB locked in
    1.7 s with no retry a minute later), so retry; after `tries` give up with a warning and leave `a` unpinned
    (the caller records how many bytes are pinned). Other errors raise."""
    import ctypes
    import ctypes.util
    import errno
    import time
    import warnings

    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    for _ in range(tries):
        if libc.mlock(ctypes.c_void_p(a.ctypes.data), ctypes.c_size_t(a.nbytes)) == 0:
            return True
        e = ctypes.get_errno()
        if e != errno.EAGAIN:
            raise OSError(e, f"mlock of {a.nbytes / 1e6:.0f} MB failed: {os.strerror(e)}")
        time.sleep(wait_s)
    warnings.warn(f"mlock of {a.nbytes / 1e6:.0f} MB still EAGAIN after {tries} tries: left unpinned "
                  "(vm.global_user_wire_limit or memory pressure)", stacklevel=2)
    return False
NP_DTYPES = {"uint8": np.uint8, "uint32": np.uint32, "bfloat16": np.uint16, "float16": np.float16,
             "float32": np.float32}


def _align(n: int) -> int:
    return (n + PAGE - 1) // PAGE * PAGE


def repack_part_major(src: str, dst: str) -> None:
    """Rewrite an ExpertStore directory (expert-major records) as part-major mapped files."""
    idx = json.load(open(os.path.join(src, "index.json")))
    E, stride, parts = idx["num_experts"], idx["stride"], idx["parts"]
    os.makedirs(dst, exist_ok=True)
    offs, o = [], 0
    for p in parts:
        offs.append(o)
        o = _align(o + E * p["bytes"])
    rec_off = np.cumsum([0] + [p["bytes"] for p in parts])
    for li in idx.get("moe_layers", range(idx["num_layers"])):
        raw = np.fromfile(os.path.join(src, f"layer_{li:02d}.bin"), np.uint8)
        out = np.zeros(o, np.uint8)
        for j, p in enumerate(parts):
            for e in range(E):
                a = e * stride + rec_off[j]
                out[offs[j] + e * p["bytes"]: offs[j] + (e + 1) * p["bytes"]] = raw[a:a + p["bytes"]]
        out.tofile(os.path.join(dst, f"layer_{li:02d}.bin"))
    meta = dict(idx, layout="part_major", part_offsets=offs, file_bytes=o, page=PAGE, source_store=src)
    json.dump(meta, open(os.path.join(dst, "index.json"), "w"), indent=1)


class MappedExperts:
    """Per layer {part name: mx.array [E, *shape]} backed by the mapped files (no copy)."""

    def __init__(self, path: str):
        self.path = path
        self.index = json.load(open(os.path.join(path, "index.json")))
        assert self.index.get("layout") == "part_major", "run repack_part_major first"
        self._maps = []  # keep the NumPy maps alive as long as the arrays
        self.locked = 0  # bytes pinned with MJ_MLOCK=1 (or so far, with MJ_MLOCK=bg)
        self._to_lock = []  # MJ_MLOCK=bg: arrays pinned by start_background_lock
        self.lock_thread = None
        self.rank = None  # MJ_MLOCK_TOP: {layer: [experts, most used first]}
        rf = os.path.join(path, "expert_rank.json")
        if MLOCK_TOP and os.path.exists(rf):
            self.rank = {int(k): v for k, v in json.load(open(rf))["layers"].items()}
        elif MLOCK_TOP:
            import warnings

            warnings.warn(f"MJ_MLOCK_TOP set but {rf} is missing: pinning every expert", stacklevel=2)

    def _pin_units(self, li: int, a: np.ndarray) -> list:
        """What to pin of one part array [E, ...]: all of it, or the top MJ_MLOCK_TOP experts' rows."""
        if self.rank is None or li not in self.rank:
            return [a]
        return [a[e] for e in self.rank[li][:MLOCK_TOP]]

    def layer(self, li: int) -> dict:
        E = self.index["num_experts"]
        f = os.path.join(self.path, f"layer_{li:02d}.bin")
        out = {}
        for p, off in zip(self.index["parts"], self.index["part_offsets"]):
            a = np.memmap(f, dtype=NP_DTYPES[p["dtype"]], mode=MAP_MODE, offset=off, shape=(E, *p["shape"]))
            self._maps.append(a)
            if MADVISE and getattr(a, "_mmap", None) is not None:
                import mmap as _mm

                a._mmap.madvise({"random": _mm.MADV_RANDOM, "willneed": _mm.MADV_WILLNEED,
                                 "sequential": _mm.MADV_SEQUENTIAL}[MADVISE])
            if MLOCK:
                for u in self._pin_units(li, a):
                    if _mlock(u):
                        self.locked += u.nbytes
            elif MLOCK_BG:
                self._to_lock += self._pin_units(li, a)
            x = mx.from_dlpack(a)
            if p["dtype"] == "bfloat16":
                x = x.view(mx.bfloat16)
            out[p["name"]] = x
        return out

    def start_background_lock(self) -> None:
        """MJ_MLOCK=bg: pin the mapped experts from a daemon thread (no-op otherwise or when already started)."""
        if not self._to_lock or self.lock_thread is not None:
            return
        import threading

        todo, self._to_lock = self._to_lock, []

        def run():
            for a in todo:
                if _mlock(a):
                    self.locked += a.nbytes

        self.lock_thread = threading.Thread(target=run, name="mj-mlock", daemon=True)
        self.lock_thread.start()

    def bytes_mapped(self) -> int:
        return sum(a.nbytes for a in self._maps)


def map_rows(cache_dir: str, src_path: str, name: str | list[str], keep: list) -> mx.array:
    """One bf16 checkpoint tensor (e.g. the embedding table or lm_head) as a zero-copy mx.array over a page-aligned
    copy in `cache_dir` (written once from the safetensors file; the tensor inside a safetensors file starts at an
    arbitrary offset, which Metal cannot wrap without copying). Rows are then gathered on the GPU, so a lean model
    needs no host round trip per token for its embedding or rescore rows, and only the touched pages are resident.
    `keep` holds the NumPy map alive as long as the array."""
    from .expert_stream import SafetensorsRows

    src = SafetensorsRows(src_path, name)
    nm = src.name
    os.makedirs(cache_dir, exist_ok=True)
    f = os.path.join(cache_dir, nm.replace("/", "_") + ".bin")
    meta_f = f[:-4] + ".json"
    want = dict(source=src.file, source_bytes=os.path.getsize(src.file), name=nm, shape=list(src.rows.shape),
                dtype="bfloat16")
    if not (os.path.exists(f) and os.path.exists(meta_f) and json.load(open(meta_f)) == want
            and os.path.getsize(f) == src.rows.nbytes):
        out = np.memmap(f + ".tmp", dtype=np.uint16, mode="w+", shape=src.rows.shape)
        step = max(1, (64 << 20) // (src.rows.shape[1] * 2))
        for i in range(0, src.rows.shape[0], step):
            out[i:i + step] = src.rows[i:i + step]
        out.flush()
        del out
        os.replace(f + ".tmp", f)
        json.dump(want, open(meta_f, "w"), indent=1)
    a = np.memmap(f, dtype=np.uint16, mode=MAP_MODE, shape=tuple(want["shape"]))
    keep.append(a)
    return mx.from_dlpack(a).view(mx.bfloat16)
