# Copyright 2026 the MilliJoule authors
# SPDX-License-Identifier: Apache-2.0
"""Stream routed MoE experts from the SSD into a fixed set of slots per layer: one model-agnostic core.

Design ideas from reading streamlx, Flash-MoE, Colibri and mlx-moe-streaming (see
results/expert_streaming_research.md). Ideas only: no code from those projects is copied or adapted here; this
file and tools/mlx_repack_experts.py were written from scratch for MilliJoule. F_NOCACHE / F_RDAHEAD are macOS
fcntl constants.

Pieces (shared by every MoE model; a model only supplies `pair_fn`, the math of one expert on (token, expert) pairs):
- `ExpertStore`: the repacked checkpoint (tools/mlx_repack_experts.py): one file per layer, expert e at e * stride,
  each record the expert's parts back to back in the engine's in-memory layout. index.json describes the parts
  (name, dtype, shape), so the store knows nothing about the model.
- `ExpertSlab`: one layer's evaluated slot arrays [S, *shape] per part; misses are preadv'd straight into the slots'
  memory (writable memoryview of the MLX array: no copy, no re-wiring); id -> slot table; least-recently-used
  eviction; background prefetch of predicted experts.
- `run_moe`: one MoE step. Syncs once on this layer's routing (and on the next layer's predicted routing, used to
  prefetch while this layer computes), runs the needed experts in rounds of at most S, puts the pair outputs back in
  (token, k) order and applies the routing weights.
Slots are only rewritten after the host has synced on this layer's routing, and the next layer's slots only while
this layer computes, so no queued kernel reads a slot being overwritten.

Accuracy: same math as each model's in-memory path. Not bitwise in bf16: MLX's gather_qmm accumulation depends on
the shape of the whole call (measured: a 160-row call and the same rows one at a time differ by 1 bf16 ulp on 40
rows), so streaming, like batching, changes accumulation order. The gate is the float32 near-tie rule (tests/mlx).
"""

from __future__ import annotations

import fcntl
import json
import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx
import numpy as np

F_NOCACHE = 48  # macOS fcntl: do not keep this fd's reads in the unified buffer cache
F_RDAHEAD = 45  # macOS fcntl: read-ahead on/off

DTYPES = {"uint8": mx.uint8, "uint32": mx.uint32, "bfloat16": mx.bfloat16, "float16": mx.float16,
          "float32": mx.float32}


SPECULATE = True  # run_moe's speculative all-resident dispatch (off: always wait for the routing first)
# Before the speculative snapshot, count prefetch reads that have already landed as resident (off: every read still
# in the pending map counts as absent, the 1d.3 behaviour).
HARVEST = os.environ.get("MJ_SPEC_HARVEST", "1") == "1"
# If this layer still has prefetch reads in flight: start its attention and router on the GPU first, wait for the
# reads while they run, then queue the speculative MoE with those experts resident (off: in-flight experts count as
# absent, so a correct but late prefetch discards the speculative call and the GPU waits for routing + read + host).
# Prefill (predicted set larger than the slots): prefetch the next layers' S most often predicted experts, so their
# reads overlap this layer's compute (off: no prefetch when the prediction does not fit, the old behaviour).
PREFILL_PREFETCH = os.environ.get("MJ_PREFILL_PREFETCH", "1") == "1"
WAIT_PENDING = os.environ.get("MJ_SPEC_WAIT", "0") == "1"  # no gain measured (RESULTS 1s)


class ExpertStore:
    """Opened per-layer expert files plus a shared reader pool."""

    def __init__(self, path: str, threads: int = 8, nocache: bool = True):
        self.path = path
        self.index = json.load(open(os.path.join(path, "index.json")))
        self.stride = self.index["stride"]
        self.parts = self.index["parts"]  # [{name, dtype, shape, bytes}] in record order
        self.fds = {}  # layer -> fd; only MoE layers have a file (models with dense layers: index "moe_layers")
        for i in self.index.get("moe_layers", range(self.index["num_layers"])):
            fd = os.open(os.path.join(path, f"layer_{i:02d}.bin"), os.O_RDONLY)
            fcntl.fcntl(fd, F_RDAHEAD, 0)
            if nocache:
                fcntl.fcntl(fd, F_NOCACHE, 1)
            self.fds[i] = fd
        self.pool = ThreadPoolExecutor(threads)
        self.bytes_read = 0

    def read_into(self, layer: int, expert: int, views: list[memoryview]) -> int:
        return os.preadv(self.fds[layer], views, expert * self.stride)


def layer_slots(expert_slots, layer: int) -> int:
    """Slots for one MoE layer: `expert_slots` is one count for every layer, or {layer: count} (uneven slots, from
    placement.uneven_slots), or a list indexed by layer."""
    if isinstance(expert_slots, dict):
        return int(expert_slots.get(layer, expert_slots.get(str(layer))))
    if isinstance(expert_slots, (list, tuple)):
        return int(expert_slots[layer])
    return int(expert_slots)


class ExpertSlab:
    """One layer's resident expert slots. `self.w[name]` is the [S, *shape] array of part `name`."""

    def __init__(self, store: ExpertStore, layer: int, num_slots: int):
        E = store.index["num_experts"]
        self.store, self.layer, self.E = store, layer, E
        self.S = max(1, min(num_slots, E))
        self.w = {p["name"]: mx.zeros((self.S, *p["shape"]), DTYPES[p["dtype"]]) for p in store.parts}
        mx.eval(list(self.w.values()))
        self._mv = [memoryview(self.w[p["name"]]).cast("B") for p in store.parts]
        self._per = [self.w[p["name"]].nbytes // self.S for p in store.parts]
        assert self._per == [p["bytes"] for p in store.parts], (self._per, store.parts)
        self.slot_of = np.full(E, -1, np.int64)
        self.expert_of = np.full(self.S, -1, np.int64)
        self.lru: OrderedDict[int, None] = OrderedDict()  # resident experts, least recently used first
        self.pending: dict[int, object] = {}  # expert -> future of an in-flight prefetch read
        self.prefetched: set[int] = set()
        self.hits = self.misses = self.pf_issued = self.pf_used = 0
        self.spec_used = self.spec_wasted = 0  # speculative dispatches kept / discarded (run_moe)
        self.spec_pending_only = 0  # discarded only because a routed expert's read was still in flight

    def arrays(self) -> list[mx.array]:
        return list(self.w.values())

    def _views(self, slot):
        return [mv[slot * n:(slot + 1) * n] for mv, n in zip(self._mv, self._per)]

    def _take_slot(self, protect: set[int]) -> int | None:
        free = np.nonzero(self.expert_of < 0)[0]
        if len(free):
            return int(free[-1])
        for old in self.lru:  # least recently used first
            if old not in protect:
                slot = int(self.slot_of[old])
                self.slot_of[old] = -1
                del self.lru[old]
                self.expert_of[slot] = -1
                return slot
        return None

    def harvest(self) -> None:
        """Retire prefetch reads that have already completed (non-blocking)."""
        for e, f in list(self.pending.items()):
            if f.done():
                self.store.bytes_read += self.pending.pop(e).result()

    def _wait(self):
        for e in list(self.pending):
            self.store.bytes_read += self.pending.pop(e).result()

    def prefetch(self, experts: np.ndarray) -> None:
        """Start background reads for predicted experts (non-blocking); evicts only experts not predicted."""
        protect = set(int(e) for e in experts) | set(self.pending)
        for e in dict.fromkeys(int(x) for x in experts):
            if self.slot_of[e] >= 0 or e in self.pending:
                continue
            slot = self._take_slot(protect)
            if slot is None:
                return
            self.slot_of[e], self.expert_of[slot] = slot, e
            self.lru[e] = None
            self.pending[e] = self.store.pool.submit(self.store.read_into, self.layer, e, self._views(slot))
            self.prefetched.add(e)
            self.pf_issued += 1

    def start_loads(self, experts) -> list:
        """Assign slots to the non-resident `experts` (evicting LRU experts not in the set) and start their reads;
        returns the futures. Resident ones get LRU/hit bookkeeping. Caller waits before using the new slots."""
        need = set(int(e) for e in experts)
        for e in need & set(self.pending):  # this step needs it: its prefetch read must land first
            self.store.bytes_read += self.pending.pop(e).result()
        self.pf_used += len(need & self.prefetched)
        self.prefetched -= need
        missing = [e for e in need if self.slot_of[e] < 0]
        self.hits += len(need) - len(missing)
        self.misses += len(missing)
        futs = []
        for e in missing:
            slot = self._take_slot(need | set(self.pending))
            self.slot_of[e], self.expert_of[slot] = slot, e
            futs.append(self.store.pool.submit(self.store.read_into, self.layer, e, self._views(slot)))
        for e in need:
            self.lru.pop(e, None)
            self.lru[e] = None
        return futs

    def ensure(self, experts) -> None:
        """Make all `experts` (len <= S) resident; evict least recently used experts not in the set."""
        need = set(int(e) for e in experts)
        if self.pending:  # this layer's prefetches were issued during the previous layer: finish them first
            self._wait()
        self.pf_used += len(need & self.prefetched)
        self.prefetched -= need
        missing = [e for e in need if self.slot_of[e] < 0]
        self.hits += len(need) - len(missing)
        self.misses += len(missing)
        jobs = []
        for e in missing:
            slot = self._take_slot(need)
            self.slot_of[e], self.expert_of[slot] = slot, e
            jobs.append((e, slot))
        for f in [self.store.pool.submit(self.store.read_into, self.layer, e, self._views(s)) for e, s in jobs]:
            self.store.bytes_read += f.result()
        for e in need:  # most recently used last
            self.lru.pop(e, None)
            self.lru[e] = None

    def touch(self, experts) -> bool:
        """Fast-path residency check: True (and LRU/hit bookkeeping done) if every expert is resident; waits only
        for in-flight prefetch reads of these experts."""
        if (self.slot_of[experts] < 0).any():
            return False
        for e in experts:
            e = int(e)
            f = self.pending.pop(e, None)
            if f is not None:
                self.store.bytes_read += f.result()
            if e in self.prefetched:
                self.prefetched.discard(e)
                self.pf_used += 1
            self.lru.move_to_end(e)
        self.hits += len(experts)
        return True

    def stats(self) -> dict:
        n = self.hits + self.misses
        return dict(hits=self.hits, misses=self.misses, hit_rate=self.hits / n if n else None,
                    prefetch_issued=self.pf_issued, prefetch_used=self.pf_used, spec_used=self.spec_used,
                    spec_wasted=self.spec_wasted, spec_pending_only=self.spec_pending_only)


def run_moe(slab: ExpertSlab, x: mx.array, idx: mx.array, wts: mx.array, pair_fn, next_pred=None) -> mx.array:
    """Routed-expert output [T, H] = sum_k wts[t, k] * expert_{idx[t, k]}(x[t]), experts streamed through `slab`.

    pair_fn(slab, x, tok, eid, slots, sorted_) -> [P, H] outputs of pairs (tok[p], eid[p]) in the given order, using
    slot ids `slots` into slab.w; sorted_ is the in-memory path's kernel choice (tokens * k >= 64).
    With sorted_ False, eid and slots may be MLX arrays (the speculative path looks slots up on the GPU).
    next_pred = (next_slab, logits) with logits [T, E] (lazy) of the next layer's router on this layer's MoE input;
    its top-k is prefetched into next_slab while this layer computes."""
    T, k = idx.shape
    tok = np.repeat(np.arange(T), k)
    # next_pred: one (slab, logits) pair or a list of them, nearest layer first (prefetch depth > 1)
    preds = [] if next_pred is None else (list(next_pred) if isinstance(next_pred, list) else [next_pred])
    tops = [(ns, mx.argpartition(lg, kth=-k, axis=-1)[..., -k:]) for ns, lg in preds]
    nxt = [t for _, t in tops] or None

    def prefetch_next():  # nearest layer first: its reads are needed soonest
        for ns, t in tops:
            pred, cnt = np.unique(np.array(t), return_counts=True)
            if len(pred) > ns.S:  # long steps (prefill): fill the slots so that layer's first round is resident
                if not PREFILL_PREFETCH:
                    continue
                res = ns.slot_of[pred] >= 0  # predicted and already resident (or in flight): keep them all
                room = ns.S - int(res.sum())
                rest = pred[~res][np.argsort(-cnt[~res], kind="stable")[:max(room, 0)]]  # most often predicted
                pred = np.concatenate([pred[res], rest])
            ns.prefetch(pred)
    # Speculative dispatch (small batches): queue the all-resident computation before the routing reaches the host,
    # with slots looked up on the GPU from a snapshot of the residency map (experts whose reads are still in flight
    # count as absent). If every routed expert was resident in the snapshot, this is exactly the fast path below
    # (same calls, same shapes) and the GPU never waits for the host; otherwise it is discarded. (Keeping the
    # resident rows and recomputing only the absent experts' rows was tried: no gain, RESULTS.md 1d.3.)
    spec = None
    if SPECULATE and T * k < 64:
        if WAIT_PENDING and slab.pending:
            mx.async_eval(idx) if nxt is None else mx.async_eval(idx, *nxt)  # GPU runs attention + router meanwhile
            slab._wait()
        elif HARVEST:
            slab.harvest()
        snap = slab.slot_of.copy()
        snap_raw = snap.copy()
        if slab.pending:
            snap[list(slab.pending)] = -1
        if (snap >= 0).any():
            sl = mx.array(np.maximum(snap, 0).astype(np.int32))[idx.reshape(-1)]
            spec_y = pair_fn(slab, x, tok, idx.reshape(-1), sl, False)
            spec = (spec_y.reshape(T, k, -1) * wts[..., None]).sum(axis=1)
            mx.async_eval(idx) if nxt is None else mx.async_eval(idx, *nxt)  # routing first, in its own batch
            mx.async_eval(spec)
    if nxt is not None:
        mx.eval(idx, *nxt)  # one host sync for all
    ids = np.array(idx)  # host sync: this layer's routing
    flat = ids.reshape(-1)
    sorted_ = flat.size >= 64
    uniq = np.unique(flat)
    if spec is not None and (snap[uniq] >= 0).all():
        slab.spec_used += 1
        slab.touch(uniq)  # bookkeeping only: all resident, none in flight
        if nxt is not None:
            prefetch_next()
        return spec
    if spec is not None:
        slab.spec_wasted += 1
        if (snap_raw[uniq] >= 0).all():
            slab.spec_pending_only += 1
    if slab.touch(uniq):  # fast path (the common case): everything resident, one call, GPU starts at once
        y = pair_fn(slab, x, tok, flat, slab.slot_of[flat].astype(np.int32), sorted_)
        out = (y.reshape(T, k, -1) * wts[..., None]).sum(axis=1)
        mx.async_eval(out)  # GPU computes this layer while the host issues the next layer's prefetch
        if nxt is not None:
            prefetch_next()
        return out
    if len(uniq) <= slab.S and not slab.pending:
        # one round, some misses: compute the resident experts' pairs on the GPU while the misses are read
        res = uniq[slab.slot_of[uniq] >= 0]
        miss = uniq[slab.slot_of[uniq] < 0]
        futs = slab.start_loads(uniq)  # assigns slots to misses, starts their reads
        parts = []
        sel_r = np.nonzero(np.isin(flat, res))[0] if len(res) else np.zeros(0, np.int64)
        if len(sel_r):
            yr = pair_fn(slab, x, tok[sel_r], flat[sel_r], slab.slot_of[flat[sel_r]].astype(np.int32), sorted_)
            mx.async_eval(yr)
            parts.append((sel_r, yr))
        if nxt is not None:
            prefetch_next()
        for f in futs:
            slab.store.bytes_read += f.result()
        sel_m = np.nonzero(np.isin(flat, miss))[0]
        ym = pair_fn(slab, x, tok[sel_m], flat[sel_m], slab.slot_of[flat[sel_m]].astype(np.int32), sorted_)
        parts.append((sel_m, ym))
        perm = np.concatenate([p for p, _ in parts])
        y = mx.concatenate([v for _, v in parts], axis=0) if len(parts) > 1 else parts[0][1]
        inv = np.empty_like(perm)
        inv[perm] = np.arange(len(perm))
        out = (y[mx.array(inv.astype(np.int32))].reshape(T, k, -1) * wts[..., None]).sum(axis=1)
        mx.async_eval(out)
        return out
    need = sorted(uniq, key=lambda e: (slab.slot_of[e] < 0, e))  # residents first: fewest reads
    rounds = [need[i:i + slab.S] for i in range(0, len(need), slab.S)]
    outs, order = [], []
    for n, r in enumerate(rounds):
        slab.ensure(r)
        if n == 0 and nxt is not None:  # overlap the next layer's reads with this layer's compute
            prefetch_next()
        sel = np.nonzero(np.isin(flat, r))[0]
        eid = flat[sel]
        y = pair_fn(slab, x, tok[sel], eid, slab.slot_of[eid].astype(np.int32), sorted_)
        if n + 1 < len(rounds):
            mx.eval(y)  # the next round overwrites slots this one read
        outs.append(y)
        order.append(sel)
    y = outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=0)
    perm = np.concatenate(order)
    if not np.array_equal(perm, np.arange(len(flat))):
        inv = np.empty_like(perm)
        inv[perm] = np.arange(len(perm))
        y = y[mx.array(inv.astype(np.int32))]
    out = (y.reshape(T, k, -1) * wts[..., None]).sum(axis=1)
    mx.async_eval(out)
    return out


def sorted_pairs(x, tok, slots, sorted_):
    """Helpers for pair_fn: (rows x[tok] shaped for gather_*mm, rhs slot ids, permutation back or None)."""
    if sorted_:
        order = np.argsort(slots, kind="stable")
        inv = mx.array(np.argsort(order).astype(np.int32))
        o = mx.array(order.astype(np.int32))
        return x[mx.array(tok.astype(np.int32))[o]][:, None, :], mx.array(slots[order]), order, inv
    return x[_mxi(tok)][:, None, None, :], _mxi(slots)[:, None], None, None


def _mxi(a):
    """int32 MLX index array from a host (numpy) or device (MLX) index array."""
    return a.astype(mx.int32) if isinstance(a, mx.array) else mx.array(np.asarray(a).astype(np.int32))


class StreamingExperts:
    """gpt-oss routed experts (MXFP4, clamped SwiGLU, per-expert biases) on the shared streaming core."""

    def __init__(self, store: ExpertStore, layer: int, num_slots: int, gu_bias: mx.array, dn_bias: mx.array, cfg,
                 dtype):
        self.slab = ExpertSlab(store, layer, num_slots)
        self.store, self.S = store, self.slab.S
        self.limit, self.alpha = cfg.swiglu_limit, cfg.swiglu_alpha
        self.gu_bias, self.dn_bias = gu_bias.astype(dtype), dn_bias.astype(dtype)  # resident, by expert id
        mx.eval(self.gu_bias, self.dn_bias)
        self.prefetch_next = None  # (next StreamingExperts, router weight, router bias, routing input)
        self.kernel_max_pairs = 0
        self.dense = False

    @property
    def slot_of(self):
        return self.slab.slot_of

    def ensure(self, experts):
        self.slab.ensure(experts)

    def stats(self):
        return self.slab.stats()

    def arrays(self):
        return self.slab.arrays() + [self.gu_bias, self.dn_bias]

    def _pair_fn(self, slab, x, tok, eid, slots, sorted_):
        from .gpt_oss import _swiglu_oai

        q = dict(transpose=True, group_size=32, bits=4, mode="mxfp4")
        w = slab.w
        xs, sl, order, inv = sorted_pairs(x, tok, slots, sorted_)
        e = eid if order is None else eid[order]
        ei = _mxi(e)
        bshape = (slice(None), None, slice(None)) if sorted_ else (slice(None), None, None, slice(None))
        h = mx.gather_qmm(xs, w["gu_w"], w["gu_s"], None, rhs_indices=sl, sorted_indices=sorted_, **q) + \
            self.gu_bias[ei][bshape]
        h = _swiglu_oai(h, self.limit, self.alpha)
        y = mx.gather_qmm(h, w["dn_w"], w["dn_s"], None, rhs_indices=sl, sorted_indices=sorted_, **q) + \
            self.dn_bias[ei][bshape]
        return y[inv][:, 0, :] if sorted_ else y[:, 0, 0, :]

    def __call__(self, x: mx.array, idx: mx.array, wts: mx.array) -> mx.array:
        nxt = None
        if self.prefetch_next is not None:
            ne, rw, rb, x_route = self.prefetch_next
            nxt = (ne.slab, mx.addmm(rb, x_route, rw))
        return run_moe(self.slab, x, idx, wts, self._pair_fn, nxt)


class SafetensorsRows:
    """Row access to one bf16 tensor of a safetensors checkpoint through a read-only memory map (no residency).

    Several rows are read with parallel pread calls: rows that macOS evicted come from the SSD, and one at a time
    16 random lm_head rows took 2.58 ms median (p90 4.07), in parallel 0.38 ms (p90 2.57), on gpt-oss-20b; the GPU
    idles for that long on every decode token (RESULTS 1z). The bytes are the same either way."""

    # take() reads in parallel for this many rows (bulk loads: the memmap); MJ_ROWS_PARALLEL=0 for A/B runs
    PARALLEL_MIN, PARALLEL_MAX = (2, 256) if os.environ.get("MJ_ROWS_PARALLEL", "1") == "1" else (1 << 62, 0)

    def __init__(self, path: str, name: str | list[str]):
        import glob
        import struct

        names = [name] if isinstance(name, str) else list(name)
        for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
            with open(f, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                hdr = json.loads(fh.read(n))
            for nm in names:
                if nm in hdr:
                    info = hdr[nm]
                    assert info["dtype"] == "BF16", info["dtype"]
                    start, end = info["data_offsets"]
                    self.name, self.file = nm, os.path.realpath(f)
                    self.offset = 8 + n + start
                    self.rows = np.memmap(os.path.realpath(f), dtype=np.uint16, mode="r", offset=self.offset,
                                          shape=tuple(info["shape"]))
                    self._fd = None
                    self._pool = None
                    return
        raise KeyError(names)

    def _read(self, ids: np.ndarray) -> np.ndarray:
        if not self.PARALLEL_MIN <= len(ids) <= self.PARALLEL_MAX:
            return np.ascontiguousarray(self.rows[ids])
        if self._pool is None:
            from concurrent.futures import ThreadPoolExecutor

            self._fd = os.open(self.file, os.O_RDONLY)
            self._pool = ThreadPoolExecutor(16, thread_name_prefix="mj-rows")
        width = self.rows.shape[1]
        nb = width * 2
        out = np.empty((len(ids), width), np.uint16)

        def rd(j):
            b = os.pread(self._fd, nb, self.offset + int(ids[j]) * nb)
            out[j] = np.frombuffer(b, np.uint16)

        list(self._pool.map(rd, range(len(ids))))
        return out

    def read(self, ids: np.ndarray) -> np.ndarray:
        """The rows as raw bf16 bits (uint16 [n, width]), on the host."""
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        if ids.size and (ids.min() < 0 or ids.max() >= self.rows.shape[0]):
            raise IndexError(f"row ids out of range [0, {self.rows.shape[0]})")
        return self._read(ids)

    def take(self, ids: np.ndarray, dtype) -> mx.array:
        a = mx.array(self.read(ids)).view(mx.bfloat16)
        return a if dtype == mx.bfloat16 else a.astype(dtype)


# ---- GPU-driven streaming (C++ extension ext/_mj_stream, prototype) -------------------------------------------------
# MJ_GPU_STREAM=1: decode steps route through the C++ Streamer. The GPU signals a Metal shared event after each
# layer's router; a C++ IO thread reads the routing from unified memory, makes the experts resident in the same slab
# arrays (preadv), writes the slot ids and signals back; the GPU waits on that event before the expert matmul. No host
# sync in the step. Long steps (prefill) keep the Python path; slot ownership moves between the two (handoff below).
GPU_STREAM = os.environ.get("MJ_GPU_STREAM") == "1"
# While a layer's expert reads are in flight, a tiny spin kernel on a separate queue keeps the GPU from dropping its
# clocks (kernels after a 2-5 ms idle run 3-4x slower). RESULTS 1u: +16% decode on the streamed 35B.
KEEPALIVE = os.environ.get("MJ_KEEPALIVE", "1") == "1"


def load_gpu_streamer():
    import sys

    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ext", "build")
    if here not in sys.path:
        sys.path.insert(0, here)
    import _mj_stream  # built by ext/CMakeLists.txt (nanobind domain "mlx", linked to this venv's libmlx)

    return _mj_stream


class GpuStreaming:
    """Owns a C++ Streamer for the slabs of one model; hands slot ownership between it and the Python slabs."""

    def __init__(self, store: ExpertStore, slabs: list[ExpertSlab], threads: int = 8):
        self.ms = load_gpu_streamer()
        # {layer: slab}; a list means every layer is MoE (models with dense layers pass the MoE layers only)
        self.store, self.slabs = store, (dict(enumerate(slabs)) if isinstance(slabs, list) else dict(slabs))
        self.cpp = None  # created on the first decode step (after prefill filled the Python slabs)
        self.owner = "py"

    def to_cpp(self):
        if self.owner == "cpp":
            return
        if self.cpp is None:
            self.cpp = self.ms.Streamer(self.store.path, self.store.stride, 8)
            self.cpp.keepalive = KEEPALIVE
        for li, s in self.slabs.items():
            s._wait()  # Python prefetch reads land first
            self.cpp.add_layer(li, s.E, [s.w[p["name"]] for p in self.store.parts], [int(v) for v in s.slot_of])
        self.owner = "cpp"

    def to_py(self):
        if self.owner == "py":
            return
        for li, s in self.slabs.items():
            so = np.array(self.cpp.slot_of(li), np.int64)
            s.slot_of[:] = so
            s.expert_of[:] = -1
            res = np.nonzero(so >= 0)[0]
            s.expert_of[so[res]] = res
            s.lru = OrderedDict((int(e), None) for e in res)
            s.pending, s.prefetched = {}, set()
        self.owner = "py"

    def route(self, li: int, idx: mx.array, preds: list) -> mx.array:
        self.to_cpp()
        return self.cpp.route(li, idx, preds)

    def stats(self) -> dict:
        return dict(self.cpp.stats()) if self.cpp is not None else {}
