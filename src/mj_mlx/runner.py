# Copyright 2026 the MilliJoule authors
# SPDX-License-Identifier: Apache-2.0
"""MLX model runner and a small engine loop around mj_mlx.scheduler.Scheduler.

The runner owns the paged KV pool (one [num_blocks * block_size, n_kv, head_dim] array per layer for K and V)
and addresses it only through the scheduler's block tables. One `forward` is one scheduler step: all items
(prefill chunks and decodes, any length) run as one flat token batch, and greedy tokens come back for the
sampled rows with a single host sync per step.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from . import scheduler

from .gpt_oss import GptOss, build_meta
from .sampler import sample

# Lean gpt-oss under overlap: build step N+1 while step N runs, fill its embedding rows once N's tokens are on the
# host, then submit (RESULTS 1z). MJ_LEAN_DEFER=0 keeps the older path (sync on N's tokens before building N+1).
LEAN_DEFER = __import__("os").environ.get("MJ_LEAN_DEFER", "1") == "1"
# With the deferred path, also queue step N+1 on the GPU before step N's token is known: its kernels wait on a Metal
# shared event (C++ HostGate, ext/mj_stream.cpp) that the host signals after writing the embedding row, so encoding
# N+1 overlaps step N instead of following it: gpt-oss-20b decode +5.6-7.5% (8-bit attention) and +11.5-17.6%
# (exact), same tokens (RESULTS 1zc). MJ_LEAN_GATE=0: off; also off when the extension is not built.
LEAN_GATE = __import__("os").environ.get("MJ_LEAN_GATE", "1") == "1"
GATE_BUFFER_OPS = GATE_BUFFER_MB = 1 << 30  # gated steps: one command buffer each
PAGE = 16384


def _host_buffer(shape, dtype=np.uint16):
    """A page-aligned host array that mx.from_dlpack wraps without copying, so writes after building a graph on
    it are seen when the graph runs. Returns (array, the mmap that owns the memory)."""
    import mmap

    n = int(np.prod(shape))
    nbytes = max(PAGE, -(-n * np.dtype(dtype).itemsize // PAGE) * PAGE)
    mm = mmap.mmap(-1, nbytes)
    return np.frombuffer(mm, dtype, count=n).reshape(shape), mm


@dataclass
class Deferred:
    """A lean step built on a host embedding buffer; submitted (armed) once its pending inputs are known."""

    s: "StepArrays"
    xbuf: np.ndarray
    mm: object
    pend: np.ndarray
    idx: np.ndarray | None
    out: mx.array | None = None  # prefill-only steps: the hidden states (KV writes)
    cand: mx.array | None = None
    x32: mx.array | None = None
    armed: bool = False
    gate_v: int = 0  # > 0: submitted behind this HostGate value; arm() signals it


@dataclass
class StepArrays:
    """The step in flat arrays (the shape of the planned Rust StepBatch)."""

    token_ids: np.ndarray  # [T] uint32
    positions: np.ndarray  # [T] int32
    request_ids: np.ndarray  # [N] uint64
    starts: np.ndarray  # [N] int32
    lens: np.ndarray  # [N] int32
    sample: np.ndarray  # [N] bool
    block_tables: np.ndarray  # [N, max_blocks] int32, -1 padded
    sample_rows: np.ndarray  # [S] int32
    sample_request_ids: np.ndarray  # [S] uint64
    released: list
    slot_mapping: np.ndarray | None = None  # [T] int32 KV write slots (from the scheduler when available)
    temperature: np.ndarray | None = None  # [N] per item (0 = greedy)
    top_p: np.ndarray | None = None
    top_k: np.ndarray | None = None
    seed: np.ndarray | None = None
    state_slots: np.ndarray | None = None  # [N] int32 recurrent-state slot per item (hybrid models)
    prev_sample_idx: np.ndarray | None = None  # [T] int32 overlap: k = input is the previous step's sampled token k

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)


def step_arrays(plan) -> StepArrays:
    """Accepts the scheduler's StepBatch (if the bindings have it) or the list of 6-tuples."""
    if hasattr(plan, "num_items"):
        d = scheduler.as_numpy(plan)
        return StepArrays(
            d["token_ids"], d["positions"], d["request_ids"], d["starts"], d["lens"], d["sample"].astype(bool),
            d["block_tables"], d["sample_rows"], d["sample_request_ids"], list(plan.released), d["slot_mapping"],
            d["temperature"], d["top_p"], d["top_k"], d["seed"], d.get("state_slots"), d.get("prev_sample_idx"),
        )
    n = len(plan)
    lens = np.fromiter((len(it[2]) for it in plan), np.int32, n)
    starts = np.fromiter((it[1] for it in plan), np.int32, n)
    rids = np.fromiter((it[0] for it in plan), np.uint64, n)
    sample = np.fromiter((it[4] for it in plan), bool, n)
    token_ids = np.fromiter((t for it in plan for t in it[2]), np.uint32, int(lens.sum()))
    cu = np.concatenate([[0], np.cumsum(lens)])
    positions = (np.repeat(starts - cu[:-1], lens) + np.arange(cu[-1])).astype(np.int32)
    mb = max(len(it[3]) for it in plan)
    bt = np.full((n, mb), -1, np.int32)
    for i, it in enumerate(plan):
        bt[i, : len(it[3])] = it[3]
    srows = (cu[1:] - 1)[sample].astype(np.int32)
    return StepArrays(token_ids, positions, rids, starts, lens, sample, bt, srows, rids[sample], [])


def load_model(path: str, dtype=mx.bfloat16, dense_experts: bool = False, **kw):
    """Pick the model class from config.json's model_type."""
    import json
    import os

    mt = json.load(open(os.path.join(path, "config.json"))).get("model_type")
    # placement: "auto" | "resident" | "mapped" | "stream": where the routed experts live, chosen by memory
    # (placement.py); the expert files are written on first use. Explicit map_experts / stream_experts /
    # expert_slots arguments win over the placement's.
    where = kw.pop("placement", None)
    if where and not kw.get("map_experts") and not kw.get("stream_experts"):
        from . import placement

        where = placement.choose(path) if where == "auto" else where
        if where == "mapped" and mt.startswith("qwen3_5") and not placement.experts_quantized(path):
            where = "stream"  # Qwen35Moe maps quantized (affine) experts only; bf16 experts stream instead
        for k, v in placement.prepare(path, where).items():
            kw.setdefault(k, v)
    if mt in ("qwen3_5_moe", "qwen3_5_moe_text", "qwen3_5", "qwen3_5_text"):
        from .qwen3_5_moe import Qwen35Moe

        return _calibrate(Qwen35Moe(path, dtype=dtype, **kw), path, kw)
    if mt == "gpt_oss":
        return GptOss(path, dtype=dtype, dense_experts=dense_experts, **kw)
    from .moe_lm import FAMILIES, MoeLM

    if mt in FAMILIES:  # the general MoE model (moe_lm.py, docs/moe-design.md)
        if kw.pop("auto_dtype", True) and dtype == mx.bfloat16:
            from .placement import checkpoint_dtype

            dtype = checkpoint_dtype(path)  # quantized checkpoints run in their scales' dtype (as mlx-lm)
        return _calibrate(MoeLM(path, dtype=dtype, **kw), path, kw)
    supported = sorted({"gpt_oss", "qwen3_5_moe", "qwen3_5", *FAMILIES})
    raise ValueError(f"unsupported model_type {mt!r} in {path}/config.json. mj-mlx runs these MoE families: "
                     f"{', '.join(supported)}. For other models (dense models included) use mlx-lm "
                     "(`pip install mlx-lm`); docs/models.md lists what is tested.")


def _calibrate(model, path: str, kw: dict):
    """MJ_SLOT_PROFILE_AUTO=1 (opt-in until checked live): a streamed model whose store has no slot profile records
    one from a short calibration (placement.calibrate_slot_profile, once per store); this load keeps uniform slots."""
    import os

    store = kw.get("stream_experts")
    if os.environ.get("MJ_SLOT_PROFILE_AUTO") == "1" and store:
        from . import placement

        store = os.path.expanduser(store)
        if not os.path.exists(os.path.join(store, placement.PROFILE)):
            from transformers import AutoTokenizer

            placement.calibrate_slot_profile(model, AutoTokenizer.from_pretrained(path), store)
    return model


class MlxRunner:
    def __init__(self, model: GptOss, block_size: int, num_blocks: int, paged_decode: bool = True,
                 num_state_slots: int = 0):
        self.model = model
        model.block_size = block_size
        # the paged decode kernel needs head_dim % 32 == 0; other models decode through gather + SDPA
        self.paged_decode = paged_decode and model.cfg.head_dim % 32 == 0
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.recurrent = getattr(model, "recurrent", False)
        self._state_slot: dict[int, int] = {}  # request id -> recurrent-state slot (from the steps' state_slots)
        if self.recurrent:
            self.pool = model.new_pools(num_blocks * block_size, num_state_slots)
        else:
            self.pool = model.new_kv_pool(num_blocks * block_size)

    def _pool_arrays(self):
        return self.model.pool_arrays(self.pool) if self.recurrent else self.pool

    @staticmethod
    def free_bytes(model: GptOss, mem_fraction: float = 0.7, reserve_bytes: int = 1 << 30) -> int:
        """GPU working-set budget left after the weights and an activation reserve."""
        info = mx.device_info()
        budget = int(info.get("max_recommended_working_set_size", info["memory_size"] * 0.66) * mem_fraction)
        weights = sum(a.nbytes for a in model.parameters())
        return max(budget - weights - reserve_bytes, 0)

    @staticmethod
    def kv_capacity(model: GptOss, block_size: int, mem_fraction: float = 0.7, reserve_bytes: int = 1 << 30,
                    max_blocks: int | None = None, other_bytes: int = 0) -> int:
        """Blocks that fit after the weights, an activation reserve and `other_bytes` (e.g. recurrent state)."""
        per_block = model.kv_bytes_per_token() * block_size
        n = max((MlxRunner.free_bytes(model, mem_fraction, reserve_bytes) - other_bytes) // per_block, 0)
        return int(min(n, max_blocks) if max_blocks else n)

    @staticmethod
    def state_capacity(model, max_running: int, state_share: float = 0.5, mem_fraction: float = 0.7,
                       reserve_bytes: int = 1 << 30) -> int:
        """Recurrent-state slots for hybrid models, sized from memory like the KV pool: at most `state_share` of
        the free budget (the rest is KV), never more than max_running, at least 1. Admission stops when no slot
        is free, so this also caps the running set. (Real Qwen3.6 state is roughly 31 MB per request in bf16 and
        63 MB in float32, by the plan's estimate: 512 requests would be 16-32 GB.)"""
        per = model.state_bytes_per_slot()
        n = int(MlxRunner.free_bytes(model, mem_fraction, reserve_bytes) * state_share) // max(per, 1)
        return max(1, min(max_running, n))

    def forward(self, s: StepArrays) -> np.ndarray:
        """Greedy tokens (uint32) for the sampled rows, in sample_rows order."""
        toks = self.launch(s)
        return np.zeros(0, np.uint32) if toks is None else np.array(toks)

    def launch(self, s: StepArrays, prev_tokens: mx.array | None = None) -> mx.array | None:
        """Build the step and start it on the GPU (async_eval) without waiting. Returns the lazy sampled tokens
        (uint32, sample_rows order), or None for a prefill-only step. With overlap scheduling, inputs marked in
        `s.prev_sample_idx` are taken from `prev_tokens` (the previous step's lazy output) on the device, so this
        step can be queued before the previous one finishes. Steps must be launched in schedule order."""
        if self.recurrent and s.state_slots is not None:
            self._state_slot.update(zip(s.request_ids.tolist(), s.state_slots.tolist()))
        if self.recurrent:
            meta = self.model.prepare(s, self.block_size, self.paged_decode)
        else:
            meta = build_meta(
                s.token_ids, s.positions, s.starts, s.lens, s.block_tables, s.sample_rows, self.block_size,
                self.model.cfg.sliding_window, s.slot_mapping, self.paged_decode,
            )
        if s.prev_sample_idx is not None and (s.prev_sample_idx >= 0).any():
            self._resolve_inputs(meta, s.prev_sample_idx, prev_tokens)
        if len(s.sample_rows) == 0:  # prefill chunks only: write KV / state, no logits
            mx.async_eval(self.model.hidden(meta, self.pool), self._pool_arrays())
            return None
        x32 = self.model.final_hidden(meta, self.pool)
        if s.temperature is None or not (s.temperature[s.sample] > 0).any():
            tokens = self.model.greedy_from(x32).astype(mx.uint32)
        else:
            sm = s.sample
            pos = (s.starts + s.lens)[sm]  # position of the token being sampled
            tokens = sample(self.model.full_logits(x32), s.temperature[sm], s.top_p[sm], s.top_k[sm], s.seed[sm],
                            pos)
        mx.async_eval(tokens, self._pool_arrays())
        return tokens

    def deferred_ok(self) -> bool:
        m = self.model
        return (LEAN_DEFER and not self.recurrent and getattr(m, "embed", 1) is None
                and getattr(m, "embed_rows", None) is not None and getattr(m, "lm_head", 1) is None
                and hasattr(m, "rescore_host"))

    def launch_deferred(self, s: StepArrays) -> Deferred:
        """Build a lean step on a host embedding buffer. Rows with known tokens are filled now; rows that take the
        previous step's token (s.prev_sample_idx >= 0) are filled by arm(). Submitted now if nothing is pending."""
        m = self.model
        meta = build_meta(
            s.token_ids, s.positions, s.starts, s.lens, s.block_tables, s.sample_rows, self.block_size,
            m.cfg.sliding_window, s.slot_mapping, self.paged_decode,
        )
        T = len(s.token_ids)
        idx = s.prev_sample_idx
        pend = (idx >= 0) if idx is not None else np.zeros(T, bool)
        xbuf, mm = _host_buffer((T, m.cfg.hidden_size))
        if (~pend).any():
            xbuf[~pend] = m.embed_rows.read(np.asarray(s.token_ids)[~pend])
        x = mx.from_dlpack(xbuf).view(mx.bfloat16)
        gv = 0
        if pend.any() and self._gate() is not None:
            gv = self._gate().next()
            x = self._gate().gate(x, gv)
        if m.dtype != mx.bfloat16:
            x = x.astype(m.dtype)
        h = m.hidden(meta, self.pool, x=x)
        d = Deferred(s, xbuf, mm, pend, idx, gate_v=gv)
        if len(s.sample_rows) == 0:
            d.out = h
        else:
            if s.temperature is not None and (s.temperature[s.sample] > 0).any():
                raise NotImplementedError("lean mode keeps lm_head on disk: greedy only")
            x32 = mx.fast.rms_norm(h[meta.sample_rows].astype(mx.float32), m.norm.astype(mx.float32),
                                   m.cfg.rms_norm_eps)
            d.x32, d.cand = x32, m.head_candidates(x32)
        if gv:  # queue it now; the GPU waits at the gate until arm() writes the pending rows and signals.
            # One command buffer for the step: MLX's eval waits while ~10 buffers are in flight, and buffers behind a
            # closed gate cannot finish, so at MLX_MAX_MB_PER_BUFFER=150 (~50 buffers per gpt-oss step) it deadlocked
            # (GPU timeout). Prefill keeps the small buffers (large ones: 280-330 vs 430 tok/s, RESULTS 1zc).
            ext = self._gate_ext
            limits = ext.buffer_limits()
            ext.set_buffer_limits(GATE_BUFFER_OPS, GATE_BUFFER_MB)
            try:
                mx.async_eval(*([d.out] if d.cand is None else [d.cand, d.x32]), self._pool_arrays())
            finally:
                ext.set_buffer_limits(*limits)
        elif not pend.any():
            self.arm(d, None)
        return d

    def _gate(self):
        if not LEAN_GATE:
            return None
        if getattr(self, "_host_gate", None) is None:
            if getattr(self, "_gate_missing", False):
                return None
            try:
                from .expert_stream import load_gpu_streamer

                self._gate_ext = load_gpu_streamer()
                self._host_gate = self._gate_ext.HostGate()
            except (ImportError, AttributeError):  # extension not built (or an old build): ungated deferred steps
                self._gate_missing = True
                return None
        return self._host_gate

    def release_gates(self) -> None:
        """Open every gate issued so far (an abandoned step must never leave the GPU waiting)."""
        g = getattr(self, "_host_gate", None)
        if g is not None:
            g.release_all()

    def __del__(self):
        self.release_gates()

    def arm(self, d: Deferred, prev_out: np.ndarray | None) -> None:
        """Fill the pending embedding rows from the previous step's host tokens and submit the step."""
        if d.armed:
            return
        if d.pend.any():
            if prev_out is None:
                raise RuntimeError("step has pending inputs but no previous step's tokens were given")
            d.xbuf[d.pend] = self.model.embed_rows.read(prev_out.astype(np.int64)[d.idx[d.pend]])
        if d.gate_v:
            self._host_gate.signal(d.gate_v)
        else:
            mx.async_eval(*([d.out] if d.cand is None else [d.cand, d.x32]), self._pool_arrays())
        d.armed = True

    def finish(self, d: Deferred) -> np.ndarray:
        """Wait for an armed step; greedy tokens (uint32, sample_rows order), chosen on the host."""
        if not d.armed:
            raise RuntimeError("finish() of a step that was never submitted")
        if d.cand is None:
            return np.zeros(0, np.uint32)
        return self.model.rescore_host(np.array(d.cand), np.array(d.x32))

    def _resolve_inputs(self, meta, idx: np.ndarray, prev_tokens: mx.array | None) -> None:
        """ids = where(idx >= 0, prev_tokens[max(idx, 0)], token_ids). On the device when the embedding is on
        the device; a host-side embedding (lean mode) needs the ids on the host, so that waits for the previous
        step (correct, just not overlapped)."""
        if prev_tokens is None:
            raise RuntimeError("step has pending inputs but no previous step's tokens were given")
        m = meta.attn if hasattr(meta, "attn") else meta
        if getattr(self.model, "embed", None) is None and getattr(getattr(self.model, "embed_lin", None), "q",
                                                                    None) is None:
            ids = np.array(m.token_ids)
            pend = idx >= 0
            ids[pend] = np.array(prev_tokens).astype(np.int32)[idx[pend]]
            m.token_ids = mx.array(ids)
            return
        i = mx.array(idx.astype(np.int32))
        m.token_ids = mx.where(i >= 0, prev_tokens.astype(mx.int32)[mx.maximum(i, 0)], m.token_ids)

    def release(self, request_ids, reason: str = "finished") -> None:
        """Paged KV is owned by the scheduler's allocator: nothing per request to drop for gpt-oss. Recurrent models
        drop the released requests' states (so they stop pinning memory until the slot is reused)."""
        if not self.recurrent or not hasattr(self.model, "release_state_slots"):
            return
        slots = [self._state_slot.pop(int(r)) for r in request_ids if int(r) in self._state_slot]
        if slots:
            self.model.release_state_slots(self.pool, slots)


_WIRED = [None]


def _set_wired_limit(n: int) -> None:
    """mx.set_wired_limit only when the value changes: re-setting it per request (the bench tools build an Engine
    per request) cost OLMoE-1B-7B ~10% of a 512-token prefill after an idle gap (RESULTS 1zd)."""
    if _WIRED[0] != n:
        mx.set_wired_limit(n)
        _WIRED[0] = n


class Engine:
    """Scheduler + runner. Greedy decoding."""

    def __init__(self, model_path: str | None = None, *, model: GptOss | None = None, dtype=mx.bfloat16,
                 block_size: int = 16, num_blocks: int | None = None, max_running: int = 256,
                 max_batch_tokens: int = 4096, max_prefill_chunk: int | None = None, max_seq_len: int = 32768,
                 enable_prefix_cache: bool = True, dense_experts: bool = False, paged_decode: bool = True,
                 cache_limit_bytes: int = 1 << 30, num_state_slots: int | None = None, overlap: bool = False,
                 wired_limit_bytes: int | None = None, prefill_area_cap: int | None = None):
        # Wire weights and KV like mlx-lm does, so macOS cannot page them out under memory pressure. Memory-mapped
        # experts need a smaller limit (wired_limit_bytes; RESULTS 1j: at the recommended limit Metal runs out).
        rec = wired_limit_bytes if wired_limit_bytes is not None else mx.device_info().get("max_recommended_working_set_size")
        if (wired_limit_bytes is None and model is not None and getattr(model, "map", None) is not None
                and os.environ.get("MJ_WIRED_RAISE") != "1"):
            # an already-loaded model with memory-mapped experts: go straight to their 1000 MB limit. Raising it to
            # the recommended size first, even briefly, wires every resident buffer, the mapped experts' no-copy
            # buffers included (Metal wires whole buffers): 2-3 s of paging on gpt-oss-20b before the first request
            # (v17 load profile). MJ_WIRED_RAISE=1 restores the old order, for A/B runs.
            rec = 1000 << 20
        if rec is not None:
            _set_wired_limit(rec)
        # Bound MLX's freed-buffer cache: prefill steps make large temporaries of varying size, and an unbounded
        # cache of wired buffers pushes a 16 GB Mac into swap (seen: GPU idle at 6 W, steps of 1-9 s).
        mx.set_cache_limit(cache_limit_bytes)
        self.model = model or load_model(model_path, dtype=dtype, dense_experts=dense_experts)
        if wired_limit_bytes is None and getattr(self.model, "map", None) is not None:
            # memory-mapped experts: MLX's wired set must stay small (RESULTS 1j: at the recommended limit Metal runs
            # out of memory). 1000 MB: at 3000 MB gpt-oss-20b logged ~1000 Metal "required > max" lines per run and
            # macOS once killed the exact arm; at 1000 MB none, for the exact and the 8-bit arm, at the same speed
            # (RESULTS 1zi, 1zk), and an hour-long soak ran clean
            _set_wired_limit(1000 << 20)
        recurrent = getattr(self.model, "recurrent", False)
        extra = {}
        state_bytes = 0
        if recurrent:  # per-request recurrent state; the radix cache cannot restore it (yet)
            if num_state_slots is None:  # sized from memory, not from max_running
                num_state_slots = MlxRunner.state_capacity(self.model, max_running)
            state_bytes = num_state_slots * self.model.state_bytes_per_slot()
            enable_prefix_cache = False
            extra["num_state_slots"] = num_state_slots
        if num_blocks is None:
            num_blocks = MlxRunner.kv_capacity(self.model, block_size, max_blocks=1 << 16, other_bytes=state_bytes)
        self.runner = MlxRunner(self.model, block_size, num_blocks, paged_decode, num_state_slots or 0)
        self.sched = scheduler.Scheduler(
            block_size=block_size, num_blocks=num_blocks, max_running=max_running,
            max_batch_tokens=max_batch_tokens, max_prefill_chunk=max_prefill_chunk or max_batch_tokens,
            prefill_area_cap=self._area_cap(prefill_area_cap),
            max_seq_len=max_seq_len, enable_prefix_cache=enable_prefix_cache, **extra,
            **({"overlap": True} if overlap else {}),
        )
        # Overlap scheduling: step N+1 is planned and queued on the GPU before step N's tokens reach the host,
        # so the host work of planning and building a step hides behind the GPU. Same tokens; stop tokens are
        # seen one step late (one discarded sample per stopping request).
        self.overlap = overlap
        self._in_flight: list[tuple[StepArrays, mx.array | None]] = []
        eos = self.model.cfg.eos_token_id
        self.eos = [eos] if isinstance(eos, int) else list(eos or [])
        self.steps = 0
        self.tokens_out = 0
        self.last_sampled = np.zeros(0, np.uint64)  # request ids that got a token in the last completed step

    def _area_cap(self, cap: int | None) -> int:
        """Prefill attention area cap (chunk x context, scheduler `prefill_area_cap`): an explicit value, else
        MJ_PREFILL_AREA_CAP, else the model's own (`prefill_area_cap()`, for models whose prefill attention memory
        grows with the area), else 0 (off)."""
        if cap is not None:
            return int(cap)
        env = os.environ.get("MJ_PREFILL_AREA_CAP")
        if env is not None:
            return int(env)
        fn = getattr(self.model, "prefill_area_cap", None)
        return int(fn()) if fn else 0

    def add_request(self, prompt_ids, max_new_tokens: int = 128, ignore_eos: bool = False, stop_token_ids=None,
                    **sampling):
        """sampling: temperature, top_p, top_k, seed (passed to the scheduler, which puts them in the StepBatch)."""
        stops = self.eos if stop_token_ids is None else list(stop_token_ids)
        return self.sched.add_request(list(prompt_ids), max_new_tokens=max_new_tokens, stop_token_ids=stops,
                                      ignore_eos=ignore_eos, **sampling)

    def step(self):
        if self.overlap:
            return self._step_overlap()
        plan = self.sched.schedule()
        s = step_arrays(plan)
        if s.released:
            self.runner.release([r for r, _ in s.released])
        toks = self.runner.forward(s)
        self.last_sampled = s.sample_request_ids  # request ids that got a token this step (for TTFT)
        if s.num_tokens > len(s.lens):  # a prefill chunk ran: drop its temporaries (as mlx-lm does)
            mx.clear_cache()
        self.steps += 1
        self.tokens_out += len(toks)
        if hasattr(self.sched, "update_sampled"):
            return self.sched.update_sampled(toks.astype(np.uint32).tobytes())
        return self.sched.update(list(zip(s.sample_request_ids.tolist(), toks.tolist())))

    def _step_overlap(self):
        """Schedule and launch step N+1 (its pending inputs read step N's lazy tokens), then wait for step N and
        complete it. An empty plan with a step in flight just completes the oldest step."""
        if self.runner.deferred_ok():
            return self._step_overlap_deferred()
        plan = self.sched.schedule()
        if plan.num_items:
            s = step_arrays(plan)
            if s.released:
                self.runner.release([r for r, _ in s.released])
            prev = self._in_flight[-1][1] if self._in_flight else None
            self._in_flight.append((s, self.runner.launch(s, prev)))
        if not self._in_flight or (len(self._in_flight) < 2 and plan.num_items):
            self.last_sampled = np.zeros(0, np.uint64)  # nothing completed this call
            return []
        s, toks = self._in_flight.pop(0)
        out = np.zeros(0, np.uint32) if toks is None else np.array(toks)  # waits for step N only
        self.last_sampled = s.sample_request_ids
        if s.num_tokens > len(s.lens):
            mx.clear_cache()
        self.steps += 1
        self.tokens_out += len(out)
        return self.sched.update_sampled(out.astype(np.uint32).tobytes())

    def _step_overlap_deferred(self):
        """Lean overlap: build step N+1 on a host embedding buffer while step N runs on the GPU, wait for N's
        tokens, fill N+1's pending rows and submit it, then complete N. The GPU waits only for the host's
        candidate readback + rescore + one embedding row, not for building a step (RESULTS 1z)."""
        plan = self.sched.schedule()
        if plan.num_items:
            s = step_arrays(plan)
            if s.released:
                self.runner.release([r for r, _ in s.released])
            self._in_flight.append((s, self.runner.launch_deferred(s)))
        if not self._in_flight or (len(self._in_flight) < 2 and plan.num_items):
            self.last_sampled = np.zeros(0, np.uint64)
            return []
        s, d = self._in_flight.pop(0)
        out = self.runner.finish(d)
        if self._in_flight:
            self.runner.arm(self._in_flight[0][1], out)
        self.last_sampled = s.sample_request_ids
        if s.num_tokens > len(s.lens):
            mx.clear_cache()
        self.steps += 1
        self.tokens_out += len(out)
        return self.sched.update_sampled(out.astype(np.uint32).tobytes())

    def has_work(self) -> bool:
        return self.sched.has_work()

    def generate(self, prompts, max_new_tokens=128, ignore_eos=False, check=False, **sampling) -> list[list[int]]:
        rids = [self.add_request(p, max_new_tokens, ignore_eos, **sampling) for p in prompts]
        out = {}
        while self.has_work():
            for f in self.step():
                out[f[0]] = list(f[2])
            if check:
                self.sched.check_invariants()
        return [out[r] for r in rids]


def timed(fn):
    t = time.perf_counter()
    r = fn()
    return r, time.perf_counter() - t
