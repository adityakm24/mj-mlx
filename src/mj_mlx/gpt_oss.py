# Copyright © 2023-2025 Apple Inc. (mlx-lm, MIT License)
# Copyright © 2026 the MilliJoule authors. Licensed under the Apache License 2.0 (LICENSE); see NOTICE.
#
# Adapted from mlx-lm (https://github.com/ml-explore/mlx-lm): mlx_lm/models/gpt_oss.py, switch_layers.py,
# activations.py, rope_utils.py. MIT License:
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
#   documentation files (the "Software"), to deal in the Software without restriction, including without limitation
#   the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
#   to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above
#   copyright notice and this permission notice shall be included in all copies or substantial portions of the
#   Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED.
#
# Changes from mlx-lm: the model runs on one flat varlen token batch per step (prefill chunks and decodes mixed)
# with a paged KV pool addressed through MilliJoule's block tables instead of per-request KV caches; YaRN RoPE
# honours `rope_scaling.truncate` (mlx-lm always truncates, transformers does not for gpt-oss); the YaRN
# magnitude scale is folded into the attention scale; q/k/v are one fused projection; gate/up experts are one
# fused MXFP4 projection; logits are computed only for sampled rows.
"""gpt-oss (GptOssForCausalLM) for the MLX backend."""

from __future__ import annotations

import glob
import json
import math
import os
import time
from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np

from . import bf16_pack
from .moe_kernels import grouped_mxfp4
from .paged_attention import paged_decode_attention


@dataclass
class GptOssConfig:
    hidden_size: int = 2880
    intermediate_size: int = 2880
    num_hidden_layers: int = 24
    num_local_experts: int = 32
    num_experts_per_tok: int = 4
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    head_dim: int = 64
    vocab_size: int = 201088
    rms_norm_eps: float = 1e-5
    sliding_window: int = 128
    rope_theta: float = 150000.0
    rope_scaling: dict | None = None
    layer_types: list[str] = field(default_factory=list)
    swiglu_limit: float = 7.0
    swiglu_alpha: float = 1.702
    eos_token_id: int | list[int] | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "GptOssConfig":
        names = {f for f in cls.__dataclass_fields__}
        c = cls(**{k: v for k, v in d.items() if k in names})
        if "rope_parameters" in d and not c.rope_scaling:
            c.rope_scaling = d["rope_parameters"]
        if not c.layer_types:
            c.layer_types = ["sliding_attention", "full_attention"] * (c.num_hidden_layers // 2)
        return c


def yarn_inv_freq(cfg: GptOssConfig) -> tuple[np.ndarray, float]:
    """Inverse frequencies and attention (magnitude) factor, as transformers' `_compute_yarn_parameters`."""
    rs = cfg.rope_scaling or {}
    dim, base = cfg.head_dim, float(cfg.rope_theta)
    pos_freqs = base ** (np.arange(0, dim, 2, dtype=np.float32) / dim)
    inv_extra = 1.0 / pos_freqs
    if rs.get("rope_type", rs.get("type")) != "yarn":
        return inv_extra.astype(np.float32), 1.0
    factor = float(rs["factor"])
    orig = rs.get("original_max_position_embeddings", 4096)
    beta_fast, beta_slow = rs.get("beta_fast", 32.0), rs.get("beta_slow", 1.0)
    truncate = rs.get("truncate", True)
    mscale = rs.get("attention_factor") or (0.1 * math.log(factor) + 1.0 if factor > 1 else 1.0)

    def corr_dim(rot):
        return (dim * math.log(orig / (rot * 2 * math.pi))) / (2 * math.log(base))

    low, high = corr_dim(beta_fast), corr_dim(beta_slow)
    if truncate:
        low, high = math.floor(low), math.ceil(high)
    low, high = max(low, 0), min(high, dim - 1)
    if low == high:
        high += 0.001
    ramp = np.clip((np.arange(dim // 2, dtype=np.float32) - low) / (high - low), 0, 1)
    extra_factor = 1 - ramp
    inv = (inv_extra / factor) * (1 - extra_factor) + inv_extra * extra_factor
    return inv.astype(np.float32), float(mscale)


@mx.compile
def _swiglu_oai(gate_up: mx.array, limit: float, alpha: float) -> mx.array:
    inter = gate_up.shape[-1] // 2
    gate, up = gate_up[..., :inter], gate_up[..., inter:]
    gate = mx.minimum(gate, limit)
    up = mx.clip(up, -limit, limit)
    return (up + 1.0) * (gate * mx.sigmoid(alpha * gate))


# Decode glue in fewer kernels (MJ_FUSE_GLUE=0: the separate ops). Each op keeps its dtype inside the compiled
# kernel, so the numbers are the same (tests/mlx/test_kv_fused.py checks the hidden states bitwise).
FUSE_GLUE = os.environ.get("MJ_FUSE_GLUE", "0") == "1"  # off: slower on gpt-oss-20b (RESULTS 2026-11-19)
# Prefill MoE glue in two kernels (Experts._fused_sorted; MJ_PREFILL_GLUE=0: the separate ops)
FUSED_PREFILL_GLUE = os.environ.get("MJ_PREFILL_GLUE", "1") == "1"


@mx.compile
def _bias_swiglu_oai(gate_up: mx.array, bias: mx.array, limit: float, alpha: float) -> mx.array:
    return _swiglu_oai(gate_up + bias, limit, alpha)


@mx.compile
def _bias_weight(y: mx.array, bias: mx.array, w: mx.array) -> mx.array:
    return (y + bias) * w


def _residual_fn(x: mx.array, a: mx.array) -> mx.array:
    return x + a


_residual = mx.compile(_residual_fn, shapeless=True)  # every step length; one kernel for the cast + add


class Experts:
    # (token, expert) pairs from which the sorted gather is used: rows routed to the same expert share one read of it.
    # Row-independent and equal to the per-pair path for one row (checked bitwise), so verify steps of 2+ tokens use
    # it and stay identical to decode.
    SORT_MIN = 8
    """All experts of one layer. MXFP4 (gather_qmm) or dense (gather_mm, used for the float32 gate)."""

    def __init__(self, w: dict, prefix: str, cfg: GptOssConfig, dtype, dense: bool):
        self.dense = dense
        inter = cfg.intermediate_size

        def blocks(name):
            b = w[f"{prefix}.{name}_blocks"]  # [E, out, G, 16] uint8, two E2M1 values per byte
            return b.view(mx.uint32).reshape(*b.shape[:2], -1), w[f"{prefix}.{name}_scales"]

        gu, gus = blocks("gate_up_proj")
        # Rows are interleaved gate/up in the checkpoint; store [gate; up] so the activation splits contiguously.
        gu = mx.concatenate([gu[:, 0::2], gu[:, 1::2]], axis=1)
        gus = mx.concatenate([gus[:, 0::2], gus[:, 1::2]], axis=1)
        gub = w[f"{prefix}.gate_up_proj_bias"]
        self.gu_bias = mx.concatenate([gub[:, 0::2], gub[:, 1::2]], axis=1).astype(dtype)
        dn, dns = blocks("down_proj")
        self.dn_bias = w[f"{prefix}.down_proj_bias"].astype(dtype)
        assert gu.shape[1] == 2 * inter
        if dense:
            deq = lambda q, s: mx.dequantize(q, s, None, group_size=32, bits=4, mode="mxfp4").astype(dtype)
            self.gu_w = deq(gu, gus).swapaxes(-1, -2)  # [E, in, out]
            self.dn_w = deq(dn, dns).swapaxes(-1, -2)
        else:
            self.gu_w, self.gu_s, self.dn_w, self.dn_s = gu, gus, dn, dns
        self.limit, self.alpha = cfg.swiglu_limit, cfg.swiglu_alpha
        self.num_experts, self.inter, self.hidden = gu.shape[0], inter, cfg.hidden_size
        # Grouped MXFP4 GEMV kernel for decode-sized steps (pairs = tokens * top_k); 0 disables it.
        self.kernel_max_pairs = 0  # slower than gather_qmm on M4 except near B=8; kept for tuning

    def _mm(self, x, which, idx, sorted_):
        if self.dense:
            w = self.gu_w if which == "gu" else self.dn_w
            return mx.gather_mm(x, w, rhs_indices=idx, sorted_indices=sorted_)
        w, s = (self.gu_w, self.gu_s) if which == "gu" else (self.dn_w, self.dn_s)
        return mx.gather_qmm(
            x, w, s, None, rhs_indices=idx, transpose=True, group_size=32, bits=4, mode="mxfp4",
            sorted_indices=sorted_,
        )

    def __call__(self, x: mx.array, idx: mx.array, wts: mx.array) -> mx.array:
        """x [T, H], idx [T, k], routing weights [T, k] -> [T, H]."""
        T, k = idx.shape
        if idx.size <= self.kernel_max_pairs:
            flat = idx.flatten()
            order = mx.argsort(flat)
            inv = mx.argsort(order)
            si = flat[order]
            counts = (si[:, None] == mx.arange(self.num_experts)[None, :]).sum(axis=0)
            offsets = mx.concatenate([mx.zeros((1,), mx.int32), mx.cumsum(counts).astype(mx.int32)])
            h = grouped_mxfp4(x[order // k], self.gu_w, self.gu_s, self.gu_bias, offsets, self.inter, True,
                              limit=self.limit, alpha=self.alpha)
            y = grouped_mxfp4(h, self.dn_w, self.dn_s, self.dn_bias, offsets, self.hidden, False,
                              rw=wts.flatten()[order].astype(x.dtype))
            return y[inv].reshape(T, k, -1).sum(axis=1)
        if FUSE_GLUE and not self.dense and idx.size < self.SORT_MIN:  # decode: bias + weights fused
            xs = x[:, None, None, :]
            h = _bias_swiglu_oai(self._mm(xs, "gu", idx, False), self.gu_bias[idx][:, :, None, :], self.limit,
                                 self.alpha)
            y = self._mm(h, "dn", idx, False)
            return _bias_weight(y, self.dn_bias[idx][:, :, None, :], wts[..., None, None]).squeeze(-2).sum(axis=1)
        if FUSED_PREFILL_GLUE and not self.dense and idx.size >= self.FUSED_GLUE_MIN:
            return self._fused_sorted(x, idx, wts)
        return (self._unweighted(x, idx) * wts[..., None]).sum(axis=1)

    # Prefill-sized steps (pairs = tokens * top_k): bias + SwiGLU and bias + un-permute + weighted sum in one kernel
    # each (moe_kernels.bias_swiglu_sorted / combine_sorted), float32 inside. Not bitwise equal to the op sequence, so
    # never used for decode or speculative verify steps (at most a few tokens), which must match decode.
    FUSED_GLUE_MIN = 256

    def _fused_sorted(self, x, idx, wts):
        from .moe_kernels import bias_swiglu_sorted, combine_sorted

        T, k = idx.shape
        flat = idx.flatten()
        order = mx.argsort(flat)
        inv = mx.argsort(order)
        si = flat[order]
        h = self._mm(x[order // k][:, None, :], "gu", si, True)
        h = bias_swiglu_sorted(h.reshape(T * k, -1), self.gu_bias, si, self.limit, self.alpha)
        y = self._mm(h[:, None, :], "dn", si, True)
        od = mx.float32 if mx.float32 in (wts.dtype, x.dtype) else x.dtype
        return combine_sorted(y.reshape(T * k, -1), self.dn_bias, inv.astype(mx.int32), flat, wts, od)

    def _unweighted(self, x: mx.array, idx: mx.array) -> mx.array:
        """[T, k, H] expert outputs before the routing weights (gather_mm / gather_qmm path)."""
        T, k = idx.shape
        if idx.size >= self.SORT_MIN:
            flat = idx.flatten()
            order = mx.argsort(flat)
            inv = mx.argsort(order)
            xs = x[order // k][:, None, :]
            si = flat[order]
            h = self._mm(xs, "gu", si, True) + self.gu_bias[si][:, None, :]
            h = _swiglu_oai(h, self.limit, self.alpha)
            y = self._mm(h, "dn", si, True) + self.dn_bias[si][:, None, :]
            return y[inv].reshape(T, k, -1)
        xs = x[:, None, None, :]
        h = self._mm(xs, "gu", idx, False) + self.gu_bias[idx][:, :, None, :]
        h = _swiglu_oai(h, self.limit, self.alpha)
        y = self._mm(h, "dn", idx, False) + self.dn_bias[idx][:, :, None, :]
        return y.squeeze(-2)


@dataclass
class AttnGroup:
    """Queries that share one attention call. rows index the step's token rows."""

    rows: mx.array  # [B*L] token rows, item-major
    B: int
    L: int
    kv_full: mx.array  # [B, Kf] slot ids for full-attention layers
    mask_full: mx.array  # [B, 1, L, Kf] bool
    kv_swa: mx.array  # [B, Ks]
    mask_swa: mx.array  # [B, 1, L, Ks]
    block_tables: mx.array | None = None  # [B, max_blocks] (single-token groups, for the paged kernel)
    kv_lens: mx.array | None = None  # [B]
    max_kv: int = 0  # host-side max of kv_lens (picks the split-K decode kernel for long contexts)


@dataclass
class StepMeta:
    """Everything one forward needs, built once per step on the host and shared by all layers."""

    token_ids: mx.array  # [T]
    rope_offsets: mx.array  # [T] int32 absolute positions
    slots: mx.array  # [T] KV write slot of each new token
    groups: list[AttnGroup]
    inv_rows: mx.array | None  # permutation back to token order, None if identity
    sample_rows: mx.array  # [S]
    fused_kv: bool = False  # decode-only step: the paged kernel writes the new K/V in place (no scatter)


PAGED_ROWS_MAX = 4
FUSED_KV = os.environ.get("MJ_KV_FUSED", "1") == "1"  # decode-only steps write K/V inside the paged kernel
# Lean decode's candidate head width (top-rescore_k of it are rescored exactly on the host). On the real weights,
# 2048 greedy-decode states: top-16 missed the exact argmax 0 times at 3 and 4 bits (RESULTS 1zb).
LEAN_HEAD_BITS = int(os.environ.get("MJ_LEAN_HEAD_BITS", "3"))


class GarbageOutputError(RuntimeError):
    """The model produced a state no real forward pass gives (checklist 4.6: one 16k run once returned constant
    logits, perplexity = vocabulary size, with no error). `rows`: the sampled rows that failed (positions in the
    step's sample rows), so a caller can end only those sequences; today the Engine step raises for all of them."""

    def __init__(self, msg: str, rows=()):
        super().__init__(f"{msg} (sample rows {list(rows)})")
        self.rows = list(rows)


def check_output(x32: np.ndarray, cand_logits: np.ndarray) -> None:
    """Cheap host check on the rows a greedy step samples from (they are on the host already in the lean path):
    the final hidden state must be finite and not all zero, and the rescored candidates' logits must be finite and
    not all equal (constant logits). MJ_CHECK_OUTPUT=0 turns it off."""
    if os.environ.get("MJ_CHECK_OUTPUT", "1") == "0":
        return
    def bad(mask, msg):
        if mask.any():
            raise GarbageOutputError(msg, np.nonzero(mask)[0].tolist())

    bad(~np.isfinite(x32).all(axis=-1), "non-finite final hidden state")
    bad(np.abs(x32).max(axis=-1) == 0, "all-zero final hidden state (constant logits)")
    bad(~np.isfinite(cand_logits).all(axis=-1), "non-finite logits")
    # "Constant" means ALL rescored candidates (16) equal in float64. A tie between the top two is legitimate and must
    # never raise (seen in bf16: RESULTS 1zz item 1167, DeepSeek-V2-Lite's haiku token); do not tighten this to a
    # top-2 or top-k margin check.
    if cand_logits.shape[-1] > 1:
        bad(np.ptp(cand_logits, axis=-1) == 0, "constant logits over the candidates")


DEFAULT_PREP_CACHE = "~/Library/Caches/mj-mlx/prep"


class PrepCache:
    """Prepared-weights cache: the arrays a load derives from the checkpoint (the low-bit candidate head, 8-bit or
    packed attention) kept in safetensors files, so later loads read them instead of computing them again (RESULTS
    1zzc: 0.9-1.0 s for the head, 0.7 s / 3.4 s for 8-bit / packed attention per gpt-oss load). The files hold exactly
    the arrays the computation returns, so the model is the same either way.

    On by default under ~/Library/Caches/mj-mlx/prep; MJ_PREP_CACHE=<dir> puts it elsewhere, MJ_PREP_CACHE=0 turns it
    off; deleting the folder clears it (each model and option set is one subfolder with a key.json saying what it is).

    Stale reads: the key covers the checkpoint's files (name, size, mtime), the options that shape the arrays, MLX's
    version and a hash of the source of the code that derives them (this class, GptOss.__init__ / load_layer,
    bf16_pack.pack), so a code change is a new key. On top, every load re-derives a spot sample (layer 0's attention
    arrays, the head's first rows) and compares it bit for bit; on a mismatch the folder is deleted and the load
    computes everything. Disk: a write is skipped when free space is under twice EST_BYTES (or twice the file)."""

    VERSION = 2
    EST_BYTES = int(2.5e9)  # gpt-oss-20b: 2.0 GB for both arms (RESULTS 1zzc)

    def __init__(self, root: str, path: str, opts: dict):
        import hashlib
        import inspect

        from . import bf16_pack as _bp

        files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
        ident = [(os.path.basename(f), os.path.getsize(f), int(os.path.getmtime(f))) for f in files]
        src = "".join(inspect.getsource(o) for o in (PrepCache, GptOss.__init__, GptOss.load_layer, _bp.pack))
        code = hashlib.sha256(src.encode()).hexdigest()[:16]
        h = hashlib.sha256(json.dumps([self.VERSION, ident, sorted(opts.items()), mx.__version__, code],
                                      default=str).encode()).hexdigest()
        self.dir = os.path.join(os.path.expanduser(root), h[:16])
        self.ok, self.written, self.said = True, 0, False
        os.makedirs(self.dir, exist_ok=True)
        meta = os.path.join(self.dir, "key.json")
        if not os.path.exists(meta):
            json.dump(dict(version=self.VERSION, checkpoint=path, files=ident, opts=opts, mlx=mx.__version__,
                           code=code), open(meta, "w"), indent=1, default=str)
            self._prune(path, opts)

    KEEP = 2  # folders kept per checkpoint and options (this one and the newest other: an A/B of two commits)

    def _prune(self, path: str, opts: dict):
        """Delete older folders for the same checkpoint and options (an earlier code / MLX version / file state),
        keeping the KEEP newest (by key.json time) counting this one, so a code change does not leave 2 GB behind
        each time and an ABAB of two commits does not rebuild the cache every block. Files are written to a temp
        name and renamed into place (get), so a process killed mid-write leaves whole files or none."""
        import shutil

        root = os.path.dirname(self.dir)
        want = json.loads(json.dumps(opts, default=str))
        same = []
        for d in os.listdir(root):
            full = os.path.join(root, d)
            if full == self.dir:
                continue
            try:
                k = json.load(open(os.path.join(full, "key.json")))
            except (OSError, ValueError):
                continue
            if k.get("checkpoint") == path and k.get("opts") == want:
                same.append((os.path.getmtime(os.path.join(full, "key.json")), full))
        for _, full in sorted(same, reverse=True)[self.KEEP - 1:]:
            shutil.rmtree(full, ignore_errors=True)

    def _drop(self, why: str):
        import shutil
        import sys

        print(f"mj-mlx: prepared-weights cache {self.dir}: {why}; deleted, computing instead", file=sys.stderr)
        shutil.rmtree(self.dir, ignore_errors=True)
        self.ok = False

    def get(self, name: str, make, verify=None):
        """make() -> dict[str, mx.array]: read from the file when it exists (then verify(d) -> bool, when given, is
        the spot check), else computed, written when there is room, and returned."""
        import shutil
        import sys

        f = os.path.join(self.dir, name + ".safetensors")
        if self.ok and os.path.exists(f):
            d = mx.load(f)
            if verify is None or verify(d):
                return d
            self._drop(f"{name} differs from a fresh derivation")
        d = make()
        mx.eval(list(d.values()))
        if not self.ok:
            return d
        nbytes = sum(v.nbytes for v in d.values())
        if shutil.disk_usage(self.dir).free < 2 * max(self.EST_BYTES, nbytes):
            if not self.said:
                print(f"mj-mlx: under {2 * self.EST_BYTES / 1e9:.0f} GB free: not writing the prepared-weights cache",
                      file=sys.stderr)
                self.said = True
            return d
        if not self.said:
            print(f"mj-mlx: writing the prepared-weights cache to {self.dir} (once per model and options, about "
                  f"{self.EST_BYTES / 1e9:.1f} GB for gpt-oss-20b; MJ_PREP_CACHE=0 turns it off, deleting the folder "
                  "clears it)", file=sys.stderr)
            self.said = True
        tmp = f + ".tmp.safetensors"
        mx.save_safetensors(tmp, d)
        os.replace(tmp, f)
        self.written += nbytes
        return d

    def finish(self):
        import sys

        if self.written:
            print(f"mj-mlx: prepared-weights cache written: {self.written / 1e9:.2f} GB in {self.dir}", file=sys.stderr)


def _same(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(a[k].shape == b[k].shape and a[k].dtype == b[k].dtype
                                        and mx.array_equal(a[k], b[k]).item() for k in a)


ROUTER_ROWWISE_MAX = 4  # steps of up to this many tokens compute router logits row by row (exact vs decode)  # items of 2..4 tokens attend row by row through the paged decode kernel


def _write_blocks_private(bt: np.ndarray, kv_lens: np.ndarray, wcol: np.ndarray, block_size: int) -> bool:
    """True when no row's write block is read by another row (the fused KV write in the paged kernel would race
    with that row's reads). A decode row writes into its own partially filled tail block; only full blocks are
    shared through the prefix cache, so this holds for the scheduler's tables. Checked, not assumed."""
    n = len(kv_lens)
    if n < 2:
        return True
    nb = (kv_lens.astype(np.int64) + block_size - 1) // block_size  # blocks each row reads
    used = np.arange(bt.shape[1])[None, :] < nb[:, None]
    rows = np.broadcast_to(np.arange(n)[:, None], bt.shape)[used]
    blks = bt[used]
    pairs = np.unique(np.stack([blks, rows], axis=1), axis=0)  # (block, row) once per row
    ub, cnt = np.unique(pairs[:, 0], return_counts=True)
    wblk = bt[np.arange(n), wcol]
    return bool(np.all(cnt[np.searchsorted(ub, wblk)] == 1))


def build_meta(
    token_ids: np.ndarray, positions: np.ndarray, starts: np.ndarray, lens: np.ndarray,
    block_tables: np.ndarray, sample_rows: np.ndarray, block_size: int, window: int,
    slot_mapping: np.ndarray | None = None, paged_decode: bool = False,
) -> StepMeta:
    """Group items for attention: all single-token items in one batched call, longer items one call each."""
    n = len(lens)
    cu = np.zeros(n + 1, np.int64)
    np.cumsum(lens, out=cu[1:])
    kv_lens = starts + lens
    bt = np.maximum(block_tables, 0).astype(np.int64)
    p_all = positions.astype(np.int64)
    if slot_mapping is None:
        item_of_row = np.repeat(np.arange(n), lens)
        slots = bt[item_of_row, p_all // block_size] * block_size + p_all % block_size
    else:
        slots = slot_mapping

    def kv_index(items, lo, hi, qpos):
        """Slots of key positions lo..hi-1 per item (padded), and the causal/window mask for queries qpos."""
        K = int((hi - lo).max())
        kp = lo[:, None] + np.arange(K)[None, :]  # [B, K] key positions
        valid = kp < hi[:, None]
        kpc = np.minimum(kp, hi[:, None] - 1)
        s = bt[items[:, None], kpc // block_size] * block_size + kpc % block_size
        m = valid[:, None, :] & (kp[:, None, :] <= qpos[:, :, None])  # [B, L, K]
        return s, m, kp

    groups = []
    order = []
    single = np.nonzero(lens == 1)[0]
    multi = np.nonzero(lens != 1)[0]
    parts = []
    if len(single):
        parts.append((single, 1))
    parts += [(np.array([i]), int(lens[i])) for i in multi]
    for items, L in parts:
        rows = (cu[items][:, None] + np.arange(L)[None, :]).reshape(-1)
        order.append(rows)
        if L == 1 and paged_decode:
            groups.append(AttnGroup(mx.array(rows.astype(np.int32)), len(items), 1, None, None, None, None,
                                    mx.array(bt[items].astype(np.int32)),
                                    mx.array(kv_lens[items].astype(np.int32)), int(kv_lens[items].max())))
            continue
        if paged_decode and L <= PAGED_ROWS_MAX:  # short item (speculative verify): each row is a decode query
            # over keys 0..its position, through the same kernel as decode, so it computes decode's numbers
            groups.append(AttnGroup(mx.array(rows.astype(np.int32)), L, 1, None, None, None, None,
                                    mx.array(np.repeat(bt[items], L, axis=0).astype(np.int32)),
                                    mx.array((int(starts[items[0]]) + 1 + np.arange(L)).astype(np.int32)),
                                    int(starts[items[0]]) + L))
            continue
        qpos = starts[items][:, None] + np.arange(L)[None, :]  # [B, L]
        hi = kv_lens[items].astype(np.int64)
        sf, mf, _ = kv_index(items, np.zeros_like(hi), hi, qpos)
        lo_w = np.maximum(starts[items].astype(np.int64) - window + 1, 0)
        ss, ms, kps = kv_index(items, lo_w, hi, qpos)
        ms &= kps[:, None, :] > qpos[:, :, None] - window
        groups.append(
            AttnGroup(
                mx.array(rows.astype(np.int32)), len(items), L,
                mx.array(sf.astype(np.int32)), mx.array(mf[:, None]),
                mx.array(ss.astype(np.int32)), mx.array(ms[:, None]),
            )
        )
    # every item one decode token through the paged kernel, and each write slot the one the kernel derives from
    # the block table at position kv_len - 1: the kernel can store the new K/V itself (FUSED_KV)
    fused = (FUSED_KV and len(groups) == 1 and groups[0].block_tables is not None and len(multi) == 0
             and bool(np.all(p_all == kv_lens - 1))
             and np.array_equal(slots, bt[np.arange(n), p_all // block_size] * block_size + p_all % block_size)
             and _write_blocks_private(bt, kv_lens, p_all // block_size, block_size))
    perm = np.concatenate(order)
    inv = None
    if not np.array_equal(perm, np.arange(len(perm))):
        inv_np = np.empty_like(perm)
        inv_np[perm] = np.arange(len(perm))
        inv = mx.array(inv_np.astype(np.int32))
    return StepMeta(
        mx.array(token_ids.astype(np.int32)), mx.array(positions.astype(np.int32)),
        mx.array(slots.astype(np.int32)), groups, inv, mx.array(sample_rows.astype(np.int32)), fused,
    )


def topk_indices(lg: mx.array, k: int, chunk: int = 128) -> mx.array:
    """Indices of the k largest entries per row (unordered), exact.

    Two stages: chunk maxima, the k best chunks (they must contain the global top-k), then top-k inside them.
    A full-vocabulary argpartition costs ~26 ms for 64 x 201k on M4; this costs ~1.3 ms."""
    S, V = lg.shape
    n = V // chunk
    if n < 2 * k:
        return mx.argpartition(lg, kth=-k, axis=-1)[:, -k:]
    head = lg[:, : n * chunk].reshape(S, n, chunk).max(axis=-1)
    if n * chunk < V:  # fold the ragged tail into one more chunk
        head = mx.concatenate([head, lg[:, n * chunk:].max(axis=-1, keepdims=True)], axis=-1)
    best = mx.argpartition(head, kth=-k, axis=-1)[:, -k:]  # [S, k]
    raw = (best[:, :, None] * chunk + mx.arange(chunk)[None, None, :]).reshape(S, k * chunk)
    idx = mx.minimum(raw, V - 1)  # the tail chunk may be short
    vals = mx.take_along_axis(lg, idx, axis=-1)
    vals = mx.where(raw < V, vals, mx.array(-mx.inf, vals.dtype))  # clamped duplicates never win
    sel = mx.argpartition(vals, kth=-k, axis=-1)[:, -k:]
    return mx.take_along_axis(idx, sel, axis=-1)


class GptOss:
    """Weights plus a forward over one step. The KV pool is passed in (owned by the runner)."""

    def __init__(self, path: str, dtype=mx.bfloat16, dense_experts: bool = False, weights: dict | None = None,
                 cfg: GptOssConfig | None = None, precise: bool = True, lm_head_bits: int | None = -1,
                 rescore_k: int = 16, head: bool = True, layers: list[int] | None = None,
                 stream_experts: str | None = None, expert_slots: int = 16, lean: bool = False, io_threads: int = 8,
                 io_nocache: bool = True, prefetch: bool = True, attn_bits: int | None = None,
                 map_experts: str | None = None, attn_pack: bool = False, map_head: str | None = None):
        """precise: keep the residual stream, RMSNorm statistics and the router in float32 (weights and KV stay
        in `dtype`). Costs ~T*H*4 bytes per layer; keeps bf16 expert routing from flipping on rounding.
        head=False / layers=[...]: build without embedding+lm_head and/or with a subset of layers (layer-streamed
        evaluation of checkpoints that do not fit in memory; see tools/mlx_real_eval.py).
        stream_experts: directory made by tools/mlx_repack_experts.py; routed experts then live on the SSD and
        `expert_slots` per layer are resident (expert_stream.StreamingExperts).
        lean: keep the embedding table and the bf16 lm_head on disk (memory-mapped rows: one embedding row per
        token, the rescore rows per sampled row); only a 4-bit candidate head is resident. Greedy only.
        attn_bits: opt-in lossy mode, e.g. 8: attention q/k/v/o weights quantized (affine, group 32), as llama.cpp's
        gpt-oss GGUF stores them (Q8_0). Halves attention bytes and memory; output no longer matches the
        reference exactly (measured in RESULTS.md). None (default) keeps the checkpoint's weights.
        map_experts: directory made by expert_map.repack_part_major; every routed expert is then a zero-copy view of
        the memory-mapped file (the page cache holds it, MLX does not copy it), for models that fit in RAM but not
        in the GPU working set.
        attn_pack: store the attention q/k/v/o weights in bf16_pack's lossless 12.25-bit layout (every weight
        bit-identical, 23% fewer bytes). Steps of up to 4 tokens read it directly (bf16_pack.matvec); longer steps
        rebuild the bf16 matrix per call and use MLX's matmul (the same numbers as without packing).
        map_head: with lean, a directory for page-aligned copies of the embedding table and the bf16 lm_head
        (expert_map.map_rows, written once): both become zero-copy GPU arrays over the page cache, so the embedding
        rows and the rescore rows are gathered on the GPU. Without it, lean reads those rows on the CPU, which needs
        the token ids on the host every step (the runner then cannot overlap decode steps). Costs RAM: Metal wires a
        whole no-copy buffer while a command buffer uses it (measured: one row gather wired 1.21 GB for the 1.16 GB
        table), so this adds ~2.3 GB wired for gpt-oss-20b; on a 16 GB Mac that thrashes (0.07 tok/s). For machines
        with the headroom only."""
        # MJ_LOAD_TIMING=1: evaluate at each stage and record its seconds in self.load_times (diagnosis only: the
        # barriers change when work runs, not what it computes)
        timing = os.environ.get("MJ_LOAD_TIMING") == "1"
        self.load_times = {}
        t_mark = [time.perf_counter()]

        def mark(name, arrays=()):
            if timing:
                mx.eval(arrays)
                t = time.perf_counter()
                self.load_times[name] = self.load_times.get(name, 0.0) + t - t_mark[0]
                t_mark[0] = t

        self.precise = precise and dtype != mx.float32
        self.attn_bits = attn_bits
        pc = os.environ.get("MJ_PREP_CACHE", DEFAULT_PREP_CACHE)  # "0" or "" turns it off (PrepCache)
        self.prep = PrepCache(pc, path, dict(dtype=str(dtype), attn_bits=attn_bits, attn_pack=attn_pack,
                                             head_bits=LEAN_HEAD_BITS)) if pc and pc != "0" and weights is None else None
        self.attn_pack = attn_pack and not attn_bits and dtype == mx.bfloat16
        self.dense_experts = dense_experts
        if cfg is None:
            with open(os.path.join(path, "config.json")) as f:
                cfg = GptOssConfig.from_dict(json.load(f))
        self.cfg = cfg
        self.dtype = dtype
        self.block_size = 16  # set by the runner
        if weights is None:
            weights = {}
            for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
                weights.update(mx.load(f))
        w = weights
        c = cfg
        mark("open_safetensors")
        cast = lambda a: a.astype(dtype)
        self.store = None
        self.map = None
        if map_experts:
            from .expert_map import MappedExperts

            self.map = MappedExperts(map_experts)
        if stream_experts:
            from .expert_stream import ExpertStore

            self.store = ExpertStore(stream_experts, threads=io_threads, nocache=io_nocache)
            self.expert_slots = expert_slots
        self.prefetch = prefetch and stream_experts is not None
        mark("open_expert_files")
        self.lean = lean
        self.embed_rows = self.head_rows = None
        lean_head_src = None
        if head and lean:
            from .expert_stream import SafetensorsRows

            self.embed = None
            mapped = None
            if map_head and dtype == mx.bfloat16:
                from .expert_map import map_rows

                self._head_maps = []
                self.embed = map_rows(map_head, path, "model.embed_tokens.weight", self._head_maps)
                mapped = map_rows(map_head, path, "lm_head.weight", self._head_maps)
            else:
                self.embed_rows = SafetensorsRows(path, "model.embed_tokens.weight")
                self.head_rows = SafetensorsRows(path, "lm_head.weight")
            self.norm = cast(w["model.norm.weight"])
            if c.hidden_size % 32:  # too narrow for a low-bit candidate head: keep the bf16 head resident
                self.lm_head = cast(w["lm_head.weight"])
                lean = False
            else:
                self.lm_head = mapped  # None: rescore rows read on the CPU (head_rows)
                lm_head_bits = LEAN_HEAD_BITS
                lean_head_src = w["lm_head.weight"]  # quantized once below, then dropped
        elif head:
            self.embed = cast(w["model.embed_tokens.weight"])
            self.lm_head = cast(w["lm_head.weight"]) if "lm_head.weight" in w else self.embed
            self.norm = cast(w["model.norm.weight"])
        else:
            self.embed = self.lm_head = self.norm = None
            lm_head_bits = None
        mark("head_setup", [a for a in (self.embed, self.lm_head, self.norm) if a is not None])
        self.layers = []
        for i in (range(c.num_hidden_layers) if layers is None else layers):
            self.layers.append(self.load_layer(w, i, mark if timing else None))
        if self.map is not None:  # MJ_MLOCK=bg: pin the mapped experts in the background from here
            self.map.start_background_lock()
        mark("lock_start")
        for j, L in enumerate(self.layers):
            L["_index"] = j
        inv_freq, mscale = yarn_inv_freq(c)
        self.rope_freqs = mx.array(1.0 / inv_freq)
        self.attn_scale = (mscale * mscale) / math.sqrt(c.head_dim)
        # Optional low-bit copy of lm_head used only to pick candidates; the final choice among the top
        # `rescore_k` is recomputed in float32 from the original rows, so greedy output does not depend on it
        # unless the true top-1 falls outside the low-bit top-k.
        # lm_head_bits: -1 = auto policy by rows (measured on M4: 4-bit is cheapest at <= 4 rows, 8-bit up to
        # ~48, bf16 above because quantized matmul turns compute-bound); an int = always that width; None = bf16.
        self.rescore_k = rescore_k
        self.lm_head_gs = 64 if c.hidden_size % 64 == 0 else 32
        if lm_head_bits == -1:
            self.head_policy = [(4, 4), (48, 8)] if dtype != mx.float32 else []
        elif lm_head_bits:
            self.head_policy = [(1 << 30, lm_head_bits)]
        else:
            self.head_policy = []
        if c.hidden_size % 32:  # too narrow to quantize: always the bf16 head
            self.head_policy = []
        src = self.lm_head if self.lm_head is not None else (
            lean_head_src.astype(dtype) if lean_head_src is not None else None)
        if src is None:  # head=False (layer-streamed evaluation): no head at all
            self.head_policy = []
        def quant_head(b):
            if self.prep is None:
                return mx.quantize(src, self.lm_head_gs, b)
            def make(rows=None):
                if rows is None:
                    x = src
                elif self.head_rows is not None:  # lean: read just these rows (slicing src would load all 1.16 GB)
                    x = mx.array(np.ascontiguousarray(self.head_rows.rows[:rows])).view(mx.bfloat16).astype(dtype)
                else:
                    x = src[:rows]
                return dict((k, a) for k, a in zip(("w", "scales", "biases"), mx.quantize(x, self.lm_head_gs, b))
                            if a is not None)

            # spot check: the first 4096 rows quantized afresh (quantization is per row, so they equal the full
            # result's first rows)
            d = self.prep.get(f"head_q{b}_g{self.lm_head_gs}", make,
                              lambda d: _same({k: v[:4096] for k, v in d.items()}, make(4096)))
            return tuple(d.get(k) for k in ("w", "scales", "biases"))

        self.lm_head_qs = {b: quant_head(b) for _, b in self.head_policy}
        self.lm_head_q = None
        del src
        mark("head_quantize", [a for q in self.lm_head_qs.values() for a in q if a is not None])
        mx.eval(self.parameters())
        mx.clear_cache()
        if self.prep is not None:
            self.prep.finish()
        mark("final_eval")  # drop load-time temporaries (pre-fusion/pre-cast copies) from MLX's buffer cache

    def load_layer(self, w: dict, i: int, mark=None) -> dict:
        """Layer i's weights in this model's layout and precision. mark: MJ_LOAD_TIMING's stage timer."""
        c, dtype = self.cfg, self.dtype
        cast = lambda a: a.astype(dtype)
        p = f"model.layers.{i}"
        a = f"{p}.self_attn"
        qkv_w = mx.concatenate([w[f"{a}.{n}_proj.weight"] for n in "qkv"], axis=0)
        qkv_b = mx.concatenate([w[f"{a}.{n}_proj.bias"] for n in "qkv"], axis=0)
        router_dt = mx.float32 if (self.precise or dtype == mx.float32) else dtype
        o_w = w[f"{a}.o_proj.weight"]
        attn_q = {}
        prep = getattr(self, "prep", None)
        if self.attn_bits:
            qa = dict(group_size=32, bits=self.attn_bits)

            def make_q():
                return {f"{n}.{j}": a for n, src in (("qkv_q", qkv_w), ("o_q", o_w))
                        for j, a in enumerate(mx.quantize(cast(src), **qa)) if a is not None}

            d = prep.get(f"attn{self.attn_bits}_L{i}", make_q, (lambda d: _same(d, make_q())) if i == 0 else None) \
                if prep else make_q()
            attn_q = {n: tuple(d.get(f"{n}.{j}") for j in range(3)) for n in ("qkv_q", "o_q")}
            mx.eval(attn_q)  # per layer, so the bf16 temporaries of all layers never coexist
            mx.clear_cache()
        elif self.attn_pack:
            keys = ("lo", "nib", "base", "esc_ptr", "esc_k", "esc_v")

            def make_p():
                return {f"{n}.{k}": v for n, src in (("qkv_p", qkv_w), ("o_p", o_w))
                        for k, v in bf16_pack.pack(cast(src)).items() if k in keys}

            d = prep.get(f"pack_L{i}", make_p, (lambda d: _same(d, make_p())) if i == 0 else None) if prep else make_p()
            attn_q = {n: dict({k: d[f"{n}.{k}"] for k in keys}, shape=tuple(d[f"{n}.lo"].shape)) for n in ("qkv_p", "o_p")}
            mx.eval([bf16_pack.arrays(attn_q[k]) for k in attn_q])
        if mark:
            mark("layers_attention", [a for v in attn_q.values() for a in (bf16_pack.arrays(v) if self.attn_pack else v)
                                      if a is not None] + [qkv_w, qkv_b, o_w])
        out = dict(
            **attn_q,
            ln1=cast(w[f"{p}.input_layernorm.weight"]),
            ln2=cast(w[f"{p}.post_attention_layernorm.weight"]),
            # float32 copies for the precise path (it cast them every layer and step)
            ln1_f32=cast(w[f"{p}.input_layernorm.weight"]).astype(mx.float32),
            ln2_f32=cast(w[f"{p}.post_attention_layernorm.weight"]).astype(mx.float32),
            qkv_wt=None if attn_q else cast(qkv_w).T, qkv_b=cast(qkv_b),
            o_wt=None if attn_q else cast(o_w).T, o_b=cast(w[f"{a}.o_proj.bias"]),
            sinks=w[f"{a}.sinks"].astype(mx.float32 if dtype == mx.float32 else dtype),
            router_wt=w[f"{p}.mlp.router.weight"].astype(router_dt).T,
            router_b=w[f"{p}.mlp.router.bias"].astype(router_dt),
            experts=self._experts(w, i),
            sliding=c.layer_types[i] == "sliding_attention",
        )
        if mark:
            e = out["experts"]
            mark("layers_experts", e.arrays() if hasattr(e, "arrays") else
                 [v for v in vars(e).values() if isinstance(v, mx.array)])
            mark("layers_rest", [v for v in out.values() if isinstance(v, mx.array)])
        return out

    def _experts(self, w: dict, i: int):
        p = f"model.layers.{i}.mlp.experts"
        if self.map is not None:
            m = self.map.layer(i)
            e = Experts.__new__(Experts)
            gub = w[f"{p}.gate_up_proj_bias"]
            e.gu_bias = mx.concatenate([gub[:, 0::2], gub[:, 1::2]], axis=1).astype(self.dtype)
            e.dn_bias = w[f"{p}.down_proj_bias"].astype(self.dtype)
            e.dense, e.gu_w, e.gu_s, e.dn_w, e.dn_s = False, m["gu_w"], m["gu_s"], m["dn_w"], m["dn_s"]
            e.limit, e.alpha = self.cfg.swiglu_limit, self.cfg.swiglu_alpha
            e.num_experts, e.inter, e.hidden = m["gu_w"].shape[0], self.cfg.intermediate_size, self.cfg.hidden_size
            e.kernel_max_pairs = 0
            return e
        if self.store is None:
            return Experts(w, p, self.cfg, self.dtype, self.dense_experts)
        from .expert_stream import StreamingExperts, layer_slots

        gub = w[f"{p}.gate_up_proj_bias"]
        gub = mx.concatenate([gub[:, 0::2], gub[:, 1::2]], axis=1)  # same de-interleave as the repacked blocks
        return StreamingExperts(self.store, i, layer_slots(self.expert_slots, i), gub, w[f"{p}.down_proj_bias"], self.cfg,
                                self.dtype)

    def expert_stats(self) -> dict:
        st = [L["experts"].stats() for L in self.layers if hasattr(L["experts"], "stats")]
        h, m = sum(x["hits"] for x in st), sum(x["misses"] for x in st)
        return dict(hits=h, misses=m, hit_rate=h / (h + m) if h + m else None,
                    prefetch_issued=sum(x["prefetch_issued"] for x in st),
                    prefetch_used=sum(x["prefetch_used"] for x in st),
                    spec_used=sum(x.get("spec_used", 0) for x in st),
                    spec_wasted=sum(x.get("spec_wasted", 0) for x in st),
                    spec_pending_only=sum(x.get("spec_pending_only", 0) for x in st),
                    bytes_read=self.store.bytes_read if self.store else 0)

    def parameters(self) -> list[mx.array]:
        out = [a for a in (self.embed, self.lm_head, self.norm, self.rope_freqs) if a is not None]
        for q in getattr(self, "lm_head_qs", {}).values():
            out += [a for a in q if a is not None]
        for L in self.layers:
            out += [v for k, v in L.items() if isinstance(v, mx.array)]
            out += [a for k in ("qkv_q", "o_q") if k in L for a in L[k]]
            out += [a for k in ("qkv_p", "o_p") if k in L for a in bf16_pack.arrays(L[k])]
            e = L["experts"]
            out += e.arrays() if hasattr(e, "arrays") else [v for v in vars(e).values() if isinstance(v, mx.array)]
        return out

    def kv_bytes_per_token(self, dtype=None) -> int:
        c = self.cfg
        return c.num_hidden_layers * 2 * c.num_key_value_heads * c.head_dim * (dtype or self.dtype).size

    def new_kv_pool(self, num_slots: int) -> list[tuple[mx.array, mx.array]]:
        c = self.cfg
        shape = (num_slots, c.num_key_value_heads, c.head_dim)
        pool = [(mx.zeros(shape, self.dtype), mx.zeros(shape, self.dtype)) for _ in range(c.num_hidden_layers)]
        mx.eval(pool)
        return pool

    def _attention(self, L, x, meta: StepMeta, pool, li):
        c = self.cfg
        T = x.shape[0]
        nq, nkv, D = c.num_attention_heads, c.num_key_value_heads, c.head_dim
        qkv = self._proj(L, "qkv", x)
        q = qkv[:, : nq * D].reshape(T, nq, 1, D)
        k = qkv[:, nq * D : (nq + nkv) * D].reshape(T, nkv, 1, D)
        v = qkv[:, (nq + nkv) * D :].reshape(T, nkv, D)
        q = mx.fast.rope(q, D, traditional=False, base=None, scale=1.0, offset=meta.rope_offsets,
                         freqs=self.rope_freqs).squeeze(2)
        k = mx.fast.rope(k, D, traditional=False, base=None, scale=1.0, offset=meta.rope_offsets,
                         freqs=self.rope_freqs).squeeze(2)
        kp, vp = pool[li]
        fused = meta.fused_kv and kp.dtype == k.dtype
        if not fused:
            kp[meta.slots] = k
            vp[meta.slots] = v
            pool[li] = (kp, vp)
        outs = []
        for g in meta.groups:
            if g.block_tables is not None:
                o = paged_decode_attention(q[g.rows], kp, vp, g.block_tables, g.kv_lens, L["sinks"],
                                           self.attn_scale, self.block_size,
                                           c.sliding_window if L["sliding"] else 0, g.max_kv,
                                           *((k, v) if fused else ()))
                outs.append(o.reshape(g.B, nq * D))
                continue
            kv_idx, mask = (g.kv_swa, g.mask_swa) if L["sliding"] else (g.kv_full, g.mask_full)
            qg = q[g.rows].reshape(g.B, g.L, nq, D).transpose(0, 2, 1, 3)
            kg = kp[kv_idx].transpose(0, 2, 1, 3)  # [B, nkv, K, D]
            vg = vp[kv_idx].transpose(0, 2, 1, 3)
            if g.B == 1 and not L["sliding"]:  # one item, keys 0..kv_len-1, queries the last rows: fused "causal"
                mask = "causal"
            o = mx.fast.scaled_dot_product_attention(qg, kg, vg, scale=self.attn_scale, mask=mask, sinks=L["sinks"])
            outs.append(o.transpose(0, 2, 1, 3).reshape(g.B * g.L, nq * D))
        o = outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=0)
        if meta.inv_rows is not None:
            o = o[meta.inv_rows]
        return self._proj(L, "o", o)

    def _proj(self, L, name, x):
        """Attention projection `name` (qkv / o) with bias, in whichever weight layout the layer holds."""
        b = L[f"{name}_b"]
        if f"{name}_p" in L:
            p = L[f"{name}_p"]
            if x.shape[0] <= bf16_pack.MAX_ROWS:
                return bf16_pack.matvec(x, p, b)
            if bf16_pack.ROWWISE:  # accuracy check of the decode kernel on long steps (tools/mlx_real_eval.py)
                R = bf16_pack.MAX_ROWS
                return mx.concatenate([bf16_pack.matvec(x[r:r + R], p, b) for r in range(0, x.shape[0], R)])
            return mx.addmm(b, x, bf16_pack.unpack(p).T)
        if f"{name}_q" in L:
            return self._qmm(x, L[f"{name}_q"]) + b
        return mx.addmm(b, x, L[f"{name}_wt"])

    def _qmm(self, x, q):
        w, sc, b = q
        return mx.quantized_matmul(x, w, sc, b, transpose=True, group_size=32, bits=self.attn_bits)

    def _moe(self, L, x, x32=None):
        k = self.cfg.num_experts_per_tok
        xr = x if x32 is None else x32
        if self.prefetch and hasattr(L["experts"], "prefetch_next"):
            li = L.get("_index")
            if li is not None and li + 1 < len(self.layers):
                N = self.layers[li + 1]
                L["experts"].prefetch_next = (N["experts"], N["router_wt"], N["router_b"], xr)
        xr2 = x if x32 is None else x32
        if 1 < xr2.shape[0] <= ROUTER_ROWWISE_MAX:  # short verify steps: each row through decode's own call, so
            # a k+1 token step routes exactly as k+1 decode steps (a 2-4 row matmul rounds differently: 2.9e-6)
            g = mx.concatenate([mx.addmm(L["router_b"], xr2[i:i + 1], L["router_wt"]) for i in range(xr2.shape[0])])
        else:
            g = mx.addmm(L["router_b"], xr2, L["router_wt"])
        idx = mx.stop_gradient(mx.argpartition(g, kth=-k, axis=-1)[..., -k:])
        wts = mx.softmax(mx.take_along_axis(g, idx, axis=-1), axis=-1, precise=True)
        return L["experts"](x, idx, wts)

    def hidden(self, meta: StepMeta, pool, x: mx.array | None = None, aux_layers=(), aux_out: list | None = None
               ) -> mx.array:
        """Residual stream after all loaded layers. x: start from these embeddings instead of meta.token_ids.
        aux_layers / aux_out: append the residual stream entering each listed layer (EAGLE-3 draft heads)."""
        eps = self.cfg.rms_norm_eps
        if x is None:
            x = self.embed[meta.token_ids] if self.embed is not None else self.embed_rows.take(
                np.array(meta.token_ids), self.dtype)
        if not self.precise:
            for li, L in enumerate(self.layers):
                if li in aux_layers:
                    aux_out.extend([x] * aux_layers.count(li))
                x = x + self._attention(L, mx.fast.rms_norm(x, L["ln1"], eps), meta, pool, li)
                x = x + self._moe(L, mx.fast.rms_norm(x, L["ln2"], eps))
            return x
        dt = self.dtype
        x = x.astype(mx.float32)
        add = _residual if FUSE_GLUE else (lambda a, b: a + b)
        for li, L in enumerate(self.layers):
            if li in aux_layers:
                aux_out.extend([x] * aux_layers.count(li))
            ln1, ln2 = (L["ln1_f32"], L["ln2_f32"]) if FUSE_GLUE else (L["ln1"], L["ln2"].astype(mx.float32))
            x = add(x, self._attention(L, mx.fast.rms_norm(x, ln1, eps).astype(dt), meta, pool, li))
            h32 = mx.fast.rms_norm(x, ln2, eps)
            x = add(x, self._moe(L, h32.astype(dt), h32))
        return x

    def logits(self, meta: StepMeta, pool, rescore: int | None = None) -> mx.array:
        """Logits [S, V] for the sampled rows only.

        rescore=k (default 8 when the model is not float32): the k largest bf16 logits of each row are recomputed
        exactly in float32 from the float32 final hidden state (k rows of lm_head, ~k*H*2 bytes), so output
        rounding cannot flip the top choice. Other entries keep their bf16 value. Returns float32 then."""
        x = self.hidden(meta, pool)[meta.sample_rows]
        if rescore is None:
            rescore = 8 if self.dtype != mx.float32 else 0
        x32 = mx.fast.rms_norm(x.astype(mx.float32), self.norm.astype(mx.float32), self.cfg.rms_norm_eps)
        lg = x32.astype(self.dtype) @ self.lm_head.T
        if not rescore:
            return lg
        cand = topk_indices(lg, rescore)  # [S, k]
        exact = (self.lm_head[cand].astype(mx.float32) * x32[:, None, :]).sum(axis=-1)  # [S, k]
        lg = lg.astype(mx.float32)
        return mx.put_along_axis(lg, cand, exact, axis=-1)

    def head_bits_for(self, rows: int) -> int | None:
        return next((b for limit, b in self.head_policy if rows <= limit), None)

    def _candidate_logits(self, x32):
        xb = x32.astype(self.dtype)
        bits = self.head_bits_for(x32.shape[0])
        if bits is None:
            return xb @ self.lm_head.T
        w, sc, b = self.lm_head_qs[bits]
        return mx.quantized_matmul(xb, w, sc, b, transpose=True, group_size=self.lm_head_gs, bits=bits)

    def final_hidden(self, meta: StepMeta, pool) -> mx.array:
        """Normalized float32 final hidden state [S, H] of the sampled rows."""
        x = self.hidden(meta, pool)[meta.sample_rows]
        return mx.fast.rms_norm(x.astype(mx.float32), self.norm.astype(mx.float32), self.cfg.rms_norm_eps)

    def full_logits(self, x32: mx.array) -> mx.array:
        """Full-vocabulary float32 logits from the full-precision head (what samplers must use)."""
        if self.lm_head is None:
            raise NotImplementedError("lean mode keeps lm_head on disk: greedy only")
        if self.dtype == mx.float32:
            return x32 @ self.lm_head.T
        return (x32.astype(self.dtype) @ self.lm_head.T).astype(mx.float32)

    def greedy_from(self, x32: mx.array) -> mx.array:
        lg = self._candidate_logits(x32)
        if self.dtype == mx.float32 and not self.head_policy:
            return mx.argmax(lg, axis=-1)
        cand = topk_indices(lg, self.rescore_k)
        if self.lm_head is None:  # lean: the rescore rows are on disk, so the choice is made on the host
            mx.eval(cand, x32)
            return mx.array(self.rescore_host(np.array(cand), np.array(x32)))
        rows = self.lm_head[cand]
        exact = (rows.astype(mx.float32) * x32[:, None, :]).sum(axis=-1)
        return mx.take_along_axis(cand, mx.argmax(exact, axis=-1)[:, None], axis=-1)[:, 0]

    def head_candidates(self, x32: mx.array) -> mx.array:
        """Lazy top-k candidate ids [S, k] from the low-bit head (lean decode: chosen on the host, rescore_host)."""
        return topk_indices(self._candidate_logits(x32), self.rescore_k)

    def rescore_host(self, cand: np.ndarray, x32: np.ndarray) -> np.ndarray:
        """Lean greedy choice among the candidates: float64 dot products of the bf16 lm_head rows (read from disk,
        in parallel) with the float32 final hidden state, on the CPU. Needs the candidates on the host anyway, so
        it costs no GPU round trip (the GPU rescore did: RESULTS 1z). uint32 [S]."""
        S, k = cand.shape
        raw = self.head_rows.read(cand.reshape(-1))  # [S*k, H] bf16 bits
        rows = (raw.astype(np.uint32) << 16).view(np.float32).reshape(S, k, -1)  # bf16 -> f32 exactly
        exact = np.einsum("skh,sh->sk", rows.astype(np.float64), x32.astype(np.float64))
        check_output(x32, exact)
        return cand[np.arange(S), np.argmax(exact, axis=-1)].astype(np.uint32)

    def greedy(self, meta: StepMeta, pool) -> mx.array:
        """Greedy token per sampled row, using the float32 rescore of the top candidates.

        Greedy only. It is exact unless the true argmax falls outside the low-bit top-k (miss rate measured by
        tools/mlx_lm_head_miss.py). A sampler with temperature > 0 must not use it (it needs the full
        distribution): use `logits(..., rescore=0)` with `lm_head_bits=None` for those rows."""
        x = self.hidden(meta, pool)[meta.sample_rows]
        x32 = mx.fast.rms_norm(x.astype(mx.float32), self.norm.astype(mx.float32), self.cfg.rms_norm_eps)
        lg = self._candidate_logits(x32)
        if self.dtype == mx.float32 and not self.head_policy:
            return mx.argmax(lg, axis=-1)
        k = self.rescore_k
        cand = topk_indices(lg, k)
        exact = (self.lm_head[cand].astype(mx.float32) * x32[:, None, :]).sum(axis=-1)
        return mx.take_along_axis(cand, mx.argmax(exact, axis=-1)[:, None], axis=-1)[:, 0]
