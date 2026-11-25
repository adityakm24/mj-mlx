# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team (transformers, Apache-2.0)
# Copyright (c) 2026 the MilliJoule authors. Licensed under the Apache License 2.0 (LICENSE); see NOTICE.
#
# Written from transformers' models/qwen3_5_moe/modeling_qwen3_5_moe.py (Apache License 2.0,
# http://www.apache.org/licenses/LICENSE-2.0). Changes: reimplemented in MLX for one flat varlen step with paged KV
# (full-attention layers) and per-request recurrent-state slots (linear-attention layers); zero-centered RMSNorm
# weights folded to 1 + w; k/v and b/a projections fused; Metal kernel for the gated delta rule (gated_delta.py);
# vectorized causal conv over all items; optional float32 residual stream and router ("precise").
"""Qwen3.5/3.6 MoE text model (Qwen3_5MoeForConditionalGeneration's language model) for the MLX backend.

Written from transformers' modeling_qwen3_5_moe.py (the reference it is tested against). Layer types:
- "linear_attention": Gated DeltaNet. A depthwise causal conv (kernel 4) over [q | k | v], SiLU, L2-normalized
  q and k, then the gated delta rule on a per-request recurrent state S [Hv, Dk, Dv] (float32):
      S <- S * exp(g);  S <- S + k (beta * (v - S^T k))^T;  o = S^T q
  followed by a gated RMSNorm (norm(o) * w * silu(z)) and out_proj. Conv and S live in per-request state slots
  (StepBatch.state_slots); an item with start == 0 zeroes its slot first.
- "full_attention": gated attention. q_proj gives [q | gate] per head, q/k get a zero-centered RMSNorm per head,
  partial rotary (first head_dim * partial_rotary_factor dims), paged KV through the block tables, output * sigmoid(gate).
- MLP: softmax router over all experts (float32), top-k renormalized, SwiGLU experts, plus a shared SwiGLU expert
  scaled by sigmoid(shared_expert_gate(x)).
Zero-centered RMSNorms (x * (1 + w)) are folded at load time to plain RMSNorm weights 1 + w.

Text only: multimodal RoPE (mrope) with equal t/h/w positions is ordinary 1-D RoPE, which is what is used here.
Dense Qwen3.5 models (num_experts absent from the config) use the same attention / DeltaNet layers with a plain
SwiGLU MLP (gate_proj, up_proj, down_proj) in place of the MoE block.
The MTP head is not used.
"""

from __future__ import annotations

import glob
import json
import math
import os
from dataclasses import dataclass
from functools import partial

import mlx.core as mx
import numpy as np

from .gpt_oss import GptOss, StepMeta, build_meta, topk_indices  # noqa: F401  (head utilities are shared)
from .prefill_attention import sdpa_prefill
from . import qmm_small, qmv_rows


@dataclass
class Qwen35MoeConfig:
    hidden_size: int
    num_hidden_layers: int
    layer_types: list
    vocab_size: int
    rms_norm_eps: float
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    partial_rotary_factor: float
    rope_theta: float
    num_experts: int  # 0: dense MLP (Qwen3.5 dense models)
    num_experts_per_tok: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    linear_conv_kernel_dim: int
    linear_key_head_dim: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_value_head_dim: int
    eos_token_id: int | list | None = None
    tie_word_embeddings: bool = False
    intermediate_size: int = 0  # dense MLP width (num_experts == 0)

    @classmethod
    def from_dict(cls, d: dict) -> "Qwen35MoeConfig":
        t = d.get("text_config", d)
        rp = t.get("rope_parameters") or {}
        return cls(
            hidden_size=t["hidden_size"], num_hidden_layers=t["num_hidden_layers"], layer_types=t["layer_types"],
            vocab_size=t["vocab_size"], rms_norm_eps=t["rms_norm_eps"], num_attention_heads=t["num_attention_heads"],
            num_key_value_heads=t["num_key_value_heads"], head_dim=t["head_dim"],
            partial_rotary_factor=rp.get("partial_rotary_factor", t.get("partial_rotary_factor", 1.0)),
            rope_theta=rp.get("rope_theta", t.get("rope_theta", 10000.0)), num_experts=t.get("num_experts", 0),
            num_experts_per_tok=t.get("num_experts_per_tok", 0), moe_intermediate_size=t.get("moe_intermediate_size", 0),
            shared_expert_intermediate_size=t.get("shared_expert_intermediate_size", 0),
            intermediate_size=t.get("intermediate_size", 0) if not t.get("num_experts") else 0,
            linear_conv_kernel_dim=t["linear_conv_kernel_dim"], linear_key_head_dim=t["linear_key_head_dim"],
            linear_num_key_heads=t["linear_num_key_heads"], linear_num_value_heads=t["linear_num_value_heads"],
            linear_value_head_dim=t["linear_value_head_dim"], eos_token_id=t.get("eos_token_id"),
            tie_word_embeddings=d.get("tie_word_embeddings", t.get("tie_word_embeddings", False)),
        )

    # attributes the shared gpt-oss head code reads
    @property
    def sliding_window(self) -> int:
        return 1 << 30

    @property
    def num_full_layers(self) -> int:
        return sum(lt == "full_attention" for lt in self.layer_types)


@dataclass
class QwenMeta:
    attn: StepMeta  # token ids, rope offsets, KV slots, attention groups, sample rows
    cu: np.ndarray  # [N+1] token row offsets per item
    lens: np.ndarray  # [N]
    state_slots: mx.array  # [N] int32
    fresh: mx.array  # [N] bool: start == 0 -> zero the slot
    decode_rows: mx.array | None  # rows of single-token items (batched path)
    decode_items: np.ndarray  # item indices of single-token items
    multi_items: np.ndarray  # item indices of longer items
    conv_idx: mx.array | None = None  # [T, K] rows of [conv_state (N*(K-1)) | qkv (T)] forming each token's window
    conv_state_idx: mx.array | None = None  # [N, K-1] rows that become each item's next conv state
    cu_mx: mx.array | None = None  # [N+1] int32
    item_of_slot: mx.array | None = None  # [N] int32 state row of each item in the gathered states (first use)
    state_slots_np: np.ndarray | None = None  # [N] host copy of state_slots
    any_fresh: bool = True  # some item starts at 0 (its conv state must be zeroed)
    fresh_u8: mx.array | None = None  # fresh as uint8 for the kernel (first use)
    capture: dict | None = None  # speculative verify: per DeltaNet layer (conv rows, state after each token)


class SlotStates:
    """Recurrent states of one layer, addressed per slot, with no cost that scales with the allocated slots.

    A single [slots, ...] pool cannot be updated in place by MLX here (passing it whole through the kernel copies
    every slot each layer; scattering into it after a gather copies it too): 9B decode, 1 request, 32 slots
    allocated: 86 ms/token vs 48 with this class. The last step's states stay as one stacked block; when the next
    step runs the same slots in the same order (steady decoding), the block goes straight back into the kernel, so
    a step costs only the kernel's own read + write of its N states. When the set changes, the N states are
    stacked once (slots that leave the block keep a view of it)."""

    def __init__(self, num_slots: int, shape: tuple, dtype):
        self.zero = mx.zeros(shape, dtype)  # shared by never-written slots (the kernel zeroes fresh items anyway)
        self.slots = [self.zero] * num_slots  # None: the slot's state is row block_pos[slot] of self.block
        self.block, self.block_ids, self.block_pos = None, (), {}
        self.shape, self.dtype = (num_slots, *shape), dtype

    def _get(self, i: int) -> mx.array:
        a = self.slots[i]
        return self.block[self.block_pos[i]] if a is None else a

    def gather(self, idx: np.ndarray) -> mx.array:
        key = tuple(int(i) for i in idx)
        if key == self.block_ids:
            return self.block
        return mx.stack([self._get(i) for i in key]) if len(key) > 1 else self._get(key[0])[None]

    def scatter(self, idx: np.ndarray, st: mx.array) -> None:
        key = tuple(int(i) for i in idx)
        keep = set(key)
        for i, j in self.block_pos.items():  # slots leaving the block keep their row (a view of the old block)
            if i not in keep:
                self.slots[i] = self.block[j]
        self.block, self.block_ids = st, key
        self.block_pos = {i: j for j, i in enumerate(key)}
        for i in key:
            self.slots[i] = None

    def reset(self, i: int) -> None:
        """Drop slot i's state (its request finished): the memory it pinned can be freed."""
        self.slots[i] = self.zero
        if i in self.block_pos:  # stays a row of the block until the next scatter; the key no longer matches
            del self.block_pos[i]
            self.block_ids = ()
            for j, p in list(self.block_pos.items()):  # remaining rows become views; the block is then unreferenced
                self.slots[j] = self.block[p]
            self.block, self.block_pos = None, {}

    def arrays(self) -> list[mx.array]:
        out = [] if self.block is None else [self.block]
        return out + list({id(a): a for a in self.slots if a is not None}.values())


def _silu(x):
    return x * mx.sigmoid(x)


_fuse = partial(mx.compile, shapeless=True)


# Elementwise chains compiled into single kernels (same ops, same order and dtypes, so the same numbers): a decode
# step is ~2x the kernels of mlx-lm's without this, and at one request each tiny kernel is pure launch latency.
# Shapeless (no retrace per token count), so no slicing inside: slices are taken by the callers.
@_fuse
def _sigmoid_f32(b):
    return mx.sigmoid(b.astype(mx.float32))


@_fuse
def _add_f32(a, bias):
    return a.astype(mx.float32) + bias


@_fuse
def _mul(a, b):
    return a * b


def _dn_gates(b, a, A, dt_bias):
    """beta = sigmoid(b), g = A * softplus(a + dt_bias), float32 (b, a in any float dtype). logaddexp stays outside
    the compiled kernels: MLX's compiled LogAddExp rounds differently from the op (1 ulp on ~1/3 of values), which
    flips near-tie greedy tokens."""
    return _sigmoid_f32(b), _mul(A, mx.logaddexp(_add_f32(a, dt_bias), 0.0))


@_fuse
def _conv_silu(win, w32, like):
    """Depthwise causal conv from gathered windows [T, K, C] with float32 weights [K, C], then SiLU, in like.dtype."""
    return _silu((win.astype(mx.float32) * w32[None]).sum(axis=1)).astype(like.dtype)


@_fuse
def _gated_out(o, z, like):
    """DeltaNet output gate after the RMSNorm: (o in the model dtype) * silu(z), computed in float32."""
    o = o.astype(like.dtype).astype(mx.float32)
    return (o * _silu(z.astype(mx.float32))).astype(like.dtype)


@_fuse
def _sigmoid_gate(o, gate):
    return o * mx.sigmoid(gate)


@_fuse
def _swiglu(g, u):
    return _silu(g) * u


@_fuse
def _residual(x, r):
    return x + r


class Lin:
    """A linear layer y = x @ W^T: bf16/float `w_t` (already transposed), or quantized (w, scales, biases) with bits
    and group size inferred from the shapes (mlx affine layout), so one model code path serves transformers-layout
    and mlx-quantized checkpoints alike."""

    def __init__(self, w: dict, name: str, in_dim: int, dtype):
        if f"{name}.scales" in w:
            self.q = (w[f"{name}.weight"], w[f"{name}.scales"], w.get(f"{name}.biases"))
            self.bits = 32 * self.q[0].shape[-1] // in_dim
            self.gs = in_dim // self.q[1].shape[-1]
            self.w_t = None
        else:
            self.q, self.w_t = None, w[f"{name}.weight"].astype(dtype).T

    @staticmethod
    def cat(lins: list["Lin"]) -> "Lin":
        """Fuse linears along the output dim (same quantization or all dense)."""
        out = Lin.__new__(Lin)
        if all(l.q is None for l in lins):
            out.q, out.w_t = None, mx.concatenate([l.w_t for l in lins], axis=1)
            return out
        assert all(l.q is not None and (l.bits, l.gs) == (lins[0].bits, lins[0].gs) for l in lins)
        out.w_t, out.bits, out.gs = None, lins[0].bits, lins[0].gs
        out.q = tuple(None if lins[0].q[j] is None else mx.concatenate([l.q[j] for l in lins], axis=0)
                      for j in range(3))
        return out

    def arrays(self) -> list[mx.array]:
        return [self.w_t] if self.q is None else [a for a in self.q if a is not None]


def mm(x, lin):
    if not isinstance(lin, Lin):
        return x @ lin
    if lin.q is None:
        return x @ lin.w_t
    if x.ndim == 2 and 2 <= x.shape[0] <= qmv_rows.MAX_ROWS:  # 2-4 rows: each row bit-identical to one-row decode
        if qmv_rows.eligible(x, *lin.q, lin.gs, lin.bits):
            return qmv_rows.quantized_matmul(x, *lin.q, group_size=lin.gs, bits=lin.bits)
        # other formats (e.g. Qwen3.6-35B's 8-bit router / shared-expert gate): one decode-shaped call per row
        return mx.concatenate([mx.quantized_matmul(x[i:i + 1], *lin.q, transpose=True, group_size=lin.gs,
                                                   bits=lin.bits) for i in range(x.shape[0])])
    if x.ndim == 2:  # decode batches of 5-16 rows take the small-tile kernel (qmm_small.py); the rest is MLX's
        return qmm_small.quantized_matmul(x, *lin.q, group_size=lin.gs, bits=lin.bits)
    return mx.quantized_matmul(x, *lin.q, transpose=True, group_size=lin.gs, bits=lin.bits)


def _swiglu_mm(x, wg, wu, wd):
    return mm(_silu(mm(x, wg)) * mm(x, wu), wd)


def _swiglu_fused(x, wgu, wd):
    """SwiGLU with gate|up fused along the output dim (one matmul; rows are independent, so the same numbers)."""
    h = mm(x, wgu)
    i = h.shape[-1] // 2
    return mm(_swiglu(h[:, :i], h[:, i:]), wd)


def _norm_key(k: str) -> str | None:
    """Map transformers (model.language_model.*) and mlx-vlm (language_model.model.*) names to model.* / lm_head."""
    for pre in ("vision_tower.", "model.visual.", "mtp.", "language_model.mtp.", "model.language_model.mtp."):
        if k.startswith(pre):
            return None
    k = k.replace("model.language_model.", "model.")
    k = k.replace("language_model.model.", "model.")
    return k.replace("language_model.lm_head.", "lm_head.")


PRED_STATS = os.environ.get("MJ_PRED_STATS") == "1"
# Streamed experts: prefetch the predicted experts of this many following layers (router of layer li+d applied to
# layer li's MoE input; decode accuracy on Qwen3.6-35B: 80-83% at d=1, 74-77% at d=2, 70-73% at d=3).
PREFETCH_DEPTH = int(os.environ.get("MJ_PREFETCH_DEPTH", "2"))  # RESULTS 1s: 2 beats 1 and 3


def _pred_stats(model, li, xr, idx):
    """Debug (MJ_PRED_STATS=1, forces host syncs): for d = 1..3, the top-k of layer li+d's router applied to layer
    li's MoE input, vs layer li+d's actual routing. model.pred_stats[d] = [hits, routed]."""
    st = model.__dict__.setdefault("pred_stats", {d: [0, 0] for d in (1, 2, 3)})
    pend = model.__dict__.setdefault("_pred_pending", {})
    k = model.cfg.num_experts_per_tok
    actual = set(np.array(idx).reshape(-1).tolist())
    for d, pred in pend.pop(li, []):
        st[d][0] += len(actual & pred)
        st[d][1] += len(actual)
    for d in (1, 2, 3):
        if li + d < len(model.layers):
            lg = mm(xr, model.layers[li + d]["router_wt"]).astype(mx.float32)
            top = mx.argpartition(-lg, kth=k - 1, axis=-1)[..., :k]
            pend.setdefault(li + d, []).append((d, set(np.array(top).reshape(-1).tolist())))


class Qwen35Moe(GptOss):
    """Weights plus one step's forward. Pools: {"kv": [per layer (K, V) or None], "conv": [...], "ssm": [...]}."""

    recurrent = True

    def __init__(self, path: str, dtype=mx.bfloat16, weights: dict | None = None, cfg: Qwen35MoeConfig | None = None,
                 precise: bool = True, lm_head_bits: int | None = -1, rescore_k: int = 16, state_dtype=mx.float32,
                 stream_experts: str | None = None, expert_slots: int = 64, lean: bool = False, io_threads: int = 8,
                 io_nocache: bool = True, prefetch: bool = True, state_layout: str = "per_slot",
                 map_experts: str | None = None):
        """state_layout: "per_slot" (SlotStates, cost O(running items)) or "pool" (the earlier single [slots, ...]
        array passed whole through the kernel, O(allocated slots); kept for A/B measurements only).
        state_dtype: storage dtype of the delta-rule state between steps (math is float32 either way).
        float32 is the accuracy-gate default; bfloat16 halves the state bytes read and written every step.
        stream_experts / expert_slots / lean / prefetch: SSD expert streaming on the shared core
        (expert_stream.py), as for gpt-oss: routed experts from the repacked files with `expert_slots` resident per
        layer; lean keeps the embedding and bf16 lm_head on disk (greedy only)."""
        self.state_dtype = state_dtype
        assert state_layout in ("per_slot", "pool")
        self.state_layout = state_layout
        self.store = None
        self.map = None
        if map_experts:  # every routed expert a zero-copy view of the memory-mapped files (expert_map.py)
            from .expert_map import MappedExperts

            self.map = MappedExperts(map_experts)
        if stream_experts:
            from .expert_stream import ExpertStore

            self.store = ExpertStore(stream_experts, threads=io_threads, nocache=io_nocache)
            self.expert_slots = expert_slots
        self.prefetch = prefetch and stream_experts is not None
        self.lean = lean
        if cfg is None:
            with open(os.path.join(path, "config.json")) as f:
                cfg = Qwen35MoeConfig.from_dict(json.load(f))
        self.cfg, self.dtype = cfg, dtype
        self.precise = precise and dtype != mx.float32
        self.block_size = 16
        if weights is None:
            weights = {}
            for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
                weights.update(mx.load(f))
        w = {}
        for k, v in weights.items():
            nk = _norm_key(k)
            if nk is not None:
                w[nk] = v
        weights = None  # only `w` holds the (lazy) source arrays from here on
        # mlx-lm-converted checkpoints store conv1d as [C, K, 1] and already fold zero-centered norms to 1 + w
        conv = next((v for k, v in w.items() if k.endswith("linear_attn.conv1d.weight")), None)
        self.mlx_layout = conv is not None and conv.shape[-1] == 1
        self.head_native_q = "lm_head.scales" in w  # a quantized head is the head: greedy = its argmax
        cast = lambda a: a.astype(dtype)
        self.embed_rows = self.head_rows = None
        lean_head_src = None
        if lean and ("model.embed_tokens.scales" in w or self.head_native_q):
            lean = False  # embedding / head already quantized (e.g. mlx 4-bit): small, kept resident as they are
        if lean:
            from .expert_stream import SafetensorsRows

            self.embed_rows = SafetensorsRows(path, ["model.language_model.embed_tokens.weight",
                                                     "model.embed_tokens.weight"])
            self.head_rows = SafetensorsRows(path, "lm_head.weight")
            w.pop("model.embed_tokens.weight")
            self.embed = None
            if cfg.hidden_size % 32:  # too narrow for a low-bit candidate head: keep the bf16 head resident
                self.lm_head = cast(w.pop("lm_head.weight"))
            else:
                lean_head_src = w.pop("lm_head.weight")
                self.lm_head = None
                lm_head_bits = 4
        elif "model.embed_tokens.scales" in w or self.head_native_q:  # quantized embedding / head (mlx 4-bit)
            H = cfg.hidden_size
            self.embed_lin = Lin(w, "model.embed_tokens", H, dtype)
            self.embed = None if self.embed_lin.q is not None else cast(w["model.embed_tokens.weight"])
            self.head_lin = Lin(w, "lm_head", H, dtype)
            self.lm_head = None
            lm_head_bits = None
        else:
            self.embed = cast(w.pop("model.embed_tokens.weight"))
            self.lm_head = cast(w.pop("lm_head.weight")) if "lm_head.weight" in w else self.embed
        nw = w.pop("model.norm.weight").astype(mx.float32)
        self.norm = nw if self.mlx_layout else 1.0 + nw
        mx.eval([a for a in (self.embed, self.lm_head, self.norm) if a is not None] +
                (self.embed_lin.arrays() + self.head_lin.arrays() if hasattr(self, "head_lin") else []))
        self.layers = []
        for i in range(cfg.num_hidden_layers):  # build, evaluate and drop sources layer by layer (bounded peak)
            L = self.load_layer(w, i)
            L["_index"] = i
            mx.eval([v for v in L.values() if isinstance(v, mx.array)] + list(L.get("gu_q", ())) +
                    list(L.get("dn_q", ())) + [a for v in L.values() if isinstance(v, Lin) for a in v.arrays()])
            for k in [k for k in w if k.startswith(f"model.layers.{i}.")]:
                del w[k]
            self.layers.append(L)
            mx.clear_cache()
        if self.map is not None:  # MJ_MLOCK=bg: pin the mapped experts in the background from here
            self.map.start_background_lock()
        rot = int(cfg.head_dim * cfg.partial_rotary_factor)
        self.rot_dims = rot
        self.attn_scale = 1.0 / math.sqrt(cfg.head_dim)
        self.rope_freqs = None
        self._init_head(lm_head_bits, rescore_k, lean_head_src)
        mx.eval(self.parameters())
        mx.clear_cache()  # drop load-time temporaries (pre-fusion/pre-cast copies) from MLX's buffer cache

    def _init_head(self, lm_head_bits, rescore_k, src=None):
        c = self.cfg
        self.rescore_k = rescore_k
        self.lm_head_gs = 64 if c.hidden_size % 64 == 0 else 32
        if lm_head_bits == -1:
            self.head_policy = [(4, 4), (48, 8)] if self.dtype != mx.float32 else []
        elif lm_head_bits:
            self.head_policy = [(1 << 30, lm_head_bits)]
        else:
            self.head_policy = []
        if c.hidden_size % 32:  # too narrow to quantize: always the bf16 head
            self.head_policy = []
        if getattr(self, "head_lin", None) is not None:
            self.head_policy = []
        src = self.lm_head if src is None else src.astype(self.dtype)
        self.lm_head_qs = {b: mx.quantize(src, self.lm_head_gs, b) for _, b in self.head_policy}
        self.lm_head_q = None

    # ---- weights ---------------------------------------------------------------------------------------------
    def load_layer(self, w: dict, i: int) -> dict:
        c, dt = self.cfg, self.dtype
        cast = lambda a: a.astype(dt)
        f32 = mx.float32
        # zero-centered RMSNorm weight, folded to 1 + w (mlx-converted checkpoints already did it); float32 (tiny)
        zc = (lambda a: a.astype(f32)) if self.mlx_layout else (lambda a: 1.0 + a.astype(f32))
        H = c.hidden_size
        lin = lambda name, din: Lin(w, name, din, dt)
        p = f"model.layers.{i}"
        L = dict(type=c.layer_types[i], ln1=zc(w[f"{p}.input_layernorm.weight"]),
                 ln2=zc(w[f"{p}.post_attention_layernorm.weight"]))
        if L["type"] == "linear_attention":
            a = f"{p}.linear_attn"
            cw = w[f"{a}.conv1d.weight"]
            cw = cw[:, :, 0] if self.mlx_layout else cw[:, 0, :]  # [C, K]
            qkv_l, z_l = lin(f"{a}.in_proj_qkv", H), lin(f"{a}.in_proj_z", H)
            ba_l = Lin.cat([lin(f"{a}.in_proj_b", H), lin(f"{a}.in_proj_a", H)])
            # one matmul for the input projections (rows are independent, so the outputs are the same numbers);
            # b|a stays separate when it is quantized differently from qkv / z
            try:
                L.update(in_wt=Lin.cat([qkv_l, z_l, ba_l]))
            except AssertionError:
                L.update(in_wt=Lin.cat([qkv_l, z_l]), ba_wt=ba_l)
            del qkv_l, z_l
            L.update(
                conv_w=cast(cw).T,  # [K, C]
                conv_w32=cast(cw).T.astype(f32),
                A=-mx.exp(w[f"{a}.A_log"].astype(f32)), dt_bias=w[f"{a}.dt_bias"].astype(f32),
                gnorm=cast(w[f"{a}.norm.weight"]),
                gnorm32=cast(w[f"{a}.norm.weight"]).astype(f32),  # cast once at load (same numbers)
                out_wt=lin(f"{a}.out_proj", c.linear_num_value_heads * c.linear_value_head_dim),
            )
        else:
            a = f"{p}.self_attn"
            L.update(
                q_wt=lin(f"{a}.q_proj", H), kv_wt=Lin.cat([lin(f"{a}.k_proj", H), lin(f"{a}.v_proj", H)]),
                q_norm=zc(w[f"{a}.q_norm.weight"]), k_norm=zc(w[f"{a}.k_norm.weight"]),
                o_wt=lin(f"{a}.o_proj", c.num_attention_heads * c.head_dim),
            )
        m = f"{p}.mlp"
        if c.num_experts == 0:  # dense SwiGLU MLP (Qwen3.5 dense models)
            g_l, u_l = lin(f"{m}.gate_proj", H), lin(f"{m}.up_proj", H)
            try:  # one gate|up matmul
                L.update(mlp_gu=Lin.cat([g_l, u_l]))
            except AssertionError:
                L.update(mlp_g=g_l, mlp_u=u_l)
            L.update(mlp_d=lin(f"{m}.down_proj", c.intermediate_size))
            return L
        router_dt = f32 if (self.precise or dt == f32) else dt
        shared = dict(
            router_wt=Lin(w, f"{m}.gate", H, router_dt),
            sh_g=lin(f"{m}.shared_expert.gate_proj", H), sh_u=lin(f"{m}.shared_expert.up_proj", H),
            sh_d=lin(f"{m}.shared_expert.down_proj", c.shared_expert_intermediate_size),
            sh_gate=Lin(w, f"{m}.shared_expert_gate", H, router_dt),
        )
        if self.store is not None:  # routed experts live on the SSD (shared streaming core)
            from .expert_stream import ExpertSlab, layer_slots

            L.update(shared, slab=ExpertSlab(self.store, i, layer_slots(self.expert_slots, i)),
                     slab_fmt=self.store.index["expert_format"])
            return L
        if self.map is not None:
            fmt = self.map.index["expert_format"]
            assert fmt["format"] == "affine", fmt
            e = self.map.layer(i)
            L.update(shared, gu_q=(e["gu_w"], e["gu_s"], e["gu_b"]), dn_q=(e["dn_w"], e["dn_s"], e["dn_b"]),
                     q_bits=fmt["bits"], q_gs=fmt["group_size"])
            return L
        quant = f"{m}.switch_mlp.gate_proj.scales" in w
        if quant:  # mlx-lm native quantized experts: fuse gate+up row-wise (groups run along the input dim)
            sw = f"{m}.switch_mlp"
            gu = None
            gu_q = tuple(mx.concatenate([w[f"{sw}.gate_proj.{t}"], w[f"{sw}.up_proj.{t}"]], axis=1)
                         for t in ("weight", "scales", "biases"))
            dn_q = tuple(w[f"{sw}.down_proj.{t}"] for t in ("weight", "scales", "biases"))
            qbits = 32 * w[f"{sw}.gate_proj.weight"].shape[-1] // c.hidden_size
            qgs = c.hidden_size // w[f"{sw}.gate_proj.scales"].shape[-1]
        elif f"{m}.experts.gate_up_proj" in w:  # fused transformers layout [E, 2I, H], gate first
            gu = w[f"{m}.experts.gate_up_proj"]
            dn = w[f"{m}.experts.down_proj"]  # [E, H, I]
        else:
            E = c.num_experts
            gu = mx.stack([mx.concatenate([w[f"{m}.experts.{e}.gate_proj.weight"],
                                           w[f"{m}.experts.{e}.up_proj.weight"]], axis=0) for e in range(E)])
            dn = mx.stack([w[f"{m}.experts.{e}.down_proj.weight"] for e in range(E)])
        if quant:
            L.update(gu_q=gu_q, dn_q=dn_q, q_bits=qbits, q_gs=qgs)
        else:
            L.update(gu_w=cast(gu).swapaxes(-1, -2), dn_w=cast(dn).swapaxes(-1, -2))  # [E, H, 2I], [E, I, H]
        L.update(shared)
        return L

    def parameters(self) -> list[mx.array]:
        out = [a for a in (self.embed, self.lm_head, self.norm) if a is not None]
        for q in getattr(self, "lm_head_qs", {}).values():
            out += [a for a in q if a is not None]
        for name in ("embed_lin", "head_lin"):
            if getattr(self, name, None) is not None:
                out += getattr(self, name).arrays()
        for L in self.layers:
            out += [v for v in L.values() if isinstance(v, mx.array)]
            out += [a for v in L.values() if isinstance(v, Lin) for a in v.arrays()]
            for k in ("gu_q", "dn_q"):
                out += list(L.get(k, ()))
            if "slab" in L:
                out += L["slab"].arrays()
        return out

    def expert_stats(self) -> dict:
        st = [L["slab"].stats() for L in self.layers if "slab" in L]
        h, m = sum(x["hits"] for x in st), sum(x["misses"] for x in st)
        return dict(hits=h, misses=m, hit_rate=h / (h + m) if h + m else None,
                    prefetch_issued=sum(x["prefetch_issued"] for x in st),
                    prefetch_used=sum(x["prefetch_used"] for x in st),
                    spec_used=sum(x.get("spec_used", 0) for x in st),
                    spec_wasted=sum(x.get("spec_wasted", 0) for x in st),
                    spec_pending_only=sum(x.get("spec_pending_only", 0) for x in st),
                    bytes_read=self.store.bytes_read if self.store else 0,
                    gpu_stream=self._gpu_stream.stats() if self.__dict__.get("_gpu_stream") is not None else None)

    def _slab_pair_fn(self, fmt: dict):
        """Qwen3.6 routed-expert math on (token, expert) pairs over slab slots (for expert_stream.run_moe)."""
        from .expert_stream import sorted_pairs

        I = self.cfg.moe_intermediate_size

        def pair(slab, x, tok, eid, slots, sorted_):
            xs, sl, order, inv = sorted_pairs(x, tok, slots, sorted_)
            w = slab.w
            if fmt["format"] == "affine":
                q = dict(transpose=True, group_size=fmt["group_size"], bits=fmt["bits"])
                h = mx.gather_qmm(xs, w["gu_w"], w["gu_s"], w["gu_b"], rhs_indices=sl, sorted_indices=sorted_, **q)
                h = _silu(h[..., :I]) * h[..., I:]
                y = mx.gather_qmm(h, w["dn_w"], w["dn_s"], w["dn_b"], rhs_indices=sl, sorted_indices=sorted_, **q)
            else:
                gu, dn = w["gu"], w["dn"]
                if gu.dtype != x.dtype:  # e.g. float32 test runs over a bf16 checkpoint
                    gu, dn = gu.astype(x.dtype), dn.astype(x.dtype)
                h = mx.gather_mm(xs, gu, rhs_indices=sl, sorted_indices=sorted_)
                h = _silu(h[..., :I]) * h[..., I:]
                y = mx.gather_mm(h, dn, rhs_indices=sl, sorted_indices=sorted_)
            return y[inv][:, 0, :] if sorted_ else y[:, 0, 0, :]

        return pair

    # ---- pools -----------------------------------------------------------------------------------------------
    def kv_bytes_per_token(self, dtype=None) -> int:
        c = self.cfg
        return c.num_full_layers * 2 * c.num_key_value_heads * c.head_dim * (dtype or self.dtype).size

    def state_bytes_per_slot(self) -> int:
        c = self.cfg
        n_lin = c.num_hidden_layers - c.num_full_layers
        conv = (c.linear_conv_kernel_dim - 1) * self._conv_dim() * self.dtype.size
        ssm = c.linear_num_value_heads * c.linear_key_head_dim * c.linear_value_head_dim * self.state_dtype.size
        return n_lin * (conv + ssm)

    def _conv_dim(self) -> int:
        c = self.cfg
        return 2 * c.linear_num_key_heads * c.linear_key_head_dim + c.linear_num_value_heads * c.linear_value_head_dim

    def new_pools(self, num_slots: int, num_state_slots: int) -> dict:
        c = self.cfg
        kv, conv, ssm = [], [], []
        for lt in c.layer_types:
            if lt == "full_attention":
                shape = (num_slots, c.num_key_value_heads, c.head_dim)
                kv.append((mx.zeros(shape, self.dtype), mx.zeros(shape, self.dtype)))
                conv.append(None)
                ssm.append(None)
            else:
                kv.append(None)
                conv.append(mx.zeros((num_state_slots, c.linear_conv_kernel_dim - 1, self._conv_dim()), self.dtype))
                # [slots, Hv, Dv, Dk]: column-major per head so the kernel's loads coalesce
                shape = (c.linear_num_value_heads, c.linear_value_head_dim, c.linear_key_head_dim)
                ssm.append(SlotStates(num_state_slots, shape, self.state_dtype) if self.state_layout == "per_slot"
                           else mx.zeros((num_state_slots, *shape), self.state_dtype))
        pools = dict(kv=kv, conv=conv, ssm=ssm)
        mx.eval(self.pool_arrays(pools))
        return pools

    def release_state_slots(self, pools, slots) -> None:
        for x in pools["ssm"]:
            if isinstance(x, SlotStates):
                for i in slots:
                    x.reset(int(i))

    def pool_arrays(self, pools) -> list[mx.array]:
        out = []
        for x in pools["kv"]:
            if x is not None:
                out += list(x)
        out += [x for x in pools["conv"] if x is not None]
        for x in pools["ssm"]:
            if x is not None:
                out += x.arrays() if isinstance(x, SlotStates) else [x]
        return out

    def prepare(self, s, block_size: int, paged_decode: bool) -> QwenMeta:
        attn = build_meta(s.token_ids, s.positions, s.starts, s.lens, s.block_tables, s.sample_rows, block_size,
                          self.cfg.sliding_window, s.slot_mapping, paged_decode)
        n = len(s.lens)
        cu = np.zeros(n + 1, np.int64)
        np.cumsum(s.lens, out=cu[1:])
        single = np.nonzero(s.lens == 1)[0]
        slots = s.state_slots if s.state_slots is not None else np.arange(n, dtype=np.int32)
        # Causal-conv windows for every token of every item, as row indices into [conv_in | qkv].
        K1 = self.cfg.linear_conv_kernel_dim - 1
        T = int(cu[-1])
        item = np.repeat(np.arange(n), s.lens)
        j = np.arange(T) - cu[item]  # local index in the item
        src = j[:, None] - K1 + np.arange(K1 + 1)[None, :]  # [T, K] local source positions
        conv_idx = np.where(src >= 0, n * K1 + cu[item][:, None] + src, item[:, None] * K1 + K1 + src)
        p = s.lens[:, None] + np.arange(K1)[None, :]  # positions in [conv_in_i | qkv_i] of the last K-1 inputs
        state_idx = np.where(p < K1, np.arange(n)[:, None] * K1 + p, n * K1 + cu[:-1][:, None] + p - K1)
        return QwenMeta(attn, cu, s.lens, mx.array(slots.astype(np.int32)), mx.array(s.starts == 0),
                        mx.array(cu[single].astype(np.int32)) if len(single) else None, single,
                        np.nonzero(s.lens != 1)[0], mx.array(conv_idx.astype(np.int32)),
                        mx.array(state_idx.astype(np.int32)), mx.array(cu.astype(np.int32)),
                        state_slots_np=np.asarray(slots, np.int64), any_fresh=bool((s.starts == 0).any()))

    # ---- layers ----------------------------------------------------------------------------------------------
    def _rms(self, x, w):
        return mx.fast.rms_norm(x, w, self.cfg.rms_norm_eps)

    def _full_attention(self, L, x, meta: QwenMeta, pools, li):
        c = self.cfg
        a = meta.attn
        T = x.shape[0]
        nq, nkv, D = c.num_attention_heads, c.num_key_value_heads, c.head_dim
        qg = mm(x, L["q_wt"]).reshape(T, nq, 2 * D)
        q, gate = qg[..., :D], qg[..., D:]
        kv = mm(x, L["kv_wt"]).reshape(T, 2 * nkv, D)
        k, v = kv[:, :nkv], kv[:, nkv:]
        q = self._rms(q.astype(mx.float32), L["q_norm"].astype(mx.float32)).astype(x.dtype)
        k = self._rms(k.astype(mx.float32), L["k_norm"].astype(mx.float32)).astype(x.dtype)
        rope = lambda t: mx.fast.rope(t[:, :, None, :], self.rot_dims, traditional=False, base=c.rope_theta,
                                      scale=1.0, offset=a.rope_offsets)[:, :, 0, :]
        q, k = rope(q), rope(k)
        kp, vp = pools["kv"][li]
        fused = a.fused_kv and kp.dtype == k.dtype  # decode-only step: the paged kernel writes K/V in place
        if not fused:
            kp[a.slots] = k
            vp[a.slots] = v
            pools["kv"][li] = (kp, vp)
        from .paged_attention import paged_decode_attention

        outs = []
        if getattr(self, "_no_sinks", None) is None:
            self._no_sinks = mx.full((nq,), -mx.inf)
            mx.eval(self._no_sinks)
        no_sinks = self._no_sinks
        for g in a.groups:
            if g.block_tables is not None:
                o = paged_decode_attention(q[g.rows], kp, vp, g.block_tables, g.kv_lens, no_sinks, self.attn_scale,
                                           self.block_size, 0, g.max_kv, *((k, v) if fused else ()))
                outs.append(o.reshape(g.B, nq * D))
                continue
            qq = q[g.rows].reshape(g.B, g.L, nq, D).transpose(0, 2, 1, 3)
            kk = kp[g.kv_full].transpose(0, 2, 1, 3)
            vv = vp[g.kv_full].transpose(0, 2, 1, 3)
            # keys are positions 0 .. kv_len-1 and the queries their last L rows: MLX's fused "causal" mode is exactly
            # that mask (bottom-right aligned; checked equal) and ~2x faster than an explicit mask at 2048 tokens
            o = sdpa_prefill(qq, kk, vv, scale=self.attn_scale, mask="causal" if g.B == 1 else g.mask_full)
            outs.append(o.transpose(0, 2, 1, 3).reshape(g.B * g.L, nq * D))
        o = outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=0)
        if a.inv_rows is not None:
            o = o[a.inv_rows]
        return mm(_sigmoid_gate(o, gate.reshape(T, nq * D)), L["o_wt"])

    use_kernel = True  # Metal gated-delta kernel + vectorized conv; False = per-token MLX ops (reference path)

    def _in_proj(self, L, x):
        """DeltaNet input projections from the fused matmul: qkv [T, C], z [T, Hv, Dv], b|a [T, 2Hv] (model dtype;
        cast to float32 by the caller)."""
        c = self.cfg
        T = x.shape[0]
        C = 2 * c.linear_num_key_heads * c.linear_key_head_dim + c.linear_num_value_heads * c.linear_value_head_dim
        Vd = c.linear_num_value_heads * c.linear_value_head_dim
        y = mm(x, L["in_wt"])
        ba = y[:, C + Vd:] if "ba_wt" not in L else mm(x, L["ba_wt"])
        return y[:, :C], y[:, C:C + Vd].reshape(T, c.linear_num_value_heads, c.linear_value_head_dim), ba

    def _linear_attention(self, L, x, meta: QwenMeta, pools, li):
        if not self.use_kernel:
            return self._linear_attention_ref(L, x, meta, pools, li)
        from .gated_delta import gated_delta_pool_packed

        c = self.cfg
        T = x.shape[0]
        Hk, Hv, Dk, Dv, K = (c.linear_num_key_heads, c.linear_num_value_heads, c.linear_key_head_dim,
                             c.linear_value_head_dim, c.linear_conv_kernel_dim)
        kd, N = Hk * Dk, len(meta.lens)
        qkv, z, ba = self._in_proj(L, x)
        beta, g = _dn_gates(ba[:, :Hv], ba[:, Hv:], L["A"], L["dt_bias"])
        conv_pool, ssm_pool = pools["conv"][li], pools["ssm"][li]
        conv_in = conv_pool[meta.state_slots]  # [N, K-1, C]
        if meta.any_fresh:  # items starting at 0 begin from a zero conv state (pure decode steps skip this)
            keep = (~meta.fresh).astype(mx.float32)
            conv_in = conv_in * keep[:, None, None].astype(conv_pool.dtype)
        ext = mx.concatenate([conv_in.reshape(N * (K - 1), -1), qkv.astype(conv_pool.dtype)], axis=0)
        # decode and short multi-token steps (speculative verify, small chunks): gather each token's K-wide window and
        # convolve in float32, as decode does. MLX's bf16 conv1d here cost accuracy: 2-token steps' mean KL vs
        # float32 went 6.9e-4 -> 2.7e-4 with this path (one-token decode: 1.8e-4; 9B, tools/mlx_vs_mlxlm.py).
        if len(meta.multi_items) == 0 or int(meta.lens.max()) <= 16:  # every item short (batched verify too)
            win = ext[meta.conv_idx]  # [T, K, C]
            co = _conv_silu(win, L["conv_w32"], x)
        else:  # prefill chunks: MLX's depthwise conv1d per item (its state prepended), in the model dtype as
            # transformers does; single-token items in the same step use the gather path
            if "conv_w_c1d" not in L:
                L["conv_w_c1d"] = L["conv_w"].T[:, :, None].astype(x.dtype)  # [C, K, 1]
            win = ext[meta.conv_idx]
            outs = []
            for it in range(N):
                a, b = int(meta.cu[it]), int(meta.cu[it + 1])
                if b - a == 1:
                    outs.append((win[a:b].astype(mx.float32) * L["conv_w32"][None]).sum(axis=1).astype(x.dtype))
                else:
                    seq = mx.concatenate([conv_in[it], qkv[a:b].astype(conv_pool.dtype)], axis=0)[None]
                    outs.append(mx.conv1d(seq.astype(x.dtype), L["conv_w_c1d"], groups=seq.shape[-1])[0])
            co = _silu(outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=0)).astype(x.dtype)
        conv_pool[meta.state_slots] = ext[meta.conv_state_idx]
        # q, k, v straight from the conv in the model dtype (bf16 halves the kernel's bytes); the kernel does the L2
        # norm of q / k and the 1/sqrt(Dk) scale in float32, as transformers does, and the recurrence in float32.
        qq = co[:, :kd].reshape(T, Hk, Dk)
        kk = co[:, kd:2 * kd].reshape(T, Hk, Dk)
        vv = co[:, 2 * kd:].reshape(T, Hv, Dv)
        # Only this step's items' states go through the kernel: its output is as large as its state input, so
        # passing a whole [slots, ...] pool cost O(allocated slots) per layer (9B decode, 1 request, 32 slots
        # allocated: 86 ms/token; per-slot states, SlotStates: 48).
        if not isinstance(ssm_pool, SlotStates):  # state_layout="pool" (A/B only): the whole pool in and out
            ios = mx.full((ssm_pool.shape[0],), -1, mx.int32)
            ios[meta.state_slots] = mx.arange(N, dtype=mx.int32)
            out, pools["ssm"][li] = gated_delta_pool_packed(qq, kk, vv, g, beta, ssm_pool, ios,
                                                            meta.fresh.astype(mx.uint8), meta.cu_mx, normalize=True)
            pools["conv"][li] = conv_pool
            o = self._rms(out, L["gnorm"].astype(mx.float32)).astype(x.dtype).astype(mx.float32)
            o = (o * _silu(z.astype(mx.float32))).astype(x.dtype).reshape(T, Hv * Dv)
            return mm(o, L["out_wt"])
        if meta.item_of_slot is None or meta.item_of_slot.shape[0] != N:
            meta.item_of_slot = mx.arange(N, dtype=mx.int32)  # compact states: item i is state row i
        if meta.fresh_u8 is None:
            meta.fresh_u8 = meta.fresh.astype(mx.uint8)
        st = ssm_pool.gather(meta.state_slots_np)
        if meta.capture is not None:  # keep every token's state and the conv inputs, to roll back to any token
            out, st, steps = gated_delta_pool_packed(qq, kk, vv, g, beta, st, meta.item_of_slot, meta.fresh_u8,
                                                     meta.cu_mx, normalize=True, keep_steps=True)
            meta.capture[li] = (ext, steps)
        else:
            out, st = gated_delta_pool_packed(qq, kk, vv, g, beta, st, meta.item_of_slot, meta.fresh_u8, meta.cu_mx,
                                              normalize=True)
        ssm_pool.scatter(meta.state_slots_np, st)
        pools["conv"][li] = conv_pool
        o = _gated_out(self._rms(out, L["gnorm32"]), z, x).reshape(T, Hv * Dv)
        return mm(o, L["out_wt"])

    def _linear_attention_ref(self, L, x, meta: QwenMeta, pools, li):
        c = self.cfg
        T = x.shape[0]
        Hk, Hv, Dk, Dv, K = (c.linear_num_key_heads, c.linear_num_value_heads, c.linear_key_head_dim,
                             c.linear_value_head_dim, c.linear_conv_kernel_dim)
        kd = Hk * Dk
        qkv, z, ba = self._in_proj(L, x)
        ba = ba.astype(mx.float32)
        beta = mx.sigmoid(ba[:, :Hv])
        g = L["A"] * mx.logaddexp(ba[:, Hv:] + L["dt_bias"], 0.0)  # A * softplus(a + dt_bias), A = -exp(A_log)
        conv_pool, ssm_pool = pools["conv"][li], pools["ssm"][li]
        keep = (~meta.fresh).astype(mx.float32)
        conv_in = conv_pool[meta.state_slots] * keep[:, None, None].astype(conv_pool.dtype)  # [N, K-1, C]
        S0 = (ssm_pool.gather(meta.state_slots_np).astype(mx.float32) * keep[:, None, None, None]).swapaxes(-1, -2)
        rows_out, conv_new, S_new = [], [None] * len(meta.lens), [None] * len(meta.lens)
        y = [None] * len(meta.lens)

        def heads(conv_out, n):
            qq, kk, vv = conv_out[:, :kd], conv_out[:, kd:2 * kd], conv_out[:, 2 * kd:]
            qq = qq.reshape(n, Hk, Dk).astype(mx.float32)
            kk = kk.reshape(n, Hk, Dk).astype(mx.float32)
            qq = qq * mx.rsqrt((qq * qq).sum(-1, keepdims=True) + 1e-6) / math.sqrt(Dk)
            kk = kk * mx.rsqrt((kk * kk).sum(-1, keepdims=True) + 1e-6)
            rep = Hv // Hk
            if rep > 1:
                qq = mx.repeat(qq, rep, axis=1)
                kk = mx.repeat(kk, rep, axis=1)
            return qq, kk, vv.reshape(n, Hv, Dv).astype(mx.float32)

        def step(S, q_t, k_t, v_t, g_t, b_t):
            S = S * mx.exp(g_t)[..., None, None]
            kv_mem = (S * k_t[..., :, None]).sum(axis=-2)  # [.., Hv, Dv]
            delta = (v_t - kv_mem) * b_t[..., None]
            S = S + k_t[..., :, None] * delta[..., None, :]
            return S, (S * q_t[..., :, None]).sum(axis=-2)

        if len(meta.decode_items):  # all single-token items at once
            di = mx.array(meta.decode_items.astype(np.int32))
            rows = meta.decode_rows
            ext = mx.concatenate([conv_in[di], qkv[rows][:, None, :]], axis=1)  # [B, K, C]
            co = _silu((ext.astype(mx.float32) * L["conv_w"].astype(mx.float32)[None]).sum(axis=1)).astype(x.dtype)
            qq, kk, vv = heads(co, len(meta.decode_items))
            Sd, od = step(S0[di], qq, kk, vv, g[rows], beta[rows])
            for j, it in enumerate(meta.decode_items):
                conv_new[it], S_new[it], y[it] = ext[j, 1:], Sd[j], od[j][None]
        for it in meta.multi_items:  # prefill chunks (and multi-token resumes): sequential recurrence
            a, b = int(meta.cu[it]), int(meta.cu[it + 1])
            n = b - a
            ext = mx.concatenate([conv_in[int(it)], qkv[a:b]], axis=0)  # [K-1+n, C]
            win = mx.stack([ext[j:j + n] for j in range(K)], axis=1)  # [n, K, C]
            co = _silu((win.astype(mx.float32) * L["conv_w"].astype(mx.float32)[None]).sum(axis=1)).astype(x.dtype)
            qq, kk, vv = heads(co, n)
            S = S0[int(it)]
            outs = []
            for t in range(n):
                S, o = step(S, qq[t], kk[t], vv[t], g[a + t], beta[a + t])
                outs.append(o)
            conv_new[it], S_new[it], y[it] = ext[-(K - 1):], S, mx.stack(outs)
        order = np.argsort(np.concatenate([meta.decode_items, meta.multi_items]))
        out = mx.concatenate(y, axis=0)  # item order == row order (items are contiguous in token rows)
        conv_pool[meta.state_slots] = mx.stack(conv_new).astype(conv_pool.dtype)
        ssm_pool.scatter(meta.state_slots_np, mx.stack(S_new).swapaxes(-1, -2).astype(ssm_pool.dtype))
        pools["conv"][li] = conv_pool
        del order
        # gated RMSNorm per value head, then out_proj
        o = self._rms(out, L["gnorm"].astype(mx.float32)).astype(x.dtype).astype(mx.float32)
        o = (o * _silu(z.astype(mx.float32))).astype(x.dtype).reshape(T, Hv * Dv)
        return mm(o, L["out_wt"])

    def _moe(self, L, x, x32=None):
        if "mlp_gu" in L:
            return _swiglu_fused(x, L["mlp_gu"], L["mlp_d"])
        if "mlp_g" in L:
            return _swiglu_mm(x, L["mlp_g"], L["mlp_u"], L["mlp_d"])
        c = self.cfg
        k = c.num_experts_per_tok
        xr = x if x32 is None else x32
        probs = mx.softmax(mm(xr, L["router_wt"]).astype(mx.float32), axis=-1, precise=True)
        idx = mx.stop_gradient(mx.argpartition(-probs, kth=k - 1, axis=-1)[..., :k])
        wts = mx.take_along_axis(probs, idx, axis=-1)
        wts = (wts / wts.sum(axis=-1, keepdims=True)).astype(x.dtype)
        T = x.shape[0]
        I = c.moe_intermediate_size
        if "slab" in L:
            from .expert_stream import run_moe

            nxt = None
            li = L["_index"]
            if self.prefetch and li + 1 < len(self.layers):
                nxt = [(self.layers[j]["slab"], mm(xr, self.layers[j]["router_wt"]).astype(mx.float32))
                       for j in range(li + 1, min(li + 1 + PREFETCH_DEPTH, len(self.layers)))]
            if PRED_STATS:  # debug: how well layer li's MoE input predicts the routing of layers li+1 .. li+3
                _pred_stats(self, li, xr, idx)
            cb = self.__dict__.get("route_trace")
            if cb is not None:  # debug / data collection (tools/mlx_router_lookahead.py): forces a host sync
                cb(li, xr, idx)
            from . import expert_stream as _es

            if _es.GPU_STREAM and T * k < 64:  # decode: GPU-driven streaming, no host sync (C++ extension)
                gs = self.__dict__.get("_gpu_stream")
                if gs is None:
                    gs = self._gpu_stream = _es.GpuStreaming(self.store, [Lx["slab"] for Lx in self.layers])
                preds = [mx.argpartition(lg, kth=-k, axis=-1)[..., -k:] for _, lg in (nxt or [])]
                slots = gs.route(li, idx, preds)
                tok = np.repeat(np.arange(T), k)
                y = self._slab_pair_fn(L["slab_fmt"])(L["slab"], x, tok, idx.reshape(-1), slots, False)
                routed = (y.reshape(T, k, -1) * wts[..., None]).sum(axis=1)
            else:
                if self.__dict__.get("_gpu_stream") is not None:
                    self._gpu_stream.to_py()
                routed = run_moe(L["slab"], x, idx, wts, self._slab_pair_fn(L["slab_fmt"]), nxt)
            shared = _swiglu_mm(x, L["sh_g"], L["sh_u"], L["sh_d"])
            sg = mx.sigmoid(mm(xr, L["sh_gate"]).astype(mx.float32)).astype(x.dtype)
            return routed + sg * shared
        if "gu_q" in L:
            qkw = dict(transpose=True, group_size=L["q_gs"], bits=L["q_bits"])
            mm_gu = lambda a, ix, srt: mx.gather_qmm(a, *L["gu_q"], rhs_indices=ix, sorted_indices=srt, **qkw)
            mm_dn = lambda a, ix, srt: mx.gather_qmm(a, *L["dn_q"], rhs_indices=ix, sorted_indices=srt, **qkw)
        else:
            mm_gu = lambda a, ix, srt: mx.gather_mm(a, L["gu_w"], rhs_indices=ix, sorted_indices=srt)
            mm_dn = lambda a, ix, srt: mx.gather_mm(a, L["dn_w"], rhs_indices=ix, sorted_indices=srt)
        if idx.size >= 64:
            flat = idx.flatten()
            order = mx.argsort(flat)
            inv = mx.argsort(order)
            si = flat[order]
            h = mm_gu(x[order // k][:, None, :], si, True)
            h = _silu(h[..., :I]) * h[..., I:]
            yv = mm_dn(h, si, True)[inv].reshape(T, k, -1)
        else:
            h = mm_gu(x[:, None, None, :], idx, False)
            h = _silu(h[..., :I]) * h[..., I:]
            yv = mm_dn(h, idx, False).squeeze(-2)
        routed = (yv * wts[..., None]).sum(axis=1)
        shared = _swiglu_mm(x, L["sh_g"], L["sh_u"], L["sh_d"])
        sg = mx.sigmoid(mm(xr, L["sh_gate"]).astype(mx.float32)).astype(x.dtype)
        return routed + sg * shared

    def hidden(self, meta: QwenMeta, pools, x: mx.array | None = None) -> mx.array:
        dt = self.dtype
        if x is None:
            x = self._embed(meta.attn.token_ids)
        if self.precise:
            x = x.astype(mx.float32)
        for li, L in enumerate(self.layers):
            h = self._rms(x, L["ln1"].astype(x.dtype)).astype(dt)
            if L["type"] == "full_attention":
                x = _residual(x, self._full_attention(L, h, meta, pools, li))
            else:
                x = _residual(x, self._linear_attention(L, h, meta, pools, li))
            h32 = self._rms(x, L["ln2"].astype(x.dtype))
            x = _residual(x, self._moe(L, h32.astype(dt), h32 if self.precise else None))
        return x

    def _embed(self, ids: mx.array) -> mx.array:
        if self.embed is not None:
            return self.embed[ids]
        if getattr(self, "embed_lin", None) is not None and self.embed_lin.q is not None:
            wq, sq, bq = self.embed_lin.q
            # dequantize in the model dtype (MLX returns the scales' dtype; bf16 scales would round a float32 run)
            sq, bq = sq[ids].astype(self.dtype), None if bq is None else bq[ids].astype(self.dtype)
            return mx.dequantize(wq[ids], sq, bq, group_size=self.embed_lin.gs, bits=self.embed_lin.bits)
        return self.embed_rows.take(np.array(ids), self.dtype)

    def greedy_from(self, x32: mx.array) -> mx.array:
        if getattr(self, "head_lin", None) is not None:  # the quantized head is the model's head: plain argmax
            return mx.argmax(mm(x32.astype(self.dtype), self.head_lin), axis=-1)
        return super().greedy_from(x32)

    def full_logits(self, x32: mx.array) -> mx.array:
        if getattr(self, "head_lin", None) is not None:
            return mm(x32.astype(self.dtype), self.head_lin).astype(mx.float32)
        return super().full_logits(x32)

    def final_hidden(self, meta: QwenMeta, pools) -> mx.array:
        x = self.hidden(meta, pools)[meta.attn.sample_rows]
        return mx.fast.rms_norm(x.astype(mx.float32), self.norm.astype(mx.float32), self.cfg.rms_norm_eps)
