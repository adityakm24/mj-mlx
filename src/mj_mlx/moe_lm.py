# Copyright (c) 2026 the MilliJoule authors. Licensed under the Apache License 2.0 (LICENSE); see NOTICE.
#
# Written from transformers' modeling files (Apache License 2.0, http://www.apache.org/licenses/LICENSE-2.0) of
# mixtral, qwen2_moe, qwen3_moe, olmoe, deepseek_v2, deepseek_v3 and glm4_moe, which are the references the tests
# compare against. Changes: one configurable model in MLX over one flat varlen step with paged KV; MLA attention as
# single-head attention over the latent cache (weight absorption); routers in float32.
"""General decoder-only MoE language model for the MLX backend (design: docs/moe-design.md).

One class, `MoeLM`, configured from config.json by `MoeLMConfig.from_hf`. Pieces (each tested against transformers
on tiny random checkpoints, tests/test_moe_lm.py):
- attention "gqa": q/k/v (+ optional bias), q/k RMSNorm per head ("head") or over the whole projection ("full"),
  rotary on the first rot_dims of each head (default / linear / llama3 / yarn inverse frequencies; half-split or
  interleaved pairs), paged KV, optional sliding window per layer.
- attention "mla" (DeepSeek-V2/V3): q (optionally through a LoRA with an RMSNorm), a latent c = norm(kv_a(x)[:r])
  and a shared rope key. The cache row is [c | k_rope] (one head); q_nope is absorbed through kv_b's key half, the
  attention output's first r dims go through kv_b's value half. The paged decode kernel and SDPA run it unchanged
  as single-head attention with K = V = the cache row.
- routers: "softmax" (top-k of softmax, optional renorm), "softmax_group" (DeepSeek-V2: greedy or group-limited
  greedy, x routed_scaling_factor), "sigmoid_group" (DeepSeek-V3 / GLM-4-MoE: sigmoid + correction bias for the
  choice, group top-k by the sum of each group's top 2, renorm, x scaling).
- experts: SwiGLU, resident (bf16 / float32 gather_mm or mlx affine gather_qmm); shared experts plain or
  sigmoid-gated; dense MLP layers where the config says so.
The head (low-bit candidates + exact rescore, quantized mlx heads) and embedding handling are Qwen35Moe's.
"""

from __future__ import annotations

import glob
import json
import math
import os
from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np

from .gpt_oss import StepMeta
from .paged_attention import paged_decode_attention
from .mla_decode import mla_decode_attention
from .prefill_attention import sdpa_prefill
from .qwen3_5_moe import Lin, Qwen35Moe, _norm_key, _residual, _silu, _swiglu, _swiglu_fused, _swiglu_mm, mm

# prefill-sized MoE steps: fused un-permute + weighted sum (moe_kernels.combine_sorted; float32 inside, not bitwise
# equal to the ops, so only for >= 256 pairs, never decode / verify). MJ_PREFILL_GLUE=0: off (as gpt-oss).
FUSED_COMBINE = os.environ.get("MJ_PREFILL_GLUE", "1") == "1"
# MLA decode rows: "sdpa" (SDPA over a gather of the cached rows), "kernel" (mla_decode.py: the paged latent rows read
# in place, split-K), "paged" (the GQA paged kernel: one KV head, very slow at length, v14 probe)
MLA_DECODE = os.environ.get("MJ_MLA_DECODE", "sdpa")
FAMILIES = ("mixtral", "qwen2_moe", "qwen3_moe", "olmoe", "deepseek_v2", "deepseek_v3", "glm4_moe")


# ---- rope -----------------------------------------------------------------------------------------------------
def rope_inv_freq(dim: int, base: float, rs: dict | None, max_pos: int | None = None) -> tuple[np.ndarray, float]:
    """Inverse frequencies [dim/2] and attention factor, as transformers' ROPE_INIT_FUNCTIONS (default, linear,
    llama3, yarn)."""
    inv = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.int64).astype(np.float32) / dim))
    t = (rs or {}).get("rope_type", (rs or {}).get("type", "default")) or "default"
    if t == "default":
        return inv.astype(np.float32), 1.0
    if t == "linear":
        return (inv / float(rs["factor"])).astype(np.float32), 1.0
    if t == "llama3":
        f, lo, hi = float(rs["factor"]), float(rs["low_freq_factor"]), float(rs["high_freq_factor"])
        old = float(rs["original_max_position_embeddings"])
        lo_w, hi_w = old / lo, old / hi
        wav = 2 * math.pi / inv
        out = np.where(wav > lo_w, inv / f, inv)
        smooth = (old / wav - lo) / (hi - lo)
        smoothed = (1 - smooth) * out / f + smooth * out
        med = ~(wav < hi_w) & ~(wav > lo_w)
        return np.where(med, smoothed, out).astype(np.float32), 1.0
    if t == "yarn":
        factor = rs.get("factor")
        orig = rs["original_max_position_embeddings"]
        if factor is None:
            factor = max_pos / orig
        factor = float(factor)

        def get_mscale(scale, m=1.0):
            return 1.0 if scale <= 1 else 0.1 * m * math.log(scale) + 1.0

        af = rs.get("attention_factor")
        if af is None:
            m, mad = rs.get("mscale"), rs.get("mscale_all_dim")
            af = float(get_mscale(factor, m) / get_mscale(factor, mad)) if (m and mad) else get_mscale(factor)
        bf, bs = rs.get("beta_fast") or 32, rs.get("beta_slow") or 1

        def corr(rot):
            return (dim * math.log(orig / (rot * 2 * math.pi))) / (2 * math.log(base))

        low, high = corr(bf), corr(bs)
        if rs.get("truncate", True):
            low, high = math.floor(low), math.ceil(high)
        low, high = max(low, 0), min(high, dim - 1)
        if low == high:
            high += 0.001
        ramp = np.clip((np.arange(dim // 2, dtype=np.float32) - low) / (high - low), 0, 1)
        ext = 1 - ramp
        pos = base ** (np.arange(0, dim, 2).astype(np.float32) / dim)
        out = (1.0 / (factor * pos)) * (1 - ext) + (1.0 / pos) * ext
        return out.astype(np.float32), float(af)
    raise NotImplementedError(f"rope_type {t!r}")


# ---- config ---------------------------------------------------------------------------------------------------
@dataclass
class MoeLMConfig:
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    vocab_size: int
    rms_norm_eps: float
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int  # the KV cache row width per head (MLA: kv_lora_rank + qk_rope_head_dim)
    rope_theta: float = 10000.0
    rope_scaling: dict | None = None
    rot_dims: int = 0  # rotary dims per head (MLA: qk_rope_head_dim)
    rope_traditional: bool = False  # interleaved pairs (DeepSeek)
    max_position_embeddings: int | None = None
    attn: str = "gqa"  # "gqa" | "mla"
    qk_norm: str = "none"  # "none" | "head" | "full"
    attn_head_dim: int = 0  # GQA head dim
    window: int = 0  # sliding window of the sliding layers (0: none)
    sliding: list = field(default_factory=list)  # per layer
    # MLA
    q_lora_rank: int | None = None
    kv_lora_rank: int = 0
    qk_nope_head_dim: int = 0
    qk_rope_head_dim: int = 0
    v_head_dim: int = 0
    # MLP / MoE
    num_experts: int = 0
    num_experts_per_tok: int = 0
    moe_intermediate_size: int = 0
    intermediate_size: int = 0  # dense MLP layers
    router: str = "softmax"  # "softmax" | "softmax_group" | "sigmoid_group"
    norm_topk_prob: bool = False
    routed_scaling_factor: float = 1.0
    topk_method: str = "greedy"
    n_group: int = 1
    topk_group: int = 1
    n_shared_experts: int = 0  # plain shared MLP of n * moe_intermediate_size (DeepSeek / GLM)
    shared_expert_intermediate_size: int = 0  # sigmoid-gated shared expert (Qwen2-MoE)
    dense_layers: list = field(default_factory=list)  # layer indices with a dense MLP
    tie_word_embeddings: bool = False
    eos_token_id: int | list | None = None

    @property
    def sliding_window(self) -> int:  # what build_meta uses for the sliding layers
        return self.window or (1 << 30)

    @classmethod
    def from_hf(cls, d: dict) -> "MoeLMConfig":
        t = d.get("text_config", d)
        mt = d.get("model_type", t.get("model_type"))
        if mt not in FAMILIES:
            raise ValueError(f"MoeLM does not know model_type {mt!r} (known: {', '.join(FAMILIES)})")
        rp = dict(t.get("rope_parameters") or t.get("rope_scaling") or {})
        theta = float(rp.get("rope_theta", t.get("rope_theta", 10000.0)))
        H, nq = t["hidden_size"], t["num_attention_heads"]
        nkv = t.get("num_key_value_heads") or nq
        L = t["num_hidden_layers"]
        hd = t.get("head_dim") or H // nq
        prf = float(rp.get("partial_rotary_factor", t.get("partial_rotary_factor", 1.0)))
        c = dict(model_type=mt, hidden_size=H, num_hidden_layers=L, vocab_size=t["vocab_size"],
                 rms_norm_eps=t.get("rms_norm_eps", 1e-6), num_attention_heads=nq, num_key_value_heads=nkv,
                 head_dim=hd, attn_head_dim=hd, rope_theta=theta, rope_scaling=rp or None,
                 rot_dims=int(hd * prf), max_position_embeddings=t.get("max_position_embeddings"),
                 tie_word_embeddings=bool(d.get("tie_word_embeddings", t.get("tie_word_embeddings", False))),
                 eos_token_id=t.get("eos_token_id", d.get("eos_token_id")))
        # sliding window: per layer from layer_types when the config has them, else every layer (Mixtral)
        win = t.get("sliding_window") or 0
        lt = t.get("layer_types")
        if win and t.get("use_sliding_window", True) is not False:
            sl = [x == "sliding_attention" for x in lt] if lt else [True] * L
        else:
            win, sl = 0, [False] * L
        c.update(window=int(win) if any(sl) else 0, sliding=sl)
        if mt == "mixtral":
            c.update(num_experts=t.get("num_local_experts") or t["num_experts"], num_experts_per_tok=t["num_experts_per_tok"],
                     moe_intermediate_size=t["intermediate_size"], norm_topk_prob=True)
        elif mt in ("qwen2_moe", "qwen3_moe", "olmoe"):
            ne = t.get("num_experts") or t.get("num_local_experts") or 0
            step = t.get("decoder_sparse_step", 1) or 1
            mlp_only = set(t.get("mlp_only_layers") or [])
            dense = [i for i in range(L) if i in mlp_only or not (ne > 0 and (i + 1) % step == 0)]
            c.update(num_experts=ne, num_experts_per_tok=t["num_experts_per_tok"],
                     moe_intermediate_size=t.get("moe_intermediate_size") or t["intermediate_size"],
                     intermediate_size=t.get("intermediate_size", 0), norm_topk_prob=bool(t.get("norm_topk_prob")),
                     shared_expert_intermediate_size=t.get("shared_expert_intermediate_size", 0) or 0,
                     dense_layers=dense, qk_norm={"qwen3_moe": "head", "olmoe": "full"}.get(mt, "none"))
        elif mt in ("deepseek_v2", "deepseek_v3"):
            r, rope = t["kv_lora_rank"], t["qk_rope_head_dim"]
            c.update(attn="mla", q_lora_rank=t.get("q_lora_rank"), kv_lora_rank=r, qk_nope_head_dim=t["qk_nope_head_dim"],
                     qk_rope_head_dim=rope, v_head_dim=t["v_head_dim"], head_dim=r + rope, num_key_value_heads=1,
                     rot_dims=rope, rope_traditional=mt == "deepseek_v2" or bool(t.get("rope_interleave", True)),
                     num_experts=t.get("n_routed_experts") or t.get("num_local_experts") or t.get("num_experts"),
                     num_experts_per_tok=t["num_experts_per_tok"], moe_intermediate_size=t["moe_intermediate_size"],
                     intermediate_size=t["intermediate_size"], routed_scaling_factor=float(t.get("routed_scaling_factor", 1.0)),
                     n_group=t.get("n_group") or 1, topk_group=t.get("topk_group") or 1,
                     n_shared_experts=t.get("n_shared_experts") or 0,
                     dense_layers=list(range(t.get("first_k_dense_replace", 0) or 0)))
            if mt == "deepseek_v2":
                c.update(router="softmax_group", topk_method=t.get("topk_method", "greedy"), norm_topk_prob=False)
            else:
                c.update(router="sigmoid_group", norm_topk_prob=bool(t.get("norm_topk_prob", True)))
        elif mt == "glm4_moe":
            c.update(num_experts=t.get("n_routed_experts") or t.get("num_local_experts"),
                     num_experts_per_tok=t["num_experts_per_tok"],
                     moe_intermediate_size=t["moe_intermediate_size"], intermediate_size=t["intermediate_size"],
                     router="sigmoid_group", norm_topk_prob=bool(t.get("norm_topk_prob", True)),
                     routed_scaling_factor=float(t.get("routed_scaling_factor", 1.0)),
                     n_group=t.get("n_group") or 1, topk_group=t.get("topk_group") or 1,
                     n_shared_experts=t.get("n_shared_experts") or 0,
                     dense_layers=list(range(t.get("first_k_dense_replace", 0) or 0)),
                     qk_norm="head" if t.get("use_qk_norm") else "none")
        return cls(**c)


def stack_switch_mlp(w: dict, m: str, num_experts: int) -> None:
    """Per-expert quantized tensors (mlx checkpoints written before mlx-lm stacked experts:
    {m}.experts.{e}.{gate,up,down}_proj.{weight,scales,biases}) -> {m}.switch_mlp.*, in place."""
    if f"{m}.experts.0.gate_proj.scales" not in w:
        return
    for proj in ("gate_proj", "up_proj", "down_proj"):
        for t in ("weight", "scales", "biases"):
            keys = [f"{m}.experts.{e}.{proj}.{t}" for e in range(num_experts)]
            if keys[0] in w:
                w[f"{m}.switch_mlp.{proj}.{t}"] = mx.stack([w.pop(k) for k in keys])


def normalize_keys(weights: dict) -> dict:
    """One key layout for every family: model.layers.{i}.{self_attn,mlp}..., per-expert gate/up/down_proj."""
    out = {}
    for k, v in weights.items():
        k = _norm_key(k)
        if k is None:
            continue
        k = k.replace(".block_sparse_moe.", ".mlp.")
        for a, b in ((".w1.", ".gate_proj."), (".w3.", ".up_proj."), (".w2.", ".down_proj.")):
            if ".mlp.experts." in k:
                k = k.replace(a, b)
        out[k] = v
    return out


# ---- the model ------------------------------------------------------------------------------------------------
class MoeLM(Qwen35Moe):
    """Weights plus one step's forward; the KV pool is a list of per-layer (K, V) pairs (MLA: the same array)."""

    recurrent = False

    def __init__(self, path: str, dtype=mx.bfloat16, weights: dict | None = None, cfg: MoeLMConfig | None = None,
                 precise: bool = True, lm_head_bits: int | None = -1, rescore_k: int = 16, lean: bool = False,
                 map_experts: str | None = None, stream_experts: str | None = None, expert_slots: int = 64,
                 io_threads: int = 8, io_nocache: bool = True, prefetch: bool = True):
        self.store = self.map = None
        if map_experts:  # routed experts as zero-copy views of part-major files (placement.prepare / expert_map)
            from .expert_map import MappedExperts

            self.map = MappedExperts(map_experts)
        if stream_experts:  # routed experts on the SSD, expert_slots per layer resident (expert_stream)
            from .expert_stream import ExpertStore

            self.store = ExpertStore(stream_experts, threads=io_threads, nocache=io_nocache)
            self.expert_slots = expert_slots
        # decode steps through the C++ GPU-driven streamer when it is built (MJ_GPU_STREAM=0: the Python path)
        self.gpu_stream = False
        if stream_experts and os.environ.get("MJ_GPU_STREAM", "1") != "0":
            try:
                from .expert_stream import load_gpu_streamer

                self.gpu_stream = hasattr(load_gpu_streamer(), "Streamer")
            except ImportError:
                pass
        self.prefetch = prefetch and stream_experts is not None
        self.lean = lean
        if cfg is None:
            with open(os.path.join(path, "config.json")) as f:
                cfg = MoeLMConfig.from_hf(json.load(f))
        self.cfg, self.dtype = cfg, dtype
        self.precise = precise and dtype != mx.float32
        self.block_size = 16
        if weights is None:
            weights = {}
            for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
                weights.update(mx.load(f))
        w = normalize_keys(weights)
        weights = None
        self.mlx_layout = True  # plain RMSNorm weights (no zero-centered folding)
        self.head_native_q = "lm_head.scales" in w
        cast = lambda a: a.astype(dtype)
        self.embed_rows = self.head_rows = None
        lean_head_src = None
        if lean and ("model.embed_tokens.scales" in w or self.head_native_q):
            lean = False
        if cfg.tie_word_embeddings and "lm_head.weight" not in w and "lm_head.scales" not in w:
            for t in ("weight", "scales", "biases"):
                if f"model.embed_tokens.{t}" in w:
                    w[f"lm_head.{t}"] = w[f"model.embed_tokens.{t}"]
            self.head_native_q = "lm_head.scales" in w
        if lean:
            from .expert_stream import SafetensorsRows

            self.embed_rows = SafetensorsRows(path, ["model.embed_tokens.weight", "model.language_model.embed_tokens.weight"])
            self.head_rows = SafetensorsRows(path, ["lm_head.weight", "model.embed_tokens.weight"]
                                             if cfg.tie_word_embeddings else "lm_head.weight")
            w.pop("model.embed_tokens.weight")
            self.embed = None
            src = w.pop("lm_head.weight")
            if cfg.hidden_size % 32:
                self.lm_head = cast(src)
            else:
                lean_head_src, self.lm_head = src, None
                from .gpt_oss import LEAN_HEAD_BITS

                lm_head_bits = LEAN_HEAD_BITS
        elif "model.embed_tokens.scales" in w or self.head_native_q:
            H = cfg.hidden_size
            self.embed_lin = Lin(w, "model.embed_tokens", H, dtype)
            self.embed = None if self.embed_lin.q is not None else cast(w["model.embed_tokens.weight"])
            self.head_lin = Lin(w, "lm_head", H, dtype)
            self.lm_head = None
            lm_head_bits = None
        else:
            self.embed = cast(w.pop("model.embed_tokens.weight"))
            self.lm_head = cast(w.pop("lm_head.weight")) if "lm_head.weight" in w else self.embed
        self.norm = w.pop("model.norm.weight").astype(mx.float32)
        mx.eval([a for a in (self.embed, self.lm_head, self.norm) if a is not None] +
                (self.embed_lin.arrays() + self.head_lin.arrays() if hasattr(self, "head_lin") else []))
        c = cfg
        inv, self.rope_af = rope_inv_freq(c.rot_dims, c.rope_theta, c.rope_scaling, c.max_position_embeddings)
        self.rope_freqs = mx.array(1.0 / inv)
        self.rot_dims = c.rot_dims
        if c.attn == "mla":
            qk = c.qk_nope_head_dim + c.qk_rope_head_dim
            s = qk ** -0.5
            rs = c.rope_scaling or {}
            if rs.get("rope_type", rs.get("type", "default")) not in (None, "default") and rs.get("mscale_all_dim"):
                m = 0.1 * rs["mscale_all_dim"] * math.log(rs["factor"]) + 1.0 if rs["factor"] > 1 else 1.0
                s = s * m * m
            self.attn_scale = s
        else:
            self.attn_scale = 1.0 / math.sqrt(c.attn_head_dim)
        self.layers = []
        for i in range(c.num_hidden_layers):
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
        self._init_head(lm_head_bits, rescore_k, lean_head_src)
        self._no_sinks = mx.full((c.num_attention_heads,), -mx.inf)
        mx.eval(self.parameters(), self._no_sinks)
        mx.clear_cache()

    # ---- weights ---------------------------------------------------------------------------------------------
    def load_layer(self, w: dict, i: int) -> dict:
        c, dt = self.cfg, self.dtype
        f32 = mx.float32
        cast = lambda a: a.astype(dt)
        H = c.hidden_size
        lin = lambda name, din: Lin(w, name, din, dt)
        bias = lambda name: cast(w[f"{name}.bias"]) if f"{name}.bias" in w else None
        p = f"model.layers.{i}"
        a = f"{p}.self_attn"
        L = dict(ln1=w[f"{p}.input_layernorm.weight"].astype(f32), ln2=w[f"{p}.post_attention_layernorm.weight"].astype(f32),
                 sliding=bool(c.sliding[i]) if c.sliding else False)
        if c.attn == "mla":
            nq, nope, rope, r, vd = (c.num_attention_heads, c.qk_nope_head_dim, c.qk_rope_head_dim, c.kv_lora_rank,
                                     c.v_head_dim)
            if c.q_lora_rank:
                L.update(q_a=lin(f"{a}.q_a_proj", H), q_a_ln=w[f"{a}.q_a_layernorm.weight"].astype(f32),
                         q_b=lin(f"{a}.q_b_proj", c.q_lora_rank))
            else:
                L.update(q=lin(f"{a}.q_proj", H))
            kvb = Lin(w, f"{a}.kv_b_proj", r, dt)
            L.update(kv_a=lin(f"{a}.kv_a_proj_with_mqa", H), kv_a_ln=w[f"{a}.kv_a_layernorm.weight"].astype(f32),
                     o=lin(f"{a}.o_proj", nq * vd))
            if kvb.q is not None:  # quantized kv_b stays quantized: per-head gather_qmm (4x fewer bytes than f16)
                per = [t.reshape(nq, nope + vd, -1) if t is not None else None for t in kvb.q]
                L.update(uk_q=tuple(t[:, :nope] for t in per), uv_q=tuple(t[:, nope:] for t in per),
                         kvb_bits=kvb.bits, kvb_gs=kvb.gs, heads=mx.arange(nq, dtype=mx.uint32))
            else:
                kvb_w = kvb.w_t.T.astype(dt).reshape(nq, nope + vd, r)
                L.update(w_uk=kvb_w[:, :nope, :], w_uv_t=kvb_w[:, nope:, :].swapaxes(-1, -2))  # [H, nope, r], [H, r, v]
        else:
            hd = c.attn_head_dim
            L.update(q=lin(f"{a}.q_proj", H), kv=Lin.cat([lin(f"{a}.k_proj", H), lin(f"{a}.v_proj", H)]),
                     o=lin(f"{a}.o_proj", c.num_attention_heads * hd), q_b_=bias(f"{a}.q_proj"),
                     o_b_=bias(f"{a}.o_proj"))
            kb, vb = bias(f"{a}.k_proj"), bias(f"{a}.v_proj")
            L["kv_b_"] = mx.concatenate([kb, vb]) if kb is not None else None
            if c.qk_norm != "none":
                L.update(q_norm=w[f"{a}.q_norm.weight"].astype(f32), k_norm=w[f"{a}.k_norm.weight"].astype(f32))
        L = {k: v for k, v in L.items() if v is not None}
        m = f"{p}.mlp"
        if i in c.dense_layers:
            g, u = lin(f"{m}.gate_proj", H), lin(f"{m}.up_proj", H)
            try:
                L.update(mlp_gu=Lin.cat([g, u]))
            except AssertionError:
                L.update(mlp_g=g, mlp_u=u)
            L.update(mlp_d=lin(f"{m}.down_proj", c.intermediate_size))
            return L
        rdt = f32 if (self.precise or dt == f32) else dt
        L["router_wt"] = Lin(w, f"{m}.gate", H, rdt)
        if f"{m}.gate.e_score_correction_bias" in w:
            L["router_bias"] = w[f"{m}.gate.e_score_correction_bias"].astype(f32)
        I = c.moe_intermediate_size
        L["I"] = I
        if c.shared_expert_intermediate_size:
            L.update(sh_g=lin(f"{m}.shared_expert.gate_proj", H), sh_u=lin(f"{m}.shared_expert.up_proj", H),
                     sh_d=lin(f"{m}.shared_expert.down_proj", c.shared_expert_intermediate_size),
                     sh_gate=Lin(w, f"{m}.shared_expert_gate", H, rdt))
        elif c.n_shared_experts:
            L.update(sh_g=lin(f"{m}.shared_experts.gate_proj", H), sh_u=lin(f"{m}.shared_experts.up_proj", H),
                     sh_d=lin(f"{m}.shared_experts.down_proj", I * c.n_shared_experts))
        stack_switch_mlp(w, m, c.num_experts)
        if self.store is not None:
            from .expert_stream import ExpertSlab, layer_slots

            L.update(slab=ExpertSlab(self.store, i, layer_slots(self.expert_slots, i)), slab_fmt=self.store.index["expert_format"])
            return L
        if self.map is not None:
            fmt = self.map.index["expert_format"]
            e = self.map.layer(i)
            if fmt["format"] == "affine":
                L.update(gu_q=(e["gu_w"], e["gu_s"], e["gu_b"]), dn_q=(e["dn_w"], e["dn_s"], e["dn_b"]),
                         q_bits=fmt["bits"], q_gs=fmt["group_size"])
            else:
                L.update(gu_w=e["gu"], dn_w=e["dn"])
            return L
        if f"{m}.switch_mlp.gate_proj.scales" in w:  # mlx-lm quantized experts
            sw = f"{m}.switch_mlp"
            L["gu_q"] = tuple(mx.concatenate([w[f"{sw}.gate_proj.{t}"], w[f"{sw}.up_proj.{t}"]], axis=1)
                              for t in ("weight", "scales", "biases"))
            L["dn_q"] = tuple(w[f"{sw}.down_proj.{t}"] for t in ("weight", "scales", "biases"))
            L["q_bits"] = 32 * w[f"{sw}.gate_proj.weight"].shape[-1] // H
            L["q_gs"] = H // w[f"{sw}.gate_proj.scales"].shape[-1]
        elif f"{m}.switch_mlp.gate_proj.weight" in w:  # mlx-lm unquantized stacked experts [E, I, H]
            sw = f"{m}.switch_mlp"
            gu = mx.concatenate([w[f"{sw}.gate_proj.weight"], w[f"{sw}.up_proj.weight"]], axis=1)
            L.update(gu_w=cast(gu).swapaxes(-1, -2), dn_w=cast(w[f"{sw}.down_proj.weight"]).swapaxes(-1, -2))
        elif f"{m}.experts.gate_up_proj" in w:  # fused transformers layout [E, 2I, H], gate first
            L.update(gu_w=cast(w[f"{m}.experts.gate_up_proj"]).swapaxes(-1, -2),
                     dn_w=cast(w[f"{m}.experts.down_proj"]).swapaxes(-1, -2))
        else:
            E = c.num_experts
            gu = mx.stack([mx.concatenate([w[f"{m}.experts.{e}.gate_proj.weight"],
                                           w[f"{m}.experts.{e}.up_proj.weight"]], axis=0) for e in range(E)])
            dn = mx.stack([w[f"{m}.experts.{e}.down_proj.weight"] for e in range(E)])
            L.update(gu_w=cast(gu).swapaxes(-1, -2), dn_w=cast(dn).swapaxes(-1, -2))  # [E, H, 2I], [E, I, H]
        return L

    # ---- pools -----------------------------------------------------------------------------------------------
    def kv_bytes_per_token(self, dtype=None) -> int:
        c = self.cfg
        per = c.head_dim * c.num_key_value_heads * (1 if c.attn == "mla" else 2)
        return c.num_hidden_layers * per * (dtype or self.dtype).size

    def new_kv_pool(self, num_slots: int) -> list:
        c = self.cfg
        shape = (num_slots, c.num_key_value_heads, c.head_dim)
        pool = []
        for _ in range(c.num_hidden_layers):
            k = mx.zeros(shape, self.dtype)
            pool.append((k, k) if c.attn == "mla" else (k, mx.zeros(shape, self.dtype)))
        mx.eval(pool)
        return pool

    def new_pools(self, num_slots: int, num_state_slots: int = 0):  # recurrent API, unused (recurrent = False)
        return self.new_kv_pool(num_slots)

    def pool_arrays(self, pool) -> list[mx.array]:
        return list({id(a): a for kv in pool for a in kv}.values())

    # ---- layers ----------------------------------------------------------------------------------------------
    def _rope(self, t, offsets):
        """t [T, h, d] with the rotary part in the first rot_dims (or all of it); float math as MLX's rope."""
        T, h, d = t.shape
        out = mx.fast.rope(t[:, :, None, :], self.rot_dims, traditional=self.cfg.rope_traditional, base=None,
                           scale=1.0, offset=offsets, freqs=self.rope_freqs)[:, :, 0, :]
        return out

    def _attend(self, L, q, k, v, meta: StepMeta, pool, li, D_out=None):
        """Write k/v, attend (paged decode kernel for single-token items, SDPA otherwise). q [T, nq, D],
        k/v [T, nkv, D]. Returns [T, nq * D_out] (D_out: leading dims of each head's output, MLA)."""
        c = self.cfg
        T, nq, D = q.shape
        kp, vp = pool[li]
        mla = c.attn == "mla"
        # MLA decode through SDPA over the gathered cache rows: the paged kernel runs one threadgroup per KV head,
        # and MLA has one, so all query heads of a sequence shared one threadgroup (RESULTS 1zd)
        mla_sdpa = mla and MLA_DECODE == "sdpa"
        fused = meta.fused_kv and kp.dtype == k.dtype and not mla_sdpa
        if not fused:
            kp[meta.slots] = k
            if mla:
                vp = kp
            else:
                vp[meta.slots] = v
            pool[li] = (kp, vp)
        win = c.window if L["sliding"] else 0
        outs = []
        for g in meta.groups:
            if g.block_tables is not None and mla_sdpa:
                pos = mx.arange(g.max_kv)
                bs = self.block_size
                slots = mx.take(g.block_tables, pos // bs, axis=1) * bs + pos % bs  # [B, K]
                kk = kp[slots].transpose(0, 2, 1, 3)  # [B, 1, K, D]
                mask = (pos[None, :] < g.kv_lens[:, None])[:, None, None, :]
                o = mx.fast.scaled_dot_product_attention(q[g.rows][:, :, None, :], kk, kk, scale=self.attn_scale,
                                                         mask=mask)[:, :, 0, :]
            elif g.block_tables is not None:
                o = paged_decode_attention(q[g.rows], kp, vp, g.block_tables, g.kv_lens, self._no_sinks,
                                           self.attn_scale, self.block_size, win, g.max_kv,
                                           *((k, v) if fused else ()))
                o = o.reshape(g.B, nq, D)
            else:
                qq = q[g.rows].reshape(g.B, g.L, nq, D).transpose(0, 2, 1, 3)
                kv_idx, mask = (g.kv_swa, g.mask_swa) if L["sliding"] else (g.kv_full, g.mask_full)
                kk = kp[kv_idx].transpose(0, 2, 1, 3)
                vv = kk if mla else vp[kv_idx].transpose(0, 2, 1, 3)
                if g.B == 1 and not L["sliding"]:
                    mask = "causal"
                o = sdpa_prefill(qq, kk, vv, scale=self.attn_scale, mask=mask)
                o = o.transpose(0, 2, 1, 3).reshape(g.B * g.L, nq, D)
            outs.append(o if D_out is None else o[..., :D_out])
        o = outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=0)
        if meta.inv_rows is not None:
            o = o[meta.inv_rows]
        return o

    def _gqa(self, L, x, meta, pool, li):
        c = self.cfg
        T = x.shape[0]
        nq, nkv, hd = c.num_attention_heads, c.num_key_value_heads, c.attn_head_dim
        q = mm(x, L["q"]).astype(x.dtype)  # quantized weights may carry float16 scales
        if "q_b_" in L:
            q = q + L["q_b_"]
        kv = mm(x, L["kv"]).astype(x.dtype)
        if "kv_b_" in L:
            kv = kv + L["kv_b_"]
        k, v = kv[:, : nkv * hd], kv[:, nkv * hd:]
        eps = c.rms_norm_eps
        if c.qk_norm == "full":  # OLMoE: RMSNorm over the whole projection
            q = mx.fast.rms_norm(q.astype(mx.float32), L["q_norm"], eps).astype(x.dtype)
            k = mx.fast.rms_norm(k.astype(mx.float32), L["k_norm"], eps).astype(x.dtype)
        q, k, v = q.reshape(T, nq, hd), k.reshape(T, nkv, hd), v.reshape(T, nkv, hd)
        if c.qk_norm == "head":
            q = mx.fast.rms_norm(q.astype(mx.float32), L["q_norm"], eps).astype(x.dtype)
            k = mx.fast.rms_norm(k.astype(mx.float32), L["k_norm"], eps).astype(x.dtype)
        q, k = self._rope(q, meta.rope_offsets), self._rope(k, meta.rope_offsets)
        if self.rope_af != 1.0:  # yarn attention factor scales cos / sin, i.e. the rotary dims of q and k
            r = self.rot_dims
            q = mx.concatenate([q[..., :r] * self.rope_af, q[..., r:]], axis=-1) if r < hd else q * self.rope_af
            k = mx.concatenate([k[..., :r] * self.rope_af, k[..., r:]], axis=-1) if r < hd else k * self.rope_af
        o = self._attend(L, q, k, v, meta, pool, li).reshape(T, nq * hd)
        o = mm(o, L["o"]).astype(x.dtype)
        return o + L["o_b_"] if "o_b_" in L else o

    def _mla(self, L, x, meta, pool, li):
        """MLA. Cache row per token: [c | k_rope] (one head). Decode rows attend in the absorbed form (q_nope through
        W_uk into latent space, single-head attention over the rows, output through W_uv); prefill items expand K / V
        from the cached latents and attend per head (absorbed attention does ~1.8x the attention math of the
        expanded form at D = r + rope vs qk + v; vLLM / SGLang split it the same way)."""
        c = self.cfg
        T = x.shape[0]
        nq, nope, rope, r, vd = (c.num_attention_heads, c.qk_nope_head_dim, c.qk_rope_head_dim, c.kv_lora_rank,
                                 c.v_head_dim)
        eps = c.rms_norm_eps
        if "q_a" in L:
            qa = mx.fast.rms_norm(mm(x, L["q_a"]).astype(mx.float32), L["q_a_ln"], eps).astype(x.dtype)
            q = mm(qa, L["q_b"]).astype(x.dtype)
        else:
            q = mm(x, L["q"]).astype(x.dtype)
        q = q.reshape(T, nq, nope + rope)
        q_nope, q_pe = q[..., :nope], q[..., nope:]
        ckv = mm(x, L["kv_a"]).astype(x.dtype)
        lat = mx.fast.rms_norm(ckv[:, :r].astype(mx.float32), L["kv_a_ln"], eps).astype(x.dtype)
        k_pe = ckv[:, r:].reshape(T, 1, rope)
        q_pe, k_pe = self._rope(q_pe, meta.rope_offsets), self._rope(k_pe, meta.rope_offsets)
        if self.rope_af != 1.0:  # cos / sin scaled on both sides: fold af^2 into the query's rope part
            q_pe = q_pe * (self.rope_af * self.rope_af)
        q_pe = q_pe.astype(x.dtype)
        row = mx.concatenate([lat, k_pe[:, 0, :].astype(lat.dtype)], axis=-1)[:, None, :]  # [T, 1, r + rope]
        kp, _ = pool[li]
        fused = meta.fused_kv and MLA_DECODE in ("paged", "kernel") and kp.dtype == row.dtype
        if not fused:
            kp[meta.slots] = row
            pool[li] = (kp, kp)
        quant = "uk_q" in L
        if quant:
            qk = dict(group_size=L["kvb_gs"], bits=L["kvb_bits"])
        bs = self.block_size
        outs = []
        for g in meta.groups:
            if g.block_tables is not None:  # single-token items: absorbed
                qn, qp = q_nope[g.rows], q_pe[g.rows]  # [B, H, nope], [B, H, rope]
                if quant:  # qn @ W_uk[h] [nope, r] (quantized along r: transpose=False)
                    q_lat = mx.gather_qmm(qn[:, :, None, :], *L["uk_q"], rhs_indices=L["heads"][None, :],
                                          transpose=False, **qk)[:, :, 0, :].astype(x.dtype)
                else:
                    q_lat = (qn.transpose(1, 0, 2) @ L["w_uk"]).transpose(1, 0, 2)
                qq = mx.concatenate([q_lat, qp], axis=-1)  # [B, H, r + rope]
                if MLA_DECODE == "kernel":
                    o = mla_decode_attention(qq, kp, g.block_tables, g.kv_lens, self.attn_scale, bs, g.max_kv, r,
                                             row[g.rows][:, 0, :] if fused else None)
                elif MLA_DECODE == "sdpa":
                    pos = mx.arange(g.max_kv)
                    slots = mx.take(g.block_tables, pos // bs, axis=1) * bs + pos % bs  # [B, K]
                    kk = kp[slots].transpose(0, 2, 1, 3)  # [B, 1, K, D]
                    mask = (pos[None, :] < g.kv_lens[:, None])[:, None, None, :]
                    o = mx.fast.scaled_dot_product_attention(qq[:, :, None, :], kk, kk, scale=self.attn_scale,
                                                             mask=mask)[:, :, 0, :r]
                else:
                    o = paged_decode_attention(qq, kp, kp, g.block_tables, g.kv_lens, self._no_sinks, self.attn_scale,
                                               bs, 0, g.max_kv, *((row[g.rows], row[g.rows]) if fused else ()))
                    o = o.reshape(g.B, nq, r + rope)[..., :r]
                if quant:  # o_lat @ W_uv[h]^T (W_uv [v, r] quantized along r: transpose=True)
                    o = mx.gather_qmm(o[:, :, None, :], *L["uv_q"], rhs_indices=L["heads"][None, :], transpose=True,
                                      **qk)[:, :, 0, :].astype(x.dtype)
                else:
                    o = (o.transpose(1, 0, 2) @ L["w_uv_t"]).transpose(1, 0, 2)
                outs.append(o)  # [B, H, v]
                continue
            # prefill item(s): expand K / V for the keys from the cached latents
            if quant:
                w_uk = mx.dequantize(*L["uk_q"], **qk).astype(x.dtype)  # [H, nope, r]
                w_uv = mx.dequantize(*L["uv_q"], **qk).astype(x.dtype)  # [H, v, r]
            else:
                w_uk, w_uv = L["w_uk"], L["w_uv_t"].swapaxes(-1, -2)
            rows = kp[g.kv_full]  # [B, K, 1, r + rope]
            cl = rows[:, None, :, 0, :r]  # [B, 1, K, r]
            kn = cl @ w_uk.swapaxes(-1, -2)  # [B, H, K, nope]
            vv = cl @ w_uv.swapaxes(-1, -2)  # [B, H, K, v]
            kr = mx.broadcast_to(rows[:, None, :, 0, r:], (*kn.shape[:3], rope))
            kk = mx.concatenate([kn, kr], axis=-1)
            qq = mx.concatenate([q_nope, q_pe], axis=-1)[g.rows].reshape(g.B, g.L, nq, nope + rope).transpose(0, 2, 1, 3)
            mask = "causal" if g.B == 1 else g.mask_full
            o = sdpa_prefill(qq, kk, vv, scale=self.attn_scale, mask=mask)  # 192 / 128 dims: scores in blocks
            outs.append(o.transpose(0, 2, 1, 3).reshape(g.B * g.L, nq, vd))
        o = outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=0)
        if meta.inv_rows is not None:
            o = o[meta.inv_rows]
        return mm(o.reshape(T, nq * vd), L["o"]).astype(x.dtype)

    def _route(self, L, xr):
        """(expert ids [T, k] int32, weights [T, k] float32) as the family's router."""
        c = self.cfg
        k = c.num_experts_per_tok
        lg = mm(xr, L["router_wt"]).astype(mx.float32)
        if c.router == "sigmoid_group":
            s = mx.sigmoid(lg)
            ch = s + L["router_bias"] if "router_bias" in L else s
            if c.n_group > 1:
                T, E = ch.shape
                g = ch.reshape(T, c.n_group, E // c.n_group)
                top2 = mx.topk(g, 2, axis=-1).sum(axis=-1)  # [T, n_group]
                keep = mx.argpartition(-top2, kth=c.topk_group - 1, axis=-1)[:, : c.topk_group]
                gm = mx.zeros((T, c.n_group), mx.bool_)
                gm = mx.put_along_axis(gm, keep, mx.array(True), axis=-1)
                ch = mx.where(mx.repeat(gm, E // c.n_group, axis=-1), ch, 0.0)
            idx = mx.argpartition(-ch, kth=k - 1, axis=-1)[:, :k]
            wts = mx.take_along_axis(s, idx, axis=-1)
            if c.norm_topk_prob:
                wts = wts / (wts.sum(axis=-1, keepdims=True) + 1e-20)
            return idx, wts * c.routed_scaling_factor
        p = mx.softmax(lg, axis=-1, precise=True)
        if c.router == "softmax_group" and c.topk_method == "group_limited_greedy" and c.n_group > 1:
            T, E = p.shape
            gmax = p.reshape(T, c.n_group, E // c.n_group).max(axis=-1)
            keep = mx.argpartition(-gmax, kth=c.topk_group - 1, axis=-1)[:, : c.topk_group]
            gm = mx.put_along_axis(mx.zeros((T, c.n_group), mx.bool_), keep, mx.array(True), axis=-1)
            p = mx.where(mx.repeat(gm, E // c.n_group, axis=-1), p, 0.0)
        idx = mx.argpartition(-p, kth=k - 1, axis=-1)[:, :k]
        wts = mx.take_along_axis(p, idx, axis=-1)
        if c.norm_topk_prob:
            wts = wts / wts.sum(axis=-1, keepdims=True)
        return idx, wts * c.routed_scaling_factor if c.router == "softmax_group" else wts

    def _experts(self, L, x, idx, wts):
        T, k = idx.shape
        I = L["I"]
        if "gu_q" in L:
            qkw = dict(transpose=True, group_size=L["q_gs"], bits=L["q_bits"])
            mm_gu = lambda a, ix, srt: mx.gather_qmm(a, *L["gu_q"], rhs_indices=ix, sorted_indices=srt, **qkw)
            mm_dn = lambda a, ix, srt: mx.gather_qmm(a, *L["dn_q"], rhs_indices=ix, sorted_indices=srt, **qkw)
        else:
            gu_w, dn_w = L["gu_w"], L["dn_w"]
            if gu_w.dtype != x.dtype:
                gu_w, dn_w = gu_w.astype(x.dtype), dn_w.astype(x.dtype)
            mm_gu = lambda a, ix, srt: mx.gather_mm(a, gu_w, rhs_indices=ix, sorted_indices=srt)
            mm_dn = lambda a, ix, srt: mx.gather_mm(a, dn_w, rhs_indices=ix, sorted_indices=srt)
        idx = idx.astype(mx.uint32)
        if idx.size >= 64:
            flat = idx.flatten()
            order = mx.argsort(flat)
            inv = mx.argsort(order)
            si = flat[order]
            h = mm_gu(x[order // k][:, None, :], si, True)
            h = _swiglu(h[..., :I], h[..., I:])  # one compiled kernel, the same numbers as the two ops
            y = mm_dn(h, si, True)
            if FUSED_COMBINE and idx.size >= 256:  # prefill: un-permute + weight + sum in one kernel (float32)
                from .moe_kernels import combine_sorted

                zb = self.__dict__.setdefault("_zero_bias", {}).get((self.cfg.num_experts, y.shape[-1], y.dtype))
                if zb is None:
                    zb = self._zero_bias[(self.cfg.num_experts, y.shape[-1], y.dtype)] = mx.zeros(
                        (self.cfg.num_experts, y.shape[-1]), y.dtype)
                return combine_sorted(y.reshape(T * k, -1), zb, inv.astype(mx.int32), flat, wts, mx.float32
                                      ).astype(x.dtype)
            yv = y[inv].reshape(T, k, -1)
        else:
            h = mm_gu(x[:, None, None, :], idx, False)
            h = _swiglu(h[..., :I], h[..., I:])
            yv = mm_dn(h, idx, False).squeeze(-2)
        return (yv * wts.astype(yv.dtype)[..., None]).sum(axis=1).astype(x.dtype)

    def _mlp(self, L, x, x32=None):
        if "mlp_gu" in L:
            return _swiglu_fused(x, L["mlp_gu"], L["mlp_d"]).astype(x.dtype)
        if "mlp_g" in L:
            return _swiglu_mm(x, L["mlp_g"], L["mlp_u"], L["mlp_d"]).astype(x.dtype)
        xr = x if x32 is None else x32
        idx, wts = self._route(L, xr)
        if "slab" in L:
            from .expert_stream import run_moe

            nxt = None
            li = L["_index"]
            cb = self.__dict__.get("route_trace")
            if cb is not None:  # data collection (placement.calibrate_slot_profile): forces a host sync
                cb(li, xr, idx)
            if self.prefetch or self.gpu_stream:  # the next MoE layers' routers on this layer's input predict their experts
                nxt = [(N["slab"], mm(xr, N["router_wt"]).astype(mx.float32))
                       for N in self.layers[li + 1: li + 3] if "slab" in N]
            from . import expert_stream as _es

            T, k = idx.shape
            # decode through the C++ Streamer; it needs room for this step's experts plus in-flight prefetches (with
            # 3 slots and top-2, Mixtral-8x7B aborted: 'no free slot'), else the Python path, which reads in rounds
            if self.gpu_stream and T * k < 64 and 4 * T * k <= L["slab"].S:
                gs = self.__dict__.get("_gpu_stream")
                if gs is None:
                    gs = self._gpu_stream = _es.GpuStreaming(
                        self.store, {j: N["slab"] for j, N in enumerate(self.layers) if "slab" in N})
                preds = [mx.argpartition(lg, kth=-k, axis=-1)[..., -k:] for _, lg in (nxt or [])]
                slots = gs.route(li, idx.astype(mx.uint32), preds)
                tok = np.repeat(np.arange(T), k)
                yp = self._slab_pair_fn(L["slab_fmt"])(L["slab"], x, tok, idx.reshape(-1), slots, False)
                y = (yp.reshape(T, k, -1) * wts.astype(yp.dtype)[..., None]).sum(axis=1)
            else:
                if self.__dict__.get("_gpu_stream") is not None:
                    self._gpu_stream.to_py()
                y = run_moe(L["slab"], x, idx.astype(mx.uint32), wts.astype(x.dtype),
                            self._slab_pair_fn(L["slab_fmt"]), nxt or None)
        else:
            y = self._experts(L, x, idx, wts)
        if "sh_g" in L:
            sh = _swiglu_mm(x, L["sh_g"], L["sh_u"], L["sh_d"]).astype(x.dtype)
            if "sh_gate" in L:
                sh = mx.sigmoid(mm(xr, L["sh_gate"]).astype(mx.float32)).astype(x.dtype) * sh
            y = y + sh
        return y.astype(x.dtype)

    def hidden(self, meta: StepMeta, pool, x: mx.array | None = None) -> mx.array:
        dt = self.dtype
        if x is None:
            x = self._embed(meta.token_ids)
        if self.precise:
            x = x.astype(mx.float32)
        attn = self._mla if self.cfg.attn == "mla" else self._gqa
        eps = self.cfg.rms_norm_eps
        for li, L in enumerate(self.layers):
            h = mx.fast.rms_norm(x, L["ln1"].astype(x.dtype), eps).astype(dt)
            x = _residual(x, attn(L, h, meta, pool, li))
            h32 = mx.fast.rms_norm(x, L["ln2"].astype(x.dtype), eps)
            x = _residual(x, self._mlp(L, h32.astype(dt), h32 if self.precise else None))
        return x

    def final_hidden(self, meta: StepMeta, pool) -> mx.array:
        x = self.hidden(meta, pool)[meta.sample_rows]
        return mx.fast.rms_norm(x.astype(mx.float32), self.norm.astype(mx.float32), self.cfg.rms_norm_eps)

    def logits(self, meta: StepMeta, pool) -> mx.array:
        """Full float32 logits of the sampled rows (tests)."""
        return self.full_logits(self.final_hidden(meta, pool))
