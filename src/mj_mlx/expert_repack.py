# Copyright 2026 the MilliJoule authors
# SPDX-License-Identifier: Apache-2.0
"""Repack the routed experts of a gpt-oss or Qwen3.5 / 3.6 checkpoint into one file per layer (ExpertStore layout).

Model-agnostic output: layer_XX.bin holds expert e at offset e * stride (stride rounded up to 16 KB); each record is
the expert's parts back to back, each part in the exact in-memory layout the engine uses, so a cache miss is one
preadv straight into the slot arrays. index.json lists the parts (name, dtype, shape, bytes) plus the expert format,
so the runtime (expert_stream.ExpertStore / ExpertSlab) needs no model knowledge. Supported sources:
- gpt_oss (MXFP4): gu_w uint32 [2I, H/8] (gate rows then up rows; the checkpoint interleaves them), gu_s uint8
  [2I, H/32] (E8M0), dn_w uint32 [H, I/8], dn_s uint8 [H, I/32]. Biases stay resident.
- qwen3_5_moe with mlx-native quantized experts (switch_mlp.*.weight/scales/biases): gu_w uint32 [2I, H*b/32]
  (gate rows then up rows), gu_s / gu_b [2I, H/gs], dn_w [H, I*b/32], dn_s / dn_b [H, I/gs].
- qwen3_5_moe with bf16 experts (fused experts.gate_up_proj or per-expert keys): gu [H, 2I] and dn [I, H] (already
  transposed for gather_mm).
The general MoE families (moe_lm.py) are repacked by placement.repack. The source checkpoint is not modified. The
per-layer, fixed-stride layout is an idea seen in Flash-MoE and mlx-moe-streaming; no code from them is used.
"""

from __future__ import annotations

import glob
import json
import os

import mlx.core as mx
import numpy as np

ALIGN = 16384
FAMILIES = ("gpt_oss", "qwen3_5_moe", "qwen3_5_moe_text")


def raw(a: mx.array) -> tuple[np.ndarray, str]:
    """MLX array -> (numpy array holding the exact bytes, dtype name)."""
    name = {mx.uint8: "uint8", mx.uint32: "uint32", mx.bfloat16: "bfloat16", mx.float16: "float16",
            mx.float32: "float32"}[a.dtype]
    if a.dtype == mx.bfloat16:
        return np.array(a.view(mx.uint16)), name
    return np.array(a), name


def gpt_oss_parts(w, i, cfg):
    p = f"model.layers.{i}.mlp.experts"
    gub, gus = w[f"{p}.gate_up_proj_blocks"], w[f"{p}.gate_up_proj_scales"]  # [E, 2I, G, 16] u8, [E, 2I, G] u8
    dnb, dns = w[f"{p}.down_proj_blocks"], w[f"{p}.down_proj_scales"]
    gub = mx.concatenate([gub[:, 0::2], gub[:, 1::2]], axis=1)
    gus = mx.concatenate([gus[:, 0::2], gus[:, 1::2]], axis=1)
    as_u32 = lambda b: b.reshape(*b.shape[:2], -1).view(mx.uint32)
    return [("gu_w", as_u32(gub)), ("gu_s", gus), ("dn_w", as_u32(dnb)), ("dn_s", dns)], {"format": "mxfp4"}


def qwen_parts(w, i, cfg):
    t = cfg.get("text_config", cfg)
    p = f"model.layers.{i}.mlp"  # names normalized by qwen3_5_moe._norm_key
    sw = f"{p}.switch_mlp"
    if f"{sw}.gate_proj.scales" in w:
        cat = lambda part: mx.concatenate([w[f"{sw}.gate_proj.{part}"], w[f"{sw}.up_proj.{part}"]], axis=1)
        q = cfg.get("quantization", {})
        bits, gs = q.get("bits", 4), q.get("group_size", 64)
        return ([("gu_w", cat("weight")), ("gu_s", cat("scales")), ("gu_b", cat("biases")),
                 ("dn_w", w[f"{sw}.down_proj.weight"]), ("dn_s", w[f"{sw}.down_proj.scales"]),
                 ("dn_b", w[f"{sw}.down_proj.biases"])],
                {"format": "affine", "bits": bits, "group_size": gs})
    if f"{p}.experts.gate_up_proj" in w:
        gu, dn = w[f"{p}.experts.gate_up_proj"], w[f"{p}.experts.down_proj"]  # [E, 2I, H], [E, H, I]
    else:
        E = t["num_experts"]
        gu = mx.stack([mx.concatenate([w[f"{p}.experts.{e}.gate_proj.weight"], w[f"{p}.experts.{e}.up_proj.weight"]],
                                      axis=0) for e in range(E)])
        dn = mx.stack([w[f"{p}.experts.{e}.down_proj.weight"] for e in range(E)])
    return [("gu", mx.contiguous(gu.swapaxes(-1, -2))), ("dn", mx.contiguous(dn.swapaxes(-1, -2)))], \
        {"format": "dense"}


def repack(path: str, out: str, log=print) -> str:
    """Write `path`'s routed experts to `out` (layer_XX.bin + index.json). Returns `out`."""
    cfg = json.load(open(os.path.join(path, "config.json")))
    mt = cfg.get("model_type")
    fn = {"gpt_oss": gpt_oss_parts, "qwen3_5_moe": qwen_parts, "qwen3_5_moe_text": qwen_parts}[mt]
    t = cfg.get("text_config", cfg)
    L = t["num_hidden_layers"]
    E = t.get("num_local_experts") or t["num_experts"]
    w = {}
    for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        w.update(mx.load(f))  # lazy: only each layer's expert tensors are read
    if fn is qwen_parts:  # transformers (model.language_model.*) or mlx-vlm (language_model.model.*) names
        from .qwen3_5_moe import _norm_key

        w = {nk: v for k, v in w.items() if (nk := _norm_key(k)) is not None}
    os.makedirs(out, exist_ok=True)
    index = None
    for i in range(L):
        parts, fmt = fn(w, i, cfg)
        parts = [(n, *raw(arr)) for n, arr in parts]
        for n, arr, _ in parts:
            assert arr.shape[0] == E, (n, arr.shape)
        rec = sum(arr[0].nbytes for _, arr, _ in parts)
        stride = (rec + ALIGN - 1) // ALIGN * ALIGN
        desc = [dict(name=n, dtype=dt, shape=list(arr.shape[1:]), bytes=int(arr[0].nbytes)) for n, arr, dt in parts]
        if index is None:
            index = dict(model_type=mt, expert_format=fmt, stride=stride, record_bytes=rec, num_layers=L,
                         num_experts=E, align=ALIGN, parts=desc)
        assert stride == index["stride"] and desc == index["parts"], f"layer {i} differs"
        with open(os.path.join(out, f"layer_{i:02d}.bin"), "wb") as fh:
            for e in range(E):
                fh.seek(e * stride)
                for _, arr, _ in parts:
                    fh.write(np.ascontiguousarray(arr[e]).tobytes())
            fh.truncate(E * stride)
        for k in [k for k in w if f".layers.{i}." in k]:
            del w[k]
        log(f"layer {i}: {E} experts x {rec / 1e6:.2f} MB (stride {stride / 1e6:.2f} MB)")
    index["source"] = os.path.realpath(path)
    json.dump(index, open(os.path.join(out, "index.json"), "w"), indent=1)
    return out
