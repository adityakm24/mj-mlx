"""Check the float32 reference on REAL gpt-oss-20b weights against transformers float32, one layer at a time.

For a real layer i, builds a 1-layer transformers GptOssModel (eager attention, experts dequantized with
transformers' own convert_moe_packed_tensors) holding the real embedding, layer i (renamed to 0, with layer i's
attention type) and the final norm, and compares its last_hidden_state with MilliJoule float32 and mlx-lm float32
on the same input. Memory: ~6 GB (embedding + one layer's experts in float32 on CPU).

    .venv/bin/python tools/mlx_real_layer_check.py --layers 0 1 --out results/real_layer_check.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import mlx.core as mx
import mlx.nn as nn
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from mj_mlx.gpt_oss import GptOss, GptOssConfig, build_meta  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig, AutoTokenizer, GptOssModel
    from transformers.integrations.mxfp4 import convert_moe_packed_tensors

    from mlx_real_eval import CONVERSATIONS

    path = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/*"))[0]
    tok = AutoTokenizer.from_pretrained(path)
    conv = CONVERSATIONS[0] + CONVERSATIONS[4]
    text = tok.apply_chat_template([{"role": r, "content": c} for r, c in conv], tokenize=False)
    ids = [int(t) for t in tok(text, add_special_tokens=False)["input_ids"]]
    T = len(ids)
    print("tokens", T, flush=True)

    files = sorted(glob.glob(os.path.join(path, "*.safetensors")))

    def tensors(prefixes):
        out = {}
        for f in files:
            with safe_open(f, "pt") as s:
                for k in s.keys():
                    if any(k.startswith(p) for p in prefixes):
                        out[k] = s.get_tensor(k)
        return out

    cfg_dict = json.load(open(os.path.join(path, "config.json")))
    results = {}
    for i in a.layers:
        lt = cfg_dict["layer_types"][i]
        sd = tensors([f"model.layers.{i}.", "model.embed_tokens.", "model.norm."])
        # transformers, float32
        hcfg = AutoConfig.from_pretrained(path)
        if hasattr(hcfg, "quantization_config"):
            del hcfg.quantization_config
        hcfg.num_hidden_layers, hcfg.layer_types = 1, [lt]
        hcfg._attn_implementation = "eager"
        hf = GptOssModel(hcfg).to(torch.float32).eval()
        new = {}
        for k, v in sd.items():
            k2 = k.replace(f"model.layers.{i}.", "layers.0.").replace("model.", "")
            if k.endswith("_blocks"):
                new[k2[: -len("_blocks")]] = convert_moe_packed_tensors(v, sd[k[: -len("_blocks")] + "_scales"],
                                                                        dtype=torch.float32)
            elif not k.endswith("_scales"):
                new[k2] = v.float()
        print(f"layer {i} ({lt}):", hf.load_state_dict(new, strict=True), flush=True)
        with torch.no_grad():
            ref = hf(torch.tensor([ids])).last_hidden_state[0].numpy()
        del hf, new
        # MilliJoule float32
        w = {k: mx.array(v.float().numpy()) if v.dtype == torch.bfloat16 else mx.array(v.numpy()) for k, v in sd.items()}
        cfg = GptOssConfig.from_dict(cfg_dict)
        m = GptOss(path, dtype=mx.float32, cfg=cfg, weights={}, head=False, layers=[])
        m.layers = [m.load_layer(w, i)]
        nb = (T + 15) // 16
        meta = build_meta(np.array(ids), np.arange(T), np.array([0]), np.array([T]), np.arange(nb)[None],
                          np.arange(T), 16, cfg.sliding_window)
        shape = (nb * 16, cfg.num_key_value_heads, cfg.head_dim)
        x0 = w["model.embed_tokens.weight"][mx.array(ids)]
        h = m.hidden(meta, [(mx.zeros(shape), mx.zeros(shape))], x=x0)
        ours = np.array(mx.fast.rms_norm(h, w["model.norm.weight"], cfg.rms_norm_eps))
        # mlx-lm float32 (its layer + its conversion)
        from mlx.utils import tree_flatten
        from mlx_lm.models import gpt_oss as G
        from mlx_lm.models.base import create_attention_mask

        blk = G.TransformerBlock(G.ModelArgs.from_dict(cfg_dict))
        lw = G.Model.sanitize(None, {k: v for k, v in w.items() if k.startswith(f"model.layers.{i}.")})
        lw = {k[len(f"model.layers.{i}."):]: v for k, v in lw.items()}
        nn.quantize(blk, group_size=32, bits=4, mode="mxfp4",
                    class_predicate=lambda p, mm: hasattr(mm, "to_quantized") and f"{p}.scales" in lw)
        assert {k for k, _ in tree_flatten(blk.parameters())} == set(lw)
        blk.load_weights(list(lw.items()), strict=True)
        blk.set_dtype(mx.float32)
        hh = x0[None]
        mask = create_attention_mask(hh, None, window_size=cfg_dict["sliding_window"] if lt == "sliding_attention"
                                     else None)
        y = blk(hh, mask, None)[0]
        theirs = np.array(mx.fast.rms_norm(y, w["model.norm.weight"], cfg.rms_norm_eps))
        scale = float(np.abs(ref).max())
        r = dict(layer=i, type=lt, tokens=T, max_abs_ref=scale,
                 millijoule_fp32_max_abs_diff=float(np.abs(ours - ref).max()),
                 mlx_lm_fp32_max_abs_diff=float(np.abs(theirs - ref).max()),
                 millijoule_fp32_rel_rms=float(np.sqrt(((ours - ref) ** 2).mean() / (ref ** 2).mean())),
                 mlx_lm_fp32_rel_rms=float(np.sqrt(((theirs - ref) ** 2).mean() / (ref ** 2).mean())))
        print(r, flush=True)
        results[f"layer{i}"] = r
        del w, sd, m, blk
        mx.clear_cache()
    json.dump(dict(model="openai/gpt-oss-20b", reference="transformers float32 (eager), 1 real layer + embedding "
                   "+ final norm", results=results), open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
