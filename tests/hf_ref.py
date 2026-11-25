"""transformers float32 reference for gpt-oss, without the accelerate-only MXFP4 loader.

Experts are unpacked with transformers' own `convert_moe_packed_tensors`, so the reference dequantization is
transformers' code, not ours."""

from __future__ import annotations

import glob
import os

import numpy as np

TINY_GPT_OSS = "trl-internal-testing/tiny-GptOssForCausalLM"


def snapshot(repo: str = TINY_GPT_OSS) -> str | None:
    hub = os.path.expanduser("~/.cache/huggingface/hub")
    hits = glob.glob(os.path.join(hub, "models--" + repo.replace("/", "--"), "snapshots", "*"))
    return hits[0] if hits else None


def load_hf_gpt_oss(path: str):
    import torch
    from safetensors.torch import load_file
    from transformers import AutoConfig, GptOssForCausalLM
    from transformers.integrations.mxfp4 import convert_moe_packed_tensors

    cfg = AutoConfig.from_pretrained(path)
    if hasattr(cfg, "quantization_config"):
        del cfg.quantization_config
    cfg._attn_implementation = "eager"
    model = GptOssForCausalLM(cfg).to(torch.float32).eval()
    sd = {}
    for f in glob.glob(os.path.join(path, "*.safetensors")):
        sd.update(load_file(f))
    new = {}
    for k, v in sd.items():
        if k.endswith("_blocks"):
            new[k[: -len("_blocks")]] = convert_moe_packed_tensors(v, sd[k[: -len("_blocks")] + "_scales"],
                                                                   dtype=torch.float32)
        elif not k.endswith("_scales"):
            new[k] = v.float()
    model.load_state_dict(new, strict=True)
    return model


def hf_greedy(model, prompt: list[int], n: int) -> tuple[list[int], np.ndarray]:
    """Greedy tokens and the logits of every generated position ([n, V])."""
    import torch

    with torch.no_grad():
        out = model.generate(torch.tensor([prompt]), max_new_tokens=n, do_sample=False, output_logits=True,
                             return_dict_in_generate=True, eos_token_id=None, pad_token_id=0)
    toks = out.sequences[0, len(prompt):].tolist()
    return toks, torch.stack(out.logits)[:, 0].numpy()
