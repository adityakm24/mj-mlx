"""transformers float32 reference for the Qwen3.5/3.6 MoE tiny checkpoint."""

from __future__ import annotations

import numpy as np

from hf_ref import snapshot

TINY_QWEN36 = "trl-internal-testing/tiny-Qwen3_5MoeForConditionalGeneration-3.6"


def qwen_snapshot():
    return snapshot(TINY_QWEN36)


def load_hf_qwen(path: str):
    import torch
    from transformers import AutoModelForImageTextToText

    return AutoModelForImageTextToText.from_pretrained(path, dtype=torch.float32).eval()


def hf_logits(model, ids: list[int]) -> np.ndarray:
    import torch

    with torch.no_grad():
        return model(input_ids=torch.tensor([ids])).logits[0].numpy()


def hf_greedy(model, prompt: list[int], n: int) -> tuple[list[int], np.ndarray]:
    import torch

    with torch.no_grad():
        out = model.generate(input_ids=torch.tensor([prompt]), max_new_tokens=n, do_sample=False, output_logits=True,
                             return_dict_in_generate=True, eos_token_id=None, pad_token_id=0)
    return out.sequences[0, len(prompt):].tolist(), torch.stack(out.logits)[:, 0].numpy()
