import os, sys, json, glob
os.environ["MJ_PRED_STATS"] = "1"
os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "150")
sys.path.insert(0, os.path.expanduser("~/Documents/Projects/MilliJoule/python"))
import mlx.core as mx, numpy as np
from transformers import AutoTokenizer
from mj_mlx import Engine, load_model
M = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/*"))[0]
model = load_model(M, stream_experts=os.path.expanduser("~/.cache/millijoule/qwen36-35b-experts-stream"), expert_slots=96,
                   lean=True, io_threads=8, io_nocache=True, prefetch=True)
tok = AutoTokenizer.from_pretrained(M)
res = {}
for name, prompt in [("chat", "Explain how a refrigerator works, step by step, for a curious ten-year-old."),
                     ("code", "Write a Python function that returns the n-th Fibonacci number using memoization, and explain it.")]:
    r = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, enable_thinking=False,
                                tokenize=True, return_dict=False)
    ids = [int(t) for t in (r["input_ids"] if hasattr(r, "keys") else r)]
    eng = Engine(model=model, num_blocks=256, max_running=1, max_batch_tokens=2048)
    eng.add_request(ids, 48, ignore_eos=True)
    eng.step()  # prefill (counted separately below)
    model.pred_stats = {d: [0, 0] for d in (1, 2, 3)}; model._pred_pending = {}
    while eng.has_work():
        eng.step()
    res[name] = {d: round(h / max(n, 1), 3) for d, (h, n) in model.pred_stats.items()}
    print(name, "decode routing predicted d layers ahead (fraction of routed experts):", res[name], flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1)
