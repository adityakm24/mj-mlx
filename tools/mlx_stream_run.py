"""Run a MoE checkpoint (gpt-oss, Qwen3.6, ...) with SSD expert streaming (MilliJoule MLX backend) and report speed,
memory, cache behaviour and GPU energy for one or more expert-slot budgets. Same tool and same metrics for every
model; --slots 0 runs the in-memory engine for comparison.

    .venv/bin/python tools/mlx_stream_run.py --experts <repacked dir> --slots 8,16 --new 64 --out results/x.json

Energy is the IOReport GPU rail (Mac whole-GPU, not comparable to NVML). Greedy decoding, one request.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "150")  # before mlx is imported: RESULTS 1p
import resource
import sys
import time

import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mj_mlx import Engine, load_model  # noqa: E402
from mj_mlx.energy import Meter  # noqa: E402

PROMPT = ("Explain, in a short paragraph, why the sky looks blue during the day and red at sunset, and then list "
          "three everyday examples of light scattering.")


PINNED_DATE = "2026-11-16"  # gpt-oss's chat template stamps today's date; pin it so runs on different days match


def pin_date(text: str) -> str:
    import re

    return re.sub(r"Current date: \d{4}-\d{2}-\d{2}", f"Current date: {PINNED_DATE}", text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="checkpoint dir (default: HF cache of openai/gpt-oss-20b)")
    ap.add_argument("--experts", default=None, help="dir from tools/mlx_repack_experts.py")
    ap.add_argument("--prompt-tokens", type=int, default=0, help="random-token prompt of this length (no tokenizer)")
    ap.add_argument("--no-lean", action="store_true")
    ap.add_argument("--slots", default="16", help="comma list of resident expert slots per layer")
    ap.add_argument("--new", type=int, default=64)
    ap.add_argument("--io-threads", type=int, default=8)
    ap.add_argument("--page-cache", action="store_true", help="let macOS cache expert reads (no F_NOCACHE)")
    ap.add_argument("--prefetch", action=argparse.BooleanOptionalAction, default=True,
                    help="predict the next layers' experts and read them early (default on; --no-prefetch: off)")
    ap.add_argument("--attn-bits", type=int, default=None, help="opt-in lossy: quantize attention weights (gpt-oss)")
    ap.add_argument("--attn-pack", action="store_true", help="lossless 12.25-bit attention weights (gpt-oss, bf16_pack)")
    ap.add_argument("--state-layout", default=None, choices=["per_slot", "pool"],
                    help="Qwen recurrent-state layout (default per_slot; pool = the old whole-pool path, A/B only)")
    ap.add_argument("--pair", default=None, help="id shared by the runs of one A/B round (for paired comparisons)")
    ap.add_argument("--no-spec", action="store_true", help="disable run_moe's speculative all-resident dispatch")
    ap.add_argument("--map-experts", default=None, help="dir from expert_map.repack_part_major: experts memory-mapped")
    ap.add_argument("--wired-mb", type=int, default=None, help="MLX wired limit (default: recommended working set)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.no_spec:
        from mj_mlx import expert_stream

        expert_stream.SPECULATE = False
    path = a.model or glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/*"))[0]
    wired = mx.device_info()["max_recommended_working_set_size"] if a.wired_mb is None else a.wired_mb << 20
    mx.set_wired_limit(wired)
    mx.set_cache_limit(1 << 30)
    tok = None
    if a.prompt_tokens:
        ids = [int(t) for t in np.random.RandomState(0).randint(0, 200000, a.prompt_tokens)]
    else:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(path)
        text = tok.apply_chat_template([{"role": "user", "content": PROMPT}], tokenize=False,
                                       add_generation_prompt=True)
        text = pin_date(text)
        ids = [int(t) for t in tok(text, add_special_tokens=False)["input_ids"]]
    rows = []
    for S in ([-1] if a.map_experts else [int(x) for x in a.slots.split(",")]):
        mx.reset_peak_memory()
        t = time.perf_counter()
        if S == 0:  # in-memory reference engine
            model = load_model(path)
        elif S == -1:  # every expert a zero-copy view of the mapped files
            kw = {"attn_bits": a.attn_bits} if a.attn_bits else {}
            if a.attn_pack:
                kw["attn_pack"] = True
            if "gpt" in json.load(open(os.path.join(path, "config.json"))).get("model_type", ""):
                kw["lean"] = not a.no_lean
            model = load_model(path, map_experts=a.map_experts, **kw)
        else:
            model = load_model(path, stream_experts=a.experts, expert_slots=S, lean=not a.no_lean,
                               io_threads=a.io_threads, io_nocache=not a.page_cache, prefetch=a.prefetch,
                               **({"attn_bits": a.attn_bits} if a.attn_bits else {}),
                               **({"state_layout": a.state_layout} if a.state_layout else {}))
        load_s = time.perf_counter() - t
        resident = sum(x.nbytes for x in model.parameters()) / 1e9
        eng = Engine(model=model, num_blocks=256, max_running=1, max_batch_tokens=2048)
        rid = eng.add_request(ids, a.new, ignore_eos=a.prompt_tokens > 0)
        t0 = time.perf_counter()
        first = None
        out = []
        with Meter() as m:
            while eng.has_work():
                fin = eng.step()
                if first is None and len(eng.last_sampled):
                    first = time.perf_counter() - t0
                    st0 = model.expert_stats() if S > 0 else dict(hits=0, misses=0, bytes_read=0)
                for f in fin:
                    out = list(f[2])
        wall = time.perf_counter() - t0
        st = model.expert_stats() if S > 0 else dict(hits=0, misses=0, bytes_read=0, prefetch_issued=0, prefetch_used=0)
        n = len(out)
        dec_s = wall - first
        dec_hits, dec_miss = st["hits"] - st0["hits"], st["misses"] - st0["misses"]
        row = dict(pair=a.pair, state_layout=getattr(model, "state_layout", None), slots=S, speculate=not a.no_spec, attn_bits=a.attn_bits, attn_pack=a.attn_pack, page_cache=a.page_cache, prefetch=a.prefetch,
                   prefetch_issued=st["prefetch_issued"], prefetch_used=st["prefetch_used"], spec_used=st.get("spec_used"), spec_wasted=st.get("spec_wasted"), spec_pending_only=st.get("spec_pending_only"), prefetch_depth=(__import__("mj_mlx.qwen3_5_moe", fromlist=["x"]).PREFETCH_DEPTH if a.prefetch and type(model).__name__ == "Qwen35Moe" else (1 if a.prefetch else 0)), resident_gb=resident, expert_budget_gb=(S * model.store.index["num_layers"] * model.store.stride / 1e9) if S > 0 else None, map_experts=a.map_experts, wired_mb=wired >> 20, load_s=load_s,
                   prompt_tokens=len(ids), new_tokens=n, ttft_s=first,
                   decode_tok_s=(n - 1) / dec_s if n > 1 else None,
                   decode_hit_rate=dec_hits / (dec_hits + dec_miss) if dec_hits + dec_miss else None,
                   decode_gb_read=(st["bytes_read"] - st0["bytes_read"]) / 1e9,
                   gpu_w=m.window.gpu_j / m.window.seconds, gpu_j_per_token=m.window.gpu_j / n,
                   gpu_active_pct=m.window.gpu_active_pct, mlx_peak_gb=mx.get_peak_memory() / 1e9,
                   tokens=out, text=tok.decode(out) if tok else "")
        rows.append(row)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if k not in ("text", "tokens")}),
              flush=True)
        print("   text:", row["text"][:300].replace("\n", " "), flush=True)
        del eng, model
        mx.clear_cache()
    res = dict(model=path, experts=a.experts, prompt=PROMPT if not a.prompt_tokens else None,
               prompt_date=PINNED_DATE if not a.prompt_tokens else None, pair=a.pair, rows=rows,
               process_peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9,
               note="greedy, batch 1, lean trunk (embed + bf16 lm_head on disk, 4-bit candidate head + fp32 rescore), "
                    "experts streamed from the repacked files with F_NOCACHE; energy = Mac GPU rail")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
