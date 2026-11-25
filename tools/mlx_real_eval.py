"""Real-weight accuracy on openai/gpt-oss-20b, layer-streamed so it runs on a 16 GB Mac.

The 13.8 GB checkpoint never sits in memory at once: each layer is loaded, run for every configuration on that
configuration's own hidden states, and dropped. Embedding and lm_head (bf16) stay resident.

Reference: MilliJoule float32 (bf16/MXFP4 weights, float32 activations, attention, KV and logits). Its math is
checked equal to transformers float32 to ~6e-8 on the architecture-exact tiny model (tests/); transformers
float32 itself would need ~84 GB for the 20B model.

Configurations, all on the same released weights:
- MilliJoule bf16 precise (default engine path; greedy via low-bit candidates + float32 rescore)
- MilliJoule bf16 plain
- mlx-lm bf16 (its own gpt_oss.py layers, loader quantization and RoPE)
- mlx-lm float32 (its layers with set_dtype(float32))

Data: chat conversations written for this test (no third-party text), formatted with the model's own tokenizer
and chat template, teacher-forced; every position after the first is scored.

    .venv/bin/python tools/mlx_real_eval.py --out results/real_gpt_oss_20b.json
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import os
os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "150")  # before mlx is imported: RESULTS 1p
import sys
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mj_mlx.gpt_oss import GptOss, GptOssConfig, build_meta  # noqa: E402

CONVERSATIONS = [
    [("user", "Can you explain why the sky looks blue during the day but red and orange at sunset?"),
     ("assistant", "Sunlight contains all visible colours. Air molecules scatter short wavelengths, like blue, much "
      "more strongly than long ones, like red; this is Rayleigh scattering. During the day, scattered blue light "
      "reaches your eyes from every direction, so the whole sky looks blue. At sunset the light crosses far more "
      "air before it reaches you, so most of the blue is scattered away along the path, and what is left is "
      "dominated by red and orange.")],
    [("user", "Write a Python function that returns the n-th Fibonacci number without recursion."),
     ("assistant", "Here is an iterative version that runs in linear time and constant memory:\n\n```python\n"
      "def fibonacci(n: int) -> int:\n    if n < 0:\n        raise ValueError(\"n must be non-negative\")\n"
      "    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n```\n\nFor example, "
      "`fibonacci(10)` returns 55.")],
    [("user", "A train leaves at 9:40 and arrives at 13:15. The trip covers 287 kilometres. What was its average "
      "speed?"),
     ("assistant", "The trip lasts from 9:40 to 13:15, which is 3 hours and 35 minutes, or 3.5833 hours. Average "
      "speed is distance divided by time: 287 km / 3.5833 h = about 80.1 km/h.")],
    [("user", "Give me three tips for keeping houseplants alive in winter."),
     ("assistant", "1. Water less often. Most plants grow slowly in winter, so let the top few centimetres of soil "
      "dry out before watering again.\n2. Move them closer to the light, but away from cold glass and from "
      "radiators, which dry the leaves.\n3. Raise the humidity by grouping plants together or placing them on a "
      "tray of wet pebbles.")],
    [("user", "What is the difference between a process and a thread?"),
     ("assistant", "A process is a running program with its own address space, file handles and other resources. "
      "A thread is a path of execution inside a process. Threads of the same process share its memory, so they "
      "can exchange data cheaply but need locks or other synchronisation to avoid races. Separate processes are "
      "isolated from each other and usually talk through pipes, sockets or shared memory set up on purpose.")],
    [("user", "Summarise the plot of a short story about a lighthouse keeper who finds a message in a bottle."),
     ("assistant", "An old lighthouse keeper on a remote island finds a bottle washed up after a storm. Inside is "
      "a child's drawing of the lighthouse and a note asking whether anyone lives there. Over the following "
      "months he sends replies with the supply boat, and slowly a friendship forms with a girl in a fishing "
      "town across the bay. When the light is automated and he must leave, she is waiting at the harbour.")],
    [("user", "Translate into French: 'The meeting has been moved to Thursday afternoon because of the strike.'"),
     ("assistant", "La réunion a été déplacée au jeudi après-midi à cause de la grève.")],
    [("user", "Is 391 a prime number? Show your reasoning."),
     ("assistant", "No. The square root of 391 is a little under 20, so it is enough to test primes up to 19. "
      "391 is odd, its digits sum to 13 so it is not divisible by 3, and it does not end in 0 or 5. Dividing by "
      "7 gives 55.86, by 11 gives 35.5, by 13 gives 30.08. Dividing by 17 gives exactly 23, so 391 = 17 x 23.")],
]


def shards(path):
    w = {}
    for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        w.update(mx.load(f))  # lazy: arrays are read when evaluated
    return w


def layer_weights(w, i):
    pre = f"model.layers.{i}."
    return {k: v for k, v in w.items() if k.startswith(pre)}


class Ours:
    def __init__(self, path, cfg, dtype, precise, attn_bits=None, attn_pack=False):
        self.m = GptOss(path, dtype=dtype, cfg=cfg, weights={}, head=False, layers=[], precise=precise,
                        attn_bits=attn_bits, attn_pack=attn_pack)
        self.m.block_size = 16
        self.rowwise = attn_pack

    def run_layer(self, w, i, xs, metas):
        self.m.layers = [self.m.load_layer(w, i)]
        c = self.m.cfg
        out = []
        for x, (meta, T) in zip(xs, metas):
            slots = ((T + 15) // 16) * 16
            shape = (slots, c.num_key_value_heads, c.head_dim)
            pool = [(mx.zeros(shape, self.m.dtype), mx.zeros(shape, self.m.dtype))]
            # hidden() loops over self.m.layers with li = 0 for the single loaded layer
            from mj_mlx import bf16_pack

            bf16_pack.ROWWISE = self.rowwise  # packed attention: every row through the decode kernel
            y = self.m.hidden(meta, pool, x=x)
            bf16_pack.ROWWISE = False
            mx.eval(y)
            out.append(y)
        self.m.layers = []
        return out


class MlxLm:
    def __init__(self, cfg_dict, dtype):
        from mlx_lm.models import gpt_oss as G

        self.G, self.cfg, self.dtype = G, cfg_dict, dtype

    def run_layer(self, w, i, xs):
        from mlx_lm.models.base import create_attention_mask

        G = self.G
        args = G.ModelArgs.from_dict(self.cfg)
        blk = G.TransformerBlock(args)  # mlx-lm's own layer
        lw = G.Model.sanitize(None, layer_weights(w, i))  # mlx-lm's own checkpoint conversion
        lw = {k[len(f"model.layers.{i}."):]: v for k, v in lw.items()}
        nn.quantize(blk, group_size=32, bits=4, mode="mxfp4",
                    class_predicate=lambda p, m: hasattr(m, "to_quantized") and f"{p}.scales" in lw)
        from mlx.utils import tree_flatten

        need = {k for k, _ in tree_flatten(blk.parameters())}
        assert need == set(lw), f"mlx-lm layer {i}: {sorted(need ^ set(lw))[:6]}"
        blk.load_weights(list(lw.items()), strict=True)
        if self.dtype == mx.float32:
            blk.set_dtype(mx.float32)
        sliding = self.cfg["layer_types"][i] == "sliding_attention"
        out = []
        for x in xs:
            h = x[None]
            mask = create_attention_mask(h, None, window_size=self.cfg["sliding_window"] if sliding else None)
            y = blk(h, mask, None)[0]
            mx.eval(y)
            out.append(y)
        del blk
        return out


def logsoftmax(a):
    a = a - a.max(-1, keepdims=True)
    return a - np.log(np.exp(a).sum(-1, keepdims=True))


PINNED_DATE = "2026-11-16"  # gpt-oss's chat template stamps today's date; pin it so runs on different days match


def pin_date(text: str) -> str:
    import re

    return re.sub(r"Current date: \d{4}-\d{2}-\d{2}", f"Current date: {PINNED_DATE}", text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="snapshot dir (default: HF cache of openai/gpt-oss-20b)")
    ap.add_argument("--layers", type=int, default=None, help="only the first N layers (debug)")
    ap.add_argument("--configs", default=None,
                    help="comma list of configs to run besides ref (default: all), e.g. mj_precise,mj_precise_attn8")
    ap.add_argument("--wikitext-chunks", type=int, default=0,
                    help="score WikiText-2 raw test instead: this many 512-token chunks, the second half of each "
                         "(llama-perplexity's convention), with perplexity per config (tools/mlx_accuracy_ppl.py)")
    ap.add_argument("--wikitext-ctx", type=int, default=512,
                    help="chunk length for --wikitext-chunks (the second half of each chunk is scored)")
    ap.add_argument("--seqs-json", default=None,
                    help="teacher-force these token sequences instead: JSON list of {name, ids, reply_start, probe}; "
                         "per config, the top 5 at each probe position and the first position from reply_start where "
                         "the config's greedy pick differs from the sequence's next token")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    path = a.model or glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/*"))[0]
    mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    mx.set_cache_limit(1 << 30)
    cfg_dict = json.load(open(os.path.join(path, "config.json")))
    cfg = GptOssConfig.from_dict(cfg_dict)
    n_layers = a.layers or cfg.num_hidden_layers

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    seqs = []
    score_from = 0  # positions scored per sequence: score_from .. len - 2 (predicting the next token)
    given = json.load(open(a.seqs_json)) if a.seqs_json else None
    if given:
        seqs = [list(map(int, g["ids"])) for g in given]
    elif a.wikitext_chunks:
        sys.path.insert(0, os.path.dirname(__file__))
        from mlx_accuracy_ppl import wikitext_raw

        ids = tok(wikitext_raw(), add_special_tokens=False)["input_ids"]
        C = a.wikitext_ctx
        seqs = [list(ids[c * C:(c + 1) * C]) for c in range(a.wikitext_chunks)]
        score_from = C // 2
    else:
        for conv in CONVERSATIONS:
            text = tok.apply_chat_template([{"role": r, "content": c} for r, c in conv], tokenize=False)
            text = pin_date(text)
            seqs.append([int(t) for t in tok(text, add_special_tokens=False)["input_ids"]])
    print("sequences:", [len(s) for s in seqs], "tokens:", sum(len(s) for s in seqs), flush=True)

    w = shards(path)
    embed = w["model.embed_tokens.weight"]  # bf16 [V, H]
    lm_head = w["lm_head.weight"]
    norm = w["model.norm.weight"]
    mx.eval(embed, lm_head, norm)

    metas = []
    for s in seqs:
        T = len(s)
        nb = (T + 15) // 16
        metas.append((build_meta(np.array(s), np.arange(T), np.array([0]), np.array([T]), np.arange(nb)[None],
                                 np.arange(T), 16, cfg.sliding_window), T))

    ours = {"ref": Ours(path, cfg, mx.float32, False), "mj_precise": Ours(path, cfg, mx.bfloat16, True),
            "mj_plain": Ours(path, cfg, mx.bfloat16, False),
            # opt-in lossy: attention weights 8-bit (as llama.cpp's GGUF stores them, Q8_0)
            "mj_precise_attn8": Ours(path, cfg, mx.bfloat16, True, attn_bits=8),
            # lossless 12.25-bit attention weights; every row through the decode matvec (chunks of 4 rows)
            "mj_precise_pack": Ours(path, cfg, mx.bfloat16, True, attn_pack=True)}
    lms = {"mlxlm_bf16": MlxLm(cfg_dict, mx.bfloat16), "mlxlm_fp32": MlxLm(cfg_dict, mx.float32)}
    if a.configs:
        keep = set(a.configs.split(","))
        ours = {k: v for k, v in ours.items() if k == "ref" or k in keep}
        lms = {k: v for k, v in lms.items() if k in keep}
    xs = {}
    for k in ours:
        dt = mx.float32 if k in ("ref", "mj_precise", "mj_precise_attn8", "mj_precise_pack") else mx.bfloat16
        xs[k] = [embed[mx.array(s)].astype(dt) for s in seqs]
    xs["mlxlm_bf16"] = [embed[mx.array(s)] for s in seqs]
    xs["mlxlm_fp32"] = [embed[mx.array(s)].astype(mx.float32) for s in seqs]

    t0 = time.time()
    for i in range(n_layers):
        lw = layer_weights(w, i)
        mx.eval(list(lw.values()))
        for k, o in ours.items():
            xs[k] = o.run_layer(lw, i, xs[k], metas)
        for k, o in lms.items():
            xs[k] = o.run_layer(w, i, xs[k])
        for k in lw:  # evaluated arrays stay alive while the shard dict holds them: drop this layer's
            del w[k]
        del lw
        gc.collect()
        mx.clear_cache()
        print(f"layer {i + 1}/{n_layers} done ({time.time() - t0:.0f}s)", flush=True)

    # Heads. Reference: float32 math on the bf16 head (exact for these weights), in vocab chunks.
    eps = cfg.rms_norm_eps
    shell = GptOss(path, dtype=mx.bfloat16, cfg=cfg, weights={"model.embed_tokens.weight": embed,
                   "lm_head.weight": lm_head, "model.norm.weight": norm}, layers=[])

    def ref_logits(h):
        x = mx.fast.rms_norm(h.astype(mx.float32), norm.astype(mx.float32), eps)
        return mx.concatenate([x @ lm_head[v:v + 32768].astype(mx.float32).T
                               for v in range(0, lm_head.shape[0], 32768)], axis=-1)

    rows = {k: dict(top1=0, kl=[], n=0, nll=[]) for k in list(ours)[1:] + list(lms)}
    if "mj_precise" in ours:
        rows["mj_precise_greedy_path"] = dict(top1=0, n=0)
    ref_nll = []
    argmax = {k: [[] for _ in seqs] for k in ["ref", *rows]}  # per config, per sequence: greedy pick per position
    probes = {k: [{} for _ in seqs] for k in ["ref", *rows] if k != "mj_precise_greedy_path"}

    def note(k, si, c0, lg):
        argmax[k][si].append(lg.argmax(-1))
        if given:
            for p in given[si].get("probe", []):
                if c0 <= p < c0 + lg.shape[0]:
                    row = lg[p - c0]
                    idx = np.argsort(-row)[:5]
                    probes[k][si][p] = [dict(id=int(i), text=tok.decode([int(i)]), logprob=float(row[i])) for i in idx]

    for si in range(len(seqs)):
        for c0 in range(score_from, len(seqs[si]) - (1 if score_from else 0), 128):
            sl = slice(c0, min(c0 + 128, len(seqs[si]) - 1) if score_from else c0 + 128)
            tgt = np.array(seqs[si][sl.start + 1: sl.stop + 1]) if score_from else None
            ref = ref_logits(xs["ref"][si][sl])
            lr = logsoftmax(np.array(ref, dtype=np.float64))
            ref_top = lr.argmax(-1)
            note("ref", si, c0, lr)
            if score_from:
                ref_nll.append(-lr[np.arange(len(tgt)), tgt])
            for k in rows:
                if k == "mj_precise_greedy_path":
                    x32 = mx.fast.rms_norm(xs["mj_precise"][si][sl].astype(mx.float32), norm.astype(mx.float32),
                                           eps)
                    g = np.array(shell.greedy_from(x32))
                    rows[k]["top1"] += int((g == ref_top).sum())
                    rows[k]["n"] += len(g)
                    argmax[k][si].append(g)
                    continue
                h = xs[k][si][sl]
                if k.startswith("mj"):
                    x32 = mx.fast.rms_norm(h.astype(mx.float32), norm.astype(mx.float32), eps)
                    lg = shell.full_logits(x32)
                elif k == "mlxlm_bf16":  # mlx-lm: bf16 norm, bf16 head, bf16 logits
                    lg = (mx.fast.rms_norm(h, norm, eps) @ lm_head.T).astype(mx.float32)
                else:
                    lg = ref_logits(h)  # fp32 head on mlx-lm's fp32 hidden state
                lg = logsoftmax(np.array(lg, dtype=np.float64))
                if score_from:
                    rows[k]["nll"].append(-lg[np.arange(len(tgt)), tgt])
                note(k, si, c0, lg)
                rows[k]["kl"].append((np.exp(lr) * (lr - lg)).sum(-1))
                rows[k]["top1"] += int((lg.argmax(-1) == ref_top).sum())
                rows[k]["n"] += lg.shape[0]
    res = {}
    for k, r in rows.items():
        d = dict(top1=r["top1"] / r["n"], positions=r["n"])
        if r.get("kl"):
            kl = np.concatenate(r["kl"])
            d.update(mean_kl=float(kl.mean()), p99_kl=float(np.quantile(kl, 0.99)), max_kl=float(kl.max()))
        if r.get("nll"):
            m = np.concatenate(r["nll"])
            d.update(ppl=float(np.exp(m.mean())), ppl_se=float(np.exp(m.mean()) * m.std(ddof=1) / np.sqrt(len(m))))
        res[k] = d
        print(k, d, flush=True)
    if ref_nll:
        m = np.concatenate(ref_nll)
        res["ref"] = dict(ppl=float(np.exp(m.mean())), ppl_se=float(np.exp(m.mean()) * m.std(ddof=1) / np.sqrt(len(m))),
                          positions=len(m))
        print("ref", res["ref"], flush=True)
    per_seq = []
    if given:
        for si, g in enumerate(given):
            d = dict(name=g.get("name"), n=len(seqs[si]), reply_start=g["reply_start"], configs={})
            nxt = np.array(seqs[si][1:])
            for k in argmax:
                am = np.concatenate(argmax[k][si])[:len(nxt)]
                off = np.nonzero(am[g["reply_start"] - 1:] != nxt[g["reply_start"] - 1:])[0]
                first = int(off[0]) + g["reply_start"] if len(off) else None  # index of the first token it would change
                d["configs"][k] = dict(first_diff=first,
                                       first_diff_pick=(tok.decode([int(am[first - 1])]) if first else None),
                                       first_diff_given=(tok.decode([int(seqs[si][first])]) if first else None),
                                       probes={str(p): v for p, v in probes.get(k, [{}] * len(seqs))[si].items()})
            per_seq.append(d)
            print(g.get("name"), {k: (v["first_diff"], [t["text"] for t in next(iter(v["probes"].values()), [])][:3])
                                  for k, v in d["configs"].items()}, flush=True)
    out = dict(model="openai/gpt-oss-20b (released MXFP4 experts, bf16 rest)", snapshot=path, layers=n_layers,
               reference="MilliJoule float32 activations on the released weights", tokens=sum(map(len, seqs)),
               prompt_date=None if a.wikitext_chunks else PINNED_DATE,
               sequences=len(seqs), results=res, per_sequence=per_seq,
               note=(f"WikiText-2 raw test, first {a.wikitext_chunks} chunks of {a.wikitext_ctx} tokens (tokens = tokens fed), "
                     f"the second half of each scored (llama.cpp perplexity convention, as tools/mlx_accuracy_ppl.py)"
                     if a.wikitext_chunks else
                     f"teacher-forced over the given sequences ({a.seqs_json})" if given else
                     "teacher-forced over chat-template-formatted conversations written for this test")
                    + "; mj_precise_greedy_path = the engine's greedy output (low-bit candidates + fp32 rescore)")
    json.dump(out, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
