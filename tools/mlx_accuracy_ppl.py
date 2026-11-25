"""WikiText-2 perplexity and KL divergence, the llama.cpp `llama-perplexity` way, for our engine and mlx-lm.

Text: WikiText-2 raw test (Salesforce/wikitext, wikitext-2-raw-v1), rows joined as in wiki.test.raw; the same file is
written to --raw-out so llama-perplexity scores the identical text. Tokens: the model's HF tokenizer over the whole
text; chunks of --ctx tokens; when the tokenizer adds BOS, a chunk's first token is replaced by BOS (as
llama-perplexity does). Scored: positions ctx/2 .. ctx-2 of each chunk predict the next token (llama.cpp's "second
half" convention). Reports perplexity = exp(mean NLL) with its standard error.

KL divergence: --save-ref DIR stores each chunk's scored log-probabilities (float16) from a reference run (the bf16
checkpoint); --ref DIR then reports, per token over the same tokens, KL(ref || this), the fraction where the top token
matches ("same top"), and the 99th percentile of KL, as llama.cpp's --kl-divergence does.

    .venv/bin/python tools/mlx_accuracy_ppl.py --engine mj --model <bf16 dir> --save-ref results/accuracy/ref_olmoe
    .venv/bin/python tools/mlx_accuracy_ppl.py --engine mj --model <4-bit dir> --ref results/accuracy/ref_olmoe --out x.json
"""

import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import mj_mlx  # noqa: E402,F401  (MLX env defaults before mlx is imported)

import numpy as np  # noqa: E402


RAW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "accuracy", "wiki.test.raw")


def wikitext_raw() -> str:
    """The WikiText-2 raw test text: the committed results/accuracy/wiki.test.raw (written by --raw-out from the
    parquet below, and the file llama-perplexity -f reads), else the Hugging Face parquet (needs pyarrow)."""
    if os.path.exists(RAW):
        with open(RAW, encoding="utf-8") as f:
            return f.read()
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    p = hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet", repo_type="dataset")
    return "".join(pq.read_table(p).column("text").to_pylist())


def log_softmax(a):
    a = a - a.max(-1, keepdims=True)
    return a - np.log(np.exp(a).sum(-1, keepdims=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["mj", "mlx-lm"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--chunks", type=int, default=80)
    ap.add_argument("--placement", default="auto")
    ap.add_argument("--mj-kw", default="{}", help="JSON kwargs for load_model (e.g. '{\"attn_bits\": 8}')")
    ap.add_argument("--save-ref", default=None)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--prefill-chunk", type=int, default=0,
                    help="mj: prefill each chunk in pieces of this many tokens over one KV pool, as the Engine does "
                         "(0: one forward per chunk)")
    ap.add_argument("--raw-out", default=None, help="write the scored text here (for llama-perplexity -f)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import mlx.core as mx
    from transformers import AutoTokenizer

    mx.set_cache_limit(1 << 30)  # as the Engine does; unbounded, a streamed 30B grew to 13 GB and swapped
    path = glob.glob(os.path.expanduser(a.model))[0]
    text = wikitext_raw()
    if a.raw_out:
        open(a.raw_out, "w").write(text)
    tok = AutoTokenizer.from_pretrained(path)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    bos = tok.bos_token_id
    add_bos = bos is not None and tok("a")["input_ids"][:1] == [bos]
    n_chunks = min(a.chunks, len(ids) // a.ctx)
    first = a.ctx // 2
    t0 = time.perf_counter()
    if a.engine == "mj":
        from mj_mlx import load_model
        from mj_mlx.gpt_oss import build_meta

        kw = json.loads(a.mj_kw)
        if a.placement != "none":  # MoeLM families; gpt-oss / Qwen3.5 take their own kwargs (--mj-kw)
            kw["placement"] = a.placement
        model = load_model(path, **kw)
        # wire the weights as the Engine does (without it macOS paged a streamed model's expert slots out)
        mx.set_wired_limit((1000 << 20) if getattr(model, "map", None) is not None
                           else mx.device_info()["max_recommended_working_set_size"])
        pool = model.new_kv_pool(a.ctx + 16) if not getattr(model, "recurrent", False) else None

        def nll_rows(x32, tgt_np):
            """Per-token NLL [n] (float64) for the rows of x32 against targets, 1024 rows at a time (8191 rows x the
            vocabulary of float32 logits is 3-7 GB at once)."""
            lean = getattr(model, "lm_head", 1) is None and getattr(model, "head_rows", None) is not None
            out = []
            for r in range(0, x32.shape[0], 1024):
                xr = x32[r:r + 1024]
                tgt = mx.array(np.asarray(tgt_np[r:r + 1024], np.int32))
                if lean:
                    # lean gpt-oss keeps the bf16 lm_head on disk: vocab blocks of its rows on the GPU (running
                    # log-sum-exp + the target's logit), never the full [tokens, 201k] logits
                    rows = model.head_rows.rows
                    xb = xr.astype(mx.bfloat16)
                    lse, tl = None, mx.zeros(tgt.shape, mx.float32)
                    for b in range(0, rows.shape[0], 32768):
                        lg = (xb @ mx.array(np.ascontiguousarray(rows[b:b + 32768])).view(mx.bfloat16).T).astype(mx.float32)
                        blk = mx.logsumexp(lg, axis=-1)
                        lse = blk if lse is None else mx.logaddexp(lse, blk)
                        inb = (tgt >= b) & (tgt < b + lg.shape[1])
                        tl = mx.where(inb, mx.take_along_axis(lg, mx.clip(tgt - b, 0, lg.shape[1] - 1)[:, None], axis=-1)[:, 0], tl)
                        mx.eval(lse, tl)
                    nl = lse - tl
                else:
                    lg = model.full_logits(xr).astype(mx.float32)
                    nl = mx.logsumexp(lg, axis=-1) - mx.take_along_axis(lg, tgt[:, None], axis=-1)[:, 0]
                mx.eval(nl)
                out.append(np.array(nl, np.float64))
            return np.concatenate(out)

        def pool_arrays():
            return model.pool_arrays(pool) if hasattr(model, "pool_arrays") else [x for kv in pool for x in kv]

        def logits(chunk):
            T = len(chunk)
            if pool is None:
                raise NotImplementedError("recurrent models: use the engine path")
            bt = np.arange((T + 15) // 16)[None]
            P = a.prefill_chunk
            if P and T > P:
                # the Engine's way: prefill in P-token pieces over the same KV pool, scoring rows as they pass
                out = []
                for lo in range(0, T, P):
                    hi = min(T, lo + P)
                    pos = np.arange(max(lo, first), min(hi, T - 1))
                    meta = build_meta(np.array(chunk[lo:hi]), np.arange(lo, hi), np.array([lo]), np.array([hi - lo]),
                                      bt, pos - lo, 16, model.cfg.sliding_window)
                    if len(pos) == 0:
                        mx.eval(model.hidden(meta, pool), *pool_arrays())
                        continue
                    x32 = model.final_hidden(meta, pool)
                    out.append(nll_rows(x32, np.array(chunk)[pos + 1]))
                    mx.eval(*pool_arrays())
                    mx.clear_cache()
                return ("nll", np.concatenate(out))
            meta = build_meta(np.array(chunk), np.arange(T), np.array([0]), np.array([T]), bt,
                              np.arange(first, T - 1), 16, model.cfg.sliding_window)
            x32 = model.final_hidden(meta, pool)
            lean = getattr(model, "lm_head", 1) is None and getattr(model, "head_rows", None) is not None
            if lean or (x32.shape[0] > 2048 and not (a.ref or a.save_ref)):
                return ("nll", nll_rows(x32, np.array(chunk[first + 1:])))
            return np.array(model.full_logits(x32))
    else:
        from pathlib import Path

        from mlx_lm.utils import load_model as lm_load

        lm, _ = lm_load(Path(path))

        def logits(chunk):
            return np.array(lm(mx.array(chunk)[None])[0, first: len(chunk) - 1].astype(mx.float32))
    load_s = time.perf_counter() - t0
    if a.save_ref:
        os.makedirs(a.save_ref, exist_ok=True)
    nll, kl_all, same, n = [], [], 0, 0
    t1 = time.perf_counter()
    for c in range(n_chunks):
        chunk = list(ids[c * a.ctx:(c + 1) * a.ctx])
        if add_bos:
            chunk[0] = bos
        out = logits(chunk)
        tgt = np.array(chunk[first + 1:])
        if isinstance(out, tuple):  # per-token NLL computed blockwise (lean head): perplexity only
            assert not (a.ref or a.save_ref), "KL needs full logits (not available with a lean head)"
            nll.append(out[1])
            n += len(tgt)
            mx.clear_cache()
            if c % 10 == 0:
                print(f"chunk {c + 1}/{n_chunks}: ppl {np.exp(np.concatenate(nll).mean()):.4f}", flush=True)
            continue
        lp = log_softmax(out.astype(np.float64))  # [ctx - 1 - first, V]
        nll.append(-lp[np.arange(len(tgt)), tgt])
        if a.save_ref:
            np.save(os.path.join(a.save_ref, f"chunk_{c:04d}.npy"), lp.astype(np.float16))
        if a.ref:
            ref = np.load(os.path.join(a.ref, f"chunk_{c:04d}.npy")).astype(np.float64)
            kl_all.append((np.exp(ref) * (ref - lp)).sum(-1))
            same += int((ref.argmax(-1) == lp.argmax(-1)).sum())
        n += len(tgt)
        mx.clear_cache()
        if c % 10 == 0:
            m = np.concatenate(nll)
            print(f"chunk {c + 1}/{n_chunks}: ppl {np.exp(m.mean()):.4f}", flush=True)
    m = np.concatenate(nll)
    ppl = float(np.exp(m.mean()))
    ppl_se = float(ppl * m.std(ddof=1) / np.sqrt(len(m)))  # delta method, as llama.cpp
    res = dict(engine=a.engine, model=path, mj_kw=json.loads(a.mj_kw), ctx=a.ctx, chunks=n_chunks, tokens_scored=n,
               prefill_chunk=a.prefill_chunk,
               text_tokens=len(ids), add_bos=add_bos, ppl=ppl, ppl_se=ppl_se, load_s=load_s,
               eval_s=time.perf_counter() - t1, ref=a.ref)
    if a.ref:
        k = np.concatenate(kl_all)
        res.update(kld_mean=float(k.mean()), kld_se=float(k.std(ddof=1) / np.sqrt(len(k))),
                   kld_p99=float(np.percentile(k, 99)), kld_median=float(np.median(k)), same_top=same / n)
    print(json.dumps({k: v for k, v in res.items() if k not in ("model",)}), flush=True)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
