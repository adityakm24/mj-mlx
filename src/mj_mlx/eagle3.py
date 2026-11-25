# Copyright 2026 the MilliJoule authors
# SPDX-License-Identifier: Apache-2.0
"""EAGLE-3 speculative decoding for gpt-oss on MLX (greedy, one request).

An EAGLE-3 head (e.g. nebius/EAGLE3-gpt-oss-20b: LlamaForCausalLM layout, 1 layer) drafts from the target's own
hidden states. As in vLLM's llama_eagle3 + gpt_oss:
- the target's residual stream entering layers (2, L/2, L-3) is concatenated ([T, 3H]) and `fc` maps it to H;
- the one decoder layer attends over [input_layernorm(embed(next token)) | hidden_norm(h)] (2H wide q/k/v input),
  then post_attention_layernorm with the residual (= h), the MLP, and the final norm with the residual; the
  pre-norm sum is the hidden fed to the next draft step (the chain), the normed one goes to the draft lm_head
  (draft_vocab_size rows; target id = draft id + d2t[draft id]);
- rope and attention scale are the target's (same yarn config), with the head's own KV cache.
The token embedding is the target's. With the target in lean mode (embedding on disk) the first draft step embeds
the verified tokens from disk rows; chained drafts use a resident 8-bit copy of the draft vocabulary's rows (drafts
only: the target always embeds exactly).

Verification runs [last token, d1..dk] through the target as one item of k+1 tokens; the longest matching prefix
is accepted. gpt-oss has no recurrent state, so a rejected draft needs no rollback: KV past the accepted length is
never read and is overwritten next step.
"""

from __future__ import annotations

import time

import mlx.core as mx
import numpy as np

from .gpt_oss import GptOss, build_meta
from .paged_attention import paged_decode_attention
from .qwen3_5_moe import Lin, mm


class Eagle3Head:
    def __init__(self, model: GptOss, path: str, bits: int | None = 4, group_size: int = 64, num_slots: int = 0,
                 emb_bits: int = 8, aux_layers=None):
        w = mx.load(path)
        c, dt = model.cfg, model.dtype
        self.model = model
        L = model.cfg.num_hidden_layers
        self.aux_layers = tuple(aux_layers) if aux_layers else (2, L // 2, L - 3)
        self.eps = c.rms_norm_eps

        def lin(name, din):
            src = w[f"{name}.weight"].astype(dt)
            if bits:
                q = mx.quantize(src, group_size, bits)
                return Lin({f"{name}.weight": q[0], f"{name}.scales": q[1], f"{name}.biases": q[2]}, name, din, dt)
            return Lin({f"{name}.weight": src}, name, din, dt)

        H = c.hidden_size
        a, m = "layers.0.self_attn", "layers.0.mlp"
        self.fc = lin("fc", 3 * H)
        nq, D = w[f"{a}.q_proj.weight"].shape[0] // c.head_dim, c.head_dim
        nkv = w[f"{a}.k_proj.weight"].shape[0] // D
        self.nq, self.nkv, self.D = nq, nkv, D
        self.qkv = Lin.cat([lin(f"{a}.q_proj", 2 * H), lin(f"{a}.k_proj", 2 * H), lin(f"{a}.v_proj", 2 * H)])
        self.o = lin(f"{a}.o_proj", nq * D)
        self.gu = Lin.cat([lin(f"{m}.gate_proj", H), lin(f"{m}.up_proj", H)])
        self.down = lin(f"{m}.down_proj", w[f"{m}.down_proj.weight"].shape[1])
        self.inter = w[f"{m}.gate_proj.weight"].shape[0]
        f32 = mx.float32
        self.n_in = w["layers.0.input_layernorm.weight"].astype(f32)
        self.n_hid = w["layers.0.hidden_norm.weight"].astype(f32)
        self.n_post = w["layers.0.post_attention_layernorm.weight"].astype(f32)
        self.n_out = w["norm.weight"].astype(f32)
        self.head = lin("lm_head", H)
        d2t = np.array(w["d2t"]).astype(np.int64)
        self.draft_to_target = mx.array((np.arange(len(d2t)) + d2t).astype(np.int32))
        tids = np.arange(len(d2t)) + d2t
        rows = model.embed[mx.array(tids.astype(np.int32))] if model.embed is not None else model.embed_rows.take(
            tids, dt)
        # draft-vocab embedding rows (drafts only), quantized unless emb_bits is None
        self.emb = mx.quantize(rows.astype(dt), 64, emb_bits) if emb_bits else (rows.astype(dt), None, None)
        self.no_sinks = mx.full((nq,), -mx.inf, dtype=dt)
        shape = (num_slots, nkv, D)
        self.kv = (mx.zeros(shape, dt), mx.zeros(shape, dt))
        mx.eval(self.arrays())
        del w

    def arrays(self):
        out = [self.n_in, self.n_hid, self.n_post, self.n_out, self.draft_to_target, self.no_sinks, *self.kv,
               *[a for a in self.emb if a is not None]]
        for l in (self.fc, self.qkv, self.o, self.gu, self.down, self.head):
            out += l.arrays()
        return out

    def embed_draft(self, d: mx.array) -> mx.array:
        """Embeddings [R, H] of draft-vocabulary ids [R] (from the resident quantized rows)."""
        w, s, b = self.emb
        if s is None:
            return w[d]
        return mx.dequantize(w[d], s[d], b[d], group_size=64, bits=8).astype(self.model.dtype)

    def forward(self, h_in: mx.array, emb: mx.array, starts, lens, block_tables: np.ndarray, block_size: int,
                use_fc: bool):
        """h_in [R, 3H] (target aux states, use_fc) or [R, H] (previous draft step's pre-norm output), emb [R, H]
        embeddings of each row's next token. Items: lens[i] rows from position starts[i]. Returns (normed [R, H]
        for the draft head, pre-norm [R, H] float32 for the next draft step)."""
        md, c = self.model, self.model.cfg
        dt = md.dtype
        starts, lens = np.asarray(starts, np.int32), np.asarray(lens, np.int32)
        T = int(lens.sum())
        cu = np.concatenate([[0], np.cumsum(lens)])
        pos = (np.repeat(starts - cu[:-1], lens) + np.arange(T)).astype(np.int32)
        meta = build_meta(np.zeros(T, np.uint32), pos, starts, lens, block_tables, (cu[1:] - 1).astype(np.int32),
                          block_size, 0, None, paged_decode=True)
        eps = self.eps
        hf = mm(h_in.astype(dt), self.fc).astype(mx.float32) if use_fc else h_in.astype(mx.float32)
        e = mx.fast.rms_norm(emb.astype(mx.float32), self.n_in, eps)
        h = mx.fast.rms_norm(hf, self.n_hid, eps)
        x = mx.concatenate([e, h], axis=-1).astype(dt)
        qkv = mm(x, self.qkv)
        nq, nkv, D = self.nq, self.nkv, self.D
        q = qkv[:, :nq * D].reshape(T, nq, 1, D)
        k = qkv[:, nq * D:(nq + nkv) * D].reshape(T, nkv, 1, D)
        v = qkv[:, (nq + nkv) * D:].reshape(T, nkv, D)
        rope = lambda t: mx.fast.rope(t, D, traditional=False, base=None, scale=1.0, offset=meta.rope_offsets,
                                      freqs=md.rope_freqs).squeeze(2)
        q, k = rope(q), rope(k)
        kp, vp = self.kv
        kp[meta.slots] = k
        vp[meta.slots] = v
        self.kv = (kp, vp)
        outs = []
        for g in meta.groups:
            if g.block_tables is not None:
                o = paged_decode_attention(q[g.rows], kp, vp, g.block_tables, g.kv_lens, self.no_sinks, md.attn_scale,
                                           block_size, 0, g.max_kv)
                outs.append(o.reshape(g.B, nq * D))
                continue
            qq = q[g.rows].reshape(g.B, g.L, nq, D).transpose(0, 2, 1, 3)
            kk = kp[g.kv_full].transpose(0, 2, 1, 3)
            vv = vp[g.kv_full].transpose(0, 2, 1, 3)
            o = mx.fast.scaled_dot_product_attention(qq, kk, vv, scale=md.attn_scale,
                                                     mask="causal" if g.B == 1 else g.mask_full)
            outs.append(o.transpose(0, 2, 1, 3).reshape(g.B * g.L, nq * D))
        o = outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=0)
        if meta.inv_rows is not None:
            o = o[meta.inv_rows]
        res = mm(o, self.o).astype(mx.float32) + hf
        h2 = mx.fast.rms_norm(res, self.n_post, eps).astype(dt)
        gu = mm(h2, self.gu)
        gate, up = gu[:, :self.inter], gu[:, self.inter:]
        mlp = mm((gate * mx.sigmoid(gate.astype(mx.float32)).astype(dt)) * up, self.down)
        pre = mlp.astype(mx.float32) + res
        return mx.fast.rms_norm(pre, self.n_out, eps), pre

    def draft(self, hn: mx.array) -> tuple[mx.array, mx.array]:
        """(draft-vocab ids [R], target ids [R]) of the greedy draft."""
        d = mx.argmax(mm(hn.astype(self.model.dtype), self.head), axis=-1).astype(mx.int32)
        return d, self.draft_to_target[d]


class Eagle3Generator:
    """Greedy single-request generation of a gpt-oss model with an EAGLE-3 head, k drafts per iteration."""

    def __init__(self, model: GptOss, head_path: str, k: int = 3, max_len: int = 4096, block_size: int = 16,
                 chunk: int = 2048, bits: int | None = 4, policy=None, emb_bits: int | None = 8, aux_layers=None):
        """k: drafts per iteration, or the policy's k_max. policy: a spec_decode.DraftPolicy (fresh per request
        is the caller's job), choosing 0..k drafts each iteration (MoE verify cost grows ~4 experts per token)."""
        self.model, self.k, self.bs, self.chunk = model, k, block_size, chunk
        self.policy = policy
        if policy is not None:
            self.k = policy.k_max
        model.block_size = block_size
        nb = (max_len + 16 + block_size - 1) // block_size
        self.bt = np.arange(nb, dtype=np.int32)
        self.pool = model.new_kv_pool(nb * block_size)
        self.head = Eagle3Head(model, head_path, bits=bits, num_slots=nb * block_size, emb_bits=emb_bits,
                               aux_layers=aux_layers)
        self.stats = dict(iters=0, drafted=0, accepted=0)

    def _step(self, ids: np.ndarray, start: int):
        md = self.model
        T = len(ids)
        meta = build_meta(np.asarray(ids, np.uint32), (start + np.arange(T)).astype(np.int32),
                          np.array([start], np.int32), np.array([T], np.int32), self.bt[None],
                          np.arange(T, dtype=np.int32), self.bs, md.cfg.sliding_window, None, True)
        aux = []
        x = md.hidden(meta, self.pool, aux_layers=self.head.aux_layers, aux_out=aux)
        x32 = mx.fast.rms_norm(x.astype(mx.float32), md.norm.astype(mx.float32), md.cfg.rms_norm_eps)
        return x32, mx.concatenate(aux, axis=-1)

    def _embed_target(self, ids: np.ndarray) -> mx.array:
        md = self.model
        if md.embed is not None:
            return md.embed[mx.array(np.asarray(ids, np.int32))]
        return md.embed_rows.take(np.asarray(ids, np.int64), md.dtype)

    SHADOW_EVERY = 8  # with no drafts, the head catches up (and checks one draft) every this many iterations

    def _drafts(self, aux, next_ids: np.ndarray, start: int, k: int):
        """Draft head over newly verified rows (aux [R, 3H], next ids on the host), then a chain of k drafts.
        Returns lazy target ids [k]. With k = 0 the rows are queued and the head only runs every SHADOW_EVERY
        iterations (over all queued rows, filling its KV), with one shadow draft checked against the target's next
        token so a policy keeps measuring acceptance; plain decoding then costs almost nothing extra."""
        hd = self.head
        self._pend.append((aux, np.asarray(next_ids), start))
        if k == 0:
            self._zero += 1
            if self.policy is None or self._zero % self.SHADOW_EVERY:
                return None
        aux = self._pend[0][0] if len(self._pend) == 1 else mx.concatenate([p[0] for p in self._pend])
        nxt = np.concatenate([p[1] for p in self._pend])
        start = self._pend[0][2]
        self._pend = []
        R = aux.shape[0]
        hn, pre = hd.forward(aux, self._embed_target(nxt), [start], [R], self.bt[None], self.bs, use_fc=True)
        if k == 0:
            self._shadow = hd.draft(hn[-1:])[1]
            mx.async_eval(self._shadow, *hd.kv)
            return None
        hn, pre = hn[-1:], pre[-1:]
        out = []
        pos = start + R
        for i in range(k):
            d, t = hd.draft(hn)
            out.append(t)
            if i + 1 < k:
                hn, pre = hd.forward(pre, hd.embed_draft(d), [pos], [1], self.bt[None], self.bs, use_fc=False)
                pos += 1
        return mx.concatenate(out)

    def _choose(self) -> int:
        return self.policy.choose() if self.policy is not None else self.k

    def generate(self, prompt, max_new: int, on_tokens=None) -> list[int]:
        md = self.model
        n = len(prompt)
        self._pend, self._zero, self._shadow = [], 0, None
        drafts = None
        kk = self._choose()
        for a in range(0, n, self.chunk):
            part = np.asarray(prompt[a:a + self.chunk])
            x32, aux = self._step(part, a)
            if a + self.chunk < n:
                self.head.forward(aux, self._embed_target(np.asarray(prompt[a + 1:a + self.chunk + 1])), [a],
                                  [len(part)], self.bt[None], self.bs, use_fc=True)
                mx.eval(self.head.kv)
            else:
                last = int(np.array(md.greedy_from(x32[-1:]))[0])
                drafts = self._drafts(aux, np.asarray(list(prompt[a + 1:]) + [last]), a, kk)
            mx.clear_cache()
        out = [last]
        if on_tokens:
            on_tokens(1)
        pos = n
        t_prev = time.perf_counter()
        while len(out) < max_new:
            d_np = np.array(drafts).astype(np.int64) if kk > 0 else np.zeros(0, np.int64)  # host ids (lean embed)
            ids = np.concatenate([[out[-1]], d_np])
            x32, aux = self._step(ids, pos)
            a_np = np.array(md.greedy_from(x32)).astype(np.int64)
            if kk == 0 and self.policy is not None and getattr(self, "_shadow", None) is not None:
                hit = int(int(np.array(self._shadow)[0]) == int(a_np[0]))
                self.policy.update(1, hit, None)  # depth-1 acceptance from the shadow draft (no time sample)
                self.stats["shadow"] = self.stats.get("shadow", 0) + 1
                self._shadow = None
            m = 0
            while m < kk and d_np[m] == a_np[m]:
                m += 1
            self.stats["iters"] += 1
            self.stats["drafted"] += kk
            self.stats["accepted"] += m
            self.stats.setdefault("k_hist", [0] * (self.k + 1))[kk] += 1
            new = [int(t) for t in a_np[:m + 1]][:max_new - len(out)]
            out += new
            if on_tokens:
                on_tokens(len(new))
            now = time.perf_counter()
            if self.policy is not None:
                self.policy.update(kk, m, now - t_prev)
            t_prev = now
            if len(out) >= max_new:
                break
            kk = self._choose()
            drafts = self._drafts(aux[:m + 1], a_np[:m + 1], pos, kk)
            if drafts is not None:
                mx.async_eval(drafts)
            pos += m + 1
        return out
