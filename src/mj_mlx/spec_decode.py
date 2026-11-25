# Copyright 2026 the MilliJoule authors
# SPDX-License-Identifier: Apache-2.0
"""Speculative decoding with the model's own MTP (multi-token prediction) head, for Qwen3.5 / Qwen3.6 on MLX.

Qwen3.5 checkpoints ship one MTP layer (`mtp.*`: fc over [norm(embed(next token)) | norm(hidden)], one full-attention
decoder layer with its own KV cache, a final norm, and the main model's lm_head). mlx-community conversions drop it;
`tools/mlx_fetch_mtp.py` fetches just those tensors. mlx-lm cannot speculate on this model family at all: its
recurrent-state cache is not trimmable (`can_trim_prompt_cache` is False), so a rejected draft cannot be undone.

One iteration, greedy, one request:
  1. verify: the main model runs [last token, d_1 .. d_k] as one k+1 token item. Every DeltaNet layer keeps the state
     after each token (gated_delta keep_steps) and its conv inputs, so the state after any prefix can be restored.
  2. accept the longest prefix of drafts equal to the main model's own argmax; emit those plus the main model's next
     token (m + 1 tokens, 1 <= m + 1 <= k + 1). Restore DeltaNet state / conv state to the last accepted token.
     Attention KV past it is stale but never read (lengths), and the next step overwrites it.
  3. draft: the MTP layer runs (hidden_q, token_{q+1}) for the m + 1 newly verified positions q (filling its KV), the
     last row gives d_1; d_2 .. d_k chain the MTP layer on its own output.
One host sync per iteration (the acceptance test). Emitted tokens are the main model's greedy tokens; the verify
step computes them on k+1 rows at once (prefill-style kernels) instead of one row, so near-ties can round differently
from plain decoding (measured by tools/mlx_spec_decode.py).
"""

from __future__ import annotations

import copy
import os
import time

import mlx.core as mx
import numpy as np

from .gpt_oss import build_meta
from .qwen3_5_moe import Lin, Qwen35Moe, QwenMeta, SlotStates, _residual, _swiglu_fused, mm
from .runner import StepArrays


class MtpHead:
    """The MTP layer of a Qwen3.5 checkpoint (raw transformers names, zero-centered norms)."""

    def __init__(self, model: Qwen35Moe, path: str, bits: int | None = 4, group_size: int = 64, num_slots: int = 0,
                 draft_vocab: np.ndarray | None = None):
        """draft_vocab: token ids the draft head scores (the most frequent ones, tools/mlx_draft_vocab.py); the full
        248k-row head costs 5.9 ms per draft on M4 vs 1.9 ms for the whole MTP layer. None = full head."""
        w = {k[len("mtp."):]: v for k, v in mx.load(path).items() if k.startswith("mtp.")}
        c, dt = model.cfg, model.dtype
        self.model = model
        f32 = mx.float32
        zc = lambda a: 1.0 + a.astype(f32)

        def lin(name, din):
            src = w[f"{name}.weight"].astype(dt)
            if bits:
                q = mx.quantize(src, group_size, bits)
                return Lin({f"{name}.weight": q[0], f"{name}.scales": q[1], f"{name}.biases": q[2]}, name, din, dt)
            return Lin({f"{name}.weight": src}, name, din, dt)

        H = c.hidden_size
        p = "layers.0"
        self.fc = lin("fc", 2 * H)
        self.n_emb, self.n_hid, self.norm = zc(w["pre_fc_norm_embedding.weight"]), zc(w["pre_fc_norm_hidden.weight"]), \
            zc(w["norm.weight"])
        a, m = f"{p}.self_attn", f"{p}.mlp"
        self.L = dict(
            type="full_attention", ln1=zc(w[f"{p}.input_layernorm.weight"]),
            ln2=zc(w[f"{p}.post_attention_layernorm.weight"]),
            q_wt=lin(f"{a}.q_proj", H), kv_wt=Lin.cat([lin(f"{a}.k_proj", H), lin(f"{a}.v_proj", H)]),
            q_norm=zc(w[f"{a}.q_norm.weight"]), k_norm=zc(w[f"{a}.k_norm.weight"]),
            o_wt=lin(f"{a}.o_proj", c.num_attention_heads * c.head_dim),
        )
        if f"{m}.experts.gate_up_proj" in w:  # MoE MTP layer (Qwen3.6-35B-A3B): resident 4-bit experts
            qb, qg = bits or 4, group_size
            gu_q = mx.quantize(w[f"{m}.experts.gate_up_proj"].astype(dt), qg, qb)  # [E, 2I, H], along H
            dn_q = mx.quantize(w[f"{m}.experts.down_proj"].astype(dt), qg, qb)  # [E, H, I], along I
            rdt = mx.float32 if (model.precise or dt == mx.float32) else dt
            self.L.update(
                router_wt=Lin({f"{m}.gate.weight": w[f"{m}.gate.weight"]}, f"{m}.gate", H, rdt),
                sh_g=lin(f"{m}.shared_expert.gate_proj", H), sh_u=lin(f"{m}.shared_expert.up_proj", H),
                sh_d=lin(f"{m}.shared_expert.down_proj", c.shared_expert_intermediate_size),
                sh_gate=Lin({f"{m}.shared_expert_gate.weight": w[f"{m}.shared_expert_gate.weight"]},
                            f"{m}.shared_expert_gate", H, rdt),
                gu_q=gu_q, dn_q=dn_q, q_bits=qb, q_gs=qg)
        else:
            self.L.update(mlp_gu=Lin.cat([lin(f"{m}.gate_proj", H), lin(f"{m}.up_proj", H)]),
                          mlp_d=lin(f"{m}.down_proj", c.intermediate_size))
        self.vocab = None
        if draft_vocab is not None:
            v = np.sort(np.asarray(draft_vocab, np.int64))
            self.vocab = mx.array(v.astype(np.int32))
            hl = getattr(model, "head_lin", None)
            sub = Lin.__new__(Lin)
            if hl is not None and hl.q is not None:
                sub.q, sub.bits, sub.gs, sub.w_t = tuple(None if t is None else t[self.vocab] for t in hl.q), hl.bits, \
                    hl.gs, None
            else:
                src = hl.w_t.T if hl is not None else model.lm_head
                sub.q, sub.w_t = None, src[self.vocab].T
            self.head = sub
        shape = (num_slots, c.num_key_value_heads, c.head_dim)
        self.pools = {"kv": [(mx.zeros(shape, dt), mx.zeros(shape, dt))]}
        mx.eval(self.arrays())

    def arrays(self) -> list[mx.array]:
        out = [self.n_emb, self.n_hid, self.norm] + self.fc.arrays() + list(self.pools["kv"][0])
        if self.vocab is not None:
            out += [self.vocab] + self.head.arrays()
        for v in self.L.values():
            if isinstance(v, mx.array):
                out.append(v)
            elif isinstance(v, Lin):
                out += v.arrays()
            elif isinstance(v, tuple):
                out += [t for t in v if isinstance(t, mx.array)]
        return out

    def _mlp(self, x: mx.array) -> mx.array:
        """MTP layer MLP on the float32 residual x: dense SwiGLU (Qwen3.5) or MoE (Qwen3.6, in-memory experts)."""
        md, L = self.model, self.L
        h = mx.fast.rms_norm(x, L["ln2"], md.cfg.rms_norm_eps)
        if "mlp_gu" in L:
            return _swiglu_fused(h.astype(md.dtype), L["mlp_gu"], L["mlp_d"])
        return md._moe(L, h.astype(md.dtype), h if md.precise else None)

    def forward(self, hid32: mx.array, next_ids: mx.array, start: int, block_table: np.ndarray, block_size: int,
                paged: bool = True):
        """hid32 [R, H] float32 (the main model's normed final hidden at positions start .. start+R-1), next_ids [R]
        (the token at each position + 1) -> (normed MTP hidden [R, H] float32: row r predicts position start+r+2)."""
        md, c = self.model, self.model.cfg
        R = hid32.shape[0]
        pos = (start + np.arange(R)).astype(np.int32)
        attn = build_meta(np.zeros(R, np.uint32), pos, np.array([start], np.int32), np.array([R], np.int32),
                          block_table[None], np.array([R - 1], np.int32), block_size, c.sliding_window, None,
                          paged_decode=paged)
        meta = QwenMeta(attn, None, None, None, None, None, None, None)
        eps = c.rms_norm_eps
        e = mx.fast.rms_norm(md._embed(next_ids).astype(mx.float32), self.n_emb, eps)
        h = mx.fast.rms_norm(hid32, self.n_hid, eps)
        x = mm(mx.concatenate([e, h], axis=-1).astype(md.dtype), self.fc).astype(mx.float32)
        L = self.L
        y = md._full_attention(L, mx.fast.rms_norm(x, L["ln1"], eps).astype(md.dtype), meta, self.pools, 0)
        x = _residual(x, y)
        x = _residual(x, self._mlp(x))
        return mx.fast.rms_norm(x, self.norm, eps)

    def forward_items(self, hid32: mx.array, next_ids: mx.array, starts, lens, block_tables: np.ndarray,
                      block_size: int, paged: bool = True):
        """forward() for several requests at once: items of lens[i] rows starting at position starts[i], each with
        its own block table row (and so its own MTP KV)."""
        md, c = self.model, self.model.cfg
        starts, lens = np.asarray(starts, np.int32), np.asarray(lens, np.int32)
        n, T = len(lens), int(lens.sum())
        cu = np.concatenate([[0], np.cumsum(lens)])
        pos = (np.repeat(starts - cu[:-1], lens) + np.arange(T)).astype(np.int32)
        attn = build_meta(np.zeros(T, np.uint32), pos, starts, lens, block_tables, (cu[1:] - 1).astype(np.int32),
                          block_size, c.sliding_window, None, paged_decode=paged)
        meta = QwenMeta(attn, None, None, None, None, None, None, None)
        eps = c.rms_norm_eps
        e = mx.fast.rms_norm(md._embed(next_ids).astype(mx.float32), self.n_emb, eps)
        h = mx.fast.rms_norm(hid32, self.n_hid, eps)
        x = mm(mx.concatenate([e, h], axis=-1).astype(md.dtype), self.fc).astype(mx.float32)
        L = self.L
        y = md._full_attention(L, mx.fast.rms_norm(x, L["ln1"], eps).astype(md.dtype), meta, self.pools, 0)
        x = _residual(x, y)
        x = _residual(x, self._mlp(x))
        return mx.fast.rms_norm(x, self.norm, eps)

    def draft(self, h32: mx.array) -> mx.array:
        """Greedy draft token ids [R] int32 from normed MTP hidden [R, H]."""
        if self.vocab is None:
            return self.model.greedy_from(h32).astype(mx.int32)
        return self.vocab[mx.argmax(mm(h32.astype(self.model.dtype), self.head), axis=-1)]


class DraftPolicy:
    """Chooses the number of drafts per iteration (0 .. k_max) to maximise expected tokens per unit cost.

    Acceptance is tracked per depth as a conditional probability p_j = P(draft j accepted | drafts < j accepted)
    (exponential moving average, rate `alpha`). A depth that is not tried drifts toward DEPTH_DECAY times the depth
    above it (rate `drift`), so a policy that dropped to fewer drafts tries more again once acceptance recovers; depth 1 does not
    drift, the k = 0 probes measure it.
    Expected tokens with k drafts: E(k) = 1 + sum_{j<=k} prod_{i<=j} p_i (the verify step always yields one token).
    Cost per iteration of k drafts: `cost[k]`, either measured wall time (mode "time": an online moving average of
    the host time between acceptance syncs, seeded by `cost` in seconds, e.g. the calibrated s_per_iter, or by a
    relative guess rescaled at the first measurement) or a fixed table (mode "energy": joules per iteration
    on this Mac, measured by tools/mlx_spec_decode.py --calibrate). k = 0 is plain decoding; it skips the MTP layer,
    whose KV is filled for the skipped positions when drafting resumes, and probes with one draft every `probe`
    iterations so acceptance keeps being observed."""

    def __init__(self, k_max: int, mode: str = "time", cost: list[float] | None = None, alpha: float = 0.15,
                 drift: float = 0.03, probe: int = 8, p0: float = 0.7):
        assert mode in ("time", "energy", "edp")
        self.k_max, self.mode, self.alpha, self.drift, self.probe = k_max, mode, alpha, drift, probe
        self.p = [p0 * self.DEPTH_DECAY ** max(j - 1, 0) for j in range(k_max + 1)]  # p[0] unused
        self.cost = list(cost) if cost else [1.0 + 0.05 * k for k in range(k_max + 1)]
        assert len(self.cost) == k_max + 1
        self.timed: set[int] = set()  # mode "time": draft counts with a measured cost (the rest keep the prior)
        self.rescale = mode == "time" and not cost  # a relative prior is rescaled to seconds at the first measurement
        self.since_draft = 0

    def expected(self, k: int) -> float:
        e, run = 1.0, 1.0
        for j in range(1, k + 1):
            run *= self.p[j]
            e += run
        return e

    def update(self, k: int, m: int, seconds: float | None) -> None:
        """k drafts were verified, m accepted, the iteration took `seconds` of host time."""
        a = self.alpha
        for j in range(1, min(m + 1, k) + 1):  # depths whose outcome was observed: 1..m accepted, m+1 rejected
            self.p[j] += a * ((1.0 if j <= m else 0.0) - self.p[j])
        for j in range(max(k + 1, 2), self.k_max + 1):  # depth 1 is re-measured by the probes instead
            self.p[j] += self.drift * (self.DEPTH_DECAY * self.p[j - 1] - self.p[j])
        if self.mode == "time" and seconds is not None:
            if self.rescale and not self.timed:  # first measurement: rescale the relative prior to seconds
                f = seconds / self.cost[k]
                self.cost = [c * f for c in self.cost]
            if k in self.timed or not self.rescale:  # a calibrated seed is blended, never replaced: one slow
                # sample (a cold kernel) could otherwise make a draft count look too costly ever to be retried
                self.cost[k] += 0.2 * (seconds - self.cost[k])
            else:
                self.cost[k] = seconds
            self.timed.add(k)
        self.since_draft = 0 if k > 0 else self.since_draft + 1

    PROBE_GAIN = 1.1
    # untried deeper depths drift toward this fraction of the depth above (measured on Qwen3.5-9B chat text: per-depth
    # acceptance falls with depth, 0.69 overall at k = 1 vs 0.47 at k = 3), so they are not over-estimated
    DEPTH_DECAY = 0.85

    def zero_run(self) -> int:
        """How many k = 0 iterations follow from now if choose() said 0: until the next probe."""
        return max(1, self.probe - 1 - self.since_draft)

    def choose(self) -> int:
        # time / energy: most tokens per second / per joule. edp: lowest energy x delay per token, i.e. the largest
        # E(k)^2 / (J(k) * T(k)), with cost[k] = J(k) * T(k) per iteration: a k that saves no energy but is faster
        # still wins (where energy alone is indifferent between draft counts).
        score = (lambda e, k: e * e / self.cost[k]) if self.mode == "edp" else (lambda e, k: e / self.cost[k])
        best = max(range(self.k_max + 1), key=lambda k: score(self.expected(k), k))
        if best == 0 and self.since_draft + 1 >= self.probe:
            # probe only if drafting could pay at all: at near-perfect acceptance (0.95 per depth) the best k must
            # beat plain decoding by PROBE_GAIN (e.g. 8 requests in a batch: it cannot, so probes only cost)
            hi = max(score(sum(0.95 ** j for j in range(k + 1)), k) for k in range(1, self.k_max + 1))
            if hi >= self.PROBE_GAIN * score(1.0, 0):
                return 1
            self.since_draft = 0
        return best


class SpecGenerator:
    """Greedy single-request generation with MTP drafts (k per iteration, or chosen per iteration by a DraftPolicy).
    Owns its own KV / state pools."""

    def __init__(self, model: Qwen35Moe, mtp_path: str, k: int = 1, max_len: int = 4096, block_size: int = 16,
                 mtp_bits: int | None = 4, chunk: int = 2048, draft_vocab: np.ndarray | None = None,
                 policy: DraftPolicy | None = None, pipeline: bool | None = None):
        """k: drafts per iteration (fixed; 0 = plain decoding through this loop), or the policy's k_max.
        pipeline: while the GPU runs this verify step, build the next one assuming every draft is accepted.
          "defer": submit it only once the acceptance test passes; on a rejection the state pools are
            restored from a snapshot and the graph is dropped unrun (no GPU time wasted).
          "eager": submit it at once; a rejection wastes its GPU time (measured slower on the streamed 35B).
          "off" / False (default): build the next step after the test (the original loop). Neither mode beat it
            (RESULTS 1v): the graph build is not on the critical path.
        Default: MJ_SPEC_PIPELINE (defer / eager / off)."""
        self.model, self.k, self.bs, self.chunk = model, k, block_size, chunk
        self.policy = policy
        if pipeline is None:
            pipeline = os.environ.get("MJ_SPEC_PIPELINE", "off")
        self.pipeline = {True: "defer", False: "off", "1": "defer", "0": "off"}.get(pipeline, pipeline)
        assert self.pipeline in ("defer", "eager", "off"), pipeline
        if policy is not None:
            self.k = policy.k_max
        self.paged = model.cfg.head_dim % 32 == 0  # the paged decode kernel's requirement (as MlxRunner)
        nb = (max_len + 16 + block_size - 1) // block_size
        self.bt = np.arange(nb, dtype=np.int32)
        model.block_size = block_size
        self.pools = model.new_pools(nb * block_size, 1)
        self.mtp = MtpHead(model, mtp_path, bits=mtp_bits, num_slots=nb * block_size, draft_vocab=draft_vocab)
        self.reset_stats()

    def reset_stats(self):
        K = max(self.k, 1)
        self.stats = dict(iters=0, drafted=0, accepted=0, k_hist=[0] * (K + 1), depth_tried=[0] * K,
                          depth_acc=[0] * K, ahead_queued=0, ahead_used=0)

    def _step(self, ids, start: int, capture: bool, token_ids_dev: mx.array | None = None):
        T = len(ids)
        s = StepArrays(np.asarray(ids, np.uint32), (start + np.arange(T)).astype(np.int32), np.zeros(1, np.uint64),
                       np.array([start], np.int32), np.array([T], np.int32), np.ones(1, bool), self.bt[None],
                       np.arange(T, dtype=np.int32), np.zeros(T, np.uint64), [], state_slots=np.zeros(1, np.int32))
        meta = self.model.prepare(s, self.bs, paged_decode=self.paged)
        meta.attn.sample_rows = mx.arange(T, dtype=mx.int32)  # every row: their hidden feeds the MTP layer too
        if token_ids_dev is not None:
            meta.attn.token_ids = token_ids_dev
        if capture:
            meta.capture = {}
        md = self.model
        x = md.hidden(meta, self.pools)
        x32 = mx.fast.rms_norm(x.astype(mx.float32), md.norm.astype(mx.float32), md.cfg.rms_norm_eps)
        return x32, meta

    def _snapshot(self):
        """Handles to the current state pools. KV / conv pools are written by __setitem__, which rebinds the array
        object in place, so each needs its own handle (copy.copy: a new object on the same graph node, no copy)."""
        P, M = self.pools, self.mtp.pools
        two = lambda t: None if t is None else (copy.copy(t[0]), copy.copy(t[1]))
        ssm = [(x, list(x.slots), x.block, x.block_ids, dict(x.block_pos)) if isinstance(x, SlotStates) else x
               for x in P["ssm"]]
        return ([two(t) for t in P["kv"]], [None if c is None else copy.copy(c) for c in P["conv"]], ssm,
                [two(t) for t in M["kv"]])

    def _restore(self, snap) -> None:
        P, M = self.pools, self.mtp.pools
        kv, conv, ssm, mkv = snap
        P["kv"][:], P["conv"][:], M["kv"][:] = kv, conv, mkv
        for li, x in enumerate(ssm):
            if isinstance(x, tuple):
                o, o.slots, o.block, o.block_ids, o.block_pos = x[0], x[1], x[2], x[3], x[4]
                P["ssm"][li] = o
            else:
                P["ssm"][li] = x

    def _queue_verify(self, last_dev: mx.array, drafts: mx.array, pos: int, kk: int, submit: bool = True) -> dict:
        """Build one verify step of [last token, kk drafts] at pos (all lazy) and, if submit, queue it on the GPU."""
        ids_dev = mx.concatenate([last_dev, drafts])
        x32, meta = self._step(np.zeros(kk + 1, np.uint32), pos, capture=True, token_ids_dev=ids_dev)
        a = self.model.greedy_from(x32).astype(mx.int32)  # [kk+1]: main model's token after each row
        if submit:
            mx.async_eval(a, drafts)
        return dict(kk=kk, pos=pos, x32=x32, meta=meta, a=a, drafts=drafts)

    def _rollback(self, meta: QwenMeta, m: int) -> None:
        """Make each DeltaNet layer's state / conv state the ones after token m of the verify item (0-based)."""
        K1 = self.model.cfg.linear_conv_kernel_dim - 1
        slot = np.zeros(1, np.int64)
        for li, (ext, steps) in meta.capture.items():
            self.pools["ssm"][li].scatter(slot, steps[m:m + 1])
            conv = self.pools["conv"][li]
            conv[mx.array([0])] = ext[m + 1:m + 1 + K1][None].astype(conv.dtype)  # ext = [K-1 old | T new] rows
            self.pools["conv"][li] = conv

    def _drafts(self, hid32, next_ids, start, k):
        """MTP over newly verified positions, then chain to k drafts. Returns lazy int32 [k]."""
        md, out = self.model, []
        h = self.mtp.forward(hid32, next_ids, start, self.bt, self.bs, self.paged)[-1:]
        pos = start + hid32.shape[0]  # position whose hidden the chain feeds next
        for i in range(k):
            d = self.mtp.draft(h)
            out.append(d)
            if i + 1 < k:
                h = self.mtp.forward(h, d, pos, self.bt, self.bs, self.paged)
                pos += 1
        return mx.concatenate(out)

    def _zero_run(self, out, pos, max_new, pend_h, pend_ids, on_tokens, t_prev):
        """Consecutive k = 0 iterations (plain decoding), pipelined like Engine(overlap=True): step i + 1 is queued
        with step i's token still on the GPU, before the host waits for step i. The run length is known ahead (the
        policy cannot change its mind without a draft to observe: it plans zeros until its next probe)."""
        md, st = self.model, self.stats
        n0 = min(self.policy.zero_run() if self.policy is not None else max_new, max_new - len(out))
        tok, queued = mx.array([out[-1]], mx.int32), []
        for i in range(n0 + 1):
            if i < n0:
                x32, _ = self._step(np.zeros(1, np.uint32), pos + i, capture=False, token_ids_dev=tok)
                tok = md.greedy_from(x32).astype(mx.int32)
                mx.async_eval(tok)
                pend_h.append(x32)
                pend_ids.append(tok)
                queued.append(tok)
            if i >= 1:
                out.append(int(queued[i - 1].item()))
                st["iters"] += 1
                st["k_hist"][0] += 1
                if on_tokens:
                    on_tokens(1)
                now = time.perf_counter()
                if self.policy is not None:
                    self.policy.update(0, 0, now - t_prev)
                t_prev = now
        return pos + n0, t_prev

    def _choose(self) -> int:
        return self.policy.choose() if self.policy is not None else self.k

    def generate(self, prompt, max_new: int, on_tokens=None) -> list[int]:
        md = self.model
        n = len(prompt)
        kk = self._choose()
        pend_h, pend_ids, pend_start = [], [], 0  # verified positions the MTP layer has not seen yet (k = 0 steps)
        for a in range(0, n, self.chunk):  # prefill, keeping every row's hidden for the MTP layer
            x32, _ = self._step(prompt[a:a + self.chunk], a, capture=False)
            last = md.greedy_from(x32[-1:]).astype(mx.int32)
            nxt = mx.concatenate([mx.array(np.asarray(prompt[a + 1:a + self.chunk], np.int32)), last]) \
                if a + self.chunk >= n else mx.array(np.asarray(prompt[a + 1:a + self.chunk + 1], np.int32))
            if a + self.chunk < n:
                self.mtp.forward(x32, nxt, a, self.bt, self.bs, self.paged)
                mx.eval(self.pools_arrays())
            elif kk > 0:
                drafts = self._drafts(x32, nxt, a, kk)
            else:
                drafts = None
                pend_h, pend_ids, pend_start = [x32], [nxt], a
            mx.clear_cache()
        mx.eval(last, *([drafts] if drafts is not None else []))
        out = [int(last.item())]
        if on_tokens:
            on_tokens(1)
        pos = n  # position of out[-1] (not yet run through the main model)
        t_prev = time.perf_counter()
        st = self.stats
        ahead = None  # verify step queued ahead of the acceptance test (pipeline)
        while len(out) < max_new:
            chosen = None  # the next k, when the pipeline already asked the policy
            if kk == 0:
                if not pend_h:
                    pend_start = pos
                pos, t_prev = self._zero_run(out, pos, max_new, pend_h, pend_ids, on_tokens, t_prev)
            else:
                cur = ahead if ahead is not None else self._queue_verify(mx.array([out[-1]], mx.int32), drafts,
                                                                        pos, kk)
                ahead, snap = None, None
                if self.pipeline != "off" and len(out) + kk + 1 < max_new:
                    kn = chosen = self._choose()  # the policy decides before seeing this step's acceptance (one behind)
                    if kn > 0:  # assume every draft is accepted: the next step's inputs are all on the GPU
                        defer = self.pipeline == "defer"
                        snap = self._snapshot() if defer else None
                        dn = self._drafts(cur["x32"], cur["a"], pos, kn)
                        ahead = self._queue_verify(cur["a"][kk:], dn, pos + kk + 1, kn, submit=not defer)
                        st["ahead_queued"] += 1
                a, x32, meta = cur["a"], cur["x32"], cur["meta"]
                a_np, d_np = np.array(a), np.array(cur["drafts"])
                m = 0
                while m < kk and d_np[m] == a_np[m]:
                    m += 1
                if ahead is not None and m == kk and snap is not None:
                    snap = None  # drop the old handles first, so the pool writes can reuse their buffers
                    mx.async_eval(ahead["a"], ahead["drafts"])  # submit before any bookkeeping: GPU gap minimal
                st["iters"] += 1
                st["drafted"] += kk
                st["accepted"] += m
                st["k_hist"][kk] += 1
                for j in range(min(m + 1, kk)):
                    st["depth_tried"][j] += 1
                    st["depth_acc"][j] += int(j < m)
                if m < kk:
                    if snap is not None:  # deferred: never submitted, so restoring the pools undoes it entirely
                        self._restore(snap)
                        snap = None
                    ahead = None  # built on rejected drafts (eager: the GPU still runs it; its results are dropped)
                    self._rollback(meta, m)
                new = [int(t) for t in a_np[:m + 1]]
                new = new[:max_new - len(out)]
                out += new
                if on_tokens:
                    on_tokens(len(new))
                now = time.perf_counter()
                if self.policy is not None:
                    self.policy.update(kk, m, now - t_prev)
                t_prev = now
                pos += m + 1
                if ahead is not None:  # its drafts already consumed this step's rows
                    st["ahead_used"] += 1
                    kk = ahead["kk"]
                    if len(out) >= max_new:
                        break
                    continue
                if not pend_h:
                    pend_start = pos - m - 1
                pend_h.append(x32[:m + 1])
                pend_ids.append(a[:m + 1])
            if len(out) >= max_new:
                break
            kk = chosen if chosen is not None else self._choose()
            if kk > 0:
                h = pend_h[0] if len(pend_h) == 1 else mx.concatenate(pend_h)
                ids = pend_ids[0] if len(pend_ids) == 1 else mx.concatenate(pend_ids)
                drafts = self._drafts(h, ids, pend_start, kk)
                pend_h, pend_ids = [], []
                mx.async_eval(drafts)  # the GPU runs the MTP layer while the host builds the next verify step
        return out

    def pools_arrays(self):
        return self.model.pool_arrays(self.pools) + list(self.mtp.pools["kv"][0])


def batch_energy_cost(j_per_iter: list[float], batch: int, draft_share: float = 0.1) -> list[float]:
    """Per-request joules per iteration for k drafts when `batch` requests share each step, from the one-request
    calibration J(k). Model: J(k) = W + A * (k + 1) + D * k, where W (reading the weights) is shared by the batch, A
    is one verified row's arithmetic and D one draft (MTP layer + draft head). A + D = J(1) - J(0); D is taken as
    `draft_share` of it (measured on 9B: 27 of ~275 mJ). An estimate, not a measurement per batch size."""
    ad = j_per_iter[1] - j_per_iter[0]
    D = draft_share * ad
    A = ad - D
    W = max(j_per_iter[0] - A, 0.0)
    return [W / batch + A * (k + 1) + D * k for k in range(len(j_per_iter))]


class SpecBatch:
    """Greedy MTP speculative decoding for several requests decoding together (a fixed batch, no scheduler).

    Every step verifies one item per request: [last token, its k_i drafts], k_i chosen per request (a DraftPolicy
    each, or a fixed k). Rejected drafts are rolled back per request (DeltaNet and conv state from the step's
    capture; attention KV past the accepted length is never read). The MTP layer runs every step for every request
    over its newly verified positions, so its KV never needs a backfill; the draft chain runs to the largest k_i.
    Each request owns state slot i and a fixed range of KV blocks. Requests run until max_new tokens (EOS ignored).
    """

    def __init__(self, model: Qwen35Moe, mtp_path: str, max_batch: int, k: int = 2, max_len: int = 2048,
                 block_size: int = 16, mtp_bits: int | None = 4, chunk: int = 2048,
                 draft_vocab: np.ndarray | None = None, policy=None, cost: list[float] | None = None):
        """policy: None = fixed k; a callable () -> DraftPolicy for one per request. Energy mode: each step the policy's
        cost table is set for the number of active requests, from `cost`: either {batch: [J per request-iteration
        for k = 0..k_max]} measured at those batch sizes (tools/mlx_spec_batch.py --calibrate; the nearest measured
        batch at or below is used), or a one-request list scaled by batch_energy_cost (an estimate)."""
        self.model, self.k, self.bs, self.chunk, self.B = model, k, block_size, chunk, max_batch
        self.new_policy, self.cost = policy, cost
        if policy is not None:
            self.k = policy().k_max
        self.paged = model.cfg.head_dim % 32 == 0
        nb = (max_len + 16 + block_size - 1) // block_size
        self.bt = np.arange(max_batch * nb, dtype=np.int32).reshape(max_batch, nb)
        model.block_size = block_size
        self.pools = model.new_pools(max_batch * nb * block_size, max_batch)
        self.mtp = MtpHead(model, mtp_path, bits=mtp_bits, num_slots=max_batch * nb * block_size,
                           draft_vocab=draft_vocab)
        self.stats = dict(steps=0, iters=0, drafted=0, accepted=0, rows=0)

    def _step(self, items, capture: bool):
        """items: list of (slot, start, host ids or None, device ids [L]) -> (x32 [T, H] normed final hidden, meta)."""
        n = len(items)
        lens = np.array([int(d.shape[0]) for _, _, _, d in items], np.int32)
        starts = np.array([st for _, st, _, _ in items], np.int32)
        slots = np.array([sl for sl, _, _, _ in items], np.int32)
        cu = np.concatenate([[0], np.cumsum(lens)])
        T = int(cu[-1])
        pos = (np.repeat(starts - cu[:-1], lens) + np.arange(T)).astype(np.int32)
        s = StepArrays(np.zeros(T, np.uint32), pos, np.arange(n, dtype=np.uint64), starts, lens, np.ones(n, bool),
                       self.bt[slots], np.arange(T, dtype=np.int32), np.zeros(T, np.uint64), [], state_slots=slots)
        meta = self.model.prepare(s, self.bs, paged_decode=self.paged)
        meta.attn.sample_rows = mx.arange(T, dtype=mx.int32)
        meta.attn.token_ids = items[0][3] if n == 1 else mx.concatenate([d for _, _, _, d in items])
        if capture:
            meta.capture = {}
        md = self.model
        x = md.hidden(meta, self.pools)
        return mx.fast.rms_norm(x.astype(mx.float32), md.norm.astype(mx.float32), md.cfg.rms_norm_eps), meta, cu

    def _rollback(self, meta, cu, slots, ms, lens) -> None:
        """Items whose drafts were not all accepted: state after their token m_i, conv inputs ending there."""
        K1 = self.model.cfg.linear_conv_kernel_dim - 1
        sel = [i for i in range(len(ms)) if ms[i] + 1 < lens[i]]
        if not sel:
            return
        N = len(lens)
        sl = np.array([slots[i] for i in sel], np.int64)
        rows = mx.array(np.array([cu[i] + ms[i] for i in sel], np.int32))
        p = np.array([ms[i] + 1 for i in sel])[:, None] + np.arange(K1)[None, :]  # local [conv_in | qkv] positions
        it = np.array(sel)[:, None]
        cidx = mx.array(np.where(p < K1, it * K1 + p, N * K1 + cu[it] + p - K1).astype(np.int32))
        for li, (ext, steps) in meta.capture.items():
            self.pools["ssm"][li].scatter(sl, steps[rows])
            conv = self.pools["conv"][li]
            conv[mx.array(sl.astype(np.int32))] = ext[cidx].astype(conv.dtype)
            self.pools["conv"][li] = conv

    def _drafts(self, h_rows, id_rows, starts, lens, slots, kmax):
        """MTP over each request's newly verified rows, then chain kmax drafts. Returns lazy int32 [n, kmax]."""
        bt = self.bt[np.asarray(slots)]
        h = self.mtp.forward_items(h_rows, id_rows, starts, lens, bt, self.bs, self.paged)
        last = mx.array((np.cumsum(lens) - 1).astype(np.int32))
        h = h[last]
        pos = np.asarray(starts) + np.asarray(lens)
        out = []
        ones = np.ones(len(lens), np.int32)
        for j in range(kmax):
            d = self.mtp.draft(h)
            out.append(d)
            if j + 1 < kmax:
                h = self.mtp.forward_items(h, d, pos + j, ones, bt, self.bs, self.paged)
        return mx.stack(out, axis=1)

    def generate(self, prompts, max_new: int, on_tokens=None) -> list[list[int]]:
        """on_tokens(n, decoding: bool) after each step (decoding = the batch is past every prefill)."""
        md = self.model
        B = len(prompts)
        assert B <= self.B
        outs, pos, pols = [], [], []
        h_rows, id_rows, starts, lens = [], [], [], []
        for i, ids in enumerate(prompts):  # prefill each request (its own slot), keeping every row for the MTP
            n = len(ids)
            for a in range(0, n, self.chunk):
                part = ids[a:a + self.chunk]
                x32, _, _ = self._step([(i, a, None, mx.array(np.asarray(part, np.int32)))], capture=False)
                if a + self.chunk < n:
                    nxt = mx.array(np.asarray(ids[a + 1:a + self.chunk + 1], np.int32))
                    self.mtp.forward_items(x32, nxt, [a], [len(part)], self.bt[[i]], self.bs, self.paged)
                    mx.eval(self.pools_arrays())
                else:
                    last = md.greedy_from(x32[-1:]).astype(mx.int32)
                    nxt = mx.concatenate([mx.array(np.asarray(ids[a + 1:], np.int32)), last])
                    h_rows.append(x32)
                    id_rows.append(nxt)
                    starts.append(a)
                    lens.append(len(part))
                    mx.eval(last)
                    outs.append([int(last.item())])
                mx.clear_cache()
            pos.append(n)
            pols.append(self.new_policy() if self.new_policy else None)
        if on_tokens:
            on_tokens(B, False)
        active = list(range(B))
        # rows verified but not yet run through the MTP layer, per request: [row hiddens], [next ids], first position
        pend = {i: [[h_rows[i]], [id_rows[i]], starts[i]] for i in range(B)}

        def flush(kmax):
            """MTP over every active request's pending rows; then kmax drafts each (lazy [n, kmax]) or None."""
            hs = [pend[i][0][0] if len(pend[i][0]) == 1 else mx.concatenate(pend[i][0]) for i in active]
            ids_ = [pend[i][1][0] if len(pend[i][1]) == 1 else mx.concatenate(pend[i][1]) for i in active]
            st = [pend[i][2] for i in active]
            ln = [int(h.shape[0]) for h in hs]
            for i in active:
                pend[i] = [[], [], None]
            if kmax > 0:
                return self._drafts(mx.concatenate(hs), mx.concatenate(ids_), st, ln, active, kmax)
            mx.async_eval(self.mtp.forward_items(mx.concatenate(hs), mx.concatenate(ids_), st, ln,
                                                 self.bt[np.asarray(active)], self.bs, self.paged))
            return None

        def add_rows(i, h, ids_, start):
            if pend[i][2] is None:
                pend[i][2] = start
            pend[i][0].append(h)
            pend[i][1].append(ids_)

        ks = [self._k(pols[i], len(active)) for i in active]
        drafts = flush(max(ks)) if max(ks) > 0 else None
        while active:
            if max(ks) == 0:  # nobody drafts: plain decoding, pipelined like Engine(overlap=True), MTP layer skipped
                n0 = min(max_new - len(outs[i]) for i in active)
                if all(p is not None for p in (pols[i] for i in active)):
                    n0 = min(n0, min(pols[i].zero_run() for i in active))
                tok = mx.array([outs[i][-1] for i in active], mx.int32)
                queued = []
                for step in range(n0 + 1):
                    if step < n0:
                        items = [(i, pos[i] + step, None, tok[j:j + 1]) for j, i in enumerate(active)]
                        x32, _, _ = self._step(items, capture=False)
                        tok = md.greedy_from(x32).astype(mx.int32)
                        mx.async_eval(tok)
                        queued.append((x32, tok))
                    if step >= 1:
                        x32p, tp = queued[step - 1]
                        t_np = np.array(tp)
                        for j, i in enumerate(active):
                            outs[i].append(int(t_np[j]))
                            add_rows(i, x32p[j:j + 1], tp[j:j + 1], pos[i] + step - 1)
                            if pols[i] is not None:
                                pols[i].update(0, 0, None)
                        self.stats["steps"] += 1
                        self.stats["iters"] += len(active)
                        self.stats["rows"] += len(active)
                        if on_tokens:
                            on_tokens(len(active), True)
                for i in active:
                    pos[i] += n0
            else:
                items = []
                for j, i in enumerate(active):
                    d = mx.array([outs[i][-1]], mx.int32)
                    if ks[j] > 0:
                        d = mx.concatenate([d, drafts[j, :ks[j]]])
                    items.append((i, pos[i], None, d))
                x32, meta, cu = self._step(items, capture=True)
                a = md.greedy_from(x32).astype(mx.int32)
                mx.eval(a, drafts)
                a_np = np.array(a)
                d_np = np.array(drafts)
                ms, got = [], 0
                for j, i in enumerate(active):
                    m = 0
                    while m < ks[j] and d_np[j, m] == a_np[cu[j] + m]:
                        m += 1
                    ms.append(m)
                    new = [int(t) for t in a_np[cu[j]:cu[j] + m + 1]][:max_new - len(outs[i])]
                    outs[i] += new
                    got += len(new)
                    if pols[i] is not None:
                        pols[i].update(ks[j], m, None)
                    self.stats["drafted"] += ks[j]
                    self.stats["accepted"] += m
                self.stats["steps"] += 1
                self.stats["iters"] += len(active)
                self.stats["rows"] += int(cu[-1])
                self._rollback(meta, cu, active, ms, [k + 1 for k in ks])
                if on_tokens:
                    on_tokens(got, True)
                for j, i in enumerate(active):
                    add_rows(i, x32[int(cu[j]):int(cu[j]) + ms[j] + 1], a[int(cu[j]):int(cu[j]) + ms[j] + 1], pos[i])
                    pos[i] += ms[j] + 1
            active = [i for i in active if len(outs[i]) < max_new]
            if not active:
                break
            ks = [self._k(pols[i], len(active)) for i in active]
            if max(ks) > 0:
                drafts = flush(max(ks))
                mx.async_eval(drafts)
            elif max(len(pend[i][0]) for i in active) >= 64:  # bound the queue during long plain stretches
                flush(0)
        return outs

    def _k(self, pol, batch: int) -> int:
        if pol is None:
            return self.k
        if self.cost is not None and pol.mode in ("energy", "edp"):
            if isinstance(self.cost, dict):
                keys = sorted(int(b) for b in self.cost)
                b = max([k for k in keys if k <= batch] or keys[:1])
                pol.cost = list(self.cost[b] if b in self.cost else self.cost[str(b)])[:pol.k_max + 1]
            else:
                pol.cost = batch_energy_cost(self.cost, batch)
        return pol.choose()

    def pools_arrays(self):
        return self.model.pool_arrays(self.pools) + list(self.mtp.pools["kv"][0])
