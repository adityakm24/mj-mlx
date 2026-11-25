"""Overlap scheduling in the MLX engine: step N+1 is planned and queued on the GPU before step N's tokens reach the
host (its decode inputs are read from step N's lazy sampled tokens on the device). The outputs must be identical
to the plain loop, under preemption, stop tokens and sampling, for every model family."""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).parent))
from hf_ref import snapshot  # noqa: E402
from hf_ref_qwen import qwen_snapshot  # noqa: E402

from mj_mlx import Engine, GptOss  # noqa: E402

GPT_OSS, QWEN = snapshot(), qwen_snapshot()
HEAD_MAP = tempfile.mkdtemp(prefix="mj_head_map_")  # page-aligned embedding / lm_head copies (lean + map_head)


def prompts(seed, lens, vocab):
    rs = np.random.RandomState(seed)
    return [[int(t) for t in rs.randint(0, vocab, n)] for n in lens]


def run(make_engine, ps, new, stops=None, **sampling):
    eng = make_engine()
    rids = []
    for i, (p, n) in enumerate(zip(ps, new)):
        kw = {k: v for k, v in sampling.items()}
        if sampling:
            kw["seed"] = 100 + i
        rids.append(eng.add_request(p, n, ignore_eos=stops is None,
                                    stop_token_ids=None if stops is None else stops[i], **kw))
    out = {}
    while eng.has_work():
        steps = eng.step()
        assert eng.last_sampled.dtype == np.uint64  # set on every step, overlap or not (benchmarks read it)
        for f in steps:
            out[f[0]] = (f[1], list(f[2]))
        eng.sched.check_invariants()
    return [out[r] for r in rids], eng.sched.stats()


CASES = [
    pytest.param("gpt_oss", dict(), marks=pytest.mark.skipif(GPT_OSS is None, reason="tiny gpt-oss not downloaded"),
                 id="gpt_oss"),
    pytest.param("gpt_oss", dict(lean=True), id="gpt_oss_lean",
                 marks=pytest.mark.skipif(GPT_OSS is None, reason="tiny gpt-oss not downloaded")),
    pytest.param("gpt_oss", dict(lean=True, map_head=HEAD_MAP), id="gpt_oss_lean_mapped",
                 marks=pytest.mark.skipif(GPT_OSS is None, reason="tiny gpt-oss not downloaded")),
    pytest.param("qwen", dict(), marks=pytest.mark.skipif(QWEN is None, reason="tiny Qwen3.6 not downloaded"),
                 id="qwen3_6"),
]


def engine_factory(family, model_kw, num_blocks, overlap):
    if family == "gpt_oss":
        def make():
            m = GptOss(GPT_OSS, dtype=mx.bfloat16 if model_kw.get("lean") else mx.float32, **model_kw)
            return Engine(model=m, num_blocks=num_blocks, max_running=6, max_batch_tokens=64, overlap=overlap)
        return make, 200000
    return (lambda: Engine(QWEN, dtype=mx.float32, num_blocks=num_blocks, max_running=6, max_batch_tokens=64,
                           overlap=overlap)), 248000


@pytest.mark.parametrize("family,model_kw", CASES)
@pytest.mark.parametrize("num_blocks", [512, 14], ids=["roomy", "tight"])
def test_overlap_greedy_same_tokens_with_stops(family, model_kw, num_blocks):
    _, vocab = engine_factory(family, model_kw, num_blocks, False)
    ps = prompts(5, [30, 140, 7, 90, 64, 15, 3, 50], vocab)
    new = [12 + 2 * i for i in range(len(ps))]
    plain, _ = run(engine_factory(family, model_kw, num_blocks, False)[0], ps, new)
    # stop half of the requests on a token they really produce mid-sequence
    stops = [[plain[i][1][len(plain[i][1]) // 2]] if i % 2 else [] for i in range(len(ps))]
    ref, ref_stats = run(engine_factory(family, model_kw, num_blocks, False)[0], ps, new, stops)
    got, stats = run(engine_factory(family, model_kw, num_blocks, True)[0], ps, new, stops)
    assert got == ref
    assert any(r == "stop" for r, _ in ref)
    assert stats["overlapped_inputs"] > 0 and ref_stats["overlapped_inputs"] == 0
    assert stats["discarded_samples"] > 0
    if num_blocks == 14:
        assert stats["preemptions"] > 0


@pytest.mark.skipif(QWEN is None, reason="tiny Qwen3.6 not downloaded")
def test_overlap_sampling_same_tokens():
    ps = prompts(9, [30, 140, 7, 90, 64, 15], 248000)
    new = [20] * len(ps)
    kw = dict(temperature=0.9, top_p=0.95, top_k=50)
    ref, _ = run(engine_factory("qwen", {}, 12, False)[0], ps, new, **kw)
    got, stats = run(engine_factory("qwen", {}, 12, True)[0], ps, new, **kw)
    assert got == ref and stats["overlapped_inputs"] > 0


@pytest.mark.skipif(GPT_OSS is None, reason="tiny gpt-oss not downloaded")
def test_lean_mapped_head_same_tokens_as_lean_rows():
    """map_head moves the embedding and rescore row reads to the GPU (zero-copy maps); the tokens must not change."""
    ps = prompts(7, [30, 140, 7, 90], 200000)
    new = [16] * len(ps)
    ref, _ = run(engine_factory("gpt_oss", dict(lean=True), 512, False)[0], ps, new)
    got, stats = run(engine_factory("gpt_oss", dict(lean=True, map_head=HEAD_MAP), 512, True)[0], ps, new)
    assert got == ref and stats["overlapped_inputs"] > 0
    m = GptOss(GPT_OSS, dtype=mx.bfloat16, lean=True, map_head=HEAD_MAP)  # second load reuses the copies
    assert m.embed_rows is None and m.head_rows is None and m.embed is not None and m.lm_head is not None


@pytest.mark.skipif(GPT_OSS is None, reason="tiny gpt-oss not downloaded")
@pytest.mark.parametrize("num_blocks", [512, 14], ids=["roomy", "tight"])
def test_lean_deferred_overlap_same_tokens_as_synced_overlap(num_blocks, monkeypatch):
    """Lean + overlap builds step N+1 on a host embedding buffer and submits it once step N's tokens are known
    (runner.launch_deferred / arm / finish). Same tokens as the older synced path, under preemption and stops."""
    from mj_mlx import runner as R

    ps = prompts(11, [30, 140, 7, 90, 64, 15], 200000)
    new = [10 + 3 * i for i in range(len(ps))]
    make = engine_factory("gpt_oss", dict(lean=True), num_blocks, True)[0]
    assert make().runner.deferred_ok()
    calls = {"finish": 0}
    real = R.MlxRunner.finish

    def counted(self, d):
        calls["finish"] += 1
        return real(self, d)
    monkeypatch.setattr(R.MlxRunner, "finish", counted)
    plain, _ = run(engine_factory("gpt_oss", dict(lean=True), num_blocks, False)[0], ps, new)
    stops = [[plain[i][1][len(plain[i][1]) // 2]] if i % 2 else [] for i in range(len(ps))]
    got, stats = run(make, ps, new, stops)
    assert calls["finish"] > 0
    monkeypatch.setattr(R, "LEAN_DEFER", False)
    ref, _ = run(engine_factory("gpt_oss", dict(lean=True), num_blocks, True)[0], ps, new, stops)
    assert got == ref and stats["overlapped_inputs"] > 0
    if num_blocks == 14:
        assert stats["preemptions"] > 0


def test_host_buffer_is_wrapped_without_copy():
    from mj_mlx.runner import _host_buffer

    buf, _mm = _host_buffer((3, 32))
    x = mx.from_dlpack(buf).view(mx.bfloat16)
    y = x.astype(mx.float32) * 2  # built before the write
    buf[1, 5] = 0x3F80  # bf16 1.0, written after building the graph
    assert float(y[1, 5].item()) == 2.0


@pytest.mark.skipif(GPT_OSS is None, reason="tiny gpt-oss not downloaded")
def test_lean_head_bits_3_same_tokens_as_4(monkeypatch):
    """MJ_LEAN_HEAD_BITS picks the candidate head width (default 3); the choice among the top-k is exact
    (rescore_host), so 3 bits give the same greedy tokens as 4 unless the true argmax leaves the top-k."""
    from mj_mlx import gpt_oss as G

    ps = prompts(5, [30, 140, 7, 90], 200000)
    new = [16] * len(ps)
    monkeypatch.setattr(G, "LEAN_HEAD_BITS", 4)
    ref, _ = run(engine_factory("gpt_oss", dict(lean=True), 512, True)[0], ps, new)
    monkeypatch.setattr(G, "LEAN_HEAD_BITS", 3)
    m = GptOss(GPT_OSS, dtype=mx.bfloat16, lean=True)
    assert [b for _, b in m.head_policy] == [3] and 3 in m.lm_head_qs
    got, _ = run(engine_factory("gpt_oss", dict(lean=True), 512, True)[0], ps, new)
    assert got == ref


def _gate_ext():
    try:
        from mj_mlx.expert_stream import load_gpu_streamer

        return hasattr(load_gpu_streamer(), "HostGate")
    except ImportError:
        return False


@pytest.mark.skipif(GPT_OSS is None or not _gate_ext(), reason="tiny gpt-oss or the C++ extension missing")
@pytest.mark.parametrize("num_blocks", [512, 14], ids=["roomy", "tight"])
def test_lean_gated_steps_same_tokens_as_deferred(num_blocks, monkeypatch):
    """MJ_LEAN_GATE: step N+1 is queued on the GPU behind a host gate before N's token is known; arm() writes the
    embedding row and opens the gate. Same tokens as the ungated deferred path (preemption, stops), every gate
    opened, and the gate really holds the GPU until signalled."""
    from mj_mlx import runner as R

    ps = prompts(11, [30, 140, 7, 90, 64, 15], 200000)
    new = [10 + 3 * i for i in range(len(ps))]
    plain, _ = run(engine_factory("gpt_oss", dict(lean=True), num_blocks, False)[0], ps, new)
    stops = [[plain[i][1][len(plain[i][1]) // 2]] if i % 2 else [] for i in range(len(ps))]
    ref, _ = run(engine_factory("gpt_oss", dict(lean=True), num_blocks, True)[0], ps, new, stops)
    monkeypatch.setattr(R, "LEAN_GATE", True)
    gates = []
    real = R.MlxRunner.arm

    def armed(self, d, prev):
        real(self, d, prev)
        if d.gate_v:
            gates.append((d.gate_v, self._host_gate.value()))
    monkeypatch.setattr(R.MlxRunner, "arm", armed)
    got, stats = run(engine_factory("gpt_oss", dict(lean=True), num_blocks, True)[0], ps, new, stops)
    assert got == ref and stats["overlapped_inputs"] > 0
    assert gates and all(val >= v for v, val in gates)
    if num_blocks == 14:
        assert stats["preemptions"] > 0


@pytest.mark.skipif(not _gate_ext(), reason="the C++ extension missing")
def test_host_gate_holds_gpu_until_signal():
    import threading
    import time

    from mj_mlx.expert_stream import load_gpu_streamer

    g = load_gpu_streamer().HostGate()
    v = g.next()
    x = mx.array([1.0, 2.0])
    y = g.gate(x, v) * 3
    mx.async_eval(y)
    t0 = time.perf_counter()
    threading.Timer(0.2, lambda: g.signal(v)).start()
    mx.eval(y)
    assert time.perf_counter() - t0 >= 0.18 and y.tolist() == [3.0, 6.0]
    g.release_all()
