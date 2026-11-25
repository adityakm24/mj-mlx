"""SSD expert streaming (expert_stream.py) on the tiny gpt-oss checkpoint, float32, vs transformers and vs the
in-memory engine. Same tolerance as the batched gate (agreed with the user): token-identical, a divergence excused
only at a true near-tie (reference top-1 vs top-2 gap < 1e-5). Streaming only changes accumulation order."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).parent))
from hf_ref import hf_greedy, load_hf_gpt_oss, snapshot  # noqa: E402

from mj_mlx import Engine, GptOss  # noqa: E402

PATH = snapshot()
ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(PATH is None, reason="tiny gpt-oss checkpoint not downloaded")
NEAR_TIE = 1e-5


@pytest.fixture(scope="module")
def experts_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("tiny_experts")
    subprocess.run([sys.executable, str(ROOT / "tools" / "mlx_repack_experts.py"), "--model", PATH, "--out",
                    str(out)], check=True, capture_output=True)
    return str(out)


@pytest.fixture(scope="module")
def hf():
    return load_hf_gpt_oss(PATH)


def prompts(seed, lens):
    rs = np.random.RandomState(seed)
    return [[int(t) for t in rs.randint(0, 200000, n)] for n in lens]


def check(ours, ref):
    excused = 0
    for i, (o, (r, lg)) in enumerate(zip(ours, ref)):
        div = next((j for j in range(len(r)) if o[j] != r[j]), None)
        if div is not None:
            top2 = np.sort(lg[div])[-2:]
            assert top2[1] - top2[0] < NEAR_TIE, f"seq {i} diverges at {div}"
            excused += 1
    return excused


def test_repacked_records_match_checkpoint(experts_dir):
    from mj_mlx.expert_stream import ExpertStore, StreamingExperts

    m = GptOss(PATH, dtype=mx.float32)
    st = GptOss(PATH, dtype=mx.float32, stream_experts=experts_dir, expert_slots=2)
    e_mem, e_st = m.layers[0]["experts"], st.layers[0]["experts"]
    assert isinstance(e_st, StreamingExperts) and isinstance(e_st.store, ExpertStore)
    e_st.ensure(np.array([3, 1]))
    for e in (3, 1):
        s = e_st.slot_of[e]
        sw = e_st.slab.w
        for a, b in ((sw["gu_w"], e_mem.gu_w), (sw["gu_s"], e_mem.gu_s), (sw["dn_w"], e_mem.dn_w), (sw["dn_s"], e_mem.dn_s)):
            assert np.array_equal(np.array(a[s]), np.array(b[e]))


@pytest.mark.parametrize("slots,lean", [(2, False), (2, True), (4, True)])
def test_streaming_greedy_matches_hf_and_in_memory(hf, experts_dir, slots, lean):
    ps = prompts(21, [5, 40, 129, 200, 3, 77])
    new = [24] * len(ps)
    ref = [hf_greedy(hf, p, n) for p, n in zip(ps, new)]
    mem = Engine(PATH, dtype=mx.float32, num_blocks=256, max_running=4, max_batch_tokens=64).generate(
        ps, 24, ignore_eos=True)
    model = GptOss(PATH, dtype=mx.float32, stream_experts=experts_dir, expert_slots=slots, lean=lean)
    eng = Engine(model=model, num_blocks=256, max_running=4, max_batch_tokens=64)
    ours = eng.generate(ps, 24, ignore_eos=True, check=True)
    st = model.expert_stats()
    print(f"slots {slots} lean {lean}: {st}, excused vs hf {check(ours, ref)}, vs in-memory "
          f"{sum(a != b for a, b in zip(ours, mem))} differing sequences")
    assert st["misses"] > 0
    assert st["spec_used"] + st["spec_wasted"] > 0  # the speculative dispatch ran (kept or discarded)
    check(mem, ref)
    if slots < 4:
        assert st["hit_rate"] < 1.0


# ---- the same streaming core on Qwen3.6 (hybrid DeltaNet + MoE with shared expert) -------------------------------
from hf_ref_qwen import hf_greedy as qwen_hf_greedy  # noqa: E402
from hf_ref_qwen import load_hf_qwen, qwen_snapshot  # noqa: E402

QPATH = qwen_snapshot()


@pytest.fixture(scope="module")
def qwen_experts_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("tiny_qwen_experts")
    subprocess.run([sys.executable, str(ROOT / "tools" / "mlx_repack_experts.py"), "--model", QPATH, "--out",
                    str(out)], check=True, capture_output=True)
    return str(out)


@pytest.mark.skipif(QPATH is None, reason="tiny Qwen3.6 checkpoint not downloaded")
@pytest.mark.parametrize("slots,lean,prefetch", [(2, False, False), (2, True, True), (4, True, False)])
def test_qwen_streaming_greedy_matches_hf_and_in_memory(qwen_experts_dir, slots, lean, prefetch):
    from mj_mlx import Qwen35Moe

    hfq = load_hf_qwen(QPATH)
    rs = np.random.RandomState(31)
    ps = [[int(t) for t in rs.randint(0, 248000, n)] for n in (5, 40, 129, 70)]
    ref = [qwen_hf_greedy(hfq, p, 20) for p in ps]
    mem = Engine(QPATH, dtype=mx.float32, num_blocks=256, max_running=4, max_batch_tokens=64).generate(
        ps, 20, ignore_eos=True)
    model = Qwen35Moe(QPATH, dtype=mx.float32, stream_experts=qwen_experts_dir, expert_slots=slots, lean=lean,
                      prefetch=prefetch)
    ours = Engine(model=model, num_blocks=256, max_running=4, max_batch_tokens=64).generate(ps, 20, ignore_eos=True)
    st = model.expert_stats()
    print(f"qwen slots {slots} lean {lean} prefetch {prefetch}: {st}, excused vs hf {check(ours, ref)}, "
          f"vs in-memory {sum(a != b for a, b in zip(ours, mem))} differing sequences")
    check(mem, ref)
    assert st["misses"] > 0


@pytest.mark.parametrize("slots", [2, 4])
def test_speculative_dispatch_is_exact(experts_dir, slots):
    """The speculative all-resident dispatch must give bit-identical results to waiting for the routing."""
    from mj_mlx import expert_stream

    ps = prompts(5, [7, 33, 2])
    outs, stats = [], []
    for spec in (False, True):
        expert_stream.SPECULATE = spec
        try:
            model = GptOss(PATH, dtype=mx.float32, stream_experts=experts_dir, expert_slots=slots, lean=True,
                           prefetch=True)
            outs.append(Engine(model=model, num_blocks=256, max_running=1, max_batch_tokens=64).generate(
                ps, 24, ignore_eos=True))
            stats.append(model.expert_stats())
        finally:
            expert_stream.SPECULATE = True
    assert stats[0]["spec_used"] == 0 and stats[1]["spec_used"] > 0
    assert outs[0] == outs[1]


def _gpu_stream_built():
    try:
        from mj_mlx import expert_stream

        expert_stream.load_gpu_streamer()
        return True
    except Exception:
        return False


@pytest.mark.skipif(QPATH is None or not _gpu_stream_built(), reason="tiny Qwen3.6 or ext/_mj_stream not built")
@pytest.mark.parametrize("slots", [8, 16])
def test_gpu_streaming_is_exact(qwen_experts_dir, slots):
    """GPU-driven streaming (C++ Streamer: Metal events, IO thread) gives the same greedy tokens as the Python path,
    across prefill (Python) / decode (C++) handoffs over several requests."""
    from mj_mlx import Qwen35Moe, expert_stream

    rs = np.random.RandomState(7)
    ps = [[int(t) for t in rs.randint(0, 248000, n)] for n in (5, 40, 129, 12, 70)]
    outs, gstats = [], None
    for gpu in (False, True):
        expert_stream.GPU_STREAM = gpu
        try:
            model = Qwen35Moe(QPATH, dtype=mx.float32, stream_experts=qwen_experts_dir, expert_slots=slots, lean=True,
                              prefetch=True)
            seq = [Engine(model=model, num_blocks=256, max_running=1, max_batch_tokens=64).generate(
                [p], 16, ignore_eos=True)[0] for p in ps]
            # several requests decoding together: multi-row steps (the routing slice is strided for T > 1)
            bat = Engine(model=model, num_blocks=256, max_running=3, max_batch_tokens=64).generate(
                ps[:3], 16, ignore_eos=True)
            outs.append((seq, bat))
            if gpu:
                gstats = model.expert_stats()["gpu_stream"]
        finally:
            expert_stream.GPU_STREAM = False
    assert gstats and gstats["jobs"] > 0 and gstats["hits"] + gstats["misses"] > 0
    assert outs[0] == outs[1]


# ---- placement= for gpt-oss and Qwen3.6 (checklist 1.0b) ----------------------------------------------------------
@pytest.mark.parametrize("which", ["gpt_oss", "qwen"])
def test_placement_for_gpt_oss_and_qwen(which, tmp_path, monkeypatch):
    """load_model(placement=...) writes the expert files on first use, finds them again by their source (also under
    another name), and runs the same model as the explicit stream_experts / map_experts arguments."""
    from mj_mlx import load_model, placement

    path = PATH if which == "gpt_oss" else QPATH
    if path is None:
        pytest.skip("tiny checkpoint not downloaded")
    monkeypatch.setattr(placement, "CACHE", str(tmp_path))
    rs = np.random.RandomState(5)
    ps = [[int(t) for t in rs.randint(0, 200000, n)] for n in (5, 40, 70)]

    def gen(**kw):
        return Engine(model=load_model(path, dtype=mx.float32, **kw), num_blocks=256, max_running=4,
                      max_batch_tokens=64).generate(ps, 12, ignore_eos=True)

    assert placement.choose(path) == "resident"  # tiny: fits easily
    ref = gen(placement="resident")
    assert ref == gen()
    streamed = gen(placement="stream", expert_slots=2)
    store, mapped = placement.cache_dirs(path)
    assert os.path.exists(os.path.join(store, "index.json")) and store.startswith(str(tmp_path))
    assert streamed == gen(stream_experts=store, expert_slots=2)
    if placement.experts_quantized(path):
        assert gen(placement="mapped") == ref  # zero-copy views of the same bytes: the resident math
        assert os.path.exists(os.path.join(mapped, "index.json"))
    else:  # the tiny Qwen has bf16 experts, which Qwen35Moe does not map: "mapped" falls back to streaming
        assert gen(placement="mapped", expert_slots=2) == streamed
    moved = str(tmp_path / "renamed-store")
    os.rename(store, moved)  # a store made by hand or by an older version, under another name
    assert placement.cache_dirs(path)[0] == moved
    t0 = os.path.getmtime(os.path.join(moved, "index.json"))
    assert gen(placement="stream", expert_slots=2) == streamed  # reused, not repacked
    assert os.path.getmtime(os.path.join(moved, "index.json")) == t0


# ---- uneven slots per layer (RESULTS 1zt, checklist 6.2b) ---------------------------------------------------------
def test_lru_miss_curve_and_allocation():
    from mj_mlx import placement

    rs = np.random.RandomState(0)
    seq = rs.randint(0, 16, (200, 2))
    curve = placement.miss_curve(placement.lru_stack_distances(seq), 16)
    for S in (2, 5, 9, 16):  # the stack-distance curve equals a direct LRU simulation at every size
        lru, miss = [], 0
        for row in seq:
            for e in dict.fromkeys(int(x) for x in row):
                if e in lru:
                    lru.remove(e)
                else:
                    miss += 1
                    if len(lru) >= S:
                        lru.pop()
                lru.insert(0, e)
        assert curve[S] == miss, (S, curve[S], miss)
    flat, steep = np.linspace(100, 90, 17).astype(np.int64), np.linspace(400, 0, 17).astype(np.int64)
    alloc = placement.allocate_slots({0: flat, 1: steep}, budget=16, floor=2, step=2)
    assert sum(alloc.values()) == 16 and alloc[1] > alloc[0] >= 2


@pytest.mark.skipif(QPATH is None, reason="tiny Qwen3.6 checkpoint not downloaded")
def test_uneven_slots_run_and_profile_tool(qwen_experts_dir, tmp_path):
    import shutil

    from mj_mlx import Qwen35Moe, placement

    hfq = load_hf_qwen(QPATH)
    rs = np.random.RandomState(7)
    ps = [[int(t) for t in rs.randint(0, 248000, n)] for n in (5, 40, 70)]
    ref = [qwen_hf_greedy(hfq, p, 16) for p in ps]
    model = Qwen35Moe(QPATH, dtype=mx.float32, stream_experts=qwen_experts_dir, expert_slots={0: 2, 1: 4}, lean=True)
    assert [L["slab"].S for L in model.layers] == [2, 4]
    ours = Engine(model=model, num_blocks=256, max_running=4, max_batch_tokens=64).generate(ps, 16, ignore_eos=True)
    check(ours, ref)
    # the profile tool on a small trace, into a copy of the store
    store = str(tmp_path / "store")
    shutil.copytree(qwen_experts_dir, store)
    trace = tmp_path / "trace"
    trace.mkdir()
    for li in (0, 1):
        np.save(trace / f"r_{li:02d}.npy", rs.randint(0, 4, (64, 2)))
    subprocess.run([sys.executable, str(ROOT / "tools" / "mlx_slot_profile.py"), "--trace", str(trace), "--store",
                    store], check=True, capture_output=True)
    al = placement.uneven_slots(store, 2)
    assert isinstance(al, dict) and sum(al.values()) <= 4 and set(al) == {0, 1}
    assert placement.uneven_slots(qwen_experts_dir, 3) == 3  # no profile: the plain count


def test_engine_wired_limit_order_for_mapped_experts(tmp_path, monkeypatch):
    """A loaded model with memory-mapped experts: the Engine goes straight to the 1000 MB wired limit, never through
    the recommended size first (that wires the mapped experts' whole no-copy buffers: 2-3 s on gpt-oss-20b).
    MJ_WIRED_RAISE=1 keeps the old order for A/B runs. Same tokens either way."""
    from mj_mlx import load_model, placement, runner

    if PATH is None or not placement.experts_quantized(PATH):
        pytest.skip("tiny gpt-oss checkpoint with quantized experts not available")
    monkeypatch.setattr(placement, "CACHE", str(tmp_path))
    model = load_model(PATH, dtype=mx.float32, placement="mapped")
    assert model.map is not None
    ps = [[int(t) for t in np.random.RandomState(6).randint(0, 200000, n)] for n in (7, 30)]
    outs = {}
    for raise_first in ("0", "1"):
        monkeypatch.setenv("MJ_WIRED_RAISE", raise_first)
        seen = []
        monkeypatch.setattr(runner, "_WIRED", [None])
        monkeypatch.setattr(runner.mx, "set_wired_limit", lambda n: seen.append(n))
        outs[raise_first] = Engine(model=model, num_blocks=256, max_running=2, max_batch_tokens=64).generate(
            ps, 8, ignore_eos=True)
        if raise_first == "0":
            assert seen == [1000 << 20]
        else:
            assert len(seen) == 2 and seen[-1] == 1000 << 20 and seen[0] != seen[-1]
    assert outs["0"] == outs["1"]


@pytest.mark.skipif(QPATH is None, reason="tiny Qwen3.6 checkpoint not downloaded")
def test_slot_profile_auto_calibration(qwen_experts_dir, tmp_path, monkeypatch):
    """MJ_SLOT_PROFILE_AUTO=1: a streamed load whose store has no profile writes one from the calibration text
    (RESULTS 1zze); the next uneven_slots call uses it; a store that has one is left alone."""
    import json
    import shutil

    from mj_mlx import placement
    from mj_mlx.runner import load_model

    store = str(tmp_path / "store")
    shutil.copytree(qwen_experts_dir, store)
    assert placement.uneven_slots(store, 2) == 2
    monkeypatch.setenv("MJ_SLOT_PROFILE_AUTO", "1")
    load_model(QPATH, dtype=mx.float32, stream_experts=store, expert_slots=2, lean=True)
    prof = json.load(open(os.path.join(store, placement.PROFILE)))
    assert prof["trace"] == "calibration" and set(prof["curves"]) == {"0", "1"} and 0 < prof["tokens"] <= 512
    assert all(len(c) == prof["num_experts"] + 1 and c[0] >= c[-1] for c in prof["curves"].values())
    al = placement.uneven_slots(store, 2)
    assert isinstance(al, dict) and sum(al.values()) <= 4
    t = os.path.getmtime(os.path.join(store, placement.PROFILE))
    load_model(QPATH, dtype=mx.float32, stream_experts=store, expert_slots=2, lean=True)
    assert os.path.getmtime(os.path.join(store, placement.PROFILE)) == t  # written once
