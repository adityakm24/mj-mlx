"""MJ_PREP_CACHE (RESULTS 1zzc): arrays derived at load (low-bit candidate head, 8-bit / packed attention) read back
from the cache are the arrays the computation returns, on a fresh write and on a read, and the model's greedy output
is unchanged."""

import os
import sys

import pytest

mx = pytest.importorskip("mlx.core")
sys.path.insert(0, os.path.dirname(__file__))
from hf_ref import snapshot  # noqa: E402

from mj_mlx import Engine, GptOss  # noqa: E402

PATH = snapshot()
pytestmark = pytest.mark.skipif(PATH is None, reason="tiny gpt-oss checkpoint not downloaded")


def greedy(m, ids, n=8):
    eng = Engine(model=m, num_blocks=64, max_running=1)
    eng.add_request(ids, n)
    out = []
    while eng.has_work():
        for f in eng.step():
            out = list(f[2])
    return out


@pytest.mark.parametrize("kw", [dict(attn_bits=8), dict(attn_pack=True), dict(lean=True), dict(lm_head_bits=4)],
                         ids=["attn8", "pack", "lean", "head4"])
def test_cache_returns_the_computed_arrays(tmp_path, monkeypatch, kw):
    monkeypatch.setenv("MJ_PREP_CACHE", "0")
    ref = GptOss(PATH, dtype=mx.bfloat16, **kw)
    monkeypatch.setenv("MJ_PREP_CACHE", str(tmp_path))
    first = GptOss(PATH, dtype=mx.bfloat16, **kw)  # computes and writes
    written = sorted(os.listdir(next(p for p in tmp_path.iterdir())))
    second = GptOss(PATH, dtype=mx.bfloat16, **kw)  # reads
    for m in (first, second):
        pa, pb = ref.parameters(), m.parameters()
        assert len(pa) == len(pb)
        for a, b in zip(pa, pb):
            assert a.shape == b.shape and a.dtype == b.dtype and mx.array_equal(a, b).item()
    ids = [1, 5, 9, 2, 7, 3]
    assert greedy(ref, ids) == greedy(second, ids)
    assert "key.json" in written


def test_cache_key_follows_the_options(tmp_path, monkeypatch):
    monkeypatch.setenv("MJ_PREP_CACHE", str(tmp_path))
    GptOss(PATH, dtype=mx.bfloat16, attn_bits=8)
    GptOss(PATH, dtype=mx.bfloat16, attn_pack=True)
    assert len(list(tmp_path.iterdir())) == 2


def test_spot_check_drops_a_stale_cache(tmp_path, monkeypatch, capfd):
    """A cached array that no longer matches a fresh derivation (here: overwritten) is caught at load, the folder is
    deleted and the load computes instead."""
    monkeypatch.setenv("MJ_PREP_CACHE", str(tmp_path))
    ref = GptOss(PATH, dtype=mx.bfloat16, attn_bits=8)
    d = next(p for p in tmp_path.iterdir())
    f = d / "attn8_L0.safetensors"
    arrs = mx.load(str(f))
    k = next(iter(arrs))
    arrs[k] = arrs[k] + 1 if arrs[k].dtype != mx.uint32 else arrs[k] ^ 1
    mx.save_safetensors(str(f), arrs)
    m = GptOss(PATH, dtype=mx.bfloat16, attn_bits=8)
    assert "differs from a fresh derivation" in capfd.readouterr().err
    for a, b in zip(ref.parameters(), m.parameters()):
        assert mx.array_equal(a, b).item()
    assert not d.exists()


def test_off_and_low_disk(tmp_path, monkeypatch, capfd):
    import mj_mlx.gpt_oss as G

    monkeypatch.setenv("MJ_PREP_CACHE", "0")
    assert GptOss(PATH, dtype=mx.bfloat16, attn_bits=8).prep is None
    monkeypatch.setenv("MJ_PREP_CACHE", str(tmp_path))
    monkeypatch.setattr(G.PrepCache, "EST_BYTES", 1 << 60)  # no disk is twice this
    GptOss(PATH, dtype=mx.bfloat16, attn_bits=8)
    assert "not writing" in capfd.readouterr().err
    assert not any(p.suffix == ".safetensors" for d in tmp_path.iterdir() for p in d.iterdir())


def test_key_follows_the_code(tmp_path, monkeypatch):
    import mj_mlx.gpt_oss as G

    monkeypatch.setenv("MJ_PREP_CACHE", str(tmp_path))
    a = G.PrepCache(str(tmp_path), PATH, dict(x=1)).dir
    monkeypatch.setattr(G.PrepCache, "VERSION", G.PrepCache.VERSION + 1)
    assert G.PrepCache(str(tmp_path), PATH, dict(x=1)).dir != a


def test_a_new_key_prunes_older_folders_of_the_same_model_and_options(tmp_path, monkeypatch):
    import mj_mlx.gpt_oss as G

    import time

    v = G.PrepCache.VERSION
    a = G.PrepCache(str(tmp_path), PATH, dict(x=1)).dir
    other = G.PrepCache(str(tmp_path), PATH, dict(x=2)).dir  # other options: never pruned
    time.sleep(0.01)
    monkeypatch.setattr(G.PrepCache, "VERSION", v + 1)
    b = G.PrepCache(str(tmp_path), PATH, dict(x=1)).dir  # two keys of one model and options: both kept (an A/B)
    assert os.path.exists(a) and os.path.exists(b)
    time.sleep(0.01)
    monkeypatch.setattr(G.PrepCache, "VERSION", v + 2)
    c = G.PrepCache(str(tmp_path), PATH, dict(x=1)).dir  # a third: the oldest goes
    assert not os.path.exists(a) and os.path.exists(b) and os.path.exists(c) and os.path.exists(other)
    monkeypatch.setattr(G.PrepCache, "VERSION", v + 1)
    assert G.PrepCache(str(tmp_path), PATH, dict(x=1)).dir == b and os.path.exists(c)  # reuse prunes nothing
