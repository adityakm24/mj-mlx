"""expert_map helpers: zero-copy row maps (map_rows) and pinning mapped pages (MJ_MLOCK)."""

import os
import tempfile

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from mj_mlx import expert_map  # noqa: E402


def test_mlock_pins_a_mapped_array_and_mlx_sees_it():
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "x.bin")
        np.arange(4 * 16384, dtype=np.uint8).tofile(f)
        a = np.memmap(f, dtype=np.uint8, mode=expert_map.MAP_MODE, shape=(4, 16384))
        assert expert_map._mlock(a)
        x = mx.from_dlpack(a)
        assert np.array_equal(np.array(x), np.asarray(a))
        a[0, 0] = 7  # zero copy: a write through NumPy is visible in the MLX array
        assert int(x[0, 0].item()) == 7


def test_safetensors_rows_parallel_take_matches_memmap():
    import json
    import struct

    from mj_mlx.expert_stream import SafetensorsRows

    rs = np.random.RandomState(0)
    w = rs.randint(0, 1 << 16, (300, 40)).astype(np.uint16)
    with tempfile.TemporaryDirectory() as d:
        hdr = json.dumps({"t": {"dtype": "BF16", "shape": [300, 40], "data_offsets": [0, w.nbytes]}}).encode()
        with open(os.path.join(d, "m.safetensors"), "wb") as f:
            f.write(struct.pack("<Q", len(hdr)) + hdr + w.tobytes())
        r = SafetensorsRows(d, "t")
        for n in (1, 2, 16, 256, 300):  # serial, parallel, bulk
            ids = rs.randint(0, 300, n)
            got = np.array(r.take(ids, mx.bfloat16).view(mx.uint16))
            assert np.array_equal(got, w[ids])
        with pytest.raises(IndexError):
            r.take(np.array([0, 300]), mx.bfloat16)


def test_mlock_top_pins_only_the_ranked_experts(tmp_path, monkeypatch):
    """MJ_MLOCK_TOP=k: per layer, only the k experts ranked first in expert_rank.json are pinned."""
    import json

    E, rows = 4, 16384  # one page per expert row
    parts = [dict(name="gu_w", dtype="uint8", shape=[rows], bytes=rows)]
    json.dump(dict(layout="part_major", num_experts=E, parts=parts, part_offsets=[0], num_layers=1),
              open(tmp_path / "index.json", "w"))
    np.zeros(E * rows, np.uint8).tofile(tmp_path / "layer_00.bin")
    json.dump(dict(layers={"0": [2, 0, 3, 1]}), open(tmp_path / "expert_rank.json", "w"))
    monkeypatch.setattr(expert_map, "MLOCK", True)
    monkeypatch.setattr(expert_map, "MLOCK_TOP", 2)
    m = expert_map.MappedExperts(str(tmp_path))
    pinned = []
    monkeypatch.setattr(expert_map, "_mlock", lambda a: pinned.append(a) or True)
    out = m.layer(0)
    assert out["gu_w"].shape == (E, rows) and m.locked == 2 * rows
    base = m._maps[0].ctypes.data
    assert sorted((p.ctypes.data - base) // rows for p in pinned) == [0, 2]
    monkeypatch.setattr(expert_map, "MLOCK_TOP", 0)  # off: the whole part
    m2 = expert_map.MappedExperts(str(tmp_path))
    pinned.clear()
    m2.layer(0)
    assert m2.locked == E * rows and len(pinned) == 1
