"""python -m mj_mlx cache (checklist 1.5): lists recognised entries with their sources, deletes only those."""

import json

from mj_mlx import cache


def make(tmp_path):
    mj, prep = tmp_path / "mj", tmp_path / "prep"
    (mj / "store").mkdir(parents=True)
    json.dump(dict(source=str(tmp_path / "ckpt")), open(mj / "store" / "index.json", "w"))
    (mj / "store" / "layer_00.bin").write_bytes(b"x" * 1000)
    (mj / "maps").mkdir()
    json.dump(dict(layout="part_major", source_store=str(mj / "store")), open(mj / "maps" / "index.json", "w"))
    (mj / "notes").mkdir()  # not ours
    (prep / "abc").mkdir(parents=True)
    json.dump(dict(checkpoint=str(tmp_path / "gone")), open(prep / "abc" / "key.json", "w"))
    (tmp_path / "ckpt").mkdir()
    return dict(millijoule=str(mj), prep=str(prep))


def test_list_and_remove(tmp_path):
    roots = make(tmp_path)
    es = {e["name"]: e for e in cache.entries(roots)}
    assert es["millijoule/store"]["kind"] == "experts-stream" and es["millijoule/store"]["source_exists"]
    assert es["millijoule/store"]["bytes"] >= 1000
    assert es["millijoule/maps"]["kind"] == "experts-mapped"
    assert es["millijoule/notes"]["kind"] == "other"
    assert es["prep/abc"]["kind"] == "prep" and es["prep/abc"]["source_exists"] is False
    log = []
    cache.remove(["millijoule/notes"], roots=roots, log=log.append)  # not ours: kept
    assert (tmp_path / "mj" / "notes").exists() and "skip" in log[0]
    cache.remove([], orphans=True, roots=roots, log=log.append)  # the prep entry's checkpoint is gone
    assert not (tmp_path / "prep" / "abc").exists() and (tmp_path / "mj" / "store").exists()
    freed = cache.remove(["millijoule/store"], roots=roots, log=log.append)
    assert freed >= 1000 and not (tmp_path / "mj" / "store").exists()


def test_repack_refuses_when_disk_is_short(tmp_path, monkeypatch):
    import collections

    import pytest

    from mj_mlx import placement

    Usage = collections.namedtuple("Usage", "total used free")
    monkeypatch.setattr("shutil.disk_usage", lambda p: Usage(10, 10, 1 << 30))
    with pytest.raises(OSError, match="not enough disk"):
        placement.check_disk(str(tmp_path), 5 << 30)
    placement.check_disk(str(tmp_path), 0)  # nothing to write: no check


def test_unsupported_model_says_what_to_do(tmp_path):
    import pytest

    from mj_mlx.runner import load_model

    json.dump(dict(model_type="llama"), open(tmp_path / "config.json", "w"))
    with pytest.raises(ValueError, match="mlx-lm"):
        load_model(str(tmp_path))


def test_too_big_models_say_so(tmp_path, monkeypatch):
    import pytest

    from mj_mlx import placement

    mem = dict(working_set=10 << 30, ram=16 << 30)
    monkeypatch.setattr(placement, "checkpoint_bytes", lambda p: (12 << 30, 0))  # dense, 12 GB
    with pytest.raises(MemoryError, match="dense model"):
        placement.choose("x", mem=mem)
    monkeypatch.setattr(placement, "checkpoint_bytes", lambda p: (40 << 30, 30 << 30))  # 10 GB not experts
    with pytest.raises(MemoryError, match="even with every routed expert"):
        placement.choose("x", mem=mem)
    monkeypatch.setattr(placement, "checkpoint_bytes", lambda p: (20 << 30, 17 << 30))
    assert placement.choose("x", mem=mem) == "stream"
    monkeypatch.setattr(placement, "checkpoint_bytes", lambda p: (12 << 30, 9 << 30))
    assert placement.choose("x", mem=mem) == "mapped"
