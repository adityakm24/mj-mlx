"""StepBatch: flat arrays, release reporting and update_sampled, driven from Python."""

import array
import sys

import pytest

from mj_mlx import scheduler

def arr(batch, name):
    dtype, _ = scheduler.STEP_BATCH_DTYPES[name]
    code = {"uint64": "Q", "int32": "i", "uint32": "I", "uint8": "B", "float32": "f"}[dtype]
    a = array.array(code)
    a.frombytes(getattr(batch, name))
    return a.tolist()


def test_arrays_line_up_with_items():
    s = scheduler.Scheduler(block_size=4, num_blocks=64, max_running=4, max_batch_tokens=64)
    a = s.add_request(list(range(10, 19)), max_new_tokens=4)
    b = s.add_request([7, 8], max_new_tokens=4, temperature=0.7, top_p=0.9, top_k=40, seed=123)
    batch = s.schedule()
    assert (len(batch), batch.num_tokens, batch.num_sample, batch.block_size) == (2, 11, 2, 4)
    assert arr(batch, "request_ids") == [a, b]
    assert arr(batch, "cu_seqlens") == [0, 9, 11]
    assert arr(batch, "positions") == list(range(9)) + [0, 1]
    assert arr(batch, "sample_rows") == [8, 10]
    assert arr(batch, "seed") == [a, 123]
    assert arr(batch, "top_k") == [0, 40]
    tables = arr(batch, "block_tables")
    width = batch.max_blocks
    slots = arr(batch, "slot_mapping")
    for row, pos in enumerate(arr(batch, "positions")):
        item = 0 if row < 9 else 1
        assert slots[row] == tables[item * width + pos // 4] * 4 + pos % 4
    assert [t[0] for t in batch] == [a, b]  # iterating still yields the old tuples
    with pytest.raises(ValueError):
        s.update_sampled([1])
    tokens = array.array("I", [5, 6])
    if sys.byteorder != "little":
        tokens.byteswap()
    assert s.update_sampled(tokens.tobytes()) == []


def test_released_reports_finish_and_abort():
    s = scheduler.Scheduler(block_size=4, num_blocks=64, max_running=4, max_batch_tokens=64)
    a = s.add_request([1, 2], max_new_tokens=1)
    b = s.add_request([3, 4], max_new_tokens=5)
    s.schedule()
    assert [f[0] for f in s.update_sampled([9, 9])] == [a]
    batch = s.schedule()
    assert batch.released == [(a, "finished")]
    s.update_sampled([9])
    assert s.abort(b)
    assert s.take_released() == [(b, "aborted")]
    assert not s.schedule()


def test_as_numpy_shapes():
    np = pytest.importorskip("numpy")
    s = scheduler.Scheduler(block_size=4, num_blocks=64, max_running=4, max_batch_tokens=64)
    s.add_request(list(range(9)))
    s.add_request([1])
    out = scheduler.as_numpy(s.schedule())
    assert out["block_tables"].shape == (2, 3)
    assert out["token_ids"].dtype == np.uint32 and out["token_ids"].shape == (10,)
    assert out["sample_rows"].tolist() == [8, 9]


def test_state_slots():
    with pytest.raises(ValueError):
        scheduler.Scheduler(num_state_slots=2)  # needs the prefix cache off
    s = scheduler.Scheduler(
        block_size=4, num_blocks=64, max_running=4, max_batch_tokens=64, enable_prefix_cache=False, num_state_slots=2
    )
    ids = [s.add_request([1, 2, 3 + i], max_new_tokens=2) for i in range(3)]
    batch = s.schedule()
    assert arr(batch, "request_ids") == ids[:2]  # two slots admit two requests
    assert arr(batch, "state_slots") == [0, 1]
    assert s.num_free_state_slots == 0
    s.update_sampled([9, 9])
    batch = s.schedule()
    assert arr(batch, "state_slots") == [0, 1]  # each keeps its slot while decoding
    s.update_sampled([9, 9])  # both finish
    batch = s.schedule()
    assert arr(batch, "request_ids") == [ids[2]]
    assert arr(batch, "starts") == [0]
    assert arr(batch, "state_slots")[0] in (0, 1)
    assert scheduler.Scheduler().schedule().state_slots == b""
