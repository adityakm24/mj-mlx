"""Drives the Rust scheduler from Python with a fake model whose next token depends only on (request, length)."""

import pytest

from mj_mlx import scheduler

def fake_token(seed: int, length: int) -> int:
    return 1000 + (seed * 2654435761 + length * 40503) % 5000


def run(sched: scheduler.Scheduler, seeds: dict[int, int]) -> dict[int, tuple]:
    done = {}
    while sched.has_work():
        plan = sched.schedule()
        assert plan, "work left but nothing scheduled"
        outputs = [
            (rid, fake_token(seeds[rid], start + len(ids)))
            for rid, start, ids, _blocks, sample, _decode in plan.items()
            if sample
        ]
        for rid, reason, tokens, cached, preempted in sched.update(outputs):
            done[rid] = (reason, tokens, cached, preempted)
        sched.check_invariants()
    return done


def test_end_to_end_with_shared_prefix_and_preemption():
    system = list(range(64))
    prompts = [system + [500 + i] * (3 + i) for i in range(16)]

    def outputs(num_blocks: int):
        sched = scheduler.Scheduler(block_size=16, num_blocks=num_blocks, max_running=8, max_batch_tokens=128)
        seeds = {sched.add_request(p, max_new_tokens=20 + i): i for i, p in enumerate(prompts)}
        done = run(sched, seeds)
        return [done[rid][1] for rid in seeds], sched.stats()

    roomy, roomy_stats = outputs(1024)
    tight, tight_stats = outputs(24)
    assert roomy == tight
    assert [len(o) for o in roomy] == [20 + i for i in range(16)]
    # Requests 0 and 1 prefill the system prompt in the same first step; the other 14 hit the cache.
    assert roomy_stats["cached_prompt_tokens"] == 14 * 64
    assert tight_stats["preemptions"] > 0


def test_errors_map_to_python_exceptions():
    sched = scheduler.Scheduler(block_size=4, num_blocks=8, max_running=2, max_batch_tokens=16)
    with pytest.raises(ValueError):
        sched.add_request([])
    with pytest.raises(ValueError):
        sched.add_request(list(range(40)))
    rid = sched.add_request([1, 2, 3], max_new_tokens=2)
    sched.schedule()
    with pytest.raises(RuntimeError):
        sched.schedule()
    with pytest.raises(RuntimeError):
        sched.update([])
    assert sched.update([(rid, 7)]) == []
    with pytest.raises(ValueError):
        scheduler.Scheduler(max_running=64, max_batch_tokens=32)


def test_stop_token_and_abort():
    sched = scheduler.Scheduler(block_size=4, num_blocks=32, max_running=4, max_batch_tokens=32)
    a = sched.add_request([1, 2], max_new_tokens=10, stop_token_ids=[42])
    b = sched.add_request([3, 4], max_new_tokens=10)
    sched.schedule()
    assert sched.abort(b)
    finished = sched.update([(a, 42), (b, 1)])
    assert finished == [(a, "stop", [42], 0, 0)]
    assert not sched.has_work()
    sched.check_invariants()


def run_overlapped(sched: scheduler.Scheduler, seeds: dict[int, int]) -> dict[int, tuple]:
    """A runner loop with two steps in flight: schedule step N+1, then complete step N."""
    import numpy as np

    done, in_flight = {}, []
    while sched.has_work():
        batch = sched.schedule()
        if batch:
            a = scheduler.as_numpy(batch)
            ids = a["token_ids"].copy()
            pending = a["prev_sample_idx"] >= 0
            if pending.any():
                prev_tokens = in_flight[-1]
                ids[pending] = prev_tokens[a["prev_sample_idx"][pending]]
            assert (ids != 2**32 - 1).all()
            rows = a["sample_rows"]
            tokens = np.array(
                [fake_token(seeds[int(r)], int(a["positions"][row]) + 1) for r, row in zip(a["sample_request_ids"], rows)],
                dtype=np.uint32,
            )
            in_flight.append(tokens)
        if len(in_flight) == 2 or (not batch and in_flight):
            for f in sched.update_sampled(in_flight.pop(0).tobytes()):
                done[f[0]] = f[1:]
        sched.check_invariants()
    return done


def test_overlap_gives_the_same_outputs():
    prompts = [[i, i + 1, i + 2] for i in range(0, 60, 3)]

    def outputs(overlap: bool):
        s = scheduler.Scheduler(block_size=4, num_blocks=24, max_running=4, max_batch_tokens=16, overlap=overlap)
        seeds = {}
        for i, p in enumerate(prompts):
            stop = [fake_token(i, len(p) + 3)] if i % 2 else None
            seeds[s.add_request(p, max_new_tokens=12, stop_token_ids=stop)] = i
        done = run_overlapped(s, seeds) if overlap else run(s, seeds)
        return [done[i][:2] for i in sorted(done)], s.stats()

    ref, base = outputs(False)
    got, stats = outputs(True)
    assert got == ref
    assert base["overlapped_inputs"] == 0 and stats["overlapped_inputs"] > 0
    assert stats["discarded_samples"] == 10  # one per stopping request

    s = scheduler.Scheduler(overlap=True)
    s.add_request([1, 2, 3], max_new_tokens=4)
    s.schedule()
    b = s.schedule()
    assert s.num_in_flight == 2 and b.num_pending_inputs == 1
    with pytest.raises(RuntimeError):
        s.schedule()
