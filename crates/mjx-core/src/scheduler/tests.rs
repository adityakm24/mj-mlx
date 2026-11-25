// Copyright 2026 the MilliJoule authors
// SPDX-License-Identifier: Apache-2.0
use super::*;

fn cfg(block_size: usize, num_blocks: usize, max_running: usize, max_batch_tokens: usize) -> EngineConfig {
    EngineConfig {
        block_size,
        num_blocks,
        max_running,
        max_batch_tokens,
        max_prefill_chunk: max_batch_tokens,
        prefill_area_cap: 0,
        max_seq_len: 1 << 20,
        enable_prefix_cache: true,
        num_state_slots: 0,
        overlap: false,
    }
}

fn params(max_new_tokens: usize) -> SamplingParams {
    SamplingParams { max_new_tokens, ..Default::default() }
}

/// A stand-in model: the next token depends only on the request's prompt and length, like greedy decoding.
fn fake_token(prompt_seed: u32, len: usize) -> u32 {
    let mut x = prompt_seed.wrapping_mul(2_654_435_761) ^ (len as u32).wrapping_mul(40_503);
    x ^= x >> 13;
    1000 + x % 5000
}

/// Runs the scheduler to completion with `fake_token`, checking invariants after every call. It also plays the
/// runner's side of `released` and `state_slots`: a request is "live" from its first item until it is released,
/// every release must name a live request, and nothing is live once the scheduler drains. A state slot changes
/// hands only at an item with `start == 0` (where the runner zeroes it), and a request keeps its slot otherwise.
/// With `overlap` on it keeps two steps in flight, resolves each pending input from the previous step's tokens and
/// checks that it names the right request and position.
fn drive(s: &mut Scheduler, seeds: &HashMap<RequestId, u32>) -> HashMap<RequestId, Finished> {
    let mut done = HashMap::new();
    let mut live: HashSet<RequestId> = HashSet::new();
    // Runner-side state: which request's recurrent state each slot holds.
    let mut slot_of: HashMap<RequestId, i32> = HashMap::new();
    let mut owner: HashMap<i32, RequestId> = HashMap::new();
    let num_slots = s.config().num_state_slots as i32;
    fn release(
        live: &mut HashSet<RequestId>,
        slot_of: &mut HashMap<RequestId, i32>,
        owner: &mut HashMap<i32, RequestId>,
        rel: &[Released],
    ) {
        for r in rel {
            assert!(live.remove(&r.request_id), "release of {} ({:?}) the runner never ran", r.request_id, r.reason);
            if let Some(slot) = slot_of.remove(&r.request_id) {
                owner.remove(&slot);
            }
        }
    }
    let depth = if s.config().overlap { 2 } else { 1 };
    // Steps "running on the device": plan and its sampled tokens, oldest first.
    let mut in_flight: VecDeque<(StepPlan, Vec<u32>)> = VecDeque::new();
    let mut guard = 0;
    while s.has_work() {
        guard += 1;
        assert!(guard < 100_000, "scheduler made no progress");
        let plan = s.schedule().unwrap();
        s.check_invariants().unwrap();
        release(&mut live, &mut slot_of, &mut owner, &plan.released);
        if plan.is_empty() {
            assert!(!in_flight.is_empty(), "work left but nothing scheduled");
            let (_, tokens) = in_flight.pop_front().unwrap();
            for f in s.update_sampled(&tokens).unwrap() {
                done.insert(f.request_id, f);
            }
            s.check_invariants().unwrap();
            continue;
        }
        assert!(plan.num_tokens <= s.config().max_batch_tokens);
        assert!(plan.num_items() <= s.config().max_running);
        assert_eq!(plan.prev_sample_idx.len(), plan.num_tokens);
        for i in 0..plan.num_items() {
            for row in plan.cu_seqlens[i] as usize..plan.cu_seqlens[i + 1] as usize {
                let k = plan.prev_sample_idx[row];
                if k < 0 {
                    assert_ne!(plan.token_ids[row], u32::MAX);
                    continue;
                }
                assert_eq!(plan.token_ids[row], u32::MAX);
                assert_eq!(plan.lens[i], 1, "a pending input is a one-token decode");
                let (prev, prev_tokens) = in_flight.back().expect("pending input with no step in flight");
                let k = k as usize;
                assert_eq!(prev.sample_request_ids[k], plan.request_ids[i]);
                assert_eq!(prev.positions[prev.sample_rows[k] as usize] + 1, plan.positions[row]);
                assert_ne!(prev_tokens[k], u32::MAX);
            }
        }
        let mut seen = HashSet::new();
        for (i, &id) in plan.request_ids.iter().enumerate() {
            live.insert(id);
            let slot = plan.state_slots[i];
            if num_slots == 0 {
                assert_eq!(slot, -1);
                continue;
            }
            assert!((0..num_slots).contains(&slot), "slot {slot} out of range");
            assert!(seen.insert(slot), "slot {slot} used twice in one step");
            if plan.starts[i] == 0 {
                assert!(owner.get(&slot).is_none_or(|&o| o == id), "slot {slot} still holds live request");
                if let Some(old) = slot_of.insert(id, slot) {
                    owner.remove(&old);
                }
                owner.insert(slot, id);
            } else {
                assert_eq!(slot_of.get(&id), Some(&slot), "request {id} changed slot mid-sequence");
            }
        }
        let tokens: Vec<u32> = plan
            .sample_rows
            .iter()
            .zip(&plan.sample_request_ids)
            .map(|(&row, id)| fake_token(seeds[id], plan.positions[row as usize] as usize + 1))
            .collect();
        in_flight.push_back((plan, tokens));
        if in_flight.len() == depth {
            let (_, tokens) = in_flight.pop_front().unwrap();
            for f in s.update_sampled(&tokens).unwrap() {
                assert!(done.insert(f.request_id, f).is_none(), "request finished twice");
            }
            s.check_invariants().unwrap();
        }
    }
    assert!(in_flight.is_empty());
    release(&mut live, &mut slot_of, &mut owner, &s.take_released());
    assert!(live.is_empty(), "requests never released: {live:?}");
    assert!(owner.is_empty() && slot_of.is_empty(), "state slots still owned after drain");
    assert_eq!(s.num_free_state_slots(), s.config().num_state_slots);
    done
}

#[test]
fn one_request_prefills_then_decodes() {
    let mut s = Scheduler::new(cfg(4, 16, 4, 64)).unwrap();
    let id = s.add_request(vec![1, 2, 3, 4, 5], params(3)).unwrap();

    let p = s.schedule().unwrap();

    let items = p.items();
    assert_eq!(items.len(), 1);
    let it = &items[0];
    assert_eq!((it.start, it.input_ids.clone(), it.sample, it.is_decode), (0, vec![1, 2, 3, 4, 5], true, false));
    assert_eq!(it.block_table.len(), 2);
    assert!(s.update(&[StepOutput { request_id: id, token: 9 }]).unwrap().is_empty());

    let p = s.schedule().unwrap();

    let items = p.items();
    let it = &items[0];
    assert_eq!((it.start, it.input_ids.clone(), it.is_decode), (5, vec![9], true));
    s.update(&[StepOutput { request_id: id, token: 10 }]).unwrap();

    s.schedule().unwrap();
    let f = s.update(&[StepOutput { request_id: id, token: 11 }]).unwrap();
    assert_eq!(f.len(), 1);
    assert_eq!(f[0].output_tokens, vec![9, 10, 11]);
    assert_eq!(f[0].reason, FinishReason::Length);
    assert!(!s.has_work());
    s.check_invariants().unwrap();
}

#[test]
fn long_prompt_is_chunked_and_samples_only_at_the_end() {
    let mut s = Scheduler::new(cfg(4, 16, 2, 4)).unwrap();
    let id = s.add_request((0..10).collect(), params(1)).unwrap();
    let mut chunks = Vec::new();
    loop {
        let p = s.schedule().unwrap();
        let items = p.items();
        let it = items[0].clone();
        chunks.push((it.start, it.input_ids.len(), it.sample));
        let out = if it.sample { vec![StepOutput { request_id: id, token: 1 }] } else { vec![] };
        if !s.update(&out).unwrap().is_empty() {
            break;
        }
    }
    assert_eq!(chunks, vec![(0, 4, false), (4, 4, false), (8, 2, true)]);
}

#[test]
fn area_cap_shrinks_chunks_as_the_context_grows() {
    let mut c = cfg(16, 4096, 1, 2048);
    c.prefill_area_cap = 2048 * 8192;
    assert_eq!(c.prefill_chunk_at(0), 2048);
    assert_eq!(c.prefill_chunk_at(6144), 2048); // 2048 x 8192: at the cap
    let at_16k = c.prefill_chunk_at(14336);
    assert!(at_16k < 2048 && at_16k * (14336 + at_16k) <= c.prefill_area_cap && at_16k % 16 == 0, "{at_16k}");
    assert_eq!(c.prefill_chunk_at(1 << 30), 256); // floor
    let mut s = Scheduler::new(c).unwrap();
    let id = s.add_request((0..16384).collect(), params(1)).unwrap();
    let (mut start, mut lens) = (0usize, Vec::new());
    loop {
        let p = s.schedule().unwrap();
        let it = p.items()[0].clone();
        assert_eq!(it.start, start);
        let n = it.input_ids.len();
        assert!(n * (start + n) <= 2048 * 8192 || n == 256, "chunk {n} at {start}");
        lens.push(n);
        start += n;
        let out = if it.sample { vec![StepOutput { request_id: id, token: 1 }] } else { vec![] };
        if !s.update(&out).unwrap().is_empty() {
            break;
        }
    }
    assert_eq!(start, 16384);
    assert_eq!(&lens[..4], &[2048, 2048, 2048, 2048]);
    assert!(lens[4] < 2048);
}

#[test]
fn decodes_and_prefill_chunks_share_a_step() {
    let mut s = Scheduler::new(cfg(4, 64, 4, 8)).unwrap();
    let a = s.add_request(vec![1, 2, 3], params(10)).unwrap();
    s.schedule().unwrap();
    s.update(&[StepOutput { request_id: a, token: 7 }]).unwrap();

    let b = s.add_request((0..20).collect(), params(10)).unwrap();
    let p = s.schedule().unwrap();
    let items = p.items();
    assert_eq!(items.len(), 2);
    assert_eq!((items[0].request_id, items[0].is_decode), (a, true));
    assert_eq!((items[1].request_id, items[1].input_ids.len()), (b, 7));
    assert_eq!(p.num_tokens, 8);
}

#[test]
fn second_request_hits_the_prefix_cache() {
    let mut s = Scheduler::new(cfg(4, 64, 4, 64)).unwrap();
    let prompt: Vec<u32> = (100..112).collect();
    let seeds: HashMap<_, _> = [(s.add_request(prompt.clone(), params(2)).unwrap(), 1)].into();
    drive(&mut s, &seeds);
    assert!(s.num_cached_blocks() >= 3);

    let b = s.add_request(prompt, params(2)).unwrap();
    let p = s.schedule().unwrap();
    let items = p.items();
    // 12 tokens = 3 blocks, but one token is always recomputed, so 2 blocks hit.
    assert_eq!((items[0].start, items[0].input_ids.len()), (8, 4));
    s.update(&[StepOutput { request_id: b, token: 5 }]).unwrap();
    assert_eq!(s.stats().cached_prompt_tokens, 8);
}

#[test]
fn concurrent_requests_share_a_prompt_once_its_prefill_is_done() {
    let mut s = Scheduler::new(cfg(4, 64, 4, 64)).unwrap();
    let system: Vec<u32> = (0..16).collect();
    let a = s.add_request([system.clone(), vec![50]].concat(), params(5)).unwrap();
    s.schedule().unwrap();
    s.update(&[StepOutput { request_id: a, token: 1 }]).unwrap();
    // `a` is still running; its prompt blocks are already shared.
    let b = s.add_request([system, vec![60]].concat(), params(5)).unwrap();
    let p = s.schedule().unwrap();
    let items = p.items();
    let item_b = items.iter().find(|i| i.request_id == b).unwrap();
    assert_eq!(item_b.start, 16);
    let item_a = items.iter().find(|i| i.request_id == a).unwrap();
    assert_eq!(item_a.block_table[..4], item_b.block_table[..4]);
}

#[test]
fn memory_pressure_preempts_and_outputs_match_an_unconstrained_run() {
    let workload: Vec<(Vec<u32>, usize)> =
        (0..12u32).map(|i| ((0..(20 + 7 * i)).map(|t| t * 31 + i).collect(), 10 + (i as usize % 5) * 6)).collect();

    let run = |num_blocks: usize| {
        let mut s = Scheduler::new(cfg(4, num_blocks, 8, 32)).unwrap();
        let mut seeds = HashMap::new();
        let mut order = Vec::new();
        for (i, (prompt, n)) in workload.iter().enumerate() {
            let id = s.add_request(prompt.clone(), params(*n)).unwrap();
            seeds.insert(id, i as u32);
            order.push(id);
        }
        let done = drive(&mut s, &seeds);
        let outputs: Vec<Vec<u32>> = order.iter().map(|id| done[id].output_tokens.clone()).collect();
        (outputs, s.stats().preemptions)
    };

    let (roomy, roomy_preempt) = run(4096);
    let (tight, tight_preempt) = run(40);
    assert_eq!(roomy_preempt, 0);
    assert!(tight_preempt > 0, "the tight pool should have forced preemptions");
    assert_eq!(roomy, tight);
    for (out, (_, n)) in roomy.iter().zip(&workload) {
        assert_eq!(out.len(), *n);
    }
}

#[test]
fn stop_token_ends_the_request_unless_ignored() {
    let mut s = Scheduler::new(cfg(4, 16, 2, 16)).unwrap();
    let stop = SamplingParams { max_new_tokens: 10, stop_token_ids: vec![2], ..Default::default() };
    let a = s.add_request(vec![1], stop.clone()).unwrap();
    let b = s.add_request(vec![1], SamplingParams { ignore_eos: true, ..stop }).unwrap();
    s.schedule().unwrap();
    let f = s.update(&[StepOutput { request_id: a, token: 2 }, StepOutput { request_id: b, token: 2 }]).unwrap();
    assert_eq!(f.len(), 1);
    assert_eq!((f[0].request_id, f[0].reason), (a, FinishReason::Stop));
}

#[test]
fn requests_stop_at_kv_capacity() {
    // 4 blocks of 2 tokens: a request can hold at most 8 tokens.
    let mut s = Scheduler::new(cfg(2, 4, 1, 8)).unwrap();
    assert!(matches!(s.add_request((0..8).collect(), params(1)), Err(SchedError::PromptTooLong { .. })));
    let id = s.add_request((0..5).collect(), params(100)).unwrap();
    let done = drive(&mut s, &[(id, 3)].into());
    assert_eq!(done[&id].reason, FinishReason::MaxSeqLen);
    assert_eq!(done[&id].output_tokens.len(), 3);
}

#[test]
fn bad_calls_are_rejected() {
    let mut s = Scheduler::new(cfg(4, 16, 2, 16)).unwrap();
    assert_eq!(s.add_request(vec![], params(1)), Err(SchedError::EmptyPrompt));
    assert_eq!(s.add_request(vec![1], params(0)), Err(SchedError::ZeroMaxNewTokens));
    assert_eq!(s.update(&[]), Err(SchedError::NoStepInFlight));
    let a = s.add_request(vec![1, 2], params(4)).unwrap();
    s.schedule().unwrap();
    assert_eq!(s.schedule(), Err(SchedError::StepInFlight));
    assert_eq!(s.update(&[]), Err(SchedError::MissingOutput(a)));
    assert_eq!(s.update(&[StepOutput { request_id: 99, token: 1 }]), Err(SchedError::UnexpectedOutput(99)));
    let dup = [StepOutput { request_id: a, token: 1 }, StepOutput { request_id: a, token: 1 }];
    assert_eq!(s.update(&dup), Err(SchedError::DuplicateOutput(a)));
    // Errors leave the step in flight, so a correct update still works.
    assert!(s.update(&[StepOutput { request_id: a, token: 1 }]).is_ok());
    assert!(Scheduler::new(cfg(4, 16, 32, 16)).is_err());
}

#[test]
fn abort_during_a_step_frees_blocks_and_ignores_its_output() {
    let mut s = Scheduler::new(cfg(4, 16, 2, 16)).unwrap();
    let a = s.add_request((0..9).collect(), params(4)).unwrap();
    let b = s.add_request((0..3).collect(), params(4)).unwrap();
    s.schedule().unwrap();
    assert!(s.abort(a));
    assert!(!s.abort(a));
    s.check_invariants().unwrap();
    let f = s.update(&[StepOutput { request_id: a, token: 1 }, StepOutput { request_id: b, token: 1 }]).unwrap();
    assert!(f.is_empty());
    assert_eq!(s.num_running(), 1);
    s.check_invariants().unwrap();
    // The aborted request's KV may be half-written, so none of it is cached.
    assert_eq!(s.num_cached_blocks(), 0);
}

#[test]
fn randomized_workloads_keep_invariants_and_drain() {
    let mut rng: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = move |n: u64| {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng % n
    };
    for round in 0..40 {
        let block_size = [1, 2, 4, 16][round % 4];
        let max_running = 1 + next(16) as usize;
        let max_batch_tokens = max_running + next(64) as usize;
        let num_blocks = 8 + next(200) as usize;
        let mut c = cfg(block_size, num_blocks, max_running, max_batch_tokens);
        c.max_prefill_chunk = 1 + next(max_batch_tokens as u64) as usize;
        c.enable_prefix_cache = round % 5 != 0;
        let mut s = Scheduler::new(c).unwrap();
        let prefixes: Vec<Vec<u32>> = (0..3).map(|p| (0..next(40) as u32).map(|t| t + 100 * p).collect()).collect();
        let mut seeds = HashMap::new();
        let mut expected = HashMap::new();
        for i in 0..(1 + next(30)) {
            let mut prompt = prefixes[next(3) as usize].clone();
            prompt.extend((0..1 + next(30) as u32).map(|t| t * 7 + i as u32));
            let n = 1 + next(40) as usize;
            let stop = if next(3) == 0 { vec![1000 + next(5000) as u32] } else { vec![] };
            match s
                .add_request(prompt, SamplingParams { max_new_tokens: n, stop_token_ids: stop, ..Default::default() })
            {
                Ok(id) => {
                    seeds.insert(id, i as u32);
                    expected.insert(id, n);
                }
                Err(SchedError::PromptTooLong { .. }) => {}
                Err(e) => panic!("{e}"),
            }
        }
        let done = drive(&mut s, &seeds);
        assert_eq!(done.len(), expected.len());
        for (id, f) in &done {
            assert!(f.output_tokens.len() <= expected[id]);
            if f.reason == FinishReason::Length {
                assert_eq!(f.output_tokens.len(), expected[id]);
            }
        }
        assert_eq!(s.num_free_blocks() + s.num_cached_blocks(), s.config().num_blocks);
    }
}

#[test]
fn resumed_decode_can_span_several_tokens() {
    // A tight pool preempts mid-decode; the resumed request hits its own cached blocks and recomputes the
    // generated tokens after the last full block in one item, so `is_decode` does not imply one token.
    let mut s = Scheduler::new(cfg(4, 6, 2, 8)).unwrap();
    let mut seeds = HashMap::new();
    for i in 0..2u32 {
        let id = s.add_request(vec![10 * i, 10 * i + 1], params(25)).unwrap();
        seeds.insert(id, i);
    }
    let mut multi = Vec::new();
    while s.has_work() {
        let plan = s.schedule().unwrap();
        let items = plan.items();
        for it in items.iter().filter(|i| i.is_decode && i.input_ids.len() > 1) {
            multi.push((it.request_id, it.start, it.input_ids.len()));
        }
        let outputs: Vec<StepOutput> = items
            .iter()
            .filter(|i| i.sample)
            .map(|i| StepOutput {
                request_id: i.request_id,
                token: fake_token(seeds[&i.request_id], i.start + i.input_ids.len()),
            })
            .collect();
        s.update(&outputs).unwrap();
        s.check_invariants().unwrap();
    }
    assert!(s.stats().preemptions > 0);
    assert!(!multi.is_empty(), "expected a multi-token decode item after a cached resume");
}

#[test]
fn flat_plan_arrays_line_up() {
    let mut s = Scheduler::new(cfg(4, 64, 4, 64)).unwrap();
    let a = s.add_request((10..19).collect(), params(4)).unwrap(); // 9 tokens: 3 blocks, last one partial
    let sp = SamplingParams { temperature: 0.7, top_p: 0.9, top_k: 40, seed: Some(123), ..params(4) };
    let b = s.add_request(vec![7, 8], sp).unwrap();
    let p = s.schedule().unwrap();
    assert_eq!(p.block_size, 4);
    assert_eq!(p.request_ids, vec![a, b]);
    assert_eq!((p.starts.clone(), p.lens.clone(), p.kv_lens.clone()), (vec![0, 0], vec![9, 2], vec![9, 2]));
    assert_eq!(p.cu_seqlens, vec![0, 9, 11]);
    assert_eq!(p.token_ids, [(10..19).collect::<Vec<u32>>(), vec![7, 8]].concat());
    assert_eq!(p.positions, [(0..9).collect::<Vec<i32>>(), vec![0, 1]].concat());
    assert_eq!(p.max_blocks, 3);
    assert_eq!(p.block_tables.len(), 2 * 3);
    assert_eq!(&p.block_tables[3..], &[p.block_tables[3], -1, -1]);
    for row in 0..p.num_tokens {
        let item = p.cu_seqlens.iter().rposition(|&c| c as usize <= row).unwrap().min(p.num_items() - 1);
        let pos = p.positions[row] as usize;
        let block = p.block_tables[item * p.max_blocks + pos / 4];
        assert_eq!(p.slot_mapping[row], block * 4 + (pos % 4) as i32);
    }
    assert_eq!(p.sample_rows, vec![8, 10]);
    assert_eq!(p.sample_request_ids, vec![a, b]);
    assert_eq!(
        (p.temperature.clone(), p.top_p.clone(), p.top_k.clone()),
        (vec![0.0, 0.7], vec![1.0, 0.9], vec![0, 40])
    );
    assert_eq!(p.seed, vec![a, 123]);
    assert!(p.released.is_empty());
    assert_eq!(s.update_sampled(&[1]), Err(SchedError::WrongOutputCount { expected: 2, got: 1 }));
    assert!(s.update_sampled(&[1, 2]).unwrap().is_empty());
    assert_eq!(s.request(b).unwrap().output_tokens(), &[2]);
    assert!(matches!(
        s.add_request(vec![1], SamplingParams { top_p: 0.0, ..params(1) }),
        Err(SchedError::InvalidSampling(_))
    ));
}

#[test]
fn releases_report_finish_abort_and_preemption() {
    let rel = |p: &StepPlan| p.released.iter().map(|r| (r.request_id, r.reason)).collect::<Vec<_>>();
    // Finish: reported by the next schedule().
    let mut s = Scheduler::new(cfg(4, 64, 4, 64)).unwrap();
    let a = s.add_request(vec![1, 2], params(1)).unwrap();
    let b = s.add_request(vec![3, 4], params(3)).unwrap();
    s.schedule().unwrap();
    assert_eq!(s.update_sampled(&[5, 6]).unwrap().len(), 1);
    let p = s.schedule().unwrap();
    assert_eq!(rel(&p), vec![(a, ReleaseReason::Finished)]);
    // Abort of a running request is reported; abort of a waiting one is not (the runner never saw it).
    s.update_sampled(&[7]).unwrap();
    let c = s.add_request(vec![9], params(3)).unwrap();
    assert!(s.abort(b) && s.abort(c));
    assert_eq!(s.take_released(), vec![Released { request_id: b, reason: ReleaseReason::Aborted }]);
    assert!(s.take_released().is_empty());

    // Preemption: reported in the plan of the schedule() that made it.
    let mut s = Scheduler::new(cfg(4, 6, 2, 8)).unwrap();
    let seeds: HashMap<_, _> = (0..2u32).map(|i| (s.add_request(vec![i, i], params(25)).unwrap(), i)).collect();
    let mut saw = false;
    while s.has_work() && !saw {
        let p = s.schedule().unwrap();
        if p.released.iter().any(|r| r.reason == ReleaseReason::Preempted) {
            saw = true;
            assert_eq!(s.stats().preemptions, 1);
        }
        let toks: Vec<u32> = p.sample_request_ids.iter().map(|id| fake_token(seeds[id], 0)).collect();
        s.update_sampled(&toks).unwrap();
    }
    assert!(saw, "the tight pool should have preempted");
}

#[test]
fn state_slots_cap_admission_and_survive_preemption() {
    let workload: Vec<(Vec<u32>, usize)> =
        (0..12u32).map(|i| ((0..(20 + 7 * i)).map(|t| t * 31 + i).collect(), 10 + (i as usize % 5) * 6)).collect();

    let run = |num_blocks: usize, num_state_slots: usize| {
        let mut c = cfg(4, num_blocks, 8, 32);
        c.enable_prefix_cache = false;
        c.num_state_slots = num_state_slots;
        let mut s = Scheduler::new(c).unwrap();
        let mut seeds = HashMap::new();
        let mut order = Vec::new();
        for (i, (prompt, n)) in workload.iter().enumerate() {
            let id = s.add_request(prompt.clone(), params(*n)).unwrap();
            seeds.insert(id, i as u32);
            order.push(id);
        }
        let done = drive(&mut s, &seeds);
        let outputs: Vec<Vec<u32>> = order.iter().map(|id| done[id].output_tokens.clone()).collect();
        (outputs, s.stats().preemptions)
    };

    let (reference, _) = run(4096, 0);
    let (roomy, roomy_preempt) = run(4096, 3);
    let (tight, tight_preempt) = run(40, 6);
    assert_eq!(roomy_preempt, 0);
    assert!(tight_preempt > 0, "the tight pool should have forced preemptions");
    assert_eq!(roomy, reference);
    assert_eq!(tight, reference);
}

#[test]
fn state_slots_need_the_prefix_cache_off() {
    let mut c = cfg(4, 16, 4, 64);
    c.num_state_slots = 4;
    assert!(matches!(Scheduler::new(c.clone()), Err(SchedError::InvalidConfig(_))));
    c.enable_prefix_cache = false;
    assert!(Scheduler::new(c).is_ok());
}

#[test]
fn aborting_a_running_request_frees_its_state_slot() {
    let mut c = cfg(4, 64, 4, 64);
    c.enable_prefix_cache = false;
    c.num_state_slots = 1;
    let mut s = Scheduler::new(c).unwrap();
    let a = s.add_request(vec![1, 2, 3], params(4)).unwrap();
    let b = s.add_request(vec![4, 5, 6], params(4)).unwrap();

    // max_running is 4, but one slot admits one request.
    let p = s.schedule().unwrap();
    assert_eq!(p.request_ids, vec![a]);
    assert_eq!(p.state_slots, vec![0]);
    assert_eq!(s.num_waiting(), 1);
    assert_eq!(s.num_free_state_slots(), 0);
    assert!(s.abort(a));
    assert_eq!(s.num_free_state_slots(), 1);
    s.update_sampled(&[7]).unwrap();
    s.check_invariants().unwrap();

    let p = s.schedule().unwrap();
    assert_eq!(p.request_ids, vec![b]);
    assert_eq!(p.starts, vec![0]);
    assert_eq!(p.state_slots, vec![0]);
    assert_eq!(p.released, vec![Released { request_id: a, reason: ReleaseReason::Aborted }]);
}

/// The same workload with and without overlap gives the same outputs; stop tokens cost one discarded sample each.
fn overlap_matches(c: EngineConfig, workload: &[(Vec<u32>, SamplingParams)]) -> (Stats, Stats) {
    let run = |overlap: bool| {
        let mut c = c.clone();
        c.overlap = overlap;
        let mut s = Scheduler::new(c).unwrap();
        let mut seeds = HashMap::new();
        let mut order = Vec::new();
        for (i, (prompt, sp)) in workload.iter().enumerate() {
            let id = s.add_request(prompt.clone(), sp.clone()).unwrap();
            seeds.insert(id, i as u32);
            order.push(id);
        }
        let done = drive(&mut s, &seeds);
        let out: Vec<(Vec<u32>, FinishReason)> =
            order.iter().map(|id| (done[id].output_tokens.clone(), done[id].reason)).collect();
        (out, s.stats().clone())
    };
    let (reference, base) = run(false);
    let (overlapped, stats) = run(true);
    assert_eq!(overlapped, reference);
    assert_eq!((base.discarded_samples, base.overlapped_inputs), (0, 0));
    let decodes = reference.iter().map(|(out, _)| out.len().saturating_sub(1)).sum::<usize>() as u64;
    assert!(decodes == 0 || stats.overlapped_inputs > 0, "overlap never ran ahead");
    let stops = reference.iter().filter(|(_, r)| *r == FinishReason::Stop).count() as u64;
    assert!(stats.discarded_samples <= stops);
    (base, stats)
}

#[test]
fn overlap_runs_one_step_ahead_and_resolves_inputs_from_the_step_in_flight() {
    let mut c = cfg(4, 64, 4, 64);
    c.overlap = true;
    let mut s = Scheduler::new(c).unwrap();
    let a = s.add_request(vec![1, 2, 3], params(3)).unwrap();
    let p1 = s.schedule().unwrap();
    assert_eq!(p1.prev_sample_idx, vec![-1, -1, -1]);
    // Step 1 is in flight; step 2 decodes the token step 1 is sampling.
    let p2 = s.schedule().unwrap();
    assert_eq!(
        (p2.starts.clone(), p2.token_ids.clone(), p2.prev_sample_idx.clone()),
        (vec![3], vec![u32::MAX], vec![0])
    );
    assert_eq!(p2.num_pending_inputs(), 1);
    assert_eq!(s.num_in_flight(), 2);
    assert_eq!(s.schedule(), Err(SchedError::StepInFlight));
    s.check_invariants().unwrap();
    assert!(s.update_sampled(&[10]).unwrap().is_empty());
    assert_eq!(s.request(a).unwrap().output_tokens(), &[10]);
    let p3 = s.schedule().unwrap();
    assert_eq!((p3.starts.clone(), p3.prev_sample_idx.clone()), (vec![4], vec![0]));
    assert!(s.update_sampled(&[11]).unwrap().is_empty());
    // The third token is the last (max_new_tokens = 3): nothing is run ahead of it.
    let p4 = s.schedule().unwrap();
    assert!(p4.is_empty());
    let f = s.update_sampled(&[12]).unwrap();
    assert_eq!(f[0].output_tokens, vec![10, 11, 12]);
    assert_eq!(s.stats().discarded_samples, 0);
    assert!(!s.has_work());
    s.check_invariants().unwrap();

    // Without overlap a second schedule() is refused.
    let mut s = Scheduler::new(cfg(4, 64, 4, 64)).unwrap();
    s.add_request(vec![1, 2, 3], params(3)).unwrap();
    s.schedule().unwrap();
    assert_eq!(s.schedule(), Err(SchedError::StepInFlight));
}

#[test]
fn overlap_drops_the_token_sampled_after_a_stop() {
    let mut c = cfg(4, 64, 4, 64);
    c.overlap = true;
    let mut s = Scheduler::new(c).unwrap();
    let sp = SamplingParams { max_new_tokens: 10, stop_token_ids: vec![99], ..Default::default() };
    let a = s.add_request(vec![1, 2, 3], sp).unwrap();
    let b = s.add_request(vec![4, 5], params(10)).unwrap();
    s.schedule().unwrap();
    s.schedule().unwrap();
    // a stops in step 1; step 2 already computes its next token.
    let f = s.update_sampled(&[99, 7]).unwrap();
    assert_eq!((f[0].request_id, f[0].reason, f[0].output_tokens.clone()), (a, FinishReason::Stop, vec![99]));
    assert_eq!(s.stats().discarded_samples, 1);
    s.check_invariants().unwrap();
    let p3 = s.schedule().unwrap();
    assert_eq!(p3.request_ids, vec![b]);
    assert_eq!(p3.released, vec![Released { request_id: a, reason: ReleaseReason::Finished }]);
    // Step 2's output for a is ignored; one per sample row is still expected.
    assert!(s.update_sampled(&[55, 8]).unwrap().is_empty());
    assert_eq!(s.request(b).unwrap().output_tokens(), &[7, 8]);
    s.check_invariants().unwrap();
    // Abort with two steps in flight: both outputs are ignored.
    s.schedule().unwrap();
    assert!(s.abort(b));
    s.check_invariants().unwrap();
    assert!(s.update_sampled(&[1]).unwrap().is_empty());
    assert!(s.update_sampled(&[2]).unwrap().is_empty());
    assert!(!s.has_work());
    assert_eq!(s.num_free_blocks() + s.num_cached_blocks(), 64);
}

#[test]
fn overlap_matches_the_plain_schedule_under_preemption_and_state_slots() {
    let workload: Vec<(Vec<u32>, SamplingParams)> = (0..12u32)
        .map(|i| {
            let stop = if i % 3 == 0 { vec![fake_token(i, 20 + 7 * i as usize + 4)] } else { vec![] };
            let sp = SamplingParams {
                max_new_tokens: 10 + (i as usize % 5) * 6,
                stop_token_ids: stop,
                ..Default::default()
            };
            ((0..(20 + 7 * i)).map(|t| t * 31 + i).collect(), sp)
        })
        .collect();
    let (_, roomy) = overlap_matches(cfg(4, 4096, 8, 32), &workload);
    assert_eq!(roomy.discarded_samples, 4, "each of the 4 stopping requests costs one sample");
    let (_, tight) = overlap_matches(cfg(4, 40, 8, 32), &workload);
    assert!(tight.preemptions > 0);
    let mut c = cfg(4, 40, 8, 32);
    c.enable_prefix_cache = false;
    c.num_state_slots = 5;
    let (_, slots) = overlap_matches(c, &workload);
    assert!(slots.preemptions > 0);
}

#[test]
fn overlap_randomized_workloads_match_the_plain_schedule() {
    let mut rng: u64 = 0x2545_F491_4F6C_DD1D;
    let mut next = move |n: u64| {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng % n
    };
    for round in 0..60 {
        let block_size = [1, 2, 4, 16][round % 4];
        let max_running = 1 + next(16) as usize;
        let max_batch_tokens = max_running + next(64) as usize;
        let num_blocks = 8 + next(200) as usize;
        let mut c = cfg(block_size, num_blocks, max_running, max_batch_tokens);
        c.max_prefill_chunk = 1 + next(max_batch_tokens as u64) as usize;
        c.max_seq_len = 16 + next(120) as usize;
        match round % 3 {
            0 => c.enable_prefix_cache = false,
            1 => {
                c.enable_prefix_cache = false;
                c.num_state_slots = 1 + next(max_running as u64) as usize;
            }
            _ => {}
        }
        let prefixes: Vec<Vec<u32>> = (0..3).map(|p| (0..next(30) as u32).map(|t| t + 100 * p).collect()).collect();
        let seq_cap = c.max_seq_len.min(c.num_blocks * c.block_size);
        let mut workload = Vec::new();
        for i in 0..(1 + next(30)) {
            let mut prompt = prefixes[next(3) as usize].clone();
            prompt.extend((0..1 + next(30) as u32).map(|t| t * 7 + i as u32));
            if prompt.len() >= seq_cap {
                continue;
            }
            // Stop on a token this request will actually produce, so stops really happen.
            let at = prompt.len() + 1 + next(20) as usize;
            let stop = if next(2) == 0 { vec![fake_token(workload.len() as u32, at)] } else { vec![] };
            let sp =
                SamplingParams { max_new_tokens: 1 + next(40) as usize, stop_token_ids: stop, ..Default::default() };
            workload.push((prompt, sp));
        }
        overlap_matches(c, &workload);
    }
}
