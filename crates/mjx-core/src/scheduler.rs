// Copyright 2026 the MilliJoule authors
// SPDX-License-Identifier: Apache-2.0
//! Continuous-batching step scheduler: decodes and chunked prefills share one token budget per step.
//!
//! The contract with the model runner is one item per request per step: run the item's tokens at positions
//! `start ..`, write their KV through its block table, and sample a token when `sample` is set.
//! `schedule()` and `update()` (or `update_sampled()`) alternate; one step is in flight at a time. With
//! `EngineConfig::overlap`, `schedule()` may run once ahead (two steps in flight), and `update()` always completes
//! the oldest.
//! A `StepPlan` is flat arrays, so a runner hands it to the device without per-item work. Before running the
//! items, the runner applies `released`: requests whose per-request state (KV is the scheduler's; recurrent state,
//! caches, RNG are the runner's) it must drop. Recurrent-state slots are handed out by the scheduler
//! (`state_slots`), but their contents are the runner's.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

use crate::config::EngineConfig;
use crate::kv::{BlockAllocator, BlockId};
use crate::radix::RadixCache;
use crate::request::{FinishReason, Request, RequestId, SamplingParams, Status};

/// One item of a plan as owned values. Built by `StepPlan::items()` for tests and debugging; runners read the
/// flat arrays instead.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StepItem {
    pub request_id: RequestId,
    /// Position of `input_ids[0]` in the request.
    pub start: usize,
    pub input_ids: Vec<u32>,
    /// KV blocks covering positions `0 .. start + input_ids.len()`.
    pub block_table: Vec<BlockId>,
    /// The last input reaches the end of the request's tokens: sample one token there.
    pub sample: bool,
    /// Every input is a generated token (the request is past its prompt). Usually one token, but a preempted
    /// request resuming from the prefix cache can recompute several generated tokens in one item.
    pub is_decode: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReleaseReason {
    Finished,
    Aborted,
    /// Sent back to the waiting queue; it will be recomputed from its start (mostly a prefix-cache hit).
    Preempted,
}

impl ReleaseReason {
    pub fn as_str(self) -> &'static str {
        match self {
            ReleaseReason::Finished => "finished",
            ReleaseReason::Aborted => "aborted",
            ReleaseReason::Preempted => "preempted",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Released {
    pub request_id: RequestId,
    pub reason: ReleaseReason,
}

/// One step as flat arrays. Items are in plan order; per-token arrays concatenate the items' inputs in that order.
#[derive(Clone, Debug, PartialEq)]
pub struct StepPlan {
    /// Tokens per KV block, so the runner can index `block_tables` without being told separately.
    pub block_size: usize,
    // Per item, length `num_items()`.
    pub request_ids: Vec<RequestId>,
    /// Position of the item's first input in its request.
    pub starts: Vec<i32>,
    pub lens: Vec<i32>,
    /// `start + len`: the request's KV length after this step.
    pub kv_lens: Vec<i32>,
    /// Sample one token at the item's last input.
    pub sample: Vec<bool>,
    /// `start >= prompt_len`. Does not imply `len == 1` (see `StepItem::is_decode`).
    pub is_decode: Vec<bool>,
    /// Length `num_items() + 1`: item `i` owns token rows `cu_seqlens[i] .. cu_seqlens[i + 1]`.
    pub cu_seqlens: Vec<i32>,
    /// `[num_items, max_blocks]` row-major, padded with -1. Position `p` of item `i` lives in block
    /// `block_tables[i * max_blocks + p / block_size]`, slot `p % block_size`.
    pub block_tables: Vec<i32>,
    pub max_blocks: usize,
    pub temperature: Vec<f32>,
    pub top_p: Vec<f32>,
    pub top_k: Vec<i32>,
    pub seed: Vec<u64>,
    /// Recurrent-state slot of each item, -1 when the model has none. A slot is the request's from admission to
    /// release. An item with `start == 0` begins a fresh state: the runner zeroes the slot before running it.
    pub state_slots: Vec<i32>,
    // Per token, length `num_tokens`.
    /// `u32::MAX` where the input is not known yet (see `prev_sample_idx`).
    pub token_ids: Vec<u32>,
    /// -1 when `token_ids` holds the input. Otherwise the input is the token that the previous plan (the one still
    /// in flight when this one was scheduled) samples at index `k` of its `sample_rows`, i.e. element `k` of that
    /// step's sampled-token output. Only with `EngineConfig::overlap`, and only for one-token decode items.
    pub prev_sample_idx: Vec<i32>,
    /// Absolute position of each token in its request.
    pub positions: Vec<i32>,
    /// KV write index of each token in a flat `[num_blocks * block_size, ..]` pool.
    pub slot_mapping: Vec<i32>,
    // Per sampled item, in item order.
    /// Token row (into the per-token arrays) whose logits are sampled.
    pub sample_rows: Vec<i32>,
    pub sample_request_ids: Vec<RequestId>,
    /// Requests the runner must drop state for before running this step's items: finishes and aborts since the
    /// previous `schedule()`, and preemptions made by this one. A preempted request can also have an item in this
    /// same plan (it was readmitted); it then recomputes from that item's start.
    pub released: Vec<Released>,
    pub num_tokens: usize,
}

impl StepPlan {
    fn new(block_size: usize) -> Self {
        Self {
            block_size,
            request_ids: Vec::new(),
            starts: Vec::new(),
            lens: Vec::new(),
            kv_lens: Vec::new(),
            sample: Vec::new(),
            is_decode: Vec::new(),
            cu_seqlens: vec![0],
            block_tables: Vec::new(),
            max_blocks: 0,
            temperature: Vec::new(),
            top_p: Vec::new(),
            top_k: Vec::new(),
            seed: Vec::new(),
            state_slots: Vec::new(),
            token_ids: Vec::new(),
            prev_sample_idx: Vec::new(),
            positions: Vec::new(),
            slot_mapping: Vec::new(),
            sample_rows: Vec::new(),
            sample_request_ids: Vec::new(),
            released: Vec::new(),
            num_tokens: 0,
        }
    }

    /// True when there are no items to run (`released` may still be non-empty).
    pub fn is_empty(&self) -> bool {
        self.request_ids.is_empty()
    }

    pub fn num_items(&self) -> usize {
        self.request_ids.len()
    }

    pub fn num_sample(&self) -> usize {
        self.sample_rows.len()
    }

    pub fn num_decode(&self) -> usize {
        self.is_decode.iter().filter(|&&d| d).count()
    }

    pub fn num_prefill(&self) -> usize {
        self.num_items() - self.num_decode()
    }

    /// Inputs taken from the previous step's sampled tokens (overlap scheduling).
    pub fn num_pending_inputs(&self) -> usize {
        self.prev_sample_idx.iter().filter(|&&k| k >= 0).count()
    }

    /// The plan as owned items, for tests and debugging.
    pub fn items(&self) -> Vec<StepItem> {
        (0..self.num_items())
            .map(|i| {
                let rows = self.cu_seqlens[i] as usize..self.cu_seqlens[i + 1] as usize;
                let table = &self.block_tables[i * self.max_blocks..(i + 1) * self.max_blocks];
                StepItem {
                    request_id: self.request_ids[i],
                    start: self.starts[i] as usize,
                    input_ids: self.token_ids[rows].to_vec(),
                    block_table: table.iter().take_while(|&&b| b >= 0).map(|&b| b as BlockId).collect(),
                    sample: self.sample[i],
                    is_decode: self.is_decode[i],
                }
            })
            .collect()
    }
}

/// Collects items while a plan is built; block tables are padded once their widest row is known.
struct PlanBuilder {
    plan: StepPlan,
    tables: Vec<BlockId>,
    table_lens: Vec<usize>,
    // Each item's request and `sample` flag, in plan order.
    items: Vec<(RequestId, bool)>,
}

impl PlanBuilder {
    fn new(block_size: usize) -> Self {
        Self { plan: StepPlan::new(block_size), tables: Vec::new(), table_lens: Vec::new(), items: Vec::new() }
    }

    fn finish(mut self) -> (StepPlan, Vec<(RequestId, bool)>) {
        let width = self.table_lens.iter().copied().max().unwrap_or(0);
        let mut padded = vec![-1i32; self.table_lens.len() * width];
        let mut off = 0;
        for (i, &n) in self.table_lens.iter().enumerate() {
            for (dst, &b) in padded[i * width..i * width + n].iter_mut().zip(&self.tables[off..off + n]) {
                *dst = b as i32;
            }
            off += n;
        }
        self.plan.block_tables = padded;
        self.plan.max_blocks = width;
        (self.plan, self.items)
    }
}

/// A step handed to the runner and not yet completed by `update()`.
struct InFlight {
    // Items in plan order, with their `sample` flag.
    items: Vec<(RequestId, bool)>,
    // Sampled items in sample-row order (what `update_sampled` lines up with).
    sampled: Vec<RequestId>,
    // Requests whose output from this step is ignored: aborted, or finished by an earlier step (overlap).
    dropped: HashSet<RequestId>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StepOutput {
    pub request_id: RequestId,
    pub token: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Finished {
    pub request_id: RequestId,
    pub reason: FinishReason,
    pub output_tokens: Vec<u32>,
    pub cached_prompt_tokens: usize,
    pub num_preemptions: u32,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    pub steps: u64,
    pub scheduled_tokens: u64,
    pub prompt_tokens: u64,
    pub cached_prompt_tokens: u64,
    pub preemptions: u64,
    pub evicted_blocks: u64,
    pub finished: u64,
    /// Tokens sampled by an overlapped step for a request that had already stopped (the stop token is seen one step
    /// late), and thrown away. Wasted work of overlap scheduling.
    pub discarded_samples: u64,
    /// Decode inputs scheduled before their token was known (taken from the step in flight).
    pub overlapped_inputs: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SchedError {
    EmptyPrompt,
    PromptTooLong { len: usize, max: usize },
    ZeroMaxNewTokens,
    InvalidSampling(String),
    InvalidConfig(String),
    StepInFlight,
    NoStepInFlight,
    UnexpectedOutput(RequestId),
    DuplicateOutput(RequestId),
    MissingOutput(RequestId),
    WrongOutputCount { expected: usize, got: usize },
}

impl fmt::Display for SchedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SchedError::EmptyPrompt => write!(f, "prompt is empty"),
            SchedError::PromptTooLong { len, max } => {
                write!(f, "prompt has {len} tokens; the limit is {} (max_seq_len or KV capacity)", max - 1)
            }
            SchedError::ZeroMaxNewTokens => write!(f, "max_new_tokens must be >= 1"),
            SchedError::InvalidSampling(msg) => write!(f, "invalid sampling params: {msg}"),
            SchedError::WrongOutputCount { expected, got } => {
                write!(f, "expected {expected} sampled tokens (one per sample row), got {got}")
            }
            SchedError::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
            SchedError::StepInFlight => {
                write!(f, "no more steps may be in flight (one, or two with overlap); call update() first")
            }
            SchedError::NoStepInFlight => write!(f, "no step is in flight; call schedule() first"),
            SchedError::UnexpectedOutput(id) => write!(f, "request {id} was not scheduled to sample this step"),
            SchedError::DuplicateOutput(id) => write!(f, "request {id} has two outputs this step"),
            SchedError::MissingOutput(id) => write!(f, "request {id} was scheduled to sample but has no output"),
        }
    }
}

impl std::error::Error for SchedError {}

pub struct Scheduler {
    cfg: EngineConfig,
    // Longest a request may grow: max_seq_len, or what the whole KV pool can hold.
    seq_cap: usize,
    alloc: BlockAllocator,
    cache: RadixCache,
    requests: HashMap<RequestId, Request>,
    waiting: VecDeque<RequestId>,
    // Admission order; preemption takes the newest first.
    running: Vec<RequestId>,
    // Oldest first; at most one, or two with overlap.
    in_flight: VecDeque<InFlight>,
    // Free recurrent-state slots; popped from the end, so the lowest slot goes first.
    free_state_slots: Vec<u32>,
    // Releases not yet handed to the runner.
    released: Vec<Released>,
    next_id: RequestId,
    stats: Stats,
}

impl Scheduler {
    pub fn new(cfg: EngineConfig) -> Result<Self, SchedError> {
        cfg.validate().map_err(SchedError::InvalidConfig)?;
        let seq_cap = cfg.max_seq_len.min(cfg.num_blocks * cfg.block_size);
        Ok(Self {
            seq_cap,
            alloc: BlockAllocator::new(cfg.num_blocks),
            cache: RadixCache::new(cfg.block_size),
            requests: HashMap::new(),
            waiting: VecDeque::new(),
            running: Vec::new(),
            in_flight: VecDeque::new(),
            free_state_slots: (0..cfg.num_state_slots as u32).rev().collect(),
            released: Vec::new(),
            next_id: 0,
            stats: Stats::default(),
            cfg,
        })
    }

    pub fn config(&self) -> &EngineConfig {
        &self.cfg
    }

    pub fn stats(&self) -> &Stats {
        &self.stats
    }

    pub fn num_waiting(&self) -> usize {
        self.waiting.len()
    }

    pub fn num_running(&self) -> usize {
        self.running.len()
    }

    pub fn num_free_blocks(&self) -> usize {
        self.alloc.num_free()
    }

    pub fn num_free_state_slots(&self) -> usize {
        self.free_state_slots.len()
    }

    pub fn num_cached_blocks(&self) -> usize {
        self.cache.num_cached_blocks()
    }

    /// Requests remain, or a step still awaits `update()`.
    pub fn has_work(&self) -> bool {
        !self.requests.is_empty() || !self.in_flight.is_empty()
    }

    /// Steps scheduled and not yet completed by `update()`.
    pub fn num_in_flight(&self) -> usize {
        self.in_flight.len()
    }

    pub fn request(&self, id: RequestId) -> Option<&Request> {
        self.requests.get(&id)
    }

    pub fn add_request(&mut self, prompt: Vec<u32>, params: SamplingParams) -> Result<RequestId, SchedError> {
        if prompt.is_empty() {
            return Err(SchedError::EmptyPrompt);
        }
        if prompt.len() >= self.seq_cap {
            return Err(SchedError::PromptTooLong { len: prompt.len(), max: self.seq_cap });
        }
        if params.max_new_tokens == 0 {
            return Err(SchedError::ZeroMaxNewTokens);
        }
        params.validate().map_err(SchedError::InvalidSampling)?;
        let id = self.next_id;
        self.next_id += 1;
        self.requests.insert(id, Request::new(id, prompt, params));
        self.waiting.push_back(id);
        Ok(id)
    }

    /// Plans the next step. A plan with no items means there is nothing to run; its `released` still counts.
    pub fn schedule(&mut self) -> Result<StepPlan, SchedError> {
        let depth = if self.cfg.overlap { 2 } else { 1 };
        if self.in_flight.len() >= depth {
            return Err(SchedError::StepInFlight);
        }
        let mut plan = PlanBuilder::new(self.cfg.block_size);
        // Where each request's pending token sits in the output of the step in flight (overlap only).
        let prev: HashMap<RequestId, i32> = self
            .in_flight
            .back()
            .map(|st| st.sampled.iter().enumerate().map(|(k, &id)| (id, k as i32)).collect())
            .unwrap_or_default();
        let mut budget = self.cfg.max_batch_tokens;
        let mut scheduled: HashSet<RequestId> = HashSet::new();

        // Running requests: decodes first (they hold the latency limit), then prefill chunks, each in admission order.
        for decode_pass in [true, false] {
            for id in self.running.clone() {
                if budget == 0 {
                    break;
                }
                let Some(r) = self.requests.get(&id) else { continue };
                if r.status != Status::Running || scheduled.contains(&id) {
                    continue;
                }
                if (r.num_uncomputed() == 1) != decode_pass {
                    continue;
                }
                // Its pending token is known to be its last (length limits): nothing to run ahead.
                if r.num_pending > 0 && ends_by_length(r, self.seq_cap) {
                    continue;
                }
                let pending = r.num_pending > 0;
                let len = r.num_uncomputed().min(budget).min(self.cfg.prefill_chunk_at(r.num_computed));
                let end = r.num_computed + len;
                if self.reserve(id, end, Some(&scheduled)) {
                    budget -= len;
                    scheduled.insert(id);
                    self.push_item(&mut plan, id, len, &prev);
                } else if !pending {
                    self.preempt(id);
                }
                // A request with a pending token is not preempted (its token would land after a restart); it waits
                // for the step in flight instead.
            }
        }

        // Waiting requests, first come first served. Admission never preempts.
        while budget > 0 && self.running.len() < self.cfg.max_running {
            let Some(&id) = self.waiting.front() else { break };
            if self.cfg.num_state_slots > 0 && self.free_state_slots.is_empty() {
                break;
            }
            self.match_prefix(id);
            let r = &self.requests[&id];
            let len = r.num_uncomputed().min(budget).min(self.cfg.prefill_chunk_at(r.num_computed));
            let end = r.num_computed + len;
            if !self.reserve(id, end, None) {
                self.release_blocks(id);
                break;
            }
            self.waiting.pop_front();
            if self.cfg.num_state_slots > 0 {
                let slot = self.free_state_slots.pop().expect("checked above");
                self.requests.get_mut(&id).expect("request exists").state_slot = Some(slot);
            }
            self.mark_admitted(id);
            self.running.push(id);
            budget -= len;
            scheduled.insert(id);
            self.push_item(&mut plan, id, len, &prev);
        }

        let (mut plan, items) = plan.finish();
        plan.released = std::mem::take(&mut self.released);
        if !plan.is_empty() {
            self.in_flight.push_back(InFlight {
                items,
                sampled: plan.sample_request_ids.clone(),
                dropped: HashSet::new(),
            });
            self.stats.steps += 1;
            self.stats.scheduled_tokens += plan.num_tokens as u64;
            self.stats.overlapped_inputs += plan.num_pending_inputs() as u64;
        }
        Ok(plan)
    }

    /// Releases not yet handed out by `schedule()`. Call after the last `update()`, when no further plan will
    /// carry them.
    pub fn take_released(&mut self) -> Vec<Released> {
        std::mem::take(&mut self.released)
    }

    /// `update()` for a runner that returns one token per sample row, in `sample_rows` order.
    pub fn update_sampled(&mut self, tokens: &[u32]) -> Result<Vec<Finished>, SchedError> {
        let Some(step) = self.in_flight.front() else {
            return Err(SchedError::NoStepInFlight);
        };
        if tokens.len() != step.sampled.len() {
            return Err(SchedError::WrongOutputCount { expected: step.sampled.len(), got: tokens.len() });
        }
        let outputs: Vec<StepOutput> =
            step.sampled.iter().zip(tokens).map(|(&request_id, &token)| StepOutput { request_id, token }).collect();
        self.update(&outputs)
    }

    /// Applies the sampled tokens of the oldest step in flight. Returns the requests that finished.
    /// The outputs are checked before anything changes, so an error leaves the step in flight.
    pub fn update(&mut self, outputs: &[StepOutput]) -> Result<Vec<Finished>, SchedError> {
        let Some(step) = self.in_flight.front() else {
            return Err(SchedError::NoStepInFlight);
        };
        Self::validate_outputs(step, outputs)?;
        let step = self.in_flight.pop_front().expect("checked above");
        let tokens: HashMap<RequestId, u32> =
            outputs.iter().filter(|o| !step.dropped.contains(&o.request_id)).map(|o| (o.request_id, o.token)).collect();

        let mut finished = Vec::new();
        for (id, sample) in step.items {
            if self.cfg.enable_prefix_cache {
                self.cache_prompt(id);
            }
            if !sample {
                continue;
            }
            let reason = {
                let r = self.requests.get_mut(&id).expect("in-flight request exists");
                r.tokens.push(tokens[&id]);
                r.num_pending -= 1;
                finish_reason(r, self.seq_cap)
            };
            if let Some(reason) = reason {
                finished.push(self.finish(id, reason));
            }
        }
        Ok(finished)
    }

    /// Drops a request. Returns false for an unknown or already finished id.
    pub fn abort(&mut self, id: RequestId) -> bool {
        let Some(mut r) = self.requests.remove(&id) else {
            return false;
        };
        self.waiting.retain(|&x| x != id);
        self.running.retain(|&x| x != id);
        // A waiting request holds no runner state: it was never run, or its preemption was already reported.
        if r.status == Status::Running {
            self.released.push(Released { request_id: id, reason: ReleaseReason::Aborted });
        }
        // A step in flight may still be writing this request's KV: free it, but do not cache it.
        self.drop_from_in_flight(id);
        for b in r.blocks.drain(..) {
            self.alloc.decref(b);
        }
        // Reusable at the next schedule(); the step in flight completes before the runner runs that plan.
        if let Some(slot) = r.state_slot.take() {
            self.free_state_slots.push(slot);
        }
        true
    }

    /// Checks block reference counts against the requests and the cache. For tests and debugging.
    pub fn check_invariants(&self) -> Result<(), String> {
        let mut expected = vec![0u32; self.alloc.num_blocks()];
        for r in self.requests.values() {
            for &b in &r.blocks {
                expected[b as usize] += 1;
            }
            match r.status {
                Status::Waiting if !r.blocks.is_empty() => {
                    return Err(format!("waiting request {} holds {} blocks", r.id, r.blocks.len()));
                }
                Status::Running if r.blocks.len() < self.cfg.blocks_for(r.num_computed) => {
                    return Err(format!("request {} has too few blocks for {} tokens", r.id, r.num_computed));
                }
                _ => {}
            }
        }
        for b in self.cache.blocks() {
            expected[b as usize] += 1;
        }
        for (b, &want) in expected.iter().enumerate() {
            let got = self.alloc.refcount(b as BlockId);
            if got != want {
                return Err(format!("block {b}: refcount {got}, expected {want}"));
            }
        }
        let free = expected.iter().filter(|&&c| c == 0).count();
        if free != self.alloc.num_free() {
            return Err(format!("{} blocks unreferenced but {} on the free list", free, self.alloc.num_free()));
        }
        if self.waiting.len() + self.running.len() != self.requests.len() {
            return Err("waiting + running != requests".into());
        }
        let mut slot_used = vec![false; self.cfg.num_state_slots];
        for &s in &self.free_state_slots {
            if std::mem::replace(&mut slot_used[s as usize], true) {
                return Err(format!("state slot {s} is free twice"));
            }
        }
        for r in self.requests.values() {
            let want = self.cfg.num_state_slots > 0 && r.status == Status::Running;
            match r.state_slot {
                Some(s) if !want => return Err(format!("request {} holds state slot {s} while not running", r.id)),
                None if want => return Err(format!("running request {} has no state slot", r.id)),
                Some(s) if std::mem::replace(&mut slot_used[s as usize], true) => {
                    return Err(format!("state slot {s} is held twice"));
                }
                _ => {}
            }
        }
        if let Some(s) = slot_used.iter().position(|&u| !u) {
            return Err(format!("state slot {s} is neither free nor held"));
        }
        let mut pending: HashMap<RequestId, usize> = HashMap::new();
        for step in &self.in_flight {
            for &(id, sample) in &step.items {
                if !self.requests.contains_key(&id) {
                    return Err(format!("step in flight has an item for unknown request {id}"));
                }
                *pending.entry(id).or_default() += sample as usize;
            }
        }
        for r in self.requests.values() {
            let want = pending.get(&r.id).copied().unwrap_or(0);
            if r.num_pending != want {
                return Err(format!(
                    "request {} has {} pending tokens, steps in flight sample {want}",
                    r.id, r.num_pending
                ));
            }
            if r.num_computed > r.tokens.len() + r.num_pending {
                return Err(format!("request {} computed past its tokens", r.id));
            }
        }
        Ok(())
    }

    /// Removes `id` from every step in flight; its outputs there are ignored.
    fn drop_from_in_flight(&mut self, id: RequestId) -> usize {
        let mut sampled = 0;
        for step in &mut self.in_flight {
            let before = step.items.len();
            step.items.retain(|&(x, s)| {
                sampled += (x == id && s) as usize;
                x != id
            });
            if step.items.len() != before {
                step.dropped.insert(id);
            }
        }
        sampled
    }

    fn validate_outputs(step: &InFlight, outputs: &[StepOutput]) -> Result<(), SchedError> {
        let sample: HashMap<RequestId, bool> = step.items.iter().copied().collect();
        let mut seen = HashSet::new();
        for o in outputs {
            if step.dropped.contains(&o.request_id) {
                continue;
            }
            if sample.get(&o.request_id) != Some(&true) {
                return Err(SchedError::UnexpectedOutput(o.request_id));
            }
            if !seen.insert(o.request_id) {
                return Err(SchedError::DuplicateOutput(o.request_id));
            }
        }
        match step.items.iter().find(|&&(id, s)| s && !seen.contains(&id)) {
            Some(&(id, _)) => Err(SchedError::MissingOutput(id)),
            None => Ok(()),
        }
    }

    /// Makes `id` hold blocks for its first `tokens` positions. On a shortage it evicts unused cache blocks,
    /// then, if `protect` is given, preempts the newest running request that is not in it.
    fn reserve(&mut self, id: RequestId, tokens: usize, protect: Option<&HashSet<RequestId>>) -> bool {
        let need = self.cfg.blocks_for(tokens).saturating_sub(self.requests[&id].blocks.len());
        if need == 0 {
            return true;
        }
        loop {
            let free = self.alloc.num_free();
            if free >= need {
                let blocks = self.alloc.allocate(need).expect("checked free count");
                self.requests.get_mut(&id).expect("request exists").blocks.extend(blocks);
                return true;
            }
            if self.cfg.enable_prefix_cache {
                let evicted = self.cache.evict(need - free, &mut self.alloc);
                self.stats.evicted_blocks += evicted as u64;
                if evicted > 0 {
                    continue;
                }
            }
            let Some(protect) = protect else { return false };
            // A request with a pending token is never a victim (see `schedule`).
            let victim = self
                .running
                .iter()
                .rev()
                .copied()
                .find(|v| *v != id && !protect.contains(v) && self.requests[v].num_pending == 0);
            match victim {
                Some(v) => self.preempt(v),
                None => return false,
            }
        }
    }

    /// Sends a running request back to the front of the queue. Its finished blocks go to the prefix cache first,
    /// so the recompute is mostly a cache hit.
    fn preempt(&mut self, id: RequestId) {
        let r = self.requests.get_mut(&id).expect("request exists");
        if self.cfg.enable_prefix_cache {
            let full = r.num_computed / self.cfg.block_size;
            self.cache.insert(&r.tokens[..full * self.cfg.block_size], &r.blocks[..full], &mut self.alloc);
        }
        for b in r.blocks.drain(..) {
            self.alloc.decref(b);
        }
        r.num_computed = 0;
        if let Some(slot) = r.state_slot.take() {
            self.free_state_slots.push(slot);
        }
        r.status = Status::Waiting;
        r.num_preemptions += 1;
        r.prompt_cached = false;
        self.running.retain(|&x| x != id);
        self.waiting.push_front(id);
        self.released.push(Released { request_id: id, reason: ReleaseReason::Preempted });
        self.stats.preemptions += 1;
    }

    fn match_prefix(&mut self, id: RequestId) {
        if !self.cfg.enable_prefix_cache {
            return;
        }
        let r = self.requests.get_mut(&id).expect("request exists");
        debug_assert!(r.blocks.is_empty() && r.num_computed == 0);
        // Leave at least one token to compute, so the step produces logits.
        let max_blocks = (r.tokens.len() - 1) / self.cfg.block_size;
        let hit = self.cache.match_prefix(&r.tokens, max_blocks, &mut self.alloc);
        r.num_computed = hit.len() * self.cfg.block_size;
        r.blocks = hit;
    }

    fn release_blocks(&mut self, id: RequestId) {
        let r = self.requests.get_mut(&id).expect("request exists");
        for b in r.blocks.drain(..) {
            self.alloc.decref(b);
        }
        r.num_computed = 0;
    }

    fn mark_admitted(&mut self, id: RequestId) {
        let r = self.requests.get_mut(&id).expect("request exists");
        r.status = Status::Running;
        if !r.admitted_once {
            r.admitted_once = true;
            r.cached_prompt_tokens = r.num_computed;
            self.stats.prompt_tokens += r.prompt_len as u64;
            self.stats.cached_prompt_tokens += r.num_computed as u64;
        }
    }

    fn push_item(&mut self, b: &mut PlanBuilder, id: RequestId, len: usize, prev: &HashMap<RequestId, i32>) {
        let bs = self.cfg.block_size;
        let r = self.requests.get_mut(&id).expect("request exists");
        let start = r.num_computed;
        let end = start + len;
        let known = r.tokens.len();
        let sample = end == known + r.num_pending;
        let p = &mut b.plan;
        p.request_ids.push(id);
        p.starts.push(start as i32);
        p.lens.push(len as i32);
        p.kv_lens.push(end as i32);
        p.sample.push(sample);
        p.is_decode.push(start >= r.prompt_len);
        p.temperature.push(r.params.temperature);
        p.top_p.push(r.params.top_p);
        p.top_k.push(r.params.top_k as i32);
        p.seed.push(r.params.seed.unwrap_or(id));
        p.state_slots.push(r.state_slot.map_or(-1, |s| s as i32));
        if end <= known {
            p.token_ids.extend_from_slice(&r.tokens[start..end]);
            p.prev_sample_idx.resize(p.prev_sample_idx.len() + len, -1);
        } else {
            // Overlap: the input is the token the step in flight samples for this request.
            debug_assert!(start == known && len == 1 && r.num_pending == 1);
            p.token_ids.push(u32::MAX);
            p.prev_sample_idx.push(prev[&id]);
        }
        for pos in start..end {
            p.positions.push(pos as i32);
            p.slot_mapping.push((r.blocks[pos / bs] as usize * bs + pos % bs) as i32);
        }
        p.num_tokens += len;
        p.cu_seqlens.push(p.num_tokens as i32);
        if sample {
            p.sample_rows.push(p.num_tokens as i32 - 1);
            p.sample_request_ids.push(id);
        }
        // TODO(perf): send block-table deltas once the runner keeps persistent tables.
        b.tables.extend_from_slice(&r.blocks);
        b.table_lens.push(r.blocks.len());
        r.num_computed = end;
        r.num_pending += sample as usize;
        b.items.push((id, sample));
    }

    /// Shares a finished prompt's full blocks with later requests as soon as its prefill completes.
    fn cache_prompt(&mut self, id: RequestId) {
        let Some(r) = self.requests.get_mut(&id) else { return };
        if r.prompt_cached || r.num_computed < r.prompt_len {
            return;
        }
        let full = r.prompt_len / self.cfg.block_size;
        self.cache.insert(&r.tokens[..full * self.cfg.block_size], &r.blocks[..full], &mut self.alloc);
        r.prompt_cached = true;
    }

    fn finish(&mut self, id: RequestId, reason: FinishReason) -> Finished {
        // Overlap: a later step in flight may already compute this request's next token; it is thrown away.
        self.stats.discarded_samples += self.drop_from_in_flight(id) as u64;
        let mut r = self.requests.remove(&id).expect("request exists");
        self.running.retain(|&x| x != id);
        if self.cfg.enable_prefix_cache {
            // The last sampled token has no KV yet; everything before it does.
            let full = r.num_computed / self.cfg.block_size;
            self.cache.insert(&r.tokens[..full * self.cfg.block_size], &r.blocks[..full], &mut self.alloc);
        }
        for b in r.blocks.drain(..) {
            self.alloc.decref(b);
        }
        if let Some(slot) = r.state_slot.take() {
            self.free_state_slots.push(slot);
        }
        self.released.push(Released { request_id: id, reason: ReleaseReason::Finished });
        self.stats.finished += 1;
        Finished {
            request_id: id,
            reason,
            output_tokens: r.output_tokens().to_vec(),
            cached_prompt_tokens: r.cached_prompt_tokens,
            num_preemptions: r.num_preemptions,
        }
    }
}

/// The request finishes once its pending tokens land, whatever they are.
fn ends_by_length(r: &Request, seq_cap: usize) -> bool {
    r.output_tokens().len() + r.num_pending >= r.params.max_new_tokens || r.tokens.len() + r.num_pending >= seq_cap
}

fn finish_reason(r: &Request, seq_cap: usize) -> Option<FinishReason> {
    let last = *r.tokens.last().expect("non-empty");
    if !r.params.ignore_eos && r.params.stop_token_ids.contains(&last) {
        Some(FinishReason::Stop)
    } else if r.output_tokens().len() >= r.params.max_new_tokens {
        Some(FinishReason::Length)
    } else if r.tokens.len() >= seq_cap {
        Some(FinishReason::MaxSeqLen)
    } else {
        None
    }
}

#[cfg(test)]
mod tests;
