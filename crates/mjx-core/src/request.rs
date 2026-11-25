// Copyright 2026 the MilliJoule authors
// SPDX-License-Identifier: Apache-2.0
use crate::kv::BlockId;

pub type RequestId = u64;

/// Per-request limits and sampling settings. The runner samples on the device; the scheduler only carries the
/// settings into each step's arrays.
#[derive(Clone, Debug)]
pub struct SamplingParams {
    pub max_new_tokens: usize,
    pub stop_token_ids: Vec<u32>,
    pub ignore_eos: bool,
    /// 0 means greedy.
    pub temperature: f32,
    /// 1.0 means off.
    pub top_p: f32,
    /// 0 means off.
    pub top_k: u32,
    /// Randomness should come from `(seed, position)`, so a recomputed request samples the same tokens.
    /// `None` uses the request id.
    pub seed: Option<u64>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            max_new_tokens: 256,
            stop_token_ids: Vec::new(),
            ignore_eos: false,
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            seed: None,
        }
    }
}

impl SamplingParams {
    pub fn validate(&self) -> Result<(), String> {
        if !(self.temperature.is_finite() && self.temperature >= 0.0) {
            return Err("temperature must be finite and >= 0".into());
        }
        if !(self.top_p > 0.0 && self.top_p <= 1.0) {
            return Err("top_p must be in (0, 1]".into());
        }
        if self.top_k > i32::MAX as u32 {
            return Err("top_k must fit in i32".into());
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinishReason {
    /// Hit `max_new_tokens`.
    Length,
    /// Sampled a stop token (kept in the output).
    Stop,
    /// Hit the engine's `max_seq_len`.
    MaxSeqLen,
}

impl FinishReason {
    pub fn as_str(self) -> &'static str {
        match self {
            FinishReason::Length => "length",
            FinishReason::Stop => "stop",
            FinishReason::MaxSeqLen => "max_seq_len",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Waiting,
    Running,
}

#[derive(Clone, Debug)]
pub struct Request {
    pub id: RequestId,
    /// Prompt followed by the tokens sampled so far.
    pub tokens: Vec<u32>,
    pub prompt_len: usize,
    pub params: SamplingParams,
    /// Leading tokens whose KV is written, or scheduled to be written by the step in flight.
    pub num_computed: usize,
    /// Covers positions `0 .. blocks.len() * block_size`.
    pub blocks: Vec<BlockId>,
    /// Recurrent-state slot, held while running (only when `num_state_slots > 0`).
    pub state_slot: Option<u32>,
    /// Tokens being sampled by steps in flight, not yet returned through `update()`. Their positions come after
    /// `tokens`; with overlap scheduling the next step may already compute the first of them.
    pub num_pending: usize,
    pub status: Status,
    /// Prompt tokens served from the prefix cache at the first admission.
    pub cached_prompt_tokens: usize,
    pub num_preemptions: u32,
    pub(crate) admitted_once: bool,
    /// The prompt's full blocks are already in the prefix cache.
    pub(crate) prompt_cached: bool,
}

impl Request {
    pub fn new(id: RequestId, prompt: Vec<u32>, params: SamplingParams) -> Self {
        let prompt_len = prompt.len();
        Self {
            id,
            tokens: prompt,
            prompt_len,
            params,
            num_computed: 0,
            blocks: Vec::new(),
            state_slot: None,
            num_pending: 0,
            status: Status::Waiting,
            cached_prompt_tokens: 0,
            num_preemptions: 0,
            admitted_once: false,
            prompt_cached: false,
        }
    }

    pub fn output_tokens(&self) -> &[u32] {
        &self.tokens[self.prompt_len..]
    }

    /// Positions not yet scheduled, counting the pending ones.
    pub fn num_uncomputed(&self) -> usize {
        self.tokens.len() + self.num_pending - self.num_computed
    }
}
