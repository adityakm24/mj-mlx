// Copyright 2026 the MilliJoule authors
// SPDX-License-Identifier: Apache-2.0
/// Static engine limits; fixed for the life of a `Scheduler`.
#[derive(Clone, Debug)]
pub struct EngineConfig {
    /// Tokens per KV block (page).
    pub block_size: usize,
    /// KV blocks in the pool, sized by the runner from free HBM.
    pub num_blocks: usize,
    /// Most requests in one step; the largest CUDA graph bucket.
    pub max_running: usize,
    /// Token budget of one step, prefill chunks and decodes together.
    pub max_batch_tokens: usize,
    /// Longest prefill chunk of one request in one step.
    pub max_prefill_chunk: usize,
    /// Prefill attention area cap, 0 for none. A chunk of `c` tokens after `n` cached ones attends over a
    /// `c x (n + c)` area; with a cap, `c` shrinks as the context grows so the area stays under it (never below
    /// `min(256, max_prefill_chunk)` tokens). Its transient memory grows with that area, and on a 16 GB Mac a
    /// 2048-token chunk over a 16k context ran into the Metal limit (mj-mlx RESULTS 1zj).
    pub prefill_area_cap: usize,
    /// Prompt plus output limit per request.
    pub max_seq_len: usize,
    pub enable_prefix_cache: bool,
    /// Recurrent-state slots (hybrid models such as Qwen3.6). 0 means the model has none. Each running request
    /// holds one slot; admission waits when none is free. Requires `enable_prefix_cache = false`: a cache hit
    /// would skip prompt tokens whose recurrent state nobody has.
    pub num_state_slots: usize,
    /// Overlap scheduling: `schedule()` may plan step N+1 while step N is still in flight (at most two open).
    /// A decode whose input is the token step N is sampling gets it by reference (`StepPlan::prev_sample_idx`).
    /// A stop token is seen one step late, so a request that stops gets one extra sampled token, which is dropped.
    /// The runner must run plans in the order `schedule()` returned them, each after the previous one's device work
    /// (one stream or queue).
    pub overlap: bool,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            block_size: 16,
            num_blocks: 4096,
            max_running: 512,
            max_batch_tokens: 8192,
            max_prefill_chunk: 8192,
            prefill_area_cap: 0,
            max_seq_len: 32768,
            enable_prefix_cache: true,
            num_state_slots: 0,
            overlap: false,
        }
    }
}

impl EngineConfig {
    /// Longest chunk for a request with `num_computed` tokens already in the cache.
    pub fn prefill_chunk_at(&self, num_computed: usize) -> usize {
        let max = self.max_prefill_chunk;
        if self.prefill_area_cap == 0 {
            return max;
        }
        // largest c with c * (n + c) <= cap: c = (sqrt(n^2 + 4 cap) - n) / 2, then step down for rounding
        let (n, cap) = (num_computed as f64, self.prefill_area_cap as f64);
        let mut c = ((n * n + 4.0 * cap).sqrt() - n) / 2.0;
        c = c.floor().max(0.0);
        let mut c = (c as usize).min(max);
        while c > 0 && c.saturating_mul(num_computed + c) > self.prefill_area_cap {
            c -= 1;
        }
        if c >= self.block_size {
            c -= c % self.block_size; // whole KV blocks, so chunks stay block aligned
        }
        c.max(256.min(max))
    }

    pub fn validate(&self) -> Result<(), String> {
        let positive = [
            ("block_size", self.block_size),
            ("num_blocks", self.num_blocks),
            ("max_running", self.max_running),
            ("max_batch_tokens", self.max_batch_tokens),
            ("max_prefill_chunk", self.max_prefill_chunk),
            ("max_seq_len", self.max_seq_len),
        ];
        for (name, value) in positive {
            if value == 0 {
                return Err(format!("{name} must be > 0"));
            }
        }
        // Step arrays are int32 (block tables padded with -1, positions, slot indices).
        if self.num_blocks > i32::MAX as usize {
            return Err("num_blocks must fit in i32".into());
        }
        if self.max_seq_len > i32::MAX as usize {
            return Err("max_seq_len must fit in i32".into());
        }
        if self.num_blocks.checked_mul(self.block_size).is_none_or(|slots| slots > i32::MAX as usize) {
            return Err("num_blocks * block_size must fit in i32".into());
        }
        if self.num_state_slots > i32::MAX as usize {
            return Err("num_state_slots must fit in i32".into());
        }
        if self.num_state_slots > 0 && self.enable_prefix_cache {
            return Err("num_state_slots > 0 requires enable_prefix_cache = false".into());
        }
        // Every running request must be able to decode in the same step.
        if self.max_running > self.max_batch_tokens {
            return Err("max_running must be <= max_batch_tokens".into());
        }
        Ok(())
    }

    pub fn blocks_for(&self, tokens: usize) -> usize {
        tokens.div_ceil(self.block_size)
    }
}
