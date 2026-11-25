// Copyright 2026 the MilliJoule authors
// SPDX-License-Identifier: Apache-2.0
//! mj-mlx scheduler core (forked from MilliJoule mj-core, 2026-11-21): request state, paged KV allocator, radix prefix cache and the step scheduler.
//!
//! Everything here runs on the CPU between GPU steps and has no CUDA dependency, so it is tested on any machine.

pub mod config;
pub mod kv;
pub mod radix;
pub mod request;
pub mod scheduler;

pub use config::EngineConfig;
pub use kv::{BlockAllocator, BlockId, OutOfBlocks};
pub use radix::RadixCache;
pub use request::{FinishReason, Request, RequestId, SamplingParams, Status};
pub use scheduler::{Finished, ReleaseReason, Released, SchedError, Scheduler, Stats, StepItem, StepOutput, StepPlan};
