// Copyright 2026 the MilliJoule authors
// SPDX-License-Identifier: Apache-2.0
//! Radix prefix cache over whole KV blocks (the page-granular form of SGLang's `RadixCache`).
//!
//! Each tree edge is one full block: its key is the block's `block_size` tokens, its value the block holding their KV.
//! The cache owns one reference on every block in the tree; a block is evictable when that is its only reference.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

use crate::kv::{BlockAllocator, BlockId};

type NodeId = usize;
const ROOT: NodeId = 0;

struct Node {
    parent: NodeId,
    key: Vec<u32>,
    block: BlockId,
    children: HashMap<Vec<u32>, NodeId>,
    last_access: u64,
}

pub struct RadixCache {
    block_size: usize,
    nodes: Vec<Node>,
    // Slots of evicted nodes, reused by later inserts.
    free_nodes: Vec<NodeId>,
    clock: u64,
    num_cached: usize,
}

impl RadixCache {
    pub fn new(block_size: usize) -> Self {
        assert!(block_size > 0);
        let root = Node { parent: ROOT, key: Vec::new(), block: 0, children: HashMap::new(), last_access: 0 };
        Self { block_size, nodes: vec![root], free_nodes: Vec::new(), clock: 0, num_cached: 0 }
    }

    pub fn num_cached_blocks(&self) -> usize {
        self.num_cached
    }

    /// Blocks of the longest cached prefix of `tokens`, at most `max_blocks` of them.
    /// Each returned block gets one extra reference, owned by the caller.
    pub fn match_prefix(&mut self, tokens: &[u32], max_blocks: usize, alloc: &mut BlockAllocator) -> Vec<BlockId> {
        self.clock += 1;
        let mut node = ROOT;
        let mut blocks = Vec::new();
        for chunk in tokens.chunks_exact(self.block_size).take(max_blocks) {
            let Some(child) = self.nodes[node].children.get(chunk).copied() else {
                break;
            };
            node = child;
            self.nodes[child].last_access = self.clock;
            alloc.incref(self.nodes[child].block);
            blocks.push(self.nodes[child].block);
        }
        blocks
    }

    /// Adds the full blocks of `tokens` (paired with `blocks`) to the tree and returns how many were new.
    /// A prefix already cached under another block keeps the cached block; the caller's copy is not adopted.
    pub fn insert(&mut self, tokens: &[u32], blocks: &[BlockId], alloc: &mut BlockAllocator) -> usize {
        self.clock += 1;
        let n = (tokens.len() / self.block_size).min(blocks.len());
        let mut node = ROOT;
        let mut added = 0;
        for (i, chunk) in tokens.chunks_exact(self.block_size).take(n).enumerate() {
            if let Some(child) = self.nodes[node].children.get(chunk).copied() {
                node = child;
                self.nodes[child].last_access = self.clock;
                continue;
            }
            alloc.incref(blocks[i]);
            let new = Node {
                parent: node,
                key: chunk.to_vec(),
                block: blocks[i],
                children: HashMap::new(),
                last_access: self.clock,
            };
            let id = match self.free_nodes.pop() {
                Some(slot) => {
                    self.nodes[slot] = new;
                    slot
                }
                None => {
                    self.nodes.push(new);
                    self.nodes.len() - 1
                }
            };
            self.nodes[node].children.insert(chunk.to_vec(), id);
            node = id;
            added += 1;
            self.num_cached += 1;
        }
        added
    }

    /// Frees up to `want` blocks by removing least recently used leaves that no request holds.
    /// Returns how many blocks went back to the pool.
    pub fn evict(&mut self, want: usize, alloc: &mut BlockAllocator) -> usize {
        if want == 0 {
            return 0;
        }
        let mut heap: BinaryHeap<Reverse<(u64, NodeId)>> = self
            .live_nodes()
            .filter(|&id| self.is_evictable(id, alloc))
            .map(|id| Reverse((self.nodes[id].last_access, id)))
            .collect();
        let mut freed = 0;
        while freed < want {
            let Some(Reverse((_, id))) = heap.pop() else { break };
            if !self.is_evictable(id, alloc) {
                continue;
            }
            let parent = self.remove_leaf(id, alloc);
            freed += 1;
            if parent != ROOT && self.is_evictable(parent, alloc) {
                heap.push(Reverse((self.nodes[parent].last_access, parent)));
            }
        }
        freed
    }

    /// Blocks currently referenced by the tree.
    pub fn blocks(&self) -> impl Iterator<Item = BlockId> + '_ {
        self.live_nodes().map(|id| self.nodes[id].block)
    }

    fn live_nodes(&self) -> impl Iterator<Item = NodeId> + '_ {
        // Depth-first walk from the root; evicted slots are unreachable.
        let mut stack: Vec<NodeId> = self.nodes[ROOT].children.values().copied().collect();
        std::iter::from_fn(move || {
            let id = stack.pop()?;
            stack.extend(self.nodes[id].children.values().copied());
            Some(id)
        })
    }

    fn is_evictable(&self, id: NodeId, alloc: &BlockAllocator) -> bool {
        id != ROOT && self.nodes[id].children.is_empty() && alloc.refcount(self.nodes[id].block) == 1
    }

    fn remove_leaf(&mut self, id: NodeId, alloc: &mut BlockAllocator) -> NodeId {
        let parent = self.nodes[id].parent;
        let key = std::mem::take(&mut self.nodes[id].key);
        self.nodes[parent].children.remove(&key);
        alloc.decref(self.nodes[id].block);
        self.free_nodes.push(id);
        self.num_cached -= 1;
        parent
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seq(n: u32) -> Vec<u32> {
        (0..n).collect()
    }

    #[test]
    fn match_returns_whole_blocks_only() {
        let mut alloc = BlockAllocator::new(8);
        let mut cache = RadixCache::new(4);
        let blocks = alloc.allocate(3).unwrap();
        assert_eq!(cache.insert(&seq(10), &blocks, &mut alloc), 2);
        // 10 tokens fill two blocks; the third, partial block is not cached.
        assert_eq!(alloc.refcount(blocks[2]), 1);
        let hit = cache.match_prefix(&seq(12), usize::MAX, &mut alloc);
        assert_eq!(hit, blocks[..2].to_vec());
        assert_eq!(alloc.refcount(blocks[0]), 3);
    }

    #[test]
    fn match_stops_at_first_difference_and_respects_max() {
        let mut alloc = BlockAllocator::new(8);
        let mut cache = RadixCache::new(2);
        let blocks = alloc.allocate(3).unwrap();
        cache.insert(&[1, 2, 3, 4, 5, 6], &blocks, &mut alloc);
        assert_eq!(cache.match_prefix(&[1, 2, 3, 9, 5, 6], 8, &mut alloc), vec![blocks[0]]);
        assert_eq!(cache.match_prefix(&[1, 2, 3, 4, 5, 6], 2, &mut alloc).len(), 2);
    }

    #[test]
    fn duplicate_insert_keeps_the_cached_block() {
        let mut alloc = BlockAllocator::new(8);
        let mut cache = RadixCache::new(2);
        let a = alloc.allocate(2).unwrap();
        let b = alloc.allocate(2).unwrap();
        assert_eq!(cache.insert(&seq(4), &a, &mut alloc), 2);
        assert_eq!(cache.insert(&seq(4), &b, &mut alloc), 0);
        assert_eq!(alloc.refcount(b[0]), 1);
        assert_eq!(cache.num_cached_blocks(), 2);
    }

    #[test]
    fn evict_frees_lru_leaves_and_skips_held_blocks() {
        let mut alloc = BlockAllocator::new(8);
        let mut cache = RadixCache::new(2);
        let old = alloc.allocate(2).unwrap();
        cache.insert(&[1, 1, 2, 2], &old, &mut alloc);
        let new = alloc.allocate(2).unwrap();
        cache.insert(&[3, 3, 4, 4], &new, &mut alloc);
        for &b in old.iter().chain(&new) {
            alloc.decref(b);
        }
        // A request still holds the newer chain's first block, so only its leaf can go.
        let held = cache.match_prefix(&[3, 3], 1, &mut alloc);
        assert_eq!(cache.evict(3, &mut alloc), 3);
        assert_eq!(cache.num_cached_blocks(), 1);
        assert_eq!(cache.blocks().collect::<Vec<_>>(), held);
        assert_eq!(alloc.num_free(), 7);
    }

    #[test]
    fn evicted_slots_are_reused() {
        let mut alloc = BlockAllocator::new(4);
        let mut cache = RadixCache::new(1);
        let b = alloc.allocate(1).unwrap();
        cache.insert(&[7], &b, &mut alloc);
        alloc.decref(b[0]);
        assert_eq!(cache.evict(1, &mut alloc), 1);
        let c = alloc.allocate(1).unwrap();
        cache.insert(&[8], &c, &mut alloc);
        assert_eq!(cache.nodes.len(), 2);
        assert_eq!(cache.match_prefix(&[8], 1, &mut alloc), c);
        assert!(cache.match_prefix(&[7], 1, &mut alloc).is_empty());
    }
}
