// Copyright 2026 the MilliJoule authors
// SPDX-License-Identifier: Apache-2.0
//! Ref-counted pool of fixed-size KV blocks.

pub type BlockId = u32;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutOfBlocks {
    pub requested: usize,
    pub free: usize,
}

/// A block is free at refcount 0. Requests and the radix cache each hold one reference on a block they use.
pub struct BlockAllocator {
    refcount: Vec<u32>,
    free: Vec<BlockId>,
}

impl BlockAllocator {
    pub fn new(num_blocks: usize) -> Self {
        assert!(num_blocks <= u32::MAX as usize, "num_blocks must fit in u32");
        // Reversed so that low ids are handed out first.
        let free = (0..num_blocks as BlockId).rev().collect();
        Self { refcount: vec![0; num_blocks], free }
    }

    pub fn num_blocks(&self) -> usize {
        self.refcount.len()
    }

    pub fn num_free(&self) -> usize {
        self.free.len()
    }

    /// All-or-nothing: hands out `n` blocks at refcount 1, or none.
    pub fn allocate(&mut self, n: usize) -> Result<Vec<BlockId>, OutOfBlocks> {
        if n > self.free.len() {
            return Err(OutOfBlocks { requested: n, free: self.free.len() });
        }
        let blocks = self.free.split_off(self.free.len() - n);
        for &b in &blocks {
            debug_assert_eq!(self.refcount[b as usize], 0);
            self.refcount[b as usize] = 1;
        }
        Ok(blocks)
    }

    pub fn incref(&mut self, block: BlockId) {
        let rc = &mut self.refcount[block as usize];
        assert!(*rc > 0, "incref of free block {block}");
        *rc += 1;
    }

    /// Returns true when the block went back to the free pool.
    pub fn decref(&mut self, block: BlockId) -> bool {
        let rc = &mut self.refcount[block as usize];
        assert!(*rc > 0, "decref of free block {block}");
        *rc -= 1;
        if *rc == 0 {
            self.free.push(block);
            true
        } else {
            false
        }
    }

    pub fn refcount(&self, block: BlockId) -> u32 {
        self.refcount[block as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocate_is_all_or_nothing() {
        let mut a = BlockAllocator::new(4);
        let got = a.allocate(3).unwrap();
        assert_eq!(got.len(), 3);
        assert_eq!(a.num_free(), 1);
        assert_eq!(a.allocate(2), Err(OutOfBlocks { requested: 2, free: 1 }));
        assert_eq!(a.num_free(), 1);
    }

    #[test]
    fn shared_block_frees_on_last_decref() {
        let mut a = BlockAllocator::new(2);
        let b = a.allocate(1).unwrap()[0];
        a.incref(b);
        assert!(!a.decref(b));
        assert_eq!(a.num_free(), 1);
        assert!(a.decref(b));
        assert_eq!(a.num_free(), 2);
    }

    #[test]
    #[should_panic(expected = "decref of free block")]
    fn double_free_panics() {
        let mut a = BlockAllocator::new(1);
        let b = a.allocate(1).unwrap()[0];
        a.decref(b);
        a.decref(b);
    }
}
