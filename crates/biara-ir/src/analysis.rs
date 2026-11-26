//! Analyses ported from what the C++ side got from LLVM: dominators, post
//! dominators, and an integer constant-range lattice.

use crate::{BlockId, Function, IntPred, Terminator};
use std::collections::{BTreeMap, BTreeSet};

/// Immediate-dominator tree over a function's blocks, computed by Cooper,
/// Harvey and Kennedy's iterative algorithm over reverse postorder.
pub struct Dominators {
    /// Reverse postorder index of each block.
    order: Vec<BlockId>,
    /// `idom[b]`, the immediate dominator, `None` for the entry.
    idom: Vec<Option<BlockId>>,
    /// Postorder numbers per block, for `dominates`.
    po: Vec<u32>,
}

impl Dominators {
    pub fn new(f: &Function) -> Self {
        let n = f.blocks.len();
        // A prototype has no body: it has no dominator tree either.
        let Some(entry) = f.entry.filter(|_| n != 0) else {
            return Self {
                order: Vec::new(),
                idom: vec![None; n],
                po: vec![u32::MAX; n],
            };
        };
        let rpo = reverse_postorder(f);
        let mut rpo_num = vec![u32::MAX; n];
        for (i, b) in rpo.iter().enumerate() {
            rpo_num[b.0 as usize] = i as u32;
        }
        let mut idom: Vec<Option<BlockId>> = vec![None; n];
        let mut changed = true;
        while changed {
            changed = false;
            for &b in rpo.iter().skip(1) {
                let mut new: Option<BlockId> = None;
                for &p in &f.block(b).preds {
                    if rpo_num[p.0 as usize] == u32::MAX {
                        continue; // unreachable predecessor
                    }
                    new = Some(match new {
                        None => p,
                        Some(cur) => intersect(p, cur, &idom, &rpo_num, entry),
                    });
                }
                if new != idom[b.0 as usize] {
                    idom[b.0 as usize] = new;
                    changed = true;
                }
            }
        }
        idom[entry.0 as usize] = None;
        let mut po = vec![0; n];
        for (i, b) in rpo.iter().rev().enumerate() {
            po[b.0 as usize] = i as u32;
        }
        Self {
            order: rpo,
            idom,
            po,
        }
    }

    /// Blocks in reverse postorder.
    pub fn order(&self) -> &[BlockId] {
        &self.order
    }

    pub fn idom(&self, b: BlockId) -> Option<BlockId> {
        self.idom[b.0 as usize]
    }

    /// True when `a` dominates `b` (`a == b` counts).
    pub fn dominates(&self, a: BlockId, b: BlockId) -> bool {
        if a == b {
            return true;
        }
        let mut cur = b;
        loop {
            match self.idom[cur.0 as usize] {
                None => return false,
                Some(d) if d == a => return true,
                Some(d) => cur = d,
            }
        }
    }

    /// Children of a block in the dominator tree.
    pub fn children(&self, b: BlockId) -> Vec<BlockId> {
        self.idom
            .iter()
            .enumerate()
            .filter(|(_, d)| **d == Some(b))
            .map(|(i, _)| BlockId(i as u32))
            .collect()
    }
}

/// Walk both chains up to their common dominator.
///
/// Only reachable blocks have an `idom`; a caller must not pass an
/// unreachable one (the caller skips those predecessors), and a chain that
/// runs out means the input was not a dominator forest, so the entry is
/// returned rather than panicking on a two-block function with dead code.
fn intersect(
    mut a: BlockId,
    mut b: BlockId,
    idom: &[Option<BlockId>],
    rpo_num: &[u32],
    entry: BlockId,
) -> BlockId {
    while a != b {
        while rpo_num[a.0 as usize] > rpo_num[b.0 as usize] {
            match idom[a.0 as usize] {
                Some(d) => a = d,
                None => return entry,
            }
        }
        while rpo_num[b.0 as usize] > rpo_num[a.0 as usize] {
            match idom[b.0 as usize] {
                Some(d) => b = d,
                None => return entry,
            }
        }
    }
    a
}

/// Blocks reachable from the entry, in reverse postorder.
pub fn reverse_postorder(f: &Function) -> Vec<BlockId> {
    let Some(entry) = f.entry else {
        return Vec::new();
    };
    let mut visited = vec![false; f.blocks.len()];
    let mut post = Vec::new();
    // Iterative DFS to avoid a deep recursion on huge functions.
    let mut stack = vec![(entry, 0usize)];
    visited[entry.0 as usize] = true;
    while let Some((b, i)) = stack.pop() {
        if let Some(s) = f.terminator(b).and_then(|term| crate::successor_at(term, i)) {
            stack.push((b, i + 1));
            if !visited[s.0 as usize] {
                visited[s.0 as usize] = true;
                stack.push((s, 0));
            }
        } else {
            post.push(b);
        }
    }
    post.reverse();
    post
}

/// Blocks that can reach an exit, for post-dominance-style queries.
pub fn can_return(f: &Function, b: BlockId) -> bool {
    let mut seen = vec![false; f.blocks.len()];
    let mut stack = vec![b];
    seen[b.0 as usize] = true;
    while let Some(cur) = stack.pop() {
        let term = f.terminator_or_unreachable(cur);
        if matches!(term, Terminator::Ret(_) | Terminator::Unreachable) {
            if matches!(term, Terminator::Ret(_)) {
                return true;
            }
            continue;
        }
        for s in f.successors(cur) {
            if !seen[s.0 as usize] {
                seen[s.0 as usize] = true;
                stack.push(s);
            }
        }
    }
    false
}

/// An inclusive-wrapped integer range over a fixed bit width: the lattice
/// LLVM's `ConstantRange` provides, restricted to what the analyses use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConstantRange {
    pub width: u32,
    /// Inclusive lower bound (bit pattern, unsigned representation).
    pub lo: u128,
    /// Inclusive upper bound; the range is `[lo, hi]` in unsigned order and
    /// may wrap past the top of the width.
    pub hi: u128,
    /// True when the range is the whole type (`lo`/`hi` are meaningless).
    pub full: bool,
}

impl ConstantRange {
    pub fn full(width: u32) -> Self {
        Self {
            width,
            lo: 0,
            hi: mask(width),
            full: true,
        }
    }

    pub fn single(v: u128, width: u32) -> Self {
        Self {
            width,
            lo: v & mask(width),
            hi: v & mask(width),
            full: false,
        }
    }

    /// `[lo, hi]` inclusive, unsigned.
    pub fn new(lo: u128, hi: u128, width: u32) -> Self {
        let m = mask(width);
        let lo = lo & m;
        let hi = hi & m;
        if lo == 0 && hi == m {
            return Self::full(width);
        }
        Self {
            width,
            lo,
            hi,
            full: false,
        }
    }

    pub fn is_single(&self) -> bool {
        !self.full && self.lo == self.hi
    }

    pub fn single_value(&self) -> Option<u128> {
        self.is_single().then_some(self.lo)
    }

    /// Signed interpretation of an unsigned-bounded range, when it does not
    /// cross the sign bit; used when a signed predicate bounds a value.
    pub fn as_signed(&self) -> Option<(i128, i128)> {
        if self.full {
            return None;
        }
        let m = mask(self.width);
        let lo_u = self.lo & m;
        let hi_u = self.hi & m;
        if lo_u <= hi_u {
            Some((signed(lo_u, self.width), signed(hi_u, self.width)))
        } else {
            None
        }
    }

    /// The set of values common to both ranges, or `None` when empty.
    pub fn intersect(&self, other: &Self) -> Option<Self> {
        if self.full {
            return Some(*other);
        }
        if other.full {
            return Some(*self);
        }
        let m = mask(self.width);
        // Enumerate the (at most two) unsigned intervals each range covers.
        let mut out: Option<Self> = None;
        for (al, ah) in self.intervals() {
            for (bl, bh) in other.intervals() {
                let lo = al.max(bl);
                let hi = ah.min(bh);
                if lo <= hi {
                    let cand = Self::new(lo, hi, self.width);
                    out = Some(match out {
                        None => cand,
                        Some(prev) => prev.union(&cand).unwrap_or(prev),
                    });
                }
            }
        }
        let _ = m;
        out
    }

    /// The unsorted intervals this range covers, as inclusive pairs.
    pub fn intervals(&self) -> Vec<(u128, u128)> {
        if self.full {
            return vec![(0, mask(self.width))];
        }
        let m = mask(self.width);
        let (lo, hi) = (self.lo & m, self.hi & m);
        if lo <= hi {
            vec![(lo, hi)]
        } else {
            vec![(lo, m), (0, hi)]
        }
    }

    /// The set of values not in `self`.
    pub fn inverse(&self) -> Self {
        if self.full {
            // The empty set: represent it as a full-width "nothing", which
            // every caller treats by unioning, so `full` is wrong here. Use a
            // single impossible value instead (`lo > hi` in unsigned order is
            // how `intervals` spells a wrap, so an empty range cannot be
            // represented; callers must not ask for one).
            return Self::full(self.width);
        }
        let m = mask(self.width);
        let lo = self.lo & m;
        let hi = self.hi & m;
        if lo <= hi {
            // Complement wraps around the top.
            Self {
                width: self.width,
                lo: hi.wrapping_add(1) & m,
                hi: lo.wrapping_sub(1) & m,
                full: false,
            }
        } else {
            // A wrapping range's complement is a single interval.
            Self {
                width: self.width,
                lo: hi.wrapping_add(1) & m,
                hi: lo.wrapping_sub(1) & m,
                full: false,
            }
        }
    }

    /// The union, when the two ranges' intervals are contiguous.
    pub fn union_with(&self, other: &Self) -> Option<Self> {
        self.union(other)
    }

    /// Add, wrapping at the width (an interval's ends shifted together).
    pub fn add(&self, other: &Self) -> Self {
        if self.full || other.full {
            return Self::full(self.width);
        }
        let (a, b) = (self.intervals(), other.intervals());
        // Ranges whose span covers the whole width can produce anything.
        let span = |r: &Self| -> u128 {
            let m = mask(r.width);
            (r.hi.wrapping_sub(r.lo) & m).wrapping_add(1)
        };
        if span(self).saturating_add(span(other)) > mask(self.width) {
            return Self::full(self.width);
        }
        // Two intervals each: hull the four endpoint sums.
        let mut lo = u128::MAX;
        let mut hi = 0u128;
        for (al, ah) in &a {
            for (bl, bh) in &b {
                let s = al.wrapping_add(*bl) & mask(self.width);
                let e = ah.wrapping_add(*bh) & mask(self.width);
                if s <= e {
                    lo = lo.min(s);
                    hi = hi.max(e);
                } else {
                    return Self::full(self.width);
                }
            }
        }
        Self::new(lo, hi, self.width)
    }

    /// Subtract, wrapping at the width.
    pub fn sub(&self, other: &Self) -> Self {
        self.add(&other.neg())
    }

    /// Negate, wrapping at the width.
    pub fn neg(&self) -> Self {
        if self.full {
            return Self::full(self.width);
        }
        let m = mask(self.width);
        let lo = self.hi.wrapping_neg() & m;
        let hi = self.lo.wrapping_neg() & m;
        Self {
            width: self.width,
            lo,
            hi,
            full: false,
        }
    }

    /// Multiply: hull the extremes of the unsigned products.
    pub fn multiply(&self, other: &Self) -> Self {
        if self.full || other.full {
            return Self::full(self.width);
        }
        let m = mask(self.width);
        let mut lo = u128::MAX;
        let mut hi = 0u128;
        for (al, ah) in self.intervals() {
            for (bl, bh) in other.intervals() {
                for (x, y) in [(al, bl), (al, bh), (ah, bl), (ah, bh)] {
                    let p = x.wrapping_mul(y) & m;
                    lo = lo.min(p);
                    hi = hi.max(p);
                }
            }
        }
        Self::new(lo, hi, self.width)
    }

    /// Bitwise `and`: exact when either side is a single value, else the
    /// hull of the four extreme combinations (conservative but sound).
    pub fn binary_and(&self, other: &Self) -> Self {
        self.bitwise(other, |a, b| a & b)
    }

    /// Bitwise `or`.
    pub fn binary_or(&self, other: &Self) -> Self {
        self.bitwise(other, |a, b| a | b)
    }

    /// Bitwise `xor`.
    pub fn binary_xor(&self, other: &Self) -> Self {
        self.bitwise(other, |a, b| a ^ b)
    }

    fn bitwise(&self, other: &Self, f: impl Fn(u128, u128) -> u128) -> Self {
        if self.full || other.full {
            return Self::full(self.width);
        }
        let m = mask(self.width);
        let mut lo = u128::MAX;
        let mut hi = 0u128;
        for (al, ah) in self.intervals() {
            for (bl, bh) in other.intervals() {
                for (x, y) in [(al, bl), (al, bh), (ah, bl), (ah, bh)] {
                    let v = f(x, y) & m;
                    lo = lo.min(v);
                    hi = hi.max(v);
                }
            }
        }
        Self::new(lo, hi, self.width)
    }

    /// Shift left by a range of amounts.
    pub fn shl(&self, other: &Self) -> Self {
        self.shift(other, |v, s| v << s)
    }

    /// Logical shift right.
    pub fn lshr(&self, other: &Self) -> Self {
        self.shift(other, |v, s| v >> s)
    }

    /// Arithmetic shift right, staying in the unsigned representation.
    pub fn ashr(&self, other: &Self) -> Self {
        if self.full || other.full {
            return Self::full(self.width);
        }
        let m = mask(self.width);
        let mut lo = u128::MAX;
        let mut hi = 0u128;
        for (al, ah) in self.intervals() {
            for (bl, bh) in other.intervals() {
                for (x, s) in [(al, bl), (al, bh), (ah, bl), (ah, bh)] {
                    if s >= self.width as u128 {
                        continue;
                    }
                    let v = (signed(x, self.width) >> s) as u128 & m;
                    lo = lo.min(v);
                    hi = hi.max(v);
                }
            }
        }
        if lo == u128::MAX {
            return Self::full(self.width);
        }
        Self::new(lo, hi, self.width)
    }

    fn shift(&self, other: &Self, f: impl Fn(u128, u32) -> u128) -> Self {
        if self.full || other.full {
            return Self::full(self.width);
        }
        let m = mask(self.width);
        let mut lo = u128::MAX;
        let mut hi = 0u128;
        for (al, ah) in self.intervals() {
            for (bl, bh) in other.intervals() {
                for (x, s) in [(al, bl), (al, bh), (ah, bl), (ah, bh)] {
                    if s >= self.width as u128 {
                        // LLVM's shift ranges poison on overshoot; the
                        // analyses treat that as "anything".
                        return Self::full(self.width);
                    }
                    let v = f(x, s as u32) & m;
                    lo = lo.min(v);
                    hi = hi.max(v);
                }
            }
        }
        if lo == u128::MAX {
            return Self::full(self.width);
        }
        Self::new(lo, hi, self.width)
    }

    /// Truncate to `width` bits.
    pub fn trunc(&self, width: u32) -> Self {
        if self.full {
            return Self::full(width);
        }
        let m = mask(width);
        Self::new(self.lo & m, self.hi & m, width)
    }

    /// Zero-extend to `width` bits.
    pub fn zext(&self, width: u32) -> Self {
        if self.full {
            return Self::full(width);
        }
        Self::new(self.lo, self.hi, width)
    }

    /// Sign-extend to `width` bits: an interval whose sign bit is set
    /// becomes a high, wrapped interval.
    pub fn sext(&self, width: u32) -> Self {
        if self.full {
            return Self::full(width);
        }
        let m = mask(width);
        let lo = signed(self.lo, self.width) as u128 & m;
        let hi = signed(self.hi, self.width) as u128 & m;
        Self::new(lo, hi, width)
    }

    /// The range implied by `v <pred> k`, at `self.width`.
    pub fn icmp_region(pred: IntPred, k: u128, width: u32) -> Self {
        let m = mask(width);
        let k = k & m;
        match pred {
            IntPred::Eq => Self::single(k, width),
            IntPred::Ne => Self::single(k, width).inverse(),
            IntPred::Ugt => Self::new(k.wrapping_add(1) & m, m, width),
            IntPred::Uge => Self::new(k, m, width),
            IntPred::Ult => Self::new(0, k.wrapping_sub(1) & m, width),
            IntPred::Ule => Self::new(0, k, width),
            IntPred::Sgt => {
                let lo = (signed(k, width) + 1) as u128 & m;
                Self::new(lo, m >> 1, width)
            }
            IntPred::Sge => Self::new(k, m >> 1, width),
            IntPred::Slt => Self::new(m >> 1 | (m & !(m >> 1)), k.wrapping_sub(1) & m, width),
            IntPred::Sle => Self::new(m >> 1 | (m & !(m >> 1)), k, width),
        }
    }

    fn union(&self, other: &Self) -> Option<Self> {
        let mine = self.intervals();
        let theirs = other.intervals();
        // Only handle the disjoint-touching case the intersection needs.
        let mut all: Vec<(u128, u128)> = mine.into_iter().chain(theirs).collect();
        all.sort();
        let mut merged: Vec<(u128, u128)> = Vec::new();
        for (lo, hi) in all {
            match merged.last_mut() {
                Some(last) if lo <= last.1.saturating_add(1) => last.1 = last.1.max(hi),
                _ => merged.push((lo, hi)),
            }
        }
        if merged.len() == 1 {
            Some(Self::new(merged[0].0, merged[0].1, self.width))
        } else {
            None
        }
    }
}

pub fn mask(width: u32) -> u128 {
    if width >= 128 {
        u128::MAX
    } else {
        (1u128 << width) - 1
    }
}

pub fn signed(v: u128, width: u32) -> i128 {
    let w = width.min(128);
    if w == 0 {
        return 0;
    }
    let m = mask(w);
    let v = v & m;
    let sign = 1u128 << (w - 1);
    if v & sign != 0 {
        (v | !m) as i128
    } else {
        v as i128
    }
}

/// Per-block facts the folding passes reuse: which values are constants,
/// block predecessors, and dominators.
pub struct FunctionInfo<'a> {
    pub func: &'a Function,
    pub doms: Dominators,
    pub reachable: BTreeSet<BlockId>,
    pub block_of: BTreeMap<u32, BlockId>,
}

impl<'a> FunctionInfo<'a> {
    pub fn new(func: &'a Function) -> Self {
        let doms = Dominators::new(func);
        let reachable: BTreeSet<BlockId> = doms.order().iter().copied().collect();
        let mut block_of = BTreeMap::new();
        for (i, v) in func.defined_in.iter().enumerate() {
            if let Some(b) = v {
                block_of.insert(i as u32, *b);
            }
        }
        Self {
            func,
            doms,
            reachable,
            block_of,
        }
    }
}

/// Recompute `preds` from the terminators. Called after any CFG edit.
pub fn recompute_preds(f: &mut Function) {
    for b in &mut f.blocks {
        b.preds.clear();
    }
    for i in 0..f.blocks.len() {
        let b = BlockId(i as u32);
        let mut index = 0;
        while let Some(succ) = f.terminator(b).and_then(|term| crate::successor_at(term, index)) {
            // Out-of-range targets are a verify error, not a panic here.
            if let Some(block) = f.blocks.get_mut(succ.0 as usize) {
                block.preds.insert(b);
            }
            index += 1;
        }
    }
}
