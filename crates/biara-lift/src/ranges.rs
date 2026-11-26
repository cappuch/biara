//! Control-flow-edge range refinement, ported from `cpp/edge_ranges.h`.
//!
//! Baseline ranges come from a value's definition; dominating branch and
//! switch outcomes narrow them further. The old baseline was LLVM's
//! `LazyValueInfo`; here it is `biara_ir`'s `transfer`, which walks the
//! defining expression with the same wrapping arithmetic.
//!
//! Soundness contract (unchanged from the C++ port, keep it):
//! - A range is only narrowed by an edge whose outcome provably held on
//!   every path to the query block. Facts from one arm of a disjunction never
//!   leak into a join: when no single in-edge dominates, only branches
//!   dominating the whole join apply.
//! - Expression transfer is width-correct wrapping arithmetic on
//!   `ConstantRange`. Unmodelled operations (div/rem, loads, calls, GEPs)
//!   stop the walk and keep the incoming range.
//! - Joins hull incoming ranges. Disjoint alternatives are never intersected.
//! - The refined set must sit inside the baseline; otherwise the walk
//!   overreached and the baseline stands.
//! - Every walk carries a hard budget; exhaustion falls back to the baseline.

use biara_ir::{
    BinOp, BlockId, Constant, ConstantRange, Dominators, Function, Inst, IntPred, Terminator, Type,
    Value, ValueId,
};

/// Dominator levels climbed per query; expression-walk nesting depth.
const DEPTH: u32 = 24;
/// Instructions and edges touched per top-level query.
const VISITS: u32 = 256;
/// Distinct phi arms given exact ranges before the rest hull to unknown.
const JOIN_PATHS: usize = 8;

/// Refinement state for one function, reused across a batch of queries.
pub struct EdgeRanges<'f> {
    f: &'f Function,
    doms: Dominators,
    /// The sole predecessor of a block, or `None` when it has 0 or 2+.
    single_pred: Vec<Option<BlockId>>,
}

impl<'f> EdgeRanges<'f> {
    pub fn new(f: &'f Function) -> Self {
        let doms = Dominators::new(f);
        let mut single_pred = vec![None; f.blocks.len()];
        let mut seen = vec![false; f.blocks.len()];
        for b in 0..f.blocks.len() {
            let from = BlockId(b as u32);
            for to in f.blocks[b].term.as_ref().map(biara_ir::successor_blocks).unwrap_or_default() {
                let slot = &mut single_pred[to.0 as usize];
                let flag = &mut seen[to.0 as usize];
                if *flag {
                    *slot = None;
                } else {
                    *flag = true;
                    *slot = Some(from);
                }
            }
        }
        Self {
            f,
            doms,
            single_pred,
        }
    }

    /// A range for `value` from a dominating `if` guard, walking the
    /// single-predecessor chain above `block`. `if (v > k) escape` with
    /// `block` on the false arm gives `v <= k`.
    /// The range a value's own definition guarantees, independent of any
    /// guard: masking with a contiguous low mask always yields at most that
    /// mask. Anything else is left full — a partial bound here would be a
    /// guess about the operand, not a fact about the result.
    pub fn defined_range(&self, value: ValueId) -> Option<ConstantRange> {
        let w = self.width_of(value)?;
        let m = biara_ir::analysis::mask(w);
        let Some(Inst::Bin { op: BinOp::And, lhs, rhs, .. }) = self.f.inst(value) else {
            return None;
        };
        let mask = match (self.const_value(*lhs), self.const_value(*rhs)) {
            (_, Some(k)) => k,
            (Some(k), _) => k,
            _ => return None,
        };
        let mask = mask & m;
        // Only a run of low ones bounds the result to `[0, mask]`.
        if mask != 0 && mask & mask.wrapping_add(1) != 0 {
            return None;
        }
        Some(ConstantRange::new(0, mask, w))
    }

    /// Whether `a` and `b` denote the same value under casts and reloads of
    /// a local slot. A guard compares a register reload that the table index
    /// was widened from, so the two ids differ but the quantity is one.
    pub fn same_under_casts(&self, a: ValueId, b: ValueId) -> bool {
        self.peel_casts(a) == self.peel_casts(b)
    }

    fn peel_casts(&self, mut v: ValueId) -> ValueId {
        for _ in 0..8 {
            match self.f.inst(v) {
                Some(Inst::Zext { value, .. })
                | Some(Inst::Sext { value, .. })
                | Some(Inst::Trunc { value, .. })
                | Some(Inst::Cast { value, .. }) => v = *value,
                Some(Inst::Load { ptr, .. }) => match self.store_into(*ptr, v) {
                    Some(w) => v = w,
                    None => return v,
                },
                _ => return v,
            }
        }
        v
    }

    /// The value stored to `slot` before `at` in its own block, if any.
    fn store_into(&self, slot: ValueId, at: ValueId) -> Option<ValueId> {
        if !matches!(self.f.inst(slot), Some(Inst::Alloca { .. })) {
            return None;
        }
        let b = self.block_of(at)?;
        let insts = &self.f.blocks[b.0 as usize].insts;
        let pos = insts.iter().position(|&i| i == at)?;
        insts[..pos].iter().rev().find_map(|&i| match self.f.inst(i) {
            Some(Inst::Store { ptr, value, .. }) if *ptr == slot => Some(*value),
            _ => None,
        })
    }

    pub fn guard_range(&self, value: ValueId, block: BlockId) -> Option<ConstantRange> {
        let mut b = block;
        for _ in 0..16 {
            let p = self.single_pred[b.0 as usize]?;
            let Some(Terminator::CondBr { cond, yes, no }) = self.f.blocks[p.0 as usize].term.clone() else {
                return None;
            };
            // `b` must be the false arm, so the guard's predicate failed.
            if yes == b || no != b {
                return None;
            }
            if let Some(Inst::Icmp {
                pred,
                ty,
                lhs,
                rhs,
            }) = self.f.inst(cond)
            {
                let w = ty.int_bits()?;
                // The guard compares the value through whatever widening or
                // truncation the lifter inserted; both preserve it.
                if *lhs == value || self.same_under_casts(*lhs, value) {
                    if let Some(k) = self.const_value(*rhs) {
                        let bound = k & biara_ir::analysis::mask(w);
                        return match pred {
                            IntPred::Ugt => Some(ConstantRange::new(0, bound, w)),
                            IntPred::Uge => {
                                Some(ConstantRange::new(0, bound.wrapping_sub(1), w))
                            }
                            _ => None,
                        };
                    }
                }
            }
            b = p;
        }
        None
    }

    /// Refine `value`'s range at `ctx` given the baseline `base`. Returns the
    /// intersection when non-empty and inside `base`, else `base`.
    pub fn refine(&self, value: ValueId, ctx: ValueId, base: ConstantRange) -> ConstantRange {
        let width = base.width;
        if width == 0 || width > 64 {
            return base;
        }
        let Some(block) = self.block_of(ctx) else {
            return base;
        };
        let mut budget = VISITS;
        let refined = self.apply_context_facts(value, block, base, &mut budget);
        // Only a narrowing inside the baseline is a fact.
        match base.intersect(&refined) {
            Some(r) if !r.full && r != base => r,
            _ => base,
        }
    }

    /// Intersect `r` with what must hold on entry to `block`.
    fn apply_context_facts(
        &self,
        value: ValueId,
        block: BlockId,
        r: ConstantRange,
        budget: &mut u32,
    ) -> ConstantRange {
        let mut cur = r;
        let mut node = Some(block);
        let mut steps = 0;
        while let Some(b) = node {
            if steps >= DEPTH || *budget == 0 {
                break;
            }
            steps += 1;
            if b != block {
                if let Some(edge) = self.edge_range(value, b, block) {
                    cur = match cur.intersect(&edge) {
                        Some(n) => n,
                        None => cur,
                    };
                }
            }
            node = self.doms.idom(b);
        }
        cur
    }

    /// The range of `value` implied by taking the edge `from -> to`.
    fn edge_range(&self, value: ValueId, from: BlockId, to: BlockId) -> Option<ConstantRange> {
        let width = self.width_of(value)?;
        match self.f.blocks[from.0 as usize].term.as_ref().unwrap_or(&Terminator::Unreachable) {
            Terminator::CondBr { cond, yes, no } => {
                if yes == no {
                    return None;
                }
                let taken = *yes == to;
                if !taken && *no != to {
                    return None;
                }
                let mut budget = VISITS;
                self.condition_range(value, *cond, taken, DEPTH, &mut budget)
            }
            Terminator::Switch {
                value: cond,
                default,
                cases,
            } => {
                if *cond != value {
                    return None;
                }
                let is_default = *default == to;
                let mut found: Option<ConstantRange> = None;
                let mut count = 0usize;
                for (k, target) in cases {
                    if is_default && *target == to {
                        return None;
                    }
                    if !is_default && *target != to {
                        continue;
                    }
                    let single = ConstantRange::single(*k, width);
                    found = Some(match found {
                        None => single,
                        Some(prev) => prev.union_with(&single).unwrap_or(prev),
                    });
                    count += 1;
                }
                let found = found?;
                if is_default {
                    // Only invert an exact case union; hull gaps are reachable.
                    let m = biara_ir::analysis::mask(width);
                    let span = (found.hi.wrapping_sub(found.lo) & m).wrapping_add(1);
                    if found.full || span != cases.len() as u128 {
                        return None;
                    }
                    return Some(found.inverse());
                }
                let _ = count;
                Some(found)
            }
            _ => None,
        }
    }

    /// The range of `value` implied by `cond` having outcome `taken`.
    fn condition_range(
        &self,
        value: ValueId,
        cond: ValueId,
        taken: bool,
        depth: u32,
        budget: &mut u32,
    ) -> Option<ConstantRange> {
        if depth == 0 || *budget == 0 {
            return None;
        }
        *budget -= 1;
        let width = self.width_of(value)?;
        if self.is_int(cond) {
            // `cond == value`: the branch proves zero or non-zero.
            if cond == value {
                return Some(ConstantRange::icmp_region(
                    if taken { IntPred::Ne } else { IntPred::Eq },
                    0,
                    width,
                ));
            }
        }
        match self.f.inst(cond) {
            Some(Inst::Zext { value: inner, .. })
            | Some(Inst::Sext { value: inner, .. })
            | Some(Inst::Trunc { value: inner, .. }) => {
                self.condition_range(value, *inner, taken, depth / 2, budget)
            }
            Some(Inst::Phi { incoming, .. }) => {
                if incoming.len() > JOIN_PATHS {
                    return None;
                }
                if let Some(def) = self.block_of(value) {
                    let here = self.block_of(cond);
                    if here == Some(def) || !self.dominates(def, here.unwrap_or(def)) {
                        return None;
                    }
                }
                let mut out: Option<ConstantRange> = None;
                for (incoming_value, _) in incoming {
                    if *budget == 0 {
                        return None;
                    }
                    *budget -= 1;
                    if let Some(c) = self.const_value(*incoming_value) {
                        if c == 0 && taken {
                            continue;
                        }
                        return None;
                    }
                    let arm = self.condition_range(value, *incoming_value, taken, depth / 2, budget)?;
                    out = Some(match out {
                        None => arm,
                        Some(prev) => prev.union_with(&arm)?,
                    });
                }
                out
            }
            Some(Inst::Icmp {
                pred,
                ty,
                lhs,
                rhs,
            }) => {
                let w = ty.int_bits()?;
                // The lifter branches on a synthesised flag word rather than
                // on the comparison itself: `ne (0 - zext(cmp)) & flags, 0`
                // says `cmp` held. The comparison underneath still bounds the
                // value it compared, so unwrap to it with the right polarity.
                if matches!(pred, IntPred::Eq | IntPred::Ne)
                    && (self.const_value(*lhs) == Some(0) || self.const_value(*rhs) == Some(0))
                {
                    let other = if self.const_value(*lhs) == Some(0) { *rhs } else { *lhs };
                    if let Some((inner, held)) = self.flag_comparison(other) {
                        let truth = (held == (*pred == IntPred::Ne)) == taken;
                        return self.condition_range(value, inner, truth, depth / 2, budget);
                    }
                }
                for k in 0..2 {
                    let (side, other) = if k == 0 { (*lhs, *rhs) } else { (*rhs, *lhs) };
                    let Some(c) = self.const_value(other) else {
                        continue;
                    };
                    let pred = if k == 1 { swap(*pred) } else { *pred };
                    let pred = if taken { pred } else { negate(pred) };
                    let region = ConstantRange::icmp_region(pred, c, w);
                    if side == value {
                        return Some(region.trunc(width));
                    }
                    // `and(X, Want)` against a dominating `and(X, Have)`.
                    if let (Some(Inst::Bin { op: BinOp::And, lhs: a0, rhs: a1, .. }), Some(Inst::Bin { op: BinOp::And, lhs: b0, rhs: b1, .. })) =
                        (self.f.inst(value), self.f.inst(side))
                    {
                        if a0 == b0 {
                            if let (Some(want), Some(have)) =
                                (self.const_value(*a1), self.const_value(*b1))
                            {
                                return masked_region(want, have, &region, w);
                            }
                        }
                    }
                    if c == 0 && matches!(pred, IntPred::Eq | IntPred::Ne) {
                        return self.condition_range(value, side, pred == IntPred::Ne, depth / 2, budget);
                    }
                }
                None
            }
            _ => None,
        }
    }

    /// `and(sub(0, zext(cmp)), flags)` names the comparison `cmp` — a
    /// non-zero result means it held. Returns `cmp` and whether the flag is
    /// set when the comparison is true (always so here; the sign of the
    /// answer lives in the caller's polarity).
    fn flag_comparison(&self, v: ValueId) -> Option<(ValueId, bool)> {
        let Some(Inst::Bin { op: BinOp::And, lhs, .. }) = self.f.inst(v) else {
            return None;
        };
        let Some(Inst::Bin { op: BinOp::Sub, lhs: z, rhs: c, .. }) = self.f.inst(*lhs) else {
            return None;
        };
        if self.const_value(*z) != Some(0) {
            return None;
        }
        let mut cur = *c;
        for _ in 0..4 {
            match self.f.inst(cur) {
                Some(Inst::Zext { value, .. })
                | Some(Inst::Trunc { value, .. })
                | Some(Inst::Sext { value, .. }) => cur = *value,
                _ => break,
            }
        }
        matches!(self.f.inst(cur), Some(Inst::Icmp { .. })).then_some((cur, true))
    }

    /// The block a value is defined in.
    fn block_of(&self, v: ValueId) -> Option<BlockId> {
        self.f.defined_in.get(v.0 as usize).copied().flatten()
    }

    /// A value from another function's arena names nothing here.
    fn in_arena(&self, v: ValueId) -> bool {
        (v.0 as usize) < self.f.values.len()
    }

    fn width_of(&self, v: ValueId) -> Option<u32> {
        if !self.in_arena(v) {
            return None;
        }
        match self.f.ty(v) {
            Type::Int(b) if b <= 64 => Some(b),
            _ => None,
        }
    }

    fn is_int(&self, v: ValueId) -> bool {
        self.in_arena(v) && matches!(self.f.ty(v), Type::Int(_))
    }

    fn const_value(&self, v: ValueId) -> Option<u128> {
        if !self.in_arena(v) {
            return None;
        }
        match self.f.value(v) {
            Value::Const(c) => c.as_u128(),
            _ => None,
        }
    }

    fn dominates(&self, a: BlockId, b: BlockId) -> bool {
        self.doms.dominates(a, b)
    }
}

/// Bound `and(X, want)` using a dominating sibling `and(X, have)` fact.
/// When the free bits form a low-bit mask, a contiguous `[0, H)` region is
/// rounded up to their tile size — every masked value is covered.
fn masked_region(
    want: u128,
    have: u128,
    region: &ConstantRange,
    width: u32,
) -> Option<ConstantRange> {
    if region.full || region.lo != 0 {
        return None;
    }
    let free = want & !have;
    let stride = free.wrapping_add(1);
    if stride == 0 || !stride.is_power_of_two() {
        return None;
    }
    let m = biara_ir::analysis::mask(width);
    if region.hi & !free == 0 || region.hi > m {
        return None;
    }
    let upper = (region.hi | free) & m;
    if upper < region.hi {
        return None;
    }
    Some(ConstantRange::new(0, upper, width))
}

fn swap(p: IntPred) -> IntPred {
    match p {
        IntPred::Slt => IntPred::Sgt,
        IntPred::Sle => IntPred::Sge,
        IntPred::Sgt => IntPred::Slt,
        IntPred::Sge => IntPred::Sle,
        other => other,
    }
}

fn negate(p: IntPred) -> IntPred {
    match p {
        IntPred::Eq => IntPred::Ne,
        IntPred::Ne => IntPred::Eq,
        IntPred::Slt => IntPred::Sge,
        IntPred::Sle => IntPred::Sgt,
        IntPred::Sgt => IntPred::Sle,
        IntPred::Sge => IntPred::Slt,
        IntPred::Ult => IntPred::Uge,
        IntPred::Ule => IntPred::Ugt,
        IntPred::Ugt => IntPred::Ule,
        IntPred::Uge => IntPred::Ult,
    }
}
