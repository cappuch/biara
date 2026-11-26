//! Indirect call/jump resolution on lifted IR, ported from
//! `cpp/jump_tables.cpp`.
//!
//! Each site is announced by `call void @biara.target(i64 %pc, i64 site)`.
//! For every such call this looks for the single non-constant value feeding
//! `%pc` whose range at the call is bounded, then evaluates `%pc` for every
//! value in that range, reading constant memory (the table) through a
//! callback into the guest image.
//!
//! The baseline range came from LLVM's `LazyValueInfo`; here it is the
//! definition's own range narrowed by [`crate::ranges::EdgeRanges`] — the same
//! refiner audit argument ranges use, so there is one implementation.
//!
//! Limits are explicit and reported: an evaluation that cannot finish sets
//! `complete = false`, and an enumeration above `max_entries` sets
//! `limited = true`. Neither silently claims exhaustiveness.

use biara_ir::{
    BinOp, BlockId, Constant, ConstantRange, Dominators, Function, Inst, IntPred, Terminator, Type,
    Value, ValueId,
};
use std::collections::{BTreeMap, BTreeSet};

/// The call that announces a site.
pub const MARKER: &str = "biara.target";

const MAX_DEPTH: u32 = 32;

/// How one site resolved.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct SiteResolution {
    pub targets: BTreeSet<u64>,
    /// True when every possible target was enumerated.
    pub complete: bool,
    /// True when an enumeration limit stopped the work.
    pub limited: bool,
}

/// Read `len` bytes of guest memory at `addr`.
pub type ReadFn<'a> = &'a dyn Fn(u64, usize) -> Option<Vec<u8>>;

/// Every site in `f`, with the targets each can reach.
pub fn indirect_targets(f: &Function, read: ReadFn<'_>, max_entries: u64) -> BTreeMap<u64, SiteResolution> {
    let mut out: BTreeMap<u64, SiteResolution> = BTreeMap::new();
    let sites = marker_sites(f);
    if sites.is_empty() {
        return out;
    }
    let doms = Dominators::new(f);
    let edges = crate::ranges::EdgeRanges::new(f);
    for (call, site, root) in sites {
        let mut res = resolve_site(f, &doms, &edges, call, root, read, max_entries);
        res.complete &= out.get(&site).map(|r| r.complete).unwrap_or(true);
        let entry = out.entry(site).or_default();
        entry.targets.extend(std::mem::take(&mut res.targets));
        entry.complete = res.complete;
        entry.limited |= res.limited;
    }
    // Keep sites whose marker had no constant site id out of the result.
    out
}

/// The `biara.target(pc, site)` calls of `f`.
fn marker_sites(f: &Function) -> Vec<(ValueId, u64, ValueId)> {
    let mut out = Vec::new();
    for b in &f.blocks {
        for &i in &b.insts {
            let Some(Inst::Call { callee, args, .. }) = f.inst(i) else {
                continue;
            };
            // Value ids are per function: an id outside this arena is not
            // this instruction's operand.
            if (*callee).0 as usize >= f.values.len() || args.iter().any(|a| a.0 as usize >= f.values.len()) {
                continue;
            }
            let is_marker = matches!(
                f.value(*callee),
                Value::Const(Constant::Function(n)) if n == MARKER
            );
            if !is_marker || args.len() < 2 {
                continue;
            }
            // The site id is the second argument and must be constant.
            let Some(Value::Const(c)) = Some(f.value(args[1])) else {
                continue;
            };
            let Some(site) = c.as_u128() else { continue };
            out.push((i, site as u64, args[0]));
        }
    }
    out
}

fn resolve_site(
    f: &Function,
    doms: &Dominators,
    edges: &crate::ranges::EdgeRanges<'_>,
    call: ValueId,
    root: ValueId,
    read: ReadFn<'_>,
    max_entries: u64,
) -> SiteResolution {
    let mut res = SiteResolution {
        targets: BTreeSet::new(),
        complete: true,
        limited: false,
    };
    let ev = Evaluator {
        f,
        read,
        holes: Vec::new(),
    };
    // A fully constant target needs no enumeration.
    if let Some(direct) = ev.eval(root, 0) {
        res.targets.insert(direct as u64);
        return res;
    }
    let chain = variable_chain(f, &ev, root);
    if chain.is_empty() {
        res.complete = false;
        return res;
    }

    // Pick the node with the fewest candidate values.
    let mut best: Option<(ValueId, ConstantRange, u64)> = None;
    for &n in chain.iter().rev() {
        // Value ids are per function: an id from another arena has no type
        // here and cannot be a candidate.
        if n.0 as usize >= f.values.len() {
            continue;
        }
        let Some(width) = f.ty(n).int_bits() else {
            continue;
        };
        if width > 64 {
            continue;
        }
        let m = biara_ir::analysis::mask(width);
        // A full baseline cannot be narrowed by refinement, so start from the
        // dominating guard's bound when there is one.
        let call_block = f
            .defined_in
            .get(call.0 as usize)
            .copied()
            .flatten()
            .unwrap_or(BlockId(0));
        // A guard above the site bounds the value on the paths that reach
        // it; failing that, the value's own definition may bound it (a mask
        // is always within its mask), which needs no guard at all.
        let base = edges
            .guard_range(n, call_block)
            .or_else(|| edges.defined_range(n))
            .unwrap_or_else(|| ConstantRange::full(width));
        let r = edges.refine(n, call, base);
        // The enumerable node of a jump table is the table's index, which the
        // IR never bounds: it is read from guest memory. Two shapes say a
        // load is that index rather than an incidental value:
        //
        //  * a `shl` scales it into a byte offset, so one unit is one entry;
        //  * a dominating guard compares it against the table's last index,
        //    so `guard_range` already narrowed it.
        //
        // Either way the window is bounded here and the entries are checked
        // by `tables` against the bodies the site can reach; a target outside
        // them is dropped and the site is reported incomplete. The program
        // counter itself is never a candidate: enumerating it would walk byte
        // offsets and invent a case per byte.
        let indexed = n != root
            && matches!(f.inst(n), Some(Inst::Load { .. }))
            && (chain.iter().any(|&c| {
                matches!(
                    f.inst(c),
                    Some(Inst::Bin { op: BinOp::Shl, lhs, .. })
                        if edges.same_under_casts(*lhs, n)
                )
            }) || edges.guard_range(n, call_block).is_some());
        let candidate: (ValueId, ConstantRange, u64) = if !(r.full || r.width == 0) {
            let span = (r.hi.wrapping_sub(r.lo) & m).wrapping_add(1);
            let step = stride_of(f, n).max(1);
            let count = span.div_ceil(step);
            if count > max_entries as u128 {
                res.limited = true;
                continue;
            }
            (n, r, count as u64)
        } else if indexed {
            // The window is a probe, not a proof: an index outside it would
            // be missed, so the site never claims to be exhaustive.
            let window = max_entries.min(64) as u128;
            res.limited = true;
            res.complete = false;
            (n, ConstantRange::new(0, window.saturating_sub(1), width), window as u64)
        } else {
            continue;
        };
        best = Some(match best {
            None => candidate,
            Some((_, br, bc)) if candidate.2 < bc => candidate,
            Some(other) => other,
        });
    }
    let Some((node, range, _)) = best else {
        res.complete = false;
        return res;
    };

    // Evaluate the root for every candidate, keeping the feasible ones.
    let step = stride_of(f, node).max(1);
    let mut k = range.lo;
    let mut seen = 0u64;
    loop {
        let ev = Evaluator {
            f,
            read,
            holes: vec![(node, k)],
        };
        match ev.eval(root, 0) {
            Some(target) => {
                res.targets.insert(target as u64);
            }
            None => res.complete = false,
        }
        seen += 1;
        if k == range.hi || seen > max_entries {
            if k != range.hi {
                res.limited = true;
                res.complete = false;
            }
            break;
        }
        k = k.wrapping_add(step) & biara_ir::analysis::mask(range.width);
    }
    let _ = doms;
    res
}

/// The power-of-two stride a value's low bits imply (a `shl` by a constant).
fn stride_of(f: &Function, v: ValueId) -> u128 {
    if let Some(Inst::Bin {
        op: BinOp::Shl,
        rhs,
        ..
    }) = f.inst(v)
    {
        if let Some(Value::Const(c)) = Some(f.value(*rhs)) {
            if let Some(n) = c.as_u128() {
                if n < 64 {
                    return 1u128 << n;
                }
            }
        }
    }
    1
}

/// The chain from `root` down to its single non-constant leaf, root first.
fn variable_chain(f: &Function, ev: &Evaluator<'_, '_>, root: ValueId) -> Vec<ValueId> {
    let mut chain = Vec::new();
    let mut v = root;
    for _ in 0..=MAX_DEPTH {
        chain.push(v);
        if matches!(
            f.inst(v),
            Some(Inst::Phi { .. }) | Some(Inst::Select { .. }) | Some(Inst::Call { .. })
        ) {
            return chain;
        }
        // A load straight through a parameter reads a guest register: that
        // is a leaf. A load through any computed address (a constant base
        // plus a variable index) is a table probe, so the chain descends
        // into it and the index becomes the enumerate variable.
        if let Some(Inst::Load { ptr, .. }) = f.inst(v)
            && ((*ptr).0 as usize) < f.values.len()
            && matches!(f.value(*ptr), Value::Param { .. })
        {
            return chain;
        }
        let Some(inst) = f.inst(v) else { return chain };
        // A store produces no value: the chain reached it by walking an
        // operand of something else, and the value it wrote is not an
        // expression this site evaluates. Stop here rather than descending
        // into the store's own operands.
        if matches!(inst, Inst::Store { .. }) {
            return chain;
        }
        // A load of a local slot reads what was stored there: follow the
        // stored value, not the slot's address. Descending into the alloca
        // would make the slot itself the enumerate variable, which no
        // enumeration can resolve.
        if let Inst::Load { ptr, .. } = inst
            && ((*ptr).0 as usize) < f.values.len()
            && matches!(f.inst(*ptr), Some(Inst::Alloca { .. }))
        {
            match ev.last_store(*ptr, v) {
                Some(stored) => {
                    v = stored;
                    continue;
                }
                None => return chain,
            }
        }
        let mut var = None;
        for op in biara_ir::operands(inst) {
            if ev.eval(op, 0).is_some() {
                continue;
            }
            if var.is_some() {
                return Vec::new();
            }
            var = Some(op);
        }
        match var {
            None => return chain,
            Some(n) => v = n,
        }
    }
    Vec::new()
}

/// A symbolic evaluator with some values fixed.
struct Evaluator<'a, 'f> {
    f: &'f Function,
    read: ReadFn<'a>,
    holes: Vec<(ValueId, u128)>,
}

impl<'f> Evaluator<'_, 'f> {
    /// The value most recently stored to `slot` before instruction `at`.
    /// A store earlier in `at`'s own block settles it. Otherwise every
    /// predecessor must agree, walking back to the entry: a path on which
    /// the slot holds something else makes the load unresolvable here
    /// rather than silently taking one side.
    fn last_store(&self, slot: ValueId, at: ValueId) -> Option<ValueId> {
        let b = self.f.defined_in.get(at.0 as usize).copied().flatten()?;
        if let Some(v) = self.store_before(slot, b, Some(at))? {
            return Some(v);
        }
        let mut pending = vec![b];
        let mut seen = std::collections::BTreeSet::new();
        let mut agreed: Option<ValueId> = None;
        let mut budget = 64u32;
        while let Some(block) = pending.pop() {
            if budget == 0 {
                return None;
            }
            budget -= 1;
            let entry = Some(block) == self.f.entry;
            let Some(v) = self.store_before(slot, block, None)? else {
                if entry {
                    return None;
                }
                for &p in &self.f.blocks[block.0 as usize].preds {
                    if seen.insert(p) {
                        pending.push(p);
                    }
                }
                continue;
            };
            match agreed {
                None => agreed = Some(v),
                Some(a) if a == v => {}
                Some(_) => return None,
            }
            if entry {
                break;
            }
        }
        agreed
    }

    /// The last store to `slot` in `block`, scanning back from `at` (or from
    /// the block's end when `at` is `None`). `Some(None)` means the block has
    /// no such store; `None` means the scan could not be trusted.
    fn store_before(
        &self,
        slot: ValueId,
        block: biara_ir::BlockId,
        at: Option<ValueId>,
    ) -> Option<Option<ValueId>> {
        let insts = &self.f.blocks[block.0 as usize].insts;
        let end = match at {
            Some(a) => insts.iter().position(|&i| i == a)?,
            None => insts.len(),
        };
        for &i in insts[..end].iter().rev() {
            if let Some(Inst::Store { ptr, value, .. }) = self.f.inst(i)
                && *ptr == slot
            {
                return Some(Some(*value));
            }
        }
        Some(None)
    }

    fn eval(&self, v: ValueId, depth: u32) -> Option<u128> {
        if depth > MAX_DEPTH {
            return None;
        }
        if let Some(&(_, x)) = self.holes.iter().find(|(h, _)| *h == v) {
            return Some(x);
        }
        let width = self.width_of(v)?;
        let m = biara_ir::analysis::mask(width);
        match self.f.value(v) {
            Value::Const(Constant::Int { bits, .. }) => Some(bits & m),
            Value::Const(Constant::Zero) => Some(0),
            Value::Const(_) | Value::Param { .. } => None,
            Value::Inst(inst) => match inst {
                Inst::Bin { op, ty, lhs, rhs } => {
                    let a = self.eval(*lhs, depth + 1)?;
                    let b = self.eval(*rhs, depth + 1)?;
                    eval_bin(*op, a, b, ty.int_bits()?)
                }
                Inst::Trunc { to, value, .. } => {
                    let w = to.int_bits()?;
                    Some(self.eval(*value, depth + 1)? & biara_ir::analysis::mask(w))
                }
                Inst::Zext { value, .. } => self.eval(*value, depth + 1),
                Inst::Sext { from, to, value } => {
                    let fw = from.int_bits()?;
                    let tw = to.int_bits()?;
                    let a = self.eval(*value, depth + 1)?;
                    Some(biara_ir::analysis::signed(a, fw) as u128 & biara_ir::analysis::mask(tw))
                }
                Inst::Cast { value, .. } => self.eval(*value, depth + 1),
                Inst::Gep { base, offset } => {
                    Some(self.eval(*base, depth + 1)?.wrapping_add(*offset as u128))
                }
                Inst::Icmp { pred, ty, lhs, rhs } => {
                    let w = ty.int_bits()?;
                    let a = self.eval(*lhs, depth + 1)?;
                    let b = self.eval(*rhs, depth + 1)?;
                    Some(u128::from(cmp(*pred, a, b, w)))
                }
                Inst::Select { cond, yes, no, .. } => {
                    let c = self.eval(*cond, depth + 1)?;
                    self.eval(if c != 0 { *yes } else { *no }, depth + 1)
                }
                Inst::Load { ty, ptr, .. } => {
                    let w = ty.int_bits()?;
                    // A load of a local slot reads what the last store to it
                    // wrote. The lifter spills every value this way, so a
                    // table base or an index is usually one load from where
                    // it was defined, not in guest memory at all.
                    let local = ((*ptr).0 as usize) < self.f.values.len()
                        && matches!(self.f.inst(*ptr), Some(Inst::Alloca { .. }));
                    if let Some(slot) = local.then_some(*ptr) {
                        if let Some(v) = self.last_store(slot, v) {
                            return self.eval(v, depth + 1);
                        }
                        return None;
                    }
                    let addr = self.eval(*ptr, depth + 1)?;
                    let bytes = (w as usize).div_ceil(8);
                    if bytes > 8 {
                        return None;
                    }
                    let buf = (self.read)(addr as u64, bytes)?;
                    let mut out = 0u128;
                    for (i, b) in buf.iter().take(bytes).enumerate() {
                        out |= (*b as u128) << (8 * i);
                    }
                    Some(out & biara_ir::analysis::mask(w))
                }
                Inst::Freeze { value, .. } => self.eval(*value, depth + 1),
                _ => None,
            },
        }
    }

    fn width_of(&self, v: ValueId) -> Option<u32> {
        // An id from another function's arena names no value here.
        if v.0 as usize >= self.f.values.len() {
            return None;
        }
        match self.f.ty(v) {
            Type::Int(b) => Some(b),
            Type::Ptr => Some(64),
            _ => None,
        }
    }
}

fn eval_bin(op: BinOp, a: u128, b: u128, width: u32) -> Option<u128> {
    let m = biara_ir::analysis::mask(width);
    Some(match op {
        BinOp::Add => a.wrapping_add(b) & m,
        BinOp::Sub => a.wrapping_sub(b) & m,
        BinOp::Mul => a.wrapping_mul(b) & m,
        BinOp::And => a & b,
        BinOp::Or => a | b,
        BinOp::Xor => a ^ b,
        BinOp::Shl => {
            if b >= width as u128 {
                0
            } else {
                (a << b) & m
            }
        }
        BinOp::Lshr => {
            if b >= width as u128 {
                0
            } else {
                (a & m) >> b
            }
        }
        BinOp::Ashr => {
            let sh = b.min((width - 1) as u128) as u32;
            (biara_ir::analysis::signed(a, width) >> sh) as u128 & m
        }
        BinOp::Udiv | BinOp::Sdiv | BinOp::Urem | BinOp::Srem => return None,
    })
}

fn cmp(pred: IntPred, a: u128, b: u128, width: u32) -> bool {
    let m = biara_ir::analysis::mask(width);
    let (a, b) = (a & m, b & m);
    let (sa, sb) = (
        biara_ir::analysis::signed(a, width),
        biara_ir::analysis::signed(b, width),
    );
    match pred {
        IntPred::Eq => a == b,
        IntPred::Ne => a != b,
        IntPred::Slt => sa < sb,
        IntPred::Sle => sa <= sb,
        IntPred::Sgt => sa > sb,
        IntPred::Sge => sa >= sb,
        IntPred::Ult => a < b,
        IntPred::Ule => a <= b,
        IntPred::Ugt => a > b,
        IntPred::Uge => a >= b,
    }
}
