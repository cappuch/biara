//! Jump tables in the C output, ported from `cpp/index_switches.cpp`.
//!
//! The lifted code switches on the jump target it loaded from a table in
//! guest memory. Where that target is a function of one bounded value (the
//! index), evaluating it for every index and switching on the index instead
//! gives `switch (i) { case 0: … }` rather than a switch on code addresses.
//!
//! Two additional rewrites fall out of the same walk:
//! - a case set that is a scaled index (`switch (x << s & m)` with every case
//!   a multiple of `1 << s`) reads as a switch on `x & (m >> s)`;
//! - a switch whose dominating guard bounds its condition and whose cases
//!   cover that bound cannot take its default, so the default goes away.

use biara_ir::{
    BinOp, BlockId, Constant, ConstantRange, Dominators, Function, Inst, IntPred, Terminator, Type,
    Value, ValueId,
};
use std::collections::{BTreeMap, BTreeSet};

/// Largest index count evaluated in full.
const MAX_INDICES: u64 = 4096;
/// The same bound as `u128`, for range spans.
const MAX_INDICES_U: u128 = MAX_INDICES as u128;
/// Nesting cap for the expression walk.
const MAX_DEPTH: u32 = 32;

/// Read `len` bytes of guest memory at `addr`; `false` when unmapped.
pub type ReadFn<'a> = &'a dyn Fn(u64, usize) -> Option<Vec<u8>>;

/// Rewrites applied to one function.
#[derive(Default, Debug, PartialEq, Eq)]
pub struct Rewrites {
    pub indices: usize,
    pub unscaled: usize,
    pub guarded_defaults: usize,
}

impl Rewrites {
    pub fn total(&self) -> usize {
        self.indices + self.unscaled + self.guarded_defaults
    }
}

/// Rewrite every table switch of `f`.
pub fn index_switches(f: &mut Function, read: ReadFn<'_>) -> Rewrites {
    let mut out = Rewrites::default();
    // One block at a time: a rewrite changes the CFG, so blocks are revisited
    // in a fresh pass rather than iterated while mutating.
    // Each block is rewritten at most once. A rewritten switch is on the
    // index, which a guard still bounds, so a second pass would find it
    // again and report progress for ever.
    let mut done: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
    let mut progress = true;
    while progress {
        progress = false;
        for b in 0..f.blocks.len() {
            let BlockId(block) = BlockId(b as u32);
            if done.contains(&block) {
                continue;
            }
            let Some(Terminator::Switch {
                value,
                default,
                cases,
            }) = f.blocks[block as usize].term.clone()
            else {
                continue;
            };
            if unscaled(f, block, value, &cases) {
                out.unscaled += 1;
                progress = true;
                continue;
            }
            if guard_default(f, block, value, &cases) {
                out.guarded_defaults += 1;
                progress = true;
                continue;
            }
            if rewrite_index(f, block, value, default, &cases, read) {
                out.indices += 1;
                progress = true;
                done.insert(block);
            }
        }
    }
    out
}

/// Ops the evaluator understands.
fn supported(f: &Function, v: ValueId) -> bool {
    match f.value(v) {
        Value::Const(Constant::Int { .. }) => true,
        Value::Const(_) => false,
        Value::Param { .. } => false,
        Value::Inst(inst) => matches!(
            inst,
            Inst::Bin { .. }
                | Inst::Un { .. }
                | Inst::Rot { .. }
                | Inst::Trunc { .. }
                | Inst::Zext { .. }
                | Inst::Sext { .. }
                | Inst::Cast { .. }
                | Inst::Load { .. }
                | Inst::Gep { .. }
                | Inst::Freeze { .. }
                | Inst::Call { .. }
        ),
    }
}

/// The non-constant values `v` depends on that the evaluator cannot compute,
/// or that it is told to stop at.
fn leaves(f: &Function, v: ValueId, stop: Option<ValueId>, out: &mut BTreeSet<ValueId>, depth: u32) {
    if depth > MAX_DEPTH || matches!(f.value(v), Value::Const(_)) {
        return;
    }
    if Some(v) == stop || !supported(f, v) {
        out.insert(v);
        return;
    }
    if let Some(inst) = f.inst(v) {
        for op in biara_ir::operands(inst) {
            leaves(f, op, stop, out, depth + 1);
        }
    }
}

/// A symbolic evaluator: computes a value's constant result with one node
/// fixed to a chosen index.
struct Eval<'a, 'f> {
    f: &'f Function,
    read: ReadFn<'a>,
    fixed: Option<ValueId>,
    fixed_value: u128,
}

impl<'f> Eval<'_, 'f> {
    fn eval(&self, v: ValueId, depth: u32) -> Option<u128> {
        if depth > MAX_DEPTH {
            return None;
        }
        if Some(v) == self.fixed {
            return Some(self.fixed_value);
        }
        let bits = self.bits_of(v)?;
        match self.f.value(v) {
            Value::Const(Constant::Int { bits: b, width }) => {
                Some(*b & biara_ir::analysis::mask(*width))
            }
            Value::Const(Constant::Zero) => Some(0),
            Value::Const(_) | Value::Param { .. } => None,
            Value::Inst(inst) => match inst {
                Inst::Trunc { value, .. } => {
                    let a = self.eval(*value, depth + 1)?;
                    Some(a & biara_ir::analysis::mask(bits))
                }
                Inst::Zext { value, .. } => self.eval(*value, depth + 1),
                Inst::Sext { from, value, .. } => {
                    let fw = from.int_bits()?;
                    let a = self.eval(*value, depth + 1)?;
                    Some(biara_ir::analysis::signed(a, fw) as u128 & biara_ir::analysis::mask(bits))
                }
                Inst::Cast { value, .. } => self.eval(*value, depth + 1),
                Inst::Freeze { value, .. } => self.eval(*value, depth + 1),
                Inst::Gep { base, offset } => {
                    let b = self.eval(*base, depth + 1)?;
                    Some(b.wrapping_add(*offset as u128))
                }
                Inst::Load { ty, ptr, .. } => {
                    let addr = self.eval(*ptr, depth + 1)?;
                    let width = ty.int_bits()?;
                    let bytes = (width as usize).div_ceil(8);
                    if bytes > 8 {
                        return None;
                    }
                    let buf = (self.read)(addr as u64, bytes)?;
                    let mut out = 0u128;
                    for (i, b) in buf.iter().take(bytes).enumerate() {
                        out |= (*b as u128) << (8 * i);
                    }
                    Some(out & biara_ir::analysis::mask(width))
                }
                Inst::Rot {
                    left,
                    ty,
                    value,
                    amount,
                } => {
                    let w = ty.int_bits()?;
                    let a = self.eval(*value, depth + 1)?;
                    let n = self.eval(*amount, depth + 1)?;
                    let n = (n % w as u128) as u32;
                    let m = biara_ir::analysis::mask(w);
                    Some(if *left {
                        ((a << n) | (a >> ((w - n) % w))) & m
                    } else {
                        ((a >> n) | (a << ((w - n) % w))) & m
                    })
                }
                Inst::Call { .. } => None,
                Inst::Bin { op, ty, lhs, rhs } => {
                    let a = self.eval(*lhs, depth + 1)?;
                    let b = self.eval(*rhs, depth + 1)?;
                    let w = ty.int_bits()?;
                    bin(*op, a, b, w)
                }
                _ => None,
            },
        }
    }

    fn bits_of(&self, v: ValueId) -> Option<u32> {
        match self.f.ty(v) {
            Type::Int(b) => Some(b),
            Type::Ptr => Some(64),
            _ => None,
        }
    }
}

fn bin(op: BinOp, a: u128, b: u128, width: u32) -> Option<u128> {
    let m = biara_ir::analysis::mask(width);
    Some(match op {
        BinOp::Add => a.wrapping_add(b) & m,
        BinOp::Sub => a.wrapping_sub(b) & m,
        BinOp::Mul => a.wrapping_mul(b) & m,
        BinOp::And => a & b,
        BinOp::Or => a | b,
        BinOp::Xor => a ^ b,
        _ if b >= width as u128 => return None,
        BinOp::Shl => (a << b) & m,
        BinOp::Lshr => (a & m) >> b,
        BinOp::Ashr => (biara_ir::analysis::signed(a, width) >> b) as u128 & m,
        BinOp::Udiv | BinOp::Sdiv | BinOp::Urem | BinOp::Srem => return None,
    })
}

/// A range for `v` from a dominating `if` guard, walking the
/// single-predecessor chain.
fn guard_range(f: &Function, doms: &Dominators, block: BlockId, v: ValueId) -> Option<ConstantRange> {
    let mut b = block;
    for _ in 0..16 {
        // The sole predecessor of `b`.
        let mut pred = None;
        for (i, blk) in f.blocks.iter().enumerate() {
            if blk.term.as_ref().map(biara_ir::successor_blocks).unwrap_or_default().contains(&b) {
                if pred.is_some() {
                    return None;
                }
                pred = Some(BlockId(i as u32));
            }
        }
        let p = pred?;
        let Some(Terminator::CondBr { cond, yes, no }) = f.blocks[p.0 as usize].term.clone() else {
            return None;
        };
        // `b` must be the false arm of a guard.
        if yes == b || no != b {
            return None;
        }
        if let Some(Inst::Icmp {
            pred: ip,
            ty,
            lhs,
            rhs,
        }) = f.inst(cond)
        {
            let w = ty.int_bits()?;
            if *lhs == v {
                if let Some(k) = const_value(f, *rhs) {
                    let bound = k & biara_ir::analysis::mask(w);
                    return match ip {
                        IntPred::Ugt => Some(ConstantRange::new(0, bound, w)),
                        IntPred::Uge => {
                            Some(ConstantRange::new(0, bound.wrapping_sub(1), w))
                        }
                        _ => None,
                    };
                }
            }
        }
        let _ = doms;
        b = p;
    }
    None
}

/// `switch ((x << s) & m)` with every case a multiple of `1 << s` reads as
/// `switch (x & (m >> s))` with divided cases.
fn unscaled(f: &mut Function, block: u32, value: ValueId, cases: &[(u128, BlockId)]) -> bool {
    let Some(Inst::Bin {
        op: BinOp::And,
        ty,
        lhs,
        rhs,
    }) = f.inst(value)
    else {
        return false;
    };
    let (shifted, mask_const) = (*lhs, *rhs);
    let Some(Inst::Bin {
        op: BinOp::Shl,
        lhs: x,
        rhs: sh,
        ..
    }) = f.inst(shifted)
    else {
        return false;
    };
    let (x, sh) = (*x, *sh);
    let (Some(mask), Some(shamt)) = (const_value(f, mask_const), const_value(f, sh))
    else {
        return false;
    };
    let width = ty.int_bits().unwrap_or(64);
    let m = biara_ir::analysis::mask(width);
    let shamt = shamt as u32;
    if shamt == 0 || shamt >= width {
        return false;
    }
    let step = 1u128 << shamt;
    if cases.iter().any(|(k, _)| (k & (m)) % step != 0) {
        return false;
    }
    // Rewrite in place: condition becomes `x & (mask >> shamt)`, cases divided.
    let new_cond = f.add_const(Constant::Int {
        bits: mask >> shamt,
        width,
    });
    let and = f.add_const(Constant::Zero); // placeholder, replaced below
    let and_id = {
        let inst = Inst::Bin {
            op: BinOp::And,
            ty: Type::Int(width),
            lhs: x,
            rhs: new_cond,
        };
        f.values.push(Value::Inst(inst));
        f.defined_in.push(Some(BlockId(block)));
        f.pc.push(None);
        let id = ValueId((f.values.len() - 1) as u32);
        // Insert before the terminator: the switch reads it.
        f.blocks[block as usize].insts.push(id);
        id
    };
    let _ = and;
    let Some(Terminator::Switch { default, .. }) = f.blocks[block as usize].term.clone() else {
        return false;
    };
    f.blocks[block as usize].term = Some(Terminator::Switch {
        value: and_id,
        default,
        cases: cases.iter().map(|(k, t)| (k >> shamt, *t)).collect(),
    });
    true
}

/// A switch whose dominating guard bounds its condition and whose cases cover
/// that bound cannot take its default.
fn guard_default(f: &mut Function, block: u32, value: ValueId, cases: &[(u128, BlockId)]) -> bool {
    let width = match f.ty(value) {
        Type::Int(b) if b <= 64 => b,
        _ => return false,
    };
    let doms = Dominators::new(f);
    let b = BlockId(block);
    let Some(r) = guard_range(f, &doms, b, value) else {
        return false;
    };
    if r.full {
        return false;
    }
    let m = biara_ir::analysis::mask(width);
    let span = (r.hi.wrapping_sub(r.lo) & m).wrapping_add(1);
    if span != cases.len() as u128 || span > MAX_INDICES_U {
        return false;
    }
    if cases.iter().any(|(k, _)| {
        let v = k & m;
        v < r.lo || v > r.hi
    }) {
        return false;
    }
    // Point the default at an unreachable block.
    let Some(Terminator::Switch { default, .. }) = f.blocks[block as usize].term.clone() else {
        return false;
    };
    let unreachable = biara_ir::BlockId(f.blocks.len() as u32);
    // The name must be unique within the function: a function with two
    // rewritten switches would otherwise hold two blocks called
    // "unreachable" and every reference to one would be ambiguous.
    f.blocks.push(biara_ir::Block {
        name: format!("unreachable{}", unreachable.0),
        insts: Vec::new(),
        term: Some(Terminator::Unreachable),
        preds: Default::default(),
    });
    if let Some(Terminator::Switch { default: d, .. }) = &mut f.blocks[block as usize].term {
        let _ = default;
        *d = unreachable;
    }
    biara_ir::analysis::recompute_preds(f);
    true
}

/// Rewrite a table switch to switch on its index. Returns whether it did.
fn rewrite_index(
    f: &mut Function,
    block: u32,
    cond: ValueId,
    _default: BlockId,
    cases: &[(u128, BlockId)],
    read: ReadFn<'_>,
) -> bool {
    if matches!(f.value(cond), Value::Const(_)) || !supported(f, cond) {
        return false;
    }
    // Only switches on something loaded: that is the table.
    let mut nodes = vec![cond];
    let mut loads = false;
    let mut i = 0;
    while i < nodes.len() && i < 256 {
        let n = nodes[i];
        i += 1;
        if matches!(f.inst(n), Some(Inst::Load { .. })) {
            loads = true;
        }
        if matches!(f.value(n), Value::Const(_)) || !supported(f, n) {
            continue;
        }
        if let Some(inst) = f.inst(n) {
            for op in biara_ir::operands(inst) {
                if !matches!(f.value(op), Value::Const(_)) {
                    nodes.push(op);
                }
            }
        }
    }
    if !loads {
        return false;
    }

    let doms = Dominators::new(f);
    let b = BlockId(block);
    let mut best: Option<(ValueId, ConstantRange)> = None;
    for &n in &nodes {
        if matches!(f.inst(n), Some(Inst::Load { .. })) {
            continue;
        }
        let Some(width) = f.ty(n).int_bits() else {
            continue;
        };
        if width > 64 {
            continue;
        }
        let m = biara_ir::analysis::mask(width);
        // The candidate's range: the definition, narrowed by a guard.
        let mut r = ConstantRange::full(width);
        if let Some(g) = guard_range(f, &doms, b, n) {
            r = g;
        }
        if r.full || r.width == 0 {
            continue;
        }
        let span = (r.hi.wrapping_sub(r.lo) & m).wrapping_add(1);
        if span > MAX_INDICES_U {
            continue;
        }
        best = Some(match best {
            None => (n, r),
            Some((_, br)) => {
                let bspan = (br.hi.wrapping_sub(br.lo) & m).wrapping_add(1);
                if span < bspan {
                    (n, r)
                } else {
                    (n, br)
                }
            }
        });
    }
    let Some((x, r)) = best else { return false };

    // Evaluate the condition for every index in the candidate's range.
    let mut rewired: BTreeMap<u128, BlockId> = BTreeMap::new();
    let step = 1u128;
    let mut k = r.lo;
    loop {
        let mut ev = Eval {
            f,
            read,
            fixed: Some(x),
            fixed_value: k,
        };
        let Some(target) = ev.eval(cond, 0) else {
            return false;
        };
        let dest = cases
            .iter()
            .find(|(c, _)| *c == target)
            .map(|(_, d)| *d)
            .unwrap_or(_default);
        if dest != _default {
            rewired.insert(k, dest);
        }
        if k == r.hi {
            break;
        }
        k = k.wrapping_add(step);
    }
    if rewired.is_empty() {
        return false;
    }
    if let Some(Terminator::Switch { .. }) = f.blocks[block as usize].term {
        f.blocks[block as usize].term = Some(Terminator::Switch {
            value: x,
            default: _default,
            cases: rewired.into_iter().collect(),
        });
        // Phis in successors now see one edge from this block instead of many.
        biara_ir::analysis::recompute_preds(f);
        true
    } else {
        false
    }
}

fn const_value(f: &Function, v: ValueId) -> Option<u128> {
    match f.value(v) {
        Value::Const(c) => c.as_u128(),
        _ => None,
    }
}
