//! The pass set the analyses depend on, reimplemented.
//!
//! LLVM provided `mem2reg`/`sroa` (spills become SSA), `instcombine`
//! (constant folding, algebraic identities), `simplifycfg` (trivial branch
//! threading, unreachable pruning), `gvn` (redundant load elimination through
//! a scoped alias model) and `dse` (dead stores). The lifters here emit one
//! alloca per promoted register and one store per write-back, so without
//! promotion no analysis sees a register as a value.

use crate::{
    BinOp, BlockId, Constant, Dominators, Function, Inst, IntPred, Terminator, Type, Value,
    ValueId, analysis::recompute_preds,
};
use std::collections::{BTreeMap, BTreeSet, HashMap};

/// Run the pipeline used before analyses read a lifted function.
pub fn simplify(f: &mut Function) {
    promote_allocas(f);
    // Repeat to a fixed point: folding exposes dead code, DCE exposes
    // single-use chains, CFG simplification exposes more folding. Every step
    // runs at least once, since folding alone finds nothing to do in a
    // function whose only defect is dead code.
    for _ in 0..4 {
        let folded = fold(f);
        dce(f);
        let simplified = simplify_cfg(f);
        mem2reg_phis(f);
        dce(f);
        if !folded && !simplified {
            break;
        }
    }
}

/// Replace an instruction's uses with `with`, everywhere in the function.
pub fn replace_all_uses(f: &mut Function, from: ValueId, with: ValueId) {
    rewrite_uses(f, |id| if id == from { with } else { id });
}

fn rewrite_uses(f: &mut Function, replace: impl Fn(ValueId) -> ValueId) {
    for b in &mut f.blocks {
        for &i in &b.insts {
            if let Value::Inst(inst) = &mut f.values[i.0 as usize] {
                for operand in crate::operands_mut(inst) {
                    *operand = replace(*operand);
                }
            }
        }
        for operand in b.term.as_mut().map(term_slots).unwrap_or_default() {
            *operand = replace(*operand);
        }
    }
}

fn term_slots(t: &mut Terminator) -> Vec<&mut ValueId> {
    match t {
        Terminator::Ret(v) => v.iter_mut().collect(),
        Terminator::CondBr { cond, .. } => vec![cond],
        Terminator::Switch { value, .. } => vec![value],
        _ => Vec::new(),
    }
}

// ---- mem2reg ---------------------------------------------------------------

struct AllocaInfo {
    ty: Type,
    /// Blocks holding a store to this slot.
    stores: BTreeSet<BlockId>,
    /// True when the address escapes (any use other than load/store of it).
    escaped: bool,
    /// Single block with a single store and no loads after it, the common case.
    single_store: Option<(BlockId, ValueId)>,
}

/// Promote allocas whose address never escapes to SSA values.
///
/// This is the subset of `mem2reg`/`SROA` the lifters need: every promoted
/// slot is written in its entry block and read with loads, so a per-block
/// value with phis at joins is enough. Slots with address escapes, or
/// partially-initialised on some path, keep their memory form.
pub fn promote_allocas(f: &mut Function) {
    let mut infos: HashMap<ValueId, AllocaInfo> = HashMap::new();
    for (i, v) in f.values.iter().enumerate() {
        if let Value::Inst(Inst::Alloca { ty }) = v {
            infos.insert(
                ValueId(i as u32),
                AllocaInfo {
                    ty: ty.clone(),
                    stores: BTreeSet::new(),
                    escaped: false,
                    single_store: None,
                },
            );
        }
    }
    if infos.is_empty() {
        return;
    }
    // Classify uses.
    for (bi, b) in f.blocks.iter().enumerate() {
        for &i in &b.insts {
            match f.inst(i) {
                Some(Inst::Load { ptr, .. }) => {
                    if let Some(info) = infos.get_mut(ptr) {
                        let _ = info;
                    }
                }
                Some(Inst::Store { ptr, value, .. }) => {
                    if let Some(info) = infos.get_mut(ptr) {
                        info.stores.insert(BlockId(bi as u32));
                        if info.single_store.is_none() {
                            info.single_store = Some((BlockId(bi as u32), *value));
                        } else {
                            info.single_store = None;
                        }
                    }
                }
                Some(other) => {
                    for o in crate::operands(other) {
                        if let Some(info) = infos.get_mut(&o) {
                            info.escaped = true;
                        }
                    }
                }
                None => {}
            }
        }
        for o in b.term.as_ref().map(crate::term_operands).unwrap_or_default() {
            if let Some(info) = infos.get_mut(&o) {
                info.escaped = true;
            }
        }
    }

    let doms = Dominators::new(f);
    let promotable: Vec<ValueId> = infos
        .iter()
        .filter(|(_, info)| !info.escaped && !info.stores.is_empty())
        .map(|(v, _)| *v)
        .collect();
    if promotable.is_empty() {
        return;
    }

    let mut replacements = vec![None; f.values.len()];
    let mut changed = Vec::new();
    for slot in promotable {
        let ty = infos[&slot].ty.clone();
        // Walk blocks in dominator-tree preorder, tracking the value the slot
        // holds. Blocks that read before any store on a path keep the slot.
        let mut current: HashMap<BlockId, ValueId> = HashMap::new();
        let mut ok = true;
        for &b in doms.order() {
            let mut cur = f
                .block(b)
                .preds
                .iter()
                .filter_map(|p| current.get(p).copied())
                .next();
            // A join needs every predecessor to agree; otherwise a phi would
            // be required, which this pass does not synthesise for slots.
            // A predecessor with no tracked value is not "no opinion" — it is
            // a path on which the slot holds something this walk cannot name,
            // which includes every loop's back edge. Promoting through it
            // would make the header read the pre-loop value forever.
            let preds = &f.block(b).preds;
            if preds.len() > 1 {
                let mut first = None;
                for p in preds {
                    match current.get(p).copied() {
                        Some(v) if first.is_none() || first == Some(v) => first = Some(v),
                        _ => {
                            ok = false;
                            break;
                        }
                    }
                }
                if !ok {
                    break;
                }
                cur = first;
            }
            for &i in &f.block(b).insts {
                match f.inst(i) {
                    Some(Inst::Store { ptr, value, .. })
                        if resolve_replacement(&replacements, *ptr) == slot =>
                    {
                        cur = Some(resolve_replacement(&replacements, *value));
                    }
                    Some(Inst::Load { ptr, .. })
                        if resolve_replacement(&replacements, *ptr) == slot =>
                    match cur {
                        Some(v) => {
                            let v = resolve_replacement(&replacements, v);
                            if i != v {
                                replacements[i.0 as usize] = Some(v);
                                changed.push(i);
                            }
                        }
                        None => {
                            ok = false;
                            break;
                        }
                    },
                    _ => {}
                }
            }
            if !ok {
                break;
            }
            if let Some(v) = cur {
                current.insert(b, v);
            }
        }
        // Rewrite every load found on the walk in one visit to the function.
        // Even a slot we cannot remove may have loads with known values.
        if !changed.is_empty() {
            rewrite_uses(f, |id| resolve_replacement(&replacements, id));
            for id in changed.drain(..) {
                replacements[id.0 as usize] = None;
            }
        }
        if ok {
            remove_slot(f, slot);
        }
        let _ = ty;
    }
    // Promoted loads rewire to their stored value; the slot and any store to
    // it are now dead. Everything else stays: a slot this pass could not
    // promote was never rewritten, so its users must survive.
    dce(f);
}

fn resolve_replacement(replacements: &[Option<ValueId>], mut value: ValueId) -> ValueId {
    while let Some(Some(next)) = replacements.get(value.0 as usize).copied() {
        value = next;
    }
    value
}

fn remove_slot(f: &mut Function, slot: ValueId) {
    let doomed: Vec<(BlockId, ValueId)> = f
        .blocks
        .iter()
        .enumerate()
        .flat_map(|(bi, b)| {
            b.insts.iter().map(move |&i| (BlockId(bi as u32), i))
        })
        .filter(|&(_, i)| {
            i == slot
                || matches!(
                    f.inst(i),
                    Some(Inst::Load { ptr, .. }) | Some(Inst::Store { ptr, .. }) if *ptr == slot
                )
        })
        .collect();
    for (b, i) in doomed {
        remove_inst(f, b, i);
    }
    dce(f);
}

// ---- folding ---------------------------------------------------------------

/// Constant-fold and apply algebraic identities. Returns whether anything
/// changed, so the pipeline can iterate.
pub fn fold(f: &mut Function) -> bool {
    let mut changed = false;
    let mut consts: HashMap<ValueId, Constant> = HashMap::new();
    for (i, v) in f.values.iter().enumerate() {
        if let Value::Const(c) = v {
            consts.insert(ValueId(i as u32), c.clone());
        }
    }
    let mut work: Vec<(BlockId, ValueId)> = Vec::new();
    for (bi, b) in f.blocks.iter().enumerate() {
        for &i in &b.insts {
            work.push((BlockId(bi as u32), i));
        }
    }
    for (block, id) in work {
        let Some(inst) = f.inst(id).cloned() else {
            continue;
        };
        if let Some(folded) = fold_inst(f, &consts, &inst) {
            // Point uses at the folded value and drop the instruction.
            replace_all_uses(f, id, folded);
            remove_inst(f, block, id);
            if let Some(c) = f.constant(folded) {
                consts.insert(folded, c.clone());
            }
            changed = true;
        }
    }
    changed
}

fn fold_inst(
    f: &mut Function,
    consts: &HashMap<ValueId, Constant>,
    inst: &Inst,
) -> Option<ValueId> {
    match inst {
        Inst::Bin { op, ty, lhs, rhs } => {
            let (l, r) = (consts.get(lhs)?, consts.get(rhs)?);
            let (lv, rv) = (l.as_u128()?, r.as_u128()?);
            let w = ty.int_bits()?;
            let m = crate::analysis::mask(w);
            let v = match op {
                BinOp::Add => lv.wrapping_add(rv) & m,
                BinOp::Sub => lv.wrapping_sub(rv) & m,
                BinOp::Mul => lv.wrapping_mul(rv) & m,
                BinOp::And => lv & rv,
                BinOp::Or => lv | rv,
                BinOp::Xor => lv ^ rv,
                BinOp::Shl => {
                    if rv >= w as u128 {
                        return None;
                    }
                    (lv << rv) & m
                }
                BinOp::Lshr => {
                    if rv >= w as u128 {
                        return None;
                    }
                    (lv & m) >> rv
                }
                BinOp::Ashr => {
                    if rv >= w as u128 {
                        return None;
                    }
                    ((crate::analysis::signed(lv, w) >> rv) as u128) & m
                }
                BinOp::Udiv | BinOp::Urem => {
                    if rv == 0 {
                        return None;
                    }
                    if *op == BinOp::Udiv { lv / rv } else { lv % rv }
                }
                BinOp::Sdiv | BinOp::Srem => {
                    let (a, b) = (crate::analysis::signed(lv, w), crate::analysis::signed(rv, w));
                    if b == 0 || (a == i128::MIN && b == -1) {
                        return None;
                    }
                    let d = if *op == BinOp::Sdiv { a / b } else { a % b };
                    (d as u128) & m
                }
            };
            Some(f.add_const(Constant::Int { bits: v & m, width: w }))
        }
        Inst::Icmp { pred, ty, lhs, rhs } if consts.contains_key(lhs) && consts.contains_key(rhs) => {
            let w = ty.int_bits()?;
            let (l, r) = (consts[lhs].as_u128()?, consts[rhs].as_u128()?);
            let (ls, rs) = (crate::analysis::signed(l, w), crate::analysis::signed(r, w));
            let b = match pred {
                IntPred::Eq => l == r,
                IntPred::Ne => l != r,
                IntPred::Slt => ls < rs,
                IntPred::Sle => ls <= rs,
                IntPred::Sgt => ls > rs,
                IntPred::Sge => ls >= rs,
                IntPred::Ult => l < r,
                IntPred::Ule => l <= r,
                IntPred::Ugt => l > r,
                IntPred::Uge => l >= r,
            };
            Some(f.add_const(Constant::int(b as i128, 1)))
        }
        Inst::Select { cond, yes, no, .. } => match consts.get(cond) {
            Some(Constant::Int { bits: 0, .. }) => Some(*no),
            Some(Constant::Int { .. }) => Some(*yes),
            _ => None,
        },
        Inst::Trunc { to, value, .. } => {
            let c = consts.get(value)?;
            let w = to.int_bits()?;
            Some(f.add_const(Constant::Int {
                bits: c.as_u128()? & crate::analysis::mask(w),
                width: w,
            }))
        }
        Inst::Zext { to, value, .. } => {
            let c = consts.get(value)?;
            let w = to.int_bits()?;
            Some(f.add_const(Constant::Int {
                bits: c.as_u128()?,
                width: w,
            }))
        }
        Inst::Sext { from, to, value } => {
            let c = consts.get(value)?;
            let fw = from.int_bits()?;
            let w = to.int_bits()?;
            let s = crate::analysis::signed(c.as_u128()?, fw);
            Some(f.add_const(Constant::Int {
                bits: (s as u128) & crate::analysis::mask(w),
                width: w,
            }))
        }
        _ => None,
    }
}

fn remove_inst(f: &mut Function, block: BlockId, id: ValueId) {
    f.blocks[block.0 as usize].insts.retain(|&i| i != id);
}

// ---- dead code -------------------------------------------------------------

/// Remove instructions with no uses that have no side effects, and blocks
/// that are unreachable from the entry.
pub fn dce(f: &mut Function) {
    // `preds` is a cache: derive it from the terminators before deciding
    // what is unreachable, or a stale cache would delete live blocks.
    recompute_preds(f);
    loop {
        let mut dropped = false;
        for (index, block) in f.blocks.iter_mut().enumerate() {
            // A block with no incoming edge can only be reached as the entry.
            if block.preds.is_empty() && Some(BlockId(index as u32)) != f.entry {
                block.insts.clear();
                block.term = Some(Terminator::Unreachable);
                dropped = true;
            }
        }
        if dropped {
            recompute_preds(f);
        }
        let removed = remove_unused_instructions(f);
        if !removed {
            break;
        }
    }
}

fn remove_unused_instructions(f: &mut Function) -> bool {
    let mut uses = vec![0usize; f.values.len()];
    let mut removable = vec![false; f.values.len()];
    for block in &f.blocks {
        for inst in block.insts.iter().filter_map(|&id| f.inst(id)) {
            for operand in crate::operands(inst) {
                if let Some(count) = uses.get_mut(operand.0 as usize) {
                    *count += 1;
                }
            }
        }
        for operand in block.term.iter().flat_map(crate::term_operands) {
            if let Some(count) = uses.get_mut(operand.0 as usize) {
                *count += 1;
            }
        }
    }

    let mut pending = Vec::new();
    for &id in f.blocks.iter().flat_map(|block| &block.insts) {
        let index = id.0 as usize;
        removable[index] = !matches!(
            f.inst(id),
            Some(Inst::Store { .. } | Inst::Call { .. } | Inst::Marker { .. } | Inst::Fence)
        );
        if removable[index] && uses[index] == 0 {
            pending.push(id);
        }
    }
    if pending.is_empty() {
        return false;
    }

    while let Some(id) = pending.pop() {
        if let Some(inst) = f.inst(id) {
            for operand in crate::operands(inst) {
                let index = operand.0 as usize;
                if let Some(count) = uses.get_mut(index) {
                    *count -= 1;
                    if *count == 0 && removable[index] {
                        pending.push(operand);
                    }
                }
            }
        }
    }
    for block in &mut f.blocks {
        block.insts.retain(|id| {
            let index = id.0 as usize;
            !removable[index] || uses[index] != 0
        });
    }
    true
}


/// Collapse trivial branches, merge a block into its only predecessor, and
/// drop unreachable blocks.
pub fn simplify_cfg(f: &mut Function) -> bool {
    let mut any = false;
    loop {
        recompute_preds(f);
        let mut changed = false;
        // Thread `br` to a block with a single unconditional successor.
        for i in 0..f.blocks.len() {
            let b = BlockId(i as u32);
            match f.blocks[i].term.clone().unwrap_or(Terminator::Unreachable) {
                Terminator::Br(t) => {
                    let mut target = t;
                    let mut hops = 0;
                    while hops < 16 {
                        let tb = &f.blocks[target.0 as usize];
                        if tb.insts.is_empty()
                            && let Some(Terminator::Br(next)) = tb.term
                            && next != target
                        {
                            target = next;
                            hops += 1;
                            continue;
                        }
                        break;
                    }
                    if target != t {
                        f.blocks[i].term = Some(Terminator::Br(target));
                        changed = true;
                    }
                    let _ = b;
                }
                Terminator::CondBr { cond, yes, no } if yes == no => {
                    f.blocks[i].term = Some(Terminator::Br(yes));
                    let _ = cond;
                    changed = true;
                }
                _ => {}
            }
        }
        if changed {
            recompute_preds(f);
        }
        // Merge a block whose single pred ends in an unconditional branch into
        // that predecessor, rewriting the pred's terminator.
        for i in 0..f.blocks.len() {
            let b = BlockId(i as u32);
            if Some(b) == f.entry {
                continue;
            }
            if f.blocks[i].preds.len() != 1 {
                continue;
            }
            let p = *f.blocks[i].preds.iter().next().unwrap();
            if !matches!(f.blocks[p.0 as usize].term, Some(Terminator::Br(t)) if t == b) {
                continue;
            }
            if !f.blocks[i].insts.is_empty() || !f.blocks[i].preds.contains(&p) {
                continue;
            }
            // Replace the pred's terminator with the successor's.
            let term = f.blocks[i].term.clone();
            f.blocks[p.0 as usize].term = term;
            f.blocks[i].term = Some(Terminator::Unreachable);
            f.blocks[i].insts.clear();
            changed = true;
        }
        recompute_preds(f);
        any |= changed;
        if !changed {
            break;
        }
    }
    any
}

/// Remove phis made redundant by the CFG edits (a phi with one incoming
/// value, or all-incoming-equal) and unused phis.
pub fn mem2reg_phis(f: &mut Function) {
    loop {
        let mut changed = false;
        for b in 0..f.blocks.len() {
            let block = BlockId(b as u32);
            let insts: Vec<ValueId> = f.blocks[b].insts.clone();
            for i in insts {
                let Some(Inst::Phi { incoming, .. }) = f.inst(i).cloned() else {
                    continue;
                };
                let mut vals: Vec<ValueId> = incoming.iter().map(|(v, _)| *v).collect();
                vals.dedup();
                if vals.len() == 1 || f.uses(i) == 0 {
                    let with = vals.first().copied();
                    if let Some(with) = with {
                        replace_all_uses(f, i, with);
                    }
                    remove_inst(f, block, i);
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
}

/// Eliminate stores overwritten before any read on a straight-line path.
pub fn dse(f: &mut Function) {
    for b in 0..f.blocks.len() {
        let block = BlockId(b as u32);
        let insts: Vec<ValueId> = f.blocks[b].insts.clone();
        let mut last_store: HashMap<ValueId, ValueId> = HashMap::new();
        let mut dead: Vec<ValueId> = Vec::new();
        for i in insts {
            match f.inst(i).cloned() {
                Some(Inst::Store { ptr, .. }) => {
                    if let Some(prev) = last_store.insert(ptr, i) {
                        dead.push(prev);
                    }
                }
                Some(inst) => {
                    if let Inst::Load { ptr, .. } = inst {
                        last_store.remove(&ptr);
                    }
                }
                None => {}
            }
        }
        for d in dead {
            remove_inst(f, block, d);
        }
    }
}

/// A cheap GVN: replace a pure instruction whose operands and operator match
/// an earlier dominating one.
pub fn gvn(f: &mut Function) {
    let doms = Dominators::new(f);
    let mut available: HashMap<String, ValueId> = HashMap::new();
    for b in doms.order().to_vec() {
        let insts: Vec<ValueId> = f.block(b).insts.clone();
        for i in insts {
            let Some(inst) = f.inst(i).cloned() else {
                continue;
            };
            if !is_pure(&inst) {
                continue;
            }
            let key = key_of(f, &inst, i);
            match available.get(&key) {
                Some(&prev) if prev != i => {
                    replace_all_uses(f, i, prev);
                    remove_inst(f, b, i);
                }
                _ => {
                    available.insert(key, i);
                }
            }
        }
        // A block with multiple preds invalidates nothing here: the key is
        // operand-based, so a dominating equal expression is still equal.
        let _ = &doms;
    }
}

fn is_pure(i: &Inst) -> bool {
    !matches!(
        i,
        Inst::Store { .. }
            | Inst::Call { .. }
            | Inst::Load { .. }
            | Inst::Alloca { .. }
            | Inst::Marker { .. }
            | Inst::Fence
    )
}

fn key_of(f: &Function, inst: &Inst, self_id: ValueId) -> String {
    let mut s = format!("{:?}", std::mem::discriminant(inst));
    // Fields that are part of the instruction's meaning but not operands:
    // a `gep` is distinguished by its offset, a cast by its type, an
    // extraction by its index.
    match inst {
        Inst::Gep { offset, .. } => s.push_str(&format!("o{offset}")),
        Inst::Bin { op, ty, .. } => s.push_str(&format!("{op:?}{ty:?}")),
        Inst::Un { op, ty, .. } => s.push_str(&format!("{op:?}{ty:?}")),
        Inst::Rot { left, ty, .. } => s.push_str(&format!("{left}{ty:?}")),
        Inst::Icmp { pred, ty, .. } => s.push_str(&format!("{pred:?}{ty:?}")),
        Inst::Trunc { from, to, .. }
        | Inst::Zext { from, to, .. }
        | Inst::Sext { from, to, .. }
        | Inst::Cast { from, to, .. } => s.push_str(&format!("{from:?}{to:?}")),
        Inst::ExtractValue { index, ty, .. } | Inst::InsertValue { index, ty, .. } => {
            s.push_str(&format!("i{index}{ty:?}"));
        }
        Inst::Select { ty, .. } => s.push_str(&format!("{ty:?}")),
        Inst::Freeze { ty, .. } => s.push_str(&format!("{ty:?}")),
        _ => {}
    }
    for o in crate::operands(inst) {
        // Value ids are per function, so an id outside this arena names
        // nothing here and must not be indexed.
        match f.values.get(o.0 as usize) {
            Some(Value::Const(c)) => s.push_str(&format!("c{:?}", c)),
            Some(Value::Param { index, .. }) => s.push_str(&format!("p{index}")),
            Some(Value::Inst(_)) => s.push_str(&format!("v{}", o.0)),
            None => s.push_str(&format!("x{}", o.0)),
        }
    }
    let _ = self_id;
    s
}

// ---- pipeline --------------------------------------------------------------

/// A pass the pipeline text can name. Anything else is rejected: silently
/// skipping a pass would change what the analyses see.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NamedPass {
    Sroa,
    EarlyCse,
    InstCombine,
    SimplifyCfg,
    Gvn,
    Dse,
    Adce,
    GlobalDce,
}

impl NamedPass {
    /// Parse LLVM's pass name, ignoring its options (`instcombine<…>`).
    pub fn parse(name: &str) -> Option<Self> {
        Some(match name.split('<').next().unwrap_or(name).trim() {
            "sroa" | "mem2reg" => Self::Sroa,
            "early-cse" => Self::EarlyCse,
            "instcombine" => Self::InstCombine,
            "simplifycfg" => Self::SimplifyCfg,
            "gvn" => Self::Gvn,
            "dse" => Self::Dse,
            "adce" => Self::Adce,
            "globaldce" => Self::GlobalDce,
            _ => return None,
        })
    }

    /// The `biara_ir` pass that stands for this LLVM one.
    pub fn run(self, f: &mut Function) {
        match self {
            // `sroa` and `mem2reg` both turn alloca traffic into SSA here.
            Self::Sroa => {
                promote_allocas(f);
                mem2reg_phis(f);
            }
            Self::EarlyCse | Self::InstCombine => {
                fold(f);
                dce(f);
            }
            Self::SimplifyCfg => {
                simplify_cfg(f);
                dce(f);
            }
            Self::Gvn => gvn(f),
            Self::Dse => dse(f),
            Self::Adce | Self::GlobalDce => {
                dce(f);
                simplify_cfg(f);
            }
        }
    }
}

/// Split on commas that are not inside parentheses.
fn split_stages(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut depth = 0i32;
    for c in text.chars() {
        match c {
            '(' => {
                depth += 1;
                cur.push(c);
            }
            ')' => {
                depth -= 1;
                cur.push(c);
            }
            ',' if depth == 0 => {
                out.push(std::mem::take(&mut cur));
            }
            _ => cur.push(c),
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur);
    }
    out
}

/// Parse a pipeline string into its passes. Accepts the `function(…)` and
/// `module(…)` wrappers the caller writes, and `default<O0..O3>` as the
/// canonicalisation chain those levels start with.
pub fn parse_pipeline(text: &str) -> Result<Vec<NamedPass>, String> {
    let mut out = Vec::new();
    for stage in split_stages(text) {
        let stage = stage.trim();
        if stage.is_empty() {
            continue;
        }
        if let Some(inner) = stage
            .strip_prefix("function(")
            .and_then(|s| s.strip_suffix(')'))
            .or_else(|| stage.strip_prefix("module(").and_then(|s| s.strip_suffix(')')))
        {
            out.extend(parse_pipeline(inner)?);
            continue;
        }
        if stage.starts_with("default<O") {
            out.extend([
                NamedPass::Sroa,
                NamedPass::EarlyCse,
                NamedPass::InstCombine,
                NamedPass::SimplifyCfg,
                NamedPass::Gvn,
                NamedPass::Dse,
                NamedPass::Adce,
            ]);
            continue;
        }
        match NamedPass::parse(stage) {
            Some(p) => out.push(p),
            None => return Err(format!("unknown pass `{stage}`")),
        }
    }
    Ok(out)
}

/// Run `pipeline` over every function of `f`, failing on an unknown name.
pub fn run_pipeline(f: &mut Function, pipeline: &str) -> Result<(), String> {
    for pass in parse_pipeline(pipeline)? {
        pass.run(f);
    }
    Ok(())
}
