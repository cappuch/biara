//! Structural checks: the invariants LLVM's verifier enforced that biara
//! relies on (typed operands, terminated blocks, phis aligned with preds).

use crate::{BlockId, Function, Inst, Terminator, Type, Value, ValueId};
use std::collections::HashMap;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifyError(pub String);

impl fmt::Display for VerifyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for VerifyError {}

fn fail<T>(msg: impl Into<String>) -> Result<T, VerifyError> {
    Err(VerifyError(msg.into()))
}

/// Check `f`. Returns the first problem found, naming the block and value.
pub fn verify(f: &Function) -> Result<(), VerifyError> {
    let name = &f.name;
    // A function without a body is a prototype: its declared type and its
    // formals are the whole of it, and there is nothing else to check.
    let Some(entry) = f.entry else {
        return check_signature(f);
    };
    if f.blocks.is_empty() {
        return check_signature(f);
    }
    if entry.0 as usize >= f.blocks.len() {
        return fail(format!("{name}: entry block out of range"));
    }
    check_signature(f)?;
    for (i, &p) in f.params.iter().enumerate() {
        match f.value(p) {
            Value::Param { ty, index, .. } => {
                if *index as usize != i {
                    return fail(format!("{name}: param {i} has index {index}"));
                }
                if *ty != f.ty.params[i] {
                    return fail(format!("{name}: param {i} type mismatch"));
                }
            }
            _ => return fail(format!("{name}: param {i} is not a parameter")),
        }
    }

    // Labels are the only way text IR and the printer name a block, so a
    // duplicate makes every branch to it ambiguous.
    let mut seen_names: HashMap<&str, u32> = HashMap::new();
    for (bi, b) in f.blocks.iter().enumerate() {
        if let Some(first) = seen_names.insert(b.name.as_str(), bi as u32) {
            return fail(format!(
                "{name}: blocks {first} and {bi} are both named {:?}",
                b.name
            ));
        }
    }
    // Every actual edge must appear in the predecessor set. Count each
    // source once per destination so we can then reject spurious entries
    // without replaying a large switch for every one of its destinations.
    let mut incoming = vec![(None, 0usize); f.blocks.len()];
    for (bi, block) in f.blocks.iter().enumerate() {
        let b = BlockId(bi as u32);
        let Some(term) = f.terminator(b) else {
            return fail(format!(
                "{name}: block {bi} ({}) has no terminator; {} insts, preds {preds:?}",
                block.name,
                block.insts.len(),
                preds = block.preds
            ));
        };
        for succ in (0..).map_while(|i| crate::successor_at(term, i)) {
            if succ.0 as usize >= f.blocks.len() {
                return fail(format!("{name}: block {bi} branches out of range"));
            }
            if !f.blocks[succ.0 as usize].preds.contains(&b) {
                return fail(format!(
                    "{name}: block {bi} -> {succ:?} missing from preds"
                ));
            }
            let (last, count) = &mut incoming[succ.0 as usize];
            if *last != Some(b) {
                *last = Some(b);
                *count += 1;
            }
        }
    }
    for (bi, block) in f.blocks.iter().enumerate() {
        let b = BlockId(bi as u32);
        if incoming[bi].1 != block.preds.len() {
            return fail(format!("{name}: block {bi} has a predecessor without a matching branch"));
        }
        for &inst in &f.block(b).insts {
            if f.defined_in[inst.0 as usize] != Some(b) {
                return fail(format!("{name}: value {inst:?} is in the wrong block"));
            }
            let Some(i) = f.inst(inst) else {
                return fail(format!("{name}: {inst:?} is not an instruction"));
            };
            check_inst(f, b, i)?;
        }
        check_term(f, b)?;
    }

    check_uses_in_range(f)?;
    Ok(())
}

/// The formals a function's type declares: one parameter value each.
fn check_signature(f: &Function) -> Result<(), VerifyError> {
    if f.params.len() != f.ty.params.len() {
        return fail(format!(
            "{}: {} params but type wants {}",
            f.name,
            f.params.len(),
            f.ty.params.len()
        ));
    }
    Ok(())
}

fn check_inst(f: &Function, b: BlockId, i: &Inst) -> Result<(), VerifyError> {
    let name = &f.name;
    // A verifier reports a fault, it never panics on one: an operand id from
    // another function's arena has no type here, and that mismatch is exactly
    // what the checks below must report.
    let ty_of = |v: ValueId| -> Option<Type> {
        if (v.0 as usize) < f.values.len() { Some(f.ty(v)) } else { None }
    };
    let same = |v: ValueId, t: &Type| ty_of(v).as_ref() == Some(t);
    match i {
        Inst::Bin { op, ty, lhs, rhs } => {
            if !ty.is_int() && !ty.is_vector() {
                return fail(format!("{name}: {op:?} on non-integer type {ty:?}"));
            }
            if !same(*lhs, ty) || !same(*rhs, ty) {
                return fail(format!(
                    "{name}: {op:?} operand types {} / {} != {ty:?}",
                    fmt_ty(&ty_of(*lhs).unwrap_or(Type::Void)),
                    fmt_ty(&ty_of(*rhs).unwrap_or(Type::Void))
                ));
            }
        }
        Inst::Icmp { pred, ty, lhs, rhs } => {
            if !same(*lhs, ty) || !same(*rhs, ty) {
                return fail(format!("{name}: icmp {pred:?} operand types differ"));
            }
            if !ty.is_int() && !ty.is_vector() {
                return fail(format!("{name}: icmp on non-integer type {ty:?}"));
            }
        }
        Inst::Select { ty, cond, yes, no } => {
            if !same(*cond, &Type::i1()) {
                return fail(format!("{name}: select condition is not i1"));
            }
            if !same(*yes, ty) || !same(*no, ty) {
                return fail(format!("{name}: select arms differ from {ty:?}"));
            }
        }
        Inst::Trunc { from, to, value } => {
            let (Some(a), Some(b)) = (from.int_bits(), to.int_bits()) else {
                return fail(format!("{name}: trunc of non-integers"));
            };
            if a <= b {
                return fail(format!("{name}: trunc {a} -> {b} does not narrow"));
            }
            if !same(*value, from) {
                return fail(format!("{name}: trunc operand type"));
            }
        }
        Inst::Zext { from, to, value } | Inst::Sext { from, to, value } => {
            let (Some(a), Some(b)) = (from.int_bits(), to.int_bits()) else {
                return fail(format!("{name}: extend of non-integers"));
            };
            if a >= b {
                return fail(format!("{name}: extend {a} -> {b} does not widen"));
            }
            if !same(*value, from) {
                return fail(format!("{name}: extend operand type"));
            }
        }
        Inst::Cast { from, to, value } => {
            if !same(*value, from) {
                return fail(format!("{name}: cast operand type"));
            }
            let bits_ok = from.int_bits().is_some() && to.int_bits().is_some()
                && from.int_bits() == to.int_bits();
            let ptr_ok = (*from == Type::Ptr) != (*to == Type::Ptr)
                && (from.int_bits().map(|b| b == 64).unwrap_or(false)
                    || to.int_bits().map(|b| b == 64).unwrap_or(false));
            if !bits_ok && !ptr_ok && from != to {
                return fail(format!(
                    "{name}: invalid cast {} -> {}",
                    fmt_ty(from),
                    fmt_ty(to)
                ));
            }
        }
        Inst::Alloca { .. } => {}
        Inst::Un { op: _, ty, value } => {
            if !ty.is_int() {
                return fail(format!("{name}: unary op on non-integer {ty:?}"));
            }
            if !same(*value, ty) {
                return fail(format!("{name}: unary operand type"));
            }
        }
        Inst::Rot { ty, value, amount, .. } => {
            if !ty.is_int() {
                return fail(format!("{name}: rotate on non-integer {ty:?}"));
            }
            if !same(*value, ty) || !same(*amount, ty) {
                return fail(format!("{name}: rotate operand types"));
            }
        }
        Inst::Fence => {}
        Inst::Load { ty, ptr, .. } => {
            if !same(*ptr, &Type::Ptr) {
                return fail(format!("{name}: load from non-pointer"));
            }
            if matches!(ty, Type::Void | Type::Func(_)) {
                return fail(format!("{name}: load of unsized type"));
            }
        }
        Inst::Store { ty, value, ptr, .. } => {
            if !same(*ptr, &Type::Ptr) {
                return fail(format!("{name}: store to non-pointer"));
            }
            if !same(*value, ty) {
                return fail(format!("{name}: store value type"));
            }
        }
        Inst::Gep { base, .. } => {
            if !same(*base, &Type::Ptr) {
                return fail(format!("{name}: gep of non-pointer"));
            }
        }
        Inst::Call { callee, args, ty } => {
            let Value::Const(crate::Constant::Function(name_)) = f.value(*callee) else {
                // Calls through a pointer are allowed (dispatch).
                if !same(*callee, &Type::Ptr) {
                    return fail(format!("{name}: call of non-function value"));
                }
                return Ok(());
            };
            let _ = name_;
            if matches!(ty, Type::Void) {
                // Return type is checked against the declaration by callers
                // that own the module; the function alone cannot see it.
            }
            let _ = args;
        }
        Inst::Phi { ty, incoming } => {
            let preds = &f.block(b).preds;
            if incoming.len() != preds.len() {
                return fail(format!(
                    "{name}: phi has {} incoming but the block has {} preds",
                    incoming.len(),
                    preds.len()
                ));
            }
            for (v, from) in incoming {
                if !preds.contains(from) {
                    return fail(format!("{name}: phi incoming from non-pred"));
                }
                if !same(*v, ty) {
                    return fail(format!("{name}: phi incoming type"));
                }
            }
        }
        Inst::ExtractValue { agg, index, .. } => {
            if let Some(Type::Struct(fields)) = ty_of(*agg)
                && (*index as usize) >= fields.len()
            {
                return fail(format!("{name}: extractvalue index out of range"));
            }
        }
        Inst::InsertValue { agg, value, .. } => {
            if ty_of(*agg) != ty_of(*value)
                && !matches!(ty_of(*agg), Some(Type::Struct(_)))
            {
                return fail(format!("{name}: insertvalue type"));
            }
        }
        Inst::Freeze { value, .. } => {
            let _ = value;
        }
        Inst::Marker { .. } => {}
    }
    Ok(())
}

fn check_term(f: &Function, b: BlockId) -> Result<(), VerifyError> {
    let name = &f.name;
    let Some(term) = f.terminator(b) else {
        return fail(format!(
            "{name}: block {b:?} is not terminated ({} insts, preds {:?})",
            f.blocks[b.0 as usize].insts.len(),
            f.blocks[b.0 as usize].preds
        ));
    };
    match term {
        Terminator::Ret(v) => match (v, &f.ty.ret) {
            (None, Type::Void) => Ok(()),
            (Some(v), ty) if f.ty(*v) == *ty => Ok(()),
            (Some(v), ty) => fail(format!(
                "{name}: ret {:?} but function returns {}",
                fmt_ty(&f.ty(*v)),
                fmt_ty(ty)
            )),
            (None, ty) => fail(format!(
                "{name}: ret void but function returns {}",
                fmt_ty(ty)
            )),
        },
        Terminator::Br(t) | Terminator::CondBr { yes: t, .. } => {
            if t.0 as usize >= f.blocks.len() {
                return fail(format!("{name}: branch target out of range"));
            }
            Ok(())
        }
        Terminator::Switch { cases, default, .. } => {
            if default.0 as usize >= f.blocks.len() {
                return fail(format!("{name}: switch default out of range"));
            }
            for (_, t) in cases {
                if t.0 as usize >= f.blocks.len() {
                    return fail(format!("{name}: switch case out of range"));
                }
            }
            Ok(())
        }
        Terminator::Unreachable => Ok(()),
    }
}

fn check_uses_in_range(f: &Function) -> Result<(), VerifyError> {
    let n = f.values.len();
    let mut check = |v: ValueId| -> Result<(), VerifyError> {
        if (v.0 as usize) >= n {
            return fail(format!("{}: use of out-of-range value {v:?}", f.name));
        }
        Ok(())
    };
    for b in &f.blocks {
        for &i in &b.insts {
            if let Some(inst) = f.inst(i) {
                for o in crate::operands(inst) {
                    check(o)?;
                }
            }
        }
        for o in b.term.as_ref().map(crate::term_operands).unwrap_or_default() {
            check(o)?;
        }
    }
    Ok(())
}

fn fmt_ty(t: &Type) -> String {
    match t {
        Type::Void => "void".to_owned(),
        Type::Int(w) => format!("i{w}"),
        Type::Ptr => "ptr".to_owned(),
        other => format!("{other:?}"),
    }
}
