//! Whole-IR rewrites that used to live in `cpp/abi_form.cpp`.
//!
//! Three small transforms that make the emitted C readable, each a pure
//! function of the IR:
//!
//! - [`fold_frames`]: `inttoptr(add(ptrtoint(alloca), k))` becomes
//!   `gep alloca, k`, so frame address arithmetic folds into one expression.
//! - [`dup_returns`]: a return-only join block is duplicated into its
//!   unconditional predecessors, so the C shows one `return` per arm instead
//!   of a `common_ret` phi.
//! - [`div_sugar`]: a `biara_sdiv*`/`biara_div*` prelude call whose high half
//!   only extends the low becomes `sdiv`/`udiv`, so the C prints `a / b`.
//!
//! Each returns how many sites it rewrote.

use biara_ir::{BinOp, BlockId, Constant, Function, Inst, Terminator, Type, Value, ValueId};

/// `inttoptr(add(ptrtoint(alloca), k))` → `gep alloca, k`.
pub fn fold_frames(f: &mut Function) -> usize {
    let mut folded = 0;
    // Collect first: replacing uses mutates the arena.
    let mut work: Vec<(ValueId, ValueId, i64)> = Vec::new();
    for b in &f.blocks {
        for &id in &b.insts {
            let Some(Inst::Cast { to, value, .. }) = f.inst(id) else {
                continue;
            };
            if *to != Type::Ptr {
                continue;
            }
            // Walk the constant offset chain.
            let mut off = 0i64;
            let mut base = *value;
            let mut ok = false;
            for _ in 0..16 {
                match f.inst(base) {
                    Some(Inst::Bin {
                        op,
                        lhs,
                        rhs,
                        ..
                    }) => {
                        let Some(k) = const_i64(f, *rhs) else { break };
                        match op {
                            BinOp::Add => off = off.wrapping_add(k),
                            BinOp::Sub => off = off.wrapping_sub(k),
                            _ => break,
                        }
                        base = *lhs;
                        ok = true;
                    }
                    _ => break,
                }
            }
            let _ = ok;
            // The base must be `ptrtoint(alloca)`.
            let Some(Inst::Cast {
                from: Type::Ptr,
                value: ptr,
                ..
            }) = f.inst(base)
            else {
                continue;
            };
            if !matches!(f.inst(*ptr), Some(Inst::Alloca { .. })) {
                continue;
            }
            work.push((id, *ptr, off));
        }
    }
    for (cast, alloca, off) in work {
        let gep = f.add_const(Constant::Zero);
        let _ = gep;
        // A fresh `gep` at the cast's own position.
        let block = f.defined_in[cast.0 as usize];
        f.values.push(Value::Inst(Inst::Gep {
            base: alloca,
            offset: off,
        }));
        f.defined_in.push(block);
        f.pc.push(f.pc_of(cast));
        let new = ValueId((f.values.len() - 1) as u32);
        if let Some(b) = block {
            let insts = &mut f.blocks[b.0 as usize].insts;
            if let Some(at) = insts.iter().position(|v| *v == cast) {
                insts.insert(at + 1, new);
            }
        }
        replace_all_uses(f, cast, new);
        remove_inst(f, cast);
        folded += 1;
    }
    folded
}

/// Duplicate a return-only join block into its unconditional predecessors.
pub fn dup_returns(f: &mut Function) -> usize {
    let mut rewired = 0;
    let mut changed = true;
    while changed {
        changed = false;
        for j in 0..f.blocks.len() {
            let jb = BlockId(j as u32);
            // The block must be `ret` with nothing else, returning a constant
            // or its own phi.
            let Some(Terminator::Ret(ret)) = f.blocks[j].term.clone() else {
                continue;
            };
            if !f.blocks[j].insts.is_empty() {
                continue;
            }
            let ret = match ret {
                None => None,
                Some(v) => {
                    // Constants pass through; a phi must live in this block.
                    if matches!(f.value(v), Value::Const(_)) {
                        Some(v)
                    } else if matches!(f.inst(v), Some(Inst::Phi { .. }))
                        && f.defined_in[v.0 as usize] == Some(jb)
                    {
                        Some(v)
                    } else {
                        continue;
                    }
                }
            };
            // Predecessors that end in an unconditional branch here.
            let preds: Vec<BlockId> = f.blocks[j].preds.iter().copied().collect();
            let mut targets = Vec::new();
            for p in preds {
                if p == jb {
                    continue;
                }
                if let Some(Terminator::Br(t)) = f.blocks[p.0 as usize].term {
                    if t == jb {
                        targets.push(p);
                    }
                }
            }
            if targets.is_empty() {
                continue;
            }
            for p in targets {
                // The value for this arm: the phi's incoming from `p`, else
                // the returned constant.
                let arm = match ret {
                    None => None,
                    Some(v) => match f.inst(v) {
                        Some(Inst::Phi { incoming, .. }) => incoming
                            .iter()
                            .find(|(_, from)| *from == p)
                            .map(|(val, _)| *val),
                        _ => Some(v),
                    },
                };
                f.blocks[p.0 as usize].term = Some(Terminator::Ret(arm));
                rewired += 1;
                changed = true;
            }
        }
    }
    if rewired > 0 {
        biara_ir::analysis::recompute_preds(f);
    }
    rewired
}

/// `biara_sdiv64(lo, hi, d)` with `hi` only extending `lo` becomes `sdiv`.
pub fn div_sugar(f: &mut Function) -> usize {
    let mut sugared = 0;
    let mut work = Vec::new();
    for b in &f.blocks {
        for &id in &b.insts {
            let Some(Inst::Call { callee, args, .. }) = f.inst(id) else {
                continue;
            };
            if args.len() != 3 {
                continue;
            }
            let Some(Value::Const(Constant::Function(name))) = Some(f.value(*callee)) else {
                continue;
            };
            let Some((width, signed)) = parse_div_name(name) else {
                continue;
            };
            let (lo, hi, d) = (args[0], args[1], args[2]);
            // `hi` must only extend `lo`.
            let extends = if signed {
                match f.inst(hi) {
                    Some(Inst::Bin {
                        op: BinOp::Ashr,
                        lhs,
                        rhs,
                        ..
                    }) => {
                        *lhs == lo
                            && const_i64(f, *rhs) == Some(width as i64 - 1)
                    }
                    Some(Inst::Sext { value, .. }) => *value == lo,
                    _ => false,
                }
            } else {
                const_i64(f, hi) == Some(0)
            };
            if !extends {
                continue;
            }
            // Every use must be an `extractvalue` of the result tuple.
            let uses: Vec<ValueId> = f
                .users(id)
                .into_iter()
                .map(|(u, _)| u)
                .collect();
            if uses.is_empty()
                || !uses
                    .iter()
                    .all(|u| matches!(f.inst(*u), Some(Inst::ExtractValue { .. })))
            {
                continue;
            }
            work.push((id, lo, d, width, signed, uses));
        }
    }
    for (call, lo, d, width, signed, uses) in work {
        let ty = Type::Int(width);
        let op = |q: bool| {
            if signed {
                if q {
                    BinOp::Sdiv
                } else {
                    BinOp::Srem
                }
            } else if q {
                BinOp::Udiv
            } else {
                BinOp::Urem
            }
        };
        let block = f.defined_in[call.0 as usize];
        let mk = |f: &mut Function, o: BinOp, b: Option<BlockId>| {
            f.values.push(Value::Inst(Inst::Bin {
                op: o,
                ty: ty.clone(),
                lhs: lo,
                rhs: d,
            }));
            f.defined_in.push(b);
            f.pc.push(f.pc_of(call));
            ValueId((f.values.len() - 1) as u32)
        };
        let quot = mk(f, op(true), block);
        let rem = mk(f, op(false), block);
        for u in uses {
            let Some(Inst::ExtractValue { index, .. }) = f.inst(u) else {
                continue;
            };
            let replacement = if *index == 0 { quot } else { rem };
            replace_all_uses(f, u, replacement);
            remove_inst(f, u);
        }
        remove_inst(f, call);
        sugared += 1;
    }
    sugared
}

/// Split `biara_sdiv64` into its width and signedness.
fn parse_div_name(name: &str) -> Option<(u32, bool)> {
    let rest = name.strip_prefix("biara_")?;
    let (signed, rest) = match rest.strip_prefix("sdiv") {
        Some(r) => (true, r),
        None => (false, rest.strip_prefix("div")?),
    };
    let width = match rest {
        "8" => 8,
        "16" => 16,
        "32" => 32,
        "64" => 64,
        _ => return None,
    };
    Some((width, signed))
}

fn const_i64(f: &Function, v: ValueId) -> Option<i64> {
    match f.value(v) {
        Value::Const(c) => c.as_i128().map(|x| x as i64),
        _ => None,
    }
}

fn replace_all_uses(f: &mut Function, from: ValueId, with: ValueId) {
    biara_ir::pass::replace_all_uses(f, from, with);
}

fn remove_inst(f: &mut Function, id: ValueId) {
    if let Some(Some(b)) = f.defined_in.get(id.0 as usize).copied() {
        f.blocks[b.0 as usize].insts.retain(|v| *v != id);
    }
}
