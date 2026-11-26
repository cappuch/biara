//! Textual output in LLVM IR syntax, so `--emit llvm` stays readable and
//! familiar even though the module is ours.

use crate::{Constant, Function, Inst, IntPred, Module, Terminator, Type, Value, ValueId};
use std::fmt::Write;

pub fn print_module(m: &Module) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "; ModuleID = '{}'", m.name);
    let _ = writeln!(out, "source_filename = \"{}\"", m.name);
    if !m.triple.is_empty() {
        let _ = writeln!(out, "target triple = \"{}\"", m.triple);
    }
    if !m.data_layout.is_empty() {
        let _ = writeln!(out, "target datalayout = \"{}\"", m.data_layout);
    }
    out.push('\n');
    for g in &m.globals {
        if let Some(init) = &g.init {
            let _ = writeln!(
                out,
                "{}@{} = {}{}",
                if g.internal { "internal " } else { "" },
                g.name,
                if g.internal { "global " } else { "global " },
                print_const(init)
            );
        }
    }
    for d in &m.decls {
        let _ = writeln!(out, "declare {} @{}({})", print_ty(&d.ty.ret), d.name, params(&d.ty));
    }
    for f in &m.functions {
        out.push('\n');
        out.push_str(&print_function(f));
    }
    out
}

fn params(t: &crate::FuncType) -> String {
    let mut s: Vec<String> = t.params.iter().map(print_ty).collect();
    if t.varargs {
        s.push("...".to_owned());
    }
    s.join(", ")
}

pub fn print_function(f: &Function) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "define {} @{}({}) {{",
        print_ty(&f.ty.ret),
        f.name,
        f.params
            .iter()
            .enumerate()
            .map(|(i, p)| format!("{} {}", print_ty(&f.ty.params[i]), name_of(f, *p)))
            .collect::<Vec<_>>()
            .join(", ")
    );
    for b in &f.blocks {
        let _ = writeln!(out, "{}:", b.name);
        for &i in &b.insts {
            if let Some(inst) = f.inst(i) {
                if matches!(inst, Inst::Marker { .. }) {
                    continue;
                }
                let _ = writeln!(
                    out,
                    "  {} = {}",
                    name_of(f, i),
                    print_inst(f, inst)
                );
            }
        }
        match &b.term {
            Some(t) => {
                let _ = writeln!(out, "  {}", print_term(f, t));
            }
            None => {
                let _ = writeln!(out, "  ; unterminated");
            }
        }
    }
    out.push_str("}\n");
    out
}

fn name_of(_f: &Function, v: ValueId) -> String {
    format!("%v{}", v.0)
}

fn print_ty(t: &Type) -> String {
    match t {
        Type::Void => "void".to_owned(),
        Type::Int(w) => format!("i{w}"),
        Type::Float(32) => "float".to_owned(),
        Type::Float(64) => "double".to_owned(),
        Type::Float(w) => format!("f{w}"),
        Type::Ptr => "ptr".to_owned(),
        Type::Array(e, n) => format!("[{} x {}]", n, print_ty(e)),
        Type::Struct(fields) => format!(
            "{{ {} }}",
            fields.iter().map(print_ty).collect::<Vec<_>>().join(", ")
        ),
        Type::Vector(e, n) => format!("<{} x {}>", n, print_ty(e)),
        Type::Func(ft) => format!("{} ({})", print_ty(&ft.ret), params(ft)),
    }
}

fn print_const(c: &Constant) -> String {
    match c {
        Constant::Int { bits, width } => {
            let s = crate::analysis::signed(*bits, *width);
            format!("{s}")
        }
        Constant::Zero => "zeroinitializer".to_owned(),
        Constant::Undef => "undef".to_owned(),
        Constant::Function(n) => format!("@{n}"),
        Constant::GlobalPtr(n) => format!("@{n}"),
        Constant::Aggregate(items) => format!(
            "[{}]",
            items.iter().map(print_const).collect::<Vec<_>>().join(", ")
        ),
    }
}

fn operand(f: &Function, v: ValueId) -> String {
    // Value ids are per function, so an operand naming another function's
    // value is simply not a constant here. `name_of` is total.
    match f.values.get(v.0 as usize) {
        Some(Value::Const(c)) => print_const(c),
        _ => name_of(f, v),
    }
}

fn print_inst(f: &Function, i: &Inst) -> String {
    match i {
        Inst::Bin { op, ty, lhs, rhs } => format!(
            "{} {} {}, {}",
            op.name(),
            print_ty(ty),
            operand(f, *lhs),
            operand(f, *rhs)
        ),
        Inst::Un { op, ty, value } => match op {
            crate::UnOp::Not => format!("xor {} {}, -1", print_ty(ty), operand(f, *value)),
            crate::UnOp::Neg => format!("sub {} 0, {}", print_ty(ty), operand(f, *value)),
            other => format!(
                "call {} @llvm.{}.{}({})",
                print_ty(ty),
                other.name(),
                print_ty(ty).trim_start_matches('i'),
                operand(f, *value)
            ),
        },
        Inst::Rot { left, ty, value, amount } => format!(
            "call {} @llvm.f{}.{}({}, {}, {})",
            print_ty(ty),
            if *left { "shl" } else { "shr" },
            print_ty(ty).trim_start_matches('i'),
            operand(f, *value),
            operand(f, *value),
            operand(f, *amount)
        ),
        Inst::Fence => "fence seq_cst".to_owned(),
        Inst::Icmp { pred, ty, lhs, rhs } => format!(
            "icmp {} {} {}, {}",
            pred.name(),
            print_ty(ty),
            operand(f, *lhs),
            operand(f, *rhs)
        ),
        Inst::Select { ty, cond, yes, no } => format!(
            "select i1 {}, {} {}, {}",
            operand(f, *cond),
            print_ty(ty),
            operand(f, *yes),
            operand(f, *no)
        ),
        Inst::Trunc { to, value, .. } => {
            format!("trunc {} to {}", operand(f, *value), print_ty(to))
        }
        Inst::Zext { to, value, .. } => {
            format!("zext {} to {}", operand(f, *value), print_ty(to))
        }
        Inst::Sext { to, value, .. } => {
            format!("sext {} to {}", operand(f, *value), print_ty(to))
        }
        Inst::Cast { from, to, value } => format!(
            "{} {} to {}",
            match (from, to) {
                (Type::Ptr, _) => "ptrtoint",
                (_, Type::Ptr) => "inttoptr",
                _ => "bitcast",
            },
            operand(f, *value),
            print_ty(to)
        ),
        Inst::Alloca { ty } => format!("alloca {}", print_ty(ty)),
        Inst::Load { ty, ptr, align } => format!(
            "load {}, ptr {}, align {align}",
            print_ty(ty),
            operand(f, *ptr)
        ),
        Inst::Store { ty, value, ptr, align } => format!(
            "store {} {}, ptr {}, align {align}",
            print_ty(ty),
            operand(f, *value),
            operand(f, *ptr)
        ),
        Inst::Gep { base, offset } => format!(
            "getelementptr inbounds i8, ptr {}, i64 {offset}",
            operand(f, *base)
        ),
        Inst::Call { callee, args, ty } => {
            let callee = match f.values.get(callee.0 as usize) {
                Some(Value::Const(Constant::Function(n))) => format!("@{n}"),
                _ => operand(f, *callee),
            };
            format!(
                "call {} {}({})",
                print_ty(ty),
                callee,
                args.iter().map(|a| operand(f, *a)).collect::<Vec<_>>().join(", ")
            )
        }
        Inst::Phi { ty, incoming } => format!(
            "phi {} {}",
            print_ty(ty),
            incoming
                .iter()
                .map(|(v, b)| format!("[ {}, %{} ]", operand(f, *v), f.block(*b).name))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Inst::ExtractValue { agg, index, .. } => {
            format!("extractvalue {}, {index}", operand(f, *agg))
        }
        Inst::InsertValue { agg, value, index, .. } => {
            format!("insertvalue {}, {}, {index}", operand(f, *agg), operand(f, *value))
        }
        Inst::Freeze { value, .. } => format!("freeze {}", operand(f, *value)),
        Inst::Marker { text } => text.clone(),
    }
}

fn print_term(f: &Function, t: &Terminator) -> String {
    match t {
        Terminator::Ret(None) => "ret void".to_owned(),
        Terminator::Ret(Some(v)) => format!("ret {} {}", print_ty(&f.ty(*v)), operand(f, *v)),
        Terminator::Br(b) => format!("br label %{}", f.block(*b).name),
        Terminator::CondBr { cond, yes, no } => format!(
            "br i1 {}, label %{}, label %{}",
            operand(f, *cond),
            f.block(*yes).name,
            f.block(*no).name
        ),
        Terminator::Switch { value, default, cases } => {
            let mut s = format!(
                "switch {} {}, label %{} [",
                print_ty(&f.ty(*value)),
                operand(f, *value),
                f.block(*default).name
            );
            for (v, b) in cases {
                let w = f.ty(*value).int_bits().unwrap_or(64);
                let _ = write!(
                    s,
                    " {} {}, label %{}",
                    print_ty(&f.ty(*value)),
                    *v as i128 as i64 as u64,
                    f.block(*b).name
                );
                let _ = w;
            }
            s.push_str(" ]");
            s
        }
        Terminator::Unreachable => "unreachable".to_owned(),
    }
}

pub fn pred_name(p: IntPred) -> &'static str {
    p.name()
}
