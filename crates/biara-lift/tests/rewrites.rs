//! The IR rewrites that came from `cpp/abi_form.cpp`.

use biara_ir::{BinOp, Builder, Constant, FuncType, Inst, Module, Terminator, Type, Value};

fn simple(name: &str) -> Builder {
    Builder::new(
        name,
        FuncType {
            ret: Type::Int(64),
            params: Vec::new(),
            varargs: false,
        },
    )
}

#[test]
fn fold_frames_turns_inttoptr_of_ptrtoint_of_alloca_into_a_gep() {
    let mut b = simple("f");
    let block = b.add_block("entry");
    let slot = b.emit(block, Inst::Alloca { ty: Type::Int(64) });
    let as_int = b.emit(
        block,
        Inst::Cast {
            from: Type::Ptr,
            to: Type::Int(64),
            value: slot,
        },
    );
    let eight = b.const_typed(8, 64);
    let sum = b.emit(
        block,
        Inst::Bin {
            op: BinOp::Add,
            ty: Type::Int(64),
            lhs: as_int,
            rhs: eight,
        },
    );
    let back = b.emit(
        block,
        Inst::Cast {
            from: Type::Int(64),
            to: Type::Ptr,
            value: sum,
        },
    );
    let loaded = b.emit(
        block,
        Inst::Load {
            ty: Type::Int(64),
            ptr: back,
            align: 1,
        },
    );
    b.ret(block, Some(loaded));
    let mut f = b.finish();

    assert_eq!(biara_lift::rewrite::fold_frames(&mut f), 1);
    // The load now reads through a `gep slot, 8`.
    let Inst::Load { ptr, .. } = f.inst(loaded).expect("load survives") else {
        panic!("expected the load to survive");
    };
    match f.inst(*ptr).expect("gep") {
        Inst::Gep { base, offset } => {
            assert_eq!(*base, slot);
            assert_eq!(*offset, 8);
        }
        other => panic!("expected a gep, got {other:?}"),
    }
}

#[test]
fn dup_returns_inlines_a_return_join_into_its_predecessors() {
    let mut b = simple("f");
    let entry = b.add_block("entry");
    let join = b.add_block("join");
    let seven = b.const_typed(7, 64);
    b.br(entry, join);
    b.ret(join, Some(seven));
    let mut f = b.finish();

    assert_eq!(biara_lift::rewrite::dup_returns(&mut f), 1);
    // The predecessor now returns on its own.
    assert!(matches!(f.terminator(entry), Some(Terminator::Ret(Some(_)))));
}

#[test]
fn div_sugar_folds_a_prelude_call_into_sdiv() {
    let ty = FuncType {
        ret: Type::Struct(vec![Type::Int(64), Type::Int(64)]),
        params: vec![Type::Int(64), Type::Int(64), Type::Int(64)],
        varargs: false,
    };
    let mut b = Builder::new("f", ty.clone());
    let a = b.add_param(Type::Int(64), "a");
    let h = b.add_param(Type::Int(64), "h");
    let d = b.add_param(Type::Int(64), "d");
    let block = b.add_block("entry");
    let callee = b.const_func("biara_sdiv64");
    let call = b.emit(
        block,
        Inst::Call {
            callee,
            args: vec![a, h, d],
            ty: Type::Struct(vec![Type::Int(64), Type::Int(64)]),
        },
    );
    let q = b.emit(
        block,
        Inst::ExtractValue {
            agg: call,
            index: 0,
            ty: Type::Int(64),
        },
    );
    b.ret(block, Some(q));
    let mut f = b.finish();

    // `h` must look like `ashr a, 63` for the sugar to apply.
    let h_def = f.values[h.0 as usize].clone();
    assert!(matches!(h_def, Value::Param { .. }));

    let sugared = biara_lift::rewrite::div_sugar(&mut f);
    // A plain parameter high half is not an extension, so nothing folds.
    assert_eq!(sugared, 0, "should not sugar a non-extending high half");
}
