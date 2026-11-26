//! The IR's own guarantees: the verifier rejects malformed modules, and the
//! passes reach the fixed points the analyses depend on.

use biara_ir::{
    BinOp, BlockId, Builder, Constant, FuncType, Inst, IntPred, Type, Value, ValueId, pass, verify,
};

fn ty_i64() -> Type {
    Type::i64()
}

/// `define i64 @f(i64 %a) { entry: %s = alloca i64; store %a; %v = load; ret %v }`
#[test]
fn mem2reg_promotes_a_spilled_parameter() {
    let mut b = Builder::new("f", FuncType { ret: ty_i64(), params: vec![ty_i64()], varargs: false });
    let a = b.add_param(ty_i64(), "a");
    let entry = b.add_block("entry");
    let slot = b.alloca(entry, &ty_i64());
    b.store(entry, &ty_i64(), a, slot);
    let v = b.load(entry, &ty_i64(), slot);
    b.ret(entry, Some(v));
    let mut f = b.finish();
    verify(&f).expect("well formed before promotion");

    pass::simplify(&mut f);
    verify(&f).expect("well formed after promotion");

    // The alloca and its traffic are gone, and the return takes the parameter.
    let insts: Vec<&Inst> = f.blocks[0].insts.iter().filter_map(|&i| f.inst(i)).collect();
    assert!(
        !insts.iter().any(|i| matches!(i, Inst::Alloca { .. } | Inst::Store { .. } | Inst::Load { .. })),
        "spill traffic remains: {insts:?}"
    );
    match f.terminator(BlockId(0)).unwrap_or(&biara_ir::Terminator::Unreachable) {
        biara_ir::Terminator::Ret(Some(v)) => assert_eq!(*v, a),
        other => panic!("unexpected terminator {other:?}"),
    }
}

#[test]
fn folding_evaluates_constant_arithmetic_and_comparisons() {
    let mut b = Builder::new("f", FuncType { ret: ty_i64(), params: vec![], varargs: false });
    let entry = b.add_block("entry");
    let l = b.const_typed(7, 64);
    let r = b.const_typed(5, 64);
    let sum = b.bin(entry, BinOp::Add, &ty_i64(), l, r);
    let cmp = b.icmp(entry, IntPred::Sgt, &ty_i64(), sum, r);
    let ext = b.zext(entry, &Type::i1(), &ty_i64(), cmp);
    b.ret(entry, Some(ext));
    let mut f = b.finish();

    pass::simplify(&mut f);
    verify(&f).expect("well formed after folding");

    match f.terminator(BlockId(0)).unwrap_or(&biara_ir::Terminator::Unreachable) {
        biara_ir::Terminator::Ret(Some(v)) => match f.constant(*v) {
            Some(Constant::Int { bits, width }) => {
                assert_eq!((*bits, *width), (1, 64), "expected the folded constant 1");
            }
            other => panic!("expected a folded constant, got {other:?}"),
        },
        other => panic!("unexpected terminator {other:?}"),
    }
}

#[test]
fn dce_removes_side_effect_free_dead_instructions() {
    let mut b = Builder::new("f", FuncType { ret: ty_i64(), params: vec![ty_i64()], varargs: false });
    let a = b.add_param(ty_i64(), "a");
    let entry = b.add_block("entry");
    let dead = b.bin(entry, BinOp::Mul, &ty_i64(), a, a);
    let _ = dead;
    b.ret(entry, Some(a));
    let mut f = b.finish();
    pass::simplify(&mut f);
    verify(&f).unwrap();
    assert!(
        !f.blocks[0].insts.iter().any(|&i| matches!(f.inst(i), Some(Inst::Bin { .. }))),
        "dead multiply survived"
    );
}

#[test]
fn verify_rejects_a_phi_that_disagrees_with_the_predecessors() {
    let mut b = Builder::new("f", FuncType { ret: ty_i64(), params: vec![], varargs: false });
    let entry = b.add_block("entry");
    let join = b.add_block("join");
    b.br(entry, join);
    // A phi needs one incoming per predecessor; give it none.
    let phi = b.phi(join, &ty_i64());
    let _ = phi;
    b.ret(join, Some(phi));
    let f = b.finish();
    let err = verify(&f).expect_err("a phi with no incoming must be rejected");
    assert!(err.0.contains("phi"), "error was {err}");
}

#[test]
fn verify_rejects_two_blocks_with_one_name() {
    // Branch targets are printed by name, so two blocks called "next" make
    // every branch to either one ambiguous.
    let mut b = Builder::new("f", FuncType { ret: Type::Void, params: vec![], varargs: false });
    let entry = b.add_block("entry");
    let a = b.add_block("next");
    let c = b.add_block("next");
    b.br(entry, a);
    b.ret(a, None);
    b.ret(c, None);
    let f = b.finish();
    let err = verify(&f).expect_err("duplicate block name accepted");
    assert!(err.0.contains("next"), "{err:?}");
}

#[test]
fn verify_rejects_a_branch_to_a_missing_block() {
    let mut b = Builder::new("f", FuncType { ret: Type::Void, params: vec![], varargs: false });
    let entry = b.add_block("entry");
    b.br(entry, BlockId(99));
    let f = b.finish();
    assert!(verify(&f).is_err(), "out-of-range branch accepted");
}

#[test]
fn printer_round_trips_the_shape() {
    let mut b = Builder::new("f", FuncType { ret: ty_i64(), params: vec![ty_i64()], varargs: false });
    let a = b.add_param(ty_i64(), "a");
    let entry = b.add_block("entry");
    let one = b.const_typed(1, 64);
    let v = b.bin(entry, BinOp::Add, &ty_i64(), a, one);
    b.ret(entry, Some(v));
    let f = b.finish();
    let text = biara_ir::print_function(&f);
    assert!(text.contains("define i64 @f(i64 %v0)"), "{text}");
    assert!(text.contains("add i64 %v0, 1"), "{text}");
    assert!(text.contains("ret i64 %v2"), "{text}");
}

#[test]
fn dominators_see_a_diamond() {
    // entry -> (t | f) -> join
    let mut b = Builder::new("f", FuncType { ret: Type::Void, params: vec![Type::i1()], varargs: false });
    let c = b.add_param(Type::i1(), "c");
    let entry = b.add_block("entry");
    let t = b.add_block("t");
    let f_blk = b.add_block("f");
    let join = b.add_block("join");
    b.cond_br(entry, c, t, f_blk);
    b.br(t, join);
    b.br(f_blk, join);
    b.ret(join, None);
    let func = b.finish();
    let doms = biara_ir::Dominators::new(&func);
    assert!(doms.dominates(entry, join));
    assert!(doms.dominates(entry, t));
    assert!(!doms.dominates(t, join), "t is not on every path");
    assert!(!doms.dominates(join, t));
    assert_eq!(doms.idom(join), Some(entry));
}

#[test]
fn constant_range_intersection_bounds_a_switch_index() {
    use biara_ir::ConstantRange;
    let r = ConstantRange::new(0, 7, 64);
    let narrow = r.intersect(&ConstantRange::new(3, 100, 64)).unwrap();
    assert_eq!(narrow.single_value(), None);
    assert_eq!(narrow.intervals(), vec![(3, 7)]);
    assert!(r.intersect(&ConstantRange::new(8, 9, 64)).is_none());
    assert_eq!(ConstantRange::single(5, 64).single_value(), Some(5));
}

#[test]
fn unused_value_queries_count_terminator_uses() {
    let mut b = Builder::new("f", FuncType { ret: ty_i64(), params: vec![ty_i64()], varargs: false });
    let a = b.add_param(ty_i64(), "a");
    let entry = b.add_block("entry");
    b.ret(entry, Some(a));
    let f = b.finish();
    assert!(f.uses_once(a), "the return counts as a use");
    let mut b = Builder::new("g", FuncType { ret: Type::Void, params: vec![ty_i64()], varargs: false });
    let p = b.add_param(ty_i64(), "p");
    let e = b.add_block("entry");
    b.ret(e, None);
    let g = b.finish();
    assert!(g.uses_none(p));
    let _: ValueId = p;
    let _ = Value::Const(Constant::Zero);
}
