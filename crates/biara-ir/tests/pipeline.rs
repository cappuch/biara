//! The pipeline runner and range transfers the ports depend on.

use biara_ir::{pass, BinOp, Builder, Constant, FuncType, Inst, IntPred, Terminator, Type, ValueId};

fn one_block(name: &str) -> (Builder, biara_ir::BlockId) {
    let mut b = Builder::new(
        name,
        FuncType {
            ret: Type::Int(64),
            params: Vec::new(),
            varargs: false,
        },
    );
    let block = b.add_block("entry");
    (b, block)
}

#[test]
fn pipeline_rejects_an_unknown_pass() {
    let err = pass::parse_pipeline("function(instcombine,nosuchpass)").unwrap_err();
    assert!(err.contains("nosuchpass"), "{err}");
}

#[test]
fn pipeline_accepts_the_names_the_decompiler_uses() {
    let names = pass::parse_pipeline(
        "function(simplifycfg<switch-range-to-icmp>,instcombine<no-verify-fixpoint>,adce,simplifycfg)",
    )
    .unwrap();
    assert_eq!(
        names,
        vec![
            pass::NamedPass::SimplifyCfg,
            pass::NamedPass::InstCombine,
            pass::NamedPass::Adce,
            pass::NamedPass::SimplifyCfg,
        ]
    );
}

#[test]
fn pipeline_expands_the_default_levels() {
    let names = pass::parse_pipeline("default<O2>").unwrap();
    assert!(names.contains(&pass::NamedPass::Sroa));
    assert!(names.contains(&pass::NamedPass::Gvn));
}

#[test]
fn running_a_pipeline_folds_a_constant_add() {
    let (mut b, block) = one_block("add");
    let a = b.const_typed(2, 64);
    let c = b.const_typed(3, 64);
    let sum = b.emit(
        block,
        Inst::Bin {
            op: BinOp::Add,
            ty: Type::Int(64),
            lhs: a,
            rhs: c,
        },
    );
    b.ret(block, Some(sum));
    let mut f = b.finish();
    pass::run_pipeline(&mut f, "function(instcombine)").unwrap();
    // The add folded, so the return names a constant.
    let Some(Terminator::Ret(Some(v))) = f.terminator(block) else {
        panic!("expected a return");
    };
    assert!(
        matches!(f.value(*v), biara_ir::Value::Const(Constant::Int { bits: 5, .. })),
        "{:?}",
        f.value(*v)
    );
}

#[test]
fn range_transfers_wrap_at_the_width() {
    use biara_ir::ConstantRange;
    let a = ConstantRange::new(250, 255, 8);
    let b = ConstantRange::new(10, 10, 8);
    // 250..255 plus 10 is 260..265, which wraps to 4..9.
    let sum = a.add(&b);
    assert!(!sum.full, "{sum:?}");
    assert_eq!((sum.lo, sum.hi), (4, 9), "{sum:?}");
}

#[test]
fn icmp_region_bounds_a_signed_compare() {
    use biara_ir::ConstantRange;
    // `x s< 0` is the negative half.
    let r = ConstantRange::icmp_region(IntPred::Slt, 0, 8);
    assert!(!r.full);
    assert!(r.lo >= 0x80, "{r:?}");
}
