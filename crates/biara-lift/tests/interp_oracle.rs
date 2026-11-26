//! The interpreter that replaced the LLVM JIT must actually compute.

use biara_ir::{BinOp, Builder, FuncType, IntPred, Module, Type};

fn module() -> Module {
    // f(a, b) = if a > 0 { b * 3 } else { -b }
    let ty = FuncType {
        ret: Type::Int(64),
        params: vec![Type::Int(64), Type::Int(64)],
        varargs: false,
    };
    let mut b = Builder::new("f", ty);
    let a = b.add_param(Type::Int(64), "a");
    let c = b.add_param(Type::Int(64), "b");
    let entry = b.add_block("entry");
    let then_bb = b.add_block("then");
    let else_bb = b.add_block("else");
    let join = b.add_block("join");
    let zero = b.const_typed(0, 64);
    let cond = b.icmp(entry, IntPred::Ugt, &Type::Int(64), a, zero);
    b.cond_br(entry, cond, then_bb, else_bb);
    let three = b.const_typed(3, 64);
    let t = b.bin(then_bb, BinOp::Mul, &Type::Int(64), c, three);
    b.br(then_bb, join);
    let e = b.bin(else_bb, BinOp::Sub, &Type::Int(64), zero, c);
    b.br(else_bb, join);
    let phi = b.phi(join, &Type::Int(64));
    b.add_incoming(phi, t, then_bb);
    b.add_incoming(phi, e, else_bb);
    b.ret(join, Some(phi));
    let mut m = Module::new("t", "", "");
    m.functions.push(b.finish());
    m
}

#[test]
fn the_interpreter_takes_the_true_arm() {
    let m = module();
    let mut interp = biara_lift::interp::Interp::new(&m, (0, 0));
    assert_eq!(interp.call("f", &[1, 7]).unwrap(), 21);
}

#[test]
fn the_interpreter_takes_the_false_arm_and_wraps() {
    let m = module();
    let mut interp = biara_lift::interp::Interp::new(&m, (0, 0));
    assert_eq!(interp.call("f", &[0, 5]).unwrap(), u128::from(u64::MAX - 4));
}

#[test]
fn the_interpreter_reports_a_missing_function() {
    let m = module();
    let mut interp = biara_lift::interp::Interp::new(&m, (0, 0));
    assert!(interp.call("nope", &[]).is_err());
}
