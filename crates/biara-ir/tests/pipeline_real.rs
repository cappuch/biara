//! The exact pipelines the decompiler runs must not empty a real body.
use biara_ir::{pass::run_pipeline, Builder, FuncType, Inst, Module, Terminator, Type};

fn build() -> biara_ir::Function {
    let ty = FuncType { ret: Type::Int(64), params: vec![Type::Ptr], varargs: false };
    let mut b = Builder::new("f", ty);
    let p = b.add_param(Type::Ptr, "p");
    let e = b.add_block("entry");
    let next = b.add_block("body");
    let slot = b.emit(e, Inst::Alloca { ty: Type::Int(64) });
    let v = b.emit(e, Inst::Load { ty: Type::Int(64), ptr: p, align: 8 });
    b.emit(e, Inst::Store { ty: Type::Int(64), value: v, ptr: slot, align: 8 });
    b.br(e, next);
    // A reachable body: reload the slot and add one.
    let back = b.emit(next, Inst::Load { ty: Type::Int(64), ptr: slot, align: 8 });
    let k = b.const_typed(1, 64);
    let r = b.bin(next, biara_ir::BinOp::Add, &Type::Int(64), back, k);
    // A call with side effects is a live root even with no users.
    let nn = b.const_func("helper_noreturn");
    b.emit(next, Inst::Call { callee: nn, args: vec![r], ty: Type::Void });
    b.ret(next, Some(r));
    b.finish()
}

#[test]
fn sroa_alone() {
    let mut f = build();
    run_pipeline(&mut f, "function(sroa)").unwrap();
    let insts: usize = f.blocks.iter().map(|b| b.insts.len()).sum();
    assert!(insts > 0, "sroa emptied the body: {insts}");
}

#[test]
fn dce_alone() {
    let mut f = build();
    run_pipeline(&mut f, "function(adce)").unwrap();
    let insts: usize = f.blocks.iter().map(|b| b.insts.len()).sum();
    assert!(insts > 0, "adce emptied the body: {insts}");
}

#[test]
fn the_first_pipeline_keeps_instructions() {
    let mut f = build();
    run_pipeline(&mut f, "function(sroa,early-cse<memssa>,instcombine<no-verify-fixpoint>)").unwrap();
    let insts: usize = f.blocks.iter().map(|b| b.insts.len()).sum();
    assert!(insts > 0, "first pipeline emptied the body");
}

#[test]
fn the_second_pipeline_keeps_instructions() {
    let mut f = build();
    run_pipeline(
        &mut f,
        "function(sroa,instcombine<no-verify-fixpoint>,simplifycfg,gvn,dse,adce,instcombine<no-verify-fixpoint>,simplifycfg)",
    )
    .unwrap();
    let insts: usize = f.blocks.iter().map(|b| b.insts.len()).sum();
    assert!(insts > 0, "second pipeline emptied the body");
}

#[test]
fn module_globaldce_keeps_defined_functions() {
    let mut m = Module::new("t", "", "");
    m.functions.push(build());
    let f = &mut m.functions[0];
    run_pipeline(f, "function(sroa,instcombine<no-verify-fixpoint>)").unwrap();
    let insts: usize = f.blocks.iter().map(|b| b.insts.len()).sum();
    assert!(insts > 0, "instructions lost");
    assert!(f.blocks.iter().any(|b| matches!(b.term, Some(Terminator::Ret(_)))));
}
