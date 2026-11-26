//! Indirect-target resolution: enumerate a bounded table of code addresses.

use biara_ir::{BinOp, Builder, FuncType, Inst, Module, Terminator, Type};

fn module() -> Module {
    // f(i): `biara.target(load(table + i*8), site=5)`, with `i` bounded to
    // [0,1] by a dominating guard.
    let ty = FuncType {
        ret: Type::Void,
        params: vec![Type::Int(64)],
        varargs: false,
    };
    let mut b = Builder::new("f", ty);
    let i = b.add_param(Type::Int(64), "i");
    let entry = b.add_block("entry");
    let body = b.add_block("body");
    let escape = b.add_block("escape");
    let one = b.const_typed(1, 64);
    let over = b.icmp(entry, biara_ir::IntPred::Ugt, &Type::Int(64), i, one);
    b.cond_br(entry, over, escape, body);
    b.ret(escape, None);

    let eight = b.const_typed(8, 64);
    let strided = b.emit(body, Inst::Bin { op: BinOp::Mul, ty: Type::Int(64), lhs: i, rhs: eight });
    let base = b.const_typed(0x8000, 64);
    let addr = b.emit(body, Inst::Bin { op: BinOp::Add, ty: Type::Int(64), lhs: base, rhs: strided });
    let ptr = b.emit(body, Inst::Cast { from: Type::Int(64), to: Type::Ptr, value: addr });
    let target = b.emit(body, Inst::Load { ty: Type::Int(64), ptr, align: 8 });
    let marker = b.const_func("biara.target");
    let site = b.const_typed(5, 64);
    b.emit(
        body,
        Inst::Call {
            callee: marker,
            args: vec![target, site],
            ty: Type::Void,
        },
    );
    b.ret(body, None);

    let mut m = Module::new("t", "", "");
    m.functions.push(b.finish());
    m
}

#[test]
fn a_bounded_table_enumerates_its_targets() {
    let m = module();
    let table = [0x1111u64, 0x2222u64];
    let read = |addr: u64, len: usize| -> Option<Vec<u8>> {
        let idx = ((addr.wrapping_sub(0x8000)) / 8) as usize;
        let v = *table.get(idx)?;
        Some(v.to_le_bytes()[..len.min(8)].to_vec())
    };
    let sites = biara_lift::indirect::indirect_targets(&m.functions[0], &read, 256);
    let res = sites.get(&5).expect("site 5 resolved");
    assert!(res.targets.contains(&0x1111), "{res:?}");
    assert!(res.targets.contains(&0x2222), "{res:?}");
}
