//! The jump-table → index-switch rewrite.

use biara_ir::{BinOp, BlockId, Builder, FuncType, Inst, Module, Terminator, Type};

fn read_table(values: Vec<u64>) -> impl Fn(u64, usize) -> Option<Vec<u8>> {
    move |addr: u64, len: usize| {
        // The table sits at 0x1000; entries are 8 bytes each.
        let idx = ((addr.wrapping_sub(0x1000)) / 8) as usize;
        let v = *values.get(idx)?;
        let bytes = v.to_le_bytes();
        Some(bytes[..len.min(8)].to_vec())
    }
}

fn module() -> (Module, BlockId) {
    // f(i): switch (load(table + i*8)) { target0, target1, default }
    let ty = FuncType {
        ret: Type::Int(64),
        params: vec![Type::Int(64)],
        varargs: false,
    };
    let mut b = Builder::new("f", ty);
    let i = b.add_param(Type::Int(64), "i");
    let entry = b.add_block("entry");
    // A guard bounds the index the way lifted code does: `if (i > 1) escape`.
    let guarded = b.add_block("guarded");
    let escape = b.add_block("escape");
    let one_const = b.const_typed(1, 64);
    let over = b.icmp(entry, biara_ir::IntPred::Ugt, &Type::Int(64), i, one_const);
    b.cond_br(entry, over, escape, guarded);
    let two = b.const_typed(2, 64);
    b.ret(escape, Some(two));
    let eight = b.const_typed(8, 64);
    let strided = b.emit(
        entry,
        Inst::Bin {
            op: BinOp::Mul,
            ty: Type::Int(64),
            lhs: i,
            rhs: eight,
        },
    );
    let base = b.const_typed(0x1000, 64);
    let addr = b.emit(
        entry,
        Inst::Bin {
            op: BinOp::Add,
            ty: Type::Int(64),
            lhs: base,
            rhs: strided,
        },
    );
    let ptr = b.emit(
        entry,
        Inst::Cast {
            from: Type::Int(64),
            to: Type::Ptr,
            value: addr,
        },
    );
    let target = b.emit(
        entry,
        Inst::Load {
            ty: Type::Int(64),
            ptr,
            align: 8,
        },
    );
    let case0 = b.add_block("case0");
    let case1 = b.add_block("case1");
    let default = b.add_block("default");
    let zero = b.const_typed(0, 64);
    b.ret(case0, Some(zero));
    let one = b.const_typed(1, 64);
    b.ret(case1, Some(one));
    let two = b.const_typed(2, 64);
    b.ret(default, Some(two));
    b.switch(guarded, target, default, vec![(0x4242, case0), (0x4343, case1)]);
    // `guarded` is entered from the guard's false arm; its terminator is the
    // switch, which `switch()` already installed.
    let mut m = Module::new("t", "", "");
    m.functions.push(b.finish());
    (m, guarded)
}

#[test]
fn a_table_switch_becomes_a_switch_on_the_index() {
    let (mut m, entry) = module();
    // The table at 0x1000 holds the code addresses the cases stand for.
    let read = read_table(vec![0x4242, 0x4343]);
    let rewrites = biara_lift::index_switches::index_switches(&mut m.functions[0], &read);
    assert_eq!(rewrites.indices, 1, "{rewrites:?}");
    let Terminator::Switch {
        value, cases, ..
    } = m.functions[0]
        .terminator(entry)
        .cloned()
        .unwrap_or(Terminator::Unreachable)
    else {
        panic!("expected a switch");
    };
    // It now switches on the index, which is the parameter.
    assert!(matches!(m.functions[0].value(value), biara_ir::Value::Param { .. }));
    assert_eq!(cases.len(), 2);
}
