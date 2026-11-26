//! The layout analysis must recover a struct from observed accesses.

use biara_ir::{
    BinOp, BlockId, Builder, FuncType, Inst, Module, Terminator, Type,
};

/// f(p): `*(u32*)(p+8) = 1; *(u32*)(p+16) = 2;` — a two-field struct.
fn accessed_through_a_parameter() -> Module {
    let ty = FuncType {
        ret: Type::Void,
        params: vec![Type::Ptr],
        varargs: false,
    };
    let mut b = Builder::new("f", ty);
    let p = b.add_param(Type::Ptr, "p");
    let entry = b.add_block("entry");
    let mut store_at = |b: &mut Builder, off: i64, value: u128| {
        let gep = b.emit(entry, Inst::Gep { base: p, offset: off });
        let ptr = b.emit(
            entry,
            Inst::Cast {
                from: Type::Ptr,
                to: Type::Ptr,
                value: gep,
            },
        );
        let v = b.const_typed(value as i128, 32);
        b.emit(
            entry,
            Inst::Store {
                ty: Type::Int(32),
                value: v,
                ptr,
                align: 4,
            },
        );
    };
    store_at(&mut b, 8, 1);
    store_at(&mut b, 16, 2);
    b.ret(entry, None);
    let mut m = Module::new("t", "", "");
    m.functions.push(b.finish());
    m
}

#[test]
fn a_parameter_accessed_at_two_offsets_becomes_a_struct() {
    let m = accessed_through_a_parameter();
    let a = biara_lift::layouts_core::Analysis::build(&m, &[]);
    let json = a.json();
    // The export must report at least one layout with fields at 8 and 16.
    assert!(json.contains("\"layouts\""), "{json}");
    let v: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
    let layouts = v["layouts"].as_array().expect("layouts array");
    assert!(!layouts.is_empty(), "{json}");
    let fields: Vec<i64> = layouts
        .iter()
        .flat_map(|l| {
            l["fields"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|f| f["off"].as_i64())
                .collect::<Vec<_>>()
        })
        .collect();
    assert!(fields.contains(&8) && fields.contains(&16), "{json}");
}

#[test]
fn a_module_without_pointer_access_recovers_nothing() {
    let ty = FuncType {
        ret: Type::Int(64),
        params: Vec::new(),
        varargs: false,
    };
    let mut b = Builder::new("g", ty);
    let entry = b.add_block("entry");
    let k = b.const_typed(7, 64);
    b.ret(entry, Some(k));
    let mut m = Module::new("t", "", "");
    m.functions.push(b.finish());
    let a = biara_lift::layouts_core::Analysis::build(&m, &[]);
    let v: serde_json::Value = serde_json::from_str(&a.json()).expect("valid JSON");
    assert!(
        v["layouts"].as_array().is_none_or(|l| l.is_empty()),
        "{}",
        a.json()
    );

    // Keep the import used when the module has no blocks.
    let _ = BlockId(0);
    let _ = BinOp::Add;
}
