//! Builder-created instructions must expose their neighbours: call-site
//! analysis walks backwards from a call to find the stores that fed it.
use biara_ir_compat as inkwell;
use biara_ir_compat::context::Context;

#[test]
fn a_stored_value_is_visible_from_the_store_that_follows_it() {
    let cx = Context::create();
    let m = cx.create_module("t");
    let b = cx.create_builder();
    let pty = cx.ptr_type(biara_ir_compat::AddressSpace::default());
    let ft = cx.void_type().fn_type(&[pty.into()], false);
    let f = m.add_function("f", ft, None);
    let entry = cx.append_basic_block(f, "entry");
    b.position_at_end(entry);
    let env = f.get_first_param().unwrap().into_pointer_value();

    // Store a constant, then make a call; the call must see the store.
    let zero = cx.i32_type().const_zero();
    let at = unsafe { b.build_gep(cx.i8_type(), env, &[cx.i64_type().const_int(40, false)], "h").unwrap() };
    unsafe { b.build_store(at, zero).unwrap() };
    let callee = m.add_function("g", cx.void_type().fn_type(&[], false), None);
    let call = b.build_call(callee, &[], "").unwrap();

    let inst = match call.try_as_basic_value() {
        biara_ir_compat::values::ValueKind::Instruction(i) => i,
        _ => panic!("void call should yield an instruction"),
    };
    let prev = inst.get_previous_instruction().expect("previous instruction");
    assert_eq!(prev.get_opcode(), biara_ir_compat::values::InstructionOpcode::Store);
    // The callee is the sole operand, so a call is reachable by a walk.
    assert_eq!(inst.get_num_operands(), 1);
}

#[test]
fn a_gep_names_its_offset_as_a_constant_operand() {
    let cx = Context::create();
    let m = cx.create_module("t");
    let b = cx.create_builder();
    let ft = cx.void_type().fn_type(&[cx.ptr_type(biara_ir_compat::AddressSpace::default()).into()], false);
    let f = m.add_function("f", ft, None);
    let entry = cx.append_basic_block(f, "entry");
    b.position_at_end(entry);
    let env = f.get_first_param().unwrap().into_pointer_value();
    let off = cx.i64_type().const_int(296, false);
    let g = unsafe { b.build_gep(cx.i8_type(), env, &[off], "h").unwrap() };
    assert_eq!(g.as_instruction_value().get_num_operands(), 2);
    let v = g
        .as_instruction_value()
        .get_operand(1)
        .and_then(|o| o.value())
        .expect("offset operand");
    assert_eq!(v.into_int_value().get_sign_extended_constant(), Some(296));
}

#[test]
fn a_terminator_names_its_successors() {
    let cx = Context::create();
    let m = cx.create_module("t");
    let b = cx.create_builder();
    let ft = cx.void_type().fn_type(&[], false);
    let f = m.add_function("f", ft, None);
    let entry = cx.append_basic_block(f, "entry");
    let then = cx.append_basic_block(f, "then");
    b.position_at_end(entry);
    b.build_conditional_branch(cx.bool_type().const_zero(), then, then)
        .unwrap();
    let term = entry.get_terminator().expect("terminator");
    assert_eq!(term.get_num_operands(), 2, "cond + both edges");
    assert!(term.get_operand(1).and_then(|o| o.block()).is_some(), "edge names its target");
}

#[test]
fn a_walk_back_from_a_call_in_a_called_function() {
    // A `call` built by the builder must be findable from its own block,
    // which is what call-site analysis walks backwards from.
    let cx = Context::create();
    let m = cx.create_module("t");
    let b = cx.create_builder();
    let pty = cx.ptr_type(biara_ir_compat::AddressSpace::default());
    let f = m.add_function("f", cx.void_type().fn_type(&[pty.into()], false), None);
    let g = m.add_function("g", cx.void_type().fn_type(&[], false), None);
    let entry = cx.append_basic_block(f, "entry");
    b.position_at_end(entry);
    let env = f.get_first_param().unwrap().into_pointer_value();
    let home = unsafe { b.build_gep(cx.i8_type(), env, &[cx.i64_type().const_int(56, false)], "h").unwrap() };
    unsafe { b.build_store(home, cx.i64_type().const_zero()).unwrap() };
    let call = b.build_call(g, &[], "").unwrap();
    let inst = match call.try_as_basic_value() {
        biara_ir_compat::values::ValueKind::Instruction(i) => i,
        _ => panic!("void call"),
    };
    let prev = inst.get_previous_instruction().expect("previous");
    assert_eq!(prev.get_opcode(), biara_ir_compat::values::InstructionOpcode::Store);
    // The store names its home as an i8 GEP offset.
    let g2 = prev.get_operand(1).and_then(|o| o.value()).expect("ptr");
    let gi = g2.as_instruction_value().expect("gep instruction");
    assert_eq!(gi.get_num_operands(), 2, "gep operands");
    assert!(gi.get_gep_source_element_type().is_ok(), "source element type");
}

#[test]
fn a_load_from_an_alloca_sees_the_store_that_filled_it() {
    let cx = Context::create();
    let m = cx.create_module("t");
    let b = cx.create_builder();
    let pty = cx.ptr_type(biara_ir_compat::AddressSpace::default());
    let f = m.add_function("f", cx.void_type().fn_type(&[pty.into()], false), None);
    let g = m.add_function("g", cx.void_type().fn_type(&[], false), None);
    let entry = cx.append_basic_block(f, "entry");
    b.position_at_end(entry);
    let env = f.get_first_param().unwrap().into_pointer_value();
    let slot = {
        let s = cx.i64_type();
        let _ = s;
        unsafe { b.build_alloca(cx.i64_type(), "slot").unwrap() }
    };
    let zero = cx.i64_type().const_zero();
    unsafe { b.build_store(slot, zero).unwrap() };
    let loaded = b.build_load(cx.i64_type(), slot, "v").unwrap();
    let home = unsafe { b.build_gep(cx.i8_type(), env, &[cx.i64_type().const_int(56, false)], "h").unwrap() };
    unsafe { b.build_store(home, loaded).unwrap() };
    let call = b.build_call(g, &[], "").unwrap();
    let inst = match call.try_as_basic_value() {
        biara_ir_compat::values::ValueKind::Instruction(i) => i,
        _ => panic!("void call"),
    };
    // From the call, walk back: the store to the home, then the load, then
    // the store into the alloca.
    let store = inst.get_previous_instruction().expect("store");
    assert_eq!(store.get_opcode(), biara_ir_compat::values::InstructionOpcode::Store);
    // The store's value operand is the load; the GEP between them is the
    // home pointer, so the walk visits it first.
    let loaded = store
        .get_operand(0)
        .and_then(|o| o.value())
        .and_then(|v| v.as_instruction_value())
        .expect("load");
    assert_eq!(loaded.get_opcode(), biara_ir_compat::values::InstructionOpcode::Load);
    let ptr = loaded.get_operand(0).and_then(|o| o.value()).expect("ptr");
    assert_eq!(
        ptr.as_instruction_value().map(|i| i.get_opcode()),
        Some(biara_ir_compat::values::InstructionOpcode::Alloca)
    );
    let fill = loaded.get_previous_instruction().expect("fill store");
    assert_eq!(fill.get_opcode(), biara_ir_compat::values::InstructionOpcode::Store);
    let filled = fill.get_operand(1).and_then(|o| o.value()).and_then(|v| v.as_instruction_value());
    assert_eq!(
        filled.map(|i| i.get_opcode()),
        Some(biara_ir_compat::values::InstructionOpcode::Alloca)
    );
}

#[test]
fn a_home_spilled_from_an_alloca_reads_back_the_alloca_value() {
    // The lifter keeps register homes in env but copies them through
    // allocas. `store (load slot), home` must be attributable to the alloca's
    // fill, not to the register's incoming value.
    let cx = Context::create();
    let m = cx.create_module("t");
    let b = cx.create_builder();
    let pty = cx.ptr_type(biara_ir_compat::AddressSpace::default());
    let f = m.add_function("f", cx.void_type().fn_type(&[pty.into()], false), None);
    let entry = cx.append_basic_block(f, "entry");
    b.position_at_end(entry);
    let env = f.get_first_param().unwrap().into_pointer_value();

    let i64t = cx.i64_type();
    let slot = b.build_alloca(i64t, "slot").unwrap();
    let home = unsafe {
        b.build_gep(cx.i8_type(), env, &[i64t.const_int(56, false)], "home")
            .unwrap()
    };
    // The alloca is filled with zero, then the home is written from it.
    unsafe { b.build_store(slot, i64t.const_zero()).unwrap() };
    let reloaded = b.build_load(i64t, slot, "r").unwrap();
    unsafe { b.build_store(home, reloaded).unwrap() };

    let callee = m.add_function("g", cx.void_type().fn_type(&[], false), None);
    let call = b.build_call(callee, &[], "").unwrap();
    let inst = match call.try_as_basic_value() {
        biara_ir_compat::values::ValueKind::Instruction(i) => i,
        _ => panic!("void call should yield an instruction"),
    };
    // Walking back from the call finds the store into the home, whose value
    // is the load; the load's pointer is the alloca, so the fill is reachable.
    let mut cur = inst.get_previous_instruction();
    let mut seen = Vec::new();
    while let Some(i) = cur {
        seen.push(i.get_opcode());
        if i.get_opcode() == biara_ir_compat::values::InstructionOpcode::Store {
            let v = i.get_operand(0).and_then(|o| o.value()).expect("stored value");
            let li = v.as_instruction_value().expect("stored value is an instruction");
            assert_eq!(li.get_opcode(), biara_ir_compat::values::InstructionOpcode::Load);
            let p = li
                .get_operand(0)
                .and_then(|o| o.value())
                .expect("load pointer");
            assert!(
                p.as_instruction_value()
                    .is_some_and(|x| x.get_opcode() == biara_ir_compat::values::InstructionOpcode::Alloca),
                "the spilled load reads the alloca"
            );
            return;
        }
        cur = i.get_previous_instruction();
    }
    panic!("no store before the call; saw {seen:?}");
}
