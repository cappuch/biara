//! ABI form: rewrite a lifted module whose functions are `void f(ptr %env)`
//! over the whole guest state into functions with the recovered calling
//! conventions — register arguments and stack arguments as parameters,
//! return registers as the result.
//!
//! Per function, every guest register lives in an alloca of its own and the
//! stack frame is an alloca too, so that SROA turns registers and frame
//! slots that never escape into SSA values. Calls become calls with
//! arguments; the rest of the CPU state (vector and FPU registers, state
//! that helpers read) is one global byte array, `biara_cpu`, that helpers
//! receive as `env`.
//!
//! Contracts preserved from `cpp/abi_form.cpp`:
//!
//! - A register's home in `env` becomes an alloca of the register's integer
//!   type; every use of `gep env, <home>` reads and writes that alloca
//!   instead. Vector/FPU registers (128 bits) have no alloca: their state
//!   stays in `biara_cpu`, so a scalar store leaves the rest of the physical
//!   lane untouched.
//! - Frame slots: `below` bytes lie below the entry stack pointer and
//!   `above` bytes from it up, so the entry stack pointer is
//!   `frame + below`. Incoming stack parameters travel in the caller's
//!   frame; they are copied into ours at the entry-stack-pointer-relative
//!   offsets, so guest loads from those slots see the passed values.
//!   Without that the slots read as undefined and the optimizer folds the
//!   body away. Declared narrow/boolean stack formals zero-extend into the
//!   64-bit slot, bit-exactly.
//! - Only registers the function actually addresses (touched) are synced
//!   around helpers: the CPU state is the only copy of the others.
//! - Guest bits never change: integer/pointer slots stay integers/pointers
//!   (never fp bitcasts, so layout recovery's integer/pointer roots keep
//!   working), and a narrow slot is extended back out at the boundary.
//! - A call `helper_foo(env)` keeps seeing the register state only if `env`
//!   (or a `gep` of it) is among its arguments; registers travel through
//!   `biara_cpu` in that case, and back for a helper that writes them.
//!   Helpers whose only pointer argument is env use the registers their
//!   bitcode reads and writes instead of all of them.
//! - Unknown indirect targets first try the lifted functions through one
//!   shared `biara_call_at(addr) -> i1` (a plain switch over entry
//!   addresses, built once, never per call site); only external addresses
//!   use the unresolved raw-register ABI.
//! - The returned value is the return registers: one slot for a single
//!   result, an aggregate literal for several, nothing for a void function.
//! - A helper with no side effects is declared read-only, so its result may
//!   be dropped when unused. A call to a symbol without a role is an error,
//!   never a silent no-op.
//! - The return address a call pushes is only read by the callee's return,
//!   which this pass replaces; the push store is removed at the call site.
//! - The old functions in `drop` (and the old bodies of the rewritten ones)
//!   are deleted.
//!
//! Documented conservative fallbacks, each forced by `biara_ir` modelling
//! less than LLVM:
//!
//! - **No floating-point types.** `biara_ir::Type` has integers, pointers,
//!   aggregates, vectors and functions but no `float`/`double`. `F32`,
//!   `F64` and `VEC128` slots are spelled as their integer bit pattern
//!   (`i32`, `i64`, the register width). A scalar store into a 128-bit register
//!   lane touches the same bytes `CreateStore(float)` did, so no guest bits
//!   move; only a printed C type would differ.
//! - **No function metadata.** `!biara.abi.args`, `!biara.abi.rets` and
//!   `!biara.abi.ret_class` have no channel in `biara_ir::FuncAttrs`, so
//!   they are not attached; the formals' IR types already carry the same
//!   facts, which is what a reader without the metadata falls back to.
//! - **No alias scopes.** `separateCpu` only ever *enabled* folding; without
//!   the `biara_cpu` scope and the `noalias` scopes on guest/frame accesses,
//!   surrounding loads and stores simply stay live (identical results, less
//!   folding).
//! - **No `alignment` on allocas.** The frame was 16-byte aligned; the entry
//!   stack pointer is computed from the alloca address itself, and guest IR
//!   otherwise imposes no alignment.
//! - **No `optnone`/`noinline` on the dispatcher** and no
//!   `nothrow`/`willreturn` on pure helpers: `FuncAttrs` has no matching
//!   bits. The dispatcher stays a plain switch either way; a pure helper only
//!   gains the read-only memory effect.
//! - **`sret_param`/`ret_class` stay descriptive.** In the C++ they only
//!   reached the (unavailable) metadata: the hidden pointer is already one of
//!   the ordinary `args`, so the parameter list is unchanged.
//! - **A prototype call to a symbol of another type** cannot carry LLVM's
//!   `bitcast`; the call is emitted by name and resolves through the one
//!   declaration of that name.
//! - Uses of a `drop`ped function that survive rewriting are left naming the
//!   absent symbol (LLVM replaced them with poison, which is unreachable by
//!   construction); no computed value depends on it.

use std::collections::{BTreeSet, HashMap};

use biara_ir::{
    BinOp, Block, BlockId, Constant, Declaration, FuncAttrs, FuncType, Function, Global, Inst,
    MemoryEffect, Module, Terminator, Type, Value, ValueId,
};

// ABI value kinds: the native type a slot lowers to. Integer/pointer slots
// stay integers/pointers (never fp bitcasts, so layout recovery's
// integer/pointer roots keep working); unknown/conflict slots stay raw
// words.
const KIND_I32: u32 = 0;
const KIND_I64: u32 = 1;
const KIND_F32: u32 = 2;
const KIND_F64: u32 = 3;
const KIND_VEC128: u32 = 4;
const KIND_PTR: u32 = 5;
const KIND_SRET: u32 = 6;
const KIND_RAW: u32 = 7;
const KIND_CONFLICT: u32 = 8;
const KIND_I8: u32 = 9;
const KIND_I16: u32 = 10;
const KIND_BOOL: u32 = 11;
const KIND_I32DECL: u32 = 12;

const RET_DIRECT: u32 = 0;
const RET_PAIR: u32 = 1;
const RET_MEMORY: u32 = 2;

const CALL_DIRECT: u32 = 0;
const CALL_IMPORT: u32 = 1;
const CALL_INDIRECT: u32 = 2;
const CALL_SYSCALL: u32 = 3;

/// The CPU-state global: everything outside the recovered register slots.
pub const CPU_GLOBAL: &str = "biara_cpu";
/// The shared indirect-target dispatcher.
pub const CALL_AT: &str = "biara_call_at";
/// The syscall thunk the convention's registers travel through.
pub const SYSCALL: &str = "biara_syscall";

/// The type one ABI slot lowers to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Kind {
    I32 = KIND_I32,
    I64 = KIND_I64,
    F32 = KIND_F32,
    F64 = KIND_F64,
    Vec128 = KIND_VEC128,
    Ptr = KIND_PTR,
    Sret = KIND_SRET,
    Raw = KIND_RAW,
    Conflict = KIND_CONFLICT,
    I8 = KIND_I8,
    I16 = KIND_I16,
    Bool = KIND_BOOL,
    I32Decl = KIND_I32DECL,
}

/// How a return value travels: direct value, two-element aggregate, or a
/// hidden-`sret` pointer parameter. `Memory` records the store-through-
/// pointer fact, never a fabricated source struct type.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum RetClass {
    #[default]
    Direct = RET_DIRECT,
    Pair = RET_PAIR,
    Memory = RET_MEMORY,
}

/// What one `call` of the lifted module stands for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum CallKind {
    Direct = CALL_DIRECT,
    Import = CALL_IMPORT,
    Indirect = CALL_INDIRECT,
    Syscall = CALL_SYSCALL,
}

/// A guest register (a TCG global in env).
#[derive(Clone, Debug)]
pub struct Reg {
    /// Offset in env.
    pub home: i64,
    pub bits: u32,
    pub name: String,
    /// Thread state (segment bases): read from the CPU state on entry rather
    /// than passed.
    pub thread: bool,
}

/// The recovered signature of a lifted function.
#[derive(Clone, Debug, Default)]
pub struct Sig {
    /// Function name in the module.
    pub function: String,
    /// Register indices, in parameter order.
    pub args: Vec<u32>,
    /// Offsets from the entry stack pointer.
    pub stack_args: Vec<i64>,
    /// Register indices holding the result.
    pub rets: Vec<u32>,
    /// Frame bytes below the entry stack pointer.
    pub below: u64,
    /// Frame bytes from the entry stack pointer up.
    pub above: u64,
    /// Guest address.
    pub entry: u64,
    /// The frame restores the stack pointer on return.
    pub balanced: bool,
    /// Registers kept, in the analysis's order.
    pub preserved: Vec<u32>,
    /// Per-parameter ABI kinds (`Kind` codes, length `args + stack_args`;
    /// stack slots carry their declared scalar kind: `I64` by default,
    /// `I8`/`I16`/`I32Decl`/`Bool` for declared narrow/boolean spilled
    /// scalars, coerced bit-exactly at the slot boundary). `None` means
    /// all-integer.
    pub arg_kinds: Option<Vec<u32>>,
    /// Per-result ABI kinds (`Kind` codes, length `rets`; `None` means
    /// all-integer). A single `F32`/`F64`/`Vec128` result is that slot type;
    /// a `Pair` return is a two-element aggregate of its member kinds.
    pub ret_kinds: Option<Vec<u32>>,
    /// Return shape.
    pub ret_class: RetClass,
    /// Hidden-`sret` parameter position in the new function (`None` when
    /// `ret_class` is not [`RetClass::Memory`]).
    pub sret_param: Option<u32>,
}

/// A call site of the lifted module and what it stands for.
#[derive(Clone, Debug)]
pub struct Call {
    /// The containing function's name.
    pub function: String,
    /// The call in that function's arena (before ABI form).
    pub inst: ValueId,
    pub kind: CallKind,
    /// The import's symbol, for [`CallKind::Import`].
    pub import: Option<String>,
    /// Arguments passed to unknown code; `None` infers them.
    pub arity: Option<u32>,
    /// Return address; `0` for tail calls and escapes.
    pub ret: u64,
}

/// A helper taking env, and how it uses the registers in it.
#[derive(Clone, Debug, Default)]
pub struct HelperRole {
    pub symbol: String,
    pub reads: bool,
    pub writes: bool,
    /// No side effects: its result may be dropped when unused.
    pub pure_: bool,
    /// With `listed`, the registers it reads and writes through env (and no
    /// others); otherwise any.
    pub listed: bool,
    pub reads_regs: Vec<u32>,
    pub writes_regs: Vec<u32>,
    /// When env itself is every pointer argument: the registers its bitcode
    /// reads and writes (`None`: not known).
    pub env_reads: Option<Vec<u32>>,
    pub env_writes: Option<Vec<u32>>,
}

/// The calling convention of the guest.
#[derive(Clone, Debug, Default)]
pub struct Conv {
    pub sp: u32,
    pub pc: u32,
    /// What returning adds to the stack pointer at the call.
    pub pop: i64,
    pub args: Vec<u32>,
    pub ret: u32,
    pub sys_args: Vec<u32>,
    pub sys_ret: u32,
    pub env_size: u64,
    /// Bytes of CPU state before env (negative offsets).
    pub margin: u64,
    /// Unknown indirect targets first try the lifted functions
    /// (`biara_call_at`); otherwise they are only called through the pointer.
    pub dispatch: bool,
}

/// Rewrite `module` into ABI form; the old functions named by `sigs` and
/// `drop` are removed. Returns the first failure, prefixed with its function.
pub fn form(
    module: &mut Module,
    conv: &Conv,
    regs: &[Reg],
    sigs: &[Sig],
    calls: &[Call],
    helpers: &[HelperRole],
    drop: &[String],
) -> Result<(), String> {
    validate(conv, regs, sigs, calls, helpers)?;
    let mut fm = Former::new(module, conv, regs, sigs, calls, helpers);
    fm.pure_helpers();
    fm.ensure_cpu_global();
    for s in sigs {
        fm.declare(s);
    }
    let dispatch = if conv.dispatch {
        Some(fm.build_dispatcher(fm.sigs))
    } else {
        None
    };
    let mut defined = Vec::new();
    for s in sigs {
        match fm.translate(s) {
            Ok(Some(f)) => defined.push(f),
            Ok(None) => {}
            Err(e) => return Err(format!("{}: {e}", s.function)),
        }
    }
    fm.finish(drop, defined, dispatch);
    // A block the rewrite created but never reached has no control-flow fact
    // of its own; stating `unreachable` is the truth, and leaving it open
    // would make the module unverifiable.
    for f in &mut fm.module.functions {
        for b in &mut f.blocks {
            if b.term.is_none() {
                b.term = Some(biara_ir::Terminator::Unreachable);
            }
        }
        biara_ir::analysis::recompute_preds(f);
    }
    Ok(())
}

/// Reject input the pass would otherwise mis-index: every register reference
/// must exist, `sp`/`pc` must be integer registers, and kind arrays sized.
fn validate(
    conv: &Conv,
    regs: &[Reg],
    sigs: &[Sig],
    calls: &[Call],
    helpers: &[HelperRole],
) -> Result<(), String> {
    let n = regs.len() as u32;
    let reg = |r: u32| -> Result<(), String> {
        if r < n {
            Ok(())
        } else {
            Err(format!("register {r} is out of range ({n} registers)"))
        }
    };
    for r in [conv.sp, conv.pc, conv.ret, conv.sys_ret] {
        reg(r)?;
    }
    for &r in conv.args.iter().chain(conv.sys_args.iter()) {
        reg(r)?;
    }
    for (name, r) in [("stack pointer", conv.sp), ("program counter", conv.pc)] {
        if regs[r as usize].bits == 128 {
            return Err(format!("the {name} is a vector register"));
        }
    }
    for s in sigs {
        for &r in s.args.iter().chain(s.rets.iter()).chain(s.preserved.iter()) {
            reg(r)?;
        }
        if let Some(k) = &s.arg_kinds
            && k.len() != s.args.len() + s.stack_args.len()
        {
            return Err(format!(
                "{}: {} ABI kinds for {} register and {} stack parameters",
                s.function,
                k.len(),
                s.args.len(),
                s.stack_args.len()
            ));
        }
        if let Some(k) = &s.ret_kinds
            && k.len() != s.rets.len()
        {
            return Err(format!(
                "{}: {} return kinds for {} results",
                s.function,
                k.len(),
                s.rets.len()
            ));
        }
    }
    for c in calls {
        if c.kind == CallKind::Import && c.import.as_deref().unwrap_or("").is_empty() {
            return Err(format!(
                "{}: an import call without an import symbol",
                c.function
            ));
        }
    }
    for h in helpers {
        for &r in h
            .reads_regs
            .iter()
            .chain(h.writes_regs.iter())
            .chain(h.env_reads.iter().flatten())
            .chain(h.env_writes.iter().flatten())
        {
            reg(r)?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Types

fn reg_ty(regs: &[Reg], r: u32) -> Type {
    Type::Int(regs[r as usize].bits)
}

/// 128-bit registers are vector/FPU state: no slot, the CPU state is their
/// home.
fn fp(regs: &[Reg], r: u32) -> bool {
    regs[r as usize].bits == 128
}

fn sig_arg_kind(s: &Sig, i: usize) -> u32 {
    s.arg_kinds
        .as_ref()
        .and_then(|k| k.get(i).copied())
        .unwrap_or(KIND_I64)
}

fn sig_ret_kind(s: &Sig, i: usize) -> u32 {
    s.ret_kinds
        .as_ref()
        .and_then(|k| k.get(i).copied())
        .unwrap_or(KIND_I64)
}

/// The native type of ABI slot `kind` over register `r`: bool/i8/i16/
/// i32-declared slots spell the IR directly; pointers stay pointers (never
/// fp bitcasts); raw/conflict/integer slots stay the register type. FP and
/// vector kinds are their integer bit pattern (see the module doc).
fn slot_ty(regs: &[Reg], r: u32, kind: u32) -> Type {
    match kind {
        KIND_BOOL => Type::i1(),
        KIND_I8 => Type::i8(),
        KIND_I16 => Type::i16(),
        KIND_I32DECL => Type::i32(),
        KIND_F32 => Type::i32(),
        KIND_F64 => Type::i64(),
        KIND_VEC128 => reg_ty(regs, r),
        KIND_PTR | KIND_SRET => Type::Ptr,
        // Raw/conflict/integer slots stay the register's integer type: no
        // established float/vector/pointer fact to narrow them.
        KIND_I32 | KIND_I64 | KIND_RAW | KIND_CONFLICT => reg_ty(regs, r),
        _ => reg_ty(regs, r),
    }
}

/// The type of stack formal `i` (declared scalar kind, else `i64`).
fn stack_ty(s: &Sig, i: usize) -> Type {
    match sig_arg_kind(s, s.args.len() + i) {
        KIND_BOOL => Type::i1(),
        KIND_I8 => Type::i8(),
        KIND_I16 => Type::i16(),
        KIND_I32DECL => Type::i32(),
        _ => Type::i64(),
    }
}

/// The type of new-function parameter `i` (register formals, then stack).
fn param_ty(s: &Sig, regs: &[Reg], i: usize) -> Type {
    if i < s.args.len() {
        slot_ty(regs, s.args[i], sig_arg_kind(s, i))
    } else {
        stack_ty(s, i - s.args.len())
    }
}

/// The type of the new function's result.
fn ret_ty(s: &Sig, regs: &[Reg]) -> Type {
    match s.rets.len() {
        0 => Type::Void,
        1 => slot_ty(regs, s.rets[0], sig_ret_kind(s, 0)),
        n => Type::Struct(
            (0..n)
                .map(|i| slot_ty(regs, s.rets[i], sig_ret_kind(s, i)))
                .collect(),
        ),
    }
}

// ---------------------------------------------------------------------------
// Value construction

/// An insertion cursor: the next instruction goes before `block.insts[pos]`.
struct At {
    block: BlockId,
    pos: usize,
}

fn emit(f: &mut Function, at: &mut At, inst: Inst) -> ValueId {
    f.values.push(Value::Inst(inst));
    f.defined_in.push(Some(at.block));
    f.pc.push(None);
    let id = ValueId((f.values.len() - 1) as u32);
    let insts = &mut f.blocks[at.block.0 as usize].insts;
    let pos = at.pos.min(insts.len());
    insts.insert(pos, id);
    at.pos = pos + 1;
    id
}

fn push_block(f: &mut Function, name: &str) -> BlockId {
    let id = BlockId(f.blocks.len() as u32);
    f.blocks.push(Block {
        name: name.to_owned(),
        insts: Vec::new(),
        term: None,
        preds: BTreeSet::new(),
    });
    if f.entry.is_none() {
        f.entry = Some(id);
    }
    id
}

fn add_param(f: &mut Function, ty: Type, name: &str) -> ValueId {
    let index = f.params.len() as u32;
    f.values.push(Value::Param {
        ty,
        index,
        name: name.to_owned(),
    });
    f.defined_in.push(None);
    f.pc.push(None);
    let id = ValueId((f.values.len() - 1) as u32);
    f.params.push(id);
    id
}

fn const_int(f: &mut Function, v: i64, width: u32) -> ValueId {
    f.add_const(Constant::int(v as i128, width))
}

fn const_u64(f: &mut Function, v: u64, width: u32) -> ValueId {
    f.add_const(Constant::int(v as i128, width))
}

fn load_at(f: &mut Function, at: &mut At, ty: &Type, ptr: ValueId) -> ValueId {
    let align = biara_ir::ty_bytes(ty);
    emit(
        f,
        at,
        Inst::Load {
            ty: ty.clone(),
            ptr,
            align,
        },
    )
}

fn store_at(f: &mut Function, at: &mut At, ty: &Type, value: ValueId, ptr: ValueId) -> ValueId {
    let align = biara_ir::ty_bytes(ty);
    emit(
        f,
        at,
        Inst::Store {
            ty: ty.clone(),
            value,
            ptr,
            align,
        },
    )
}

/// Convert `v` from `from` to `to`, bit-exactly: integers extend/truncate,
/// pointers cross through `i64` (never an invalid width cast).
fn convert(f: &mut Function, at: &mut At, from: &Type, to: &Type, v: ValueId) -> ValueId {
    if from == to {
        return v;
    }
    match (from, to) {
        (Type::Int(a), Type::Int(b)) => {
            if a < b {
                emit(
                    f,
                    at,
                    Inst::Zext {
                        from: from.clone(),
                        to: to.clone(),
                        value: v,
                    },
                )
            } else {
                emit(
                    f,
                    at,
                    Inst::Trunc {
                        from: from.clone(),
                        to: to.clone(),
                        value: v,
                    },
                )
            }
        }
        (Type::Ptr, _) => {
            let i = emit(
                f,
                at,
                Inst::Cast {
                    from: Type::Ptr,
                    to: Type::i64(),
                    value: v,
                },
            );
            convert(f, at, &Type::i64(), to, i)
        }
        (_, Type::Ptr) => {
            let i = convert(f, at, from, &Type::i64(), v);
            emit(
                f,
                at,
                Inst::Cast {
                    from: Type::i64(),
                    to: Type::Ptr,
                    value: i,
                },
            )
        }
        _ => emit(
            f,
            at,
            Inst::Cast {
                from: from.clone(),
                to: to.clone(),
                value: v,
            },
        ),
    }
}

fn slot_of(slots: &[Option<ValueId>], r: u32) -> ValueId {
    slots[r as usize].expect("an integer register has a slot")
}

/// `env`-relative byte address of `reg`'s home.
fn cpu_at(f: &mut Function, at: &mut At, env: ValueId, reg: &Reg) -> ValueId {
    emit(
        f,
        at,
        Inst::Gep {
            base: env,
            offset: reg.home,
        },
    )
}

/// Load register `r` from the CPU state as `want` (vector/FPU only, and the
/// dispatcher, which has no slots).
fn read_cpu(
    f: &mut Function,
    at: &mut At,
    regs: &[Reg],
    env: ValueId,
    r: u32,
    want: &Type,
) -> ValueId {
    let p = cpu_at(f, at, env, &regs[r as usize]);
    let rty = reg_ty(regs, r);
    if fp(regs, r) {
        return load_at(f, at, want, p);
    }
    let v = load_at(f, at, &rty, p);
    convert(f, at, &rty, want, v)
}

/// Store `v` into register `r`'s home in the CPU state.
fn write_cpu(f: &mut Function, at: &mut At, regs: &[Reg], env: ValueId, r: u32, v: ValueId) {
    let rty = reg_ty(regs, r);
    let v = if fp(regs, r) {
        v
    } else {
        let have = f.ty(v);
        convert(f, at, &have, &rty, v)
    };
    let p = cpu_at(f, at, env, &regs[r as usize]);
    let vt = f.ty(v);
    store_at(f, at, &vt, v, p);
}

/// Load register `r` from its slot as `want` (the CPU state for vectors).
fn read_slot(
    f: &mut Function,
    at: &mut At,
    regs: &[Reg],
    env: ValueId,
    slots: &[Option<ValueId>],
    r: u32,
    want: &Type,
) -> ValueId {
    if fp(regs, r) {
        let p = cpu_at(f, at, env, &regs[r as usize]);
        return load_at(f, at, want, p);
    }
    let rty = reg_ty(regs, r);
    let v = load_at(f, at, &rty, slot_of(slots, r));
    convert(f, at, &rty, want, v)
}

/// Store `v` into register `r`'s slot (the CPU state for vectors).
fn write_slot(
    f: &mut Function,
    at: &mut At,
    regs: &[Reg],
    env: ValueId,
    slots: &[Option<ValueId>],
    r: u32,
    v: ValueId,
) {
    if fp(regs, r) {
        // A scalar store leaves the remaining physical lane bytes untouched.
        let p = cpu_at(f, at, env, &regs[r as usize]);
        let vt = f.ty(v);
        store_at(f, at, &vt, v, p);
        return;
    }
    let rty = reg_ty(regs, r);
    let have = f.ty(v);
    let v = convert(f, at, &have, &rty, v);
    store_at(f, at, &rty, v, slot_of(slots, r));
}

/// Load register `r` as integer `ty` (zero-extended or truncated): the
/// convention's raw argument registers.
fn load_as(
    f: &mut Function,
    at: &mut At,
    regs: &[Reg],
    env: ValueId,
    slots: &[Option<ValueId>],
    r: u32,
    ty: &Type,
) -> ValueId {
    let rty = reg_ty(regs, r);
    let v = if fp(regs, r) {
        let p = cpu_at(f, at, env, &regs[r as usize]);
        load_at(f, at, &rty, p)
    } else {
        load_at(f, at, &rty, slot_of(slots, r))
    };
    convert(f, at, &rty, ty, v)
}

/// A dead value no translated instruction references.
fn placeholder(f: &mut Function, ty: &Type) -> ValueId {
    // A `Marker` has no result type, so it cannot stand in for a value a kept
    // instruction still reads: `freeze ty` gives the placeholder the type its
    // uses require.
    f.values.push(Value::Inst(Inst::Marker {
        text: String::new(),
    }));
    f.defined_in.push(None);
    f.pc.push(None);
    let m = ValueId((f.values.len() - 1) as u32);
    f.values.push(Value::Inst(Inst::Freeze {
        ty: ty.clone(),
        value: m,
    }));
    f.defined_in.push(None);
    f.pc.push(None);
    ValueId((f.values.len() - 1) as u32)
}

/// A zero aggregate of `ty` — the aggregate `insertvalue` fills, without
/// introducing `undef`.
fn zero_of(f: &mut Function, ty: &Type) -> ValueId {
    fn zero(t: &Type) -> Constant {
        match t {
            Type::Int(w) => Constant::zero_int(*w),
            Type::Vector(e, n) => Constant::Aggregate(vec![zero(e); *n as usize]),
            Type::Array(e, n) => Constant::Aggregate(vec![zero(e); *n as usize]),
            Type::Struct(fields) => Constant::Aggregate(fields.iter().map(zero).collect()),
            _ => Constant::Zero,
        }
    }
    let c = zero(ty);
    f.add_const(c)
}

// ---------------------------------------------------------------------------
// The pass

struct Former<'a> {
    module: &'a mut Module,
    conv: &'a Conv,
    regs: &'a [Reg],
    sigs: &'a [Sig],
    helpers: &'a [HelperRole],
    /// Register home -> register, for registers that have a slot.
    reg_at: HashMap<i64, u32>,
    sig_of: HashMap<String, usize>,
    helper_of: HashMap<String, usize>,
    /// Call records, by containing function, then by value id.
    calls: HashMap<String, HashMap<u32, usize>>,
    call_list: Vec<Call>,
    dispatch_reads: BTreeSet<u32>,
    dispatch_writes: BTreeSet<u32>,
    dispatch: bool,
}

impl<'a> Former<'a> {
    fn new(
        module: &'a mut Module,
        conv: &'a Conv,
        regs: &'a [Reg],
        sigs: &'a [Sig],
        calls: &[Call],
        helpers: &'a [HelperRole],
    ) -> Self {
        let mut reg_at = HashMap::new();
        for (r, reg) in regs.iter().enumerate() {
            if reg.bits != 128 {
                reg_at.insert(reg.home, r as u32);
            }
        }
        let mut by_fn: HashMap<String, HashMap<u32, usize>> = HashMap::new();
        let mut call_list = Vec::new();
        for c in calls {
            let i = call_list.len();
            by_fn
                .entry(c.function.clone())
                .or_default()
                .insert(c.inst.0, i);
            call_list.push(c.clone());
        }
        Self {
            module,
            conv,
            regs,
            sigs,
            helpers,
            reg_at,
            sig_of: sigs
                .iter()
                .enumerate()
                .map(|(i, s)| (s.function.clone(), i))
                .collect(),
            helper_of: helpers
                .iter()
                .enumerate()
                .map(|(i, h)| (h.symbol.clone(), i))
                .collect(),
            calls: by_fn,
            call_list,
            dispatch_reads: BTreeSet::new(),
            dispatch_writes: BTreeSet::new(),
            dispatch: conv.dispatch,
        }
    }

    fn sig(&self, name: &str) -> Option<&Sig> {
        self.sig_of.get(name).map(|&i| &self.sigs[i])
    }

    fn helper(&self, name: &str) -> Option<&HelperRole> {
        self.helper_of.get(name).map(|&i| &self.helpers[i])
    }

    fn call_record(&self, function: &str, v: u32) -> Option<&Call> {
        let i = *self.calls.get(function)?.get(&v)?;
        Some(&self.call_list[i])
    }

    /// Helpers without side effects go when their result is unused: only the
    /// read-only memory effect is modellable (see the module doc).
    fn pure_helpers(&mut self) {
        let read_only = MemoryEffect {
            argmem_read: false,
            argmem_write: false,
            inaccessible_read: true,
            inaccessible_write: false,
        };
        for h in self.helpers {
            if !h.pure_ {
                continue;
            }
            if let Some(d) = self.module.decls.iter_mut().find(|d| d.name == h.symbol) {
                d.attrs.memory = Some(read_only);
            }
            if let Some(f) = self
                .module
                .functions
                .iter_mut()
                .find(|f| f.name == h.symbol)
            {
                f.attrs.memory = Some(read_only);
            }
        }
    }

    /// `biara_cpu`: the rest of the guest state, as one byte array.
    fn ensure_cpu_global(&mut self) {
        if self.module.globals.iter().any(|g| g.name == CPU_GLOBAL) {
            return;
        }
        self.module.globals.push(Global {
            name: CPU_GLOBAL.to_owned(),
            ty: Type::Array(
                Box::new(Type::i8()),
                self.conv.margin.saturating_add(self.conv.env_size),
            ),
            init: None,
            internal: false,
        });
    }

    fn has_definition(&self, name: &str) -> bool {
        self.module
            .functions
            .iter()
            .any(|f| f.name == name && f.entry.is_some() && !f.blocks.is_empty())
    }

    /// State the new native signature (as a declaration when the body belongs
    /// to another job) and record what the dispatcher must pass.
    fn declare(&mut self, s: &Sig) {
        if !self.has_definition(&s.function) {
            let ty = FuncType {
                ret: ret_ty(s, self.regs),
                params: (0..s.args.len() + s.stack_args.len())
                    .map(|i| param_ty(s, self.regs, i))
                    .collect(),
                varargs: false,
            };
            if let Some(d) = self.module.decls.iter_mut().find(|d| d.name == s.function) {
                d.ty = ty;
            } else {
                self.module.decls.push(Declaration {
                    name: s.function.clone(),
                    ty,
                    attrs: FuncAttrs::default(),
                });
            }
        }
        for &r in &s.args {
            if !fp(self.regs, r) {
                self.dispatch_reads.insert(r);
            }
        }
        if !s.stack_args.is_empty() {
            self.dispatch_reads.insert(self.conv.sp);
        }
        for &r in &s.rets {
            if !fp(self.regs, r) {
                self.dispatch_writes.insert(r);
            }
        }
    }

    /// The shared dispatcher: one switch on the guest address, calling the
    /// lifted function whose entry it is.
    fn build_dispatcher(&self, sigs: &[Sig]) -> Function {
        let fty = FuncType {
            ret: Type::i1(),
            params: vec![Type::i64()],
            varargs: false,
        };
        let mut f = Function::new(CALL_AT, fty);
        let addr = add_param(&mut f, Type::i64(), "addr");
        let entry = push_block(&mut f, "entry");
        let cpu = f.add_const(Constant::GlobalPtr(CPU_GLOBAL.to_owned()));
        let env = emit(
            &mut f,
            &mut At { block: entry, pos: 0 },
            Inst::Gep {
                base: cpu,
                offset: self.conv.margin as i64,
            },
        );
        let mut cases = Vec::new();
        let mut seen = BTreeSet::new();
        for s in sigs {
            // One case per address: a repeated entry would make two.
            if !seen.insert(s.entry) {
                continue;
            }
            // A case block is named by the guest entry it dispatches to:
            // every case needs a distinct label or the branches are
            // ambiguous.
            let cb = push_block(&mut f, &format!("at_{:x}", s.entry));
            cases.push((s.entry as u128, cb));
            let mut at = At { block: cb, pos: 0 };
            let mut args = Vec::new();
            for i in 0..s.args.len() {
                let want = param_ty(s, self.regs, i);
                args.push(read_cpu(&mut f, &mut at, self.regs, env, s.args[i], &want));
            }
            for i in 0..s.stack_args.len() {
                let sp = read_cpu(
                    &mut f,
                    &mut at,
                    self.regs,
                    env,
                    self.conv.sp,
                    &Type::i64(),
                );
                let off = const_int(&mut f, s.stack_args[i], 64);
                let a = emit(
                    &mut f,
                    &mut at,
                    Inst::Bin {
                        op: BinOp::Add,
                        ty: Type::i64(),
                        lhs: sp,
                        rhs: off,
                    },
                );
                let p = emit(
                    &mut f,
                    &mut at,
                    Inst::Cast {
                        from: Type::i64(),
                        to: Type::Ptr,
                        value: a,
                    },
                );
                let v = load_at(&mut f, &mut at, &Type::i64(), p);
                let want = param_ty(s, self.regs, s.args.len() + i);
                args.push(trunc_if_narrow(&mut f, &mut at, v, &want));
            }
            let callee = f.add_const(Constant::Function(s.function.clone()));
            let rty = ret_ty(s, self.regs);
            let r = emit(
                &mut f,
                &mut at,
                Inst::Call {
                    callee,
                    args,
                    ty: rty,
                },
            );
            for i in 0..s.rets.len() {
                let v = if s.rets.len() == 1 {
                    r
                } else {
                    emit(
                        &mut f,
                        &mut at,
                        Inst::ExtractValue {
                            agg: r,
                            index: i as u32,
                            ty: slot_ty(self.regs, s.rets[i], sig_ret_kind(s, i)),
                        },
                    )
                };
                write_cpu(&mut f, &mut at, self.regs, env, s.rets[i], v);
            }
            let yes = const_int(&mut f, 1, 1);
            f.blocks[cb.0 as usize].term = Some(Terminator::Ret(Some(yes)));
        }
        let other = push_block(&mut f, "other");
        let no = const_int(&mut f, 0, 1);
        f.blocks[other.0 as usize].term = Some(Terminator::Ret(Some(no)));
        f.blocks[entry.0 as usize].term = Some(Terminator::Switch {
            value: addr,
            default: other,
            cases,
        });
        biara_ir::analysis::recompute_preds(&mut f);
        f
    }

    /// Translate one lifted function into its ABI form. `None` when there is
    /// no body to translate (the native prototype stands as declared).
    fn translate(&mut self, s: &Sig) -> Result<Option<Function>, String> {
        let Some(pos) = self
            .module
            .functions
            .iter()
            .position(|f| f.name == s.function)
        else {
            return Ok(None);
        };
        let old = &self.module.functions[pos];
        if old.entry.is_none() || old.blocks.is_empty() {
            // A prototype without a body: `declare` already stated its native
            // signature as a declaration.
            return Ok(None);
        }
        let old_env = match old.params.len() {
            0 => None,
            1 => Some(old.params[0]),
            n => {
                return Err(format!(
                    "lifted function takes {n} parameters; only the env pointer is understood"
                ));
            }
        };
        let old = self.module.functions.remove(pos);
        let fty = FuncType {
            ret: ret_ty(s, self.regs),
            params: (0..s.args.len() + s.stack_args.len())
                .map(|i| param_ty(s, self.regs, i))
                .collect(),
            varargs: false,
        };
        let mut f = Function::new(s.function.clone(), fty);
        // The formed function keeps the facts the signature summary reads:
        // its guest address and the per-formal travel kinds.
        f.attrs.entry_addr = Some(s.entry);
        f.attrs.abi_arg_kinds = s.arg_kinds.clone().unwrap_or_default();
        f.attrs.abi_ret_kinds = s.ret_kinds.clone().unwrap_or_default();
        f.attrs.abi_ret_class = Some(s.ret_class as u32);
        for i in 0..s.args.len() {
            let ty = param_ty(s, self.regs, i);
            add_param(&mut f, ty, &self.regs[s.args[i] as usize].name);
        }
        for i in 0..s.stack_args.len() {
            let ty = param_ty(s, self.regs, s.args.len() + i);
            add_param(&mut f, ty, &format!("stack_{}", s.stack_args[i]));
        }
        let regs = self.regs;
        let sp_reg = self.conv.sp;
        let pc_reg = self.conv.pc;
        let entry = push_block(&mut f, "entry");
        let mut at = At { block: entry, pos: 0 };
        // The env pointer: the CPU state past the margin.
        let cpu = f.add_const(Constant::GlobalPtr(CPU_GLOBAL.to_owned()));
        let env = emit(
            &mut f,
            &mut at,
            Inst::Gep {
                base: cpu,
                offset: self.conv.margin as i64,
            },
        );
        // One slot per integer register.
        let mut slots: Vec<Option<ValueId>> = vec![None; regs.len()];
        for r in 0..regs.len() as u32 {
            if !fp(regs, r) {
                let ty = reg_ty(regs, r);
                slots[r as usize] = Some(emit(&mut f, &mut at, Inst::Alloca { ty }));
            }
        }
        // The stack frame, and the entry stack pointer in it.
        let frame_ty = Type::Array(Box::new(Type::i8()), s.below.saturating_add(s.above));
        let frame = emit(&mut f, &mut at, Inst::Alloca { ty: frame_ty });
        let base = emit(
            &mut f,
            &mut at,
            Inst::Cast {
                from: Type::Ptr,
                to: Type::i64(),
                value: frame,
            },
        );
        let below = const_u64(&mut f, s.below, 64);
        let sp_in = emit(
            &mut f,
            &mut at,
            Inst::Bin {
                op: BinOp::Add,
                ty: Type::i64(),
                lhs: base,
                rhs: below,
            },
        );
        write_slot(&mut f, &mut at, regs, env, &slots, sp_reg, sp_in);
        let pcv = const_u64(&mut f, s.entry, regs[pc_reg as usize].bits);
        write_slot(&mut f, &mut at, regs, env, &slots, pc_reg, pcv);
        for r in 0..regs.len() as u32 {
            if !regs[r as usize].thread {
                continue;
            }
            let rty = reg_ty(regs, r);
            let v = read_cpu(&mut f, &mut at, regs, env, r, &rty);
            write_slot(&mut f, &mut at, regs, env, &slots, r, v);
        }
        for i in 0..s.args.len() {
            let p = f.params[i];
            write_slot(&mut f, &mut at, regs, env, &slots, s.args[i], p);
        }
        // Incoming stack parameters arrive in the caller's frame.
        for i in 0..s.stack_args.len() {
            let p = f.params[s.args.len() + i];
            let off = const_int(&mut f, s.stack_args[i], 64);
            let a = emit(
                &mut f,
                &mut at,
                Inst::Bin {
                    op: BinOp::Add,
                    ty: Type::i64(),
                    lhs: sp_in,
                    rhs: off,
                },
            );
            let ap = emit(
                &mut f,
                &mut at,
                Inst::Cast {
                    from: Type::i64(),
                    to: Type::Ptr,
                    value: a,
                },
            );
            let v = join_to_i64(&mut f, &mut at, p);
            store_at(&mut f, &mut at, &Type::i64(), v, ap);
        }

        // Which register homes this body addresses, and the env-addressing
        // sites that become slot accesses or disappear.
        let mut touched = vec![false; regs.len()];
        touched[sp_reg as usize] = true;
        touched[pc_reg as usize] = true;
        let mut owned_gep: HashMap<u32, u32> = HashMap::new();
        let mut elided: BTreeSet<u32> = BTreeSet::new();
        if let Some(ev) = old_env {
            for b in &old.blocks {
                for &v in &b.insts {
                    match old.inst(v) {
                        Some(Inst::Gep { base, offset }) if *base == ev => {
                            if let Some(&r) = self.reg_at.get(offset) {
                                touched[r as usize] = true;
                                owned_gep.insert(v.0, r);
                                elided.insert(v.0);
                            }
                        }
                        Some(Inst::Load { ptr, .. }) | Some(Inst::Store { ptr, .. })
                            if *ptr == ev =>
                        {
                            if let Some(&r) = self.reg_at.get(&0) {
                                touched[r as usize] = true;
                            }
                        }
                        Some(Inst::Store { ptr, .. }) => {
                            // Discard writes to icount and I/O flags before env.
                            // Loads still read from the zeroed CPU-state margin.
                            if let Some(Inst::Gep { base, offset }) = old.inst(*ptr)
                                && *base == ev
                                && *offset < 0
                                && !self.reg_at.contains_key(offset)
                            {
                                elided.insert(v.0);
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        // Translate the body: one id remap, placeholders for forward
        // references (phis), and every block gets its new identity first.
        let mut block_map: Vec<BlockId> = vec![entry; old.blocks.len()];
        for (bi, b) in old.blocks.iter().enumerate() {
            if Some(BlockId(bi as u32)) == old.entry {
                block_map[bi] = entry;
            } else {
                let name = if b.name.is_empty() { "block" } else { &b.name };
                block_map[bi] = push_block(&mut f, name);
            }
        }
        let mut map: Vec<Option<ValueId>> = vec![None; old.values.len()];
        let mut rev: HashMap<u32, u32> = HashMap::new();
        for (i, val) in old.values.iter().enumerate() {
            let id = ValueId(i as u32);
            map[i] = match val {
                Value::Param { .. } => {
                    if Some(id) != old_env {
                        return Err("lifted function has a non-env parameter".to_owned());
                    }
                    Some(env)
                }
                Value::Const(c) => Some(f.add_const(c.clone())),
                Value::Inst(_) => match owned_gep.get(&id.0) {
                    Some(&r) => Some(slot_of(&slots, r)),
                    None => Some(placeholder(&mut f, &old.ty(id))),
                },
            };
            if let Some(nv) = map[i] {
                rev.insert(nv.0, i as u32);
            }
        }
        // The register-at-home-0 form: `load/store ... ptr %env` itself.
        let home0 = self.reg_at.get(&0).map(|&r| slot_of(&slots, r));
        for (bi, b) in old.blocks.iter().enumerate() {
            let nb = block_map[bi];
            for &ov in &b.insts {
                if owned_gep.contains_key(&ov.0) {
                    continue;
                }
                let target = map[ov.0 as usize].expect("every value is mapped");
                if elided.contains(&ov.0) {
                    // A dropped store: its value stays an unused marker.
                    continue;
                }
                let mut inst = translate_inst(&old, ov, &map, &block_map, old_env, home0);
                // A store's declared width must match the value it stores. A
                // placeholder stands in for a value the rewrite could not
                // preserve and carries the original type, while the declared
                // width comes from the original instruction: once a use is
                // remapped the two can disagree.
                let mut drop = false;
                if let Inst::Store { value, .. } = &inst {
                    if (value.0 as usize) >= f.values.len() {
                        // The remap produced an id this arena does not hold:
                        // the store states nothing and would mistype the body.
                        drop = true;
                    } else {
                        let have = f.ty(*value);
                        // A `Void` value is a placeholder that survived as a
                        // store's operand: nothing can be stored.
                        if have == Type::Void {
                            drop = true;
                        } else if let Inst::Store { ty: t, .. } = &mut inst
                            && *t != have
                        {
                            *t = have;
                        }
                    }
                }
                if drop {
                    // Leave a typed constant: the slot must not keep the stale
                    // instruction an earlier block wrote there.
                    f.values[target.0 as usize] = Value::Const(Constant::Zero);
                    continue;
                }
                f.values[target.0 as usize] = Value::Inst(inst);
                f.defined_in[target.0 as usize] = Some(nb);
                f.pc[target.0 as usize] = old.pc_of(ov);
                f.blocks[nb.0 as usize].insts.push(target);
            }
            let term = translate_term(
                &old.terminator_or_unreachable(BlockId(bi as u32)),
                &map,
                &block_map,
            );
            f.blocks[nb.0 as usize].term = Some(term);
        }
        // Instructions the old arena held outside any block (dead code an
        // earlier pass unlinked): keep them typed but unplaced. A `Void`
        // marker would mistype any surviving use.
        for (i, val) in old.values.iter().enumerate() {
            let id = ValueId(i as u32);
            if !matches!(val, Value::Inst(_)) || owned_gep.contains_key(&id.0) {
                continue;
            }
            if old.defined_in[i].is_some() {
                continue;
            }
            let target = map[i].expect("every value is mapped");
            if elided.contains(&id.0) || (target.0 as usize) >= f.values.len() {
                continue;
            }
            let inst = translate_inst(&old, id, &map, &block_map, old_env, home0);
            f.values[target.0 as usize] = Value::Inst(inst);
            f.defined_in[target.0 as usize] = None;
            f.pc[target.0 as usize] = old.pc_of(id);
        }
        biara_ir::analysis::recompute_preds(&mut f);

        // The lowering's promote alloca per register (its slot before ABI
        // form): a sync store loads from it, so it sets nothing either.
        let mut promote: Vec<Option<ValueId>> = vec![None; regs.len()];
        let mut slot_owner: HashMap<ValueId, u32> = HashMap::new();
        for (r, slot) in slots.iter().enumerate() {
            if let Some(v) = slot {
                slot_owner.insert(*v, r as u32);
            }
        }
        for &v in &f.blocks[entry.0 as usize].insts.clone() {
            let Some(Inst::Store { value, ptr, .. }) = f.inst(v) else {
                continue;
            };
            let Some(Inst::Load { ptr: lp, .. }) = f.inst(*value) else {
                continue;
            };
            let Some(&r) = slot_owner.get(lp) else {
                continue;
            };
            if matches!(f.inst(*ptr), Some(Inst::Alloca { .. })) {
                promote[r as usize] = Some(*ptr);
            }
        }

        // Call sites, in program order.
        let sites: Vec<ValueId> = f
            .blocks
            .iter()
            .flat_map(|b| b.insts.iter().copied())
            .filter(|&v| matches!(f.inst(v), Some(Inst::Call { .. })))
            .collect();
        for v in sites {
            if !in_block(&f, v) {
                continue;
            }
            self.rewrite_call(&mut f, s, env, &slots, &promote, &touched, &rev, v)?;
        }

        // Returns: the return registers become the result.
        let nrets = s.rets.len();
        for bi in 0..f.blocks.len() {
            if !matches!(f.terminator(BlockId(bi as u32)), Some(Terminator::Ret(_))) {
                continue;
            }
            let b = BlockId(bi as u32);
            let mut at = At {
                block: b,
                pos: f.blocks[bi].insts.len(),
            };
            let value = match nrets {
                0 => None,
                1 => {
                    let want = slot_ty(regs, s.rets[0], sig_ret_kind(s, 0));
                    Some(read_slot(&mut f, &mut at, regs, env, &slots, s.rets[0], &want))
                }
                _ => {
                    let fields: Vec<Type> = (0..nrets)
                        .map(|i| slot_ty(regs, s.rets[i], sig_ret_kind(s, i)))
                        .collect();
                    let agg_ty = Type::Struct(fields.clone());
                    let mut agg = zero_of(&mut f, &agg_ty);
                    for (i, field) in fields.iter().enumerate() {
                        let v = read_slot(&mut f, &mut at, regs, env, &slots, s.rets[i], field);
                        agg = emit(
                            &mut f,
                            &mut at,
                            Inst::InsertValue {
                                agg,
                                value: v,
                                index: i as u32,
                                ty: agg_ty.clone(),
                            },
                        );
                    }
                    Some(agg)
                }
            };
            f.blocks[bi].term = Some(Terminator::Ret(value));
        }
        biara_ir::analysis::recompute_preds(&mut f);
        Ok(Some(f))
    }

    // ---- calls ----------------------------------------------------------

    /// Registers a call may change: everything, unless it is a helper (its
    /// listed writes, or none) or a lifted callee (what its ABI does not
    /// preserve).
    fn clobbered(&self, callee: Option<&str>) -> BTreeSet<u32> {
        let mut w: BTreeSet<u32> = (0..self.regs.len() as u32).collect();
        let Some(name) = callee else {
            return w;
        };
        if let Some(s) = self.sig(name) {
            w.clear();
            if s.balanced {
                let kept: BTreeSet<u32> = s.preserved.iter().copied().collect();
                for r in 0..self.regs.len() as u32 {
                    if !kept.contains(&r) {
                        w.insert(r);
                    }
                }
            }
        } else if let Some(h) = self.helper(name) {
            w.clear();
            if h.writes {
                if h.listed {
                    w.extend(h.writes_regs.iter().copied());
                } else {
                    w.extend(0..self.regs.len() as u32);
                }
            }
        }
        w
    }

    /// Arguments passed to unknown code: the convention's registers up to the
    /// last one set on the way here (single predecessors, back through calls
    /// that preserve them, to the entry), or that the function itself
    /// received and no call clobbered.
    fn infer_arity(
        &self,
        f: &Function,
        s: &Sig,
        slots: &[Option<ValueId>],
        promote: &[Option<ValueId>],
        block: BlockId,
        upto: usize,
    ) -> u32 {
        let conv = &self.conv.args;
        let n = conv.len();
        let mut set = vec![false; n];
        let mut live = vec![true; n];
        let mut at_entry = false;
        let mut seen: BTreeSet<BlockId> = BTreeSet::new();
        let mut cur = block;
        let mut pos = upto;
        for _ in 0..=f.blocks.len() {
            for i in (0..pos).rev() {
                let v = f.blocks[cur.0 as usize].insts[i];
                match f.inst(v) {
                    Some(Inst::Store { value, ptr, .. }) => {
                        for k in 0..n {
                            if !live[k] {
                                continue;
                            }
                            let slot = slots[conv[k] as usize];
                            let prom = promote[conv[k] as usize];
                            if Some(*ptr) != slot && prom != Some(*ptr) {
                                continue;
                            }
                            // A copy of the register's own value (entry loads,
                            // syncs, reloads) sets nothing; anything else
                            // does, and ends the search for this register.
                            let copy = match f.inst(*value) {
                                Some(Inst::Load { ptr: lp, .. }) => {
                                    Some(*lp) == slot || prom == Some(*lp)
                                }
                                _ => false,
                            };
                            if !copy {
                                set[k] = true;
                                live[k] = false;
                            }
                        }
                    }
                    Some(Inst::Call { callee, .. }) => {
                        let name = match f.value(*callee) {
                            Value::Const(Constant::Function(n)) => Some(n.as_str()),
                            _ => None,
                        };
                        for r in self.clobbered(name) {
                            for k in 0..n {
                                if conv[k] == r {
                                    live[k] = false;
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
            if Some(cur) == f.entry {
                at_entry = true;
                break;
            }
            let preds = &f.blocks[cur.0 as usize].preds;
            if preds.len() != 1 {
                break;
            }
            let p = *preds.iter().next().expect("one predecessor");
            if !seen.insert(p) {
                break;
            }
            cur = p;
            pos = f.blocks[p.0 as usize].insts.len();
        }
        let mut count = 0;
        for k in 0..n {
            let param = at_entry && live[k] && s.args.contains(&conv[k]);
            if set[k] || param {
                count = (k + 1) as u32;
            }
        }
        count
    }

    /// Rewrite one call site in place.
    fn rewrite_call(
        &mut self,
        f: &mut Function,
        s: &Sig,
        env: ValueId,
        slots: &[Option<ValueId>],
        promote: &[Option<ValueId>],
        touched: &[bool],
        rev: &HashMap<u32, u32>,
        v: ValueId,
    ) -> Result<(), String> {
        let Some(block) = f.defined_in[v.0 as usize] else {
            return Ok(());
        };
        // Call records name the pre-translation arena; find this call's old id.
        let record = rev
            .get(&v.0)
            .and_then(|&old| self.call_record(&s.function, old))
            .cloned();
        let kind = record.as_ref().map(|c| c.kind);
        let ret_addr = record.as_ref().map_or(0, |c| c.ret);
        let callee_name = match f.inst(v) {
            Some(Inst::Call { callee, .. }) => match f.value(*callee) {
                Value::Const(Constant::Function(n)) => Some(n.clone()),
                _ => None,
            },
            _ => None,
        };
        // The pushed return address is only read by the callee's return,
        // which ABI form replaces.
        if ret_addr != 0 && self.conv.pop > 0 {
            let idx = f.blocks[block.0 as usize]
                .insts
                .iter()
                .position(|&x| x == v)
                .unwrap_or(0);
            if let Some((b, i)) = find_pushed_return(f, block, idx, ret_addr) {
                f.blocks[b.0 as usize].insts.remove(i);
            }
        }
        let Some(idx) = f.blocks[block.0 as usize]
            .insts
            .iter()
            .position(|&x| x == v)
        else {
            return Ok(());
        };
        let regs = self.regs;
        let sp_reg = self.conv.sp;
        let mut at = At { block, pos: idx };
        let sp = read_slot(f, &mut at, regs, env, slots, sp_reg, &reg_ty(regs, sp_reg));

        // A direct call to a lifted function goes to its exact native type.
        if let Some(name) = &callee_name
            && let Some(t) = self.sig(name).cloned()
            && kind.is_none_or(|k| k == CallKind::Direct)
        {
            invoke(f, &mut at, regs, self.conv, env, slots, &t, sp, ret_addr);
            erase(f, block, v);
            return Ok(());
        }
        match kind {
            Some(CallKind::Import | CallKind::Indirect) => {
                // Native direct calls already use the callee signature.
                // Infer register arguments only for the raw-register ABI.
                let arity = record
                    .as_ref()
                    .and_then(|c| c.arity)
                    .filter(|&arity| arity != u32::MAX)
                    .or_else(|| callee_name.as_deref().and_then(crate::cgen::known_proto_arity))
                    .unwrap_or_else(|| self.infer_arity(f, s, slots, promote, block, idx));
                // Known guest targets are called through their exact native
                // types; only external addresses use the unresolved
                // raw-register ABI.
                let pc_reg = self.conv.pc;
                let target = read_slot(f, &mut at, regs, env, slots, pc_reg, &reg_ty(regs, pc_reg));
                let target = convert(f, &mut at, &reg_ty(regs, pc_reg), &Type::i64(), target);
                if kind == Some(CallKind::Indirect) && !self.dispatch {
                    let p = emit(
                        f,
                        &mut at,
                        Inst::Cast {
                            from: Type::i64(),
                            to: Type::Ptr,
                            value: target,
                        },
                    );
                    let r = call_unknown(f, &mut at, regs, env, slots, p, &self.conv.args, arity);
                    write_slot(f, &mut at, regs, env, slots, self.conv.ret, r);
                    resumed(f, &mut at, regs, self.conv, env, slots, sp, ret_addr);
                    erase(f, block, v);
                    return Ok(());
                }
                if kind == Some(CallKind::Indirect) {
                    for &r in &self.dispatch_reads {
                        let rty = reg_ty(regs, r);
                        let x = read_slot(f, &mut at, regs, env, slots, r, &rty);
                        write_cpu(f, &mut at, regs, env, r, x);
                    }
                    let callee = f.add_const(Constant::Function(CALL_AT.to_owned()));
                    let known = emit(
                        f,
                        &mut at,
                        Inst::Call {
                            callee,
                            args: vec![target],
                            ty: Type::i1(),
                        },
                    );
                    // Split the block after the call: the tail moves to
                    // `indirect.resume`; this block branches on whether a
                    // lifted function answered.
                    let call_index = f.blocks[block.0 as usize]
                        .insts
                        .iter()
                        .position(|&x| x == v)
                        .expect("the call is in its block");
                    let tail: Vec<ValueId> = f.blocks[block.0 as usize]
                        .insts
                        .split_off(call_index + 1);
                    f.blocks[block.0 as usize].insts.pop();
                    let cont = push_block(f, &format!("indirect.resume.{}", v.0));
                    f.blocks[cont.0 as usize].insts = tail.clone();
                    for x in tail {
                        f.defined_in[x.0 as usize] = Some(cont);
                    }
                    let old_term = f.blocks[block.0 as usize]
                        .term
                        .take()
                        .unwrap_or(Terminator::Unreachable);
                    f.blocks[cont.0 as usize].term = Some(old_term);
                    let lifted = push_block(f, &format!("indirect.lifted.{}", v.0));
                    let external = push_block(f, &format!("indirect.external.{}", v.0));
                    f.blocks[block.0 as usize].term = Some(Terminator::CondBr {
                        cond: known,
                        yes: lifted,
                        no: external,
                    });
                    let mut lat = At { block: lifted, pos: 0 };
                    for &r in &self.dispatch_writes {
                        let rty = reg_ty(regs, r);
                        let x = read_cpu(f, &mut lat, regs, env, r, &rty);
                        write_slot(f, &mut lat, regs, env, slots, r, x);
                    }
                    resumed(f, &mut lat, regs, self.conv, env, slots, sp, ret_addr);
                    f.blocks[lifted.0 as usize].term = Some(Terminator::Br(cont));
                    let mut eat = At {
                        block: external,
                        pos: 0,
                    };
                    let p = emit(
                        f,
                        &mut eat,
                        Inst::Cast {
                            from: Type::i64(),
                            to: Type::Ptr,
                            value: target,
                        },
                    );
                    let r = call_unknown(f, &mut eat, regs, env, slots, p, &self.conv.args, arity);
                    write_slot(f, &mut eat, regs, env, slots, self.conv.ret, r);
                    resumed(f, &mut eat, regs, self.conv, env, slots, sp, ret_addr);
                    f.blocks[external.0 as usize].term = Some(Terminator::Br(cont));
                    biara_ir::analysis::recompute_preds(f);
                    return Ok(());
                }
                let symbol = record.and_then(|c| c.import).unwrap_or_default();
                let name = self.import_decl(&symbol);
                let callee = f.add_const(Constant::Function(name));
                let r = call_unknown(f, &mut at, regs, env, slots, callee, &self.conv.args, arity);
                let want = reg_ty(regs, self.conv.ret);
                let have = f.ty(r);
                let r = convert(f, &mut at, &have, &want, r);
                write_slot(f, &mut at, regs, env, slots, self.conv.ret, r);
                resumed(f, &mut at, regs, self.conv, env, slots, sp, ret_addr);
                erase(f, block, v);
                Ok(())
            }
            Some(CallKind::Syscall) => {
                let params: Vec<Type> = self.conv.sys_args.iter().map(|_| Type::i64()).collect();
                let name = self.declare_fn(
                    SYSCALL,
                    FuncType {
                        ret: Type::i64(),
                        params,
                        varargs: false,
                    },
                );
                let callee = f.add_const(Constant::Function(name));
                let r = call_unknown(
                    f,
                    &mut at,
                    regs,
                    env,
                    slots,
                    callee,
                    &self.conv.sys_args,
                    self.conv.sys_args.len() as u32,
                );
                let want = reg_ty(regs, self.conv.sys_ret);
                let have = f.ty(r);
                let r = convert(f, &mut at, &have, &want, r);
                write_slot(f, &mut at, regs, env, slots, self.conv.sys_ret, r);
                erase(f, block, v);
                Ok(())
            }
            _ => {
                // A division prelude call, then helpers.
                if let Some(name) = &callee_name
                    && self.div_op(f, &mut at, regs, env, slots, name, v)?
                {
                    erase(f, block, v);
                    return Ok(());
                }
                let Some(name) = callee_name else {
                    return Err("call through a pointer in lifted code".to_owned());
                };
                // A role decides how the registers reach the callee. Only a
                // call that has none *and* is a pure intrinsic needs no sync:
                // an intrinsic reads its operands and writes its result,
                // never the guest register state.
                let Some(h) = self.helper(&name).cloned() else {
                    if name.starts_with("llvm.") {
                        return Ok(());
                    }
                    return Err(format!("call to {name} without a role"));
                };
                self.sync_helper(f, &mut at, regs, env, slots, touched, v, &h);
                Ok(())
            }
        }
    }

    /// A helper sees the register state only through env: sync the registers
    /// it reads before the call and the ones it writes after.
    #[allow(clippy::too_many_arguments)]
    fn sync_helper(
        &self,
        f: &mut Function,
        at: &mut At,
        regs: &[Reg],
        env: ValueId,
        slots: &[Option<ValueId>],
        touched: &[bool],
        v: ValueId,
        h: &HelperRole,
    ) {
        let call_args: Vec<ValueId> = match f.inst(v) {
            Some(Inst::Call { args, .. }) => args.clone(),
            _ => Vec::new(),
        };
        let mut sees_env = false;
        for &a in &call_args {
            let is_gep_of_env = matches!(f.inst(a), Some(Inst::Gep { base, .. }) if *base == env);
            sees_env |= a == env || is_gep_of_env;
        }
        let all: Vec<u32> = (0..regs.len() as u32).collect();
        let mut reads = if h.listed {
            h.reads_regs.clone()
        } else {
            all.clone()
        };
        let mut writes = if h.listed {
            h.writes_regs.clone()
        } else {
            all
        };
        let mut env_only = h.env_reads.is_some();
        for &a in &call_args {
            env_only &= f.ty(a) != Type::Ptr || a == env;
        }
        if env_only {
            reads = h.env_reads.clone().unwrap_or_default();
            writes = h.env_writes.clone().unwrap_or_default();
        }
        let live = |r: u32| (r as usize) < regs.len() && touched[r as usize];
        reads.retain(|&r| live(r));
        writes.retain(|&r| live(r));
        if sees_env && h.reads {
            for &r in &reads {
                if fp(regs, r) {
                    continue;
                }
                let rty = reg_ty(regs, r);
                let x = read_slot(f, at, regs, env, slots, r, &rty);
                write_cpu(f, at, regs, env, r, x);
            }
        }
        if sees_env && h.writes {
            let after = f.blocks[at.block.0 as usize]
                .insts
                .iter()
                .position(|&x| x == v)
                .map_or(0, |i| i + 1);
            let mut wat = At {
                block: at.block,
                pos: after,
            };
            for &r in &writes {
                if fp(regs, r) {
                    continue;
                }
                let rty = reg_ty(regs, r);
                let x = read_cpu(f, &mut wat, regs, env, r, &rty);
                write_slot(f, &mut wat, regs, env, slots, r, x);
            }
        }
    }

    /// An x86 division helper (`idivq_EAX` etc.): a call to a prelude
    /// function with QEMU's semantics (quotient of the RDX:RAX pair, or the
    /// low half for widths below 64; a zero divisor or an overflowing
    /// quotient raises, so a kept result is the plain C one). The caller
    /// already guards a zero divisor. `Ok(false)` when this is not a div op.
    #[allow(clippy::too_many_arguments)]
    fn div_op(
        &mut self,
        f: &mut Function,
        at: &mut At,
        regs: &[Reg],
        env: ValueId,
        slots: &[Option<ValueId>],
        callee: &str,
        v: ValueId,
    ) -> Result<bool, String> {
        const OPS: &[(&str, &str)] = &[
            ("divb_AL", "biara_div8"),
            ("idivb_AL", "biara_sdiv8"),
            ("divw_AX", "biara_div16"),
            ("idivw_AX", "biara_sdiv16"),
            ("divl_EAX", "biara_div32"),
            ("idivl_EAX", "biara_sdiv32"),
            ("divq_EAX", "biara_div64"),
            ("idivq_EAX", "biara_sdiv64"),
        ];
        let bare = callee.strip_prefix("helper_").unwrap_or(callee);
        let Some(&(_, target)) = OPS.iter().find(|(n, _)| *n == bare) else {
            return Ok(false);
        };
        let Some(args) = (match f.inst(v) {
            Some(Inst::Call { args, .. }) => Some(args.clone()),
            _ => None,
        }) else {
            return Ok(false);
        };
        if args.len() < 2 {
            // Not the shape the prelude takes; the helper path decides.
            return Ok(false);
        }
        let find = |want: &str| -> Result<u32, String> {
            regs.iter()
                .position(|r| r.name == want)
                .map(|i| i as u32)
                .ok_or_else(|| format!("div without {want}"))
        };
        let rax = find("rax")?;
        let rdx = find("rdx")?;
        let qr = Type::Struct(vec![Type::i64(), Type::i64()]);
        let name = self.declare_fn(
            target,
            FuncType {
                ret: qr.clone(),
                params: vec![Type::i64(), Type::i64(), Type::i64()],
                varargs: false,
            },
        );
        let lo_ty = reg_ty(regs, rax);
        let hi_ty = reg_ty(regs, rdx);
        let lo = read_slot(f, at, regs, env, slots, rax, &lo_ty);
        let hi = read_slot(f, at, regs, env, slots, rdx, &hi_ty);
        let lo = convert(f, at, &lo_ty, &Type::i64(), lo);
        let hi = convert(f, at, &hi_ty, &Type::i64(), hi);
        let d_have = f.ty(args[1]);
        let d = convert(f, at, &d_have, &Type::i64(), args[1]);
        let callee = f.add_const(Constant::Function(name));
        let r = emit(
            f,
            at,
            Inst::Call {
                callee,
                args: vec![lo, hi, d],
                ty: qr.clone(),
            },
        );
        for (i, reg) in [(0u32, rax), (1u32, rdx)] {
            let field = emit(
                f,
                at,
                Inst::ExtractValue {
                    agg: r,
                    index: i,
                    ty: Type::i64(),
                },
            );
            write_slot(f, at, regs, env, slots, reg, field);
        }
        Ok(true)
    }

    /// An import declaration: a varargs `i64`-returning symbol with the raw
    /// register ABI.
    fn import_decl(&mut self, name: &str) -> String {
        let fty = FuncType {
            ret: Type::i64(),
            params: Vec::new(),
            varargs: true,
        };
        if let Some(d) = self.module.decls.iter().find(|d| d.name == name)
            && d.ty == fty
        {
            return name.to_owned();
        }
        // A name taken by another symbol: the import keeps its own.
        let taken = self.module.decls.iter().any(|d| d.name == name)
            || self.module.functions.iter().any(|f| f.name == name);
        let n = if taken {
            format!("{name}.import")
        } else {
            name.to_owned()
        };
        if !self.module.decls.iter().any(|d| d.name == n)
            && !self.module.functions.iter().any(|f| f.name == n)
        {
            self.module.decls.push(Declaration {
                name: n.clone(),
                ty: fty,
                attrs: FuncAttrs::default(),
            });
        }
        n
    }

    /// Make sure `name` is declared with `ty`; a pre-existing symbol keeps
    /// its own declaration (see the module doc).
    fn declare_fn(&mut self, name: &str, ty: FuncType) -> String {
        if !self.module.decls.iter().any(|d| d.name == name)
            && !self.module.functions.iter().any(|f| f.name == name)
        {
            self.module.decls.push(Declaration {
                name: name.to_owned(),
                ty,
                attrs: FuncAttrs::default(),
            });
        }
        name.to_owned()
    }

    /// Drop the old bodies and install the rewritten ones.
    fn finish(&mut self, drop: &[String], defined: Vec<Function>, dispatch: Option<Function>) {
        let gone: BTreeSet<String> = self
            .sigs
            .iter()
            .map(|s| s.function.clone())
            .chain(drop.iter().cloned())
            .collect();
        let dropped: BTreeSet<String> = drop.iter().cloned().collect();
        self.module
            .functions
            .retain(|f| !gone.contains(&f.name));
        // A signature's native prototype stays (its body belongs to another
        // job); the driver declarations being dropped go.
        self.module
            .decls
            .retain(|d| !dropped.contains(&d.name));
        self.module.functions.extend(defined);
        if let Some(d) = dispatch {
            self.module.functions.push(d);
        }
    }
}

// ---------------------------------------------------------------------------
// Body translation

/// Is `v` still an instruction of a block?
fn in_block(f: &Function, v: ValueId) -> bool {
    f.defined_in
        .get(v.0 as usize)
        .copied()
        .flatten()
        .is_some_and(|b| f.blocks[b.0 as usize].insts.contains(&v))
}

fn erase(f: &mut Function, block: BlockId, v: ValueId) {
    f.blocks[block.0 as usize].insts.retain(|&x| x != v);
    if f.defined_in[v.0 as usize] == Some(block) {
        f.defined_in[v.0 as usize] = None;
    }
}

fn mapped(map: &[Option<ValueId>], v: ValueId) -> ValueId {
    map[v.0 as usize].expect("every operand is mapped")
}

/// Truncate a 64-bit stack slot to a declared narrow formal.
fn trunc_if_narrow(f: &mut Function, at: &mut At, v: ValueId, want: &Type) -> ValueId {
    match want.int_bits() {
        Some(w) if w < 64 => emit(
            f,
            at,
            Inst::Trunc {
                from: Type::i64(),
                to: want.clone(),
                value: v,
            },
        ),
        _ => v,
    }
}

/// Zero-extend a declared narrow/boolean stack formal into the 64-bit slot,
/// bit-exactly.
fn join_to_i64(f: &mut Function, at: &mut At, p: ValueId) -> ValueId {
    let ty = f.ty(p);
    match ty.int_bits() {
        Some(w) if w < 64 => emit(
            f,
            at,
            Inst::Zext {
                from: ty,
                to: Type::i64(),
                value: p,
            },
        ),
        _ => p,
    }
}

/// Rebuild `inst` in the new function: operands are remapped, and the two
/// env-addressing forms become the register's slot.
fn translate_inst(
    old: &Function,
    v: ValueId,
    map: &[Option<ValueId>],
    block_map: &[BlockId],
    old_env: Option<ValueId>,
    home0: Option<ValueId>,
) -> Inst {
    let m = |x: ValueId| mapped(map, x);
    let inst = old.inst(v).expect("an instruction value").clone();
    let env_ptr = |ptr: ValueId| {
        if Some(ptr) == old_env {
            // `load/store ... ptr %env`: the register whose home is 0.
            home0.unwrap_or_else(|| m(ptr))
        } else {
            m(ptr)
        }
    };
    match inst {
        Inst::Bin { op, ty, lhs, rhs } => Inst::Bin {
            op,
            ty,
            lhs: m(lhs),
            rhs: m(rhs),
        },
        Inst::Un { op, ty, value } => Inst::Un {
            op,
            ty,
            value: m(value),
        },
        Inst::Rot {
            left,
            ty,
            value,
            amount,
        } => Inst::Rot {
            left,
            ty,
            value: m(value),
            amount: m(amount),
        },
        Inst::Fence => Inst::Fence,
        Inst::Icmp { pred, ty, lhs, rhs } => Inst::Icmp {
            pred,
            ty,
            lhs: m(lhs),
            rhs: m(rhs),
        },
        Inst::Select { ty, cond, yes, no } => Inst::Select {
            ty,
            cond: m(cond),
            yes: m(yes),
            no: m(no),
        },
        Inst::Trunc { from, to, value } => Inst::Trunc {
            from,
            to,
            value: m(value),
        },
        Inst::Zext { from, to, value } => Inst::Zext {
            from,
            to,
            value: m(value),
        },
        Inst::Sext { from, to, value } => Inst::Sext {
            from,
            to,
            value: m(value),
        },
        Inst::Cast { from, to, value } => Inst::Cast {
            from,
            to,
            value: m(value),
        },
        Inst::Alloca { ty } => Inst::Alloca { ty },
        Inst::Load { ty, ptr, align } => Inst::Load {
            ty,
            ptr: env_ptr(ptr),
            align,
        },
        Inst::Store {
            ty,
            value,
            ptr,
            align,
        } => Inst::Store {
            ty,
            value: m(value),
            ptr: env_ptr(ptr),
            align,
        },
        Inst::Gep { base, offset } => Inst::Gep {
            base: m(base),
            offset,
        },
        Inst::Call { callee, args, ty } => Inst::Call {
            callee: m(callee),
            args: args.into_iter().map(m).collect(),
            ty,
        },
        Inst::Phi { ty, incoming } => Inst::Phi {
            ty,
            incoming: incoming
                .into_iter()
                .map(|(x, b)| (m(x), block_map[b.0 as usize]))
                .collect(),
        },
        Inst::ExtractValue { agg, index, ty } => Inst::ExtractValue {
            agg: m(agg),
            index,
            ty,
        },
        Inst::InsertValue {
            agg,
            value,
            index,
            ty,
        } => Inst::InsertValue {
            agg: m(agg),
            value: m(value),
            index,
            ty,
        },
        Inst::Freeze { ty, value } => Inst::Freeze {
            ty,
            value: m(value),
        },
        Inst::Marker { text } => Inst::Marker { text },
    }
}

fn translate_term(term: &Terminator, map: &[Option<ValueId>], block_map: &[BlockId]) -> Terminator {
    let m = |x: ValueId| mapped(map, x);
    let b = |x: BlockId| block_map[x.0 as usize];
    match term {
        Terminator::Ret(v) => Terminator::Ret(v.map(m)),
        Terminator::Br(t) => Terminator::Br(b(*t)),
        Terminator::CondBr { cond, yes, no } => Terminator::CondBr {
            cond: m(*cond),
            yes: b(*yes),
            no: b(*no),
        },
        Terminator::Switch {
            value,
            default,
            cases,
        } => Terminator::Switch {
            value: m(*value),
            default: b(*default),
            cases: cases.iter().map(|(k, t)| (*k, b(*t))).collect(),
        },
        Terminator::Unreachable => Terminator::Unreachable,
    }
}

// ---------------------------------------------------------------------------
// Calls

/// Call `callee` with the convention's registers up to `arity`.
fn call_unknown(
    f: &mut Function,
    at: &mut At,
    regs: &[Reg],
    env: ValueId,
    slots: &[Option<ValueId>],
    callee: ValueId,
    conv: &[u32],
    arity: u32,
) -> ValueId {
    let mut args = Vec::new();
    for k in 0..arity.min(conv.len() as u32) {
        args.push(load_as(
            f,
            at,
            regs,
            env,
            slots,
            conv[k as usize],
            &Type::i64(),
        ));
    }
    emit(
        f,
        at,
        Inst::Call {
            callee,
            args,
            ty: Type::i64(),
        },
    )
}

/// After a call returns to `ret`: the stack pointer as the return leaves it.
fn resumed(
    f: &mut Function,
    at: &mut At,
    regs: &[Reg],
    conv: &Conv,
    env: ValueId,
    slots: &[Option<ValueId>],
    sp_at_call: ValueId,
    ret: u64,
) {
    let sp_ty = reg_ty(regs, conv.sp);
    let pop = const_int(f, conv.pop, regs[conv.sp as usize].bits);
    let v = emit(
        f,
        at,
        Inst::Bin {
            op: BinOp::Add,
            ty: sp_ty.clone(),
            lhs: sp_at_call,
            rhs: pop,
        },
    );
    write_slot(f, at, regs, env, slots, conv.sp, v);
    if ret != 0 {
        let pc = const_u64(f, ret, regs[conv.pc as usize].bits);
        write_slot(f, at, regs, env, slots, conv.pc, pc);
    }
}

/// The store of return address `ret` before a call: back from the call to
/// the previous call, also through blocks with a single predecessor.
fn find_pushed_return(
    f: &Function,
    block: BlockId,
    upto: usize,
    ret: u64,
) -> Option<(BlockId, usize)> {
    let mut cur = block;
    let mut i = upto;
    for _ in 0..=f.blocks.len() {
        while i > 0 {
            i -= 1;
            let v = f.blocks[cur.0 as usize].insts[i];
            match f.inst(v) {
                Some(Inst::Call { .. }) => return None,
                Some(Inst::Store { value, .. }) => {
                    if let Value::Const(Constant::Int { bits, width }) = f.value(*value)
                        && *width == 64
                        && *bits == ret as u128
                    {
                        return Some((cur, i));
                    }
                }
                _ => {}
            }
        }
        let preds = &f.blocks[cur.0 as usize].preds;
        if preds.len() != 1 {
            return None;
        }
        cur = *preds.iter().next().expect("one predecessor");
        i = f.blocks[cur.0 as usize].insts.len();
    }
    None
}

/// Call a lifted function through its exact native type and keep its
/// results in the register slots.
fn invoke(
    f: &mut Function,
    at: &mut At,
    regs: &[Reg],
    conv: &Conv,
    env: ValueId,
    slots: &[Option<ValueId>],
    t: &Sig,
    sp: ValueId,
    ret_addr: u64,
) {
    let mut args = Vec::new();
    for i in 0..t.args.len() {
        let want = param_ty(t, regs, i);
        args.push(read_slot(f, at, regs, env, slots, t.args[i], &want));
    }
    if !t.stack_args.is_empty() {
        let sp64 = convert(f, at, &reg_ty(regs, conv.sp), &Type::i64(), sp);
        for i in 0..t.stack_args.len() {
            let off = const_int(f, t.stack_args[i], 64);
            let a = emit(
                f,
                at,
                Inst::Bin {
                    op: BinOp::Add,
                    ty: Type::i64(),
                    lhs: sp64,
                    rhs: off,
                },
            );
            let p = emit(
                f,
                at,
                Inst::Cast {
                    from: Type::i64(),
                    to: Type::Ptr,
                    value: a,
                },
            );
            let v = load_at(f, at, &Type::i64(), p);
            let want = param_ty(t, regs, t.args.len() + i);
            args.push(trunc_if_narrow(f, at, v, &want));
        }
    }
    let rty = ret_ty(t, regs);
    let callee = f.add_const(Constant::Function(t.function.clone()));
    let r = emit(
        f,
        at,
        Inst::Call {
            callee,
            args,
            ty: rty,
        },
    );
    for i in 0..t.rets.len() {
        let v = if t.rets.len() == 1 {
            r
        } else {
            let field = slot_ty(regs, t.rets[i], sig_ret_kind(t, i));
            emit(
                f,
                at,
                Inst::ExtractValue {
                    agg: r,
                    index: i as u32,
                    ty: field,
                },
            )
        };
        write_slot(f, at, regs, env, slots, t.rets[i], v);
    }
    resumed(f, at, regs, conv, env, slots, sp, ret_addr);
}

#[cfg(test)]
mod tests {
    use super::*;
    use biara_ir::{Builder, IntPred};

    fn regs() -> Vec<Reg> {
        vec![
            // 0: rax, 1: rdx, 2: rsp, 3: rip
            Reg { home: 0, bits: 64, name: "rax".into(), thread: false },
            Reg { home: 8, bits: 64, name: "rdx".into(), thread: false },
            Reg { home: 16, bits: 64, name: "rsp".into(), thread: false },
            Reg { home: 24, bits: 64, name: "rip".into(), thread: false },
        ]
    }

    fn conv() -> Conv {
        Conv {
            sp: 2,
            pc: 3,
            pop: 8,
            args: vec![0, 1],
            ret: 0,
            sys_args: vec![0],
            sys_ret: 0,
            env_size: 32,
            margin: 64,
            dispatch: true,
        }
    }

    /// A lifted `void f(ptr %env)`: `rax = rax + 1; return`.
    fn lifted(name: &str, entry: u64) -> Function {
        let ty = FuncType {
            ret: Type::Void,
            params: vec![Type::Ptr],
            varargs: false,
        };
        let mut b = Builder::new(name, ty);
        let env = b.add_param(Type::Ptr, "env");
        let blk = b.add_block("entry");
        let p = b.emit(blk, Inst::Gep { base: env, offset: 0 });
        let rax = b.load(blk, &Type::i64(), p);
        let one = b.const_typed(1, 64);
        let sum = b.bin(blk, BinOp::Add, &Type::i64(), rax, one);
        let q = b.emit(blk, Inst::Gep { base: env, offset: 8 });
        let cond = b.icmp(blk, IntPred::Eq, &Type::i64(), rax, sum);
        b.store(blk, &Type::i64(), sum, q);
        b.cond_br(blk, cond, blk, blk);
        // A second block so preds are non-trivial.
        let out = b.add_block("out");
        let q2 = b.emit(out, Inst::Gep { base: env, offset: 0 });
        b.store(out, &Type::i64(), sum, q2);
        b.ret(out, None);
        b.finish()
    }

    #[test]
    fn forms_a_lifted_function() {
        let mut m = Module::new("t", "", "");
        m.functions.push(lifted("f", 0x1000));
        let sigs = vec![Sig {
            function: "f".into(),
            args: vec![0],
            stack_args: vec![8],
            rets: vec![0],
            below: 16,
            above: 16,
            entry: 0x1000,
            balanced: true,
            preserved: vec![1],
            arg_kinds: None,
            ret_kinds: None,
            ret_class: RetClass::Direct,
            sret_param: None,
        }];
        form(&mut m, &conv(), &regs(), &sigs, &[], &[], &[]).expect("abi form");
        let f = m.function("f").expect("rewritten function");
        assert_eq!(f.ty.params.len(), 2, "one register and one stack formal");
        assert_eq!(f.ty.ret, Type::i64());
        assert_eq!(f.params.len(), 2);
        assert!(m.function("biara_call_at").is_some(), "dispatcher built");
        assert!(m.globals.iter().any(|g| g.name == CPU_GLOBAL));
        biara_ir::verify(f).expect("verifies");
        biara_ir::verify(m.function("biara_call_at").unwrap()).expect("dispatcher verifies");
    }

    #[test]
    fn rejects_a_non_env_parameter() {
        let mut m = Module::new("t", "", "");
        let ty = FuncType {
            ret: Type::Void,
            params: vec![Type::Ptr, Type::i64()],
            varargs: false,
        };
        let mut b = Builder::new("f", ty);
        b.add_param(Type::Ptr, "env");
        b.add_param(Type::i64(), "x");
        let blk = b.add_block("entry");
        b.ret(blk, None);
        m.functions.push(b.finish());
        let sigs = vec![Sig { function: "f".into(), entry: 1, ..Default::default() }];
        let err = form(&mut m, &conv(), &regs(), &sigs, &[], &[], &[]).unwrap_err();
        assert!(err.contains("only the env pointer"), "{err}");
    }
}
