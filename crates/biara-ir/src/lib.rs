//! A small SSA IR: what the lifter builds and the decompiler reads.
//!
//! The design follows LLVM's shape because the analyses here were ported from
//! it (values with uses, basic blocks with predecessors, phis at joins), but
//! nothing beyond what biara emits and reads is modelled: no debug info, no
//! exception handling, no arbitrary constant folds into operands.
//!
//! Values live in a per-function arena and are referred to by [`ValueId`].

use std::collections::BTreeSet;

mod verify;
mod print;
pub mod analysis;
pub mod pass;
pub mod build;

pub use analysis::{ConstantRange, Dominators, recompute_preds};
pub use build::{ty_bytes, Builder};
pub use analysis::recompute_preds as update_preds;
pub use print::{print_function, print_module};
pub use verify::{verify, VerifyError};

/// An integer comparison predicate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IntPred {
    Eq,
    Ne,
    Slt,
    Sle,
    Sgt,
    Sge,
    Ult,
    Ule,
    Ugt,
    Uge,
}

impl IntPred {
    /// True when the predicate is true for every pair of equal operands.
    pub fn reflexive(self) -> bool {
        matches!(self, Self::Eq | Self::Sle | Self::Sge | Self::Ule | Self::Uge)
    }

    /// The predicate with its operands swapped (`a < b` becomes `b > a`).
    pub fn swapped(self) -> Self {
        match self {
            Self::Eq => Self::Eq,
            Self::Ne => Self::Ne,
            Self::Slt => Self::Sgt,
            Self::Sle => Self::Sge,
            Self::Sgt => Self::Slt,
            Self::Sge => Self::Sle,
            Self::Ult => Self::Ugt,
            Self::Ule => Self::Uge,
            Self::Ugt => Self::Ult,
            Self::Uge => Self::Ule,
        }
    }

    /// The predicate that is false exactly when `self` is true.
    pub fn negated(self) -> Self {
        match self {
            Self::Eq => Self::Ne,
            Self::Ne => Self::Eq,
            Self::Slt => Self::Sge,
            Self::Sle => Self::Sgt,
            Self::Sgt => Self::Sle,
            Self::Sge => Self::Slt,
            Self::Ult => Self::Uge,
            Self::Ule => Self::Ugt,
            Self::Ugt => Self::Ule,
            Self::Uge => Self::Ult,
        }
    }

    /// `(lo, hi)` bounds a value satisfying this predicate against `rhs`
    /// implies, as a signed predicate. Used by range analysis.
    pub fn is_signed(self) -> bool {
        matches!(self, Self::Slt | Self::Sle | Self::Sgt | Self::Sge)
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Eq => "eq",
            Self::Ne => "ne",
            Self::Slt => "slt",
            Self::Sle => "sle",
            Self::Sgt => "sgt",
            Self::Sge => "sge",
            Self::Ult => "ult",
            Self::Ule => "ule",
            Self::Ugt => "ugt",
            Self::Uge => "uge",
        }
    }
}

/// Binary operators.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    /// Unsigned division; non-zero divisor is the caller's responsibility.
    Udiv,
    /// Signed division; non-zero divisor is the caller's responsibility.
    Sdiv,
    Urem,
    Srem,
    Shl,
    Lshr,
    Ashr,
    And,
    Or,
    Xor,
}

impl BinOp {
    pub fn name(self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Sub => "sub",
            Self::Mul => "mul",
            Self::Udiv => "udiv",
            Self::Sdiv => "sdiv",
            Self::Urem => "urem",
            Self::Srem => "srem",
            Self::Shl => "shl",
            Self::Lshr => "lshr",
            Self::Ashr => "ashr",
            Self::And => "and",
            Self::Or => "or",
            Self::Xor => "xor",
        }
    }

    /// Shift and division take their right operand as a count/divisor, which
    /// is never folded into a constant operand by the printer.
    pub fn is_commutative(self) -> bool {
        matches!(
            self,
            Self::Add | Self::Mul | Self::And | Self::Or | Self::Xor
        )
    }

    pub fn is_shift(self) -> bool {
        matches!(self, Self::Shl | Self::Lshr | Self::Ashr)
    }
}

/// A unary operator: the bit-manipulation primitives TCG lowers to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UnOp {
    Not,
    Neg,
    /// Count leading/trailing zeros, and set bits.
    Ctlz,
    Cttz,
    Ctpop,
    Bswap,
}

impl UnOp {
    pub fn name(self) -> &'static str {
        match self {
            Self::Not => "not",
            Self::Neg => "neg",
            Self::Ctlz => "ctlz",
            Self::Cttz => "cttz",
            Self::Ctpop => "ctpop",
            Self::Bswap => "bswap",
        }
    }
}

/// A type. Integers carry their bit width; `Ptr` is opaque (the address
/// space is uniform: guest memory is host memory).
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Type {
    Void,
    /// `i1`, `i8`, `i16`, `i32`, `i64`, `i128`.
    Int(u32),
    Ptr,
    Array(Box<Type>, u64),
    Struct(Vec<Type>),
    Vector(Box<Type>, u32),
    Func(Box<FuncType>),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FuncType {
    pub ret: Type,
    pub params: Vec<Type>,
    pub varargs: bool,
}

impl Type {
    pub fn i1() -> Self {
        Self::Int(1)
    }
    pub fn i8() -> Self {
        Self::Int(8)
    }
    pub fn i16() -> Self {
        Self::Int(16)
    }
    pub fn i32() -> Self {
        Self::Int(32)
    }
    pub fn i64() -> Self {
        Self::Int(64)
    }
    pub fn i128() -> Self {
        Self::Int(128)
    }

    pub fn is_int(&self) -> bool {
        matches!(self, Self::Int(_))
    }

    pub fn int_bits(&self) -> Option<u32> {
        match self {
            Self::Int(b) => Some(*b),
            _ => None,
        }
    }

    pub fn is_vector(&self) -> bool {
        matches!(self, Self::Vector(..))
    }

    /// Number of bits one lane occupies for a packed vector, else `None`.
    pub fn lane_bits(&self) -> Option<u32> {
        match self {
            Self::Vector(elem, _) => elem.int_bits(),
            _ => None,
        }
    }
}

/// A constant value.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Constant {
    /// An integer of a given width, as a two's-complement bit pattern in the
    /// low bits of `bits` (`bits` holds at most 128 significant bits).
    Int { bits: u128, width: u32 },
    /// A zero-initialised aggregate or pointer (`null`).
    Zero,
    /// `undef`: any value of the type will do.
    Undef,
    /// The address of a function, by name.
    Function(String),
    /// The address of a global, by name.
    GlobalPtr(String),
    /// An array/struct literal.
    Aggregate(Vec<Constant>),
}

impl Constant {
    pub fn int(value: impl Into<i128>, width: u32) -> Self {
        let v: i128 = value.into();
        let mask = if width >= 128 {
            u128::MAX
        } else {
            (1u128 << width) - 1
        };
        Self::Int {
            bits: (v as u128) & mask,
            width,
        }
    }

    pub fn zero_int(width: u32) -> Self {
        Self::Int { bits: 0, width }
    }

    pub fn all_ones(width: u32) -> Self {
        let mask = if width >= 128 {
            u128::MAX
        } else {
            (1u128 << width) - 1
        };
        Self::Int { bits: mask, width }
    }

    /// The constant as a signed integer of its width.
    pub fn as_i128(&self) -> Option<i128> {
        match self {
            Self::Int { bits, width } => {
                let w = (*width).min(128);
                if w == 0 {
                    return Some(0);
                }
                let sign = 1u128 << (w - 1);
                let raw = if w >= 128 { *bits } else { bits & ((1u128 << w) - 1) };
                Some(if raw & sign != 0 {
                    (raw | !((1u128 << w) - 1)) as i128
                } else {
                    raw as i128
                })
            }
            _ => None,
        }
    }

    /// The constant as an unsigned integer, zero-extended.
    pub fn as_u128(&self) -> Option<u128> {
        match self {
            Self::Int { bits, width } => {
                let w = (*width).min(128);
                Some(if w >= 128 { *bits } else { bits & ((1u128 << w) - 1) })
            }
            _ => None,
        }
    }

    pub fn is_zero(&self) -> bool {
        matches!(self, Self::Zero) || self.as_u128() == Some(0)
    }
}

/// Index of a value in its function's arena.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ValueId(pub u32);

/// A value: an instruction result, a parameter, or a constant.
#[derive(Clone)]
#[doc(hidden)]
pub enum Value {
    Param {
        ty: Type,
        index: u32,
        name: String,
    },
    Const(Constant),
    Inst(Inst),
}

impl std::fmt::Debug for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Param { name, .. } => write!(f, "param {name}"),
            Self::Const(c) => write!(f, "const {c:?}"),
            Self::Inst(i) => write!(f, "inst {i:?}"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FuncId(pub u32);

/// One instruction. The result type is implied by the operator.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Inst {
    Bin { op: BinOp, ty: Type, lhs: ValueId, rhs: ValueId },
    Un { op: UnOp, ty: Type, value: ValueId },
    /// `rotl`/`rotr` by `amount`, masked to the type width.
    Rot { left: bool, ty: Type, value: ValueId, amount: ValueId },
    /// A sequentially-consistent fence (TCG's `mb`).
    Fence,
    Icmp { pred: IntPred, ty: Type, lhs: ValueId, rhs: ValueId },
    Select { ty: Type, cond: ValueId, yes: ValueId, no: ValueId },
    Trunc { from: Type, to: Type, value: ValueId },
    Zext { from: Type, to: Type, value: ValueId },
    Sext { from: Type, to: Type, value: ValueId },
    /// `inttoptr`/`ptrtoint`/`bitcast`: representation-preserving.
    Cast { from: Type, to: Type, value: ValueId },
    Alloca { ty: Type },
    /// Alignment in bytes; guest memory is `1` (QEMU imposes none).
    Load { ty: Type, ptr: ValueId, align: u32 },
    Store { ty: Type, value: ValueId, ptr: ValueId, align: u32 },
    /// `getelementptr inbounds i8, ptr %base, i64 <offset>` — the only form
    /// biara emits, with a constant offset.
    Gep { base: ValueId, offset: i64 },
    Call { callee: ValueId, args: Vec<ValueId>, ty: Type },
    Phi { ty: Type, incoming: Vec<(ValueId, BlockId)> },
    /// `extractvalue` of a single element (the overflow intrinsics).
    ExtractValue { agg: ValueId, index: u32, ty: Type },
    /// `insertvalue`.
    InsertValue { agg: ValueId, value: ValueId, index: u32, ty: Type },
    Freeze { ty: Type, value: ValueId },
    /// Metadata carrier for `!biara.pc`: no runtime effect.
    Marker { text: String },
}

/// A basic block's terminator.
#[derive(Clone, Debug)]
pub enum Terminator {
    Ret(Option<ValueId>),
    Br(BlockId),
    CondBr { cond: ValueId, yes: BlockId, no: BlockId },
    Switch {
        value: ValueId,
        default: BlockId,
        cases: Vec<(u128, BlockId)>,
    },
    Unreachable,
}

#[derive(Clone)]
pub struct Block {
    pub name: String,
    pub insts: Vec<ValueId>,
    /// The block's terminator: `None` until the builder sets one.
    ///
    /// `None` and [`Terminator::Unreachable`] are different facts. `None`
    /// means the block is still open — a builder that stops mid-block leaves
    /// it that way, and no control-flow fact may be inferred. `Unreachable`
    /// is a deliberate statement that control does not continue. Treating
    /// the two alike made a partly built block look finished.
    pub term: Option<Terminator>,
    pub preds: BTreeSet<BlockId>,
}

#[derive(Clone)]
pub struct Function {
    pub name: String,
    pub ty: FuncType,
    pub params: Vec<ValueId>,
    pub blocks: Vec<Block>,
    pub entry: Option<BlockId>,
    pub attrs: FuncAttrs,
    pub values: Vec<Value>,
    /// Block index each value is defined in (`None` for params/constants).
    pub defined_in: Vec<Option<BlockId>>,
    /// Guest address each value was lowered from, when one applied. This is
    /// the `!biara.pc` metadata: decompiler provenance and audit locations.
    pub pc: Vec<Option<u64>>,
}

/// Attributes the analyses and the C emitter consult.
#[derive(Clone, Debug, Default)]
pub struct FuncAttrs {
    /// `noreturn`: calls never fall through.
    pub noreturn: bool,
    /// Parameter indices that are `noalias`.
    pub noalias_params: BTreeSet<u32>,
    /// Parameter indices carrying `captures(none)`.
    pub nocapture_params: BTreeSet<u32>,
    /// `memory(...)` read/write bits, as emitted for markers/helpers.
    pub memory: Option<MemoryEffect>,
    /// `biara.c_type` on a parameter, or `biara.c_return`.
    pub c_type: Option<String>,
    pub c_return: Option<String>,
    /// `!biara.entry`: the guest address this function lifts.
    pub entry_addr: Option<u64>,
    /// `!biara.callee`: the guest address a declaration stands for.
    pub callee_addr: Option<u64>,
    /// `!biara.present`: presentation ordinal.
    pub present: Option<u64>,
    /// The `!biara.os` tag.
    pub os: Option<u32>,
    /// Per-formal ABI kinds (`biara.abi.args`), in formal order.
    ///
    /// The ABI form of a function states how each parameter travels, which
    /// its IR type alone cannot always express (a declared narrow lane and
    /// a recovered raw word can both be an integer of the same width). The
    /// kind vector preserves that evidence for the signature summary.
    pub abi_arg_kinds: Vec<u32>,
    /// Per-result ABI kinds (`biara.abi.rets`).
    pub abi_ret_kinds: Vec<u32>,
    /// The return shape (`biara.abi.ret_class`): `0` direct, `1` pair,
    /// `2` memory (hidden `sret`).
    pub abi_ret_class: Option<u32>,
}

/// Which memory a call may touch, mirroring `memory(...)`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemoryEffect {
    pub argmem_read: bool,
    pub argmem_write: bool,
    pub inaccessible_read: bool,
    pub inaccessible_write: bool,
}

#[derive(Clone)]
pub struct Global {
    pub name: String,
    pub ty: Type,
    pub init: Option<Constant>,
    pub internal: bool,
}

#[derive(Clone)]
pub struct Module {
    pub name: String,
    pub functions: Vec<Function>,
    pub globals: Vec<Global>,
    pub decls: Vec<Declaration>,
    /// Triple/layout text kept for `--emit llvm` compatibility.
    pub triple: String,
    pub data_layout: String,
}

/// A function declared but not defined here (an import, a helper, a driver).
#[derive(Clone, Debug)]
pub struct Declaration {
    pub name: String,
    pub ty: FuncType,
    pub attrs: FuncAttrs,
}

impl Module {
    pub fn new(name: impl Into<String>, triple: impl Into<String>, layout: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            functions: Vec::new(),
            globals: Vec::new(),
            decls: Vec::new(),
            triple: triple.into(),
            data_layout: layout.into(),
        }
    }

    pub fn decl(&self, name: &str) -> Option<&Declaration> {
        self.decls.iter().find(|d| d.name == name)
    }

    pub fn function(&self, name: &str) -> Option<&Function> {
        self.functions.iter().find(|f| f.name == name)
    }

    pub fn function_mut(&mut self, name: &str) -> Option<&mut Function> {
        self.functions.iter_mut().find(|f| f.name == name)
    }
}

impl Function {
    pub fn new(name: impl Into<String>, ty: FuncType) -> Self {
        Self {
            name: name.into(),
            ty,
            params: Vec::new(),
            blocks: Vec::new(),
            entry: None,
            attrs: FuncAttrs::default(),
            values: Vec::new(),
            defined_in: Vec::new(),
            pc: Vec::new(),
        }
    }

    /// Create the formal parameters this function's type declares, if they
    /// do not exist yet. Idempotent: a function built through the module
    /// API always has its formals, which is what makes parameter lookup
    /// answer for a function whose body has not been written.
    pub fn ensure_params(&mut self) {
        if !self.params.is_empty() {
            return;
        }
        for (i, ty) in self.ty.params.clone().into_iter().enumerate() {
            let index = self.params.len() as u32;
            self.values.push(Value::Param {
                ty,
                index,
                name: format!("arg{i}"),
            });
            self.defined_in.push(None);
            self.pc.push(None);
            let id = ValueId((self.values.len() - 1) as u32);
            self.params.push(id);
        }
    }

    pub fn value(&self, id: ValueId) -> &Value {
        &self.values[id.0 as usize]
    }

    pub fn value_mut(&mut self, id: ValueId) -> &mut Value {
        &mut self.values[id.0 as usize]
    }

    /// The instruction `id` names, if it is one of this function's.
    ///
    /// Value ids are per function, so an id from another function is simply
    /// not this function's instruction: callers that scan every function for
    /// a value must not have to pre-check the bounds themselves.
    pub fn inst(&self, id: ValueId) -> Option<&Inst> {
        match self.values.get(id.0 as usize)? {
            Value::Inst(i) => Some(i),
            _ => None,
        }
    }

    pub fn constant(&self, id: ValueId) -> Option<&Constant> {
        match self.values.get(id.0 as usize)? {
            Value::Const(c) => Some(c),
            _ => None,
        }
    }

    /// Type of a value.
    pub fn ty(&self, id: ValueId) -> Type {
        match self.value(id) {
            Value::Param { ty, .. } => ty.clone(),
            Value::Const(c) => const_ty(c, self),
            Value::Inst(i) => inst_ty(i, self),
        }
    }

    pub fn block(&self, id: BlockId) -> &Block {
        &self.blocks[id.0 as usize]
    }

    pub fn block_mut(&mut self, id: BlockId) -> &mut Block {
        &mut self.blocks[id.0 as usize]
    }

    /// The block's terminator, once one is set.
    pub fn terminator(&self, b: BlockId) -> Option<&Terminator> {
        self.blocks[b.0 as usize].term.as_ref()
    }

    /// The block's terminator, or [`Terminator::Unreachable`] when none is
    /// set — for the passes that only need "does control continue".
    pub fn terminator_or_unreachable(&self, b: BlockId) -> Terminator {
        self.blocks[b.0 as usize]
            .term
            .clone()
            .unwrap_or(Terminator::Unreachable)
    }

    /// Successors of `b`; an open block has none.
    pub fn successors(&self, b: BlockId) -> Vec<BlockId> {
        self.blocks[b.0 as usize]
            .term
            .as_ref()
            .map(successor_blocks)
            .unwrap_or_default()
    }

    /// Add a constant to the arena.
    /// The guest address a value was lowered from, if any.
    pub fn pc_of(&self, v: ValueId) -> Option<u64> {
        self.pc.get(v.0 as usize).copied().flatten()
    }

    /// Add a constant, reusing an identical one already in this function.
    ///
    /// Constants are values, so two literals with the same bit pattern and
    /// width must be the same value: a caller that compares handles (as the
    /// `inkwell` API does) sees identity, not two equal copies.
    pub fn add_const(&mut self, c: Constant) -> ValueId {
        if let Some(i) = self
            .values
            .iter()
            .position(|v| matches!(v, Value::Const(x) if *x == c))
        {
            return ValueId(i as u32);
        }
        self.append_const(c)
    }

    /// Index constants before a batch that only appends constants to the arena.
    pub fn constant_index(&self) -> std::collections::HashMap<Constant, ValueId> {
        let mut index = std::collections::HashMap::new();
        for (i, value) in self.values.iter().enumerate() {
            if let Value::Const(constant) = value {
                index.entry(constant.clone()).or_insert(ValueId(i as u32));
            }
        }
        index
    }

    /// Add a constant using the index from this function. Every constant
    /// inserted during the batch must update the same index.
    pub fn add_indexed_const(
        &mut self,
        index: &mut std::collections::HashMap<Constant, ValueId>,
        constant: Constant,
    ) -> ValueId {
        *index.entry(constant.clone()).or_insert_with(|| self.append_const(constant))
    }

    fn append_const(&mut self, c: Constant) -> ValueId {
        self.values.push(Value::Const(c));
        self.defined_in.push(None);
        self.pc.push(None);
        ValueId((self.values.len() - 1) as u32)
    }

    /// Every instruction that mentions `v` as an operand, with its block.
    pub fn users(&self, v: ValueId) -> Vec<(ValueId, BlockId)> {
        let mut out = Vec::new();
        for (bi, b) in self.blocks.iter().enumerate() {
            for &i in &b.insts {
                if let Value::Inst(inst) = self.value(i)
                    && operands(inst).any(|o| o == v)
                {
                    out.push((i, BlockId(bi as u32)));
                }
            }
            for u in b.term.as_ref().map(term_operands).unwrap_or_default() {
                if u == v {
                    // The terminator is not a value; report it via the block.
                    out.push((ValueId(u32::MAX), BlockId(bi as u32)));
                }
            }
        }
        out
    }

    /// Operands of every instruction and terminator: the raw use list.
    pub fn uses(&self, v: ValueId) -> usize {
        let insts = self
            .blocks
            .iter()
            .flat_map(|b| b.insts.iter())
            .filter(|&&i| {
                self.inst(i).is_some_and(|inst| operands(inst).any(|o| o == v))
            })
            .count();
        let terms = self
            .blocks
            .iter()
            .flat_map(|b| b.term.as_ref().map(term_operands).unwrap_or_default())
            .filter(|&o| o == v)
            .count();
        insts + terms
    }

    pub fn uses_once(&self, v: ValueId) -> bool {
        self.uses(v) == 1
    }

    pub fn uses_none(&self, v: ValueId) -> bool {
        self.uses(v) == 0
    }
}

/// The declared type of a constant.
pub fn const_ty(c: &Constant, f: &Function) -> Type {
    match c {
        Constant::Int { width, .. } => Type::Int(*width),
        Constant::Zero => Type::Ptr,
        Constant::Undef => Type::i1(),
        Constant::Function(name) => {
            let _ = name;
            Type::Ptr
        }
        Constant::GlobalPtr(_) => Type::Ptr,
        Constant::Aggregate(items) => {
            Type::Struct(items.iter().map(|c| const_ty(c, f)).collect())
        }
    }
}

/// The result type of an instruction.
pub fn inst_ty(i: &Inst, f: &Function) -> Type {
    match i {
        Inst::Bin { ty, .. }
        | Inst::Un { ty, .. }
        | Inst::Rot { ty, .. }
        | Inst::Icmp { ty, .. }
        | Inst::Select { ty, .. }
        | Inst::Load { ty, .. }
        | Inst::Phi { ty, .. }
        | Inst::ExtractValue { ty, .. }
        | Inst::InsertValue { ty, .. }
        | Inst::Freeze { ty, .. } => {
            if matches!(i, Inst::Icmp { .. }) {
                Type::i1()
            } else {
                ty.clone()
            }
        }
        Inst::Trunc { to, .. } | Inst::Zext { to, .. } | Inst::Sext { to, .. } | Inst::Cast { to, .. } => {
            to.clone()
        }
        Inst::Alloca { .. } => Type::Ptr,
        Inst::Store { .. } | Inst::Marker { .. } | Inst::Fence => Type::Void,
        Inst::Gep { .. } => Type::Ptr,
        Inst::Call { ty, .. } => ty.clone(),
    }
}

/// Iterate the value operands of an instruction.
pub fn operands(i: &Inst) -> impl Iterator<Item = ValueId> + '_ {
    let v: Vec<ValueId> = match i {
        Inst::Bin { lhs, rhs, .. } | Inst::Icmp { lhs, rhs, .. } => vec![*lhs, *rhs],
        Inst::Un { value, .. } => vec![*value],
        Inst::Rot { value, amount, .. } => vec![*value, *amount],
        Inst::Fence => Vec::new(),
        Inst::Select { cond, yes, no, .. } => vec![*cond, *yes, *no],
        Inst::Trunc { value, .. }
        | Inst::Zext { value, .. }
        | Inst::Sext { value, .. }
        | Inst::Cast { value, .. }
        | Inst::Freeze { value, .. } => vec![*value],
        Inst::Alloca { .. } | Inst::Marker { .. } => Vec::new(),
        Inst::Load { ptr, .. } => vec![*ptr],
        Inst::Store { value, ptr, .. } => vec![*value, *ptr],
        Inst::Gep { base, .. } => vec![*base],
        Inst::Call { callee, args, .. } => {
            let mut v = vec![*callee];
            v.extend(args.iter().copied());
            v
        }
        Inst::Phi { incoming, .. } => incoming.iter().map(|(v, _)| *v).collect(),
        Inst::ExtractValue { agg, .. } => vec![*agg],
        Inst::InsertValue { agg, value, .. } => vec![*agg, *value],
    };
    v.into_iter()
}

pub fn operands_mut(i: &mut Inst) -> Vec<&mut ValueId> {
    match i {
        Inst::Bin { lhs, rhs, .. } | Inst::Icmp { lhs, rhs, .. } => vec![lhs, rhs],
        Inst::Un { value, .. } => vec![value],
        Inst::Rot { value, amount, .. } => vec![value, amount],
        Inst::Fence => Vec::new(),
        Inst::Select { cond, yes, no, .. } => vec![cond, yes, no],
        Inst::Trunc { value, .. }
        | Inst::Zext { value, .. }
        | Inst::Sext { value, .. }
        | Inst::Cast { value, .. }
        | Inst::Freeze { value, .. } => vec![value],
        Inst::Alloca { .. } | Inst::Marker { .. } => Vec::new(),
        Inst::Load { ptr, .. } => vec![ptr],
        Inst::Store { value, ptr, .. } => vec![value, ptr],
        Inst::Gep { base, .. } => vec![base],
        Inst::Call { callee, args, .. } => {
            let mut v: Vec<&mut ValueId> = vec![callee];
            v.extend(args.iter_mut());
            v
        }
        Inst::Phi { incoming, .. } => incoming.iter_mut().map(|(v, _)| v).collect(),
        Inst::ExtractValue { agg, .. } => vec![agg],
        Inst::InsertValue { agg, value, .. } => vec![agg, value],
    }
}

pub fn term_operands(t: &Terminator) -> Vec<ValueId> {
    match t {
        Terminator::Ret(v) => v.iter().copied().collect(),
        Terminator::Br(_) | Terminator::Unreachable => Vec::new(),
        Terminator::CondBr { cond, .. } => vec![*cond],
        Terminator::Switch { value, .. } => vec![*value],
    }
}

pub fn successor_blocks(t: &Terminator) -> Vec<BlockId> {
    match t {
        Terminator::Ret(_) | Terminator::Unreachable => Vec::new(),
        Terminator::Br(b) => vec![*b],
        Terminator::CondBr { yes, no, .. } => vec![*yes, *no],
        Terminator::Switch { default, cases, .. } => {
            let mut v = vec![*default];
            v.extend(cases.iter().map(|(_, b)| *b));
            v
        }
    }
}
