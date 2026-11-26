//! Value handles.

use crate::types::{BasicMetadataTypeEnum, BasicTypeEnum};
use crate::Context;
use std::cell::RefCell;
use std::marker::PhantomData;

/// Which value a call returned.
pub enum ValueKind<'ctx> {
    Basic(BasicValueEnum<'ctx>),
    Instruction(InstructionValue<'ctx>),
}

impl<'ctx> ValueKind<'ctx> {
    /// The basic value, when the call returned one.
    pub fn basic(self) -> Option<BasicValueEnum<'ctx>> {
        match self {
            Self::Basic(v) => Some(v),
            Self::Instruction(_) => None,
        }
    }

    pub fn instruction(self) -> Option<InstructionValue<'ctx>> {
        match self {
            Self::Instruction(v) => Some(v),
            Self::Basic(_) => None,
        }
    }
}

/// Anything usable as a value.
pub trait BasicValue<'ctx>: Clone {
    fn as_basic_value_enum(&self) -> BasicValueEnum<'ctx>;
}

/// The object-safe subset `build_return(Option<&dyn …>)` needs.
pub trait DynBasicValue {
    fn as_basic_value_enum_dyn(&self) -> BasicValueEnum<'_>;
}

impl<'ctx> DynBasicValue for BasicValueEnum<'ctx> {
    fn as_basic_value_enum_dyn(&self) -> BasicValueEnum<'_> {
        self.clone()
    }
}

impl<'ctx> DynBasicValue for IntValue<'ctx> {
    fn as_basic_value_enum_dyn(&self) -> BasicValueEnum<'_> {
        BasicValueEnum::IntValue(*self)
    }
}

impl<'ctx> DynBasicValue for PointerValue<'ctx> {
    fn as_basic_value_enum_dyn(&self) -> BasicValueEnum<'_> {
        BasicValueEnum::PointerValue(*self)
    }
}

/// A typed value.
#[derive(Clone, Copy)]
pub enum BasicValueEnum<'ctx> {
    IntValue(IntValue<'ctx>),
    PointerValue(PointerValue<'ctx>),
    StructValue(StructValue<'ctx>),
}

impl<'ctx> BasicValueEnum<'ctx> {
    pub fn owner(&self) -> Option<usize> {
        match self {
            Self::IntValue(v) => v.owner(),
            Self::PointerValue(v) => v.owner(),
            Self::StructValue(_) => None,
        }
    }

    pub fn into_int_value(self) -> IntValue<'ctx> {
        match self {
            Self::IntValue(v) => v,
            _ => panic!("expected an integer value"),
        }
    }

    pub fn into_pointer_value(self) -> PointerValue<'ctx> {
        match self {
            Self::PointerValue(v) => v,
            _ => panic!("expected a pointer value"),
        }
    }

    pub fn into_struct_value(self) -> StructValue<'ctx> {
        match self {
            Self::StructValue(v) => v,
            _ => panic!("expected a struct value"),
        }
    }

    pub fn get_type(&self) -> BasicTypeEnum<'ctx> {
        match self {
            Self::IntValue(v) => BasicTypeEnum::IntType(v.get_type()),
            Self::PointerValue(v) => BasicTypeEnum::PointerType(v.get_type()),
            Self::StructValue(v) => BasicTypeEnum::StructType(v.get_type()),
        }
    }

    pub fn is_int_value(&self) -> bool {
        matches!(self, Self::IntValue(_))
    }

    pub fn is_pointer_value(&self) -> bool {
        matches!(self, Self::PointerValue(_))
    }

    pub fn is_struct_value(&self) -> bool {
        matches!(self, Self::StructValue(_))
    }

    pub fn set_name(&self, _name: &str) -> Result<(), String> {
        Ok(())
    }

    pub fn as_instruction_value(&self) -> Option<InstructionValue<'ctx>> {
        match self {
            Self::IntValue(v) => v.as_instruction_value(),
            Self::PointerValue(v) => Some(v.as_instruction_value()),
            Self::StructValue(v) => Some(InstructionValue {
                ctx: v.ctx,
                id: v.id,
                term: None,
                func: None,
                _m: PhantomData,
            }),
        }
    }

    pub fn const_zero(ty: &BasicTypeEnum<'ctx>) -> Self {
        match ty {
            BasicTypeEnum::IntType(t) => Self::IntValue(t.const_zero()),
            BasicTypeEnum::PointerType(t) => {
                Self::PointerValue(PointerValue::null(t.ctx))
            }
            BasicTypeEnum::StructType(t) => Self::StructValue(StructValue {
                ctx: t.ctx,
                ty: t.ir,
                id: biara_ir::ValueId(u32::MAX),
            }),
            BasicTypeEnum::VectorType(t) => Self::IntValue(IntValue::all_ones(t.ctx, 0)),
            BasicTypeEnum::ArrayType(t) => Self::StructValue(StructValue {
                ctx: t.ctx,
                ty: t.ir,
                id: biara_ir::ValueId(u32::MAX),
            }),
            BasicTypeEnum::FloatType(t) => Self::IntValue(IntValue::constant(t.ctx, 0, t.bits)),
        }
    }
}

impl<'ctx> std::fmt::Debug for BasicValueEnum<'ctx> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IntValue(v) => write!(f, "i{} v{}", v.bits, v.value_id().map(|i| i.0).unwrap_or(u32::MAX)),
            Self::PointerValue(v) => write!(f, "ptr v{}", v.id.0),
            Self::StructValue(v) => write!(f, "struct v{}", v.id.0),
        }
    }
}

impl<'ctx> PartialEq for BasicValueEnum<'ctx> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::IntValue(a), Self::IntValue(b)) => a.equal(b),
            (Self::PointerValue(a), Self::PointerValue(b)) => a.id == b.id,
            (Self::StructValue(a), Self::StructValue(b)) => a.id == b.id,
            _ => false,
        }
    }
}
impl<'ctx> Eq for BasicValueEnum<'ctx> {}

impl<'ctx> PartialEq<PointerValue<'ctx>> for BasicValueEnum<'ctx> {
    fn eq(&self, other: &PointerValue<'ctx>) -> bool {
        matches!(self, Self::PointerValue(p) if p.id == other.id)
    }
}

impl<'ctx> PartialEq<IntValue<'ctx>> for BasicValueEnum<'ctx> {
    fn eq(&self, other: &IntValue<'ctx>) -> bool {
        matches!(self, Self::IntValue(i) if i.equal(other))
    }
}

impl<'ctx> BasicValue<'ctx> for BasicValueEnum<'ctx> {
    fn as_basic_value_enum(&self) -> BasicValueEnum<'ctx> {
        *self
    }
}

/// Where an integer handle's value comes from.
#[derive(Clone, Copy)]
pub(crate) enum IntSource {
    /// A literal, interned into the target function when first used.
    Const(u128),
    /// A value already in a function arena.
    Value(biara_ir::ValueId),
}

/// An integer value.
#[derive(Clone, Copy)]
pub struct IntValue<'ctx> {
    pub(crate) ctx: &'ctx Context,
    pub(crate) src: IntSource,
    pub(crate) bits: u32,
    /// Owning function, when known. Value ids repeat across functions, so
    /// without it a lookup can name another function's value.
    pub(crate) func: Option<usize>,
    _m: PhantomData<&'ctx ()>,
}

/// `raw` sign-extended from `bits` to 64.
fn extend_sign(raw: u128, bits: u32) -> i64 {
    let w = bits.min(128);
    let masked = if w >= 128 { raw } else { raw & ((1u128 << w) - 1) };
    let signed = if w > 0 && w < 128 && masked & (1u128 << (w - 1)) != 0 {
        (masked | !((1u128 << w) - 1)) as i128
    } else {
        masked as i128
    };
    signed as i64
}

impl<'ctx> IntValue<'ctx> {
    pub(crate) fn new(ctx: &'ctx Context, id: biara_ir::ValueId, bits: u32) -> Self {
        Self {
            ctx,
            src: IntSource::Value(id),
            bits,
            func: None,
            _m: PhantomData,
        }
    }

    pub(crate) fn with_owner(self, func: Option<usize>) -> Self {
        Self { func, ..self }
    }

    pub fn owner(&self) -> Option<usize> {
        self.func
    }

    pub(crate) fn constant(ctx: &'ctx Context, v: i128, bits: u32) -> Self {
        let mask = if bits >= 128 {
            u128::MAX
        } else {
            (1u128 << bits) - 1
        };
        Self {
            ctx,
            src: IntSource::Const((v as u128) & mask),
            bits,
            func: None,
            _m: PhantomData,
        }
    }

    pub(crate) fn all_ones(ctx: &'ctx Context, bits: u32) -> Self {
        Self {
            ctx,
            src: IntSource::Const(if bits >= 128 {
                u128::MAX
            } else {
                (1u128 << bits) - 1
            }),
            bits,
            func: None,
            _m: PhantomData,
        }
    }

    /// A constant's bit pattern. Panics on a non-constant, as LLVM's cast did.
    pub fn bits_value(&self) -> u128 {
        match self.src {
            IntSource::Const(v) => v,
            IntSource::Value(_) => panic!("bits_value on a non-constant"),
        }
    }

    pub fn is_const(&self) -> bool {
        // A literal is a constant; so is an arena value holding one.
        // `operand` rebuilds handles as `IntSource::Value`, so testing the
        // source alone reported every arena constant — an offset like the
        // `16` in `sub %rsp-8, 16` — as a non-constant, and each caller that
        // gates on `is_const` then dropped it.
        matches!(self.src, IntSource::Const(_)) || self.get_zero_extended_constant().is_some()
    }

    pub(crate) fn equal(&self, other: &Self) -> bool {
        match (self.src, other.src) {
            (IntSource::Const(a), IntSource::Const(b)) => a == b && self.bits == other.bits,
            (IntSource::Value(a), IntSource::Value(b)) => a == b,
            // A literal and the value that holds it are the same number: an
            // operand read back from a store is the literal, while a handle
            // the builder interned is the value.
            (IntSource::Const(a), IntSource::Value(b))
            | (IntSource::Value(b), IntSource::Const(a)) => {
                self.bits == other.bits
                    && self
                        .ctx
                        .with(self.ctx.module_id(), |m| {
                            m.module
                                .functions
                                .iter()
                                .find_map(|f| {
                                    // Value ids are per function: only an arena
                                    // that holds this id can describe it.
                                    let entry = f.values.get(b.0 as usize)?;
                                    match entry {
                                        biara_ir::Value::Const(biara_ir::Constant::Int {
                                            bits,
                                            width,
                                        }) => Some(*bits == a && *width == self.bits),
                                        _ => None,
                                    }
                                })
                        })
                        .unwrap_or(false)
            }
        }
    }

    /// The widening constant, when this is one.
    pub fn get_zero_extended_constant(&self) -> Option<u64> {
        match self.src {
            IntSource::Const(v) => Some(v as u64),
            // A constant reached through the arena is still a constant; only
            // an instruction has no fixed value.
            // Only the owning function can say what this id is: value ids
            // repeat across functions, so scanning for one whose arena is
            // merely long enough reads another function's constant, which
            // then corrupts whatever offset it feeds.
            IntSource::Value(id) => self.ctx.with(self.ctx.module_id(), |m| {
                let f = m.module.functions.get(self.func?)?;
                if (id.0 as usize) >= f.values.len() {
                    return None;
                }
                match f.constant(id) {
                    Some(biara_ir::Constant::Int { bits, .. }) => Some(*bits as u64),
                    _ => None,
                }
            }),
        }
    }

    pub fn get_sign_extended_constant(&self) -> Option<i64> {
        match self.src {
            // A constant reached through the arena sign-extends by its own
            // width, exactly as a literal does.
            IntSource::Value(_) => {
                let raw = self.get_zero_extended_constant()? as u128;
                Some(extend_sign(raw, self.bits))
            }
            IntSource::Const(v) => Some(extend_sign(v, self.bits)),
        }
    }

    pub fn get_type(&self) -> crate::types::IntType<'ctx> {
        crate::types::IntType::new(self.ctx, self.bits)
    }

    pub fn get_bit_width(&self) -> u32 {
        self.bits
    }

    /// The arena id, materialising a constant if needed is the builder's job.
    pub(crate) fn value_id(&self) -> Option<biara_ir::ValueId> {
        match self.src {
            IntSource::Value(id) => Some(id),
            IntSource::Const(_) => None,
        }
    }

    pub fn as_instruction(&self) -> Option<InstructionValue<'ctx>> {
        self.as_instruction_value()
    }

    pub fn as_instruction_value(&self) -> Option<InstructionValue<'ctx>> {
        self.value_id().map(|id| InstructionValue {
            ctx: self.ctx,
            id,
            term: None,
            func: self.func,
            _m: PhantomData,
        })
    }
}

impl<'ctx> PartialEq for IntValue<'ctx> {
    fn eq(&self, other: &Self) -> bool {
        self.equal(other)
    }
}
impl<'ctx> Eq for IntValue<'ctx> {}

impl<'ctx> BasicValue<'ctx> for IntValue<'ctx> {
    fn as_basic_value_enum(&self) -> BasicValueEnum<'ctx> {
        BasicValueEnum::IntValue(*self)
    }
}

/// A pointer value.
#[derive(Clone, Copy)]
pub struct PointerValue<'ctx> {
    pub(crate) ctx: &'ctx Context,
    pub(crate) id: biara_ir::ValueId,
    /// Owning function, when known.
    pub(crate) func: Option<usize>,
    _m: PhantomData<&'ctx ()>,
}

impl<'ctx> PartialEq for PointerValue<'ctx> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
impl<'ctx> Eq for PointerValue<'ctx> {}
impl<'ctx> std::hash::Hash for PointerValue<'ctx> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl<'ctx> PointerValue<'ctx> {
    pub(crate) fn new(ctx: &'ctx Context, id: biara_ir::ValueId) -> Self {
        Self {
            ctx,
            id,
            func: None,
            _m: PhantomData,
        }
    }

    pub(crate) fn with_owner(self, func: Option<usize>) -> Self {
        Self { func, ..self }
    }

    pub fn owner(&self) -> Option<usize> {
        self.func
    }

    pub(crate) fn null(ctx: &'ctx Context) -> Self {
        Self {
            ctx,
            id: biara_ir::ValueId(u32::MAX),
            func: None,
            _m: PhantomData,
        }
    }

    pub fn get_type(&self) -> crate::types::PointerType<'ctx> {
        crate::types::PointerType::new(self.ctx)
    }

    pub fn as_instruction_value(&self) -> InstructionValue<'ctx> {
        // Dropping the owner here would let every later lookup fall back to
        // "any function holding this id", which names another function's
        // instruction.
        InstructionValue {
            ctx: self.ctx,
            id: self.id,
            term: None,
            func: self.func,
            _m: PhantomData,
        }
    }

    pub fn as_instruction(&self) -> Option<InstructionValue<'ctx>> {
        Some(self.as_instruction_value())
    }
}

impl<'ctx> BasicValue<'ctx> for PointerValue<'ctx> {
    fn as_basic_value_enum(&self) -> BasicValueEnum<'ctx> {
        BasicValueEnum::PointerValue(*self)
    }
}

/// A struct value: what `with.overflow` returns.
#[derive(Clone, Copy)]
pub struct StructValue<'ctx> {
    pub(crate) ctx: &'ctx Context,
    pub(crate) ty: &'ctx biara_ir::Type,
    pub(crate) id: biara_ir::ValueId,
}

impl<'ctx> StructValue<'ctx> {
    pub fn get_type(&self) -> crate::types::StructType<'ctx> {
        crate::types::StructType::new(self.ctx, self.ty.clone())
    }
}

impl<'ctx> BasicValue<'ctx> for StructValue<'ctx> {
    fn as_basic_value_enum(&self) -> BasicValueEnum<'ctx> {
        BasicValueEnum::StructValue(*self)
    }
}

/// A function handle.
#[derive(Clone, Copy)]
pub struct FunctionValue<'ctx> {
    pub(crate) ctx: &'ctx Context,
    pub(crate) module: usize,
    pub(crate) func: usize,
    _m: PhantomData<&'ctx ()>,
}

impl<'ctx> FunctionValue<'ctx> {
    pub(crate) fn new(ctx: &'ctx Context, func: usize) -> Self {
        Self {
            ctx,
            module: ctx.module_id(),
            func,
            _m: PhantomData,
        }
    }

    /// This function's IR attributes, cloned (see [`biara_ir::FuncAttrs`]).
    pub fn attrs(&self) -> biara_ir::FuncAttrs {
        self.ctx.with(self.module, |m| {
            m.module
                .functions
                .get(self.func)
                .map(|f| f.attrs.clone())
                .unwrap_or_default()
        })
    }


    /// A function in a specific module.
    pub(crate) fn in_module(ctx: &'ctx Context, module: usize, func: usize) -> Self {
        Self {
            ctx,
            module,
            func,
            _m: PhantomData,
        }
    }

    pub fn module_id(&self) -> usize {
        self.module
    }

    /// The owning context, so callers can reach the IR directly.
    pub fn ctx(&self) -> &'ctx Context {
        self.ctx
    }

    /// The function's index in its module.
    pub fn index(&self) -> usize {
        self.func
    }

    pub fn get_name(&self) -> crate::values::StringRef {
        StringRef(self.ctx.intern_str(
            self.ctx
                .with(self.module, |m| {
                    m.module.functions.get(self.func).map(|f| f.name.clone())
                })
                .unwrap_or_default(),
        ))
    }

    pub fn count_basic_blocks(&self) -> u32 {
        self.ctx.with(self.module, |m| {
            m.module
                .functions
                .get(self.func)
                .map_or(0, |f| f.blocks.len() as u32)
        })
    }

    pub fn count_params(&self) -> u32 {
        self.ctx.with(self.module, |m| {
            m.module
                .functions
                .get(self.func)
                .map_or(0, |f| f.params.len() as u32)
        })
    }

    pub fn get_nth_param(&self, i: u32) -> Option<BasicValueEnum<'ctx>> {
        self.ctx.with(self.module, |m| {
            let f = &m.module.functions[self.func];
            let id = *f.params.get(i as usize)?;
            // The parameter belongs to this function; carrying the owner keeps
            // later lookups from resolving the id elsewhere.
            let owner = Some(self.func);
            Some(match f.ty(id) {
                biara_ir::Type::Ptr => BasicValueEnum::PointerValue(
                    PointerValue::new(self.ctx, id).with_owner(owner),
                ),
                biara_ir::Type::Int(b) => BasicValueEnum::IntValue(
                    IntValue::new(self.ctx, id, b).with_owner(owner),
                ),
                _ => return None,
            })
        })
    }

    pub fn get_type(&self) -> crate::types::FunctionType<'ctx> {
        let ty = self
            .ctx
            .with(self.ctx.module_id(), |m| m.module.functions[self.func].ty.clone());
        let ir = match self
            .ctx
            .intern(biara_ir::Type::Func(Box::new(ty)))
        {
            biara_ir::Type::Func(f) => &**f,
            _ => unreachable!(),
        };
        crate::types::FunctionType {
            ir,
            ctx: self.ctx,
            _m: PhantomData,
        }
    }

    pub fn set_name(&self, _name: &str) -> Result<(), String> {
        Ok(())
    }

    pub fn add_attribute(&self, loc: crate::attributes::AttributeLoc, attr: crate::attributes::Attribute) {
        self.ctx.with_mut(self.module, |m| {
            let f = &mut m.module.functions[self.func];
            match loc {
                crate::attributes::AttributeLoc::Function => match attr.kind {
                    "noalias" => {}
                    "noreturn" => f.attrs.noreturn = true,
                    "memory" => {
                        let v = attr.value;
                        f.attrs.memory = Some(biara_ir::MemoryEffect {
                            argmem_read: v & 0b10 == 0,
                            argmem_write: v & 0b01 == 0,
                            inaccessible_read: v & 0b10_00 == 0,
                            inaccessible_write: v & 0b01_00 == 0,
                        });
                    }
                    _ => {}
                },
                crate::attributes::AttributeLoc::Param(i) => match attr.kind {
                    "noalias" => {
                        f.attrs.noalias_params.insert(i);
                    }
                    "captures" => {
                        f.attrs.nocapture_params.insert(i);
                    }
                    _ => {}
                },
            }
        });
    }

    pub fn get_first_basic_block(&self) -> Option<crate::basic_block::BasicBlock<'ctx>> {
        let b = self.ctx.with(self.module, |m| m.module.functions[self.func].entry)?;
        Some(crate::basic_block::BasicBlock::new(
            self.ctx,
            self.func,
            b,
        ))
    }

    pub fn get_basic_blocks(&self) -> Vec<crate::basic_block::BasicBlock<'ctx>> {
        let n = self.ctx.with(self.module, |m| m.module.functions[self.func].blocks.len());
        (0..n)
            .map(|i| {
                crate::basic_block::BasicBlock::new(
                    self.ctx,
                    self.func,
                    biara_ir::BlockId(i as u32),
                )
            })
            .collect()
    }

    pub fn set_linkage(&self, _linkage: crate::module::Linkage) {}

    pub fn get_first_param(&self) -> Option<BasicValueEnum<'ctx>> {
        self.get_nth_param(0)
    }

    pub fn get_params(&self) -> Vec<BasicValueEnum<'ctx>> {
        (0..self.count_params())
            .filter_map(|i| self.get_nth_param(i))
            .collect()
    }

    pub fn get_basic_block_iter(&self) -> std::vec::IntoIter<crate::basic_block::BasicBlock<'ctx>> {
        self.get_basic_blocks().into_iter()
    }

    pub fn get_instruction_iter(&self) -> std::vec::IntoIter<InstructionValue<'ctx>> {
        self.get_basic_blocks()
            .into_iter()
            .flat_map(|b| b.get_instructions())
            .collect::<Vec<_>>()
            .into_iter()
    }

    pub fn get_string_attribute(
        &self,
        loc: crate::attributes::AttributeLoc,
        kind: impl Into<String>,
    ) -> Option<crate::values::StringRef> {
        let kind = kind.into();
        self.ctx.with(self.module, |m| {
            let f = &m.module.functions[self.func];
            let v = match (&loc, kind.as_str()) {
                (crate::attributes::AttributeLoc::Function, "biara.c_return") => {
                    f.attrs.c_return.clone()
                }
                (crate::attributes::AttributeLoc::Param(_), "biara.c_type") => {
                    f.attrs.c_type.clone()
                }
                _ => None,
            }?;
            Some(crate::values::StringRef(self.ctx.intern_str(v)))
        })
    }

    /// The function as a "global" (metadata attachment target).
    pub fn as_global_value(&self) -> GlobalValue<'ctx> {
        GlobalValue {
            ctx: self.ctx,
            func: self.func,
        }
    }

    pub fn get_enum_attribute(
        &self,
        loc: crate::attributes::AttributeLoc,
        kind: impl Into<String>,
    ) -> Option<crate::attributes::Attribute> {
        let kind: String = kind.into();
        self.ctx.with(self.module, |m| {
            let f = &m.module.functions[self.func];
            let v = match (loc, kind.as_str()) {
                (crate::attributes::AttributeLoc::Function, "memory") => {
                    f.attrs.memory.map(|e| {
                        let mut v = 0u64;
                        if !e.argmem_read { v |= 0b10 }
                        if !e.argmem_write { v |= 0b01 }
                        if !e.inaccessible_read { v |= 0b10_00 }
                        if !e.inaccessible_write { v |= 0b01_00 }
                        v
                    })?
                }
                _ => return None,
            };
            Some(crate::attributes::Attribute {
                kind: Box::leak(String::from(kind).into_boxed_str()),
                value: v,
            })
        })
    }
}

/// A function viewed as a global (what `set_metadata` attaches to).
#[derive(Clone, Copy)]
pub struct GlobalValue<'ctx> {
    pub(crate) ctx: &'ctx Context,
    pub(crate) func: usize,
}

impl<'ctx> GlobalValue<'ctx> {
    pub fn set_name(&self, name: &str) -> Result<(), String> {
        self.ctx.with_mut(self.ctx.module_id(), |m| {
            if let Some(f) = m.module.functions.get_mut(self.func) {
                f.name = name.to_owned();
            }
            m.reindex();
        });
        Ok(())
    }

    pub fn set_metadata(&self, node: MetadataValue<'ctx>, kind: u32) -> Result<(), String> {
        let key = format!("{kind}");
        self.ctx.with_mut(self.ctx.module_id(), |m| {
            m.metadata.entry(key).or_default();
        });
        self.ctx.with_mut(self.ctx.module_id(), |m| {
            // The handle outlives a rewrite that can remove its function, so
            // a stale index attaches to nothing rather than panicking — the
            // same contract `set_name` already keeps.
            let Some(f) = m.module.functions.get_mut(self.func) else {
                return;
            };
            match kind_kind(self.ctx, kind).as_str() {
                "biara.entry" => f.attrs.entry_addr = Some(const_u64(&node.node())),
                "biara.callee" => f.attrs.callee_addr = Some(const_u64(&node.node())),
                "biara.present" => f.attrs.present = Some(const_u64(&node.node())),
                "biara.os" => f.attrs.os = Some(const_u64(&node.node()) as u32),
                "biara.c_type" => {
                    if let crate::MetadataNode::Str(s) = node.node() {
                        f.attrs.c_type = Some(s.clone());
                    }
                }
                _ => {}
            }
        });
        Ok(())
    }
}

/// The metadata kind carrying a value's guest address.
const PC_KIND: &str = "biara.pc";

fn kind_kind(ctx: &Context, kind: u32) -> String {
    let arena = ctx.kinds.borrow();
    arena
        .iter()
        .find(|(_, v)| **v == kind)
        .map(|(k, _)| k.clone())
        .unwrap_or_default()
}

fn const_u64(n: &crate::MetadataNode) -> u64 {
    match n {
        crate::MetadataNode::Int(v) => *v as u64,
        crate::MetadataNode::Node(items) => items.first().map(const_u64).unwrap_or(0),
        _ => 0,
    }
}

/// A string result, borrowed from the context as inkwell's was.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StringRef<'ctx>(pub &'ctx str);

impl<'ctx> StringRef<'ctx> {
    /// The value itself (inkwell's `get_string_value` returns a `StringRef`).
    pub fn get_string_value(&self) -> StringRef {
        self.clone()
    }

    pub fn to_str(&self) -> Result<&'ctx str, std::str::Utf8Error> {
        Ok(self.0)
    }

    pub fn to_string_lossy(&self) -> std::borrow::Cow<'ctx, str> {
        std::borrow::Cow::Borrowed(self.0)
    }

    pub fn to_bytes(&self) -> &'ctx [u8] {
        self.0.as_bytes()
    }
}

impl<'ctx> std::fmt::Display for StringRef<'ctx> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// An instruction handle.
#[derive(Clone, Copy)]
pub struct InstructionValue<'ctx> {
    pub(crate) ctx: &'ctx Context,
    pub(crate) id: biara_ir::ValueId,
    /// `Some` when this handle names a block's terminator.
    pub(crate) term: Option<biara_ir::BlockId>,
    /// The function whose arena holds `id`. Value ids repeat across
    /// functions, so resolving by id alone would name another function's
    /// instruction; `None` asks the accessors to find the owner by block.
    pub(crate) func: Option<usize>,
    pub(crate) _m: PhantomData<&'ctx ()>,
}

impl<'ctx> std::fmt::Debug for InstructionValue<'ctx> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "inst v{}", self.id.0)
    }
}

impl<'ctx> PartialEq for InstructionValue<'ctx> {
    fn eq(&self, other: &Self) -> bool {
        // Value ids repeat across functions, so two handles with the same id
        // are only the same instruction when they name the same function.
        // The comment above said this; the code did not, and maps keyed by
        // instruction (`entry_loads`, `entry_loads`'s users) then matched a
        // different function's value of the same id.
        self.id == other.id && self.term == other.term && self.func == other.func
    }
}
impl<'ctx> Eq for InstructionValue<'ctx> {}
impl<'ctx> std::hash::Hash for InstructionValue<'ctx> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
        self.term.hash(state);
        self.func.hash(state);
    }
}
impl<'ctx> PartialOrd for InstructionValue<'ctx> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl<'ctx> Ord for InstructionValue<'ctx> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.id, self.term).cmp(&(other.id, other.term))
    }
}

impl<'ctx> InstructionValue<'ctx> {
    /// The opcode, which the analyses match on.
    pub fn get_opcode(&self) -> InstructionOpcode {
        if let Some(b) = self.term {
            return self.ctx.with(self.module_id_public(), |m| {
                m.module
                    .functions
                    .iter()
                    .find_map(|f| f.blocks.get(b.0 as usize))
                    .map(|blk| match blk.term.as_ref().unwrap_or(&biara_ir::Terminator::Unreachable) {
                        biara_ir::Terminator::Ret(_) => InstructionOpcode::Ret,
                        biara_ir::Terminator::Br(_) => InstructionOpcode::Br,
                        biara_ir::Terminator::CondBr { .. } => InstructionOpcode::CondBr,
                        biara_ir::Terminator::Switch { .. } => InstructionOpcode::Switch,
                        biara_ir::Terminator::Unreachable => InstructionOpcode::Unreachable,
                    })
                    .unwrap_or(InstructionOpcode::Unreachable)
            });
        }
        self.ctx
            .with(self.ctx.module_id(), |m| {
                // Value ids repeat across functions, so the id alone does not
                // say which function owns this instruction. The one whose
                // block actually lists it does.
                // The stated owner answers only when its blocks really list
                // the id; otherwise the function that lists it does. Taking
                // `self.func` on trust reports another function's opcode,
                // because ids repeat across functions.
                let listed = |f: &biara_ir::Function| {
                    f.defined_in
                        .get(self.id.0 as usize)
                        .copied()
                        .flatten()
                        .is_some_and(|b| {
                            f.blocks
                                .get(b.0 as usize)
                                .is_some_and(|x| x.insts.contains(&self.id))
                        })
                };
                let owner = |m: &crate::ModuleData| -> Option<usize> {
                    let by_owner = self
                        .func
                        .and_then(|i| m.module.functions.get(i))
                        .filter(|f| listed(f))
                        .map(|_| self.func.unwrap_or(0));
                    by_owner.or_else(|| {
                        m.module.functions.iter().position(|f| listed(f))
                    })
                };
                let f = owner(m).and_then(|i| m.module.functions.get(i))?;
                f.inst(self.id)
                    .map(|i| InstructionOpcode::from_inst(i, f))
            })
            .unwrap_or(InstructionOpcode::Call)
    }

    /// Operand `i`, as a value.
    pub fn get_operand(&self, i: u32) -> Option<Operand<'ctx>> {
        if let Some(b) = self.term {
            let (owner, target) = self.ctx.with(self.ctx.module_id(), |m| {
                let (at, f) = match self.func.and_then(|i| m.module.functions.get(i)) {
                    Some(f) if f.blocks.get(b.0 as usize).is_some() => {
                        (self.func.unwrap_or(0), f)
                    }
                    _ => m
                        .module
                        .functions
                        .iter()
                        .enumerate()
                        .find(|(_, f)| f.blocks.get(b.0 as usize).is_some())?,
                };
                let target = f.successors(b)
                    .get(i as usize)
                    .copied()?;
                Some((at, target))
            })?;
            return Some(Operand::Block(crate::basic_block::BasicBlock::new(
                self.ctx, owner, target,
            )));
        }
        let id = self.ctx.with(self.ctx.module_id(), |m| {
            let candidates: Vec<&biara_ir::Function> = match self.func {
                Some(i) => m.module.functions.get(i).into_iter().collect(),
                None => m.module.functions.iter().collect(),
            };
            candidates.iter().find_map(|f| {
                let inst = f.inst(self.id)?;
                biara_ir::operands(inst).nth(i as usize)
            })
        });
        if let Some(id) = id {
            return Some(self.operand(id));
        }
        // A `gep`'s offset is an immediate in the IR; expose it as the second
        // operand's constant, which is how inkwell's shape spells it.
        if i == 1 {
            // Instruction ids are per-function, so the owner decides which
            // `gep` this is; another function's instruction with the same id
            // is a different offset.
            let off = match self.resolve_inst() {
                Some(biara_ir::Inst::Gep { offset, .. }) => offset,
                _ => return None,
            };
            return Some(Operand::Value(BasicValueEnum::IntValue(IntValue::constant(
                self.ctx,
                off as i128,
                64,
            ))));
        }
        None
    }

    fn operand(&self, id: biara_ir::ValueId) -> Operand<'ctx> {
        // Value ids repeat across functions: resolve in the owner, or in the
        // function whose block actually lists the value.
        // Value ids repeat across functions, so the owner decides the type:
        // the same id in another function may be a different value entirely.
        // Any value the function holds counts, not only block instructions:
        // a constant is a value with no block, and requiring one sent every
        // constant to the "arena long enough" fallback below, which answered
        // with an unrelated function's instruction of the same id.
        let listed = |f: &biara_ir::Function| match f.values.get(id.0 as usize) {
            Some(biara_ir::Value::Inst(_)) => f
                .defined_in
                .get(id.0 as usize)
                .copied()
                .flatten()
                .is_some_and(|b| {
                    f.blocks
                        .get(b.0 as usize)
                        .is_some_and(|b| b.insts.contains(&id))
                }),
            Some(_) => true,
            None => false,
        };
        // The stated owner decides when its blocks actually list the id; the
        // function that lists it answers otherwise. Falling back only on
        // "arena long enough" would read another function's value.
        let owner = self.ctx.with(self.ctx.module_id(), |m| {
            let by_owner = self
                .func
                .and_then(|i| m.module.functions.get(i))
                .filter(|f| listed(f))
                .map(|_| self.func.unwrap_or(0));
            by_owner.or_else(|| {
                m.module
                    .functions
                    .iter()
                    .position(|f| listed(f))
                    .or_else(|| {
                        m.module.functions.iter().position(|f| {
                            (id.0 as usize) < f.values.len()
                        })
                    })
            })
        });
        let ty = self.ctx.with(self.ctx.module_id(), |m| {
            owner
                .and_then(|i| m.module.functions.get(i))
                .map(|f| f.ty(id))
                .unwrap_or(biara_ir::Type::Int(64))
        });
        let mk = |v: BasicValueEnum<'ctx>| match v {
            BasicValueEnum::IntValue(i) => {
                Operand::Value(BasicValueEnum::IntValue(i.with_owner(owner)))
            }
            BasicValueEnum::PointerValue(p) => {
                Operand::Value(BasicValueEnum::PointerValue(p.with_owner(owner)))
            }
            v => Operand::Value(v),
        };
        match ty {
            biara_ir::Type::Ptr => mk(BasicValueEnum::PointerValue(PointerValue::new(
                self.ctx, id,
            ))),
            biara_ir::Type::Int(b) => {
                mk(BasicValueEnum::IntValue(IntValue::new(self.ctx, id, b)))
            }
            _ => mk(BasicValueEnum::IntValue(IntValue::new(self.ctx, id, 64))),
        }
    }

    pub fn get_num_operands(&self) -> u32 {
        if let Some(b) = self.term {
            // Block ids repeat across functions, so the successor count is
            // read from the function that owns this terminator.
            let counted = |f: &biara_ir::Function| {
                f.blocks
                    .get(b.0 as usize)?
                    .term
                    .as_ref()
                    .map(|t| biara_ir::successor_blocks(t).len() as u32)
            };
            return self.ctx.with(self.module_id_public(), |m| {
                self.func
                    .and_then(|i| m.module.functions.get(i))
                    .and_then(counted)
                    .or_else(|| m.module.functions.iter().find_map(counted))
                    .unwrap_or(0)
            });
        }
        // A `gep` in inkwell's shape is (base, offset): the offset is an
        // immediate here, so it is still named. An unowned handle falls back
        // to any function that lists the id, as it always did.
        let inst = self.resolve_inst().or_else(|| {
            self.ctx.with(self.ctx.module_id(), |m| {
                m.module
                    .functions
                    .iter()
                    .find_map(|f| f.inst(self.id).cloned())
            })
        });
        match inst {
            Some(biara_ir::Inst::Gep { .. }) => 2,
            Some(other) => biara_ir::operands(&other).count() as u32,
            None => 0,
        }
    }

    pub(crate) fn module_id_public(&self) -> usize {
        self.ctx.module_id()
    }

    /// The arena id this instruction names.
    pub fn ir_id(&self) -> biara_ir::ValueId {
        self.id
    }

    /// The type of this instruction's result.
    pub fn get_type(&self) -> crate::types::AnyTypeEnum<'ctx> {
        let ty = self.ctx.with(self.ctx.module_id(), |m| {
            m.module
                .functions
                .iter()
                .find_map(|f| {
                    ((self.id.0 as usize) < f.values.len()).then(|| f.ty(self.id))
                })
        });
        match ty {
            Some(t) => crate::types::AnyTypeEnum::from_ir(self.ctx, &t),
            None => crate::types::AnyTypeEnum::IntType(crate::types::IntType::new(self.ctx, 64)),
        }
    }

    /// For a `getelementptr`, the source element type (always `i8` here).
    pub fn get_gep_source_element_type(&self) -> Result<crate::types::BasicTypeEnum<'ctx>, String> {
        if self.get_opcode() != InstructionOpcode::GetElementPtr {
            return Err("not a getelementptr".into());
        }
        Ok(crate::types::BasicTypeEnum::IntType(
            crate::types::IntType::new(self.ctx, 8),
        ))
    }

    /// Every operand, in order.
    pub fn get_operands(&self) -> std::vec::IntoIter<Option<Operand<'ctx>>> {
        (0..self.get_num_operands())
            .map(|i| self.get_operand(i))
            .collect::<Vec<_>>()
            .into_iter()
    }

    /// The instruction before this one in its block.
    pub fn func_owner(&self) -> Option<usize> {
        self.func
    }

    /// The instruction this handle names. The owner decides which one it is:
    /// value ids repeat across functions, so an unqualified search can name
    /// an unrelated instruction with the same id.
    pub fn resolve_inst(&self) -> Option<biara_ir::Inst> {
        if self.term.is_some() {
            return None;
        }
        self.ctx.with(self.ctx.module_id(), |m| {
            // With a stated owner, only that function may answer, and only if
            // it really holds this instruction: an id can be inside the
            // arena while naming a constant, and then `inst` says so.
            let by_owner = self
                .func
                .and_then(|i| m.module.functions.get(i))
                .filter(|f| f.inst(self.id).is_some());
            match self.func {
                Some(_) => by_owner.and_then(|f| f.inst(self.id).cloned()),
                None => m
                    .module
                    .functions
                    .iter()
                    .find(|f| {
                        f.defined_in
                            .get(self.id.0 as usize)
                            .copied()
                            .flatten()
                            .is_some_and(|b| {
                                f.blocks
                                    .get(b.0 as usize)
                                    .is_some_and(|x| x.insts.contains(&self.id))
                            })
                    })
                    .and_then(|f| f.inst(self.id).cloned()),
            }
        })
    }

    pub fn get_previous_instruction(&self) -> Option<InstructionValue<'ctx>> {
        let ids = self.owning_block_insts()?;
        let at = ids.iter().position(|v| *v == self.id)?;
        at.checked_sub(1).map(|i| Self {
            ctx: self.ctx,
            func: self.func,
            id: ids[i],
            term: None,
            _m: PhantomData,
        })
    }

    /// The instruction after this one in its block.
    pub fn get_next_instruction(&self) -> Option<InstructionValue<'ctx>> {
        // The owning block is the one that lists this instruction: block ids
        // repeat across functions, so a global search can name the wrong one.
        let ids = self.owning_block_insts()?;
        let at = ids.iter().position(|v| *v == self.id)?;
        ids.get(at + 1).map(|id| Self {
            ctx: self.ctx,
            func: self.func,
            id: *id,
            term: None,
            _m: PhantomData,
        })
    }

    /// The instruction list of the block that actually holds this value.
    ///
    /// Block ids repeat across functions, so the block and its owner must be
    /// resolved together: looking a block id up in every function would find
    /// another function's block with the same number.
    fn owning_block_insts(&self) -> Option<Vec<biara_ir::ValueId>> {
        self.ctx.with(self.ctx.module_id(), |m| {
            let holds = |f: &biara_ir::Function| {
                f.blocks.iter().find_map(|blk| {
                    blk.insts
                        .contains(&self.id)
                        .then(|| blk.insts.clone())
                })
            };
            // Value ids are per-function, so another function listing the
            // same id is a different instruction. Only an instruction whose
            // owner is unknown searches the whole module.
            match self.func {
                Some(i) => m.module.functions.get(i).and_then(&holds),
                None => m.module.functions.iter().find_map(&holds),
            }
        })
    }

    fn block_insts(&self, b: biara_ir::BlockId) -> Option<Vec<biara_ir::ValueId>> {
        self.ctx.with(self.ctx.module_id(), |m| {
            m.module
                .functions
                .iter()
                .find_map(|f| f.blocks.get(b.0 as usize).map(|blk| blk.insts.clone()))
        })
    }

    /// Remove this instruction from its block.
    pub fn erase_from_basic_block(&self) {
        if let Some(b) = self.term {
            self.ctx.with_mut(self.ctx.module_id(), |m| {
                for f in &mut m.module.functions {
                    if let Some(blk) = f.blocks.get_mut(b.0 as usize) {
                        blk.term = Some(biara_ir::Terminator::Unreachable);
                        return;
                    }
                }
            });
            return;
        }
        self.ctx.with_mut(self.ctx.module_id(), |m| {
            for f in &mut m.module.functions {
                for blk in &mut f.blocks {
                    blk.insts.retain(|v| *v != self.id);
                }
            }
        });
    }

    /// Rewrite every use of this value to `other`.
    pub fn replace_all_uses_with(&self, other: &InstructionValue<'ctx>) {
        self.ctx.with_mut(self.ctx.module_id(), |m| {
            for f in &mut m.module.functions {
                biara_ir::pass::replace_all_uses(f, self.id, other.id);
            }
        });
    }

    /// Attach `node` under `kind`.
    ///
    /// The only per-instruction metadata the IR models is the guest address,
    /// kept as `Function::pc`; that is what `!biara.pc` carries.
    pub fn set_metadata(&self, node: MetadataValue<'ctx>, kind: u32) -> Result<(), String> {
        let is_pc = kind_kind(self.ctx, kind) == PC_KIND;
        if !is_pc {
            return Ok(());
        }
        let addr = const_u64(&node.node());
        let owner = self.owner();
        self.ctx.with_mut(self.ctx.module_id(), |m| {
            if let Some(f) = owner.and_then(|i| m.module.functions.get_mut(i)) {
                let at = self.id.0 as usize;
                if f.pc.len() <= at {
                    f.pc.resize(at + 1, None);
                    f.defined_in.resize(at + 1, None);
                }
                f.pc[at] = Some(addr);
            }
        });
        Ok(())
    }

    /// The node attached under `kind`, for the metadata the IR models.
    pub fn get_metadata(&self, kind: u32) -> Option<MetadataValue<'ctx>> {
        if kind_kind(self.ctx, kind) != PC_KIND {
            return None;
        }
        let owner = self.owner()?;
        let addr = self.ctx.with(self.ctx.module_id(), |m| {
            m.module
                .functions
                .get(owner)
                .and_then(|f| f.pc.get(self.id.0 as usize).copied().flatten())
        })?;
        // `!biara.pc` is `!{i64 <addr>}`, matching what the lifter attached.
        Some(MetadataValue::new(
            self.ctx,
            crate::MetadataNode::Node(vec![crate::MetadataNode::Int(addr as i128)]),
        ))
    }

    /// The function owning this instruction: its stated owner, else the one
    /// whose block lists the id.
    fn owner(&self) -> Option<usize> {
        self.ctx.with(self.ctx.module_id(), |m| {
            if let Some(i) = self.func
                && (self.id.0 as usize) < m.module.functions.get(i)?.values.len()
            {
                return Some(i);
            }
            m.module.functions.iter().position(|f| {
                f.defined_in
                    .get(self.id.0 as usize)
                    .copied()
                    .flatten()
                    .is_some_and(|b| {
                        f.blocks
                            .get(b.0 as usize)
                            .is_some_and(|b| b.insts.contains(&self.id))
                    })
            })
        })
    }

    pub fn get_volatile(&self) -> Result<bool, String> {
        Ok(false)
    }

    pub fn get_atomic_ordering(&self) -> Result<crate::AtomicOrdering, String> {
        Ok(crate::AtomicOrdering::NotAtomic)
    }

    pub fn set_alignment(&self, _align: u32) -> Result<(), String> {
        Ok(())
    }

    pub fn get_parent(&self) -> Option<crate::basic_block::BasicBlock<'ctx>> {
        // The block and its owner are one fact: block ids repeat across
        // functions, so resolving them separately can name another function's
        // block. Prefer the stated owner, then whichever function lists the
        // id in a block.
        let found = self.ctx.with(self.ctx.module_id(), |m| {
            // `defined_in` is only filled in where the builder recorded it, so
            // the block's own instruction list is authoritative.
            let listed = |f: &biara_ir::Function| {
                f.blocks.iter().position(|blk| blk.insts.contains(&self.id))
            };
            let by_owner = self
                .func
                .and_then(|i| m.module.functions.get(i))
                .and_then(listed)
                .map(|b| (self.func.unwrap_or_default(), b));
            // Value ids are per-function; a match in another function is a
            // different block.
            by_owner.or_else(|| match self.func {
                Some(_) => None,
                None => m
                    .module
                    .functions
                    .iter()
                    .enumerate()
                    .find_map(|(i, f)| listed(f).map(|b| (i, b))),
            })
        });
        let (owner, b) = found?;
        Some(crate::basic_block::BasicBlock::new(
            self.ctx,
            owner,
            biara_ir::BlockId(b as u32),
        ))
    }
}

/// An instruction operand.
#[derive(Clone)]
pub enum Operand<'ctx> {
    Value(BasicValueEnum<'ctx>),
    Metadata(MetadataValue<'ctx>),
    Block(crate::basic_block::BasicBlock<'ctx>),
}

impl<'ctx> Operand<'ctx> {
    pub fn value(&self) -> Option<BasicValueEnum<'ctx>> {
        match self {
            Self::Value(v) => Some(v.clone()),
            _ => None,
        }
    }

    pub fn block(&self) -> Option<crate::basic_block::BasicBlock<'ctx>> {
        match self {
            Self::Block(b) => Some(*b),
            _ => None,
        }
    }
}

/// The opcode inventory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InstructionOpcode {
    Add,
    Sub,
    Mul,
    And,
    Or,
    Xor,
    Shl,
    LShr,
    AShr,
    UDiv,
    SDiv,
    URem,
    SRem,
    ICmp,
    Select,
    Trunc,
    ZExt,
    SExt,
    IntToPtr,
    PtrToInt,
    BitCast,
    GetElementPtr,
    Load,
    Store,
    Alloca,
    Call,
    Phi,
    Fence,
    Ret,
    Br,
    CondBr,
    Switch,
    Unreachable,
    ExtractValue,
    InsertValue,
    Freeze,
    AtomicRMW,
    AtomicCmpXchg,
    AddrSpaceCast,
    ExtractElement,
    InsertElement,
    ShuffleVector,
    FPTrunc,
    FPExt,
    FAdd,
    FSub,
    FMul,
    FDiv,
    FNeg,
    FCmp,
}

impl InstructionOpcode {
    pub(crate) fn from_inst(i: &biara_ir::Inst, _f: &biara_ir::Function) -> Self {
        use biara_ir::Inst as I;
        match i {
            I::Bin { op, .. } => match op {
                biara_ir::BinOp::Add => Self::Add,
                biara_ir::BinOp::Sub => Self::Sub,
                biara_ir::BinOp::Mul => Self::Mul,
                biara_ir::BinOp::And => Self::And,
                biara_ir::BinOp::Or => Self::Or,
                biara_ir::BinOp::Xor => Self::Xor,
                biara_ir::BinOp::Shl => Self::Shl,
                biara_ir::BinOp::Lshr => Self::LShr,
                biara_ir::BinOp::Ashr => Self::AShr,
                biara_ir::BinOp::Udiv => Self::UDiv,
                biara_ir::BinOp::Sdiv => Self::SDiv,
                biara_ir::BinOp::Urem => Self::URem,
                biara_ir::BinOp::Srem => Self::SRem,
            },
            I::Un { .. } | I::Rot { .. } | I::Marker { .. } => Self::Call,
            I::Icmp { .. } => Self::ICmp,
            I::Select { .. } => Self::Select,
            I::Trunc { .. } => Self::Trunc,
            I::Zext { .. } => Self::ZExt,
            I::Sext { .. } => Self::SExt,
            I::Cast { from, .. } => match from {
                biara_ir::Type::Ptr => Self::PtrToInt,
                _ => Self::IntToPtr,
            },
            I::Alloca { .. } => Self::Alloca,
            I::Load { .. } => Self::Load,
            I::Store { .. } => Self::Store,
            I::Gep { .. } => Self::GetElementPtr,
            I::Call { .. } => Self::Call,
            I::Phi { .. } => Self::Phi,
            I::ExtractValue { .. } => Self::ExtractValue,
            I::InsertValue { .. } => Self::InsertValue,
            I::Freeze { .. } => Self::Freeze,
            I::Fence => Self::Fence,
        }
    }
}

/// A call site view.
pub struct CallSiteValue<'ctx> {
    pub(crate) inner: InstructionValue<'ctx>,
}

impl<'ctx> CallSiteValue<'ctx> {
    pub fn try_from(i: InstructionValue<'ctx>) -> Result<Self, String> {
        let is_call = i.ctx.with(i.ctx.module_id(), |m| {
            m.module
                .functions
                .iter()
                .enumerate()
                .filter(|(at, _)| i.func.is_none_or(|o| o == *at))
                .any(|(_, f)| {
                    matches!(
                        f.defined_in.get(i.id.0 as usize).copied().flatten()
                            .and_then(|b| f.blocks.get(b.0 as usize))
                            .filter(|b| b.insts.contains(&i.id))
                            .and_then(|_| f.inst(i.id)),
                        Some(biara_ir::Inst::Call { .. })
                    )
                })
        });
        if is_call {
            Ok(Self { inner: i })
        } else {
            Err("not a call".into())
        }
    }

    pub fn get_called_fn_value(&self) -> Option<FunctionValue<'ctx>> {
        // The call's own function decides which instruction this is, and only
        // that function can interpret its operands. An unowned handle (made
        // by a port that had no context) falls back to any function that
        // holds the id.
        let owner = self.inner.func;
        let name = self.inner.ctx.with(self.inner.ctx.module_id(), |m| {
            let f = match owner {
                Some(i) => m.module.functions.get(i)?,
                None => m
                    .module
                    .functions
                    .iter()
                    .find(|f| {
                        f.inst(self.inner.id)
                            .is_some_and(|i| matches!(i, biara_ir::Inst::Call { .. }))
                    })?,
            };
            let biara_ir::Inst::Call { callee, .. } = f.inst(self.inner.id)? else {
                return None;
            };
            let callee = *callee;
            if callee.0 as usize >= f.values.len() {
                return None;
            }
            let biara_ir::Value::Const(biara_ir::Constant::Function(n)) = f.value(callee) else {
                return None;
            };
            Some(n.clone())
        })?;
        let idx = self.inner.ctx.with(self.inner.ctx.module_id(), |m| m.index.get(&name).copied())?;
        Some(FunctionValue::new(self.inner.ctx, idx))
    }

    pub fn count_arguments(&self) -> u32 {
        self.get_num_arg_operands()
    }

    pub fn get_num_arg_operands(&self) -> u32 {
        self.inner.ctx.with(self.inner.ctx.module_id(), |m| {
            m.module
                .functions
                .iter()
                .find_map(|f| match f.inst(self.inner.id) {
                    Some(biara_ir::Inst::Call { args, .. }) => Some(args.len() as u32),
                    _ => None,
                })
                .unwrap_or(0)
        })
    }
}

/// A metadata handle. `Copy`, like inkwell's: the node lives in the context.
#[derive(Clone, Copy, Debug)]
pub struct MetadataValue<'ctx> {
    pub(crate) node: &'ctx crate::MetadataNode,
    pub(crate) ctx: &'ctx Context,
    pub(crate) _m: PhantomData<&'ctx ()>,
}

impl<'ctx> PartialEq for MetadataValue<'ctx> {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self.node, other.node) || self.node == other.node
    }
}
impl<'ctx> Eq for MetadataValue<'ctx> {}

impl<'ctx> MetadataValue<'ctx> {
    pub(crate) fn new(ctx: &'ctx Context, node: crate::MetadataNode) -> Self {
        Self {
            node: ctx.intern_node(node),
            ctx,
            _m: PhantomData,
        }
    }

    pub(crate) fn node(&self) -> &crate::MetadataNode {
        self.node
    }

    /// A metadata integer as a value (used for `!biara.pc`).
    pub fn into_int_value(&self) -> IntValue<'ctx> {
        let v = match self.node {
            crate::MetadataNode::Int(v) => *v,
            _ => 0,
        };
        IntValue::constant(self.ctx, v, 64)
    }

    pub fn get_node_values(&self) -> Option<Vec<MetadataValue<'ctx>>> {
        match self.node {
            crate::MetadataNode::Node(items) => Some(
                items
                    .iter()
                    .map(|n| MetadataValue::new(self.ctx_of(), n.clone()))
                    .collect(),
            ),
            _ => None,
        }
    }

    fn ctx_of(&self) -> &'ctx Context {
        // The node's owner is the context that interned it; a fresh metadata
        // value needs one for interning. Store it alongside.
        self.ctx
    }
}

/// A call argument.
#[derive(Clone)]
pub enum BasicMetadataValueEnum<'ctx> {
    BasicValue(BasicValueEnum<'ctx>),
    Metadata(MetadataValue<'ctx>),
}

impl<'ctx> BasicMetadataValueEnum<'ctx> {
    pub(crate) fn to_value(&self) -> BasicValueEnum<'ctx> {
        match self {
            Self::BasicValue(v) => v.clone(),
            Self::Metadata(_) => panic!("metadata passed as an argument"),
        }
    }
}

impl<'ctx> From<IntValue<'ctx>> for BasicMetadataValueEnum<'ctx> {
    fn from(v: IntValue<'ctx>) -> Self {
        Self::BasicValue(BasicValueEnum::IntValue(v))
    }
}

impl<'ctx> From<PointerValue<'ctx>> for BasicMetadataValueEnum<'ctx> {
    fn from(v: PointerValue<'ctx>) -> Self {
        Self::BasicValue(BasicValueEnum::PointerValue(v))
    }
}

impl<'ctx> From<BasicValueEnum<'ctx>> for BasicMetadataValueEnum<'ctx> {
    fn from(v: BasicValueEnum<'ctx>) -> Self {
        Self::BasicValue(v)
    }
}

impl<'ctx> From<StructValue<'ctx>> for BasicMetadataValueEnum<'ctx> {
    fn from(v: StructValue<'ctx>) -> Self {
        Self::BasicValue(BasicValueEnum::StructValue(v))
    }
}

impl<'ctx> From<IntValue<'ctx>> for MetadataValue<'ctx> {
    fn from(v: IntValue<'ctx>) -> Self {
        let node = match v.src {
            IntSource::Const(bits) => crate::MetadataNode::Int(bits as i128),
            IntSource::Value(id) => crate::MetadataNode::Int(id.0 as i128),
        };
        Self {
            node: v.ctx.intern_node(node),
            ctx: v.ctx,
            _m: PhantomData,
        }
    }
}

impl<'ctx> From<MetadataValue<'ctx>> for BasicMetadataValueEnum<'ctx> {
    fn from(v: MetadataValue<'ctx>) -> Self {
        Self::Metadata(v)
    }
}

impl<'ctx> From<IntValue<'ctx>> for BasicValueEnum<'ctx> {
    fn from(v: IntValue<'ctx>) -> Self {
        Self::IntValue(v)
    }
}

impl<'ctx> From<PointerValue<'ctx>> for BasicValueEnum<'ctx> {
    fn from(v: PointerValue<'ctx>) -> Self {
        Self::PointerValue(v)
    }
}

/// Duck-typed access to the arena view used by `AsValueRef` call sites.
pub trait AsValueRef {
    fn as_value_ref(&self) -> ValueRef;
}

/// An opaque value reference; the compat layer carries ids, not pointers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ValueRef(pub u32);

impl ValueRef {
    /// Reinterpret as a foreign (C++) handle. The payload is the arena id.
    pub fn cast<T>(self) -> T
    where
        T: FromValueRef,
    {
        T::from_value_ref(self)
    }

    pub fn as_ptr(self) -> *mut u8 {
        self.0 as usize as *mut u8
    }
}

/// Types that can be rebuilt from an arena id.
pub trait FromValueRef {
    fn from_value_ref(v: ValueRef) -> Self;
}

impl FromValueRef for usize {
    fn from_value_ref(v: ValueRef) -> Self {
        v.0 as usize
    }
}

impl FromValueRef for *mut std::ffi::c_void {
    fn from_value_ref(v: ValueRef) -> Self {
        v.0 as usize as *mut std::ffi::c_void
    }
}

impl<'ctx> AsValueRef for FunctionValue<'ctx> {
    fn as_value_ref(&self) -> ValueRef {
        ValueRef(self.func as u32)
    }
}

impl<'ctx> AsValueRef for IntValue<'ctx> {
    fn as_value_ref(&self) -> ValueRef {
        ValueRef(self.value_id().map(|i| i.0).unwrap_or(u32::MAX))
    }
}

impl<'ctx> AsValueRef for PointerValue<'ctx> {
    fn as_value_ref(&self) -> ValueRef {
        ValueRef(self.id.0)
    }
}

impl<'ctx> AsValueRef for InstructionValue<'ctx> {
    fn as_value_ref(&self) -> ValueRef {
        ValueRef(self.id.0)
    }
}

impl<'ctx> AsValueRef for BasicValueEnum<'ctx> {
    fn as_value_ref(&self) -> ValueRef {
        self.as_instruction_value()
            .map(|i| ValueRef(i.id.0))
            .unwrap_or(ValueRef(u32::MAX))
    }
}

impl<'ctx> AsValueRef for BasicBlockValueMarker<'ctx> {
    fn as_value_ref(&self) -> ValueRef {
        ValueRef(self.0)
    }
}

/// Marker so `BasicBlock` can implement `AsValueRef` without a cycle.
#[derive(Clone, Copy)]
pub struct BasicBlockValueMarker<'ctx>(pub(crate) u32, pub(crate) PhantomData<&'ctx ()>);

/// Shared position state so `Builder` methods can take `&self`, as inkwell's do.
pub(crate) struct Position {
    pub func: Option<usize>,
    pub block: RefCell<Option<biara_ir::BlockId>>,
    pub before: RefCell<Option<biara_ir::ValueId>>,
}

thread_local! {
    pub(crate) static _UNUSED: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// A phi node, seen through its incoming edges.
#[derive(Clone, Copy)]
pub struct PhiValue<'ctx> {
    pub(crate) inner: InstructionValue<'ctx>,
}

impl<'ctx> TryFrom<InstructionValue<'ctx>> for PhiValue<'ctx> {
    type Error = String;

    fn try_from(v: InstructionValue<'ctx>) -> Result<Self, String> {
        let is_phi = v.ctx.with(v.module_id_public(), |m| {
            m.module
                .functions
                .iter()
                .any(|f| matches!(f.inst(v.id), Some(biara_ir::Inst::Phi { .. })))
        });
        if is_phi {
            Ok(Self { inner: v })
        } else {
            Err("not a phi".into())
        }
    }
}

impl<'ctx> PhiValue<'ctx> {
    pub fn as_instruction(&self) -> InstructionValue<'ctx> {
        self.inner
    }

    pub fn count_incoming(&self) -> u32 {
        let ctx = self.inner.ctx;
        ctx.with(ctx.module_id(), |m| {
            m.module
                .functions
                .iter()
                .find_map(|f| match f.inst(self.inner.id) {
                    Some(biara_ir::Inst::Phi { incoming, .. }) => Some(incoming.len() as u32),
                    _ => None,
                })
                .unwrap_or(0)
        })
    }

    /// Incoming edge `i`, if it exists.
    pub fn get_incoming(
        &self,
        i: u32,
    ) -> Option<(BasicValueEnum<'ctx>, crate::basic_block::BasicBlock<'ctx>)> {
        self.incoming_all().into_iter().nth(i as usize)
    }

    pub fn incoming_all(
        &self,
    ) -> Vec<(BasicValueEnum<'ctx>, crate::basic_block::BasicBlock<'ctx>)> {
        let ctx = self.inner.ctx;
        let edges = ctx
            .with(ctx.module_id(), |m| {
                m.module
                    .functions
                    .iter()
                    .find_map(|f| match f.inst(self.inner.id) {
                        Some(biara_ir::Inst::Phi { incoming, .. }) => Some(incoming.clone()),
                        _ => None,
                    })
            })
            .unwrap_or_default();
        edges
            .into_iter()
            .map(|(v, b)| {
                // The phi's own function types its operands: value ids repeat
                // across functions, so any arena merely long enough can
                // answer with the wrong width or kind.
                let owner = self.inner.func;
                let ty = ctx.with(ctx.module_id(), |m| {
                    owner
                        .and_then(|i| m.module.functions.get(i))
                        .filter(|f| (v.0 as usize) < f.values.len())
                        .map(|f| f.ty(v))
                });
                let value = match ty {
                    Some(biara_ir::Type::Ptr) => BasicValueEnum::PointerValue(
                        PointerValue::new(ctx, v).with_owner(owner),
                    ),
                    Some(biara_ir::Type::Int(bits)) => BasicValueEnum::IntValue(
                        IntValue::new(ctx, v, bits).with_owner(owner),
                    ),
                    _ => BasicValueEnum::IntValue(IntValue::new(ctx, v, 64).with_owner(owner)),
                };
                let owner = ctx.with(ctx.module_id(), |m| {
                    m.module
                        .functions
                        .iter()
                        .position(|f| f.blocks.get(b.0 as usize).is_some())
                        .unwrap_or(0)
                });
                (
                    value,
                    crate::basic_block::BasicBlock::new(ctx, owner, b),
                )
            })
            .collect()
    }
}
