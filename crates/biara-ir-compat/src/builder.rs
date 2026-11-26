//! The instruction builder. Methods take `&self`, as inkwell's do, so
//! closures capturing the builder keep working.

use crate::basic_block::BasicBlock;
use crate::types::{BasicType, BasicTypeEnum, IntType};
use crate::values::{
    BasicMetadataValueEnum, BasicValue, BasicValueEnum, FunctionValue, InstructionValue, IntSource,
    IntValue, PointerValue, StructValue, ValueKind,
};
use crate::Context;
use std::cell::Cell;
use std::marker::PhantomData;

#[derive(Debug, thiserror::Error)]
pub enum BuilderError {
    #[error("builder used without a position")]
    NoPosition,
    #[error("value does not belong to this function")]
    ForeignValue,
}

/// The position a builder writes at.
#[derive(Clone, Copy)]
struct Pos {
    func: Option<usize>,
    block: biara_ir::BlockId,
    before: Option<biara_ir::ValueId>,
}

/// Emits instructions into a function's blocks.
pub struct Builder<'ctx> {
    ctx: &'ctx Context,
    pos: Cell<Option<Pos>>,
    _m: PhantomData<&'ctx ()>,
}

impl<'ctx> Builder<'ctx> {
    pub(crate) fn new(ctx: &'ctx Context) -> Self {
        Self {
            ctx,
            pos: Cell::new(None),
            _m: PhantomData,
        }
    }

    pub fn position_at_end(&self, block: BasicBlock<'ctx>) {
        // The block states its owner: block ids repeat across functions, so
        // scanning for "a function with this index" would pick the wrong one.
        self.pos.set(Some(Pos {
            func: Some(block.func),
            block: block.block,
            before: None,
        }));
    }

    pub fn position_before(&self, inst: &InstructionValue<'ctx>) {
        // Block ids repeat across functions, so the block and its owner are one
        // fact: resolve them together, in the instruction's own function.
        let found = self.ctx.with(self.ctx.module_id(), |m| {
            let listed = |f: &biara_ir::Function| {
                f.blocks
                    .iter()
                    .position(|b| b.insts.contains(&inst.id))
                    .map(|at| biara_ir::BlockId(at as u32))
            };
            if let Some(i) = inst.func
                && let Some(f) = m.module.functions.get(i)
                && let Some(b) = listed(f)
            {
                return Some((i, b));
            }
            if inst.term.is_some() {
                return None;
            }
            m.module
                .functions
                .iter()
                .enumerate()
                .find_map(|(i, f)| listed(f).map(|b| (i, b)))
        });
        let Some((func, block)) = found else { return };
        self.pos.set(Some(Pos {
            func: Some(func),
            block,
            before: Some(inst.id),
        }));
    }

    pub fn get_insert_block(&self) -> Option<BasicBlock<'ctx>> {
        self.pos.get().map(|p| {
            // The block states its owner; `Pos` may not, when it was set
            // from a block handle that only carried an id.
            let owner = p.func.unwrap_or_else(|| {
                self.ctx.with(self.ctx.module_id(), |m| {
                    m.module
                        .functions
                        .iter()
                        .position(|f| f.blocks.get(p.block.0 as usize).is_some())
                        .unwrap_or(0)
                })
            });
            BasicBlock::new(self.ctx, owner, p.block)
        })
    }

    /// The function being built.
    fn func(&self) -> Option<usize> {
        self.pos.get().and_then(|p| p.func)
    }

    /// Resolve an integer handle to an arena id, interning constants.
    fn id(&self, v: IntValue<'ctx>) -> biara_ir::ValueId {
        match v.src {
            IntSource::Value(id) => id,
            IntSource::Const(bits) => {
                let func = self.func();
                self.ctx.with_mut(self.ctx.module_id(), |m| {
                    m.build_constant(func.unwrap_or(0), biara_ir::Constant::Int {
                        bits,
                        width: v.bits,
                    })
                })
            }
        }
    }

    fn emit(&self, inst: biara_ir::Inst) -> biara_ir::ValueId {
        let Some(p) = self.pos.get() else {
            panic!("builder used without a position");
        };
        self.ctx.with_mut(self.ctx.module_id(), |m| {
            let f = match p.func {
                Some(i) => &mut m.module.functions[i],
                None => m
                    .module
                    .functions
                    .iter_mut()
                    .find(|f| f.blocks.get(p.block.0 as usize).is_some())
                    .expect("a block belongs to a function"),
            };
            let mut b = biara_ir::Builder::resume(std::mem::replace(
                f,
                biara_ir::Function::new("", biara_ir::FuncType {
                    ret: biara_ir::Type::Void,
                    params: Vec::new(),
                    varargs: false,
                }),
            ));
            let id = b.emit(p.block, inst);
            *f = b.func;
            if let Some(before) = p.before {
                let insts = &mut f.blocks[p.block.0 as usize].insts;
                if let Some(at) = insts.iter().position(|v| *v == before) {
                    insts.pop();
                    insts.insert(at, id);
                }
            }
            id
        })
    }

    /// The arena type of `id` in the function being built, if known.
    fn arena_int_bits(&self, id: biara_ir::ValueId) -> Option<u32> {
        self.ctx.with(self.ctx.module_id(), |m| {
            let func = match self.pos.get().and_then(|p| p.func) {
                Some(i) => m.module.functions.get(i)?,
                None => m.module.functions.first()?,
            };
            // Value ids are per function: an id outside this arena has no
            // width here, and indexing past the end would take the build down.
            let entry = func.values.get(id.0 as usize)?;
            match entry {
                biara_ir::Value::Const(biara_ir::Constant::Int { width, .. }) => Some(*width),
                biara_ir::Value::Inst(_) | biara_ir::Value::Param { .. } => func.ty(id).int_bits(),
                _ => None,
            }
        })
    }

    fn int_of(&self, id: biara_ir::ValueId) -> IntValue<'ctx> {
        // Value ids are per function: resolve in the function being built,
        // never by scanning (another function's id would answer wrongly).
        let bits = self.arena_int_bits(id).unwrap_or(64);
        IntValue::new(self.ctx, id, bits)
    }

    // ---- arithmetic -----------------------------------------------------

    fn bin(&self, op: biara_ir::BinOp, a: IntValue<'ctx>, b: IntValue<'ctx>) -> IntValue<'ctx> {
        // The operands' handle widths say how the caller typed the
        // expression; an arena lookup of a constant reports the constant's own
        // narrow width, which is not the operation's. Fall back to the arena
        // only when a handle's width was never set.
        let (abits, bbits) = (a.bits, b.bits);
        let (a, b) = (self.id(a), self.id(b));
        let bits = abits.max(bbits).max(1);
        let id = self.emit(biara_ir::Inst::Bin {
            op,
            ty: biara_ir::Type::Int(bits),
            lhs: a,
            rhs: b,
        });
        IntValue::new(self.ctx, id, bits)
    }

    pub fn build_int_add(&self, a: IntValue<'ctx>, b: IntValue<'ctx>, _n: &str) -> Result<IntValue<'ctx>, BuilderError> {
        Ok(self.bin(biara_ir::BinOp::Add, a, b))
    }
    pub fn build_int_sub(&self, a: IntValue<'ctx>, b: IntValue<'ctx>, _n: &str) -> Result<IntValue<'ctx>, BuilderError> {
        Ok(self.bin(biara_ir::BinOp::Sub, a, b))
    }
    pub fn build_int_mul(&self, a: IntValue<'ctx>, b: IntValue<'ctx>, _n: &str) -> Result<IntValue<'ctx>, BuilderError> {
        Ok(self.bin(biara_ir::BinOp::Mul, a, b))
    }
    pub fn build_and(&self, a: IntValue<'ctx>, b: IntValue<'ctx>, _n: &str) -> Result<IntValue<'ctx>, BuilderError> {
        Ok(self.bin(biara_ir::BinOp::And, a, b))
    }
    pub fn build_or(&self, a: IntValue<'ctx>, b: IntValue<'ctx>, _n: &str) -> Result<IntValue<'ctx>, BuilderError> {
        Ok(self.bin(biara_ir::BinOp::Or, a, b))
    }
    pub fn build_xor(&self, a: IntValue<'ctx>, b: IntValue<'ctx>, _n: &str) -> Result<IntValue<'ctx>, BuilderError> {
        Ok(self.bin(biara_ir::BinOp::Xor, a, b))
    }
    pub fn build_left_shift(&self, a: IntValue<'ctx>, b: IntValue<'ctx>, _n: &str) -> Result<IntValue<'ctx>, BuilderError> {
        Ok(self.bin(biara_ir::BinOp::Shl, a, b))
    }
    pub fn build_right_shift(&self, a: IntValue<'ctx>, b: IntValue<'ctx>, signed: bool, _n: &str) -> Result<IntValue<'ctx>, BuilderError> {
        Ok(self.bin(
            if signed { biara_ir::BinOp::Ashr } else { biara_ir::BinOp::Lshr },
            a,
            b,
        ))
    }
    pub fn build_int_signed_div(&self, a: IntValue<'ctx>, b: IntValue<'ctx>, _n: &str) -> Result<IntValue<'ctx>, BuilderError> {
        Ok(self.bin(biara_ir::BinOp::Sdiv, a, b))
    }
    pub fn build_int_unsigned_div(&self, a: IntValue<'ctx>, b: IntValue<'ctx>, _n: &str) -> Result<IntValue<'ctx>, BuilderError> {
        Ok(self.bin(biara_ir::BinOp::Udiv, a, b))
    }
    pub fn build_int_signed_rem(&self, a: IntValue<'ctx>, b: IntValue<'ctx>, _n: &str) -> Result<IntValue<'ctx>, BuilderError> {
        Ok(self.bin(biara_ir::BinOp::Srem, a, b))
    }
    pub fn build_int_unsigned_rem(&self, a: IntValue<'ctx>, b: IntValue<'ctx>, _n: &str) -> Result<IntValue<'ctx>, BuilderError> {
        Ok(self.bin(biara_ir::BinOp::Urem, a, b))
    }

    pub fn build_not(&self, a: IntValue<'ctx>, _n: &str) -> Result<IntValue<'ctx>, BuilderError> {
        let v = self.id(a);
        let id = self.emit(biara_ir::Inst::Un {
            op: biara_ir::UnOp::Not,
            ty: biara_ir::Type::Int(a.bits),
            value: v,
        });
        Ok(self.int_of(id))
    }

    pub fn build_int_neg(&self, a: IntValue<'ctx>, _n: &str) -> Result<IntValue<'ctx>, BuilderError> {
        let v = self.id(a);
        let id = self.emit(biara_ir::Inst::Un {
            op: biara_ir::UnOp::Neg,
            ty: biara_ir::Type::Int(a.bits),
            value: v,
        });
        Ok(self.int_of(id))
    }

    pub fn build_int_compare(&self, pred: crate::support::IntPredicate, a: IntValue<'ctx>, b: IntValue<'ctx>, _n: &str) -> Result<IntValue<'ctx>, BuilderError> {
        let handle_bits = a.bits;
        let (a, b) = (self.id(a), self.id(b));
        let bits = self
            .arena_int_bits(a)
            .or_else(|| self.arena_int_bits(b))
            .unwrap_or(handle_bits);
        let ty = biara_ir::Type::Int(bits);
        let id = self.emit(biara_ir::Inst::Icmp {
            pred: pred.to_ir(),
            ty,
            lhs: a,
            rhs: b,
        });
        Ok(IntValue::new(self.ctx, id, 1))
    }

    pub fn build_select(&self, cond: IntValue<'ctx>, yes: IntValue<'ctx>, no: IntValue<'ctx>, _n: &str) -> Result<BasicValueEnum<'ctx>, BuilderError> {
        let handle_bits = yes.bits;
        let (c, y, n) = (self.id(cond), self.id(yes), self.id(no));
        let bits = self
            .arena_int_bits(y)
            .or_else(|| self.arena_int_bits(n))
            .unwrap_or(handle_bits);
        let ty = biara_ir::Type::Int(bits);
        let id = self.emit(biara_ir::Inst::Select { ty, cond: c, yes: y, no: n });
        Ok(BasicValueEnum::IntValue(IntValue::new(self.ctx, id, bits)))
    }

    pub fn build_int_truncate(&self, v: IntValue<'ctx>, ty: IntType<'ctx>, _n: &str) -> Result<IntValue<'ctx>, BuilderError> {
        let value = self.id(v);
        let from_bits = self.arena_int_bits(value).unwrap_or(v.bits);
        let from = biara_ir::Type::Int(from_bits);
        let to = biara_ir::Type::Int(ty.get_bit_width());
        // A cast to the width it already has changes nothing; emitting it
        // would be an ill-formed truncate, so it is the identity.
        if from_bits == ty.get_bit_width() {
            return Ok(v);
        }
        let id = self.emit(biara_ir::Inst::Trunc { from, to, value });
        // The result width is the destination's, whatever the arena says.
        Ok(IntValue::new(self.ctx, id, ty.get_bit_width()))
    }

    pub fn build_int_z_extend(&self, v: IntValue<'ctx>, ty: IntType<'ctx>, _n: &str) -> Result<IntValue<'ctx>, BuilderError> {
        let value = self.id(v);
        let from_bits = self.arena_int_bits(value).unwrap_or(v.bits);
        let from = biara_ir::Type::Int(from_bits);
        let to = biara_ir::Type::Int(ty.get_bit_width());
        if from_bits == ty.get_bit_width() {
            return Ok(v);
        }
        let id = self.emit(biara_ir::Inst::Zext { from, to, value });
        // The result width is the destination's, whatever the arena says.
        Ok(IntValue::new(self.ctx, id, ty.get_bit_width()))
    }

    pub fn build_int_s_extend(&self, v: IntValue<'ctx>, ty: IntType<'ctx>, _n: &str) -> Result<IntValue<'ctx>, BuilderError> {
        let value = self.id(v);
        let from_bits = self.arena_int_bits(value).unwrap_or(v.bits);
        let from = biara_ir::Type::Int(from_bits);
        let to = biara_ir::Type::Int(ty.get_bit_width());
        if from_bits == ty.get_bit_width() {
            return Ok(v);
        }
        let id = self.emit(biara_ir::Inst::Sext { from, to, value });
        // The result width is the destination's, whatever the arena says.
        Ok(IntValue::new(self.ctx, id, ty.get_bit_width()))
    }

    // ---- memory ---------------------------------------------------------

    pub fn build_alloca(
        &self,
        ty: impl Into<BasicTypeEnum<'ctx>>,
        _n: &str,
    ) -> Result<PointerValue<'ctx>, BuilderError> {
        let ir = ty.into().to_ir();
        let id = self.emit(biara_ir::Inst::Alloca { ty: ir });
        Ok(PointerValue::new(self.ctx, id))
    }

    pub fn build_load(
        &self,
        ty: impl Into<BasicTypeEnum<'ctx>>,
        ptr: PointerValue<'ctx>,
        _n: &str,
    ) -> Result<BasicValueEnum<'ctx>, BuilderError> {
        let ty = ty.into();
        let ir = ty.to_ir();
        let id = self.emit(biara_ir::Inst::Load {
            ty: ir.clone(),
            ptr: ptr.id,
            align: 1,
        });
        Ok(match &ir {
            biara_ir::Type::Int(bits) => BasicValueEnum::IntValue(IntValue::new(self.ctx, id, *bits)),
            biara_ir::Type::Ptr => BasicValueEnum::PointerValue(PointerValue::new(self.ctx, id)),
            _ => {
                let t = self.ctx.intern(ir);
                BasicValueEnum::StructValue(StructValue {
                    ctx: self.ctx,
                    ty: t,
                    id,
                })
            }
        })
    }

    pub fn build_store(
        &self,
        ptr: PointerValue<'ctx>,
        v: impl Into<BasicValueEnum<'ctx>>,
    ) -> Result<InstructionValue<'ctx>, BuilderError> {
        let v = v.into();
        let (value, bits) = match v {
            BasicValueEnum::IntValue(i) => (self.id(i), i.bits),
            BasicValueEnum::PointerValue(p) => (p.id, 64),
            BasicValueEnum::StructValue(s) => (s.id, 64),
        };
        let id = self.emit(biara_ir::Inst::Store {
            ty: biara_ir::Type::Int(bits),
            value,
            ptr: ptr.id,
            align: 1,
        });
        Ok(InstructionValue {
            ctx: self.ctx,
            id,
            term: None,
            func: self.pos.get().and_then(|p| p.func),
            _m: PhantomData,
        })
    }

    pub fn build_int_to_ptr(&self, v: IntValue<'ctx>, _ty: crate::types::PointerType<'ctx>, _n: &str) -> Result<PointerValue<'ctx>, BuilderError> {
        let value = self.id(v);
        let from = biara_ir::Type::Int(v.bits);
        let id = self.emit(biara_ir::Inst::Cast {
            from,
            to: biara_ir::Type::Ptr,
            value,
        });
        Ok(PointerValue::new(self.ctx, id))
    }

    pub fn build_ptr_to_int(&self, p: PointerValue<'ctx>, ty: IntType<'ctx>, _n: &str) -> Result<IntValue<'ctx>, BuilderError> {
        let id = self.emit(biara_ir::Inst::Cast {
            from: biara_ir::Type::Ptr,
            to: biara_ir::Type::Int(ty.get_bit_width()),
            value: p.id,
        });
        Ok(IntValue::new(self.ctx, id, ty.get_bit_width()))
    }

    pub fn build_gep(&self, _elem: IntType<'ctx>, base: PointerValue<'ctx>, offsets: &[IntValue<'ctx>], _n: &str) -> Result<PointerValue<'ctx>, BuilderError> {
        let offset = offsets
            .first()
            .map(|v| match v.src {
                IntSource::Const(bits) => bits as i64,
                IntSource::Value(_) => panic!("non-constant GEP offset"),
            })
            .unwrap_or(0);
        let id = self.emit(biara_ir::Inst::Gep {
            base: base.id,
            offset,
        });
        Ok(PointerValue::new(self.ctx, id))
    }

    // ---- control flow ---------------------------------------------------

    fn set_term(&self, term: biara_ir::Terminator) -> Result<(), BuilderError> {
        let Some(p) = self.pos.get() else {
            return Err(BuilderError::NoPosition);
        };
        // Block ids repeat across functions, so the position's function must
        // own the block. Falling back to "any function with this block index"
        // writes the terminator into another function's block.
        let owner = self.ctx.with(self.ctx.module_id(), |m| {
            p.func
                .filter(|&i| {
                    m.module
                        .functions
                        .get(i)
                        .is_some_and(|f| f.blocks.get(p.block.0 as usize).is_some())
                })
                .or_else(|| {
                    m.module
                        .functions
                        .iter()
                        .position(|f| f.blocks.get(p.block.0 as usize).is_some())
                })
        });
        let Some(owner) = owner else {
            return Err(BuilderError::NoPosition);
        };
        self.ctx.with_mut(self.ctx.module_id(), |m| {
            m.module.functions[owner].blocks[p.block.0 as usize].term = Some(term);
        });
        Ok(())
    }

    pub fn build_unconditional_branch(&self, to: BasicBlock<'ctx>) -> Result<(), BuilderError> {
        self.set_term(biara_ir::Terminator::Br(to.block))
    }

    pub fn build_conditional_branch(&self, cond: IntValue<'ctx>, yes: BasicBlock<'ctx>, no: BasicBlock<'ctx>) -> Result<(), BuilderError> {
        let c = self.id(cond);
        self.set_term(biara_ir::Terminator::CondBr {
            cond: c,
            yes: yes.block,
            no: no.block,
        })
    }

    pub fn build_switch(&self, value: IntValue<'ctx>, default: BasicBlock<'ctx>, cases: &[(IntValue<'ctx>, BasicBlock<'ctx>)]) -> Result<(), BuilderError> {
        let v = self.id(value);
        let cases = cases
            .iter()
            .map(|(c, b)| {
                let key = match c.src {
                    IntSource::Const(bits) => bits,
                    IntSource::Value(_) => panic!("non-constant switch case"),
                };
                (key, b.block)
            })
            .collect();
        self.set_term(biara_ir::Terminator::Switch {
            value: v,
            default: default.block,
            cases,
        })
    }

    /// `ret`. `None` returns void; a value returns it.
    pub fn build_return(
        &self,
        v: Option<&(dyn crate::values::DynBasicValue + 'ctx)>,
    ) -> Result<(), BuilderError> {
        let v = match v {
            None => None,
            Some(v) => Some(match v.as_basic_value_enum_dyn() {
                BasicValueEnum::IntValue(i) => self.id(i),
                BasicValueEnum::PointerValue(p) => p.id,
                BasicValueEnum::StructValue(s) => s.id,
            }),
        };
        self.set_term(biara_ir::Terminator::Ret(v))
    }

    pub fn build_unreachable(&self) -> Result<(), BuilderError> {
        self.set_term(biara_ir::Terminator::Unreachable)
    }

    pub fn build_fence(&self, _o: crate::support::AtomicOrdering, _single: bool, _n: &str) -> Result<InstructionValue<'ctx>, BuilderError> {
        let id = self.emit(biara_ir::Inst::Fence);
        Ok(InstructionValue {
            ctx: self.ctx,
            id,
            term: None,
            func: self.pos.get().and_then(|p| p.func),
            _m: PhantomData,
        })
    }

    // ---- calls ----------------------------------------------------------

    pub fn build_call(&self, callee: FunctionValue<'ctx>, args: &[BasicMetadataValueEnum<'ctx>], _n: &str) -> Result<CallReturn<'ctx>, BuilderError> {
        let name = callee.get_name().0;
        let ret = callee.get_type().ir.ret.clone();
        let args: Vec<biara_ir::ValueId> = args
            .iter()
            .map(|a| match a.to_value() {
                BasicValueEnum::IntValue(v) => self.id(v),
                BasicValueEnum::PointerValue(v) => v.id,
                BasicValueEnum::StructValue(v) => v.id,
            })
            .collect();
        let callee_id = self.ctx.with_mut(self.ctx.module_id(), |m| {
            m.build_constant(
                self.func().unwrap_or(0),
                biara_ir::Constant::Function(name.to_owned()),
            )
        });
        let id = self.emit(biara_ir::Inst::Call {
            callee: callee_id,
            args,
            ty: ret.clone(),
        });
        Ok(CallReturn {
            ctx: self.ctx,
            id,
            ret,
            func: self.func(),
        })
    }

    /// `extractvalue` of one element of an aggregate result.
    pub fn build_extract_value(&self, agg: StructValue<'ctx>, index: u32, _n: &str) -> Result<BasicValueEnum<'ctx>, BuilderError> {
        let elem_ty = match &agg.ty {
            biara_ir::Type::Struct(f) => f
                .get(index as usize)
                .cloned()
                .unwrap_or(biara_ir::Type::Int(64)),
            _ => biara_ir::Type::Int(64),
        };
        let id = self.emit(biara_ir::Inst::ExtractValue {
            agg: agg.id,
            index,
            ty: elem_ty.clone(),
        });
        Ok(match elem_ty.int_bits() {
            Some(b) => BasicValueEnum::IntValue(IntValue::new(self.ctx, id, b)),
            None => BasicValueEnum::PointerValue(PointerValue::new(self.ctx, id)),
        })
    }
}

/// The result of `build_call`.
pub struct CallReturn<'ctx> {
    ctx: &'ctx Context,
    id: biara_ir::ValueId,
    ret: biara_ir::Type,
    /// The function holding the call: ids repeat across functions, so the
    /// owning function has to travel with the handle.
    func: Option<usize>,
}

impl<'ctx> CallReturn<'ctx> {
    pub fn try_as_basic_value(&self) -> ValueKind<'ctx> {
        match &self.ret {
            biara_ir::Type::Void => ValueKind::Instruction(InstructionValue {
                ctx: self.ctx,
                id: self.id,
                term: None,
                func: self.func,
                _m: PhantomData,
            }),
            biara_ir::Type::Int(b) => {
                ValueKind::Basic(BasicValueEnum::IntValue(IntValue::new(self.ctx, self.id, *b)))
            }
            biara_ir::Type::Ptr => ValueKind::Basic(BasicValueEnum::PointerValue(
                PointerValue::new(self.ctx, self.id),
            )),
            other => {
                let ty = self.ctx.intern(other.clone());
                ValueKind::Basic(BasicValueEnum::StructValue(StructValue {
                    ctx: self.ctx,
                    ty,
                    id: self.id,
                }))
            }
        }
    }
}





