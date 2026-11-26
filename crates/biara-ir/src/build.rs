//! A builder for one function: values, blocks, and a cursor.

use crate::{
    BinOp, BlockId, Constant, FuncType, Function, Inst, IntPred, Terminator, Type, Value, ValueId,
};

pub struct Builder {
    pub func: Function,
    /// Guest address recorded on values emitted next (see `Function::pc`).
    pub pc: Option<u64>,
}

/// A block under construction.
pub struct Cursor {
    pub block: BlockId,
}

impl Builder {
    pub fn new(name: impl Into<String>, ty: FuncType) -> Self {
        Self {
            func: Function::new(name, ty),
            pc: None,
        }
    }

    /// Continue building an existing function (handles hold one across calls).
    pub fn resume(func: Function) -> Self {
        Self { func, pc: None }
    }

    pub fn const_int(&mut self, v: i128) -> ValueId {
        self.func.add_const(Constant::int(v, 64))
    }

    pub fn const_typed(&mut self, v: i128, width: u32) -> ValueId {
        self.func.add_const(Constant::int(v, width))
    }

    pub fn const_zero(&mut self, ty: &Type) -> ValueId {
        let c = match ty {
            Type::Int(w) => Constant::zero_int(*w),
            _ => Constant::Zero,
        };
        self.func.add_const(c)
    }

    pub fn const_all_ones(&mut self, ty: &Type) -> ValueId {
        let w = ty.int_bits().unwrap_or(64);
        self.func.add_const(Constant::all_ones(w))
    }

    pub fn const_undef(&mut self) -> ValueId {
        self.func.add_const(Constant::Undef)
    }

    pub fn const_func(&mut self, name: &str) -> ValueId {
        self.func.add_const(Constant::Function(name.to_owned()))
    }

    pub fn add_param(&mut self, ty: Type, name: &str) -> ValueId {
        let index = self.func.params.len() as u32;
        self.func.values.push(Value::Param {
            ty: ty.clone(),
            index,
            name: name.to_owned(),
        });
        self.func.defined_in.push(None);
        self.func.pc.push(None);
        let id = ValueId((self.func.values.len() - 1) as u32);
        self.func.params.push(id);
        id
    }

    pub fn add_block(&mut self, name: &str) -> BlockId {
        let id = BlockId(self.func.blocks.len() as u32);
        self.func.blocks.push(crate::Block {
            name: name.to_owned(),
            insts: Vec::new(),
            term: None,
            preds: Default::default(),
        });
        if self.func.entry.is_none() {
            self.func.entry = Some(id);
        }
        id
    }

    /// Emit `inst` into `b`, returning its result. Instructions whose type is
    /// `Void` still get an id (no uses).
    pub fn emit(&mut self, b: BlockId, inst: Inst) -> ValueId {
        self.func.values.push(Value::Inst(inst));
        self.func.defined_in.push(Some(b));
        self.func.pc.push(self.pc);
        let id = ValueId((self.func.values.len() - 1) as u32);
        self.func.blocks[b.0 as usize].insts.push(id);
        id
    }

    pub fn bin(&mut self, b: BlockId, op: BinOp, ty: &Type, lhs: ValueId, rhs: ValueId) -> ValueId {
        self.emit(
            b,
            Inst::Bin {
                op,
                ty: ty.clone(),
                lhs,
                rhs,
            },
        )
    }

    pub fn icmp(&mut self, b: BlockId, pred: IntPred, ty: &Type, lhs: ValueId, rhs: ValueId) -> ValueId {
        self.emit(
            b,
            Inst::Icmp {
                pred,
                ty: ty.clone(),
                lhs,
                rhs,
            },
        )
    }

    pub fn select(&mut self, b: BlockId, ty: &Type, cond: ValueId, yes: ValueId, no: ValueId) -> ValueId {
        self.emit(
            b,
            Inst::Select {
                ty: ty.clone(),
                cond,
                yes,
                no,
            },
        )
    }

    pub fn load(&mut self, b: BlockId, ty: &Type, ptr: ValueId) -> ValueId {
        self.load_aligned(b, ty, ptr, ty_bytes(ty))
    }

    /// A load at an explicit alignment; guest memory uses 1.
    pub fn load_aligned(&mut self, b: BlockId, ty: &Type, ptr: ValueId, align: u32) -> ValueId {
        self.emit(
            b,
            Inst::Load {
                ty: ty.clone(),
                ptr,
                align,
            },
        )
    }

    pub fn store(&mut self, b: BlockId, ty: &Type, value: ValueId, ptr: ValueId) -> ValueId {
        self.store_aligned(b, ty, value, ptr, ty_bytes(ty))
    }

    /// A store at an explicit alignment; guest memory uses 1.
    pub fn store_aligned(
        &mut self,
        b: BlockId,
        ty: &Type,
        value: ValueId,
        ptr: ValueId,
        align: u32,
    ) -> ValueId {
        self.emit(
            b,
            Inst::Store {
                ty: ty.clone(),
                value,
                ptr,
                align,
            },
        )
    }

    pub fn un(&mut self, b: BlockId, op: crate::UnOp, ty: &Type, value: ValueId) -> ValueId {
        self.emit(
            b,
            Inst::Un {
                op,
                ty: ty.clone(),
                value,
            },
        )
    }

    pub fn rot(
        &mut self,
        b: BlockId,
        left: bool,
        ty: &Type,
        value: ValueId,
        amount: ValueId,
    ) -> ValueId {
        self.emit(
            b,
            Inst::Rot {
                left,
                ty: ty.clone(),
                value,
                amount,
            },
        )
    }

    pub fn fence(&mut self, b: BlockId) -> ValueId {
        self.emit(b, Inst::Fence)
    }

    pub fn alloca(&mut self, b: BlockId, ty: &Type) -> ValueId {
        self.emit(
            b,
            Inst::Alloca { ty: ty.clone() },
        )
    }

    pub fn gep(&mut self, b: BlockId, base: ValueId, offset: i64) -> ValueId {
        self.emit(b, Inst::Gep { base, offset })
    }

    pub fn cast(&mut self, b: BlockId, from: &Type, to: &Type, value: ValueId) -> ValueId {
        self.emit(
            b,
            Inst::Cast {
                from: from.clone(),
                to: to.clone(),
                value,
            },
        )
    }

    pub fn trunc(&mut self, b: BlockId, from: &Type, to: &Type, value: ValueId) -> ValueId {
        self.emit(
            b,
            Inst::Trunc {
                from: from.clone(),
                to: to.clone(),
                value,
            },
        )
    }

    pub fn zext(&mut self, b: BlockId, from: &Type, to: &Type, value: ValueId) -> ValueId {
        self.emit(
            b,
            Inst::Zext {
                from: from.clone(),
                to: to.clone(),
                value,
            },
        )
    }

    pub fn sext(&mut self, b: BlockId, from: &Type, to: &Type, value: ValueId) -> ValueId {
        self.emit(
            b,
            Inst::Sext {
                from: from.clone(),
                to: to.clone(),
                value,
            },
        )
    }

    pub fn call(&mut self, b: BlockId, callee: ValueId, args: Vec<ValueId>, ty: &Type) -> ValueId {
        self.emit(
            b,
            Inst::Call {
                callee,
                args,
                ty: ty.clone(),
            },
        )
    }

    pub fn phi(&mut self, b: BlockId, ty: &Type) -> ValueId {
        self.emit(
            b,
            Inst::Phi {
                ty: ty.clone(),
                incoming: Vec::new(),
            },
        )
    }

    pub fn add_incoming(&mut self, phi: ValueId, value: ValueId, from: BlockId) {
        if let Value::Inst(Inst::Phi { incoming, .. }) = self.func.value_mut(phi) {
            incoming.push((value, from));
        }
    }

    pub fn ret(&mut self, b: BlockId, v: Option<ValueId>) {
        self.func.blocks[b.0 as usize].term = Some(Terminator::Ret(v));
    }

    pub fn br(&mut self, b: BlockId, to: BlockId) {
        self.func.blocks[b.0 as usize].term = Some(Terminator::Br(to));
    }

    pub fn cond_br(&mut self, b: BlockId, cond: ValueId, yes: BlockId, no: BlockId) {
        self.func.blocks[b.0 as usize].term = Some(Terminator::CondBr { cond, yes, no });
    }

    pub fn switch(&mut self, b: BlockId, value: ValueId, default: BlockId, cases: Vec<(u128, BlockId)>) {
        self.func.blocks[b.0 as usize].term = Some(Terminator::Switch {
            value,
            default,
            cases,
        });
    }

    pub fn unreachable(&mut self, b: BlockId) {
        self.func.blocks[b.0 as usize].term = Some(Terminator::Unreachable);
    }

    pub fn finish(mut self) -> Function {
        crate::recompute_preds(&mut self.func);
        self.func
    }
}

/// Size in bytes of a type, for the default alignment.
pub fn ty_bytes(t: &Type) -> u32 {
    match t {
        Type::Int(w) | Type::Float(w) => (w.div_ceil(8)).max(1),
        Type::Ptr => 8,
        Type::Array(e, n) => ty_bytes(e) * (*n as u32),
        Type::Struct(fields) => fields.iter().map(ty_bytes).sum(),
        Type::Vector(e, n) => ty_bytes(e) * *n,
        Type::Void | Type::Func(_) => 1,
    }
}
