//! Basic block handles.

use crate::Context;
use std::marker::PhantomData;

/// A block in the function currently addressed.
#[derive(Clone, Copy)]
pub struct BasicBlock<'ctx> {
    pub(crate) ctx: &'ctx Context,
    /// The function that owns this block. Block ids repeat across functions,
    /// so a block handle must state its owner rather than guess by scanning.
    pub(crate) func: usize,
    pub(crate) block: biara_ir::BlockId,
    _m: PhantomData<&'ctx ()>,
}

impl<'ctx> PartialEq for BasicBlock<'ctx> {
    fn eq(&self, other: &Self) -> bool {
        self.block == other.block
    }
}
impl<'ctx> Eq for BasicBlock<'ctx> {}
impl<'ctx> std::hash::Hash for BasicBlock<'ctx> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.block.hash(state);
    }
}
impl<'ctx> PartialOrd for BasicBlock<'ctx> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl<'ctx> Ord for BasicBlock<'ctx> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.block.cmp(&other.block)
    }
}

impl<'ctx> BasicBlock<'ctx> {
    pub(crate) fn new(
        ctx: &'ctx Context,
        func: usize,
        block: biara_ir::BlockId,
    ) -> Self {
        Self {
            ctx,
            func,
            block,
            _m: PhantomData,
        }
    }

    /// The block's terminator, when one is set.
    ///
    /// A block with instructions but no terminator is still being built;
    /// reporting a terminator there would make a half-built block look
    /// finished.
    pub fn get_terminator(&self) -> Option<crate::values::InstructionValue<'ctx>> {
        let has = self
            .ctx
            .with(self.ctx.module_id(), |m| {
                m.module
                    .functions
                    .get(self.func)
                    .and_then(|f| f.blocks.get(self.block.0 as usize))
                    .is_some_and(|b| b.term.is_some())
            });
        has.then(|| crate::values::InstructionValue {
            ctx: self.ctx,
            id: biara_ir::ValueId(u32::MAX),
            term: Some(self.block),
            func: Some(self.func),
            _m: PhantomData,
        })
    }

    /// The block's instructions, in order.
    pub fn get_instructions(&self) -> std::vec::IntoIter<crate::values::InstructionValue<'ctx>> {
        let ids = self.ctx.with(self.ctx.module_id(), |m| {
            m.module
                .functions
                .get(self.func)
                .and_then(|f| f.blocks.get(self.block.0 as usize))
                .map(|b| b.insts.clone())
                .unwrap_or_default()
        });
        ids.into_iter()
            .map(|id| crate::values::InstructionValue {
                ctx: self.ctx,
                id,
                term: None,
                func: Some(self.func),
                _m: PhantomData,
            })
            .collect::<Vec<_>>()
            .into_iter()
    }

    pub fn get_first_instruction(&self) -> Option<crate::values::InstructionValue<'ctx>> {
        self.get_instructions().next()
    }

    pub fn get_last_instruction(&self) -> Option<crate::values::InstructionValue<'ctx>> {
        self.get_instructions().next_back()
    }

    pub fn get_context(&self) -> &'_ Context {
        self.ctx
    }
}

impl<'ctx> crate::values::AsValueRef for BasicBlock<'ctx> {
    fn as_value_ref(&self) -> crate::values::ValueRef {
        crate::values::ValueRef(self.block.0)
    }
}
