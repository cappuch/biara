//! The context, as inkwell exposed it (`Module::get_context`).

use crate::types::BasicType;

pub use crate::Context;

/// Anything holding a context.
pub trait AsContextRef<'ctx> {
    fn as_ctx(&self) -> ContextRef<'ctx>;
}

impl<'ctx> AsContextRef<'ctx> for ContextRef<'ctx> {
    fn as_ctx(&self) -> ContextRef<'ctx> {
        *self
    }
}

impl<'ctx> AsContextRef<'ctx> for crate::module::Module<'ctx> {
    fn as_ctx(&self) -> ContextRef<'ctx> {
        ContextRef(self.ctx)
    }
}

/// A borrowed context.
#[derive(Clone, Copy)]
pub struct ContextRef<'ctx>(pub(crate) &'ctx crate::Context);

impl<'ctx> ContextRef<'ctx> {
    /// Reinterpret as a foreign handle (the context's address).
    pub fn cast<T>(&self) -> T
    where
        T: FromContextRef,
    {
        T::from_context_ref(*self)
    }

    pub fn get_kind_id(&self, name: &str) -> u32 {
        self.0.get_kind_id(name)
    }

    pub fn i8_type(&self) -> crate::types::IntType<'ctx> {
        crate::types::IntType::new(self.0, 8)
    }
    pub fn i16_type(&self) -> crate::types::IntType<'ctx> {
        crate::types::IntType::new(self.0, 16)
    }
    pub fn i32_type(&self) -> crate::types::IntType<'ctx> {
        crate::types::IntType::new(self.0, 32)
    }
    pub fn i64_type(&self) -> crate::types::IntType<'ctx> {
        crate::types::IntType::new(self.0, 64)
    }
    pub fn i128_type(&self) -> crate::types::IntType<'ctx> {
        crate::types::IntType::new(self.0, 128)
    }
    pub fn bool_type(&self) -> crate::types::IntType<'ctx> {
        crate::types::IntType::new(self.0, 1)
    }
    pub fn void_type(&self) -> crate::types::VoidType<'ctx> {
        crate::types::VoidType::new(self.0)
    }
    pub fn ptr_type(&self, _space: crate::AddressSpace) -> crate::types::PointerType<'ctx> {
        crate::types::PointerType::new(self.0)
    }
    pub fn struct_type(&self, fields: &[crate::types::BasicTypeEnum<'ctx>], packed: bool) -> crate::types::StructType<'ctx> {
        crate::types::StructType::new(
            self.0,
            biara_ir::Type::Struct(fields.iter().map(|f| f.to_ir()).collect()),
        )
    }

    /// A metadata string node.
    pub fn metadata_string<'a>(&'a self, s: &str) -> crate::values::MetadataValue<'ctx> {
        crate::values::MetadataValue::new(self.0, crate::MetadataNode::Str(s.to_owned()))
    }

    /// A metadata tuple.
    pub fn metadata_node<'a>(&'a self, items: &[crate::values::MetadataValue<'ctx>]) -> crate::values::MetadataValue<'ctx> {
        crate::values::MetadataValue::new(
            self.0,
            crate::MetadataNode::Node(items.iter().map(|m| m.node().clone()).collect()),
        )
    }

    /// Add a block to `f`, returning it.
    pub fn append_basic_block(
        &self,
        f: crate::values::FunctionValue<'ctx>,
        name: &str,
    ) -> crate::basic_block::BasicBlock<'ctx> {
        let idx = self.0.with_mut(self.0.module_id(), |m| {
            let func_index = f.func;
            let func = &mut m.module.functions[f.func];
            let id = biara_ir::BlockId(func.blocks.len() as u32);
            func.blocks.push(biara_ir::Block {
                name: name.to_owned(),
                insts: Vec::new(),
                term: None,
                preds: Default::default(),
            });
            if func.entry.is_none() {
                func.entry = Some(id);
            }
            (func_index, id)
        });
        crate::basic_block::BasicBlock::new(self.0, idx.0, idx.1)
    }

    /// A string attribute (`target-features` and friends).
    pub fn create_string_attribute(
        &self,
        kind: impl Into<String>,
        _value: &str,
    ) -> crate::attributes::Attribute {
        crate::attributes::Attribute {
            kind: Box::leak(kind.into().into_boxed_str()),
            value: 0,
        }
    }

    pub fn as_ctx_ref(&self) -> ContextRef<'ctx> {
        *self
    }

    pub fn create_builder(&self) -> crate::builder::Builder<'ctx> {
        crate::builder::Builder::new(self.0)
    }

    pub fn create_enum_attribute(
        &self,
        kind: impl Into<String>,
        value: u64,
    ) -> crate::attributes::Attribute {
        let kind: String = kind.into();
        crate::attributes::Attribute {
            kind: Box::leak(kind.into_boxed_str()),
            value,
        }
    }
}

/// Types that can be rebuilt from a borrowed context.
pub trait FromContextRef {
    fn from_context_ref(ctx: ContextRef<'_>) -> Self;
}

impl FromContextRef for usize {
    fn from_context_ref(ctx: ContextRef<'_>) -> Self {
        ctx.0 as *const crate::Context as usize
    }
}

impl FromContextRef for *mut std::ffi::c_void {
    fn from_context_ref(ctx: ContextRef<'_>) -> Self {
        ctx.0 as *const crate::Context as *mut std::ffi::c_void
    }
}
