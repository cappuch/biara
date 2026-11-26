//! The `inkwell` API shape `biara-lift` was written against, over [`biara_ir`].
//!
//! Handles borrow `&'ctx Context` and are `Copy`, exactly like inkwell's, so
//! the 18k lines of analysis code compile unchanged. The IR arena lives behind
//! `RefCell`s inside `Context`; every handle is `{ &'ctx Context, indices }`.
//!
//! What LLVM supplied as a *library* — passes, analyses, verification, printing
//! — is in `biara_ir`.

use std::cell::RefCell;
use std::collections::HashMap;
use std::marker::PhantomData;

pub use biara_ir as ir;

pub mod attributes;
pub mod basic_block;
pub mod builder;
pub mod context;
pub mod intrinsics;
pub mod memory_buffer;
pub mod module;
pub mod passes;
pub mod targets;
pub mod types;
pub mod values;

pub mod support;
pub use builder::{Builder, CallReturn};
pub use module::Module;
pub use support::{AddressSpace, AtomicOrdering, IntPredicate, OptimizationLevel, PassBuilderOptions};
pub use values::{
    BasicMetadataValueEnum, BasicValue, BasicValueEnum, CallSiteValue, FunctionValue,
    InstructionOpcode, InstructionValue, IntValue, MetadataValue, Operand, PhiValue, PointerValue,
    StringRef, ValueKind,
};


/// Module storage: the IR plus a name index and module metadata.
pub struct ModuleData {
    pub module: biara_ir::Module,
    /// Function name to index into `module.functions`.
    pub index: HashMap<String, usize>,
    /// Module-level metadata, by kind.
    pub metadata: HashMap<String, Vec<MetadataNode>>,
    /// Globals by name (data the C renderer reads).
    pub globals: HashMap<String, ()>,
}

/// One metadata attachment: either a value payload or a list of sub-nodes.
#[derive(Clone, Debug, PartialEq)]
pub enum MetadataNode {
    Int(i128),
    Str(String),
    /// A metadata tuple.
    Node(Vec<MetadataNode>),
    /// An empty node (`metadata !{}`).
    Empty,
}

impl ModuleData {
    pub fn new(name: &str) -> Self {
        Self {
            module: biara_ir::Module::new(name, "", ""),
            index: HashMap::new(),
            metadata: HashMap::new(),
            globals: HashMap::new(),
        }
    }

    pub fn reindex(&mut self) {
        self.index.clear();
        for (i, f) in self.module.functions.iter().enumerate() {
            self.index.insert(f.name.clone(), i);
        }
    }
}

/// The arena one `Context` owns.
#[derive(Default)]
pub struct Arena {
    pub modules: Vec<ModuleData>,
    /// The module new handles bind to (`Module::add_function` sets it).
    pub current: usize,
}

/// Owns the IR. Handles borrow it for `'ctx`.
pub struct Context {
    pub(crate) arena: RefCell<Arena>,
    /// Metadata kind names interned to ids (`get_kind_id`).
    kinds: RefCell<HashMap<String, u32>>,
    /// Types interned for the handles' lifetimes. Entries are leaked on
    /// purpose: handles need `&'ctx Type`, and a context lives for the
    /// analysis, so the set stays small and bounded by the module's types.
    types: RefCell<HashMap<biara_ir::Type, &'static biara_ir::Type>>,
    /// Metadata nodes interned for the same reason.
    nodes: RefCell<Vec<&'static MetadataNode>>,
    /// Strings interned so `StringRef` can borrow the context like inkwell's.
    strings: RefCell<Vec<&'static str>>,
}

impl Context {
    pub fn create() -> Self {
        Self {
            arena: RefCell::new(Arena {
                modules: vec![ModuleData::new("lifted")],
                current: 0,
            }),
            kinds: RefCell::new(HashMap::new()),
            types: RefCell::new(HashMap::new()),
            nodes: RefCell::new(Vec::new()),
            strings: RefCell::new(Vec::new()),
        }
    }

    /// A string borrowable for `'ctx`.
    pub(crate) fn intern_str(&self, s: String) -> &'static str {
        let leaked: &'static str = Box::leak(s.into_boxed_str());
        self.strings.borrow_mut().push(leaked);
        leaked
    }

    /// A metadata node whose address outlives every handle.
    pub(crate) fn intern_node(&self, node: MetadataNode) -> &'static MetadataNode {
        let leaked: &'static MetadataNode = Box::leak(Box::new(node));
        self.nodes.borrow_mut().push(leaked);
        leaked
    }

    /// A type whose address outlives every handle.
    pub(crate) fn intern(&self, ty: biara_ir::Type) -> &'static biara_ir::Type {
        let mut types = self.types.borrow_mut();
        if let Some(&t) = types.get(&ty) {
            return t;
        }
        let leaked: &'static biara_ir::Type = Box::leak(Box::new(ty.clone()));
        types.insert(ty, leaked);
        leaked
    }

    pub fn create_module(&self, name: &str) -> module::Module<'_> {
        let id = {
            let mut arena = self.arena.borrow_mut();
            arena.modules.push(ModuleData::new(name));
            let id = arena.modules.len() - 1;
            arena.current = id;
            id
        };
        module::Module::new(self, id)
    }

    /// The context's first module.
    pub fn module(&self) -> module::Module<'_> {
        module::Module::new(self, 0)
    }

    /// The module index a handle addresses, defaulting to the first.
    /// The module this context owns, for callers that must mutate it
    /// directly (a cross-crate finalisation pass, a diagnostics helper).
    pub fn module_id_public(&self) -> usize {
        self.module_id()
    }

    pub(crate) fn module_id(&self) -> usize {
        let arena = self.arena.borrow();
        arena.current
    }

    /// Intern a metadata kind name.
    pub fn get_kind_id(&self, name: &str) -> u32 {
        let mut kinds = self.kinds.borrow_mut();
        if let Some(&id) = kinds.get(name) {
            return id;
        }
        let id = kinds.len() as u32;
        kinds.insert(name.to_owned(), id);
        id
    }

    pub fn i8_type(&self) -> types::IntType<'_> {
        types::IntType::new(self, 8)
    }
    pub fn i16_type(&self) -> types::IntType<'_> {
        types::IntType::new(self, 16)
    }
    pub fn i32_type(&self) -> types::IntType<'_> {
        types::IntType::new(self, 32)
    }
    pub fn i64_type(&self) -> types::IntType<'_> {
        types::IntType::new(self, 64)
    }
    pub fn i128_type(&self) -> types::IntType<'_> {
        types::IntType::new(self, 128)
    }
    pub fn bool_type(&self) -> types::IntType<'_> {
        types::IntType::new(self, 1)
    }

    pub fn void_type(&self) -> types::VoidType<'_> {
        types::VoidType::new(self)
    }

    pub fn ptr_type(&self, _space: AddressSpace) -> types::PointerType<'_> {
        types::PointerType::new(self)
    }

    pub fn struct_type(&self, fields: &[types::BasicTypeEnum<'_>], _packed: bool) -> types::StructType<'_> {
        use types::BasicType as _;
        types::StructType::new(
            self,
            biara_ir::Type::Struct(fields.iter().map(|f| f.to_ir()).collect()),
        )
    }

    /// `double`, `float`, … — the guest's vector lanes lower to these.
    pub fn f32_type(&self) -> types::FloatType<'_> {
        types::FloatType::new(self, 32)
    }
    pub fn f64_type(&self) -> types::FloatType<'_> {
        types::FloatType::new(self, 64)
    }

    /// Add a block to `f`.
    pub fn append_basic_block<'a>(
        &'a self,
        f: values::FunctionValue<'a>,
        name: &str,
    ) -> basic_block::BasicBlock<'a> {
        let id = self.with_mut(f.module, |m| {
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
            id
        });
        basic_block::BasicBlock::new(f.ctx, f.func, id)
    }

    /// A string attribute (`target-features`, declared-parameter spellings).
    pub fn create_string_attribute(
        &self,
        kind: impl Into<String>,
        _value: &str,
    ) -> attributes::Attribute {
        attributes::Attribute {
            kind: Box::leak(kind.into().into_boxed_str()),
            value: 0,
        }
    }

    /// A borrowed view, for call sites that take `impl AsContext`.
    pub fn as_ctx_ref(&self) -> context::ContextRef<'_> {
        context::ContextRef(self)
    }

    pub fn create_builder(&self) -> builder::Builder<'_> {
        builder::Builder::new(self)
    }

    /// An enum attribute (`noalias`, `captures`, `memory`).
    pub fn create_enum_attribute(
        &self,
        kind: impl Into<String>,
        value: u64,
    ) -> attributes::Attribute {
        attributes::Attribute {
            kind: Box::leak(kind.into().into_boxed_str()),
            value,
        }
    }

    /// A metadata string node.
    pub fn metadata_string(&self, s: &str) -> values::MetadataValue<'_> {
        values::MetadataValue::new(self, MetadataNode::Str(s.to_owned()))
    }

    /// A metadata tuple built from values and strings.
    pub fn metadata_node(&self, items: &[values::MetadataValue<'_>]) -> values::MetadataValue<'_> {
        values::MetadataValue::new(
            self,
            MetadataNode::Node(items.iter().map(|m| m.node().clone()).collect()),
        )
    }

    /// Run `f` over the current module's data (direct IR access for ports).
    pub fn update<R>(&self, f: impl FnOnce(&mut ModuleData) -> R) -> R {
        let id = self.module_id();
        self.with_mut(id, f)
    }

    /// Borrow module `id` immutably.
    pub fn with<R>(&self, id: usize, f: impl FnOnce(&ModuleData) -> R) -> R {
        let arena = self.arena.borrow();
        f(&arena.modules[id])
    }

    /// Borrow module `id` mutably.
    pub fn with_mut<R>(&self, id: usize, f: impl FnOnce(&mut ModuleData) -> R) -> R {
        let mut arena = self.arena.borrow_mut();
        f(&mut arena.modules[id])
    }
}

impl Default for Context {
    fn default() -> Self {
        Self::create()
    }
}

impl std::fmt::Debug for Context {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Context").finish_non_exhaustive()
    }
}
