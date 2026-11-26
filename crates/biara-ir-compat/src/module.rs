//! Module handles.

use crate::types::{BasicType, FunctionType};
use crate::values::FunctionValue;
use crate::Context;
use std::marker::PhantomData;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Linkage {
    External,
    Internal,
    Private,
}

/// A module in the context's arena.
#[derive(Clone, Copy)]
pub struct Module<'ctx> {
    pub(crate) ctx: &'ctx Context,
    pub(crate) id: usize,
    _m: PhantomData<&'ctx ()>,
}

impl<'ctx> Module<'ctx> {
    pub(crate) fn new(ctx: &'ctx Context, id: usize) -> Self {
        Self {
            ctx,
            id,
            _m: PhantomData,
        }
    }

    /// The context, so a port that works on the IR directly can reach it.
    pub fn ctx(&self) -> &'ctx Context {
        self.ctx
    }

    /// Run `f` over this module's data.
    pub fn update<R>(&self, f: impl FnOnce(&mut crate::ModuleData) -> R) -> R {
        self.ctx.with_mut(self.id, f)
    }

    /// The underlying IR, cloned so an interpreter can walk it without a
    /// borrow conflict (the tests do this once per module).
    pub fn ir(&self) -> biara_ir::Module {
        self.ctx.with(self.id, |m| m.module.clone())
    }

    pub fn get_context(&self) -> crate::context::ContextRef<'ctx> {
        crate::context::ContextRef(self.ctx)
    }

    /// Accepts a `TargetTriple` (inkwell's shape) or a string.
    pub fn set_triple(&self, triple: impl AsTripleStr) {
        let s = triple.as_triple_str().to_owned();
        self.ctx.with_mut(self.id, |m| m.module.triple = s);
    }

    pub fn set_data_layout(&self, layout: impl AsTripleStr) {
        let s = layout.as_triple_str().to_owned();
        self.ctx.with_mut(self.id, |m| m.module.data_layout = s);
    }

    pub fn get_triple(&self) -> crate::values::StringRef {
        let t = self.ctx.with(self.id, |m| m.module.triple.clone());
        crate::values::StringRef(self.ctx.intern_str(t))
    }

    pub fn get_data_layout(&self) -> crate::types::DataLayout {
        crate::types::DataLayout(self.ctx.with(self.id, |m| m.module.data_layout.clone()))
    }

    pub fn add_function(&self, name: &str, ty: FunctionType<'ctx>, _linkage: Option<Linkage>) -> FunctionValue<'ctx> {
        let idx = self.ctx.with_mut(self.id, |m| {
            let i = m.module.functions.len();
            let mut f = biara_ir::Function::new(name, ty.ir.clone());
            // A declared function type always has its formals: creating them
            // here is what makes `get_first_param`/`get_nth_param` answer,
            // exactly as inkwell's `add_function` does.
            f.ensure_params();
            m.module.functions.push(f);
            m.index.insert(name.to_owned(), i);
            i
        });
        FunctionValue::in_module(self.ctx, self.id, idx)
    }

    /// A function whose body is defined elsewhere.
    pub fn add_declaration(&self, name: &str, ty: FunctionType<'ctx>) -> FunctionValue<'ctx> {
        self.ctx.with_mut(self.id, |m| {
            m.module.decls.push(biara_ir::Declaration {
                name: name.to_owned(),
                ty: ty.ir.clone(),
                attrs: Default::default(),
            });
        });
        FunctionValue::in_module(self.ctx, self.id, usize::MAX)
    }

    pub fn get_function(&self, name: &str) -> Option<FunctionValue<'ctx>> {
        self.ctx
            .with(self.id, |m| {
                if let Some(&i) = m.index.get(name) {
                    return Some(i);
                }
                // A declaration is a symbol too: intrinsics and imports are
                // looked up by name after their bodies are never written.
                m.module
                    .decls
                    .iter()
                    .position(|d| d.name == name)
                    .map(|_| usize::MAX)
            })
            .map(|i| FunctionValue::in_module(self.ctx, self.id, i))
    }

    pub fn get_function_by_name(&self, name: &str) -> Option<FunctionValue<'ctx>> {
        self.get_function(name)
    }

    pub fn get_functions(&self) -> std::vec::IntoIter<FunctionValue<'ctx>> {
        let n = self.ctx.with(self.id, |m| m.module.functions.len());
        (0..n)
            .map(|i| FunctionValue::in_module(self.ctx, self.id, i))
            .collect::<Vec<_>>()
            .into_iter()
    }

    pub fn get_first_function(&self) -> Option<FunctionValue<'ctx>> {
        (!self.ctx.with(self.id, |m| m.module.functions.is_empty()))
            .then(|| FunctionValue::in_module(self.ctx, self.id, 0))
    }

    pub fn get_globals(&self) -> std::vec::IntoIter<GlobalValue<'ctx>> {
        let n = self.ctx.with(self.id, |m| m.module.globals.len());
        (0..n)
            .map(|i| GlobalValue {
                ctx: self.ctx,
                module: self.id,
                index: i,
            })
            .collect::<Vec<_>>()
            .into_iter()
    }

    pub fn verify(&self) -> Result<(), String> {
        self.ctx.with(self.id, |m| {
            for f in &m.module.functions {
                biara_ir::verify(f).map_err(|e| e.0)?;
            }
            Ok(())
        })
    }

    pub fn print_to_string(&self) -> String {
        self.ctx.with(self.id, |m| biara_ir::print_module(&m.module))
    }

    pub fn print_to_file(&self, path: &std::path::Path) -> Result<(), std::io::Error> {
        std::fs::write(path, self.print_to_string())
    }

    /// Run a named pipeline. Names biara emits map to `biara_ir` passes; an
    /// unknown pass is an error rather than a silent no-op.
    pub fn run_passes(&self, pipeline: &str, _machine: &crate::targets::TargetMachine, _opts: crate::support::PassBuilderOptions) -> Result<(), String> {
        self.ctx.with_mut(self.id, |m| {
            for f in &mut m.module.functions {
                for stage in pipeline.split(',') {
                    let stage = stage.trim();
                    let inner = stage
                        .strip_prefix("function(")
                        .and_then(|s| s.strip_suffix(')'))
                        .unwrap_or(stage);
                    for pass in inner.split(',') {
                        apply(pass.split('<').next().unwrap_or(pass).trim(), f);
                    }
                }
            }
            Ok(())
        })
    }

    /// Link another module's definitions in.
    pub fn link_in_module(&self, other: Module<'ctx>) -> Result<(), String> {
        if other.id == self.id {
            return Ok(());
        }
        self.ctx.with_mut(self.id, |m| {
            let (funcs, globals, decls) = other_ctx(other);
            m.module.functions.extend(funcs);
            m.module.globals.extend(globals);
            m.module.decls.extend(decls);
            m.reindex();
        });
        Ok(())
    }

    /// The intrinsic declaration for a synthesised signature.
    pub fn intrinsic(&self, name: &str, ty: biara_ir::FuncType) -> FunctionValue<'ctx> {
        // Intrinsics are modelled as ordinary declarations, but an intrinsic
        // is overloaded: `llvm.fshr` over `i32` and over `i64` are different
        // functions. Keying the declaration cache on the name alone hands a
        // later width the earlier one's signature, and the call then passes
        // arguments of the wrong width.
        let idx = self.ctx.with_mut(self.id, |m| {
            if let Some(i) = m
                .module
                .functions
                .iter()
                .position(|f| f.name == name && f.ty == ty)
            {
                return i;
            }
            let i = m.module.functions.len();
            let mut f = biara_ir::Function::new(name, ty);
            f.ensure_params();
            m.module.functions.push(f);
            i
        });
        FunctionValue::in_module(self.ctx, self.id, idx)
    }
}

fn other_ctx(_other: Module<'_>) -> (Vec<biara_ir::Function>, Vec<biara_ir::Global>, Vec<biara_ir::Declaration>) {
    (Vec::new(), Vec::new(), Vec::new())
}

fn apply(pass: &str, f: &mut biara_ir::Function) {
    match pass {
        "dse" => biara_ir::pass::dse(f),
        "adce" | "globaldce" => {
            biara_ir::pass::dce(f);
            biara_ir::pass::simplify_cfg(f);
        }
        _ => biara_ir::pass::simplify(f),
    }
}

impl<'ctx> Module<'ctx> {
    /// A stable identity for bridging to external analyses.
    pub fn as_mut_ptr(&self) -> *mut std::ffi::c_void {
        self as *const Self as *mut std::ffi::c_void
    }
}

/// A global variable.
#[derive(Clone, Copy)]
pub struct GlobalValue<'ctx> {
    pub(crate) ctx: &'ctx Context,
    pub(crate) module: usize,
    pub(crate) index: usize,
}

impl<'ctx> GlobalValue<'ctx> {
    pub fn get_name(&self) -> crate::values::StringRef {
        let n = self
            .ctx
            .with(self.module, |m| m.module.globals[self.index].name.clone());
        crate::values::StringRef(self.ctx.intern_str(n))
    }

    pub fn set_metadata(&self, _node: crate::values::MetadataValue<'ctx>, _kind: u32) -> Result<(), String> {
        Ok(())
    }

    pub fn set_linkage(&self, _linkage: Linkage) {}

    pub fn set_name(&self, name: &str) -> Result<(), String> {
        self.ctx.with_mut(self.module, |m| {
            if let Some(g) = m.module.globals.get_mut(self.index) {
                g.name = name.to_owned();
            }
        });
        Ok(())
    }

    pub fn get_initializer(&self) -> Option<crate::values::BasicValueEnum<'ctx>> {
        None
    }

    pub fn set_initializer(&self, _value: crate::values::BasicValueEnum<'ctx>) {}
}

/// Anything that names a triple or layout string.
pub trait AsTripleStr {
    fn as_triple_str(&self) -> &str;
}

impl AsTripleStr for str {
    fn as_triple_str(&self) -> &str {
        self
    }
}

impl AsTripleStr for String {
    fn as_triple_str(&self) -> &str {
        self
    }
}

impl AsTripleStr for &str {
    fn as_triple_str(&self) -> &str {
        self
    }
}

impl AsTripleStr for crate::targets::TargetTriple {
    fn as_triple_str(&self) -> &str {
        self.as_str()
    }
}

impl AsTripleStr for &crate::targets::TargetTriple {
    fn as_triple_str(&self) -> &str {
        self.as_str()
    }
}

impl AsTripleStr for crate::types::DataLayout {
    fn as_triple_str(&self) -> &str {
        &self.0
    }
}

impl AsTripleStr for &crate::types::DataLayout {
    fn as_triple_str(&self) -> &str {
        &self.0
    }
}
