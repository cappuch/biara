//! Reusable interprocedural layout facts, extracted from the C renderer.
//!
//! The native analysis (`cpp/layouts.h`, `cpp/layouts.cpp`, formerly the
//! renderer-local `Types` in `cpp/cgen.cpp`) runs once per transformed
//! module and is cached in `TransformedC` ahead of the LLVM module. The C
//! renderer consumes that same cached analysis; it establishes no second
//! type convention. [`TransformedC::layout_facts`](crate::TransformedC::layout_facts)
//! returns these owned facts, and `--emit types` serializes them.
//!
//! What the facts mean (consumers rely on this; keep it):
//! - Observed access shapes only. [`SharedLayout::size`] is the accessed
//!   extent (last field end), never a declared object bound: audit object
//!   bounds must not consume it.
//! - Pointer classes deliberately do not meet across calls. Cross-call
//!   propagation is directional and context-sensitive (offset-aware
//!   actual-to-formal constraints for arguments, returns, and pointer
//!   fields, bounded fixpoint), so a generic callee called with unrelated
//!   objects keeps separate layouts while one object observed through two
//!   callees accumulates combined reusable fields under one shared struct.
//! - Overlapping incompatible accesses are explicit [`LayoutConflict`]
//!   facts; the renderer leaves those to casts, never asserted types.
//! - [`FactStatus`] and [`LayoutLimits`] report every materially dropped
//!   fact (field-offset cap, unsupported/nonconstant address relationships,
//!   overflowing offset translation, fixpoint truncation): a `Complete`
//!   status is never claimed when evidence was discarded.
//! - Layout ids are 0-based indices into [`LayoutFacts::layouts`], assigned
//!   in deterministic emission order (by struct name). Attaching debug
//!   names renames display fields only; ids, sizes, offsets, and bindings
//!   never move. `ArgLayout.arg` is the LLVM parameter index (position `i`
//!   describes LLVM parameter `i`, including any hidden-sret pointer and
//!   stack slots); `function` is the guest entry address.
//!
//! `Handle` is a private RAII wrapper around the opaque native analysis:
//! build once per module (CPU offsets arrive env-relative; the margin is
//! added here, matching the renderer's `MARGIN`), apply debug metadata,
//! export facts, expose the raw pointer for the cached emitter. The native
//! side holds LLVM references, so the parent stores the `Handle` ahead of
//! the module and drops it first. Facts are small owned serde data:
//! layouts, bindings, conflicts, status, limits — no per-instruction data.

use std::{
    ffi::{CString, c_char, c_void},
    os::raw::c_int,
};

use biara_ir_compat::module::Module;


/// One observed field of a shared layout: `bytes` accessed at `off`, what
/// it points to, and its display name. `asserted` carries a user-declared
/// C type when a struct binding bound this field (`""`/missing means
/// inferred only): asserted vs inferred provenance for consumers.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LayoutField {
    pub off: i64,
    pub bytes: u32,
    pub pointee: Option<u32>,
    pub name: String,
    /// User-asserted C type (`""` when inferred only).
    #[serde(default)]
    pub asserted: String,
}

/// One reusable inferred layout: fields at the offsets accessed, widest
/// access wins, plus the guest entries whose evidence converged here.
/// `size` is the accessed extent, not a declared object bound.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SharedLayout {
    pub id: u32,
    pub name: String,
    pub size: u64,
    pub scalar: bool,
    pub fields: Vec<LayoutField>,
    pub functions: Vec<u64>,
}

/// An embedded struct inside a frame layout.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Embedded {
    pub off: i64,
    pub layout: u32,
}

/// One frame (stack alloca or CPU-state global) layout: scalar fields plus
/// structs embedded where a callee layout was placed.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct FrameLayout {
    pub function: u64,
    pub name: String,
    pub layout: String,
    pub size: u64,
    pub fields: Vec<LayoutField>,
    pub embedded: Vec<Embedded>,
}

/// Argument binding: LLVM parameter `arg` of guest entry `function` carries
/// the bound layout (`None`: no layout established), observed at `offset`.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ArgLayout {
    pub function: u64,
    pub arg: u32,
    pub layout: Option<u32>,
    pub offset: i64,
}

/// Return binding: every return path of guest entry `function` aliases this
/// layout (`None`: absent or incompatible layout evidence across paths).
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ReturnLayout {
    pub function: u64,
    pub layout: Option<u32>,
}

/// Overlapping incompatible accesses at one offset of one layout: the
/// observed widths that could not both be asserted. The renderer uses
/// casts there.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LayoutConflict {
    pub layout: Option<u32>,
    pub off: i64,
    pub widths: Vec<u32>,
    pub detail: String,
}

/// Bounded-analysis status: `Complete` only when no evidence was dropped;
/// otherwise how many propagation rounds ran against their cap.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FactStatus {
    Complete,
    Truncated { iterations: u32, cap: u32 },
}

/// What the bounded analysis dropped instead of asserting.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LayoutLimits {
    pub iterations: u32,
    pub iteration_cap: u32,
    pub max_field_offset: i64,
    pub capped_offsets: u32,
    pub unsupported_bases: u32,
    pub overflowed_offsets: u32,
}

/// Owned reusable layout evidence for one transformed module.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LayoutFacts {
    pub status: FactStatus,
    pub layouts: Vec<SharedLayout>,
    pub frames: Vec<FrameLayout>,
    pub args: Vec<ArgLayout>,
    pub returns: Vec<ReturnLayout>,
    pub conflicts: Vec<LayoutConflict>,
    pub limits: LayoutLimits,
}

/// Analyze an ABI-form module and return module-independent layout evidence.
/// CPU fields use env-relative offsets; `margin` locates env within CPU state.
/// For repeated export/render operations, use [`crate::TransformedC`] instead,
/// which retains one cached analysis.
pub fn analyze_layouts(
    module: &Module<'_>,
    cpu: &[(i64, u32, String)],
    margin: u64,
) -> Result<LayoutFacts, crate::Error> {
    Handle::build(module, cpu, margin)?.facts()
}


/// One cached layout analysis for a transformed module.
///
/// Built once per module; `facts` exports the same JSON the renderer and the
/// signature summary consume.
pub(crate) struct Handle {
    analysis: crate::layouts_core::Analysis,
}

impl Handle {
    /// Build and cache the layout analysis for `module`. `cpu` carries
    /// env-relative `(offset, bytes, name)` descriptors; `margin` (the
    /// renderer's `MARGIN`) is added here so the analysis's offsets match
    /// emission.
    pub(crate) fn build(
        module: &Module<'_>,
        cpu: &[(i64, u32, String)],
        margin: u64,
    ) -> Result<Self, crate::Error> {
        // The analysis takes owned IR, so a borrowed module is enough.
        let ir = module.ir();
        let fields: Vec<(i64, u32, String)> = cpu
            .iter()
            .map(|(off, bytes, name)| (margin as i64 + off, *bytes, name.clone()))
            .collect();
        Ok(Handle {
            analysis: crate::layouts_core::Analysis::build(&ir, &fields),
        })
    }

    pub(crate) fn apply_debug(&mut self, module: &Module<'_>) -> Result<(), crate::Error> {
        // Debug metadata is not modelled by the IR, so this renames nothing;
        // structural facts (ids, sizes, offsets, bindings) never move.
        let ir = module.ir();
        self.analysis.apply_debug(&ir);
        Ok(())
    }

    /// Apply user-declared struct bindings to referenced layout ids.
    pub(crate) fn bind_structs(
        &mut self,
        classes: &[i32],
        names: &[String],
        structs: &[Vec<(i64, u32, String, String)>],
    ) -> Result<(), crate::Error> {
        let binds: Vec<(i64, String, Vec<crate::layouts_core::Field>)> = classes
            .iter()
            .zip(names)
            .zip(structs)
            .map(|((&class, name), fields)| {
                (
                    class as i64,
                    name.clone(),
                    fields
                        .iter()
                        .map(|(off, bytes, fname, _ty)| crate::layouts_core::Field {
                            off: *off,
                            bytes: *bytes,
                            name: fname.clone(),
                        })
                        .collect(),
                )
            })
            .collect();
        self.analysis
            .bind_structs(&binds)
            .map_err(crate::Error::Llvm)
    }

    /// Bind user-declared structs by name to every referenced layout id.
    pub(crate) fn bind_by_name(
        &mut self,
        binds: &[(String, biara_proto::annotations::StructAnnotation)],
        referenced: &std::collections::BTreeSet<u32>,
    ) -> Result<(), crate::Error> {
        if binds.is_empty() {
            return Ok(());
        }
        // Expand (struct, referenced ids) into (class, struct) pairs; sort
        // and dedup by the rendered key so repeated runs bind identically.
        let mut native: Vec<(i32, String, biara_proto::annotations::StructAnnotation)> = Vec::new();
        for (name, decl) in binds {
            for id in referenced {
                native.push((*id as i32, name.clone(), decl.clone()));
            }
        }
        native.sort_by_key(|(id, name, decl)| {
            (
                *id,
                name.clone(),
                serde_json::to_string(decl).unwrap_or_default(),
            )
        });
        native.dedup_by(|a, b| {
            a.0 == b.0
                && a.1 == b.1
                && serde_json::to_string(&a.2).unwrap_or_default()
                    == serde_json::to_string(&b.2).unwrap_or_default()
        });
        let binds: Vec<(i64, String, Vec<crate::layouts_core::Field>)> = native
            .iter()
            .map(|(class, name, decl)| {
                (
                    *class as i64,
                    name.clone(),
                    decl.fields
                        .iter()
                        .map(|field| crate::layouts_core::Field {
                            off: field.offset,
                            bytes: field.bytes,
                            name: field.name.clone(),
                        })
                        .collect(),
                )
            })
            .collect();
        self.analysis
            .bind_structs(&binds)
            .map_err(crate::Error::Llvm)
    }

    /// The analysis, for the renderer.
    pub(crate) fn analysis(&self) -> &crate::layouts_core::Analysis {
        &self.analysis
    }

    /// Raw handle for the native renderer. The renderer is being ported to
    /// Rust; until that lands this names the analysis so the call site keeps
    /// a stable shape.
    pub(crate) fn as_ptr(&self) -> *mut std::ffi::c_void {
        &self.analysis as *const crate::layouts_core::Analysis
            as *const std::ffi::c_void as *mut std::ffi::c_void
    }

    /// Export owned facts: layouts, frames, bindings, conflicts, status,
    /// and limits.
    pub(crate) fn facts(&self) -> Result<LayoutFacts, crate::Error> {
        serde_json::from_str(&self.analysis.json())
            .map_err(|e| crate::Error::Llvm(format!("layout facts: {e}")))
    }
}
