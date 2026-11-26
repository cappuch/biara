//! Interprocedural pointer-layout recovery, ported from `cpp/layouts.cpp`.
//!
//! The analysis runs once per transformed module over the `biara_ir` arena
//! (no LLVM types, no `unsafe`, no foreign calls) and produces the same
//! facts the native implementation produced for rendering, fact export and
//! the `Layouts` wrapper.
//!
//! ## Contract (kept from the native module doc, verbatim in spirit)
//!
//! - **Observed access shapes only.** Sizes are accessed extents, never
//!   declared object bounds; audit object bounds must not consume them.
//! - **Pointer classes deliberately do not meet across calls.** A generic
//!   callee taking any pointer must not merge every struct passed to it.
//!   Cross-call propagation is directional and context-sensitive instead:
//!   offset-aware actual-to-formal constraints for arguments, returns and
//!   pointer fields, refined to a bounded fixpoint. Unrelated actuals at one
//!   generic callee keep separate layouts; its formal receives only their
//!   common evidence. One actual accumulates evidence from all its callees.
//! - **Overlapping incompatible accesses are explicit conflict facts**,
//!   never silently asserted source types. The renderer falls back to casts
//!   on conflict, exactly as before extraction.
//! - **Frame embedding** (a callee layout placed into a caller frame slot)
//!   and **structural sharing** (equal layouts share one printed struct) are
//!   preserved byte-for-byte from the extracted implementation.
//! - **Bounded work**: the propagation fixpoint caps at 8 rounds; every
//!   dropped fact (field-offset cap, unsupported/nonconstant address
//!   relationships, overflowing offset translation) is reported through
//!   [`Analysis::limits`], never a misleading "complete" claim.
//! - **Stable identities**: layout ids are 0-based indices into the exported
//!   `layouts` array, assigned in deterministic emission order (struct name,
//!   then class id). Attaching debug names only renames display fields; ids,
//!   sizes, offsets and bindings never move.
//! - **Deterministic iteration only**: module/function/block/instruction
//!   order and ordered maps. Pointer-address ordering is never used, because
//!   there are no addresses here and there never were meaningful ones.
//!
//! ## Documented conservative fallbacks (evidence dropped, never invented)
//!
//! - `!biara.debug_name` / `!biara.debug_objects` are not modelled by
//!   [`biara_ir::FuncAttrs`], so [`Analysis::apply_debug`] cannot rename
//!   display fields: it is a no-op, exactly the behaviour of the native code
//!   when those attachments are absent (which is what the emitted module
//!   contains when no DWARF was supplied). Structural facts are untouched
//!   either way. The per-instruction guest PC needed by the access check
//!   exists ([`biara_ir::Function::pc_of`]); the declared frame objects do
//!   not.
//! - `or <base>, <const>` address arithmetic is only reduced for a zero
//!   constant. The native code required `isDisjoint()` on a possibly-disjoint
//!   `or`; that disjointness proof is not expressible over this IR, and
//!   assuming it would invent offsets. Unreduced `or` roots are recorded as
//!   `unsupported_bases`, so the status stays honest.
//! - Allocas carry no name in `biara_ir`, so the native `frame.<n>`
//!   entry-stack-pointer convention (and its `arg_`/`var_` slot naming)
//!   cannot be recovered: every frame reports no entry-SP base and slots are
//!   named by absolute offset. This affects display names only; embedding,
//!   sizes and offsets are unchanged.
//! - `biara.abi.args` / `biara.abi.rets` metadata lists are not modelled by
//!   [`biara_ir::FuncAttrs`]; [`u32_metadata`] therefore returns `None` and
//!   documents the gap. `signatures.rs` reads those operands through the
//!   native accessor today and does not need this helper.

use std::collections::{BTreeMap, BTreeSet};

use biara_ir::{
    ty_bytes, Constant, Function, Inst, Module, Terminator, Type, Value, ValueId,
};

/// Largest offset a struct field is recovered at; larger offsets index
/// arrays (or are absolute addresses) and stay casts.
pub const MAX_FIELD_OFFSET: i64 = 1 << 12;
/// Rounds of cross-call constraint propagation (bounded fixpoint).
pub const PROPAGATE_ROUNDS: u32 = 8;

/// One observed field of a recovered layout: `bytes` accessed at `off`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Field {
    pub off: i64,
    pub bytes: u32,
    pub name: String,
}

/// One recovered struct layout. `bytes` is the accessed extent (last field
/// end), not a declared object bound.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    pub bytes: u64,
    pub fields: Vec<Field>,
    pub name: String,
}

/// Exported layout facts.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Layouts {
    pub layouts: Vec<Layout>,
    /// `(class id, layout id)` for each pointer class that became a struct.
    pub bindings: Vec<(i64, usize)>,
    /// Human-readable notes on bounded work and explicit conflicts.
    pub limits: Vec<String>,
}

/// Function entry address from `FuncAttrs::entry_addr` (`!biara.entry`).
pub fn entry_of(f: &Function) -> Option<u64> {
    f.attrs.entry_addr
}

/// A `u32` metadata item by index from `FuncAttrs` (`biara.abi.args` etc.).
///
/// The native `biara_layouts_u32_metadata` read an `MDNode` operand list.
/// [`biara_ir::FuncAttrs`] models no such lists, so this returns `None` for
/// every key and index: the evidence is not carried by this IR. Callers that
/// need it (the signature exporter) read the native accessor while the ABI
/// form still exists; nothing in this module depends on it.
pub fn u32_metadata(_f: &Function, _key: &str, _index: u32) -> Option<u32> {
    None
}

// ---------------------------------------------------------------------------
// Small helpers (ported from the C++ anonymous helpers).
// ---------------------------------------------------------------------------

fn json_str(s: &str) -> String {
    let mut r = String::with_capacity(s.len() + 2);
    r.push('"');
    for b in s.bytes() {
        match b {
            b'"' => r.push_str("\\\""),
            b'\\' => r.push_str("\\\\"),
            0x08 => r.push_str("\\b"),
            0x0c => r.push_str("\\f"),
            b'\n' => r.push_str("\\n"),
            b'\r' => r.push_str("\\r"),
            b'\t' => r.push_str("\\t"),
            c if c < 0x20 => {
                r.push_str("\\u00");
                r.push_str(&format!("{c:02x}"));
            }
            c => r.push(c as char),
        }
    }
    r.push('"');
    r
}

fn is_keyword(s: &str) -> bool {
    matches!(
        s,
        "auto"
            | "break"
            | "case"
            | "char"
            | "const"
            | "continue"
            | "default"
            | "do"
            | "double"
            | "else"
            | "enum"
            | "extern"
            | "float"
            | "for"
            | "goto"
            | "if"
            | "inline"
            | "int"
            | "long"
            | "register"
            | "restrict"
            | "return"
            | "short"
            | "signed"
            | "sizeof"
            | "static"
            | "struct"
            | "switch"
            | "typedef"
            | "union"
            | "unsigned"
            | "void"
            | "volatile"
            | "while"
            | "bool"
            | "true"
            | "false"
            | "biara_cpu"
    )
}

fn ident(s: &str) -> String {
    let mut r: String = s
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if r.is_empty() || r.as_bytes()[0].is_ascii_digit() {
        r.insert(0, '_');
    }
    if is_keyword(&r) {
        r.push('_');
    }
    r
}

/// Container width for a bit count: 1, 8, 16, 32, 64, 128, else 0.
fn container(bits: u32) -> u32 {
    if bits == 1 {
        return 1;
    }
    for w in [8u32, 16, 32, 64, 128] {
        if bits <= w {
            return w;
        }
    }
    0
}

fn uint_ty(bits: u32) -> &'static str {
    match container(bits) {
        1 => "bool",
        8 => "uint8_t",
        16 => "uint16_t",
        32 => "uint32_t",
        64 => "uint64_t",
        128 => "biara_u128",
        _ => "",
    }
}

fn hex(v: u64) -> String {
    format!("0x{v:x}")
}

fn field_name(off: i64) -> String {
    format!("f_{:x}", off as u64)
}

/// Frame slot display name: `arg_<n>` above the entry stack pointer,
/// `var_<n>` below it, plain offsets when the base is unknown.
fn slot_name(below: Option<i64>, off: i64) -> String {
    match below {
        None => field_name(off),
        Some(below) => {
            let rel = off - below;
            if rel < 0 {
                format!("var_{:x}", (-rel) as u64)
            } else {
                format!("arg_{:x}", rel as u64)
            }
        }
    }
}

/// Canonicalize a declared integer alias to its width/signedness-stable
/// spelling: `long` is 8 bytes on the host but may be 4 on the guest (Win64
/// LLP64), so printing the alias literally would move offsets. Width-implied
/// `intN_t`/`uintN_t` spell the same bytes everywhere; non-integer spellings
/// (float/double/bool/pointers/structs/enums) pass through verbatim.
fn canon_int(bytes: u32, asserted: &str) -> String {
    let flat: String = asserted
        .chars()
        .map(|c| c.to_ascii_lowercase())
        .collect();
    let words: Vec<&str> = flat
        .split([' ', '\t'])
        .filter(|w| !w.is_empty() && *w != "const" && *w != "volatile" && *w != "signed")
        .collect();
    let has = |w: &str| words.contains(&w);
    let unsigned = has("unsigned");
    let non_int = has("bool")
        || has("_bool")
        || has("float")
        || has("double")
        || asserted.contains('*')
        || asserted.contains("struct")
        || asserted.contains("union")
        || asserted.contains("enum");
    if non_int {
        return asserted.to_string();
    }
    let is_int = has("char")
        || has("short")
        || has("int")
        || has("long")
        || has("int8_t")
        || has("int16_t")
        || has("int32_t")
        || has("int64_t")
        || has("uint8_t")
        || has("uint16_t")
        || has("uint32_t")
        || has("uint64_t")
        || has("u8")
        || has("u16")
        || has("u32")
        || has("u64")
        || has("i8")
        || has("i16")
        || has("i32")
        || has("i64");
    if !is_int {
        return asserted.to_string();
    }
    let prefix = if unsigned { "uint" } else { "int" };
    match bytes {
        1 => format!("{prefix}8_t"),
        2 => format!("{prefix}16_t"),
        4 => format!("{prefix}32_t"),
        8 => format!("{prefix}64_t"),
        16 => if unsigned { "biara_u128" } else { "biara_i128" }.to_string(),
        _ => asserted.to_string(),
    }
}

// ---------------------------------------------------------------------------
// Analysis value model.
// ---------------------------------------------------------------------------

/// A value identity across the whole module: the arena value in its function.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct ValRef {
    func: u32,
    value: ValueId,
}

/// Root plus constant offset of an address (integer or pointer).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Base {
    /// `None`: an absolute address.
    root: Option<ValRef>,
    off: i64,
}

/// A frame root: a stack alloca or a module global, in module order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum FrameKey {
    Global(usize),
    Alloca(ValRef),
}

/// One internal field, with provenance the public export keeps private.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LField {
    off: i64,
    bytes: u32,
    pointee: i32,
    embed: i32,
    name: String,
    debug: bool,
    asserted: String,
}

impl LField {
    fn new(off: i64, bytes: u32) -> Self {
        Self {
            off,
            bytes,
            pointee: -1,
            embed: -1,
            name: String::new(),
            debug: false,
            asserted: String::new(),
        }
    }
}

/// One internal layout.
#[derive(Clone, Debug, Default)]
pub(crate) struct StructLayout {
    name: String,
    fields: Vec<LField>,
    /// Accessed extent, not a declared object bound.
    size: u64,
    /// A pointer to the one field at offset 0.
    scalar: bool,
    /// Frames: offset of the entry stack pointer.
    below: Option<i64>,
}

impl StructLayout {
    fn starts_at(&self, off: i64) -> Option<&LField> {
        // `fields_of` and `embed` keep offsets sorted and unique.
        self.fields.iter().find(|f| f.off == off)
    }

    fn at(&self, off: i64, bytes: u32) -> Option<&LField> {
        self.starts_at(off).filter(|f| f.bytes == bytes)
    }
}

/// Evidence dropped by the bounded analysis, reported rather than hidden.
#[derive(Clone, Debug, Default)]
struct Limits {
    iterations: u32,
    iteration_cap: u32,
    capped_offsets: u32,
    unsupported_bases: u32,
    overflowed_offsets: u32,
    fixpoint_truncated: bool,
}

impl Limits {
    fn new() -> Self {
        Self {
            iteration_cap: PROPAGATE_ROUNDS,
            ..Self::default()
        }
    }
}

/// Union-find of pointer classes without path compression: representatives
/// are identical to the compressing version, and read-only queries stay
/// borrow-friendly.
#[derive(Clone, Debug, Default)]
struct Uf {
    parent: Vec<usize>,
}

impl Uf {
    fn add(&mut self) -> usize {
        let i = self.parent.len();
        self.parent.push(i);
        i
    }

    fn rep(&self, mut x: usize) -> usize {
        while self.parent[x] != x {
            x = self.parent[x];
        }
        x
    }

    fn unite(&mut self, a: usize, b: usize) {
        let (a, b) = (self.rep(a), self.rep(b));
        if a != b {
            self.parent[a.max(b)] = a.min(b);
        }
    }
}

/// Export state, computed once in module/function/IR order. Names are read
/// live from the layouts at serialization time, so display renames and user
/// bindings show up without moving ids.
#[derive(Clone, Debug, Default)]
struct Export {
    /// Classes in emission order; index is the layout id.
    order: Vec<usize>,
    /// Class representative -> layout id.
    id_of: BTreeMap<usize, usize>,
    args: Vec<Binding>,
    rets: Vec<Binding>,
    frames: Vec<FrameRef>,
}

#[derive(Clone, Debug)]
struct Binding {
    function: u64,
    /// Parameter index, or `u32::MAX` for a return.
    index: u32,
    offset: i64,
    class: Option<usize>,
}

#[derive(Clone, Debug)]
struct FrameRef {
    key: FrameKey,
    function: u64,
    name: String,
}

/// The interprocedural layout analysis for one module.
pub struct Analysis {
    var_of: BTreeMap<ValRef, usize>,
    uf: Uf,
    /// class -> offset -> widest bytes.
    acc: BTreeMap<usize, BTreeMap<i64, u32>>,
    /// Every observed width per (class, offset), including widths that lost
    /// widest-wins: conflict evidence, never narrowed.
    observed: BTreeMap<usize, BTreeMap<i64, BTreeSet<u32>>>,
    acc_sizes: BTreeMap<usize, BTreeSet<u32>>,
    nonzero: BTreeMap<usize, bool>,
    field_ptr: BTreeMap<(usize, i64), usize>,
    /// Shared layouts by class representative.
    layouts: BTreeMap<usize, StructLayout>,
    /// Frame layouts in `frame_order` order.
    frames: Vec<(FrameKey, StructLayout)>,
    frame_order: BTreeMap<FrameKey, u32>,
    dropped: Limits,
    export: Export,
    cpu: Vec<(i64, u32, String)>,
    /// Binds that landed on no covered layout, reported through
    /// [`Layouts::limits`] instead of disappearing.
    skipped_binds: Vec<String>,
}

impl Analysis {
    /// Build from the module plus the named CPU-state fields
    /// (`offset`, `bytes`, `name`), which arrive already margin-adjusted.
    pub fn build(module: &Module, cpu: &[(i64, u32, String)]) -> Self {
        let mut a = Self {
            var_of: BTreeMap::new(),
            uf: Uf::default(),
            acc: BTreeMap::new(),
            observed: BTreeMap::new(),
            acc_sizes: BTreeMap::new(),
            nonzero: BTreeMap::new(),
            field_ptr: BTreeMap::new(),
            layouts: BTreeMap::new(),
            frames: Vec::new(),
            frame_order: BTreeMap::new(),
            dropped: Limits::new(),
            export: Export::default(),
            cpu: cpu.to_vec(),
            skipped_binds: Vec::new(),
        };
        a.run(module);
        a
    }

    /// Recompute after debug metadata changes (display names only).
    ///
    /// `!biara.debug_name` / `!biara.debug_objects` are not modelled by the
    /// IR, so no rename can be justified: this is a no-op, which is exactly
    /// the native behaviour when those attachments are absent. Structural
    /// facts (ids, sizes, offsets, bindings) are untouched by construction.
    pub fn apply_debug(&mut self, _module: &Module) {}

    /// Bind named structs to class ids.
    ///
    /// Applied per bound lane only (never by shape matching across unrelated
    /// layouts): the class takes the declared struct name, covered fields
    /// take declared names, declared padding gaps are preserved, and extra
    /// declared fields beyond the recovered extent ride along as asserted
    /// members. As in the native code, a bind whose class is unknown, scalar,
    /// or not covered by the declared fields leaves the layout untouched (the
    /// caller validates widths first); such a skip is recorded in
    /// [`Layouts::limits`] rather than silently swallowed, and never changes
    /// a layout that the native code would also have skipped.
    ///
    /// The native binding also carried a declared C type per field; this
    /// API's [`Field`] has no type slot, so inferred fields keep the
    /// width-implied integer spelling in [`Analysis::field_ty`].
    pub fn bind_structs(&mut self, binds: &[(i64, String, Vec<Field>)]) -> Result<(), String> {
        for (class, name, fields) in binds {
            let rep = if *class < 0 {
                None
            } else {
                let class = *class as usize;
                (class < self.uf.parent.len()).then(|| self.uf.rep(class))
            };
            let Some(rep) = rep.filter(|r| {
                self.layouts.get(r).is_some_and(|l| {
                    !l.scalar
                        && !l.fields.is_empty()
                        && l.fields.iter().all(|f| {
                            fields.iter().any(|d| d.off == f.off && d.bytes == f.bytes)
                        })
                })
            }) else {
                self.skipped_binds
                    .push(format!("struct {name} over class {class}: no covered layout"));
                continue;
            };
            let l = self.layouts.get_mut(&rep).expect("checked above");
            l.name = format!("struct {name}");
            let mut merged: BTreeMap<i64, LField> =
                l.fields.iter().map(|f| (f.off, f.clone())).collect();
            let mut size = l.size;
            for d in fields {
                match merged.get_mut(&d.off) {
                    Some(f) if f.bytes == d.bytes => {
                        f.name = d.name.clone();
                        f.debug = true;
                    }
                    Some(_) => {}
                    None => {
                        let mut extra = LField::new(d.off, d.bytes);
                        extra.name = d.name.clone();
                        extra.debug = true;
                        merged.insert(d.off, extra);
                        size = size.max((d.off + d.bytes as i64).max(0) as u64);
                    }
                }
            }
            l.fields = merged.into_values().collect();
            l.size = size;
        }
        Ok(())
    }

    /// JSON export, matching the native `exportJson` shape.
    pub fn json(&self) -> String {
        let d = &self.dropped;
        let complete = !d.fixpoint_truncated
            && d.capped_offsets == 0
            && d.unsupported_bases == 0
            && d.overflowed_offsets == 0;
        let mut s = String::from("{\"status\":");
        if complete {
            s.push_str("\"complete\"");
        } else {
            s.push_str(&format!(
                "{{\"truncated\":{{\"iterations\":{},\"cap\":{}}}}}",
                d.iterations, d.iteration_cap
            ));
        }
        s.push_str(",\"layouts\":[");
        let mut first = true;
        for (id, &class) in self.export.order.iter().enumerate() {
            let Some(l) = self.layouts.get(&class) else {
                continue;
            };
            if !first {
                s.push(',');
            }
            first = false;
            s.push_str(&format!(
                "{{\"id\":{id},\"name\":{},\"size\":{},\"scalar\":{},\"fields\":[",
                json_str(&l.name),
                l.size,
                if l.scalar { "true" } else { "false" }
            ));
            let mut ff = true;
            for f in &l.fields {
                if !ff {
                    s.push(',');
                }
                ff = false;
                let pointee = if f.pointee >= 0 {
                    self.id_of_class(f.pointee as usize)
                } else {
                    None
                };
                s.push_str(&format!(
                    "{{\"off\":{},\"bytes\":{},\"pointee\":{},\"name\":{},\"asserted\":{}}}",
                    f.off,
                    f.bytes,
                    pointee.map_or_else(|| "null".to_string(), |p| p.to_string()),
                    json_str(&f.name),
                    json_str(&f.asserted)
                ));
            }
            s.push_str("],\"functions\":[");
            let mut gf = true;
            for b in &self.export.args {
                let Some(c) = b.class else { continue };
                if self.id_of_class(c) == Some(id) {
                    if !gf {
                        s.push(',');
                    }
                    gf = false;
                    s.push_str(&b.function.to_string());
                }
            }
            s.push_str("]}");
        }
        s.push_str("],\"frames\":[");
        let mut first = true;
        for fr in &self.export.frames {
            let Some((_, l)) = self.frames.iter().find(|(k, _)| *k == fr.key) else {
                continue;
            };
            if !first {
                s.push(',');
            }
            first = false;
            s.push_str(&format!(
                "{{\"function\":{},\"name\":{},\"layout\":{},\"size\":{},\"fields\":[",
                fr.function,
                json_str(&fr.name),
                json_str(&l.name),
                l.size
            ));
            let mut ff = true;
            for f in &l.fields {
                if !ff {
                    s.push(',');
                }
                ff = false;
                let pointee = if f.pointee >= 0 {
                    self.id_of_class(f.pointee as usize)
                } else {
                    None
                };
                s.push_str(&format!(
                    "{{\"off\":{},\"bytes\":{},\"pointee\":{},\"name\":{}}}",
                    f.off,
                    f.bytes,
                    pointee.map_or_else(|| "null".to_string(), |p| p.to_string()),
                    json_str(&f.name)
                ));
            }
            s.push_str("],\"embedded\":[");
            let mut ef = true;
            for f in &l.fields {
                if f.embed < 0 {
                    continue;
                }
                if !ef {
                    s.push(',');
                }
                ef = false;
                s.push_str(&format!(
                    "{{\"off\":{},\"layout\":{}}}",
                    f.off,
                    self.id_of_class(f.embed as usize)
                        .map_or_else(|| "null".to_string(), |p| p.to_string())
                ));
            }
            s.push_str("]}");
        }
        s.push_str("],\"args\":[");
        let mut first = true;
        for b in &self.export.args {
            if !first {
                s.push(',');
            }
            first = false;
            let id = b.class.and_then(|c| self.id_of_class(c));
            s.push_str(&format!(
                "{{\"function\":{},\"arg\":{},\"layout\":{},\"offset\":{}}}",
                b.function,
                b.index,
                id.map_or_else(|| "null".to_string(), |p| p.to_string()),
                b.offset
            ));
        }
        s.push_str("],\"returns\":[");
        let mut first = true;
        for b in &self.export.rets {
            if !first {
                s.push(',');
            }
            first = false;
            let id = b.class.and_then(|c| self.id_of_class(c));
            s.push_str(&format!(
                "{{\"function\":{},\"layout\":{}}}",
                b.function,
                id.map_or_else(|| "null".to_string(), |p| p.to_string())
            ));
        }
        s.push_str("],\"conflicts\":");
        s.push_str(&self.conflicts_json());
        s.push_str(&format!(
            ",\"limits\":{{\"iterations\":{},\"iteration_cap\":{},\"max_field_offset\":{},\
             \"capped_offsets\":{},\"unsupported_bases\":{},\"overflowed_offsets\":{}}}}}",
            d.iterations,
            d.iteration_cap,
            MAX_FIELD_OFFSET,
            d.capped_offsets,
            d.unsupported_bases,
            d.overflowed_offsets
        ));
        s
    }

    /// The exported facts in the compact owned shape consumers use.
    pub fn layouts(&self) -> Layouts {
        let mut layouts = Vec::with_capacity(self.export.order.len());
        for &class in &self.export.order {
            let Some(l) = self.layouts.get(&class) else {
                continue;
            };
            layouts.push(Layout {
                bytes: l.size,
                fields: l
                    .fields
                    .iter()
                    .map(|f| Field {
                        off: f.off,
                        bytes: f.bytes,
                        name: f.name.clone(),
                    })
                    .collect(),
                name: l.name.clone(),
            });
        }
        let bindings: Vec<(i64, usize)> = self
            .layouts
            .keys()
            .filter_map(|&c| self.id_of_class(c).map(|id| (c as i64, id)))
            .collect();
        let mut limits = Vec::new();
        if self.dropped.fixpoint_truncated {
            limits.push(format!(
                "cross-call propagation truncated after {} of {} rounds",
                self.dropped.iterations, self.dropped.iteration_cap
            ));
        } else {
            limits.push(format!(
                "cross-call propagation converged within {} of {} rounds",
                self.dropped.iterations, self.dropped.iteration_cap
            ));
        }
        if self.dropped.capped_offsets > 0 {
            limits.push(format!(
                "{} access(es) at or beyond the field-offset cap {MAX_FIELD_OFFSET}",
                self.dropped.capped_offsets
            ));
        }
        if self.dropped.unsupported_bases > 0 {
            limits.push(format!(
                "{} nonconstant or conflicting address relationship(s) dropped",
                self.dropped.unsupported_bases
            ));
        }
        if self.dropped.overflowed_offsets > 0 {
            limits.push(format!(
                "{} offset translation(s) overflowed and were dropped",
                self.dropped.overflowed_offsets
            ));
        }
        limits.extend(self.skipped_binds.iter().cloned());
        for (id, off, widths) in self.conflicts() {
            limits.push(format!(
                "layout {} offset {off}: overlapping widths {widths:?}; renderer uses casts",
                id.map_or_else(|| "?".to_string(), |i| i.to_string())
            ));
        }
        Layouts {
            layouts,
            bindings,
            limits,
        }
    }

    // -- queries used by the renderer (parent integration) --------------------

    /// C type of a pointer to class `c`'s layout.
    pub fn ptr_ty(&self, c: usize) -> String {
        match self.layouts.get(&self.uf.rep(c)) {
            Some(l) if !l.scalar => format!("{} *", l.name),
            Some(l) => l
                .fields
                .first()
                .map(|f| format!("{} *", self.field_ty(f)))
                .unwrap_or_default(),
            None => String::new(),
        }
    }

    /// C type of one field.
    pub(crate) fn field_ty(&self, f: &LField) -> String {
        if !f.asserted.is_empty() {
            return canon_int(f.bytes, &f.asserted);
        }
        if f.embed >= 0 {
            if let Some(l) = self.layouts.get(&self.uf.rep(f.embed as usize)) {
                return l.name.clone();
            }
        }
        if f.pointee >= 0 {
            let p = self.uf.rep(f.pointee as usize);
            if let Some(pl) = self.layouts.get(&p) {
                // Scalar pointees of scalar pointers are kept out of recursion.
                let recurse = pl.scalar
                    && pl
                        .fields
                        .first()
                        .map(|pf| pf.pointee >= 0)
                        .unwrap_or(false);
                if !recurse {
                    return self.ptr_ty(p);
                }
            }
        }
        uint_ty(f.bytes.saturating_mul(8)).to_string()
    }

    // -- internals -----------------------------------------------------------

    fn class_of(&self, v: ValRef) -> Option<usize> {
        self.var_of.get(&v).map(|&x| self.uf.rep(x))
    }

    fn var(&mut self, v: ValRef) -> usize {
        if let Some(&x) = self.var_of.get(&v) {
            return x;
        }
        let x = self.uf.add();
        self.var_of.insert(v, x);
        x
    }

    fn id_of_class(&self, c: usize) -> Option<usize> {
        self.export.id_of.get(&self.uf.rep(c)).copied()
    }

    fn shared_id(&self, c: Option<usize>) -> Option<usize> {
        c.and_then(|c| self.id_of_class(c))
    }

    /// Conflicts: overlapping incompatible accesses at one offset.
    ///
    /// Two sources feed this, both explicit (the renderer leaves overlaps to
    /// casts):
    /// - `observed`: every width ever seen at (class, offset), including
    ///   widths that lost widest-wins. Two distinct widths at one offset are
    ///   a conflict even when one dominates the other.
    /// - overlapping ranges: an access starting strictly inside another
    ///   access with a different width, as the field builder resolved it by
    ///   first-come order rather than type assertion.
    fn conflicts(&self) -> Vec<(Option<usize>, i64, Vec<u32>)> {
        let mut out = Vec::new();
        let mut done: BTreeSet<(usize, i64)> = BTreeSet::new();
        for (&c, offs) in &self.observed {
            for (&off, widths) in offs {
                if widths.len() >= 2 {
                    out.push((self.id_of_class(c), off, widths.iter().copied().collect()));
                    done.insert((c, off));
                }
            }
        }
        for (&c, offs) in &self.acc {
            let mut prev_end: Option<i64> = None;
            let mut prev_off = i64::MIN;
            let mut prev_bytes = 0u32;
            for (&off, &bytes) in offs {
                if let Some(pe) = prev_end {
                    if off < pe
                        && off != prev_off
                        && !done.contains(&(c, off))
                        && !done.contains(&(c, prev_off))
                    {
                        out.push((self.id_of_class(c), off, vec![prev_bytes, bytes]));
                        done.insert((c, off));
                    }
                }
                let end = off.saturating_add(bytes as i64);
                if prev_end.is_none_or(|pe| end > pe) {
                    prev_off = off;
                    prev_bytes = bytes;
                    prev_end = Some(end);
                }
            }
        }
        // `emit` sorted and deduplicated before requiring two widths, so an
        // equal-width overlap is not a conflict.
        for c in &mut out {
            c.2.sort_unstable();
            c.2.dedup();
        }
        out.retain(|c| c.2.len() >= 2);
        out
    }

    fn conflicts_json(&self) -> String {
        let mut s = String::from("[");
        let mut first = true;
        for (id, off, mut widths) in self.conflicts() {
            widths.sort_unstable();
            widths.dedup();
            if !first {
                s.push(',');
            }
            first = false;
            s.push_str(&format!(
                "{{\"layout\":{},\"off\":{off},\"widths\":[",
                id.map_or_else(|| "null".to_string(), |i| i.to_string())
            ));
            for (k, w) in widths.iter().enumerate() {
                if k > 0 {
                    s.push(',');
                }
                s.push_str(&w.to_string());
            }
            s.push_str("],\"detail\":");
            s.push_str(&json_str("overlapping accesses at one offset; renderer uses casts"));
            s.push('}');
        }
        s.push(']');
        s
    }

    /// Frame root of a value, if it is a stack alloca or a module global.
    fn frame_key(&self, module: &Module, f: &Function, r: ValRef) -> Option<FrameKey> {
        match f.value(r.value) {
            Value::Inst(Inst::Alloca { .. }) => Some(FrameKey::Alloca(r)),
            Value::Const(Constant::GlobalPtr(name)) => module
                .globals
                .iter()
                .position(|g| &g.name == name)
                .map(FrameKey::Global),
            _ => None,
        }
    }

    /// The whole pipeline, in the native order.
    fn run(&mut self, module: &Module) {
        // Frame order follows module/global/instruction order, never
        // allocator addresses.
        for gi in 0..module.globals.len() {
            let n = self.frame_order.len() as u32;
            self.frame_order.insert(FrameKey::Global(gi), n);
        }
        for (fi, f) in module.functions.iter().enumerate() {
            for b in &f.blocks {
                for &id in &b.insts {
                    if matches!(f.inst(id), Some(Inst::Alloca { .. })) {
                        let n = self.frame_order.len() as u32;
                        self.frame_order.insert(
                            FrameKey::Alloca(ValRef {
                                func: fi as u32,
                                value: id,
                            }),
                            n,
                        );
                    }
                }
            }
        }

        let mut same: Vec<(ValRef, ValRef)> = Vec::new();
        let mut loaded: Vec<(Base, ValRef)> = Vec::new();
        let mut stored: Vec<(Base, ValRef)> = Vec::new();
        // (root, offset, bytes) accesses; root `None` is an absolute address.
        let mut accesses: Vec<(Option<ValRef>, i64, u32)> = Vec::new();
        let mut frame_passes: Vec<(FrameKey, i64, ValRef)> = Vec::new();
        let mut frame_stores: Vec<(Base, Base)> = Vec::new();
        let mut arg_passes: Vec<(ValRef, ValRef)> = Vec::new();
        // Direct calls: (actual base, formal parameter, call) for propagation.
        let mut call_args: Vec<(Base, ValRef, ValRef)> = Vec::new();
        // Direct-call returns feeding pointer variables: (call, user).
        let mut call_rets: Vec<(ValRef, ValRef)> = Vec::new();
        // Call results used directly by a `ret` (class seeding only).
        let mut ret_calls: Vec<ValRef> = Vec::new();
        // Callee return bases: function -> reduced bases of returned values.
        let mut ret_bases: BTreeMap<u32, Vec<Base>> = BTreeMap::new();

        for (fi, f) in module.functions.iter().enumerate() {
            let fi = fi as u32;
            frame_spill_aliases(f, fi, &mut same);
            let mut call_users = None;
            for b in &f.blocks {
                for &id in &b.insts {
                    let here = ValRef { func: fi, value: id };
                    let mut access = |ptr: ValueId, bytes: u32| -> Base {
                        let base = base(f, fi, ptr, 0);
                        accesses.push((base.root, base.off, bytes));
                        base
                    };
                    match f.value(id) {
                        Value::Inst(Inst::Load { ty, ptr, .. }) => {
                            // The load's own instruction is the loaded value.
                            loaded.push((access(*ptr, ty_bytes(ty)), here));
                        }
                        Value::Inst(Inst::Store { ty, value, ptr, .. }) => {
                            let dest = access(*ptr, ty_bytes(ty));
                            stored.push((dest, ValRef {
                                func: fi,
                                value: *value,
                            }));
                        }
                        Value::Inst(Inst::Phi { incoming, .. }) => {
                            for (v, _) in incoming {
                                let vb = base(f, fi, *v, 0);
                                if vb.off == 0 {
                                    if let Some(r) = vb.root {
                                        if pointer_root(f, r.value) {
                                            same.push((here, r));
                                        }
                                    }
                                }
                            }
                        }
                        Value::Inst(Inst::Select { yes, no, .. }) => {
                            for v in [*yes, *no] {
                                let vb = base(f, fi, v, 0);
                                if vb.off == 0 {
                                    if let Some(r) = vb.root {
                                        if pointer_root(f, r.value) {
                                            same.push((here, r));
                                        }
                                    }
                                }
                            }
                        }
                        Value::Inst(Inst::Call { args, .. }) => {
                            let Some(callee) = callee_index(module, f, id) else {
                                continue;
                            };
                            let callee_f = &module.functions[callee];
                            let n = args.len().min(callee_f.params.len());
                            for k in 0..n {
                                let actual = ValRef {
                                    func: fi,
                                    value: args[k],
                                };
                                let param = ValRef {
                                    func: callee as u32,
                                    value: callee_f.params[k],
                                };
                                arg_passes.push((actual, param));
                                let ab = base(f, fi, args[k], 0);
                                call_args.push((ab, param, here));
                            }
                            // Per-site result users: values derived from this
                            // site's result. Loads/stores/calls/returns that
                            // consume the result so the result's facts
                            // instantiate at this site only.
                            if pointer_root(f, id) {
                                let users = call_users.get_or_insert_with(|| {
                                    let mut users: BTreeMap<ValueId, Vec<ValueId>> = BTreeMap::new();
                                    for &user in f.blocks.iter().flat_map(|b| &b.insts) {
                                        let Some(inst) = f.inst(user) else { continue };
                                        for operand in biara_ir::operands(inst) {
                                            if matches!(f.inst(operand), Some(Inst::Call { .. })) {
                                                let list = users.entry(operand).or_default();
                                                if list.last() != Some(&user) {
                                                    list.push(user);
                                                }
                                            }
                                        }
                                    }
                                    users
                                });
                                if let Some(users) = users.get(&id) {
                                    call_rets.extend(users.iter().map(|&value| {
                                        (here, ValRef { func: fi, value })
                                    }));
                                }
                            }
                        }
                        _ => {}
                    }
                }
                if let Some(Terminator::Ret(Some(v))) = &b.term {
                    ret_bases
                        .entry(fi)
                        .or_default()
                        .push(base(f, fi, *v, 0));
                    // `ret <callResult>` is a use of the call in the native
                    // user list; it never instantiates result facts (a return
                    // is neither a load nor a join), but it does seed the
                    // call's class, which `ret_calls` records.
                    if pointer_root(f, *v) {
                        if let Some(Inst::Call { .. }) = f.inst(*v) {
                            ret_calls.push(ValRef { func: fi, value: *v });
                        }
                    }
                }
            }
        }

        // Frame slots that only ever hold one frame address (spills at -O0):
        // a load from one is that address, for typing.
        let mut holds: BTreeMap<(ValRef, i64), Option<Base>> = BTreeMap::new();
        for (dest, v) in &stored {
            let Some(root) = dest.root else { continue };
            if !matches!(f_at(module, root), Some(Inst::Alloca { .. })) {
                continue;
            }
            let vb = base(f_of(module, root), root.func, v.value, 0);
            match holds.get_mut(&(root, dest.off)) {
                None => {
                    holds.insert((root, dest.off), Some(vb));
                }
                Some(slot) => {
                    if slot.is_none_or(|b| b.root != vb.root || b.off != vb.off) {
                        *slot = None;
                    }
                }
            }
        }

        for (dest, v) in &stored {
            let vb = type_base(module, v, &holds);
            if let Some(root) = vb.root {
                if matches!(f_at(module, root), Some(Inst::Alloca { .. })) {
                    frame_stores.push((*dest, vb));
                }
            }
        }
        for (arg, param) in &arg_passes {
            let b = type_base(module, arg, &holds);
            if let Some(root) = b.root {
                if matches!(f_at(module, root), Some(Inst::Alloca { .. })) {
                    frame_passes.push((FrameKey::Alloca(root), b.off, *param));
                }
            }
        }

        // Classes: roots accessed through, and what they are unified with.
        // Call actuals/formals get classes too, so offset-shifted facts have
        // somewhere to land even when the actual is never accessed directly.
        for (root, _, _) in &accesses {
            if let Some(r) = root {
                if !is_frame_value(module, *r) && is_pointerish(&f_of(module, *r).ty(r.value)) {
                    self.var(*r);
                }
            }
        }
        for (ab, param, _) in &call_args {
            if let Some(r) = ab.root {
                if pointer_root(f_of(module, r), r.value) {
                    self.var(r);
                }
            }
            if pointer_root(f_of(module, *param), param.value) {
                self.var(*param);
            }
        }
        for (cb, _) in &call_rets {
            if pointer_root(f_of(module, *cb), cb.value) {
                self.var(*cb);
            }
        }
        for cb in &ret_calls {
            if pointer_root(f_of(module, *cb), cb.value) {
                self.var(*cb);
            }
        }
        // Stored/loaded value roots get classes too: a child stored into a
        // container slot (or loaded from one) must have somewhere for pointee
        // facts to land even when it is never addressed directly.
        for (_, v) in &loaded {
            let vb = base(f_of(module, *v), v.func, v.value, 0);
            if vb.off == 0 {
                if let Some(r) = vb.root {
                    if pointer_root(f_of(module, r), r.value) {
                        self.var(r);
                    }
                }
            }
        }
        for (_, v) in &stored {
            let vb = base(f_of(module, *v), v.func, v.value, 0);
            if vb.off == 0 {
                if let Some(r) = vb.root {
                    if pointer_root(f_of(module, r), r.value) {
                        self.var(r);
                    }
                }
            }
        }
        for (a, b) in &same {
            let va = self.var(*a);
            let vb = self.var(*b);
            self.uf.unite(va, vb);
        }

        // Pointer fields: loaded or stored values that are pointer variables.
        let mut changed = true;
        while changed {
            changed = false;
            self.field_ptr.clear();
            for (b, v) in loaded.iter().chain(stored.iter()) {
                if self.link_field(module, *b, *v) {
                    changed = true;
                }
            }
        }

        let mut frame_acc: BTreeMap<FrameKey, BTreeMap<i64, u32>> = BTreeMap::new();
        for (root, off, bytes) in &accesses {
            let off = *off;
            let bytes = *bytes;
            let Some(r) = *root else {
                // Absolute addresses: not fields, but not silently dropped.
                self.dropped.unsupported_bases += 1;
                continue;
            };
            let ty = f_of(module, r).ty(r.value);
            if is_frame_value(module, r) {
                let key = self
                    .frame_key(module, f_of(module, r), r)
                    .expect("frame value has a key");
                let w = frame_acc.entry(key).or_default().entry(off).or_insert(0);
                *w = (*w).max(bytes);
            } else if is_pointerish(&ty) {
                // Far offsets index arrays (or are absolute addresses), not
                // fields.
                let Some(class) = self.class_of(r) else {
                    self.dropped.unsupported_bases += 1;
                    continue;
                };
                if !(0..MAX_FIELD_OFFSET).contains(&off) {
                    self.dropped.capped_offsets += 1;
                    continue;
                }
                let w = self
                    .acc
                    .entry(class)
                    .or_default()
                    .entry(off)
                    .or_insert(0);
                *w = (*w).max(bytes);
                self.acc_sizes.entry(class).or_default().insert(bytes);
                *self.nonzero.entry(class).or_default() |= off != 0;
                self.observed
                    .entry(class)
                    .or_default()
                    .entry(off)
                    .or_default()
                    .insert(bytes);
            } else if !pointer_root(f_of(module, r), r.value) {
                // Nonconstant, non-rooted address relationships (computed
                // indices, unreduced operators): recorded, not fields.
                self.dropped.unsupported_bases += 1;
            }
        }

        // Cross-call constraint propagation over the seeded access sets.
        self.propagate(
            module,
            &call_args,
            &call_rets,
            &ret_bases,
            &loaded,
            &stored,
        );

        // Layouts from observed accesses (class ids in map order).
        let mut n = 0usize;
        let acc = self.acc.clone();
        for (&class, offs) in &acc {
            let mut l = StructLayout {
                fields: fields_of(offs, 0, i64::MAX),
                ..StructLayout::default()
            };
            if l.fields.is_empty() {
                continue;
            }
            n += 1;
            l.scalar = !*self.nonzero.get(&class).unwrap_or(&false)
                && self.acc_sizes.get(&class).is_some_and(|s| s.len() == 1);
            l.name = format!("struct s{n}");
            let ptr_of = |off: i64| self.field_ptr.get(&(class, off)).copied();
            for f in &mut l.fields {
                f.name = field_name(f.off);
                if f.bytes == 8 {
                    if let Some(p) = ptr_of(f.off) {
                        f.pointee = self.uf.rep(p) as i32;
                    }
                }
                l.size = (f.off + f.bytes as i64).max(0) as u64;
            }
            self.layouts.insert(class, l);
        }
        // Classes do not meet across calls; instead, layouts that come out
        // the same (fields and what they point to, recursively) share one
        // struct.
        self.share();

        // Pointees without layouts of their own stay integers.
        let known: BTreeSet<usize> = self.layouts.keys().copied().collect();
        for l in self.layouts.values_mut() {
            for f in &mut l.fields {
                if f.pointee >= 0 && !known.contains(&(f.pointee as usize)) {
                    f.pointee = -1;
                }
            }
        }

        // Frame layouts.
        let mut frames: Vec<(FrameKey, StructLayout)> = Vec::new();
        for (key, offs) in frame_acc {
            let l = match key {
                FrameKey::Alloca(r) => {
                    let f = f_of(module, r);
                    let size = match f.inst(r.value) {
                        Some(Inst::Alloca { ty }) => ty_bytes(ty) as u64,
                        _ => 0,
                    };
                    let mut l = StructLayout {
                        fields: fields_of(&offs, 0, size as i64),
                        size,
                        name: format!(
                            "struct {}_{}",
                            ident(&f.name),
                            ident(&format!("t{}", r.value.0))
                        ),
                        // Allocas carry no name, so the `frame.<n>` entry-SP
                        // convention cannot be recovered (documented fallback).
                        below: None,
                        scalar: false,
                    };
                    for f in &mut l.fields {
                        f.name = field_name(f.off);
                    }
                    l
                }
                FrameKey::Global(gi) => {
                    let g = &module.globals[gi];
                    let size = ty_bytes(&g.ty) as u64;
                    let mut all = offs;
                    let mut names: BTreeMap<i64, String> = BTreeMap::new();
                    for (off, _, name) in &self.cpu {
                        if all.contains_key(off) {
                            names.insert(*off, name.clone());
                        }
                    }
                    let mut l = StructLayout {
                        size,
                        name: "struct cpu_state".to_string(),
                        fields: fields_of(&all, 0, size as i64),
                        below: None,
                        scalar: false,
                    };
                    all.clear();
                    let mut used: BTreeSet<String> = BTreeSet::new();
                    for f in &mut l.fields {
                        f.name = match names.get(&f.off) {
                            Some(n) => ident(n),
                            None => field_name(f.off),
                        };
                        // Globals can share a name (AArch64's two views of pc).
                        if !used.insert(f.name.clone()) {
                            f.name.push('_');
                            f.name.push_str(&hex(f.off as u64)[2..]);
                        }
                    }
                    l
                }
            };
            frames.push((key, l));
        }
        self.frames = frames;

        // A frame slot handed to a parameter whose pointees are a struct
        // holds that struct; so does one whose address is stored in a field
        // that points to a struct (until no more such slots turn up).
        for (key, off, param) in &frame_passes {
            if let Some(c) = self.class_of(*param) {
                self.embed(*key, *off, c);
            }
        }
        let mut changed = true;
        while changed {
            changed = false;
            for (dest, val) in &frame_stores {
                let p = self.pointee_at(module, *dest);
                if p >= 0 {
                    if let Some(root) = val.root {
                        changed |= self.embed(FrameKey::Alloca(root), val.off, p as usize);
                    }
                }
            }
        }

        self.compute_export(module);
    }

    /// One field-link step of the intra-function pointer-flow fixpoint.
    fn link_field(&mut self, module: &Module, b: Base, v: ValRef) -> bool {
        let Some(root) = b.root else { return false };
        if !self.var_of.contains_key(&root) {
            return false;
        }
        let vb = base(f_of(module, v), v.func, v.value, 0);
        let Some(vroot) = vb.root else { return false };
        if vb.off != 0 || !self.var_of.contains_key(&vroot) {
            return false;
        }
        let (Some(c), Some(p)) = (self.class_of(root), self.class_of(vroot)) else {
            return false;
        };
        match self.field_ptr.get(&(c, b.off)) {
            Some(&existing) if self.uf.rep(existing) != p => {
                self.uf.unite(existing, p);
                true
            }
            Some(_) => false,
            None => {
                self.field_ptr.insert((c, b.off), p);
                false
            }
        }
    }

    /// Directional, context-sensitive cross-call constraints. See the module
    /// doc: classes are never united across a call; four offset-aware fact
    /// families (arguments, returns, derived values, pointer fields) are
    /// propagated for at most [`PROPAGATE_ROUNDS`] rounds, and leftovers are
    /// reported, never asserted.
    #[allow(clippy::too_many_arguments)]
    fn propagate(
        &mut self,
        module: &Module,
        call_args: &[(Base, ValRef, ValRef)],
        call_rets: &[(ValRef, ValRef)],
        ret_bases: &BTreeMap<u32, Vec<Base>>,
        loaded: &[(Base, ValRef)],
        stored: &[(Base, ValRef)],
    ) {
        let mut xacc: BTreeMap<usize, BTreeMap<i64, u32>> = BTreeMap::new();
        // Bumped whenever a class's facts (widest of `acc` and `xacc` per
        // offset) grow. Merging `xacc` into `acc` leaves it unchanged.
        let mut version: BTreeMap<usize, u64> = BTreeMap::new();
        // The source version each (target, pointee) field flow last carried.
        let mut flowed: BTreeMap<(usize, usize), u64> = BTreeMap::new();

        // Classes are stable throughout propagation. Index each caller-side
        // pointer store once, in the original order for each slot.
        let mut stored_owners: BTreeMap<(usize, i64), Vec<usize>> = BTreeMap::new();
        for (slot, value) in stored {
            let Some(container) = slot.root.and_then(|root| self.class_of(root)) else {
                continue;
            };
            let value_base = base(f_of(module, *value), value.func, value.value, 0);
            if value_base.off == 0
                && let Some(owner) = value_base.root.and_then(|root| self.class_of(root))
            {
                stored_owners.entry((container, slot.off)).or_default().push(owner);
            }
        }

        for round in 0..PROPAGATE_ROUNDS {
            self.dropped.iterations = round;
            let mut widths_changed = false;
            let before: BTreeMap<usize, BTreeMap<i64, u32>> = xacc.clone();

            // Arguments, per call site: formal facts flow to this site's
            // actual at `Off + delta`, where delta is the actual's own base
            // offset (an offset pointer shifts).
            for (ab, param, _) in call_args {
                let (Some(aroot), true) = (ab.root, pointer_root(f_of(module, *param), param.value))
                else {
                    continue;
                };
                if !pointer_root(f_of(module, aroot), aroot.value)
                    || !self.var_of.contains_key(&aroot)
                    || !self.var_of.contains_key(param)
                {
                    continue;
                }
                let (Some(ac), Some(fc)) = (self.class_of(aroot), self.class_of(*param)) else {
                    continue;
                };
                if ac == fc {
                    continue;
                }
                if let Some(offs) = self.acc.get(&fc).cloned() {
                    let observed_fc = self.observed.get(&fc).cloned().unwrap_or_default();
                    for (off, bytes) in &offs {
                        let t = *off as i128 + ab.off as i128;
                        if !(0..=i64::MAX as i128).contains(&t) {
                            self.dropped.overflowed_offsets += 1;
                            continue;
                        }
                        let t = t as i64;
                        note(
                            &mut xacc,
                            &mut self.observed,
                            &mut version,
                            &mut self.dropped,
                            &mut widths_changed,
                            Some(ac),
                            t,
                            *bytes,
                        );
                        if let Some(widths) = observed_fc.get(off) {
                            let set = self
                                .observed
                                .entry(ac)
                                .or_default()
                                .entry(t)
                                .or_default();
                            for w in widths {
                                widths_changed |= set.insert(*w);
                            }
                        }
                    }
                }
                // A formal's own propagated facts flow onward, except facts
                // this same site just contributed (no self-amplification).
                if let Some(jt) = xacc.get(&fc).cloned() {
                    if fc != ac {
                        let mut novel: BTreeMap<i64, u32> = BTreeMap::new();
                        for (off, bytes) in &jt {
                            let t = *off as i128 + ab.off as i128;
                            if !(0..=i64::MAX as i128).contains(&t) {
                                self.dropped.overflowed_offsets += 1;
                                continue;
                            }
                            let d = t as i64;
                            let have = xacc
                                .get(&ac)
                                .and_then(|m| m.get(&d))
                                .copied()
                                .unwrap_or(0);
                            if *bytes > have {
                                novel.insert(d, *bytes);
                            }
                        }
                        for (d, bytes) in novel {
                            note(
                                &mut xacc,
                                &mut self.observed,
                                &mut version,
                                &mut self.dropped,
                                &mut widths_changed,
                                Some(ac),
                                d,
                                bytes,
                            );
                        }
                    }
                }
            }

            // Returns, per call site: the callee's returned roots instantiate
            // at this site's actuals only.
            {
                let mut by_site: BTreeMap<ValRef, Vec<(Base, ValRef)>> = BTreeMap::new();
                for (ab, param, cb) in call_args {
                    by_site.entry(*cb).or_default().push((*ab, *param));
                }
                for (cb, _) in call_rets {
                    let Some(callee) = self.callee_of(module, *cb) else {
                        continue;
                    };
                    let Some(rb) = ret_bases.get(&callee) else {
                        continue;
                    };
                    if !self.var_of.contains_key(cb) {
                        continue;
                    }
                    let Some(cc) = self.class_of(*cb) else { continue };
                    let Some(site) = by_site.get(cb) else { continue };
                    for ret in rb {
                        let Some(rroot) = ret.root else { continue };
                        if !matches!(
                            f_of(module, rroot).value(rroot.value),
                            Value::Param { .. }
                        ) {
                            continue;
                        }
                        for (ab, param) in site {
                            if *param != rroot {
                                continue;
                            }
                            let Some(aroot) = ab.root else { continue };
                            if !self.var_of.contains_key(&aroot)
                                || !pointer_root(f_of(module, aroot), aroot.value)
                            {
                                continue;
                            }
                            let Some(ac) = self.class_of(aroot) else { continue };
                            let delta = ab.off as i128 + ret.off as i128;
                            let rf = self.result_facts(&xacc, *cb);
                            for (off, bytes) in rf {
                                let t = off as i128 + delta;
                                if !(0..=i64::MAX as i128).contains(&t) {
                                    self.dropped.overflowed_offsets += 1;
                                    continue;
                                }
                                // Never flow a class into itself through its
                                // own call.
                                if ac == cc && delta == 0 {
                                    continue;
                                }
                                note(
                                    &mut xacc,
                                    &mut self.observed,
                                    &mut version,
                                    &mut self.dropped,
                                    &mut widths_changed,
                                    Some(ac),
                                    t as i64,
                                    bytes,
                                );
                            }
                        }
                    }
                }
            }

            // Derived values: a load whose value is a pointer root aliases
            // this site's result, and likewise a phi/select rooted at it.
            // Nothing flows back into the result class, so sites stay
            // separate.
            for (cb, ui) in call_rets {
                if !self.var_of.contains_key(cb) {
                    continue;
                }
                let Some(cc) = self.class_of(*cb) else { continue };
                let rf = self.result_facts(&xacc, *cb);
                if rf.is_empty() {
                    continue;
                }
                let f = f_of(module, *ui);
                let uic = self.class_of(*ui);
                let is_load = matches!(f.inst(ui.value), Some(Inst::Load { .. }));
                let is_join = matches!(
                    f.inst(ui.value),
                    Some(Inst::Phi { .. }) | Some(Inst::Select { .. })
                );
                let eligible = pointer_root(f, ui.value)
                    && uic.is_some()
                    && uic != Some(cc)
                    && (is_load || is_join);
                if !eligible {
                    continue;
                }
                let Some(uc) = uic else { continue };
                for (off, bytes) in &rf {
                    note(
                        &mut xacc,
                        &mut self.observed,
                        &mut version,
                        &mut self.dropped,
                        &mut widths_changed,
                        Some(uc),
                        *off,
                        *bytes,
                    );
                }
            }

            // Pointer fields, per call site: a load/store the callee performs
            // at `(formal + FO)` instantiates at the caller's actual
            // `(root + delta + FO)`.
            {
                let mut by_site: BTreeMap<ValRef, Vec<(Base, ValRef)>> = BTreeMap::new();
                for (ab, param, cb) in call_args {
                    by_site.entry(*cb).or_default().push((*ab, *param));
                }
                let mut callee_mems: BTreeMap<u32, Vec<(Base, ValRef)>> = BTreeMap::new();
                for (b, v) in loaded.iter().chain(stored.iter()) {
                    if let Some(r) = b.root {
                        if matches!(f_of(module, r).value(r.value), Value::Param { .. }) {
                            callee_mems.entry(r.func).or_default().push((*b, *v));
                        }
                    }
                }
                for (cb, sites) in &by_site {
                    let Some(callee) = self.callee_of(module, *cb) else {
                        continue;
                    };
                    let Some(mems) = callee_mems.get(&callee) else {
                        continue;
                    };
                    let mut actual_for: BTreeMap<ValRef, Base> = BTreeMap::new();
                    for (ab, param) in sites {
                        actual_for.insert(*param, *ab);
                    }
                    for (b, v) in mems {
                        let Some(broot) = b.root else { continue };
                        let Some(af) = actual_for.get(&broot).copied() else {
                            continue;
                        };
                        let Some(afroot) = af.root else { continue };
                        if !self.var_of.contains_key(&afroot) {
                            continue;
                        }
                        let slot = af.off as i128 + b.off as i128;
                        if !(0..=i64::MAX as i128).contains(&slot) {
                            self.dropped.overflowed_offsets += 1;
                            continue;
                        }
                        let slot = slot as i64;
                        let Some(ac) = self.class_of(afroot) else { continue };
                        // The value side maps like the address side: a value
                        // rooted at a formal of the same callee resolves to
                        // this site's actual for that formal (offset-aware);
                        // a value rooted outside keeps its own class.
                        // `None` plus `Some(true)` means an offset
                        // translation overflowed (reported), `None` alone is
                        // simply no mapping.
                        let map_value =
                            |me: &Self, w: ValRef| -> (Option<(usize, i64)>, bool) {
                                let wb = base(f_of(module, w), w.func, w.value, 0);
                                let Some(wroot) = wb.root else {
                                    return (None, false);
                                };
                                let is_callee_arg = wroot.func == callee
                                    && matches!(
                                        f_of(module, wroot).value(wroot.value),
                                        Value::Param { .. }
                                    );
                                if is_callee_arg {
                                    let Some(wf) = actual_for.get(&wroot).copied() else {
                                        return (None, false);
                                    };
                                    let Some(wfroot) = wf.root else {
                                        return (None, false);
                                    };
                                    let t = wf.off as i128 + wb.off as i128;
                                    if !(0..=i64::MAX as i128).contains(&t) {
                                        return (None, true);
                                    }
                                    if !me.var_of.contains_key(&wfroot) {
                                        return (None, false);
                                    }
                                    (me.class_of(wfroot).map(|c| (c, t as i64)), false)
                                } else if wb.off == 0 && me.var_of.contains_key(&wroot) {
                                    (me.class_of(wroot).map(|c| (c, 0)), false)
                                } else {
                                    (None, false)
                                }
                            };
                        // Is the value side rooted inside the callee (a
                        // callee-side child such as a loaded pointer), rather
                        // than mapped to one of this site's actuals?
                        let vbraw = base(f_of(module, *v), v.func, v.value, 0);
                        let callee_side = match vbraw.root {
                            None => false,
                            Some(r) => {
                                let vroot_in_callee = r.func == callee
                                    && matches!(
                                        f_of(module, r).value(r.value),
                                        Value::Param { .. }
                                    );
                                !vroot_in_callee
                                    && !(vbraw.off == 0
                                        && self.var_of.contains_key(&r)
                                        && actual_for.contains_key(&r))
                            }
                        };
                        // Caller-side children stored at this slot: their
                        // classes alias the callee child at this site.
                        let slot_owners = stored_owners
                            .get(&(self.uf.rep(ac), slot))
                            .map_or(&[][..], Vec::as_slice);
                        if callee_side {
                            let Some(vroot) = vbraw.root else { continue };
                            if !self.var_of.contains_key(&vroot) || slot_owners.is_empty() {
                                continue;
                            }
                            let cc = self.class_of(vroot).expect("checked above");
                            let mut flow: Vec<(usize, i64, u32)> = Vec::new();
                            if let Some(offs) = self.acc.get(&cc) {
                                for (off, bytes) in offs {
                                    for o in slot_owners {
                                        if *o != cc {
                                            flow.push((*o, *off, *bytes));
                                        }
                                    }
                                }
                            }
                            if let Some(offs) = xacc.get(&cc) {
                                for (off, bytes) in offs {
                                    for o in slot_owners {
                                        if *o != cc {
                                            flow.push((*o, *off, *bytes));
                                        }
                                    }
                                }
                            }
                            for (o, off, bytes) in flow {
                                note(
                                    &mut xacc,
                                    &mut self.observed,
                                    &mut version,
                                    &mut self.dropped,
                                    &mut widths_changed,
                                    Some(o),
                                    off,
                                    bytes,
                                );
                            }
                            // The slot's field fact names the caller's owner
                            // (first one, deterministic).
                            let owner = *slot_owners.iter().min().expect("non-empty");
                            match self.field_ptr.get(&(ac, slot)) {
                                Some(&existing) if self.uf.rep(existing) != owner => {
                                    // Conflicting children at one slot stay
                                    // explicit: keep the first, count the
                                    // disagreement.
                                    self.dropped.unsupported_bases += 1;
                                }
                                Some(_) => {}
                                None => {
                                    self.field_ptr.insert((ac, slot), owner);
                                }
                            }
                            continue;
                        }
                        let vc = match map_value(self, *v) {
                            (Some((c, _)), _) => c,
                            (None, overflowed) => {
                                if overflowed {
                                    self.dropped.overflowed_offsets += 1;
                                }
                                continue;
                            }
                        };
                        match self.field_ptr.get(&(ac, slot)) {
                            Some(&existing) if self.uf.rep(existing) != vc => {
                                self.dropped.unsupported_bases += 1;
                            }
                            Some(_) => {}
                            None => {
                                self.field_ptr.insert((ac, slot), vc);
                            }
                        }
                    }
                }
            }

            // Intra-function pointer flow: only the pointee's established
            // accesses flow into the loaded/stored value's class. The
            // container's own offsets are never copied into the child.
            for (b, v) in loaded.iter().chain(stored.iter()) {
                let Some(broot) = b.root else { continue };
                if !self.var_of.contains_key(&broot) {
                    continue;
                }
                let vb = base(f_of(module, *v), v.func, v.value, 0);
                let Some(vroot) = vb.root else { continue };
                if vb.off != 0 || !self.var_of.contains_key(&vroot) {
                    continue;
                }
                let Some(p) = self.class_of(vroot) else { continue };
                // The pointee class is whatever `field_ptr` established for
                // (container, B.Off); without that evidence there is nothing
                // directional to say (no blind container->child copy).
                let Some(bc) = self.class_of(broot) else { continue };
                let Some(&fp) = self.field_ptr.get(&(bc, b.off)) else {
                    continue;
                };
                let pc = self.uf.rep(fp);
                let src_version = *version.get(&pc).unwrap_or(&0);
                if flowed.get(&(p, pc)).copied() == Some(src_version) {
                    continue;
                }
                let mut facts: Vec<(i64, u32)> = Vec::new();
                if let Some(offs) = self.acc.get(&pc) {
                    facts.extend(offs.iter().map(|(o, b)| (*o, *b)));
                }
                if pc != p {
                    if let Some(offs) = xacc.get(&pc) {
                        facts.extend(offs.iter().map(|(o, b)| (*o, *b)));
                    }
                }
                for (off, bytes) in facts {
                    note(
                        &mut xacc,
                        &mut self.observed,
                        &mut version,
                        &mut self.dropped,
                        &mut widths_changed,
                        Some(p),
                        off,
                        bytes,
                    );
                }
                flowed.insert((p, pc), src_version);
            }

            if xacc == before && !widths_changed {
                return;
            }
            // Merge into `acc` so the next round sees the new facts through
            // the same widest-wins rule (sizes, nonzero flags, and observed
            // widths follow, so the scalar test and conflict evidence account
            // for propagated accesses too).
            for (c, offs) in &xacc {
                for (off, bytes) in offs {
                    let w = self.acc.entry(*c).or_default().entry(*off).or_insert(0);
                    *w = (*w).max(*bytes);
                    self.acc_sizes.entry(*c).or_default().insert(*bytes);
                    *self.nonzero.entry(*c).or_default() |= *off != 0;
                    self.observed
                        .entry(*c)
                        .or_default()
                        .entry(*off)
                        .or_default()
                        .insert(*bytes);
                }
            }
        }
        self.dropped.iterations = PROPAGATE_ROUNDS;
        self.dropped.fixpoint_truncated = true;
    }

    fn callee_of(&self, module: &Module, call: ValRef) -> Option<u32> {
        let f = f_of(module, call);
        callee_index(module, f, call.value).map(|i| i as u32)
    }

    /// Per-site return instantiation: the call result's classes observed at
    /// that site only, keyed by the call site, never by the callee.
    fn result_facts(
        &self,
        xacc: &BTreeMap<usize, BTreeMap<i64, u32>>,
        cb: ValRef,
    ) -> BTreeMap<i64, u32> {
        let mut out = BTreeMap::new();
        let Some(c) = self.class_of(cb) else {
            return out;
        };
        if let Some(offs) = self.acc.get(&c) {
            out = offs.clone();
        }
        if let Some(offs) = xacc.get(&c) {
            for (off, bytes) in offs {
                let w = out.entry(*off).or_insert(0);
                *w = (*w).max(*bytes);
            }
        }
        out
    }

    /// Partition refinement over the layouts: equal fields and pointees in
    /// equal blocks, until stable; each block takes its first member's name.
    fn share(&mut self) {
        let mut block: BTreeMap<usize, usize> =
            self.layouts.keys().map(|&c| (c, 0)).collect();
        let mut count = 1usize;
        loop {
            let mut keys: BTreeMap<Vec<i64>, usize> = BTreeMap::new();
            let mut next: BTreeMap<usize, usize> = BTreeMap::new();
            for (&c, l) in &self.layouts {
                let mut k: Vec<i64> = vec![block.get(&c).copied().unwrap_or(0) as i64];
                k.push(i64::from(l.scalar));
                for f in &l.fields {
                    k.push(f.off);
                    k.push(f.bytes as i64);
                    k.push(match block.get(&(f.pointee as usize)) {
                        Some(b) if f.pointee >= 0 => *b as i64,
                        _ => -1,
                    });
                }
                let n = keys.len();
                let id = *keys.entry(k).or_insert(n);
                next.insert(c, id);
            }
            block = next;
            if keys.len() == count {
                break;
            }
            count = keys.len();
        }
        let mut names: BTreeMap<usize, String> = BTreeMap::new();
        for (&c, l) in self.layouts.iter_mut() {
            let n = names.len();
            let name = names
                .entry(block[&c])
                .or_insert_with(|| format!("struct s{}", n + 1))
                .clone();
            l.name = name;
        }
    }

    /// The class a pointer stored at `b` points to, if known.
    fn pointee_at(&self, module: &Module, b: Base) -> i32 {
        let Some(root) = b.root else { return -1 };
        if pointer_root(f_of(module, root), root.value) {
            let Some(c) = self.class_of(root) else {
                return -1;
            };
            let Some(l) = self.layouts.get(&c) else {
                return -1;
            };
            if l.scalar {
                return -1;
            }
            return l.at(b.off, 8).map_or(-1, |f| f.pointee);
        }
        // Frame roots (stack allocas and CPU-state globals alike).
        let Some(key) = self.frame_key(module, f_of(module, root), root) else {
            return -1;
        };
        let Some((_, l)) = self.frames.iter().find(|(k, _)| *k == key) else {
            return -1;
        };
        for e in &l.fields {
            if e.embed >= 0 && e.off <= b.off && b.off < e.off + e.bytes as i64 {
                let embedded = self.uf.rep(e.embed as usize);
                let Some(el) = self.layouts.get(&embedded) else {
                    return -1;
                };
                return el.at(b.off - e.off, 8).map_or(-1, |f| f.pointee);
            }
        }
        -1
    }

    /// Place a struct of class `c` at `off` in frame `key`, over the scalar
    /// fields there. False if it does not fit or is there already.
    fn embed(&mut self, key: FrameKey, off: i64, c: usize) -> bool {
        let Some(l) = self.layouts.get(&c) else {
            return false;
        };
        if l.scalar {
            return false;
        }
        let size = l.size as i64;
        let end = off + size;
        let Some((_, frame)) = self.frames.iter_mut().find(|(k, _)| *k == key) else {
            return false;
        };
        if off < 0 || end > frame.size as i64 {
            return false;
        }
        for f in &frame.fields {
            if f.embed >= 0 && f.off < end && off < f.off + f.bytes as i64 {
                return false;
            }
        }
        frame
            .fields
            .retain(|f| !(f.off < end && off < f.off + f.bytes as i64));
        let name = slot_name(frame.below, off);
        let mut e = LField::new(off, size as u32);
        e.embed = c as i32;
        e.name = name;
        let at = frame
            .fields
            .partition_point(|f| f.off < e.off);
        frame.fields.insert(at, e);
        true
    }

    /// Export state in deterministic module/function/IR order.
    fn compute_export(&mut self, module: &Module) {
        self.export = Export::default();
        // Shared layouts in deterministic order: one id per printed structure.
        // `share()` gives structurally equal classes the same name; the id is
        // assigned per unique name (first class in deterministic order wins),
        // so two classes sharing `struct s1` export one layout id, and
        // bindings referencing either class resolve to it.
        let mut sorted: Vec<(usize, String)> = self
            .layouts
            .iter()
            .map(|(&c, l)| (c, l.name.clone()))
            .collect();
        sorted.sort_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));
        let mut id_by_name: BTreeMap<String, usize> = BTreeMap::new();
        for (c, name) in sorted {
            let n = self.export.order.len();
            let id = *id_by_name.entry(name).or_insert(n);
            if id == n {
                self.export.order.push(c);
            }
            self.export.id_of.insert(c, id);
        }

        // Argument/return bindings in module/function/parameter order. `arg`
        // is the parameter index; `function` is the guest entry address.
        for (fi, f) in module.functions.iter().enumerate() {
            let Some(entry) = entry_of(f) else { continue };
            for (k, p) in f.params.iter().enumerate() {
                let v = ValRef {
                    func: fi as u32,
                    value: *p,
                };
                let class = self
                    .class_of(v)
                    .filter(|c| self.layouts.contains_key(c));
                self.export.args.push(Binding {
                    function: entry,
                    index: k as u32,
                    offset: 0,
                    class,
                });
            }
            if !matches!(f.ty.ret, Type::Void) {
                // A shared return layout requires agreement on every path.
                let mut class: Option<usize> = None;
                let mut seen = false;
                let mut consistent = true;
                for b in &f.blocks {
                    let Some(Terminator::Ret(Some(v))) = &b.term else {
                        continue;
                    };
                    let bb = base(f, fi as u32, *v, 0);
                    let cc = match bb.root {
                        Some(r) if bb.off == 0 && pointer_root(f, r.value) => {
                            self.class_of(r)
                        }
                        _ => None,
                    };
                    let id = self.shared_id(cc);
                    if id.is_none() || (seen && id != self.shared_id(class)) {
                        consistent = false;
                    }
                    if !seen {
                        class = cc;
                    }
                    seen = true;
                }
                self.export.rets.push(Binding {
                    function: entry,
                    index: u32::MAX,
                    offset: 0,
                    class: if seen && consistent { class } else { None },
                });
            }
        }

        // Frames in deterministic `frame_order` (module/global/instruction
        // order), ties by layout name.
        let mut ordered: Vec<(FrameKey, StructLayout)> = std::mem::take(&mut self.frames);
        ordered.sort_by(|a, b| {
            let oa = self.frame_order.get(&a.0).copied().unwrap_or(u32::MAX);
            let ob = self.frame_order.get(&b.0).copied().unwrap_or(u32::MAX);
            oa.cmp(&ob).then_with(|| a.1.name.cmp(&b.1.name))
        });
        for (key, _) in &ordered {
            let (function, name) = match *key {
                FrameKey::Alloca(r) => {
                    let f = f_of(module, r);
                    (entry_of(f).unwrap_or(0), f.name.clone())
                }
                FrameKey::Global(gi) => (0, module.globals[gi].name.clone()),
            };
            self.export.frames.push(FrameRef {
                key: *key,
                function,
                name,
            });
        }
        self.frames = ordered;
    }
}

/// Record one propagated `(offset, bytes)` fact, reporting overflow/cap
/// drops instead of hiding them.
#[allow(clippy::too_many_arguments)]
fn note(
    xacc: &mut BTreeMap<usize, BTreeMap<i64, u32>>,
    observed: &mut BTreeMap<usize, BTreeMap<i64, BTreeSet<u32>>>,
    version: &mut BTreeMap<usize, u64>,
    dropped: &mut Limits,
    widths_changed: &mut bool,
    c: Option<usize>,
    off: i64,
    bytes: u32,
) {
    if !(0..MAX_FIELD_OFFSET).contains(&off) {
        dropped.capped_offsets += 1;
        return;
    }
    let Some(c) = c else {
        dropped.unsupported_bases += 1;
        return;
    };
    let widths = observed.entry(c).or_default().entry(off).or_default();
    *widths_changed |= widths.insert(bytes);
    let entry = xacc.entry(c).or_default();
    match entry.get_mut(&off) {
        Some(w) => {
            if bytes > *w {
                *version.entry(c).or_insert(0) += 1;
                *w = bytes;
            }
        }
        None => {
            entry.insert(off, bytes);
            *version.entry(c).or_insert(0) += 1;
        }
    }
}

/// Fields from `(offset, bytes)` accesses: widest per offset, then first-come
/// non-overlapping.
fn fields_of(acc: &BTreeMap<i64, u32>, lo: i64, hi: i64) -> Vec<LField> {
    let mut out = Vec::new();
    let mut end = lo;
    for (&off, &bytes) in acc {
        if off < end || off + bytes as i64 > hi {
            continue;
        }
        out.push(LField::new(off, bytes));
        end = off + bytes as i64;
    }
    out
}

fn is_pointerish(t: &Type) -> bool {
    matches!(t, Type::Ptr | Type::Int(64))
}

/// A root a pointer variable can stand for: an argument, load, call result,
/// phi or select of 64-bit-integer or pointer type.
fn pointer_root(f: &Function, v: ValueId) -> bool {
    let shape = matches!(
        f.value(v),
        Value::Param { .. }
            | Value::Inst(Inst::Load { .. })
            | Value::Inst(Inst::Call { .. })
            | Value::Inst(Inst::Phi { .. })
            | Value::Inst(Inst::Select { .. })
    );
    shape && is_pointerish(&f.ty(v))
}

/// Root plus constant offset of an address (integer or pointer). Reduces
/// through `gep`, `inttoptr`/`ptrtoint` and constant adds.
fn base(f: &Function, func: u32, v: ValueId, depth: u32) -> Base {
    let raw = Base {
        root: Some(ValRef { func, value: v }),
        off: 0,
    };
    if depth > 32 {
        return raw;
    }
    // Value ids are per function: `func` names which arena `v` belongs to, so
    // an id outside it says nothing about this function's layout.
    if v.0 as usize >= f.values.len() {
        return raw;
    }
    match &f.values[v.0 as usize] {
        Value::Const(Constant::Int { bits, width }) if *width <= 64 => Base {
            root: None,
            off: *bits as u64 as i64,
        },
        Value::Inst(Inst::Gep { base: b, offset }) => {
            let mut bx = base(f, func, *b, depth + 1);
            bx.off = bx.off.wrapping_add(*offset);
            bx
        }
        Value::Inst(Inst::Cast { value, .. }) => {
            // `inttoptr`/`ptrtoint`: representation-preserving here.
            let t = f.ty(*value);
            if is_pointerish(&t) {
                base(f, func, *value, depth + 1)
            } else {
                raw
            }
        }
        Value::Inst(Inst::Bin { op, lhs, rhs, .. })
            if matches!(op, biara_ir::BinOp::Add | biara_ir::BinOp::Sub | biara_ir::BinOp::Or) =>
        {
            let Some(k) = const_i64(f, *rhs) else {
                return raw;
            };
            if *op == biara_ir::BinOp::Or {
                // The native code required `isDisjoint()` here; that proof is
                // not expressible over this IR, so only the identity (k == 0)
                // is reduced rather than inventing offsets (documented).
                if k == 0 {
                    return base(f, func, *lhs, depth + 1);
                }
                return raw;
            }
            let mut bx = base(f, func, *lhs, depth + 1);
            bx.off = bx.off.wrapping_add(if *op == biara_ir::BinOp::Sub {
                -k
            } else {
                k
            });
            bx
        }
        _ => raw,
    }
}

fn const_i64(f: &Function, v: ValueId) -> Option<i64> {
    match f.value(v) {
        Value::Const(Constant::Int { bits, width }) if *width == 64 => Some(*bits as u64 as i64),
        _ => None,
    }
}

/// Frame slots that only ever hold one frame address (spills at -O0): a load
/// from one is that address, for typing.
/// Recover equality through a unique, dominating stack spill.
/// A frame address exposed anywhere in the function can be modified by a call.
fn frame_spill_aliases(f: &Function, fi: u32, same: &mut Vec<(ValRef, ValRef)>) {
    let frame_base = |id| {
        let b = base(f, fi, id, 0);
        let root = b.root?;
        matches!(f.inst(root.value), Some(Inst::Alloca { .. })).then_some((root, b.off))
    };
    let mut stores: BTreeMap<_, Vec<_>> = BTreeMap::new();
    let mut loads = Vec::new();
    for (index, block) in f.blocks.iter().enumerate() {
        let block_id = biara_ir::BlockId(index as u32);
        for (position, &id) in block.insts.iter().enumerate() {
            match f.inst(id) {
                Some(Inst::Store { ptr, value, ty, .. }) => {
                    if let Some((frame, offset)) = frame_base(*ptr) {
                        stores.entry(frame).or_default().push((
                            offset, i64::from(ty_bytes(ty)), block_id, position, *value,
                        ));
                    }
                }
                Some(Inst::Load { ptr, ty, .. }) if is_pointerish(ty) => {
                    if let Some((frame, offset)) = frame_base(*ptr) {
                        loads.push((frame, offset, ty, block_id, position, id));
                    }
                }
                _ => {}
            }
        }
    }
    if loads.is_empty() || stores.is_empty() {
        return;
    }
    let mut escapes = BTreeSet::new();
    for block in &f.blocks {
        for &id in &block.insts {
            let Some(inst) = f.inst(id) else { continue };
            for operand in biara_ir::operands(inst) {
                let Some((root, _)) = frame_base(operand) else { continue };
                let internal = match inst {
                    Inst::Load { ptr, .. } => operand == *ptr,
                    Inst::Store { ptr, value, .. } => operand == *ptr && operand != *value,
                    Inst::Gep { .. } | Inst::Cast { .. } | Inst::Bin { .. } => {
                        is_pointerish(&f.ty(id)) && frame_base(id).is_some_and(|(r, _)| r == root)
                    }
                    _ => false,
                };
                if !internal {
                    escapes.insert(root);
                }
            }
        }
        for operand in block.term.as_ref().into_iter().flat_map(biara_ir::term_operands) {
            if let Some((root, _)) = frame_base(operand) {
                escapes.insert(root);
            }
        }
    }
    let mut dom = None;
    for (frame, offset, ty, block, position, load) in loads {
        if escapes.contains(&frame) {
            continue;
        }
        let Some(stores) = stores.get(&frame) else { continue };
        let bytes = i64::from(ty_bytes(ty));
        let mut overlapping = stores.iter().filter(|&&(start, width, ..)| {
            offset < start.saturating_add(width) && start < offset.saturating_add(bytes)
        });
        let Some(&(start, width, from, before, value)) = overlapping.next() else { continue };
        if start != offset || width != bytes || overlapping.next().is_some()
            || f.ty(value) != *ty || !pointer_root(f, value)
        {
            continue;
        }
        let dominates = if from == block {
            before < position
        } else {
            dom.get_or_insert_with(|| biara_ir::analysis::Dominators::new(f))
                .dominates(from, block)
        };
        if dominates {
            same.push((ValRef { func: fi, value: load }, ValRef { func: fi, value }));
        }
    }
}

fn type_base(module: &Module, v: &ValRef, holds: &BTreeMap<(ValRef, i64), Option<Base>>) -> Base {
    let f = f_of(module, *v);
    let b = base(f, v.func, v.value, 0);
    let Some(broot) = b.root else { return b };
    if let Some(Inst::Load { ptr, .. }) = f.inst(broot.value) {
        let p = base(f, v.func, *ptr, 0);
        if let Some(p_root) = p.root {
            if let Some(Some(held)) = holds.get(&(p_root, p.off)) {
                if let Some(hr) = held.root {
                    if matches!(f_of(module, hr).inst(hr.value), Some(Inst::Alloca { .. })) {
                        return *held;
                    }
                }
            }
        }
    }
    b
}

fn f_of(module: &Module, r: ValRef) -> &Function {
    &module.functions[r.func as usize]
}

fn f_at(module: &Module, r: ValRef) -> Option<&Inst> {
    f_of(module, r).inst(r.value)
}

/// A frame root: a stack alloca or a module global.
fn is_frame_value(module: &Module, r: ValRef) -> bool {
    // A pair from another function's arena names no value in this one.
    let f = f_of(module, r);
    if (r.value.0 as usize) >= f.values.len() {
        return false;
    }
    match f.value(r.value) {
        Value::Inst(Inst::Alloca { .. }) => true,
        Value::Const(Constant::GlobalPtr(name)) => module.globals.iter().any(|g| &g.name == name),
        _ => false,
    }
}

/// Index of the function a direct call refers to, when it is defined here.
fn callee_index(module: &Module, f: &Function, call: ValueId) -> Option<usize> {
    let Some(Inst::Call { callee, .. }) = f.inst(call) else {
        return None;
    };
    let Value::Const(Constant::Function(name)) = f.value(*callee) else {
        return None;
    };
    // A declaration is not a body: nothing to propagate through.
    module.functions.iter().position(|g| &g.name == name)
}


