//! C from a module in ABI form.
//!
//! Ported from the former `cpp/cgen.cpp` renderer, now walking [`biara_ir`]
//! directly: no FFI, no LLVM, no unsafe.
//!
//! Mode 0 (executable C): whole-module source streamed in bounded chunks,
//! dependencies before first use. Compilable by construction: casts, wrapping
//! (`wrap`), and side effects are preserved; only names/folding/structure are
//! simplified, never semantics.
//!
//! Mode 1 (function packages): one complete UTF-8 JSON object per guest
//! function carrying `!biara.entry` (here `FuncAttrs::entry_addr`), emitted
//! with a single sink call per object. Dispatcher/runtime helpers
//! (`biara.run`, `biara.dispatch`, `biara.syscall`, `biara_function_at`,
//! `biara_div*`, `biara_sdiv*`, declarations without an entry address,
//! intrinsics) are excluded. When any function carries `!biara.present`
//! (`FuncAttrs::present`), only those are emitted in ordinal order; otherwise
//! every entry carrier in module order. Each package is self-contained:
//! `body` is the full definition from the same renderer as Mode 0,
//! `dependencies` holds only what the function uses (types recursively,
//! strings, globals, imports, helpers), captured independently per function
//! (no cross-function dedup).
//!
//! Provenance: `statements[].addresses` are surviving `!biara.pc`
//! contributors captured BEFORE folding/inlining (including inline
//! expressions, edge phi copies, and return-merge RHS), unioned on merges.
//! They are NOT exhaustive original coverage: biara_ir does not model
//! debug metadata per se and
//! structural lines (braces/labels/gotos/breaks) never guess a PC.
//! `provenance` is `exact` only when every contributor had surviving
//! metadata, `inferred` on partial loss, `unknown` when none survives. Lost
//! metadata is reported, never fabricated.
//!
//! biara_ir records one optional PC per value (`Function::pc`), not a list,
//! and terminators carry none. The port keeps the contract by treating a
//! missing value PC as loss and by marking every terminator-sourced
//! statement (edge copies, returns, branch headers) `inferred` rather than
//! inventing an address for it.
//!
//! Names: arguments default to neutral `arg0...` (the architectural name is
//! kept in `names[].original` from the IR parameter name), values to neutral
//! `vN` except justified producer names (`<callee>_result`, evidence
//! `inferred`). Unique source bindings for entry registers have evidence
//! `debug` and retain the original architectural name in the report.
//! Frame fields from `!biara.debug_objects` are not modelled either.
//!
//! Types: every address is reduced to a root plus a constant offset (through
//! `gep`, `inttoptr`/`ptrtoint` and constant adds). Roots are function
//! parameters, loaded values, call results, phis and selects (pointer
//! variables), the stack frame allocas, the CPU state global, or nothing (an
//! absolute address).
//!
//! The layout analysis is consumed from [`crate::layouts_core::Layouts`],
//! which exposes the shared struct layouts (used here for printed struct
//! declarations and mode-1 type dependencies) but no per-value pointer-class
//! map and no frame layouts. That part of the C++ renderer cannot be
//! expressed against this API, so — explicitly, not silently — pointer
//! variables are printed as their transport type (`uint8_t *`) and accesses
//! through them are typed byte casts at the same offsets, which preserves
//! every byte accessed. Stack frames become byte arrays of the alloca's size
//! (C++'s own fallback when no frame layout exists), and the CPU state struct
//! is built from [`Options::cpu`] (env-relative offsets plus `margin`).
//!
//! Code: one C statement per instruction that is not folded into its single
//! user (folding bounded by a size budget so output stays readable),
//! gotos between basic blocks, phis as assignments on the edges. Natural
//! loops with a proven pure-header single-exit condition read as `while`;
//! a unique constant-step induction phi permits `for`. A unique conditional
//! latch without loop/exit phi copies permits `do/while`, preserving effects
//! before the test. Other loops retain `for (;;)` and explicit transfers.

use crate::layouts_core::{Layout as LLayout, Layouts};
use biara_ir::{
    operands, successor_blocks, BinOp, BlockId, Constant, FuncAttrs, FuncType, Function, Inst,
    IntPred, Module, Terminator, Type, UnOp, Value, ValueId,
};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::io::{self, Write};

/// A named field of the CPU state, offset relative to the start of
/// `biara_cpu` (before `Options::margin` is added).
#[derive(Clone, Debug)]
pub struct CpuField {
    pub off: i64,
    pub bytes: u32,
    pub name: String,
}

/// Copied data (`R_*_COPY`): address and size in the image, host-linked name.
#[derive(Clone, Debug)]
pub struct DataSym {
    pub addr: u64,
    pub size: u64,
    pub name: String,
}

/// A guest string literal at a call argument: address and C-escaped text.
#[derive(Clone, Debug)]
pub struct GuestStr {
    pub addr: u64,
    pub text: String,
}

/// A called import's declaration: parameter count, which take strings, and
/// whether calls vary in count (then `, ...`).
#[derive(Clone, Debug)]
pub struct Proto {
    pub name: String,
    pub nargs: u32,
    pub strmask: u64,
    pub ellipsis: bool,
}

/// Everything the renderer needs besides the module itself.
pub struct Options<'a> {
    pub cpu: &'a [CpuField],
    pub conv_args: &'a [String],
    pub margin: u64,
    pub banner: &'a str,
    pub data: &'a [DataSym],
    pub strings: &'a [GuestStr],
    pub protos: &'a [Proto],
    pub syms: &'a [GuestStr],
    /// Layouts from the layout analysis (`crate::layouts_core::Layouts`).
    pub layouts: &'a Layouts,
}

/// Mode 0: write C to `out`; returns the count of unsupported constructs.
pub fn emit(
    module: &Module,
    opts: &Options<'_>,
    out: &mut dyn Write,
) -> Result<usize, String> {
    let mut e = Emitter::new(module, opts);
    e.run(out)?;
    Ok(e.unsupported)
}

/// Mode 1: one JSON object per function.
pub fn emit_packages(
    module: &Module,
    opts: &Options<'_>,
    out: &mut dyn FnMut(&str) -> io::Result<()>,
) -> Result<usize, String> {
    let mut e = Emitter::new(module, opts);
    e.packages(out)?;
    Ok(e.unsupported)
}

// ---------------------------------------------------------------------------
// small text helpers
// ---------------------------------------------------------------------------

fn json_str(s: &str) -> String {
    let mut r = String::with_capacity(s.len() + 16);
    r.push('"');
    for &c in s.as_bytes() {
        match c {
            b'"' => r.push_str("\\\""),
            b'\\' => r.push_str("\\\\"),
            0x08 => r.push_str("\\b"),
            0x0c => r.push_str("\\f"),
            b'\n' => r.push_str("\\n"),
            b'\r' => r.push_str("\\r"),
            b'\t' => r.push_str("\\t"),
            c if c < 0x20 => {
                r.push_str(&format!("\\u00{:02x}", c));
            }
            c => r.push(c as char),
        }
    }
    r.push('"');
    r
}

/// A byte offset: decimal while small, hex beyond.
fn num(v: i64) -> String {
    if v < 0 {
        return format!("-{}", num(v.wrapping_neg()));
    }
    if v < 10 {
        v.to_string()
    } else {
        hexv(v as u64)
    }
}

fn hexv(v: u64) -> String {
    format!("0x{:x}", v)
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
    let mut r = String::with_capacity(s.len());
    for c in s.chars() {
        r.push(if c.is_ascii_alphanumeric() || c == '_' {
            c
        } else {
            '_'
        });
    }
    if r.is_empty() || r.as_bytes()[0].is_ascii_digit() {
        r.insert(0, '_');
    }
    if is_keyword(&r) {
        r.push('_');
    }
    r
}

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

fn uint_ty(bits: u32) -> String {
    match container(bits) {
        1 => "bool",
        8 => "uint8_t",
        16 => "uint16_t",
        32 => "uint32_t",
        64 => "uint64_t",
        128 => "biara_u128",
        _ => "",
    }
    .to_string()
}

fn int_ty(bits: u32) -> String {
    match container(bits) {
        8 => "int8_t",
        16 => "int16_t",
        32 => "int32_t",
        64 => "int64_t",
        128 => "biara_i128",
        _ => "",
    }
    .to_string()
}

fn standard(bits: u32) -> bool {
    container(bits) == bits
}

fn zero_initializer(constant: &Constant) -> bool {
    match constant {
        Constant::Zero | Constant::Int { bits: 0, .. } => true,
        Constant::Aggregate(fields) => fields.iter().all(zero_initializer),
        _ => false,
    }
}

fn width_mask(w: u32) -> u128 {
    if w >= 128 {
        u128::MAX
    } else {
        (1u128 << w) - 1
    }
}

fn sign_extend(bits: u128, w: u32) -> i128 {
    if w == 0 {
        return 0;
    }
    let w = w.min(128);
    let raw = bits & width_mask(w);
    let sign = 1u128 << (w - 1);
    if raw & sign != 0 {
        (raw | !width_mask(w)) as i128
    } else {
        raw as i128
    }
}

fn is_all_ones(bits: u128, w: u32) -> bool {
    (bits & width_mask(w)) == width_mask(w)
}

/// An unsigned constant without a suffix.
fn lit_v(bits: u128, w: u32) -> String {
    if w == 1 {
        return if bits & 1 == 1 { "true" } else { "false" }.to_string();
    }
    if w > 64 {
        return suffixed(bits, w);
    }
    let x = (bits & width_mask(w)) as u64;
    if is_all_ones(bits, w) && container(w) >= 32 {
        return "-1".to_string();
    }
    if x <= 1000 {
        x.to_string()
    } else {
        hexv(x)
    }
}

/// `lit` with the type in the literal (`1ull`, `5u`).
fn suffixed(bits: u128, w: u32) -> String {
    if w == 1 {
        return lit_v(bits, w);
    }
    if w > 64 {
        let hi = (bits >> 64) as u64;
        let lo = bits as u64;
        if hi == 0 {
            return format!("((biara_u128){})", suffixed(lo as u128, 64));
        }
        return format!("(((biara_u128){}ull << 64) | {}ull)", hexv(hi), hexv(lo));
    }
    let x = (bits & width_mask(w)) as u64;
    let suffix = if container(w) == 64 { "ull" } else { "u" };
    if is_all_ones(bits, w) {
        return format!("-1{}", suffix);
    }
    let s = if x <= 1000 { x.to_string() } else { hexv(x) };
    format!("{}{}", s, suffix)
}

/// `lit`, but for a signed C type (`2`, not `2u`; `-1`, not `0xff...`).
fn slit(bits: u128, w: u32) -> String {
    if w > 64 {
        return lit_v(bits, w);
    }
    if is_all_ones(bits, w) {
        return "-1".to_string();
    }
    let x = sign_extend(bits, w);
    if x == i64::MIN as i128 {
        return "(-0x7fffffffffffffff - 1)".to_string();
    }
    if x < 0 {
        x.to_string()
    } else if (x as u64) <= 1000 {
        x.to_string()
    } else {
        hexv(x as u64)
    }
}

/// `E` reduced to `Bits` (after arithmetic that may leave the width).
fn wrap(e: &str, bits: u32) -> String {
    if bits == 1 {
        return format!("(({}) & 1)", e);
    }
    if standard(bits) {
        if bits >= 32 {
            e.to_string()
        } else {
            format!("({})({})", uint_ty(bits), e)
        }
    } else {
        let mask = width_mask(bits) & width_mask(container(bits));
        format!("(({}) & {})", e, lit_v(mask, container(bits)))
    }
}

fn is_ptr_ty(s: &str) -> bool {
    s.ends_with('*') || s.starts_with("biara_callback_")
}

/// `e` (of C type `from`) as C type `to`.
fn conv(e: &str, from: &str, to: &str) -> String {
    if from == to {
        return e.to_string();
    }
    if is_ptr_ty(from) && !is_ptr_ty(to) && to == "uint64_t" {
        return format!("({}){}", to, e);
    }
    if is_ptr_ty(from) && !is_ptr_ty(to) {
        return format!("({})(uintptr_t){}", to, e);
    }
    if !is_ptr_ty(from) && is_ptr_ty(to) && from == "uint64_t" {
        return format!("({}){}", to, e);
    }
    if !is_ptr_ty(from) && is_ptr_ty(to) {
        return format!("({})(uintptr_t){}", to, e);
    }
    format!("({}){}", to, e)
}

/// `E` without one pair of parentheses around all of it.
fn bare(e: &str) -> String {
    let b = e.as_bytes();
    if b.len() < 2 || b[0] != b'(' || b[b.len() - 1] != b')' {
        return e.to_string();
    }
    let mut depth: i32 = 0;
    for (i, &c) in b.iter().enumerate().take(b.len() - 1) {
        match c {
            b'(' => depth += 1,
            b')' => depth -= 1,
            _ => {}
        }
        if depth == 0 {
            let _ = i;
            return e.to_string();
        }
    }
    e[1..e.len() - 1].to_string()
}

/// Whether statement `t` leaves the code after it unreachable.
fn ends(t: &str) -> bool {
    t.starts_with("return")
        || t.starts_with("goto ")
        || t == "continue;"
        || t == "break;"
        || t == "__builtin_unreachable();"
}

/// C name of a function; the guest's `main` is not C's.
fn fn_name(name: &str, decl: bool) -> String {
    if name == "main" && !decl {
        "guest_main".to_string()
    } else {
        ident(name)
    }
}

// ---------------------------------------------------------------------------
// bounded, newline-normalizing output
// ---------------------------------------------------------------------------

/// Append-only, bounded output with newline normalization across every
/// section and function boundary. A failed sink stops both writing and code
/// generation.
struct COutput<'a> {
    w: &'a mut dyn Write,
    pending: String,
    newlines: u32,
    error: Option<String>,
}

impl<'a> COutput<'a> {
    fn new(w: &'a mut dyn Write) -> Self {
        Self {
            w,
            pending: String::new(),
            newlines: 0,
            error: None,
        }
    }

    fn flush(&mut self) {
        if !self.pending.is_empty() && self.error.is_none() {
            if let Err(e) = self.w.write_all(self.pending.as_bytes()) {
                self.error = Some(e.to_string());
            }
            self.pending.clear();
        }
    }

    fn append(&mut self, text: &str) -> bool {
        self.pending.push_str(text);
        if self.pending.len() >= 65536 {
            self.flush();
        }
        self.error.is_none()
    }

    fn flush_newlines(&mut self) -> bool {
        let n = self.newlines;
        self.newlines = 0;
        match n {
            0 => true,
            1 => self.append("\n"),
            _ => self.append("\n\n"),
        }
    }

    fn write(&mut self, text: &str) -> bool {
        if self.error.is_some() {
            return false;
        }
        let mut rest = text;
        while !rest.is_empty() {
            if rest.starts_with('\n') {
                self.newlines = (self.newlines + 1).min(2);
                rest = &rest[1..];
                continue;
            }
            if !self.flush_newlines() {
                return false;
            }
            match rest.find('\n') {
                Some(i) => {
                    if !self.append(&rest[..i]) {
                        return false;
                    }
                    rest = &rest[i..];
                }
                None => {
                    if !self.append(rest) {
                        return false;
                    }
                    rest = "";
                }
            }
        }
        true
    }

    fn err(&self) -> String {
        self.error.clone().unwrap_or_else(|| "output failed".into())
    }

    fn finish(&mut self) -> Result<(), String> {
        if self.error.is_none() && self.newlines > 0 {
            self.flush_newlines();
        }
        self.flush();
        match &self.error {
            Some(e) => Err(e.clone()),
            None => Ok(()),
        }
    }
}

// ---------------------------------------------------------------------------
// instruction records
// ---------------------------------------------------------------------------

/// One C statement and the values it reads. `pcs` are surviving PC
/// contributors captured BEFORE folding/inlining into this statement:
/// direct from instructions emitted here, folded from values folded into it
/// (inline expressions, edge phi inputs, return-merge RHS). `lost_fold` marks
/// a folded value with no surviving PC (partial metadata loss).
#[derive(Clone, Default)]
struct Stmt {
    text: String,
    /// a pure assignment to this value
    def: Option<ValueId>,
    /// the variable it writes, if any
    assigns: Option<ValueId>,
    /// for a pure `v = RHS`, the RHS (mergeable into `return RHS`)
    rhs: String,
    /// writes `<phi>_next`
    staged: bool,
    refs: Vec<ValueId>,
    depth: u32,
    /// a label statement for this block
    label: Option<BlockId>,
    pcs: BTreeSet<u64>,
    lost_fold: bool,
}

impl Stmt {
    fn plain(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            ..Default::default()
        }
    }
}

/// Per-function CFG and use facts, built once.
struct Facts {
    uses: Vec<u32>,
    inst_users: Vec<Vec<(ValueId, BlockId)>>,
    /// Blocks whose terminator reads the value (a use like any other).
    term_uses: Vec<Vec<BlockId>>,
    pos: HashMap<ValueId, u32>,
    preds: Vec<Vec<BlockId>>,
    succs: Vec<Vec<BlockId>>,
}

impl Facts {
    fn build(f: &Function) -> Self {
        let n = f.values.len();
        let mut uses = vec![0u32; n];
        let mut inst_users: Vec<Vec<(ValueId, BlockId)>> = vec![Vec::new(); n];
        let mut term_uses: Vec<Vec<BlockId>> = vec![Vec::new(); n];
        let mut pos = HashMap::new();
        for (bi, b) in f.blocks.iter().enumerate() {
            let bid = BlockId(bi as u32);
            for (k, &v) in b.insts.iter().enumerate() {
                pos.insert(v, k as u32);
                if let Some(inst) = f.inst(v) {
                    for o in operands(inst) {
                        if let Some(slot) = uses.get_mut(o.0 as usize) {
                            *slot += 1;
                        }
                        if let Some(list) = inst_users.get_mut(o.0 as usize) {
                            list.push((v, bid));
                        }
                    }
                }
            }
            for o in b.term.as_ref().map(biara_ir::term_operands).unwrap_or_default() {
                if let Some(slot) = uses.get_mut(o.0 as usize) {
                    *slot += 1;
                }
                if let Some(list) = term_uses.get_mut(o.0 as usize) {
                    if !list.contains(&bid) {
                        list.push(bid);
                    }
                }
            }
        }
        let mut preds = vec![Vec::new(); f.blocks.len()];
        let mut succs = vec![Vec::new(); f.blocks.len()];
        for (bi, b) in f.blocks.iter().enumerate() {
            let from = BlockId(bi as u32);
            for to in b.term.as_ref().map(successor_blocks).unwrap_or_default() {
                succs[bi].push(to);
                if let Some(slot) = preds.get_mut(to.0 as usize) {
                    slot.push(from);
                }
            }
        }
        Self {
            uses,
            inst_users,
            term_uses,
            pos,
            preds,
            succs,
        }
    }

    fn uses_of(&self, v: ValueId) -> u32 {
        self.uses.get(v.0 as usize).copied().unwrap_or(0)
    }

    fn pos_of(&self, v: ValueId) -> Option<u32> {
        self.pos.get(&v).copied()
    }
}

// ---------------------------------------------------------------------------
// dominators and post-dominators over the CFG the terminators define
// ---------------------------------------------------------------------------

/// Immediate dominators, computed from the successor edges the terminators
/// spell out. [`biara_ir::Dominators`] trusts `Block::preds`; a module that
/// was edited without recomputing them would otherwise lose every loop.
struct Dom {
    idom: Vec<Option<BlockId>>,
    rpo_num: Vec<u32>,
}

impl Dom {
    fn build(f: &Function) -> Self {
        let n = f.blocks.len();
        let mut preds: Vec<Vec<BlockId>> = vec![Vec::new(); n];
        let mut succs: Vec<Vec<BlockId>> = vec![Vec::new(); n];
        for i in 0..n {
            for to in f.blocks[i].term.as_ref().map(successor_blocks).unwrap_or_default() {
                succs[i].push(to);
                if let Some(slot) = preds.get_mut(to.0 as usize) {
                    slot.push(BlockId(i as u32));
                }
            }
        }
        let mut this = Dom {
            idom: vec![None; n],
            rpo_num: vec![u32::MAX; n],
        };
        let Some(entry) = f.entry else { return this };
        // Reverse postorder from the entry.
        let mut post = Vec::new();
        let mut seen = vec![false; n];
        let mut stack = vec![(entry, 0usize)];
        seen[entry.0 as usize] = true;
        while let Some((b, k)) = stack.pop() {
            if k < succs[b.0 as usize].len() {
                stack.push((b, k + 1));
                let s = succs[b.0 as usize][k];
                if !seen[s.0 as usize] {
                    seen[s.0 as usize] = true;
                    stack.push((s, 0));
                }
            } else {
                post.push(b);
            }
        }
        let rpo: Vec<BlockId> = post.into_iter().rev().collect();
        for (i, b) in rpo.iter().enumerate() {
            this.rpo_num[b.0 as usize] = i as u32;
        }
        let mut changed = true;
        while changed {
            changed = false;
            for &b in rpo.iter().skip(1) {
                let mut new: Option<BlockId> = None;
                for &p in &preds[b.0 as usize] {
                    if this.rpo_num[p.0 as usize] == u32::MAX {
                        continue;
                    }
                    new = Some(match new {
                        None => p,
                        Some(cur) => this.intersect(p, cur),
                    });
                }
                if new.is_some() && new != this.idom[b.0 as usize] {
                    this.idom[b.0 as usize] = new;
                    changed = true;
                }
            }
        }
        this
    }

    fn intersect(&self, mut a: BlockId, mut b: BlockId) -> BlockId {
        while a != b {
            let an = self.rpo_num[a.0 as usize];
            let bn = self.rpo_num[b.0 as usize];
            while an > bn {
                match self.idom[a.0 as usize] {
                    Some(d) => a = d,
                    None => return b,
                }
            }
            let an = self.rpo_num[a.0 as usize];
            let bn = self.rpo_num[b.0 as usize];
            while bn > an {
                match self.idom[b.0 as usize] {
                    Some(d) => b = d,
                    None => return a,
                }
            }
        }
        a
    }

    /// True when `a` dominates `b` (`a == b` counts).
    fn dominates(&self, a: BlockId, b: BlockId) -> bool {
        if a == b {
            return true;
        }
        if self.rpo_num.get(a.0 as usize).copied().unwrap_or(u32::MAX) == u32::MAX {
            return false;
        }
        let mut cur = b;
        while let Some(d) = self.idom.get(cur.0 as usize).copied().flatten() {
            if d == a {
                return true;
            }
            cur = d;
        }
        false
    }
}

// ---------------------------------------------------------------------------
// post-dominators
// ---------------------------------------------------------------------------

fn intersect_num(mut a: usize, mut b: usize, idom: &[usize], num: &[usize]) -> usize {
    // A node outside the reverse-postorder (or one whose chain has not been
    // established yet) cannot be intersected: keep the other.
    let valid = |x: usize| num.get(x).copied().unwrap_or(usize::MAX) != usize::MAX;
    while a != b {
        while valid(a) && valid(b) && num[a] > num[b] {
            let next = idom[a];
            if next == usize::MAX || !valid(next) {
                return b;
            }
            a = next;
        }
        while valid(a) && valid(b) && num[b] > num[a] {
            let next = idom[b];
            if next == usize::MAX || !valid(next) {
                return a;
            }
            b = next;
        }
        if !valid(a) || !valid(b) || (num[a] <= num[b] && num[b] <= num[a]) {
            break;
        }
    }
    a
}

/// Immediate post-dominator of each block (`None` when the chain reaches the
/// virtual exit or the block cannot reach one). Cooper-Harvey-Kennedy on the
/// reverse CFG with a synthetic exit node.
fn postdominators(f: &Function) -> Vec<Option<BlockId>> {
    let n = f.blocks.len();
    let mut out = vec![None; n];
    if n == 0 {
        return out;
    }
    let succs: Vec<Vec<BlockId>> = (0..n).map(|i| f.blocks[i].term.as_ref().map(successor_blocks).unwrap_or_default()).collect();
    let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];
    for i in 0..n {
        for s in &succs[i] {
            preds[s.0 as usize].push(i);
        }
    }
    let root = n;
    let exits: Vec<usize> = (0..n).filter(|i| succs[*i].is_empty()).collect();
    // Reverse graph: rsucc[b] = CFG predecessors, rsucc[root] = exits.
    let mut rsucc: Vec<Vec<usize>> = vec![Vec::new(); n + 1];
    for i in 0..n {
        rsucc[i] = preds[i].clone();
    }
    rsucc[root] = exits.clone();
    let mut rpred: Vec<Vec<usize>> = vec![Vec::new(); n + 1];
    for i in 0..n {
        rpred[i] = succs[i].iter().map(|b| b.0 as usize).collect();
    }
    for e in &exits {
        rpred[*e].push(root);
    }
    // Reverse postorder from the virtual exit.
    let mut rpo: Vec<usize> = Vec::new();
    let mut seen = vec![false; n + 1];
    let mut stack: Vec<(usize, usize)> = vec![(root, 0)];
    seen[root] = true;
    while let Some((b, k)) = stack.pop() {
        if k < rsucc[b].len() {
            stack.push((b, k + 1));
            let s = rsucc[b][k];
            if !seen[s] {
                seen[s] = true;
                stack.push((s, 0));
            }
        } else {
            rpo.push(b);
        }
    }
    rpo.reverse();
    let mut numv = vec![usize::MAX; n + 1];
    for (i, b) in rpo.iter().enumerate() {
        numv[*b] = i;
    }
    let mut idom = vec![usize::MAX; n + 1];
    idom[root] = root;
    let mut changed = true;
    while changed {
        changed = false;
        for &b in rpo.iter().skip(1) {
            let mut new = usize::MAX;
            for &p in &rpred[b] {
                if numv[p] == usize::MAX {
                    continue;
                }
                new = if new == usize::MAX {
                    p
                } else {
                    intersect_num(p, new, &idom, &numv)
                };
            }
            if new != usize::MAX && new != idom[b] {
                idom[b] = new;
                changed = true;
            }
        }
    }
    for i in 0..n {
        let d = idom[i];
        if d != usize::MAX && d != root {
            out[i] = Some(BlockId(d as u32));
        }
    }
    out
}

// ---------------------------------------------------------------------------
// natural loops (biara_ir has no LoopInfo)
// ---------------------------------------------------------------------------

struct Loop {
    header: BlockId,
    blocks: HashSet<BlockId>,
    exits: Vec<BlockId>,
    latches: Vec<BlockId>,
    preheader: Option<BlockId>,
}

struct LoopSet {
    loops: Vec<Loop>,
    of_block: HashMap<BlockId, usize>,
}

impl LoopSet {
    fn build(f: &Function, dom: &Dom) -> Self {
        let n = f.blocks.len();
        let succs: Vec<Vec<BlockId>> = (0..n).map(|i| f.blocks[i].term.as_ref().map(successor_blocks).unwrap_or_default()).collect();
        let mut preds: Vec<Vec<BlockId>> = vec![Vec::new(); n];
        for i in 0..n {
            for s in &succs[i] {
                preds[s.0 as usize].push(BlockId(i as u32));
            }
        }
        let mut back: BTreeMap<BlockId, Vec<BlockId>> = BTreeMap::new();
        for i in 0..n {
            let from = BlockId(i as u32);
            for to in &succs[i] {
                if dom.dominates(*to, from) {
                    back.entry(*to).or_default().push(from);
                }
            }
        }
        let mut loops = Vec::new();
        for (header, latches) in back {
            let mut blocks: HashSet<BlockId> = HashSet::new();
            blocks.insert(header);
            let mut stack = latches.clone();
            while let Some(x) = stack.pop() {
                if !blocks.insert(x) {
                    continue;
                }
                if let Some(ps) = preds.get(x.0 as usize) {
                    for p in ps {
                        if !blocks.contains(p) {
                            stack.push(*p);
                        }
                    }
                }
            }
            let mut exits: Vec<BlockId> = Vec::new();
            for bi in 0..n {
                let b = BlockId(bi as u32);
                if !blocks.contains(&b) {
                    continue;
                }
                for s in &succs[bi] {
                    if !blocks.contains(s) && !exits.contains(s) {
                        exits.push(*s);
                    }
                }
            }
            let outside: Vec<BlockId> = preds[header.0 as usize]
                .iter()
                .copied()
                .filter(|p| !blocks.contains(p))
                .collect();
            let preheader = if outside.len() == 1 {
                Some(outside[0])
            } else {
                None
            };
            loops.push(Loop {
                header,
                blocks,
                exits,
                latches,
                preheader,
            });
        }
        let mut of_block: HashMap<BlockId, usize> = HashMap::new();
        for (i, l) in loops.iter().enumerate() {
            for b in &l.blocks {
                match of_block.get(b) {
                    Some(&prev) if loops[prev].blocks.len() <= l.blocks.len() => {}
                    _ => {
                        of_block.insert(*b, i);
                    }
                }
            }
        }
        Self { loops, of_block }
    }

    /// Innermost loop containing `b`.
    fn loop_for(&self, b: BlockId) -> Option<usize> {
        self.of_block.get(&b).copied()
    }

    fn is_header(&self, b: BlockId) -> bool {
        self.loops.iter().any(|l| l.header == b)
    }

    fn unique_exit(&self, i: usize) -> Option<BlockId> {
        let l = &self.loops[i];
        if l.exits.len() == 1 {
            Some(l.exits[0])
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// import prototypes
// ---------------------------------------------------------------------------

struct ImportPrototype {
    name: &'static str,
    result: &'static str,
    args: &'static [&'static str],
    variadic: bool,
    posix_only: bool,
}

const fn proto(
    name: &'static str,
    result: &'static str,
    args: &'static [&'static str],
    variadic: bool,
    posix_only: bool,
) -> ImportPrototype {
    ImportPrototype {
        name,
        result,
        args,
        variadic,
        posix_only,
    }
}

/// The parameter count of a C library function biara knows the signature
/// of, so the ABI form can read exactly that many argument registers.
pub(crate) fn known_proto_arity(name: &str) -> Option<u32> {
    KNOWN_PROTOS
        .iter()
        .find(|p| p.name == name && !p.variadic)
        .map(|p| p.args.len() as u32)
}

static KNOWN_PROTOS: &[ImportPrototype] = &[
    proto("malloc", "uint8_t *", &["uint64_t"], false, false),
    proto("calloc", "uint8_t *", &["uint64_t", "uint64_t"], false, false),
    proto("realloc", "uint8_t *", &["uint8_t *", "uint64_t"], false, false),
    proto(
        "aligned_alloc",
        "uint8_t *",
        &["uint64_t", "uint64_t"],
        false,
        false,
    ),
    proto("free", "void", &["uint8_t *"], false, false),
    proto(
        "memcpy",
        "uint8_t *",
        &["uint8_t *", "const uint8_t *", "uint64_t"],
        false,
        false,
    ),
    proto(
        "memmove",
        "uint8_t *",
        &["uint8_t *", "const uint8_t *", "uint64_t"],
        false,
        false,
    ),
    proto(
        "memset",
        "uint8_t *",
        &["uint8_t *", "int32_t", "uint64_t"],
        false,
        false,
    ),
    proto("strlen", "uint64_t", &["const char *"], false, false),
    proto("atoi", "int32_t", &["const char *"], false, false),
    proto("getenv", "char *", &["const char *"], false, false),
    proto("strchr", "char *", &["const char *", "int32_t"], false, false),
    proto("strstr", "char *", &["const char *", "const char *"], false, false),
    proto("memcmp", "int32_t", &["const void *", "const void *", "uint64_t"], false, false),
    proto(
        "strcmp",
        "int32_t",
        &["const char *", "const char *"],
        false,
        false,
    ),
    proto(
        "strncmp",
        "int32_t",
        &["const char *", "const char *", "uint64_t"],
        false,
        false,
    ),
    proto("puts", "int32_t", &["const char *"], false, false),
    proto("putchar", "int32_t", &["int32_t"], false, false),
    proto("putc", "int32_t", &["int32_t", "uint8_t *"], false, false),
    proto("printf", "int32_t", &["const char *"], true, false),
    proto(
        "fprintf",
        "int32_t",
        &["uint8_t *", "const char *"],
        true,
        false,
    ),
    proto(
        "snprintf",
        "int32_t",
        &["char *", "uint64_t", "const char *"],
        true,
        false,
    ),
    proto(
        "read",
        "int64_t",
        &["int32_t", "uint8_t *", "uint64_t"],
        false,
        true,
    ),
    proto(
        "write",
        "int64_t",
        &["int32_t", "const uint8_t *", "uint64_t"],
        false,
        true,
    ),
    proto("atol", "int64_t", &["const char *"], false, true),
    proto(
        "strtol",
        "int64_t",
        &["const char *", "char **", "int32_t"],
        false,
        true,
    ),
    proto(
        "strtoll",
        "int64_t",
        &["const char *", "char **", "int32_t"],
        false,
        true,
    ),
    proto(
        "__isoc23_strtol",
        "int64_t",
        &["const char *", "char **", "int32_t"],
        false,
        true,
    ),
    proto(
        "__isoc23_strtoll",
        "int64_t",
        &["const char *", "char **", "int32_t"],
        false,
        true,
    ),
];

// ---------------------------------------------------------------------------
// lexer over emitted C (statement metrics / package dependencies)
// ---------------------------------------------------------------------------

#[derive(Default)]
struct CText {
    tokens: u64,
    casts: u64,
    gotos: u64,
    identifiers: BTreeSet<String>,
}

fn is_type_token(t: &[u8]) -> bool {
    matches!(
        t,
        b"bool"
            | b"char"
            | b"void"
            | b"float"
            | b"double"
            | b"int8_t"
            | b"int16_t"
            | b"int32_t"
            | b"int64_t"
            | b"uint8_t"
            | b"uint16_t"
            | b"uint32_t"
            | b"uint64_t"
            | b"uintptr_t"
            | b"biara_u128"
            | b"biara_i128"
    )
}

#[derive(Clone)]
struct CalleeRef {
    name: String,
    decl: bool,
    ty: FuncType,
}

// ---------------------------------------------------------------------------
// the emitter
// ---------------------------------------------------------------------------

struct Live {
    sites: Vec<(BlockId, Option<ValueId>)>,
    in_blocks: HashSet<BlockId>,
}

struct LoopCx {
    l: usize,
    /// its single exit, emitted after it
    follow: Option<BlockId>,
    /// switches opened inside it (`break` is theirs)
    switches: u32,
    induction: Option<ValueId>,
    do_latch: Option<BlockId>,
    footer: Option<Stmt>,
}

struct Emitter<'m, 'a, 'b> {
    module: &'m Module,
    opts: &'a Options<'b>,
    unsupported: usize,

    strings: BTreeMap<u64, String>,
    string_uses: BTreeMap<u64, u32>,
    syms: BTreeMap<u64, String>,
    protos: BTreeMap<String, Proto>,
    posix: bool,
    data: Vec<DataSym>,

    helpers: BTreeMap<String, bool>,
    support_defs: BTreeMap<String, String>,
    aggregates: BTreeMap<Vec<Type>, String>,
    callback_types: HashMap<(usize, ValueId), String>,
    callback_arguments: BTreeMap<String, Vec<String>>,
    prelude: String,
    global_names: BTreeSet<String>,
    data_used: BTreeMap<String, bool>,
    strs_used: BTreeMap<u64, bool>,
    atol_used: bool,
    atol_declared: bool,

    emitted_functions: BTreeMap<String, u64>,
    emitted_globals: BTreeMap<String, u64>,
    emitted_types: BTreeMap<String, u32>,

    calls_by_name: HashMap<String, Vec<(usize, ValueId)>>,
    func_refs: HashMap<String, u32>,

    current_fn: usize,
    names: HashMap<ValueId, String>,
    originals: HashMap<ValueId, String>,
    name_evidence: HashMap<ValueId, &'static str>,
    taken: BTreeSet<String>,
    next_suffix: BTreeMap<String, u32>,
    inline: HashSet<ValueId>,
    canon_map: HashMap<ValueId, ValueId>,
    /// Per-value C type spelling, fixed once per function so folded
    /// expressions and declarations agree on aggregate names.
    value_cty: HashMap<(usize, ValueId), String>,
    order: HashMap<BlockId, u32>,
    targeted: HashSet<BlockId>,

    statement_count: u64,
    mapped_count: u64,
    max_depth: u64,
    temp_count: u64,
    packaging: bool,
    statement_json: String,

    cur_refs: Option<Vec<ValueId>>,
    cur_pcs: Option<BTreeSet<u64>>,
    cur_lost: bool,

    facts: Facts,
    dom: Option<Dom>,
    pdom: Vec<Option<BlockId>>,
    loops: LoopSet,
    live_cache: HashMap<ValueId, Live>,

    out: Vec<Stmt>,
    depth: u32,
    emitted: HashSet<BlockId>,
    pending: Vec<BlockId>,
    loop_stack: Vec<LoopCx>,
}

impl<'m, 'a, 'b> Emitter<'m, 'a, 'b> {
    fn new(module: &'m Module, opts: &'a Options<'b>) -> Self {
        let mut strings = BTreeMap::new();
        for s in opts.strings {
            strings.insert(s.addr, s.text.clone());
        }
        let mut syms = BTreeMap::new();
        for s in opts.syms {
            syms.insert(s.addr, s.text.clone());
        }
        let mut protos = BTreeMap::new();
        for p in opts.protos {
            protos.insert(
                p.name.clone(),
                Proto {
                    name: p.name.clone(),
                    nargs: p.nargs,
                    strmask: p.strmask,
                    ellipsis: p.ellipsis,
                },
            );
        }
        let posix = module
            .functions
            .iter()
            .find_map(|f| f.attrs.os)
            .or_else(|| module.decls.iter().find_map(|d| d.attrs.os))
            .map(|os| os == 1 || os == 2)
            .unwrap_or(false);
        let mut global_names: BTreeSet<String> = [
            "uint8_t",
            "uint16_t",
            "uint32_t",
            "uint64_t",
            "int8_t",
            "int16_t",
            "int32_t",
            "int64_t",
            "uintptr_t",
            "biara_u128",
            "biara_i128",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();

        let mut calls_by_name: HashMap<String, Vec<(usize, ValueId)>> = HashMap::new();
        let mut func_refs: HashMap<String, u32> = HashMap::new();
        let mut global_refs: HashMap<String, u32> = HashMap::new();
        for (fi, f) in module.functions.iter().enumerate() {
            for b in &f.blocks {
                for &v in &b.insts {
                    let Some(inst) = f.inst(v) else { continue };
                    for o in operands(inst) {
                        match f.value(o) {
                            Value::Const(Constant::Function(n)) => {
                                *func_refs.entry(n.clone()).or_insert(0) += 1
                            }
                            Value::Const(Constant::GlobalPtr(n)) => {
                                *global_refs.entry(n.clone()).or_insert(0) += 1
                            }
                            _ => {}
                        }
                    }
                    if let Inst::Call { callee, args, .. } = inst {
                        if let Value::Const(Constant::Function(n)) = f.value(*callee) {
                            calls_by_name.entry(n.clone()).or_default().push((fi, v));
                            let _ = args;
                        }
                    }
                }
            }
        }
        for g in &module.globals {
            if let Some(init) = &g.init {
                count_const_refs(init, &mut func_refs, &mut global_refs);
            }
        }
        let _ = &mut global_names;
        let data = opts.data.to_vec();
        let mut e = Self {
            module,
            opts,
            unsupported: 0,
            strings,
            string_uses: BTreeMap::new(),
            syms,
            protos,
            posix,
            data,
            helpers: BTreeMap::new(),
            support_defs: BTreeMap::new(),
            aggregates: BTreeMap::new(),
            callback_types: HashMap::new(),
            callback_arguments: BTreeMap::new(),
            prelude: String::new(),
            global_names,
            data_used: BTreeMap::new(),
            strs_used: BTreeMap::new(),
            atol_used: false,
            atol_declared: false,
            emitted_functions: BTreeMap::new(),
            emitted_globals: BTreeMap::new(),
            emitted_types: BTreeMap::new(),
            calls_by_name,
            func_refs,
            current_fn: 0,
            names: HashMap::new(),
            originals: HashMap::new(),
            name_evidence: HashMap::new(),
            taken: BTreeSet::new(),
            next_suffix: BTreeMap::new(),
            inline: HashSet::new(),
            canon_map: HashMap::new(),
            value_cty: HashMap::new(),
            order: HashMap::new(),
            targeted: HashSet::new(),
            statement_count: 0,
            mapped_count: 0,
            max_depth: 0,
            temp_count: 0,
            packaging: false,
            statement_json: String::new(),
            cur_refs: None,
            cur_pcs: None,
            cur_lost: false,
            facts: Facts {
                uses: Vec::new(),
                inst_users: Vec::new(),
                term_uses: Vec::new(),
                pos: HashMap::new(),
                preds: Vec::new(),
                succs: Vec::new(),
            },
            dom: None,
            pdom: Vec::new(),
            loops: LoopSet {
                loops: Vec::new(),
                of_block: HashMap::new(),
            },
            live_cache: HashMap::new(),
            out: Vec::new(),
            depth: 1,
            emitted: HashSet::new(),
            pending: Vec::new(),
            loop_stack: Vec::new(),
        };
        e.infer_callbacks();
        if !e.strings.is_empty() {
            for f in &module.functions {
                for b in &f.blocks {
                    for &v in &b.insts {
                        let Some(inst) = f.inst(v) else { continue };
                        for o in operands(inst) {
                            if let Value::Const(c) = f.value(o) {
                                e.count_string_uses(c, true);
                            }
                        }
                    }
                    for o in b.term.as_ref().map(biara_ir::term_operands).unwrap_or_default() {
                        if let Value::Const(c) = f.value(o) {
                            e.count_string_uses(c, true);
                        }
                    }
                }
            }
            for g in &module.globals {
                if let Some(init) = &g.init {
                    e.count_string_uses(init, false);
                }
            }
        }
        e
    }

    // -- naming ----------------------------------------------------------

    fn declare_name(&mut self, v: ValueId, hint: &str, original: String) {
        let base = if hint.is_empty() {
            "value".to_string()
        } else {
            ident(hint)
        };
        let mut name = base.clone();
        let collides = |e: &Self, c: &str| {
            e.taken.contains(c)
                || e.global_names.contains(c)
                || e.emitted_functions.contains_key(c)
                || e.emitted_globals.contains_key(c)
                || e.emitted_types.contains_key(c)
        };
        if collides(self, &name) {
            let mut next = *self.next_suffix.get(&base).unwrap_or(&1);
            loop {
                name = format!("{}_{}", base, next);
                next += 1;
                if !collides(self, &name) {
                    break;
                }
            }
            self.next_suffix.insert(base, next);
        }
        self.taken.insert(name.clone());
        self.names.insert(v, name);
        self.originals.insert(v, original);
        self.name_evidence.insert(v, "unknown");
    }

    fn argument_names(&mut self, f: &Function) {
        for (i, &p) in f.params.iter().enumerate() {
            let original = match f.value(p) {
                Value::Param { name, .. } => name.clone(),
                _ => String::new(),
            };
            if let Some(debug_name) = f.attrs.debug_param_names.get(&original) {
                self.declare_name(p, debug_name, original);
                self.name_evidence.insert(p, "debug");
            } else {
                self.declare_name(p, &format!("arg{}", i), original);
            }
        }
    }

    fn name(&self, v: ValueId) -> String {
        self.names.get(&v).cloned().unwrap_or_default()
    }

    fn canon(&self, v: ValueId) -> ValueId {
        self.canon_map.get(&v).copied().unwrap_or(v)
    }

    // -- provenance ------------------------------------------------------

    fn begin(&mut self) {
        self.cur_refs = Some(Vec::new());
        self.cur_pcs = Some(BTreeSet::new());
        self.cur_lost = false;
    }

    fn capture_pc(&mut self, f: &Function, v: ValueId) {
        if !self.packaging {
            return;
        }
        if self.cur_pcs.is_none() {
            return;
        }
        if !matches!(f.value(v), Value::Inst(_)) {
            return;
        }
        match f.pc_of(v) {
            Some(pc) => {
                if let Some(s) = self.cur_pcs.as_mut() {
                    s.insert(pc);
                }
            }
            None => self.cur_lost = true,
        }
    }

    /// A terminator has no PC in `biara_ir`: record the loss rather than
    /// inventing an address for the statement it produces.
    fn note_terminator(&mut self) {
        if self.packaging {
            self.cur_lost = true;
        }
    }

    fn end_stmt(&mut self, s: &mut Stmt) {
        s.refs = self.cur_refs.take().unwrap_or_default();
        s.pcs = self.cur_pcs.take().unwrap_or_default();
        s.lost_fold = self.cur_lost;
        self.cur_lost = false;
    }

    // -- type names ------------------------------------------------------

    fn aggregate(&mut self, fields: &[Type]) -> String {
        if let Some(n) = self.aggregates.get(fields) {
            return n.clone();
        }
        let name = format!("struct biara_tuple{}", self.aggregates.len() + 1);
        let mut def = format!("{} {{", name);
        for (i, t) in fields.iter().enumerate() {
            let ty = self.cty(t);
            def.push_str(&format!(" {} f{};", ty, i));
        }
        def.push_str(" };\n");
        self.prelude.push_str(&def);
        self.support_defs
            .insert(name[7..].to_string(), def.clone());
        self.aggregates.insert(fields.to_vec(), name.clone());
        name
    }

    fn cty(&mut self, ty: &Type) -> String {
        match ty {
            Type::Void => "void".into(),
            Type::Int(bits) => {
                let u = uint_ty(*bits);
                if u.is_empty() {
                    self.unsupported += 1;
                    "uint64_t /* unsupported type */".into()
                } else {
                    u
                }
            }
            Type::Float(32) => "float".into(),
            Type::Float(64) => "double".into(),
            Type::Ptr => "uint8_t *".into(),
            Type::Struct(fields) => self.aggregate(fields),
            Type::Vector(elem, n) => {
                let bits = match **elem {
                    Type::Int(b) | Type::Float(b) => b,
                    Type::Ptr => 64,
                    _ => 0,
                };
                let bytes = bits * *n / 8;
                if matches!(bits, 8 | 16 | 32 | 64)
                    && bytes != 0
                    && bytes <= 16
                    && (bytes & (bytes - 1)) == 0
                {
                    let name = match **elem {
                        Type::Float(32) => format!("biara_v{n}f"),
                        Type::Float(64) => format!("biara_v{n}d"),
                        _ => format!("biara_v{n}u{bits}"),
                    };
                    if !self.support_defs.contains_key(&name) {
                        let elem_ty = self.cty(elem);
                        let def = format!(
                            "typedef {} {} __attribute__((vector_size({})));\n",
                            elem_ty, name, bytes
                        );
                        self.support_defs.insert(name.clone(), def.clone());
                        self.prelude.push_str(&def);
                    }
                    name
                } else {
                    self.unsupported += 1;
                    "uint64_t /* unsupported type */".into()
                }
            }
            Type::Array(..) | Type::Func(_) | Type::Float(_) => {
                self.unsupported += 1;
                "uint64_t /* unsupported type */".into()
            }
        }
    }

    fn callback_type(&self, v: ValueId) -> Option<String> {
        self.callback_types
            .get(&(self.current_fn, v))
            .cloned()
    }

    fn declared_param(&self, attrs: &FuncAttrs, nparams: usize, i: usize) -> Option<String> {
        attrs.c_params.get(&(i as u32)).cloned().or_else(|| {
            (nparams == 1 && i == 0).then(|| attrs.c_type.clone()).flatten()
        })
    }

    fn cty_of(&self, f: &Function, v: ValueId) -> String {
        if let Some(t) = self.callback_type(v) {
            return t;
        }
        if let Value::Param { index, .. } = f.value(v) {
            if let Some(d) = self.declared_param(&f.attrs, f.params.len(), *index as usize) {
                if !d.is_empty() {
                    return d;
                }
            }
        }
        match self.value_cty.get(&(self.current_fn, v)) {
            Some(t) => t.clone(),
            None => cty_static(&f.ty(v)),
        }
    }

    fn declared_return(&self, attrs: &FuncAttrs) -> Option<String> {
        attrs.c_return.clone().filter(|s| !s.is_empty())
    }

    fn declared_signed(s: &str) -> bool {
        let t = s.trim();
        matches!(
            t,
            "int8_t"
                | "int16_t"
                | "int32_t"
                | "int64_t"
                | "biara_i128"
                | "int"
                | "signed"
                | "signed int"
                | "short"
                | "signed short"
                | "short int"
                | "long"
                | "signed long"
                | "long int"
                | "long long"
                | "signed long long"
                | "char"
                | "signed char"
        )
    }

    fn intrinsic_fn(&mut self, base: &str, bits: u32) -> String {
        let n = format!("biara_{}{}", base, bits);
        self.helpers.entry(n.clone()).or_insert(false);
        n
    }
}

fn count_const_refs(
    c: &Constant,
    func_refs: &mut HashMap<String, u32>,
    global_refs: &mut HashMap<String, u32>,
) {
    match c {
        Constant::Function(n) => *func_refs.entry(n.clone()).or_insert(0) += 1,
        Constant::GlobalPtr(n) => *global_refs.entry(n.clone()).or_insert(0) += 1,
        Constant::Aggregate(items) => {
            for i in items {
                count_const_refs(i, func_refs, global_refs);
            }
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// address reduction: root plus constant offset
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Root {
    /// a pointer variable (parameter, load, call result, phi, select)
    Value(ValueId),
    /// the CPU state global
    Cpu,
    /// nothing: an absolute address
    Abs,
}

#[derive(Clone, Copy, Debug)]
struct Base {
    root: Root,
    off: i64,
}

impl<'m, 'a, 'b> Emitter<'m, 'a, 'b> {
    fn has_cpu_global(&self) -> bool {
        self.module.globals.iter().any(|g| g.name == "biara_cpu")
    }

    fn base_depth(&self, f: &Function, v: ValueId, depth: u32) -> Base {
        if depth > 32 {
            return Base {
                root: Root::Value(v),
                off: 0,
            };
        }
        match f.value(v) {
            Value::Const(Constant::Int { bits, width }) if *width <= 64 => Base {
                root: Root::Abs,
                off: (*bits & width_mask(*width)) as u64 as i64,
            },
            Value::Const(Constant::GlobalPtr(name)) if name == "biara_cpu" => Base {
                root: Root::Cpu,
                off: 0,
            },
            Value::Inst(Inst::Gep { base, offset }) => {
                let mut b = self.base_depth(f, *base, depth + 1);
                b.off = b.off.wrapping_add(*offset);
                b
            }
            Value::Inst(Inst::Cast { from, value, .. })
                if matches!(from, Type::Ptr) || matches!(from, Type::Int(64)) =>
            {
                self.base_depth(f, *value, depth + 1)
            }
            Value::Inst(Inst::Bin { op, ty, lhs, rhs })
                if matches!(op, BinOp::Add | BinOp::Sub) && matches!(ty, Type::Int(64)) =>
            {
                match f.value(*rhs) {
                    Value::Const(Constant::Int { bits, width }) if *width == 64 => {
                        let mut b = self.base_depth(f, *lhs, depth + 1);
                        let k = sign_extend(*bits, 64) as i64;
                        b.off = b.off.wrapping_add(if *op == BinOp::Sub {
                            k.wrapping_neg()
                        } else {
                            k
                        });
                        b
                    }
                    _ => Base {
                        root: Root::Value(v),
                        off: 0,
                    },
                }
            }
            _ => Base {
                root: Root::Value(v),
                off: 0,
            },
        }
    }

    fn base(&self, f: &Function, v: ValueId) -> Base {
        self.base_depth(f, v, 0)
    }

    fn is_alloca(&self, f: &Function, v: ValueId) -> bool {
        matches!(f.inst(v), Some(Inst::Alloca { .. }))
    }

    fn is_frame_root(&self, f: &Function, r: Root) -> bool {
        match r {
            Root::Value(v) => self.is_alloca(f, v),
            Root::Cpu => self.has_cpu_global(),
            Root::Abs => false,
        }
    }

    /// Address `off` bytes into frame/CPU root `r`, as a pointer.
    fn frame_addr(&self, f: &Function, r: Root, off: i64) -> String {
        let var = match r {
            Root::Value(v) if self.is_alloca(f, v) => self.name(v),
            _ => "biara_cpu".to_string(),
        };
        if r == Root::Cpu && off == self.opts.margin as i64 {
            return "BIARA_ENV".to_string();
        }
        if off == 0 {
            return format!("(uint8_t *)&{}", var);
        }
        format!("((uint8_t *)&{} + {})", var, num(off))
    }

    /// Copied data holding `off` (whole access inside one symbol), if any.
    fn data_at(&self, off: u64, bytes: u32) -> Option<&DataSym> {
        if bytes == 0 {
            return None;
        }
        self.data.iter().find(|d| {
            d.size != 0 && d.addr <= off && off - d.addr + bytes as u64 <= d.size
        })
    }

    /// The lvalue for data `name` at `delta` bytes in, and its C type.
    fn deref(
        &mut self,
        name: &str,
        delta: u64,
        size: u64,
        bytes: u32,
        raw: &str,
    ) -> (String, String) {
        self.data_used.entry(name.to_string()).or_insert(false);
        let sym = ident(name);
        if delta == 0 && bytes == 8 && size == 8 {
            return (sym, "uint8_t *".to_string());
        }
        let mut p = format!("((uint8_t *)&{}", sym);
        if delta != 0 {
            p.push_str(&format!(" + {})", hexv(delta)));
        } else {
            p.push(')');
        }
        (format!("*({} *){}", raw, p), raw.to_string())
    }

    /// The lvalue of type `ty` at address `ptr`, and its C type.
    fn resolve(&mut self, f: &Function, ptr: ValueId, ty: &Type) -> (String, String) {
        let b = self.base(f, ptr);
        let bytes = biara_ir::ty_bytes(ty);
        let raw = self.cty(ty);
        if self.is_frame_root(f, b.root) {
            let var = match b.root {
                Root::Value(v) if self.is_alloca(f, v) => self.name(v),
                _ => "biara_cpu".to_string(),
            };
            if b.off == 0 {
                return (format!("*({} *)&{}", raw, var), raw);
            }
            return (
                format!("*({} *){}", raw, self.frame_addr(f, b.root, b.off)),
                raw,
            );
        }
        if b.root == Root::Abs {
            if let Some(d) = self.data_at(b.off as u64, bytes) {
                let (name, size, addr) = (d.name.clone(), d.size, d.addr);
                return self.deref(&name, b.off as u64 - addr, size, bytes, &raw);
            }
            return (format!("*({} *){}", raw, hexv(b.off as u64)), raw);
        }
        let Root::Value(rv) = b.root else {
            unreachable!()
        };
        let declared = match f.value(rv) {
            Value::Param { index, .. } => {
                self.declared_param(&f.attrs, f.params.len(), *index as usize)
            }
            _ => None,
        };
        let pointee = declared
            .as_deref()
            .and_then(|spelling| spelling.trim().strip_suffix('*'))
            .map(str::trim_end);
        if let Some(pointee) = pointee
            && let Some(field) = self
                .opts
                .layouts
                .layouts
                .iter()
                .find(|layout| layout.name == pointee)
                .and_then(|layout| {
                    layout.fields.iter().find(|field| {
                        field.off == b.off && field.bytes == bytes && uint_ty(bytes * 8) == raw
                    })
                })
        {
            let member = field.name.clone();
            return (format!("{}->{member}", self.val(f, rv)), raw);
        }
        let p = self.val(f, rv);
        let off = if b.off == 0 {
            String::new()
        } else if b.off < 0 {
            format!(" - {}", hexv(b.off.unsigned_abs()))
        } else {
            format!(" + {}", hexv(b.off as u64))
        };
        if pointee.is_some() || matches!(f.ty(rv), Type::Ptr) {
            let p = if b.off != 0 && pointee.is_some_and(|ty| ty != "uint8_t") {
                format!("((uint8_t *)({p}))")
            } else {
                p
            };
            (
                format!("*({} *)({}{})", raw, p, off),
                raw,
            )
        } else {
            (
                format!("*({} *)(uintptr_t)({}{})", raw, p, off),
                raw,
            )
        }
    }

    fn mem(&mut self, f: &Function, ptr: ValueId, ty: &Type) -> String {
        self.resolve(f, ptr, ty).0
    }

    /// `resolve` without recording references.
    fn mem_ty(&mut self, f: &Function, ptr: ValueId, ty: &Type) -> String {
        let saved = self.cur_refs.take();
        let s = self.resolve(f, ptr, ty).1;
        self.cur_refs = saved;
        s
    }

    // -- values ----------------------------------------------------------

    fn val(&mut self, f: &Function, v: ValueId) -> String {
        self.capture_pc(f, v);
        match f.value(v).clone() {
            Value::Const(c) => self.val_const(f, &c),
            Value::Param { .. } => self.name(v),
            Value::Inst(inst) => {
                // Frame addresses print from the root, as pointers or integers.
                let ty = biara_ir::inst_ty(&inst, f);
                if matches!(ty, Type::Ptr | Type::Int(64)) {
                    let b = self.base(f, v);
                    if self.is_frame_root(f, b.root) {
                        let p = self.frame_addr(f, b.root, b.off);
                        return if matches!(ty, Type::Ptr) {
                            p
                        } else {
                            format!("(uint64_t){}", p)
                        };
                    }
                }
                if self.inline.contains(&v) {
                    return self.inline_expr(f, v);
                }
                if let Some(refs) = self.cur_refs.as_mut() {
                    refs.push(v);
                }
                self.name(v)
            }
        }
    }

    fn val_const(&mut self, f: &Function, c: &Constant) -> String {
        match c {
            Constant::Int { bits, width } => lit_v(*bits, *width),
            Constant::Zero => "(uint8_t *)0".to_string(),
            Constant::Undef => "0".to_string(),
            Constant::Function(name) => format!(
                "(uint8_t *)&{}",
                fn_name(name, self.module.decl(name).is_some())
            ),
            Constant::GlobalPtr(name) => format!("(uint8_t *)&{}", ident(name)),
            Constant::Aggregate(items) => {
                let fields: Vec<Type> = items.iter().map(|i| biara_ir::const_ty(i, f)).collect();
                let name = self.aggregate(&fields);
                let mut s = format!("({}){{", name);
                for (k, item) in items.iter().enumerate() {
                    if k > 0 {
                        s.push_str(", ");
                    }
                    s.push_str(&self.val_const(f, item));
                }
                s.push('}');
                s
            }
        }
    }

    /// An inlined instruction's expression, parenthesized unless it is
    /// already an atomic cast.
    fn inline_expr(&mut self, f: &Function, v: ValueId) -> String {
        let e = self.expr(f, v);
        let bytes = e.as_bytes();
        if bytes.len() > 2 && bytes[0] == b'(' {
            let mut depth: i32 = 0;
            let mut j = 0usize;
            for (i, &c) in bytes.iter().enumerate() {
                match c {
                    b'(' => depth += 1,
                    b')' => depth -= 1,
                    _ => {}
                }
                if depth == 0 {
                    j = i;
                    break;
                }
            }
            if j > 0 && j + 1 < bytes.len() {
                let inner = &e[1..j];
                let rest = &e[j + 1..];
                let atomic_cast = matches!(
                    inner,
                    "uint8_t"
                        | "uint16_t"
                        | "uint32_t"
                        | "uint64_t"
                        | "int8_t"
                        | "int16_t"
                        | "int32_t"
                        | "int64_t"
                        | "bool"
                        | "uint8_t *"
                        | "int8_t *"
                        | "uint16_t *"
                        | "int16_t *"
                        | "uint32_t *"
                        | "int32_t *"
                        | "uint64_t *"
                        | "int64_t *"
                        | "const char *"
                );
                if atomic_cast && !rest.is_empty() {
                    let r = rest.as_bytes();
                    let skip = if matches!(r[0], b'-' | b'!' | b'~') { 1 } else { 0 };
                    let mut rd: i32 = 0;
                    let mut binary = false;
                    for &c in r.iter().skip(skip) {
                        match c {
                            b'(' | b'[' => rd += 1,
                            b')' | b']' => rd -= 1,
                            _ => {}
                        }
                        if rd == 0 && b"+-*/%|&^?:<>=!".contains(&c) {
                            binary = true;
                            break;
                        }
                    }
                    if !binary {
                        return e;
                    }
                }
            }
        }
        format!("({})", e)
    }

    /// `val` converted to C type `to`.
    fn val_as(&mut self, f: &Function, v: ValueId, to: &str) -> String {
        let ty = f.ty(v);
        if is_ptr_ty(to) && matches!(ty, Type::Ptr | Type::Int(64)) {
            let b = self.base(f, v);
            if self.is_frame_root(f, b.root) {
                let p = self.frame_addr(f, b.root, b.off);
                return if to == "uint8_t *" {
                    p
                } else {
                    format!("({}){}", to, p)
                };
            }
        }
        if let Value::Const(Constant::Int { bits, .. }) = f.value(v) {
            if *bits == 0 && is_ptr_ty(to) {
                return "0".to_string();
            }
        }
        let from = self.cty_of(f, v);
        let e = self.val(f, v);
        conv(&e, &from, to)
    }

    /// Integer operations use unsigned bit transport, even when a formal has
    /// a declared signed or pointer C type.
    fn ival(&mut self, f: &Function, v: ValueId) -> String {
        let to = self.cty(&f.ty(v));
        self.val_as(f, v, &to)
    }

    fn sval(&mut self, f: &Function, v: ValueId) -> String {
        if let Value::Param { index, .. } = f.value(v) {
            if let Some(d) = self.declared_param(&f.attrs, f.params.len(), *index as usize) {
                if !d.is_empty() && Self::declared_signed(&d) {
                    return self.val(f, v);
                }
            }
        }
        let Some(bits) = f.ty(v).int_bits() else {
            return self.val(f, v);
        };
        // A folded truncation is one signed cast, of a constant its value.
        if self.inline.contains(&v) {
            if let Some(Inst::Trunc { to, value, .. }) = f.inst(v) {
                if let Some(w) = to.int_bits() {
                    if standard(w) {
                        if let Value::Const(Constant::Int { bits: cb, .. }) = f.value(*value) {
                            return slit(*cb & width_mask(w), container(w));
                        }
                        let e = self.ival(f, *value);
                        return format!("({}){}", int_ty(w), e);
                    }
                }
            }
        }
        if bits != 1 {
            if let Value::Const(Constant::Int { bits: cb, width }) = f.value(v) {
                return slit(*cb, (*width).min(128));
            }
        }
        let e = self.ival(f, v);
        if bits == 1 {
            return format!("(-(int64_t){})", e);
        }
        if standard(bits) {
            return format!("({}){}", int_ty(bits), e);
        }
        self.helpers.entry("sext".to_string()).or_insert(false);
        format!("biara_sext({}, {})", e, bits)
    }

    /// The left operand of a shift, which alone sets the shift's type.
    fn shifted(&mut self, f: &Function, a: ValueId, x: &str) -> String {
        if let Value::Const(Constant::Int { bits, width }) = f.value(a) {
            return suffixed(*bits, *width);
        }
        if self.inline.contains(&a) {
            // An inline expression can have a narrower C type than its IR
            // type, for example a sign extension written as an int32_t cast.
            if f.ty(a).int_bits().is_some_and(|bits| bits >= 64) {
                return format!("({})({})", self.cty(&f.ty(a)), x);
            }
            if let Some(Inst::Select { yes, no, .. }) = f.inst(a) {
                let consts = matches!(f.value(*yes), Value::Const(_))
                    && matches!(f.value(*no), Value::Const(_));
                if consts {
                    let ty = self.cty(&f.ty(a));
                    return format!("({})({})", ty, x);
                }
            }
        }
        x.to_string()
    }

    /// What a constant function address names: ` /* guest_main */`, else
    /// nothing.
    fn sym_comment(&self, f: &Function, v: ValueId) -> String {
        if let Value::Const(Constant::Int { bits, width }) = f.value(v) {
            if *width <= 64 {
                if let Some(s) = self.syms.get(&(*bits as u64)) {
                    return format!(" /* {} */", s);
                }
            }
        }
        String::new()
    }

    /// What a constant address holds: ` /* "read ..." */`, else nothing.
    fn str_comment(&self, f: &Function, v: ValueId) -> String {
        if let Value::Const(Constant::Int { bits, width }) = f.value(v) {
            if *width <= 64 {
                if let Some(s) = self.strings.get(&(*bits as u64)) {
                    return format!(" /* \"{}\" */", s);
                }
            }
        }
        String::new()
    }

    fn const_u64(&self, f: &Function, v: ValueId) -> Option<u64> {
        match f.value(v) {
            Value::Const(Constant::Int { bits, width }) if *width <= 64 => {
                Some((*bits & width_mask(*width)) as u64)
            }
            _ => None,
        }
    }

    fn const_u128(&self, f: &Function, v: ValueId) -> Option<(u128, u32)> {
        match f.value(v) {
            Value::Const(Constant::Int { bits, width }) => Some((*bits, *width)),
            Value::Const(Constant::Zero) => Some((0, 64)),
            _ => None,
        }
    }

    // -- comparisons and expressions -------------------------------------

    fn cmp(&mut self, f: &Function, pred: IntPred, ty: &Type, lhs: ValueId, rhs: ValueId) -> String {
        let mut pred = pred;
        let a = lhs;
        let b = rhs;
        // `x >= 1` is `x > 0`; a sign test `x > -1` is `x >= 0` (and swaps).
        // The right constant becomes `0`, spelled without a value of its own.
        let mut b_zero = false;
        if matches!(pred, IntPred::Uge | IntPred::Sge) {
            if let Some((k, w)) = self.const_u128(f, b) {
                if k == 1 && w != 1 {
                    b_zero = true;
                    pred = if pred == IntPred::Uge {
                        IntPred::Ugt
                    } else {
                        IntPred::Sgt
                    };
                }
            }
        }
        if matches!(pred, IntPred::Sgt | IntPred::Sle) {
            if let Some((k, w)) = self.const_u128(f, b) {
                if is_all_ones(k, w) {
                    b_zero = true;
                    pred = if pred == IntPred::Sgt {
                        IntPred::Sge
                    } else {
                        IntPred::Slt
                    };
                }
            }
        }
        if b_zero {
            return self.cmp_zero(f, pred, a);
        }
        let op = match pred {
            IntPred::Eq => "==",
            IntPred::Ne => "!=",
            IntPred::Ugt | IntPred::Sgt => ">",
            IntPred::Uge | IntPred::Sge => ">=",
            IntPred::Ult | IntPred::Slt => "<",
            _ => "<=",
        };
        let a_ptr = matches!(f.ty(a), Type::Ptr);
        let ca = self.cty_of(f, a);
        let cb = self.cty_of(f, b);
        if a_ptr || is_ptr_ty(&ca) || is_ptr_ty(&cb) {
            if let Some(k) = self.const_u64(f, b) {
                if k == 0 && is_ptr_ty(&ca) && (op == "==" || op == "!=") {
                    let av = self.val(f, a);
                    return format!("{} {} 0", av, op);
                }
            }
            let x = self.val_as(f, a, "uint64_t");
            let y = self.val_as(f, b, "uint64_t");
            return format!("{} {} {}", x, op, y);
        }
        let _ = ty;
        if pred.is_signed() {
            let x = self.sval(f, a);
            let y = self.sval(f, b);
            format!("{} {} {}", x, op, y)
        } else {
            let x = self.ival(f, a);
            let y = self.ival(f, b);
            format!("{} {} {}", x, op, y)
        }
    }

    /// A comparison whose right operand is the constant `0`.
    fn cmp_zero(&mut self, f: &Function, pred: IntPred, a: ValueId) -> String {
        let op = match pred {
            IntPred::Eq => "==",
            IntPred::Ne => "!=",
            IntPred::Ugt | IntPred::Sgt => ">",
            IntPred::Uge | IntPred::Sge => ">=",
            IntPred::Ult | IntPred::Slt => "<",
            _ => "<=",
        };
        if pred.is_signed() {
            let x = self.sval(f, a);
            format!("{} {} 0", x, op)
        } else {
            let x = self.ival(f, a);
            format!("{} {} 0", x, op)
        }
    }

    fn expr(&mut self, f: &Function, v: ValueId) -> String {
        self.capture_pc(f, v);
        let inst = match f.inst(v) {
            Some(i) => i.clone(),
            None => return self.name(v),
        };
        match inst {
            Inst::Bin { op, ty, lhs, rhs } => self.bin_expr(f, v, op, &ty, lhs, rhs),
            Inst::Un { op, ty, value } => self.un_expr(f, op, &ty, value),
            Inst::Rot { left, ty, value, amount } => {
                let bits = ty.int_bits().unwrap_or(0);
                if standard(bits) && bits >= 8 {
                    let base = if left { "rotl" } else { "rotr" };
                    let n = self.intrinsic_fn(base, bits);
                    let x = self.ival(f, value);
                    let a = self.ival(f, amount);
                    format!("{}({}, {})", n, x, a)
                } else {
                    self.unsupported += 1;
                    "0 /* unsupported rotate */".to_string()
                }
            }
            Inst::Icmp { pred, ty, lhs, rhs } => self.cmp(f, pred, &ty, lhs, rhs),
            Inst::Select { ty, cond, yes, no } => {
                if matches!(ty, Type::Int(1)) {
                    if let (Some((b1, _)), Some((b0, _))) =
                        (self.const_u128(f, yes), self.const_u128(f, no))
                    {
                        if b1 == 1 {
                            let c = self.ival(f, cond);
                            let n = self.ival(f, no);
                            return format!("{} || {}", c, n);
                        }
                        if b0 == 0 {
                            let c = self.ival(f, cond);
                            let y = self.ival(f, yes);
                            return format!("{} && {}", c, y);
                        }
                    }
                }
                let to = self.cty_of(f, v);
                let c = self.ival(f, cond);
                let y = self.val_as(f, yes, &to);
                let n = self.val_as(f, no, &to);
                format!("{} ? {} : {}", c, y, n)
            }
            Inst::Trunc { from, to, value } => self.cast_expr(f, v, "trunc", &from, &to, value),
            Inst::Zext { from, to, value } => self.cast_expr(f, v, "zext", &from, &to, value),
            Inst::Sext { from, to, value } => self.cast_expr(f, v, "sext", &from, &to, value),
            Inst::Cast { from, to, value } if matches!((&from, &to),
                (Type::Int(_), Type::Float(_) | Type::Vector(..))
                    | (Type::Float(_) | Type::Vector(..), Type::Int(_))) => {
                let from_type = self.cty(&from);
                let to_type = self.cty(&to);
                let value = self.val_as(f, value, &from_type);
                format!("((union {{ {from_type} source; {to_type} target; }}){{ .source = {value} }}).target")
            }
            Inst::Cast { to, value, .. } => {
                let toty = self.cty(&to);
                let fromty = self.cty_of(f, value);
                let _ = fromty;
                // Representation-preserving: inttoptr/ptrtoint/bitcast.
                if matches!(to, Type::Ptr) {
                    let target = self.cty_of(f, v);
                    self.val_as(f, value, &target)
                } else if f.ty(value) == to {
                    let target = self.cty_of(f, v);
                    self.val_as(f, value, &target)
                } else {
                    let _ = toty;
                    let target = self.cty_of(f, v);
                    self.val_as(f, value, &target)
                }
            }
            Inst::Freeze { value, .. } => {
                let target = self.cty_of(f, v);
                self.val_as(f, value, &target)
            }
            Inst::Gep { base, offset } => {
                let p = self.val_as(f, base, "uint8_t *");
                let target = self.cty_of(f, v);
                let mut s = format!("({}", p);
                if offset != 0 {
                    s.push_str(&format!(
                        " + {}",
                        if offset < 0 {
                            format!("- {}", hexv(offset.unsigned_abs()))
                        } else {
                            num(offset)
                        }
                    ));
                }
                s.push(')');
                conv(&s, "uint8_t *", &target)
            }
            Inst::ExtractValue { agg, index, ty } => {
                let a = self.val(f, agg);
                let _ = ty;
                format!("{}.f{}", a, index)
            }
            Inst::Load { ty, ptr, .. } => {
                let lv = self.mem(f, ptr, &ty);
                let lt = self.mem_ty(f, ptr, &ty);
                let target = self.cty_of(f, v);
                conv(&lv, &lt, &target)
            }
            Inst::Call { .. } => self.call(f, v),
            Inst::InsertValue { .. } => {
                self.unsupported += 1;
                "0 /* unsupported insertvalue */".to_string()
            }
            Inst::Alloca { .. } | Inst::Store { .. } | Inst::Marker { .. } | Inst::Fence
            | Inst::Phi { .. } => {
                self.unsupported += 1;
                "0 /* unsupported: not a value */".to_string()
            }
        }
    }

    fn bin_expr(
        &mut self,
        f: &Function,
        _v: ValueId,
        op: BinOp,
        ty: &Type,
        lhs: ValueId,
        rhs: ValueId,
    ) -> String {
        let bits = ty.int_bits().unwrap_or(0);
        let mut x = self.ival(f, lhs);
        let mut y = self.ival(f, rhs);
        let wide = if bits < 32 { "(uint32_t)" } else { "" };
        if matches!(op, BinOp::And | BinOp::Or | BinOp::Xor) && bits != 1 {
            // Bitwise operands print as hex where constant.
            if let Some((k, w)) = self.const_u128(f, lhs) {
                if w <= 64 {
                    x = hexv((k & width_mask(w)) as u64);
                }
            }
            if let Some((k, w)) = self.const_u128(f, rhs) {
                if w <= 64 {
                    y = hexv((k & width_mask(w)) as u64);
                }
            }
        }
        let k_rhs = self.const_u128(f, rhs);
        match op {
            BinOp::Add => {
                if let Some((k, w)) = k_rhs {
                    let s = sign_extend(k, w);
                    if s < 0 && bits > 1 && s != sign_extend(1u128 << (bits.min(128) - 1), bits.min(128)) {
                        let neg = (-s) as u128;
                        return wrap(&format!("{} - {}", x, lit_v(neg, w)), bits);
                    }
                }
                wrap(&format!("{} + {}", x, y), bits)
            }
            BinOp::Sub => wrap(&format!("{} - {}", x, y), bits),
            BinOp::Mul => wrap(&format!("{}{} * {}", wide, x, y), bits),
            BinOp::Shl => {
                let s = self.shifted(f, lhs, &x);
                wrap(&format!("{}{} << {}", wide, s, y), bits)
            }
            BinOp::Lshr => {
                let s = self.shifted(f, lhs, &x);
                format!("{} >> {}", s, y)
            }
            BinOp::Ashr => {
                let sx = self.sval(f, lhs);
                wrap(
                    &format!("({})({} >> {})", uint_ty(bits), sx, y),
                    bits,
                )
            }
            BinOp::Udiv => format!("{} / {}", x, y),
            BinOp::Urem => format!("{} % {}", x, y),
            BinOp::Sdiv => {
                let sx = self.sval(f, lhs);
                let sy = self.sval(f, rhs);
                wrap(
                    &format!("({})({} / {})", uint_ty(bits), sx, sy),
                    bits,
                )
            }
            BinOp::Srem => {
                let sx = self.sval(f, lhs);
                let sy = self.sval(f, rhs);
                wrap(
                    &format!("({})({} % {})", uint_ty(bits), sx, sy),
                    bits,
                )
            }
            BinOp::And => {
                let (m, other) = if let Some((mk, mw)) = k_rhs {
                    (Some((mk, mw)), Some(lhs))
                } else if let Some((mk, mw)) = self.const_u128(f, lhs) {
                    (Some((mk, mw)), Some(rhs))
                } else {
                    (None, None)
                };
                if let (Some((mk, mw)), Some(o)) = (m, other) {
                    if mw <= 64 {
                        let mask = mk & width_mask(mw);
                        let n = mask.count_ones();
                        // A low-bits mask is a narrowing cast.
                        if (n == 8 || n == 16 || n == 32) && bits > n && mask == width_mask(n) {
                            let oe = self.ival(f, o);
                            return wrap(&format!("({}){}", uint_ty(n), oe), bits);
                        }
                        if n >= 1 && n < bits && mask == width_mask(n) {
                            let oe = self.ival(f, o);
                            return wrap(
                                &format!("{} % {}", oe, lit_v(1u128 << n, bits)),
                                bits,
                            );
                        }
                    }
                }
                format!("{} & {}", x, y)
            }
            BinOp::Or => format!("{} | {}", x, y),
            BinOp::Xor => {
                if bits == 1 {
                    if let Some((k, _)) = k_rhs {
                        if k == 1 {
                            let o = rhs;
                            let _ = o;
                            // `!(a < b)` reads as `a >= b`.
                            if let Some(Inst::Icmp { pred, ty, lhs, rhs }) = f.inst(lhs) {
                                return self.cmp(f, pred.negated(), ty, *lhs, *rhs);
                            }
                            return format!("!{}", x);
                        }
                    }
                }
                if let Some((k, w)) = k_rhs {
                    if is_all_ones(k, w) {
                        return wrap(&format!("~{}", x), bits);
                    }
                }
                format!("{} ^ {}", x, y)
            }
        }
    }

    fn un_expr(&mut self, f: &Function, op: UnOp, ty: &Type, value: ValueId) -> String {
        let bits = ty.int_bits().unwrap_or(0);
        match op {
            UnOp::Not => {
                let x = self.ival(f, value);
                if bits == 1 {
                    format!("!{}", x)
                } else {
                    wrap(&format!("~{}", x), bits)
                }
            }
            UnOp::Neg => {
                let x = self.ival(f, value);
                wrap(&format!("({})0 - {}", uint_ty(bits), x), bits)
            }
            UnOp::Ctlz | UnOp::Cttz | UnOp::Ctpop | UnOp::Bswap => {
                if !standard(bits) || bits < 8 {
                    self.unsupported += 1;
                    return "0 /* unsupported bit operation */".to_string();
                }
                let name = match op {
                    UnOp::Ctlz => "clz",
                    UnOp::Cttz => "ctz",
                    UnOp::Ctpop => "popcount",
                    _ => "bswap",
                };
                let n = self.intrinsic_fn(name, bits);
                let x = self.ival(f, value);
                format!("{}({})", n, x)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn cast_expr(
        &mut self,
        f: &Function,
        _v: ValueId,
        kind: &str,
        from: &Type,
        to: &Type,
        value: ValueId,
    ) -> String {
        let _ = from;
        let target = self.cty(to);
        // A cast of a constant is the folded value (`2`, not `(int32_t)2u`).
        if let (Some(w), Some((cb, cw))) = (to.int_bits(), self.const_u128(f, value)) {
            let signed = int_ty(w).starts_with("int") || target == "biara_i128";
            match kind {
                "zext" => {
                    let z = cb & width_mask(cw);
                    return if signed {
                        slit(z, w)
                    } else {
                        lit_v(z, w)
                    };
                }
                "sext" => {
                    let s = sign_extend(cb, cw) as u128;
                    return if signed {
                        slit(s, w)
                    } else {
                        lit_v(s, w)
                    };
                }
                "trunc" => {
                    let t = cb & width_mask(w);
                    if w == 1 {
                        return if t & 1 == 1 { "true" } else { "false" }.to_string();
                    }
                    return if signed {
                        slit(t, w)
                    } else {
                        lit_v(t, w)
                    };
                }
                _ => {}
            }
        }
        match kind {
            "zext" => {
                let e = self.ival(f, value);
                format!("({}){}", target, e)
            }
            "sext" => {
                let bits = to.int_bits().unwrap_or(0);
                if let Some(Inst::Trunc { to: tw, value: tv, .. }) = f.inst(value) {
                    let w = tw.int_bits().unwrap_or(0);
                    if let Some((cb, cw)) = self.const_u128(f, *tv) {
                        let _ = cw;
                        let s = sign_extend(cb, w) as u128;
                        return wrap(&slit(s, bits), bits);
                    }
                    let e = self.ival(f, *tv);
                    return wrap(&format!("({}){}", int_ty(w), e), bits);
                }
                let e = self.sval(f, value);
                wrap(&format!("({}){}", target, e), bits)
            }
            "trunc" => {
                let bits = to.int_bits().unwrap_or(0);
                let e = self.ival(f, value);
                if bits == 1 {
                    return format!("(bool)({} & 1)", e);
                }
                wrap(&format!("({}){}", target, e), bits)
            }
            _ => {
                self.val_as(f, value, &target)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// calls
// ---------------------------------------------------------------------------

impl<'m, 'a, 'b> Emitter<'m, 'a, 'b> {
    fn fn_ref(&self, name: &str) -> Option<CalleeRef> {
        if let Some(f) = self.module.function(name) {
            return Some(CalleeRef {
                name: name.to_string(),
                decl: f.entry.is_none() && f.blocks.is_empty(),
                ty: f.ty.clone(),
            });
        }
        if let Some(d) = self.module.decl(name) {
            return Some(CalleeRef {
                name: name.to_string(),
                decl: true,
                ty: d.ty.clone(),
            });
        }
        None
    }

    fn prototype(&self, name: &str) -> Option<&'static ImportPrototype> {
        KNOWN_PROTOS
            .iter()
            .find(|p| p.name == name && (!p.posix_only || self.posix))
    }

    /// Every live consumer of the import's result truncates it, so the
    /// declaration may narrow `int32_t` to the guest ABI width.
    fn uses_only_truncated(&self, name: &str) -> bool {
        let Some(sites) = self.calls_by_name.get(name) else {
            return true;
        };
        for (fi, callv) in sites {
            let f = &self.module.functions[*fi];
            let mut any = false;
            for b in &f.blocks {
                for &i in &b.insts {
                    let Some(inst) = f.inst(i) else { continue };
                    if operands(inst).any(|o| o == *callv) {
                        any = true;
                        match inst {
                            Inst::Trunc { to, .. } if to.int_bits().unwrap_or(0) <= 32 => {}
                            _ => return false,
                        }
                    }
                }
            }
            if !any {
                return false;
            }
        }
        true
    }

    fn import_result_ty(
        &mut self,
        callee: &CalleeRef,
        p: &ImportPrototype,
    ) -> String {
        let ret = callee.ty.ret.clone();
        if p.result == "void" && !matches!(ret, Type::Void) {
            return self.cty(&ret);
        }
        if p.result == "int32_t" && matches!(ret, Type::Int(64)) && self.uses_only_truncated(&callee.name)
        {
            return "int32_t".to_string();
        }
        p.result.to_string()
    }

    fn call_type(&mut self, f: &Function, callv: ValueId) -> String {
        if let Some(Inst::Call { callee, ty, .. }) = f.inst(callv) {
            if let Some(name) = self.constant_fn_name(f, *callee) {
                if let Some(p) = self.prototype(&name) {
                    let r = {
                        let cr = self.fn_ref(&name).unwrap();
                        self.import_result_ty(&cr, self.prototype(&name).unwrap())
                    };
                    let _ = p;
                    return if r == "int32_t" {
                        "uint32_t".to_string()
                    } else {
                        r
                    };
                }
            }
            let ty = ty.clone();
            return self.cty(&ty);
        }
        String::new()
    }

    fn constant_fn_name(&self, f: &Function, v: ValueId) -> Option<String> {
        match f.value(v) {
            Value::Const(Constant::Function(n)) => Some(n.clone()),
            _ => None,
        }
    }

    fn str_param(&self, callee: Option<&CalleeRef>, k: usize) -> bool {
        let Some(c) = callee else { return false };
        if !c.decl || k >= 64 {
            return false;
        }
        match self.protos.get(&c.name) {
            Some(p) => (p.strmask >> k) & 1 == 1,
            None => false,
        }
    }

    fn string_arg(&mut self, f: &Function, a: ValueId, can_inline: bool) -> String {
        if let Some(addr) = self.const_u64(f, a) {
            if let Some(text) = self.strings.get(&addr).cloned() {
                if can_inline && self.string_uses.get(&addr).copied().unwrap_or(0) == 1 {
                    return format!("\"{}\"", text);
                }
                self.strs_used.entry(addr).or_insert(false);
                return format!("str_{}", hexv(addr).trim_start_matches("0x"));
            }
        }
        let v = self.val_as(f, a, "const char *");
        let c = self.str_comment(f, a);
        format!("{}{}", v, c)
    }

    fn inline_string_arg(&self, callee: Option<&CalleeRef>) -> bool {
        let Some(c) = callee else { return false };
        let Some(p) = self.prototype(&c.name) else {
            return false;
        };
        if p.result.ends_with('*') {
            return false;
        }
        true
    }

    fn unknown_arity(&self, name: &str) -> u32 {
        self.calls_by_name
            .get(name)
            .map(|sites| {
                sites
                    .iter()
                    .filter_map(|(fi, cv)| match self.module.functions[*fi].inst(*cv) {
                        Some(Inst::Call { args, .. }) => Some(args.len() as u32),
                        _ => None,
                    })
                    .max()
                    .unwrap_or(0)
            })
            .unwrap_or(0)
    }

    fn fixed_params(&self, c: &CalleeRef) -> Option<usize> {
        if let Some(p) = self.prototype(&c.name) {
            return if p.variadic {
                Some(p.args.len())
            } else {
                None
            };
        }
        if c.decl {
            if let Some(p) = self.protos.get(&c.name) {
                return if p.ellipsis {
                    Some(p.nargs.max(1) as usize)
                } else {
                    None
                };
            }
        }
        if c.ty.varargs && c.ty.params.is_empty() {
            return if self.unknown_arity(&c.name) > 0 {
                Some(1)
            } else {
                None
            };
        }
        if c.ty.varargs {
            Some(c.ty.params.len())
        } else {
            None
        }
    }

    fn call_args(&mut self, f: &Function, callv: ValueId, callee: Option<&CalleeRef>) -> String {
        let Some(Inst::Call { args, callee: callee_v, .. }) = f.inst(callv) else {
            return String::new();
        };
        let args = args.clone();
        let callee_v = *callee_v;
        let fixed = callee.and_then(|c| self.fixed_params(c));
        let mut s = String::new();
        for (k, a) in args.iter().enumerate() {
            if k > 0 {
                s.push_str(", ");
            }
            // Past the declared parameters nothing converts an argument: a
            // constant carries its type (`5ull`).
            if let Some((bits, w)) = self.const_u128(f, *a) {
                let past = fixed.map(|fx| k >= fx).unwrap_or(false);
                if past && w <= 64 && !self.str_param(callee, k) {
                    s.push_str(&suffixed(bits, w));
                    s.push_str(&self.sym_comment(f, *a));
                    s.push_str(&self.str_comment(f, *a));
                    continue;
                }
            }
            let mut to = match callee {
                Some(c) if !c.decl && k < c.ty.params.len() => {
                    let t = c.ty.params[k].clone();
                    self.cty(&t)
                }
                Some(c) if k < c.ty.params.len() => {
                    let t = c.ty.params[k].clone();
                    self.cty(&t)
                }
                _ => {
                    let t = f.ty(*a);
                    self.cty(&t)
                }
            };
            if callee.is_none() {
                to = self.cty_of(f, *a);
                if let Some(name) = self.callback_type(callee_v) {
                    if let Some(list) = self.callback_arguments.get(&name) {
                        if let Some(t) = list.get(k) {
                            to = t.clone();
                        }
                    }
                }
            }
            if let Some(p) = callee.and_then(|c| self.prototype(&c.name)) {
                if k < p.args.len() {
                    to = p.args[k].to_string();
                }
            }
            // A character constant to putc/putchar prints as the character.
            if k == 0 {
                if let Some(c) = callee {
                    if c.decl && (c.name == "putc" || c.name == "putchar") {
                        if let Some(ch) = self.const_u64(f, *a) {
                            if let Some(cc) = char_literal(ch) {
                                s.push_str(&cc);
                                continue;
                            }
                        }
                    }
                }
            }
            // A constant function address names itself.
            if self.const_u64(f, *a).is_some() && !self.sym_comment(f, *a).is_empty() {
                let v = self.val_as(f, *a, &to);
                s.push_str(&v);
                s.push_str(&self.sym_comment(f, *a));
                continue;
            }
            if self.str_param(callee, k) {
                let can = self.inline_string_arg(callee);
                let v = self.string_arg(f, *a, can);
                s.push_str(&v);
                continue;
            }
            let v = self.val_as(f, *a, &to);
            s.push_str(&v);
            s.push_str(&self.str_comment(f, *a));
        }
        s
    }

    fn call(&mut self, f: &Function, callv: ValueId) -> String {
        let Some(Inst::Call { args, callee: callee_v, ty }) = f.inst(callv).cloned() else {
            return String::new();
        };
        let direct = self.constant_fn_name(f, callee_v);
        if let Some(name) = &direct {
            // `strtol(s, 0, 10)` is `atol(s)` (likewise `strtoll`).
            if args.len() == 3
                && matches!(
                    name.as_str(),
                    "strtol" | "strtoll" | "__isoc23_strtol" | "__isoc23_strtoll"
                )
            {
                let end_zero = matches!(f.value(args[1]), Value::Const(c) if c.is_zero());
                let base_ten = self.const_u64(f, args[2]) == Some(10);
                if end_zero && base_ten {
                    self.atol_used = true;
                    let a = self.string_arg(f, args[0], true);
                    return format!("atol({})", a);
                }
            }
            if let Some(p) = self.known_intrinsic(name) {
                return self.intrinsic_call(f, callv, &p, &args, &ty);
            }
            if name.starts_with("biara_div") || name.starts_with("biara_sdiv") {
                if let Type::Struct(fields) = &ty {
                    let agg = self.aggregate(&fields.clone());
                    self.helpers
                        .entry(format!("{}|{}", name, agg))
                        .or_insert(false);
                }
            }
            // A helper the emitter models itself must be defined in the
            // output, not merely declared, or the program cannot link.
            if name == "helper_cc_compute_c" {
                self.helpers.entry(name.to_string()).or_insert(false);
            }
            let cr = self.fn_ref(name);
            // Calls to unknown code that set no argument still pass the
            // first, as its prototype has one.
            if let Some(c) = &cr {
                if c.ty.varargs && c.ty.params.is_empty() && args.is_empty()
                    && self.unknown_arity(name) > 0
                {
                    return format!("{}(0)", fn_name(name, c.decl));
                }
            }
            let argtext = self.call_args(f, callv, cr.as_ref());
            let mut result = format!("{}({})", fn_name(name, cr.as_ref().map(|c| c.decl).unwrap_or(true)), argtext);
            if let Some(c) = &cr {
                if let Some(p) = self.prototype(&c.name) {
                    if self.import_result_ty(c, p) == "int32_t" {
                        result = format!("(uint32_t){}", result);
                    }
                }
            }
            return result;
        }
        if self.callback_type(callee_v).is_some() {
            let callee = self.val(f, callee_v);
            let args = self.call_args(f, callv, None);
            return format!("{}({})", callee, args);
        }
        let ret = self.cty(&ty);
        let mut sig = format!("{} (*)(", ret);
        for (k, a) in args.iter().enumerate() {
            if k > 0 {
                sig.push_str(", ");
            }
            let t = self.cty_of(f, *a);
            sig.push_str(&t);
        }
        if args.is_empty() {
            sig.push_str("void");
        }
        sig.push(')');
        let callee = self.val_as(f, callee_v, "uint8_t *");
        let argtext = self.call_args(f, callv, None);
        format!("(({}){})({})", sig, callee, argtext)
    }

    /// A recognized `llvm.*` intrinsic, if the module calls one by name.
    fn known_intrinsic(&self, name: &str) -> Option<String> {
        let known = [
            "llvm.ctlz",
            "llvm.cttz",
            "llvm.ctpop",
            "llvm.bswap",
            "llvm.bitreverse",
            "llvm.abs",
            "llvm.umax",
            "llvm.umin",
            "llvm.smax",
            "llvm.smin",
            "llvm.uadd.with.overflow",
            "llvm.usub.with.overflow",
            "llvm.umul.with.overflow",
            "llvm.sadd.with.overflow",
            "llvm.ssub.with.overflow",
            "llvm.smul.with.overflow",
            "llvm.fshl",
            "llvm.fshr",
            "llvm.memcpy",
            "llvm.memmove",
            "llvm.memset",
            "llvm.assume",
            "llvm.lifetime.start",
            "llvm.lifetime.end",
            "llvm.donothing",
        ];
        if !name.starts_with("llvm.") {
            return None;
        }
        known
            .iter()
            .find(|k| name == **k || name.starts_with(&format!("{}.", k)))
            .map(|k| (*k).to_string())
    }

    fn intrinsic_call(
        &mut self,
        f: &Function,
        callv: ValueId,
        base: &str,
        args: &[ValueId],
        ty: &Type,
    ) -> String {
        let _ = callv;
        let bits = ty.int_bits().unwrap_or_else(|| match ty {
            Type::Struct(fields) => fields.first().and_then(|t| t.int_bits()).unwrap_or(0),
            _ => 0,
        });
        let std_ok = standard(bits) && bits >= 8;
        let a = |e: &mut Self, k: usize| -> String {
            match args.get(k) {
                Some(v) => e.ival(f, *v),
                None => "0".to_string(),
            }
        };
        match base {
            "llvm.fshl" | "llvm.fshr" => {
                if !std_ok {
                    self.unsupported += 1;
                    return format!("0 /* unsupported intrinsic {} */", base);
                }
                let left = base == "llvm.fshl";
                if args.len() >= 2 && args[0] == args[1] {
                    let n = self.intrinsic_fn(if left { "rotl" } else { "rotr" }, bits);
                    let x = a(self, 0);
                    let amt = a(self, 2);
                    return format!("{}({}, {})", n, x, amt);
                }
                let n = self.intrinsic_fn(if left { "fshl" } else { "fshr" }, bits);
                let x = a(self, 0);
                let y = a(self, 1);
                let amt = a(self, 2);
                format!("{}({}, {}, {})", n, x, y, amt)
            }
            "llvm.ctlz" | "llvm.cttz" | "llvm.ctpop" | "llvm.bswap" | "llvm.bitreverse"
            | "llvm.abs" => {
                if !std_ok {
                    self.unsupported += 1;
                    return format!("0 /* unsupported intrinsic {} */", base);
                }
                let short = match base {
                    "llvm.ctlz" => "clz",
                    "llvm.cttz" => "ctz",
                    "llvm.ctpop" => "popcount",
                    "llvm.bswap" => "bswap",
                    "llvm.bitreverse" => "bitreverse",
                    _ => "abs",
                };
                let n = self.intrinsic_fn(short, bits);
                let x = a(self, 0);
                format!("{}({})", n, x)
            }
            "llvm.umax" | "llvm.umin" | "llvm.smax" | "llvm.smin" => {
                if !std_ok {
                    self.unsupported += 1;
                    return format!("0 /* unsupported intrinsic {} */", base);
                }
                let short = &base[5..];
                let n = self.intrinsic_fn(short, bits);
                let x = a(self, 0);
                let y = a(self, 1);
                format!("{}({}, {})", n, x, y)
            }
            "llvm.uadd.with.overflow"
            | "llvm.usub.with.overflow"
            | "llvm.umul.with.overflow"
            | "llvm.sadd.with.overflow"
            | "llvm.ssub.with.overflow"
            | "llvm.smul.with.overflow" => {
                if !std_ok {
                    self.unsupported += 1;
                    return format!("0 /* unsupported intrinsic {} */", base);
                }
                let short = format!("{}_{}", &base[5..9], "ovf");
                let agg = match ty {
                    Type::Struct(fields) => self.aggregate(&fields.clone()),
                    _ => {
                        self.unsupported += 1;
                        return format!("0 /* unsupported intrinsic {} */", base);
                    }
                };
                let name = format!("biara_{}{}", short, bits);
                self.helpers
                    .entry(format!("{}|{}", name, agg))
                    .or_insert(false);
                let x = a(self, 0);
                let y = a(self, 1);
                format!("{}({}, {})", name, x, y)
            }
            "llvm.memcpy" | "llvm.memmove" => {
                let b = if base == "llvm.memcpy" {
                    "__builtin_memcpy"
                } else {
                    "__builtin_memmove"
                };
                let d = match args.first() {
                    Some(v) => self.val_as(f, *v, "uint8_t *"),
                    None => "0".to_string(),
                };
                let s = match args.get(1) {
                    Some(v) => self.val_as(f, *v, "uint8_t *"),
                    None => "0".to_string(),
                };
                let n = a(self, 2);
                format!("{}({}, {}, {})", b, d, s, n)
            }
            "llvm.memset" => {
                let d = match args.first() {
                    Some(v) => self.val_as(f, *v, "uint8_t *"),
                    None => "0".to_string(),
                };
                let v = a(self, 1);
                let n = a(self, 2);
                format!("__builtin_memset({}, {}, {})", d, v, n)
            }
            // These carry no runtime effect; the statement printer drops
            // them before reaching here.
            "llvm.assume" | "llvm.lifetime.start" | "llvm.lifetime.end" | "llvm.donothing" => {
                String::new()
            }
            _ => {
                self.unsupported += 1;
                format!("0 /* unsupported intrinsic {} */", base)
            }
        }
    }
}

fn char_literal(ch: u64) -> Option<String> {
    let s = match ch {
        10 => "'\\n'".to_string(),
        9 => "'\\t'".to_string(),
        13 => "'\\r'".to_string(),
        92 => "'\\\\'".to_string(),
        39 => "'\\''".to_string(),
        c if (32..127).contains(&c) => format!("'{}'", c as u8 as char),
        _ => return None,
    };
    Some(s)
}

// ---------------------------------------------------------------------------
// folding and copy coalescing
// ---------------------------------------------------------------------------

fn is_pure(inst: &Inst) -> bool {
    matches!(
        inst,
        Inst::Bin { .. }
            | Inst::Icmp { .. }
            | Inst::Un { .. }
            | Inst::Rot { .. }
            | Inst::Trunc { .. }
            | Inst::Zext { .. }
            | Inst::Sext { .. }
            | Inst::Cast { .. }
            | Inst::Select { .. }
            | Inst::Gep { .. }
            | Inst::Freeze { .. }
            | Inst::ExtractValue { .. }
    )
}

fn is_ignored(f: &Function, v: ValueId) -> bool {
    match f.value(v) {
        Value::Inst(Inst::Alloca { .. }) => true,
        Value::Inst(Inst::Marker { .. }) => true,
        Value::Inst(Inst::Call { callee, .. }) => match f.value(*callee) {
            Value::Const(Constant::Function(n)) => {
                n.starts_with("llvm.assume")
                    || n.starts_with("llvm.lifetime.")
                    || n.starts_with("llvm.donothing")
                    || n.contains("noalias_scope")
            }
            _ => false,
        },
        _ => false,
    }
}

impl<'m, 'a, 'b> Emitter<'m, 'a, 'b> {
    fn small_expression(&self, f: &Function, v: ValueId, budget: &mut u32) -> bool {
        let Some(inst) = f.inst(v) else { return true };
        if !is_pure(inst) || self.facts.uses_of(v) != 1 {
            return true;
        }
        if *budget == 0 {
            return false;
        }
        *budget -= 1;
        for o in operands(inst) {
            if !self.small_expression(f, o, budget) {
                return false;
            }
        }
        true
    }

    /// Whether `v`'s expression reads a phi of `bb` (whose copies on an edge
    /// into `bb` may already have overwritten it).
    fn reads_phi_of(&self, f: &Function, v: ValueId, bb: BlockId, depth: u32) -> bool {
        let Some(inst) = f.inst(v) else { return false };
        if depth > 8 {
            return false;
        }
        if matches!(inst, Inst::Phi { .. }) {
            return f.defined_in[v.0 as usize] == Some(bb);
        }
        operands(inst).any(|o| self.reads_phi_of(f, o, bb, depth + 1))
    }

    fn foldable(&self, f: &Function, v: ValueId) -> bool {
        let Some(inst) = f.inst(v) else { return false };
        if !is_pure(inst) || self.facts.uses_of(v) != 1 {
            return false;
        }
        let mut budget = 16u32;
        if !self.small_expression(f, v, &mut budget) {
            return false;
        }
        let users = self.facts.inst_users[v.0 as usize].clone();
        let my_cty = self.cty_name_of(f, v);
        if users.is_empty() {
            // Folded into the block's terminator (a branch condition, a
            // switch value, a return), which prints in the same block.
            let terms = &self.facts.term_uses[v.0 as usize];
            if terms.len() != 1 {
                return false;
            }
            return !is_ptr_ty(&my_cty) || self.cty_name_ty(f, v) == my_cty;
        }
        let (user, _) = users[0];
        let def_block = f.defined_in[v.0 as usize];
        if let Some(Inst::Phi { incoming, .. }) = f.inst(user) {
            let incoming_block = incoming
                .iter()
                .find(|(iv, _)| *iv == v)
                .map(|(_, b)| *b);
            if incoming_block != def_block {
                return false;
            }
            let user_block = f.defined_in[user.0 as usize];
            if let Some(ub) = user_block {
                if self.reads_phi_of(f, v, ub, 0) {
                    return false;
                }
            }
            return !is_ptr_ty(&my_cty) || self.cty_name_ty(f, v) == my_cty;
        }
        if f.defined_in[user.0 as usize] != def_block {
            return false;
        }
        !is_ptr_ty(&my_cty) || self.cty_name_ty(f, v) == my_cty
    }

    /// `ctyOf`, same spelling (`value_cty` makes it side-effect free).
    fn cty_name_of(&self, f: &Function, v: ValueId) -> String {
        self.cty_of(f, v)
    }

    fn cty_name_ty(&self, f: &Function, v: ValueId) -> String {
        cty_static(&f.ty(v))
    }

    // -- liveness --------------------------------------------------------

    fn read_sites(&self, f: &Function, v: ValueId) -> Vec<(BlockId, Option<ValueId>)> {
        let mut out = Vec::new();
        self.read_sites_into(f, v, &mut out, 0);
        out
    }

    fn read_sites_into(
        &self,
        f: &Function,
        v: ValueId,
        out: &mut Vec<(BlockId, Option<ValueId>)>,
        depth: u32,
    ) {
        for (user, _) in &self.facts.inst_users[v.0 as usize] {
            let Some(inst) = f.inst(*user) else { continue };
            if let Inst::Phi { incoming, .. } = inst {
                if let Some((_, b)) = incoming.iter().find(|(iv, _)| *iv == v) {
                    out.push((*b, None));
                }
            } else if self.inline.contains(user) && depth < 16 {
                self.read_sites_into(f, *user, out, depth + 1);
            } else {
                let b = f.defined_in[user.0 as usize];
                if let Some(b) = b {
                    out.push((b, Some(*user)));
                }
            }
        }
        // A terminator read ends the block.
        for b in &self.facts.term_uses[v.0 as usize] {
            out.push((*b, None));
        }
    }

    fn live(&mut self, f: &Function, v: ValueId) -> Live {
        if let Some(l) = self.live_cache.get(&v) {
            return Live {
                sites: l.sites.clone(),
                in_blocks: l.in_blocks.clone(),
            };
        }
        let sites = self.read_sites(f, v);
        let def = f.defined_in[v.0 as usize];
        let mut in_blocks: HashSet<BlockId> = HashSet::new();
        let mut work: Vec<BlockId> = Vec::new();
        for (sb, _) in &sites {
            if Some(*sb) != def && in_blocks.insert(*sb) {
                work.push(*sb);
            }
        }
        while let Some(b) = work.pop() {
            for p in &self.facts.preds[b.0 as usize] {
                if Some(*p) != def && in_blocks.insert(*p) {
                    work.push(*p);
                }
            }
        }
        let l = Live { sites, in_blocks };
        self.live_cache.insert(
            v,
            Live {
                sites: l.sites.clone(),
                in_blocks: l.in_blocks.clone(),
            },
        );
        l
    }

    /// Whether `v` is read after the point just past `at` in `bb` (`bb`'s
    /// start, after its phis, when `at` is `None`).
    fn live_at(
        &mut self,
        f: &Function,
        v: ValueId,
        bb: BlockId,
        at: Option<ValueId>,
    ) -> bool {
        let d = f.defined_in[v.0 as usize];
        let is_phi = matches!(f.inst(v), Some(Inst::Phi { .. }));
        if d == Some(bb) {
            let at_phi = at.map(|a| matches!(f.inst(a), Some(Inst::Phi { .. }))).unwrap_or(true);
            if !is_phi {
                match at {
                    None => return false,
                    Some(_a) if at_phi => return false,
                    Some(a) => {
                        let vp = self.facts.pos_of(v);
                        let ap = self.facts.pos_of(a);
                        match (vp, ap) {
                            (Some(vp), Some(ap)) if vp >= ap => return false,
                            _ => {}
                        }
                    }
                }
            }
        } else if let Some(d) = d {
            match &self.dom {
                Some(dom) if dom.dominates(d, bb) => {}
                _ => return false,
            }
        }
        let l = self.live(f, v);
        for (sb, si) in &l.sites {
            if *sb != bb {
                continue;
            }
            match (*si, at) {
                (None, _) => return true,
                (Some(_), None) => return true,
                (Some(si), Some(a)) => {
                    let a_phi = matches!(f.inst(a), Some(Inst::Phi { .. }));
                    if a_phi {
                        return true;
                    }
                    let sp = self.facts.pos_of(si);
                    let ap = self.facts.pos_of(a);
                    match (sp, ap) {
                        (Some(sp), Some(ap)) if sp > ap => return true,
                        (Some(_), None) => return true,
                        _ => {}
                    }
                }
            }
        }
        for s in &self.facts.succs[bb.0 as usize] {
            if l.in_blocks.contains(s) {
                return true;
            }
        }
        false
    }

    fn feeds_other_phi(f: &Function, x: ValueId, phi: ValueId) -> bool {
        let Some(Inst::Phi { .. }) = f.inst(phi) else {
            return false;
        };
        let Some(pb) = f.defined_in[phi.0 as usize] else {
            return false;
        };
        // Every phi of the same block that is not `phi`.
        for b in &f.blocks {
            let _ = b;
        }
        for &other in &f.blocks[pb.0 as usize].insts {
            if other == phi {
                continue;
            }
            if let Some(Inst::Phi { incoming, .. }) = f.inst(other) {
                if incoming.iter().any(|(iv, _)| *iv == x) {
                    return true;
                }
            }
        }
        false
    }

    fn interfere(&mut self, f: &Function, a: ValueId, b: ValueId) -> bool {
        let a_phi = matches!(f.inst(a), Some(Inst::Phi { .. }));
        let b_phi = matches!(f.inst(b), Some(Inst::Phi { .. }));
        let a_block = f.defined_in[a.0 as usize];
        let b_block = f.defined_in[b.0 as usize];
        if a_phi && b_phi && a_block == b_block {
            return true;
        }
        if Self::feeds_other_phi(f, a, b) || Self::feeds_other_phi(f, b, a) {
            return true;
        }
        let at = |_e: &Self, v: ValueId| -> Option<ValueId> {
            if matches!(f.inst(v), Some(Inst::Phi { .. })) {
                None
            } else {
                Some(v)
            }
        };
        let (a_at, b_at) = (at(self, a), at(self, b));
        match b_block {
            Some(bb) if self.live_at(f, a, bb, b_at) => return true,
            _ => {}
        }
        match a_block {
            Some(ab) if self.live_at(f, b, ab, a_at) => return true,
            _ => {}
        }
        false
    }

    fn coalesce(&mut self, f: &Function) {
        // Groups keyed by representative.
        let mut groups: BTreeMap<ValueId, Vec<ValueId>> = BTreeMap::new();
        let blocks = f.blocks.len();
        for bi in 0..blocks {
            let bid = BlockId(bi as u32);
            if self.loops.is_header(bid) {
                continue;
            }
            let phis: Vec<ValueId> = f.blocks[bi]
                .insts
                .iter()
                .copied()
                .filter(|v| matches!(f.inst(*v), Some(Inst::Phi { .. })))
                .collect();
            for p in phis {
                if !self.names.contains_key(&p) || self.inline.contains(&p) {
                    continue;
                }
                let incoming: Vec<(ValueId, BlockId)> = match f.inst(p) {
                    Some(Inst::Phi { incoming, .. }) => incoming.clone(),
                    _ => continue,
                };
                for (x, _) in incoming {
                    if x == p
                        || !self.names.contains_key(&x)
                        || self.inline.contains(&x)
                        || self.is_alloca(f, x)
                        || matches!(f.value(x), Value::Param { .. })
                        || self.cty_name_of(f, x) != self.cty_name_of(f, p)
                        || self.canon(x) == self.canon(p)
                    {
                        continue;
                    }
                    if let Some(Inst::Phi { .. }) = f.inst(x) {
                        if f.defined_in[x.0 as usize] == Some(bid) {
                            continue;
                        }
                    }
                    let a_rep = self.canon(p);
                    let b_rep = self.canon(x);
                    let mut a_group = groups.get(&a_rep).cloned().unwrap_or_else(|| vec![p]);
                    let mut b_group = groups.get(&b_rep).cloned().unwrap_or_else(|| vec![x]);
                    if a_group.len() + b_group.len() > 64 {
                        continue;
                    }
                    let mut clash = false;
                    'outer: for i in &a_group {
                        for j in &b_group {
                            if self.interfere(f, *i, *j) {
                                clash = true;
                                break 'outer;
                            }
                        }
                    }
                    if clash {
                        continue;
                    }
                    let rep = a_group[0];
                    a_group.append(&mut b_group);
                    groups.remove(&b_rep);
                    groups.remove(&a_rep);
                    for i in &a_group {
                        self.canon_map.insert(*i, rep);
                    }
                    groups.insert(rep, a_group);
                }
            }
        }
        // One name per group, the phi's: a producer's name would be wrong on
        // the other paths into it.
        for (rep, group) in groups {
            let name = self.names.get(&rep).cloned().unwrap_or_default();
            for i in group {
                self.names.insert(i, name.clone());
            }
        }
    }
}

/// `cty` without `&mut self`: only width/pointer/struct shapes.
fn cty_static(ty: &Type) -> String {
    match ty {
        Type::Void => "void".into(),
        Type::Int(bits) => {
            let u = uint_ty(*bits);
            if u.is_empty() {
                "uint64_t /* unsupported type */".into()
            } else {
                u
            }
        }
        Type::Float(32) => "float".into(),
        Type::Float(64) => "double".into(),
        Type::Ptr => "uint8_t *".into(),
        Type::Struct(fields) => format!("struct biara_tuple@{}", fields.len()),
        _ => "uint64_t /* unsupported type */".into(),
    }
}

// ---------------------------------------------------------------------------
// structure and statements
// ---------------------------------------------------------------------------

impl<'m, 'a, 'b> Emitter<'m, 'a, 'b> {
    fn cur(&self) -> &'m Function {
        &self.module.functions[self.current_fn]
    }

    fn put(&mut self, mut s: Stmt) {
        s.depth = self.depth;
        self.out.push(s);
    }

    fn put_text(&mut self, t: &str) {
        self.put(Stmt::plain(t));
    }

    fn label(&self, b: BlockId) -> String {
        format!("L{}", self.order.get(&b).copied().unwrap_or(0))
    }

    fn jump_to(&mut self, to: BlockId) {
        self.targeted.insert(to);
        let l = self.label(to);
        self.put_text(&format!("goto {};", l));
        if !self.emitted.contains(&to) {
            self.pending.push(to);
        }
    }

    /// `Some(ret)` when `bb` only returns (its phis, at most).
    fn returns_only(&self, f: &Function, bb: BlockId) -> Option<Option<ValueId>> {
        let block = f.block(bb);
        let ret = match block.term.as_ref().unwrap_or(&Terminator::Unreachable) {
            Terminator::Ret(v) => v,
            _ => return None,
        };
        for &v in &block.insts {
            match f.inst(v) {
                Some(Inst::Phi { .. }) => {
                    if Some(v) != *ret {
                        return None;
                    }
                }
                _ => return None,
            }
        }
        Some(*ret)
    }

    /// Where the arms of `b`'s branch meet, if that can follow them.
    fn join(&self, _f: &Function, b: BlockId) -> Option<BlockId> {
        let j = self.pdom.get(b.0 as usize).copied().flatten()?;
        if self.emitted.contains(&j) {
            return None;
        }
        match &self.dom {
            Some(d) if d.dominates(b, j) => {}
            _ => return None,
        }
        if let Some(cx) = self.loop_stack.last() {
            let lp = &self.loops.loops[cx.l];
            if !lp.blocks.contains(&j) || j == lp.header {
                return None;
            }
        }
        Some(j)
    }

    /// Whether `from` is the only way into `to` other than its back edges.
    fn only_entry(&self, from: BlockId, to: BlockId) -> bool {
        for p in &self.facts.preds[to.0 as usize] {
            if *p != from {
                match &self.dom {
                    Some(d) if d.dominates(to, *p) => {}
                    _ => return false,
                }
            }
        }
        true
    }

    /// Assignments to the phis of `to` for the edge from `from`.
    fn edge(&mut self, from: BlockId, to: BlockId) {
        let f = self.cur();
        let mut phis: Vec<ValueId> = Vec::new();
        for &v in &f.block(to).insts {
            let Some(Inst::Phi { .. }) = f.inst(v) else {
                continue;
            };
            let skip = match self.loop_stack.last() {
                Some(cx) => {
                    let lp = &self.loops.loops[cx.l];
                    lp.header == to && cx.induction == Some(v)
                }
                None => false,
            };
            if !skip {
                phis.push(v);
            }
        }
        if phis.is_empty() {
            return;
        }
        let incoming_of = |e: &Self, p: ValueId| -> Option<ValueId> {
            match e.cur().inst(p) {
                Some(Inst::Phi { incoming, .. }) => {
                    incoming.iter().find(|(_, b)| *b == from).map(|(v, _)| *v)
                }
                _ => None,
            }
        };
        // Parallel copy: through temporaries when one phi reads another.
        let mut temps = false;
        for &p in &phis {
            if let Some(inp) = incoming_of(self, p) {
                if inp != p
                    && matches!(self.cur().inst(inp), Some(Inst::Phi { .. }))
                    && self.cur().defined_in[inp.0 as usize] == Some(to)
                {
                    temps = true;
                }
            }
        }
        let mut tail: Vec<Stmt> = Vec::new();
        for &p in &phis {
            let Some(inp) = incoming_of(self, p) else {
                continue;
            };
            let mut s = Stmt {
                def: Some(p),
                assigns: Some(p),
                ..Default::default()
            };
            self.begin();
            self.note_terminator();
            let to_ty = self.cty_of(self.cur(), p);
            let v = self.val_as(self.cur(), inp, &to_ty);
            self.end_stmt(&mut s);
            let note = format!(
                "{}{}",
                self.str_comment(self.cur(), inp),
                self.sym_comment(self.cur(), inp)
            );
            let pname = self.name(p);
            if temps {
                s.text = format!("{}_next = {}{};", pname, v, note);
                s.staged = true;
                self.put(s.clone());
                let mut s2 = Stmt {
                    text: format!("{} = {}_next;", pname, pname),
                    def: Some(p),
                    assigns: Some(p),
                    pcs: s.pcs.clone(),
                    lost_fold: true,
                    ..Default::default()
                };
                s2.lost_fold = true;
                tail.push(s2);
            } else if v != pname {
                s.text = format!("{} = {}{};", pname, bare(&v), note);
                self.put(s);
            }
        }
        for s in tail {
            self.put(s);
        }
    }

    /// Control from `from` to `to`, where `next` follows naturally. Returns
    /// whether control can fall through past this edge.
    fn transfer(&mut self, from: BlockId, to: BlockId, next: Option<BlockId>) -> bool {
        let f = self.cur();
        if let Some(ret) = self.returns_only(f, to) {
            let mut s = Stmt::default();
            self.begin();
            self.note_terminator();
            match ret {
                None => s.text = "return;".to_string(),
                Some(v) => {
                    let mut inp = v;
                    if matches!(f.inst(v), Some(Inst::Phi { .. }))
                        && f.defined_in[v.0 as usize] == Some(to)
                    {
                        if let Some(Inst::Phi { incoming, .. }) = f.inst(v) {
                            if let Some((iv, _)) = incoming.iter().find(|(_, b)| *b == from) {
                                inp = *iv;
                            }
                        }
                    }
                    let ty = f.ty(v);
                    let cty = self.cty(&ty);
                    let e = self.val_as(f, inp, &cty);
                    s.text = format!("return {};", bare(&e));
                }
            }
            self.end_stmt(&mut s);
            self.put(s);
            return false;
        }
        self.edge(from, to);
        if Some(to) == next {
            return true;
        }
        if let Some(cx) = self.loop_stack.last() {
            let lp = &self.loops.loops[cx.l];
            if to == lp.header {
                self.put_text("continue;");
                return false;
            }
            if Some(to) == cx.follow && cx.switches == 0 {
                self.put_text("break;");
                return false;
            }
        }
        if !self.emitted.contains(&to) && self.only_entry(from, to) {
            return self.emit_block(to, next);
        }
        self.jump_to(to);
        false
    }

    /// A branch condition, negated when the arms swap.
    fn neg_cond(&mut self, f: &Function, v: ValueId, neg: bool) -> String {
        if neg {
            if let Some(Inst::Icmp { pred, ty, lhs, rhs }) = f.inst(v).cloned() {
                return self.cmp(f, pred.negated(), &ty, lhs, rhs);
            }
        }
        let c = self.ival(f, v);
        let c = bare(&c);
        if neg {
            format!("!({})", c)
        } else {
            c
        }
    }

    fn emit_block(&mut self, b: BlockId, next: Option<BlockId>) -> bool {
        if Some(b) == next {
            return true;
        }
        if self.emitted.contains(&b) {
            self.jump_to(b);
            return false;
        }
        let f = self.cur();
        // A return-only join whose every incoming edge already returned
        // directly (and nothing jumps to it) is unreachable: skip it.
        if self.returns_only(f, b).is_some() && !self.targeted.contains(&b) {
            let preds = self.facts.preds[b.0 as usize].clone();
            if !preds.is_empty() && preds.iter().all(|p| self.emitted.contains(p)) {
                return false;
            }
        }
        self.emitted.insert(b);
        let mut lb = Stmt::default();
        lb.label = Some(b);
        self.put(lb);
        if let Some(li) = self.loops.loop_for(b) {
            if self.loops.loops[li].header == b {
                return self.emit_loop(li, b, next);
            }
        }
        self.body(b, next)
    }

    fn emit_loop(&mut self, li: usize, b: BlockId, next: Option<BlockId>) -> bool {
        let f = self.cur();
        let exit = self.loops.unique_exit(li);
        let after = match exit {
            Some(e)
                if !self.emitted.contains(&e)
                    && self.dom.as_ref().map(|d| d.dominates(b, e)).unwrap_or(false)
                    && self.returns_only(f, e).is_none() =>
            {
                Some(e)
            }
            _ => None,
        };
        // A pure header with one inside edge and no exit phi can be expressed
        // as a pre-test loop without moving effects or parallel copies.
        if let Some(Terminator::CondBr { cond, yes, no }) = f.block(b).term.clone() {
            if let Some(exit) = exit {
                let exit_has_phis = f
                    .block(exit)
                    .insts
                    .iter()
                    .any(|v| matches!(f.inst(*v), Some(Inst::Phi { .. })));
                if !exit_has_phis {
                    let neg = yes == exit;
                    let inside = if neg { no } else { yes };
                    let other = if neg { yes } else { no };
                    let mut clean = other == exit
                        && inside != b
                        && self.loops.loops[li].blocks.contains(&inside);
                    if clean {
                        for &v in &f.block(b).insts {
                            if matches!(f.inst(v), Some(Inst::Phi { .. })) {
                                continue;
                            }
                            if !is_ignored(f, v) && !self.inline.contains(&v) {
                                clean = false;
                                break;
                            }
                        }
                    }
                    if clean {
                        let lp = &self.loops.loops[li];
                        let pre = lp.preheader;
                        let latch = if lp.latches.len() == 1 {
                            Some(lp.latches[0])
                        } else {
                            None
                        };
                        let phis: Vec<ValueId> = f
                            .block(b)
                            .insts
                            .iter()
                            .copied()
                            .filter(|v| matches!(f.inst(*v), Some(Inst::Phi { .. })))
                            .collect();
                        let mut induction: Option<ValueId> = None;
                        let mut update: Option<ValueId> = None;
                        if pre.is_some() && latch.is_some() && phis.len() == 1 {
                            let latch = latch.unwrap();
                            if matches!(f.block(latch).term, Some(Terminator::Br(_))) {
                                let p = phis[0];
                                let phi_in: Vec<(ValueId, BlockId)> =
                                    match f.inst(p) {
                                        Some(Inst::Phi { incoming, .. }) => incoming.clone(),
                                        _ => Vec::new(),
                                    };
                                if phi_in.len() == 2 {
                                    if let Some((u, _)) =
                                        phi_in.iter().find(|(_, bl)| *bl == latch)
                                    {
                                        if let Some(Inst::Bin { op, lhs, rhs, .. }) = f.inst(*u) {
                                            let step_ok = matches!(op, BinOp::Add | BinOp::Sub)
                                                && *lhs == p
                                                && self.facts.uses_of(*u) == 1
                                                && f.defined_in[u.0 as usize] == Some(latch)
                                                && matches!(f.value(*rhs), Value::Const(_));
                                            if step_ok {
                                                induction = Some(p);
                                                update = Some(*u);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        if let Some(u) = update {
                            self.inline.insert(u);
                        }
                        let mut header = Stmt::default();
                        self.begin();
                        self.note_terminator();
                        let cond_text = self.neg_cond(f, cond, neg);
                        if let Some(ind) = induction {
                            let mut init = String::new();
                            if self.out.len() >= 2 {
                                let last_is_label = self.out[self.out.len() - 1].label == Some(b);
                                if last_is_label {
                                    let idx = self.out.len() - 2;
                                    let is_ind = self.out[idx].assigns == Some(ind)
                                        && !self.out[idx].staged;
                                    if is_ind {
                                        init = self.out[idx].text.clone();
                                        if init.ends_with(';') {
                                            init.pop();
                                        }
                                        let pcs = self.out[idx].pcs.clone();
                                        header.pcs.extend(pcs);
                                        header.lost_fold |= self.out[idx].lost_fold;
                                        let refs = self.out[idx].refs.clone();
                                        header.refs.extend(refs);
                                        self.out.remove(idx);
                                    }
                                }
                            }
                            header.assigns = Some(ind);
                            let upd = update.unwrap();
                            let e = self.expr(self.cur(), upd);
                            let name = self.name(ind);
                            header.text = format!(
                                "for ({}; {}; {} = {}) {{",
                                init,
                                cond_text,
                                name,
                                bare(&e)
                            );
                        } else {
                            header.text = format!("while ({}) {{", cond_text);
                        }
                        self.end_stmt(&mut header);
                        self.put(header);
                        self.depth += 1;
                        self.loop_stack.push(LoopCx {
                            l: li,
                            follow: Some(exit),
                            switches: 0,
                            induction,
                            do_latch: None,
                            footer: None,
                        });
                        self.transfer(b, inside, Some(b));
                        self.loop_stack.pop();
                        self.depth -= 1;
                        self.put_text("}");
                        return self.transfer(b, exit, next);
                    }
                }
            }
        }
        // Post-test loops require a unique conditional latch and no loop/exit
        // phi copies.
        let lp = &self.loops.loops[li];
        let latch = if lp.latches.len() == 1 {
            Some(lp.latches[0])
        } else {
            None
        };
        let latch_br = latch.and_then(|l| match f.block(l).term.clone().unwrap_or(Terminator::Unreachable) {
            Terminator::CondBr { cond, yes, no } => Some((l, cond, yes, no)),
            _ => None,
        });
        let header_has_phis = f
            .block(b)
            .insts
            .iter()
            .any(|v| matches!(f.inst(*v), Some(Inst::Phi { .. })));
        let exit_clean = exit
            .map(|e| {
                !f.block(e)
                    .insts
                    .iter()
                    .any(|v| matches!(f.inst(*v), Some(Inst::Phi { .. })))
            })
            .unwrap_or(false);
        let mut do_loop = latch_br.is_some() && !header_has_phis && exit.is_some() && exit_clean;
        let mut do_latch = None;
        if let (true, Some((l, _, yes, no))) = (do_loop, latch_br.clone()) {
            let exit_b = exit.unwrap();
            do_loop = (yes == b && no == exit_b) || (no == b && yes == exit_b);
            if do_loop {
                let blocks: Vec<BlockId> = self.loops.loops[li].blocks.iter().copied().collect();
                for blk in blocks {
                    if blk != l && !matches!(f.block(blk).term, Some(Terminator::Br(_))) {
                        do_loop = false;
                    }
                    for to in f.block(blk).term.as_ref().map(successor_blocks).unwrap_or_default() {
                        if !self.loops.loops[li].blocks.contains(&to)
                            && (blk != l || to != exit_b)
                        {
                            do_loop = false;
                        }
                    }
                }
                if do_loop {
                    do_latch = Some(l);
                }
            }
        }
        if let Some(l) = do_latch {
            self.put_text("do {");
            self.depth += 1;
            self.loop_stack.push(LoopCx {
                l: li,
                follow: exit,
                switches: 0,
                induction: None,
                do_latch: Some(l),
                footer: None,
            });
            self.body(b, Some(b));
            let footer = self.loop_stack.pop().and_then(|c| c.footer);
            self.depth -= 1;
            if let Some(ft) = footer {
                self.put(ft);
            }
            return self.transfer(l, exit.unwrap(), next);
        }
        self.put_text("for (;;) {");
        self.depth += 1;
        let follow = if after.is_some() || exit == next {
            exit
        } else {
            None
        };
        self.loop_stack.push(LoopCx {
            l: li,
            follow,
            switches: 0,
            induction: None,
            do_latch: None,
            footer: None,
        });
        self.body(b, Some(b));
        self.loop_stack.pop();
        self.depth -= 1;
        self.put_text("}");
        if let Some(e) = after {
            return self.emit_block(e, next);
        }
        true
    }

    fn body(&mut self, b: BlockId, next: Option<BlockId>) -> bool {
        self.statements(b);
        let f = self.cur();
        let term = f.block(b).term.clone().unwrap_or(Terminator::Unreachable);
        let mut s = Stmt::default();
        if let Some(cx) = self.loop_stack.last() {
            if cx.do_latch == Some(b) {
                if let Terminator::CondBr { cond, yes, no } = term {
                    let header = self.loops.loops[cx.l].header;
                    let neg = no == header;
                    self.begin();
                    self.note_terminator();
                    let c = self.neg_cond(f, cond, neg);
                    s.text = format!("}} while ({});", c);
                    self.end_stmt(&mut s);
                    let yi = self.loop_stack.len() - 1;
                    self.loop_stack[yi].footer = Some(s);
                    let _ = yes;
                    return true;
                }
            }
        }
        match term {
            Terminator::Br(to) => self.transfer(b, to, next),
            Terminator::CondBr { cond, yes, no } => {
                let j = self.join(f, b);
                let rn = j.or(next);
                let empty = |e: &Self, x: Option<BlockId>| -> bool {
                    match (x, rn) {
                        (Some(x), Some(rn)) => {
                            x == rn
                                && !e.cur().block(x).insts.iter().any(|v| {
                                    matches!(e.cur().inst(*v), Some(Inst::Phi { .. }))
                                })
                        }
                        _ => false,
                    }
                };
                let mut t = yes;
                let mut fa = no;
                let swap = empty(self, Some(t)) && !empty(self, Some(fa));
                self.begin();
                self.note_terminator();
                let c = self.neg_cond(f, cond, swap);
                self.end_stmt(&mut s);
                if swap {
                    std::mem::swap(&mut t, &mut fa);
                }
                s.text = format!("if ({}) {{", c);
                self.put(s);
                self.depth += 1;
                let mut ft = self.transfer(b, t, rn);
                self.depth -= 1;
                let leaves = self
                    .out
                    .last()
                    .map(|x| x.label.is_none() && ends(&x.text))
                    .unwrap_or(false);
                if !empty(self, Some(fa)) {
                    if leaves {
                        self.put_text("}");
                    } else {
                        self.put_text("} else {");
                        self.depth += 1;
                    }
                    ft |= self.transfer(b, fa, rn);
                    if !leaves {
                        self.depth -= 1;
                    }
                }
                if !leaves || empty(self, Some(fa)) {
                    self.put_text("}");
                }
                if let Some(j) = j {
                    if ft || Some(j) != rn {
                        return self.emit_block(j, next);
                    }
                }
                ft
            }
            Terminator::Switch { value, default, cases } => {
                let j = self.join(f, b);
                let rn = j.or(next);
                self.begin();
                self.note_terminator();
                let e = self.ival(f, value);
                s.text = format!("switch ({}) {{", bare(&e));
                self.end_stmt(&mut s);
                self.put(s);
                if let Some(cx) = self.loop_stack.last_mut() {
                    cx.switches += 1;
                }
                let mut falls = false;
                let default_unreachable = matches!(f.block(default).term, Some(Terminator::Unreachable))
                    && f.block(default).insts.is_empty();
                let nodefault = default_unreachable && !cases.is_empty();
                let mut left = cases.len();
                let arm = |e: &mut Self, case: Option<&str>, to: BlockId| -> bool {
                    match case {
                        Some(c) => e.put_text(c),
                        None => e.put_text("default:"),
                    }
                    e.depth += 1;
                    let before = e.out.len();
                    let mut ff = e.transfer(b, to, rn);
                    let fell_out = e.out.len() == before
                        || e.out.last().map(|x| !ends(&x.text)).unwrap_or(true);
                    if fell_out {
                        e.put_text("break;");
                        ff = true;
                    }
                    e.depth -= 1;
                    ff
                };
                for (val, to) in &cases {
                    let label = format!("case {}:", lit_v(*val, 64));
                    if left > 1 || !nodefault {
                        left -= 1;
                        falls |= arm(self, Some(&label), *to);
                        continue;
                    }
                    left -= 1;
                    self.put_text(&label);
                    falls |= arm(self, None, *to);
                }
                if !nodefault {
                    falls |= arm(self, None, default);
                }
                if let Some(cx) = self.loop_stack.last_mut() {
                    cx.switches -= 1;
                }
                self.put_text("}");
                if let Some(j) = j {
                    if falls || Some(j) != rn {
                        return self.emit_block(j, next);
                    }
                }
                falls
            }
            Terminator::Ret(v) => {
                self.begin();
                self.note_terminator();
                match v {
                    Some(v) => {
                        let ty = f.ty(v);
                        let tyn = self.cty(&ty);
                        let e = self.val_as(f, v, &tyn);
                        s.text = format!("return {};", bare(&e));
                    }
                    None => s.text = "return;".to_string(),
                }
                self.end_stmt(&mut s);
                self.put(s);
                true
            }
            Terminator::Unreachable => {
                self.begin();
                self.note_terminator();
                s.text = "__builtin_unreachable();".to_string();
                self.end_stmt(&mut s);
                self.put(s);
                true
            }
        }
    }

    fn statements(&mut self, bb: BlockId) {
        let f = self.cur();
        let insts = f.block(bb).insts.clone();
        for v in insts {
            if matches!(f.inst(v), Some(Inst::Phi { .. })) || is_ignored(f, v) {
                continue;
            }
            if self.inline.contains(&v) {
                continue;
            }
            let mut s = Stmt::default();
            self.begin();
            self.capture_pc(f, v);
            let ty = biara_ir::inst_ty(f.inst(v).unwrap(), f);
            match f.inst(v).cloned().unwrap() {
                Inst::Store { ty, value, ptr, .. } => {
                    let lv = self.mem(f, ptr, &ty);
                    let lt = self.mem_ty(f, ptr, &ty);
                    let vv = self.val_as(f, value, &lt);
                    s.text = format!("{} = {};", lv, bare(&vv));
                    s.assigns = None;
                }
                Inst::Fence => {
                    s.text = "__atomic_thread_fence(__ATOMIC_SEQ_CST);".to_string();
                }
                Inst::InsertValue { agg, value, index, .. } => {
                    let n = self.name(v);
                    let agg_val = if f.constant(agg).is_some_and(zero_initializer) {
                        format!("({}){{0}}", self.cty(&ty))
                    } else {
                        self.val(f, agg)
                    };
                    let ins = self.ival(f, value);
                    let is_undef = matches!(f.value(agg), Value::Const(Constant::Undef));
                    let field = format!("{}.f{}", n, index);
                    s.text = format!(
                        "{}{} = {};",
                        if is_undef {
                            String::new()
                        } else {
                            format!("{} = {}; ", n, agg_val)
                        },
                        field,
                        ins
                    );
                }
                Inst::Call { .. } => {
                    let e = self.call(f, v);
                    let empty_use = self.facts.uses_of(v) == 0;
                    let void = matches!(ty, Type::Void);
                    s.text = if void || empty_use {
                        format!("{};", e)
                    } else {
                        let ct = self.call_type(f, v);
                        let target = self.cty_of(f, v);
                        let conv_e = conv(&e, &ct, &target);
                        let n = self.name(v);
                        let text = format!("{} = {};", n, conv_e);
                        s.rhs = bare(&conv_e);
                        text
                    };
                }
                _ => {
                    let e = self.expr(f, v);
                    s.rhs = bare(&e);
                    let n = self.name(v);
                    s.text = format!("{} = {};", n, s.rhs);
                    if is_pure(f.inst(v).unwrap()) || matches!(f.inst(v), Some(Inst::Load { .. })) {
                        s.def = Some(v);
                    }
                }
            }
            if !matches!(ty, Type::Void)
                && (self.facts.uses_of(v) > 0 || !matches!(f.inst(v), Some(Inst::Call { .. })))
            {
                s.assigns = Some(v);
            }
            self.end_stmt(&mut s);
            self.put(s);
        }
    }
}

// ---------------------------------------------------------------------------
// signatures, bodies, and output
// ---------------------------------------------------------------------------

impl<'m, 'a, 'b> Emitter<'m, 'a, 'b> {
    fn count_string_uses(&mut self, c: &Constant, direct: bool) {
        match c {
            Constant::Int { bits, width } if *width <= 64 => {
                let addr = (*bits & width_mask(*width)) as u64;
                if self.strings.contains_key(&addr) {
                    let uses = self.string_uses.entry(addr).or_insert(0);
                    *uses = if direct {
                        (*uses + 1).min(2)
                    } else {
                        2
                    };
                }
            }
            Constant::Function(_) | Constant::GlobalPtr(_) => {}
            Constant::Aggregate(items) => {
                for i in items {
                    self.count_string_uses(i, false);
                }
            }
            _ => {}
        }
    }

    /// Indirect calls whose uses agree on one signature become one typedef,
    /// like the C++ renderer's `inferCallbacks`.
    fn infer_callbacks(&mut self) {
        #[derive(Clone)]
        struct Shape {
            sig: String,
            conflict: bool,
            args: Vec<String>,
        }
        // Collect uses first: naming aggregate argument/result types needs
        // `&mut self` (one struct definition per shape), and the walk needs
        // only the immutable module.
        struct Use {
            key: (usize, ValueId),
            ret: Type,
            args: Vec<Type>,
        }
        let mut uses: Vec<Use> = Vec::new();
        for (fi, f) in self.module.functions.iter().enumerate() {
            if f.entry.is_none() {
                continue;
            }
            for b in &f.blocks {
                for &v in &b.insts {
                    let Some(Inst::Call { callee, args, ty }) = f.inst(v) else {
                        continue;
                    };
                    if matches!(f.value(*callee), Value::Const(Constant::Function(_))) {
                        continue;
                    }
                    let arg_tys: Vec<Type> = args.iter().map(|a| f.ty(*a)).collect();
                    uses.push(Use {
                        key: (fi, *callee),
                        ret: ty.clone(),
                        args: arg_tys.clone(),
                    });
                    // `biara_function_at(<root>)` names the root directly.
                    if let Some(Inst::Call { callee: inner, args: inner_args, .. }) = f.inst(*callee)
                    {
                        if let Value::Const(Constant::Function(n)) = f.value(*inner) {
                            if n == "biara_function_at" && inner_args.len() == 1 {
                                let b = self.base(f, inner_args[0]);
                                if let Root::Value(rv) = b.root {
                                    if b.off == 0 {
                                        uses.push(Use {
                                            key: (fi, rv),
                                            ret: ty.clone(),
                                            args: arg_tys.clone(),
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        let mut shapes: BTreeMap<(usize, ValueId), Shape> = BTreeMap::new();
        for u in &uses {
            let ret = self.cty(&u.ret);
            let mut sig = format!("{} (*)(", ret);
            let mut arg_tys = Vec::new();
            for (k, t) in u.args.iter().enumerate() {
                if k > 0 {
                    sig.push_str(", ");
                }
                let tys = self.cty(t);
                sig.push_str(&tys);
                arg_tys.push(tys);
            }
            if u.args.is_empty() {
                sig.push_str("void");
            }
            sig.push(')');
            match shapes.get_mut(&u.key) {
                Some(existing) => {
                    if existing.sig != sig {
                        existing.conflict = true;
                    }
                }
                None => {
                    shapes.insert(
                        u.key,
                        Shape {
                            sig,
                            conflict: false,
                            args: arg_tys,
                        },
                    );
                }
            }
        }
        let mut aliases: BTreeMap<String, (usize, ValueId)> = BTreeMap::new();
        for (key, sh) in &shapes {
            if !sh.conflict {
                aliases.entry(sh.sig.clone()).or_insert(*key);
            }
        }
        let mut names: BTreeMap<String, String> = BTreeMap::new();
        for (n, (sig, key)) in aliases.iter().enumerate() {
            let name = format!("biara_callback_{}_t", n);
            let shape = &shapes[key];
            self.callback_arguments.insert(name.clone(), shape.args.clone());
            let mut def = sig.clone();
            if let Some(pos) = def.find("(*)") {
                def.insert_str(pos + 2, &name);
            }
            let def = format!(
                "/* inferred from agreeing call uses; targets may remain open */\ntypedef {};\n",
                def
            );
            self.global_names.insert(name.clone());
            self.support_defs.insert(name.clone(), def.clone());
            self.prelude.push_str(&def);
            names.insert(sig.clone(), name);
        }
        for (key, sh) in shapes {
            if sh.conflict {
                continue;
            }
            if let Some(n) = names.get(&sh.sig) {
                self.callback_types.insert(key, n.clone());
            }
        }
    }

    /// Does the body of function `fi` read parameter `p`?
    ///
    /// An unread parameter keeps its name (dropping it would be a C23
    /// extension in a definition) and the body says `(void)p;` so `-Wextra`
    /// does not reject it.
    #[allow(dead_code)]
    fn param_used(&self, fi: usize, p: ValueId) -> bool {
        self.module.functions[fi]
            .blocks
            .iter()
            .flat_map(|b| b.insts.iter().copied())
            .any(|i| {
                self.module.functions[fi]
                    .inst(i)
                    .is_some_and(|inst| biara_ir::operands(inst).any(|o| o == p))
            })
            || self.module.functions[fi].blocks.iter().any(|b| {
                b.term.as_ref().map(biara_ir::term_operands).unwrap_or_default().contains(&p)
            })
    }

    fn declaration_signature(&mut self, d: &biara_ir::Declaration) -> String {
        if let Some(p) = KNOWN_PROTOS
            .iter()
            .find(|p| p.name == d.name && (!p.posix_only || self.posix))
        {
            let cr = CalleeRef {
                name: d.name.clone(),
                decl: true,
                ty: d.ty.clone(),
            };
            let ret = self.import_result_ty(&cr, p);
            let mut args = p.args.join(", ");
            if p.variadic {
                if !args.is_empty() {
                    args.push_str(", ");
                }
                args.push_str("...");
            } else if args.is_empty() {
                args.push_str("void");
            }
            format!("{} {}({})", ret, fn_name(&d.name, true), args)
        } else if let Some(p) = self.protos.get(&d.name).cloned() {
            let ret = self.cty(&d.ty.ret);
            let mut args = (0..p.nargs)
                .map(|k| {
                        if p.strmask.checked_shr(k).unwrap_or(0) & 1 == 1 {
                            "const char *"
                        } else {
                            "uint64_t"
                        }
                    })
                .collect::<Vec<_>>();
            if p.ellipsis {
                if args.is_empty() {
                    args.push("uint64_t");
                }
                args.push("...");
            } else if args.is_empty() {
                args.push("void");
            }
            format!("{} {}({})", ret, fn_name(&d.name, true), args.join(", "))
        } else {
            let params: Vec<String> = d.ty.params.iter().map(|t| self.cty(t)).collect();
            let ret = self.cty(&d.ty.ret);
            let mut list = params.join(", ");
            // An open signature has no named parameter before `...`, so an
            // empty list is the K&R spelling that accepts any arguments.
            if d.ty.varargs && !list.is_empty() {
                list.push_str(", ...");
            }
            format!("{} {}({})", ret, ident(&d.name), list)
        }
    }

    fn signature_of(&mut self, fi: usize, with_names: bool) -> String {
        let f = &self.module.functions[fi];
        let name = f.name.clone();
        if f.entry.is_none() && f.blocks.is_empty() {
            if let Some(p) = KNOWN_PROTOS
                .iter()
                .find(|p| p.name == name && (!p.posix_only || self.posix))
            {
                let cr = CalleeRef {
                    name: name.clone(),
                    decl: true,
                    ty: f.ty.clone(),
                };
                let ret = self.import_result_ty(&cr, p);
                let mut s = format!("{} {}({}", ret, fn_name(&name, true), {
                    let mut a = String::new();
                    for (k, t) in p.args.iter().enumerate() {
                        if k > 0 {
                            a.push_str(", ");
                        }
                        a.push_str(t);
                    }
                    if p.variadic {
                        if p.args.is_empty() {
                            a.push_str("void");
                        } else {
                            a.push_str(", ...");
                        }
                    } else if p.args.is_empty() {
                        a.push_str("void");
                    }
                    a
                });
                s.push(')');
                return s;
            }
        }
        let mut ret = self.declared_return(&f.attrs).unwrap_or_default();
        if ret.is_empty() {
            let t = f.ty.ret.clone();
            ret = self.cty(&t);
        }
        let mut s = format!("{} {}(", ret, fn_name(&name, f.entry.is_none() && f.blocks.is_empty()));
        let params = f.ty.params.clone();
        if f.entry.is_none() && f.blocks.is_empty() {
            if let Some(p) = self.protos.get(&name).cloned() {
                for k in 0..p.nargs {
                    if k > 0 {
                        s.push_str(", ");
                    }
                    s.push_str(if (p.strmask >> k) & 1 == 1 {
                        "const char *"
                    } else {
                        "uint64_t"
                    });
                }
                if p.nargs == 0 && !p.ellipsis {
                    s.push_str("void");
                }
                if p.ellipsis {
                    s.push_str(if p.nargs > 0 { ", ..." } else { "uint64_t, ..." });
                }
                s.push(')');
                return s;
            }
        }
        if f.ty.varargs && params.is_empty() {
            s.push_str(if self.unknown_arity(&name) > 0 {
                "uint64_t, ...)"
            } else {
                "void)"
            });
            return s;
        }
        let decl = f.entry.is_none() && f.blocks.is_empty();
        for k in 0..params.len() {
            if k > 0 {
                s.push_str(", ");
            }
            let mut t = if decl {
                self.cty(&params[k])
            } else {
                // An inferred callback type spells the whole parameter.
                let cb = f
                    .params
                    .get(k)
                    .and_then(|p| self.callback_types.get(&(fi, *p)).cloned());
                match cb {
                    Some(cb) => cb,
                    None => {
                        let declared = self.declared_param(&f.attrs, f.params.len(), k);
                        match declared {
                            Some(d) if !d.is_empty() => d,
                            _ => self.cty(&params[k]),
                        }
                    }
                }
            };
            if !decl && with_names {
                if let Some(pv) = f.params.get(k) {
                    t.push(' ');
                    t.push_str(&self.name(*pv));
                }
            }
            s.push_str(&t);
        }
        if f.ty.varargs {
            s.push_str(", ...");
        }
        if params.is_empty() {
            s.push_str("void");
        }
        s.push(')');
        s
    }
}

/// Helper bodies for every intrinsic/operation the renderer referenced.
/// QEMU's carry-flag helper, which the lifter calls as an opaque function:
/// the emitted C must define it, or the program cannot link.
const CC_COMPUTE_C: &str = "\
#ifndef helper_cc_compute_c
uint64_t helper_cc_compute_c(uint64_t dst, uint64_t src1, uint64_t src2, uint32_t op) {
    uint32_t width = 8U << (op & 3);
    uint64_t mask = UINT64_MAX >> (64U - width);
    uint64_t result = dst & mask;
    uint64_t operand = src1 & mask;
    unsigned __int128 with_carry = (unsigned __int128)operand + (src2 & mask);
    switch (op) {
    case 0: case 2: case 40: case 41: case 42: case 43: return src1 & 1;
    case 1: case 3: return dst;
    case 4: case 5: case 6: case 7: return src1 != 0;
    case 8: case 9: case 10: case 11: return result < operand;
    case 12: case 13: case 14: return result < with_carry;
    case 15: return src2 ? dst <= src1 : dst < src1;
    case 16: case 17: case 18: case 19: return ((result + operand) & mask) < operand;
    case 20: case 21: case 22:
        return ((result + (uint64_t)with_carry) & mask) < with_carry;
    case 23: {
        uint64_t minuend = dst + src1 + src2;
        return src2 ? minuend <= src1 : minuend < src1;
    }
    case 28: case 29: case 30: case 31:
    case 32: case 33: case 34: case 35: return src1;
    case 36: case 37: case 38: case 39: return operand >> (width - 1);
    case 44: case 45: case 46: case 47: return operand == 0;
    case 48: case 49: case 50: case 51: return operand != 0;
    case 58: case 59: return dst & 1;
    default: return 0;
    }
}
#endif
";

fn helper_defs(helpers: &mut BTreeMap<String, bool>, support: &mut BTreeMap<String, String>) -> String {
    let mut out = String::new();
    let keys: Vec<String> = helpers.keys().cloned().collect();
    for h in keys {
        if *helpers.get(&h).unwrap_or(&false) {
            continue;
        }
        helpers.insert(h.clone(), true);
        if h == "helper_cc_compute_c" {
            out.push_str(CC_COMPUTE_C);
            support.insert(h.clone(), CC_COMPUTE_C.to_string());
            continue;
        }
        if h == "sext" {
            let def = "static inline int64_t biara_sext(uint64_t x, unsigned n) { return (int64_t)(x << (64 - n)) >> (64 - n); }\n".to_string();
            out.push_str(&def);
            support.insert("biara_sext".to_string(), def);
            continue;
        }
        let (name, agg) = match h.find('|') {
            Some(bar) => (h[..bar].to_string(), h[bar + 1..].to_string()),
            None => (h.clone(), String::new()),
        };
        let mut op = name.clone();
        while op.ends_with(|c: char| c.is_ascii_digit()) {
            op.pop();
        }
        let digits = &name[op.len()..];
        let Ok(bits) = digits.parse::<u32>() else {
            continue;
        };
        let op = op.strip_prefix("biara_").unwrap_or(&op).to_string();
        let u = uint_ty(bits);
        let s = int_ty(bits);
        let w = bits.to_string();
        let mut f = String::from("static inline ");
        match op.as_str() {
            "div" | "sdiv" => {
                let signed = op == "sdiv";
                let t = if signed { s.clone() } else { u.clone() };
                let i128 = if signed { "biara_i128" } else { "biara_u128" };
                let pair = if bits == 64 {
                    format!("({})((({})hi << 64) | ({})lo)", i128, i128, i128)
                } else {
                    format!("({})lo", t)
                };
                f.push_str(&format!(
                    "{} {}(uint64_t lo, uint64_t hi, uint64_t d) {{ {} r; r.f0 = ({})({} / ({})(d)); r.f1 = ({})({} % ({})(d)); return r; }}",
                    agg, name, agg, u, pair, t, u, pair, t
                ));
            }
            "rotl" | "rotr" => {
                let l = op == "rotl";
                f.push_str(&format!(
                    "{} {}({} x, {} n) {{ n %= {}; return n ? ({})((x {} n) | (x {} ({} - n))) : x; }}",
                    u,
                    name,
                    u,
                    u,
                    w,
                    u,
                    if l { "<<" } else { ">>" },
                    if l { ">>" } else { "<<" },
                    w
                ));
            }
            "fshl" | "fshr" => {
                let l = op == "fshl";
                f.push_str(&format!(
                    "{} {}({} xa, {} xb, {} n) {{ n %= {}; return n ? ({})({}) : {}; }}",
                    u,
                    name,
                    u,
                    u,
                    u,
                    w,
                    u,
                    if l {
                        format!("(xa << n) | (xb >> ({} - n))", w)
                    } else {
                        format!("(xb >> n) | (xa << ({} - n))", w)
                    },
                    if l { "xa" } else { "xb" }
                ));
            }
            "clz" | "ctz" => {
                let body = if bits == 128 {
                    if op == "clz" {
                        format!("(uint64_t)(x >> 64) ? __builtin_clzll((uint64_t)(x >> 64)) : 64 + __builtin_clzll((uint64_t)x)")
                    } else {
                        format!("(uint64_t)x ? __builtin_ctzll((uint64_t)x) : 64 + __builtin_ctzll((uint64_t)(x >> 64))")
                    }
                } else if op == "clz" {
                    format!("__builtin_clzll((uint64_t)x) - {}", 64 - bits)
                } else {
                    "__builtin_ctzll((uint64_t)x)".to_string()
                };
                f.push_str(&format!(
                    "{} {}({} x) {{ return x ? ({})({}) : {}; }}",
                    u, name, u, u, body, w
                ));
            }
            "popcount" => {
                let body = if bits == 128 {
                    "(__builtin_popcountll((uint64_t)x) + __builtin_popcountll((uint64_t)(x >> 64)))"
                } else {
                    "__builtin_popcountll((uint64_t)x)"
                };
                f.push_str(&format!("{} {}({} x) {{ return ({}){}; }}", u, name, u, u, body));
            }
            "bswap" => {
                let body = match bits {
                    16 => "__builtin_bswap16(x)".to_string(),
                    32 => "__builtin_bswap32(x)".to_string(),
                    64 => "__builtin_bswap64(x)".to_string(),
                    128 => "((biara_u128)__builtin_bswap64((uint64_t)x) << 64) | __builtin_bswap64((uint64_t)(x >> 64))".to_string(),
                    _ => "x".to_string(),
                };
                f.push_str(&format!("{} {}({} x) {{ return {}; }}", u, name, u, body));
            }
            "bitreverse" => {
                f.push_str(&format!(
                    "{} {}({} x) {{ {} r = 0; for (unsigned i = 0; i < {}; ++i) r |= ({})((x >> i) & 1) << ({} - 1 - i); return r; }}",
                    u, name, u, u, w, u, w
                ));
            }
            "abs" => {
                f.push_str(&format!(
                    "{} {}({} x) {{ return ({})x < 0 ? ({})(0 - x) : x; }}",
                    u, name, u, s, u
                ));
            }
            "uadd_sat" => {
                f.push_str(&format!(
                    "{} {}({} a, {} b) {{ {} s = a + b; return s < a ? ({})~({})0 : s; }}",
                    u, name, u, u, u, u, u
                ));
            }
            "umax" | "umin" => {
                f.push_str(&format!(
                    "{} {}({} a, {} b) {{ return a {} b ? a : b; }}",
                    u,
                    name,
                    u,
                    u,
                    if op == "umax" { ">" } else { "<" }
                ));
            }
            "smax" | "smin" => {
                f.push_str(&format!(
                    "{} {}({} a, {} b) {{ return ({})a {} ({})b ? a : b; }}",
                    u,
                    name,
                    u,
                    u,
                    s,
                    if op == "smax" { ">" } else { "<" },
                    s
                ));
            }
            o if o.ends_with("_ovf") => {
                let kind = &o[1..4];
                let signed = o.starts_with('s');
                let t = if signed { s.clone() } else { u.clone() };
                f.push_str(&format!(
                    "{} {}({} a, {} b) {{ {} r; {} v; r.f1 = __builtin_{}_overflow(({})a, ({})b, &v); r.f0 = ({})(v); return r; }}",
                    agg, name, u, u, agg, t, kind, t, t, u
                ));
            }
            _ => continue,
        }
        out.push_str(&f);
        out.push('\n');
        support.insert(name, format!("{}\n", f));
    }
    out
}

impl<'m, 'a, 'b> Emitter<'m, 'a, 'b> {
    fn emit_pending(&mut self, out: &mut COutput<'_>) -> Result<(), String> {
        let helpers = helper_defs(&mut self.helpers, &mut self.support_defs);
        self.prelude.push_str(&helpers);
        let prelude = std::mem::take(&mut self.prelude);
        if !out.write(&prelude) {
            return Err(out.err());
        }
        let names: Vec<String> = self.data_used.keys().cloned().collect();
        for n in names {
            if !*self.data_used.get(&n).unwrap_or(&false) {
                if !out.write(&format!("extern uint8_t *{};\n", ident(&n))) {
                    return Err(out.err());
                }
                self.data_used.insert(n, true);
            }
        }
        let addrs: Vec<u64> = self.strs_used.keys().copied().collect();
        for a in addrs {
            if !*self.strs_used.get(&a).unwrap_or(&false) {
                let text = self.strings.get(&a).cloned().unwrap_or_default();
                if !out.write(&format!(
                    "static const char str_{}[] = \"{}\";\n",
                    hexv(a).trim_start_matches("0x"),
                    text
                )) {
                    return Err(out.err());
                }
                self.strs_used.insert(a, true);
            }
        }
        if self.atol_used && !self.atol_declared {
            let t = if self.posix { "int64_t" } else { "uint64_t" };
            if !out.write(&format!("{} atol(const char *);\n", t)) {
                return Err(out.err());
            }
            self.atol_declared = true;
        }
        Ok(())
    }

    /// One function's C definition, plus its per-statement JSON record.
    fn function(&mut self, fi: usize) -> Result<(String, String), String> {
        self.current_fn = fi;
        self.names.clear();
        self.originals.clear();
        self.name_evidence.clear();
        self.taken.clear();
        self.next_suffix.clear();
        self.inline.clear();
        self.canon_map.clear();
        self.order.clear();
        self.targeted.clear();
        self.live_cache.clear();
        self.statement_json = String::from("[");
        self.statement_count = 0;
        self.mapped_count = 0;
        self.max_depth = 0;
        self.temp_count = 0;
        // Block ids are per function: a set carried across functions would
        // mark this function's blocks as already emitted.
        self.emitted.clear();

        let f = &self.module.functions[fi];
        self.facts = Facts::build(f);
        self.dom = Some(Dom::build(f));
        self.pdom = postdominators(f);
        self.loops = LoopSet::build(f, self.dom.as_ref().unwrap());

        self.argument_names(f);
        let nparams = f.params.len();
        // Fix every value's C spelling before folding: aggregates are named
        // once, so declarations and folded expressions agree.
        self.value_cty.clear();
        let mut typed: Vec<ValueId> = f.params.clone();
        typed.extend(f.blocks.iter().flat_map(|b| b.insts.iter().copied()));
        for v in typed {
            let t = if let Some(cb) = self.callback_type(v) {
                cb
            } else if let Value::Param { index, .. } = f.value(v) {
                match self.declared_param(&f.attrs, f.params.len(), *index as usize) {
                    Some(d) if !d.is_empty() => d,
                    _ => {
                        let ty = f.ty(v);
                        self.cty(&ty)
                    }
                }
            } else {
                let ty = f.ty(v);
                self.cty(&ty)
            };
            self.value_cty.insert((fi, v), t);
        }
        for bi in 0..f.blocks.len() {
            self.order.insert(BlockId(bi as u32), bi as u32);
        }
        let all: Vec<ValueId> = f.blocks.iter().flat_map(|b| b.insts.iter().copied()).collect();
        for v in all {
            if is_ignored(f, v) && !self.is_alloca(f, v) {
                continue;
            }
            if self.foldable(f, v) {
                self.inline.insert(v);
                continue;
            }
            if matches!(biara_ir::inst_ty(f.inst(v).unwrap(), f), Type::Void) {
                continue;
            }
            let (hint, producer) = if self.is_alloca(f, v) {
                ("frame".to_string(), false)
            } else if let Some(Inst::Call { callee, .. }) = f.inst(v) {
                match f.value(*callee) {
                    Value::Const(Constant::Function(n)) if !n.starts_with("llvm.") => {
                        (format!("{}_result", fn_name(n, false)), true)
                    }
                    _ => (format!("value{}", self.names.len().saturating_sub(nparams)), false),
                }
            } else {
                (format!("value{}", self.names.len().saturating_sub(nparams)), false)
            };
            self.declare_name(v, &hint, String::new());
            if producer {
                self.name_evidence.insert(v, "inferred");
            }
        }
        self.coalesce(f);
        self.out.clear();
        self.emitted.clear();
        self.pending.clear();
        self.loop_stack.clear();
        self.depth = 1;
        let entry = f.entry.ok_or_else(|| format!("function {} has no entry", f.name))?;
        self.emit_block(entry, None);
        while let Some(b) = self.pending.pop() {
            if !self.emitted.contains(&b) {
                self.emit_block(b, None);
            }
        }

        // Drop pure assignments nothing reads.
        let mut uses: HashMap<ValueId, u32> = HashMap::new();
        let mut by_def: HashMap<ValueId, Vec<usize>> = HashMap::new();
        for (i, s) in self.out.iter().enumerate() {
            for r in &s.refs {
                *uses.entry(self.canon(*r)).or_insert(0) += 1;
            }
            if let Some(d) = s.def {
                by_def.entry(self.canon(d)).or_default().push(i);
            }
        }
        let mut work: Vec<usize> = Vec::new();
        let mut dead: HashSet<usize> = HashSet::new();
        for (def, idxs) in &by_def {
            if uses.get(def).copied().unwrap_or(0) == 0 {
                work.extend(idxs.iter().copied());
            }
        }
        while let Some(i) = work.pop() {
            if !dead.insert(i) {
                continue;
            }
            let refs = self.out[i].refs.clone();
            for r in refs {
                let c = self.canon(r);
                let e = uses.entry(c).or_insert(0);
                if *e > 0 {
                    *e -= 1;
                }
                if *e == 0 {
                    if let Some(idxs) = by_def.get(&c) {
                        work.extend(idxs.iter().copied());
                    }
                }
            }
        }

        // `v = RHS; return v;` with `v` used nowhere else becomes
        // `return RHS;`.
        let names = self.names.clone();
        for si in 0..self.out.len().saturating_sub(1) {
            if dead.contains(&si) || dead.contains(&(si + 1)) {
                continue;
            }
            let (a_assigns, a_rhs, a_refs, a_pcs, a_lost) = {
                let a = &self.out[si];
                (
                    a.assigns,
                    a.rhs.clone(),
                    a.refs.clone(),
                    a.pcs.clone(),
                    a.lost_fold,
                )
            };
            let Some(assigns) = a_assigns else { continue };
            if a_rhs.is_empty() {
                continue;
            }
            let aname = names.get(&assigns).cloned().unwrap_or_default();
            let b_text = self.out[si + 1].text.clone();
            let ret_named = format!("return {};", aname);
            let ret_rhs = format!("return {};", a_rhs);
            if b_text != ret_named && b_text != ret_rhs {
                continue;
            }
            if uses.get(&self.canon(assigns)).copied().unwrap_or(0) != 1 && b_text != ret_rhs {
                continue;
            }
            if b_text != ret_rhs {
                self.out[si + 1].text = ret_rhs;
            }
            let b = &mut self.out[si + 1];
            b.refs.retain(|r| *r != assigns);
            b.refs.extend(a_refs);
            b.pcs.extend(a_pcs);
            b.lost_fold |= a_lost;
            dead.insert(si);
        }

        // Recompute live uses after the merge.
        uses.clear();
        for (i, s) in self.out.iter().enumerate() {
            if dead.contains(&i) {
                continue;
            }
            for r in &s.refs {
                *uses.entry(self.canon(*r)).or_insert(0) += 1;
            }
        }
        let mut assigned: HashSet<ValueId> = HashSet::new();
        let mut staged: HashSet<ValueId> = HashSet::new();
        for (i, s) in self.out.iter().enumerate() {
            if dead.contains(&i) {
                continue;
            }
            if let Some(a) = s.assigns {
                let c = self.canon(a);
                if s.staged {
                    staged.insert(c);
                } else {
                    assigned.insert(c);
                }
            }
        }

        let f = &self.module.functions[fi];
        let insts: Vec<ValueId> = f.blocks.iter().flat_map(|b| b.insts.iter().copied()).collect();
        let mut locals = String::new();
        for v in insts {
            if !self.names.contains_key(&v) || (self.canon_map.contains_key(&v) && self.canon(v) != v)
            {
                continue;
            }
            if let Some(Inst::Alloca { ty }) = f.inst(v) {
                let n = self.name(v);
                let size = biara_ir::ty_bytes(ty) as u64;
                locals.push_str(&format!(
                    "    uint8_t {}[{}] __attribute__((aligned(16)));\n",
                    n, size
                ));
                continue;
            }
            let n = self.name(v);
            let ct = self.cty_name_of(f, v);
            let live = uses.get(&v).copied().unwrap_or(0) > 0;
            if assigned.contains(&v) || live {
                locals.push_str(&format!("    {} {};\n", ct, n));
                self.temp_count += 1;
            }
            if staged.contains(&v) {
                locals.push_str(&format!("    {} {}_next;\n", ct, n));
            }
        }
        // A `return;` closing a void function returns to nothing: drop it.
        if matches!(f.ty.ret, Type::Void) {
            for si in (0..self.out.len()).rev() {
                if self.out[si].label.is_some() {
                    continue;
                }
                if self.out[si].text == "return;" && self.out[si].depth == 1 {
                    dead.insert(si);
                }
                break;
            }
        }
        let sig = self.signature_of(fi, true);
        // An unread formal keeps its name (a definition may not omit it) and
        // is named in a `(void)` statement so `-Wextra` accepts it.
        let mut unused = String::new();
        for &p in &f.params {
            if !self.param_used(fi, p) {
                unused.push_str(&format!("    (void){};\n", self.name(p)));
            }
        }
        let mut body = format!(
            "\n{}\n{{\n{}{}{}",
            sig,
            locals,
            unused,
            if locals.is_empty() && unused.is_empty() {
                ""
            } else {
                "\n"
            }
        );
        let mut line = body.matches('\n').count() as u64 + 1;
        let entry_addr = f.attrs.entry_addr;
        let mut json = String::from("[");
        for i in 0..self.out.len() {
            if dead.contains(&i) {
                continue;
            }
            let label = self.out[i].label;
            let text = if let Some(lb) = label {
                if !self.targeted.contains(&lb) {
                    continue;
                }
                format!("{}:;", self.label(lb))
            } else {
                self.out[i].text.clone()
            };
            if self.packaging {
                if self.statement_count > 0 {
                    json.push(',');
                }
                let id = format!(
                    "function_{}/s{}",
                    hexv(entry_addr.unwrap_or(0)).trim_start_matches("0x"),
                    self.statement_count
                );
                self.statement_count += 1;
                let pcs: Vec<String> = self.out[i].pcs.iter().map(|p| p.to_string()).collect();
                let provenance = if self.out[i].pcs.is_empty() {
                    "unknown"
                } else if self.out[i].lost_fold {
                    "inferred"
                } else {
                    "exact"
                };
                json.push_str(&format!(
                    "{{\"id\":{},\"line\":{},\"text\":{},\"addresses\":[{}],\"provenance\":{}}}",
                    json_str(&id),
                    line,
                    json_str(&text),
                    pcs.join(","),
                    json_str(provenance)
                ));
                if !self.out[i].pcs.is_empty() {
                    self.mapped_count += 1;
                }
                self.max_depth = self.max_depth.max(self.out[i].depth as u64);
            }
            let indent = if label.is_some() {
                self.out[i].depth.saturating_sub(1)
            } else {
                self.out[i].depth
            };
            body.push_str(&" ".repeat(4 * indent as usize));
            body.push_str(&text);
            body.push('\n');
            line += 1;
        }
        json.push(']');
        self.statement_json = json;
        body.push_str("}\n");
        Ok((body, String::new()))
    }

    /// C's `main` over the guest's: argc, argv and envp go to the registers
    /// that carry the first three arguments.
    fn entry_wrapper(&mut self, fi: usize) -> String {
        let f = &self.module.functions[fi];
        let params = f.params.clone();
        let ftypes = f.ty.params.clone();
        let ret = f.ty.ret.clone();
        let cn = ["argc", "argv", "envp"];
        let mut args = String::new();
        for (i, p) in params.iter().enumerate() {
            let pname = match f.value(*p) {
                Value::Param { name, .. } => name.clone(),
                _ => String::new(),
            };
            let mut v = "0".to_string();
            for (k, ca) in self.opts.conv_args.iter().enumerate().take(3) {
                if &pname == ca {
                    // The guest takes these as integers, so a pointer
                    // argument needs an explicit conversion.
                    v = if k == 0 {
                        "(uint64_t)(int64_t)argc".to_string()
                    } else {
                        format!("(uint64_t)(uintptr_t){}", cn[k])
                    };
                }
            }
            let ty = ftypes.get(i).cloned().unwrap_or(Type::Void);
            let cty = self.cty(&ty);
            if !args.is_empty() {
                args.push_str(", ");
            }
            args.push_str(&conv(&v, "uint64_t", &cty));
        }
        let call = format!("guest_main({})", args);
        let body = if ret.int_bits().is_some() {
            format!("    return (int){};\n", call)
        } else {
            format!("    {};\n    return 0;\n", call)
        };
        format!(
            "\n/* The program's entry: the guest's main, with C's arguments. */\nint main(int argc, char **argv, char **envp)\n{{\n{}}}\n",
            body
        )
    }

    fn emit_layout(&mut self, l: &LLayout) -> String {
        let mut s = format!("{} {{\n", l.name);
        let mut at: i64 = 0;
        for f in &l.fields {
            if f.off > at {
                s.push_str(&format!(
                    "    uint8_t _pad_{}[{}];\n",
                    hexv(at as u64).trim_start_matches("0x"),
                    f.off - at
                ));
            }
            let ty = uint_ty(f.bytes * 8);
            s.push_str(&format!("    {} {};\n", ty, f.name));
            at = f.off + f.bytes as i64;
        }
        if (l.bytes as i64) > at {
            s.push_str(&format!(
                "    uint8_t _pad_{}[{}];\n",
                hexv(at as u64).trim_start_matches("0x"),
                l.bytes as i64 - at
            ));
        }
        s.push_str("} __attribute__((packed));\n");
        s
    }

    /// The CPU state struct: `cpu_state`'s fields at the offsets the
    /// convention names, env included at `margin`.
    fn emit_cpu_state(&mut self) -> String {
        let mut fields: Vec<(i64, u32, String)> = self
            .opts
            .cpu
            .iter()
            .map(|c| (c.off + self.opts.margin as i64, c.bytes, ident(&c.name)))
            .collect();
        fields.sort_by_key(|f| f.0);
        let storage = self.module.globals.iter().find_map(|global| {
            if global.name == crate::abi_form::CPU_GLOBAL
                && let Type::Array(element, bytes) = &global.ty
                && **element == Type::i8()
            {
                i64::try_from(*bytes).ok()
            } else {
                None
            }
        });
        let size = fields
            .iter()
            .map(|f| f.0 + f.1 as i64)
            .max()
            .unwrap_or(0)
            .max(self.opts.margin as i64 + 8)
            .max(storage.unwrap_or(0));
        let mut s = String::from("struct cpu_state {\n");
        let mut at: i64 = 0;
        let mut used: BTreeSet<String> = BTreeSet::new();
        for (off, bytes, name) in &fields {
            if *off > at {
                s.push_str(&format!(
                    "    uint8_t _pad_{}[{}];\n",
                    hexv(at as u64).trim_start_matches("0x"),
                    off - at
                ));
            }
            let mut n = name.clone();
            if !used.insert(n.clone()) {
                n = format!("{}_{:x}", n, off);
            }
            s.push_str(&format!("    {} {};\n", uint_ty(bytes * 8), n));
            at = *off + *bytes as i64;
        }
        if size > at {
            s.push_str(&format!(
                "    uint8_t _pad_{}[{}];\n",
                hexv(at as u64).trim_start_matches("0x"),
                size - at
            ));
        }
        s.push_str("} __attribute__((packed));\n");
        s
    }

    // -- mode 0 ----------------------------------------------------------

    fn run(&mut self, w: &mut dyn Write) -> Result<(), String> {
        let mut out = COutput::new(w);
        for f in &self.module.functions {
            let decl = f.entry.is_none() && f.blocks.is_empty();
            self.global_names.insert(fn_name(&f.name, decl));
        }
        self.global_names.insert("biara_cpu".to_string());
        for d in &self.data {
            self.global_names.insert(ident(&d.name));
        }
        let banner = format!(
            "/*\n * {}\n */\n\n#include <stdbool.h>\n#include <stdint.h>\n\ntypedef unsigned __int128 biara_u128;\ntypedef __int128 biara_i128;\n\n",
            self.opts.banner
        );
        if !out.write(&banner) {
            return Err(out.err());
        }

        // Layouts precede all prototypes and bodies; emit one at a time.
        let layouts: Vec<LLayout> = self.opts.layouts.layouts.clone();
        let mut declared: BTreeSet<String> = BTreeSet::new();
        for l in &layouts {
            if declared.insert(l.name.clone()) {
                if !out.write(&format!("{};\n", l.name)) {
                    return Err(out.err());
                }
            }
        }
        let mut defined: BTreeSet<String> = BTreeSet::new();
        for l in &layouts {
            if defined.insert(l.name.clone()) {
                let text = self.emit_layout(l);
                if !out.write(&format!("\n{}", text)) {
                    return Err(out.err());
                }
            }
        }
        if self.has_cpu_global() {
            let cpu = self.emit_cpu_state();
            if !out.write(&format!(
                "\n/* CPU state that registers do not cover (vector and FPU registers,\n * what helpers read through env). */\n{}extern struct cpu_state biara_cpu;\n#define BIARA_ENV ((uint8_t *)&biara_cpu + {})\n",
                cpu, self.opts.margin
            )) {
                return Err(out.err());
            }
        }
        if !out.write("\n") {
            return Err(out.err());
        }

        // Forward declarations allow bodies to stream in module order. One
        // C name may stand for several IR declarations (an overloaded
        // intrinsic); the first spelling wins.
        let mut declared_names: BTreeSet<String> = BTreeSet::new();
        let n = self.module.functions.len();
        for fi in 0..n {
            let f = &self.module.functions[fi];
            let decl = f.entry.is_none() && f.blocks.is_empty();
            if decl && self.func_refs.get(&f.name).copied().unwrap_or(0) == 0 {
                continue;
            }
            self.names.clear();
            self.taken.clear();
            self.next_suffix.clear();
            for (i, p) in f.params.iter().enumerate() {
                let original = match f.value(*p) {
                    Value::Param { name, .. } => name.clone(),
                    _ => String::new(),
                };
                self.declare_name(*p, &original, original.clone());
                let _ = i;
            }
            if decl && (f.name.starts_with("biara_div") || f.name.starts_with("biara_sdiv")) {
                if let Type::Struct(fields) = &f.ty.ret {
                    let agg = self.aggregate(&fields.clone());
                    self.helpers
                        .entry(format!("{}|{}", f.name, agg))
                        .or_insert(false);
                }
                self.emit_pending(&mut out)?;
                continue;
            }
            // A helper the emitter defines itself must not also be declared
            // here: the prototype and the `static inline` definition would
            // disagree on linkage.
            // An overloaded intrinsic has one C spelling; a second
            // declaration of the same name would only conflict.
            if !declared_names.insert(self.module.functions[fi].name.clone()) {
                continue;
            }
            let sig = self.signature_of(fi, true);
            self.emit_pending(&mut out)?;
            if !out.write(&format!("{};\n", sig)) {
                return Err(out.err());
            }
        }
        // `Module::decls` holds the prototypes the ABI form introduced
        // (`biara_syscall`, `biara_call_at`, and the lifted helpers). They
        // are not in `functions`, so the loop above never reaches them and
        // the calls they satisfy would be implicitly declared. A C library
        // name takes the library's own prototype: the guest's register types
        // would otherwise restate `memcpy` as returning `uint64_t`.
        // An overloaded intrinsic is several IR declarations under one C
        // name; the spelling below widens it to `...`, so the first
        // declaration counts and a repeat would conflict.
        for d in self.module.decls.clone() {
            if !declared.insert(fn_name(&d.name, true).to_owned()) {
                continue;
            }
            if (d.name.starts_with("biara_div") || d.name.starts_with("biara_sdiv"))
                && let Type::Struct(fields) = &d.ty.ret
            {
                let aggregate = self.aggregate(fields);
                self.helpers
                    .entry(format!("{}|{}", d.name, aggregate))
                    .or_insert(false);
                self.emit_pending(&mut out)?;
                continue;
            }
            let sig = self.declaration_signature(&d);
            self.emit_pending(&mut out)?;
            if !out.write(&format!("{sig};\n")) {
                return Err(out.err());
            }
        }
        let defs: Vec<usize> = (0..n)
            .filter(|fi| {
                let f = &self.module.functions[*fi];
                !(f.entry.is_none() && f.blocks.is_empty())
            })
            .collect();
        for fi in defs {
            let (body, _) = self.function(fi)?;
            self.emit_pending(&mut out)?;
            if !out.write(&body) {
                return Err(out.err());
            }
        }
        if let Some(mi) = self
            .module
            .functions
            .iter()
            .position(|f| f.name == "main" && !(f.entry.is_none() && f.blocks.is_empty()))
        {
            let w = self.entry_wrapper(mi);
            if !out.write(&w) {
                return Err(out.err());
            }
        }
        out.finish()
    }

    // -- mode 1 ----------------------------------------------------------

    fn packages(&mut self, out: &mut dyn FnMut(&str) -> io::Result<()>) -> Result<(), String> {
        self.packaging = true;
        self.support_defs
            .insert("biara_u128".into(), "typedef unsigned __int128 biara_u128;\n".into());
        self.support_defs
            .insert("biara_i128".into(), "typedef __int128 biara_i128;\n".into());
        for f in &self.module.functions {
            let decl = f.entry.is_none() && f.blocks.is_empty();
            self.global_names.insert(fn_name(&f.name, decl));
        }
        let mut declarations = BTreeMap::new();
        for declaration in &self.module.decls {
            let name = fn_name(&declaration.name, true);
            self.global_names.insert(name.clone());
            declarations.entry(name).or_insert(declaration);
        }
        let selected = self
            .module
            .functions
            .iter()
            .any(|f| f.attrs.present.is_some() || f.attrs.omit_from_presentation);
        let mut functions: Vec<usize> = self
            .module
            .functions
            .iter()
            .enumerate()
            .filter(|(_, f)| {
                f.attrs.entry_addr.is_some()
                    && !(f.entry.is_none() && f.blocks.is_empty())
                    && (!selected || f.attrs.present.is_some())
            })
            .map(|(i, _)| i)
            .collect();

        // Authoritative registry: emitted nonlocal spellings with identity.
        self.emitted_functions.clear();
        self.emitted_globals.clear();
        self.emitted_types.clear();
        for f in &self.module.functions {
            let decl = f.entry.is_none() && f.blocks.is_empty();
            let name = fn_name(&f.name, decl);
            if let Some(e) = f.attrs.entry_addr {
                self.emitted_functions.entry(name).or_insert(e);
            } else if let Some(c) = f.attrs.callee_addr {
                self.emitted_functions.entry(name).or_insert(c);
            } else if let Some(d) = self.module.decl(&f.name) {
                if let Some(c) = d.attrs.callee_addr {
                    self.emitted_functions.entry(name).or_insert(c);
                }
            }
        }
        for (name, declaration) in &declarations {
            if let Some(entry) = declaration.attrs.callee_addr {
                self.emitted_functions.entry(name.clone()).or_insert(entry);
            }
        }
        for d in &self.data {
            self.emitted_globals.entry(ident(&d.name)).or_insert(d.addr);
        }
        for (i, l) in self.opts.layouts.layouts.iter().enumerate() {
            if let Some(short) = l.name.strip_prefix("struct ") {
                self.emitted_types
                    .entry(short.to_string())
                    .or_insert(i as u32);
            }
        }
        functions.sort_by_key(|fi| {
            let f = &self.module.functions[*fi];
            if selected {
                f.attrs.present.unwrap_or(u64::MAX)
            } else {
                f.attrs.entry_addr.unwrap_or(u64::MAX)
            }
        });

        for fi in functions {
            let before = self.unsupported;
            let (body, _) = self.function(fi)?;
            let sig = self.signature_of(fi, true);
            let text = scan_text(&body, &self.callback_arguments);
            helper_defs(&mut self.helpers, &mut self.support_defs);
            self.prelude.clear();

            let mut deps: Vec<(String, String, String)> = Vec::new();
            let mut seen_ids: BTreeSet<String> = BTreeSet::new();
            let mut needed: BTreeSet<String> = text.identifiers.clone();
            for name in &text.identifiers {
                let Some(declaration) = declarations.get(name) else { continue };
                if self.support_defs.contains_key(name) {
                    continue;
                }
                let key = format!("import:{name}");
                seen_ids.insert(key.clone());
                let signature = self.declaration_signature(declaration);
                needed.extend(scan_text(&signature, &self.callback_arguments).identifiers);
                deps.push((key, "import".into(), format!("{signature};\n")));
            }
            let mut expanded: BTreeSet<String> = BTreeSet::new();
            while let Some(id) = needed.iter().next().cloned() {
                needed.remove(&id);
                if !expanded.insert(id.clone()) {
                    continue;
                }
                let mut def = String::new();
                if let Some(d) = self.support_defs.get(&id) {
                    def = d.clone();
                    let is_type = id.starts_with("biara_tuple")
                        || id.starts_with("biara_callback_")
                        || id == "biara_u128"
                        || id == "biara_i128"
                        || id.starts_with("biara_v");
                    let kind = if is_type { "type" } else { "helper" };
                    let key = format!("{}:{}", kind, id);
                    if seen_ids.insert(key) {
                        deps.push((format!("{}:{}", kind, id), kind.into(), def.clone()));
                    }
                } else if let Some((i, l)) = self
                    .opts
                    .layouts
                    .layouts
                    .iter()
                    .enumerate()
                    .find(|(_, l)| l.name.strip_prefix("struct ") == Some(id.as_str()))
                {
                    let _ = i;
                    let l = l.clone();
                    def = self.emit_layout(&l);
                    let key = format!("type:{}", id);
                    if seen_ids.insert(key) {
                        deps.push((format!("type:{}", id), "type".into(), def.clone()));
                    }
                }
                if !def.is_empty() {
                    for d in scan_text(&def, &self.callback_arguments).identifiers {
                        if !expanded.contains(&d) {
                            needed.insert(d);
                        }
                    }
                }
            }
            for (addr, literal) in self.strings.clone() {
                let id = format!("str_{}", hexv(addr).trim_start_matches("0x"));
                if text.identifiers.contains(&id) {
                    let key = format!("string:{}", id);
                    if seen_ids.insert(key) {
                        deps.push((
                            format!("string:{}", id),
                            "string".into(),
                            format!("static const char {}[] = \"{}\";\n", id, literal),
                        ));
                    }
                }
            }
            for name in self.data_used.keys().cloned().collect::<Vec<_>>() {
                if text.identifiers.contains(&ident(&name)) {
                    let key = format!("global:{}", name);
                    if seen_ids.insert(key) {
                        deps.push((
                            format!("global:{}", name),
                            "global".into(),
                            format!("extern uint8_t *{};\n", ident(&name)),
                        ));
                    }
                }
            }
            if text.identifiers.contains("biara_cpu") || text.identifiers.contains("BIARA_ENV") {
                let cpu = self.emit_cpu_state();
                if seen_ids.insert("type:cpu_state".into()) {
                    deps.push(("type:cpu_state".into(), "type".into(), cpu));
                }
                if seen_ids.insert("global:biara_cpu".into()) {
                    deps.push((
                        "global:biara_cpu".into(),
                        "global".into(),
                        format!(
                            "extern struct cpu_state biara_cpu;\n#define BIARA_ENV ((uint8_t *)&biara_cpu + {})\n",
                            self.opts.margin
                        ),
                    ));
                }
            }
            for ci in 0..self.module.functions.len() {
                if ci == fi {
                    continue;
                }
                let cf = &self.module.functions[ci];
                let decl = cf.entry.is_none() && cf.blocks.is_empty();
                let nm = fn_name(&cf.name, decl);
                if !text.identifiers.contains(&nm) || self.support_defs.contains_key(&nm) {
                    continue;
                }
                let kind = if decl { "import" } else { "function" };
                let key = format!("{}:{}", kind, nm);
                if seen_ids.insert(key) {
                    let sig = self.signature_of(ci, false);
                    deps.push((
                        format!("{}:{}", kind, nm),
                        kind.into(),
                        format!("{};\n", sig),
                    ));
                }
            }
            if text.identifiers.contains("atol") && seen_ids.insert("import:atol".into()) {
                let t = if self.posix { "int64_t" } else { "uint64_t" };
                deps.push((
                    "import:atol".into(),
                    "import".into(),
                    format!("{} atol(const char *);\n", t),
                ));
            }
            for integer in [
                "uint8_t", "uint16_t", "uint32_t", "uint64_t", "int8_t", "int16_t", "int32_t",
                "int64_t", "uintptr_t",
            ] {
                if expanded.contains(integer) && seen_ids.insert("include:stdint".into()) {
                    deps.push((
                        "include:stdint".into(),
                        "include".into(),
                        "#include <stdint.h>\n".into(),
                    ));
                    break;
                }
            }
            if (expanded.contains("bool") || expanded.contains("true") || expanded.contains("false"))
                && seen_ids.insert("include:stdbool".into())
            {
                deps.push((
                    "include:stdbool".into(),
                    "include".into(),
                    "#include <stdbool.h>\n".into(),
                ));
            }
            deps.sort_by(|a, b| a.0.cmp(&b.0));
            let dep_json = format!(
                "[{}]",
                deps.iter()
                    .map(|(id, kind, def)| format!(
                        "{{\"id\":{},\"kind\":{},\"text\":{}}}",
                        json_str(id),
                        json_str(kind),
                        json_str(def)
                    ))
                    .collect::<Vec<_>>()
                    .join(",")
            );

            // Names, deduplicated by emitted spelling.
            let mut name_json = String::from("[");
            let mut first = true;
            let mut listed: BTreeSet<String> = BTreeSet::new();
            let f = &self.module.functions[fi];
            let mut add_name = |n: &Option<String>, original: &str, evidence: &str| {
                let Some(n) = n else { return };
                if !text.identifiers.contains(n) || !listed.insert(n.clone()) {
                    return;
                }
                if !first {
                    name_json.push(',');
                }
                first = false;
                name_json.push_str(&format!(
                    "{{\"name\":{},\"original\":{},\"evidence\":{}}}",
                    json_str(n),
                    json_str(original),
                    json_str(evidence)
                ));
            };
            for p in &f.params {
                add_name(
                    &self.names.get(p).cloned(),
                    self.originals.get(p).map(|s| s.as_str()).unwrap_or(""),
                    self.name_evidence.get(p).copied().unwrap_or("unknown"),
                );
            }
            let insts: Vec<ValueId> =
                f.blocks.iter().flat_map(|b| b.insts.iter().copied()).collect();
            for v in insts {
                add_name(
                    &self.names.get(&v).cloned(),
                    self.originals.get(&v).map(|s| s.as_str()).unwrap_or(""),
                    self.name_evidence.get(&v).copied().unwrap_or("unknown"),
                );
            }
            drop(add_name);
            name_json.push(']');

            let regs = text
                .identifiers
                .iter()
                .filter(|i| register_name(i))
                .count();
            let metrics = format!(
                "{{\"tokens\":{},\"temporaries\":{},\"register_names\":{},\"casts\":{},\"gotos\":{},\"max_nesting\":{},\"statements\":{},\"mapped_statements\":{},\"unsupported\":{}}}",
                text.tokens,
                self.temp_count,
                regs,
                text.casts,
                text.gotos,
                self.max_depth,
                self.statement_count,
                self.mapped_count,
                self.unsupported - before
            );

            let mut json = format!(
                "{{\"address\":{},\"name\":{},\"signature\":{},\"body\":{},\"statements\":{},\"dependencies\":{},\"names\":{},\"metrics\":{}",
                f.attrs.entry_addr.unwrap_or(0),
                json_str(&fn_name(&f.name, false)),
                json_str(&sig),
                json_str(&body),
                self.statement_json,
                dep_json,
                name_json,
                metrics
            );
            json.push_str(",\"symbols\":{\"functions\":[");
            json.push_str(
                &self
                    .emitted_functions
                    .iter()
                    .filter(|(s, _)| text.identifiers.contains(*s))
                    .map(|(s, a)| format!("{{\"text\":{},\"address\":{}}}", json_str(s), a))
                    .collect::<Vec<_>>()
                    .join(","),
            );
            json.push_str("],\"globals\":[");
            json.push_str(
                &self
                    .emitted_globals
                    .iter()
                    .filter(|(s, _)| text.identifiers.contains(*s))
                    .map(|(s, a)| format!("{{\"text\":{},\"address\":{}}}", json_str(s), a))
                    .collect::<Vec<_>>()
                    .join(","),
            );
            json.push_str("],\"types\":[");
            json.push_str(
                &self
                    .emitted_types
                    .iter()
                    .filter(|(s, _)| text.identifiers.contains(*s))
                    .map(|(s, i)| format!("{{\"text\":{},\"layout\":{}}}", json_str(s), i))
                    .collect::<Vec<_>>()
                    .join(","),
            );
            json.push_str("]}}");
            out(&json).map_err(|e| e.to_string())?;
        }
        Ok(())
    }
}

fn register_name(name: &str) -> bool {
    const REGS: &[&str] = &[
        "rax", "rbx", "rcx", "rdx", "rdi", "rsi", "rsp", "rbp", "rip", "r8", "r9", "r10", "r11",
        "r12", "r13", "r14", "r15",
    ];
    if REGS.contains(&name) {
        return true;
    }
    if let Some(rest) = name.strip_prefix('x') {
        if let Ok(n) = rest.parse::<u32>() {
            return n < 31;
        }
    }
    false
}

fn scan_text(text: &str, callbacks: &BTreeMap<String, Vec<String>>) -> CText {
    let mut r = CText::default();
    let b = text.as_bytes();
    let mut i = 0usize;
    let mut open = false;
    let mut is_type = false;
    let mut struct_name = false;
    let mut body = false;
    while i < b.len() {
        if b[i].is_ascii_whitespace() {
            i += 1;
            continue;
        }
        if text[i..].starts_with("/*") {
            match text[i + 2..].find("*/") {
                Some(e) => i = i + 2 + e + 2,
                None => i = b.len(),
            }
            continue;
        }
        if text[i..].starts_with("//") {
            match text[i + 2..].find('\n') {
                Some(e) => i = i + 2 + e,
                None => i = b.len(),
            }
            continue;
        }
        let start = i;
        i += 1;
        let ident_start = b[start].is_ascii_alphabetic() || b[start] == b'_';
        if ident_start || b[start].is_ascii_digit() {
            while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
                i += 1;
            }
        } else if b[start] == b'"' || b[start] == b'\'' {
            let quote = b[start];
            while i < b.len() {
                let c = b[i];
                i += 1;
                if c == b'\\' && i < b.len() {
                    i += 1;
                } else if c == quote {
                    break;
                }
            }
        }
        let token = &text[start..i];
        r.tokens += 1;
        if ident_start {
            r.identifiers.insert(token.to_string());
        }
        if token == "goto" {
            r.gotos += 1;
        }
        if token == "{" {
            body = true;
        }
        if token == "(" {
            open = true;
            is_type = false;
            struct_name = false;
        } else if token == ")" {
            if open && is_type && body {
                r.casts += 1;
            }
            open = false;
        } else if open {
            if token == "const" || token == "volatile" || token == "*" {
                // qualifiers and pointer stars do not decide it
            } else if token == "struct" {
                is_type = true;
                struct_name = true;
            } else if struct_name && ident_start {
                struct_name = false;
            } else if is_type_token(token.as_bytes()) || callbacks.contains_key(token) {
                is_type = true;
            } else {
                open = false;
            }
        }
    }
    r
}
