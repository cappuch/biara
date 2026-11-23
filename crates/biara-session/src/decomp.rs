//! Decompiling one function on demand.
//!
//! The engine lowers a selected function (plus declarations of what it calls) to LLVM and
//! renders C. That costs far more than a frame, so a worker thread does it and the UI asks
//! with [`Decompiler::request`]. The newest request runs first, so jumping through ten
//! functions decompiles the one you land on, not the nine you passed. Finished functions are
//! cached for the session, the least recently asked for leaving once they hold more than
//! [`CACHE_BYTES`]: one huge function's pseudocode takes hundreds of megabytes.
//!
//! Names (of functions, data and locals) and comments apply when a line is drawn or printed,
//! from the tokens' identities, so changing them never decompiles again. Signatures and
//! declared structs change the code itself: [`Decompiler::set_types`] drops every cached
//! result.

use std::{
    collections::{BTreeMap, BTreeSet, HashMap, VecDeque},
    sync::Arc,
};

use biara_lift::DecompileOptions;
use biara_proto::{
    FunctionPackage, PackageSymbols,
    annotations::Annotations,
    packages::{PackageStatement, place_comments},
    tokens::{SymbolRef, Token, TokenKind},
};
use parking_lot::{Condvar, Mutex};

use super::{
    Notify, Program,
    types::{StructLayout, Types, layouts_of},
};

/// One decompiled function, as the pseudocode view shows it.
#[derive(Clone, Debug)]
pub struct Decompiled {
    pub entry: u64,
    pub name: String,
    pub signature: String,
    /// The full C text of the function.
    pub body: String,
    /// Lexical tokens over `body`, whitespace excluded, with symbol identities for renaming
    /// and navigation.
    pub tokens: Vec<Token>,
    pub symbols: PackageSymbols,
    /// Where each line of `body` is, and which tokens are on it.
    pub lines: Vec<LineSpan>,
    /// The names the decompiler gave the function's locals and parameters; tokens refer to
    /// them as [`SymbolRef::Local`].
    pub locals: Vec<String>,
    /// The line (0-based) defining the function, if a token names its entry.
    pub signature_line: Option<usize>,
    /// The structs the function uses, recovered or declared.
    pub layouts: Vec<StructLayout>,
    /// The guest instructions behind each line: line `i`'s are
    /// `addresses[address_ends[i - 1]..address_ends[i]]` (from 0 for the first line).
    addresses: Vec<u64>,
    address_ends: Vec<u32>,
}

/// One line of the body: its byte range (newline excluded) and the tokens inside it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LineSpan {
    pub start: usize,
    pub end: usize,
    pub first_token: usize,
    pub end_token: usize,
}

impl Decompiled {
    /// The tokens on line `i` (0-based).
    pub fn line_tokens(&self, i: usize) -> &[Token] {
        let l = &self.lines[i];
        &self.tokens[l.first_token..l.end_token]
    }

    /// The text of line `i` (0-based).
    pub fn line_text(&self, i: usize) -> &str {
        let l = &self.lines[i];
        &self.body[l.start..l.end]
    }

    /// Roughly the memory it holds.
    fn footprint(&self) -> usize {
        self.body.len()
            + self.tokens.len() * std::mem::size_of::<Token>()
            + self.lines.len() * std::mem::size_of::<LineSpan>()
            + self.addresses.len() * 8
            + self.address_ends.len() * 4
            + self.locals.iter().map(|l| l.len() + 24).sum::<usize>()
    }

    /// Addresses of the instructions that produced line `i` (0-based), if known.
    pub fn line_addresses(&self, i: usize) -> &[u64] {
        let start = i
            .checked_sub(1)
            .map_or(0, |p| self.address_ends[p] as usize);
        &self.addresses[start..self.address_ends[i] as usize]
    }

    /// Which of `comments` show on which line (0-based): each on the first line mapped to its
    /// address, a comment on the entry nothing maps on the signature line. Scans every line,
    /// so callers cache it per document and comment set.
    pub fn comment_lines<'c>(
        &self,
        comments: &'c BTreeMap<u64, String>,
    ) -> BTreeMap<usize, Vec<(u64, &'c str)>> {
        place_comments(
            (0..self.lines.len()).map(|i| (i, self.line_addresses(i))),
            self.entry,
            self.signature_line,
            comments,
        )
    }

    /// Line `i` (0-based) as text, each token with a symbol spelled as `name` says (or as
    /// emitted when it says `None`).
    pub fn render_line(
        &self,
        i: usize,
        name: &dyn Fn(&SymbolRef, &str) -> Option<String>,
    ) -> String {
        let l = &self.lines[i];
        let mut out = String::with_capacity(l.end - l.start);
        let mut at = l.start;
        for t in self.line_tokens(i) {
            let (start, end) = (t.start as usize, (t.end as usize).min(l.end));
            let Some(symbol) = &t.symbol else { continue };
            if start < at || end <= start {
                continue;
            }
            let raw = &self.body[start..end];
            if let Some(new) = name(symbol, raw) {
                out.push_str(&self.body[at..start]);
                out.push_str(&new);
                at = end;
            }
        }
        out.push_str(&self.body[at..l.end]);
        out
    }
}

/// The addresses of each of `lines` lines, flat, from the first statement on each line.
fn index_addresses(lines: usize, statements: &[PackageStatement]) -> (Vec<u64>, Vec<u32>) {
    let mut of_line: Vec<Option<&[u64]>> = vec![None; lines];
    for s in statements {
        if let Some(slot) = (s.line as usize)
            .checked_sub(1)
            .and_then(|i| of_line.get_mut(i))
        {
            slot.get_or_insert(&s.addresses);
        }
    }
    let mut addresses = Vec::new();
    let ends = of_line
        .into_iter()
        .map(|a| {
            addresses.extend_from_slice(a.unwrap_or_default());
            addresses.len() as u32
        })
        .collect();
    (addresses, ends)
}

/// Index `body` by line. A token never spans a newline except a block comment; such a token is
/// listed on the line it starts on.
fn index_lines(body: &str, tokens: &[Token]) -> Vec<LineSpan> {
    let mut lines = Vec::new();
    let mut start = 0;
    let mut token = 0;
    for text in body.split('\n') {
        let end = start + text.len();
        let first = token;
        while token < tokens.len()
            && (tokens[token].start as usize) < end + 1
            && (tokens[token].start as usize) >= start
        {
            token += 1;
        }
        lines.push(LineSpan {
            start,
            end,
            first_token: first,
            end_token: token,
        });
        start = end + 1;
    }
    lines
}

impl From<FunctionPackage> for Decompiled {
    fn from(mut p: FunctionPackage) -> Self {
        // Rows print the text between tokens as it is: whitespace tokens add nothing but
        // memory, which for a function of a million lines is hundreds of megabytes.
        p.tokens.retain(|t| t.kind != TokenKind::Whitespace);
        p.tokens.shrink_to_fit();
        let lines = index_lines(&p.body, &p.tokens);
        let (addresses, address_ends) = index_addresses(lines.len(), &p.statements);
        let signature_line = p
            .tokens
            .iter()
            .find(|t| matches!(t.symbol, Some(SymbolRef::Function { address }) if address == p.address))
            .map(|t| bytecount_newlines(&p.body[..t.start as usize]));
        Decompiled {
            entry: p.address,
            name: p.name,
            signature: p.signature,
            body: p.body,
            tokens: p.tokens,
            symbols: p.symbols.unwrap_or_default(),
            lines,
            locals: p.names.into_iter().map(|n| n.name).collect(),
            signature_line,
            layouts: Vec::new(),
            addresses,
            address_ends,
        }
    }
}

fn bytecount_newlines(s: &str) -> usize {
    s.bytes().filter(|&b| b == b'\n').count()
}

/// Where a function's decompilation stands.
#[derive(Clone, Debug)]
pub enum Status {
    /// Never requested.
    None,
    /// Queued or running.
    Pending,
    Done(Arc<Decompiled>),
    Failed(Arc<str>),
}

/// Decompile `entry` now, on the calling thread, with `types` (recovery's, or what the notes
/// declare).
pub fn decompile(program: &Program, types: &Types, entry: u64) -> Result<Decompiled, String> {
    run(program, types, entry).map_err(|e| format!("{e:#}"))
}

fn run(program: &Program, types: &Types, entry: u64) -> anyhow::Result<Decompiled> {
    use anyhow::Context as _;
    let cx = inkwell::context::Context::create();
    let machine = biara_lift::analysis_machine()?;
    let selected = BTreeSet::from([entry]);
    let name_of = |e: u64| {
        program
            .image
            .function_at(e)
            .map(|f| f.name.clone())
            .or_else(|| program.cfg.stub_import(e).map(|s| format!("{s}@plt")))
    };
    let banner = format!(
        "Decompiled by biara from {} ({}).",
        program.path.display(),
        program.image.arch
    );
    let prepared = {
        let lib = program.lib.lock();
        biara_lift::prepare_c_selected(
            &cx,
            &machine,
            lib.info(),
            &program.cfg,
            &types.abis,
            &name_of,
            &banner,
            &program.image.data,
            &selected,
        )?
    };
    let mut transformed = prepared.transform(
        &machine,
        &|addr, buf| program.image.read_const(addr, buf),
        DecompileOptions::default(),
    )?;
    transformed.apply_image_debug(&program.image);
    if !types.structs.is_empty() {
        transformed.apply_annotations(&Annotations {
            structs: types.structs.clone(),
            ..Annotations::default()
        })?;
    }
    transformed.select_functions(&[entry]);
    let layouts = layouts_of(&transformed.layout_facts()?, entry, &types.structs);
    let mut found = None;
    transformed.visit_functions(&mut |package| {
        if package.address == entry {
            found = Some(package);
        }
        Ok(())
    })?;
    let mut doc = found
        .map(Decompiled::from)
        .with_context(|| format!("no output for {entry:#x}"))?;
    doc.layouts = layouts;
    Ok(doc)
}

struct Shared {
    queue: Mutex<Queue>,
    wake: Condvar,
    /// What decompiles use: recovery's calling conventions, or the notes' declarations.
    types: Mutex<Arc<Types>>,
}

/// How much finished pseudocode the cache keeps, beyond the latest result.
pub const CACHE_BYTES: usize = 1 << 30;

#[derive(Default)]
struct Queue {
    pending: VecDeque<u64>,
    done: HashMap<u64, Status>,
    /// Finished functions, least recently asked for first, with their footprints.
    kept: VecDeque<(u64, usize)>,
    kept_bytes: usize,
    closed: bool,
    /// Results delivered so far; the UI compares it to know when to redraw.
    completed: u64,
    /// Bumped when the declared types change; a result begun before is stale.
    generation: u64,
}

impl Queue {
    /// Record `entry`'s result, then drop the least recently asked-for results until the rest
    /// fit `budget` bytes. The newest always stays: it is what the view is waiting for.
    fn finish(&mut self, entry: u64, status: Status, budget: usize) {
        if let Status::Done(d) = &status {
            let bytes = d.footprint();
            self.kept.push_back((entry, bytes));
            self.kept_bytes += bytes;
            while self.kept_bytes > budget && self.kept.len() > 1 {
                let (old, bytes) = self.kept.pop_front().expect("more than one kept");
                self.kept_bytes -= bytes;
                self.done.remove(&old);
            }
        }
        self.done.insert(entry, status);
        self.completed += 1;
    }

    /// Mark `entry` as just asked for, keeping it longest.
    fn touch(&mut self, entry: u64) {
        if let Some(i) = self.kept.iter().position(|&(e, _)| e == entry) {
            let kept = self.kept.remove(i).expect("position is in range");
            self.kept.push_back(kept);
        }
    }
}

/// A background decompiler for one program.
pub struct Decompiler {
    shared: Arc<Shared>,
    program: Arc<Program>,
}

impl Decompiler {
    pub fn new(program: Arc<Program>, notify: Notify) -> Decompiler {
        let shared = Arc::new(Shared {
            queue: Mutex::new(Queue::default()),
            wake: Condvar::new(),
            types: Mutex::new(Arc::new(Types::recovered(&program))),
        });
        let worker = shared.clone();
        let worker_program = program.clone();
        std::thread::Builder::new()
            .name("biara-decompiler".into())
            .spawn(move || {
                let program = worker_program;
                loop {
                    let (entry, generation) = {
                        let mut q = worker.queue.lock();
                        loop {
                            if q.closed {
                                return;
                            }
                            // Newest first: the function you are looking at, not the ones you passed.
                            if let Some(e) = q.pending.pop_back() {
                                break (e, q.generation);
                            }
                            worker.wake.wait(&mut q);
                        }
                    };
                    let types = worker.types.lock().clone();
                    let status = match decompile(&program, &types, entry) {
                        Ok(d) => Status::Done(Arc::new(d)),
                        Err(e) => Status::Failed(e.into()),
                    };
                    let mut q = worker.queue.lock();
                    if q.generation == generation {
                        q.finish(entry, status, CACHE_BYTES);
                    } else if !q.pending.contains(&entry) {
                        // The declared types changed while it ran: do it again.
                        q.pending.push_back(entry);
                        continue;
                    }
                    drop(q);
                    notify();
                }
            })
            .expect("spawn decompiler thread");
        Decompiler { shared, program }
    }

    /// Decompile from now on with the signatures and structs `annotations` declare, dropping
    /// every cached result. When a signature does not fit its function, nothing changes and
    /// the error says which.
    pub fn set_types(&self, annotations: &Annotations) -> Result<(), String> {
        let types = Types::declared(&self.program, annotations)?;
        *self.shared.types.lock() = Arc::new(types);
        let mut q = self.shared.queue.lock();
        q.generation += 1;
        q.done.retain(|_, s| matches!(s, Status::Pending));
        q.kept.clear();
        q.kept_bytes = 0;
        q.completed += 1;
        Ok(())
    }

    /// Bumped by [`Decompiler::set_types`]: a result fetched before is out of date.
    pub fn generation(&self) -> u64 {
        self.shared.queue.lock().generation
    }

    /// Ask for `entry`. Does nothing when it is cached or already queued; a queued request
    /// moves to the front.
    pub fn request(&self, entry: u64) {
        let mut q = self.shared.queue.lock();
        if let Some(Status::Done(_) | Status::Failed(_)) = q.done.get(&entry) {
            q.touch(entry);
            return;
        }
        q.pending.retain(|&e| e != entry);
        q.pending.push_back(entry);
        q.done.insert(entry, Status::Pending);
        self.shared.wake.notify_one();
    }

    /// How many results have been delivered; changes whenever a function finishes.
    pub fn completed(&self) -> u64 {
        self.shared.queue.lock().completed
    }

    /// The current state of `entry`.
    pub fn status(&self, entry: u64) -> Status {
        self.shared
            .queue
            .lock()
            .done
            .get(&entry)
            .cloned()
            .unwrap_or(Status::None)
    }
}

impl Drop for Decompiler {
    fn drop(&mut self) {
        self.shared.queue.lock().closed = true;
        self.shared.wake.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tok(start: u32, end: u32) -> Token {
        Token {
            start: u64::from(start),
            end: u64::from(end),
            kind: TokenKind::Identifier,
            symbol: None,
        }
    }

    fn doc(entry: u64, bytes: usize) -> Status {
        Status::Done(Arc::new(Decompiled {
            entry,
            name: String::new(),
            signature: String::new(),
            body: "x".repeat(bytes),
            tokens: vec![],
            symbols: Default::default(),
            lines: vec![],
            locals: vec![],
            signature_line: None,
            layouts: vec![],
            addresses: vec![],
            address_ends: vec![],
        }))
    }

    #[test]
    fn the_cache_lets_the_least_recently_asked_for_results_go() {
        let mut q = Queue::default();
        let cached = |q: &Queue| {
            let mut e: Vec<u64> = q.done.keys().copied().collect();
            e.sort();
            e
        };
        q.finish(1, doc(1, 400), 1000);
        q.finish(2, doc(2, 400), 1000);
        assert_eq!(cached(&q), [1, 2], "within the budget, both stay");
        // Asking for 1 again makes 2 the least recent, so 2 goes when 3 arrives.
        q.touch(1);
        q.finish(3, doc(3, 400), 1000);
        assert_eq!(cached(&q), [1, 3]);
        // A result larger than the whole budget still stays: the view is waiting for it.
        q.finish(4, doc(4, 5000), 1000);
        assert_eq!(cached(&q), [4]);
        assert_eq!(q.kept_bytes, 5000);
        // Failures cost nothing and are kept.
        q.finish(5, Status::Failed("no".into()), 1000);
        assert_eq!(cached(&q), [4, 5]);
    }

    #[test]
    fn lines_know_their_text_and_tokens() {
        let body = "int f(int a)\n{\n    return a;\n}\n";
        // Tokens: `int` `f` on line 0, `{` on line 1, `return` `a` on line 2.
        let tokens = vec![tok(0, 3), tok(4, 5), tok(13, 14), tok(20, 26), tok(27, 28)];
        let lines = index_lines(body, &tokens);
        assert_eq!(
            lines.len(),
            5,
            "four lines plus the empty tail after the last newline"
        );
        let statement = |line: u64, addresses: Vec<u64>| PackageStatement {
            id: String::new(),
            line,
            text: String::new(),
            addresses,
            provenance: String::new(),
        };
        let (addresses, address_ends) = index_addresses(
            lines.len(),
            &[
                statement(3, vec![0x10, 0x14]),
                statement(3, vec![0x99]),
                statement(1, vec![0x8]),
            ],
        );
        let d = Decompiled {
            entry: 0,
            name: String::new(),
            signature: String::new(),
            body: body.into(),
            tokens,
            symbols: Default::default(),
            lines,
            locals: vec![],
            signature_line: Some(0),
            layouts: vec![],
            addresses,
            address_ends,
        };
        assert_eq!(d.line_text(0), "int f(int a)");
        assert_eq!(d.line_tokens(0).len(), 2);
        assert_eq!((d.line_text(1), d.line_tokens(1).len()), ("{", 1));
        assert_eq!(
            (d.line_text(2), d.line_tokens(2).len()),
            ("    return a;", 2)
        );
        assert_eq!(d.line_tokens(3).len(), 0);
        // Each line has the addresses of its first statement; lines without one have none.
        assert_eq!(d.line_addresses(0), [0x8]);
        assert_eq!(d.line_addresses(1), [] as [u64; 0]);
        assert_eq!(d.line_addresses(2), [0x10, 0x14]);
        assert_eq!(d.line_addresses(4), [] as [u64; 0]);
    }

    #[test]
    fn lines_print_with_names_and_take_their_comments() {
        let body = "int f(int arg0)\n{\n    return g(arg0) + arg0x;\n}";
        let sym = |start: u64, end: u64, symbol: SymbolRef| Token {
            start,
            end,
            kind: TokenKind::Identifier,
            symbol: Some(symbol),
        };
        let local = || SymbolRef::Local {
            name: "arg0".into(),
        };
        let tokens = vec![
            sym(4, 5, SymbolRef::Function { address: 0x10 }),
            sym(10, 14, local()),
            sym(29, 30, SymbolRef::Function { address: 0x20 }),
            sym(31, 35, local()),
            tok(39, 44),
        ];
        let lines = index_lines(body, &tokens);
        let statement = |line: u64, addresses: Vec<u64>| PackageStatement {
            id: String::new(),
            line,
            text: String::new(),
            addresses,
            provenance: String::new(),
        };
        let (addresses, address_ends) =
            index_addresses(lines.len(), &[statement(3, vec![0x14, 0x18])]);
        let d = Decompiled {
            entry: 0x10,
            name: "f".into(),
            signature: String::new(),
            body: body.into(),
            tokens,
            symbols: Default::default(),
            lines,
            locals: vec!["arg0".into()],
            signature_line: Some(0),
            layouts: vec![],
            addresses,
            address_ends,
        };
        let names = |s: &SymbolRef, _: &str| match s {
            SymbolRef::Local { name } if name == "arg0" => Some("count".to_owned()),
            SymbolRef::Function { address: 0x20 } => Some("helper".to_owned()),
            _ => None,
        };
        assert_eq!(d.render_line(0, &names), "int f(int count)");
        // `arg0x` has no symbol, so it stays even though it starts with a renamed name.
        assert_eq!(
            d.render_line(2, &names),
            "    return helper(count) + arg0x;"
        );
        let comments = BTreeMap::from([(0x10, "entry".to_owned()), (0x18, "sum".to_owned())]);
        assert_eq!(
            d.comment_lines(&comments),
            BTreeMap::from([(0, vec![(0x10, "entry")]), (2, vec![(0x18, "sum")])])
        );
    }
}
