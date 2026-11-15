//! Decompiling one function on demand.
//!
//! The engine lowers a selected function (plus declarations of what it calls) to LLVM and
//! renders C. That costs far more than a frame, so a worker thread does it and the UI asks
//! with [`Decompiler::request`]. The newest request runs first, so jumping through ten
//! functions decompiles the one you land on, not the nine you passed. Finished functions are
//! cached for the session, the least recently asked for leaving once they hold more than
//! [`CACHE_BYTES`]: one huge function's pseudocode takes hundreds of megabytes.

use std::{
    collections::{BTreeSet, HashMap, VecDeque},
    sync::Arc,
};

use biara_lift::DecompileOptions;
use biara_proto::{
    FunctionPackage, PackageSymbols,
    packages::PackageStatement,
    tokens::{Token, TokenKind},
};
use parking_lot::{Condvar, Mutex};

use super::{Notify, Program};

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

    /// Addresses of the instructions that produced line `i` (0-based), if known.
    /// Roughly the memory it holds.
    fn footprint(&self) -> usize {
        self.body.len()
            + self.tokens.len() * std::mem::size_of::<Token>()
            + self.lines.len() * std::mem::size_of::<LineSpan>()
            + self.addresses.len() * 8
            + self.address_ends.len() * 4
    }

    pub fn line_addresses(&self, i: usize) -> &[u64] {
        let start = i
            .checked_sub(1)
            .map_or(0, |p| self.address_ends[p] as usize);
        &self.addresses[start..self.address_ends[i] as usize]
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
        Decompiled {
            entry: p.address,
            name: p.name,
            signature: p.signature,
            body: p.body,
            tokens: p.tokens,
            symbols: p.symbols.unwrap_or_default(),
            lines,
            addresses,
            address_ends,
        }
    }
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

/// Decompile `entry` now, on the calling thread.
pub fn decompile(program: &Program, entry: u64) -> Result<Decompiled, String> {
    run(program, entry).map_err(|e| format!("{e:#}"))
}

fn run(program: &Program, entry: u64) -> anyhow::Result<Decompiled> {
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
            &program.abis,
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
    transformed.select_functions(&[entry]);
    let mut found = None;
    transformed.visit_functions(&mut |package| {
        if package.address == entry {
            found = Some(package);
        }
        Ok(())
    })?;
    found
        .map(Decompiled::from)
        .with_context(|| format!("no output for {entry:#x}"))
}

struct Shared {
    queue: Mutex<Queue>,
    wake: Condvar,
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
}

impl Decompiler {
    pub fn new(program: Arc<Program>, notify: Notify) -> Decompiler {
        let shared = Arc::new(Shared {
            queue: Mutex::new(Queue::default()),
            wake: Condvar::new(),
        });
        let worker = shared.clone();
        std::thread::Builder::new()
            .name("biara-decompiler".into())
            .spawn(move || {
                loop {
                    let entry = {
                        let mut q = worker.queue.lock();
                        loop {
                            if q.closed {
                                return;
                            }
                            // Newest first: the function you are looking at, not the ones you passed.
                            if let Some(e) = q.pending.pop_back() {
                                break e;
                            }
                            worker.wake.wait(&mut q);
                        }
                    };
                    let status = match decompile(&program, entry) {
                        Ok(d) => Status::Done(Arc::new(d)),
                        Err(e) => Status::Failed(e.into()),
                    };
                    worker.queue.lock().finish(entry, status, CACHE_BYTES);
                    notify();
                }
            })
            .expect("spawn decompiler thread");
        Decompiler { shared }
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
}
