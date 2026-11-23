//! The pseudocode view: the decompiler's C for the current function.
//!
//! Colour comes from the engine's own tokens, not a second lexer: every identifier the
//! renderer knows (functions, globals, types) arrives with its identity, so calls link to their
//! targets and renames apply by address at draw time, without decompiling again.

use std::{borrow::Cow, collections::BTreeMap, sync::Arc};

use biara_proto::tokens::{SymbolRef, TokenKind};
use rune::{Color, El, GrowthSpec, ScrollSpec, Tx, col, el, row, text};

use biara_session::{Decompiled, Notes, decomp::Status};

use super::{
    App,
    code::{Line, row_key},
    theme::*,
};

/// What the view is showing.
#[derive(Default)]
pub struct State {
    /// The function `doc` belongs to.
    pub entry: Option<u64>,
    pub doc: Option<Arc<Decompiled>>,
    /// An identifier the user clicked; every occurrence is highlighted.
    pub word: Option<String>,
    /// The decompiler generation the shown document came from. A newer generation makes it
    /// stale, so the view asks for the function again.
    pub generation: u64,
    /// Comments to draw at the end of each line, by line index, cached from `doc` and the
    /// notes version: placing them scans every line, and documents can be 740k lines.
    comments: BTreeMap<usize, String>,
    /// The `(version, notes version)` that `comments` was built from.
    comments_for: Option<(u64, u64)>,
    /// Bumped when a new document replaces the rows. Row keys are indices, so nothing else
    /// changes them: a rename or a highlighted word only changes what mounted rows draw, and
    /// those are rebuilt anyway. A bump makes the next commit compare every row's key.
    pub version: u64,
    seen: u64,
}

impl State {
    /// Point the view at `entry`, picking up its result if the decompiler has one.
    /// Returns true if what the view shows changed.
    pub fn sync(
        &mut self,
        app_entry: Option<u64>,
        status: Status,
        completed: u64,
        generation: u64,
    ) -> bool {
        let mut changed = self.seen != completed;
        self.seen = completed;
        if self.entry != app_entry {
            self.entry = app_entry;
            self.doc = None;
            self.word = None;
            self.version += 1;
            changed = true;
        }
        if let Status::Done(d) = status
            && self.doc.as_ref().is_none_or(|have| !Arc::ptr_eq(have, &d))
        {
            self.doc = Some(d);
            self.generation = generation;
            self.version += 1;
            changed = true;
        }
        changed
    }

    /// Re-place the comments on the shown document's lines when the document or the notes
    /// changed. Cached because placing them scans every line of the document. True if the
    /// drawn comments changed.
    pub fn refresh_comments(&mut self, notes: &Notes) -> bool {
        let key = (self.version, notes.version());
        if self.comments_for == Some(key) {
            return false;
        }
        self.comments.clear();
        if let Some(doc) = &self.doc {
            for (line, placed) in doc.comment_lines(notes.comments()) {
                let text = placed
                    .iter()
                    .map(|(_, text)| *text)
                    .collect::<Vec<_>>()
                    .join("; ");
                self.comments.insert(line, text);
            }
        }
        self.comments_for = Some(key);
        true
    }
}

pub fn view(app: &App) -> El {
    let p = colors();
    let message = |title: &str, detail: &str| {
        col()
            .w_full()
            .h_full()
            .center()
            .gap(6.0)
            .child(text(title.to_owned()).font(UI_MEDIUM).color(p.text))
            .child(text(detail.to_owned()).font(UI).color(p.muted))
    };
    let Some(entry) = app.pseudo.entry else {
        return message(
            "No function here",
            "Go to an address inside a function to see its pseudocode.",
        );
    };
    let name = app.label(entry);
    if let Some(doc) = &app.pseudo.doc {
        let lines = doc.lines.len();
        return col()
            .w_full()
            .h_full()
            .scroll(ScrollSpec {
                stick_bottom: false,
            })
            .child(
                col()
                    .key("pseudo-rows")
                    .w_full()
                    .py(8.0)
                    .growth(GrowthSpec::default())
                    .virtual_rows(
                        lines,
                        ROW_H,
                        |_: &App, i| Cow::Owned(row_key("p", i)),
                        build_row,
                    )
                    .virtual_rows_version((app.pseudo.version, lines)),
            );
    }
    match app.decomp.as_ref().map(|d| d.status(entry)) {
        Some(Status::Failed(e)) => message(&format!("Could not decompile {name}"), &e),
        Some(Status::Pending) => message(
            &format!("Decompiling {name}"),
            "Lowering to LLVM and rendering C…",
        ),
        _ => message(&name, "Pseudocode has not been requested yet."),
    }
}

fn build_row(s: &App, i: usize) -> El {
    let p = colors();
    let doc = s.pseudo.doc.as_ref().expect("rows need a document");
    let mut line = Line::new();
    let base = doc.lines[i].start;
    let tokens = doc.line_tokens(i);
    let mut at = base;
    for (n, t) in tokens.iter().enumerate() {
        let (start, end) = (t.start as usize, (t.end as usize).min(doc.lines[i].end));
        if start > at {
            line.push(&doc.body[at..start], p.text);
        }
        if end <= start {
            continue;
        }
        let raw = &doc.body[start..end];
        let (shown, link) = match &t.symbol {
            Some(SymbolRef::Function { address }) | Some(SymbolRef::Global { address }) => {
                (s.token_name(*address, raw), Some(format!("a{address:x}")))
            }
            Some(SymbolRef::Local { name }) => (
                s.notes
                    .local(doc.entry, name)
                    .map_or_else(|| s.local_name(raw), str::to_owned),
                Some(format!("w{name}")),
            ),
            _ if t.kind == TokenKind::Identifier => (s.local_name(raw), Some(format!("w{raw}"))),
            _ => (raw.to_owned(), None),
        };
        let called = tokens[n + 1..]
            .iter()
            .find(|u| u.kind != TokenKind::Whitespace)
            .is_some_and(|u| doc.body[u.start as usize..u.end as usize].starts_with('('));
        let color = token_color(t.kind, &t.symbol, raw, called);
        let marked = t.kind == TokenKind::Identifier
            && match &t.symbol {
                Some(SymbolRef::Local { name }) => s.pseudo.word.as_deref() == Some(name),
                _ => s.pseudo.word.as_deref() == Some(raw),
            };
        line.run(&shown, color, link, marked.then_some(p.accent_soft));
        at = end;
    }
    if at < doc.lines[i].end {
        line.push(&doc.body[at..doc.lines[i].end], p.text);
    }
    if let Some(comment) = s.pseudo.comments.get(&i) {
        line.push(&format!("  // {comment}"), p.comment);
    }
    let addresses = doc.line_addresses(i).first().copied();
    let mut r = row()
        .w_full()
        .h(ROW_H)
        .items_center()
        .child(
            text(format!("{}", i + 1))
                .font(SMALL.mono())
                .color(p.faint)
                .w(48.0)
                .align(rune::Align::End),
        )
        .child(el().w(14.0));
    if let Some(addr) = addresses {
        r = r
            .hover(p.hover)
            .on_press(move |s: &mut App| s.sync_cursor(addr));
    }
    r.child(
        line.build(CODE)
            .on_link(|s: &mut App, id: &str, tx: &mut Tx| link(s, id, tx)),
    )
}

fn token_color(kind: TokenKind, symbol: &Option<SymbolRef>, raw: &str, called: bool) -> Color {
    let p = colors();
    match kind {
        TokenKind::Keyword => p.keyword,
        TokenKind::Number => p.number,
        TokenKind::String => p.string,
        TokenKind::Comment => p.comment,
        TokenKind::Punct => p.punct,
        TokenKind::Identifier => match symbol {
            Some(SymbolRef::Function { .. }) => p.function,
            Some(SymbolRef::Global { .. }) => p.global,
            Some(SymbolRef::Type { .. }) => p.ty,
            Some(SymbolRef::Local { name }) if is_argument(name) => p.register,
            Some(SymbolRef::Local { .. }) => p.text,
            None if is_type_name(raw) => p.ty,
            None if called => p.function,
            None => p.text,
        },
        _ => p.text,
    }
}

/// A parameter the decompiler named by position (`arg0`); colour follows the emitted name.
fn is_argument(name: &str) -> bool {
    name.strip_prefix("arg")
        .is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
}

fn is_type_name(s: &str) -> bool {
    s.ends_with("_t") || matches!(s, "struct" | "union" | "enum" | "biara_cpu")
}

fn link(s: &mut App, id: &str, tx: &mut Tx) {
    if let Some(addr) = id
        .strip_prefix('a')
        .and_then(|h| u64::from_str_radix(h, 16).ok())
    {
        s.goto(addr, tx);
    } else if let Some(word) = id.strip_prefix('w') {
        s.pseudo.word = if s.pseudo.word.as_deref() == Some(word) {
            None
        } else {
            Some(word.to_owned())
        };
    }
}
