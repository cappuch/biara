//! Modal dialogs: go to anything, the command palette, rename and the block limit.
//!
//! One overlay serves them all. It owns a text field, a ranked list of results and a
//! selection; what a result does depends on the mode.

use rune::{Color, El, FileDialogOptions, Key, KeyEvent, TextEdit, Tx, col, el, row, text};
use rune_kit::TextInput;

use biara_proto::annotations::StructAnnotation;
use biara_proto::tokens::{SymbolRef, Token, TokenKind};
use biara_session::types::{bind_param, declare_from_layout};
use biara_session::{NameKind, StructLayout};

use super::{App, MainTab, bottom::BottomTab, parse_addr, parse_count, theme::*, widgets};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Mode {
    Goto,
    Commands,
    Rename {
        addr: u64,
        kind: NameKind,
    },
    /// Rename the local or parameter the pseudocode emitted as `local` in `function`.
    RenameLocal {
        function: u64,
        local: String,
    },
    /// Name the recovered struct a field belongs to before its fields can be renamed.
    NameStruct {
        function: u64,
        layout: StructLayout,
        field: String,
        offset: i64,
    },
    /// Rename the field at `offset` of the declared struct `struct_name`.
    RenameField {
        struct_name: String,
        offset: i64,
        field: String,
    },
    /// A one-line comment at `addr`.
    Comment {
        addr: u64,
    },
    /// `function`'s C signature.
    Signature {
        function: u64,
    },
    /// Why the selected field cannot be renamed; nothing to edit.
    Notice {
        message: String,
    },
    /// How many blocks recovery may reach; changing it analyses the binary again.
    BlockLimit,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Command {
    Goto,
    Rename,
    RenameLocal,
    RenameField,
    Comment,
    Signature,
    OpenFile,
    Tab(MainTab),
    ToggleGraph,
    Back,
    Forward,
    Xrefs,
    Output,
    ToggleDemangle,
    BlockLimit,
}

impl Command {
    const ALL: [(Command, &'static str, &'static str); 18] = [
        (Command::Goto, "Go to address or name…", "G"),
        (Command::Rename, "Rename the current symbol…", "N"),
        (Command::RenameLocal, "Rename the highlighted local…", "N"),
        (
            Command::RenameField,
            "Rename the highlighted struct field…",
            "N",
        ),
        (Command::Comment, "Comment at the cursor…", ";"),
        (Command::Signature, "Set the function signature…", "Y"),
        (Command::OpenFile, "Open binary…", "Ctrl+O"),
        (Command::Tab(MainTab::Listing), "Show disassembly", ""),
        (Command::Tab(MainTab::Graph), "Show graph", ""),
        (Command::Tab(MainTab::Pseudo), "Show pseudocode", "F5"),
        (Command::Tab(MainTab::Hex), "Show hex", ""),
        (
            Command::ToggleGraph,
            "Toggle graph and disassembly",
            "Space",
        ),
        (Command::Back, "Go back", "Esc"),
        (Command::Forward, "Go forward", "Alt+Right"),
        (Command::Xrefs, "Show cross-references", "X"),
        (Command::Output, "Show output", ""),
        (Command::ToggleDemangle, "Toggle demangled names", ""),
        (Command::BlockLimit, "Set the block limit…", ""),
    ];
}

#[derive(Clone, Debug)]
pub enum Action {
    Goto(u64),
    Command(Command),
}

#[derive(Clone, Debug)]
pub struct Hit {
    pub label: String,
    pub detail: String,
    pub action: Action,
}

pub struct State {
    pub mode: Option<Mode>,
    pub input: TextEdit,
    pub hits: Vec<Hit>,
    pub selected: usize,
    /// Why the last submit was rejected (rename).
    pub error: Option<String>,
}

impl Default for State {
    fn default() -> Self {
        State {
            mode: None,
            input: TextEdit::new(""),
            hits: Vec::new(),
            selected: 0,
            error: None,
        }
    }
}

const MAX_HITS: usize = 14;

/// Open the overlay in `mode`.
pub fn open(app: &mut App, mode: Mode) {
    let initial = match &mode {
        Mode::Rename { addr, .. } => app.name_of(*addr).unwrap_or_default(),
        Mode::RenameLocal { function, local } => app
            .notes
            .local(*function, local)
            .map_or_else(|| local.clone(), str::to_owned),
        Mode::NameStruct { .. } => String::new(),
        Mode::RenameField { field, .. } => field.clone(),
        Mode::Comment { addr } => app.notes.comment(*addr).unwrap_or_default().to_owned(),
        Mode::Signature { function } => function_signature(app, *function).unwrap_or_default(),
        // Stopped at the limit: offer twice it, the usual next step.
        Mode::BlockLimit => app
            .block_limit_reached()
            .map_or(app.block_limit(), |limit| limit.saturating_mul(2))
            .to_string(),
        _ => String::new(),
    };
    app.overlay = State {
        mode: Some(mode),
        input: TextEdit::new(initial),
        hits: Vec::new(),
        selected: 0,
        error: None,
    };
    update(app);
}

pub fn close(app: &mut App) {
    app.overlay = State::default();
}

/// Recompute the results for the current text.
pub fn update(app: &mut App) {
    let q = app.overlay.input.text.trim().to_owned();
    app.overlay.error = None;
    app.overlay.selected = 0;
    app.overlay.hits = match &app.overlay.mode {
        Some(Mode::Goto) => goto_hits(app, &q),
        Some(Mode::Commands) => command_hits(&q),
        _ => Vec::new(),
    };
}

/// How well `hay` matches `needle` (lower is better); `None` if it does not.
fn rank(hay: &str, needle: &str) -> Option<u8> {
    let (h, n) = (hay.to_lowercase(), needle.to_lowercase());
    if h == n {
        Some(0)
    } else if h.starts_with(&n) {
        Some(1)
    } else if h.contains(&n) {
        Some(2)
    } else {
        None
    }
}

fn goto_hits(app: &App, q: &str) -> Vec<Hit> {
    if q.is_empty() {
        return Vec::new();
    }
    let mut hits = Vec::new();
    if let Some(addr) = parse_addr(q) {
        let place = match app.function_of(addr) {
            Some(f) if f != addr => format!("{}+{:#x}", app.label(f), addr - f),
            Some(f) => app.label(f),
            None => "address".into(),
        };
        hits.push(Hit {
            label: format!("{addr:#x}"),
            detail: place,
            action: Action::Goto(addr),
        });
    }
    let mut named: Vec<(u8, Hit)> = Vec::new();
    for f in &app.index.functions {
        let name = app.display_of(f.entry).unwrap_or_else(|| f.name.clone());
        if let Some(r) = rank_any(&app.search_names(f.entry, &f.name), q) {
            named.push((
                r,
                Hit {
                    label: name,
                    detail: format!("function  {:#x}", f.entry),
                    action: Action::Goto(f.entry),
                },
            ));
        }
    }
    for i in &app.index.imports {
        let shown = match &i.demangled {
            Some(d) if app.demangle => d.short.clone(),
            _ => i.name.clone(),
        };
        let mut forms = vec![i.name.as_str()];
        forms.extend(
            i.demangled
                .iter()
                .flat_map(|d| [d.short.as_str(), d.full.as_str()]),
        );
        if let Some(r) = rank_any(&forms, q) {
            named.push((
                r + 1,
                Hit {
                    label: shown,
                    detail: format!("import  {:#x}", i.slot),
                    action: Action::Goto(i.slot),
                },
            ));
        }
    }
    named.sort_by(|a, b| {
        a.0.cmp(&b.0)
            .then_with(|| a.1.label.len().cmp(&b.1.label.len()))
    });
    hits.extend(named.into_iter().map(|(_, h)| h).take(MAX_HITS));
    if hits.len() < MAX_HITS {
        let room = MAX_HITS - hits.len();
        hits.extend(
            app.index
                .strings
                .iter()
                .filter(|s| rank(&s.text, q).is_some())
                .take(room.min(5))
                .map(|s| Hit {
                    label: format!("\"{}\"", s.text.replace('\n', " ")),
                    detail: format!("string  {:#x}", s.addr),
                    action: Action::Goto(s.addr),
                }),
        );
    }
    hits.truncate(MAX_HITS);
    hits
}

fn command_hits(q: &str) -> Vec<Hit> {
    // Real substring matches first (exact, prefix, contains), then letters-in-order matches.
    let mut hits: Vec<(u8, Hit)> = Command::ALL
        .iter()
        .filter_map(|&(c, label, keys)| {
            let rank = if q.is_empty() {
                0
            } else {
                rank(label, q).or_else(|| subsequence(label, q).then_some(3))?
            };
            Some((
                rank,
                Hit {
                    label: label.to_owned(),
                    detail: keys.to_owned(),
                    action: Action::Command(c),
                },
            ))
        })
        .collect();
    hits.sort_by_key(|(rank, _)| *rank);
    hits.into_iter().map(|(_, h)| h).collect()
}

/// Whether the letters of `needle` appear in order in `hay` (a fuzzy match).
fn subsequence(hay: &str, needle: &str) -> bool {
    let mut h = hay.chars().flat_map(char::to_lowercase);
    needle
        .chars()
        .flat_map(char::to_lowercase)
        .filter(|c| !c.is_whitespace())
        .all(|n| h.any(|c| c == n))
}

/// Why `name` cannot rename `local`: an identifier in the shown document already prints as it.
/// Empty and emitted names clear the rename instead, so they never collide.
fn local_name_is_free(app: &App, function: u64, local: &str, name: &str) -> Result<(), String> {
    if name.is_empty() || name == local {
        return Ok(());
    }
    let Some(doc) = app.pseudo.doc.as_ref().filter(|doc| doc.entry == function) else {
        return Ok(());
    };
    for other in &doc.locals {
        if other == local {
            continue;
        }
        let shown = app.notes.local(function, other).unwrap_or(other);
        if shown == name {
            return Err(format!("{other} is already named {name}"));
        }
    }
    for token in &doc.tokens {
        if token.kind != TokenKind::Identifier
            || matches!(&token.symbol, Some(SymbolRef::Local { name }) if name == local)
        {
            continue;
        }
        let text = &doc.body[token.start as usize..token.end as usize];
        if text == name {
            return Err(format!("{name} is already an identifier in this function"));
        }
    }
    Ok(())
}

/// The text of `token` in the document it belongs to.
fn token_text<'a>(doc: &'a biara_session::Decompiled, token: &Token) -> &'a str {
    &doc.body[token.start as usize..token.end as usize]
}

/// Whether the highlighted word is a member field: an identifier right after `->` or `.`.
fn is_field_word(doc: &biara_session::Decompiled, word: &str) -> bool {
    doc.tokens.windows(2).any(|pair| {
        pair[1].kind == TokenKind::Identifier
            && token_text(doc, &pair[1]) == word
            && matches!(token_text(doc, &pair[0]), "->" | ".")
    })
}

/// The name a signature gives parameter `param`, or `arg<param>` when it has none.
fn parameter_label(signature: &str, param: usize) -> String {
    let param_text = signature
        .find('(')
        .and_then(|open| {
            signature
                .rfind(')')
                .filter(|&close| close > open)
                .map(|close| (open, close))
        })
        .and_then(|(open, close)| signature[open + 1..close].split(',').nth(param))
        .map(str::trim);
    let Some(text) = param_text else {
        return format!("arg{param}");
    };
    let start = text
        .rfind(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .map_or(0, |i| i + 1);
    let (ty, ident) = (text[..start].trim(), &text[start..]);
    if ty.is_empty() || ident.is_empty() || ty.ends_with("struct") {
        format!("arg{param}")
    } else {
        ident.to_owned()
    }
}

/// The signature to edit for `function`: the declared one, else the shown decompilation's.
fn function_signature(app: &App, function: u64) -> Option<String> {
    app.notes
        .signature(function)
        .map(str::to_owned)
        .or_else(|| {
            app.pseudo
                .doc
                .as_ref()
                .filter(|doc| doc.entry == function)
                .map(|doc| doc.signature.clone())
        })
}

/// What `N` does to a highlighted struct field: rename it, name the struct it belongs to
/// first, or explain why neither is possible.
fn field_rename_mode(app: &App) -> Option<Mode> {
    if app.main_tab != MainTab::Pseudo {
        return None;
    }
    let word = app.pseudo.word.as_deref()?;
    let doc = app.pseudo.doc.as_ref()?;
    if !is_field_word(doc, word) {
        return None;
    }
    let layouts: Vec<&StructLayout> = doc
        .layouts
        .iter()
        .filter(|layout| layout.fields.iter().any(|field| field.name == word))
        .collect();
    let layout = match layouts.as_slice() {
        [] => {
            return Some(Mode::Notice {
                message: format!("No struct in this function has a field named {word}."),
            });
        }
        [layout] => *layout,
        several => {
            let names: Vec<&str> = several.iter().map(|layout| layout.name.as_str()).collect();
            return Some(Mode::Notice {
                message: format!(
                    "{word} is a field of more than one struct: {}.",
                    names.join(", ")
                ),
            });
        }
    };
    let Some(field) = layout.fields.iter().find(|field| field.name == word) else {
        return None;
    };
    if layout.declared {
        return Some(Mode::RenameField {
            struct_name: layout.name.clone(),
            offset: field.offset,
            field: word.to_owned(),
        });
    }
    if layout.params.is_empty() {
        return Some(Mode::Notice {
            message: format!(
                "Struct {} is recovered from this decompile, but no parameter points at it. \
                 Only structs a parameter points at can be named.",
                layout.name
            ),
        });
    }
    Some(Mode::NameStruct {
        function: doc.entry,
        layout: layout.clone(),
        field: word.to_owned(),
        offset: field.offset,
    })
}

/// Put `previous` back as struct `name`'s declaration; a failed restore is logged, not hidden.
fn restore_struct(app: &mut App, name: &str, previous: Option<StructAnnotation>) {
    match app.notes.set_struct(name, previous) {
        Ok(Some(problem)) => app
            .log
            .push(format!("notes: could not restore struct {name}: {problem}")),
        Ok(None) => {}
        Err(problem) => app
            .log
            .push(format!("notes: could not restore struct {name}: {problem}")),
    }
}

/// Put `previous` back as `function`'s signature; a failed restore is logged, not hidden.
fn restore_signature(app: &mut App, function: u64, previous: Option<String>) {
    let signature = previous.as_deref().unwrap_or("");
    match app.notes.set_signature(function, signature) {
        Ok(Some(problem)) => app
            .log
            .push(format!("notes: could not restore signature: {problem}")),
        Ok(None) => {}
        Err(problem) => app
            .log
            .push(format!("notes: could not restore signature: {problem}")),
    }
}

/// Declare the recovered struct under the typed name, retype the parameter that points at it,
/// then ask for the field name.
fn submit_struct_name(
    app: &mut App,
    function: u64,
    layout: StructLayout,
    field: String,
    offset: i64,
) {
    let name = app.overlay.input.text.trim().to_owned();
    let Some(&param) = layout.params.first() else {
        app.overlay.error = Some("Only structs a parameter points at can be named.".into());
        return;
    };
    let previous_struct = app.notes.structs().get(&name).cloned();
    let previous_signature = app.notes.signature(function).map(str::to_owned);
    let signature = function_signature(app, function).unwrap_or_default();
    match app
        .notes
        .set_struct(&name, Some(declare_from_layout(&layout)))
    {
        Ok(Some(problem)) => app.log.push(format!("notes: could not save: {problem}")),
        Ok(None) => {}
        Err(e) => {
            app.overlay.error = Some(e.to_string());
            return;
        }
    }
    let bound = match bind_param(&signature, param as usize, &name, app.notes.structs()) {
        Ok(bound) => bound,
        Err(e) => {
            restore_struct(app, &name, previous_struct);
            app.overlay.error = Some(e);
            return;
        }
    };
    match app.notes.set_signature(function, &bound) {
        Ok(Some(problem)) => app.log.push(format!("notes: could not save: {problem}")),
        Ok(None) => {}
        Err(e) => {
            restore_struct(app, &name, previous_struct);
            app.overlay.error = Some(e.to_string());
            return;
        }
    }
    let annotations = app.notes.annotations();
    let applied = app
        .decomp
        .as_ref()
        .map_or(Ok(()), |d| d.set_types(annotations));
    match applied {
        Ok(()) => {
            app.names_changed();
            open(
                app,
                Mode::RenameField {
                    struct_name: name,
                    offset,
                    field,
                },
            );
        }
        Err(e) => {
            restore_signature(app, function, previous_signature);
            restore_struct(app, &name, previous_struct);
            app.overlay.error = Some(e);
        }
    }
}

/// Rename a field of a declared struct and rebind the declarations.
fn submit_field_name(app: &mut App, struct_name: String, offset: i64) {
    let field = app.overlay.input.text.trim().to_owned();
    let previous = app.notes.structs().get(&struct_name).cloned();
    match app.notes.rename_field(&struct_name, offset, &field) {
        Ok(Some(problem)) => app.log.push(format!("notes: could not save: {problem}")),
        Ok(None) => {}
        Err(e) => {
            app.overlay.error = Some(e.to_string());
            return;
        }
    }
    let annotations = app.notes.annotations();
    let applied = app
        .decomp
        .as_ref()
        .map_or(Ok(()), |d| d.set_types(annotations));
    match applied {
        Ok(()) => {
            app.names_changed();
            close(app);
        }
        Err(e) => {
            restore_struct(app, &struct_name, previous);
            app.overlay.error = Some(e);
        }
    }
}

/// Enter: run the selected result, or apply the rename.
pub fn submit(app: &mut App, tx: &mut Tx) {
    match app.overlay.mode.clone() {
        Some(Mode::Rename { addr, kind }) => {
            let name = app.overlay.input.text.trim().to_owned();
            match app.notes.set(addr, kind, &name) {
                Ok(saved_error) => {
                    if let Some(e) = saved_error {
                        app.log.push(format!("notes: could not save: {e}"));
                    }
                    app.names_changed();
                    close(app);
                }
                Err(e) => app.overlay.error = Some(e.to_string()),
            }
        }
        Some(Mode::RenameLocal { function, local }) => {
            let name = app.overlay.input.text.trim().to_owned();
            if let Err(e) = local_name_is_free(app, function, &local, &name) {
                app.overlay.error = Some(e);
                return;
            }
            match app.notes.set_local(function, &local, &name) {
                Ok(saved_error) => {
                    if let Some(e) = saved_error {
                        app.log.push(format!("notes: could not save: {e}"));
                    }
                    app.names_changed();
                    close(app);
                }
                Err(e) => app.overlay.error = Some(e.to_string()),
            }
        }
        Some(Mode::Comment { addr }) => {
            let text = app.overlay.input.text.trim().to_owned();
            match app.notes.set_comment(addr, &text) {
                Ok(saved_error) => {
                    if let Some(e) = saved_error {
                        app.log.push(format!("notes: could not save: {e}"));
                    }
                    app.names_changed();
                    close(app);
                }
                Err(e) => app.overlay.error = Some(e.to_string()),
            }
        }
        Some(Mode::Signature { function }) => {
            let signature = app.overlay.input.text.trim().to_owned();
            let previous = app.notes.signature(function).map(str::to_owned);
            match app.notes.set_signature(function, &signature) {
                Ok(saved_error) => {
                    let annotations = app.notes.annotations();
                    let applied = app
                        .decomp
                        .as_ref()
                        .map_or(Ok(()), |d| d.set_types(annotations));
                    match applied {
                        Ok(()) => {
                            if let Some(e) = saved_error {
                                app.log.push(format!("notes: could not save: {e}"));
                            }
                            app.names_changed();
                            close(app);
                        }
                        Err(e) => {
                            // The engine rejected the signature, so put the previous one back:
                            // a failed edit changes nothing.
                            restore_signature(app, function, previous);
                            app.overlay.error = Some(e);
                        }
                    }
                }
                Err(e) => app.overlay.error = Some(e.to_string()),
            }
        }
        Some(Mode::NameStruct {
            function,
            layout,
            field,
            offset,
        }) => submit_struct_name(app, function, layout, field, offset),
        Some(Mode::RenameField {
            struct_name,
            offset,
            ..
        }) => submit_field_name(app, struct_name, offset),
        Some(Mode::Notice { .. }) => close(app),
        Some(Mode::BlockLimit) => match parse_count(&app.overlay.input.text) {
            Some(limit) => {
                close(app);
                app.set_block_limit(limit);
            }
            None => {
                app.overlay.error =
                    Some("Enter a positive number of blocks, such as 2000000 or 2M".into())
            }
        },
        Some(_) => {
            let Some(hit) = app.overlay.hits.get(app.overlay.selected).cloned() else {
                return;
            };
            close(app);
            run(app, hit.action, tx);
        }
        None => {}
    }
}

pub fn run(app: &mut App, action: Action, tx: &mut Tx) {
    match action {
        Action::Goto(addr) => app.goto(addr, tx),
        Action::Command(c) => command(app, c, tx),
    }
}

pub fn command(app: &mut App, c: Command, tx: &mut Tx) {
    match c {
        Command::Goto => open(app, Mode::Goto),
        Command::Rename => {
            if let Some(m) = rename_mode(app) {
                open(app, m);
            }
        }
        Command::RenameLocal => {
            if let Some(m) = local_rename_mode(app) {
                open(app, m);
            }
        }
        Command::RenameField => {
            if let Some(m) = field_rename_mode(app) {
                open(app, m);
            }
        }
        Command::Comment => open(app, Mode::Comment { addr: app.cur }),
        Command::Signature => {
            if let Some(function) = app.function_of(app.cur) {
                open(app, Mode::Signature { function });
            }
        }
        Command::OpenFile => open_file(tx),
        Command::Tab(t) => app.select_tab(t, tx),
        Command::ToggleGraph => {
            let next = if app.main_tab == MainTab::Graph {
                MainTab::Listing
            } else {
                MainTab::Graph
            };
            app.select_tab(next, tx);
        }
        Command::Back => app.go_back(tx),
        Command::Forward => app.go_forward(tx),
        Command::Xrefs => app.bottom_tab = BottomTab::Xrefs,
        Command::Output => app.bottom_tab = BottomTab::Output,
        Command::ToggleDemangle => app.toggle_demangle(),
        Command::BlockLimit => open(app, Mode::BlockLimit),
    }
}

/// The local or parameter highlighted in the pseudocode, if there is one.
fn local_rename_mode(app: &App) -> Option<Mode> {
    if app.main_tab != MainTab::Pseudo {
        return None;
    }
    let word = app.pseudo.word.as_deref()?;
    let doc = app.pseudo.doc.as_ref()?;
    doc.locals
        .iter()
        .any(|local| local == word)
        .then(|| Mode::RenameLocal {
            function: doc.entry,
            local: word.to_owned(),
        })
}

/// What a rename would apply to: the highlighted field or local in the pseudocode; else the
/// symbol at the cursor, else the current function.
fn rename_mode(app: &App) -> Option<Mode> {
    if let Some(mode) = local_rename_mode(app) {
        return Some(mode);
    }
    if let Some(mode) = field_rename_mode(app) {
        return Some(mode);
    }
    let target = if app.index.base_name(app.cur).is_some() || app.notes.get(app.cur).is_some() {
        app.cur
    } else {
        app.function_of(app.cur)?
    };
    let kind = if app.index.function(target).is_some() {
        NameKind::Function
    } else {
        NameKind::Data
    };
    Some(Mode::Rename { addr: target, kind })
}

/// Pick a binary with the system file dialog.
pub fn open_file(tx: &mut Tx) {
    let options = FileDialogOptions {
        title: "Open binary".into(),
        ..Default::default()
    };
    tx.open_file(options, |s: &mut App, result, _tx| match result {
        Ok(Some(path)) => s.open(path),
        Ok(None) => {}
        Err(e) => s.log.push(format!("open: {e}")),
    });
}

/// Keys for the whole window that open or act on dialogs. True if handled.
pub fn global_key(app: &mut App, key: &KeyEvent, tx: &mut Tx) -> bool {
    match &key.key {
        Key::Command('k') | Key::Command('p') if key.mods.cmd => {
            open(app, Mode::Commands);
            true
        }
        Key::Command('o') if key.mods.cmd => {
            open_file(tx);
            true
        }
        Key::Text(t) if !key.mods.cmd && !key.mods.alt => match t.as_str() {
            "g" => {
                open(app, Mode::Goto);
                true
            }
            "n" => match rename_mode(app) {
                Some(m) => {
                    open(app, m);
                    true
                }
                None => false,
            },
            "x" => {
                app.bottom_tab = BottomTab::Xrefs;
                true
            }
            ";" => {
                open(app, Mode::Comment { addr: app.cur });
                true
            }
            "y" => match app.function_of(app.cur) {
                Some(function) => {
                    open(app, Mode::Signature { function });
                    true
                }
                None => false,
            },
            _ => false,
        },
        _ => false,
    }
}

// View -------------------------------------------------------------------------------------

/// The overlay, if one is open.
pub fn view(app: &App) -> Option<El> {
    let mode = app.overlay.mode.as_ref()?;
    let p = colors();
    let (title, placeholder) = match mode {
        Mode::Goto => ("Go to", "Address, function, import or string".to_owned()),
        Mode::Commands => ("Commands", "Type a command".to_owned()),
        Mode::Rename { addr, .. } => ("Rename", format!("New name for {}", app.label(*addr))),
        Mode::RenameLocal { function, local } => (
            "Rename local",
            format!(
                "New name for {}",
                app.notes
                    .local(*function, local)
                    .map_or_else(|| app.local_name(local), str::to_owned)
            ),
        ),
        Mode::NameStruct {
            function, layout, ..
        } => {
            let param = layout.params.first().copied().unwrap_or(0) as usize;
            let signature = function_signature(app, *function).unwrap_or_default();
            (
                "Name struct",
                format!(
                    "Name struct {} (what {} points to) to rename its fields",
                    layout.name,
                    parameter_label(&signature, param)
                ),
            )
        }
        Mode::RenameField {
            struct_name, field, ..
        } => (
            "Rename field",
            format!("New name for {struct_name}.{field}"),
        ),
        Mode::Comment { addr } => ("Comment", format!("One line about {addr:#x}")),
        Mode::Signature { function } => (
            "Signature",
            format!("C signature of {}", app.label(*function)),
        ),
        Mode::Notice { .. } => ("Cannot rename field", String::new()),
        Mode::BlockLimit => (
            "Block limit",
            format!(
                "Blocks to recover before stopping (now {})",
                app.block_limit()
            ),
        ),
    };
    let notice = match mode {
        Mode::Notice { message } => Some(message.clone()),
        _ => None,
    };
    let mut card = col()
        .key("overlay-card")
        .w(580.0)
        .radius(12.0)
        .fill(p.raised)
        .border(1.0, p.line_strong)
        .shadow(18.0, 48.0, Color::BLACK.alpha(0.5))
        .child(
            row()
                .w_full()
                .px(16.0)
                .pt(12.0)
                .child(text(title).font(SMALL_MEDIUM).color(p.muted)),
        );
    if let Some(message) = notice {
        card = card.child(
            row()
                .w_full()
                .px(16.0)
                .pb(14.0)
                .child(text(message).font(UI).color(p.bad)),
        );
    } else {
        let input = TextInput::new(&app.overlay.input)
            .placeholder(placeholder)
            .leading_icon(&widgets::SEARCH)
            .width(Some(554.0))
            .on_edit(|s: &mut App| &mut s.overlay.input)
            .on_change(|s: &mut App, _tx: &mut Tx| update(s))
            .on_submit(|s: &mut App, tx: &mut Tx| submit(s, tx))
            .on_key(|s: &mut App, k: &KeyEvent, _tx: &mut Tx| match k.key {
                Key::Down if !s.overlay.hits.is_empty() => {
                    s.overlay.selected = (s.overlay.selected + 1) % s.overlay.hits.len();
                    true
                }
                Key::Up if !s.overlay.hits.is_empty() => {
                    s.overlay.selected =
                        (s.overlay.selected + s.overlay.hits.len() - 1) % s.overlay.hits.len();
                    true
                }
                _ => false,
            })
            .build(kit());
        card = card.child(col().w_full().px(12.0).py(8.0).child(input));
    }
    if let Some(e) = &app.overlay.error {
        card = card.child(
            row()
                .w_full()
                .px(16.0)
                .pb(10.0)
                .child(text(e.clone()).font(SMALL).color(p.bad)),
        );
    }
    if !app.overlay.hits.is_empty() {
        let mut list = col().w_full().pb(8.0).px(6.0);
        for (i, hit) in app.overlay.hits.iter().enumerate() {
            let action = hit.action.clone();
            let selected = i == app.overlay.selected;
            list = list.child(
                row()
                    .key(format!("hit{i}"))
                    .w_full()
                    .h(30.0)
                    .px(10.0)
                    .gap(12.0)
                    .items_center()
                    .radius(6.0)
                    .fill(if selected {
                        p.selected
                    } else {
                        Color::TRANSPARENT
                    })
                    .hover(p.hover)
                    .on_press_tx(move |s: &mut App, tx| {
                        close(s);
                        run(s, action.clone(), tx);
                    })
                    .child(
                        text(hit.label.clone())
                            .font(CODE)
                            .color(p.text)
                            .truncate()
                            .grow(1.0)
                            .min_w(0.0),
                    )
                    .child(text(hit.detail.clone()).font(SMALL).color(p.faint)),
            );
        }
        card = card.child(el().w_full().h(1.0).fill(p.line)).child(list);
    }
    Some(
        el().key("overlay")
            .absolute(0.0, 0.0)
            .w_full()
            .h_full()
            .fill(Color::BLACK.alpha(0.45))
            .items_start()
            .justify_center()
            .pt(110.0)
            .modal_tx(true, |s: &mut App, _tx: &mut Tx| close(s))
            .child(card),
    )
}

/// The best rank of `q` among several spellings of one name.
fn rank_any<S: AsRef<str>>(names: &[S], q: &str) -> Option<u8> {
    names.iter().filter_map(|n| rank(n.as_ref(), q)).min()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn substring_matches_outrank_scattered_letters() {
        let labels: Vec<_> = command_hits("gr").into_iter().map(|h| h.label).collect();
        let show = labels
            .iter()
            .position(|l| l == "Show graph")
            .expect("substring match listed");
        let goto = labels
            .iter()
            .position(|l| l == "Go to address or name…")
            .expect("fuzzy match listed");
        assert!(show < goto, "{labels:?}");
    }

    #[test]
    fn an_empty_query_lists_every_command_in_order() {
        assert_eq!(command_hits("").len(), Command::ALL.len());
        assert_eq!(
            command_hits("").first().map(|h| h.label.as_str()),
            Some("Go to address or name…")
        );
        assert!(command_hits("zzzz").is_empty());
    }
}
