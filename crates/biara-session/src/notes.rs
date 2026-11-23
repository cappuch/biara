//! What the user (or an agent) has written about a binary: names for functions and data,
//! names for a function's locals and parameters, function signatures, struct declarations,
//! and comments.
//!
//! Stored in the engine's own annotation format (`biara_proto::annotations::Annotations`), so
//! `biara lift --annotations <file>` reads what was saved. The file lives in the user's data
//! directory, keyed by the binary's path, so read-only locations such as `/usr/lib` work and
//! nothing is written next to the binary.
//!
//! Several processes may edit one file: the workbench and `biara mcp` open the same one.
//! Every change first takes in what the others saved, then writes atomically, and
//! [`watch`] tells a long-lived reader when to call [`Notes::reload_if_changed`].

use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, SystemTime},
};

use biara_proto::annotations::{Annotations, FunctionAnnotation, StructAnnotation};

use super::Notify;

/// What a name is attached to; the annotation file keeps functions and data apart.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NameKind {
    Function,
    Data,
}

/// The notes on one binary, plus where they are saved.
#[derive(Debug, Default)]
pub struct Notes {
    saved: Annotations,
    file: Option<PathBuf>,
    /// The file as last read or written; anything else means another process wrote it.
    stamp: Option<Stamp>,
    /// Bumped on every change; views cache against it.
    version: u64,
    /// Bumped when a signature or struct changes: decompilations made before are stale.
    types_version: u64,
}

/// Identifies one write of the file: atomic saves replace it, so the inode changes too.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Stamp {
    modified: SystemTime,
    len: u64,
    inode: u64,
}

impl Stamp {
    fn of(path: &Path) -> Option<Stamp> {
        let m = std::fs::metadata(path).ok()?;
        #[cfg(unix)]
        let inode = std::os::unix::fs::MetadataExt::ino(&m);
        #[cfg(not(unix))]
        let inode = 0;
        Some(Stamp {
            modified: m.modified().ok()?,
            len: m.len(),
            inode,
        })
    }
}

/// Why a note was rejected.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NameError(pub String);

impl std::fmt::Display for NameError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// A valid C identifier that is not a keyword: the renderer needs names it can print as code.
pub fn validate(name: &str) -> Result<(), NameError> {
    let mut chars = name.chars();
    let first = chars
        .next()
        .ok_or_else(|| NameError("a name cannot be empty".into()))?;
    if !(first.is_ascii_alphabetic() || first == '_') {
        return Err(NameError(
            "a name starts with a letter or underscore".into(),
        ));
    }
    if let Some(bad) = chars.find(|c| !(c.is_ascii_alphanumeric() || *c == '_')) {
        return Err(NameError(format!(
            "{bad:?} is not allowed in a name; use letters, digits and underscores"
        )));
    }
    if !biara_proto::annotations::is_identifier(name) {
        return Err(NameError(format!("{name:?} is a C keyword")));
    }
    Ok(())
}

/// What one save did, for the caller to report: `Some(error)` when writing the file failed.
/// The change still applies for this session.
pub type Saved = Option<String>;

impl Notes {
    /// The notes saved for `binary`, if any. A missing file is an empty set; an unreadable one
    /// is too, with the error returned so the caller can say so.
    pub fn load(binary: &Path) -> (Notes, Option<String>) {
        let mut notes = Notes {
            file: Self::file_for(binary),
            ..Notes::default()
        };
        let problem = notes.read().err();
        (notes, problem)
    }

    /// A set that is never written to disk.
    pub fn in_memory() -> Notes {
        Notes::default()
    }

    /// Where `binary`'s notes live: `<data dir>/biara/names/<name>-<hash of the path>.json`.
    pub fn file_for(binary: &Path) -> Option<PathBuf> {
        let data = std::env::var_os("XDG_DATA_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| Path::new(&h).join(".local/share")))?;
        let canonical = binary.canonicalize().unwrap_or_else(|_| binary.to_owned());
        let hash = blake3::hash(canonical.to_string_lossy().as_bytes()).to_hex();
        let stem = canonical
            .file_name()
            .map_or_else(|| "binary".into(), |n| n.to_string_lossy().into_owned());
        Some(
            data.join("biara/names")
                .join(format!("{stem}-{}.json", &hash[..16])),
        )
    }

    pub fn path(&self) -> Option<&Path> {
        self.file.as_deref()
    }

    pub fn version(&self) -> u64 {
        self.version
    }

    /// Changes with any signature or struct declaration; see [`crate::Decompiler::set_types`].
    pub fn types_version(&self) -> u64 {
        self.types_version
    }

    /// Everything, in the annotation format.
    pub fn annotations(&self) -> &Annotations {
        &self.saved
    }

    /// The name given to the function or data at `addr`.
    pub fn get(&self, addr: u64) -> Option<&str> {
        self.saved
            .functions
            .get(&addr)
            .and_then(|f| f.name.as_deref())
            .or_else(|| self.saved.globals.get(&addr).map(String::as_str))
    }

    /// The name given to the local or parameter the decompiler emitted as `emitted` in
    /// `function`.
    pub fn local(&self, function: u64, emitted: &str) -> Option<&str> {
        self.locals(function)?.get(emitted).map(String::as_str)
    }

    /// `function`'s renamed locals, emitted name to given name.
    pub fn locals(&self, function: u64) -> Option<&BTreeMap<String, String>> {
        self.saved
            .functions
            .get(&function)
            .map(|f| &f.locals)
            .filter(|l| !l.is_empty())
    }

    pub fn signature(&self, function: u64) -> Option<&str> {
        self.saved.functions.get(&function)?.signature.as_deref()
    }

    /// Every declared signature, by function entry.
    pub fn signatures(&self) -> BTreeMap<u64, String> {
        self.saved
            .functions
            .iter()
            .filter_map(|(&a, f)| Some((a, f.signature.clone()?)))
            .collect()
    }

    /// Declared structs, by name (without `struct `).
    pub fn structs(&self) -> &BTreeMap<String, StructAnnotation> {
        &self.saved.structs
    }

    pub fn comment(&self, addr: u64) -> Option<&str> {
        self.saved.comments.get(&addr).map(String::as_str)
    }

    pub fn comments(&self) -> &BTreeMap<u64, String> {
        &self.saved.comments
    }

    /// Name `addr`, or remove its name when `name` is empty.
    pub fn set(&mut self, addr: u64, kind: NameKind, name: &str) -> Result<Saved, NameError> {
        if !name.is_empty() {
            validate(name)?;
        }
        Ok(self.change(|a| {
            let name = (!name.is_empty()).then(|| name.to_owned());
            match kind {
                NameKind::Function => function_entry(a, addr, |f| f.name = name),
                NameKind::Data => match name {
                    Some(n) => {
                        a.globals.insert(addr, n);
                    }
                    None => {
                        a.globals.remove(&addr);
                    }
                },
            }
        }))
    }

    /// Name the local or parameter emitted as `emitted` in `function`, or restore the emitted
    /// name when `name` is empty or equal to it. The caller checks that the local exists and
    /// that the name is free in the function's body; this checks the name and that no other
    /// local of the function already has it.
    pub fn set_local(
        &mut self,
        function: u64,
        emitted: &str,
        name: &str,
    ) -> Result<Saved, NameError> {
        validate(emitted)?;
        let clear = name.is_empty() || name == emitted;
        if !clear {
            validate(name)?;
            if let Some((other, _)) = self
                .locals(function)
                .and_then(|l| l.iter().find(|(e, n)| *n == name && *e != emitted))
            {
                return Err(NameError(format!("{other} is already named {name}")));
            }
        }
        Ok(self.change(|a| {
            function_entry(a, function, |f| {
                if clear {
                    f.locals.remove(emitted);
                } else {
                    f.locals.insert(emitted.to_owned(), name.to_owned());
                }
            })
        }))
    }

    /// Declare `function`'s C signature (`int parse(char *buf, size_t len)`), or drop it when
    /// `signature` is empty. Only the shape is checked here; the decompiler checks it against
    /// the function's calling convention ([`crate::Decompiler::set_signatures`]).
    pub fn set_signature(&mut self, function: u64, signature: &str) -> Result<Saved, NameError> {
        let signature = signature.trim();
        if signature.contains(['\n', '\r']) {
            return Err(NameError("a signature is one line".into()));
        }
        let saved = self.change(|a| {
            function_entry(a, function, |f| {
                f.signature = (!signature.is_empty()).then(|| signature.to_owned());
            })
        });
        self.types_version += 1;
        Ok(saved)
    }

    /// Declare struct `name` with `decl`'s fields, replacing any earlier declaration, or drop
    /// it when `decl` is `None`. Checks the name and that the fields are well formed; whether
    /// it fits a layout is up to the decompiler, which binds it only through a parameter a
    /// signature types `struct name *`.
    pub fn set_struct(
        &mut self,
        name: &str,
        decl: Option<StructAnnotation>,
    ) -> Result<Saved, NameError> {
        validate(name)?;
        if let Some(decl) = &decl {
            for field in &decl.fields {
                validate(&field.name)?;
            }
            let check = Annotations {
                structs: BTreeMap::from([(name.to_owned(), decl.clone())]),
                ..Annotations::default()
            };
            check.validate().map_err(|e| NameError(e.to_string()))?;
        } else if let Some(user) = self.saved.functions.iter().find_map(|(&a, f)| {
            f.signature
                .as_deref()
                .filter(|s| names_struct(s, name))
                .map(|_| a)
        }) {
            return Err(NameError(format!(
                "the signature of {user:#x} uses struct {name}; change it first"
            )));
        }
        let saved = self.change(|a| match decl {
            Some(decl) => {
                a.structs.insert(name.to_owned(), decl);
            }
            None => {
                a.structs.remove(name);
            }
        });
        self.types_version += 1;
        Ok(saved)
    }

    /// Name the field at `offset` of declared struct `name`.
    pub fn rename_field(
        &mut self,
        name: &str,
        offset: i64,
        field: &str,
    ) -> Result<Saved, NameError> {
        validate(field)?;
        let decl = self
            .saved
            .structs
            .get(name)
            .ok_or_else(|| NameError(format!("struct {name} is not declared")))?;
        if let Some(other) = decl
            .fields
            .iter()
            .find(|f| f.name == field && f.offset != offset)
        {
            return Err(NameError(format!(
                "the field at {:#x} is already named {field}",
                other.offset
            )));
        }
        let mut decl = decl.clone();
        let slot = decl
            .fields
            .iter_mut()
            .find(|f| f.offset == offset)
            .ok_or_else(|| NameError(format!("struct {name} has no field at {offset:#x}")))?;
        slot.name = field.to_owned();
        self.set_struct(name, Some(decl))
    }

    /// Comment `addr`, or remove its comment when `text` is empty. One line.
    pub fn set_comment(&mut self, addr: u64, text: &str) -> Result<Saved, NameError> {
        let text = text.trim();
        if text.contains(['\n', '\r']) {
            return Err(NameError("a comment is one line".into()));
        }
        Ok(self.change(|a| {
            if text.is_empty() {
                a.comments.remove(&addr);
            } else {
                a.comments.insert(addr, text.to_owned());
            }
        }))
    }

    /// Take in what another process saved since this one last read or wrote the file. Returns
    /// whether anything changed; an unreadable file keeps the current notes and is an error.
    pub fn reload_if_changed(&mut self) -> Result<bool, String> {
        let Some(path) = &self.file else {
            return Ok(false);
        };
        if Stamp::of(path) == self.stamp {
            return Ok(false);
        }
        let before = (self.signatures(), self.saved.structs.clone());
        self.read()?;
        self.version += 1;
        if (self.signatures(), self.saved.structs.clone()) != before {
            self.types_version += 1;
        }
        Ok(true)
    }

    /// Replace the notes with the file's.
    fn read(&mut self) -> Result<(), String> {
        let Some(path) = &self.file else {
            return Ok(());
        };
        let stamp = Stamp::of(path);
        let text = match std::fs::read_to_string(path) {
            Ok(text) => text,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                self.saved = Annotations::default();
                self.stamp = None;
                return Ok(());
            }
            Err(e) => return Err(format!("{}: {e}", path.display())),
        };
        let parsed: Annotations =
            serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        self.saved = parsed;
        self.stamp = stamp;
        Ok(())
    }

    /// Apply `edit` on top of the file's latest contents, then save.
    fn change(&mut self, edit: impl FnOnce(&mut Annotations)) -> Saved {
        // Another process's edits survive ours; if the file cannot be read, ours still apply.
        let reload = self.reload_if_changed().err();
        edit(&mut self.saved);
        self.version += 1;
        reload.or_else(|| self.save().err())
    }

    /// Write the notes atomically (temp file, then rename).
    fn save(&mut self) -> Result<(), String> {
        let Some(path) = &self.file else {
            return Ok(());
        };
        let write = || -> std::io::Result<()> {
            if let Some(dir) = path.parent() {
                std::fs::create_dir_all(dir)?;
            }
            let tmp = path.with_extension(format!("json.{}.tmp", std::process::id()));
            let json = serde_json::to_vec_pretty(&self.saved).map_err(std::io::Error::other)?;
            std::fs::write(&tmp, json)?;
            std::fs::rename(&tmp, path)
        };
        write().map_err(|e| format!("{}: {e}", path.display()))?;
        self.stamp = Stamp::of(path);
        Ok(())
    }
}

/// Whether C `signature` mentions `struct name`.
fn names_struct(signature: &str, name: &str) -> bool {
    signature.match_indices("struct").any(|(i, _)| {
        let rest = signature[i + "struct".len()..].trim_start();
        rest.strip_prefix(name)
            .is_some_and(|r| !r.starts_with(|c: char| c.is_ascii_alphanumeric() || c == '_'))
    })
}

/// Edit `addr`'s function entry, dropping it once it holds nothing.
fn function_entry(a: &mut Annotations, addr: u64, edit: impl FnOnce(&mut FunctionAnnotation)) {
    let f = a.functions.entry(addr).or_default();
    edit(f);
    if f.name.is_none() && f.signature.is_none() && f.locals.is_empty() {
        a.functions.remove(&addr);
    }
}

/// Calls `notify` whenever the file at `path` changes, checking twice a second, until the
/// returned handle drops.
pub fn watch(path: PathBuf, notify: Notify) -> Watch {
    let stop = Arc::new(AtomicBool::new(false));
    let stopped = stop.clone();
    std::thread::Builder::new()
        .name("biara-notes-watch".into())
        .spawn(move || {
            let mut last = Stamp::of(&path);
            while !stopped.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(500));
                let now = Stamp::of(&path);
                if now != last {
                    last = now;
                    notify();
                }
            }
        })
        .expect("spawn notes watcher");
    Watch { stop }
}

/// Stops its watcher when dropped.
pub struct Watch {
    stop: Arc<AtomicBool>,
}

impl Drop for Watch {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn in_dir(dir: &Path) -> Notes {
        Notes {
            file: Some(dir.join("n.json")),
            ..Notes::default()
        }
    }

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("biara-notes-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn names_must_be_c_identifiers() {
        assert!(validate("parse_header2").is_ok());
        assert!(validate("_x").is_ok());
        assert!(validate("").is_err());
        assert!(validate("2fast").is_err());
        assert!(validate("has space").unwrap_err().0.contains("' '"));
        assert!(validate("a-b").is_err());
        assert!(validate("while").unwrap_err().0.contains("keyword"));
    }

    #[test]
    fn set_get_clear_round_trip_through_the_annotation_file() {
        let dir = temp_dir("round");
        let mut n = in_dir(&dir);
        let v0 = n.version();
        assert_eq!(n.set(0x401000, NameKind::Function, "main_loop"), Ok(None));
        assert_eq!(n.set(0x404010, NameKind::Data, "config"), Ok(None));
        assert_eq!(n.set_local(0x401000, "value3", "count"), Ok(None));
        assert_eq!(n.set_comment(0x401004, "  reads the header  "), Ok(None));
        assert!(n.version() > v0);
        assert_eq!(n.get(0x401000), Some("main_loop"));

        // The file is the CLI's annotation format.
        let text = std::fs::read_to_string(n.path().unwrap()).unwrap();
        let parsed: Annotations = serde_json::from_str(&text).unwrap();
        parsed.validate().unwrap();
        let f = &parsed.functions[&0x401000];
        assert_eq!(f.name.as_deref(), Some("main_loop"));
        assert_eq!(f.locals["value3"], "count");
        assert_eq!(parsed.globals[&0x404010], "config");
        assert_eq!(parsed.comments[&0x401004], "reads the header");

        // A fresh load sees the same notes; clearing removes them from the file, and a
        // function entry left with nothing goes.
        let mut again = in_dir(&dir);
        again.read().unwrap();
        assert_eq!(again.local(0x401000, "value3"), Some("count"));
        n.set(0x401000, NameKind::Function, "").unwrap();
        n.set_local(0x401000, "value3", "value3").unwrap();
        n.set_comment(0x401004, "").unwrap();
        assert_eq!((n.get(0x401000), n.local(0x401000, "value3")), (None, None));
        let parsed: Annotations =
            serde_json::from_str(&std::fs::read_to_string(n.path().unwrap()).unwrap()).unwrap();
        assert!(parsed.functions.is_empty() && parsed.comments.is_empty());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn two_writers_keep_each_others_changes() {
        let dir = temp_dir("two");
        let (mut ui, mut agent) = (in_dir(&dir), in_dir(&dir));
        ui.set(0x10, NameKind::Function, "from_ui").unwrap();
        agent.set_comment(0x14, "from the agent").unwrap();
        ui.set_signature(0x10, "int from_ui(int a)").unwrap();
        // Each change took in the other's first, so the file holds all three.
        let mut reader = in_dir(&dir);
        reader.read().unwrap();
        assert_eq!(reader.get(0x10), Some("from_ui"));
        assert_eq!(reader.comment(0x14), Some("from the agent"));
        assert_eq!(reader.signature(0x10), Some("int from_ui(int a)"));
        // The agent sees the UI's signature change as one.
        let before = agent.types_version();
        assert_eq!(agent.reload_if_changed(), Ok(true));
        assert!(agent.types_version() > before);
        assert_eq!(agent.reload_if_changed(), Ok(false));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn struct_fields_rename_in_place_and_a_struct_in_use_stays() {
        let mut n = Notes::in_memory();
        let field = |offset, name: &str| biara_proto::annotations::StructField {
            offset,
            bytes: 8,
            name: name.into(),
            type_name: String::new(),
        };
        let decl = StructAnnotation {
            fields: vec![field(0, "f_0"), field(8, "f_8")],
        };
        n.set_struct("args", Some(decl)).unwrap();
        let v = n.types_version();
        n.rename_field("args", 8, "count").unwrap();
        assert!(n.types_version() > v, "a field rename re-decompiles");
        assert_eq!(n.structs()["args"].fields[1].name, "count");
        assert!(n.rename_field("args", 0, "count").is_err(), "taken at 8");
        assert!(n.rename_field("args", 16, "x").is_err(), "no field there");
        assert!(n.rename_field("other", 0, "x").is_err(), "not declared");
        let overlapping = StructAnnotation {
            fields: vec![field(0, "a"), field(4, "b")],
        };
        assert!(n.set_struct("bad", Some(overlapping)).is_err());
        n.set_signature(0x10, "void f(struct args *a)").unwrap();
        assert!(n.set_struct("args", None).is_err(), "a signature uses it");
        n.set_signature(0x10, "void f(struct args_v2 *a)").unwrap();
        n.set_struct("args", None).unwrap();
        assert!(n.structs().is_empty());
    }

    #[test]
    fn invalid_notes_change_nothing() {
        let mut n = Notes::in_memory();
        let v = n.version();
        assert!(n.set(0x10, NameKind::Function, "not valid").is_err());
        assert!(n.set_comment(0x10, "two\nlines").is_err());
        n.set_local(0x10, "value1", "count").unwrap();
        let v1 = n.version();
        assert!(
            n.set_local(0x10, "value2", "count").is_err(),
            "taken by value1"
        );
        assert_eq!((v1 > v, n.version(), n.get(0x10)), (true, v1, None));
    }
}
