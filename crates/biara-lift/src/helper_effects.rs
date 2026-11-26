//! Which TCG globals a QEMU helper reads and writes through `env`, from the
//! helper bitcode (cpp/helper_effects.cpp). TCG only says whether a helper
//! touches globals at all; ABI form (`decompile`) uses this to store to the
//! CPU state before a helper only the globals it reads, and to reload after it
//! only those it writes.
//! Per bitcode file, results are computed once per helper and kept for the
//! process.

use std::{
    collections::HashMap,
    ffi::{CStr, CString, c_char},
    path::{Path, PathBuf},
    sync::{Arc, LazyLock, Mutex},
};

use biara_tcg::ContextInfo;

use crate::Error;

/// Per global (by `TempId`): may the helper read it, may it write it.
#[derive(Debug)]
pub(crate) struct Access {
    pub(crate) reads: Vec<bool>,
    pub(crate) writes: Vec<bool>,
}

/// Helper symbol (`helper_<name>`) to its access.
pub(crate) type Effects = HashMap<String, Arc<Access>>;

static KNOWN: LazyLock<Mutex<HashMap<PathBuf, Effects>>> = LazyLock::new(Default::default);

/// The accesses of `symbols` through env, from `info.helpers`; empty without
/// helper bitcode (every helper then stays conservative).
pub(crate) fn effects<'a>(
    info: &ContextInfo,
    symbols: impl IntoIterator<Item = &'a str>,
) -> Result<Effects, Error> {
    let Some(path) = info.helpers.as_deref() else {
        return Ok(Effects::new());
    };
    let mut known = KNOWN.lock().unwrap_or_else(|e| e.into_inner());
    let cached = known.entry(path.to_owned()).or_default();
    let symbols: Vec<&str> = symbols.into_iter().collect();
    let missing: Vec<&str> = symbols
        .iter()
        .copied()
        .filter(|s| !cached.contains_key(*s))
        .collect();
    if !missing.is_empty() {
        for (symbol, access) in missing.iter().zip(analyze(path, info, &missing)?) {
            cached.insert((*symbol).to_owned(), Arc::new(access));
        }
    }
    Ok(symbols
        .iter()
        .filter_map(|s| cached.get(*s).map(|a| ((*s).to_owned(), a.clone())))
        .collect())
}

/// Helper effects came from decoding the helper bitcode and following each
/// helper's pointer parameters to its accesses. That needs LLVM's bitcode
/// reader and value tracking, which are gone, so every helper is reported
/// conservatively: it may read and write every global. That is what the
/// analysis did before a helper's bitcode was available, and it is the safe
/// direction — lifted code stores and reloads more, never fewer, values.
fn analyze(_path: &Path, info: &ContextInfo, symbols: &[&str]) -> Result<Vec<Access>, Error> {
    let m = info.globals.len();
    Ok((0..symbols.len())
        .map(|_| Access {
            reads: vec![true; m],
            writes: vec![true; m],
        })
        .collect())
}
