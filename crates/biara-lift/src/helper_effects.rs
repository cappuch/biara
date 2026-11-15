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

unsafe extern "C" {
    fn biara_helper_effects(
        path: *const c_char,
        symbols: *const *const c_char,
        n: usize,
        offsets: *const i64,
        sizes: *const u32,
        m: usize,
        reads: *mut u8,
        writes: *mut u8,
        err: *mut *mut c_char,
    ) -> i32;
    #[link_name = "free"]
    fn libc_free(p: *mut std::ffi::c_void);
}

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

fn analyze(path: &Path, info: &ContextInfo, symbols: &[&str]) -> Result<Vec<Access>, Error> {
    let path_c = CString::new(path.as_os_str().as_encoded_bytes())
        .map_err(|_| Error::Llvm(format!("{}: NUL in path", path.display())))?;
    let names: Vec<CString> = symbols.iter().map(|s| CString::new(*s).unwrap()).collect();
    let name_ptrs: Vec<*const c_char> = names.iter().map(|s| s.as_ptr()).collect();
    let offsets: Vec<i64> = info.globals.iter().map(|g| g.mem_offset).collect();
    let sizes: Vec<u32> = info.globals.iter().map(|g| g.ty.bits() / 8).collect();
    let m = offsets.len();
    let mut reads = vec![1u8; symbols.len() * m];
    let mut writes = vec![1u8; symbols.len() * m];
    let mut error = std::ptr::null_mut();
    let failed = unsafe {
        biara_helper_effects(
            path_c.as_ptr(),
            name_ptrs.as_ptr(),
            name_ptrs.len(),
            offsets.as_ptr(),
            sizes.as_ptr(),
            m,
            reads.as_mut_ptr(),
            writes.as_mut_ptr(),
            &mut error,
        )
    };
    if failed != 0 {
        let message = unsafe { CStr::from_ptr(error) }
            .to_string_lossy()
            .into_owned();
        unsafe { libc_free(error.cast()) };
        return Err(Error::Llvm(message));
    }
    Ok((0..symbols.len())
        .map(|i| Access {
            reads: reads[i * m..(i + 1) * m].iter().map(|&b| b != 0).collect(),
            writes: writes[i * m..(i + 1) * m].iter().map(|&b| b != 0).collect(),
        })
        .collect())
}
