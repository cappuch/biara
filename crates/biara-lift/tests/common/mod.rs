#![allow(dead_code)] // each test binary uses a subset
//! Building test programs with the pinned clang (BIARA_CLANG, set by
//! .cargo/config.toml).

pub mod corpus;

use std::{
    path::{Path, PathBuf},
    process::Command,
    sync::OnceLock,
};

use biara_cfg::{Cfg, Discovery};
use biara_lift::{ResolveOptions, analysis_machine, recover_resolving};
use biara_loader::Image;
use biara_tcg::Libtcg;

pub fn tmp() -> PathBuf {
    PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
}

/// A tool related to the clang at `BIARA_CLANG` (`llvm-strip`, `llvm-objcopy`,
/// ...). Homebrew and distro LLVM installs keep those next to it; when the
/// install is a minimal one, take the tool from `PATH` instead (LLVM also
/// ships versioned names, hence the `<name>-<major>` fallback).
pub fn llvm_tool(name: &str) -> PathBuf {
    let dir = Path::new(env!("BIARA_CLANG")).parent().unwrap();
    let sibling = dir.join(name);
    sibling.exists().then_some(sibling).unwrap_or_else(|| {
        let on_path = std::env::var_os("PATH").unwrap_or_default();
        std::env::split_paths(&on_path)
            .map(|d| d.join(name))
            .find(|p| p.exists())
            .unwrap_or_else(|| panic!("{name} is neither next to BIARA_CLANG nor on PATH"))
    })
}

/// `ld.lld` for clang: the Rust toolchain's lld, which picks its flavour
/// from the name it is run as. Also links `lld-link` and `ld64.lld` to the
/// same binary, for the PE and Mach-O tests.
///
/// `rust-lld` looks for `libLLVM.dylib` beside itself, which is not where
/// rustup puts it, and it dispatches on argv[0], which a wrapper changes, so
/// each flavour gets a one-line script naming both.
pub fn linker() -> PathBuf {
    static LD: OnceLock<PathBuf> = OnceLock::new();
    LD.get_or_init(|| {
        let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
        let out = Command::new(rustc)
            .args(["--print", "sysroot"])
            .output()
            .unwrap();
        let sysroot = String::from_utf8(out.stdout).unwrap();
        let sysroot = sysroot.trim();
        let rustlib = PathBuf::from(sysroot).join("lib/rustlib");
        let lld = std::fs::read_dir(&rustlib)
            .unwrap()
            .map(|e| e.unwrap().path().join("bin/rust-lld"))
            .find(|p| p.exists())
            .unwrap_or_else(|| panic!("no rust-lld under {}", rustlib.display()));
        let lib = PathBuf::from(sysroot).join("lib");
        // One per test binary: they run concurrently.
        let dir = tmp().join(concat!("ld-", env!("CARGO_CRATE_NAME")));
        std::fs::create_dir_all(&dir).unwrap();
        for (name, flavour) in [
            ("ld.lld", "gnu"),
            ("lld-link", "link"),
            ("ld64.lld", "darwin"),
        ] {
            let script = dir.join(name);
            let body = format!(
                "#!/bin/sh\nset -e\nexport DYLD_LIBRARY_PATH=\"{}:$DYLD_LIBRARY_PATH\"\nexec \"{}\" -flavor {flavour} \"$@\"\n",
                lib.display(),
                lld.display()
            );
            if std::fs::read_to_string(&script).ok().as_deref() != Some(body.as_str()) {
                // `script` may be a symlink from an earlier run, and writing
                // through one would overwrite the linker it points at.
                let _ = std::fs::remove_file(&script);
                std::fs::write(&script, body).unwrap();
                let mut perm = std::fs::metadata(&script).unwrap().permissions();
                std::os::unix::fs::PermissionsExt::set_mode(&mut perm, 0o755);
                std::fs::set_permissions(&script, perm).unwrap();
            }
        }
        dir.join("ld.lld")
    })
    .clone()
}

/// The whole program, as the CLI recovers it without `--at`.
pub fn recover_program(lib: &mut Libtcg, image: &Image) -> Cfg {
    let opts = ResolveOptions {
        cfg: biara_cfg::Options {
            os: image.os,
            imports: image.imports.clone(),
            discovery: Some(Discovery {
                code: image.code.clone(),
                extents: image.function_extents(),
                data_pointers: image.code_pointers.clone(),
            }),
            function_starts: image.function_starts(),
            ..Default::default()
        },
        ..Default::default()
    };
    let starts: Vec<u64> = image.function_starts().into_iter().collect();
    recover_resolving(
        lib,
        &analysis_machine().unwrap(),
        &|pc| image.code_at(pc),
        &|addr, buf| image.read_const(addr, buf),
        &starts,
        opts,
    )
    .unwrap()
}
