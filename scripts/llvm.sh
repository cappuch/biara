#!/usr/bin/env bash
# Locate a usable LLVM 23 and materialize the two prefixes
# .cargo/config.toml expects:
#
#   .deps/llvm-$LLVM_VERSION        symlinks into the install: include/, lib/,
#                                   llvm-* tools, a linker, and a clang wrapper
#
# LLVM is a build-time dependency only: its clang compiles the QEMU TCG lifters
# from bitcode and cross-compiles the test corpora. Nothing links LLVM as a
# library any more, so no llvm-config wrapper is generated.
#
# Search order, first compatible install wins:
#   1. $LLVM_SYS_231_PREFIX (explicit user choice; the name is historical, it is
#      any llvm-sys-shaped prefix)
#   2. .deps/llvm-release-$LLVM_VERSION (the pinned release, if downloaded)
#   3. .deps/llvm-local (an install placed by hand), then .deps/llvm-$LLVM_VERSION
#      (a previous run)
#   4. Homebrew: brew --prefix llvm@<major>, then llvm
#   5. /usr/local/opt/llvm*, /opt/homebrew/opt/llvm*
#   6. llvm-config in PATH
#
# An install qualifies when `llvm-config --version` reports major $LLVM_MAJOR and
# lib/libLLVM*.a exists. The static-archive requirement is not about our link
# (nothing links LLVM now): the pinned 23.1.2 release ships static archives, and
# requiring them keeps the search deterministic across Homebrew layouts.
#
# Run by fetch-deps.sh. Exits 0 and prints the install it used; exits 1 with a
# diagnostic when none qualifies.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/versions.sh"
LLVM_MAJOR="${LLVM_VERSION%%.*}"
host=$(rustc -vV | sed -n 's/^host: //p')

compatible() {
    local p=$1 v
    [[ -n $p ]] || return 1
    [[ -x $p/bin/llvm-config ]] || return 1
    v=$("$p/bin/llvm-config" --version 2>/dev/null) || return 1
    [[ ${v%%.*} == "$LLVM_MAJOR" ]] || return 1
    compgen -G "$p/lib/libLLVM*.a" >/dev/null || return 1
    echo "$p"
}

find_llvm() {
    local c brew
    if [[ -n ${LLVM_SYS_231_PREFIX:-} ]] && compatible "$LLVM_SYS_231_PREFIX"; then
        return
    fi
    for c in "${LLVM_RELEASE_PREFIX:-}" "$DEPS/llvm-local" "$LLVM_PREFIX" \
        /opt/homebrew/opt/llvm@"$LLVM_MAJOR" \
        /usr/local/opt/llvm@"$LLVM_MAJOR" \
        /opt/homebrew/opt/llvm /usr/local/opt/llvm; do
        compatible "$c" && return
    done
    if command -v brew >/dev/null 2>&1; then
        for c in "llvm@$LLVM_MAJOR" llvm; do
            brew --prefix "$c" >/dev/null 2>&1 || continue
            compatible "$(brew --prefix "$c")" && return
        done
    fi
    compgen -G "/usr/lib/llvm-$LLVM_MAJOR*" >/dev/null && \
        for c in /usr/lib/llvm-"$LLVM_MAJOR"*; do compatible "$c" && return; done
    if command -v llvm-config >/dev/null 2>&1; then
        c=$(dirname "$(dirname "$(command -v llvm-config)")")
        compatible "$c" && return
    fi
    return 1
}

# Some LLVM 23 builds ship libLLVM*.a members as bitcode, and rustc's own LLVM
# (older: 1.98 embeds 22) parses them when it archives the rlib. Those are close
# enough for the pinned release, but an LLVM 23.1.3 bitcode build fails with
# `error: Not an int attribute (Producer: 'LLVM23.1.3' Reader: 'LLVM 22.1.8')`.
# Homebrew and distro packages ship real object files and are unaffected, so
# prefer those; only warn, since whether bitcode links depends on the build.
warns_bitcode_archives() {
    local p=$1 f
    f=$(compgen -G "$p/lib/libLLVMCore.a" | head -1) || return 1
    file -b "$f" 2>/dev/null | grep -qi 'LLVM IR bitcode'
}

llvm_major() {
    "$1/bin/llvm-config" --version 2>/dev/null | cut -d. -f1
}

# The macOS release ships its static archives as LLVM bitcode, which Apple's
# linker cannot read; that install needs its own clang + lld. A distro or
# Homebrew install ships real object files, so the system linker is correct --
# and Apple's clang rejects -fuse-ld=lld outright.
uses_bitcode_archives() {
    local p=$1 f
    f=$(compgen -G "$p/lib/libLLVMCore.a" | head -1) || return 1
    file -b "$f" 2>/dev/null | grep -qi 'LLVM IR bitcode'
}

# Write a generated file without ever following a symlink: the prefix holds
# links into the real install, and `cat > $prefix/bin/clang` would otherwise
# write through one into the user's toolchain. `mv` replaces the link itself.
emit() {
    local dest=$1 content
    content=$(cat)
    printf '%s\n' "$content" > "$dest.$$.tmp"
    chmod "$2" "$dest.$$.tmp"
    mv -f "$dest.$$.tmp" "$dest"
}

materialize() {
    local src=$1
    # A previous run's prefix links into the real install; materialize from
    # that, never from the prefix itself, whose bin/ this function clears.
    if [[ $src == "$LLVM_PREFIX" && -f $LLVM_PREFIX/.source ]]; then
        src=$(<"$LLVM_PREFIX/.source")
    fi

    # The prefix is a directory of symlinks this script owns, never the install
    # itself, so re-running (or a different install) is just re-linking.
    mkdir -p "$LLVM_PREFIX/bin"
    rm -rf "$LLVM_PREFIX/bin"
    mkdir -p "$LLVM_PREFIX/bin"
    ln -sfn "$src/include" "$LLVM_PREFIX/include"
    ln -sfn "$src/lib" "$LLVM_PREFIX/lib"
    # clang is not in that list: below it becomes a wrapper (or a link to the
    # system compiler), and linking it here first would make the wrapper write
    # through the link into the user's toolchain.
    for t in llvm-config llvm-link llvm-strip llvm-objcopy llvm-objdump llvm-ar llvm-nm \
        ld.lld ld64.lld lld opt llc; do
        [[ -e $src/bin/$t ]] && ln -s "$src/bin/$t" "$LLVM_PREFIX/bin/$t"
    done
    # An install without its own lld links ELF/PE through the linker Rust
    # ships. That binary looks for libLLVM beside itself, which is not where
    # rustup keeps it, so it is wrapped with the path it needs.
    if [[ ! -e $src/bin/ld.lld ]]; then
        toolchain=$(rustc --print sysroot 2>/dev/null || true)
        lld=$toolchain/lib/rustlib/${host}/bin/rust-lld
        if [[ -x $lld ]]; then
            emit "$LLVM_PREFIX/bin/ld.lld" 755 <<LLD
#!/bin/sh
# Generated by scripts/llvm.sh: $lld needs libLLVM.dylib from the toolchain's
# lib/, which is not on its own search path.
set -e
DYLD_LIBRARY_PATH="$toolchain/lib\${DYLD_LIBRARY_PATH:+:\$DYLD_LIBRARY_PATH}"
export DYLD_LIBRARY_PATH
# The binary dispatches on argv[0], which stays `rust-lld` through a wrapper,
# so the GNU flavour ELF and PE links want must be named explicitly.
exec "$lld" -flavor gnu "\$@"
LLD
        fi
    fi
    emit "$LLVM_PREFIX/.source" 644 <<<"$src"

    # This path has two jobs, and they need different linkers:
    #
    #   * the workspace linker (.cargo/config.toml), which links host Mach-O
    #     binaries. Apple clang is right for those, and it rejects -fuse-ld=lld.
    #   * BIARA_CLANG (scripts/llvm.sh sets it below via .cargo/config.toml's
    #     CLANG_PATH), which cross-compiles the test corpora for linux/PE with
    #     -fuse-ld=lld. Apple clang needs that spelled --ld-path=<lld>.
    #
    # The invocation says which: a cross target other than Darwin. Apple's
    # clang cannot emit ELF or PE objects at all, so a cross build runs the
    # LLVM install's own clang (found through $0's directory, so the rpath
    # that clang needs still resolves). The one exception is an LLVM whose
    # archives are bitcode (the official release): only its own clang + lld
    # can link those, so run it unchanged.
    if [[ $(uname -s) == Darwin ]] && ! uses_bitcode_archives "$src"; then
        emit "$LLVM_PREFIX/bin/clang" 755 <<EOF
#!/bin/sh
# Generated by scripts/llvm.sh: $src ships object-file archives, so Apple's
# toolchain links them. Rewrite the LLVM-only link flags for the target.
set -e
src_dir="$src"
ldl="$LLVM_PREFIX/bin/ld.lld"
cross=0
for a in "\$@"; do
  case "\$a" in
    --target=*) case "\$a" in *darwin*|*apple*) ;; *) cross=1 ;; esac ;;
  esac
done
args=()
for a in "\$@"; do
  case "\$a" in
    -fuse-ld=lld)
      # Cross-linking needs lld and its ELF/PE driver options; host links use
      # the SDK linker and must not see either.
      [ "\$cross" = 1 ] && args+=("--ld-path=\$ldl")
      continue ;;
    -Wl,--hash-style=*|-Wl,--eh-frame-hdr*) [ "\$cross" = 1 ] && args+=("\$a"); continue ;;
    -Wl,-cache_path_lto,*) continue ;;
  esac
  args+=("\$a")
done
if [ "\$cross" = 1 ]; then
  # Apple's clang cannot emit ELF or PE objects; the LLVM install's own clang
  # can, and it finds its dylibs beside itself.
  exec "\$src_dir/bin/clang" "\${args[@]}"
fi
exec /usr/bin/clang "\${args[@]}"
EOF
        ln -sf clang "$LLVM_PREFIX/bin/clang++"
    elif [[ -e $src/bin/clang ]]; then
        ln -s "$src/bin/clang" "$LLVM_PREFIX/bin/clang"
        ln -sf clang "$LLVM_PREFIX/bin/clang++"
    else
        # Dev packages only (Debian's llvm-N-dev ships no clang); any system
        # clang cross-compiles these corpora when driven by lld.
        for c in /usr/bin/clang /usr/bin/clang-$(llvm_major "$src") clang; do
            command -v "$c" >/dev/null 2>&1 || continue
            ln -s "$(command -v "$c")" "$LLVM_PREFIX/bin/clang"
            break
        done
        [[ -e $LLVM_PREFIX/bin/clang ]] && ln -sf clang "$LLVM_PREFIX/bin/clang++"
    fi

    # Cross-compiling (the test corpora, the release smoke test) needs a linker
    # that understands ELF/PE: lld. The official release and distro `lld`
    # packages ship one; Homebrew's llvm does not, so fall back to the Rust
    # toolchain's rust-lld, which picks its flavour from argv[0].
    if [[ ! -e $LLVM_PREFIX/bin/ld.lld ]]; then
        local lld
        lld=$(command -v ld.lld 2>/dev/null || true)
        if [[ -z $lld ]]; then
            local rustc_bin rustlib
            rustc_bin=$(command -v rustc 2>/dev/null || true)
            if [[ -n $rustc_bin ]]; then
                rustlib="$("$rustc_bin" --print sysroot)/lib/rustlib"
                lld=$(compgen -G "$rustlib/*/bin/rust-lld" | head -1) || true
            fi
        fi
        # Never link a path onto itself: `ln -sf a a` writes through `a` and
        # truncates the linker it points at. The search can hand back this very
        # prefix's ld.lld, so compare before replacing.
        if [[ -n ${lld:-} && -x $lld ]]; then
            for name in ld.lld ld64.lld lld-link; do
                dest=$LLVM_PREFIX/bin/$name
                [[ -e $dest && $lld -ef $dest ]] && continue
                ln -sf "$lld" "$dest"
            done
        else
            echo "warning: no lld and no rust-lld; cross-compiled tests will fail." >&2
            echo "         Install lld (brew install lld, apt install lld)." >&2
        fi
    fi

}

if ! src=$(find_llvm); then
    echo "no LLVM $LLVM_MAJOR with static archives found; run scripts/fetch-deps.sh" >&2
    exit 1
fi
materialize "$src"
if warns_bitcode_archives "$src"; then
    echo "warning: $src ships bitcode objects; rustc's older LLVM cannot read them." >&2
    echo "         Install LLVM from Homebrew or your distribution for a working build." >&2
fi
echo "$src"
