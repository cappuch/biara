#!/usr/bin/env bash
# Download the pinned QEMU source, and the pinned LLVM release only when the
# system has no usable LLVM (scripts/llvm.sh finds one first).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/versions.sh"

fetch() {
    local url=$1 sha=$2 dest=$3
    if [[ -f $dest ]] && echo "$sha  $dest" | sha256_check -c --status; then
        return
    fi
    curl -fL --retry 3 -o "$dest.part" "$url"
    echo "$sha  $dest.part" | sha256_check -c --quiet
    mv "$dest.part" "$dest"
}

mkdir -p "$DEPS/dl"

fetch "$QEMU_URL" "$QEMU_SHA256" "$DEPS/dl/qemu-$QEMU_VERSION.tar.xz"

# LLVM: use what the host has (scripts/llvm.sh probes Homebrew, /usr/local,
# /usr/lib/llvm-N and PATH) and only unpack the pinned release when nothing
# qualifies. A previous download counts as a candidate too, so this is a no-op
# on the second run.
if bash "$ROOT/scripts/llvm.sh" >/dev/null 2>&1; then
    echo "llvm $("$LLVM_PREFIX/bin/llvm-config" --version): $(cat "$LLVM_PREFIX/.source")"
else
    fetch "$LLVM_URL" "$LLVM_SHA256" "$DEPS/dl/$LLVM_ASSET"
    if [[ ! -x $LLVM_RELEASE_PREFIX/bin/llvm-config ]]; then
        rm -rf "$LLVM_RELEASE_PREFIX"
        mkdir -p "$LLVM_RELEASE_PREFIX"
        # The release archive uses a 1 GiB zstd window.
        zstd -dc --long=31 "$DEPS/dl/$LLVM_ASSET" \
            | tar -x -C "$LLVM_RELEASE_PREFIX" --strip-components=1
    fi

    # The macOS release also ships its own libc++ (headers and libraries),
    # libc++abi and libunwind. Its newer headers call into a libc++ macOS does
    # not have; without them the release's clang uses the SDK's, like the
    # system's compiler.
    if [[ $(uname -s) == Darwin ]]; then
        rm -rf "$LLVM_RELEASE_PREFIX"/include/c++ "$LLVM_RELEASE_PREFIX"/lib/libc++* \
            "$LLVM_RELEASE_PREFIX"/lib/libunwind*
    fi
    LLVM_SYS_231_PREFIX= bash "$ROOT/scripts/llvm.sh" >/dev/null
    echo "llvm $("$LLVM_PREFIX/bin/llvm-config" --version): $LLVM_RELEASE_PREFIX"
fi

# LLVM wants zstd and biara-loader liblzma, which macOS lacks. Offer only
# Homebrew's static archives (.cargo/config.toml searches here first), so the
# binaries run without Homebrew. Skip whatever is already provided by the LLVM
# install or the SDK.
if [[ $(uname -s) == Darwin ]]; then
    mkdir -p "$DEPS/macos-static"
    for lib in zstd xz; do
        archive=$(brew --prefix "$lib" 2>/dev/null)/lib/lib$( [[ $lib == xz ]] && echo lzma || echo zstd).a
        [[ -f $archive ]] && ln -sf "$archive" "$DEPS/macos-static/"
    done
fi

echo "qemu $QEMU_VERSION: $DEPS/dl/qemu-$QEMU_VERSION.tar.xz"
echo "llvm $("$LLVM_PREFIX/bin/llvm-config" --version): $LLVM_PREFIX"
