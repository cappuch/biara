# Pinned toolchain versions. Sourced by the other scripts.
# Bumping a version: update the sha256s and rebase patches/qemu-<ver>/.
# LLVM is a build-time tool (clang, llvm-link) only; bumping it also means
# updating the paths .cargo/config.toml names.
#
# Hosts: Linux x86-64 and macOS arm64 (the hosts LLVM publishes both
# releases for).

QEMU_VERSION=11.1.2
QEMU_URL="https://download.qemu.org/qemu-${QEMU_VERSION}.tar.xz"
QEMU_SHA256=731b5681e4bb18be313231579b8efd0296c5b015fa36dc533874b639ba838016

LLVM_VERSION=23.1.2
case "$(uname -s)-$(uname -m)" in
Linux-x86_64)
    LLVM_ASSET="LLVM-${LLVM_VERSION}-Linux-X64.tar.zst"
    LLVM_SHA256=6382de1c1a210ce5a5cc49d18bc8444d137742e7cbf9b19f4ae602bb1ab52534
    # Shared-library suffix of the lifters.
    DSO=so
    ;;
Darwin-arm64)
    LLVM_ASSET="LLVM-${LLVM_VERSION}-macOS-ARM64.tar.zst"
    LLVM_SHA256=3da0e91b5dfe3a5ec795ad2be79b3f5e6f28c8b23edcd3847fad7742b25e0507
    DSO=dylib
    ;;
*)
    echo "unsupported host $(uname -s) $(uname -m): Linux x86-64 and macOS arm64 only" >&2
    exit 1
    ;;
esac
LLVM_URL="https://github.com/llvm/llvm-project/releases/download/llvmorg-${LLVM_VERSION}/${LLVM_ASSET}"

# GNU `sha256sum`; macOS's own sha256sum lacks --status and --quiet, so
# `shasum` there.
sha256_check() {
    if [[ $(uname -s) == Darwin ]]; then shasum -a 256 "$@"; else sha256sum "$@"; fi
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPS="$ROOT/.deps"
# The pinned release, when it has to be downloaded, unpacks here; LLVM_PREFIX
# is a prefix scripts/llvm.sh owns (symlinks into whichever install it found).
LLVM_RELEASE_PREFIX="$DEPS/llvm-release-$LLVM_VERSION"
LLVM_PREFIX="$DEPS/llvm-$LLVM_VERSION"
QEMU_SRC="$DEPS/build/qemu-src-$QEMU_VERSION"
QEMU_BUILD="$DEPS/build/qemu-$QEMU_VERSION"
OUT="$DEPS/out"
