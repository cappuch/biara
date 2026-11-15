//! Readable decompilation: what the C for people leaves out (or folds) compared
//! with the lifted program, and what it must keep for the C to stay true.
#![cfg(all(target_os = "linux", target_arch = "x86_64"))]
use serde_json::Value;
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    sync::LazyLock,
};

/// A program with resolved and unresolved indirect calls, x86 flags read
/// across blocks, and a variadic call; built once, as tests run in parallel.
fn fixture() -> PathBuf {
    static BUILT: LazyLock<PathBuf> = LazyLock::new(build);
    BUILT.clone()
}

fn build() -> PathBuf {
    let dir =
        PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("readable-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let source = dir.join("program.c");
    let binary = dir.join("program");
    fs::write(
        &source,
        r#"
extern int printf(const char *, ...);
#define KEEP __attribute__((noinline, used))
KEEP long leaf(long v) { __asm__ volatile("" : "+r"(v)); return v * 3 + 1; }
KEEP long twice(long v) { __asm__ volatile("" : "+r"(v)); return v * 2; }
KEEP long thrice(long v) { __asm__ volatile("" : "+r"(v)); return v * 3; }
/* Defined in assembly, so the compiler cannot devirtualize the calls. */
extern long (*const ops[4])(long);
__asm__(".section .rodata\n.balign 8\n.globl ops\nops: .quad leaf, twice, thrice, leaf\n.text\n");
KEEP long by_table(unsigned long index, long v) { return ops[index & 3](v) + 5; }
/* One constant slot, like a GOT entry the loader fills before the code runs. */
extern long (*const slot)(long);
__asm__(".section .rodata\n.balign 8\n.globl slot\nslot: .quad thrice\n.text\n");
KEEP void sink(long *p) { p[0]++; }
/* The frame escapes to `sink`, so stores to it stay visible. */
KEEP long via_slot(long v) { long kept[2] = {v, v}; sink(kept); return slot(v) + kept[0]; }
KEEP long open_callback(long (*callback)(long), long v) { return callback(v) + 1; }
KEEP int show(void) { return printf("%ld\n", -1L); }
/* `je` tests the flags of the `test` before `jns`, in another block. */
long sign(long);
__asm__(".text\n.globl sign\n.type sign,@function\nsign:\n"
        "  test %rdi, %rdi\n  jns 1f\n  mov $-1, %rax\n  ret\n"
        "1: je 2f\n  mov $1, %eax\n  ret\n"
        "2: xor %eax, %eax\n  ret\n.size sign, .-sign\n");
/* `jns`: a branch on `x >= 0`. */
long nonneg(long);
__asm__(".text\n.globl nonneg\n.type nonneg,@function\nnonneg:\n"
        "  test %rdi, %rdi\n  jns 1f\n  mov $9, %eax\n  ret\n"
        "1: push %rbx\n  call leaf\n  add $1, %rax\n  pop %rbx\n  ret\n.size nonneg, .-nonneg\n");
/* `pushfq` calls a helper that reads the flags through env. */
long eflags_twice(long *p, long v);
__asm__(".text\n.globl eflags_twice\n.type eflags_twice,@function\neflags_twice:\n"
        "  pushfq\n  pop %rax\n  mov %rax, (%rdi)\n  add $1, %rsi\n"
        "  pushfq\n  pop %rdx\n  add %rdx, %rax\n  add %rsi, %rax\n  ret\n"
        ".size eflags_twice, .-eflags_twice\n");
/* `rdtsc` calls a helper whose effect on env is unknown (it calls code the
   bitcode does not define); the flags of `cmp` are live across it. */
long tsc_twice(long a, long b, long *p);
__asm__(".text\n.globl tsc_twice\n.type tsc_twice,@function\ntsc_twice:\n"
        "  mov %rdx, %rcx\n  cmp %rsi, %rdi\n  setl %r8b\n"
        "  rdtsc\n  mov %rax, (%rcx)\n"
        "  rdtsc\n  movzbl %r8b, %edx\n  add %rdx, %rax\n  ret\n.size tsc_twice, .-tsc_twice\n");
/* Unknown helpers on both arms of a branch, then another after the join. */
long tsc_join(long a);
__asm__(".text\n.globl tsc_join\n.type tsc_join,@function\ntsc_join:\n"
        "  test %rdi, %rdi\n  je 1f\n  rdtsc\n  jmp 2f\n1: rdtsc\n  rdtsc\n"
        "2: rdtsc\n  ret\n.size tsc_join, .-tsc_join\n");
int main(int argc, char **argv) {
  (void)argv;
  long argc_copy = argc;
  return by_table(argc, 2) + via_slot(argc) + nonneg(argc) + open_callback(leaf, argc) + sign(argc) + show() + eflags_twice(&argc_copy, argc) + tsc_twice(argc, 1, &argc_copy) + tsc_join(argc) == 0;
}
"#,
    )
    .unwrap();
    let built = Command::new(env!("BIARA_CLANG"))
        .args([
            "-O2",
            "-g",
            "-fno-builtin",
            "-fno-pie",
            "-no-pie",
            "-fuse-ld=bfd",
        ])
        .arg(&source)
        .arg("-o")
        .arg(&binary)
        .output()
        .unwrap();
    assert!(
        built.status.success(),
        "{}",
        String::from_utf8_lossy(&built.stderr)
    );
    binary
}

fn lift(binary: &Path, args: &[&str]) -> String {
    let output = Command::new(env!("CARGO_BIN_EXE_biara"))
        .arg("lift")
        .arg(binary)
        .args(args)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap()
}

/// The decompiled body of `function`.
fn decompiled(binary: &Path, function: &str) -> String {
    let report: Value = serde_json::from_str(&lift(
        binary,
        &[function, "--emit", "decomp", "--format", "json"],
    ))
    .unwrap();
    let record = report["functions"]
        .as_array()
        .unwrap()
        .iter()
        .find(|f| f["function"]["name"] == function)
        .unwrap_or_else(|| panic!("no package for {function}"));
    record["chunks"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c["text"].as_str().unwrap())
        .collect()
}

#[test]
fn resolved_indirect_calls_are_direct_calls_that_return_their_results() {
    let binary = fixture();
    let body = decompiled(&binary, "by_table");
    // Every table entry is known: no call through the loaded pointer, and no
    // arm for an index the table cannot hold.
    assert!(
        !body.contains("biara_callback") && !body.contains("(*)("),
        "{body}"
    );
    assert!(!body.contains("__builtin_unreachable"), "{body}");
    // `thrice` is only ever called through the table; the read of the result
    // after that call is still what makes it return a value.
    for callee in ["leaf", "twice", "thrice"] {
        assert!(
            body.contains(&format!(" = {callee}(v);")),
            "{callee}:\n{body}"
        );
    }
    // The three results meet in one variable: no copies into it.
    assert!(!body.contains("_result;"), "{body}");
}

#[test]
fn a_call_through_a_constant_slot_is_a_direct_call() {
    let binary = fixture();
    let body = decompiled(&binary, "via_slot");
    assert!(body.contains("thrice(v)"), "{body}");
    assert!(
        !body.contains("biara_callback") && !body.contains("(*)("),
        "{body}"
    );
    // The return address the `call` pushes is not program data.
    let constant_store = body.lines().any(|l| {
        let l = l.trim();
        l.starts_with("frame.") && l.contains(" = 0x")
    });
    assert!(!constant_store, "{body}");
}

#[test]
fn unknown_callbacks_are_one_call_through_the_pointer_unless_the_c_runs() {
    let binary = fixture();
    let body = decompiled(&binary, "open_callback");
    assert!(
        !body.contains("biara_call_at") && !body.contains("biara_cpu"),
        "{body}"
    );
    assert_eq!(
        body.matches("(v, ").count() + body.matches("(v)").count(),
        1,
        "{body}"
    );
    // Executable C still tries the lifted functions first.
    let c = lift(&binary, &["open_callback", "--emit", "c"]);
    assert!(c.contains("= biara_call_at(callback);"), "{c}");
}

#[test]
fn condition_code_helpers_fold_where_the_operation_is_constant() {
    let binary = fixture();
    let body = decompiled(&binary, "sign");
    assert!(!body.contains("helper_cc_compute"), "{body}");
    assert!(
        body.contains("(int64_t)arg0 > 0") && body.contains("(int64_t)arg0 < 0"),
        "{body}"
    );
}

#[test]
fn sign_tests_compare_with_zero() {
    // LLVM's canonical `x > -1` is the `x >= 0` that `jns` tests.
    let body = decompiled(&fixture(), "nonneg");
    assert!(body.contains("if ((int64_t)arg0 >= 0)"), "{body}");
}

#[test]
fn literals_carry_their_type_only_where_c_would_lose_it() {
    let binary = fixture();
    let body = decompiled(&binary, "leaf");
    assert!(body.contains("(v * 3) + 1"), "{body}");
    // A variadic argument has no parameter to convert it: `-1` alone would
    // pass a 32-bit int where the callee reads a 64-bit long.
    let body = decompiled(&binary, "show");
    assert!(body.contains(r#"printf("%ld\n", -1ull)"#), "{body}");
}

#[test]
fn helpers_see_only_the_cpu_state_they_use() {
    let body = decompiled(&fixture(), "eflags_twice");
    // `read_eflags` reads the lazy flags and nothing else: no segment bases,
    // bound registers or pc stored for it, and nothing reloaded after it.
    assert_eq!(
        body.matches("helper_read_eflags(BIARA_ENV)").count(),
        2,
        "{body}"
    );
    for state in ["_base", "bnd", "biara_cpu.rip", "= biara_cpu."] {
        assert!(!body.contains(state), "{state}:\n{body}");
    }
    assert!(body.contains("biara_cpu.cc_dst = "), "{body}");
}

#[test]
fn unknown_helpers_sync_only_state_the_function_uses() {
    let body = decompiled(&fixture(), "tsc_twice");
    assert_eq!(body.matches("helper_rdtsc(BIARA_ENV)").count(), 2, "{body}");
    // The CPU state is the only copy of what the function never names.
    assert!(!body.contains("_base") && !body.contains("bnd"), "{body}");
    // The flags go to the CPU state once: a store through a guest pointer
    // between the helpers cannot change them.
    assert_eq!(body.matches("biara_cpu.cc_dst = ").count(), 1, "{body}");
    // Where paths join, what each path reloaded is not the CPU state to LLVM.
    let body = decompiled(&fixture(), "tsc_join");
    assert!(!body.contains("_base") && !body.contains("bnd"), "{body}");
}
