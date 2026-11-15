//! Concrete candidates survive callback/return summaries; writable pointer
//! contents never become an exhaustive static target set.
#![cfg(target_os = "linux")]
mod common;
use biara_cfg::ExitKind;
use biara_loader::Image;
use biara_tcg::Libtcg;
use common::{linker, recover_program, tmp};
use std::{collections::BTreeSet, fs, process::Command};

#[test]
fn callback_returns_and_readonly_tables() {
    for arch in ["x86_64", "aarch64"] {
        for opt in [0, 2] {
            let dir = tmp().join(format!("indirect-{arch}-O{opt}"));
            fs::create_dir_all(&dir).unwrap();
            let factory = dir.join("factory.c");
            let callers = dir.join("callers.c");
            let binary = dir.join("program");
            fs::write(
                &factory,
                r#"
typedef long (*Callback)(long);
__attribute__((noinline,used)) long increment(long x) { return x+1; }
__attribute__((noinline,used)) long decrement(long x) { return x-1; }
static Callback const table[] = {increment,decrement};
Callback volatile writable = increment;
__attribute__((noinline,used)) Callback factory(void) { return increment; }
__attribute__((noinline,used)) Callback mutable_factory(void) { return writable; }
__attribute__((noinline,used)) long table_call(unsigned i,long x) {
    Callback cb=table[i&1];
    __asm__ volatile("" : "+r"(cb));
    return cb(x)+3;
}
__attribute__((noinline,used)) long projected(unsigned long n,long x) {
    unsigned i=(unsigned)n;
    if(i<2 && (n >= (1UL<<32) || i==0)) {
        Callback cb=table[i];
        __asm__ volatile("" : "+r"(cb));
        return cb(x)+3;
    }
    return 0;
}
__attribute__((noinline,used)) long growing(unsigned i,long x) {
    static void *const labels[]={&&a,&&b};
    Callback cb=increment;
    if(i) goto *labels[i&1];
    goto join;
a: cb=increment; goto join;
b: cb=decrement;
join: return cb(x)+3;
}
"#,
            )
            .unwrap();
            fs::write(
                &callers,
                r#"
typedef long (*Callback)(long);
extern Callback factory(void),mutable_factory(void);
extern long table_call(unsigned,long);
extern long projected(unsigned long,long),growing(unsigned,long);
__attribute__((noinline,used)) long apply(Callback cb,long x) { return cb(x)+2; }
__attribute__((noinline,used)) long apply_mutable(Callback cb,long x) { return cb(x)+4; }
__attribute__((noreturn,used)) void _start(void) {
    long value=apply(factory(),5)+table_call(1,8)+apply_mutable(mutable_factory(),3)
        +projected((1UL<<32)+1,6)+growing(1,5);
#if defined(__x86_64__)
    __asm__ volatile("syscall"::"a"(60L),"D"(value==41 ? 0L : 1L):"rcx","r11","memory");
#else
    register long number __asm__("x8")=93;
    register long result __asm__("x0")=value==41 ? 0 : 1;
    __asm__ volatile("svc #0"::"r"(number),"r"(result):"memory");
#endif
    __builtin_unreachable();
}
"#,
            )
            .unwrap();
            let output = Command::new(env!("BIARA_CLANG"))
                .args([
                    format!("--target={arch}-linux-gnu"),
                    format!("-O{opt}"),
                    format!("--ld-path={}", linker().display()),
                ])
                .args([
                    "-ffreestanding",
                    "-nostdlib",
                    "-fno-pie",
                    "-static",
                    "-funwind-tables",
                ])
                .arg(&factory)
                .arg(&callers)
                .arg("-o")
                .arg(&binary)
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            if arch == "x86_64" {
                assert!(
                    Command::new(&binary).status().unwrap().success(),
                    "native callback ground truth"
                );
            }
            let image = Image::load(&binary).unwrap();
            let address = |name: &str| image.symbol(name).unwrap().addr;
            let lib = Libtcg::load(&Libtcg::find(arch).unwrap(), None).unwrap();
            let cfg = recover_program(&mut lib.lock(), &image);
            let indirect_site = |name: &str| {
                *cfg.functions[&address(name)]
                    .blocks
                    .iter()
                    .find(|&&pc| {
                        cfg.blocks[&pc]
                            .exits
                            .iter()
                            .any(|e| matches!(e.kind, ExitKind::IndirectCall { .. }))
                    })
                    .unwrap_or_else(|| panic!("{arch} O{opt} {name}: no indirect call"))
            };
            let apply = indirect_site("apply");
            let candidates = &cfg.indirect_calls[&apply];
            assert!(
                candidates.targets.contains(&address("increment")),
                "{arch} O{opt}: callback return summary"
            );
            assert!(
                !candidates.complete,
                "known callers are not proof of all runtime callers"
            );
            assert!(
                cfg.functions[&address("apply")]
                    .calls
                    .contains(&address("increment"))
            );
            let table = indirect_site("table_call");
            let targets = &cfg.indirect_calls[&table];
            assert_eq!(
                targets.targets,
                BTreeSet::from([address("increment"), address("decrement")])
            );
            assert!(
                targets.complete,
                "readonly bounded table must be exhaustive"
            );
            let projected = &cfg.indirect_calls[&indirect_site("projected")];
            assert_eq!(
                projected.targets,
                BTreeSet::from([address("increment"), address("decrement")]),
                "{arch} O{opt}: upper-bit conditions cannot discard a feasible low-bit index"
            );
            assert!(projected.complete);
            let grown = &cfg.indirect_calls[&indirect_site("growing")];
            assert_eq!(
                grown.targets,
                BTreeSet::from([address("increment"), address("decrement")]),
                "{arch} O{opt}: recovered predecessor paths must invalidate target proofs"
            );
            let mutable = indirect_site("apply_mutable");
            assert!(!cfg.indirect_calls.get(&mutable).is_some_and(|t| t.complete));
            assert!(
                !cfg.indirect_calls
                    .get(&mutable)
                    .is_some_and(|t| t.targets.contains(&address("increment"))),
                "writable initializer is not immutable evidence"
            );
        }
    }
}

/// A switch without a bounds check (unreachable default) has an index range
/// wider than its table (`& 7` for five cases, a whole byte for `state_b`), so
/// enumeration reads past the table into the next function's. Those entries
/// are not this switch's cases: they must not join its function.
#[test]
fn unbounded_tables_stay_in_their_function() {
    let dir = tmp().join("indirect-unbounded");
    fs::create_dir_all(&dir).unwrap();
    let source = dir.join("states.c");
    let binary = dir.join("program");
    fs::write(
        &source,
        r#"
__attribute__((noinline,used)) long state_a(const unsigned char *s, long x) {
    switch (*s & 7) {
    case 0: x = x * 3 + 1; break;
    case 1: x = x ^ 0x55; break;
    case 2: x = x - 17; break;
    case 3: x = x << 3; break;
    case 4: x = x / 7; break;
    default: __builtin_unreachable();
    }
    return x + s[1];
}
__attribute__((noinline,used)) long state_b(const unsigned char *s, long x) {
    switch (*s) {
    case 0: x = x * 5 + 2; break;
    case 1: x = x ^ 0x99; break;
    case 2: x = x - 23; break;
    case 3: x = x << 5; break;
    case 4: x = x / 9; break;
    default: __builtin_unreachable();
    }
    return x + s[2];
}
__attribute__((noinline,used)) long twice(long x) { return 2 * x; }
__attribute__((noinline,used)) long negate(long x) { return -x; }
static long (*const handlers[])(long) = {twice, negate};
__attribute__((noinline,used)) long dispatch(const unsigned char *s, long x) {
    long (*h)(long) = handlers[*s & 1];
    __asm__ volatile("" : "+r"(h));
    return h(x);
}
__attribute__((noreturn,used)) void _start(void) {
    static const unsigned char a[] = {2, 1}, b[] = {3, 0, 4}, c[] = {1};
    long value = state_a(a, 20) + state_b(b, 1) + dispatch(c, 7);
    __asm__ volatile("syscall"::"a"(60L),"D"(value == 33 ? 0L : 1L):"rcx","r11","memory");
    __builtin_unreachable();
}
"#,
    )
    .unwrap();
    let output = Command::new(env!("BIARA_CLANG"))
        .args([
            "--target=x86_64-linux-gnu".to_owned(),
            "-O2".to_owned(),
            format!("--ld-path={}", linker().display()),
        ])
        .args([
            "-ffreestanding",
            "-nostdlib",
            "-fno-pie",
            "-static",
            "-funwind-tables",
        ])
        .arg(&source)
        .arg("-o")
        .arg(&binary)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(Command::new(&binary).status().unwrap().success());
    let image = Image::load(&binary).unwrap();
    let body = |name: &str| {
        let symbol = image.symbol(name).unwrap();
        symbol.addr..symbol.addr + symbol.size
    };
    let (a, b) = (body("state_a"), body("state_b"));
    let lib = Libtcg::load(&Libtcg::find("x86_64").unwrap(), None).unwrap();
    let cfg = recover_program(&mut lib.lock(), &image);
    let jump_site = |entry: u64| {
        *cfg.functions[&entry]
            .blocks
            .iter()
            .find(|&&pc| {
                cfg.blocks[&pc]
                    .exits
                    .iter()
                    .any(|e| matches!(e.kind, ExitKind::IndirectJump { .. }))
            })
            .expect("an indirect jump")
    };
    let function = &cfg.functions[&a.start];
    let site = jump_site(a.start);
    let targets = &cfg.jump_tables[&site];
    assert_eq!(
        targets.iter().filter(|t| a.contains(t)).count(),
        5,
        "the five cases: {targets:x?}"
    );
    // Past the table: other functions' entries at most (tail calls, which
    // keep bodies apart), never a place inside another function.
    let starts = image.function_starts();
    assert!(
        targets.iter().all(|t| a.contains(t) || starts.contains(t)),
        "{targets:x?}"
    );
    assert!(
        function.blocks.iter().all(|pc| !b.contains(pc)),
        "state_b's code joined state_a"
    );
    // Read past its end, the table cannot be shown to have only these.
    assert!(cfg.incomplete_jump_tables.contains(&site));
    // Other functions' entries are what a tail call through a table reaches.
    let tail = jump_site(body("dispatch").start);
    assert_eq!(
        cfg.jump_tables[&tail],
        BTreeSet::from([body("twice").start, body("negate").start])
    );
}
