# Remaining test failures in biara

State of the tree when this was written: `HEAD = db46187`, working tree clean,
`cargo build --workspace` succeeds with 0 errors.

Start with §1 if you only read one section. §2 is the one worth fixing first.
§3 and §4 are the other two, independent.

**Status:** a fix for the `get_operand` half of the compat layer is in the
working tree and is being gated on aarch64 as this was written. It replaces
`get_operand`'s own instruction search with a call to `resolve_inst`, which is
the version that checks the stated owner actually lists the id. That fix alone
made `tail` pass where two earlier attempts had broken it.

---

## 0. How to run the tests (read this before running anything)

The aarch64 suite is the regression gate. It takes 5-25 minutes depending on
load. **Never run two `cargo test` invocations at once** — they contend and
produce meaningless counts. This repeatedly invalidated results while the
remaining failures were being investigated.

```sh
# The gate. 18 tests. Everything below must keep these passing.
cargo test -p biara-lift --test aarch64 -- --test-threads=4

# Fast, focused suites (each 5-60 s)
cargo test -p biara-lift --test audit          # 3 tests, all failing
cargo test -p biara-lift --test discovery      # 4 tests, 1 failing
cargo test -p biara-cli --test engine          # 9 tests, 1 failing

# A single aarch64 test
cargo test -p biara-lift --test aarch64 -- tail --test-threads=1
```

Environment: `.cargo/config.toml` sets `BIARA_CLANG` and `BIARA_LIBTCG_DIR`
relative to the workspace root. Running a test binary directly out of
`target/debug/deps/` bypasses that config and will fail to find the TCG
lifters — always go through `cargo test`.

Useful debugging hook (intentional, keep it): set
`BIARA_DUMP_ANALYSIS_IR=/tmp/x.txt` and run the audit test. `calls.rs` writes
the analysis module as LLVM-syntax text. **This is the module `CallSites`
actually analyses.** Do not confuse it with `biara lift --emit llvm` output:
that is built with `abis: None` and has a structurally different shape
(register homes are formal parameters, not allocas). Reasoning about analysis
behaviour from `--emit llvm` output has misled every attempt so far.

---

## 1. Current failure inventory

| Suite | Result | Failing |
|---|---|---|
| aarch64 | 18 / 18 | — (gate, currently green) |
| pe | 62 / 62 | — |
| gep_operands | 30 / 30 | — |
| ir, pipeline, pipeline_real | 21 / 21 | — |
| layouts, layouts_core, rewrites | 14 / 14 | — |
| everything else | 62 / 62 | — |
| **audit** | **0 / 3** | `elf`, `pe`, `macho` — all same defect |
| **discovery** | **3 / 4** | `x86_64` (`aarch64` passes) |
| **engine** | **8 / 9** | `persistent_resolution_keeps_shared_memory_and_named_register_results` |

---

## 2. audit: `args[1]` (the buffer pointer) is `Arg::Unknown`

**Correction.** An earlier revision of this file claimed the analysis modules
were malformed — that functions referenced value ids from other functions'
arenas. That was wrong. It came from reading a stale `/tmp` dump, and from
confusing two different modules. Both are explained below; do not repeat that
mistake.

### The two modules, and why they are easy to confuse

Two separate `lift_with` calls produce modules that both reach
`CallSites::collect_unindexed`:

1. `crates/biara-lift/src/calls.rs` — module name `"calls"`, built with
   `abis: Some(...)` and `only: batch` (31 entries for the audit corpus).
   `CallSites::add` drives this one. **This is the module `audit.rs` asserts
   on.** Every one of its 31 functions gets a body (`sub_1a10` has 6 blocks).
2. `crates/biara-lift/src/tables.rs` — module name `"targets"`, built with
   `abis: None` and `only: pending`, then `calls.collect(&module)` on it.
   For the audit corpus `pending` holds very few entries, so this module has
   ~32 functions of which only 3 have bodies.

Both write `BIARA_DUMP_ANALYSIS_IR` to the same path, so the file you get is
whichever ran last — and the `"targets"` one is nearly empty. Reading it and
concluding the lifter was dropping bodies cost a long detour.

Verified: `lift_with` returns 31 functions **all with bodies** for the `"calls"`
module, and `memory::simplify` does not change that (measured: 33 with blocks
before, 33 after). The printer, the verifier and `dce` are all innocent.

### Actual state of the `audit` failure

With the `operand` fix now in the tree, `read(0, buf, 64)` resolves as:

- `args[0]` → `Const(0)`  ✅ (was `Unknown`)
- `args[1]` → `Unknown`   ❌ (needs `Arg::Stack(o)` with `o <= -16`)
- `args[2]` → `Const(64)` ✅ (was `Unknown`)

So `audit.rs:87` is the only remaining assertion.

`args[1]` is the buffer pointer. In the `"calls"` module for `sub_1a10`:

```llvm
%v31 = load i64, ptr %v173        ; entry rsp
%v52 = add i64 %v31, -8
%v66 = add i64 %v52, -16          ; = rsp - 24, the buffer
%v113 = store i64 %v66, ptr %v37  ; %v37 = gep %v0, 48 = rsi home
%v118 = call void @sub_2180(%v0)  ; the read() call
```

`reaching_store(call, 48)` does find `%v113` and does hand back `%v66`
(confirmed by instrumentation: correct id, opcode `Add`). The failure is in
folding `%v66` to `Sym::Reg(rsp, -24)` / `Arg::Stack(-24)`. Instrumentation
showed `fx.symbolic(%v66)` returning `None` while `symbolic` on the *inner*
`add %v31, -8` succeeded — so the chain breaks at one link. Next step: run
`symbolic_at` on `%v66` specifically and print which sub-step returns `None`.
Likely suspects: `entry_loads` lookup for `%v31`, or the `constant(1)` helper
for the `-16` literal.

### Why the previous attempt was reverted (still relevant)

`db46187` reverted `0e57599`, which had widened the `listed` closure inside
`InstructionValue::operand` to accept constants. That change broke aarch64
from 18/18 to 1/17. Bisect: `tail` passes at `a725eb6`, fails at `0e57599`;
reverting `is_const` alone still failed; reverting `listed` as well made it
pass. Narrowing to `Value::Const` did not help either.

**The fix that works instead**, and is in the tree now: `operand` uses
`self.func` whenever that function's arena contains `id`
(`(id.0 as usize) < f.values.len()`), leaving `listed` untouched. This is
sound because `operand` is only ever called with ids from
`biara_ir::operands(inst)`, which belong to the instruction's own function.

---

## 2b. Symptom: instruction handles carry the wrong owning function

Everything below describes a real but secondary defect. Fix it only after §2's
malformation is resolved, because a wrong owner is what *makes* a leaked id
resolve to something plausible instead of failing loudly.

### Symptom

`crates/biara-lift/tests/audit.rs` asserts, for the call `read(0, buf, 64)`:

```rust
let read = call("bad_read", "read");
assert_eq!(read.args[0], Arg::Const(0), "{at}");        // line 86 — FAILS: Unknown
assert!(matches!(read.args[1], Arg::Stack(o) if o <= -16));  // line 87
assert_eq!(read.args[2], Arg::Const(64), "{at}");       // line 91
```

All three arguments come back `Arg::Unknown`.

### Mechanism, established by instrumentation

For the function containing `read(0, buf, 64)`, in the analysis IR:

```llvm
%v66 = add i64 %v52, -16        ; the buf pointer: (rsp-8) - 16
%v115 = store i64 0, ptr %v41   ; rdi home <- literal 0
```

`%v66`'s operand 1 must be the literal `-16`. What actually arrives:

- The operand handle is an `IntValue` with `src = IntSource::Value(ValueId(184))`
  and `func = Some(1)`.
- In function index 1, `values[184]` is **`Value::Inst`** — not a constant.
- In function index 21, `values[184]` **is** `Value::Const`.
- So the operand resolves to an instruction, `is_const()` is false,
  `get_zero_extended_constant()` returns `None`, and the whole pointer chain
  from the stack pointer fails to fold. The argument comes back `Unknown`.

Consequence chain: `symbolic_at` cannot fold `add(load(rsp), -16)` to
`Sym::Reg(sp, -24)`; `argument_at` cannot produce `Arg::Stack(-24)`; and the
same failure swallows the constant `0` and `64` because those also arrive as
arena constants whose owner is wrong.

### Why the obvious fix does not work

`InstructionValue::operand()` in `crates/biara-ir-compat/src/values.rs`
computes an `owner` and applies it to each operand with `.with_owner(owner)`.
The `owner` it picks is derived from `self.func` — the *instruction's* function.
Value ids are per function, so for an operand that is a constant, the
instruction's function is not necessarily where that constant lives.

`operand()` has a `listed` closure used to decide the owner:

```rust
let listed = |f: &biara_ir::Function| {
    f.defined_in.get(id.0 as usize).copied().flatten()
        .is_some_and(|b| f.blocks.get(b.0 as usize)
            .is_some_and(|b| b.insts.contains(&id)))
};
```

Constants have no `defined_in` and are never in `insts`, so `listed` is false
for every constant. The code then falls through to a last-resort branch that
picks *any* function whose arena is long enough:

```rust
.or_else(|| m.module.functions.iter().position(|f| (id.0 as usize) < f.values.len()))
```

That is the "arena long enough" fallback and it is exactly what returns the
wrong function. It answers with function 1 for id 184 because function 1's
arena has ≥185 entries, even though function 21 is the one holding the
constant.

### The trap — read this before you touch `listed`

Widening `listed` to accept constants was tried and **broke aarch64 from
18/18 to 1/17**. Both of these variants fail identically:

```rust
Some(biara_ir::Value::Inst(_)) => /* existing block-membership test */,
Some(_) => true,                            // accepts any value kind
```

```rust
Some(biara_ir::Value::Inst(_)) => /* existing block-membership test */,
Some(biara_ir::Value::Const(_)) => true,    // accepts only constants
```

Narrowing to `Value::Const` did **not** help. That is the important datum: the
predicate is not the problem. By the time `listed` runs, `self.func` is already
wrong and is being *trusted* in the `by_owner` branch:

```rust
let by_owner = self.func
    .and_then(|i| m.module.functions.get(i))
    .filter(|f| listed(f))
    .map(|_| self.func.unwrap_or(0));
```

Changing the predicate only changes which wrong answer you get.

### What to actually find

Why does an instruction handle's `func` disagree with where the operand
constant really lives? Candidates worth checking (not yet eliminated):

- `IntValue::new(ctx, id, bits)` sets `func: None`; only `with_owner` sets it.
  Enumerate every construction site and see which paths leave it `None` or set
  it to the wrong function.
- `basic_block.rs:get_instructions()` builds handles with
  `func: Some(self.func)`. Is `self.func` correct there?
- `InstructionValue::get_previous_instruction()` / `get_next_instruction()`
  build handles too — check the `func` they assign.
- Constant ids may be *shared or interned* across functions by the lifter. If
  the lifter puts a constant in several arenas at different indices, or reuses
  one arena's constant id for a different value in another, no owner-based
  scheme can work and the fix belongs in `biara-ir` or the lifter, not in the
  compat layer.

**The decisive experiment:** dump, for id 184 across all 47 functions, which
hold it and what kind of `Value` it is. If the constant `-16` exists in the
function that owns `%v66` but at a *different* id, then the handle's id is
itself wrong and the bug is in handle construction, not owner resolution.

### Acceptance for this defect

- `cargo test -p biara-lift --test audit` — all 3 pass.
- `cargo test -p biara-lift --test aarch64 -- --test-threads=4` — still 18/18.
  Run this before committing. Committing without it is what caused the revert
  in §5.

---

## 3. discovery: one jump-table arm becomes a function (x86_64, Static, -O2)

`cargo test -p biara-lift --test discovery` reports:

```
x86_64 Static unwind=false -O2: missed [], extra [2011af, 2011b3]
```

`missed` is empty everywhere — nothing is lost; two extra entries appear.

Those two addresses are the `one`/`two` labels in `tests/discovery/prog.c`,
which the corpus explicitly says are **not** functions:

```c
NOINLINE long dispatch(long op) {
  static void *const labels[] = {&&one, &&two};
  goto *labels[op & 1];
one: return op + 1;
two: return op * 3;
}
```

Disassembly of the failing build:

```
2011a8: ff 24 cd 60 01 20 00    jmpq *0x200160(,%rcx,8)
2011af: 48 ff c0                incq %rax          <- label "one"
2011b3: 48 8d 04 40             leaq (%rax,%rax,2) <- label "two"
```

and `.rodata` at `0x200160` holds exactly `[0x2011af, 0x2011b3]`, so the table
is readable and unambiguous.

### Why it fails (established)

`slot` on an `IndirectJump` is set only when the branch target is `Sym::Mem(a)`
— a plain memory read. See `crates/biara-cfg/src/exits.rs`, `exit_kind`:

```rust
let slot = match pc {
    Sym::Mem(a) => Some(a),
    _ => None,
};
```

Here the target is `load(0x200160 + rcx*8)`, an **indexed** load, so `pc` is a
computed value and `slot` is `None`. With no slot, `tables::find` cannot
enumerate the table, the site stays in `incomplete_jump_tables`, and recovery
never learns the arms are arms. `discover()` then adds `2011af`/`2011b3` from
the `data_pointers` set as functions, because by the time it runs the blocks
that would prove them interior do not yet exist.

### Two partial fixes already landed (do not remove)

`a725eb6` — `could_be_entry` in `crates/biara-cfg/src/lib.rs` rejects a
candidate strictly inside the span of a recovered function it is not the entry
of. That fixed 3 of 4 discovery configurations (Pie, PieRelr and aarch64 at
-O0). Static `-O2` still fails.

`37bfd2f` — `get_opcode` verifies the stated owner holds the instruction, like
`resolve_inst` already did.

### Where the fix belongs

Either (a) teach `exit_kind` to record the table's constant base for an
indexed load — `0x200160` is recoverable at lift time — or (b) make
`discover()` defer candidates that a resolved indirect jump already covers.
(a) is the smaller, more honest fix and would also help `indirect.rs`.

### Acceptance

`cargo test -p biara-lift --test discovery` — 4/4. Then the aarch64 gate.

---

## 4. engine: `frame`'s indirect call target is not resolved

```
cargo test -p biara-cli --test engine
persistent_resolution_keeps_shared_memory_and_named_register_results ... FAILED
  left: Array []          right: Array [Number(2101832)]   (0x201248, `chosen`)
```

The test builds this:

```c
long run_frame(Callback first) {
    Word slots[2] = { (Word)chosen ^ 0x1234, 0xdeadbeef00001234UL };
    return frame(slots);
}
extern long frame(Word *);
```

```asm
frame:
  movq   (%rdi), %rax        ; slots[0]  = chosen ^ 0x1234
  movzwl 0x8(%rdi), %ecx     ; slots[1] low half = 0x1234
  xorq   %rcx, %rax          ; -> chosen
  callq  *%rax
```

The target is `load(rdi) ^ zext16(load(rdi+8))` — arithmetic over **two guest
memory loads**, where `rdi` is the caller's constant array. `publish` in the
same test resolves fine because its target comes from a register.

This needs interprocedural memory forwarding: `frame`'s `rdi` is `run_frame`'s
`slots` array, whose contents are constants. `Arg::Deref` and `Leaf::Loaded`
exist for a *single* `load(reg+off)`; there is no representation for `xor` of
two loads, and `Arg::Expr` over two `Deref`s is never resolved against the
caller's frame.

Reproduce the current state directly:

```sh
cd target/tmp/engine-shared-frame   # created by the test
/Users/mikus/biara/target/debug/biara lift program _start \
  --emit cfg --format json --max-blocks 64 --jobs 1 --database /tmp/e.db \
  | python3 -c "import json,sys; [print(hex(b['address']),b['call_targets']) for b in json.load(sys.stdin)['analysis']['blocks'] if b.get('call_targets')]"
```

`0x201210` (`frame`) reports `{'complete': False, 'targets': []}`;
`0x20122a` (`publish`) reports `{'complete': False, 'targets': [2101832]}`.

Note `--emit cfg` prints **text** by default; `--format json` is required, which
is what `tests/engine.rs::run()` does.

### Acceptance

`cargo test -p biara-cli --test engine` — 9/9, with `--jobs 2` and `--jobs 1`
both checked (the test does both). Then the aarch64 gate.

---

## 5. What was already fixed, and the one reverting

Landed and verified — do not undo:

| Commit | Defect |
|---|---|
| `28cf4b2` | `print_module`'s `operand` indexed `values` unguarded; a cross-function operand panicked the printer, which is why `BIARA_DUMP_ANALYSIS_IR` appeared never to fire |
| `37bfd2f` | `get_opcode` trusted `self.func` without checking, unlike `resolve_inst` |
| `a725eb6` | `could_be_entry` accepted candidates at a recovered block's *end* even when strictly inside the function's span |
| `8835da1`, `35b5152`, `3669ef8`, etc. | the LLVM removal itself — `biara-ir` replaces it, `otool -L target/debug/biara` shows no llvm |

**Reverted:** `0e57599` → reverted by `db46187`. It bundled three changes; the
`listed` widening broke aarch64 to 1/17. Bisect evidence: `tail` passes at
`a725eb6`, fails at `0e57599`; reverting `is_const` alone still failed;
reverting `listed` as well made it pass. Its two companion changes
(`InstructionValue` equality ignoring `func`, and `is_const` ignoring arena
constants) were reverted with it and may be worth re-landing **separately and
individually gated** — but the equality one is suspicious in its own right,
because it made maps keyed by instruction distinguish handles that the rest of
the code treats as equal. If you re-land it, run the gate on it alone.

---

## 6. Ground rules that will save you time

1. **Run the aarch64 gate before every commit.** It is the only thing that
   catches regressions in instruction handling, ranges, indirect jumps and the
   lifter. A change that improves `audit` while breaking aarch64 is a net loss.
2. One `cargo test` at a time. Always.
3. No debug hooks left behind. The tree was repeatedly polluted with
   `eprintln!`/`BIARA_*_DBG` scaffolding; grep before committing:
   `rg 'BIARA_[A-Z_]*DBG|eprintln!\("(DBG|TRACE|B66)' crates/`
4. Never leave a speculative change that has no measurement behind it. Several
   were added and reverted during this work (extent probing, range filters,
   `last_store` cross-block lookback, a post-fixpoint label purge).
5. `aarch64` corpus is 289 inputs × 4 lift variants × 18 tests. It is slow but
   it is the truth.
