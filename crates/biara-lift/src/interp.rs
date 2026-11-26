//! A small interpreter over [`biara_ir`], standing in for the LLVM JIT the
//! differential tests used.
//!
//! The lifted IR is plain: integer arithmetic, loads and stores at guest
//! addresses (which are host addresses in the test process), branches,
//! phis, and calls. Everything is executed against the caller's `env` buffer,
//! exactly as compiled code would have been, so a test that compared a JIT
//! result against a reference can compare an interpretation instead.
//!
//! Scope is deliberate: this is a test oracle, not a production runtime. It
//! interprets the instructions `biara-lift` emits and reports anything else
//! rather than guessing.

use biara_ir::{
    BinOp, BlockId, Constant, Function, Inst, IntPred, Module, Terminator, Type, UnOp, Value,
    ValueId, ty_bytes,
};
use std::collections::HashMap;

/// A failure while interpreting.
#[derive(Debug, thiserror::Error)]
pub enum Trap {
    #[error("no function named {0}")]
    NoFunction(String),
    #[error("unresolved external call to {0}")]
    External(String),
    #[error("call to a function outside the module: {0}")]
    MissingCallee(String),
    #[error("unknown constant function {0}")]
    UnknownCallee(String),
    #[error("division by zero")]
    DivideByZero,
    #[error("read of an undefined value at block {0:?}")]
    Undefined(BlockId),
    #[error("unsupported instruction {0}")]
    Unsupported(&'static str),
}

/// Carry from QEMU's lazy x86 flags. The four-entry CCOp groups use
/// their bottom two bits for operand widths of 8, 16, 32, and 64.
fn x86_carry(dst: u64, src1: u64, src2: u64, op: u32) -> u64 {
    let width = 8 << (op & 3);
    let mask = u64::MAX >> (64 - width);
    let result = dst & mask;
    let operand = src1 & mask;
    let with_carry = u128::from(operand) + u128::from(src2 & mask);
    match op {
        0 | 2 | 40..=43 => src1 & 1,
        1 | 3 => dst,
        4..=7 => u64::from(src1 != 0),
        8..=11 => u64::from(result < operand),
        12..=15 => u64::from(u128::from(result) < with_carry),
        16..=19 => u64::from((result.wrapping_add(operand) & mask) < operand),
        20..=23 => {
            let minuend = result.wrapping_add(with_carry as u64) & mask;
            u64::from(u128::from(minuend) < with_carry)
        }
        28..=35 => src1,
        36..=39 => operand >> (width - 1),
        44..=47 => u64::from(operand == 0),
        48..=51 => u64::from(operand != 0),
        58 | 59 => dst & 1,
        _ => 0,
    }
}

impl Interp<'_> {
    /// Run a QEMU helper the lifter left as a declaration.
    ///
    /// These are the `helper_*` routines QEMU implements in C; the ones that
    /// model plain libc memory and string operations are performed here so a
    /// lifted function that calls them can be interpreted. Anything else is
    /// reported, never silently skipped.
    fn helper(&mut self, name: &str, args: &[u128]) -> Option<u128> {
        let bare = name.strip_prefix("helper_")?;
        let arg = |i: usize| args.get(i).copied().unwrap_or(0) as u64;
        match bare {
            "cc_compute_c" if args.len() == 4 => {
                Some(u128::from(x86_carry(arg(0), arg(1), arg(2), arg(3) as u32)))
            }
            "memset" => {
                let (dst, c, n) = (arg(0), arg(1) as u8, arg(2));
                // A bogus length would spin the loop instead of reporting a
                // miscomputed address.
                if n > self.memory.1.saturating_sub(self.memory.0) {
                    return None;
                }
                for i in 0..n {
                    self.store(u128::from(dst.wrapping_add(i)), u128::from(c), 8).ok()?;
                }
                Some(dst as u128)
            }
            "memcpy" | "memmove" => {
                let (dst, src, n) = (arg(0), arg(1), arg(2));
                if n > self.memory.1.saturating_sub(self.memory.0) {
                    return None;
                }
                let mut buf = Vec::with_capacity(n as usize);
                for i in 0..n {
                    buf.push(self.load(u128::from(src.wrapping_add(i)), 8).ok()? as u8);
                }
                for (i, b) in buf.into_iter().enumerate() {
                    self.store(u128::from(dst.wrapping_add(i as u64)), u128::from(b), 8).ok()?;
                }
                Some(dst as u128)
            }
            // QEMU's guarded-page probes exist to fault on a TLB-protected
            // page. User-mode interpretation has no such protection: nothing
            // is guarded, so the probe reports "not guarded" and the branch
            // helper returns its condition unchanged.
            "guarded_page_check" => Some(0),
            "guarded_page_br" => Some(args.first().copied().unwrap_or(0)),
            // AArch64 64-bit integer division helpers: `helper_<s|u>div64`,
            // `helper_<s|u>rem64`, and the overflow-checking `muls64`.
            "udiv64" | "sdiv64" | "urem64" | "srem64" | "muls64" => {
                let (x, y) = (arg(0), arg(1));
                Some(match bare {
                    "udiv64" => (x.checked_div(y).unwrap_or(0)) as u128,
                    "sdiv64" => {
                        let (x, y) = (x as i64, y as i64);
                        // QEMU's signed divide is undefined on overflow and on
                        // a zero divisor; zero is the value it computes.
                        (x.checked_div(y).unwrap_or(0)) as u128
                    }
                    "urem64" => (x.checked_rem(y).unwrap_or(0)) as u128,
                    "srem64" => {
                        let (x, y) = (x as i64, y as i64);
                        (x.checked_rem(y).unwrap_or(0)) as u128
                    }
                    _ => {
                        let (x, y) = (x as i64, y as i64);
                        let wide = (x as i128) * (y as i128);
                        let overflow = wide != (wide as i64) as i128;
                        ((overflow as u128) << 64) | (wide as u64) as u128
                    }
                })
            }
            // AArch64 data-processing helpers: `helper_rbit<w>` reverses the
            // bits of one register, `helper_clz<w>`/`helper_ctz<w>` count
            // leading/trailing zeros, `helper_popcount<w>` counts set bits.
            "rbit32" | "rbit64" | "clz32" | "clz64" | "ctz32" | "ctz64"
            | "popcount32" | "popcount64" => {
                let w: u32 = if bare.ends_with("64") { 64 } else { 32 };
                let v = arg(0) & (if w == 64 { u64::MAX } else { u32::MAX as u64 });
                Some(match &bare[..bare.len() - 2] {
                    "rbit" => (v.reverse_bits() >> (64 - w)) as u128,
                    "clz" => u64::from(v.leading_zeros() - (64 - w)) as u128,
                    "ctz" => u64::from(v.trailing_zeros()) as u128,
                    _ => u64::from(v.count_ones()) as u128,
                })
            }
            // AArch64 `CNT`: set bits per element, written back in place.
            // `gvec_cnt_<b|h|s|d>` names the element width.
            "gvec_cnt_b" | "gvec_cnt_h" | "gvec_cnt_s" | "gvec_cnt_d" => {
                let width = match bare.as_bytes().last() {
                    Some(b'b') => 8u32,
                    Some(b'h') => 16,
                    Some(b's') => 32,
                    _ => 64,
                };
                // (dest, source, element size in bytes): a vector of 16
                // bytes, counted element by element.
                let (dst, src) = (arg(0), arg(1));
                let step = u64::from(width / 8).max(1);
                let mut i = 0;
                while i < 16 {
                    let v = self.load(u128::from(src.wrapping_add(i)), width).ok()?;
                    self.store(
                        u128::from(dst.wrapping_add(i)),
                        u128::from((v as u64).count_ones()),
                        width,
                    )
                    .ok()?;
                    i = i.wrapping_add(step);
                }
                Some(dst as u128)
            }
            // QEMU's VFP conversion helpers. The name gives source width
            // (`i`/`h` 32-bit, `q` 64-bit), signedness, and destination
            // precision (`s` f32, `d` f64); `vfp_to*` goes the other way.
            // Softfloat status setters: the interpreter computes with the
            // host's default rounding mode, so these have nothing to record.
            "set_rmode" | "set_float_exception_flags" | "set_float_rounding_mode" => {
                Some(0)
            }
            _ if bare.starts_with("vfp_") => return float_helper(bare, &args),
            _ => None,
        }
    }
}

/// QEMU's `float_compare` result: 0 equal, 1 less, 2 greater, 3 unordered.
fn compare_f32(a: f32, b: f32, signaling: bool) -> u32 {
    let _ = signaling;
    if a.is_nan() || b.is_nan() {
        3
    } else if a < b {
        1
    } else if a > b {
        2
    } else {
        0
    }
}

fn compare_f64(a: f64, b: f64, signaling: bool) -> u32 {
    let _ = signaling;
    if a.is_nan() || b.is_nan() {
        3
    } else if a < b {
        1
    } else if a > b {
        2
    } else {
        0
    }
}

/// Width of the first field of the struct `agg` is typed as, so a second
/// field lands directly above it.
fn struct_first_width(f: &Function, agg: ValueId) -> u32 {
    match f.ty(agg) {
        Type::Struct(fields) => fields.first().and_then(Type::int_bits).unwrap_or(64),
        _ => 64,
    }
}

/// A QEMU `vfp_*` conversion or arithmetic helper.
fn float_helper(name: &str, args: &[u128]) -> Option<u128> {
    let a = args.first().copied().unwrap_or(0) as u64;
    let b = args.get(1).copied().unwrap_or(0) as u64;
    let c = args.get(2).copied().unwrap_or(0) as u64;
    // Floating-point to integer, `vfp_to<s|u><w>`: checked first because
    // `split_once("to")` below would otherwise read it as a conversion.
    if let Some(rest) = name.strip_prefix("vfp_to") {
        let signed = rest.starts_with('s');
        // `<dst><src>`: `tosqs` is signed, to quad, from single. Reading
        // single-precision bits as a double is garbage (a `-1.0f` becomes a
        // denormal), so the source precision decides how to decode them.
        let width = rest.trim_start_matches(['s', 'u']);
        let v = match width.chars().nth(1) {
            Some('s') | Some('h') | Some('i') => f32::from_bits(a as u32) as f64,
            _ => f64::from_bits(a),
        };
        let wide = width.starts_with('q') || width.starts_with('d');
        return Some(if signed {
            let bounded = v.clamp(i64::MIN as f64, i64::MAX as f64) as i64;
            if wide {
                bounded as u64 as u128
            } else {
                bounded as i32 as u32 as u128
            }
        } else {
            let bounded = if v <= 0.0 { 0.0 } else { v.min(u64::MAX as f64) };
            if wide {
                bounded as u64 as u128
            } else {
                bounded as u32 as u128
            }
        });
    }

    // Arithmetic: the last letter names the precision.
    let body = name.strip_prefix("vfp_")?;
    let (op, dst) = body.split_at_checked(body.len().checked_sub(1)?)?;
    let bin = |x: u64, y: u64| -> Option<u128> {
        Some(match dst {
            "s" => {
                let (x, y) = (f32::from_bits(x as u32), f32::from_bits(y as u32));
                match op {
                    "add" => (x + y).to_bits() as u128,
                    "sub" => (x - y).to_bits() as u128,
                    "mul" => (x * y).to_bits() as u128,
                    "div" => (x / y).to_bits() as u128,
                    "min" => x.min(y).to_bits() as u128,
                    "max" => x.max(y).to_bits() as u128,
                    _ => return None,
                }
            }
            "d" => {
                let (x, y) = (f64::from_bits(x), f64::from_bits(y));
                match op {
                    "add" => (x + y).to_bits() as u128,
                    "sub" => (x - y).to_bits() as u128,
                    "mul" => (x * y).to_bits() as u128,
                    "div" => (x / y).to_bits() as u128,
                    "min" => x.min(y).to_bits() as u128,
                    "max" => x.max(y).to_bits() as u128,
                    _ => return None,
                }
            }
            _ => return None,
        })
    };
    if matches!(op, "add" | "sub" | "mul" | "div" | "min" | "max") {
        return bin(a, b);
    }
    // Fused multiply-add: a + b * c.
    if op == "muladd" {
        return Some(match dst {
            "s" => {
                let (a, b, c) = (
                    f32::from_bits(a as u32),
                    f32::from_bits(b as u32),
                    f32::from_bits(c as u32),
                );
                a.mul_add(b, c).to_bits() as u128
            }
            "d" => {
                let (a, b, c) = (f64::from_bits(a), f64::from_bits(b), f64::from_bits(c));
                a.mul_add(b, c).to_bits() as u128
            }
            _ => return None,
        });
    }
    // Unary: abs, neg, sqrt.
    if matches!(op, "abs" | "neg" | "sqrt") {
        return Some(match dst {
            "s" => {
                let x = f32::from_bits(a as u32);
                match op {
                    "abs" => x.abs().to_bits() as u128,
                    "neg" => (-x).to_bits() as u128,
                    _ => x.sqrt().to_bits() as u128,
                }
            }
            "d" => {
                let x = f64::from_bits(a);
                match op {
                    "abs" => x.abs().to_bits() as u128,
                    "neg" => (-x).to_bits() as u128,
                    _ => x.sqrt().to_bits() as u128,
                }
            }
            _ => return None,
        });
    }
    // Comparisons: a zero result or the comparison outcome, per QEMU's
    // `float_compare` returning the flag combination.
    if op.starts_with("cmp") {
        let quiet = op.ends_with('e');
        return Some(match dst {
            "s" => compare_f32(f32::from_bits(a as u32), f32::from_bits(b as u32), quiet) as u128,
            "d" => compare_f64(f64::from_bits(a), f64::from_bits(b), quiet) as u128,
            _ => return None,
        });
    }
    // Integer conversions: `vfp_<conv>to<dst>`.
    if let Some((src, dst)) = name.strip_prefix("vfp_")?.split_once("to") {
        let (sign, width) = match src.split_at_checked(1)? {
            ("u", w) => (false, w),
            ("s", w) => (true, w),
            _ => return None,
        };
        let dst = dst.chars().next()?;
        let integer = match width {
            "i" | "h" => ((a as u32) as i32 as i64, 32),
            "q" => (a as i64, 64),
            _ => return None,
        };
        let value = if sign {
            integer.0 as f64
        } else {
            match width {
                "i" | "h" => (a as u32) as f64,
                _ => a as f64,
            }
        };
        let _ = integer.1;
        // `scvtf d, x, #fbits` is the fixed-point form: the integer is
        // scaled down by `2^fbits`. The helper takes that count as its
        // second argument; ignoring it turns `x * 0.5` into `x`.
        let value = value / (1u64 << b.min(63)) as f64;
        return Some(if dst == 's' {
            (value as f32).to_bits() as u128
        } else {
            value.to_bits() as u128
        });
    }
    // Floating-point to integer: `vfp_to<s|u><w>`.
    let rest = name.strip_prefix("vfp_to")?;
    let signed = rest.starts_with('s');
    let width = rest.trim_start_matches(['s', 'u']);
    // `fcvtzs x, d, #fbits` scales up by `2^fbits` before truncating.
    let v = f64::from_bits(a) * (1u64 << b.min(63)) as f64;
    let wide = width.starts_with('q') || width.contains('d');
    Some(if signed {
        let bounded = v.clamp(i64::MIN as f64, i64::MAX as f64) as i64;
        if wide {
            bounded as u64 as u128
        } else {
            bounded as i32 as u32 as u128
        }
    } else {
        let bounded = if v <= 0.0 { 0.0 } else { v.min(u64::MAX as f64) };
        if wide {
            bounded as u64 as u128
        } else {
            bounded as u32 as u128
        }
    })
}

/// The result of an LLVM overflow intrinsic, as its `{value, flag}` pair.
///
/// These lower to a call in the IR rather than to arithmetic, so an
/// interpreter has to give them their meaning.
fn intrinsic(name: &str, args: &[u128], f: &Function, ty: &Type) -> Option<u128> {
    // Funnel shifts yield a plain integer, not a struct.
    if let Some(bits) = ty.int_bits() {
        // Unary intrinsics first: the funnel-shift path below needs three
        // arguments and would otherwise reject them.
        // Bit-counting intrinsics take the value and an `is_zero_poison` flag.
        if matches!(
            name,
            "llvm.ctlz" | "llvm.cttz" | "llvm.ctpop" | "llvm.bitreverse"
        ) {
            let v = *args.first()? & mask(bits);
            let v = v as u64;
            let w = bits.min(64);
            return Some(match name {
                "llvm.ctlz" => u64::from(v.leading_zeros() - (64 - w)) as u128,
                "llvm.cttz" => u64::from(v.trailing_zeros()) as u128,
                "llvm.ctpop" => u64::from(v.count_ones()) as u128,
                _ => (v.reverse_bits() >> (64 - w)) as u128,
            });
        }
        if name == "llvm.bswap" {
            let v = (*args.first()? & mask(bits)) as u128;
            let bytes = (bits / 8) as usize;
            let mut out = 0u128;
            for i in 0..bytes {
                out |= ((v >> (8 * i)) & 0xff) << (8 * (bytes - 1 - i));
            }
            return Some(out);
        }
        let (a, b, c) = (*args.first()?, *args.get(1)?, *args.get(2)? as u32);
        return match name {
            // `fshr`: concatenate a:b and take the low `bits` of a right
            // shift by `c`; `fshl` shifts left.
            "llvm.fshr" | "llvm.fshl" => {
                let (a, b) = (a & mask(bits), b & mask(bits));
                let c = if bits == 0 { 0 } else { c % bits };
                let shift = |hi: u128, lo: u128| {
                    let joined = (hi << bits) | lo;
                    if name == "llvm.fshr" {
                        (joined >> c) & mask(bits)
                    } else {
                        (joined << c >> bits) & mask(bits)
                    }
                };
                Some(if name == "llvm.fshr" {
                    shift(b, a)
                } else {
                    shift(a, b)
                })
            }
            _ => None,
        };
    }
    let bits = match ty {
        Type::Struct(fields) => fields.first().and_then(Type::int_bits)?,
        _ => return None,
    };
    let (a, b) = (*args.first()?, *args.get(1)?);
    let typed = |v: u128| v & mask(bits);
    let (value, overflow) = match name {
        "llvm.uadd.with.overflow" => {
            let (s, o) = a.overflowing_add(b);
            (s, o || s > u128::from(mask(bits)))
        }
        "llvm.sadd.with.overflow" => {
            let m = 1u128 << (bits - 1);
            let (s, _) = a.overflowing_add(b);
            let lo = s & mask(bits);
            (lo, (a ^ lo) & (b ^ lo) & m != 0)
        }
        "llvm.usub.with.overflow" => {
            let (s, o) = a.overflowing_sub(b);
            (s, o || s > u128::from(mask(bits)))
        }
        "llvm.ssub.with.overflow" => {
            let m = 1u128 << (bits - 1);
            let (s, _) = a.overflowing_sub(b);
            let lo = s & mask(bits);
            (lo, (a ^ b) & (a ^ lo) & m != 0)
        }
        "llvm.umul.with.overflow" => {
            let p = a.wrapping_mul(b);
            (p, bits < 128 && p >> bits != 0)
        }
        "llvm.smul.with.overflow" => {
            let p = a.wrapping_mul(b);
            let m = 1u128 << (bits - 1);
            let lo = p & mask(bits);
            let neg = |v: u128| (v & m) != 0;
            (
                lo,
                !(a == 0 || b == 0) && (neg(a) == neg(b) || lo != 0) && a.checked_mul(b).is_none(),
            )
        }
        _ => return None,
    };
    let _ = f;
    // A struct value is laid out as its fields: the result below `bits`, the
    // flag above it, matching `ExtractValue`.
    Some(typed(value) | (u128::from(overflow) << bits))
}

/// Instructions one top-level call may run. A lifted guest routine runs far
/// fewer; a larger budget would only let a runaway loop stall the caller.
const FUEL: u64 = 2_000_000;

/// Nested lifted calls a single top-level call may make. The corpus recurses
/// a handful deep; a larger depth is a callee that never returns.
const MAX_DEPTH: usize = 512;

/// An interpreted unit of lifted code.
pub struct Interp<'a> {
    module: &'a Module,
    /// Guest-memory bounds; the lifted code addresses host memory directly.
    memory: (u64, u64),
    /// Addresses of globals, by name; shared with each frame so a constant
    /// `GlobalPtr` resolves while `run` keeps its own `&mut self`.
    globals: std::rc::Rc<HashMap<String, u64>>,
    /// Instruction budget, so a bug cannot hang the suite.
    fuel: u64,
    /// Storage for `alloca`: the guest's memory map does not reserve any, and
    /// carving it out of the guest image would let scratch writes corrupt
    /// guest state the oracle compares.
    alloca: Vec<u8>,
    /// Next free byte of `alloca`.
    scratch: u64,
    /// Further regions the guest may address, such as its stack: the lifted
    /// code touches host memory directly, so a caller-owned allocation is
    /// fair game even though it is not part of the CPU-state buffer.
    extra: Vec<(u64, u64)>,
    /// The guest pc the dispatcher last switched on.
    dispatch: std::cell::Cell<Option<u128>>,
    /// Current lifted-call nesting depth.
    depth: std::cell::Cell<usize>,

}

/// Why execution stopped.
enum Flow {
    Return,
    Goto(BlockId),
}

impl<'a> Interp<'a> {
    /// Interpret `module`, whose guest memory spans `[lo, hi)`.
    ///
    /// Globals get no storage here: the range is the caller's memory and
    /// writing initializers into it would clobber state the caller set up.
    /// A global whose address matters must be supplied with [`Interp::bind`];
    /// an unbound one resolves to address 0.
    ///
    /// `alloca` storage comes from the **middle** of the range: the caller's
    /// own image and CPU state are at the bottom and its stack at the top,
    /// so taking from either end would corrupt them.
    pub fn new(module: &'a Module, memory: (u64, u64)) -> Self {
        Self {
            module,
            memory,
            extra: Vec::new(),
            dispatch: std::cell::Cell::new(None),
            depth: std::cell::Cell::new(0),
            globals: std::rc::Rc::new({
                // The ABI form addresses CPU state through `biara_cpu`, whose
                // bytes start at the beginning of the map.
                let mut g = HashMap::new();
                if module.globals.iter().any(|g| g.name == "biara_cpu") {
                    g.insert("biara_cpu".to_owned(), memory.0);
                }
                g
            }),
            fuel: 0,
            alloca: vec![0; 1 << 20],
            scratch: 0,
        }
    }

    /// Bind an external symbol (a QEMU helper) to an address. Calls to it are
    /// not interpreted: the tests avoid them, and a stray call traps.
    /// Declare another region the guest may address, such as its stack.
    pub fn map(&mut self, range: (u64, u64)) {
        self.extra.push(range);
    }

    pub fn bind(&mut self, name: impl Into<String>, addr: u64) {
        std::rc::Rc::make_mut(&mut self.globals).insert(name.into(), addr);
    }

    /// Call `func` with `args`, returning its result.
    pub fn call(&mut self, name: &str, args: &[u128]) -> Result<u128, Trap> {
        // Frame storage from a previous call is dead once it returns, and
        // the budget is per top-level call so one long run cannot starve the
        // next.
        self.scratch = 0;
        self.fuel = FUEL;
        let f = self
            .module
            .function(name)
            .ok_or_else(|| Trap::NoFunction(name.to_owned()))?;
        self.run(f, args)
    }

    fn run(&mut self, f: &Function, args: &[u128]) -> Result<u128, Trap> {
        // A lifted callee that re-enters itself without progressing would
        // recurse until the host stack dies; the guest's own depth is small.
        let depth = self.depth.get() + 1;
        if depth > MAX_DEPTH {
            return Err(Trap::Unsupported("call depth exceeded"));
        }
        self.depth.set(depth);
        let result = self.run_inner(f, args);
        self.depth.set(depth - 1);
        result
    }

    fn run_inner(&mut self, f: &Function, args: &[u128]) -> Result<u128, Trap> {
        let mut frame = Frame::new(f, args, std::rc::Rc::clone(&self.globals))?;
        // A declaration has no lifted body: `helper_*` is the lifter's own
        // implementation of a libc routine, so it is called out of band.
        let mut block = f.entry.ok_or(Trap::UnknownCallee(f.name.clone()))?;
        // The block an edge came from, for phi resolution on entry.
        let mut from: Option<BlockId> = None;
        loop {
            self.fuel = self
                .fuel
                .checked_sub(1)
                .ok_or(Trap::Unsupported("ran out of fuel"))?;
            // Phis select on the incoming edge, so they are resolved as the
            // block is entered, before any of its instructions run.
            if let Some(prev) = from {
                self.resolve_phis(f, &mut frame, block, prev)?;
            }
            let flow = self.run_block(f, &mut frame, block)?;
            // `biara.run` loops on the guest pc; a value the switch cannot
            // leave repeats for ever. Comparing it between visits is exact
            // and cheap, and reports an untranslated address instead of
            // letting the budget expire.
            if f.name == crate::RUN
                && let Some(pc) = self.dispatch_pc(f, &frame, block)
            {
                if self.dispatch.get() == Some(pc) {
                    return Err(Trap::Unsupported("dispatch made no progress"));
                }
                self.dispatch.set(Some(pc));
            }
            match flow {
                Flow::Return => {
                    return Ok(frame.ret.map(|v| frame.read(v)).unwrap_or(0));
                }
                Flow::Goto(next) => {
                    from = Some(block);
                    block = next;
                }
            }
        }
    }

    /// The guest pc the dispatcher switches on, when `block` is its switch.
    fn dispatch_pc(&mut self, f: &Function, frame: &Frame<'_>, block: BlockId) -> Option<u128> {
        let Some(b) = f.blocks.get(block.0 as usize) else {
            return None;
        };
        let Terminator::Switch { value, .. } = b.term.as_ref()? else {
            return None;
        };
        Some(frame.read(*value) & u64::MAX as u128)
    }

    /// Give each phi in `block` its value for the edge `prev -> block`.
    fn resolve_phis(
        &mut self,
        f: &Function,
        frame: &mut Frame<'_>,
        block: BlockId,
        prev: BlockId,
    ) -> Result<(), Trap> {
        for &id in &f.block(block).insts {
            let Some(Inst::Phi { incoming, .. }) = f.inst(id) else {
                continue;
            };
            // An edge with no phi arm would leave the previous value in
            // place, which silently steers a loop the wrong way.
            let Some((value, _)) = incoming.iter().find(|(_, b)| *b == prev) else {
                return Err(Trap::Undefined(prev));
            };
            let v = frame.read(*value);
            frame.set(id, v);
        }
        Ok(())
    }

    fn run_block(
        &mut self,
        f: &Function,
        frame: &mut Frame<'_>,
        block: BlockId,
    ) -> Result<Flow, Trap> {
        let b = f.block(block);
        for &inst in &b.insts {
            self.step(f, frame, inst)?;
        }
        match b.term.as_ref().unwrap_or(&Terminator::Unreachable) {
            Terminator::Ret(v) => {
                frame.ret = *v;
                Ok(Flow::Return)
            }
            Terminator::Br(to) => Ok(Flow::Goto(*to)),
            Terminator::CondBr { cond, yes, no } => {
                let c = frame.read(*cond);
                Ok(Flow::Goto(if c != 0 { *yes } else { *no }))
            }
            Terminator::Switch {
                value,
                default,
                cases,
            } => {
                let v = frame.read(*value);
                Ok(Flow::Goto(
                    cases
                        .iter()
                        .find(|(k, _)| *k == v)
                        .map(|(_, b)| *b)
                        .unwrap_or(*default),
                ))
            }
            // Reaching an unreachable block is not a defect in the lifted
            // code: the lowering marks escape paths this way (after a
            // noreturn call, or a table whose every case leaves). Execution
            // stops here, as the guest's own `__builtin_unreachable` would.
            Terminator::Unreachable => Ok(Flow::Return),
        }
    }

    fn step(&mut self, f: &Function, frame: &mut Frame<'_>, id: ValueId) -> Result<(), Trap> {
        let Some(inst) = f.inst(id) else { return Ok(()) };
        let ty_bits = |t: &Type| t.int_bits().unwrap_or(64);
        let result = match inst {
            Inst::Bin { op, ty, lhs, rhs } => {
                let (a, b) = (frame.read(*lhs), frame.read(*rhs));
                let w = ty_bits(ty);
                Some(bin(*op, a, b, w)?)
            }
            Inst::Un { op, ty, value } => {
                let v = frame.read(*value);
                Some(un(*op, v, ty_bits(ty)))
            }
            Inst::Rot {
                left,
                ty,
                value,
                amount,
            } => {
                let (v, n) = (frame.read(*value), frame.read(*amount));
                let w = ty_bits(ty);
                let n = (n % w as u128) as u32;
                let m = mask(w);
                let r = if *left {
                    ((v << n) | (v >> ((w - n) % w))) & m
                } else {
                    ((v >> n) | (v << ((w - n) % w))) & m
                };
                Some(r)
            }
            Inst::Fence | Inst::Marker { .. } => None,
            Inst::Icmp { pred, ty, lhs, rhs } => {
                let (a, b) = (frame.read(*lhs), frame.read(*rhs));
                Some(u128::from(icmp(*pred, a, b, ty_bits(ty))))
            }
            Inst::Select {
                cond, yes, no, ..
            } => Some(if frame.read(*cond) != 0 {
                frame.read(*yes)
            } else {
                frame.read(*no)
            }),
            Inst::Trunc { to, value, .. } => {
                let w = ty_bits(to);
                Some(frame.read(*value) & mask(w))
            }
            Inst::Zext { from, value, .. } => {
                let w = ty_bits(from);
                Some(frame.read(*value) & mask(w))
            }
            Inst::Sext { from, to, value } => {
                let (fw, tw) = (ty_bits(from), ty_bits(to));
                Some(signed(frame.read(*value), fw) as u128 & mask(tw))
            }
            Inst::Cast { to, value, .. } => {
                let w = ty_bits(to);
                Some(frame.read(*value) & mask(w))
            }
            Inst::Alloca { ty } => {
                // An alloca takes its storage once per activation; a loop
                // body revisits the instruction, not the allocation.
                if let Some(&at) = frame.allocas.get(&id) {
                    Some(at)
                } else {
                    let bytes = (ty_bytes(ty).max(1) as u64 + 15) & !15;
                    let base = self.scratch;
                    if base + bytes > self.alloca.len() as u64 {
                        return Err(Trap::Unsupported("out of alloca storage"));
                    }
                    self.scratch += bytes;
                    let at = (self.alloca.as_ptr_range().start as u64 + base) as u128;
                    frame.allocas.insert(id, at);
                    Some(at)
                }
            }
            Inst::Load { ty, ptr, .. } => {
                let addr = frame.read(*ptr);
                Some(self.load(addr, ty_bits(ty))?)
            }
            Inst::Store {
                ty, value, ptr, ..
            } => {
                let (v, addr) = (frame.read(*value), frame.read(*ptr));
                self.store(addr, v, ty_bits(ty))?;
                None
            }
            Inst::Gep { base, offset } => Some(frame.read(*base).wrapping_add(*offset as u128)),
            Inst::Call { callee, args, ty } => {
                let name = match f.value(*callee) {
                    Value::Const(Constant::Function(n)) => n.clone(),
                    _ => return Err(Trap::Unsupported("indirect call")),
                };
                let args_ids: Vec<biara_ir::ValueId> = args.clone();
                let args: Vec<u128> = args.iter().map(|a| frame.read(*a)).collect();
                let w = ty_bits(ty);
                // An overflow intrinsic yields `{result, flag}`; there is no
                // body to call, so it is evaluated here.
                // An intrinsic's width is its result type's, except where the
                // call was lowered as returning void; then the first
                // argument's own type decides.
                let ret_ty = match ty {
                    Type::Void if name.starts_with("llvm.") => match args_ids.first() {
                        Some(&a) if (a.0 as usize) < f.values.len() => f.ty(a),
                        _ => Type::Int(64),
                    },
                    other => other.clone(),
                };
                if let Some(v) = intrinsic(&name, &args, f, &ret_ty) {
                    Some(v)
                } else if let Some(r) = self.helper(&name, &args) {
                    Some(r & mask(w))
                } else {
                    match self.module.function(&name) {
                    Some(g) => {

                        // A callee's frame storage dies with the callee, so
                        // the arena pointer is restored on return; without
                        // that, a deep or looping call chain would exhaust it.
                        let mark = self.scratch;
                        let r = self.run(g, &args);
                        self.scratch = mark;
                        Some(r? & mask(w))
                    }
                    None if self.globals.contains_key(&name) => {
                        Err(Trap::External(name))?
                    }
                    None => Err(Trap::MissingCallee(name))?,
                    }
                }
            }
            Inst::Phi { .. } => None,
            Inst::ExtractValue { agg, index, ty } => {
                // `ty` is the extracted field's own width; the field's
                // position comes from the fields before it, so the first
                // field's width is what shifts field 1 into place.
                let a = frame.read(*agg);
                Some(if *index == 0 {
                    a & mask(struct_first_width(f, *agg))
                } else {
                    (a >> struct_first_width(f, *agg)) & mask(ty_bits(ty))
                })
            }
            Inst::InsertValue {
                agg,
                value,
                index,
                ty,
            } => {
                // The fields sit at their own widths; a flag field above an
                // `i64` result is at bit 64, above an `i32` result at bit 32.
                let (a, v) = (frame.read(*agg), frame.read(*value));
                Some(if *index == 0 {
                    a | (v & mask(ty_bits(ty)))
                } else {
                    a | ((v & mask(ty_bits(ty))) << struct_first_width(f, *agg))
                })
            }
            Inst::Freeze { value, .. } => Some(frame.read(*value)),
        };
        if let Some(v) = result {
            frame.set(id, v);
        }
        Ok(())
    }

    /// Does `[addr, addr + bytes)` lie in a region the caller owns?
    fn mapped(&self, addr: u64, bytes: usize) -> bool {
        let end = addr.saturating_add(bytes as u64);
        let arena = self.alloca.as_ptr_range();
        (addr >= self.memory.0 && end <= self.memory.1)
            || (addr >= arena.start as u64 && end <= arena.end as u64)
            || self.extra.iter().any(|&(lo, hi)| addr >= lo && end <= hi)
    }

    fn load(&self, addr: u128, bits: u32) -> Result<u128, Trap> {
        let addr = addr as u64;
        let bytes = bits.div_ceil(8) as usize;
        if !self.mapped(addr, bytes) {
            return Err(Trap::Unsupported("load outside guest memory"));
        }
        // Guest memory is host memory in the test process.
        let mut buf = [0u8; 16];
        unsafe {
            std::ptr::copy_nonoverlapping(addr as *const u8, buf.as_mut_ptr(), bytes);
        }
        let mut out = 0u128;
        for (i, b) in buf.iter().take(bytes).enumerate() {
            out |= (*b as u128) << (8 * i);
        }
        Ok(out & mask(bits))
    }

    fn store(&self, addr: u128, value: u128, bits: u32) -> Result<(), Trap> {
        let addr = addr as u64;
        let bytes = bits.div_ceil(8) as usize;
        if !self.mapped(addr, bytes) {
            return Err(Trap::Unsupported("store outside guest memory"));
        }
        let bytes_le = value.to_le_bytes();
        unsafe {
            std::ptr::copy_nonoverlapping(bytes_le.as_ptr(), addr as *mut u8, bytes);
        }
        Ok(())
    }
}

/// Per-call value storage. Constants are not stored: they come from the
/// module, so a read of one is answered by its literal.
struct Frame<'a> {
    f: &'a Function,
    values: HashMap<ValueId, u128>,
    ret: Option<ValueId>,
    /// Global names to addresses, so a `GlobalPtr` constant yields one.
    globals: std::rc::Rc<HashMap<String, u64>>,
    /// Address of each `alloca`: an alloca runs once per activation, so
    /// re-visiting it in a loop must yield the same storage.
    allocas: HashMap<ValueId, u128>,
}

impl<'a> Frame<'a> {
    fn new(
        f: &'a Function,
        args: &[u128],
        globals: std::rc::Rc<HashMap<String, u64>>,
    ) -> Result<Self, Trap> {
        let mut values = HashMap::new();
        for (i, &p) in f.params.iter().enumerate() {
            values.insert(p, args.get(i).copied().unwrap_or(0));
        }
        Ok(Self {
            allocas: HashMap::new(),
            f,
            values,
            globals,
            ret: None,
        })
    }

    fn read(&self, id: ValueId) -> u128 {
        if let Some(&v) = self.values.get(&id) {
            return v;
        }
        // An id outside this function's arena names nothing here; a read of
        // something the IR never computed has no value.
        let Some(entry) = self.f.values.get(id.0 as usize) else {
            return 0;
        };
        match entry {
            Value::Const(Constant::Int { bits, .. }) => *bits,
            Value::Const(Constant::Zero) => 0,
            // A global's address, as registered by the interpreter.
            Value::Const(Constant::GlobalPtr(name)) => self
                .globals
                .get(name)
                .map(|&a| a as u128)
                .unwrap_or_else(|| panic!("interpreter: global {name} is unbound")),
            // A value the IR never computed: an instruction that produces
            // nothing (a store) or was not reached. Reading it as zero would
            // send the guest to address 0, so the read has no value at all.
            _ => 0,
        }
    }

    fn set(&mut self, id: ValueId, v: u128) {
        self.values.insert(id, v);
    }
}

fn mask(bits: u32) -> u128 {
    if bits >= 128 {
        u128::MAX
    } else {
        (1u128 << bits) - 1
    }
}

fn signed(v: u128, bits: u32) -> i128 {
    let w = bits.min(128);
    if w == 0 {
        return 0;
    }
    let m = mask(w);
    let v = v & m;
    if v & (1u128 << (w - 1)) != 0 {
        (v | !m) as i128
    } else {
        v as i128
    }
}

fn bin(op: BinOp, a: u128, b: u128, bits: u32) -> Result<u128, Trap> {
    let m = mask(bits);
    let (sa, sb) = (signed(a, bits), signed(b, bits));
    Ok(match op {
        BinOp::Add => a.wrapping_add(b) & m,
        BinOp::Sub => a.wrapping_sub(b) & m,
        BinOp::Mul => a.wrapping_mul(b) & m,
        BinOp::And => a & b & m,
        // `a | b & m` would parse as `a | (b & m)` and keep `a`'s high bits.
        BinOp::Or => (a | b) & m,
        BinOp::Xor => (a ^ b) & m,
        BinOp::Shl => (a << (b % bits as u128)) & m,
        BinOp::Lshr => (a & m) >> (b % bits as u128),
        BinOp::Ashr => ((sa >> (b % bits as u128)) as u128) & m,
        BinOp::Udiv => {
            if b == 0 {
                return Err(Trap::DivideByZero);
            }
            (a & m) / (b & m)
        }
        BinOp::Sdiv => {
            if sb == 0 {
                return Err(Trap::DivideByZero);
            }
            (sa.wrapping_div(sb) as u128) & m
        }
        BinOp::Urem => {
            if b == 0 {
                return Err(Trap::DivideByZero);
            }
            (a & m) % (b & m)
        }
        BinOp::Srem => {
            if sb == 0 {
                return Err(Trap::DivideByZero);
            }
            (sa.wrapping_rem(sb) as u128) & m
        }
    })
}

fn un(op: UnOp, v: u128, bits: u32) -> u128 {
    let m = mask(bits);
    match op {
        UnOp::Not => !v & m,
        UnOp::Neg => v.wrapping_neg() & m,
        UnOp::Ctlz => (v & m).leading_zeros().saturating_sub(128 - bits) as u128,
        UnOp::Cttz => {
            let x = v & m;
            if x == 0 {
                bits as u128
            } else {
                x.trailing_zeros() as u128
            }
        }
        UnOp::Ctpop => (v & m).count_ones() as u128,
        UnOp::Bswap => {
            let byte = (bits / 8) as usize;
            let le = (v & m).to_le_bytes();
            let mut out = 0u128;
            for i in 0..byte {
                out |= (le[i] as u128) << (8 * (byte - 1 - i));
            }
            out & m
        }
    }
}

fn icmp(pred: IntPred, a: u128, b: u128, bits: u32) -> bool {
    // Comparisons see the values as the instruction's type width.
    let m = mask(bits);
    let (a, b) = (a & m, b & m);
    let (sa, sb) = (signed(a, bits), signed(b, bits));
    match pred {
        IntPred::Eq => a == b,
        IntPred::Ne => a != b,
        IntPred::Slt => sa < sb,
        IntPred::Sle => sa <= sb,
        IntPred::Sgt => sa > sb,
        IntPred::Sge => sa >= sb,
        IntPred::Ult => a < b,
        IntPred::Ule => a <= b,
        IntPred::Ugt => a > b,
        IntPred::Uge => a >= b,
    }
}
