//! The LLVM intrinsics `biara-lift` calls, modelled as the IR operations they
//! stand for. `llvm.*.with.overflow` returns `{ iN, i1 }`; `biara-ir` spells
//! that as a two-field struct built from the value and the overflow bit.

use crate::module::Module;
use crate::types::BasicType;
use crate::values::FunctionValue;

pub struct Intrinsic {
    name: String,
}

impl Intrinsic {
    /// Look up an intrinsic by its LLVM name.
    ///
    /// Every name the lowering and the renderer use is recognised; an
    /// unknown name is not an intrinsic and the caller must handle it.
    pub fn find(name: &str) -> Option<Self> {
        const KNOWN: &[&str] = &[
            "llvm.uadd.with.overflow",
            "llvm.sadd.with.overflow",
            "llvm.usub.with.overflow",
            "llvm.ssub.with.overflow",
            "llvm.umul.with.overflow",
            "llvm.smul.with.overflow",
            "llvm.ctlz",
            "llvm.cttz",
            "llvm.ctpop",
            "llvm.bswap",
            "llvm.bitreverse",
            "llvm.fshl",
            "llvm.fshr",
            "llvm.abs",
            "llvm.smax",
            "llvm.smin",
            "llvm.umax",
            "llvm.umin",
            "llvm.memcpy",
            "llvm.memmove",
            "llvm.memset",
        ];
        KNOWN.contains(&name).then(|| Self {
            name: name.to_owned(),
        })
    }

    /// The declaration this intrinsic's call has, given its argument types.
    ///
    /// The shapes are LLVM's: the `with.overflow` family returns
    /// `{ iN, i1 }`, the bit-count family takes `(iN, i1)`, `fshl`/`fshr`
    /// take `(iN, iN, iN)`, `abs` takes `(iN, i1)`, and the min/max family
    /// take `(iN, iN)`. The memory intrinsics are variadic by nature and keep
    /// their pointer/iN shape from the first argument's spelling.
    pub fn get_declaration<'ctx>(
        &self,
        module: &Module<'ctx>,
        types: &[crate::types::BasicMetadataTypeEnum<'ctx>],
    ) -> Option<FunctionValue<'ctx>> {
        let ty = types.first()?.to_ir();
        let bits = ty.int_bits().unwrap_or(64);
        let i = biara_ir::Type::Int(bits);
        let b = biara_ir::Type::Int(1);
        let ir = match self.name.as_str() {
            "llvm.uadd.with.overflow"
            | "llvm.sadd.with.overflow"
            | "llvm.usub.with.overflow"
            | "llvm.ssub.with.overflow"
            | "llvm.umul.with.overflow"
            | "llvm.smul.with.overflow" => biara_ir::FuncType {
                ret: biara_ir::Type::Struct(vec![i.clone(), b]),
                params: vec![i.clone(), i.clone()],
                varargs: false,
            },
            "llvm.bswap" | "llvm.bitreverse" => biara_ir::FuncType {
                ret: i.clone(),
                params: vec![i.clone()],
                varargs: false,
            },
            "llvm.memcpy" | "llvm.memmove" | "llvm.memset" => biara_ir::FuncType {
                ret: biara_ir::Type::Void,
                params: types.iter().map(|t| t.to_ir()).collect(),
                varargs: false,
            },
            "llvm.fshl" | "llvm.fshr" => biara_ir::FuncType {
                ret: i.clone(),
                params: vec![i.clone(), i.clone(), i.clone()],
                varargs: false,
            },
            _ => biara_ir::FuncType {
                ret: i.clone(),
                params: vec![i.clone(), i.clone()],
                varargs: false,
            },
        };
        Some(module.intrinsic(&self.name, ir))
    }
}

