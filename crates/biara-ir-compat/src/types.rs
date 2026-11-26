//! Type handles: a `biara_ir::Type` plus the context it belongs to.

use crate::Context;
use std::marker::PhantomData;

/// Anything usable as a value type.
pub trait BasicType<'ctx>: Clone {
    fn to_ir(&self) -> biara_ir::Type;
    fn get_context(&self) -> &'_ Context;
}

/// Anything usable as a type (including `void` and function types).
pub trait AnyType<'ctx>: Clone {
    fn to_any_ir(&self) -> biara_ir::Type;
}

macro_rules! int_type {
    ($name:ident) => {
        #[derive(Clone, Copy)]
        pub struct $name<'ctx> {
            pub(crate) bits: u32,
            pub(crate) ctx: &'ctx Context,
            _m: PhantomData<&'ctx ()>,
        }

        impl<'ctx> $name<'ctx> {
            pub(crate) fn new(ctx: &'ctx Context, bits: u32) -> Self {
                Self {
                    bits,
                    ctx,
                    _m: PhantomData,
                }
            }

            pub fn get_bit_width(&self) -> u32 {
                self.bits
            }

            pub fn get_context(&self) -> &'_ Context {
                self.ctx
            }

            /// A constant; constants are interned by the builder on use.
            pub fn const_int(&self, value: u64, sign_extend: bool) -> crate::values::IntValue<'ctx> {
                let v = if sign_extend {
                    value as i64 as i128
                } else {
                    value as i128
                };
                crate::values::IntValue::constant(self.ctx, v, self.bits)
            }

            pub fn const_zero(&self) -> crate::values::IntValue<'ctx> {
                crate::values::IntValue::constant(self.ctx, 0, self.bits)
            }

            pub fn const_all_ones(&self) -> crate::values::IntValue<'ctx> {
                crate::values::IntValue::all_ones(self.ctx, self.bits)
            }

            pub fn const_unsigned_max(&self) -> crate::values::IntValue<'ctx> {
                self.const_all_ones()
            }

            /// An array `[len x self]`.
            pub fn array_type(&self, len: u32) -> crate::types::ArrayType<'ctx> {
                ArrayType::new(
                    self.ctx,
                    biara_ir::Type::Array(Box::new(biara_ir::Type::Int(self.bits)), len as u64),
                )
            }

            pub fn vec_type(&self, len: u32) -> crate::types::VectorType<'ctx> {
                VectorType::new(
                    self.ctx,
                    biara_ir::Type::Vector(Box::new(biara_ir::Type::Int(self.bits)), len),
                )
            }

            /// A function signature returning this type.
            pub fn fn_type(
                &self,
                params: &[BasicMetadataTypeEnum<'ctx>],
                varargs: bool,
            ) -> FunctionType<'ctx> {
                FunctionType::new(
                    self.ctx,
                    biara_ir::Type::Int(self.bits),
                    params,
                    varargs,
                )
            }

            pub fn as_basic_type_enum(&self) -> BasicTypeEnum<'ctx> {
                BasicTypeEnum::IntType(*self)
            }
        }

        impl<'ctx> BasicType<'ctx> for $name<'ctx> {
            fn to_ir(&self) -> biara_ir::Type {
                biara_ir::Type::Int(self.bits)
            }
            fn get_context(&self) -> &'_ Context {
                self.ctx
            }
        }

        impl<'ctx> AnyType<'ctx> for $name<'ctx> {
            fn to_any_ir(&self) -> biara_ir::Type {
                biara_ir::Type::Int(self.bits)
            }
        }
    };
}

int_type!(IntType);

#[derive(Clone, Copy)]
pub struct VoidType<'ctx> {
    pub(crate) ctx: &'ctx Context,
    pub(crate) _m: PhantomData<&'ctx ()>,
}

impl<'ctx> VoidType<'ctx> {
    pub(crate) fn new(ctx: &'ctx Context) -> Self {
        Self {
            ctx,
            _m: PhantomData,
        }
    }

    pub fn fn_type(
        &self,
        params: &[BasicMetadataTypeEnum<'ctx>],
        varargs: bool,
    ) -> FunctionType<'ctx> {
        FunctionType::new(self.ctx, biara_ir::Type::Void, params, varargs)
    }

    pub fn get_context(&self) -> &'_ Context {
        self.ctx
    }
}

impl<'ctx> AnyType<'ctx> for VoidType<'ctx> {
    fn to_any_ir(&self) -> biara_ir::Type {
        biara_ir::Type::Void
    }
}

#[derive(Clone, Copy)]
pub struct FloatType<'ctx> {
    pub(crate) bits: u32,
    pub(crate) ctx: &'ctx Context,
    pub(crate) _m: PhantomData<&'ctx ()>,
}

impl<'ctx> FloatType<'ctx> {
    pub(crate) fn new(ctx: &'ctx Context, bits: u32) -> Self {
        Self {
            bits,
            ctx,
            _m: PhantomData,
        }
    }

    pub fn get_bit_width(&self) -> u32 {
        self.bits
    }

    pub fn fn_type(
        &self,
        params: &[BasicMetadataTypeEnum<'ctx>],
        varargs: bool,
    ) -> FunctionType<'ctx> {
        FunctionType::new(
            self.ctx,
            biara_ir::Type::Struct(Vec::new()),
            params,
            varargs,
        )
    }
}

impl<'ctx> BasicType<'ctx> for FloatType<'ctx> {
    fn to_ir(&self) -> biara_ir::Type {
        biara_ir::Type::Struct(Vec::new())
    }
    fn get_context(&self) -> &'_ Context {
        self.ctx
    }
}

impl<'ctx> AnyType<'ctx> for FloatType<'ctx> {
    fn to_any_ir(&self) -> biara_ir::Type {
        biara_ir::Type::Struct(Vec::new())
    }
}

#[derive(Clone, Copy)]
pub struct PointerType<'ctx> {
    pub(crate) ctx: &'ctx Context,
    pub(crate) _m: PhantomData<&'ctx ()>,
}

impl<'ctx> PointerType<'ctx> {
    pub(crate) fn new(ctx: &'ctx Context) -> Self {
        Self {
            ctx,
            _m: PhantomData,
        }
    }

    pub fn fn_type(
        &self,
        params: &[BasicMetadataTypeEnum<'ctx>],
        varargs: bool,
    ) -> FunctionType<'ctx> {
        FunctionType::new(self.ctx, biara_ir::Type::Ptr, params, varargs)
    }

    pub fn get_context(&self) -> &'_ Context {
        self.ctx
    }
}

impl<'ctx> BasicType<'ctx> for PointerType<'ctx> {
    fn to_ir(&self) -> biara_ir::Type {
        biara_ir::Type::Ptr
    }
    fn get_context(&self) -> &'_ Context {
        self.ctx
    }
}

impl<'ctx> AnyType<'ctx> for PointerType<'ctx> {
    fn to_any_ir(&self) -> biara_ir::Type {
        biara_ir::Type::Ptr
    }
}

/// A struct or array type.
#[derive(Clone, Copy)]
pub struct StructType<'ctx> {
    pub(crate) ir: &'ctx biara_ir::Type,
    pub(crate) ctx: &'ctx Context,
    pub(crate) _m: PhantomData<&'ctx ()>,
}

impl<'ctx> StructType<'ctx> {
    pub(crate) fn new(ctx: &'ctx Context, ir: biara_ir::Type) -> Self {
        Self {
            ir: ctx.intern(ir),
            ctx,
            _m: PhantomData,
        }
    }

    /// Array length, when this is an array.
    pub fn len(&self) -> u32 {
        match self.ir {
            biara_ir::Type::Array(_, n) => *n as u32,
            _ => 0,
        }
    }

    pub fn count_fields(&self) -> u32 {
        match self.ir {
            biara_ir::Type::Struct(f) => f.len() as u32,
            _ => 0,
        }
    }

    pub fn get_field_type_at_index(&self, i: u32) -> Option<BasicTypeEnum<'ctx>> {
        match self.ir {
            biara_ir::Type::Struct(f) => f
                .get(i as usize)
                .and_then(|t| BasicTypeEnum::from_ir(self.ctx, t)),
            _ => None,
        }
    }

    pub fn as_basic_type_enum(&self) -> BasicTypeEnum<'ctx> {
        BasicTypeEnum::StructType(*self)
    }

    pub fn get_context(&self) -> &'_ Context {
        self.ctx
    }
}

impl<'ctx> BasicType<'ctx> for StructType<'ctx> {
    fn to_ir(&self) -> biara_ir::Type {
        self.ir.clone()
    }
    fn get_context(&self) -> &'_ Context {
        self.ctx
    }
}

impl<'ctx> AnyType<'ctx> for StructType<'ctx> {
    fn to_any_ir(&self) -> biara_ir::Type {
        self.ir.clone()
    }
}

/// An array type.
pub type ArrayType<'ctx> = StructType<'ctx>;

#[derive(Clone, Copy)]
pub struct VectorType<'ctx> {
    pub(crate) ir: &'ctx biara_ir::Type,
    pub(crate) ctx: &'ctx Context,
    pub(crate) _m: PhantomData<&'ctx ()>,
}

impl<'ctx> VectorType<'ctx> {
    pub(crate) fn new(ctx: &'ctx Context, ir: biara_ir::Type) -> Self {
        Self {
            ir: ctx.intern(ir),
            ctx,
            _m: PhantomData,
        }
    }

    pub fn get_size(&self) -> u32 {
        match self.ir {
            biara_ir::Type::Vector(_, n) => *n,
            _ => 0,
        }
    }

    pub fn get_element_type(&self) -> BasicTypeEnum<'ctx> {
        match self.ir {
            biara_ir::Type::Vector(e, _) => {
                BasicTypeEnum::from_ir(self.ctx, e).unwrap_or(BasicTypeEnum::IntType(IntType::new(self.ctx, 8)))
            }
            _ => BasicTypeEnum::IntType(IntType::new(self.ctx, 8)),
        }
    }

    pub fn get_context(&self) -> &'_ Context {
        self.ctx
    }
}

impl<'ctx> BasicType<'ctx> for VectorType<'ctx> {
    fn to_ir(&self) -> biara_ir::Type {
        self.ir.clone()
    }
    fn get_context(&self) -> &'_ Context {
        self.ctx
    }
}

impl<'ctx> AnyType<'ctx> for VectorType<'ctx> {
    fn to_any_ir(&self) -> biara_ir::Type {
        self.ir.clone()
    }
}

/// A function signature.
#[derive(Clone, Copy)]
pub struct FunctionType<'ctx> {
    pub(crate) ir: &'ctx biara_ir::FuncType,
    pub(crate) ctx: &'ctx Context,
    pub(crate) _m: PhantomData<&'ctx ()>,
}

impl<'ctx> FunctionType<'ctx> {
    pub(crate) fn new(
        ctx: &'ctx Context,
        ret: biara_ir::Type,
        params: &[BasicMetadataTypeEnum<'ctx>],
        varargs: bool,
    ) -> Self {
        let ty = biara_ir::Type::Func(Box::new(biara_ir::FuncType {
            ret,
            params: params.iter().map(|p| p.to_ir()).collect(),
            varargs,
        }));
        let interred = ctx.intern(ty);
        let ir = match interred {
            biara_ir::Type::Func(f) => &**f,
            _ => unreachable!(),
        };
        Self {
            ir,
            ctx,
            _m: PhantomData,
        }
    }

    /// The return type, or `None` when the function returns `void` — the
    /// shape inkwell call sites use with `map_or`.
    pub fn get_return_type(&self) -> Option<BasicTypeEnum<'ctx>> {
        BasicTypeEnum::from_ir(self.ctx, &self.ir.ret)
    }

    pub fn get_param_types(&self) -> Vec<BasicMetadataTypeEnum<'ctx>> {
        self.ir
            .params
            .iter()
            .map(|t| match t {
                biara_ir::Type::Ptr => BasicMetadataTypeEnum::PointerType(PointerType::new(self.ctx)),
                biara_ir::Type::Int(b) => BasicMetadataTypeEnum::IntType(IntType::new(self.ctx, *b)),
                other => BasicMetadataTypeEnum::Type(
                    BasicTypeEnum::from_ir(self.ctx, other)
                        .unwrap_or(BasicTypeEnum::IntType(IntType::new(self.ctx, 8))),
                ),
            })
            .collect()
    }

    pub fn count_param_types(&self) -> u32 {
        self.ir.params.len() as u32
    }

    pub fn get_context(&self) -> &'_ Context {
        self.ctx
    }
}

impl<'ctx> BasicType<'ctx> for FunctionType<'ctx> {
    fn to_ir(&self) -> biara_ir::Type {
        biara_ir::Type::Func(Box::new(self.ir.clone()))
    }
    fn get_context(&self) -> &'_ Context {
        self.ctx
    }
}

impl<'ctx> AnyType<'ctx> for FunctionType<'ctx> {
    fn to_any_ir(&self) -> biara_ir::Type {
        biara_ir::Type::Func(Box::new(self.ir.clone()))
    }
}

/// Any type, including `void` and signatures.
#[derive(Clone, Copy)]
pub enum AnyTypeEnum<'ctx> {
    IntType(IntType<'ctx>),
    PointerType(PointerType<'ctx>),
    FloatType(FloatType<'ctx>),
    VoidType(VoidType<'ctx>),
    StructType(StructType<'ctx>),
    VectorType(VectorType<'ctx>),
    FunctionType(FunctionType<'ctx>),
}

impl<'ctx> AnyTypeEnum<'ctx> {
    pub fn from_ir(ctx: &'ctx Context, ty: &biara_ir::Type) -> Self {
        match ty {
            biara_ir::Type::Int(b) => Self::IntType(IntType::new(ctx, *b)),
            biara_ir::Type::Ptr => Self::PointerType(PointerType::new(ctx)),
            biara_ir::Type::Void => Self::VoidType(VoidType::new(ctx)),
            biara_ir::Type::Func(_) => {
                let interred = ctx.intern(ty.clone());
                let ir = match interred {
                    biara_ir::Type::Func(f) => &**f,
                    _ => unreachable!(),
                };
                Self::FunctionType(FunctionType {
                    ir,
                    ctx,
                    _m: PhantomData,
                })
            }
            biara_ir::Type::Vector(..) => Self::VectorType(VectorType::new(ctx, ty.clone())),
            _ => Self::StructType(StructType::new(ctx, ty.clone())),
        }
    }

    /// `Option::map_or` shape for call sites that only care about real types.
    pub fn map_or<U>(
        &self,
        default: U,
        f: impl FnOnce(AnyTypeEnum<'ctx>) -> U,
    ) -> U {
        match self {
            Self::VoidType(_) | Self::FunctionType(_) => default,
            other => f(*other),
        }
    }

    pub fn as_basic_type_enum(&self) -> Option<BasicTypeEnum<'ctx>> {
        Some(match self {
            Self::IntType(t) => BasicTypeEnum::IntType(*t),
            Self::PointerType(t) => BasicTypeEnum::PointerType(*t),
            Self::FloatType(t) => BasicTypeEnum::FloatType(*t),
            Self::StructType(t) => BasicTypeEnum::StructType(*t),
            Self::VectorType(t) => BasicTypeEnum::VectorType(*t),
            Self::VoidType(_) | Self::FunctionType(_) => return None,
        })
    }
}

impl<'ctx> AnyType<'ctx> for AnyTypeEnum<'ctx> {
    fn to_any_ir(&self) -> biara_ir::Type {
        match self {
            Self::IntType(t) => t.to_ir(),
            Self::PointerType(t) => t.to_ir(),
            Self::FloatType(t) => t.to_ir(),
            Self::VoidType(t) => t.to_any_ir(),
            Self::StructType(t) => t.to_ir(),
            Self::VectorType(t) => t.to_ir(),
            Self::FunctionType(t) => t.to_ir(),
        }
    }
}

/// The type enum builder methods accept.
#[derive(Clone, Copy)]
pub enum BasicTypeEnum<'ctx> {
    IntType(IntType<'ctx>),
    PointerType(PointerType<'ctx>),
    FloatType(FloatType<'ctx>),
    StructType(StructType<'ctx>),
    VectorType(VectorType<'ctx>),
    /// Arrays share `StructType`'s representation; the variant records intent.
    ArrayType(StructType<'ctx>),
}

impl<'ctx> BasicTypeEnum<'ctx> {
    pub(crate) fn from_ir(ctx: &'ctx Context, ty: &biara_ir::Type) -> Option<Self> {
        Some(match ty {
            biara_ir::Type::Int(b) => Self::IntType(IntType::new(ctx, *b)),
            biara_ir::Type::Ptr => Self::PointerType(PointerType::new(ctx)),
            biara_ir::Type::Vector(..) => Self::VectorType(VectorType::new(ctx, ty.clone())),
            biara_ir::Type::Array(..) => Self::ArrayType(StructType::new(ctx, ty.clone())),
            biara_ir::Type::Struct(_) => Self::StructType(StructType::new(ctx, ty.clone())),
            _ => return None,
        })
    }

    pub fn as_any_type_enum(&self) -> AnyTypeEnum<'ctx> {
        match self {
            Self::IntType(t) => AnyTypeEnum::IntType(*t),
            Self::PointerType(t) => AnyTypeEnum::PointerType(*t),
            Self::FloatType(t) => AnyTypeEnum::FloatType(*t),
            Self::StructType(t) => AnyTypeEnum::StructType(*t),
            Self::VectorType(t) => AnyTypeEnum::VectorType(*t),
            Self::ArrayType(t) => AnyTypeEnum::StructType(*t),
        }
    }

    pub fn get_context(&self) -> &'_ Context {
        self.get_context_ref()
    }

    /// A signature returning this type.
    pub fn fn_type(
        &self,
        params: &[BasicMetadataTypeEnum<'ctx>],
        varargs: bool,
    ) -> FunctionType<'ctx> {
        FunctionType::new(self.get_context_ref(), self.to_ir(), params, varargs)
    }

    fn get_context_ref(&self) -> &'ctx Context {
        match self {
            Self::IntType(t) => t.ctx,
            Self::PointerType(t) => t.ctx,
            Self::FloatType(t) => t.ctx,
            Self::StructType(t) => t.ctx,
            Self::VectorType(t) => t.ctx,
            Self::ArrayType(t) => t.ctx,
        }
    }
}

impl<'ctx> BasicType<'ctx> for BasicTypeEnum<'ctx> {
    fn to_ir(&self) -> biara_ir::Type {
        match self {
            Self::IntType(t) => t.to_ir(),
            Self::PointerType(t) => t.to_ir(),
            Self::FloatType(t) => t.to_ir(),
            Self::StructType(t) => t.to_ir(),
            Self::VectorType(t) => t.to_ir(),
            Self::ArrayType(t) => t.to_ir(),
        }
    }
    fn get_context(&self) -> &'_ Context {
        self.get_context_ref()
    }
}

impl<'ctx> AnyType<'ctx> for BasicTypeEnum<'ctx> {
    fn to_any_ir(&self) -> biara_ir::Type {
        self.to_ir()
    }
}

/// A parameter type in a signature.
#[derive(Clone, Copy)]
pub enum BasicMetadataTypeEnum<'ctx> {
    IntType(IntType<'ctx>),
    PointerType(PointerType<'ctx>),
    FloatType(FloatType<'ctx>),
    Type(BasicTypeEnum<'ctx>),
}

impl<'ctx> BasicType<'ctx> for BasicMetadataTypeEnum<'ctx> {
    fn to_ir(&self) -> biara_ir::Type {
        match self {
            Self::IntType(t) => t.to_ir(),
            Self::PointerType(t) => t.to_ir(),
            Self::FloatType(t) => t.to_ir(),
            Self::Type(t) => t.to_ir(),
        }
    }
    fn get_context(&self) -> &'_ Context {
        match self {
            Self::IntType(t) => t.ctx,
            Self::PointerType(t) => t.ctx,
            Self::FloatType(t) => t.ctx,
            Self::Type(t) => t.get_context(),
        }
    }
}

impl<'ctx> From<IntType<'ctx>> for BasicMetadataTypeEnum<'ctx> {
    fn from(t: IntType<'ctx>) -> Self {
        Self::IntType(t)
    }
}

impl<'ctx> From<PointerType<'ctx>> for BasicMetadataTypeEnum<'ctx> {
    fn from(t: PointerType<'ctx>) -> Self {
        Self::PointerType(t)
    }
}

impl<'ctx> From<BasicTypeEnum<'ctx>> for BasicMetadataTypeEnum<'ctx> {
    fn from(t: BasicTypeEnum<'ctx>) -> Self {
        Self::Type(t)
    }
}

impl<'ctx> From<IntType<'ctx>> for BasicTypeEnum<'ctx> {
    fn from(t: IntType<'ctx>) -> Self {
        Self::IntType(t)
    }
}

impl<'ctx> From<PointerType<'ctx>> for BasicTypeEnum<'ctx> {
    fn from(t: PointerType<'ctx>) -> Self {
        Self::PointerType(t)
    }
}

impl<'ctx> From<StructType<'ctx>> for BasicTypeEnum<'ctx> {
    fn from(t: StructType<'ctx>) -> Self {
        Self::StructType(t)
    }
}

impl<'ctx> From<FunctionType<'ctx>> for BasicTypeEnum<'ctx> {
    fn from(_t: FunctionType<'ctx>) -> Self {
        unreachable!("a function type is not a value type")
    }
}

impl<'ctx> From<VectorType<'ctx>> for BasicTypeEnum<'ctx> {
    fn from(t: VectorType<'ctx>) -> Self {
        Self::VectorType(t)
    }
}

/// A data layout string.
#[derive(Clone, Debug)]
pub struct DataLayout(pub String);

impl DataLayout {
    /// The layout string (inkwell returns a `StringRef` here).
    pub fn as_str(&self) -> crate::values::StringRef {
        crate::values::StringRef(self.0.as_str())
    }

    /// The pointer width the layout declares, in bits.
    pub fn get_pointer_size(&self, _space: Option<&crate::AddressSpace>) -> u32 {
        if self.0.is_empty() {
            if cfg!(target_pointer_width = "64") { 64 } else { 32 }
        } else {
            64
        }
    }

    pub fn get_pointer_byte_size(&self, space: Option<&crate::AddressSpace>) -> u32 {
        self.get_pointer_size(space) / 8
    }
}
