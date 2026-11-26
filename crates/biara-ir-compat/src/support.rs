//! Small enums and markers the inkwell API exposed.

/// Address space placeholder; `biara-ir` has one uniform pointer type.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AddressSpace(pub u32);

impl AddressSpace {
    #[allow(clippy::should_implement_trait)]
    pub fn default() -> Self {
        Self(0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptimizationLevel {
    None,
    Less,
    Default,
    Aggressive,
}

impl OptimizationLevel {
    pub fn from_u8(level: u8) -> Self {
        match level {
            0 => Self::None,
            1 => Self::Less,
            2 => Self::Default,
            _ => Self::Aggressive,
        }
    }
}

/// Integer comparison predicates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntPredicate {
    EQ,
    NE,
    SLT,
    SLE,
    SGT,
    SGE,
    ULT,
    ULE,
    UGT,
    UGE,
}

impl IntPredicate {
    pub fn to_ir(self) -> biara_ir::IntPred {
        use biara_ir::IntPred as P;
        match self {
            Self::EQ => P::Eq,
            Self::NE => P::Ne,
            Self::SLT => P::Slt,
            Self::SLE => P::Sle,
            Self::SGT => P::Sgt,
            Self::SGE => P::Sge,
            Self::ULT => P::Ult,
            Self::ULE => P::Ule,
            Self::UGT => P::Ugt,
            Self::UGE => P::Uge,
        }
    }
}

/// Atomic orderings; `fence` is always sequentially consistent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AtomicOrdering {
    NotAtomic,
    SequentiallyConsistent,
}

/// A pass pipeline name.
#[derive(Clone, Debug, Default)]
pub struct PassPipeline(pub String);

impl PassPipeline {
    pub fn name(&self) -> &str {
        &self.0
    }
}

/// A pass pipeline. `biara-ir` understands the names biara emits.
#[derive(Clone, Debug, Default)]
pub struct PassBuilderOptions;

impl PassBuilderOptions {
    pub fn create() -> Self {
        Self
    }
}
