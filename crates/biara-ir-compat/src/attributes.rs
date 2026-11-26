//! Function and parameter attributes.

/// Where an attribute attaches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttributeLoc {
    Function,
    Param(u32),
}

/// A named enum attribute (`noalias`, `captures`, `memory`, `noreturn`).
/// `Copy`, as inkwell's was: the name is a shared `&'static str`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Attribute {
    pub(crate) kind: &'static str,
    pub(crate) value: u64,
}

impl Attribute {
    /// LLVM's `get_named_enum_kind_id` is just the attribute's name.
    pub fn get_named_enum_kind_id(name: &str) -> &'static str {
        // Attribute names are a fixed, small set.
        match name {
            "noalias" => "noalias",
            "captures" => "captures",
            "memory" => "memory",
            "noreturn" => "noreturn",
            "target-features" => "target-features",
            other => Box::leak(other.to_owned().into_boxed_str()),
        }
    }

    /// The attribute's enum value (bit flags for `memory`).
    pub fn get_enum_value(&self) -> u64 {
        self.value
    }

    pub fn get_string_value(&self) -> Option<crate::values::StringRef> {
        None
    }
}
