//! Target descriptions. `biara-ir` emits portable C, so a target is just the
//! triple and data layout its analyses consult.

use std::cell::Cell;

/// A target triple.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TargetTriple(String);

impl TargetTriple {
    pub fn create(triple: &str) -> Self {
        Self(triple.to_owned())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn normalize(&self) -> Self {
        self.clone()
    }
}

impl std::fmt::Display for TargetTriple {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CodeModel {
    Default,
    Small,
    Kernel,
    Medium,
    Large,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RelocMode {
    Default,
    Static,
    PIC,
    DynamicNoPic,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct InitializationConfig;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FileType {
    Object,
    Assembly,
    Bitcode,
}

/// Registering native targets is a no-op: nothing here needs a code generator.
pub struct Target;

impl Target {
    pub fn initialize_x86(_config: &InitializationConfig) {}
    pub fn initialize_aarch64(_config: &InitializationConfig) {}
    pub fn initialize_native(_config: &InitializationConfig) -> Result<(), String> {
        Ok(())
    }

    /// Every triple is "supported": nothing here generates code.
    pub fn from_triple(_triple: &TargetTriple) -> Result<Target, String> {
        Ok(Target)
    }

    /// A machine for this target.
    pub fn create_target_machine(
        &self,
        triple: &TargetTriple,
        cpu: &str,
        features: &str,
        level: crate::OptimizationLevel,
        reloc: RelocMode,
        model: CodeModel,
    ) -> Option<TargetMachine> {
        TargetMachine::create(triple, cpu, features, level, reloc, model).ok()
    }
}

/// A machine description: a triple plus the pointer width it implies.
pub struct TargetMachine {
    triple: TargetTriple,
    pointer_bits: Cell<u32>,
    /// Textual layout of this target, as `get_target_data` reports it.
    layout: String,
}

impl TargetMachine {
    pub fn create(
        triple: &TargetTriple,
        _cpu: &str,
        _features: &str,
        _level: crate::OptimizationLevel,
        _reloc: RelocMode,
        _model: CodeModel,
    ) -> Result<Self, String> {
        let bits = if triple.as_str().contains("x86_64") || triple.as_str().contains("aarch64") {
            64
        } else {
            32
        };
        Ok(Self {
            layout: match bits {
                // The layout the analysis pipeline assumes for a 64-bit
                // little-endian target; every pointer is 64 bits and the
                // integer widths match the guest.
                64 => "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
                    .to_owned(),
                _ => "e-m:e-p:32:32-i64:64-n8:16:32-S128".to_owned(),
            },
            triple: triple.clone(),
            pointer_bits: Cell::new(bits),
        })
    }

    pub fn get_default_triple() -> TargetTriple {
        let arch = if cfg!(target_arch = "aarch64") {
            "aarch64"
        } else {
            "x86_64"
        };
        let os = if cfg!(target_os = "macos") {
            "apple-darwin"
        } else if cfg!(target_os = "windows") {
            "pc-windows-msvc"
        } else {
            "unknown-linux-gnu"
        };
        TargetTriple::create(&format!("{arch}-{os}"))
    }

    /// A stable identity for bridging to external analyses.
    pub fn as_mut_ptr(&self) -> *mut std::ffi::c_void {
        self as *const Self as *mut std::ffi::c_void
    }

    /// Create a machine for `target`.
    pub fn get_triple(&self) -> TargetTriple {
        self.triple.clone()
    }

    pub fn get_target_data(&self) -> TargetData {
        TargetData {
            pointer_bits: self.pointer_bits.get(),
            layout: self.layout.clone(),
        }
    }
}

/// Pointer widths and the textual layout of the machine's target.
pub struct TargetData {
    pointer_bits: u32,
    layout: String,
}

impl TargetData {
    pub fn get_pointer_byte_size(&self, _space: Option<&crate::AddressSpace>) -> u32 {
        self.pointer_bits / 8
    }

    pub fn get_pointer_size(&self, _space: Option<&crate::AddressSpace>) -> u32 {
        self.pointer_bits
    }

    pub fn get_data_layout(&self) -> crate::types::DataLayout {
        crate::types::DataLayout(self.layout.clone())
    }
}
