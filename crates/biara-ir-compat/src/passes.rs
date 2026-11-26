//! Pass manager handles. `biara-ir` takes a pipeline name instead, so these
//! exist only so inkwell-shaped signatures keep compiling.

pub use crate::support::PassBuilderOptions;
use crate::support::PassPipeline;
use crate::targets::TargetMachine;

/// A pass manager over a module.
#[derive(Clone, Copy, Default)]
pub struct PassManager;

impl PassManager {
    pub fn create() -> Self {
        Self
    }
}

/// A function pass manager.
#[derive(Clone, Copy, Default)]
pub struct PassManagerBuilder;

impl PassManagerBuilder {
    pub fn create() -> Self {
        Self
    }
}

/// Apply a pipeline to a module (inkwell's `run_passes` entry point).
pub fn run_passes(
    module: &crate::module::Module<'_>,
    pipeline: &PassPipeline,
    machine: &TargetMachine,
    options: PassBuilderOptions,
) -> Result<(), String> {
    module.run_passes(pipeline.name(), machine, options)
}
