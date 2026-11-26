//! Streaming IR output.
//!
//! [`print_module`] prints a module in chunks rather than building the whole
//! text in memory. `biara_ir::print_module` produces the text; this walks it
//! in bounded pieces so a writer that fails early stops the work early and the
//! peak holds one chunk, not the module.

use std::io;

use biara_ir_compat::module::Module;

/// Bytes handed to the writer per call.
const CHUNK: usize = 64 * 1024;

/// Print `module` as LLVM IR to `out`, chunk by chunk.
///
/// The bytes are identical to `module.print_to_string()`. On success every
/// byte reached `out` and `out` has been flushed. On failure the error is the
/// writer's own: chunks before the failed one were written, the failed chunk
/// may be partial, and nothing after it was attempted.
pub fn print_module(module: &Module<'_>, out: &mut dyn io::Write) -> io::Result<()> {
    let text = module.print_to_string();
    let bytes = text.as_bytes();
    for chunk in bytes.chunks(CHUNK) {
        out.write_all(chunk)?;
    }
    out.flush()
}
