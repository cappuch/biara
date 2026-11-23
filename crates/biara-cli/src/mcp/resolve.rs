//! Turning what an agent types into an address.
//!
//! Every tool that takes an address or a function accepts the same forms: hexadecimal
//! (`0x…`), decimal, a symbol the binary defines, a name the user gave it, or a demangled
//! name. User names win over symbols, matching what the views display; the precedence is
//! documented here and the error names every form when nothing matches.

use super::Server;

impl Server {
    /// Resolve an address given as hex, decimal, symbol, user name or demangled name.
    pub(crate) fn resolve(&self, text: &str) -> Result<u64, String> {
        let text = text.trim();
        if text.is_empty() {
            return Err("an address or name cannot be empty".into());
        }
        if let Some(hex) = text.strip_prefix("0x").or_else(|| text.strip_prefix("0X")) {
            return u64::from_str_radix(hex, 16)
                .map_err(|_| format!("{text:?} is not a hexadecimal address"));
        }
        if text.bytes().all(|b| b.is_ascii_digit()) {
            return text
                .parse()
                .map_err(|_| format!("{text:?} is too large for an address"));
        }
        if let Some(addr) = self.lookup_name(text) {
            return Ok(addr);
        }
        Err(format!(
            "{text:?} is not an address or a known name. Use 0x…, decimal, a symbol, \
             a name you set, or a demangled name (list_functions lists the functions)"
        ))
    }

    /// The address a name stands for. Names the user wrote come first so a rename shadows
    /// the binary's symbol, as it does in the pseudocode.
    fn lookup_name(&self, name: &str) -> Option<u64> {
        let notes = self.notes.annotations();
        for (&addr, function) in &notes.functions {
            if function.name.as_deref() == Some(name) {
                return Some(addr);
            }
        }
        for (&addr, data) in &notes.globals {
            if data == name {
                return Some(addr);
            }
        }

        let program = self.program.as_ref()?;
        if let Some(symbol) = program.image.symbol(name) {
            return Some(symbol.addr);
        }
        for (&slot, import) in &program.image.imports {
            if import == name || name == format!("{import}@got") {
                return Some(slot);
            }
        }
        if let Some(function) = program.index.functions.iter().find(|f| f.name == name) {
            return Some(function.entry);
        }
        for function in &program.index.functions {
            if let Some(demangled) = program.index.demangled(function.entry) {
                if demangled.short == name || demangled.full == name {
                    return Some(function.entry);
                }
            }
        }
        for import in &program.index.imports {
            if let Some(demangled) = &import.demangled {
                if demangled.short == name || demangled.full == name {
                    return Some(import.slot);
                }
            }
        }
        for symbol in &program.image.symbols {
            if let Some(demangled) = program.index.demangled(symbol.addr) {
                if demangled.short == name || demangled.full == name {
                    return Some(symbol.addr);
                }
            }
        }
        None
    }
}
