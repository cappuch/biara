//! User notes on rendered functions: local and parameter renames and address comments,
//! applied to each finished [`FunctionPackage`] so every consumer of decompiled text (the
//! `decomp` emitter, JSON records, the worker service) sees them.

use std::collections::{BTreeMap, BTreeSet};

use biara_proto::{
    FunctionPackage,
    annotations::Annotations,
    packages::place_comments,
    tokens::{SymbolRef, TokenKind, substitute},
};

use crate::tokens::{LexSymbol, lex};

/// The parts of [`Annotations`] applied after rendering.
#[derive(Default)]
pub(crate) struct Notes {
    locals: BTreeMap<u64, BTreeMap<String, String>>,
    comments: BTreeMap<u64, String>,
}

impl Notes {
    pub(crate) fn new(annotations: &Annotations) -> Notes {
        Notes {
            locals: annotations
                .functions
                .iter()
                .filter(|(_, f)| !f.locals.is_empty())
                .map(|(&a, f)| (a, f.locals.clone()))
                .collect(),
            comments: annotations.comments.clone(),
        }
    }

    /// Rename `package`'s locals and append its comments. `table` is the lexer table its
    /// tokens came from; tokens are lexed again when the body changes.
    pub(crate) fn apply(
        &self,
        package: &mut FunctionPackage,
        table: &[LexSymbol],
    ) -> Result<(), String> {
        let locals = self.locals.get(&package.address);
        let mut changed = false;
        let mut table = table.to_vec();
        if let Some(locals) = locals {
            self.rename(package, locals, &mut table)?;
            changed = true;
        }
        let signature_line = signature_line(package);
        let mut lines: Vec<(usize, &[u64])> = package
            .statements
            .iter()
            .filter_map(|s| {
                Some((
                    usize::try_from(s.line).ok()?.checked_sub(1)?,
                    &s.addresses[..],
                ))
            })
            .collect();
        lines.sort_by_key(|&(line, _)| line);
        let placed = place_comments(lines, package.address, signature_line, &self.comments);
        if !placed.is_empty() {
            let mut body: Vec<String> = package.body.split('\n').map(str::to_owned).collect();
            for (line, notes) in placed {
                let Some(text) = body.get_mut(line) else {
                    continue;
                };
                for (_, note) in notes {
                    text.push_str(" // ");
                    text.push_str(note);
                }
            }
            package.body = body.join("\n");
            changed = true;
        }
        if changed {
            package.tokens = lex(&package.body, &package.signature, &table);
            let lines: Vec<&str> = package.body.split('\n').collect();
            for statement in &mut package.statements {
                if let Some(line) = (statement.line as usize)
                    .checked_sub(1)
                    .and_then(|i| lines.get(i))
                {
                    statement.text = line.trim_start_matches(' ').to_owned();
                }
            }
        }
        Ok(())
    }

    /// Substitute `locals` (emitted name to display name) into the body, signature and name
    /// table, and point `table`'s local entries at the new spellings.
    fn rename(
        &self,
        package: &mut FunctionPackage,
        locals: &BTreeMap<String, String>,
        table: &mut [LexSymbol],
    ) -> Result<(), String> {
        let address = package.address;
        let emitted: BTreeSet<&str> = package.names.iter().map(|n| n.name.as_str()).collect();
        if let Some(missing) = locals.keys().find(|k| !emitted.contains(k.as_str())) {
            return Err(format!(
                "function {address:#x} has no local named {missing:?}"
            ));
        }
        // A new name may not be any identifier the body keeps: another local, a symbol, a
        // type or a field.
        let kept: BTreeSet<&str> = package
            .tokens
            .iter()
            .filter(|t| t.kind == TokenKind::Identifier)
            .filter(|t| !matches!(&t.symbol, Some(SymbolRef::Local { name }) if locals.contains_key(name)))
            .map(|t| &package.body[t.start as usize..t.end as usize])
            .collect();
        if let Some(name) = locals.values().find(|n| kept.contains(n.as_str())) {
            return Err(format!(
                "function {address:#x}: local name {name:?} is already used in its body"
            ));
        }
        let renames: BTreeMap<SymbolRef, String> = locals
            .iter()
            .map(|(e, n)| (SymbolRef::Local { name: e.clone() }, n.clone()))
            .collect();
        package.body = substitute(&package.body, &package.tokens, &renames)?;
        let signature_tokens = lex(&package.signature, &package.signature, table);
        package.signature = substitute(&package.signature, &signature_tokens, &renames)?;
        for name in &mut package.names {
            if let Some(new) = locals.get(&name.name) {
                name.name = new.clone();
            }
        }
        for entry in table {
            if let SymbolRef::Local { name } = &entry.symbol
                && let Some(new) = locals.get(name)
            {
                entry.text = new.clone();
            }
        }
        Ok(())
    }
}

/// The body line (0-based) defining the function: the first token naming its own entry.
pub(crate) fn signature_line(package: &FunctionPackage) -> Option<usize> {
    let token = package.tokens.iter().find(
        |t| matches!(t.symbol, Some(SymbolRef::Function { address }) if address == package.address),
    )?;
    Some(
        package.body.as_bytes()[..token.start as usize]
            .iter()
            .filter(|&&b| b == b'\n')
            .count(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use biara_proto::annotations::FunctionAnnotation;

    const BODY: &str = "\nuint64_t f(uint64_t arg0)\n{\n    uint64_t value0;\n\n    value0 = arg0 + 1;\n    return value0;\n}\n";

    fn package() -> (FunctionPackage, Vec<LexSymbol>) {
        let mut package: FunctionPackage = serde_json::from_value(serde_json::json!({
            "address": 0x1000, "name": "f", "signature": "uint64_t f(uint64_t arg0)",
            "body": BODY,
            "statements": [
                {"id": "function_1000/s0", "line": 4, "text": "uint64_t value0;", "addresses": [], "provenance": "unknown"},
                {"id": "function_1000/s1", "line": 6, "text": "value0 = arg0 + 1;", "addresses": [0x1004], "provenance": "exact"},
                {"id": "function_1000/s2", "line": 7, "text": "return value0;", "addresses": [0x1008], "provenance": "exact"},
            ],
            "dependencies": [],
            "names": [
                {"name": "arg0", "original": "rdi", "evidence": "unknown"},
                {"name": "value0", "original": "", "evidence": "unknown"},
            ],
            "metrics": {"tokens": 0, "temporaries": 0, "register_names": 0, "casts": 0, "gotos": 0,
                        "max_nesting": 0, "statements": 3, "mapped_statements": 2, "unsupported": 0},
        }))
        .unwrap();
        let mut table = vec![LexSymbol {
            text: "f".into(),
            symbol: SymbolRef::Function { address: 0x1000 },
        }];
        table.extend(["arg0", "value0"].map(|n| LexSymbol {
            text: n.into(),
            symbol: SymbolRef::Local { name: n.into() },
        }));
        package.tokens = lex(&package.body, &package.signature, &table);
        (package, table)
    }

    fn notes(locals: &[(&str, &str)], comments: &[(u64, &str)]) -> Notes {
        let mut annotations = Annotations::default();
        annotations.functions.insert(
            0x1000,
            FunctionAnnotation {
                locals: locals.iter().map(|&(e, n)| (e.into(), n.into())).collect(),
                ..Default::default()
            },
        );
        annotations.comments = comments.iter().map(|&(a, t)| (a, t.into())).collect();
        Notes::new(&annotations)
    }

    #[test]
    fn locals_and_comments_apply_to_body_signature_statements_and_tokens() {
        let (mut package, table) = package();
        notes(
            &[("arg0", "count"), ("value0", "next")],
            &[(0x1004, "bump"), (0x1000, "entry")],
        )
        .apply(&mut package, &table)
        .unwrap();
        assert_eq!(
            package.body,
            "\nuint64_t f(uint64_t count) // entry\n{\n    uint64_t next;\n\n    next = count + 1; // bump\n    return next;\n}\n"
        );
        assert_eq!(package.signature, "uint64_t f(uint64_t count)");
        assert_eq!(package.statements[1].text, "next = count + 1; // bump");
        assert_eq!(package.statements[2].text, "return next;");
        assert_eq!(package.names[0].name, "count");
        // Renamed tokens keep the emitted identity, so clients can rename again.
        let count = package
            .tokens
            .iter()
            .find(|t| &package.body[t.start as usize..t.end as usize] == "count")
            .unwrap();
        assert_eq!(
            count.symbol,
            Some(SymbolRef::Local {
                name: "arg0".into()
            })
        );
    }

    #[test]
    fn a_local_cannot_take_a_name_the_body_keeps_or_rename_what_is_not_there() {
        let (mut package, table) = package();
        let err = notes(&[("value0", "f")], &[]).apply(&mut package, &table);
        assert!(err.unwrap_err().contains("already used"));
        let err = notes(&[("value0", "arg0")], &[]).apply(&mut package, &table);
        assert!(err.unwrap_err().contains("already used"));
        let err = notes(&[("value9", "x")], &[]).apply(&mut package, &table);
        assert!(err.unwrap_err().contains("no local"));
        // Swapping two names is one simultaneous substitution.
        notes(&[("value0", "arg0"), ("arg0", "value0")], &[])
            .apply(&mut package, &table)
            .unwrap();
        assert!(package.body.contains("value0 + 1;"));
        assert!(package.body.contains("    arg0 = value0 + 1;"));
    }
}
