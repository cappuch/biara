//! Types the user declares and the layouts the decompiler recovers.
//!
//! The layout analysis infers a struct for each pointer a function dereferences at fixed
//! offsets (`struct s1 { uint64_t f_0; uint64_t f_8; }`). Those names are per decompile: `s1`
//! in one function is unrelated to `s1` in another. To name a struct and its fields for good,
//! declare it (the annotation format's `structs`) and point a function parameter at it with a
//! signature (`struct args *argv`): the engine then binds the declared struct to the layout
//! that parameter reaches, provided the declaration covers every recovered field by offset and
//! width. [`declare_from_layout`] and [`bind_param`] build both from what was recovered.

use std::{collections::BTreeMap, sync::Arc};

use biara_abi::FunctionAbi;
use biara_lift::LayoutFacts;
use biara_proto::annotations::{
    Annotations, FunctionAnnotation, StructAnnotation, StructField, is_identifier,
};

use super::Program;

/// What decompiling takes from the notes: the calling conventions with declared signatures
/// applied, and the declared structs to bind.
#[derive(Clone, Default)]
pub struct Types {
    pub abis: Arc<BTreeMap<u64, FunctionAbi>>,
    pub structs: BTreeMap<String, StructAnnotation>,
}

impl Types {
    /// Recovery's calling conventions, nothing declared.
    pub fn recovered(program: &Program) -> Types {
        Types {
            abis: program.abis.clone(),
            structs: BTreeMap::new(),
        }
    }

    /// `annotations`' signatures and structs over `program`. A signature the function's
    /// convention cannot carry, or one naming an undeclared struct, is an error naming it.
    pub fn declared(program: &Program, annotations: &Annotations) -> Result<Types, String> {
        let declared = Annotations {
            functions: annotations
                .functions
                .iter()
                .filter_map(|(&a, f)| {
                    let f = FunctionAnnotation {
                        signature: Some(f.signature.clone()?),
                        ..Default::default()
                    };
                    Some((a, f))
                })
                .collect(),
            structs: annotations.structs.clone(),
            ..Annotations::default()
        };
        if declared.functions.is_empty() && declared.structs.is_empty() {
            return Ok(Types::recovered(program));
        }
        let mut abis = (*program.abis).clone();
        let lib = program.lib.lock();
        biara_lift::annotations::apply_abi_annotations(
            &program.cfg,
            lib.info(),
            &mut abis,
            &declared,
        )
        .map_err(|e| e.to_string())?;
        Ok(Types {
            abis: Arc::new(abis),
            structs: declared.structs,
        })
    }
}

/// A struct one decompiled function uses.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StructLayout {
    /// Its C name without `struct `: recovered (`s1`, this decompile only) or declared.
    pub name: String,
    /// Whether a declared struct of this name bound it.
    pub declared: bool,
    /// Bytes from the start to the end of the last field accessed.
    pub size: u64,
    pub fields: Vec<LayoutField>,
    /// Parameters of the function (from 0) that point at its start.
    pub params: Vec<u32>,
}

/// One field of a [`StructLayout`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayoutField {
    pub offset: i64,
    pub bytes: u32,
    pub name: String,
    /// The declared C type, when a declaration gave one.
    pub type_name: Option<String>,
    /// The struct this field points to, when it is a pointer the analysis followed.
    pub pointee: Option<String>,
}

/// The structs `entry` uses, from the layout analysis of its decompile: those it accesses and
/// those their fields point to.
pub fn layouts_of(
    facts: &LayoutFacts,
    entry: u64,
    structs: &BTreeMap<String, StructAnnotation>,
) -> Vec<StructLayout> {
    let name_of = |id: u32| {
        facts
            .layouts
            .iter()
            .find(|l| l.id == id)
            .map(|l| l.name.strip_prefix("struct ").unwrap_or(&l.name).to_owned())
    };
    let mut wanted: Vec<u32> = facts
        .layouts
        .iter()
        .filter(|l| !l.scalar && l.functions.contains(&entry))
        .map(|l| l.id)
        .chain(
            facts
                .args
                .iter()
                .filter(|a| a.function == entry)
                .filter_map(|a| a.layout),
        )
        .collect();
    let mut i = 0;
    while i < wanted.len() {
        if let Some(l) = facts.layouts.iter().find(|l| l.id == wanted[i]) {
            wanted.extend(l.fields.iter().filter_map(|f| f.pointee));
        }
        i += 1;
        let mut seen = std::collections::BTreeSet::new();
        wanted.retain(|id| seen.insert(*id));
    }
    wanted
        .iter()
        .filter_map(|&id| {
            let l = facts.layouts.iter().find(|l| l.id == id)?;
            let name = name_of(id)?;
            Some(StructLayout {
                declared: structs.contains_key(&name),
                size: l.size,
                fields: l
                    .fields
                    .iter()
                    .map(|f| LayoutField {
                        offset: f.off,
                        bytes: f.bytes,
                        name: f.name.clone(),
                        type_name: (!f.asserted.is_empty()).then(|| f.asserted.clone()),
                        pointee: f.pointee.and_then(name_of),
                    })
                    .collect(),
                params: facts
                    .args
                    .iter()
                    .filter(|a| a.function == entry && a.layout == Some(id) && a.offset == 0)
                    .map(|a| a.arg)
                    .collect(),
                name,
            })
        })
        .collect()
}

/// A declaration covering every field of `layout`, keeping their names: what a struct needs
/// to bind to it.
pub fn declare_from_layout(layout: &StructLayout) -> StructAnnotation {
    StructAnnotation {
        fields: layout
            .fields
            .iter()
            .map(|f| StructField {
                offset: f.offset,
                bytes: f.bytes,
                name: f.name.clone(),
                type_name: f.type_name.clone().unwrap_or_default(),
            })
            .collect(),
    }
}

/// `signature` (`uint64_t f(uint64_t arg0, struct s1 * arg1)`) with parameter `param` typed
/// `struct <name> *`. Other pointers to structs that are not in `declared` become `void *`,
/// since a declared signature may only name declared structs.
pub fn bind_param(
    signature: &str,
    param: usize,
    name: &str,
    declared: &BTreeMap<String, StructAnnotation>,
) -> Result<String, String> {
    if !is_identifier(name) {
        return Err(format!("{name:?} is not a C identifier"));
    }
    let open = signature
        .find('(')
        .ok_or_else(|| format!("{signature:?} has no parameter list"))?;
    let close = signature
        .rfind(')')
        .filter(|&c| c > open)
        .ok_or_else(|| format!("{signature:?} has no parameter list"))?;
    let inner = signature[open + 1..close].trim();
    let mut params: Vec<String> = if inner.is_empty() || inner == "void" {
        Vec::new()
    } else {
        inner.split(',').map(|p| p.trim().to_owned()).collect()
    };
    if param >= params.len() {
        return Err(format!(
            "the signature has {} parameters; there is no parameter {param}",
            params.len()
        ));
    }
    for (i, p) in params.iter_mut().enumerate() {
        let (ty, ident) = split_param(p);
        let ty = if i == param {
            format!("struct {name} *")
        } else {
            match struct_pointer(&ty) {
                Some(s) if !declared.contains_key(s) => "void *".to_owned(),
                _ => ty,
            }
        };
        *p = match ident {
            Some(ident) => format!("{ty} {ident}").replace("* ", "*"),
            None => ty,
        };
    }
    Ok(format!(
        "{}({})",
        signature[..open].trim_end(),
        params.join(", ")
    ))
}

/// `ty` and the parameter's name, if it has one (`struct s1 * arg1` gives `struct s1 *` and
/// `arg1`).
fn split_param(p: &str) -> (String, Option<String>) {
    let p = p.trim();
    let start = p
        .rfind(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .map_or(0, |i| i + 1);
    let ident = &p[start..];
    let ty = p[..start].trim();
    // A lone word is a type (`uint64_t`), not a name.
    if ty.is_empty() || ident.is_empty() || ty.ends_with("struct") {
        (p.to_owned(), None)
    } else {
        (ty.to_owned(), Some(ident.to_owned()))
    }
}

/// `s1` for `struct s1 *`.
fn struct_pointer(ty: &str) -> Option<&str> {
    ty.strip_prefix("struct ")?
        .trim_end()
        .strip_suffix('*')
        .map(str::trim)
        .filter(|s| !s.contains(['*', ' ']))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binding_a_parameter_retypes_it_and_drops_undeclared_struct_pointers() {
        let declared = BTreeMap::from([("node".to_owned(), StructAnnotation::default())]);
        let sig =
            "uint64_t f(uint64_t arg0, struct s1 * arg1, struct s2 * arg2, struct node * arg3)";
        assert_eq!(
            bind_param(sig, 1, "args", &declared).unwrap(),
            "uint64_t f(uint64_t arg0, struct args *arg1, void *arg2, struct node *arg3)"
        );
        // Unnamed parameters, as in a declaration.
        assert_eq!(
            bind_param("void g(uint64_t, struct s1 *)", 1, "args", &declared).unwrap(),
            "void g(uint64_t, struct args *)"
        );
        assert!(bind_param(sig, 4, "args", &declared).is_err());
        assert!(bind_param(sig, 1, "two words", &declared).is_err());
    }
}
