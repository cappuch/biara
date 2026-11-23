//! The tools `biara mcp` exposes, and the text each returns.
//!
//! The text is the agent's whole interface: compact, one fact per line, every address in
//! hexadecimal, and a paging footer on every list that says the total and the next offset.
//! Tool errors (a bad argument, a name that cannot be used) come back as results with
//! `isError: true`; only protocol faults are JSON-RPC errors. "Not ready yet" is deliberately
//! not an error: it carries the recovery stage and tells the agent to call again.

use std::{collections::BTreeMap, fmt::Write as _, ops::Range};

use biara_proto::{
    annotations::{StructAnnotation, StructField},
    tokens::{SymbolRef, TokenKind},
};
use biara_session::{
    FunctionKind, NameKind, Notes, Program, StructLayout, XrefKind, disasm,
    types::{LayoutField, bind_param, declare_from_layout},
};
use serde_json::{Value, json};

use super::{DEFAULT_WAIT, Failure, MAX_WAIT, Server, hex};

/// What one tool call returns.
pub struct ToolResult {
    pub text: String,
    pub is_error: bool,
}

/// The names, descriptions and input schemas of every tool, in a fixed order so clients can
/// cache the list and LLM prompts stay stable.
pub fn list_json() -> Value {
    json!({"tools": [
        {
            "name": "status",
            "description": "Report the open binary (path, architecture, format, entry point), \
                the analysis stage and progress, function/string/import counts, and where the \
                notes file is saved. Never waits; call it first and after `open`.",
            "inputSchema": {"type": "object", "properties": {}, "additionalProperties": false}
        },
        {
            "name": "open",
            "description": "Open another binary, replacing the current one. Recovery starts in \
                the background; call `status` or any tool that needs the program, which waits. \
                The path must be a file biara can read.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the executable or library to analyse."}
                },
                "required": ["path"],
                "additionalProperties": false
            }
        },
        {
            "name": "list_functions",
            "description": "List recovered functions in address order: entry, display name \
                (your name if one was set, otherwise the symbol), size in bytes, basic blocks, \
                and caller and call counts. `filter` is a case-insensitive substring of the \
                display name, symbol, demangled name or hexadecimal entry. Paged: the last \
                line gives the total and the next offset.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "filter": {"type": "string", "description": "Only functions whose display name, symbol, demangled name or hex entry contains this text (case-insensitive)."},
                    "offset": page_offset(),
                    "limit": page_limit(100),
                    "wait_seconds": wait_property()
                },
                "additionalProperties": false
            }
        },
        {
            "name": "list_imports",
            "description": "List imported symbols by the address of the pointer the dynamic \
                linker fills (the GOT or IAT slot), with the readable form of a mangled name. \
                Paged: the last line gives the total and the next offset.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "offset": page_offset(),
                    "limit": page_limit(100),
                    "wait_seconds": wait_property()
                },
                "additionalProperties": false
            }
        },
        {
            "name": "list_strings",
            "description": "List ASCII and UTF-16 strings found in the image's data segments. \
                `filter` is a case-insensitive substring of the text (this is also the string \
                search; there is no separate search tool). Paged: the last line gives the \
                total and the next offset.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "filter": {"type": "string", "description": "Only strings whose text contains this text (case-insensitive)."},
                    "offset": page_offset(),
                    "limit": page_limit(100),
                    "wait_seconds": wait_property()
                },
                "additionalProperties": false
            }
        },
        {
            "name": "list_segments",
            "description": "List mapped segments with address, size and permissions. \
                Paged: the last line gives the total and the next offset.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "offset": page_offset(),
                    "limit": page_limit(50),
                    "wait_seconds": wait_property()
                },
                "additionalProperties": false
            }
        },
        {
            "name": "list_structs",
            "description": "List the persistent struct declarations for this binary with \
                their fields. A declaration binds to code only where a function signature \
                types a parameter `struct name *`; recovered layouts are per-decompile \
                (struct_layouts). Paged: the last line gives the total and the next offset.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "offset": page_offset(),
                    "limit": page_limit(50)
                },
                "additionalProperties": false
            }
        },
        {
            "name": "function_info",
            "description": "Everything known about one function: its names (display, symbol, \
                demangled), entry, size, blocks, stack frame, argument count, declared \
                signature, renamed locals, comments inside its blocks, and its callers and \
                callees with names. Use xrefs_to or xrefs_from for complete reference lists.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "function": function_property(),
                    "wait_seconds": wait_property()
                },
                "required": ["function"],
                "additionalProperties": false
            }
        },
        {
            "name": "struct_layouts",
            "description": "List the structs a function's pointers reach in its \
                decompilation: recovered or declared name, size, the parameters (0-based) \
                pointing at the struct, and each field's offset, width, name, declared type \
                and pointee struct. Recovered names (s1, f_8) live only for this decompile; \
                use struct_from_param to declare one permanently and bind it.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "function": function_property(),
                    "wait_seconds": wait_property()
                },
                "required": ["function"],
                "additionalProperties": false
            }
        },
        {
            "name": "decompile",
            "description": "Render a function's pseudocode with every name applied (functions, \
                global data and locals) and your comments appended as `// text`. Each line is \
                prefixed with its 0-based line number; with `addresses`, also its first machine \
                address. Paged for large functions: the first line says `lines a-b of N` and \
                the last gives the next offset.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "function": function_property(),
                    "offset": {"type": "integer", "minimum": 0, "default": 0, "description": "First line to return (0-based)."},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 2000, "default": 400, "description": "Maximum lines to return."},
                    "addresses": {"type": "boolean", "default": false, "description": "Prefix each line with the first machine address mapped to it."},
                    "wait_seconds": wait_property()
                },
                "required": ["function"],
                "additionalProperties": false
            }
        },
        {
            "name": "disassemble",
            "description": "Disassemble `count` instructions from an address or function \
                entry. Direct branch and data targets get the symbol or user name that names \
                them, and comments at an instruction are appended. The last line gives where \
                to continue.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "address_or_function": {"type": ["string", "integer"], "description": "Start address (0x…, decimal, symbol, user name or demangled name) or a function name."},
                    "count": {"type": "integer", "minimum": 1, "maximum": 500, "default": 20, "description": "Instructions to disassemble."},
                    "wait_seconds": wait_property()
                },
                "required": ["address_or_function"],
                "additionalProperties": false
            }
        },
        {
            "name": "xrefs_to",
            "description": "List references to an address (calls, jumps and data references) \
                with the referring instruction and the function containing it. Paged: the last \
                line gives the total and the next offset.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "address": {"type": ["string", "integer"], "description": "Target address (0x…, decimal, symbol, user name or demangled name)."},
                    "offset": page_offset(),
                    "limit": page_limit(100),
                    "wait_seconds": wait_property()
                },
                "required": ["address"],
                "additionalProperties": false
            }
        },
        {
            "name": "xrefs_from",
            "description": "List references made by one instruction (an instruction address) \
                or by every instruction of a function (a function name or entry). Each line \
                gives the kind, the target, the referring instruction and the containing \
                function. Paged: the last line gives the total and the next offset.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "address_or_function": {"type": ["string", "integer"], "description": "An instruction address, or a function name or entry to list every reference the function makes."},
                    "offset": page_offset(),
                    "limit": page_limit(100),
                    "wait_seconds": wait_property()
                },
                "required": ["address_or_function"],
                "additionalProperties": false
            }
        },
        {
            "name": "read_bytes",
            "description": "Hex and ASCII dump of mapped bytes at an address. Fails when any \
                requested byte is unmapped. At most 4096 bytes.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "address": {"type": ["string", "integer"], "description": "Address to read (0x…, decimal, symbol, user name or demangled name)."},
                    "length": {"type": "integer", "minimum": 1, "maximum": 4096, "default": 64, "description": "Bytes to read."},
                    "wait_seconds": wait_property()
                },
                "required": ["address"],
                "additionalProperties": false
            }
        },
        {
            "name": "search_bytes",
            "description": "Search the file-backed bytes of every mapped segment for a byte \
                pattern: space-separated hexadecimal bytes, with `??` for one wildcard byte, \
                for example `48 8b ?? 05`. Each hit shows its address, the matched bytes and \
                the string starting there, if any. Paged: the last line gives the total and \
                the next offset.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Space-separated hex bytes; `??` matches any byte."},
                    "offset": page_offset(),
                    "limit": page_limit(50),
                    "wait_seconds": wait_property()
                },
                "required": ["pattern"],
                "additionalProperties": false
            }
        },
        {
            "name": "rename_function",
            "description": "Give a function a display name, or clear it with an empty name so \
                its symbol shows again. The name must be a C identifier that is not a keyword. \
                Names apply to later listings, decompiles and disassembly without re-analysis.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "function": function_property(),
                    "name": {"type": "string", "description": "New name; empty clears it."},
                    "wait_seconds": wait_property()
                },
                "required": ["function", "name"],
                "additionalProperties": false
            }
        },
        {
            "name": "rename_data",
            "description": "Give the data at an address a display name, or clear it with an \
                empty name. Use rename_function for code. The name must be a C identifier that \
                is not a keyword.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "address": {"type": ["string", "integer"], "description": "Data address (0x…, decimal, symbol, user name or demangled name)."},
                    "name": {"type": "string", "description": "New name; empty clears it."},
                    "wait_seconds": wait_property()
                },
                "required": ["address", "name"],
                "additionalProperties": false
            }
        },
        {
            "name": "rename_local",
            "description": "Rename one local or parameter of a function. `name` is the \
                decompiler's name (arg0, value12) or the current display name; `new_name` must \
                be a C identifier that is not a keyword and must not appear anywhere in the \
                function body, including another local's display name. Empty `new_name` \
                restores the decompiler's name.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "function": function_property(),
                    "name": {"type": "string", "description": "The local's emitted name (arg0, value12) or its current display name."},
                    "new_name": {"type": "string", "description": "New name; empty restores the emitted name."},
                    "wait_seconds": wait_property()
                },
                "required": ["function", "name", "new_name"],
                "additionalProperties": false
            }
        },
        {
            "name": "set_signature",
            "description": "Declare a function's C signature, for example \
                `int parse(char *buf, size_t len)`, or clear it with an empty string. The \
                signature must fit the recovered calling convention: otherwise nothing \
                changes and the error says why. A change re-decompiles affected functions.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "function": function_property(),
                    "signature": {"type": "string", "description": "One-line C signature; empty clears it."},
                    "wait_seconds": wait_property()
                },
                "required": ["function", "signature"],
                "additionalProperties": false
            }
        },
        {
            "name": "declare_struct",
            "description": "Declare or replace a persistent struct `name` from explicit \
                fields, or drop it with `null` or `[]` (refused while a signature names it). \
                Offsets must not overlap and `type`, if given, must match `bytes`. A \
                declaration does not retype code by itself: a signature parameter typed \
                `struct name *` binds it to the layout that parameter reaches, provided the \
                fields cover every recovered field by offset and width; struct_from_param \
                builds both from a recovered layout.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Struct name without `struct `; a C identifier that is not a keyword."},
                    "fields": {
                        "type": ["array", "null"],
                        "description": "Fields in offset order; null or [] drops the declaration.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "offset": {"type": "integer", "minimum": 0, "description": "Byte offset from the struct's start."},
                                "bytes": {"type": "integer", "enum": [1, 2, 4, 8, 16], "description": "Field width in bytes."},
                                "name": {"type": "string", "description": "Field name: a C identifier that is not a keyword."},
                                "type": {"type": "string", "description": "C spelling of the field type (`u32`, `char *`, `struct node *`); it must match `bytes`, and empty means the width-implied unsigned integer."}
                            },
                            "required": ["offset", "bytes", "name"],
                            "additionalProperties": false
                        }
                    },
                    "wait_seconds": wait_property()
                },
                "required": ["name", "fields"],
                "additionalProperties": false
            }
        },
        {
            "name": "struct_from_param",
            "description": "Declare a persistent struct `name` covering the recovered layout \
                behind parameter `param` (0-based) of `function`, and bind it by typing that \
                parameter `struct name *` in the function's signature. Starts from the \
                declared signature if there is one, otherwise the decompiled one; other \
                undeclared struct pointers in that signature become `void *`. Re-decompiles \
                affected functions. On failure both the declaration and the signature are \
                restored.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "function": function_property(),
                    "param": {"type": "integer", "minimum": 0, "description": "0-based parameter index of the function."},
                    "name": {"type": "string", "description": "Struct name to declare and bind."},
                    "wait_seconds": wait_property()
                },
                "required": ["function", "param", "name"],
                "additionalProperties": false
            }
        },
        {
            "name": "rename_field",
            "description": "Rename a field of a declared struct, chosen by byte `offset` or \
                by current `field` name. Recovered layouts have no persistent field names; \
                declare and bind one with struct_from_param first. Re-decompiles affected \
                functions.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "struct": {"type": "string", "description": "Declared struct name without `struct `."},
                    "offset": {"type": ["integer", "string"], "description": "Field byte offset (decimal or 0x…); give this or `field`, not both."},
                    "field": {"type": "string", "description": "Current field name; give this or `offset`, not both."},
                    "new_name": {"type": "string", "description": "New field name: a C identifier that is not a keyword."},
                    "wait_seconds": wait_property()
                },
                "required": ["struct", "new_name"],
                "anyOf": [{"required": ["offset"]}, {"required": ["field"]}],
                "additionalProperties": false
            }
        },
        {
            "name": "set_comment",
            "description": "Attach a one-line comment to an address, or clear it with an empty \
                comment. It shows on the first pseudocode line mapped to the address (the \
                signature line for an entry with no mapped line), in disassembly, and in \
                list_notes.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "address": {"type": ["string", "integer"], "description": "Address to comment (0x…, decimal, symbol, user name or demangled name)."},
                    "comment": {"type": "string", "description": "One line; empty clears it."},
                    "wait_seconds": wait_property()
                },
                "required": ["address", "comment"],
                "additionalProperties": false
            }
        },
        {
            "name": "list_notes",
            "description": "List everything written about this binary: function and data \
                names, renamed locals, declared signatures, and comments. Paged: the last line \
                gives the total and the next offset.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "offset": page_offset(),
                    "limit": page_limit(200)
                },
                "additionalProperties": false
            }
        }
    ]})
}

fn wait_property() -> Value {
    json!({
        "type": "integer",
        "minimum": 0,
        "maximum": MAX_WAIT,
        "default": DEFAULT_WAIT,
        "description": "Seconds to wait for recovery before reporting its stage; call again to keep waiting."
    })
}

fn page_offset() -> Value {
    json!({"type": "integer", "minimum": 0, "default": 0, "description": "First item to return (0-based)."})
}

fn page_limit(default: u64) -> Value {
    json!({"type": "integer", "minimum": 1, "maximum": 500, "default": default, "description": "Maximum items to return."})
}

fn function_property() -> Value {
    json!({
        "type": ["string", "integer"],
        "description": "Function entry, or an address inside the function (0x…, decimal, symbol, user name or demangled name)."
    })
}

/// Run one tool by name. `Err` means the name itself is unknown, which is a protocol error.
pub fn call(server: &mut Server, name: &str, args: &Value) -> Result<ToolResult, String> {
    let result = match name {
        "status" => tool_status(server),
        "open" => tool_open(server, args),
        "list_functions" => tool_list_functions(server, args),
        "list_imports" => tool_list_imports(server, args),
        "list_strings" => tool_list_strings(server, args),
        "list_segments" => tool_list_segments(server, args),
        "list_structs" => tool_list_structs(server, args),
        "function_info" => tool_function_info(server, args),
        "struct_layouts" => tool_struct_layouts(server, args),
        "decompile" => tool_decompile(server, args),
        "disassemble" => tool_disassemble(server, args),
        "xrefs_to" => tool_xrefs_to(server, args),
        "xrefs_from" => tool_xrefs_from(server, args),
        "read_bytes" => tool_read_bytes(server, args),
        "search_bytes" => tool_search_bytes(server, args),
        "rename_function" => tool_rename_function(server, args),
        "rename_data" => tool_rename_data(server, args),
        "rename_local" => tool_rename_local(server, args),
        "set_signature" => tool_set_signature(server, args),
        "declare_struct" => tool_declare_struct(server, args),
        "struct_from_param" => tool_struct_from_param(server, args),
        "rename_field" => tool_rename_field(server, args),
        "set_comment" => tool_set_comment(server, args),
        "list_notes" => tool_list_notes(server, args),
        other => return Err(format!("unknown tool {other:?}")),
    };
    Ok(match result {
        Ok(text) => ToolResult {
            text,
            is_error: false,
        },
        Err(failure) => ToolResult {
            text: failure.text,
            is_error: failure.is_error,
        },
    })
}

// Arguments ------------------------------------------------------------------------------

fn required_str<'a>(args: &'a Value, key: &str) -> Result<&'a str, Failure> {
    match args.get(key) {
        Some(Value::String(text)) => Ok(text),
        Some(_) => Err(Failure::error(format!("`{key}` must be a string"))),
        None => Err(Failure::error(format!("missing required argument `{key}`"))),
    }
}

fn optional_str<'a>(args: &'a Value, key: &str) -> Result<Option<&'a str>, Failure> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(text)) => Ok(Some(text)),
        Some(_) => Err(Failure::error(format!("`{key}` must be a string"))),
    }
}

fn integer(args: &Value, key: &str, default: u64) -> Result<u64, Failure> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(value) => value
            .as_u64()
            .ok_or_else(|| Failure::error(format!("`{key}` must be a non-negative integer"))),
    }
}

fn wait_seconds(args: &Value) -> Result<u64, Failure> {
    Ok(integer(args, "wait_seconds", DEFAULT_WAIT)?.min(MAX_WAIT))
}

fn address_arg(server: &Server, args: &Value, key: &str) -> Result<u64, Failure> {
    match args.get(key) {
        Some(Value::Number(number)) => number.as_u64().ok_or_else(|| {
            Failure::error(format!(
                "`{key}` must be a non-negative integer or a string"
            ))
        }),
        Some(Value::String(text)) => server.resolve(text).map_err(Failure::error),
        Some(_) => Err(Failure::error(format!(
            "`{key}` must be an address or name (0x…, decimal, symbol, user name or demangled name)"
        ))),
        None => Err(Failure::error(format!("missing required argument `{key}`"))),
    }
}

fn function_arg(
    server: &Server,
    program: &Program,
    args: &Value,
    key: &str,
) -> Result<u64, Failure> {
    let addr = address_arg(server, args, key)?;
    if let Some(function) = program.index.function(addr) {
        return Ok(function.entry);
    }
    if let Some(entry) = program.function_containing(addr) {
        return Ok(entry);
    }
    Err(Failure::error(format!(
        "{} is not a function entry and is not inside a known function; \
         list_functions shows the functions",
        hex(addr)
    )))
}

fn need_program(server: &mut Server, args: &Value) -> Result<std::sync::Arc<Program>, Failure> {
    server.await_program(wait_seconds(args)?)
}

/// Push the notes' signatures and struct declarations into the decompiler. Returns the error
/// so a mutation tool can restore its notes and say what happened.
fn apply_declared_types(server: &mut Server) -> Result<(), String> {
    let applied = match &server.decompiler {
        Some(decompiler) => decompiler.set_types(server.notes.annotations()),
        None => Ok(()),
    };
    server.types_applied = Some(server.notes.types_version());
    applied
}

/// The width-implied unsigned spelling for a field with no declared type.
fn width_type(bytes: u32) -> &'static str {
    match bytes {
        1 => "u8",
        2 => "u16",
        4 => "u32",
        8 => "u64",
        16 => "u128",
        _ => "?",
    }
}

fn layout_field_line(field: &LayoutField) -> String {
    format!(
        "  +{} bytes={} name={} type={} pointee={}",
        hex(field.offset as u64),
        field.bytes,
        field.name,
        field
            .type_name
            .as_deref()
            .unwrap_or_else(|| width_type(field.bytes)),
        field.pointee.as_deref().unwrap_or("-")
    )
}

fn declared_field_line(field: &StructField) -> String {
    format!(
        "  +{} bytes={} name={} type={}",
        hex(field.offset as u64),
        field.bytes,
        field.name,
        if field.type_name.is_empty() {
            width_type(field.bytes)
        } else {
            field.type_name.as_str()
        }
    )
}

/// The end of the last declared field: the declared struct's size.
fn struct_size(decl: &StructAnnotation) -> i64 {
    decl.fields
        .iter()
        .map(|field| field.offset + i64::from(field.bytes))
        .max()
        .unwrap_or(0)
}

fn required_integer(args: &Value, key: &str) -> Result<u64, Failure> {
    match args.get(key) {
        None | Some(Value::Null) => {
            Err(Failure::error(format!("missing required argument `{key}`")))
        }
        Some(value) => value
            .as_u64()
            .ok_or_else(|| Failure::error(format!("`{key}` must be a non-negative integer"))),
    }
}

/// An offset given as a JSON integer or as a decimal/`0x…` string.
fn offset_arg(args: &Value, key: &str) -> Result<Option<i64>, Failure> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(number)) => number
            .as_i64()
            .map(Some)
            .ok_or_else(|| Failure::error(format!("`{key}` must be an integer"))),
        Some(Value::String(text)) => {
            let text = text.trim();
            let parsed = match text.strip_prefix("0x").or_else(|| text.strip_prefix("0X")) {
                Some(hex) => i64::from_str_radix(hex, 16),
                None => text.parse(),
            };
            parsed
                .map(Some)
                .map_err(|_| Failure::error(format!("`{key}` must be a decimal or 0x… offset")))
        }
        Some(_) => Err(Failure::error(format!(
            "`{key}` must be an integer or a 0x… string"
        ))),
    }
}

/// `declare_struct`'s `fields`: absent is an error, `null` or `[]` drops the declaration.
fn declared_fields(args: &Value) -> Result<Option<StructAnnotation>, Failure> {
    let value = args.get("fields").ok_or_else(|| {
        Failure::error("missing required argument `fields` (use null or [] to drop a struct)")
    })?;
    match value {
        Value::Null => Ok(None),
        Value::Array(items) => {
            if items.is_empty() {
                return Ok(None);
            }
            let mut fields = Vec::with_capacity(items.len());
            for item in items {
                if !item.is_object() {
                    return Err(Failure::error("each field must be an object"));
                }
                let offset = offset_arg(item, "offset")?
                    .ok_or_else(|| Failure::error("a field needs an `offset`"))?;
                let bytes = required_integer(item, "bytes")?;
                let bytes = u32::try_from(bytes)
                    .ok()
                    .filter(|bytes| matches!(bytes, 1 | 2 | 4 | 8 | 16))
                    .ok_or_else(|| Failure::error("a field's `bytes` must be 1, 2, 4, 8 or 16"))?;
                let name = required_str(item, "name")?;
                let type_name = optional_str(item, "type")?.unwrap_or("").to_owned();
                fields.push(StructField {
                    offset,
                    bytes,
                    name: name.to_owned(),
                    type_name,
                });
            }
            Ok(Some(StructAnnotation { fields }))
        }
        _ => Err(Failure::error(
            "`fields` must be an array of fields, or null to drop the struct",
        )),
    }
}

fn layout_block(layout: &StructLayout) -> String {
    let mut text = format!(
        "struct {} ({}) size={} params=[{}]\n",
        layout.name,
        if layout.declared {
            "declared"
        } else {
            "recovered"
        },
        layout.size,
        layout
            .params
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    );
    for field in &layout.fields {
        text.push_str(&layout_field_line(field));
        text.push('\n');
    }
    text
}

// Paging ---------------------------------------------------------------------------------

struct Page {
    offset: usize,
    limit: usize,
}

impl Page {
    fn parse(args: &Value, default_limit: u64, max_limit: u64) -> Result<Page, Failure> {
        let offset = usize::try_from(integer(args, "offset", 0)?).unwrap_or(usize::MAX);
        let limit = integer(args, "limit", default_limit)?;
        if limit == 0 {
            return Err(Failure::error("`limit` must be at least 1"));
        }
        Ok(Page {
            offset,
            limit: usize::try_from(limit.min(max_limit)).unwrap_or(usize::MAX),
        })
    }

    fn bounds(&self, total: usize) -> Range<usize> {
        let start = self.offset.min(total);
        start..(start.saturating_add(self.limit)).min(total)
    }

    /// What the last line of a paged list says.
    fn footer(&self, total: usize) -> String {
        let range = self.bounds(total);
        if total == 0 {
            return "no items".into();
        }
        if range.start >= total {
            return format!(
                "offset {} is past the end: {total} items total",
                self.offset
            );
        }
        if range.end < total {
            format!(
                "items {}-{} of {total}; next page: offset={}, limit={}",
                range.start + 1,
                range.end,
                range.end,
                self.limit
            )
        } else {
            format!(
                "items {}-{} of {total}; end of list",
                range.start + 1,
                range.end
            )
        }
    }
}

fn paged(
    title: &str,
    page: &Page,
    total: usize,
    lines: impl IntoIterator<Item = String>,
) -> String {
    let mut text = format!("{title} ({total} total)\n");
    for line in lines {
        text.push_str(&line);
        text.push('\n');
    }
    text.push_str(&page.footer(total));
    text
}

// Names and small formatters -------------------------------------------------------------

fn name_at(program: &Program, notes: &Notes, addr: u64) -> Option<String> {
    notes
        .get(addr)
        .map(str::to_owned)
        .or_else(|| program.index.base_name(addr).map(str::to_owned))
}

fn display_function(program: &Program, notes: &Notes, entry: u64) -> String {
    name_at(program, notes, entry)
        .or_else(|| program.index.function(entry).map(|f| f.name.clone()))
        .unwrap_or_else(|| format!("sub_{entry:x}"))
}

fn kind_name(kind: XrefKind) -> &'static str {
    match kind {
        XrefKind::Call => "call",
        XrefKind::Jump => "jump",
        XrefKind::Data => "data",
    }
}

fn quote(text: &str) -> String {
    serde_json::to_string(text).unwrap_or_else(|_| format!("{text:?}"))
}

fn bytes_hex(bytes: &[u8]) -> String {
    let mut text = String::new();
    for (i, byte) in bytes.iter().enumerate() {
        if i > 0 {
            text.push(' ');
        }
        let _ = write!(text, "{byte:02x}");
    }
    text
}

/// The instruction at `addr`, decoded from the executable segment, or `<unmapped>`.
fn instruction_text(program: &Program, addr: u64) -> String {
    let width = if program.arch.fixed_width().is_some() {
        4
    } else {
        16
    };
    let Some((code, base)) = program.image.code_at(addr) else {
        return "<unmapped>".into();
    };
    let start = (addr - base) as usize;
    if start >= code.len() {
        return "<unmapped>".into();
    }
    let end = start.saturating_add(width).min(code.len());
    disasm::decode(program.arch, addr, &code[start..end]).text()
}

/// The instruction plus the names of its direct targets and its comment, for disassembly.
fn annotated_instruction(program: &Program, notes: &Notes, instruction: &disasm::Insn) -> String {
    let mut text = instruction.text();
    for target in [instruction.target, instruction.data_ref]
        .into_iter()
        .flatten()
    {
        if let Some(name) = name_at(program, notes, target) {
            let _ = write!(text, " ; {name}");
        }
    }
    if let Some(comment) = notes.comment(instruction.addr) {
        let _ = write!(text, " // {comment}");
    }
    text
}

fn block_ranges(program: &Program, entry: u64) -> Vec<Range<u64>> {
    program
        .cfg
        .functions
        .get(&entry)
        .map(|function| {
            function
                .blocks
                .iter()
                .filter_map(|block| program.cfg.blocks.get(block))
                .map(|block| block.pc()..block.end())
                .collect()
        })
        .unwrap_or_default()
}

// Tools ----------------------------------------------------------------------------------

fn tool_status(server: &mut Server) -> Result<String, Failure> {
    server.pump();
    let mut text = String::new();
    match &server.binary {
        Some(path) => {
            let _ = writeln!(text, "binary: {}", path.display());
        }
        None => {
            text.push_str(
                "binary: none\nNo binary is open. Call `open` with a path, or start the server \
                 as `biara mcp <binary>`.",
            );
            return Ok(text);
        }
    }
    let image = server
        .program
        .as_ref()
        .map(|program| program.image.clone())
        .or_else(|| server.image.clone());
    if let Some(image) = &image {
        let _ = writeln!(text, "arch: {}", image.arch);
        let format = match image.os {
            "linux" => "ELF",
            "windows" => "PE",
            "darwin" => "Mach-O",
            other => other,
        };
        let _ = writeln!(text, "format: {format}");
        let _ = writeln!(text, "entry: {}", hex(image.entry));
    }
    match &server.program {
        Some(program) => {
            text.push_str("analysis: ready\n");
            let _ = writeln!(text, "functions: {}", program.index.functions.len());
            let _ = writeln!(text, "strings: {}", program.index.strings.len());
            let _ = writeln!(text, "imports: {}", program.index.imports.len());
        }
        None => {
            match &server.failure {
                Some(error) => {
                    let _ = writeln!(text, "analysis: failed: {error}");
                }
                None => match server.stage {
                    Some(stage) => {
                        let _ = writeln!(
                            text,
                            "analysis: {} ({} functions, {} blocks recovered so far)",
                            stage.label(),
                            server.progress.0,
                            server.progress.1
                        );
                    }
                    None => text.push_str("analysis: starting\n"),
                },
            }
            if let Some(preview) = &server.preview {
                let _ = writeln!(
                    text,
                    "functions: {} (preliminary, from symbols)",
                    preview.functions.len()
                );
                text.push_str("strings: not counted yet\n");
                let _ = writeln!(text, "imports: {}", preview.imports.len());
            }
        }
    }
    match server.notes.path() {
        Some(path) => {
            let _ = writeln!(text, "notes: {}", path.display());
        }
        None => text.push_str("notes: not saved (XDG_DATA_HOME and HOME are unset)\n"),
    }
    Ok(text.trim_end().to_owned())
}

fn tool_open(server: &mut Server, args: &Value) -> Result<String, Failure> {
    let path = std::path::PathBuf::from(required_str(args, "path")?);
    if !path.is_file() {
        return Err(Failure::error(format!(
            "{}: no such file, or it is not a regular file",
            path.display()
        )));
    }
    if server.binary.as_deref() == Some(path.as_path()) {
        let state = if server.program.is_some() {
            "already open"
        } else if server.failure.is_some() {
            "open, but analysis failed; call status"
        } else {
            "already opening; analysis is still running"
        };
        return Ok(format!("{}: {state}", path.display()));
    }
    server.start_binary(path.clone());
    Ok(format!(
        "opening {}: recovery runs in the background. Call `status`, or call any tool that \
         needs the program and it will wait.",
        path.display()
    ))
}

fn tool_list_functions(server: &mut Server, args: &Value) -> Result<String, Failure> {
    let program = need_program(server, args)?;
    let page = Page::parse(args, 100, 500)?;
    let filter_text = optional_str(args, "filter")?.unwrap_or("");
    let filter = filter_text.to_lowercase();
    let notes = &server.notes;
    let mut lines = Vec::new();
    for function in &program.index.functions {
        let display = display_function(&program, notes, function.entry);
        let base = program.index.base_name(function.entry);
        let demangled = program.index.demangled(function.entry);
        if !filter.is_empty() {
            let mut haystack = format!("{} {} {}", hex(function.entry), display, function.name);
            if let Some(base) = base {
                haystack.push(' ');
                haystack.push_str(base);
            }
            if let Some(demangled) = demangled {
                haystack.push(' ');
                haystack.push_str(&demangled.short);
                haystack.push(' ');
                haystack.push_str(&demangled.full);
            }
            if !haystack.to_lowercase().contains(&filter) {
                continue;
            }
        }
        let mut line = format!(
            "{} {} size={} blocks={} callers={} calls={}",
            hex(function.entry),
            display,
            function.size,
            function.blocks,
            function.callers,
            function.calls
        );
        if function.kind == FunctionKind::Stub {
            line.push_str(" stub");
        }
        if let Some(base) = base {
            if base != display {
                let _ = write!(line, " symbol={base}");
            }
        }
        if let Some(demangled) = demangled {
            if demangled.short != display && Some(demangled.short.as_str()) != base {
                let _ = write!(line, " demangled={}", demangled.short);
            }
        }
        lines.push(line);
    }
    let total = lines.len();
    let title = if filter.is_empty() {
        "functions".to_owned()
    } else {
        format!("functions matching {filter_text:?}")
    };
    Ok(paged(
        &title,
        &page,
        total,
        lines[page.bounds(total)].iter().cloned(),
    ))
}

fn tool_list_imports(server: &mut Server, args: &Value) -> Result<String, Failure> {
    let program = need_program(server, args)?;
    let page = Page::parse(args, 100, 500)?;
    let total = program.index.imports.len();
    let lines = program.index.imports[page.bounds(total)]
        .iter()
        .map(|import| {
            let mut line = format!("{} {}", hex(import.slot), import.name);
            if let Some(demangled) = &import.demangled {
                if demangled.full != import.name {
                    let _ = write!(line, " demangled={}", demangled.full);
                }
            }
            line
        });
    Ok(paged("imports", &page, total, lines))
}

fn tool_list_strings(server: &mut Server, args: &Value) -> Result<String, Failure> {
    let program = need_program(server, args)?;
    let page = Page::parse(args, 100, 500)?;
    let filter_text = optional_str(args, "filter")?.unwrap_or("");
    let filter = filter_text.to_lowercase();
    let mut lines = Vec::new();
    for string in &program.index.strings {
        if !filter.is_empty() && !string.text.to_lowercase().contains(&filter) {
            continue;
        }
        let mut line = format!(
            "{} {} bytes={}{}",
            hex(string.addr),
            quote(&string.text),
            string.bytes,
            if string.wide { " utf16" } else { "" }
        );
        if let Some(comment) = server.notes.comment(string.addr) {
            let _ = write!(line, " // {comment}");
        }
        lines.push(line);
    }
    let total = lines.len();
    let title = if filter.is_empty() {
        "strings".to_owned()
    } else {
        format!("strings matching {filter_text:?}")
    };
    Ok(paged(
        &title,
        &page,
        total,
        lines[page.bounds(total)].iter().cloned(),
    ))
}

fn tool_list_segments(server: &mut Server, args: &Value) -> Result<String, Failure> {
    let program = need_program(server, args)?;
    let page = Page::parse(args, 50, 500)?;
    let segments = &program.image.segments;
    let total = segments.len();
    let lines = segments[page.bounds(total)].iter().map(|segment| {
        format!(
            "{} size={} {}{}{}",
            hex(segment.addr),
            hex(segment.size),
            'r',
            if segment.write { 'w' } else { '-' },
            if segment.exec { 'x' } else { '-' }
        )
    });
    Ok(paged("segments", &page, total, lines))
}

fn tool_list_structs(server: &mut Server, args: &Value) -> Result<String, Failure> {
    if server.binary.is_none() {
        return Err(Failure::stage(
            "no binary is open. Call `open` with a path, or start `biara mcp <binary>`.",
        ));
    }
    let page = Page::parse(args, 50, 500)?;
    let structs: Vec<_> = server.notes.structs().iter().collect();
    let total = structs.len();
    let mut lines = Vec::new();
    for (name, decl) in &structs[page.bounds(total)] {
        lines.push(format!(
            "struct {name} fields={} size={}",
            decl.fields.len(),
            struct_size(decl)
        ));
        for field in &decl.fields {
            lines.push(declared_field_line(field));
        }
    }
    Ok(paged("declared structs", &page, total, lines))
}

fn tool_function_info(server: &mut Server, args: &Value) -> Result<String, Failure> {
    let program = need_program(server, args)?;
    let entry = function_arg(server, &program, args, "function")?;
    let notes = &server.notes;
    let function = program
        .index
        .function(entry)
        .ok_or_else(|| Failure::error(format!("{} is not in the function table", hex(entry))))?;
    let display = display_function(&program, notes, entry);
    let mut text = format!("function {} ({})\n", display, hex(entry));
    if function.name != display {
        let _ = writeln!(text, "symbol: {}", function.name);
    }
    if let Some(demangled) = program.index.demangled(entry) {
        if demangled.full != display && demangled.full != function.name {
            let _ = writeln!(text, "demangled: {}", demangled.full);
        }
    }
    let _ = writeln!(text, "size: {} bytes", function.size);
    let _ = writeln!(text, "blocks: {}", function.blocks);
    match function.frame {
        Some(frame) => {
            let _ = writeln!(text, "frame: {frame} bytes ({})", hex(frame));
        }
        None => text.push_str("frame: unknown\n"),
    }
    let _ = writeln!(text, "args: {}", function.args);
    match notes.signature(entry) {
        Some(signature) => {
            let _ = writeln!(text, "signature: {signature} (declared)");
        }
        None => text.push_str("signature: not declared\n"),
    }

    const SHOWN: usize = 50;
    match notes.locals(entry) {
        Some(locals) => {
            let _ = writeln!(text, "renamed locals: {}", locals.len());
            for (emitted, name) in locals.iter().take(SHOWN) {
                let _ = writeln!(text, "  {emitted} -> {name}");
            }
            if locals.len() > SHOWN {
                let _ = writeln!(text, "  ... {} more", locals.len() - SHOWN);
            }
        }
        None => text.push_str("renamed locals: none\n"),
    }

    let ranges = block_ranges(&program, entry);
    let comments: Vec<(u64, &str)> = notes
        .comments()
        .iter()
        .filter(|(addr, _)| ranges.iter().any(|range| range.contains(addr)))
        .map(|(&addr, comment)| (addr, comment.as_str()))
        .collect();
    let _ = writeln!(text, "comments: {}", comments.len());
    for (addr, comment) in comments.iter().take(SHOWN) {
        let _ = writeln!(text, "  {} {comment}", hex(*addr));
    }
    if comments.len() > SHOWN {
        let _ = writeln!(text, "  ... {} more", comments.len() - SHOWN);
    }

    let mut callers: BTreeMap<u64, usize> = BTreeMap::new();
    for xref in program.index.xrefs_to(entry) {
        // A jump into this entry from another function is a tail call; a jump inside this
        // function (a loop back to the entry) and a data reference are not callers.
        let caller = xref.kind == XrefKind::Call
            || (xref.kind == XrefKind::Jump && xref.function != 0 && xref.function != entry);
        if caller && xref.function != 0 {
            *callers.entry(xref.function).or_default() += 1;
        }
    }
    let _ = writeln!(text, "callers: {}", callers.len());
    for (&caller, &count) in callers.iter().take(SHOWN) {
        let name = display_function(&program, notes, caller);
        let _ = writeln!(text, "  {} {name} ({count} call sites)", hex(caller));
    }
    if callers.len() > SHOWN {
        let _ = writeln!(
            text,
            "  ... {} more; xrefs_to {} lists every call site",
            callers.len() - SHOWN,
            hex(entry)
        );
    }

    let mut callees: BTreeMap<u64, usize> = BTreeMap::new();
    if let Some(forward) = &server.forward {
        if let Some(refs) = forward.by_function.get(&entry) {
            for &(target, xref) in refs {
                let callee = xref.kind == XrefKind::Call
                    || (xref.kind == XrefKind::Jump
                        && target != entry
                        && program.index.function(target).is_some());
                if callee {
                    *callees.entry(target).or_default() += 1;
                }
            }
        }
    }
    let _ = writeln!(text, "callees: {}", callees.len());
    for (&callee, &count) in callees.iter().take(SHOWN) {
        let name = name_at(&program, notes, callee).unwrap_or_else(|| format!("sub_{callee:x}"));
        let kind = if program.image.imports.contains_key(&callee) {
            " import"
        } else {
            ""
        };
        let _ = writeln!(text, "  {} {name}{kind} ({count} call sites)", hex(callee));
    }
    if callees.len() > SHOWN {
        let _ = writeln!(
            text,
            "  ... {} more; xrefs_from {} lists every reference",
            callees.len() - SHOWN,
            hex(entry)
        );
    }
    Ok(text.trim_end().to_owned())
}

fn tool_struct_layouts(server: &mut Server, args: &Value) -> Result<String, Failure> {
    let program = need_program(server, args)?;
    let wait = wait_seconds(args)?;
    let entry = function_arg(server, &program, args, "function")?;
    let document = server.await_decompile(entry, wait)?;
    let display = display_function(&program, &server.notes, entry);
    let count = document.layouts.len();
    let mut text = format!(
        "struct layouts of {display} ({}): {count} struct{}\n",
        hex(entry),
        if count == 1 { "" } else { "s" }
    );
    if document.layouts.is_empty() {
        text.push_str("none: no pointer-reached struct was recovered for this function");
        return Ok(text);
    }
    for layout in &document.layouts {
        text.push_str(&layout_block(layout));
    }
    Ok(text.trim_end().to_owned())
}

fn tool_decompile(server: &mut Server, args: &Value) -> Result<String, Failure> {
    let program = need_program(server, args)?;
    let wait = wait_seconds(args)?;
    let entry = function_arg(server, &program, args, "function")?;
    let page = Page::parse(args, 400, 2000)?;
    let addresses = match args.get("addresses") {
        None | Some(Value::Null) => false,
        Some(Value::Bool(on)) => *on,
        Some(_) => return Err(Failure::error("`addresses` must be a boolean")),
    };
    let document = server.await_decompile(entry, wait)?;
    let total = document.lines.len();
    let range = page.bounds(total);

    let page_comments: BTreeMap<usize, Vec<(u64, String)>> = {
        let all = server.comment_lines(&document);
        all.range(range.clone())
            .map(|(&line, found)| (line, found.clone()))
            .collect()
    };
    let notes = &server.notes;
    let display = display_function(&program, notes, entry);
    let mut text = if total == 0 {
        format!("{display} ({}): no lines\n", hex(entry))
    } else if range.start >= total {
        format!(
            "{display} ({}): offset {} is past the end ({total} lines)\n",
            hex(entry),
            page.offset
        )
    } else {
        format!(
            "{display} ({}): lines {}-{} of {total}\n",
            hex(entry),
            range.start,
            range.end - 1
        )
    };
    for line in range.clone() {
        let rendered = document.render_line(line, &|symbol, _emitted| match symbol {
            SymbolRef::Local { name } => notes.local(entry, name).map(str::to_owned),
            SymbolRef::Function { address } | SymbolRef::Global { address } => {
                notes.get(*address).map(str::to_owned)
            }
            SymbolRef::Type { .. } => None,
        });
        if addresses {
            match document.line_addresses(line).first() {
                Some(addr) => {
                    let _ = write!(text, "{line}: {} {rendered}", hex(*addr));
                }
                None => {
                    let _ = write!(text, "{line}: {rendered}");
                }
            }
        } else {
            let _ = write!(text, "{line}: {rendered}");
        }
        if let Some(found) = page_comments.get(&line) {
            for (_, comment) in found {
                let _ = write!(text, " // {comment}");
            }
        }
        text.push('\n');
    }
    if range.end < total {
        let _ = writeln!(
            text,
            "next page: offset={}, limit={}",
            range.end, page.limit
        );
    } else {
        text.push_str("end of list");
    }
    Ok(text)
}

fn tool_disassemble(server: &mut Server, args: &Value) -> Result<String, Failure> {
    let program = need_program(server, args)?;
    let addr = address_arg(server, args, "address_or_function")?;
    let count = integer(args, "count", 20)?;
    if count == 0 {
        return Err(Failure::error("`count` must be at least 1"));
    }
    let count = usize::try_from(count.min(500)).unwrap_or(500);
    let (code, base) = program
        .image
        .code_at(addr)
        .ok_or_else(|| Failure::error(format!("{} is not in executable code", hex(addr))))?;
    let start = (addr - base) as usize;
    if start >= code.len() {
        return Err(Failure::error(format!(
            "{} is past the file-backed bytes of its segment",
            hex(addr)
        )));
    }
    let width = if program.arch.fixed_width().is_some() {
        4
    } else {
        16
    };
    let want = count
        .saturating_mul(width)
        .min(code.len().saturating_sub(start));
    let instructions: Vec<_> = disasm::sweep(program.arch, addr, &code[start..start + want])
        .into_iter()
        .take(count)
        .collect();
    let notes = &server.notes;
    let mut text = String::new();
    let mut next = None;
    for instruction in &instructions {
        let line = annotated_instruction(&program, notes, instruction);
        let _ = writeln!(text, "{}  {line}", hex(instruction.addr));
        next = Some(instruction.addr + u64::from(instruction.len));
    }
    match next {
        Some(next) if instructions.len() == count => {
            let _ = write!(text, "next: address={} count={count}", hex(next));
        }
        Some(next) => {
            let _ = write!(text, "end of executable bytes at {}", hex(next));
        }
        None => text.push_str("no instruction decoded"),
    }
    Ok(text)
}

fn tool_xrefs_to(server: &mut Server, args: &Value) -> Result<String, Failure> {
    let program = need_program(server, args)?;
    let addr = address_arg(server, args, "address")?;
    let page = Page::parse(args, 100, 500)?;
    let references = program.index.xrefs_to(addr);
    let total = references.len();
    let notes = &server.notes;
    let lines = references[page.bounds(total)].iter().map(|xref| {
        let function = if xref.function != 0 {
            display_function(&program, notes, xref.function)
        } else {
            "unknown".into()
        };
        let mut line = format!(
            "{} {} in {function}: {}",
            kind_name(xref.kind),
            hex(xref.from),
            instruction_text(&program, xref.from)
        );
        if let Some(comment) = notes.comment(xref.from) {
            let _ = write!(line, " // {comment}");
        }
        line
    });
    Ok(paged(
        &format!("references to {}", hex(addr)),
        &page,
        total,
        lines,
    ))
}

fn tool_xrefs_from(server: &mut Server, args: &Value) -> Result<String, Failure> {
    let program = need_program(server, args)?;
    let addr = address_arg(server, args, "address_or_function")?;
    let page = Page::parse(args, 100, 500)?;
    let Some(forward) = &server.forward else {
        return Err(Failure::error(
            "the analysis has not produced references yet",
        ));
    };
    let (title, references) = if let Some(function) = program.index.function(addr) {
        (
            format!(
                "references made by {} ({})",
                display_function(&program, &server.notes, function.entry),
                hex(function.entry)
            ),
            forward
                .by_function
                .get(&function.entry)
                .cloned()
                .unwrap_or_default(),
        )
    } else {
        (
            format!("references made by the instruction at {}", hex(addr)),
            forward
                .by_instruction
                .get(&addr)
                .cloned()
                .unwrap_or_default(),
        )
    };
    if references.is_empty() && program.index.function(addr).is_none() {
        return Err(Failure::error(format!(
            "no references are recorded from the instruction at {}; pass the instruction's \
             first address or a function entry",
            hex(addr)
        )));
    }
    let total = references.len();
    let notes = &server.notes;
    let lines = references[page.bounds(total)].iter().map(|(target, xref)| {
        let name = name_at(&program, notes, *target).unwrap_or_else(|| format!("sub_{target:x}"));
        let function = if xref.function != 0 {
            display_function(&program, notes, xref.function)
        } else {
            "unknown".into()
        };
        format!(
            "{} -> {} {name} at {} in {function}: {}",
            kind_name(xref.kind),
            hex(*target),
            hex(xref.from),
            instruction_text(&program, xref.from)
        )
    });
    Ok(paged(&title, &page, total, lines))
}

fn tool_read_bytes(server: &mut Server, args: &Value) -> Result<String, Failure> {
    let program = need_program(server, args)?;
    let addr = address_arg(server, args, "address")?;
    let length = integer(args, "length", 64)?;
    if length == 0 || length > 4096 {
        return Err(Failure::error("`length` must be between 1 and 4096"));
    }
    let length = length as usize;
    let bytes = program.bytes(addr, length).ok_or_else(|| {
        Failure::error(format!(
            "{length} bytes at {} are not all mapped",
            hex(addr)
        ))
    })?;
    Ok(hexdump(addr, &bytes))
}

fn hexdump(addr: u64, bytes: &[u8]) -> String {
    let mut text = String::new();
    for (index, chunk) in bytes.chunks(16).enumerate() {
        let at = addr + (index * 16) as u64;
        let _ = write!(text, "{}  ", hex(at));
        for i in 0..16 {
            match chunk.get(i) {
                Some(byte) => {
                    let _ = write!(text, "{byte:02x} ");
                }
                None => text.push_str("   "),
            }
            if i == 7 {
                text.push(' ');
            }
        }
        text.push_str(" |");
        for &byte in chunk {
            text.push(if (0x20..0x7f).contains(&byte) {
                byte as char
            } else {
                '.'
            });
        }
        text.push_str("|\n");
    }
    text.trim_end().to_owned()
}

fn tool_search_bytes(server: &mut Server, args: &Value) -> Result<String, Failure> {
    const CAP: usize = 100_000;
    let program = need_program(server, args)?;
    let page = Page::parse(args, 50, 500)?;
    let text_pattern = required_str(args, "pattern")?;
    let pattern = parse_pattern(text_pattern)?;
    let mut hits = Vec::new();
    let mut capped = false;
    'segments: for segment in &program.image.segments {
        let data = program.image.data(segment);
        if pattern.len() > data.len() {
            continue;
        }
        for start in 0..=data.len() - pattern.len() {
            let matched = pattern
                .iter()
                .enumerate()
                .all(|(offset, byte)| byte.is_none_or(|byte| data[start + offset] == byte));
            if matched {
                hits.push(segment.addr + start as u64);
                if hits.len() >= CAP {
                    capped = true;
                    break 'segments;
                }
            }
        }
    }
    let total = hits.len();
    let notes = &server.notes;
    let mut lines = Vec::new();
    for &hit in &hits[page.bounds(total)] {
        let matched = program
            .bytes(hit, pattern.len())
            .map(|bytes| bytes_hex(&bytes))
            .unwrap_or_default();
        let mut line = format!("{} {matched}", hex(hit));
        if let Some(string) = program.index.string_at(hit) {
            let _ = write!(line, " ; {}", quote(&string.text));
        } else if let Some(name) = name_at(&program, notes, hit) {
            let _ = write!(line, " ; {name}");
        }
        lines.push(line);
    }
    let mut title = format!("matches for {text_pattern:?}");
    if capped {
        let _ = write!(
            title,
            " (search stopped after {CAP} hits; narrow the pattern)"
        );
    }
    Ok(paged(&title, &page, total, lines))
}

fn parse_pattern(text: &str) -> Result<Vec<Option<u8>>, Failure> {
    let mut pattern = Vec::new();
    for token in text.split_whitespace() {
        let token = token
            .strip_prefix("0x")
            .or_else(|| token.strip_prefix("0X"))
            .unwrap_or(token);
        if token == "?" || token == "??" {
            pattern.push(None);
            continue;
        }
        if token.len() != 2 {
            return Err(Failure::error(format!(
                "{token:?} is not a byte; write two hex digits or `??` for a wildcard"
            )));
        }
        let byte = u8::from_str_radix(token, 16).map_err(|_| {
            Failure::error(format!(
                "{token:?} is not a byte; write two hex digits or `??` for a wildcard"
            ))
        })?;
        pattern.push(Some(byte));
    }
    if pattern.is_empty() {
        return Err(Failure::error("`pattern` cannot be empty"));
    }
    if pattern.len() > 256 {
        return Err(Failure::error(
            "`pattern` is longer than 256 bytes; use a shorter sequence",
        ));
    }
    Ok(pattern)
}

fn tool_rename_function(server: &mut Server, args: &Value) -> Result<String, Failure> {
    let program = need_program(server, args)?;
    let entry = function_arg(server, &program, args, "function")?;
    let name = required_str(args, "name")?;
    let saved = server
        .notes
        .set(entry, NameKind::Function, name)
        .map_err(|error| Failure::error(format!("not renamed: {error}")))?;
    let mut text = if name.is_empty() {
        format!(
            "cleared the name of function {}; its symbol name shows again",
            hex(entry)
        )
    } else {
        format!("function {} is now {name}", hex(entry))
    };
    if let Some(problem) = saved {
        let _ = write!(
            text,
            "\nwarning: the name is in memory but was not saved: {problem}"
        );
    }
    Ok(text)
}

fn tool_rename_data(server: &mut Server, args: &Value) -> Result<String, Failure> {
    let program = need_program(server, args)?;
    let addr = address_arg(server, args, "address")?;
    if program.index.function(addr).is_some() {
        return Err(Failure::error(format!(
            "{} is a function; use rename_function (an empty name clears its name)",
            hex(addr)
        )));
    }
    let name = required_str(args, "name")?;
    if !name.is_empty() && program.image.segment_at(addr).is_none() {
        return Err(Failure::error(format!(
            "{} is not mapped; data names need an address in the image",
            hex(addr)
        )));
    }
    let saved = server
        .notes
        .set(addr, NameKind::Data, name)
        .map_err(|error| Failure::error(format!("not renamed: {error}")))?;
    let mut text = if name.is_empty() {
        format!("cleared the name of the data at {}", hex(addr))
    } else {
        format!("the data at {} is now {name}", hex(addr))
    };
    if let Some(problem) = saved {
        let _ = write!(
            text,
            "\nwarning: the name is in memory but was not saved: {problem}"
        );
    }
    Ok(text)
}

fn tool_rename_local(server: &mut Server, args: &Value) -> Result<String, Failure> {
    let program = need_program(server, args)?;
    let wait = wait_seconds(args)?;
    let entry = function_arg(server, &program, args, "function")?;
    let name = required_str(args, "name")?;
    let new_name = required_str(args, "new_name")?;
    let document = server.await_decompile(entry, wait)?;
    if document.locals.is_empty() {
        return Err(Failure::error(format!(
            "{} has no named locals or parameters",
            display_function(&program, &server.notes, entry)
        )));
    }
    let renamed: BTreeMap<String, String> = server.notes.locals(entry).cloned().unwrap_or_default();
    let effective = |emitted: &str| {
        renamed
            .get(emitted)
            .cloned()
            .unwrap_or_else(|| emitted.to_owned())
    };
    let emitted = document
        .locals
        .iter()
        .find(|local| local.as_str() == name)
        .cloned()
        .or_else(|| {
            document
                .locals
                .iter()
                .find(|local| effective(local) == name)
                .cloned()
        })
        .ok_or_else(|| {
            Failure::error(format!(
                "{name:?} is not a local or parameter of {}; the locals are: {}",
                display_function(&program, &server.notes, entry),
                document.locals.join(", ")
            ))
        })?;

    if !new_name.is_empty() {
        if let Some((other, _)) = renamed
            .iter()
            .find(|(local, display)| local.as_str() != emitted && display.as_str() == new_name)
        {
            return Err(Failure::error(format!(
                "{new_name:?} is already the display name of {other:?} in this function; \
                 choose another name"
            )));
        }
        for token in &document.tokens {
            if token.kind != TokenKind::Identifier {
                continue;
            }
            let text = document
                .body
                .get(token.start as usize..token.end as usize)
                .unwrap_or("");
            if text != new_name {
                continue;
            }
            if matches!(&token.symbol, Some(SymbolRef::Local { name }) if name == &emitted) {
                continue;
            }
            return Err(Failure::error(format!(
                "{new_name:?} already appears in the function body (at byte {}); \
                 choose a name that is not used there",
                token.start
            )));
        }
    }

    let saved = server
        .notes
        .set_local(entry, &emitted, new_name)
        .map_err(|error| Failure::error(format!("not renamed: {error}")))?;
    let display = display_function(&program, &server.notes, entry);
    let mut text = if new_name.is_empty() {
        format!("{display}: {emitted} uses the decompiler's name again")
    } else {
        format!("{display}: {emitted} is now {new_name}")
    };
    if let Some(problem) = saved {
        let _ = write!(
            text,
            "\nwarning: the name is in memory but was not saved: {problem}"
        );
    }
    Ok(text)
}

fn tool_set_signature(server: &mut Server, args: &Value) -> Result<String, Failure> {
    let program = need_program(server, args)?;
    let entry = function_arg(server, &program, args, "function")?;
    let signature = required_str(args, "signature")?;
    let previous = server.notes.signature(entry).map(str::to_owned);
    let save_problem = server
        .notes
        .set_signature(entry, signature)
        .map_err(|error| Failure::error(format!("signature not set: {error}")))?;
    let applied = apply_declared_types(server);
    if let Err(error) = applied {
        let restore = previous.as_deref().unwrap_or("");
        if let Err(problem) = server.notes.set_signature(entry, restore) {
            eprintln!("biara mcp: restoring the previous signature failed: {problem}");
        }
        let restored = apply_declared_types(server);
        if let Err(problem) = &restored {
            eprintln!("biara mcp: the restored signature is not in effect: {problem}");
        }
        server.types_problem = restored.err();
        return Err(Failure::error(format!(
            "signature not applied: {error}\nthe previous signature was restored"
        )));
    }
    server.types_problem = None;
    let display = display_function(&program, &server.notes, entry);
    let mut text = if signature.is_empty() {
        format!(
            "cleared the declared signature of {display} ({})",
            hex(entry)
        )
    } else {
        format!(
            "declared {display} ({}) as `{signature}`; decompilation now uses it",
            hex(entry)
        )
    };
    if let Some(problem) = save_problem {
        let _ = write!(
            text,
            "\nwarning: the signature is in memory but was not saved: {problem}"
        );
    }
    Ok(text)
}

fn tool_declare_struct(server: &mut Server, args: &Value) -> Result<String, Failure> {
    need_program(server, args)?;
    let name = required_str(args, "name")?.to_owned();
    let declaration = declared_fields(args)?;
    let previous = server.notes.structs().get(&name).cloned();
    let save_problem = server
        .notes
        .set_struct(&name, declaration.clone())
        .map_err(|error| Failure::error(format!("struct {name} not declared: {error}")))?;
    let applied = apply_declared_types(server);
    if let Err(error) = applied {
        if let Err(problem) = server.notes.set_struct(&name, previous) {
            eprintln!("biara mcp: restoring struct {name} failed: {problem}");
        }
        let restored = apply_declared_types(server);
        if let Err(problem) = &restored {
            eprintln!("biara mcp: the restored struct is not in effect: {problem}");
        }
        server.types_problem = restored.err();
        return Err(Failure::error(format!(
            "struct {name} not applied: {error}\nthe previous declaration was restored"
        )));
    }
    server.types_problem = None;
    let mut text = match &declaration {
        Some(declaration) => format!(
            "declared struct {name}: {} fields, {} bytes\n{}\n\
             bind it by typing a parameter `struct {name} *`, or use struct_from_param",
            declaration.fields.len(),
            struct_size(declaration),
            declaration
                .fields
                .iter()
                .map(declared_field_line)
                .collect::<Vec<_>>()
                .join("\n")
        ),
        None => format!("dropped struct {name}"),
    };
    if let Some(problem) = save_problem {
        let _ = write!(
            text,
            "\nwarning: the declaration is in memory but was not saved: {problem}"
        );
    }
    Ok(text)
}

fn tool_struct_from_param(server: &mut Server, args: &Value) -> Result<String, Failure> {
    let program = need_program(server, args)?;
    let wait = wait_seconds(args)?;
    let entry = function_arg(server, &program, args, "function")?;
    let param = required_integer(args, "param")?;
    let param = usize::try_from(param)
        .map_err(|_| Failure::error("`param` is too large for a parameter index"))?;
    let name = required_str(args, "name")?.to_owned();
    let document = server.await_decompile(entry, wait)?;
    let display = display_function(&program, &server.notes, entry);
    let layout = document
        .layouts
        .iter()
        .find(|layout| {
            layout
                .params
                .iter()
                .any(|&pointed| pointed as usize == param)
        })
        .cloned()
        .ok_or_else(|| {
            Failure::error(format!(
                "parameter {param} of {display} does not point at a recovered struct; \
                 struct_layouts {{function: {}}} lists what its pointers reach",
                hex(entry)
            ))
        })?;
    let declaration = declare_from_layout(&layout);
    let base = server
        .notes
        .signature(entry)
        .map(str::to_owned)
        .unwrap_or_else(|| document.signature.clone());
    let mut declared = server.notes.structs().clone();
    declared.insert(name.clone(), declaration.clone());
    let bound = bind_param(&base, param, &name, &declared).map_err(|error| {
        Failure::error(format!(
            "cannot bind parameter {param} of {display} to struct {name}: {error}"
        ))
    })?;

    let previous_struct = server.notes.structs().get(&name).cloned();
    let previous_signature = server.notes.signature(entry).map(str::to_owned);
    let save_problem = server
        .notes
        .set_struct(&name, Some(declaration.clone()))
        .map_err(|error| Failure::error(format!("struct {name} not declared: {error}")))?;
    let signature_save = match server.notes.set_signature(entry, &bound) {
        Ok(saved) => saved,
        Err(error) => {
            let previous = previous_signature.clone().unwrap_or_default();
            if let Err(problem) = server.notes.set_signature(entry, &previous) {
                eprintln!("biara mcp: restoring the previous signature failed: {problem}");
            }
            if let Err(problem) = server.notes.set_struct(&name, previous_struct) {
                eprintln!("biara mcp: restoring struct {name} failed: {problem}");
            }
            let restored = apply_declared_types(server);
            if let Err(problem) = &restored {
                eprintln!("biara mcp: the restored declaration is not in effect: {problem}");
            }
            server.types_problem = restored.err();
            return Err(Failure::error(format!(
                "signature not set: {error}\nthe previous declaration and signature were restored"
            )));
        }
    };
    let applied = apply_declared_types(server);
    if let Err(error) = applied {
        if let Err(problem) = server
            .notes
            .set_signature(entry, previous_signature.as_deref().unwrap_or(""))
        {
            eprintln!("biara mcp: restoring the previous signature failed: {problem}");
        }
        if let Err(problem) = server.notes.set_struct(&name, previous_struct) {
            eprintln!("biara mcp: restoring struct {name} failed: {problem}");
        }
        let restored = apply_declared_types(server);
        if let Err(problem) = &restored {
            eprintln!("biara mcp: the restored declaration is not in effect: {problem}");
        }
        server.types_problem = restored.err();
        return Err(Failure::error(format!(
            "struct {name} not bound: {error}\n\
             the previous declaration and signature were restored"
        )));
    }
    server.types_problem = None;
    let mut text = format!(
        "declared struct {name} and bound parameter {param} of {display} ({})\n\
         signature: {bound}\n\
         struct {name}: {} fields, {} bytes\n{}",
        hex(entry),
        declaration.fields.len(),
        struct_size(&declaration),
        declaration
            .fields
            .iter()
            .map(declared_field_line)
            .collect::<Vec<_>>()
            .join("\n")
    );
    for problem in [save_problem, signature_save].into_iter().flatten() {
        let _ = write!(
            text,
            "\nwarning: the change is in memory but was not saved: {problem}"
        );
    }
    Ok(text)
}

fn field_names(declaration: &StructAnnotation) -> String {
    declaration
        .fields
        .iter()
        .map(|field| field.name.as_str())
        .collect::<Vec<_>>()
        .join(", ")
}

fn tool_rename_field(server: &mut Server, args: &Value) -> Result<String, Failure> {
    need_program(server, args)?;
    let name = required_str(args, "struct")?.to_owned();
    let new_name = required_str(args, "new_name")?;
    let offset = offset_arg(args, "offset")?;
    let field = optional_str(args, "field")?;
    if offset.is_some() && field.is_some() {
        return Err(Failure::error("give either `offset` or `field`, not both"));
    }
    let declaration = server.notes.structs().get(&name).cloned().ok_or_else(|| {
        Failure::error(format!(
            "struct {name} is not declared; recovered layouts are per-decompile. \
             Inspect one with struct_layouts and declare and bind it with struct_from_param, \
             then rename its fields"
        ))
    })?;
    let offset = match (offset, field) {
        (Some(offset), None) => {
            if offset < 0 {
                return Err(Failure::error("a field offset cannot be negative"));
            }
            offset
        }
        (None, Some(field)) => declaration
            .fields
            .iter()
            .filter(|declared| declared.name == field)
            .min_by_key(|declared| declared.offset)
            .map(|declared| declared.offset)
            .ok_or_else(|| {
                Failure::error(format!(
                    "struct {name} has no field named {field:?}; its fields are: {}",
                    field_names(&declaration)
                ))
            })?,
        (None, None) => {
            return Err(Failure::error(
                "give a field `offset` or a current `field` name",
            ));
        }
        (Some(_), Some(_)) => unreachable!("checked above"),
    };
    let previous = declaration
        .fields
        .iter()
        .find(|declared| declared.offset == offset)
        .map(|declared| declared.name.clone())
        .ok_or_else(|| {
            Failure::error(format!(
                "struct {name} has no field at {}; its fields are: {}",
                hex(offset as u64),
                field_names(&declaration)
            ))
        })?;
    let bytes = declaration
        .fields
        .iter()
        .find(|declared| declared.offset == offset)
        .map_or(0, |declared| declared.bytes);
    let save_problem = server
        .notes
        .rename_field(&name, offset, new_name)
        .map_err(|error| Failure::error(format!("field not renamed: {error}")))?;
    let applied = apply_declared_types(server);
    if let Err(error) = applied {
        if let Err(problem) = server.notes.rename_field(&name, offset, &previous) {
            eprintln!("biara mcp: restoring field {previous} failed: {problem}");
        }
        let restored = apply_declared_types(server);
        if let Err(problem) = &restored {
            eprintln!("biara mcp: the restored field name is not in effect: {problem}");
        }
        server.types_problem = restored.err();
        return Err(Failure::error(format!(
            "field not renamed: {error}\nthe previous name was restored"
        )));
    }
    server.types_problem = None;
    let mut text = format!(
        "struct {name} field {} (bytes={bytes}) is now {new_name}",
        hex(offset as u64)
    );
    if let Some(problem) = save_problem {
        let _ = write!(
            text,
            "\nwarning: the name is in memory but was not saved: {problem}"
        );
    }
    Ok(text)
}

fn tool_set_comment(server: &mut Server, args: &Value) -> Result<String, Failure> {
    let program = need_program(server, args)?;
    let addr = address_arg(server, args, "address")?;
    let comment = required_str(args, "comment")?;
    let saved = server
        .notes
        .set_comment(addr, comment)
        .map_err(|error| Failure::error(format!("comment not set: {error}")))?;
    let mut text = if comment.is_empty() {
        format!("cleared the comment at {}", hex(addr))
    } else {
        let target = name_at(&program, &server.notes, addr)
            .map(|name| format!(" ({name})"))
            .unwrap_or_default();
        format!("commented {}{target}: {comment}", hex(addr))
    };
    if let Some(problem) = saved {
        let _ = write!(
            text,
            "\nwarning: the comment is in memory but was not saved: {problem}"
        );
    }
    Ok(text)
}

fn tool_list_notes(server: &mut Server, args: &Value) -> Result<String, Failure> {
    let Some(binary) = &server.binary else {
        return Ok(
            "no binary is open. Call `open` with a path, or start `biara mcp <binary>`.".into(),
        );
    };
    let page = Page::parse(args, 200, 500)?;
    let annotations = server.notes.annotations();
    let mut lines = Vec::new();
    for (&addr, function) in &annotations.functions {
        if let Some(name) = &function.name {
            lines.push(format!("name function {} {name}", hex(addr)));
        }
        for (emitted, name) in &function.locals {
            lines.push(format!("local function={} {emitted} -> {name}", hex(addr)));
        }
        if let Some(signature) = &function.signature {
            lines.push(format!("signature function={} {signature}", hex(addr)));
        }
    }
    for (name, declaration) in &annotations.structs {
        lines.push(format!(
            "struct {name} fields={} size={}",
            declaration.fields.len(),
            struct_size(declaration)
        ));
        for field in &declaration.fields {
            lines.push(format!(
                "field struct={name} +{} bytes={} name={} type={}",
                hex(field.offset as u64),
                field.bytes,
                field.name,
                if field.type_name.is_empty() {
                    width_type(field.bytes)
                } else {
                    field.type_name.as_str()
                }
            ));
        }
    }
    for (&addr, name) in &annotations.globals {
        lines.push(format!("name data {} {name}", hex(addr)));
    }
    for (&addr, comment) in &annotations.comments {
        lines.push(format!("comment {} {comment}", hex(addr)));
    }
    let total = lines.len();
    let title = match server.notes.path() {
        Some(path) => format!("notes for {} ({})", binary.display(), path.display()),
        None => format!("notes for {} (not saved)", binary.display()),
    };
    Ok(paged(
        &title,
        &page,
        total,
        lines[page.bounds(total)].iter().cloned(),
    ))
}
