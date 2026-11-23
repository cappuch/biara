//! `biara mcp`: one open binary served to an AI agent over stdio.
//!
//! The transport is newline-delimited JSON-RPC 2.0, as the Model Context Protocol's stdio
//! binding specifies. Two protocol eras are answered on the same stream: the stateless
//! 2026-07-28 revision, where every request names its protocol version in `_meta`, and the
//! `initialize` handshake of 2025-11-25 and earlier. The era is decided per request, never
//! stored as a connection's identity, so a client may mix the two.
//!
//! Nothing native may reach the protocol stream: the engine, LLVM and QEMU workers write to
//! file descriptor 1 freely, and one stray byte would corrupt a JSON line. At startup the
//! original stdout is duplicated into a private handle kept for protocol messages, and
//! descriptor 1 is pointed at stderr. Logging therefore goes to stderr like everything else.
//!
//! The binary is optional; `open` switches. Recovery always runs on a background thread and
//! tools that need the program wait for it, reporting the stage instead of failing when the
//! wait runs out.

use std::{
    collections::{BTreeMap, HashMap},
    io::{self, BufRead, Write},
    num::NonZeroUsize,
    path::PathBuf,
    sync::{
        Arc,
        mpsc::{Receiver, RecvTimeoutError, TryRecvError},
    },
    time::{Duration, Instant},
};

use anyhow::{Context as _, Result, bail};
use biara_loader::Image;
use biara_session::{
    Decompiler, Event, Index, Notes, OpenOptions, Program, Stage, Xref,
    decomp::{Decompiled, Status},
    recovery,
};
use serde_json::{Value, json};

mod resolve;
mod tools;

/// The revision whose requests carry their version and identity inline.
const MODERN_VERSION: &str = "2026-07-28";
/// Revisions served by the `initialize` handshake, newest first.
const LEGACY_VERSIONS: [&str; 4] = ["2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"];
/// Longest a tool waits for recovery, in seconds.
const MAX_WAIT: u64 = 600;
/// How long a tool waits for recovery unless `wait_seconds` says otherwise.
const DEFAULT_WAIT: u64 = 60;
/// Cache hints on `tools/list` and `server/discover`.
const TTL_MS: u64 = 3_600_000;

const INSTRUCTIONS: &str = "biara analyses one binary at a time: functions, disassembly, \
cross-references, pseudocode, struct layouts, bytes and strings, plus names, local renames, \
signatures, struct declarations and comments that persist in a notes file shared with the \
biara workbench. Call `status` first. Functions and addresses accept hex (0x...), decimal, a \
symbol name, a name you set, or a demangled name. Lists are paged with offset/limit and \
report the total and the next offset.";

#[derive(clap::Args)]
pub struct McpArgs {
    /// Native executable or DLL to analyse. Omit it to start empty and use the `open` tool.
    pub binary: Option<PathBuf>,
    /// Persistent analysis database; default the content-addressed user cache.
    #[arg(long)]
    pub database: Option<PathBuf>,
    /// Analysis worker processes; defaults from the core count and free memory.
    #[arg(long)]
    pub jobs: Option<NonZeroUsize>,
    /// Analysis memory budget in MiB, including recovery workers.
    #[arg(long, default_value_t = 4096)]
    pub memory_mib: u64,
    /// Stop recovery after this many translation blocks.
    #[arg(long)]
    pub max_blocks: Option<usize>,
}

/// Serve one binary (or none) over stdio until stdin closes.
pub fn run(args: McpArgs) -> Result<()> {
    let mut out = protocol_writer()?;
    let mut server = Server::new(&args);
    if let Some(path) = &args.binary {
        if !path.is_file() {
            bail!("{}: no such file", path.display());
        }
        server.start_binary(path.clone());
    }
    match &args.binary {
        Some(path) => eprintln!("biara mcp: serving {} over stdio", path.display()),
        None => {
            eprintln!("biara mcp: no binary; use the open tool or pass one on the command line")
        }
    }
    for line in io::stdin().lock().lines() {
        let line = line.context("reading stdin")?;
        if line.trim().is_empty() {
            continue;
        }
        let Some(reply) = server.handle_line(&line) else {
            continue;
        };
        let mut bytes = serde_json::to_vec(&reply).context("serialising a protocol reply")?;
        bytes.push(b'\n');
        out.write_all(&bytes)
            .and_then(|()| out.flush())
            .context("writing the protocol stream")?;
    }
    Ok(())
}

/// The original stdout, before descriptor 1 is pointed at stderr.
///
/// Native code (LLVM, QEMU, the engine and its workers) may write to fd 1 at any time; if that
/// were the protocol stream, one diagnostic would break the JSON framing. Duplicating the
/// descriptor first keeps the real stream private to this module.
#[cfg(unix)]
fn protocol_writer() -> Result<Box<dyn Write>> {
    io::stdout().flush().context("flushing stdout")?;
    // SAFETY: `dup` and `dup2` are plain descriptor operations; the returned descriptor is
    // owned by the `File` and never aliased elsewhere.
    let fd = unsafe { libc::dup(1) };
    if fd < 0 {
        return Err(io::Error::last_os_error()).context("duplicating stdout");
    }
    // SAFETY: `fd` was just returned by `dup` and is not owned by anything else.
    let out = unsafe { <std::fs::File as std::os::fd::FromRawFd>::from_raw_fd(fd) };
    // Worker processes must not inherit the protocol descriptor: a surviving worker would
    // hold the client's stdout pipe open after this process exits.
    // SAFETY: `fd` is open and `fcntl` only sets a descriptor flag.
    if unsafe { libc::fcntl(fd, libc::F_SETFD, libc::FD_CLOEXEC) } < 0 {
        return Err(io::Error::last_os_error())
            .context("marking the protocol stream close-on-exec");
    }
    // SAFETY: 2 is the process's stderr and always open here.
    if unsafe { libc::dup2(2, 1) } < 0 {
        return Err(io::Error::last_os_error()).context("pointing stdout at stderr");
    }
    Ok(Box::new(out))
}

/// Without POSIX descriptor operations there is no safe way to keep native writes off stdout;
/// the protocol uses it directly and native logging may interfere.
#[cfg(not(unix))]
fn protocol_writer() -> Result<Box<dyn Write>> {
    Ok(Box::new(io::stdout()))
}

/// A tool failure. `is_error` is false for "not ready yet", which the protocol delivers as an
/// ordinary result so the agent reads the stage and calls again rather than treating it as a
/// failure.
pub struct Failure {
    pub text: String,
    pub is_error: bool,
}

impl Failure {
    pub fn error(text: impl Into<String>) -> Failure {
        Failure {
            text: text.into(),
            is_error: true,
        }
    }

    pub fn stage(text: impl Into<String>) -> Failure {
        Failure {
            text: text.into(),
            is_error: false,
        }
    }
}

/// A JSON-RPC protocol error (as opposed to a tool error, which is a result).
pub struct RpcError {
    pub code: i64,
    pub message: String,
    pub data: Option<Value>,
}

impl RpcError {
    fn parse(message: impl Into<String>) -> RpcError {
        RpcError {
            code: -32700,
            message: message.into(),
            data: None,
        }
    }

    fn invalid_request(message: impl Into<String>) -> RpcError {
        RpcError {
            code: -32600,
            message: message.into(),
            data: None,
        }
    }

    fn method_not_found(method: &str) -> RpcError {
        RpcError {
            code: -32601,
            message: format!("method not found: {method}"),
            data: None,
        }
    }

    fn invalid_params(message: impl Into<String>) -> RpcError {
        RpcError {
            code: -32602,
            message: message.into(),
            data: None,
        }
    }

    fn unsupported_version(requested: &str) -> RpcError {
        RpcError {
            code: -32022,
            message: "Unsupported protocol version".into(),
            data: Some(json!({
                "supported": [MODERN_VERSION],
                "requested": requested,
            })),
        }
    }
}

/// Calls made by every instruction of a function, and every reference made by each
/// instruction, keyed both ways. [`Index`] keeps incoming references only, so the MCP server
/// builds this once when recovery finishes; it is cheap and stays out of `biara-session`.
#[derive(Default)]
struct Forward {
    by_function: HashMap<u64, Vec<(u64, Xref)>>,
    by_instruction: HashMap<u64, Vec<(u64, Xref)>>,
}

impl Forward {
    fn build(index: &Index) -> Forward {
        let mut forward = Forward::default();
        for (&target, refs) in &index.xrefs {
            for xref in refs {
                forward
                    .by_instruction
                    .entry(xref.from)
                    .or_default()
                    .push((target, *xref));
                if xref.function != 0 {
                    forward
                        .by_function
                        .entry(xref.function)
                        .or_default()
                        .push((target, *xref));
                }
            }
        }
        for refs in forward
            .by_instruction
            .values_mut()
            .chain(forward.by_function.values_mut())
        {
            refs.sort_by_key(|(target, xref)| (xref.from, *target, xref.kind));
        }
        forward
    }
}

/// Comments placed on a decompiled document, cached until the document or the notes change.
struct CommentCache {
    entry: u64,
    notes_version: u64,
    document: usize,
    lines: BTreeMap<usize, Vec<(u64, String)>>,
}

/// Everything the running server knows: the open binary, its recovery, the decompiler, the
/// notes, and the small caches derived from them.
pub struct Server {
    database: Option<PathBuf>,
    workers: Option<usize>,
    memory_mib: u64,
    max_blocks: Option<usize>,

    binary: Option<PathBuf>,
    image: Option<Arc<Image>>,
    preview: Option<Arc<Index>>,
    program: Option<Arc<Program>>,
    recovery: Option<Receiver<Event>>,
    stage: Option<Stage>,
    progress: (usize, usize),
    failure: Option<String>,

    notes: Notes,
    notes_problem: Option<String>,
    decompiler: Option<Decompiler>,
    types_applied: Option<u64>,
    types_problem: Option<String>,

    forward: Option<Forward>,
    comments: Option<CommentCache>,
}

impl Server {
    fn new(args: &McpArgs) -> Server {
        Server {
            database: args.database.clone(),
            workers: args.jobs.map(NonZeroUsize::get),
            memory_mib: args.memory_mib,
            max_blocks: args.max_blocks,
            binary: None,
            image: None,
            preview: None,
            program: None,
            recovery: None,
            stage: None,
            progress: (0, 0),
            failure: None,
            notes: Notes::in_memory(),
            notes_problem: None,
            decompiler: None,
            types_applied: None,
            types_problem: None,
            forward: None,
            comments: None,
        }
    }

    /// Make `path` the open binary; recovery starts on its own thread.
    fn start_binary(&mut self, path: PathBuf) {
        self.binary = Some(path.clone());
        self.image = None;
        self.preview = None;
        self.program = None;
        self.recovery = None;
        self.stage = Some(Stage::Loading);
        self.progress = (0, 0);
        self.failure = None;
        self.decompiler = None;
        self.types_applied = None;
        self.types_problem = None;
        self.forward = None;
        self.comments = None;

        let (notes, problem) = Notes::load(&path);
        self.notes = notes;
        self.notes_problem = problem;
        if let Some(problem) = &self.notes_problem {
            eprintln!("biara mcp: notes: {problem}");
        }
        self.recovery = Some(recovery::open(
            path,
            noop_notify(),
            OpenOptions {
                database: self.database.clone(),
                worker_exe: None,
                workers: self.workers,
                memory_mib: Some(self.memory_mib),
                max_blocks: self.max_blocks,
            },
        ));
    }

    /// Take every queued recovery event.
    fn pump(&mut self) {
        loop {
            let event = match self.recovery.as_ref() {
                Some(rx) => rx.try_recv(),
                None => break,
            };
            match event {
                Ok(event) => self.handle_event(event),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    self.recovery = None;
                    break;
                }
            }
        }
    }

    fn handle_event(&mut self, event: Event) {
        match event {
            Event::Opened { image, preview, .. } => {
                self.image = Some(image);
                self.preview = Some(preview);
            }
            Event::Progress {
                stage,
                functions,
                blocks,
            } => {
                self.stage = Some(stage);
                self.progress = (functions, blocks);
            }
            Event::Ready(program) => {
                self.program = Some(program.clone());
                self.stage = None;
                self.forward = Some(Forward::build(&program.index));
                let decompiler = Decompiler::new(program.clone(), noop_notify());
                self.decompiler = Some(decompiler);
                self.types_applied = Some(self.notes.types_version());
                if !self.notes.signatures().is_empty() || !self.notes.structs().is_empty() {
                    self.apply_types();
                }
                eprintln!(
                    "biara mcp: ready: {} functions, {} strings, {} imports",
                    program.index.functions.len(),
                    program.index.strings.len(),
                    program.index.imports.len()
                );
            }
            Event::Failed(error) => {
                self.stage = None;
                eprintln!("biara mcp: analysis failed: {error}");
                self.failure = Some(error);
            }
        }
    }

    /// Apply the notes' declared signatures and structs to the decompiler, remembering a
    /// failure so tools can warn instead of silently decompiling under the old types.
    fn apply_types(&mut self) {
        let Some(decompiler) = &self.decompiler else {
            return;
        };
        let applied = decompiler.set_types(self.notes.annotations());
        self.types_applied = Some(self.notes.types_version());
        match applied {
            Ok(()) => self.types_problem = None,
            Err(error) => {
                eprintln!("biara mcp: declared types are not in effect: {error}");
                self.types_problem = Some(error);
            }
        }
    }

    /// Take in the workbench's edits before every tool call, and re-decompile under new
    /// signatures or struct declarations when one appeared.
    fn before_call(&mut self) -> Vec<String> {
        let mut warnings = Vec::new();
        if let Some(problem) = self.notes_problem.take() {
            warnings.push(format!("warning: notes: {problem}"));
        }
        if self.binary.is_none() {
            return warnings;
        }
        match self.notes.reload_if_changed() {
            Ok(_) => {}
            Err(error) => {
                let warning = format!("warning: could not reload notes: {error}");
                eprintln!("biara mcp: {warning}");
                warnings.push(warning);
            }
        }
        if self.decompiler.is_some() && self.types_applied != Some(self.notes.types_version()) {
            self.apply_types();
        }
        if let Some(problem) = &self.types_problem {
            warnings.push(format!(
                "warning: declared types are not in effect: {problem}"
            ));
        }
        warnings
    }

    /// The program, once recovery finishes or `wait_seconds` runs out. Running out is not a
    /// tool error: the agent reads the stage and calls again.
    fn await_program(&mut self, wait_seconds: u64) -> Result<Arc<Program>, Failure> {
        self.pump();
        if let Some(program) = &self.program {
            return Ok(program.clone());
        }
        if let Some(error) = &self.failure {
            return Err(Failure::error(format!("analysis failed: {error}")));
        }
        if self.binary.is_none() {
            return Err(Failure::stage(
                "no binary is open. Call `open` with a path, or start `biara mcp <binary>`.",
            ));
        }
        let deadline = Instant::now() + Duration::from_secs(wait_seconds.min(MAX_WAIT));
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break;
            }
            let received = match self.recovery.as_ref() {
                Some(rx) => rx.recv_timeout(remaining),
                None => break,
            };
            match received {
                Ok(event) => {
                    self.handle_event(event);
                    if let Some(program) = &self.program {
                        return Ok(program.clone());
                    }
                    if let Some(error) = &self.failure {
                        return Err(Failure::error(format!("analysis failed: {error}")));
                    }
                }
                Err(RecvTimeoutError::Timeout) => break,
                Err(RecvTimeoutError::Disconnected) => {
                    self.recovery = None;
                    break;
                }
            }
        }
        if let Some(program) = &self.program {
            return Ok(program.clone());
        }
        if let Some(error) = &self.failure {
            return Err(Failure::error(format!("analysis failed: {error}")));
        }
        Err(Failure::stage(self.not_ready_text()))
    }

    /// Recover `entry`'s pseudocode, waiting at most `wait_seconds` for the decompiler.
    fn await_decompile(
        &mut self,
        entry: u64,
        wait_seconds: u64,
    ) -> Result<Arc<Decompiled>, Failure> {
        let Some(decompiler) = &self.decompiler else {
            return Err(Failure::stage(
                "analysis is not ready; call `status` and try again",
            ));
        };
        decompiler.request(entry);
        let deadline = Instant::now() + Duration::from_secs(wait_seconds.min(MAX_WAIT));
        loop {
            match decompiler.status(entry) {
                Status::Done(document) => return Ok(document),
                Status::Failed(error) => {
                    return Err(Failure::error(format!(
                        "decompiling {} failed: {error}",
                        hex(entry)
                    )));
                }
                Status::Pending | Status::None => {}
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break;
            }
            std::thread::sleep(remaining.min(Duration::from_millis(20)));
        }
        Err(Failure::stage(format!(
            "decompiling {} has not finished. Call decompile again to keep waiting \
             (wait_seconds up to {MAX_WAIT}), or use `status`.",
            hex(entry)
        )))
    }

    fn not_ready_text(&self) -> String {
        let mut text = match self.stage {
            Some(stage) => format!(
                "not ready: {} ({} functions, {} blocks recovered so far)",
                stage.label(),
                self.progress.0,
                self.progress.1
            ),
            None => "not ready: starting analysis".into(),
        };
        text.push_str(&format!(
            "\nThe program is still being recovered. Call this tool again to keep waiting \
             (wait_seconds up to {MAX_WAIT}), or call `status` for progress."
        ));
        text
    }

    /// Comments on `document`'s lines, cached against the document and the notes version.
    fn comment_lines(
        &mut self,
        document: &Arc<Decompiled>,
    ) -> &BTreeMap<usize, Vec<(u64, String)>> {
        let key = (
            document.entry,
            self.notes.version(),
            Arc::as_ptr(document) as usize,
        );
        let stale = self
            .comments
            .as_ref()
            .is_none_or(|cached| (cached.entry, cached.notes_version, cached.document) != key);
        if stale {
            let lines = document
                .comment_lines(self.notes.comments())
                .into_iter()
                .map(|(line, found)| {
                    (
                        line,
                        found
                            .into_iter()
                            .map(|(addr, text)| (addr, text.to_owned()))
                            .collect(),
                    )
                })
                .collect();
            self.comments = Some(CommentCache {
                entry: key.0,
                notes_version: key.1,
                document: key.2,
                lines,
            });
        }
        &self.comments.as_ref().expect("just filled").lines
    }

    // Protocol ---------------------------------------------------------------------------

    fn handle_line(&mut self, line: &str) -> Option<Value> {
        match serde_json::from_str::<Value>(line) {
            Ok(value) => self.handle_message(value),
            Err(error) => Some(error_reply(
                Value::Null,
                RpcError::parse(format!("parse error: {error}")),
            )),
        }
    }

    fn handle_message(&mut self, value: Value) -> Option<Value> {
        let Some(message) = value.as_object() else {
            return Some(error_reply(
                Value::Null,
                RpcError::invalid_request("a JSON-RPC message must be an object"),
            ));
        };
        if message.get("jsonrpc").and_then(Value::as_str) != Some("2.0") {
            return Some(error_reply(
                message.get("id").cloned().unwrap_or(Value::Null),
                RpcError::invalid_request("jsonrpc must be \"2.0\""),
            ));
        }
        let Some(method) = message.get("method").and_then(Value::as_str) else {
            return Some(error_reply(
                message.get("id").cloned().unwrap_or(Value::Null),
                RpcError::invalid_request("a request needs a string method"),
            ));
        };
        let params = message.get("params").cloned().unwrap_or(Value::Null);
        let Some(id) = message.get("id") else {
            self.handle_notification(method);
            return None;
        };
        if !(id.is_string() || id.is_u64() || id.is_i64()) {
            return Some(error_reply(
                Value::Null,
                RpcError::invalid_request("id must be a string or an integer"),
            ));
        }
        let id = id.clone();
        match self.handle_request(method, &params) {
            Ok(result) => Some(json!({"jsonrpc": "2.0", "id": id, "result": result})),
            Err(error) => Some(error_reply(id, error)),
        }
    }

    fn handle_notification(&mut self, method: &str) {
        match method {
            // The legacy handshake's completion; this server does not gate requests on it.
            "notifications/initialized" => {}
            "notifications/cancelled" => {}
            other => eprintln!("biara mcp: ignoring notification {other}"),
        }
    }

    fn handle_request(&mut self, method: &str, params: &Value) -> Result<Value, RpcError> {
        let version = params
            .get("_meta")
            .and_then(Value::as_object)
            .and_then(|meta| meta.get("io.modelcontextprotocol/protocolVersion"))
            .and_then(Value::as_str);
        if let Some(version) = version {
            if version != MODERN_VERSION {
                return Err(RpcError::unsupported_version(version));
            }
            // The version selects the era. `clientCapabilities` is part of the revision's
            // required metadata, but this server needs no client capability; a probe that
            // carries only the version is served rather than rejected.
            let result = match method {
                "server/discover" => self.discover(),
                "tools/list" => {
                    let mut result = tools::list_json();
                    if let Some(object) = result.as_object_mut() {
                        object.insert("ttlMs".into(), json!(TTL_MS));
                        object.insert("cacheScope".into(), json!("public"));
                    }
                    Ok(result)
                }
                "tools/call" => self.tools_call(params),
                "initialize" => Err(RpcError::invalid_params(format!(
                    "initialize was removed in {MODERN_VERSION}; call server/discover instead"
                ))),
                other => Err(RpcError::method_not_found(other)),
            }?;
            return Ok(modern_result(result));
        }
        match method {
            "initialize" => self.initialize(params),
            "ping" => Ok(json!({})),
            "tools/list" => Ok(tools::list_json()),
            "tools/call" => self.tools_call(params),
            "server/discover" => Err(RpcError::invalid_params(
                "server/discover needs params._meta[\"io.modelcontextprotocol/protocolVersion\"] \
                 = \"2026-07-28\"",
            )),
            other => Err(RpcError::method_not_found(other)),
        }
    }

    fn initialize(&self, params: &Value) -> Result<Value, RpcError> {
        let requested = params
            .get("protocolVersion")
            .and_then(Value::as_str)
            .ok_or_else(|| RpcError::invalid_params("initialize needs params.protocolVersion"))?;
        if !LEGACY_VERSIONS.contains(&requested) {
            return Err(RpcError {
                code: -32602,
                message: "Unsupported protocol version".into(),
                data: Some(json!({"supported": LEGACY_VERSIONS, "requested": requested})),
            });
        }
        Ok(json!({
            "protocolVersion": requested,
            "capabilities": {"tools": {}},
            "serverInfo": server_info(),
            "instructions": INSTRUCTIONS,
        }))
    }

    fn discover(&mut self) -> Result<Value, RpcError> {
        Ok(json!({
            "supportedVersions": [MODERN_VERSION],
            "capabilities": {"tools": {}},
            "instructions": INSTRUCTIONS,
            "ttlMs": TTL_MS,
            "cacheScope": "public",
        }))
    }

    fn tools_call(&mut self, params: &Value) -> Result<Value, RpcError> {
        let name = params
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| RpcError::invalid_params("tools/call needs a string name"))?;
        let empty = json!({});
        let arguments = match params.get("arguments") {
            None | Some(Value::Null) => &empty,
            Some(arguments) => arguments,
        };
        if !arguments.is_object() {
            return Err(RpcError::invalid_params(
                "tools/call arguments must be an object",
            ));
        }
        let warnings = self.before_call();
        let result = tools::call(self, name, arguments).map_err(|message| {
            RpcError::invalid_params(format!(
                "{message}; call tools/list for the available tools"
            ))
        })?;
        let mut text = result.text;
        if !warnings.is_empty() {
            text = format!("{}\n{text}", warnings.join("\n"));
        }
        Ok(json!({
            "content": [{"type": "text", "text": text}],
            "isError": result.is_error,
        }))
    }
}

fn noop_notify() -> Arc<dyn Fn() + Send + Sync> {
    Arc::new(|| {})
}

fn server_info() -> Value {
    json!({
        "name": "biara",
        "title": "biara binary analysis",
        "version": env!("CARGO_PKG_VERSION"),
    })
}

/// Add the fields every modern result carries: its completion kind and the server's identity.
fn modern_result(mut result: Value) -> Value {
    let Some(object) = result.as_object_mut() else {
        return result;
    };
    object.insert("resultType".into(), json!("complete"));
    let meta = object.entry("_meta").or_insert_with(|| json!({}));
    if let Some(meta) = meta.as_object_mut() {
        meta.insert("io.modelcontextprotocol/serverInfo".into(), server_info());
    }
    result
}

fn error_reply(id: Value, error: RpcError) -> Value {
    let mut body = json!({"code": error.code, "message": error.message});
    if let Some(data) = error.data {
        body["data"] = data;
    }
    json!({"jsonrpc": "2.0", "id": id, "error": body})
}

/// An address as the tools print it.
pub(crate) fn hex(addr: u64) -> String {
    format!("0x{addr:x}")
}
