//! `biara mcp` end to end: the two protocol eras it answers, the tool list, and one agent
//! session that decompiles, renames, comments and is refused a bad name.
#![cfg(all(target_os = "linux", target_arch = "x86_64"))]

use serde_json::{Value, json};
use std::{
    fs,
    io::{BufRead, BufReader, Read, Write},
    path::{Path, PathBuf},
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
    sync::{Arc, LazyLock, Mutex},
    time::Duration,
};

/// `examples/sample`: it has `calc` with parameters `arg0..arg2` and a `main` that calls it.
fn sample() -> PathBuf {
    static SAMPLE: LazyLock<PathBuf> = LazyLock::new(|| {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let built = Command::new("bash")
            .arg(root.join("examples/sample/build.sh"))
            .output()
            .expect("run examples/sample/build.sh");
        assert!(
            built.status.success(),
            "building examples/sample failed: {}",
            String::from_utf8_lossy(&built.stderr)
        );
        let binary = root.join("target/sample/sample");
        assert!(binary.is_file(), "{} was not built", binary.display());
        binary.canonicalize().expect("canonical sample path")
    });
    SAMPLE.clone()
}

fn temp_dir(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("mcp-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

/// The metadata every 2026-07-28 request carries.
fn modern_meta() -> Value {
    json!({
        "io.modelcontextprotocol/protocolVersion": "2026-07-28",
        "io.modelcontextprotocol/clientInfo": {"name": "biara-test", "version": "0"},
        "io.modelcontextprotocol/clientCapabilities": {},
    })
}

struct Client {
    child: Child,
    stdin: Option<ChildStdin>,
    stdout: BufReader<ChildStdout>,
    stderr: Arc<Mutex<String>>,
    next_id: u64,
}

impl Client {
    fn start(binary: &Path, database: &Path, data_home: &Path) -> Client {
        let mut child = Command::new(env!("CARGO_BIN_EXE_biara"))
            .arg("mcp")
            .arg(binary)
            .arg("--database")
            .arg(database)
            .env("XDG_DATA_HOME", data_home)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn biara mcp");
        let stdin = child.stdin.take().expect("piped stdin");
        let stdout = BufReader::new(child.stdout.take().expect("piped stdout"));
        let stderr = Arc::new(Mutex::new(String::new()));
        let mut pipe = child.stderr.take().expect("piped stderr");
        let collected = stderr.clone();
        std::thread::spawn(move || {
            let mut text = String::new();
            let _ = pipe.read_to_string(&mut text);
            collected.lock().unwrap().push_str(&text);
        });
        Client {
            child,
            stdin: Some(stdin),
            stdout,
            stderr,
            next_id: 0,
        }
    }

    fn stderr(&self) -> String {
        self.stderr.lock().unwrap().clone()
    }

    fn send(&mut self, message: Value) {
        let line = serde_json::to_string(&message).unwrap();
        let stdin = self.stdin.as_mut().expect("stdin is open");
        writeln!(stdin, "{line}").unwrap();
        stdin.flush().unwrap();
    }

    fn request(&mut self, method: &str, params: Value) -> Value {
        self.next_id += 1;
        let id = json!(self.next_id);
        self.send(json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}));
        loop {
            let mut line = String::new();
            let read = self.stdout.read_line(&mut line).expect("read a reply");
            assert!(
                read > 0,
                "the server closed stdout while waiting for {id}; stderr:\n{}",
                self.stderr()
            );
            let reply: Value = serde_json::from_str(&line).unwrap_or_else(|error| {
                panic!(
                    "a protocol line was not JSON ({error}): {line:?}\nstderr:\n{}",
                    self.stderr()
                )
            });
            if reply.get("id") == Some(&id) {
                return reply;
            }
            // Notifications carry no id; the server sends none today.
        }
    }

    fn call(&mut self, name: &str, arguments: Value) -> Value {
        let reply = self.request(
            "tools/call",
            json!({"name": name, "arguments": arguments, "_meta": modern_meta()}),
        );
        assert!(
            reply.get("error").is_none(),
            "tools/call {name} was a protocol error: {reply}"
        );
        assert_eq!(reply["result"]["resultType"], "complete");
        assert!(reply["result"]["_meta"]["io.modelcontextprotocol/serverInfo"]["name"].is_string());
        reply["result"].clone()
    }

    fn text(&mut self, name: &str, arguments: Value) -> String {
        let result = self.call(name, arguments);
        assert_eq!(
            result["isError"],
            false,
            "{name} failed: {}",
            result["content"][0]["text"].as_str().unwrap_or("")
        );
        result["content"][0]["text"].as_str().unwrap().to_owned()
    }
}

impl Drop for Client {
    fn drop(&mut self) {
        // Closing stdin is the transport's shutdown signal.
        self.stdin.take();
        for _ in 0..100 {
            match self.child.try_wait() {
                Ok(Some(_)) => return,
                Ok(None) => std::thread::sleep(Duration::from_millis(20)),
                Err(_) => break,
            }
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn only_file(dir: &Path) -> PathBuf {
    let mut files: Vec<_> = fs::read_dir(dir)
        .unwrap_or_else(|error| panic!("reading {}: {error}", dir.display()))
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "json")
        })
        .collect();
    assert_eq!(
        files.len(),
        1,
        "expected one notes file in {}",
        dir.display()
    );
    files.remove(0)
}

#[test]
fn two_protocol_eras_and_one_annotated_session() {
    let binary = sample();
    let root = temp_dir("session");
    let database = root.join("database");
    let data_home = root.join("data");
    fs::create_dir_all(&data_home).unwrap();
    let mut client = Client::start(&binary, &database, &data_home);

    // A 2025-11-25 client: the handshake, then a ping that must be the next reply (the
    // initialized notification gets no answer).
    let initialized = client.request(
        "initialize",
        json!({
            "protocolVersion": "2025-11-25",
            "capabilities": {},
            "clientInfo": {"name": "biara-test", "version": "0"},
        }),
    );
    assert_eq!(initialized["result"]["protocolVersion"], "2025-11-25");
    assert_eq!(initialized["result"]["capabilities"]["tools"], json!({}));
    assert!(
        initialized["result"].get("resultType").is_none(),
        "legacy results do not carry resultType: {initialized}"
    );
    client.send(json!({"jsonrpc": "2.0", "method": "notifications/initialized"}));
    let ping = client.request("ping", json!({}));
    assert_eq!(ping["result"], json!({}));

    // A 2026-07-28 client needs no handshake: every request names its version.
    let discover = client.request("server/discover", json!({"_meta": modern_meta()}));
    assert_eq!(discover["result"]["resultType"], "complete");
    assert_eq!(
        discover["result"]["supportedVersions"],
        json!(["2026-07-28"])
    );
    assert_eq!(
        discover["result"]["_meta"]["io.modelcontextprotocol/serverInfo"]["name"],
        "biara"
    );
    assert!(discover["result"]["capabilities"]["tools"].is_object());
    assert_eq!(discover["result"]["cacheScope"], "public");

    let unsupported = client.request(
        "server/discover",
        json!({"_meta": {"io.modelcontextprotocol/protocolVersion": "1900-01-01"}}),
    );
    assert_eq!(unsupported["error"]["code"], -32022);
    assert_eq!(
        unsupported["error"]["message"],
        "Unsupported protocol version"
    );
    assert_eq!(unsupported["error"]["data"]["requested"], "1900-01-01");
    assert_eq!(
        unsupported["error"]["data"]["supported"],
        json!(["2026-07-28"])
    );

    // The tool list is complete, in a fixed order, and every call needs a schema.
    let listed = client.request("tools/list", json!({"_meta": modern_meta()}));
    let tools = listed["result"]["tools"].as_array().unwrap();
    let names: Vec<&str> = tools
        .iter()
        .map(|tool| tool["name"].as_str().unwrap())
        .collect();
    assert_eq!(
        names,
        [
            "status",
            "open",
            "list_functions",
            "list_imports",
            "list_strings",
            "list_segments",
            "list_structs",
            "function_info",
            "struct_layouts",
            "decompile",
            "disassemble",
            "xrefs_to",
            "xrefs_from",
            "read_bytes",
            "search_bytes",
            "rename_function",
            "rename_data",
            "rename_local",
            "set_signature",
            "declare_struct",
            "struct_from_param",
            "rename_field",
            "set_comment",
            "list_notes",
        ]
    );
    for tool in tools {
        assert_eq!(
            tool["inputSchema"]["type"], "object",
            "{} has no object schema",
            tool["name"]
        );
        assert!(
            tool["description"]
                .as_str()
                .is_some_and(|text| text.len() > 40),
            "{} has no useful description",
            tool["name"]
        );
    }
    assert!(listed["result"]["ttlMs"].is_u64());
    assert_eq!(listed["result"]["cacheScope"], "public");

    // Tools that need the program wait for recovery; one call gives status a ready program.
    let functions = client.text(
        "list_functions",
        json!({"filter": "calc", "wait_seconds": 600}),
    );
    assert!(functions.contains("calc"), "{functions}");

    let status = client.text("status", json!({}));
    assert!(
        status.contains(&format!("binary: {}", binary.display())),
        "{status}"
    );
    assert!(status.contains("arch: x86_64"), "{status}");
    assert!(status.contains("format: ELF"), "{status}");
    assert!(status.contains("entry: 0x"), "{status}");
    assert!(status.contains("analysis: ready"), "{status}");
    assert!(
        status.contains(&data_home.display().to_string()),
        "status must name the notes file: {status}"
    );

    // The pseudocode starts with the signature and the decompiler's parameter names.
    let pseudocode = client.text(
        "decompile",
        json!({"function": "calc", "wait_seconds": 600}),
    );
    assert!(pseudocode.contains("calc("), "{pseudocode}");
    assert!(pseudocode.contains("arg0"), "{pseudocode}");
    assert!(pseudocode.contains("lines 0-"), "{pseudocode}");

    // Renaming a parameter applies without decompiling again and reaches the notes file.
    let renamed = client.text(
        "rename_local",
        json!({"function": "calc", "name": "arg1", "new_name": "lhs", "wait_seconds": 600}),
    );
    assert!(renamed.contains("arg1 is now lhs"), "{renamed}");
    let after = client.text(
        "decompile",
        json!({"function": "calc", "wait_seconds": 600}),
    );
    assert!(after.contains("lhs"), "{after}");
    assert!(!after.contains("arg1"), "{after}");

    let notes_file = only_file(&data_home.join("biara/names"));
    let saved: Value = serde_json::from_str(&fs::read_to_string(&notes_file).unwrap()).unwrap();
    assert!(
        saved["functions"]
            .as_object()
            .unwrap()
            .values()
            .any(|function| function["locals"]["arg1"] == "lhs"),
        "the notes file has no local rename: {saved}"
    );
    assert!(client.text("list_notes", json!({})).contains("arg1 -> lhs"));

    // A comment on an instruction address lands on the pseudocode line that maps it.
    let with_addresses = client.text(
        "decompile",
        json!({"function": "calc", "addresses": true, "limit": 200, "wait_seconds": 600}),
    );
    let address = with_addresses
        .lines()
        .find_map(|line| {
            let (number, rest) = line.split_once(": ")?;
            if !number.bytes().all(|byte| byte.is_ascii_digit()) {
                return None;
            }
            let rest = rest.strip_prefix("0x")?;
            Some(format!("0x{}", rest.split_whitespace().next()?))
        })
        .unwrap_or_else(|| panic!("no line carries an address:\n{with_addresses}"));
    client.text(
        "set_comment",
        json!({"address": address.clone(), "comment": "reads the op", "wait_seconds": 600}),
    );
    let commented = client.text(
        "decompile",
        json!({"function": "calc", "addresses": true, "limit": 200, "wait_seconds": 600}),
    );
    assert!(
        commented
            .lines()
            .any(|line| { line.contains(&address) && line.ends_with(" // reads the op") }),
        "the comment is not on the mapped line:\n{commented}"
    );

    // A renamed function shows through every caller.
    client.text(
        "rename_function",
        json!({"function": "calc", "name": "compute_sum", "wait_seconds": 600}),
    );
    let caller = client.text(
        "decompile",
        json!({"function": "main", "wait_seconds": 600}),
    );
    assert!(caller.contains("compute_sum("), "{caller}");
    let info = client.text(
        "function_info",
        json!({"function": "compute_sum", "wait_seconds": 600}),
    );
    assert!(info.contains("compute_sum"), "{info}");
    assert!(info.contains("arg1 -> lhs"), "{info}");
    assert!(info.contains("reads the op"), "{info}");
    assert!(info.contains("callers: 1"), "main calls it: {info}");

    // A name another local already displays is refused, as a tool error rather than a
    // protocol error.
    let reply = client.call(
        "rename_local",
        json!({"function": "calc", "name": "arg2", "new_name": "lhs", "wait_seconds": 600}),
    );
    assert_eq!(reply["isError"], true, "{reply}");
    let refusal = reply["content"][0]["text"].as_str().unwrap();
    assert!(refusal.contains("already"), "{refusal}");

    // `main` receives a pointer to a recovered four-field struct as parameter 1. The
    // recovered layout is per-decompile; declaring it and typing the parameter keeps it.
    let layouts = client.text(
        "struct_layouts",
        json!({"function": "main", "wait_seconds": 600}),
    );
    assert!(layouts.contains("struct s1 (recovered)"), "{layouts}");
    assert!(layouts.contains("params=[1]"), "{layouts}");
    assert!(layouts.contains("+0x8 bytes=8 name=f_8"), "{layouts}");

    let bound = client.text(
        "struct_from_param",
        json!({"function": "main", "param": 1, "name": "args", "wait_seconds": 600}),
    );
    assert!(bound.contains("struct args *arg1"), "{bound}");
    assert!(bound.contains("4 fields, 32 bytes"), "{bound}");
    let bound_view = client.text(
        "decompile",
        json!({"function": "main", "limit": 300, "wait_seconds": 600}),
    );
    assert!(bound_view.contains("struct args *"), "{bound_view}");
    assert!(bound_view.contains("arg1->f_8"), "{bound_view}");

    let renamed_field = client.text(
        "rename_field",
        json!({"struct": "args", "offset": 8, "new_name": "first", "wait_seconds": 600}),
    );
    assert!(renamed_field.contains("is now first"), "{renamed_field}");
    let renamed_view = client.text(
        "decompile",
        json!({"function": "main", "limit": 300, "wait_seconds": 600}),
    );
    assert!(renamed_view.contains("arg1->first"), "{renamed_view}");
    assert!(!renamed_view.contains("arg1->f_8"), "{renamed_view}");

    let declared = client.text("list_structs", json!({}));
    assert!(
        declared.contains("struct args fields=4 size=32"),
        "{declared}"
    );
    assert!(declared.contains("name=first"), "{declared}");

    // Overlapping fields are rejected before the declaration reaches the notes.
    let reply = client.call(
        "declare_struct",
        json!({
            "name": "bad",
            "fields": [
                {"offset": 0, "bytes": 8, "name": "a"},
                {"offset": 4, "bytes": 8, "name": "b"},
            ],
            "wait_seconds": 600,
        }),
    );
    assert_eq!(reply["isError"], true, "{reply}");
    let overlap = reply["content"][0]["text"].as_str().unwrap();
    assert!(overlap.contains("overlap"), "{overlap}");

    // The struct and the binding signature are in the shared notes file, and list_notes
    // reports them.
    let saved: Value = serde_json::from_str(&fs::read_to_string(&notes_file).unwrap()).unwrap();
    assert_eq!(saved["structs"]["args"]["fields"][1]["name"], "first");
    assert!(
        saved["functions"]
            .as_object()
            .unwrap()
            .values()
            .any(|function| function["signature"]
                .as_str()
                .is_some_and(|signature| signature.contains("struct args *"))),
        "the binding signature is not in the notes file: {saved}"
    );
    let notes = client.text("list_notes", json!({"limit": 100}));
    assert!(notes.contains("struct args fields=4 size=32"), "{notes}");
    assert!(
        notes.contains("field struct=args +0x8 bytes=8 name=first"),
        "{notes}"
    );

    // Bytes and strings round-trip through the same paging and address rules.
    let dump = client.text(
        "read_bytes",
        json!({"address": address.clone(), "length": 16, "wait_seconds": 600}),
    );
    assert!(dump.starts_with(&address), "{dump}");
    let search = client.text(
        "search_bytes",
        json!({"pattern": "?? ??", "limit": 5, "wait_seconds": 600}),
    );
    assert!(
        search.contains("next page") || search.contains("end of list"),
        "{search}"
    );
}
