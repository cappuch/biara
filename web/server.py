"""Serve the Biara browser UI and run bounded native decompilation jobs."""

import argparse
import json
import os
import signal
import subprocess
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

UPLOAD_LIMIT = 8 * 1024 * 1024
OUTPUT_LIMIT = 8 * 1024 * 1024
TIME_LIMIT = 60
ASSETS = {"/": ("index.html", "text/html"), "/app.js": ("app.js", "text/javascript"),
          "/style.css": ("style.css", "text/css")}
INSPECTORS = {"disassembly": "disassemble", "imports": "list_imports",
              "strings": "list_strings", "segments": "list_segments"}


class Handler(BaseHTTPRequestHandler):
    def setup(self):
        self.request.settimeout(15)
        super().setup()

    def reply(self, status, content, content_type="text/plain; charset=utf-8"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(content)

    def error(self, status, message):
        self.reply(status, json.dumps({"error": message}).encode(), "application/json")

    def do_GET(self):
        if self.path == "/health":
            self.reply(200, b"ok\n")
            return
        asset = ASSETS.get(self.path)
        if asset is None:
            self.error(404, "Page not found.")
            return
        name, kind = asset
        self.reply(200, (Path(__file__).parent / name).read_bytes(), kind + "; charset=utf-8")

    def do_POST(self):
        url = urlsplit(self.path)
        if url.path != "/decompile":
            self.error(404, "Endpoint not found.")
            return
        query = parse_qs(url.query)
        root = query.get("root", [""])[0].strip()
        view = query.get("view", ["pseudocode"])[0]
        if view not in (*INSPECTORS, "pseudocode", "graph"):
            self.error(400, "Unknown workbench view.")
            return
        if view in ("disassembly", "graph") and not root:
            self.error(400, "Select a function first.")
            return
        structured = view != "pseudocode" or query.get("format", ["text"])[0] == "json"
        if len(root) > 256 or root.startswith("-") or (root and not root.isprintable()):
            self.error(400, "Use a function name or address.")
            return
        if self.headers.get_content_type() != "application/octet-stream":
            self.error(415, "Send the binary as application/octet-stream.")
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            size = 0
        if not 0 < size <= UPLOAD_LIMIT:
            self.error(413, "Choose a binary between 1 byte and 8 MiB.")
            return
        if not self.server.slot.acquire(blocking=False):
            self.error(503, "A decompilation is already running. Try again shortly.")
            return
        try:
            self.decompile(size, root, structured, view)
        except (TimeoutError, subprocess.TimeoutExpired):
            self.error(408, "The upload or decompilation timed out.")
        except OSError:
            self.error(500, "The decompiler could not start.")
        finally:
            self.server.slot.release()

    def decompile(self, size, root, structured, view):
        with tempfile.TemporaryDirectory(prefix="biara-web-") as directory:
            work = Path(directory)
            binary = work / "input"
            with binary.open("wb") as output:
                remaining = size
                while remaining:
                    chunk = self.rfile.read(min(remaining, 65536))
                    if not chunk:
                        self.error(400, "The upload ended early.")
                        return
                    output.write(chunk)
                    remaining -= len(chunk)
            limits = ["--jobs", "1", "--max-blocks", "2000", "--memory-mib", "2048",
                      "--database", str(work / "analysis.db")]
            if view in INSPECTORS:
                args = [self.server.biara, "mcp", str(binary), *limits]
                options = {"wait_seconds": 45}
                if view == "disassembly":
                    options.update(address_or_function=root, count=100)
                else:
                    options["limit"] = 100
                requests = [
                    {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                     "params": {"protocolVersion": "2025-11-25", "capabilities": {},
                                "clientInfo": {"name": "biara-workbench", "version": "1"}}},
                    {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                     "params": {"name": INSPECTORS[view], "arguments": options}},
                ]
                (work / "requests").write_text("".join(json.dumps(request) + "\n" for request in requests))
            else:
                args = [self.server.biara, "lift", str(binary)]
                if root:
                    args.append(root)
                args += ["--emit", "cfg" if view == "graph" else "decomp", *limits]
                if structured:
                    args += ["--format", "json"]
                (work / "requests").touch()
            with ((work / "requests").open("rb") as stdin,
                  (work / "output").open("w+b") as stdout,
                  (work / "errors").open("w+b") as stderr):
                with subprocess.Popen(args, stdin=stdin, stdout=stdout,
                                      stderr=stderr, cwd=work, start_new_session=True) as process:
                    try:
                        returncode = process.wait(timeout=TIME_LIMIT)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        raise
                if returncode != 0:
                    stderr.seek(0)
                    message = stderr.read(16384).decode(errors="replace").splitlines()
                    detail = next((line for line in reversed(message) if line.startswith("Error:")), "")
                    self.error(422, detail.replace(str(binary), "input") or "Decompilation failed.")
                    return
                stdout.seek(0)
                source = stdout.read(OUTPUT_LIMIT + 1)
            if len(source) > OUTPUT_LIMIT:
                self.error(422, "Output exceeds 8 MiB. Select a function to narrow the result.")
                return
            if view in INSPECTORS:
                try:
                    reply = next(response for response in map(json.loads, source.splitlines())
                                 if response.get("id") == 2)
                    result = reply["result"]
                    lines = "\n".join(item["text"] for item in result["content"])
                except (KeyError, ValueError, StopIteration):
                    self.error(502, "The inspector returned an invalid response.")
                    return
                if result.get("isError"):
                    self.error(422, lines.replace(str(binary), "input"))
                    return
                source = json.dumps({"text": lines}).encode()
            self.reply(200, source, "application/json" if structured else "text/plain; charset=utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--biara", type=Path, default=Path("target/release/biara"))
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    with ThreadingHTTPServer(("127.0.0.1", args.port), Handler) as server:
        server.biara = str(args.biara.resolve(strict=True))
        server.slot = threading.BoundedSemaphore(1)
        server.serve_forever()


if __name__ == "__main__":
    main()
