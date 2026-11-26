const byId = (id) => document.getElementById(id);
const source = byId("source");
const status = byId("status");
let report;
let active;
let openedFile;
let inspecting = false;
let mainView = "pseudocode";
const inspectionCache = new Map();

const address = (number) => `0x${number.toString(16)}`;
const codeFor = (record) => record.chunks.map((chunk) => chunk.text).join("");

function navigate(location) {
  const match = report.functions.find((record) => record.function.address === location);
  if (match) {
    select(match);
  } else {
    byId("root").value = address(location);
    document.querySelector("form").requestSubmit();
  }
}

function renderCode(record) {
  const code = codeFor(record);
  const tokens = record.function.tokens ?? [];
  if (tokens.length > 50000) {
    source.textContent = code;
    return;
  }
  const bytes = new TextEncoder().encode(code);
  const decoder = new TextDecoder();
  const fragment = document.createDocumentFragment();
  const colors = new Set(["keyword", "number", "punct", "comment", "string", "type"]);
  let end = 0;
  for (const token of tokens) {
    if (token.start < end || token.end > bytes.length || token.end <= token.start) continue;
    const link = token.symbol?.kind === "function" && Number.isSafeInteger(token.symbol.address)
      && token.symbol.address !== record.function.address;
    if (!link && !colors.has(token.kind)) continue;
    fragment.append(decoder.decode(bytes.subarray(end, token.start)));
    const run = document.createElement(link ? "button" : "span");
    run.className = link ? "code-link" : `syntax-${token.kind}`;
    run.textContent = decoder.decode(bytes.subarray(token.start, token.end));
    if (link) {
      run.type = "button";
      run.addEventListener("click", () => navigate(token.symbol.address));
    }
    fragment.append(run);
    end = token.end;
  }
  fragment.append(decoder.decode(bytes.subarray(end)));
  source.replaceChildren(fragment);
}

function renderReferences(context) {
  const relations = byId("relations");
  relations.replaceChildren();
  for (const [title, references] of [["Callers", context.callers], ["Callees", context.callees]]) {
    const heading = document.createElement("h3");
    heading.textContent = title;
    relations.append(heading);
    if (!references?.length) {
      const empty = document.createElement("p");
      empty.textContent = "None recovered";
      relations.append(empty);
    }
    for (const reference of references ?? []) {
      const hasAddress = Number.isSafeInteger(reference.address);
      const label = reference.name || reference.import || (hasAddress ? address(reference.address) : "Unknown target");
      const item = document.createElement(hasAddress && !reference.import ? "button" : "span");
      item.textContent = hasAddress ? `${label} · ${address(reference.address)}` : label;
      if (item.tagName === "BUTTON") {
        item.type = "button";
        item.addEventListener("click", () => navigate(reference.address));
      }
      relations.append(item);
    }
  }
}

function select(record) {
  active = record;
  for (const button of byId("functions").children) {
    button.setAttribute("aria-pressed", String(button.dataset.address === address(record.function.address)));
  }
  byId("title").textContent = record.function.name;
  byId("title").hidden = false;
  renderCode(record);
  byId("result").hidden = false;
  byId("references").hidden = false;
  renderReferences(record.context);
  byId("analysis").hidden = false;
  byId("evidence").textContent = JSON.stringify({
    signature: record.function.signature,
    context: record.context,
    dependencies: record.function.dependencies,
    completeness: report.metadata.completeness,
  }, null, 2);
  byId("copy").disabled = false;
  byId("save").disabled = false;
  showMain("pseudocode");
}

async function requestView(file, params) {
  const response = await fetch(`/decompile?${new URLSearchParams(params)}`, {
    method: "POST",
    headers: { "Content-Type": "application/octet-stream" },
    body: file,
  });
  const result = await response.json().catch(() => null);
  if (!response.ok || !result) {
    throw new Error(result?.error || `Decompilation failed (${response.status}).`);
  }
  return result;
}

async function inspection(view, root = "") {
  const key = `${view}:${root}`;
  if (!inspectionCache.has(key)) {
    if (inspecting) throw new Error("A workbench view is already loading.");
    inspecting = true;
    try {
      inspectionCache.set(key, await requestView(openedFile, { view, root }));
    } finally {
      inspecting = false;
    }
  }
  return inspectionCache.get(key);
}

function drawGraph(analysis) {
  const canvas = byId("graph").querySelector("canvas");
  const fn = analysis.functions.find((item) => item.address === active.function.address);
  const blocks = fn ? new Set(fn.blocks) : new Set();
  const nodes = analysis.blocks.filter((block) => blocks.has(block.address));
  const maxNodes = 100;
  if (nodes.length > maxNodes) {
    throw new Error(`Graph contains ${nodes.length} blocks. Open Disassembly to inspect this function.`);
  }
  const columns = Math.min(5, Math.max(1, Math.ceil(Math.sqrt(nodes.length))));
  const positions = new Map(nodes.map((block, i) =>
    [block.address, [18 + i % columns * 179, 18 + Math.floor(i / columns) * 92]]));
  canvas.width = columns * 179 + 18;
  canvas.height = Math.max(110, Math.ceil(nodes.length / columns) * 92 + 18);
  const ctx = canvas.getContext("2d");
  for (const block of nodes) {
    const [x, y] = positions.get(block.address);
    const targets = new Set([
      ...block.exits.map((exit) => exit.continuation ?? exit.target),
      ...(Array.isArray(block.jump_targets) ? block.jump_targets : []),
    ]);
    for (const destination of targets) {
      if (!positions.has(destination)) continue;
      const [dx, dy] = positions.get(destination);
      const angle = Math.atan2(dy - y - 42, dx - x);
      ctx.strokeStyle = "#7c8cff";
      ctx.beginPath();
      ctx.moveTo(x + 71, y + 42);
      ctx.lineTo(dx + 71, dy);
      ctx.lineTo(dx + 71 - Math.cos(angle - .45) * 8, dy - Math.sin(angle - .45) * 8);
      ctx.moveTo(dx + 71, dy);
      ctx.lineTo(dx + 71 - Math.cos(angle + .45) * 8, dy - Math.sin(angle + .45) * 8);
      ctx.stroke();
    }
  }
  ctx.font = "13px monospace";
  for (const block of nodes) {
    const [x, y] = positions.get(block.address);
    ctx.fillStyle = "#12151c";
    ctx.strokeStyle = "#2a3140";
    ctx.fillRect(x, y, 142, 42);
    ctx.strokeRect(x, y, 142, 42);
    ctx.fillStyle = "#d9dde6";
    ctx.fillText(address(block.address), x + 8, y + 25);
  }
  byId("graph").hidden = false;
}

async function showMain(view) {
  mainView = view;
  for (const button of document.querySelectorAll("[data-view]")) {
    button.setAttribute("aria-pressed", String(button.dataset.view === view));
    button.disabled = !active;
  }
  byId("result").hidden = view !== "pseudocode" || !active;
  byId("inspector").hidden = true;
  byId("graph").hidden = true;
  byId("copy").disabled = byId("save").disabled = view !== "pseudocode" || !active;
  if (!active || view === "pseudocode") return;
  try {
    if (view === "hex") {
      const bytes = new Uint8Array(await openedFile.slice(0, 4096).arrayBuffer());
      if (mainView !== view) return;
      const lines = ["File offsets · first 4 KiB", ""];
      for (let offset = 0; offset < bytes.length; offset += 16) {
        const row = Array.from(bytes.slice(offset, offset + 16));
        lines.push(`${offset.toString(16).padStart(8, "0")}  ${row.map(n => n.toString(16).padStart(2, "0")).join(" ").padEnd(47)}  ${row.map(n => n >= 32 && n < 127 ? String.fromCharCode(n) : ".").join("")}`);
      }
      byId("inspector").textContent = lines.join("\n");
    } else {
      const root = address(active.function.address);
      const result = await inspection(view, root);
      if (mainView !== view || !active || address(active.function.address) !== root) return;
      if (view === "graph") {
        drawGraph(result.analysis);
        return;
      }
      byId("inspector").textContent = result.text;
    }
    byId("inspector").hidden = false;
  } catch (error) {
    if (mainView === view) {
      byId("inspector").textContent = error.message;
      byId("inspector").hidden = false;
    }
  }
}

async function showSide(view) {
  for (const button of document.querySelectorAll("[data-side]")) {
    button.setAttribute("aria-pressed", String(button.dataset.side === view));
    button.disabled = !openedFile;
  }
  const functions = view === "functions";
  byId("functions").hidden = byId("filter").hidden = byId("count").hidden = !functions;
  byId("side-inspector").hidden = functions;
  if (functions) return;
  byId("side-inspector").textContent = "Loading…";
  try {
    const result = await inspection(view);
    if (document.querySelector(`[data-side="${view}"]`).getAttribute("aria-pressed") === "true") {
      byId("side-inspector").textContent = result.text;
    }
  } catch (error) {
    byId("side-inspector").textContent = error.message;
  }
}

for (const button of document.querySelectorAll("[data-view]")) {
  button.addEventListener("click", () => showMain(button.dataset.view));
}
for (const button of document.querySelectorAll("[data-side]")) {
  button.addEventListener("click", () => showSide(button.dataset.side));
}

function renderFunctions() {
  const list = byId("functions");
  list.replaceChildren();
  const filter = byId("filter").value.toLowerCase().trim();
  for (const record of report.functions) {
    const name = record.function.name;
    const location = address(record.function.address);
    if (!name.toLowerCase().includes(filter) && !location.includes(filter)) continue;
    const button = document.createElement("button");
    const label = document.createElement("span");
    const offset = document.createElement("small");
    label.textContent = name;
    offset.textContent = location;
    button.type = "button";
    button.dataset.address = location;
    button.setAttribute("aria-pressed", String(active === record));
    button.append(label, offset);
    button.addEventListener("click", () => select(record));
    list.append(button);
  }
  byId("count").textContent = `${list.children.length} / ${report.functions.length}`;
}

byId("filter").addEventListener("input", renderFunctions);
document.querySelector("form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = byId("binary").files[0];
  if (!file || !file.size || file.size > 8 * 1024 * 1024) {
    status.textContent = "Choose a binary between 1 byte and 8 MiB.";
    return;
  }
  byId("submit").disabled = true;
  byId("filter").disabled = true;
  byId("copy").disabled = true;
  byId("save").disabled = true;
  byId("result").hidden = true;
  byId("references").hidden = true;
  byId("analysis").hidden = true;
  byId("graph").hidden = true;
  byId("inspector").hidden = true;
  byId("functions").replaceChildren();
  byId("title").textContent = "Pseudocode";
  byId("count").textContent = "";
  status.textContent = "Decompiling…";
  try {
    report = await requestView(file, { root: byId("root").value.trim(), format: "json" });
    if (openedFile !== file) inspectionCache.clear();
    openedFile = file;
    showSide("functions");
    active = undefined;
    byId("filter").value = "";
    byId("filter").disabled = !report.functions.length;
    renderFunctions();
    if (report.functions.length) {
      select(report.functions[0]);
      status.textContent = file.name;
    } else {
      status.textContent = "No functions were recovered. Try a known symbol or address.";
    }
  } catch (error) {
    status.textContent = error.message;
  } finally {
    byId("submit").disabled = false;
  }
});

byId("copy").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(source.textContent);
    status.textContent = "Pseudocode copied.";
  } catch {
    status.textContent = "Could not access the clipboard.";
  }
});

byId("save").addEventListener("click", () => {
  const link = document.createElement("a");
  const url = URL.createObjectURL(new Blob([source.textContent], { type: "text/x-c" }));
  link.href = url;
  link.download = `${active.function.name.replace(/[^a-zA-Z0-9_.-]/g, "_")}.c`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
