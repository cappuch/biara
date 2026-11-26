const byId = (id) => document.getElementById(id);
const form = document.querySelector("form");
const source = byId("source");
form.addEventListener("submit", async (event) => {
  event.preventDefault();
  byId("submit").disabled = true;
  byId("result").hidden = true;
  source.textContent = "";
  byId("status").textContent = "Decompiling…";
  try {
    const file = byId("binary").files[0];
    if (!file || file.size > 8 * 1024 * 1024) throw new Error("Choose a binary up to 8 MiB.");
    const query = new URLSearchParams({ root: byId("root").value.trim() });
    const response = await fetch(`/decompile?${query}`, {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream" },
      body: file,
    });
    if (!response.ok) {
      const result = await response.json().catch(() => ({}));
      throw new Error(result.error || `Decompilation failed (${response.status}).`);
    }
    source.textContent = await response.text();
    byId("result").hidden = false;
    byId("status").textContent = "Decompilation complete.";
  } catch (error) {
    byId("status").textContent = error.message;
  } finally {
    byId("submit").disabled = false;
  }
});

byId("save").addEventListener("click", () => {
  const link = document.createElement("a");
  const url = URL.createObjectURL(new Blob([source.textContent], { type: "text/x-c" }));
  link.href = url;
  link.download = "decompiled.c";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
