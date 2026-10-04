/* TrustLayer small UI helpers + reusable HTML fragments.
   No framework: just template strings and DOM utilities. */

const UI = (() => {
  const esc = (v) => String(v == null ? "" : v)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");

  const pct = (v) => `${Math.round((Number(v) || 0) * 100)}%`;
  const num = (v, d = 2) => (Number(v) || 0).toFixed(d);

  function fmtBytes(n) {
    n = Number(n) || 0;
    if (n < 1024) return `${n} B`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
    return `${(n / 1024 / 1024).toFixed(1)} MB`;
  }

  function fmtDate(iso) {
    if (!iso) return "—";
    try { return new Date(iso).toLocaleString(); } catch (e) { return iso; }
  }

  function modalityTag(m) {
    return `<span class="tag ${esc(m)}">${esc(String(m).toUpperCase())}</span>`;
  }

  function assessmentTag(a) {
    const cls = a === "AUTHENTIC" ? "green" : a === "SUSPICIOUS" ? "red" : "amber";
    return `<span class="tag ${cls}">${esc(a || "—")}</span>`;
  }

  function severityClass(sev) {
    return `sev-${(sev || "info").toLowerCase()}`;
  }

  function findingHTML(f) {
    return `<div class="finding ${severityClass(f.severity)}">
      ${f.source || f.src ? `<div class="fsrc">${esc(f.source || f.src)}</div>` : ""}
      <div class="ftitle">${esc(f.title)}</div>
      <div class="fdetail">${esc(f.detail)}</div>
    </div>`;
  }

  function barRow(label, value, risk) {
    const v = Math.max(0, Math.min(1, Number(value) || 0));
    return `<div class="bar-row">
      <div class="label">${esc(label)}</div>
      <div class="bar ${risk ? "risk" : ""}" role="meter" aria-label="${esc(label)}"
           aria-valuemin="0" aria-valuemax="1" aria-valuenow="${v.toFixed(2)}"
           title="${(v * 100).toFixed(0)}%"><span style="width:${v * 100}%"></span></div>
      <div class="val">${v.toFixed(2)}</div>
    </div>`;
  }

  function toast(message, type = "") {
    const el = document.getElementById("toast");
    el.textContent = message;
    el.className = `toast ${type}`;
    el.hidden = false;
    clearTimeout(toast._t);
    toast._t = setTimeout(() => { el.hidden = true; }, 4200);
  }

  function setStatus(text, cls = "") {
    const el = document.getElementById("status-pill");
    el.textContent = text;
    el.className = `pill ${cls}`;
  }

  function kv(pairs) {
    const rows = pairs.map(([k, v]) =>
      `<dt>${esc(k)}</dt><dd>${typeof v === "string" ? esc(v) : v}</dd>`).join("");
    return `<dl class="kv">${rows}</dl>`;
  }

  function empty(message, actionHTML = "") {
    return `<div class="card center"><p class="muted">${esc(message)}</p>${actionHTML}</div>`;
  }

  function loading(message = "Working…") {
    return `<div class="loading"><div class="spinner"></div>${esc(message)}</div>`;
  }

  return { esc, pct, num, fmtBytes, fmtDate, modalityTag, assessmentTag,
           findingHTML, barRow, toast, setStatus, kv, empty, loading, severityClass };
})();
