/* TrustLayer API client.
   All calls talk to the same origin the page was served from, so the app works
   whether you run the FastAPI static server or a Vite dev server with a proxy. */

const API = (() => {
  const base = ""; // same-origin

  async function handle(res) {
    let body = null;
    try { body = await res.json(); } catch (e) { body = null; }
    if (!res.ok) {
      const detail = body && body.detail ? body.detail : res.statusText;
      const msg = Array.isArray(detail)
        ? detail.map(d => d.msg || JSON.stringify(d)).join(", ")
        : detail;
      throw new Error(msg || `Request failed (${res.status})`);
    }
    return body;
  }

  const get = (path) => fetch(base + path).then(handle);

  const postJSON = (path, payload) =>
    fetch(base + path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload || {}),
    }).then(handle);

  const postForm = (path, formData) =>
    fetch(base + path, { method: "POST", body: formData }).then(handle);

  const del = (path) => fetch(base + path, { method: "DELETE" }).then(handle);

  return {
    health: () => get("/api/health"),
    capabilities: () => get("/api/system/capabilities"),
    evaluation: () => get("/api/evaluation"),
    limits: () => get("/api/limits"),

    createInvestigation: (name) => postJSON("/api/investigations", { name }),
    listInvestigations: () => get("/api/investigations"),
    getInvestigation: (id) => get(`/api/investigations/${id}`),
    deleteInvestigation: (id) => del(`/api/investigations/${id}`),

    upload: (id, file) => {
      const fd = new FormData();
      fd.append("file", file);
      return postForm(`/api/investigations/${id}/upload`, fd);
    },
    deleteEvidence: (evidenceId) => del(`/api/evidence/${evidenceId}`),

    analyze: (id) => postJSON(`/api/investigations/${id}/analyze`, {}),
    results: (id) => get(`/api/investigations/${id}/results`),
    graph: (id) => get(`/api/investigations/${id}/graph`),
    evidence: (id) => get(`/api/investigations/${id}/evidence`),

    createSample: () => postJSON("/api/sample", {}),
  };
})();
