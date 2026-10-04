/* TrustLayer evidence graph renderer.
   A dependency-free circular/force-lite layout in SVG. Evidence nodes are laid
   out on a ring; entity/claim nodes hang near their document. Good enough for a
   prototype and keeps the frontend build-free. */

const Graph = (() => {
  const COLORS = {
    image: "#38bdf8",
    video: "#a78bfa",
    audio: "#2dd4bf",
    document: "#4ade80",
    entity_name: "#fbbf24",
    entity_date: "#fde68a",
    entity_location: "#fb923c",
    claim: "#c7d2fe",
  };
  const EDGE_COLORS = {
    similarity: "#4ade80",
    consistency: "#22d3ee",
    partial_consistency: "#fbbf24",
    contradiction: "#fb7185",
    timestamp_conflict: "#f43f5e",
    missing_alignment: "#fbbf24",
    identity_uncertain: "#a855f7",
    mentions: "#5b6bb0",
    asserts_claim: "#5b6bb0",
    conflict: "#fb7185",
    same_event_time: "#38bdf8",
    different_event_time: "#f97316",
    corroborates: "#34d399",
  };

  function render(container, graph) {
    if (!graph || !graph.nodes || graph.nodes.length === 0) {
      container.innerHTML = `<div class="empty muted" style="padding:40px;text-align:center">No graph yet. Run an analysis first.</div>`;
      return;
    }

    const W = container.clientWidth || 900;
    const H = container.clientHeight || 460;
    const cx = W / 2, cy = H / 2;

    const evidenceNodes = graph.nodes.filter(n => n.kind === "evidence");
    const otherNodes = graph.nodes.filter(n => n.kind !== "evidence");

    const pos = {};
    const ringR = Math.min(W, H) * 0.28;
    evidenceNodes.forEach((n, i) => {
      const angle = (Math.PI * 2 * i) / Math.max(evidenceNodes.length, 1) - Math.PI / 2;
      pos[n.id] = {
        x: cx + ringR * Math.cos(angle),
        y: cy + ringR * Math.sin(angle),
      };
    });
    // entity/claim nodes on an outer ring, biased outward from centre
    otherNodes.forEach((n, i) => {
      const angle = (Math.PI * 2 * i) / Math.max(otherNodes.length, 1) + 0.4;
      const r = ringR + Math.min(W, H) * 0.16;
      pos[n.id] = { x: cx + r * Math.cos(angle), y: cy + r * Math.sin(angle) };
    });

    const svg = [];
    svg.push(`<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" role="img" aria-label="Evidence relationship graph">
      <defs>
        <filter id="tlGlow" x="-70%" y="-70%" width="240%" height="240%">
          <feGaussianBlur stdDeviation="6" result="b"/>
          <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
        <radialGradient id="tlCore"><stop offset="0%" stop-color="#fff" stop-opacity=".95"/>
          <stop offset="100%" stop-color="#fff" stop-opacity="0"/></radialGradient>
      </defs>`);

    // edges
    graph.edges.forEach(e => {
      const a = pos[e.source], b = pos[e.target];
      if (!a || !b) return;
      const color = EDGE_COLORS[e.relationship] || "#4a5da8";
      const score = Math.max(0, Math.min(1, Number(e.score) || 0));
      const width = e.relationship === "similarity" || e.relationship === "consistency"
      || e.relationship === "partial_consistency" || e.relationship === "corroborates"
        ? 1.2 + score * 3 : 1.8;
      const dash = (e.relationship === "contradiction" || e.relationship.includes("conflict")
      || e.relationship === "different_event_time")
        ? 'stroke-dasharray="6 4"' : "";
      const title = `${e.relationship.replace(/_/g, " ")}${e.score != null ? " · score " + Number(e.score).toFixed(2) : ""}`;
      svg.push(`<line x1="${a.x}" y1="${a.y}" x2="${b.x}" y2="${b.y}"
        stroke="${color}" stroke-width="${width}" stroke-linecap="round" ${dash}
        opacity="${e.relationship === "mentions" || e.relationship === "asserts_claim" ? "0.35" : "0.85"}">
        <title>${UI.esc(title)}</title></line>`);
    });

    // nodes
    graph.nodes.forEach(n => {
      const p = pos[n.id];
      if (!p) return;
      const color = COLORS[n.modality || n.kind] || "#94a3d8";
      const r = n.kind === "evidence" ? 26 : 13;
      const label = String(n.label || "Unknown");
      const short = label.length > 16 ? label.slice(0, 15) + "…" : label;
      const score = Math.max(0, Math.min(1, Number(n.score) || 0));
      const ring = score >= 0.65 ? "#fb7185" : score >= 0.4 ? "#fbbf24" : "#34e5a0";
      svg.push(`<g class="gnode" style="color:${color}">
        ${n.kind === "evidence" ? `<circle cx="${p.x}" cy="${p.y}" r="${r + 12}" fill="${color}" opacity="0.16"/>
        <circle class="gnode-halo" cx="${p.x}" cy="${p.y}" r="${r + 5}" fill="none" stroke="${ring}" stroke-width="2.5" stroke-opacity="0.9" filter="url(#tlGlow)"/>` : ""}
        <circle cx="${p.x}" cy="${p.y}" r="${r}" fill="${color}" fill-opacity="${n.kind === "evidence" ? 0.95 : 0.78}" stroke="#05071a" stroke-width="1.5">
          <title>${UI.esc(label)}${n.kind === "evidence" ? " · " + UI.esc(n.modality) + " · indicator " + score.toFixed(2) + " · " + UI.esc(n.label_text || "") : ""}</title>
        </circle>
        <circle cx="${p.x}" cy="${p.y}" r="${r * 0.42}" fill="url(#tlCore)" opacity="${n.kind === "evidence" ? 0.55 : 0.35}"/>
        <text x="${p.x}" y="${p.y + r + 15}" text-anchor="middle">${UI.esc(short)}</text>
      </g>`);
    });

    svg.push("</svg>");
    container.innerHTML = svg.join("");
  }

  function legendHTML() {
    const items = [
      ["Evidence node", "#38bdf8"],
      ["Entity / claim", "#fbbf24"],
      ["Lower indicator", "#34e5a0"],
      ["Elevated indicator", "#fbbf24"],
      ["High indicator / conflict", "#fb7185"],
      ["Similarity / consistency", "#22d3ee"],
      ["Mentions", "#5b6bb0"],
    ];
    return `<div class="legend">${items.map(([t, c]) =>
      `<span><i style="background:${c};color:${c}"></i>${t}</span>`).join("")}</div>`;
  }

  return { render, legendHTML };
})();
