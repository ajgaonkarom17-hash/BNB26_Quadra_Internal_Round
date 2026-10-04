/* TrustLayer page views. Each view renders HTML into #view and wires events. */

const Views = (() => {
  const PIPELINE_STEPS = [
    "Image / media extraction",
    "Video frame extraction",
    "Audio feature extraction",
    "Document extraction",
    "Individual analysis",
    "Embedding generation",
    "Cross-modal analysis",
    "Evidence graph",
    "Uncertainty analysis",
    "Final assessment",
  ];

  /* ------------------------------------------------------------------ Home */
  async function home(view) {
    view.innerHTML = `
      <div class="hero">
        <div class="kicker">AI-Powered Digital Authenticity &amp; Trust</div>
        <h1>Do these pieces of evidence tell <span class="grad-text">one consistent story</span>?</h1>
        <p class="lead">
          TrustLayer analyzes multiple digital artifacts, checks each one on its own,
          then reasons about the <strong>relationships between them</strong>. Individual
          files can each look plausible — but inconsistencies across files reveal manipulation.
        </p>
        <div class="flow">
          <span class="chip image">Image</span><span class="arrow">+</span>
          <span class="chip video">Video</span><span class="arrow">+</span>
          <span class="chip audio">Audio</span><span class="arrow">+</span>
          <span class="chip document">Document</span>
          <span class="arrow">→</span>
          <span class="chip">Individual analysis</span>
          <span class="arrow">→</span>
          <span class="chip">Common representation</span>
          <span class="arrow">→</span>
          <span class="chip">Multimodal fusion</span>
          <span class="arrow">→</span>
          <span class="chip">Cross-modal reasoning</span>
          <span class="arrow">→</span>
          <span class="chip">Evidence graph</span>
          <span class="arrow">→</span>
          <span class="chip">Assessment + explanation</span>
        </div>
        <div class="stat-strip">
          <div class="mini-stat"><span class="ms-label">Modalities</span><span class="ms-value">4</span></div>
          <div class="mini-stat"><span class="ms-label">Pipeline stages</span><span class="ms-value">10</span></div>
          <div class="mini-stat"><span class="ms-label">Upload location</span><span class="ms-value" style="font-size:15px">Local</span></div>
          <div class="mini-stat"><span class="ms-label">Verdicts</span><span class="ms-value">3</span></div>
        </div>
        <div style="display:flex;gap:12px;flex-wrap:wrap;margin-top:20px">
          <button class="btn" id="home-new">Create Investigation</button>
          <button class="btn secondary" id="home-sample">Run Sample Investigation</button>
        </div>
      </div>

      <div class="grid cols-3">
        <div class="card feature f1">
          <div class="ficon">🔍</div>
          <h3>① Individual authenticity</h3>
          <p>Each file gets an independent heuristic analysis: image ELA/noise, video
          frame temporal analysis, audio spectrogram features, document entities.</p>
        </div>
        <div class="card feature f2">
          <div class="ficon">🔗</div>
          <h3>② Cross-modal consistency</h3>
          <p>Pairwise reasoning compares timestamps, durations, entities and shared
          embeddings to surface <em>conflicts</em> between evidence.</p>
        </div>
        <div class="card feature f3">
          <div class="ficon">⚖</div>
          <h3>③ Honest uncertainty</h3>
          <p>Coverage, confidence and missing evidence are reported separately.
          When evidence is thin, the result is INCONCLUSIVE — never forced.</p>
        </div>
      </div>
    `;
    view.querySelector("#home-new").onclick = () => TrustLayer.go("#/new");
    view.querySelector("#home-sample").onclick = () => runSample(view);
  }

  async function runSample(view) {
    TrustLayer.setStatus("Building sample…", "busy");
    try {
      const res = await API.createSample();
      TrustLayer.toast("Sample investigation analyzed.", "ok");
      TrustLayer.go(`#/results/${res.investigation.investigation_id}`);
    } catch (e) {
      UI.toast(`Sample failed: ${e.message}`, "err");
      TrustLayer.setStatus("Error", "warn");
    }
  }

  /* ----------------------------------------------------------- New / upload */
  async function newInvestigation(view, params) {
    let invId = params && params.id ? params.id : TrustLayer.state.draftId;
    let inv = null;

    if (invId) {
      try { inv = (await API.getInvestigation(invId)).investigation; }
      catch (e) { invId = null; }
    }

    view.innerHTML = `
      <h1>New Investigation</h1>
      <p class="lead">Name the investigation, then add any combination of image,
      video, audio and document files. You can analyze with as few as one file, but
      cross-modal reasoning needs more than one modality.</p>

      <div class="card">
        <label class="field">
          <span>Investigation name</span>
          <input type="text" id="inv-name" placeholder="e.g. Event Authenticity Test"
                 value="${UI.esc(inv ? inv.name : "")}" ${inv ? "disabled" : ""} />
        </label>
        <div id="create-row">
          ${inv ? "" : `<button class="btn" id="create-btn">Create Investigation</button>`}
        </div>
        <div id="inv-meta" class="${inv ? "" : "hidden"}">
          ${inv ? `<p class="muted mono" style="margin:0">ID: ${UI.esc(inv.id)} · created ${UI.fmtDate(inv.created_at)}</p>` : ""}
        </div>
      </div>

      <div id="upload-section" class="${inv ? "" : "hidden"}">
        <div class="card" style="margin-top:18px">
          <h2 style="margin-top:0">Build one multimodal investigation</h2>
          <p class="muted" style="margin-top:-6px">Add evidence for as many modalities as
          you can. TrustLayer analyses each file, then reasons <strong>across</strong> them.
          Two or more modalities enable cross-modal reasoning.</p>
          <div class="modality-slots" id="modality-slots"></div>
          <div class="dropzone" id="dropzone">
            <div class="dz-icon">⬆</div>
            <strong>Drop files here or click to browse</strong>
            <span class="muted">Image · Video · Audio · Document (TXT, PDF)</span>
            <input type="file" id="file-input" multiple hidden
                   accept="image/*,video/*,audio/*,.txt,.pdf,.md,.csv,.json,.log" />
          </div>
          <div id="file-list" style="margin-top:16px"></div>
        </div>
        <div style="display:flex;gap:12px;margin-top:18px;flex-wrap:wrap;align-items:center">
          <button class="btn" id="analyze-btn" disabled>Analyze Evidence</button>
          <button class="btn secondary" id="back-history">Back to History</button>
          <span class="muted" id="multimodal-hint"></span>
        </div>
      </div>
    `;

    if (!inv) {
      view.querySelector("#create-btn").onclick = async () => {
        const name = view.querySelector("#inv-name").value.trim();
        if (!name) { UI.toast("Please enter a name.", "err"); return; }
        try {
          const created = await API.createInvestigation(name);
          TrustLayer.state.draftId = created.id;
          TrustLayer.toast("Investigation created.", "ok");
          TrustLayer.refresh(`#/new/${created.id}`, true);
        } catch (e) { UI.toast(e.message, "err"); }
      };
      return;
    }

    const fileList = view.querySelector("#file-list");
    const slotsEl = view.querySelector("#modality-slots");
    const hintEl = view.querySelector("#multimodal-hint");
    const analyzeBtn = view.querySelector("#analyze-btn");
    let uploaded = [];

    const MODALITIES = ["image", "video", "audio", "document"];
    const MOD_ICONS = { image: "🖼", video: "🎬", audio: "🔊", document: "📄" };

    function renderSlots() {
      const present = new Set(uploaded.map(ev => ev.modality));
      slotsEl.innerHTML = MODALITIES.map(m => {
        const count = uploaded.filter(ev => ev.modality === m).length;
        const on = count > 0;
        return `<div class="mslot ${on ? "on " + m : ""}">
          <span class="micon">${MOD_ICONS[m]}</span>
          <span class="mlabel">${m.toUpperCase()}</span>
          <span class="mcount">${on ? count + " file" + (count > 1 ? "s" : "") : "empty"}</span>
        </div>`;
      }).join(`<div class="mplus">+</div>`) + `<div class="meq">→</div>
        <div class="mslot result ${present.size >= 2 ? "on" : ""}">
          <span class="micon">🧩</span>
          <span class="mlabel">FUSION</span>
          <span class="mcount">${present.size} modalit${present.size === 1 ? "y" : "ies"}</span>
        </div>`;

      if (present.size >= 2) {
        hintEl.innerHTML = `<span class="tag green">Multimodal ready</span>
          ${present.size} modalities will be fused and cross-checked.`;
      } else if (present.size === 1) {
        hintEl.innerHTML = `<span class="tag amber">Single modality</span>
          Add another modality to enable cross-modal reasoning.`;
      } else {
        hintEl.textContent = "";
      }
    }

    async function refreshFiles() {
      try {
        const data = await API.evidence(invId);
        uploaded = data.evidence || [];
      } catch (e) { uploaded = []; }
      if (uploaded.length === 0) {
        fileList.innerHTML = `<p class="muted">No files yet.</p>`;
      } else {
        fileList.innerHTML = uploaded.map(ev => `
          <div class="file-row">
            ${UI.modalityTag(ev.modality)}
            <span class="fname">${UI.esc(ev.filename)}</span>
            <span class="fsize">${UI.fmtBytes(ev.size_bytes)}</span>
            <button class="btn ghost small" data-del="${UI.esc(ev.id)}">Remove</button>
          </div>
        `).join("");
        fileList.querySelectorAll("[data-del]").forEach(btn => {
          btn.onclick = async () => {
            try { await API.deleteEvidence(btn.dataset.del); refreshFiles(); }
            catch (e) { UI.toast(e.message, "err"); }
          };
        });
      }
      analyzeBtn.disabled = uploaded.length === 0;
      analyzeBtn.textContent = uploaded.length
        ? `Analyze Evidence (${uploaded.length})` : "Analyze Evidence";
      renderSlots();
    }

    async function handleFiles(fileListObj) {
      const files = Array.from(fileListObj);
      if (!files.length) return;
      TrustLayer.setStatus(`Uploading ${files.length} file(s)…`, "busy");
      for (const f of files) {
        try {
          await API.upload(invId, f);
        } catch (e) {
          UI.toast(`${f.name}: ${e.message}`, "err");
        }
      }
      TrustLayer.setStatus("Ready", "ok");
      refreshFiles();
    }

    const dz = view.querySelector("#dropzone");
    const input = view.querySelector("#file-input");
    dz.onclick = () => input.click();
    input.onchange = () => handleFiles(input.files);
    dz.addEventListener("dragover", e => { e.preventDefault(); dz.classList.add("drag"); });
    dz.addEventListener("dragleave", () => dz.classList.remove("drag"));
    dz.addEventListener("drop", e => {
      e.preventDefault(); dz.classList.remove("drag");
      handleFiles(e.dataTransfer.files);
    });

    analyzeBtn.onclick = () => TrustLayer.go(`#/processing/${invId}`);
    view.querySelector("#back-history").onclick = () => TrustLayer.go("#/history");

    refreshFiles();
  }

  /* ----------------------------------------------------------- Processing */
  async function processing(view, params) {
    const invId = params.id;
    view.innerHTML = `
      <h1>Analysis Processing</h1>
      <p class="lead">Running the TrustLayer pipeline. This can take a few seconds
      depending on media size.</p>
      <div class="card">
        <ul class="pipeline" id="pipeline">
          ${PIPELINE_STEPS.map((s, i) => `<li data-step="${i}">
            <span class="pstep">${i + 1}</span><span>${UI.esc(s)}</span></li>`).join("")}
        </ul>
      </div>`;

    const items = view.querySelectorAll("#pipeline li");
    let current = 0;
    const timer = setInterval(() => {
      if (current < items.length) {
        items[current].classList.add("active");
        current++;
      }
    }, 240);

    TrustLayer.setStatus("Analyzing…", "busy");
    try {
      const result = await API.analyze(invId);
      clearInterval(timer);
      items.forEach(li => { li.classList.remove("active"); li.classList.add("done"); });
      TrustLayer.state.lastAnalysis = result;
      TrustLayer.toast("Analysis complete.", "ok");
      TrustLayer.setStatus("Analyzed", "ok");
      setTimeout(() => TrustLayer.go(`#/results/${invId}`), 350);
    } catch (e) {
      clearInterval(timer);
      items.forEach(li => li.classList.remove("active"));
      TrustLayer.setStatus("Error", "warn");
      view.insertAdjacentHTML("beforeend", `
        <div class="card accent-red" style="margin-top:18px">
          <h3>Analysis failed</h3>
          <p>${UI.esc(e.message)}</p>
          <button class="btn secondary" onclick="TrustLayer.go('#/new/${UI.esc(invId)}')">
            Back to upload</button>
        </div>`);
    }
  }

  /* -------------------------------------------------------------- Results */
  function multimodalBanner(f, cross, u) {
    const MODS = ["image", "video", "audio", "document"];
    const ICONS = { image: "🖼", video: "🎬", audio: "🔊", document: "📄" };
    const present = new Set(f.available_modalities);
    const individual = f.breakdown.individual_risk ?? 0;
    const crossRisk = f.breakdown.cross_modal_risk ?? 0;
    const override = f.breakdown.conflict_override ?? 0;

    const slots = MODS.map(m => {
      const on = present.has(m);
      return `<div class="mslot ${on ? "on " + m : ""}">
        <span class="micon">${ICONS[m]}</span>
        <span class="mlabel">${m.toUpperCase()}</span>
        <span class="mcount">${on ? "included" : "missing"}</span>
      </div>`;
    }).join(`<div class="mplus">+</div>`);

    const verdict = override > 0.02
      ? `<span class="tag red">Cross-modal conflict escalated the assessment</span>`
      : crossRisk < 0.4
        ? `<span class="tag green">Modalities agree</span>`
        : `<span class="tag amber">Partial agreement</span>`;

    return `<div class="card accent-violet" style="margin-bottom:18px">
      <div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:10px;align-items:center">
        <h3 style="margin:0">Multimodal evidence → one decision</h3>
        <span class="tag neutral">Multimodal AI</span>
      </div>
      <div class="modality-slots" style="margin:14px 0 12px">
        ${slots}<div class="meq">→</div>
        <div class="mslot result on">
          <span class="micon">🧩</span>
          <span class="mlabel">FUSION</span>
          <span class="mcount">${present.size} modalit${present.size === 1 ? "y" : "ies"}</span>
        </div>
      </div>
      <div class="grid cols-3" style="gap:12px">
        <div class="mini-stat">
          <span class="ms-label">Individual risk</span>
          <span class="ms-value">${individual.toFixed(2)}</span>
        </div>
        <div class="mini-stat">
          <span class="ms-label">Cross-modal risk</span>
          <span class="ms-value">${crossRisk.toFixed(2)}</span>
        </div>
        <div class="mini-stat">
          <span class="ms-label">Cross-modal conflict bonus</span>
          <span class="ms-value">+${override.toFixed(2)}</span>
        </div>
      </div>
      <div style="margin-top:12px;display:flex;gap:10px;flex-wrap:wrap;align-items:center">
        ${verdict}
        <span class="muted" style="font-size:12.5px">
          ${cross.pairs_compared} pair comparison(s), ${cross.conflict_count} conflict flag(s).
          ${present.size < 2 ? "Only one modality — cross-modal reasoning unavailable." : ""}
        </span>
      </div>
    </div>`;
  }

  async function results(view, params) {
    const invId = params.id;
    view.innerHTML = UI.loading("Loading results…");
    let data;
    try {
      data = await API.results(invId);
      TrustLayer.state.lastAnalysis = data;
    } catch (e) {
      view.innerHTML = UI.empty(`Could not load results: ${e.message}`);
      return;
    }

    const u = data.uncertainty;
    const f = data.fusion;
    const ex = data.explanation;
    const assessment = u.final_assessment;

    const parts = [];

    // multimodal banner — the core idea of the project
    parts.push(multimodalBanner(f, data.cross_modal.summary, u));

    // assessment banner
    parts.push(`
      <div class="assessment-card border-${assessment} card">
        <div>
          <div class="assessment-label">Overall Assessment</div>
          <div class="assessment-value assess-${assessment}">${UI.esc(assessment)}</div>
          <div class="assessment-sub">
            ${u.downgraded
              ? "Downgraded from " + UI.esc(u.pre_fusion_assessment) + " due to insufficient / conflicting evidence."
              : "Based on " + f.available_modalities.length + " modality(ies) and " +
                data.cross_modal.summary.pairs_compared + " cross-modal comparison(s)."}
          </div>
        </div>
        <div>
          <div class="assessment-label">Confidence</div>
          <div class="big-metric">${UI.esc(f.confidence_label)}</div>
          <div class="assessment-sub">${UI.pct(f.confidence)} · uncertainty ${UI.esc(u.uncertainty)}</div>
        </div>
        <div>
          <div class="assessment-label">Evidence Coverage</div>
          <div class="big-metric">${UI.pct(u.coverage)}</div>
          <div class="assessment-sub">${f.available_modalities.length} of 4 modalities</div>
        </div>
      </div>
    `);

    // top reasons
    if (ex.top_reasons.length) {
      parts.push(`<div class="card" style="margin-top:18px">
        <h3>Why was this flagged?</h3>
        ${ex.why_flagged.slice(0, 5).map(UI.findingHTML).join("")}
      </div>`);
    }

    // mode / mask note
    parts.push(`<div class="card accent-cyan" style="margin-top:18px">
      <div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:10px;align-items:center">
        <h3 style="margin:0">Modality coverage &amp; fusion inputs</h3>
        <span class="tag neutral">${UI.esc(f.engine)}</span>
      </div>
      <div style="margin-top:10px">
        ${["image","video","audio","document"].map((m,i) => {
          const on = f.available_modalities.includes(m);
          return `<span class="tag ${on ? m : "neutral"}" style="margin-right:6px;${on?"":"opacity:.5"}">
            ${on ? "✓" : "✗"} ${m.toUpperCase()}</span>`;
        }).join("")}
        <span class="tag neutral mono" style="margin-left:8px">mask [${(data.cross_modal && data.individual) ? ["image","video","audio","document"].map(m => f.available_modalities.includes(m)?1:0).join(",") : ""}]</span>
      </div>
      <div style="margin-top:14px">
        ${UI.barRow("Individual risk", f.breakdown.individual_risk, true)}
        ${UI.barRow("Cross-modal risk", f.breakdown.cross_modal_risk, true)}
        ${UI.barRow("Consistency", f.consistency, false)}
        ${UI.barRow("Coverage", u.coverage, false)}
      </div>
    </div>`);

    // ---- cross-modal findings (first-class section) -------------------
    {
      const comps = data.cross_modal.comparisons || [];
      const rows = [];
      comps.forEach(c => {
        const [a, b] = c.pair.split("_");
        (c.conflicts || []).forEach(cf => {
          rows.push(`<div class="finding ${cf.type === "timestamp_missing" || cf.type.endsWith("_unavailable") ? "sev-low" : "sev-high"}">
            <div class="ftitle">⚠ ${UI.esc(cf.type.replace(/_/g, " ").toUpperCase())}</div>
            <div class="fdetail">${UI.esc(cf.description)}</div></div>`);
        });
        const agrees = (c.breakdown && c.breakdown.claim_agreements) || [];
        agrees.forEach(d => {
          rows.push(`<div class="finding sev-info">
            <div class="ftitle">✓ CONSISTENCY</div>
            <div class="fdetail">${UI.esc(d)}</div></div>`);
        });
      });
      const consistentPairs = comps.filter(c => c.label === "CONSISTENT");
      consistentPairs.forEach(c => {
        rows.push(`<div class="finding sev-info">
          <div class="ftitle">✓ CONSISTENT</div>
          <div class="fdetail">${UI.esc(c.pair.replace("_", " ↔ "))} (heuristic score ${Number(c.consistency).toFixed(2)})</div></div>`);
      });
      parts.push(`<h2>Cross-Modal Findings</h2><div class="card">${
        rows.length ? rows.join("") :
        '<p class="muted">No cross-modal findings — a single modality cannot be cross-checked.</p>'
      }</div>`);
    }

    // individual analysis
    parts.push(`<h2>Individual Modality Analysis</h2><div class="grid cols-2">`);
    Object.entries(data.individual).forEach(([m, a]) => {
      const findings = (a.findings || []).map(UI.findingHTML).join("");
      const detail = a.detail || {};
      const extra = [];
      if (m === "image" && detail.exif && Object.keys(detail.exif).length) {
        extra.push(`<details><summary>Metadata (${Object.keys(detail.exif).length} EXIF fields)</summary>
          ${UI.kv(Object.entries(detail.exif).slice(0, 12).map(([k,v]) => [k, String(v)]))}</details>`);
      }
      if (m === "document" && detail.entities) {
        const e = detail.entities;
        extra.push(`<details><summary>Extracted entities</summary>${UI.kv([
          ["Dates", (e.dates||[]).join(", ") || "—"],
          ["Names", (e.names||[]).join(", ") || "—"],
          ["Locations", (e.locations||[]).join(", ") || "—"],
        ])}</details>`);
      }
      parts.push(`<div class="card accent-${UI.esc(m)}">
        <div style="display:flex;justify-content:space-between;align-items:center;gap:10px">
          <h3 style="margin:0">${UI.modalityTag(m)} <span class="muted" style="font-size:12px">${UI.esc(a.evidence_id ? (a.filename||"") : "")}</span></h3>
          ${UI.assessmentTag(a.label)}
        </div>
        <div style="margin:12px 0 6px">
          ${UI.barRow("Manipulation indicator", a.score, true)}
        </div>
        <span class="tag neutral">${UI.esc(a.engine)}</span>
        <div style="margin-top:12px">${findings || '<p class="muted">No findings.</p>'}</div>
        ${extra.join("")}
      </div>`);
    });
    parts.push(`</div>`);

    // cross-modal
    const comps = data.cross_modal.comparisons;
    parts.push(`<h2>Cross-Modal Consistency <span class="muted" style="font-size:13px;font-weight:400">Heuristic consistency score (not a probability)</span></h2>`);
    if (!comps.length) {
      parts.push(`<div class="card"><p class="muted">No cross-modal comparisons were possible — only one modality is available.</p></div>`);
    } else {
      parts.push(`<div class="card">`);
      comps.forEach(c => {
        const cls = c.label === "CONSISTENT" ? "green" : c.label === "INCONSISTENT" ? "red" : "amber";
        const [a, b] = c.pair.split("_");
        parts.push(`<div class="bar-row">
          <div class="label">${UI.modalityTag(a)} ↔ ${UI.modalityTag(b)}</div>
          <div class="bar ${c.label === "INCONSISTENT" ? "risk" : ""}"><span style="width:${c.consistency*100}%"></span></div>
          <div class="val">${Number(c.consistency).toFixed(2)}</div>
          <div style="width:110px;text-align:right"><span class="tag ${cls}">${UI.esc(c.label)}</span></div>
        </div>`);
        if (c.conflicts && c.conflicts.length) {
          parts.push(`<div style="margin:2px 0 12px 142px">
            ${c.conflicts.map(cf => `<div class="muted" style="font-size:12.5px">• ${UI.esc(cf.description)}</div>`).join("")}
          </div>`);
        }
      });
      parts.push(`</div>`);
    }

    // video timeline
    if (data.video_timeline && (data.video_timeline.duration || 0) > 0) {
      parts.push(`<h2>Video Timeline — Suspicious Frame / Timestamp Indicators</h2>`);
      parts.push(renderTimeline(data.video_timeline));
    }

    // what changed
    parts.push(`<h2>What Changed?</h2>`);
    parts.push(`<div class="card"><div class="grid cols-3">${
      ex.what_changed.map(w => `<div>
        <h3>${UI.esc(w.aspect)}</h3>
        <div style="font-size:13.5px;font-weight:600;color:${
          w.status.includes("Conflict") || w.status.includes("alteration") ? "var(--amber)" :
          w.status.includes("No significant") ? "var(--green)" : "var(--text-dim)"
        }">→ ${UI.esc(w.status)}</div>
        <p class="muted" style="font-size:12.5px;margin-top:4px">${UI.esc(w.detail)}</p>
      </div>`).join("")
    }</div></div>`);

    // explanation columns
    parts.push(`<h2>Explanation</h2><div class="grid cols-2">`);
    parts.push(`<div class="card accent-red"><h3 style="color:var(--red)">⚠ Why flagged</h3>
      ${ex.why_flagged.length ? ex.why_flagged.map(UI.findingHTML).join("") : '<p class="muted">Nothing strongly flagged.</p>'}</div>`);
    parts.push(`<div class="card accent-green"><h3 style="color:var(--green)">✓ Supporting evidence</h3>
      ${ex.supporting.length ? ex.supporting.map(UI.findingHTML).join("") : '<p class="muted">No strong supporting signals.</p>'}</div>`);
    parts.push(`<div class="card accent-amber"><h3 style="color:var(--amber)">✕ Conflicting evidence</h3>
      ${ex.conflicting.length ? ex.conflicting.map(UI.findingHTML).join("") : '<p class="muted">No cross-modal conflicts detected.</p>'}</div>`);
    parts.push(`<div class="card accent-cyan"><h3>○ Missing evidence</h3>
      ${ex.missing.length ? ex.missing.map(UI.findingHTML).join("") : '<p class="muted">No missing modalities.</p>'}
      ${u.low_quality.length ? `<h3 style="margin-top:12px">Low-quality media</h3>` +
        u.low_quality.map(q => `<div class="muted" style="font-size:12.5px">• ${UI.esc(q.modality)}: ${UI.esc(q.note)}</div>`).join("") : ""}
    </div>`);
    parts.push(`</div>`);

    // graph preview + actions
    parts.push(`<h2>Evidence Graph</h2>
      <div class="card">
        <div class="graph-wrap" id="graph-preview"></div>
        ${Graph.legendHTML()}
        <div style="margin-top:14px;display:flex;gap:10px;flex-wrap:wrap">
          <button class="btn secondary" onclick="TrustLayer.go('#/graph/${UI.esc(invId)}')">Open full graph</button>
          <button class="btn ghost" onclick="TrustLayer.go('#/history')">Back to History</button>
        </div>
      </div>`);

    view.innerHTML = parts.join("");

    // render graph preview
    try {
      const g = await API.graph(invId);
      const el = document.getElementById("graph-preview");
      if (el) Graph.render(el, g.graph);
    } catch (e) { /* non-fatal */ }
  }

  function renderTimeline(tl) {
    const duration = Number(tl.duration) || 1;
    const markers = (tl.suspicious || []).map(s => {
      const left = Math.max(0, Math.min(100, (Number(s.timestamp) / duration) * 100));
      return `<div class="marker" style="left:${left}%" title="${UI.esc(s.reason)} (indicator ${s.indicator_score})">
        <div class="dot"></div><div class="ts">⚠ ${fmtTime(s.timestamp)}</div></div>`;
    }).join("");
    return `<div class="card">
      <div class="timeline">
        <div class="track"></div>
        ${markers}
        <div class="labels"><span>00:00</span><span>${fmtTime(duration)}</span></div>
      </div>
      ${(tl.suspicious && tl.suspicious.length)
        ? `<div style="margin-top:34px">${
            tl.suspicious.map(s => `<div class="finding sev-medium">
              <div class="ftitle">${fmtTime(s.timestamp)} — ${UI.esc(s.reason)}</div>
              <div class="fdetail">Indicator score ${Number(s.indicator_score).toFixed(2)} · frame ${s.frame_index}.
              These are indicators, not confirmed manipulation localization.</div></div>`).join("")}</div>`
        : `<p class="muted">No suspicious frame indicators were found across sampled frames.</p>`}
    </div>`;
  }

  function fmtTime(sec) {
    sec = Math.max(0, Number(sec) || 0);
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  }

  /* ---------------------------------------------------------- Full graph */
  async function graphPage(view, params) {
    const invId = params.id;
    view.innerHTML = `<h1>Evidence Graph</h1>
      <p class="lead">Nodes are uploaded files and the entities/claims extracted from them.
      Edges are similarity, consistency, contradiction, timestamp and mentions relationships.</p>
      <div class="card"><div class="graph-wrap" id="graph-full" style="height:560px"></div>
      ${Graph.legendHTML()}
      <div style="margin-top:14px"><button class="btn ghost" onclick="TrustLayer.go('#/results/${UI.esc(invId)}')">Back to results</button></div></div>
      <div class="card" style="margin-top:18px"><h3>Edge detail</h3><div id="edge-table"></div></div>`;
    try {
      const g = (await API.graph(invId)).graph;
      Graph.render(view.querySelector("#graph-full"), g);
      const rows = g.edges.map(e => `<tr>
        <td>${UI.esc(e.source.slice(0, 8))}</td>
        <td>${UI.esc(e.target.slice(0, 8))}</td>
        <td>${UI.esc(e.relationship)}</td>
        <td class="mono">${e.score != null ? Number(e.score).toFixed(2) : "—"}</td>
      </tr>`).join("");
      view.querySelector("#edge-table").innerHTML = `<table>
        <thead><tr><th>Source</th><th>Target</th><th>Relationship</th><th>Score</th></tr></thead>
        <tbody>${rows || '<tr><td colspan="4" class="muted">No edges.</td></tr>'}</tbody></table>`;
    } catch (e) {
      view.querySelector("#graph-full").innerHTML = `<p class="muted">${UI.esc(e.message)}</p>`;
    }
  }

  /* ------------------------------------------------------------- History */
  async function history(view) {
    view.innerHTML = UI.loading("Loading history…");
    let list = [];
    try { list = (await API.listInvestigations()).investigations || []; }
    catch (e) { view.innerHTML = UI.empty(e.message); return; }

    const rows = list.map(inv => `
      <tr>
        <td><a href="#/results/${UI.esc(inv.id)}">${UI.esc(inv.name)}</a>
            <div class="muted mono" style="font-size:11px">${UI.esc(inv.id.slice(0,8))}</div></td>
        <td>${UI.assessmentTag(inv.assessment)}</td>
        <td>${inv.confidence ? UI.esc(inv.confidence) : "—"}</td>
        <td class="mono">${inv.coverage != null ? UI.pct(inv.coverage) : "—"}</td>
        <td>${UI.esc(inv.status)}</td>
        <td class="muted">${UI.fmtDate(inv.created_at)}</td>
        <td>
          <button class="btn ghost small" data-open="${UI.esc(inv.id)}">Open</button>
          <button class="btn danger small" data-del="${UI.esc(inv.id)}">Delete</button>
        </td>
      </tr>`).join("");

    view.innerHTML = `
      <h1>Investigation History</h1>
      <p class="lead">All investigations stored locally in SQLite.</p>
      <div class="card">
        ${list.length ? `<table>
          <thead><tr><th>Name</th><th>Assessment</th><th>Confidence</th><th>Coverage</th><th>Status</th><th>Created</th><th></th></tr></thead>
          <tbody>${rows}</tbody></table>` : `<p class="muted">No investigations yet.</p>`}
      </div>
      <div style="margin-top:16px;display:flex;gap:10px;flex-wrap:wrap">
        <button class="btn" onclick="TrustLayer.go('#/new')">New Investigation</button>
        <button class="btn secondary" id="h-sample">Run Sample Investigation</button>
      </div>`;

    view.querySelectorAll("[data-open]").forEach(b => b.onclick = () => TrustLayer.go(`#/results/${b.dataset.open}`));
    view.querySelectorAll("[data-del]").forEach(b => b.onclick = async () => {
      if (!confirm("Delete this investigation and its evidence?")) return;
      try { await API.deleteInvestigation(b.dataset.del); TrustLayer.toast("Deleted.", "ok"); TrustLayer.refresh("#/history", true); }
      catch (e) { UI.toast(e.message, "err"); }
    });
    const hs = view.querySelector("#h-sample");
    if (hs) hs.onclick = () => runSample(view);
  }

  /* ---------------------------------------------------------- Evaluation */
  async function evaluation(view) {
    view.innerHTML = UI.loading("Loading evaluation data…");
    let ev, caps;
    try { ev = await API.evaluation(); caps = await API.capabilities(); }
    catch (e) { view.innerHTML = UI.empty(e.message); return; }

    const metricRow = (k, v) => `<tr><td>${UI.esc(k)}</td><td class="mono">${UI.esc(v)}</td></tr>`;
    const modelRow = (k, v) => `<tr><td>${UI.esc(k)}</td>
      <td>${v ? '<span class="tag green">present</span>' : '<span class="tag neutral">not found</span>'}</td></tr>`;

    // Real measured evaluation report (if `training.evaluate_image` was run).
    let reportCard = "";
    if (ev.report && ev.report.status === "ok") {
      const tm = ev.report.test_metrics, cm = ev.report.confusion;
      const metricCell = (label, val) => `
        <div class="mini-stat">
          <span class="ms-label">${label}</span>
          <span class="ms-value">${val == null ? "—" : Number(val).toFixed(3)}</span>
        </div>`;
      const total = (cm.tp + cm.tn + cm.fp + cm.fn) || 1;
      reportCard = `
        <div class="card accent-green">
          <div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:10px;align-items:center">
            <h3 style="margin:0">Measured results — Image detector</h3>
            <span class="tag green">REAL EVALUATION</span>
          </div>
          <p class="muted" style="margin-top:6px">
            ${UI.esc(ev.report.dataset || "")} · trained on ${ev.report.n_train}
            images · tested on <strong>${ev.report.n_test} held-out images</strong>
            never seen during training.
          </p>
          <div class="grid cols-3" style="gap:12px;margin:14px 0">
            ${metricCell("Accuracy", tm.accuracy)}
            ${metricCell("Precision", tm.precision)}
            ${metricCell("Recall", tm.recall)}
            ${metricCell("F1", tm.f1)}
            ${metricCell("ROC-AUC", tm.roc_auc)}
            ${metricCell("Mean score (fake)", ev.report.mean_score_fake)}
          </div>
          <h3>Confusion matrix</h3>
          <table style="max-width:460px">
            <thead><tr><th></th><th>Predicted REAL</th><th>Predicted FAKE</th></tr></thead>
            <tbody>
              <tr><td><strong>Actual REAL</strong></td>
                  <td class="mono" style="color:var(--green)">${cm.tn}</td>
                  <td class="mono" style="color:var(--amber)">${cm.fp}</td></tr>
              <tr><td><strong>Actual FAKE</strong></td>
                  <td class="mono" style="color:var(--amber)">${cm.fn}</td>
                  <td class="mono" style="color:var(--green)">${cm.tp}</td></tr>
            </tbody>
          </table>
          <p class="muted" style="font-size:12.5px;margin-top:10px">
            Mean score on REAL images: ${UI.esc(String(ev.report.mean_score_real))} ·
            on FAKE images: ${UI.esc(String(ev.report.mean_score_fake))}.
            These are real, non-inflated hold-out numbers.
          </p>
        </div>`;
    }

    view.innerHTML = `
      <h1>Evaluation &amp; Model</h1>
      <p class="lead">How the prototype performs, what is actually trained, and what has
      <strong>not</strong> been evaluated. Results are never invented.</p>

      ${reportCard}

      <div class="grid cols-2" style="margin-top:18px">
        <div class="card">
          <h3>Model architecture comparison</h3>
          <table><tbody>
            ${metricRow("Individual-only model (image)", ev.metrics.individual_only)}
            ${metricRow("Multimodal model", ev.metrics.multimodal)}
            ${metricRow("Multimodal + cross-modal reasoning", ev.metrics.multimodal_plus_cross_modal)}
          </tbody></table>
          <p class="muted" style="font-size:12.5px;margin-top:10px">${UI.esc(ev.note)}</p>
        </div>
        <div class="card">
          <h3>Generalization</h3>
          <table><tbody>
            ${metricRow("Known manipulations (train set)", ev.generalization.known_manipulations)}
            ${metricRow("Unseen manipulations (test set)", ev.generalization.unseen_manipulations)}
            ${metricRow("Different compression", ev.generalization.different_compression)}
            ${metricRow("Different conditions", ev.generalization.different_conditions)}
          </tbody></table>
          <p class="muted" style="font-size:12.5px;margin-top:10px">
            Run <code class="mono">python -m training.evaluate_image --cifake</code> to
            refresh these numbers.</p>
        </div>

        <div class="card">
          <h3>Runtime &amp; mode</h3>
          ${UI.kv([
            ["Current mode", ev.mode],
            ["Any trained model", ev.any_model_trained ? "yes" : "no"],
            ["Fusion input", ev.fusion_input],
            ["Modality dropout", ev.modality_dropout],
            ["Video decoder", caps.video_decoder],
          ])}
        </div>
        <div class="card">
          <h3>Trained artifacts in models/</h3>
          <table><tbody>
            ${Object.entries(ev.models_present).map(([k,v]) => modelRow(k, v)).join("")}
          </tbody></table>
        </div>

        <div class="card">
          <h3>Optional libraries detected</h3>
          <table><tbody>
            ${Object.entries(caps.libraries).map(([k,v]) => modelRow(k, v)).join("")}
          </tbody></table>
        </div>
        <div class="card">
          <h3>Evidence coverage definition</h3>
          <p class="muted">Coverage = fraction of the four modalities present:
          image, video, audio, document. It is reported separately from confidence.</p>
          <p class="muted" style="font-size:12.5px">${UI.esc(caps.disclaimer)}</p>
        </div>
      </div>`;
  }

  /* -------------------------------------------------------- (Limitations page removed) */

  return { home, newInvestigation, processing, results, graphPage, history,
           evaluation };
})();
