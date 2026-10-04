/* TrustLayer router + app bootstrap. Hash-based routing, no build tools. */

const TrustLayer = (() => {
  const state = { draftId: null, lastAnalysis: null, capabilities: null };

  const TITLES = {
    home: "Home",
    new: "New Investigation",
    processing: "Analysis Processing",
    results: "Investigation Results",
    graph: "Evidence Graph",
    history: "Investigation History",
    evaluation: "Evaluation / Model",
  };

  // Maps URL route -> view function key. This is why #/new works even though
  // the view is exported as `newInvestigation`.
  const ROUTES = {
    home: "home",
    new: "newInvestigation",
    processing: "processing",
    results: "results",
    graph: "graphPage",
    history: "history",
    evaluation: "evaluation",
  };

  function parseHash() {
    const raw = (location.hash || "#/").replace(/^#\/?/, "");
    const [route, ...rest] = raw.split("/");
    const params = {};
    if (rest.length) params.id = rest.join("/");
    return { route: route || "home", params };
  }

  async function render() {
    const { route, params } = parseHash();
    const view = document.getElementById("view");
    const viewKey = ROUTES[route] || "home";
    const handler = Views[viewKey] || Views.home;

    // nav highlight
    document.querySelectorAll("#nav a").forEach(a => {
      a.classList.toggle("active", a.dataset.route === route ||
        (route === "processing" && a.dataset.route === "new") ||
        (route === "results" && a.dataset.route === "history") ||
        (route === "graph" && a.dataset.route === "history"));
    });
    document.getElementById("page-title").textContent = TITLES[route] || "TrustLayer";
    document.body.classList.remove("nav-open");

    try {
      await handler(view, params);
    } catch (e) {
      console.error(e);
      view.innerHTML = UI.empty(`Something went wrong: ${e.message}`);
    }
    window.scrollTo({ top: 0, behavior: "instant" in window ? "instant" : "auto" });
  }

  function go(hash) {
    if (location.hash === hash) render();
    else location.hash = hash;
  }

  function refresh(hash, force) {
    if (force || location.hash === hash) { location.hash = hash; render(); }
    else location.hash = hash;
  }

  async function boot() {
    window.addEventListener("hashchange", render);
    // load capabilities for the mode badge
    try {
      state.capabilities = await API.capabilities();
      const mode = state.capabilities.mode;
      document.getElementById("mode-badge").textContent = `MODE: ${mode.toUpperCase()}`;
      if (mode === "demo") {
        const flag = document.getElementById("demo-flag");
        flag.hidden = false;
      }
      UI.setStatus("Ready", "ok");
    } catch (e) {
      UI.setStatus("Backend offline", "warn");
      UI.toast("Could not reach the TrustLayer backend.", "err");
    }
    if (!location.hash) location.hash = "#/";
    render();
  }

  return {
    go,
    render,
    refresh,
    boot,
    state,
    setStatus: UI.setStatus,
    toast: UI.toast,          // used by every view
  };
})();

document.addEventListener("DOMContentLoaded", () => TrustLayer.boot());
