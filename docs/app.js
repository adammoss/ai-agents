const state = {
  tests: [],
  filters: {
    q: "",
    model: "All",
    verdict: "All",
    novelty: "All",
    family: "All",
    sort: "pvalue",
  },
};

const el = (id) => document.getElementById(id);

function fmt(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "n/a";
  const n = Number(value);
  if (Math.abs(n) >= 100) return n.toPrecision(4);
  if (Math.abs(n) >= 1) return n.toFixed(3).replace(/0+$/, "").replace(/\.$/, "");
  return n.toPrecision(3);
}

function labelClass(verdict) {
  return String(verdict || "unknown").toLowerCase().replace(/[^a-z]+/g, "-");
}

function tailLabel(tail) {
  const text = String(tail || "").toLowerCase();
  if (text.includes("lower")) return "lower-tail";
  if (text.includes("upper")) return "upper-tail";
  if (text.includes("two")) return "two-sided";
  return tail || "";
}

function significanceText(test) {
  const m = test.metrics || {};
  if (m.anomaly_sigma === null || m.anomaly_sigma === undefined) return "n/a";
  const tail = tailLabel(test.tail);
  const suffix = tail ? ` ${tail}` : "";
  if ((m.sigma || 0) < 0 && m.anomaly_sigma === 0) {
    return `0σ${tail ? ` (${tail} not supported)` : ""}`;
  }
  return `${fmt(m.anomaly_sigma)}σ${suffix}`;
}

function uniq(values) {
  return [...new Set(values.filter(Boolean))].sort((a, b) => a.localeCompare(b));
}

function fillSelect(node, values) {
  node.innerHTML = "";
  ["All", ...values].forEach((value) => {
    const opt = document.createElement("option");
    opt.value = value;
    opt.textContent = value;
    node.appendChild(opt);
  });
}

function renderSummary(metadata) {
  const counts = metadata.verdict_counts || {};
  const novelty = metadata.novelty_counts || {};
  const cards = [
    ["Tests", metadata.n_tests],
    ["Models", Object.keys(metadata.model_counts || {}).length],
    ["Anomaly", counts.Anomaly || 0],
    ["Borderline", counts.Borderline || 0],
    ["Novel", novelty.Novel || 0],
    ["Minimum p", fmt(metadata.minimum_primary_p_value)],
  ];
  el("summary-grid").innerHTML = cards
    .map(([label, value]) => `<article class="metric-card"><span>${label}</span><strong>${value}</strong></article>`)
    .join("");
}

function searchableText(test) {
  return [
    test.title,
    test.description,
    test.hypothesis,
    test.interpretation,
    test.family,
    test.model,
    test.verdict,
    test.novelty,
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();
}

function filteredTests() {
  const q = state.filters.q.trim().toLowerCase();
  let tests = state.tests.filter((test) => {
    if (state.filters.model !== "All" && test.model !== state.filters.model) return false;
    if (state.filters.verdict !== "All" && test.verdict !== state.filters.verdict) return false;
    if (state.filters.novelty !== "All" && test.novelty !== state.filters.novelty) return false;
    if (state.filters.family !== "All" && test.family !== state.filters.family) return false;
    if (q && !searchableText(test).includes(q)) return false;
    return true;
  });

  tests.sort((a, b) => {
    if (state.filters.sort === "number") return a.model.localeCompare(b.model) || a.test_number - b.test_number || a.title.localeCompare(b.title);
    if (state.filters.sort === "sigma") return ((b.metrics.anomaly_sigma || 0) - (a.metrics.anomaly_sigma || 0)) || a.title.localeCompare(b.title);
    if (state.filters.sort === "title") return a.title.localeCompare(b.title);
    const ap = a.metrics.primary_p_value ?? 999;
    const bp = b.metrics.primary_p_value ?? 999;
    return ap - bp || a.title.localeCompare(b.title);
  });

  return tests;
}

function renderTests() {
  const tests = filteredTests();
  el("result-count").textContent = `${tests.length} shown`;
  el("test-list").innerHTML = tests
    .map((test) => {
      const m = test.metrics || {};
      const thumb = test.assets && test.assets.figure_png
        ? `<div class="test-thumb"><img src="${test.assets.figure_png}" alt=""></div>`
        : `<div class="test-thumb" aria-hidden="true"></div>`;
      return `<article class="test-row">
        ${thumb}
        <div class="test-title">
          <a href="${test.page}">${test.title}</a>
          <p>${test.description || "No description recorded."}</p>
        </div>
        <div><span class="cell-label">Verdict</span><span class="label ${labelClass(test.verdict)}">${test.verdict}</span></div>
        <div><span class="cell-label">Model</span>${test.model || "Unknown"}</div>
        <div><span class="cell-label">p-value</span><span class="number">${fmt(m.primary_p_value)}</span></div>
        <div><span class="cell-label">Significance</span><span class="number">${significanceText(test)}</span></div>
        <div><span class="cell-label">Novelty</span>${test.novelty || "Unknown"}</div>
        <div><span class="cell-label">Family</span>${test.family || "other"}</div>
      </article>`;
    })
    .join("");
}

function bindControls() {
  const search = el("search");
  search.addEventListener("input", (event) => {
    state.filters.q = event.target.value;
    renderTests();
  });

  [
    ["model-filter", "model"],
    ["verdict-filter", "verdict"],
    ["novelty-filter", "novelty"],
    ["family-filter", "family"],
    ["sort-order", "sort"],
  ].forEach(([id, key]) => {
    el(id).addEventListener("change", (event) => {
      state.filters[key] = event.target.value;
      renderTests();
    });
  });

  el("reset").addEventListener("click", () => {
    state.filters = { q: "", model: "All", verdict: "All", novelty: "All", family: "All", sort: "pvalue" };
    search.value = "";
    el("model-filter").value = "All";
    el("verdict-filter").value = "All";
    el("novelty-filter").value = "All";
    el("family-filter").value = "All";
    el("sort-order").value = "pvalue";
    renderTests();
  });
}

fetch("data/tests.json")
  .then((response) => response.json())
  .then((catalogue) => {
    state.tests = catalogue.tests;
    renderSummary(catalogue.metadata);
    fillSelect(el("model-filter"), uniq(state.tests.map((test) => test.model)));
    fillSelect(el("verdict-filter"), uniq(state.tests.map((test) => test.verdict)));
    fillSelect(el("novelty-filter"), uniq(state.tests.map((test) => test.novelty)));
    fillSelect(el("family-filter"), uniq(state.tests.map((test) => test.family)));
    bindControls();
    renderTests();
  })
  .catch((error) => {
    el("test-list").innerHTML = `<article class="test-row"><div class="test-title"><strong>Could not load catalogue data.</strong><p>${error}</p></div></article>`;
  });
