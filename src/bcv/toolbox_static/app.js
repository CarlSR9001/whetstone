"use strict";

const META = {
  inspector: {
    badge: "exposure + paired gate",
    short: "Full promotion decision",
    boundary: "Requires a complete post-quarantine cohort. Partial baseline/candidate intersections are never treated as evidence.",
    hint: "Exam, exposure, baseline, and candidate files map by filename.",
    steps: ["Audit exposure", "Prove cohort", "Issue verdict"],
  },
  leakage: {
    badge: "three evidence tiers",
    short: "Find compromised evals",
    boundary: "Exact identity and finite-corpus DSL behavior can quarantine. Text similarity creates a review queue, never an automatic semantic verdict.",
    hint: "Drop exam and exposure JSON/JSONL files.",
    steps: ["Exact identity", "Behavioral proof", "Human review"],
  },
  gate: {
    badge: "paired exact statistics",
    short: "PASS / HOLD / BLOCK",
    boundary: "Baseline and candidate must cover identical item IDs. A higher aggregate score cannot erase a regression.",
    hint: "Drop baseline and candidate results, or edit the payload.",
    steps: ["Pair items", "Count discordance", "Enforce policy"],
  },
  health: {
    badge: "history, not vibes",
    short: "Diagnose the exam bank",
    boundary: "Flakiness requires repeated outcomes for the same named system. Sparse history remains visibly under-observed.",
    hint: "Drop history rows with item_id, system, passed, and optional domain.",
    steps: ["Measure items", "Map frontier", "Queue actions"],
  },
  safepatch: {
    badge: "conservation verifier",
    short: "Patch without collateral damage",
    boundary: "The hosted surface applies caller-specified edits and checks conservation. It does not generate edits with a model.",
    hint: "Drop one Markdown document or a complete SafePatch payload.",
    steps: ["Scope sections", "Protect tokens", "Prove conservation"],
  },
  counterexample: {
    badge: "bounded exact witness",
    short: "Attack a graph conjecture",
    boundary: "A returned witness is exact. No witness within the declared search budget is not a proof.",
    hint: "Edit the supported graph-DSL expression and bounded search budget.",
    steps: ["Search graphs", "Verify exactly", "Render witness"],
  },
  memory: {
    badge: "relevance vs salience",
    short: "Catch attention hijacks",
    boundary: "The ranking is a mechanism playground. Published accuracy belongs to the exact TinySeasons decision procedure, not arbitrary prose.",
    hint: "Drop a memory array or edit the objective and token budget.",
    steps: ["Score salience", "Condition on goal", "Spend budget"],
  },
  replay: {
    badge: "event flight recorder",
    short: "See branches and rewinds",
    boundary: "Reconstructs explicit emulator events only. It does not expose or claim hidden chain of thought.",
    hint: "Drop an emulator result or JSONL event log.",
    steps: ["Parse controls", "Rebuild branches", "Trace outcome"],
  },
};

const state = {
  catalog: [],
  examples: {},
  active: "inspector",
  result: null,
  artifact: null,
  loadedFiles: [],
};

const $ = id => document.getElementById(id);
const SVG_NS = "http://www.w3.org/2000/svg";

function node(tag, className = "", text = null) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== null) element.textContent = String(text);
  return element;
}

function svgNode(tag, attributes = {}, text = null) {
  const element = document.createElementNS(SVG_NS, tag);
  Object.entries(attributes).forEach(([name, value]) => element.setAttribute(name, String(value)));
  if (text !== null) element.textContent = String(text);
  return element;
}

function setText(id, value) {
  $(id).textContent = value == null ? "\u2014" : String(value);
}

function parseJsonOrJsonl(text) {
  const clean = text.trim();
  if (!clean) return [];
  try { return JSON.parse(clean); }
  catch (_) {
    return clean.split(/\r?\n/).filter(line => line.trim()).map((line, index) => {
      try { return JSON.parse(line); }
      catch (error) { throw new Error(`line ${index + 1} is not valid JSON`); }
    });
  }
}

function classifyFile(name) {
  const value = name.toLowerCase();
  const pairs = [
    ["exposure", ["exposure", "training", "train"]],
    ["baseline", ["baseline", "base"]],
    ["candidate", ["candidate", "contender"]],
    ["history", ["history", "grades", "runs"]],
    ["memories", ["memory", "memories"]],
    ["events", ["event", "trace", "replay", "transcript"]],
    ["exam", ["exam", "bank", "eval"]],
  ];
  for (const [field, words] of pairs) {
    if (words.some(word => value.includes(word))) return field;
  }
  return null;
}

function renderFileChips() {
  const container = $("fileChips");
  container.replaceChildren();
  state.loadedFiles.forEach(file => container.appendChild(node("span", "file-chip", file)));
}

async function consumeFiles(files) {
  let payload;
  try { payload = JSON.parse($("payloadEditor").value); }
  catch (_) { payload = {}; }
  for (const file of files) {
    const text = await file.text();
    if (state.active === "safepatch" && /\.(md|markdown|txt)$/i.test(file.name)) {
      payload.document = text;
    } else {
      const parsed = parseJsonOrJsonl(text);
      const field = classifyFile(file.name);
      const completePayload = files.length === 1 && parsed && !Array.isArray(parsed) && typeof parsed === "object" &&
        Object.keys(parsed).some(key => ["exam", "baseline", "candidate", "events", "memories", "history", "document"].includes(key));
      if (completePayload) payload = parsed;
      else if (field) payload[field] = parsed && parsed.results !== undefined ? parsed.results : parsed;
      else if (state.active === "replay") payload.events = Array.isArray(parsed) ? parsed : parsed.events || parsed.result?.events || [];
      else if (state.active === "memory") payload.memories = Array.isArray(parsed) ? parsed : parsed.memories || [];
      else if (state.active === "health") payload.history = Array.isArray(parsed) ? parsed : parsed.history || parsed.results || [];
      else throw new Error(`cannot map ${file.name}; include the data role in its filename`);
    }
    state.loadedFiles.push(file.name);
  }
  $("payloadEditor").value = JSON.stringify(payload, null, 2);
  renderFileChips();
  renderPayloadSummary();
}

function renderTabs() {
  const tabs = $("toolTabs");
  tabs.replaceChildren();
  state.catalog.forEach((tool, index) => {
    const button = node("button", `tool-tab${tool.id === state.active ? " active" : ""}`);
    button.type = "button";
    button.setAttribute("role", "tab");
    button.setAttribute("aria-selected", tool.id === state.active ? "true" : "false");
    button.append(
      node("span", "", String(index + 1).padStart(2, "0")),
      node("b", "", tool.name),
      node("small", "", META[tool.id].short),
    );
    button.addEventListener("click", () => selectTool(tool.id, true));
    tabs.appendChild(button);
  });
}

function renderMechanismSteps(steps) {
  const container = $("mechanismSteps");
  container.replaceChildren();
  steps.forEach((step, index) => {
    const item = node("div", "mechanism-step");
    item.append(node("span", "", `0${index + 1}`), node("b", "", step));
    container.appendChild(item);
  });
}

function payloadFacts(id, payload) {
  const size = value => Array.isArray(value) ? value.length : value && typeof value === "object" ? Object.keys(value).length : 0;
  if (id === "inspector") return [["exam", size(payload.exam)], ["exposure", size(payload.exposure)], ["baseline", size(payload.baseline)], ["candidate", size(payload.candidate)]];
  if (id === "leakage") return [["exam", size(payload.exam)], ["exposure", size(payload.exposure)], ["similarity", payload.similarity_threshold ?? 0.6]];
  if (id === "gate") return [["paired items", size(payload.baseline)], ["domains", size(payload.domains)], ["alpha", payload.policy?.confidence_alpha ?? .05]];
  if (id === "health") return [["history rows", size(payload.history)], ["defined items", size(payload.items)]];
  if (id === "safepatch") return [["characters", payload.document?.length || 0], ["operations", size(payload.operations)]];
  if (id === "counterexample") return [["graph sizes", size(payload.ns)], ["restarts", payload.restarts], ["steps", payload.steps]];
  if (id === "memory") return [["memories", size(payload.memories)], ["token budget", payload.token_budget], ["objective entities", size(payload.objective_entities)]];
  if (id === "replay") return [["events", size(payload.events)], ["notes", size(payload.notes)]];
  return [];
}

function renderPayloadSummary() {
  const container = $("payloadSummary");
  container.replaceChildren();
  try {
    const payload = JSON.parse($("payloadEditor").value);
    payloadFacts(state.active, payload).forEach(([label, value]) => {
      const chip = node("span", "payload-chip");
      chip.append(node("b", "", value ?? "\u2014"), document.createTextNode(` ${label}`));
      container.appendChild(chip);
    });
  } catch (_) {
    container.appendChild(node("span", "payload-chip invalid", "JSON needs attention"));
  }
}

function renderEmptyBlueprint() {
  const meta = META[state.active];
  const stage = node("div", "empty-stage");
  meta.steps.forEach((step, index) => {
    stage.appendChild(node("span", "", step));
    if (index < meta.steps.length - 1) stage.appendChild(node("i"));
  });
  $("emptyBlueprint").replaceChildren(stage);
  setText("emptyMessage", `Run the verified sample to watch Whetstone ${meta.steps.join(" -> ").toLowerCase()}.`);
}

function selectTool(id, updateHash = false) {
  state.active = id;
  state.result = null;
  state.artifact = null;
  state.loadedFiles = [];
  const index = state.catalog.findIndex(tool => tool.id === id);
  const tool = state.catalog[index];
  const meta = META[id];
  setText("toolNumber", `${String(index + 1).padStart(2, "0")} / ${String(state.catalog.length).padStart(2, "0")}`);
  setText("activeTitle", tool.name);
  setText("activePromise", tool.promise);
  setText("activeBoundary", meta.boundary);
  setText("boundaryBadge", meta.badge);
  setText("dropHint", meta.hint);
  $("payloadEditor").value = JSON.stringify(state.examples[id], null, 2);
  renderFileChips();
  renderMechanismSteps(meta.steps);
  renderTabs();
  renderPayloadSummary();
  resetResult();
  if (updateHash) history.replaceState(null, "", `#tool=${id}`);
}

function resetResult() {
  setText("resultHeading", "Ready for a run");
  setText("resultStatus", "IDLE");
  $("resultStatus").className = "status idle";
  $("runProgress").hidden = true;
  $("resultEmpty").hidden = false;
  $("resultContent").hidden = true;
  $("resultNarrative").replaceChildren();
  $("resultViz").replaceChildren();
  $("summaryGrid").replaceChildren();
  $("resultDetail").replaceChildren();
  renderEmptyBlueprint();
}

function metric(value, label) {
  const box = node("div", "summary");
  box.append(node("b", "", value == null ? "\u2014" : value), node("span", "", label));
  return box;
}

function table(columns, rows, rowClass = null) {
  const element = node("table");
  const head = node("thead");
  const headRow = node("tr");
  columns.forEach(column => headRow.appendChild(node("th", "", column.label)));
  head.appendChild(headRow);
  const body = node("tbody");
  rows.forEach(row => {
    const tr = node("tr", rowClass ? rowClass(row) : "");
    columns.forEach(column => {
      const value = column.value(row);
      tr.appendChild(node("td", "", value == null ? "\u2014" : value));
    });
    body.appendChild(tr);
  });
  element.append(head, body);
  return element;
}

function vizCard(title, note = "") {
  const card = node("div", "viz-card");
  const heading = node("div", "viz-title");
  heading.append(node("b", "", title), node("span", "", note));
  card.appendChild(heading);
  return card;
}

function setNarrative(verdict, headline, body) {
  const container = $("resultNarrative");
  container.className = `result-narrative ${verdict}`;
  container.replaceChildren(node("strong", "", headline), node("p", "", body));
}

function setArtifact(label, filename, content, type = "application/json") {
  state.artifact = { filename, content, type };
  setText("artifactButton", label);
  $("artifactButton").hidden = false;
}

function scoreComparison(title, scorecard) {
  const card = vizCard(title, `${scorecard.pass_delta >= 0 ? "+" : ""}${scorecard.pass_delta} pass delta`);
  const rows = node("div", "score-compare");
  [["baseline", scorecard.baseline], ["candidate", scorecard.candidate]].forEach(([label, score]) => {
    const row = node("div", "score-row");
    const svg = svgNode("svg", { class: "mini-bar", viewBox: "0 0 100 12", role: "img", "aria-label": `${label} pass rate ${Math.round(score.rate * 100)} percent` });
    svg.append(svgNode("rect", { x: 0, y: 0, width: 100, height: 12, class: "mini-bar-track" }));
    svg.append(svgNode("rect", { x: 0, y: 0, width: Math.max(0, Math.min(100, score.rate * 100)), height: 12, class: `mini-bar-value ${label}` }));
    row.append(node("span", "", label), svg, node("b", "", `${score.passed}/${score.total}`));
    rows.appendChild(row);
  });
  card.appendChild(rows);
  return card;
}

function renderPipeline(pipeline) {
  const card = vizCard("Promotion pipeline", "each boundary is explicit");
  const track = node("div", "pipeline-track");
  pipeline.forEach((stage, index) => {
    const item = node("div", `pipeline-stage ${stage.state}`);
    item.append(node("span", "", `0${index + 1}`), node("b", "", stage.stage.replaceAll("_", " ")), node("small", "", stage.detail));
    track.appendChild(item);
  });
  card.appendChild(track);
  return card;
}

function renderInspector(result, summary, detail, viz) {
  const verdict = result.gate.verdict;
  setNarrative(verdict, `${verdict} after removing ${result.audit.quarantined_items} compromised item(s).`, result.gate.next_action || result.gate.reasons[0]);
  viz.append(renderPipeline(result.pipeline), scoreComparison("Post-quarantine scorecard", result.scorecard.clean));
  summary.append(
    metric(verdict, "decision"),
    metric(result.audit.quarantined_items, "quarantined"),
    metric(result.audit.review_queue.length, "human review"),
    metric(result.gate.paired_evidence.gains, "clean gains"),
    metric(result.gate.paired_evidence.regressions, "regressions"),
    metric(result.gate.paired_evidence.exact_mcnemar_two_sided_p?.toFixed(6), "exact paired p"),
  );
  detail.appendChild(table([
    { label: "item", value: row => row.item_id },
    { label: "domain", value: row => row.domain },
    { label: "baseline", value: row => row.baseline ? "pass" : "fail" },
    { label: "candidate", value: row => row.candidate ? "pass" : "fail" },
    { label: "outcome", value: row => row.outcome },
  ], result.gate.paired_evidence.items_detail || [], row => `row-${row.outcome}`));
  setArtifact("Download clean exam", "whetstone-clean-exam.jsonl", result.audit.clean_exam.map(row => JSON.stringify(row)).join("\n") + "\n", "application/x-ndjson");
  return verdict;
}

function renderLeakage(result, summary, detail, viz) {
  setNarrative("ready", `${result.quarantined_items} item(s) quarantined across proof-backed tiers.`, `${result.review_queue.length} additional text near-match(es) remain a human decision, not a semantic claim.`);
  const card = vizCard("Exposure evidence ladder", "stronger evidence gets stronger action");
  const tiers = node("div", "tier-grid");
  const exact = result.analysis_tiers.exact_identity;
  const behavioral = result.analysis_tiers.behavioral_fingerprint;
  const review = result.analysis_tiers.text_similarity;
  [["exact", "Exact identity", exact.items, "automatic quarantine"], ["behavioral", "Behavioral", behavioral.items, `${behavioral.observations} graph observations`], ["review", "Text similarity", review.candidates, "human review only"]].forEach(([className, label, count, note]) => {
    const tier = node("div", `tier-card ${className}`);
    tier.append(node("span", "", label), node("b", "", count), node("small", "", note));
    tiers.appendChild(tier);
  });
  card.appendChild(tiers);
  viz.appendChild(card);
  summary.append(metric(result.quarantined_items, "quarantined"), metric(result.clean_items, "clean"), metric(result.review_queue.length, "review queue"), metric(`${(result.exposure_rate * 100).toFixed(1)}%`, "confirmed exposure"), metric(behavioral.observations, "fingerprint graphs"), metric(review.threshold, "review threshold"));
  const rows = [
    ...(result.quarantined || []).map(row => ({ item_id: row.item_id, tier: row.matches.map(match => match.tier).join(", "), evidence: row.matches.map(match => match.reason).join(", "), source: row.matches.map(match => match.source).join(", "), action: "quarantine" })),
    ...(result.review_queue || []).map(row => ({ item_id: row.item_id, tier: "text review", evidence: `score ${row.score}`, source: row.source, action: "human review" })),
  ];
  detail.appendChild(table([
    { label: "item", value: row => row.item_id }, { label: "tier", value: row => row.tier }, { label: "evidence", value: row => row.evidence }, { label: "source", value: row => row.source }, { label: "action", value: row => row.action },
  ], rows));
  setArtifact("Download clean exam", "whetstone-clean-exam.jsonl", result.clean_exam.map(row => JSON.stringify(row)).join("\n") + "\n", "application/x-ndjson");
  return "ready";
}

function renderGate(result, summary, detail, viz) {
  setNarrative(result.verdict, `${result.verdict}: ${result.reasons[0]}`, result.next_action);
  const card = vizCard("Decision path", "policy checks execute in order");
  const path = node("div", "decision-path");
  result.decision_path.forEach(check => {
    const item = node("div", `decision-check ${check.state}`);
    item.append(node("span", "", check.state), node("b", "", check.check.replaceAll("_", " ")), node("small", "", `${check.observed ?? "\u2014"} / ${check.requirement}`));
    path.appendChild(item);
  });
  card.appendChild(path);
  viz.append(card, scoreComparison("Paired cohort scorecard", result.scorecard));
  const evidence = result.paired_evidence;
  summary.append(metric(result.verdict, "decision"), metric(evidence.gains, "gains"), metric(evidence.regressions, "regressions"), metric(evidence.ties, "ties"), metric(evidence.exact_mcnemar_two_sided_p.toFixed(6), "exact paired p"), metric(evidence.additional_clean_gains_needed ?? "blocked", "clean gains to threshold"));
  const domains = Object.entries(evidence.by_domain).map(([domain, counts]) => ({ domain, ...counts }));
  detail.appendChild(table([
    { label: "domain", value: row => row.domain }, { label: "items", value: row => row.items }, { label: "baseline", value: row => row.baseline_passes }, { label: "candidate", value: row => row.candidate_passes }, { label: "delta", value: row => row.delta }, { label: "regressions", value: row => row.regressions },
  ], domains));
  return result.verdict;
}

function readinessRing(readiness) {
  const wrap = node("div", "health-ring");
  const svg = svgNode("svg", { viewBox: "0 0 130 130", role: "img", "aria-label": `bank readiness ${readiness.index} out of 100` });
  svg.append(svgNode("circle", { cx: 65, cy: 65, r: 48, pathLength: 100, class: "ring-track" }));
  svg.append(svgNode("circle", { cx: 65, cy: 65, r: 48, pathLength: 100, "stroke-dasharray": `${readiness.index} ${100 - readiness.index}`, class: "ring-value" }));
  svg.append(svgNode("text", { x: 65, y: 63, class: "ring-number" }, readiness.index));
  svg.append(svgNode("text", { x: 65, y: 80, class: "ring-label" }, "readiness"));
  wrap.appendChild(svg);
  return wrap;
}

function renderHealth(result, summary, detail, viz) {
  const gaps = result.frontier_gaps.length ? `Frontier gaps: ${result.frontier_gaps.join(", ")}.` : "Every domain has at least one discriminator.";
  setNarrative("ready", `Bank readiness ${result.readiness.index}/100 with ${result.classification_counts.discriminating || 0} active discriminator(s).`, `${gaps} ${result.action_queue.length} item action(s) are queued.`);
  const card = vizCard("Bank operability", "transparent four-component heuristic");
  const overview = node("div", "health-overview");
  overview.appendChild(readinessRing(result.readiness));
  const components = node("div", "component-grid");
  Object.entries(result.readiness.components).forEach(([name, value]) => {
    const item = node("div", "component");
    item.append(node("span", "", name.replaceAll("_", " ")), node("b", "", `${Math.round(value * 100)}%`));
    components.appendChild(item);
  });
  overview.appendChild(components);
  card.appendChild(overview);
  viz.appendChild(card);
  summary.append(metric(result.items, "items"), metric(result.systems.length, "systems"), metric(result.classification_counts.discriminating || 0, "discriminators"), metric(result.classification_counts.saturated || 0, "saturated"), metric(result.classification_counts.flaky || 0, "flaky"), metric(result.frontier_gaps.length, "domain gaps"));
  detail.appendChild(table([
    { label: "item", value: row => row.item_id }, { label: "domain", value: row => row.domain }, { label: "class", value: row => row.classification }, { label: "difficulty", value: row => row.difficulty }, { label: "utility", value: row => row.utility }, { label: "next action", value: row => row.recommended_action },
  ], result.items_detail || []));
  setArtifact("Download action queue", "whetstone-bank-actions.json", JSON.stringify(result.action_queue, null, 2) + "\n");
  return "ready";
}

function renderSafePatch(result, summary, detail, viz) {
  const verdict = result.accepted ? "PASS" : "BLOCK";
  setNarrative(verdict, `${result.changed_sections.length} section(s) changed; ${result.untouched_sections.length} conserved byte-for-byte.`, "Protected dates, numbers, names, and references were checked before the patch was accepted.");
  const card = vizCard("Conservation map", "blue changed / white locked");
  const grid = node("div", "conservation-grid");
  [...result.changed_sections.map(heading => [heading, true]), ...result.untouched_sections.map(heading => [heading, false])].forEach(([heading, changed]) => {
    const item = node("div", `section-lock${changed ? " changed" : ""}`);
    item.append(node("span", "", changed ? "verified change" : "byte locked"), node("b", "", heading));
    grid.appendChild(item);
  });
  const stats = node("div", "diff-stats");
  stats.append(node("span", "plus", `+${result.diff_stats.added_lines} lines`), node("span", "minus", `-${result.diff_stats.removed_lines} lines`), node("span", "", `${result.diff_stats.character_delta >= 0 ? "+" : ""}${result.diff_stats.character_delta} characters`));
  card.append(grid, stats);
  viz.appendChild(card);
  summary.append(metric(result.accepted ? "ACCEPT" : "REJECT", "patch"), metric(result.diff_stats.operations, "operations"), metric(result.changed_sections.length, "sections changed"), metric(result.untouched_sections.length, "sections locked"), metric(result.diff_stats.added_lines, "lines added"), metric(result.diff_stats.removed_lines, "lines removed"));
  detail.appendChild(node("pre", "", result.unified_diff));
  setArtifact("Download patched Markdown", "whetstone-patched.md", result.updated_document, "text/markdown");
  return verdict;
}

function renderGraph(result, container) {
  const card = vizCard("Exact witness", `certificate ${result.certificate_sha256.slice(0, 12)}...`);
  const toolbar = node("div", "graph-toolbar");
  const modes = node("div", "graph-modes");
  const greedyButton = node("button", "mode-button active", `Greedy: ${result.find.greedy_colors} colors`);
  const optimalButton = node("button", "mode-button", `Exact: ${result.find.chromatic_number} colors`);
  greedyButton.type = optimalButton.type = "button";
  modes.append(greedyButton, optimalButton);
  toolbar.append(modes, node("span", "graph-proof", `strict gap +${result.witness.gap} / ${result.find.edges.length} edges`));

  const width = 520, height = 330, cx = width / 2, cy = height / 2, radius = 118;
  const positions = new Map();
  result.witness.nodes.forEach((entry, index) => {
    const angle = -Math.PI / 2 + (2 * Math.PI * index / result.witness.nodes.length);
    positions.set(entry.id, [cx + radius * Math.cos(angle), cy + radius * Math.sin(angle)]);
  });
  const svg = svgNode("svg", { class: "graph-canvas", viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": `Counterexample graph with ${result.find.n} vertices and ${result.find.edges.length} edges` });
  result.find.edges.forEach(([left, right]) => {
    const [x1, y1] = positions.get(left), [x2, y2] = positions.get(right);
    svg.appendChild(svgNode("line", { x1, y1, x2, y2, class: "graph-edge" }));
  });
  const circles = [];
  result.witness.nodes.forEach(entry => {
    const [x, y] = positions.get(entry.id);
    const circle = svgNode("circle", { cx: x, cy: y, r: 16, class: `graph-node color-${entry.greedy_color % 6}`, "data-node": entry.id });
    const title = svgNode("title", {}, `vertex ${entry.id}; degree ${entry.degree}; greedy order ${entry.greedy_order}`);
    circle.appendChild(title);
    circles.push({ circle, entry });
    svg.append(circle, svgNode("text", { x, y: y + .5, class: "graph-label" }, entry.id));
  });
  const legend = node("div", "graph-legend");

  function setMode(mode) {
    circles.forEach(({ circle, entry }) => circle.setAttribute("class", `graph-node color-${entry[`${mode}_color`] % 6}`));
    greedyButton.classList.toggle("active", mode === "greedy");
    optimalButton.classList.toggle("active", mode === "optimal");
    legend.replaceChildren();
    const count = mode === "greedy" ? result.find.greedy_colors : result.find.chromatic_number;
    for (let color = 0; color < count; color += 1) {
      const item = node("span");
      item.append(node("i", `color-${color % 6}`), document.createTextNode(` color ${color + 1}`));
      legend.appendChild(item);
    }
  }
  greedyButton.addEventListener("click", () => setMode("greedy"));
  optimalButton.addEventListener("click", () => setMode("optimal"));
  setMode("greedy");
  card.append(toolbar, svg, legend);
  container.appendChild(card);
}

function renderCounterexample(result, summary, detail, viz) {
  const verdict = result.falsified ? "BLOCK" : "HOLD";
  const headline = result.falsified ? `Conjecture falsified by an exact ${result.find.n}-vertex witness.` : "No witness found inside the declared budget.";
  const body = result.falsified ? `Greedy uses ${result.find.greedy_colors} colors; the exact optimum is ${result.find.chromatic_number}. The certificate verifies both colorings.` : result.claim_boundary;
  setNarrative(verdict, headline, body);
  if (result.find) renderGraph(result, viz);
  summary.append(metric(result.status, "search result"), metric(result.find?.n, "vertices"), metric(result.find?.edges?.length, "edges"), metric(result.find?.chromatic_number, "exact chromatic"), metric(result.find?.greedy_colors, "greedy colors"), metric(`${result.elapsed_seconds}s`, "elapsed"));
  if (result.find) {
    const features = Object.entries(result.witness.features).map(([feature, value]) => ({ feature, value }));
    detail.appendChild(table([{ label: "verified feature", value: row => row.feature }, { label: "value", value: row => row.value }], features));
    setArtifact("Download witness", "whetstone-counterexample.json", JSON.stringify({ find: result.find, witness: result.witness, certificate_sha256: result.certificate_sha256 }, null, 2) + "\n");
  }
  return verdict;
}

function renderMemoryChart(result) {
  const card = vizCard("Attention map", "salience grabs; relevance decides");
  const width = 520, height = 270, left = 44, right = 18, top = 18, bottom = 38;
  const valuesX = result.ranking.map(row => row.salience);
  const valuesY = result.ranking.map(row => row.relevance);
  const minX = Math.min(...valuesX), maxX = Math.max(...valuesX);
  const minY = Math.min(...valuesY, 0), maxY = Math.max(...valuesY, 0);
  const x = value => left + (value - minX) / Math.max(.0001, maxX - minX) * (width - left - right);
  const y = value => top + (maxY - value) / Math.max(.0001, maxY - minY) * (height - top - bottom);
  const svg = svgNode("svg", { class: "memory-chart", viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": "Memory salience versus objective-conditioned relevance scatter plot" });
  svg.append(svgNode("line", { x1: left, y1: height - bottom, x2: width - right, y2: height - bottom, class: "chart-axis" }));
  svg.append(svgNode("line", { x1: left, y1: top, x2: left, y2: height - bottom, class: "chart-axis" }));
  if (minY <= 0 && maxY >= 0) svg.append(svgNode("line", { x1: left, y1: y(0), x2: width - right, y2: y(0), class: "chart-zero" }));
  svg.append(svgNode("text", { x: width / 2, y: height - 10, class: "chart-label", "text-anchor": "middle" }, "query-free salience ->"));
  svg.append(svgNode("text", { x: 11, y: height / 2, class: "chart-label", transform: `rotate(-90 11 ${height / 2})`, "text-anchor": "middle" }, "objective relevance ->"));
  result.ranking.forEach(row => {
    const circle = svgNode("circle", { cx: x(row.salience), cy: y(row.relevance), r: row.selected_by_relevance ? 8 : 6, class: `memory-point ${row.classification}` });
    circle.appendChild(svgNode("title", {}, `#${row.id} ${row.classification}; salience ${row.salience}; relevance ${row.relevance}`));
    svg.append(circle, svgNode("text", { x: x(row.salience) + 9, y: y(row.relevance) + 3, class: "chart-label" }, `#${row.id}`));
  });
  const budgets = node("div", "budget-duel");
  [["Relevance pager", result.budget_comparison.relevance], ["Salience pager", result.budget_comparison.salience]].forEach(([label, data]) => {
    const item = node("div", "budget-card");
    item.append(node("span", "", label), node("b", "", `${data.selected.length} memories / ${data.tokens} tokens`), node("small", "", `${data.negative_relevance_tokens} negative-relevance tokens selected`));
    budgets.appendChild(item);
  });
  card.append(svg, budgets);
  return card;
}

function renderMemory(result, summary, detail, viz) {
  setNarrative("ready", `${result.shiny_traps.length} attention hijack(s) exposed; ${result.boring_but_decisive.length} quiet fact(s) recovered.`, `Objective-conditioned paging avoided ${result.budget_comparison.attention_waste_avoided_tokens} token(s) of negative-relevance attention under this budget.`);
  viz.appendChild(renderMemoryChart(result));
  summary.append(metric(result.ranking.length, "memories"), metric(result.shiny_traps.length, "shiny traps"), metric(result.boring_but_decisive.length, "quiet decisive"), metric(result.selected_by_relevance.length, "selected"), metric(result.selected_tokens, "tokens used"), metric(result.rank_correlation, "rank correlation"));
  detail.appendChild(table([
    { label: "rel rank", value: row => row.relevance_rank }, { label: "sal rank", value: row => row.salience_rank }, { label: "memory", value: row => row.content }, { label: "relevance", value: row => row.relevance }, { label: "selected", value: row => row.selected_by_relevance ? "yes" : "no" }, { label: "class", value: row => row.classification },
  ], result.ranking || []));
  const selectedRows = result.ranking.filter(row => row.selected_by_relevance);
  setArtifact("Download selected memory", "whetstone-selected-memory.json", JSON.stringify(selectedRows, null, 2) + "\n");
  return "ready";
}

function renderReplayMap(result) {
  const card = vizCard("Branch flight recorder", `${result.branches.length} branch lane(s); critical path ${result.critical_path.join(" -> ")}`);
  const map = node("div", "branch-map");
  result.branches.forEach(branch => {
    const lane = node("div", "branch-lane");
    const label = node("div", "branch-label", `BRANCH ${branch.id}`);
    const events = node("div", "branch-events");
    result.timeline.filter(event => event.branch === branch.id).forEach(event => {
      const normalized = event.detail.toUpperCase();
      const outcome = event.kind === "verifier" ? normalized.includes("ACCEPT") ? " accept" : normalized.includes("REJECT") ? " reject" : "" : "";
      const item = node("div", `event-node ${event.kind}${outcome}`, `${event.step} ${event.detail}`);
      item.title = `${event.kind}; source ${event.source}`;
      events.appendChild(item);
    });
    lane.append(label, events);
    map.appendChild(lane);
  });
  card.appendChild(map);
  if (result.notes.length) card.appendChild(node("div", "branch-note", `Surviving note: ${result.notes.join(" | ")}`));
  return card;
}

function renderReplay(result, summary, detail, viz) {
  setNarrative("ready", `${result.events} events reconstructed into ${result.branches.length} branch lane(s).`, `${result.verifier_summary.rejects} rejection(s) became structured rewinds; the final path ends on branch ${result.final_branch}.`);
  viz.appendChild(renderReplayMap(result));
  summary.append(metric(result.events, "events"), metric(result.branches.length, "branches"), metric(result.checkpoints.length, "checkpoints"), metric(result.rewinds.length, "rewinds"), metric(result.verifier_summary.rejects, "rejections"), metric(result.notes.length, "surviving notes"));
  detail.appendChild(table([
    { label: "step", value: row => row.step }, { label: "branch", value: row => row.branch }, { label: "kind", value: row => row.kind }, { label: "control", value: row => row.control }, { label: "detail", value: row => row.detail },
  ], result.timeline || []));
  setArtifact("Download event trace", "whetstone-replay.jsonl", result.timeline.map(row => JSON.stringify(row)).join("\n") + "\n", "application/x-ndjson");
  return "ready";
}

function renderResult(result) {
  state.result = result;
  state.artifact = null;
  $("artifactButton").hidden = true;
  $("resultEmpty").hidden = true;
  $("resultContent").hidden = false;
  const summary = $("summaryGrid");
  const detail = $("resultDetail");
  const viz = $("resultViz");
  summary.replaceChildren();
  detail.replaceChildren();
  viz.replaceChildren();
  let verdict = "ready";
  if (state.active === "inspector") verdict = renderInspector(result, summary, detail, viz);
  else if (state.active === "leakage") verdict = renderLeakage(result, summary, detail, viz);
  else if (state.active === "gate") verdict = renderGate(result, summary, detail, viz);
  else if (state.active === "health") verdict = renderHealth(result, summary, detail, viz);
  else if (state.active === "safepatch") verdict = renderSafePatch(result, summary, detail, viz);
  else if (state.active === "counterexample") verdict = renderCounterexample(result, summary, detail, viz);
  else if (state.active === "memory") verdict = renderMemory(result, summary, detail, viz);
  else if (state.active === "replay") verdict = renderReplay(result, summary, detail, viz);
  setText("resultHeading", result.request_id ? `Run ${result.request_id}` : "Completed run");
  setText("resultStatus", verdict.toUpperCase());
  $("resultStatus").className = `status ${verdict}`;
  setText("receiptHash", result.receipt_sha256 ? `${result.receipt_sha256.slice(0, 12)}...` : "");
  $("resultJson").textContent = JSON.stringify(result, null, 2);
}

async function runActive() {
  let payload;
  try { payload = JSON.parse($("payloadEditor").value); }
  catch (error) { return showError(`Request JSON is invalid: ${error.message}`); }
  const tool = state.catalog.find(item => item.id === state.active);
  $("runButton").disabled = true;
  $("runProgress").hidden = false;
  setText("resultHeading", "Running verifier");
  setText("resultStatus", "RUNNING");
  $("resultStatus").className = "status running";
  try {
    const response = await fetch(tool.endpoint, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
    renderResult(result);
  } catch (error) {
    showError(error.message);
  } finally {
    $("runButton").disabled = false;
    $("runProgress").hidden = true;
  }
}

function showError(message) {
  state.result = null;
  state.artifact = null;
  $("artifactButton").hidden = true;
  $("resultEmpty").hidden = false;
  $("resultContent").hidden = true;
  $("resultNarrative").replaceChildren();
  $("resultViz").replaceChildren();
  $("summaryGrid").replaceChildren();
  $("resultDetail").replaceChildren();
  $("emptyBlueprint").replaceChildren(node("div", "empty-stage", ""));
  setText("emptyMessage", message);
  setText("resultHeading", "Run refused safely");
  setText("resultStatus", "ERROR");
  $("resultStatus").className = "status error";
}

function downloadBlob(filename, content, type) {
  const blob = new Blob([content], { type });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(link.href);
}

function downloadResult() {
  if (state.result) downloadBlob(`whetstone-${state.active}-receipt.json`, JSON.stringify(state.result, null, 2) + "\n", "application/json");
}

async function copyText(text, button, confirmation) {
  try {
    if (!navigator.clipboard?.writeText) throw new Error("Clipboard API unavailable");
    await navigator.clipboard.writeText(text);
  } catch (_) {
    const fallback = document.createElement("textarea");
    fallback.value = text;
    fallback.setAttribute("readonly", "");
    fallback.style.position = "fixed";
    fallback.style.opacity = "0";
    document.body.appendChild(fallback);
    fallback.select();
    const copied = document.execCommand("copy");
    fallback.remove();
    if (!copied) throw new Error("Clipboard access was denied");
  }
  const original = button.textContent;
  button.textContent = confirmation;
  setTimeout(() => { button.textContent = original; }, 1200);
}

function selectToolFromHash() {
  if (!state.catalog.length || !location.hash.startsWith("#tool=")) return;
  const requested = location.hash.slice(6);
  if (requested !== state.active && state.catalog.some(tool => tool.id === requested)) {
    selectTool(requested, false);
  }
}

async function init() {
  const [catalogResponse, examplesResponse, evidenceResponse, healthResponse] = await Promise.all([
    fetch("/api/catalog"), fetch("/api/examples"), fetch("/api/evidence"), fetch("/api/health"),
  ]);
  if (![catalogResponse, examplesResponse, evidenceResponse, healthResponse].every(response => response.ok)) throw new Error("service bootstrap failed");
  state.catalog = (await catalogResponse.json()).tools;
  state.examples = await examplesResponse.json();
  const receipt = await evidenceResponse.json();
  const health = await healthResponse.json();
  const metrics = [
    [receipt.cross_scale.models, `models on one ${receipt.cross_scale.bank_items}-item private bank`],
    [receipt.cross_scale.largest_contrast.verdict, `${receipt.cross_scale.largest_contrast.gains} gains / ${receipt.cross_scale.largest_contrast.regressions} regressions / p=${receipt.cross_scale.largest_contrast.p}`],
    [receipt.relevance.accuracy.relevance, `${receipt.relevance.probes} exact relevance probes`],
    [receipt.redteam.paraphrase_caught && receipt.redteam.inflation_caught ? "2 / 2" : "partial", "specified hostile attacks caught"],
  ];
  const grid = $("evidenceGrid");
  grid.replaceChildren();
  metrics.forEach(([value, label]) => {
    const box = node("div", "metric");
    box.append(node("b", "", value), node("span", "", label));
    grid.appendChild(box);
  });
  setText("serviceVersion", `v${health.version}`);
  setText("serviceState", health.private_bank_loaded ? "unexpected bank state" : "healthy / stateless / private bank absent");
  const requested = location.hash.startsWith("#tool=") ? location.hash.slice(6) : "inspector";
  selectTool(state.catalog.some(tool => tool.id === requested) ? requested : "inspector", false);
}

$("dropzone").addEventListener("click", () => $("fileInput").click());
$("dropzone").addEventListener("keydown", event => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    $("fileInput").click();
  }
});
$("fileInput").addEventListener("change", event => consumeFiles([...event.target.files]).catch(error => showError(error.message)));
["dragenter", "dragover"].forEach(name => $("dropzone").addEventListener(name, event => { event.preventDefault(); $("dropzone").classList.add("dragging"); }));
["dragleave", "drop"].forEach(name => $("dropzone").addEventListener(name, event => { event.preventDefault(); $("dropzone").classList.remove("dragging"); }));
$("dropzone").addEventListener("drop", event => consumeFiles([...event.dataTransfer.files]).catch(error => showError(error.message)));
$("payloadEditor").addEventListener("input", renderPayloadSummary);
$("sampleButton").addEventListener("click", () => selectTool(state.active, false));
$("runButton").addEventListener("click", runActive);
$("copyButton").addEventListener("click", () => state.result && copyText(JSON.stringify(state.result, null, 2), $("copyButton"), "Receipt copied"));
$("shareButton").addEventListener("click", () => copyText(`${location.origin}${location.pathname}#tool=${state.active}`, $("shareButton"), "Link copied"));
$("downloadButton").addEventListener("click", downloadResult);
$("artifactButton").addEventListener("click", () => state.artifact && downloadBlob(state.artifact.filename, state.artifact.content, state.artifact.type));
window.addEventListener("hashchange", selectToolFromHash);

init().catch(error => {
  setText("serviceState", `service unavailable: ${error.message}`);
  showError("The product service did not start cleanly.");
});
