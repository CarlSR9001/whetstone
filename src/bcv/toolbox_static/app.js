"use strict";

const META = {
  inspector: {
    badge: "exact exposure + paired gate",
    boundary: "Requires a complete post-quarantine item cohort. It never silently intersects partial baseline and candidate runs.",
    hint: "Drop exam, exposure, baseline, and candidate JSON/JSONL files; filenames map automatically.",
  },
  leakage: {
    badge: "declared identity only",
    boundary: "Catches explicit IDs, hashes, and exact content identity. It does not claim semantic near-duplicate detection.",
    hint: "Drop exam and exposure JSON/JSONL files.",
  },
  gate: {
    badge: "paired exact statistics",
    boundary: "Baseline and candidate must cover identical item IDs. Aggregate score alone is never accepted as evidence.",
    hint: "Drop baseline and candidate results, or edit the JSON payload.",
  },
  health: {
    badge: "history, not vibes",
    boundary: "Flakiness is estimated only from repeated outcomes for the same named system.",
    hint: "Drop history JSONL rows with item_id, system, passed, and optional domain.",
  },
  safepatch: {
    badge: "deterministic verifier",
    boundary: "This hosted surface applies a precise patch and checks conservation. It does not generate edits with a model.",
    hint: "Drop one Markdown document or a complete SafePatch JSON payload.",
  },
  counterexample: {
    badge: "bounded CPU search",
    boundary: "A returned witness is exact. No witness within the stated budget is not a proof of the conjecture.",
    hint: "Edit the supported graph-DSL expression and bounded search budget.",
  },
  memory: {
    badge: "objective-conditioned ranking",
    boundary: "The estimator is a mechanism playground. The 0.988 result is specific to the exact TinySeasons decision procedure.",
    hint: "Drop a memory array or edit objective, objective_entities, and memories.",
  },
  replay: {
    badge: "event flight recorder",
    boundary: "Reconstructs explicit emulator events. It does not expose or claim access to hidden chain of thought.",
    hint: "Drop an emulator result or JSONL event log.",
  },
};

const state = { catalog: [], examples: {}, active: "inspector", result: null, loadedFiles: [] };
const $ = id => document.getElementById(id);

function setText(id, value) { $(id).textContent = value == null ? "—" : String(value); }

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
  for (const [field, words] of pairs) if (words.some(word => value.includes(word))) return field;
  return null;
}

function renderFileChips() {
  const container = $("fileChips");
  container.replaceChildren();
  state.loadedFiles.forEach(file => {
    const chip = document.createElement("span");
    chip.className = "file-chip";
    chip.textContent = file;
    container.appendChild(chip);
  });
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
      if (files.length === 1 && parsed && !Array.isArray(parsed) && typeof parsed === "object" && Object.keys(parsed).some(key => ["exam", "baseline", "candidate", "events", "memories", "history", "document"].includes(key))) {
        payload = parsed;
      } else if (field) {
        payload[field] = parsed && parsed.results !== undefined ? parsed.results : parsed;
      } else if (state.active === "replay") {
        payload.events = Array.isArray(parsed) ? parsed : parsed.events || parsed.result?.events || [];
      } else if (state.active === "memory") {
        payload.memories = Array.isArray(parsed) ? parsed : parsed.memories || [];
      } else if (state.active === "health") {
        payload.history = Array.isArray(parsed) ? parsed : parsed.history || parsed.results || [];
      } else {
        throw new Error(`cannot map ${file.name}; include exam/exposure/baseline/candidate/history in the filename`);
      }
    }
    state.loadedFiles.push(file.name);
  }
  $("payloadEditor").value = JSON.stringify(payload, null, 2);
  renderFileChips();
}

function renderTabs() {
  const tabs = $("toolTabs");
  tabs.replaceChildren();
  state.catalog.forEach((tool, index) => {
    const button = document.createElement("button");
    button.className = "tool-tab" + (tool.id === state.active ? " active" : "");
    button.type = "button";
    button.setAttribute("role", "tab");
    button.setAttribute("aria-selected", tool.id === state.active ? "true" : "false");
    const number = document.createElement("span");
    number.textContent = String(index + 1).padStart(2, "0");
    const name = document.createElement("b");
    name.textContent = tool.name;
    button.append(number, name);
    button.addEventListener("click", () => selectTool(tool.id));
    tabs.appendChild(button);
  });
}

function selectTool(id) {
  state.active = id;
  state.result = null;
  state.loadedFiles = [];
  const index = state.catalog.findIndex(tool => tool.id === id);
  const tool = state.catalog[index];
  const meta = META[id];
  setText("toolNumber", String(index + 1).padStart(2, "0"));
  setText("activeTitle", tool.name);
  setText("activePromise", tool.promise);
  setText("activeBoundary", meta.boundary);
  setText("boundaryBadge", meta.badge);
  setText("dropHint", meta.hint);
  $("payloadEditor").value = JSON.stringify(state.examples[id], null, 2);
  renderFileChips();
  renderTabs();
  resetResult();
}

function resetResult() {
  setText("resultHeading", "Waiting for a run");
  setText("resultStatus", "IDLE");
  $("resultStatus").className = "status idle";
  $("resultEmpty").hidden = false;
  $("resultContent").hidden = true;
  $("summaryGrid").replaceChildren();
  $("resultDetail").replaceChildren();
}

function metric(value, label) {
  const box = document.createElement("div");
  box.className = "summary";
  const strong = document.createElement("b");
  strong.textContent = value == null ? "—" : String(value);
  const span = document.createElement("span");
  span.textContent = label;
  box.append(strong, span);
  return box;
}

function table(columns, rows) {
  const element = document.createElement("table");
  const head = document.createElement("thead");
  const headRow = document.createElement("tr");
  columns.forEach(column => { const th = document.createElement("th"); th.textContent = column.label; headRow.appendChild(th); });
  head.appendChild(headRow);
  const body = document.createElement("tbody");
  rows.forEach(row => {
    const tr = document.createElement("tr");
    columns.forEach(column => { const td = document.createElement("td"); const value = column.value(row); td.textContent = value == null ? "—" : String(value); tr.appendChild(td); });
    body.appendChild(tr);
  });
  element.append(head, body);
  return element;
}

function renderResult(result) {
  state.result = result;
  $("resultEmpty").hidden = true;
  $("resultContent").hidden = false;
  const summary = $("summaryGrid");
  const detail = $("resultDetail");
  summary.replaceChildren(); detail.replaceChildren();
  let verdict = "ready";
  if (state.active === "inspector") {
    verdict = result.gate.verdict;
    summary.append(metric(result.gate.verdict, "decision"), metric(result.audit.quarantined_items, "quarantined"), metric(result.gate.paired_evidence.gains, "gains"), metric(result.gate.paired_evidence.regressions, "regressions"), metric(result.audit.clean_items, "clean cohort"), metric(result.gate.paired_evidence.exact_mcnemar_two_sided_p?.toFixed(6), "paired p"));
    detail.appendChild(table([
      {label:"item", value:row=>row.item_id}, {label:"domain", value:row=>row.domain}, {label:"outcome", value:row=>row.outcome}
    ], result.gate.paired_evidence.items_detail || []));
  } else if (state.active === "leakage") {
    summary.append(metric(result.quarantined_items, "quarantined"), metric(result.clean_items, "clean"), metric(`${(result.exposure_rate * 100).toFixed(1)}%`, "exposure rate"));
    detail.appendChild(table([
      {label:"item", value:row=>row.item_id}, {label:"reason", value:row=>row.matches?.map(match=>match.reason).join(", ")}, {label:"source", value:row=>row.matches?.map(match=>match.source).join(", ")}
    ], result.quarantined || []));
  } else if (state.active === "gate") {
    verdict = result.verdict;
    const evidence = result.paired_evidence;
    summary.append(metric(result.verdict, "decision"), metric(evidence.gains, "gains"), metric(evidence.regressions, "regressions"), metric(evidence.ties, "ties"), metric(evidence.exact_mcnemar_two_sided_p.toFixed(6), "paired p"), metric(evidence.resolution.minimum_clean_discordant_wins, "clean wins needed"));
    detail.appendChild(table([
      {label:"item", value:row=>row.item_id}, {label:"domain", value:row=>row.domain}, {label:"baseline", value:row=>row.baseline?"pass":"fail"}, {label:"candidate", value:row=>row.candidate?"pass":"fail"}, {label:"outcome", value:row=>row.outcome}
    ], evidence.items_detail || []));
  } else if (state.active === "health") {
    summary.append(metric(result.items, "items"), metric(result.systems.length, "systems"), metric(result.classification_counts.discriminating || 0, "discriminators"), metric(result.classification_counts.saturated || 0, "saturated"), metric(result.classification_counts.flaky || 0, "flaky"), metric(result.frontier_gaps.length, "domain gaps"));
    detail.appendChild(table([
      {label:"item", value:row=>row.item_id}, {label:"domain", value:row=>row.domain}, {label:"class", value:row=>row.classification}, {label:"discrimination", value:row=>row.discrimination}, {label:"flip rate", value:row=>row.max_within_system_flip_rate}
    ], result.items_detail || []));
  } else if (state.active === "safepatch") {
    verdict = result.accepted ? "PASS" : "BLOCK";
    summary.append(metric(result.accepted ? "ACCEPT" : "REJECT", "patch"), metric(result.changed_sections.length, "sections changed"), metric(result.checks.untargeted_sections_byte_identical ? "YES" : "NO", "untargeted identical"));
    const pre = document.createElement("pre"); pre.textContent = result.unified_diff; detail.appendChild(pre);
  } else if (state.active === "counterexample") {
    verdict = result.falsified ? "BLOCK" : "HOLD";
    summary.append(metric(result.status, "search result"), metric(result.find?.n, "vertices"), metric(result.find?.edges?.length, "edges"), metric(result.find?.chromatic_number, "exact χ"), metric(result.find?.greedy_colors, "greedy colors"), metric(`${result.elapsed_seconds}s`, "elapsed"));
    if (result.find) detail.appendChild(table([{label:"edge", value:row=>`${row[0]} — ${row[1]}`}], result.find.edges));
  } else if (state.active === "memory") {
    summary.append(metric(result.ranking.length, "memories"), metric(result.shiny_traps.length, "shiny traps"), metric(result.boring_but_decisive.length, "boring decisive"), metric(result.selected_by_relevance.length, "selected"), metric(result.selected_tokens, "tokens used"));
    detail.appendChild(table([
      {label:"rel rank", value:row=>row.relevance_rank}, {label:"sal rank", value:row=>row.salience_rank}, {label:"memory", value:row=>row.content}, {label:"relevance", value:row=>row.relevance}, {label:"class", value:row=>row.classification}
    ], result.ranking || []));
  } else if (state.active === "replay") {
    summary.append(metric(result.events, "events"), metric(result.checkpoints.length, "checkpoints"), metric(result.rewinds.length, "rewinds"), metric(result.controls.CHECK || 0, "checks"), metric(result.notes.length, "surviving notes"), metric(result.external_interventions, "external events"));
    detail.appendChild(table([
      {label:"step", value:row=>row.step}, {label:"branch", value:row=>row.branch}, {label:"kind", value:row=>row.kind}, {label:"detail", value:row=>row.detail}
    ], result.timeline || []));
  }
  setText("resultHeading", result.request_id ? `Run ${result.request_id}` : "Completed run");
  setText("resultStatus", verdict.toUpperCase());
  $("resultStatus").className = `status ${verdict}`;
  $("resultJson").textContent = JSON.stringify(result, null, 2);
}

async function runActive() {
  let payload;
  try { payload = JSON.parse($("payloadEditor").value); }
  catch (error) { return showError(`Request JSON is invalid: ${error.message}`); }
  const tool = state.catalog.find(item => item.id === state.active);
  $("runButton").disabled = true;
  setText("resultHeading", "Running verifier");
  setText("resultStatus", "RUNNING");
  $("resultStatus").className = "status running";
  try {
    const response = await fetch(tool.endpoint, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(payload)});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
    renderResult(result);
  } catch (error) { showError(error.message); }
  finally { $("runButton").disabled = false; }
}

function showError(message) {
  $("resultEmpty").hidden = false;
  $("resultContent").hidden = true;
  $("resultEmpty").querySelector("p").textContent = message;
  setText("resultHeading", "Run refused safely");
  setText("resultStatus", "ERROR");
  $("resultStatus").className = "status error";
}

function downloadResult() {
  if (!state.result) return;
  const blob = new Blob([JSON.stringify(state.result, null, 2) + "\n"], {type:"application/json"});
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `whetstone-${state.active}-receipt.json`;
  document.body.appendChild(link); link.click(); link.remove(); URL.revokeObjectURL(link.href);
}

async function init() {
  const [catalogResponse, examplesResponse, evidenceResponse, healthResponse] = await Promise.all([
    fetch("/api/catalog"), fetch("/api/examples"), fetch("/api/evidence"), fetch("/api/health")
  ]);
  if (![catalogResponse, examplesResponse, evidenceResponse, healthResponse].every(response => response.ok)) throw new Error("service bootstrap failed");
  state.catalog = (await catalogResponse.json()).tools;
  state.examples = await examplesResponse.json();
  const receipt = await evidenceResponse.json();
  const health = await healthResponse.json();
  const metrics = [
    [receipt.cross_scale.models, `models · one ${receipt.cross_scale.bank_items}-item bank`],
    [receipt.cross_scale.largest_contrast.verdict, `${receipt.cross_scale.largest_contrast.gains} gains · ${receipt.cross_scale.largest_contrast.regressions} regressions · p=${receipt.cross_scale.largest_contrast.p}`],
    [receipt.relevance.accuracy.relevance, `${receipt.relevance.probes} exact relevance probes`],
    [receipt.redteam.paraphrase_caught && receipt.redteam.inflation_caught ? "2 / 2" : "partial", "specified hostile attacks caught"],
  ];
  const grid = $("evidenceGrid"); grid.replaceChildren();
  metrics.forEach(([value,label]) => { const box=document.createElement("div"); box.className="metric"; const strong=document.createElement("b"); strong.textContent=value; const span=document.createElement("span"); span.textContent=label; box.append(strong,span); grid.appendChild(box); });
  setText("serviceVersion", `v${health.version}`);
  setText("serviceState", health.private_bank_loaded ? "unexpected bank state" : "healthy · stateless · private bank absent");
  selectTool("inspector");
}

$("dropzone").addEventListener("click", () => $("fileInput").click());
$("dropzone").addEventListener("keydown", event => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); $("fileInput").click(); } });
$("fileInput").addEventListener("change", event => consumeFiles([...event.target.files]).catch(error => showError(error.message)));
["dragenter","dragover"].forEach(name => $("dropzone").addEventListener(name, event => { event.preventDefault(); $("dropzone").classList.add("dragging"); }));
["dragleave","drop"].forEach(name => $("dropzone").addEventListener(name, event => { event.preventDefault(); $("dropzone").classList.remove("dragging"); }));
$("dropzone").addEventListener("drop", event => consumeFiles([...event.dataTransfer.files]).catch(error => showError(error.message)));
$("sampleButton").addEventListener("click", () => selectTool(state.active));
$("runButton").addEventListener("click", runActive);
$("copyButton").addEventListener("click", async () => { if (state.result) { await navigator.clipboard.writeText(JSON.stringify(state.result, null, 2)); setText("copyButton", "Copied"); setTimeout(() => setText("copyButton", "Copy JSON"), 1200); } });
$("downloadButton").addEventListener("click", downloadResult);

init().catch(error => { setText("serviceState", `service unavailable: ${error.message}`); showError("The product service did not start cleanly."); });
