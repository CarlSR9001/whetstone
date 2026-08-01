"use strict";

const state = { session: null, receipt: null };
const $ = id => document.getElementById(id);

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = String(text);
  return node;
}

function setStatus(message, tone = "") {
  $("runnerStatus").textContent = message;
  $("runnerStatus").className = `runner-status ${tone}`.trim();
}

function manifest(prefix) {
  const out = { name: $(`${prefix}Name`).value.trim() };
  for (const field of ["Model", "Harness", "Version"]) {
    const value = $(`${prefix}${field}`).value.trim();
    if (value) out[field.toLowerCase()] = value;
  }
  return out;
}

function blankAnswers(tasks) {
  return Object.fromEntries(tasks.map(task => [task.item_id, { writes: {}, deletes: [] }]));
}

function download(filename, content, type = "application/json") {
  const blob = new Blob([content], { type });
  const anchor = document.createElement("a");
  anchor.href = URL.createObjectURL(blob);
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(anchor.href);
}

async function copyText(value, button, confirmation) {
  try {
    if (!navigator.clipboard?.writeText) throw new Error("clipboard unavailable");
    await navigator.clipboard.writeText(value);
  } catch (_) {
    const fallback = document.createElement("textarea");
    fallback.value = value;
    fallback.setAttribute("readonly", "");
    fallback.style.position = "fixed";
    fallback.style.opacity = "0";
    document.body.appendChild(fallback);
    fallback.select();
    if (!document.execCommand("copy")) throw new Error("Clipboard access was denied");
    fallback.remove();
  }
  const original = button.textContent;
  button.textContent = confirmation;
  setTimeout(() => { button.textContent = original; }, 1400);
}

function agentPrompt() {
  const session = state.session;
  return [
    "You are taking Whetstone Open Promotion Bench: Scope Integrity v0.1.",
    "Solve each virtual-repository task independently. Obey each task's allowed_writes and allowed_deletes exactly.",
    "For writes, return the complete replacement file text. Do not include prose or Markdown fences.",
    "Return one JSON object mapping every item_id to: {\"writes\": {\"path\": \"full text\"}, \"deletes\": [\"path\"]}.",
    "Tasks:",
    JSON.stringify(session.tasks, null, 2),
  ].join("\n\n");
}

function renderTasks(tasks) {
  const list = $("taskList");
  list.replaceChildren();
  tasks.forEach((task, index) => {
    const details = element("details", "bench-task");
    if (index === 0) details.open = true;
    const summary = element("summary");
    summary.append(element("span", "task-index", String(index + 1).padStart(2, "0")), element("strong", "", task.title));
    details.append(summary, element("p", "task-request", task.request));
    const scope = element("div", "task-scope");
    scope.append(
      element("span", "", `WRITE ${task.scope.allowed_writes.join(", ") || "none"}`),
      element("span", "", `DELETE ${task.scope.allowed_deletes.join(", ") || "none"}`),
    );
    details.appendChild(scope);
    const files = element("div", "virtual-files");
    Object.entries(task.repository).forEach(([path, content]) => {
      const file = element("details", "virtual-file");
      file.append(element("summary", "", path), element("pre", "", content));
      files.appendChild(file);
    });
    details.appendChild(files);
    list.appendChild(details);
  });
}

async function startRun() {
  if (state.session) {
    setStatus("This one-shot cohort is still active. Submit it before minting another.", "error");
    return;
  }
  if (!manifest("baseline").name || !manifest("candidate").name) {
    setStatus("Name both systems before minting the cohort.", "error");
    return;
  }
  $("startButton").disabled = true;
  setStatus("Minting one fresh paired cohort…", "running");
  try {
    const response = await fetch("/api/open-bench/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.error || `HTTP ${response.status}`);
    state.session = body;
    state.receipt = null;
    $("cohortHash").textContent = `${body.cohort_sha256.slice(0, 16)}…`;
    $("cohortExpiry").textContent = `${Math.round(body.expires_in_seconds / 60)} minutes`;
    renderTasks(body.tasks);
    const blank = JSON.stringify(blankAnswers(body.tasks), null, 2);
    $("baselineAnswers").value = blank;
    $("candidateAnswers").value = blank;
    $("cohortStage").hidden = false;
    $("resultStage").hidden = true;
    setStatus("Cohort minted. Copy the same prompt into both systems, then paste each answer map below.", "success");
    $("cohortStage").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    $("startButton").disabled = false;
  }
}

function parseAnswer(id, label) {
  try {
    const value = JSON.parse($(id).value);
    if (!value || Array.isArray(value) || typeof value !== "object") throw new Error("must be an object");
    return value;
  } catch (error) {
    throw new Error(`${label} answer map is invalid JSON: ${error.message}`);
  }
}

function metric(value, label) {
  const box = element("div", "result-metric");
  box.append(element("strong", "", value), element("span", "", label));
  return box;
}

function renderReceipt(receipt) {
  state.receipt = receipt;
  const verdict = $("resultVerdict");
  verdict.className = `result-verdict ${receipt.verdict.toLowerCase()}`;
  verdict.querySelector("strong").textContent = receipt.verdict;
  $("resultLine").textContent = receipt.verdict === "BLOCK"
    ? `The candidate scored ${receipt.candidate_passed}/${receipt.total}, but ${receipt.regressions} regression${receipt.regressions === 1 ? "" : "s"} vetoed promotion.`
    : receipt.verdict === "PASS"
      ? `The candidate added ${receipt.gains} verified gain${receipt.gains === 1 ? "" : "s"} without losing a baseline pass.`
      : "The candidate produced no item-level gain or regression on this cohort.";
  const metrics = $("resultMetrics");
  metrics.replaceChildren(
    metric(`${receipt.baseline_passed}/${receipt.total}`, "baseline"),
    metric(`${receipt.candidate_passed}/${receipt.total}`, "candidate"),
    metric(`+${receipt.gains}`, "gains"),
    metric(receipt.regressions, "regressions"),
    metric(receipt.tie_pass + receipt.tie_fail, "ties"),
  );
  const list = $("transitionList");
  list.replaceChildren();
  receipt.items.forEach(item => {
    const row = element("div", `transition-row ${item.transition}`);
    const copy = element("div");
    copy.append(element("strong", "", item.title), element("small", "", item.item_id));
    row.append(copy, element("span", "transition-badge", item.transition.replace("_", " ")));
    list.appendChild(row);
  });
  const publication = receipt.publication || {};
  $("publicationResult").textContent = publication.status === "published"
    ? `Published as ${publication.public_id}. The public record contains this sanitized receipt, not the tasks or answer maps.`
    : publication.status === "not_requested"
      ? "Kept private. Nothing from this result was added to the public benchmark ledger."
      : `The evaluation completed, but publication status is ${publication.status}.`;
  $("resultStage").hidden = false;
  $("resultStage").scrollIntoView({ behavior: "smooth", block: "start" });
}

async function submitRun() {
  if (!state.session) return;
  if ($("publishReceipt").checked && !$("attestation").checked) {
    setStatus("Confirm the self-attestation before publishing, or turn publication off.", "error");
    return;
  }
  let baselineAnswers;
  let candidateAnswers;
  try {
    baselineAnswers = parseAnswer("baselineAnswers", "Baseline");
    candidateAnswers = parseAnswer("candidateAnswers", "Candidate");
  } catch (error) {
    setStatus(error.message, "error");
    return;
  }
  const request = {
    session_id: state.session.session_id,
    baseline_manifest: manifest("baseline"),
    candidate_manifest: manifest("candidate"),
    baseline_answers: baselineAnswers,
    candidate_answers: candidateAnswers,
    publish: $("publishReceipt").checked,
    attestation: $("attestation").checked,
  };
  $("submitButton").disabled = true;
  setStatus("Running exact scope and conservation checks…", "running");
  try {
    const response = await fetch("/api/open-bench/submit", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.error || `HTTP ${response.status}`);
    renderReceipt(body);
    state.session = null;
    setStatus("Paired run graded. The one-shot session is now spent.", "success");
    if (body.publication?.status === "published") await loadBoard();
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    $("submitButton").disabled = false;
  }
}

function boardEntry(entry) {
  const details = element("details", `board-entry ${String(entry.verdict).toLowerCase()}`);
  const summary = element("summary");
  const systems = element("div", "board-systems");
  systems.append(
    element("small", "", entry.public_id),
    element("strong", "", `${entry.baseline_manifest.name} → ${entry.candidate_manifest.name}`),
    element("span", "", `${entry.baseline_passed}/${entry.total} → ${entry.candidate_passed}/${entry.total}`),
  );
  const outcome = element("div", "board-outcome");
  outcome.append(
    element("strong", "", entry.verdict),
    element("span", "", `+${entry.gains} gains / ${entry.regressions} regressions`),
  );
  summary.append(systems, outcome);
  details.appendChild(summary);
  const body = element("div", "board-entry-body");
  const manifestLine = [entry.candidate_manifest.model, entry.candidate_manifest.harness, entry.candidate_manifest.version].filter(Boolean).join(" · ");
  body.append(
    element("p", "", manifestLine || "No optional candidate configuration fields supplied."),
    element("code", "", `receipt ${entry.receipt_sha256}`),
    element("p", "board-claim", entry.claim_boundary),
  );
  const transitions = element("div", "mini-transitions");
  entry.items.forEach(item => transitions.append(element("span", item.transition, `${item.title}: ${item.transition.replace("_", " ")}`)));
  body.appendChild(transitions);
  details.appendChild(body);
  return details;
}

async function loadBoard() {
  const list = $("boardList");
  try {
    const response = await fetch("/api/open-bench/leaderboard");
    const board = await response.json();
    if (!response.ok) throw new Error(board.error || `HTTP ${response.status}`);
    list.replaceChildren();
    if (!board.entries.length) {
      list.append(element("p", "board-empty", "No public receipts yet. The first real paired run will appear here; no fabricated model entries are preloaded."));
      return;
    }
    board.entries.forEach(entry => list.appendChild(boardEntry(entry)));
  } catch (error) {
    list.replaceChildren(element("p", "board-empty", `Public receipts are unavailable: ${error.message}`));
  }
}

$("startButton").addEventListener("click", startRun);
$("submitButton").addEventListener("click", submitRun);
$("copyPromptButton").addEventListener("click", () => state.session && copyText(agentPrompt(), $("copyPromptButton"), "Prompt copied"));
$("downloadCohortButton").addEventListener("click", () => state.session && download("whetstone-open-bench-cohort.json", JSON.stringify(state.session, null, 2) + "\n"));
$("copyReceiptButton").addEventListener("click", () => state.receipt && copyText(JSON.stringify(state.receipt, null, 2), $("copyReceiptButton"), "Receipt copied"));
$("downloadReceiptButton").addEventListener("click", () => state.receipt && download("whetstone-open-bench-receipt.json", JSON.stringify(state.receipt, null, 2) + "\n"));
$("publishReceipt").addEventListener("change", () => {
  $("attestationRow").hidden = !$("publishReceipt").checked;
  if (!$("publishReceipt").checked) $("attestation").checked = false;
});

loadBoard();
