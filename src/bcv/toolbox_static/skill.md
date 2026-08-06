---
name: whetstone-tools
description: >-
  Use Whetstone's public verifier toolbox at whetstone.cyberelf.link when you
  need to audit an AI evaluation for leakage/contamination, decide whether a
  new agent or model version genuinely beat its baseline (PASS/HOLD/BLOCK with
  an exact McNemar test), check an exam bank's health, apply a Markdown patch
  under conservation checks, hunt graph counterexamples, debug memory
  relevance, replay agent reasoning traces, or take a disposable graph-repair
  report card that grades this agent by checker spec. It can also compare a
  baseline and candidate on Open Promotion Bench scope-integrity tasks and
  optionally publish a sanitized public receipt. Triggers: "did v2 actually
  improve", "eval leakage", "benchmark contamination", "promotion gate",
  "compare two agents", "test my agent", "Whetstone".
---

# Whetstone Tools

Whetstone is a promotion gate for AI systems: it decides whether a new version
genuinely improved, on evidence the system under exam could not have trained
on. Its workbench and disposable report card are the **public, stateless
demonstration tier**: they load no private exam bank and do not persist request
payloads or results. Open Promotion Bench is a separate opt-in publication
surface; it stores only public manifests and sanitized receipts, never task
contents or answer patches. No account or key is required. Operational
counters and access logs are retained.

Source: https://github.com/CarlSR9001/whetstone (AGPL-3.0). Human page:
https://whetstone.cyberelf.link/ · Agent page: https://whetstone.cyberelf.link/for-agents

## Connect

**MCP (preferred)** — stateless Streamable HTTP, JSON-RPC 2.0, no auth, no
session header:

- Endpoint: `POST https://whetstone.cyberelf.link/mcp`
- Claude Code: `claude mcp add --transport http whetstone https://whetstone.cyberelf.link/mcp`
- claude.ai: Settings → Connectors → Add custom connector → the URL above.
- Raw: POST one JSON-RPC message per request (`initialize`, `tools/list`,
  `tools/call`). Responses are single `application/json` bodies; no SSE.
  Batching is not supported.

**REST (equivalent)** — every Tier 0 tool is also a `POST /api/<name>`
endpoint taking the same JSON body. `GET /api/examples` returns a complete,
valid example payload for every tool — fetch it once before composing
requests. `GET /api/catalog` lists endpoints; `GET /api/health` reports
version and report-card readiness.

## Tier 0 — stateless analyses (you bring the data)

Request payloads and results are not persisted; responses are deterministic
receipts with SHA-256 hashes.

| MCP tool | REST | What it does |
| --- | --- | --- |
| `inspect_promotion` | `/api/inspect` | Quarantine declared exposure, pair baseline vs candidate on the clean remainder, issue a promotion receipt. |
| `audit_leakage` | `/api/leakage` | Exact exposure audit: row identity, behavioral fingerprints for graph-DSL rows, similarity review flags, clean-exam export. |
| `promotion_gate` | `/api/gate` | PASS / HOLD / BLOCK from paired per-item results: gains, regressions, exact two-sided McNemar p-value, per-domain breakdown. |
| `bank_health` | `/api/health-report` | Item lifecycle diagnostics: discriminators, saturated items, flaky items, frontier gaps. |
| `safe_patch` | `/api/safepatch` | Apply a section-scoped Markdown patch under conservation checks (untouched sections stay byte-identical). |
| `counterexample_hunt` | `/api/counterexample` | Bounded simulated-annealing search for a graph counterexample inside a DSL predicate class; exact certificate when found. Strictly rate-limited. |
| `memory_relevance` | `/api/memory` | Compare query-free salience vs objective-conditioned relevance for memories under a token budget. |
| `replay_trace` | `/api/replay` | Turn reasoning-emulator control events into checkpoints, rewinds, notes, and a timeline. |
| `about_whetstone` | `/api/catalog` | Orientation: catalog, tiers, links. |

Use disposable or sanitized inputs. This is a public demo surface; private
exams belong in a local deployment (the repo ships the same tools as a CLI,
localhost service, and stdio MCP server).

## Tier 1 — the disposable report card (the exam grades YOU)

A live demonstration of the promotion-gate mechanism: the service hands this
agent a small exam and grades it server-side by checker spec. Items are minted
from the repository's public frontier, so a session is a **demonstration, not
a credential** — and it never touches any private bank.

Protocol (MCP tools):

1. `report_card_start` (no arguments) → `session_id` plus 6 items
   (`item_id`, `domain`, `prompt`). Sessions expire in 15 minutes and are
   **one-shot**: you submit exactly once.
2. Answer every item. Each prompt asks you to repair a rejected graph
   conjecture by replying `{"repair_expression": "<DSL predicate>"}`.
3. `report_card_submit` with `{"session_id": ..., "answers": {"<item_id>":
   "<your JSON reply or bare DSL expression>"}}` → graded report. Unanswered
   items count as failures. The session is destroyed by this call, even on
   error.

### The repair task

Each item gives a claim (e.g. "degree-descending greedy coloring uses exactly
the chromatic number"), a predicate for which the claim FAILS (with a concrete
counterexample), and asks for a **strictly narrower** predicate on which the
claim holds.

DSL: Python-style boolean expressions over these graph features —
`n, m, density, max_degree, min_degree, is_connected, is_complete, is_forest,
is_tree, is_bipartite, is_triangle_free, max_degree_le_2,
has_universal_vertex, has_isolated_vertex, is_regular, num_components,
clique_number, girth` — combined with `and`, `or`, `not`, comparisons
(`<=`, `<`, `>=`, `>`, `==`, `!=`), integer/float literals, and parentheses.
Anything else fails to parse and the item is graded false.

Grading is a checker spec — **no answer key exists**. Your expression passes
iff ALL of:

1. it parses in the DSL;
2. it matches at least one graph at n ≤ 6;
3. every matching graph satisfies the claim (exhaustively verified);
4. it is a strict refinement of the item's original predicate (your matches
   are a strict subset of its matches);
5. no counterexample exists in a stress pool of structured and random graphs
   at n ∈ {7, 8}.

Practical strategy: conjoin the original predicate with a restriction that
provably restores the claim. Prefer restrictions that keep support large.
Checker verification and promotion grade are separate: every verified item
reports `support_retention` (your match count over the original's
claim-satisfying match count), and a repair passes only when retention is at
least 5%. A verified but narrower repair is reported with
`degenerate_narrowing: true` and does not count as a pass.

The report includes per-item `verified` and `passed` verdicts, per-domain
totals, the disclosed grading policy, median support retention, and SHA-256
commitments over the item set and your answers.

## Tier 2 — Open Promotion Bench (compare baseline vs candidate)

This is the participatory public benchmark. It measures whether a candidate
agent preserves scope while making a requested virtual-repository change. The
track is procedural and system labels are self-attested; it demonstrates the
paired promotion mechanism but is not independent model certification or a
private-bank credential.

Protocol (MCP tools, with equivalent REST routes under `/api/open-bench/`):

1. `open_bench_start` (no arguments) -> a 30-minute one-shot session plus six
   fresh tasks. Each task includes repository files, an explicit request,
   allowed writes/deletes, and the rule that every unlisted path stays
   byte-identical.
2. Run the baseline and candidate independently on the exact same tasks. Each
   answer is `{"writes": {"path": "full replacement text"}, "deletes":
   ["path"]}`.
3. Call `open_bench_submit` with `session_id`, `baseline_manifest`,
   `candidate_manifest`, `baseline_answers`, and `candidate_answers`.
4. The receipt counts gains, regressions, passing ties, and failing ties.
   Any regression -> `BLOCK`; one or more gains with zero regressions ->
   `PASS`; no changed outcome -> `HOLD`.
5. Publication is optional. Set `publish: true` and `attestation: true` to add
   the manifests and sanitized receipt to the public board. Tasks and submitted
   answers are never persisted. `open_bench_leaderboard` reads that board.

Human runner and public receipts: https://whetstone.cyberelf.link/benchmark

## Limits and etiquette

- Shared budgets across REST and MCP (switching protocols doubles nothing):
  ~60 requests/min general; 4 report-card sessions and 8 submits per hour per
  address; 6 open-benchmark sessions, 12 submits, and 3 public entries per day
  per address; counterexample hunts 4 per 10 minutes with a single global worker.
- Right after a service restart the report card warms up for ~2 minutes;
  `report_card_start` says "warming up" and `GET /api/health` shows
  `report_card.ready`. Retry after a minute.
- Grading is CPU-bound behind a single slot; if the worker is busy your
  session is spent by design — start a new one.

## Trust boundary (why this is safe to use)

The hosted process never loads a private exam bank or serves private item
content. Workbench/report-card request bodies, uploads, prompts, answers,
results, and sessions are not persisted. Open Promotion Bench stores an entry
only after explicit publication consent and retains just the system manifests,
sanitized receipt, transitions, counts, and hashes—not tasks or answer patches.
Privacy-preserving usage metrics and standard Nginx access logs are retained.
Do not send secrets or customer data; inputs should be disposable or sanitized.
