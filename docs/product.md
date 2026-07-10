# Whetstone as a product: the promotion gate, operable

The research harness proved the mechanism. This layer makes it operable by a
team that has never read FINDINGS.md: one CLI, adapters for any system under
exam, pluggable verifiers, a CI contract, and a minimal service.

## The five-minute tour

```powershell
$env:PYTHONPATH='src'
python -m bcv.cli init                                  # bank + whetstone.toml
python -m bcv.cli mint --domain code --max-items 8      # exam items, checkers stay private
python -m bcv.cli grade --system v1 --command "python my_agent.py"
python -m bcv.cli grade --system v2 --command "python my_agent_new.py"
python -m bcv.cli gate --baseline v1 --candidate v2     # exit code IS the verdict
python -m bcv.cli status                                # buckets + metabolism
```

(Installed via `pip install -e .`, the same commands are just `whetstone ...`.)

For local models, point the same adapter at Ollama or LM Studio; localhost is
inside the bank trust boundary and does not burn the exam. Reasoning models may
need a larger completion budget than the default:

```powershell
whetstone grade --system qwen3 --api-base http://127.0.0.1:11434/v1 `
  --model qwen3:8b --max-tokens 2048
```

`results/local_qwen_code_bakeoff.json` records the local code runs. After the
bank grew to 16 independent hidden-check items, qwen2.5-1.5B scored 6/16 and
qwen3-8B scored 13/16. The paired gate issued **PASS**: seven gains, zero
regressions, exact p=.015625. This is a reproducible local-model comparison,
not a frontier-model claim.

The next fresh 20-item bank is the counterweight that makes the evidence
credible: qwen2.5-1.5B scored 7/20 while qwen3-8B scored 10/20, but the paired
gate issued **BLOCK** (6 gains, 3 regressions, p=.5078125). Three full qwen3
replays were item-for-item identical, so reliability-aware gating also BLOCKED:
the regressions were stable rather than noisy. A single local PASS is evidence,
not a license to stop measuring.

The original gen-2 adapter's retained-coloring caveat is also closed in
`results/gen2_gate_receipt.json`: the adapter verified 4/8 retained items
against the base model's 0/8, so the probe did not regress. That does **not**
turn the round into a promotion: the paired 33-item gate remained **BLOCK**
with one playground regression and p=.375. The receipt is sanitized; private
prompts, outputs, item IDs, and adapter paths stay in ignored local artifacts.

The cross-scale bank now has the missing trained-student comparison on the
identical 48-item cohort. The cached FastContext-4B base scored 22/48
(21/24 code, 1/24 graph); gen-2 scored 26/48 (19/24 code, 7/24 graph) but
correctly BLOCKED on two stable code regressions across three identical fresh
loads. Gen-3 is a mechanism change, not another training round: one PEFT model
enables gen-2 only for graph-repair prompts and disables the adapter elsewhere.
It scored 28/48 and earned PASS over base with six gains, zero regressions, and
exact p=.03125, again with one identical item vector across three fresh loads.
See `results/local_fastcontext_same_bank_receipt.json` and
`results/local_fastcontext_gen3_routed_receipt.json`.

Against stock Qwen2.5-32B on that same bank, routed gen-3 has nine gains and one
regression (28/48 vs 20/48, p=.021484375), so the strict zero-regression policy
still BLOCKS that comparison. Aggregate superiority does not erase an item-level
regression.

`results/redteam_gate_receipt.json` records the hostile self-test: a semantic
paraphrase evaded row identity but the behavioral fingerprint quarantined it;
and a toy memorizer's six leaked-item wins would have produced an unsafe PASS,
while the protected bank issued HOLD. It is evidence for these attacks, not a
claim that the leakage defense is complete.

The former chess/Go volume ceiling has been addressed in a separate private
engine bank. After the published-log burn and randomized-opening replenishment,
the retained bank holds 17 Stockfish and 52 KataGo items. Three fresh-load
grades of shallow and mid engine tiers produced a latest 17/69 vs 27/69 result;
the gate still BLOCKED on one stable chess regression and p=.0524788. See
`results/engine_bank_bakeoff_receipt.json` and
`results/engine_bank_replenishment_receipt.json`. The mills accept explicit
tool and bank paths, so they can run without copying gitignored engines into a
worktree:

The published-GTP-log incident is enforced in defaults now: HEAD carries only
SHA-256 commitments to the 22 permanently public move prefixes, so a fresh
clone can quarantine collisions without republishing the trajectories. Go
milling uses an unseeded 4-8-stone random opening by default; disabling the
mint-time check remains an explicit research override.

```powershell
python -m bcv.grandmaster --mill 500 --mint-exams --per-bank 500 `
  --engine-path "C:\path\to\stockfish.exe" --root .bcv_runs/chess_mill --bank-root .bcv_runs/engine_bank
python -m bcv.baduk --mill 100 --mint-exams --per-bank 100 `
  --katago-dir "C:\path\to\katago" --root .bcv_runs/go_mill --bank-root .bcv_runs/engine_bank
```

Code-bank minting now selects a reproducible shuffled library slice from
`--seed`, rather than always taking a fixed prefix. A new seed-4, 12-item
cohort included four previously untested families (islands, Sudoku validity,
POSIX normalization, and spiral traversal). It also **BLOCKED**: qwen3-8B
scored 5/12 against qwen2.5-1.5B's 4/12, with 4 gains, 3 regressions, and
p=1.0. The local evidence file records task-level outcomes rather than hiding
that result behind aggregate scores.

Run the same local-only protocol without reconstructing it by hand:

```powershell
whetstone local-bakeoff --out .bcv_runs/my_bakeoff --seed 4 --items 12 `
  --baseline-model qwen2.5:1.5b --candidate-model qwen3:8b --candidate-repeats 3
```

The command refuses non-local endpoints and a non-empty output directory. It
writes the private bank, run manifests, `local_bakeoff.json`, and the signed
JSON/HTML promotion-gate receipt under `--out`.

## The pieces

**Candidates** (`bcv/candidates.py`) — anything that answers can be graded:
an OpenAI-compatible endpoint (`--api-base http://localhost:1234/v1 --model X`,
key via `--api-key-env NAME`, never on argv), a shell command (prompt on
stdin, answer on stdout), or stored answers (`--answers file.jsonl`, rows of
`{"item_id": ..., "answer": ...}`).

**The burn rule, enforced in code** — grading through a non-local endpoint
sends private items outside the trust boundary, so every exposed item is
permanently burned: recorded with provider and timestamp, removed from the
reusable pools, never again a promotion item or training fuel. Skipping this
requires the on-the-record flag `--allow-external-no-burn`. The bank has a
metabolism; `whetstone status` shows it.

**Verifier registry** (`bcv/registry.py`) — a domain is a plug with three
verbs: mint, prompt, grade. One bank can mix graph repairs, game moves, code
tasks, and panel cases; one `grade_bank` call grades across all of them.
Adding a customer domain means adding one `DomainPlugin`.

**Code domain** (`bcv/code_domain.py`) — the first non-toy domain. Tasks are
graded by hidden property checks run in an isolated interpreter with a hard
timeout: round-trips, invariants, brute-force oracles on small inputs. A
topological-sort task has exponentially many correct answers; all pass, none
are stored. There is no answer key to leak.

**Verifier panels** (`bcv/panel.py`) — the fuzzy-domain fallback: several
cheap independent checks, each allowed to abstain, aggregated by VETO (the
only aggregation that survived reward hacking in the spectrum experiments;
averages and majorities were gameable). Panels are calibrated against labeled
human verdicts — `whetstone calibrate-panel` reports agreement, false-accepts
(the dangerous direction), false-rejects, and per-check attribution. The
shipped support-agent panel scores zero false-accepts on its labeled corpus,
and the test suite pins that as a tripwire.

That clean corpus is only a smoke calibration, not evidence that support is
solved. `sample_docs/support_hard_calibration.jsonl` is an explicitly
expert-authored adversarial corpus: it contains source-worded false claims and
valid paraphrases. The committed v1 baseline is **0/13 agreement** (6 false
accepts, 7 false rejects) in `results/support_hard_panel_baseline.json`.
That failure is intentional and load-bearing: lexical overlap is not support
correctness. It is the regression target while adding case-specific
policy/claim checks and later replacing authored labels with independent
spot-checks.

Accordingly, support minting now fails closed by default. The clean 14-case
artifact is available only when supplied explicitly as a smoke fixture; the
registry cannot silently use it to mint promotion-capable support items after
the hard corpus disproved the panel.

`results/support_hard_semantic_qwen3_baseline.json` records the first
local-model research pass: a qwen3:8b semantic veto panel reaches 12/13
agreement on the same adversarial cases (0 false rejects), but has **one false
accept**. It is therefore deliberately *not* admitted to mint promotion-bank
items. A condition-extraction prompt variant was also tested and lost (11/13,
still one false accept), so it was discarded rather than presented as progress.

For real support calibration, `whetstone panel-export` makes a blind JSONL
queue with the existing labels and provenance stripped. `whetstone
panel-adjudicate` accepts one vote file per named reviewer and emits
calibration triples only when at least two distinct reviewers unanimously
agree; missing and split votes go to a separate disagreement file. This is an
auditable intake mechanism, not a claim that two reviewer IDs prove human
independence.

Use `panel-export --reviewers reviewer-a reviewer-b --templates-dir ballots`
to create separate blank vote templates; each reviewer receives the same blind
queue plus only their own ballot. After both verdict fields are filled with
`pass` or `fail`, submit those files to `panel-adjudicate`.

**The service** (`bcv/service.py`, `whetstone serve`) — GET /status,
POST /grade, POST /gate over localhost JSON. It accepts answers, never model
endpoints, so it cannot be tricked into shipping private prompts to an
arbitrary URL; item contents never transit the API in either direction.

## The CI contract

`whetstone gate` exits 0 on PASS, 2 on HOLD, 3 on BLOCK. A release pipeline
step is therefore one line, and the build fails unless the candidate earned
promotion. See [ci-example.yml](ci-example.yml). The gate report (JSON + a
self-contained HTML page with the paired evidence, the exact McNemar p-value,
and a SHA-256 commitment to the bank state) is the artifact you attach to the
release — the receipt for why this version shipped.

The repository's own test suite runs in `.github/workflows/tests.yml`; the
private-bank promotion example remains separate because it requires a
self-hosted runner holding the gitignored bank.

After repeated grading, use `whetstone gate --regression-policy
reliability_aware` to distinguish a flip on a historically stable item (fatal)
from one on a measured noisy item (an explicit budget) or an item with too
little history (HOLD). The report records both the selected policy and its
per-item reliability evidence.

New CLI, HTTP, command, ACP, and endpoint grade events also carry a
non-secret run manifest into the append-only ledger and gate report: adapter
kind, local endpoint host/model or command hash, token and timeout limits,
seed, item count, elapsed time, and burn mode. It deliberately excludes exam
content, answers, API keys, and raw command strings.

Promotion comparison is cohort-strict: the two latest grade events must cover
the identical private item IDs. Whetstone records a SHA-256 commitment to that
item set and refuses to silently intersect partial grade runs into an easier
gate decision.

## What this layer deliberately does not do

- It does not publish exam items, ever — no CLI or API surface serves them.
- It does not grade what it cannot verify: a panel that has not been
  calibrated says so in the item payload, and all-abstain cases fail closed.
- It does not treat process isolation as a security sandbox; run graders for
  untrusted code inside your existing containment.

## Agent-native surfaces: MCP and ACP

**MCP server** (`bcv/whetstone_mcp.py`, `whetstone mcp`, registered in
`.mcp.json`) — any MCP client (Claude, an orchestrator, a CI bot) gets the
gate as tools: `whetstone_status`, `whetstone_mint`, `whetstone_grade_answers`,
`whetstone_grade_command`, `whetstone_grade_endpoint`, `whetstone_gate`,
`whetstone_burn`, `whetstone_calibrate_panel`, `whetstone_use_bank`. The trust
boundary is the same as everywhere else, and it matters more here: an MCP
client may be (or may be steering) the very system under exam, so NO tool
returns item prompts or payloads — grading runs server-side, and gate results
return summaries plus report paths, never item text.

**ACP candidate** (`bcv/acp.py`, `whetstone grade --acp "<agent command>"`) —
Whetstone speaks the Agent Client Protocol as the client, so any ACP-exposing
agent (Claude Code, Gemini CLI, custom agents) is a system under exam with
zero integration work: spawn, handshake, one session, each exam item as a
prompt turn, chunks reassembled into the graded answer. The grader advertises
no filesystem or terminal capabilities and answers every permission request
with "cancelled" — an agent that cannot answer an exam without touching the
world fails that item honestly.

## The demo stage

`whetstone stage` (or `python -m bcv.cli stage`) serves a one-page presentation
shell at http://127.0.0.1:8990 — a thin translator over the real engine for
showing the gate to a non-technical room:

- Every number is compiled at load time from the committed receipts in
  `results/`; nothing on the page is typed in.
- One live button runs the real investor demo (mint, quarantine, grade, gate)
  on the presenting machine's CPU and shows what it returns.
- The spine is the promotion story: base 22/48, a gen-2 fine-tune that scores
  higher (26/48) but is BLOCKED for two stable code regressions, and a
  task-routed gen-3 (28/48) that earns a statistical PASS (6 gains, 0
  regressions, p=0.031, identical across three fresh loads).
- Exam item contents never appear on the page — same trust boundary as every
  other surface.
