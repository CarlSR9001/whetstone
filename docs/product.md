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
