# Branching Continual Verification Blueprint

## Objective

Build an agent architecture that treats reasoning, memory, document work, tool use, and self-improvement as versioned, testable state.

The core bet:

> The next useful jump is not just a larger transformer. It is a runtime and training loop where the model can branch, verify, merge, remember, and eventually update itself without losing provenance or stability.

This blueprint combines six bets:

1. Training stages should converge into one continuous experience objective.
2. Inference should produce durable learning signals, not dead transcripts.
3. Tokenization and compute should adapt to difficulty instead of charging every unit equally.
4. Feedback should be typed and process-level, not mostly collapsed into a scalar reward.
5. Self-generated experience should be filtered by verifiers and fed back into training.
6. The model should have a Git-like cognitive workspace for branches, commits, diffs, merge conflicts, blame, revert, and bisect.

## Non-Negotiable Design Constraints

- No hidden mushy memory. Every persistent belief needs source, confidence, timestamp, and expiry policy.
- No untracked state mutation. Every durable change is a commit or event.
- No branch merge without validation. Merges must pass domain hooks or preserve explicit unresolved conflicts.
- No training example without provenance. Self-generated data must retain generator, verifier, test result, and lineage.
- No full-document rewrite when a patch can express the edit. Long artifacts are structured objects with invariants.
- No pretending weight updates are safe by default. Runtime memory comes first; trainable deltas come later behind gates and rollback.

## System Shape

```text
User / Environment
        |
        v
Task Contract Compiler
        |
        v
Cognitive Git Runtime <----> Artifact Stores
        |                         |
        v                         v
Branch Workers              Code / Docs / Sheets / Memory / Sources
        |
        v
Verifier Layer
        |
        v
Merge Controller
        |
        v
Experience Ledger
        |
        v
Training / Adapter / Memory Update Pipeline
```

The system starts as a runtime around existing models. The model does not need native mutable weights on day one. It needs a branch manager that can rehydrate prior state into context on demand, plus verifiers that decide what can enter `main`.

## Integration Map From The Source Notes

The notes point at a family of missing layers. In this blueprint, each becomes either a core subsystem, an MVP domain, or a later research hook.

| Source-note idea | Blueprint role |
| --- | --- |
| Conservation-law editing | First MVP. Documents become structured artifacts with patch invariants. |
| Agent flight recorder | Core ledger. Every belief, action, tool result, and artifact change becomes replayable state. |
| Test-first prompting outside code | Task Contract Compiler. Soft tasks get rejection tests before generation. |
| Memory as living index | Third MVP. Memories carry source, confidence, expiry, contradiction links, and use policy. |
| Contradiction harvester | Merge conflict engine. Conflicts become first-class state instead of being smoothed away. |
| RAG with adversarial faithfulness checks | Claim/evidence hooks. Answers need support maps, not just retrieved context. |
| Inference budget router | Compute router. The system chooses direct answer, branch, retrieve, tool, verify, stronger model, or stop. |
| Model self-doubt detectors | Instability hooks. Rephrase variance, unsupported numbers, vague claims, and irreversible plans trigger verification. |
| Mechanistic tripwires | Later signal layer. Internal or external predictors can flag likely unsupported claims, unsafe tool calls, or destructive edits. |
| Artifact-native AI | Artifact Stores. Code, docs, sheets, calendars, and memory are operated on through native structure. |
| Model-as-fuzzer for concepts | Hypothesis generator. Category errors become branch seeds for new experiments. |
| Anti-slop datasets | Experience mining. Revision trails, failed trajectories, and repairs become training candidates. |

## Core Primitive: Cognitive Git

### Objects

`branch`
: A named alternate state, such as `hypothesis/adaptive-tokenization`, `impl/document-editor`, or `critique/verifier-failure`.

`commit`
: A typed state transition with message, source, evidence, affected objects, tests run, and confidence delta.

`semantic diff`
: A diff over claims, constraints, memories, artifacts, and model assumptions, not just text lines.

`merge conflict`
: A first-class contradiction. It must be resolved, preserved as unresolved, or used to fork a new experiment.

`blame`
: Provenance lookup for a claim, decision, memory, artifact change, or learned example.

`bisect`
: Search over commits to find where a failure, hallucination, bad edit, or unsupported assumption entered.

`hook`
: Validator attached to a domain or state type. Examples: tests pass, citations support claims, untouched text remains byte-identical, no memory promoted without source.

### Minimal State Layout

```text
state/
  branches/
    main/
      beliefs.jsonl
      constraints.jsonl
      open_questions.jsonl
      evidence.jsonl
      plan.json
      memories.jsonl
      artifacts/
      tool_log.jsonl
      commits.jsonl
  refs/
    tags.json
    branch_index.json
experience/
  episodes.jsonl
  verifier_results.jsonl
  training_candidates.jsonl
```

Actual Git can back file-level history, but the important layer is semantic: claim IDs, evidence IDs, artifact IDs, branch IDs, and test IDs must be addressable.

## Experience Event Schema

Every useful interaction becomes a typed event.

```json
{
  "event_id": "evt_...",
  "episode_id": "ep_...",
  "branch_id": "branch/main",
  "event_type": "claim_added | artifact_patch | verifier_result | memory_update | model_delta | conflict",
  "actor": "model | tool | verifier | user | trainer",
  "input_refs": ["..."],
  "output_refs": ["..."],
  "evidence_refs": ["..."],
  "artifact_refs": ["..."],
  "tests": [
    {
      "test_id": "test_...",
      "result": "pass | fail | inconclusive",
      "details_ref": "..."
    }
  ],
  "confidence_before": 0.42,
  "confidence_after": 0.78,
  "timestamp": "ISO-8601"
}
```

This ledger is the bridge between runtime cognition and training. It lets the system later mine good traces, failed traces, repairs, contradictions, and verifier-approved outputs.

## The Six Research Tracks

### 1. Continuous Objective Instead Of Bolted Stages

Current pattern:

```text
Pretrain -> SFT -> RLHF/RLAIF -> freeze -> serve
```

Target pattern:

```text
Experience stream -> mixed objective -> evaluated model/runtime -> more experience
```

The system should interleave:

- next-token prediction on high-quality text,
- instruction-following traces,
- tool-use traces,
- process-supervised reasoning steps,
- verifier-approved self-generated examples,
- failed trajectory repairs,
- memory hygiene events,
- artifact patch histories.

Near-term experiment:

- Use a small open model or adapter target.
- Build one unified dataset format from the experience ledger.
- Train or fine-tune on mixed batches instead of isolated phases.
- Compare against staged fine-tuning on retention, task performance, and failure regression.

Falsifier:

- The mixed objective improves one domain while causing broad regressions or unstable instruction following.

### 2. Persistent Test-Time Learning

Do not start with unsafe weight mutation. Use a ladder:

1. External memory commits.
2. Retrieved branch state.
3. User/session-specific preference files.
4. Verifier-approved exemplar cache.
5. Adapter deltas gated by replay tests.
6. Weight updates only after rollback, eval, and provenance exist.

The important mechanism is not "the model remembers everything." It is:

> Every interaction can produce candidate learning, but only verified learning is promoted.

Runtime learning loop:

```text
observe -> branch -> act -> verify -> commit -> promote memory/example/delta -> replay tests -> merge
```

Safety hook:

- A memory cannot become high-confidence without source.
- A learned adapter cannot activate unless it passes a replay suite against old tasks and current domain hooks.
- Every model delta is tagged and revertible.

First measurable target:

- Personal assistant memory with source/confidence/expiry.
- Evaluate stale-memory use, contradiction handling, and ability to explain provenance.

### 3. Adaptive Representation And Compute

Tokenization is treated as a fixed tollbooth today. This track tests whether the runtime can get useful gains before training tokenizer-free models.

Near-term runtime version:

- Keep the base model tokenizer.
- Add a surprisal/difficulty router that decides when to:
  - answer directly,
  - retrieve,
  - branch,
  - call tools,
  - ask for missing input,
  - allocate stronger model compute,
  - split text into byte/character-sensitive inspection,
  - run artifact-native parsers.

Longer-term training version:

- Compare BPE/token models against byte/character/adaptive segmentation models on:
  - code edits,
  - math notation,
  - names/dates/numbers,
  - corrupted text,
  - multilingual input,
  - proof steps where small symbol changes matter.

Falsifier:

- Adaptive routing costs more than it saves, or routes too late to prevent failures.

### 4. Typed Process Feedback

Scalar reward is too thin for complex correction. The system should preserve why something failed.

Feedback types:

- unsupported claim,
- wrong final answer,
- wrong intermediate assumption,
- tool misuse,
- stale memory,
- unverified citation,
- document invariant violation,
- code test failure,
- bad branch merge,
- overconfident answer,
- missed user constraint.

Each verifier result should attach to specific state objects, not just the final response.

Example:

```json
{
  "failure_type": "unsupported_claim",
  "claim_id": "claim_42",
  "entered_at_commit": "commit_16",
  "required_hook": "claim_has_evidence",
  "repair": "weaken_claim_or_attach_source"
}
```

This enables bisect, replay, and process-data training.

### 5. Self-Generated Verified Experience

The system should generate its own candidate tasks, solutions, critiques, and repairs, but only train on what can be checked.

Verifier-bounded domains:

- code: tests, type checks, lint, benchmark harnesses,
- math: symbolic checks, proof assistants where available, numeric substitution,
- documents: patch invariants, citation maps, entity preservation,
- RAG: claim-to-source support maps,
- memory: source/confidence/expiry/contradiction checks,
- planning: constraint satisfaction and reversible-action gates.

Self-training loop:

```text
generate task -> solve on branch -> verify -> repair failures -> commit trace -> promote passing trace -> train/evaluate
```

The training data should include both:

- successful final artifacts,
- failed trajectory to repaired trajectory.

The second category is likely more valuable because it captures how cognition moved.

### 6. Branch-Native Inference

The model should not run one fragile chain. It should create and manage branches.

Default branch policy for hard tasks:

- `main`: current accepted state.
- `hypothesis/*`: competing interpretations.
- `implementation/*`: concrete build paths.
- `critique/*`: adversarial checks.
- `evidence/*`: retrieval/source-grounded state.
- `repair/*`: fixes after failure.

Merge policy:

- Merge evidence, constraints, and verified artifacts freely.
- Merge hypotheses only after conflict checks.
- Merge memories only after source/confidence/expiry validation.
- Merge model deltas only after replay tests.

The model should see compact rehydration views, not raw branch dumps:

```text
Branch hypothesis/runtime-learning:
- Assumption: durable learning starts external, not in weights.
- Evidence: stability risk from catastrophic interference.
- Open risk: adapter promotion may overfit to verifier quirks.
- Useful invariant from critique/verifier-loop: no training candidate without lineage.
```

## First MVP: Conservation-Law Document Editor

This is the best first proving ground because failure is measurable.

### Build

Agent edits long Markdown or DOCX-derived structured text through patches only.

State objects:

- sections,
- paragraphs,
- claims,
- citations,
- entities,
- dates,
- numbers,
- requested edits,
- forbidden changes.

Hooks:

- unchanged sections byte-identical,
- section order preserved unless explicitly targeted,
- names/dates/numbers preserved unless targeted,
- citations remain attached to claims,
- no new factual claim without provenance,
- patch applies cleanly,
- final rendered document passes structural checks.

Experiment:

- Give baseline chat editing and branch-runtime editing the same 10-20 sequential edits over long documents.
- Measure accidental deletion, unrequested rewrite, number drift, citation drift, section drift, and explainability of changes.

Success criterion:

- Branch-runtime editing substantially reduces unrequested changes and can blame every meaningful edit.

## Second MVP: Claim-Level Research Synthesizer

State graph:

- claim nodes,
- source nodes,
- evidence edges,
- contradiction edges,
- interpretation branches.

Hooks:

- every factual claim has support,
- unsupported claims are weakened or removed,
- contradictory sources produce explicit conflicts,
- final answer generated only from merged supported claims.

Success criterion:

- On a source bundle with planted contradictions, the system surfaces conflicts instead of smoothing them away.

## Third MVP: Memory Hygiene Assistant

Memory fields:

- content,
- source,
- confidence,
- type: observation, preference, fact, inference, joke, instruction,
- created_at,
- last_verified_at,
- expiry_policy,
- contradiction_refs,
- use_policy: silent, ask-before-use, never-use-without-surfacing.

Hooks:

- no high-confidence memory without source,
- stale memories are downgraded,
- contradictions block silent use,
- user correction creates a superseding commit.

Success criterion:

- The assistant can explain why a memory is active, stale, contradicted, or retired.

## Runtime Loop

```text
1. Compile user request into task contract:
   - objective
   - constraints
   - tests
   - allowed tools
   - irreversible actions

2. Create or update main branch.

3. Route compute:
   - direct answer for low-risk tasks
   - branch for ambiguity or high cost of error
   - retrieve for factual claims
   - use artifact-native parser for structured artifacts

4. Work on isolated branch.

5. Commit typed deltas.

6. Run hooks.

7. Merge passing deltas into main.

8. Write experience events.

9. Promote verified memories/examples/training candidates.

10. If a failure is found, bisect and create repair hook.
```

## Completion Criteria For The Blueprint Program

This architecture is not validated by a nice demo. It is validated when it beats a baseline on measurable failure rates.

Minimum evidence:

- Document editor reduces accidental corruption across sequential edits.
- Research synthesizer produces claim support maps and catches contradictions.
- Memory assistant avoids stale or unsourced memory use.
- Branch runtime can bisect at least one induced hallucination or bad edit to the commit where it entered.
- Experience ledger can produce training candidates with full provenance.

## Main Risks

`Verifier gaming`
: Self-training can overfit to weak verifiers. Mitigation: adversarial verifier branches, held-out checks, and human spot audits.

`State explosion`
: Branches and commits can become noise. Mitigation: semantic compaction, tags, branch GC, and rehydration views.

`False confidence from structure`
: A clean ledger can still contain wrong claims. Mitigation: require evidence hooks for factual promotion and keep conflict states visible.

`Adapter instability`
: Persistent weight or adapter updates can damage old behavior. Mitigation: external memory first, replay tests, reversible deltas.

`Cost creep`
: Branch-native inference can spend too much compute. Mitigation: inference budget router with explicit cost accounting.

## Near-Term Build Order

1. Implement the semantic state schema and commit ledger.
2. Implement branch operations: create, checkout, diff, merge, blame, revert.
3. Build document-editing hooks for Markdown first.
4. Run baseline-vs-branch document corruption benchmark.
5. Add claim/evidence graph for research synthesis.
6. Add memory hygiene schema and promotion hooks.
7. Add bisect over commits.
8. Convert verified traces into training-candidate JSONL.
9. Add adapter-training experiments only after the runtime ledger is producing clean supervised traces.

## Whiteboard Version

The model does not need to magically become honest, persistent, and self-improving in one leap.

Give it a cognitive filesystem:

- branch uncertainty,
- commit evidence,
- diff beliefs,
- merge only verified state,
- blame failures,
- bisect hallucinations,
- train from repaired traces.

That is the shortest path from frozen chatbot to a system that can accumulate experience without turning into untracked mush.
