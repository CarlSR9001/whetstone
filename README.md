# whetstone

**Two learners grinding each other sharp: a student that must earn every
promotion, and an examiner that must earn the right to judge it — with every
label anchored to an oracle that cannot practically be wrong.**

whetstone is a self-improving research harness built around one rule: *nothing is
believed until something exact has tried to kill it.* A verifier-gated student
(QLoRA adapters on a local 4B) trains on experience milled from root oracles —
exhaustive graph enumeration, simulated-annealing counterexample search, exact
game simulators, Monte-Carlo consensus, Stockfish, KataGo — and is graded by an
evolving exam bank that quarantines anything the student trained on, learns which
items discriminate, retires the saturated, and feeds its retired exams back as
training fuel.

Selected results (full ledger with methods, nulls, and receipts in
[FINDINGS.md](FINDINGS.md); artifacts in [results/](results/)):

- Settled an open finite conjecture: degree-descending greedy coloring is NOT
  optimal on connected triangle-free non-bipartite graphs (minimal
  counterexample: 11 vertices, 13 edges — found by annealing, minimized, exact).
- Quantified the verifier-horizon regress: 99/121 conjectures with precision 1.0
  at n<=6 are false at n=7/8; acceptance horizons must sit strictly inside
  stress horizons or self-training loops train on artifacts.
- Measured salience vs relevance: the most salient memory is the counterfactually
  useful one 4% of the time; a query-conditioned relevance score, 81%.
- Ran a full unattended continual-learning loop: cold-start 0/8 -> 8/8 in one
  gated round; the multi-domain generation cracked a cross-domain frontier its
  own examiner had exposed (base 1/33 -> gen-2 4/33, first MIS pass).
- Closed the same-bank scale comparison locally: stock Qwen2.5 1.5B->32B rose
  8->20/48 while every stock model scored 0/24 on graph repairs; the task-routed
  verifier-trained FastContext-4B scored 28/48 and earned PASS over its 22/48
  base (6 gains, 0 regressions, exact p=.03125), repeated identically three times.
- The private promotion exam bank is deliberately NOT in this repo: publishing it
  would let future models train on it — the leakage rule at internet scale.

Licensed AGPL-3.0. Commercial licensing available from the author.

Install the dependency-light core (CLI and stateless toolbox) with
`pip install .`. Optional integrations are grouped as `agents`, `engines`, and
`local-models`; `pip install ".[all]"` installs every integration.

The public HTTP MCP endpoint serves both the sessionless `2026-07-28`
protocol (`server/discover` plus per-request metadata and headers) and the
initialize-based `2025-06-18`, `2025-03-26`, and `2024-11-05` revisions. The
stdlib server remains dependency-free; `mcp>=2,<3` is needed only for SDK
clients and the stdio adapter.

Tagged releases contain archive-built wheels, source archives, and
`SHA256SUMS`; GitHub Actions attaches build-provenance attestations. For
v0.7.0, verify the wheel against the repository with:

```powershell
gh release download v0.7.0 -R CarlSR9001/whetstone -p "*.whl" -p "SHA256SUMS"
gh attestation verify .\branching_continual_verification-0.7.0-py3-none-any.whl -R CarlSR9001/whetstone
```

## The product layer

The research harness above is operable as a product: one CLI, adapters for any
system under exam, pluggable verifiers, a CI contract, and agent-native
surfaces. Full guide in [docs/product.md](docs/product.md).

```powershell
$env:PYTHONPATH='src'
python -m bcv.cli init                                   # bank + whetstone.toml
python -m bcv.cli mint --domain code --max-items 8       # hidden-check exam items
python -m bcv.cli grade --system v2 --command "python my_agent.py"
python -m bcv.cli grade --system v2 --acp "my-acp-agent" # any Agent Client Protocol agent
python -m bcv.cli gate --baseline v1 --candidate v2      # exit code IS the verdict: 0 PASS, 2 HOLD, 3 BLOCK
python -m bcv.cli sweep                                  # retire saturated items downward
python -m bcv.cli redteam                                # hostile self-test; nonzero exit on escape
python -m bcv.cli serve --port 8977                      # localhost JSON service
python -m bcv.cli mcp                                    # MCP server (also in .mcp.json)
python -m bcv.cli inspect --exam exam.jsonl --exposure exposure.jsonl --baseline v1.json --candidate v2.json
python -m bcv.cli toolbox --port 8988                    # eight stateless tools on localhost
```

Grading an external endpoint burns every exposed item — permanently — via the
bank's exposure accounting; the promotion report carries paired evidence, an
exact McNemar p-value, the bank's own resolution statement, and a SHA-256
commitment to bank state. No surface, human or agent, ever serves exam item
contents.

The stateless toolbox turns eight mechanisms already in the repo into usable
file-in/receipt-out surfaces: Whetstone Inspector, Eval Leak Auditor, Promotion
Gate, Bank Health, SafePatch, Counterexample Hunter, Memory Relevance Debugger,
and Agent Replay Console. Each run exposes the decision path, an inspectable
visual certificate, a SHA-256 receipt, and a task-specific downloadable
artifact. Leakage analysis keeps exact identity, finite-corpus behavioral
equivalence, and text-similarity review as separate evidence tiers; only the
first two quarantine automatically. The public instance at
[whetstone.cyberelf.link](https://whetstone.cyberelf.link/) accepts only caller
supplied disposable data; it has no examiner bank, accounts, cookies, or upload
persistence. Private exams belong in the CLI or a customer-controlled instance.

The same public service now includes
[Open Promotion Bench](https://whetstone.cyberelf.link/benchmark), a paired
scope-integrity track. `open_bench_start` gives a baseline and candidate the
same six procedurally varied virtual-repository tasks; `open_bench_submit`
checks requested state plus edit-scope conservation and counts gains,
regressions, and ties before issuing PASS/HOLD/BLOCK. A submitter may explicitly
publish a self-attested sanitized receipt. The ledger stores manifests,
transitions, counts, and hashes only—never task contents or answer patches—so
the existing workbench/report-card non-retention boundary remains intact.

Support-panel minting currently fails closed by default: the lexical checker
passed its clean smoke corpus but failed 13/13 adversarial cases, so it cannot
mint promotion-capable support items until an independently reviewed hard
calibration is supplied explicitly.

## The sixty-second demo

The whole promotion gate, live, on CPU, in about ten seconds:

```powershell
$env:PYTHONPATH='src'; python -m bcv.demo_investor
```

It mints exam items at the verifier frontier, quarantines the ones a toy
student trained on (row-identity leakage check against a real buffer on disk),
grades two systems through the checker specs — a memorizer that echoes its
training data and a candidate that proposes repairs — computes per-item
discrimination, issues a gains-vs-regressions promotion decision, retires
saturated items into the downward-only training flow, and writes an append-only
ledger. The two systems are stored answer policies (this keeps the demo off the
GPU); every mint, quarantine, and grade is the production code path, computed
live. A canonical transcript is in
[results/demo_investor_transcript.txt](results/demo_investor_transcript.txt).

## Research context

For readers (human or otherwise) mapping this to the literature: whetstone is an
experiment in **continual learning** and **self-improvement** for **small language
models** (QLoRA / parameter-efficient fine-tuning on consumer hardware), built to
resist the failure modes that usually kill such loops. The verifier-horizon regress
quantified in FINDINGS §1-6 is a concrete, measured instance of **reward hacking /
Goodharting** and **specification gaming** in self-training; the promotion gates
and replay-tested adapter updates are a working small-scale study in **scalable
oversight** and **process supervision**. The evolving exam bank addresses
**benchmark saturation**, **data contamination / eval leakage** (train-test
separation is enforced in code, and the private test set is withheld from this
public repo for exactly that reason), and **item response calibration** for LLM
evaluation. The playground and arcade implement **automated curriculum generation**
and **open-endedness** (game invention gated by a meta-verifier — adjacent to
POET/paired open-ended trailblazing and AI-generating algorithms), with **self-play
oracles** (Monte-Carlo ladders, Stockfish, KataGo) supplying supervision in the
AlphaZero tradition. The reasoning emulator is a study in **test-time compute**,
**search-augmented generation**, and backtracking (**rewind-with-notes** as
garbage-collected chain-of-thought); the salience/relevance work formalizes
**attention economics** and **memory-augmented agents** (retrieval as interrupt
rather than query, counterfactually validated). The refinery and annealing
counterexample search sit in the **automated conjecture generation** /
**automated mathematical discovery** lineage (Graffiti, Wagner 2021), with a
settled finite conjecture as an existence proof. Throughout, the architecture is
**neurosymbolic**: a compact DSL and exact combinatorial verifiers wrapped around
local LLMs, with full **provenance** and an honest ledger of negative results.

---

This repo grew from the blueprint in `BLUEPRINT.md`; the original scaffold notes follow.

What exists now:

- A JSONL-backed cognitive branch store.
- Typed commits/events with provenance.
- Branch creation, checkout, merge, diff, blame, and bisect.
- A patch-only Markdown editor with conservation hooks.
- A benchmark harness that injects long-document corruption and verifies that hooks catch it.
- A local Ollama-backed Markdown edit agent that writes only verifier-accepted patches.

Run the tests:

```powershell
python -m pytest
```

Run tests plus every current experiment:

```powershell
.\scripts\run_all.ps1
```

Run a real local-model Markdown edit against the sample agreement:

```powershell
.\scripts\run_sample_edit.ps1
```

Run the accept/retry/block usefulness benchmark:

```powershell
$env:PYTHONPATH='src'; python -m bcv.usefulness
```

Run the multi-document sequential edit benchmark:

```powershell
$env:PYTHONPATH='src'; python -m bcv.corpus_benchmark
```

Run Cognitive Git recall and coding benchmarks:

```powershell
$env:PYTHONPATH='src'; python -m bcv.recall_benchmark
$env:PYTHONPATH='src'; python -m bcv.coding_benchmark --quick
$env:PYTHONPATH='src'; python -m bcv.coding_benchmark
```

Export the seed Taste-RL preference and SFT datasets:

```powershell
$env:PYTHONPATH='src'; python -m bcv.taste
```

Generate a scaled synthetic Taste-RL dataset:

```powershell
$env:PYTHONPATH='src'; python -m bcv.taste_data --variants-per-prompt 40
```

Run the first verifier-backed graph discovery loop:

```powershell
$env:PYTHONPATH='src'; python -m bcv.discovery --max-n 6
```

Evaluate parseable graph conjectures from a proposal file:

```powershell
$env:PYTHONPATH='src'; python -m bcv.graph_agent --max-n 6 --proposal-file sample_docs/graph_proposals.json
```

Ask the local Ollama/LM Studio model to propose graph conjectures, then verify them:

```powershell
$env:PYTHONPATH='src'; python -m bcv.graph_agent --max-n 6 --use-model --max-rules 6
```

Run a second model proposal pass conditioned on the previous verifier failures:

```powershell
$env:PYTHONPATH='src'; python -m bcv.graph_agent --max-n 6 --use-model --max-rules 6 --feedback-file .bcv_runs/graph_agent/proposal_evaluation.json
```

Run the larger research foundry comparison: stateless proposal loops versus an internal-git feedback loop that accumulates accepted conjectures, counterexamples, repairs, and SFT data:

```powershell
$env:PYTHONPATH='src'; python -m bcv.research_foundry --mode scripted --rounds 3 --max-n 6
```

Run the same foundry through the local Ollama/LM Studio model:

```powershell
$env:PYTHONPATH='src'; python -m bcv.research_foundry --mode model --rounds 3 --max-n 6 --max-rules 4
```

Run the foundry with a post-run scale check that stress-tests every accepted rule and
repair at larger n and commits scale falsifications to the ledger:

```powershell
$env:PYTHONPATH='src'; python -m bcv.research_foundry --mode scripted --rounds 3 --max-n 6 --stress-ns 7 8
```

Run the self-contained frontier loop: FastContext 4B proposes on the GPU, the CPU
verifier kills, per-round scale falsifications feed back into the next proposal
prompt, and every proposal is judged for semantic novelty against the miner's
two-atom hull:

```powershell
$env:PYTHONPATH='src'; python -m bcv.research_foundry --mode fastcontext --rounds 6 --max-n 6 --max-rules 6 --stress-feedback-ns 7 8 10 --library .bcv_runs/adversary_library.jsonl
```

Hunt counterexamples by simulated annealing inside a conjecture's predicate class
(finds persist to the adversary library, which any pool can load with --library):

```powershell
$env:PYTHONPATH='src'; python -m bcv.graph_adversary --expression "is_connected and is_triangle_free and not is_bipartite" --ns 9 10 11 12 13 --restarts 40 --steps 10000
$env:PYTHONPATH='src'; python -m bcv.graph_adversary --report-file .bcv_runs/graph_generalize_rich/generalization_report.json
```

Run the model-vs-annealer counterexample duel:

```powershell
$env:PYTHONPATH='src'; python scripts/run_duel.py
```

Run the reasoning-emulator benchmark (save/load/rewind-with-notes/verifier-check as
model-invocable controls, vs linear CoT, verifier-scored):

```powershell
$env:PYTHONPATH='src'; python -m bcv.emulator --dataset-path .bcv_runs/graph_repair_hard_rich/hard_heldout.jsonl --model Qwen/Qwen3-1.7B --limit 6
```

Serve the Pro Action Replay MCP server (external agents can step, dump, and poke a
live reasoning session; registered in `.mcp.json` as `reasoning-emulator`):

```powershell
$env:PYTHONPATH='src'; python -m bcv.emulator_mcp
```

Run the full conjecture refinery for a domain (enumerate -> verify -> stress ->
anneal-attack -> closure-expand library -> THEOREMS ledger + falsification museum):

```powershell
$env:PYTHONPATH='src'; python -m bcv.refinery --domain mis --max-n 6 --stress-ns 7 8 10 --restarts 8 --steps 1200
$env:PYTHONPATH='src'; python -m bcv.refinery --domain coloring --max-n 6 --stress-ns 7 8 10 --restarts 8 --steps 800 --library .bcv_runs/adversary_library.jsonl
```

Train a FastContext QLoRA smoke adapter from the foundry's accumulated verifier-repair data:

```powershell
$env:PYTHONPATH='src'; python -m bcv.research_foundry --mode model --rounds 3 --max-n 6 --max-rules 4 --train-adapter
```

Generate a larger verifier-repair dataset for graph conjecture repair:

```powershell
$env:PYTHONPATH='src'; python -m bcv.graph_repair_data --max-n 6 --max-proposals 48 --root .bcv_runs/graph_repair_data_full
```

Train a FastContext QLoRA graph-repair adapter on JSON-only verifier targets:

```powershell
$env:PYTHONPATH='src'; python -m bcv.graph_lora --dataset-path .bcv_runs/graph_repair_data_full/repair_json_sft.jsonl --output-dir .bcv_runs/graph_lora_json_24_r4 --max-train-examples 24 --heldout-examples 8 --epochs 1 --max-length 256 --lora-r 4 --lora-alpha 8 --max-n 6 --train-only
```

Generate the hard repair dataset (the prompt contains counterexample evidence but not
the answer constraint, and heldout groups share no original expression with train):

```powershell
$env:PYTHONPATH='src'; python -m bcv.graph_repair_data --hard --max-n 6 --max-proposals 48 --heldout-groups 8 --root .bcv_runs/graph_repair_hard
```

Train and strictly evaluate the hard-task adapter (eval requires the output to be a
verifier-accepted strict refinement of the prompt's original expression, and reports
support retention plus mode-collapse diagnostics):

```powershell
$env:PYTHONPATH='src'; python -m bcv.graph_lora --dataset-path .bcv_runs/graph_repair_hard/hard_train.jsonl --heldout-path .bcv_runs/graph_repair_hard/hard_heldout.jsonl --output-dir .bcv_runs/graph_lora_hard_29_r8 --max-train-examples 29 --epochs 2 --max-length 640 --lora-r 8 --lora-alpha 16 --max-n 6
```

Stress-test every rule and repair accepted at n<=6 on larger graphs (random G(n,p)
plus deterministic adversaries: interleaved crown graphs and greedy-adversarial trees):

```powershell
$env:PYTHONPATH='src'; python -m bcv.graph_generalize --evaluation-file .bcv_runs/graph_repair_hard/proposal_evaluation.json --ns 7 8 --samples-per-np 120 --relabels 3
```

Build the stress-mined hard dataset, whose repair targets must also survive sampled
and adversarial graphs at larger n (targets stop being horizon artifacts by construction):

```powershell
$env:PYTHONPATH='src'; python -m bcv.graph_repair_data --hard --max-n 6 --max-proposals 48 --heldout-groups 6 --stress-ns 7 8 --root .bcv_runs/graph_repair_hard_stress
```

Run the first benchmark:

```powershell
$env:PYTHONPATH='src'; python -m bcv.benchmark
```

Run every current probe and write ledgers/training candidates:

```powershell
$env:PYTHONPATH='src'; python -m bcv.experiments
```

Run only the local-model document probe:

```powershell
$env:PYTHONPATH='src'; python -m bcv.model_probe
```

Train the current learned document-corruption tripwire:

```powershell
$env:PYTHONPATH='src'; python -m bcv.tripwire
```

Export verifier-ledger training datasets:

```powershell
$env:PYTHONPATH='src'; python -m bcv.sft_export
```

Run the cached FastContext 4B QLoRA smoke trainer:

```powershell
$env:PYTHONPATH='src'; python -m bcv.lora_smoke
```

The benchmark is deliberately deterministic. It does not prove an LLM will edit safely. It proves the state and verifier layer can distinguish a clean patch from a corrupt full-document rewrite before a model is plugged in.
