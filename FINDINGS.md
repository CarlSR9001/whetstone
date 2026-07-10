# Graph-Conjecture Track Findings (2026-07-01)

This documents what the verifier-repair QLoRA loop actually established once the eval
was hardened, and what it revealed about verifier horizons. All artifacts live under
`.bcv_runs/`. Model: `microsoft/FastContext-1.0-4B-RL`, 4-bit QLoRA on an 8 GB RTX 5060.

## 1. The original 8/8-vs-0/8 result was real but trivial

The first adapter run (`graph_lora_json_24_r4`) trained on prompts that contained both
`original_expression` and `added_constraint`; the target was literally
`(original) and (constraint)` — string assembly, not repair. The heldout split also
shared original expressions with train, and the old eval accepted *any*
verifier-passing expression, not one tied to the prompt's rule.

Under the new strict metric (repair must be verifier-accepted AND a strict refinement
of the prompt's original expression) the old adapter still scores 8/8
(`graph_lora_json_24_r4_strict/eval/eval_result.json`) — its outputs were correct,
just trivially derivable. The claim survives; the difficulty didn't.

## 2. The hard task: the win is real once the trainer is fixed

`graph_repair_data --hard` builds a no-leak dataset: the prompt carries the original
expression plus deduplicated counterexample/kept-example feature evidence; the model
must choose the repair constraint itself. Train/heldout are split by original
expression (no group overlap).

Two trainer bugs surfaced on the way, both now fixed in `graph_lora.py`:

- **Full-sequence loss**: loss over prompt+completion made the adapter memorize
  prompts (loss 0.97 → regurgitated training prompts on heldout, 0/8). Fixed by
  masking prompt tokens (`mask_prompt_loss`, default on).
- **Truncation-induced NaN**: at `max_length=640` every hard example truncated to
  prompt-only, all labels masked, NaN loss, and a silently no-op adapter (lora_B
  stayed zero — outputs byte-identical to base). Fixed with a fully-masked-example
  skip, a non-finite-loss skip, `skipped_steps` reporting, and a hard error when every
  step is skipped.

Result on group-disjoint heldout (`graph_lora_hard_29_r8_masked_v2`):

| metric | base | adapter |
| --- | --- | --- |
| parseable | 6/8 | 8/8 |
| verified strict refinement | 2/8 | **7/8** |
| distinct expressions | — | 8/8 (no collapse) |
| mean support retention | — | 0.74 |

The adapter reads the evidence and picks correct constraints on rules it never saw
(e.g. `max_degree >= 3` when every counterexample has `max_degree: 2`). Its one miss
(`m <= 5` where the counterexample itself has `m = 5`) is caught by the verifier —
which is the point of the architecture.

The stress-target variant (`graph_lora_stress_31_r8`, targets mined to survive n≤8)
scores 6/6 verified strict refinements vs base 1/6.

## 3. The bigger finding: verifier truth horizons

`graph_generalize.py` re-checks every expression accepted at n≤6 on larger graphs:
random G(n,p) plus deterministic adversaries — `crown_graph_interleaved` (labels walk
the removed matching; greedy provably uses k colors on a 2-chromatic graph) and
`greedy_adversarial_tree` (two same-colored hubs force 3 colors on a tree). The
deterministic families matter: random relabelings find these failures only by luck,
which made single-seed sweeps unstable (22 vs 53 survivors across seeds before the
adversaries were added).

The regress, quantified (`graph_generalize/generalization_report.json` and the
scratch reports at `graph_generalize_n10/`, `graph_generalize_n12/`):

- **99 of 121** expressions with precision 1.0 at n≤6 are falsified at n=7/8,
  including **7 of 10** verifier-accepted rules.
- Stress-mining targets against an n≤8 pool changes 28/37 targets; **13 of 37** still
  die at n=9/10 (all `max_degree >= 4` variants, killed by degree-4 hub adversaries).
- Mining against an n≤10 pool converges: **0/37** falsified at n=11/12, and the
  non-vacuous survivors are genuine mathematics — stars (`is_tree and
  has_universal_vertex`), odd cycles (`max_degree_le_2 and not is_bipartite`), small
  bounded families, and one open survivor (`is_connected and is_triangle_free and not
  is_bipartite`, which even the Grötzsch graph fails to falsify: greedy hits χ=4 there).
- Adapter-generated repairs inherit their targets' horizon: hard-adapter repairs
  survive 2/7 at n≤10; stress-adapter repairs survive 4/6.

This is the blueprint's "verifier gaming / false confidence from structure" risk made
concrete: **a finite verifier at horizon h mints precision-1.0 conjectures that die at
h+1 — training targets must be mined against a strictly larger adversarial horizon
than the one used for acceptance.** `--stress-ns` does exactly that.

## 4. Second pass (same day): ablation, richer DSL, foundry integration

**Evidence ablation.** Re-evaluating the trained hard-task adapter on heldout prompts
with the counterexample/kept-example evidence stripped drops it from 7/8 to 3/8
verified refinements (`graph_lora_hard_ablation_noevidence/`). The adapter still
emits well-formed `(original) and (constraint)` guesses without evidence but picks
wrong constraints — so roughly half its skill is *reading the evidence*, the rest is
learned priors about which constraints tend to work.

**Richer DSL.** `num_components`, `clique_number`, and `girth` (999 = acyclic) are now
graph features, DSL names, miner atoms, and prompt evidence. Retraining on the
enriched dataset (`graph_repair_hard_rich`, 48 groups, 40/8 split) yields a perfect
heldout score: adapter **8/8** verified strict refinements vs base **0/8**
(`graph_lora_hard_rich_r8`), support retention 0.79.

**Two new blind spots found and closed.** The enriched sweep initially left
`(max_degree_le_2 and not has_isolated_vertex) and (num_components >= 2)` and
`(max_degree_le_2 and is_regular) and (num_components >= 2)` standing. Both are false:
a bad-labeled P6 ∪ P2 (n=8) and a bad-labeled C6 ∪ C4 (n=10) match them and defeat
greedy. Random G(n,p) essentially never draws an adversarially-labeled *disconnected*
graph, so `adversarial_path_union` and `adversarial_cycle_union` are now deterministic
pool members (note the parity trap: the cycle union falsifies only when the filler
cycle is even, so the pool must include n=10, not just n=9). Final enriched sweep:
**132 of 159 falsified** (`graph_generalize_rich/`).

The rich adapter also produced a live example of target-horizon inheritance: its
verified repair `(max_degree_le_2) and (not is_connected)` is precision-1.0 at n≤6
solely because the smallest counterexample (P6 ∪ P2) has 8 vertices.

**Foundry integration.** `research_foundry --stress-ns 7 8` now stress-checks every
accepted rule and repair after the run, records survival counts in
`comparison.json`, and commits each scale falsification to the cognitive-git ledger
as a `scale_falsification` event. The test suite asserts that the scripted foundry's
n≤6-accepted repairs do die at scale — the loop can no longer grade its own homework.

## 5. Third pass: the frontier engine (optimization-driven falsification)

The 4B foundry run hard-rebooted the machine mid-run — Windows event log shows kernel
bugcheck 0x1E / access violation, i.e. an NVIDIA driver crash under sustained decode
(not an OOM; driver 591.86 on a fresh Blackwell card). Response: per-round resume in
the foundry (a crash now costs one round), model-agnostic 4-bit loaders
(`model_zoo.py`), per-round GPU telemetry, and a switch to Qwen3-1.7B for sustained
loops. The rerun completed cleanly.

**The adversarial optimizer replaces curated families** (`graph_adversary.py`):
simulated annealing over edge flips constrained to a conjecture's predicate class,
objective = cheap greedy count, exact chromatic computed only on trigger states.
Every confirmed counterexample persists to `adversary_library.jsonl`, which any pool
loads via `--library` — the harness's hostility now grows from search, not curation.
Controls behave: provably-true classes (stars, odd cycles) survive full budgets; the
16 bounded `m <= 4`-style survivors of the rich sweep also survive, consistent with
being real theorems.

**The open conjecture is settled.** `is_connected and is_triangle_free and not
is_bipartite` — which random pools, hand families, and the Grötzsch graph all failed
to kill — is FALSE: the annealer found an 11-vertex counterexample in 3 restarts,
minimized to n=11, m=13, max degree 3, girth 4 (chi=3, greedy=4;
`graph_adversary_deep/minimized_counterexample.json`). With the library loaded, the
standard sweep now falsifies both forms automatically (134/161).

**Semantic novelty is now measurable** (`novelty.py`): an expression only counts as
outside the miner's reach if its match-set over the exhaustive n<=6 universe differs
from every <=2-atom conjunction (bitmask hull, paraphrase-proof).

**Frozen-vs-learning A/B on Qwen3-1.7B** (`--learn`, `foundry_learn_qwen17/`): the
learning arm's round 1 produced 2 verifier-accepted rules and 3 semantically novel
proposals — including `n == m and is_connected and ...`, a feature-vs-feature
comparison the miner's atom vocabulary cannot express at all (rejected by the
verifier, but a genuine hull escape). The frozen arm accepted nothing in 6 rounds.
The continual learner trained each round but its gate correctly HELD every adapter
(probe never improved on ~8 buffer examples x 1 epoch) — the promotion gate working
as designed, and an honest null: this buffer is too thin to lift a 1.7B proposer.

**The loop audits its own accepts.** Both round-1 accepted rules
(`has_isolated_vertex`, `has_universal_vertex`) looked suspicious — padding
constructions (bad-P6 + isolated vertex; bad-graph + universal vertex) should kill
them past the horizon, since the smallest greedy failures need 6 vertices. The
annealer falsified both at n=7 in seconds; the finds are in the library. Padding
transforms are a systematic closure operator on the library worth automating.

**Model-vs-annealer duel** (`scripts/run_duel.py`): annealer falsified both target
classes in <1s; the 1.7B produced valid in-class graphs but no adversarial labeling
in 8 feedback-guided tries per class. Verdict: optimization owns instance
construction; the model's frontier value (so far) is hull-escaping proposals.

## 6. Fourth pass: the conjecture refinery (domain generality proven)

Everything above generalizes past greedy coloring. `bcv/domains.py` reduces a domain
to one observe function (features + exact value + greedy value + claim bit);
`bcv/refinery.py` is the one-command pipeline: enumerate candidates -> exhaustive
small-n verification -> hostile stress pool (random + deterministic families +
persistent adversary library) -> stress-mined repairs -> domain-generic annealing
attack on every survivor -> closure-operator expansion of the library
(pad-isolated / pad-universal / union-edge, each variant re-verified) -> a
THEOREMS_<domain>.md ledger of survival certificates plus a falsification museum.

Second domain, first run (`--domain mis`: static degree-ascending greedy independent
set vs exact independence number): 64 candidates -> 34 certified survivors, 64
falsified. The funnel earned each stage: 35 kills at small n, 23 at the stress pool,
and **6 caught only by the annealer** — conjectures both exhaustive verification and
the hostile pool had blessed. Notable machine finding: `is_forest`/`is_tree` FAIL for
static greedy MIS (7-vertex tree, greedy 3 vs alpha 4) even though adaptive
leaf-removal is classically optimal on forests — the refinery empirically separated
static from adaptive greedy. `is_tree and max_degree >= 3` and star classes survive.

Coloring rerun through the refinery: 51 certificates, 60 falsified — and the annealer
found **nothing new**, because the accumulated adversary library already made the
stress pool that hostile. The library is doing the optimizer's old work; falsification
knowledge compounds across runs and domains.

Ops note: the earlier hard reboot was a kernel bugcheck (0x1E) in the NVIDIA driver
under sustained 4B decode, not an OOM; the driver has since been updated, and the
refinery itself is CPU-only and crash-safe.

## 7. Fifth pass: the reasoning emulator (save/load/rewind as tools) + PAR MCP server

`bcv/emulator.py` gives a local model console-emulator controls over its own
inference: SAVE checkpoints its transcript, LOAD rewinds — with the deliberate
deviation that the model's note about the dead branch survives in a rewind-proof
notepad (pure save-state semantics would restore its ignorance and loop it), SKETCH
fast-forwards a throwaway high-temperature rollout distilled to a note, CHECK runs
the exact verifier so rewind decisions are grounded, ANSWER commits. Dead branches
are garbage-collected from context but preserved in the event log (the flight
recorder). Tool-effect boundaries would be save-barriers; this testbed is pure
reasoning + read-only verifier, so the whole tape is rewindable.

`bcv/emulator_mcp.py` (registered in `.mcp.json`) is the Pro Action Replay: an MCP
server over live sessions where an external agent can single-step the model's
reasoning, dump the full tape, and poke memory directly — inject notes, write into
the transcript as if the model had thought it, force saves and rewinds. Every poke
is logged, so the recorder distinguishes native thoughts from external writes.

Benchmark (Qwen3-1.7B, 6 heldout repair problems, verifier-scored): the MECHANISM
works — the model presses the buttons (SAVE/CHECK/LOAD with real rewind-notes),
argument sanitization + repeat-check dedup cut invalid verifier calls 36 -> 3 — but
solve rate is 0/6 in both emulator and linear arms across 17 distinct grounded
candidate checks. Conclusion, stated plainly: the emulator is a search-control
surface, and it cannot inject skill the model lacks. Candidate *generation* is the
bottleneck for a raw 1.7B, exactly matching the evidence-ablation and adapter
results (the trained 4B adapter solved 8/8 of these). The known path is controller-
trace SFT: synthesize save/check/load episodes (the ScriptedClient format in the
tests is the trace schema) and train with the existing QLoRA pipeline — search
mechanics from the harness, candidate skill from the weights.

FastContext 4B rerun (updated NVIDIA driver, partial: 4/6 problems before the run
was stopped for a performance pathology): **emulator 1/4, linear 0/4 on the same
model** — and the split is purely CHECK discipline. The one candidate the model
CHECKed to ACCEPTED became the one solve (`(is_bipartite) and (max_degree_le_2) and
(n <= 5 or m <= 4)` — verified strict refinement; the novelty judge correctly rates
its `or` as decorative: same match-set as an enumerable 2-atom repair). Every
unchecked ANSWER failed, including one that resubmitted the original expression
verbatim. Grounded search converts exactly when it is used; the next lever is a
policy nudge or trace-SFT to make CHECK-before-ANSWER unconditional. Ops: unbounded
transcript rendering inflated prefill until the KV cache spilled to system memory
(~10x decode slowdown, 100% GPU util at 41 W); fixed with a render window
(`RENDER_WINDOW = 12`) — full transcript stays in state, only the prompt is
windowed. No driver crash this time.

## 8. Sixth pass: TinySeasons x the salience paper (Eq. 1 implemented, Eq. 16 falsified-in-part)

`bcv/tinyseasons.py` generates serialized episodes with ground truth by construction:
heat -> payoff threads (the paper's §13 booking arc as a generative grammar),
computed per-episode ratings, and ORACLE labels for which prior beats each episode
depends on. `bcv/salience.py` implements the paper's Eq. 1 salience score literally
(with the §7 "GR analogy" tightened to a theorem: S_i = rho_i * exp(-Phi_i), so
softmax selection IS the Gibbs measure over the potential — verified by unit test),
plus the full §12.4 ablation battery. `bcv/salience_eval.py` runs the §12.5
fixed-budget comparison; `TransformersLocalClient.score_nll` does prefill-only
Eq. 16 scoring.

Results (149 transitions CPU / 20 transitions GPU, budget 90 tokens):

- Selection F1 vs oracle: salience 0.288 >> uniform 0.096, novelty-only 0.209,
  recency 0.217 — the controller beats every §12.5 baseline. But the paper's own
  ablations bite: additive form 0.344 and no-decay 0.336 BEAT the multiplicative
  form, and shuffled-surprise ~= salience (dA contributes nothing here). Mechanism:
  in serialized fiction the load-bearing beats are the OLD setups of long threads —
  heat is old by construction — so exp(-lambda*dt) suppresses exactly what matters.
- Eq. 16 NLL: the ranking INVERTS — full salience best (1.590), recency ~ties
  (1.597), the F1-winning ablations are worst (1.705/1.711). A frozen LM's
  next-episode NLL rewards surface/stylistic recency, not semantic state carriage.

The two-metric dissociation is the finding: (1) Eq. 16 as specified is gameable by
recency and needs a state-probe outcome (or an oracle) to detect retention value;
(2) lambda is not a nuisance parameter — it parameterizes the Pareto front between
semantic retention (facts) and surface continuity (fluency), and its correct value
is a property of the downstream consumer, not of the controller. Both are concrete
refinements the paper can absorb; both were produced by the paper's own
falsification battery, run for the first time.

## 9. Seventh pass: salience-paged long-term memory (recall-as-interrupt)

`bcv/memstore.py` is the blueprint's memory-hygiene MVP with the salience paper as
its pager: SQLite rows carry kind/source/confidence/lineage/reinforcement telemetry;
writes are gated and deduped-by-reinforcement; the pager scores every live memory
against the current context each step (Eq. 1; lambda on the retention side of the
Pareto front per §8) and PUSHES the top set under budget into the working context —
recall-as-interrupt, not Letta-style recall-as-query. Paged-in memories are
reinforced (Δt = time since last reinforcement), so useful old facts stay warm.
Consolidation is an episodic->semantic state-fold (transfer chains collapse to one
current-holder fact that exists in no single memory, with full parent lineage;
superseded state facts retire). Ponder writes derived connections at LOW confidence
with lineage — gated entry, per the drift lesson. The reasoning emulator now takes a
`memory` and pages it into the prompt uninvited.

Four-arm benchmark (24 seasons, 1,457 ground-truth QA probes, ~88-token equal
injection budget, extractive answerer so accuracy measures ONLY whether the needed
fact surfaced):

| arm | accuracy |
| --- | --- |
| none (current episode only) | 0.361 |
| recency buffer (Letta-shaped) | 0.386 |
| salience paging | 0.496 |
| salience + consolidation | **0.623** |
| oracle (needed fact directly, ~7 tokens) | 1.000 |

The recency buffer is nearly useless for long-range state (+2.5 pts over nothing) —
same mechanism as §8: the facts that matter are old. Paging adds +11 pts;
consolidation adds +13 more (521 derived state facts). The oracle line is the
sharpest statement of the problem: the needed fact averages 7 tokens — capacity was
never the constraint, addressing is.

## 10. Eighth pass: relevance is not salience — and it's worth +38 points

`bcv/relevance.py` gives relevance its own math, per the salience/relevance split:
salience is a query-free attention prior ("what grabs"); relevance is
objective-conditioned counterfactual usefulness R(x|Q) = E[Delta DecisionQuality|x]
— which on TinySeasons is EXACTLY computable (the extractive answerer is the
decision procedure), so the estimator is validated against the true quantity. The
estimator implements VoI + CausalReach + ModelRepair − AttentionCost − the
adversarial salience-hijack term rho*Sal*(1−VoI): the shinier something is while
not touching the question, the more it is penalized.

Results (1,457 probes, 90-token budget, same store):

| pager | accuracy |
| --- | --- |
| salience (query-blind attention prior) | 0.608 |
| two-stage (salience proposes, relevance disposes) | 0.877 |
| relevance (query-conditioned) | **0.988** |

Estimator validation (125 queries with exact counterfactuals): the top-1 salience
pick is truly relevant **4%** of the time; the top-1 relevance-estimate pick, **81%**.
Quadrant counts: 2,856 high-salience/low-relevance (shiny traps) vs 101
low-salience/high-relevance (boring footnotes that decide answers) vs 59 aligned —
salience agrees with counterfactual usefulness ~2% of the time here. "We overweight
salience and then retroactively call it relevance" is now a measured failure mode.

Honest scope: TinySeasons state-queries are needle-shaped, which maximizes the
split, and relevance is query-conditioned by construction — that IS the thesis
(relevance does not exist without a Q/G), but transfer to fuzzier objectives is
untested. Architecturally the split is the attention-level instance of
propose-cheap/verify-exact: salience is the proposer, relevance is the verifier,
and the two-stage pager (0.877 at a fraction of the scoring cost) is the practical
deployment shape.

## 11. Ninth pass: the pilot seat (gates tuned from inside; a channel-capacity null)

Changes made from inside the cockpit: (1) the ANSWER gate — unverified answers now
auto-convert to CHECK, closing the discipline hole the 4B run exposed; (2) a
chronometer — the prompt carries a monotonic step clock ("rewinds restore your
transcript, not the clock") and notes are timestamped, giving episodes a real arrow
of time; (3) a loop-breaker — consecutive REPEAT verdicts heat the sampler, since
perseveration is a low-temperature attractor; (4) the PAR MCP server now owns a
persistent memory store with emu_memory_remember / emu_memory_page, so an external
agent can plant cross-session memories that page into any future episode.

The Claude-as-pilot experiment (scripts/gundam_run.py): inject the pilot's analysis
— INCLUDING the literal correct expression — into a Qwen3-1.7B episode mid-flight,
gate on, and measure conversion. Result: solo 0/6, piloted 0/6, across two channel
designs (notepad injection, then transcript-tail injection after diagnosing the
notepad as positionally low-bandwidth). Episode traces show the failure mode:
the 1.7B perseverates on its own kitchen-sink expression, and even at the recency
position cannot faithfully read-copy-CHECK a supplied expression. The finding,
stated plainly: **there is a competence floor below which piloting degenerates into
puppeteering** — the poke channel can carry strategy to a model that can transcribe
(the 4B converts checked hints), but not to one that cannot; below the floor the
only working intervention is the harness pressing CHECK itself, at which point the
small model contributes nothing. Channel capacity is bounded by the receiver's
transduction fidelity, not the sender's intelligence.

Friction log from the pilot seat (each now fixed or filed): benchmark rows dropped
episode events (fixed in probes); notepad-vs-transcript injection asymmetry
(documented above; both poke tools exist for this reason); no live closed-loop
piloting from inside a Claude Code session driving Python directly — the MCP
cockpit is wired and registered in .mcp.json for exactly that: start a session in
this repo and the reasoning-emulator server gives interactive emu_step /
emu_poke_* / emu_memory_* controls.

## 12. Tenth pass: PLAY — verifier bootstrapping by game invention (all five gaps wired)

The user's insight, taken literally: the answer to the verifier-coverage gap is
play. A game is a voluntarily adopted rule-set that makes activity gradeable —
"the floor is lava" compiles a fuzzy world into a decision procedure, and kids
arguing "that doesn't count!" are doing adversarial verifier refinement.
`bcv/playground.py` mechanizes it on a sticks-and-stones substrate (row of cells,
place/shift/capture, win-condition grammar): candidate games are proposed from a
grammar and CERTIFIED by a meta-verifier — terminates decisively, seat-balanced,
skill beats luck (1-ply >> random), depth beats skill (2-ply >> 1-ply: the Reactor
knob readout of WHICH cognition the game trains), and resists rule-lawyering (a
degenerate spam policy must not dominate).

Run: 72 invented games -> **5 certified**, all training lookahead. The rejections
are interpretable playground arguments: k=2 games die as "hackable" (spam wins —
too easy to be a game), 3-in-row on 7 cells dies as "unbalanced" (first seat wins
84.5%). The certified set — mobility/blocking games with shift, capture-variant
4-in-a-row, a race with capture — reads like games a child would actually keep.

The other gaps, wired in the same pass:
- Gap 2 (experience supply): every certified game is a mill — 1,937 exactly-graded
  (state -> deeper-search move) rows from one run, free, unlimited. The continual
  learner's starvation problem now has a supply line.
- Gap 3 (standing question): memstore gained a goal ledger (kind='goal', active
  goals expose Q-entities for relevance conditioning) — relevance does not exist
  without a Q, so the Q now persists.
- Gap 4 (transduction): the emulator accepts structured JSON control packets
  ({"control": "CHECK", "arg": ...}) alongside prose — an exact channel for
  receivers below the prose read-copy-emit floor.
- Gap 5 (consolidation): skills folded across games TRANSFER — the "extend your
  line" prior consolidated from two 3-in-row games beats plain greedy on a
  held-out 4-in-row game at **0.71** winrate, despite the prior agreeing with
  depth-2 play only 51.7% of the time: a lossy abstraction that is nonetheless
  portable, which is what a skill IS.

## 13. The cook: the full loop, unattended, cold start to ceiling

scripts/cook.py ran the whole machine on the 4B without supervision: growing
verified-experience buffer (40 -> 94 -> 214 rows, round 3 mixing in playground
game experience), QLoRA retrain each round, promotion gated on the strict
refinement probe. Results: round 1 went **0/8 -> 8/8** (cold start to probe
ceiling on 40 milled examples, promoted); rounds 2 and 3 held 8/8 and the gate
correctly declined lateral promotions. The round-3 answer to the open question:
cross-task game experience did NOT interfere with the repair skill (8/8 held, all
distinct outputs) — no catastrophic interference at this scale, consistent with
the continuity results. Training loss rose with buffer diversity (0.03 -> 0.18)
while probe performance stayed at ceiling, which is the multi-task tax landing in
the loss, not in the skill. GPU peaked at 53 C; ~2 h unattended; resume never
needed. Known limitation, stated plainly: the probe is saturated — rounds 2-3
"no gain" is indistinguishable from "gain masked by ceiling." Next probe needs
headroom: stress heldout + MIS-domain repairs + game-move accuracy as a composite.

## 14. Twelfth pass: the examiner becomes the second learner

`bcv/examiner.py` implements the evolving exam bank with stricter gates than the
student: root oracles per domain (exact verifiers + stress pools + simulator) end
the examiner-of-examiner regress; repair items carry a CHECKER SPEC instead of an
answer key (any verified strict refinement surviving the stress pool passes — no
answer distribution exists to leak); leakage is row-identity against the student's
actual training buffers; discrimination is learned from grading history; downward-
only bucket flow (private_promotion_exam -> public_regression -> trainable) turns
retired exams into student fuel automatically.

First minting: 24 items, **13 quarantined for leakage** — the student trained on
nearly the whole coloring candidate space, and the quarantine caught it, so the
promoted bank (11 items) automatically tilted toward MIS repairs and game moves:
cross-domain transfer testing EMERGED from the leakage rule rather than being
designed in.

First grading: base 4B 0/11, cook-promoted adapter **0/11**. The saturated 8/8
probe concealed a frontier the examiner exposed in one round: the student's repair
skill does not survive the domain shift (coloring->MIS) plus prompt-format shift
(evidence-rich training format -> terse exam format). Two lessons recorded: (i) a
game-item validity bug (rules omitted from the prompt — no player could compute
safe moves) was caught by exactly the review the lifecycle prescribes and fixed;
(ii) 0-vs-0 yields zero discrimination — an exam everyone fails is as blind as one
everyone aces, so promotion policy should require items that BRACKET the frontier
(some graded system passes, some fails), which failure-mining will supply once the
next student round trains against MIS + exam-format data. The examiner's first act
was to unsaturate the ceiling and hand the student loop its next hill: cross-domain
+ cross-format generalization.

## 15. Thirteenth pass: the arcade — real games, real cognitive reactors

`bcv/arcade.py` replaces the 1D toys with exact simulators for the classics:
Connect-4 (real gravity rules), Gomoku 7x7, Othello 6x6 (full flipping + pass
rules), and Hex 7x7 (connection game, provably drawless). One uniform strength
ladder with nothing to hand-tune: Monte-Carlo move choice with R rollouts
(random < MC-4 < MC-16 < MC-64) — more rollouts is strictly more compute, and the
meta-verifier confirms it is more strength.

All four CERTIFIED by the same gates that graded the toys, and the certification
numbers are a capability fingerprint per game (the Reactor thesis on real games):

| game | skill (mc4 vs random) | depth (mc16 vs mc4) | spam resistance |
| --- | --- | --- | --- |
| connect4 | 1.00 | 0.75 | 0.10 |
| gomoku | 1.00 | 0.65 | 0.50 |
| othello6 | 0.95 | 0.775 | 0.05 |
| hex7 | 0.95 | **1.00** | 0.00 |

Hex is the purest search-depth reactor of the four (MC-16 never lost to MC-4),
matching its reputation for deep connectivity reasoning; Othello shows the largest
depth gradient among the fill-the-board games, consistent with shallow evaluation
being famously punished there.

Also shipped: 120 oracle-labeled positions milled (state -> MC-64 move) as student
fuel; 12 frontier exam items (positions where MC-64 disagrees with MC-4) promoted
into the exam bank — leakage-clean by construction, new domain; and the SKILL
LEDGER (`.bcv_runs/arcade/skill_ledger.jsonl`): an append-only growth curve of
every system's winrate per game per ladder rung, seeded with the MC-4 baseline.
"Getting smarter" is now a monotonicity requirement on a file: each adapter
generation must not regress its ladder ratings to promote. Honest boundary noted:
chess and Go stay out until a real engine oracle (Stockfish/KataGo) is wired in —
weak-oracle supervision would poison the mill.

## 16. Fourteenth pass: grandmaster oracles — Stockfish wired, KataGo verified

The rule that kept M.U.G.E.N and Unity out (no exact oracle -> no admission) is the
rule that let these in. `tools/stockfish/` holds the official avx2 build;
`bcv/grandmaster.py` runs the uniform pipeline on real chess with python-chess:
ladder = random < depth-1 < depth-4 < depth-8 (strictly increasing by
construction), fingerprint CERTIFIED (skill d1-vs-random 1.0, depth d4-vs-d1 0.65,
spam 0.0), 40 positions milled from shallow self-play labeled with Stockfish
depth-12 best moves (frontier rate 0.15 — d12 disagrees with d2 on 15% of
positions), 6 frontier exam items promoted into the bank (new domain,
leakage-clean), chess rungs appended to the skill ledger.

KataGo: opencl binary + b6c96 net downloaded to `tools/katago/`; GTP smoke test
passed — 9x9 `genmove b` returned F5 from the neural net. The Go mill (two-process
visit ladder: maxVisits=2 shallow vs maxVisits=64 oracle, same
mill/exam/ledger stages) is the queued next module; the oracle itself is verified
working. M.U.G.E.N and Unity declined with reasons on record: closed real-time
engines with no programmatic ground truth would poison the mill — the native
discrete fighting-game abstraction (frame-data as a certified game) is the
harness-compatible path to that cognition.

The pipeline is now uniform across five game domains plus three research domains,
every one anchored to an oracle that cannot practically be wrong: exhaustive
enumeration, simulated annealing proof, simulators, Monte-Carlo consensus,
Stockfish, and (verified, pending mill) KataGo.

## 17. Fifteenth pass: cook 2 — one student, six ladders, first frontier crack

`bcv/baduk.py` completed the oracle set: two KataGo engines over the same net,
shallow (maxVisits 2) driving self-play, oracle (48) labeling sampled positions via
genmove+undo+resync. Frontier rate **0.417** — v48 disagrees with v2 on 42% of Go
positions vs chess's 15%, the branching factor made visible. 4 Go exam items
promoted; the bank now spans six domains (coloring, MIS, playground, arcade,
chess, go), 33 promoted items.

`scripts/cook2.py` ran the multi-domain round with the hard asymmetry enforced in
code: every FEN, board state, move sequence, and original expression in the
current promoted bank was filtered from the 412-row training buffer before
training (26 rows excluded). Buffer composition: 214 round-3 rows + 48 terse
exam-format repairs (coloring AND MIS — aimed at the two frontiers the examiner
exposed) + 34 chess + 108 arcade + 8 go.

Grading on the 33-item bank: base **1/33**; gen-2 **4/33** — and the movement is
exactly where the buffer aimed: **MIS 1/2 (the cross-domain repair frontier
cracked — first MIS exam pass in the system's history)**, playground 3/20 (from
1/20), chess 0/6 and go 0/4 unchanged (34 and 8 rows cannot move a 4B on engine
move-prediction; those ladders need orders more mill volume, which is free).
The bank's earlier blindness is also cured: four items now DISCRIMINATE (gen-2
passes, base fails) — the exam brackets the frontier for the first time, so
promotion decisions have signal. The retained coloring check was subsequently
closed: gen-2 verified 4/8 against base 0/8. The full 33-item promotion decision
still BLOCKED on four gains, one playground regression, and exact p=.375.

## 18. Product gate, exposure incident, and adversarial calibration

The product layer now emits signed JSON/HTML promotion reports with exact
McNemar evidence, bank and grade-event hashes, an instrument-resolution
statement, strict or reliability-aware regression policy, and CLI exit codes.
MCP, ACP, HTTP, command, stored-answer, and local Transformers candidates all
use the same registry path.

A public KataGo transcript exposed move prefixes that later low-visit mills
reproduced. Nine affected Go items across the isolated and original banks were
burned. The permanent remediation is fresh-clone safe: HEAD contains only
SHA-256 commitments to 22 public prefixes, minting checks those commitments by
default, and Go mills start from an unseeded 4-8-stone random opening.

The 32B paraphrase attack generated 360 DSL rewrites. Against the stated truth
horizon, n<=6 fingerprints caught 143/143 equivalent rewrites (zero quarantine
evasions) and over-quarantined 0/200 distinct rewrites. The one-sided 95% upper
bound after zero false positives is 1.4867%; this is finite-horizon behavioral
calibration, not universal semantic duplicate detection.

The fuzzy-domain result failed in the useful direction. The lexical support
panel scored 0/13 on the hard authored corpus with six false accepts; a local
qwen3:8b semantic veto reached 12/13 but retained one false accept. Production
support minting now fails closed until an independently reviewed hard
calibration is supplied explicitly.

## 19. Same-bank scale ladder and the routed gen-3 PASS

Eight stock models were graded on one 48-item bank. The Qwen2.5 generalist
ladder rose monotonically from 8/48 at 1.5B to 20/48 at 32B, but every stock
model scored 0/24 on graph repairs. The 1.5B-vs-32B paired contrast produced
12 gains, zero regressions, and p=.00048828125; it remains significant after
Bonferroni correction across all 28 possible pairs (adjusted p=.0137).

The missing trained-student comparison then ran locally on the RTX 5060 using
the exact same bank, item set, prompt protocol, and 384-token ceiling:

- FastContext-4B base: **22/48** (code 21/24, graph 1/24).
- Gen-2 adapter: **26/48** (code 19/24, graph 7/24), but **BLOCK** on six gains,
  two stable code regressions, p=.2890625. Three fresh loads produced one
  identical 48-item outcome vector.
- Routed gen-3: **28/48** (code 21/24, graph 7/24). One PEFT model enables the
  adapter only for repair prompts and disables it elsewhere. Against base it
  earned **PASS**: six gains, zero regressions, 42 ties, exact p=.03125. Three
  fresh loads again produced one identical outcome vector.

Against stock Qwen2.5-32B, routed gen-3 has nine gains and one regression
(p=.021484375), so the strict zero-regression gate still BLOCKS that comparison
despite the 28/48 vs 20/48 aggregate. This is the intended distinction between
"higher score" and "earned promotion."

## Commands

See README ("hard repair dataset", "stress-test", "stress-mined") for the exact
invocations. `python -m pytest` covers the DSL analysis, dataset invariants
(no leak, group disjointness, distinct evidence), and the adversarial families
(214 tests in the current suite).
