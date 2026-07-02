# Whetstone — the pitch

**One line: eval and promotion infrastructure that stops AI agents from grading
their own homework.**

## Problem

Every team shipping agents faces the same unanswerable question: is the new
version actually better, or did it memorize the eval? Static benchmarks saturate
and leak into training data. Self-improvement loops are worse: they generate their
own training signal, and without hard separation they train on their own
artifacts and report progress that isn't there.

This repo contains the measured version of that failure, not the anecdote:

- 99 of 121 rules that scored precision 1.0 under a finite verifier were false
  one horizon step later (FINDINGS §3) — self-graded confidence is manufactured
  by the grading horizon itself.
- A saturated 8/8 internal eval concealed a frontier on which the same model
  scored 0/11 the moment a leakage-clean exam existed (FINDINGS §13-14).

## Product wedge

**A private examiner for agentic systems.** Whetstone's exam bank is a second
learner with stricter gates than the student it judges:

- **Leakage quarantine enforced in code** — any item overlapping the training
  corpus is quarantined before promotion; train/test separation is a property of
  the pipeline, not a policy document.
- **Checker-spec grading** — items carry verification procedures, not answer
  keys, so there is no answer distribution to leak.
- **Discrimination learned from use** — items accrue per-system pass/fail
  history; items that stop separating systems retire automatically
  (item-response calibration for agents).
- **Downward-only flow** — retired private exams become public regression items
  and then training fuel; nothing exposed to training re-enters the private bank.
- **Promotion gates** — a new model/adapter/agent version ships only if it
  improves on the private bank without regressing retained probes.

Working end-to-end today on consumer hardware (demonstrated with QLoRA students
on a local 4B): the loop went cold-start to eval-ceiling unattended, the examiner
exposed the hidden frontier, targeted training cracked it, and every step is in
the ledger (FINDINGS §13-17).

## Why this repo is the evidence

- The private promotion exam bank is deliberately absent from this public repo —
  publishing it would let future models train on it. The discipline being sold
  is practiced in the artifact itself.
- Root oracles anchor every label: exhaustive enumeration, adversarial
  counterexample search (which settled a real open conjecture — FINDINGS §5),
  exact game simulators, Stockfish, KataGo.
- An MCP server (`reasoning-emulator`) already lets an external agent step,
  inspect, rewind, poke, and verify a live reasoning session with every
  intervention logged — the audit surface, implemented.

## Market and competition

Agent observability and static evals exist (LangSmith, Braintrust, Patronus,
et al.). None offer leakage-quarantined *evolving* exam banks, checker-spec
grading, or verifier-gated promotion for continually-updated agents. As teams
move from prompt iteration to weight/adapter iteration, the "did it really
improve" question becomes a deployment gate, and the examiner becomes
infrastructure, not tooling.

## Stage, honestly

Public research prototype (repo is days old), solo founder plus AI agents,
pre-revenue, no customers. What exists is a working architecture with receipts
and a documented failure museum. Seeking pre-seed to turn the examiner into a
hosted product: domain adapters for coding/support/research agents, a managed
private-bank service, and CI-style promotion gates.

## Founder

Independent AI researcher building agent harnesses and evaluation
infrastructure; this repo (16 research passes, 118 tests, honest nulls
included) is the working sample.
