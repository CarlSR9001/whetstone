# Changelog

All notable public changes are recorded here. Whetstone follows semantic
versioning for its Python package and public service contract.

## [0.8.0] - 2026-08-08

### Added

- Challenge-bound Ed25519 SSHSIG receipts, a public rotation-aware key bundle,
  offline verification, and one-command hosted report-card/Open Bench runners.
- Trajectory-disjoint Stockfish/KataGo milling and a resumable CUDA Gen-4
  engine-student runner with immutable checkpoints and sanitized commitments.
- SPDX 2.3 release SBOMs, artifact attestations, Dependabot, CodeQL, MCP Registry
  metadata/publication, and opt-in PyPI Trusted Publishing.
- A Windows PowerShell deployment publisher and concise server-side release-state
  validation for the forge, toolbox, and signed-receipt health gates.

### Changed

- Report-card sessions now survive a busy grading worker and can be retried
  safely instead of being consumed before a grade exists.
- CI uses a universal uv lock, pinned third-party actions, official MCP
  conformance coverage, and one stable required-check aggregate.
- Deployment requires an immutable source commit, synchronized forge state,
  successful report-card warm-up, configured publication ledger, and a ready
  persistent receipt signer before activation.

### Research

- The first Gen-4 adapter trained for 1,313 CUDA steps with zero skipped steps
  on 1,313 trajectory-disjoint engine examples. Across three fresh model loads,
  aggregate score rose from 7/69 to 8/69, but the paired gate BLOCKED on three
  gains, two regressions, and exact McNemar p=1.0. No promotion is claimed.

## [0.7.0] - 2026-08-05

### Added

- Sessionless MCP `2026-07-28` discovery and per-request transport alongside
  the initialize-based 2025 protocol revisions.
- Official MCP conformance coverage for the stateless lifecycle, tool listing,
  cache hints, and HTTP header validation.
- Bounded per-tool, per-transport outcome and failure-reason telemetry.
- Python 3.11-3.13 core compatibility and coverage gates, plus provenance-
  attested tag releases with SHA-256 manifests.

### Changed

- Promotion receipts now describe the regression policy actually applied
  instead of implying that PASS requires zero regressions.
- Report cards distinguish checker verification from promotion eligibility;
  verified repairs retaining less than 5% of clean support are flagged as
  degenerate narrowing and do not pass.
- Public contamination language now states the evidence boundary explicitly:
  declared-exposure controls do not prove absence from every training corpus.

## [0.6.0] - 2026-07-31

### Added

- Open Promotion Bench paired scope-integrity cohorts, sanitized opt-in public
  receipts, and its public leaderboard surface.
- Disposable report-card and public toolbox deployment hardening.

[0.8.0]: https://github.com/CarlSR9001/whetstone/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/CarlSR9001/whetstone/compare/a37fe57c945ee2a8eb7f10d8cd4e17ebd7f123a0...v0.7.0
[0.6.0]: https://github.com/CarlSR9001/whetstone/compare/v0.5.3...a37fe57c945ee2a8eb7f10d8cd4e17ebd7f123a0
