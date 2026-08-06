# Changelog

All notable public changes are recorded here. Whetstone follows semantic
versioning for its Python package and public service contract.

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

[0.7.0]: https://github.com/CarlSR9001/whetstone/compare/a37fe57c945ee2a8eb7f10d8cd4e17ebd7f123a0...v0.7.0
[0.6.0]: https://github.com/CarlSR9001/whetstone/compare/v0.5.3...a37fe57c945ee2a8eb7f10d8cd4e17ebd7f123a0
