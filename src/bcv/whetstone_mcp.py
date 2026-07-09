"""Whetstone as an MCP server: the promotion gate as agent-callable tools.

Any MCP client — Claude, an orchestrator, a CI bot — can mint exam items,
grade candidates, ask for a promotion verdict, and inspect bank health. The
trust boundary is identical to the HTTP service and it is the whole point:

- NO tool returns exam item prompts or payloads. There is no "list items"
  tool. An MCP client may be (or may be steering) the very system under exam;
  serving it the test would be the contamination this product exists to stop.
- Grading happens server-side. The client names a candidate (a shell command,
  a local endpoint, or submits answers by item id); whetstone talks to the
  candidate itself, so private prompts never transit the MCP channel.
- External endpoints trigger burn accounting exactly as in the CLI, and the
  override is a loud explicit argument, never a default.

Run: $env:PYTHONPATH='src'; python -m bcv.whetstone_mcp
(registered in the repo's .mcp.json as `whetstone`)

Implementation functions are module-level and MCP-free so the test suite can
call them directly; the @mcp.tool wrappers only serialize.
"""

from __future__ import annotations

import json
import os
import shlex

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("whetstone")

_BANK_ROOT = os.environ.get("WHETSTONE_ROOT", ".bcv_runs/examiner")


def _bank():
    from bcv.examiner import ExaminerBank

    return ExaminerBank(_BANK_ROOT)


# ------------------------------------------------------------ implementations


def use_bank_impl(root: str) -> dict:
    global _BANK_ROOT
    _BANK_ROOT = root
    bank = _bank()
    return {"root": str(bank.root), "items": len(bank.items)}


def status_impl() -> dict:
    from bcv.service import status_payload

    return {"root": _BANK_ROOT, **status_payload(_bank())}


def metabolism_impl(output_dir: str | None = None) -> dict:
    from bcv.metabolism import metabolism_report, write_metabolism_report

    bank = _bank()
    report = metabolism_report(bank.root)
    json_path, html_path = write_metabolism_report(bank.root, output_dir or str(bank.root / "metabolism"))
    return {
        "sustainability": report["sustainability"],
        "events_total": report["events_total"],
        "current_promoted_supply": report["current_promoted_supply"],
        "report_json": str(json_path),
        "report_html": str(html_path),
    }


def mint_impl(domain: str, max_items: int = 8, seed: int = 0, buffers: list[str] | None = None) -> dict:
    from bcv.registry import MINTABLE_DOMAINS, mint_domain

    if domain not in MINTABLE_DOMAINS:
        raise ValueError(f"unknown domain {domain!r}; mintable: {MINTABLE_DOMAINS}")
    return mint_domain(_bank(), domain, buffers=buffers or [], max_items=max_items, seed=seed)


def grade_answers_impl(system: str, answers: dict) -> dict:
    from bcv.service import grade_payload

    return grade_payload(_bank(), {"system": system, "answers": answers}, config={})


def grade_command_impl(system: str, command: str, max_items: int | None = None, timeout_seconds: float = 120.0) -> dict:
    from bcv.candidates import CommandCandidate
    from bcv.registry import grade_bank

    candidate = CommandCandidate(shlex.split(command), timeout_seconds=timeout_seconds)
    report = grade_bank(_bank(), system=system, candidate=candidate, max_items=max_items)
    report.pop("results", None)
    return report


def grade_endpoint_impl(
    system: str,
    api_base: str,
    model: str,
    api_key_env: str | None = None,
    max_items: int | None = None,
    timeout_seconds: float = 120.0,
    max_tokens: int = 512,
    allow_external_no_burn: bool = False,
) -> dict:
    from bcv.candidates import OpenAICompatibleCandidate
    from bcv.registry import grade_bank

    candidate = OpenAICompatibleCandidate(
        base_url=api_base,
        model=model,
        api_key=os.environ.get(api_key_env) if api_key_env else None,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
    )
    report = grade_bank(
        _bank(),
        system=system,
        candidate=candidate,
        max_items=max_items,
        burn_external=not allow_external_no_burn,
    )
    report.pop("results", None)
    if candidate.is_external and allow_external_no_burn:
        report["warning"] = "external endpoint graded WITHOUT burn accounting (explicit override)"
    return report


def gate_impl(
    baseline: str,
    candidate: str,
    alpha: float | None = None,
    min_gains: int | None = None,
    max_regressions: int | None = None,
    regression_policy: str = "strict",
    max_noisy_regressions: int = 1,
    reliability_min_observations: int = 3,
    stable_flip_rate: float = 0.05,
    out_dir: str | None = None,
) -> dict:
    from bcv.gate import GatePolicy, build_gate_report, latest_grade_event, write_gate_report

    bank = _bank()
    policy = GatePolicy(
        min_gains=1 if min_gains is None else min_gains,
        max_regressions=0 if max_regressions is None else max_regressions,
        confidence_alpha=0.05 if alpha is None else alpha,
        require_retained_probe=False,
        regression_policy=regression_policy,
        max_noisy_regressions=max_noisy_regressions,
        reliability_min_observations=reliability_min_observations,
        stable_flip_rate=stable_flip_rate,
    )
    events = bank.root / "grade_events.jsonl"
    baseline_event = latest_grade_event(events, baseline)
    candidate_event = latest_grade_event(events, candidate)
    baseline_results = baseline_event["results"]
    candidate_results = candidate_event["results"]
    if set(baseline_results) != set(candidate_results):
        raise ValueError("baseline and candidate must be graded on the identical item cohort")
    shared = sorted(baseline_results)
    report = build_gate_report(
        bank,
        baseline=baseline,
        candidate=candidate,
        baseline_results={item: baseline_results[item] for item in shared},
        candidate_results={item: candidate_results[item] for item in shared},
        retained_probe=None,
        policy=policy,
        grade_runs={
            "baseline": baseline_event.get("run_manifest", {}),
            "candidate": candidate_event.get("run_manifest", {}),
        },
    )
    json_path, html_path = write_gate_report(report, out_dir or str(bank.root / f"gate_{candidate}"))
    evidence = report["paired_evidence"]
    return {
        "verdict": report["verdict"],
        "reasons": report["reasons"],
        "gains": evidence["gains"],
        "regressions": evidence["regressions"],
        "ties": evidence["ties"],
        "p_value": evidence["exact_mcnemar_two_sided_p"],
        "resolution": evidence["resolution"],
        "report_json": str(json_path),
        "report_html": str(html_path),
    }


def burn_impl(item_id: str, provider: str, reason: str) -> dict:
    bank = _bank()
    bank.burn(item_id, provider=provider, reason=reason)
    bank.save()
    return {"burned": item_id, "provider": provider, "reason": reason}


def calibrate_panel_impl(labeled_path: str = "sample_docs/support_calibration.jsonl") -> dict:
    from pathlib import Path

    from bcv.panel import SUPPORT_PANEL, calibrate_panel

    triples = []
    for line in Path(labeled_path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            triples.append((row["case"], row["answer"], bool(row["human_pass"])))
    return calibrate_panel(SUPPORT_PANEL, triples).to_dict()


# ------------------------------------------------------------------ MCP tools


@mcp.tool()
def whetstone_use_bank(root: str) -> str:
    """Point subsequent tools at a bank root (default: WHETSTONE_ROOT env or .bcv_runs/examiner)."""
    return json.dumps(use_bank_impl(root), sort_keys=True)


@mcp.tool()
def whetstone_status() -> str:
    """Bank health: bucket counts, promoted-by-domain, discrimination, graded systems.
    Item contents are never returned by this or any other tool."""
    return json.dumps(status_impl(), sort_keys=True)


@mcp.tool()
def whetstone_metabolism(output_dir: str | None = None) -> str:
    """Write a safe bank-sustainability report from append-only mint, promotion,
    retirement, and burn events. It exposes rates and no item contents."""
    return json.dumps(metabolism_impl(output_dir), sort_keys=True)


@mcp.tool()
def whetstone_mint(domain: str, max_items: int = 8, seed: int = 0, buffers: list[str] | None = None) -> str:
    """Mint exam items for a domain (coloring, mis, playground, code, support).
    Pass training-buffer paths so the leakage quarantine can check overlap."""
    return json.dumps(mint_impl(domain, max_items, seed, buffers), sort_keys=True)


@mcp.tool()
def whetstone_grade_answers(system: str, answers_json: str) -> str:
    """Grade stored answers: answers_json is a JSON object {item_id: answer}.
    Use when the caller already ran its candidate elsewhere."""
    answers = json.loads(answers_json)
    if not isinstance(answers, dict):
        raise ValueError("answers_json must be a JSON object keyed by item id")
    return json.dumps(grade_answers_impl(system, answers), sort_keys=True)


@mcp.tool()
def whetstone_grade_command(system: str, command: str, max_items: int | None = None, timeout_seconds: float = 120.0) -> str:
    """Grade a candidate agent invoked as a shell command (prompt on stdin, answer
    on stdout). Grading runs server-side: exam prompts never transit MCP."""
    return json.dumps(grade_command_impl(system, command, max_items, timeout_seconds), sort_keys=True)


@mcp.tool()
def whetstone_grade_endpoint(
    system: str,
    api_base: str,
    model: str,
    api_key_env: str | None = None,
    max_items: int | None = None,
    timeout_seconds: float = 120.0,
    max_tokens: int = 512,
    allow_external_no_burn: bool = False,
) -> str:
    """Grade a candidate behind an OpenAI-compatible endpoint. Non-local hosts
    trigger burn accounting: every exposed item is permanently consumed.
    allow_external_no_burn=True is an on-the-record override, never a default."""
    return json.dumps(
        grade_endpoint_impl(system, api_base, model, api_key_env, max_items, timeout_seconds, max_tokens, allow_external_no_burn),
        sort_keys=True,
    )


@mcp.tool()
def whetstone_gate(
    baseline: str,
    candidate: str,
    alpha: float | None = None,
    min_gains: int | None = None,
    max_regressions: int | None = None,
    regression_policy: str = "strict",
    max_noisy_regressions: int = 1,
    reliability_min_observations: int = 3,
    stable_flip_rate: float = 0.05,
    out_dir: str | None = None,
) -> str:
    """Promotion decision from recorded grades: PASS, HOLD, or BLOCK, with paired
    evidence, exact McNemar p-value, the bank's resolution statement, and paths
    to the written JSON/HTML gate report."""
    return json.dumps(
        gate_impl(
            baseline, candidate, alpha, min_gains, max_regressions,
            regression_policy, max_noisy_regressions, reliability_min_observations, stable_flip_rate, out_dir,
        ),
        sort_keys=True,
    )


@mcp.tool()
def whetstone_burn(item_id: str, provider: str, reason: str) -> str:
    """Record an external exposure by hand: the item permanently leaves the
    reusable pools (audit trail keeps it)."""
    return json.dumps(burn_impl(item_id, provider, reason), sort_keys=True)


@mcp.tool()
def whetstone_calibrate_panel(labeled_path: str = "sample_docs/support_calibration.jsonl") -> str:
    """Measure support-panel agreement against labeled human verdicts: agreement,
    false-accepts (the dangerous direction), false-rejects, per-check attribution."""
    return json.dumps(calibrate_panel_impl(labeled_path), sort_keys=True)


if __name__ == "__main__":
    mcp.run()
