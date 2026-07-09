"""Promotion decisions as auditable artifacts, not score deltas in a log.

The examiner is useful only if its decision contract is stricter than "the
candidate got a bigger number."  This module keeps the evidence paired at the
item level, checks retained probes, performs an exact McNemar test, and emits a
self-contained JSON/HTML gate report.  It never writes to the private bank.
"""

from __future__ import annotations

import hashlib
import html
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bcv.examiner import ExaminerBank


@dataclass(frozen=True)
class GatePolicy:
    min_gains: int = 1
    max_regressions: int = 0
    confidence_alpha: float = 0.05
    require_retained_probe: bool = True
    regression_policy: str = "strict"  # strict | reliability_aware
    max_noisy_regressions: int = 1
    reliability_min_observations: int = 3
    stable_flip_rate: float = 0.05


def exact_mcnemar_p_value(gains: int, regressions: int) -> float:
    """Two-sided exact McNemar/binomial p-value for paired binary outcomes."""
    discordant = gains + regressions
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(gains, regressions) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def minimum_clean_discordant_wins(alpha: float) -> int:
    """Smallest clean win-only paired signal that can clear ``alpha``."""
    wins = 0
    while exact_mcnemar_p_value(wins, 0) > alpha:
        wins += 1
    return wins


def power_statement(gains: int, regressions: int, alpha: float) -> dict[str, int | float]:
    """State this bank's observed paired-evidence resolution honestly.

    This is not a prospective sample-size guarantee. It says what the observed
    non-tie capacity could certify if every discordant item were a clean gain,
    and how many more independent clean discriminators are needed under the
    same exact test.
    """
    discordant = gains + regressions
    minimum_clean = minimum_clean_discordant_wins(alpha)
    additional_with_current_regressions = 0
    while exact_mcnemar_p_value(gains + additional_with_current_regressions, regressions) > alpha:
        additional_with_current_regressions += 1
    return {
        "alpha": alpha,
        "observed_discordant_items": discordant,
        "minimum_clean_discordant_wins": minimum_clean,
        "best_case_p_if_observed_discordants_were_all_clean": exact_mcnemar_p_value(discordant, 0),
        "additional_clean_discriminators_needed_if_current_outcomes_were_clean": max(0, minimum_clean - discordant),
        "additional_clean_wins_needed_if_current_regressions_persist": additional_with_current_regressions,
    }


def item_flakiness(item, min_observations: int) -> dict[str, int | float] | None:
    """Estimate repeated-grade instability without mistaking model differences for noise."""
    minority = total = systems = 0
    for stats in item.graded.values():
        observations = stats["pass"] + stats["fail"]
        if observations < min_observations:
            continue
        systems += 1
        total += observations
        minority += min(stats["pass"], stats["fail"])
    if not total:
        return None
    return {"systems": systems, "observations": total, "flip_rate": minority / total}


def bank_hash(bank: ExaminerBank) -> str:
    """Commit to the complete private-bank state without disclosing its contents."""
    rows = [asdict(bank.items[item_id]) for item_id in sorted(bank.items)]
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def latest_grade_event(path: str | Path, system: str) -> dict:
    """Load the latest complete item-level grade event for one system."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    for line in reversed(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("event") != "grade" or event.get("system") != system:
            continue
        results = event.get("results")
        if not isinstance(results, dict) or not all(isinstance(item_id, str) and isinstance(passed, bool) for item_id, passed in results.items()):
            raise ValueError(f"malformed grade event for {system}")
        return event
    raise ValueError(f"no grade event found for {system} in {path}")


def latest_grade_event_results(path: str | Path, system: str) -> dict[str, bool]:
    return latest_grade_event(path, system)["results"]


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def retained_probe_summary(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    payload = payload.get("eval", payload)
    base = payload.get("base_verified")
    candidate = payload.get("adapter_verified")
    total = payload.get("eval_examples")
    if not all(isinstance(value, int) for value in (base, candidate, total)):
        raise ValueError("retained probe must include integer base_verified, adapter_verified, and eval_examples")
    return {
        "base_verified": base,
        "candidate_verified": candidate,
        "items": total,
        "delta": candidate - base,
        "no_regression": candidate >= base,
    }


def build_gate_report(
    bank: ExaminerBank,
    baseline: str,
    candidate: str,
    baseline_results: dict[str, bool],
    candidate_results: dict[str, bool],
    retained_probe: dict[str, Any] | None = None,
    policy: GatePolicy = GatePolicy(),
    grade_runs: dict[str, dict] | None = None,
) -> dict[str, Any]:
    if set(baseline_results) != set(candidate_results):
        raise ValueError("baseline and candidate must be graded on the same item ids")
    missing = set(baseline_results) - set(bank.items)
    if missing:
        raise ValueError(f"results reference items missing from bank: {sorted(missing)}")

    rows = []
    gains = regressions = ties = 0
    by_domain: dict[str, dict[str, int]] = {}
    regression_reliability = []
    for item_id in sorted(baseline_results):
        base = bool(baseline_results[item_id])
        contender = bool(candidate_results[item_id])
        outcome = "gain" if contender and not base else "regression" if base and not contender else "tie"
        gains += outcome == "gain"
        regressions += outcome == "regression"
        ties += outcome == "tie"
        domain = bank.items[item_id].domain
        counts = by_domain.setdefault(domain, {"items": 0, "gains": 0, "regressions": 0, "ties": 0})
        counts["items"] += 1
        counts[f"{outcome}s"] += 1
        rows.append({"item_id": item_id, "domain": domain, "baseline": base, "candidate": contender, "outcome": outcome})
        if outcome == "regression":
            reliability = item_flakiness(bank.items[item_id], policy.reliability_min_observations)
            if reliability is None:
                classification = "unknown"
            elif reliability["flip_rate"] <= policy.stable_flip_rate:
                classification = "stable"
            else:
                classification = "noisy"
            regression_reliability.append({"item_id": item_id, "domain": domain, "classification": classification, "reliability": reliability})

    retained = retained_probe_summary(retained_probe)
    p_value = exact_mcnemar_p_value(gains, regressions)
    resolution = power_statement(gains, regressions, policy.confidence_alpha)
    reasons = []
    stable_regressions = [row for row in regression_reliability if row["classification"] == "stable"]
    noisy_regressions = [row for row in regression_reliability if row["classification"] == "noisy"]
    unknown_regressions = [row for row in regression_reliability if row["classification"] == "unknown"]
    if policy.regression_policy not in {"strict", "reliability_aware"}:
        raise ValueError(f"unknown regression policy: {policy.regression_policy}")
    if policy.regression_policy == "strict" and regressions > policy.max_regressions:
        verdict = "BLOCK"
        reasons.append(f"{regressions} regression(s) exceed the policy limit of {policy.max_regressions}")
    elif policy.regression_policy == "reliability_aware" and stable_regressions:
        verdict = "BLOCK"
        reasons.append(f"{len(stable_regressions)} regression(s) are on historically stable item(s)")
    elif policy.regression_policy == "reliability_aware" and len(noisy_regressions) > policy.max_noisy_regressions:
        verdict = "BLOCK"
        reasons.append(f"{len(noisy_regressions)} noisy regression(s) exceed the budget of {policy.max_noisy_regressions}")
    elif policy.require_retained_probe and (retained is None or not retained["no_regression"]):
        verdict = "BLOCK"
        reasons.append("retained probe is absent or regressed")
    elif policy.regression_policy == "reliability_aware" and unknown_regressions:
        verdict = "HOLD"
        reasons.append(f"{len(unknown_regressions)} regression(s) lack repeated-grade reliability evidence")
    elif gains < policy.min_gains:
        verdict = "HOLD"
        reasons.append(f"{gains} gain(s) are below the policy minimum of {policy.min_gains}")
    elif p_value > policy.confidence_alpha:
        verdict = "HOLD"
        reasons.append(
            f"paired exact McNemar p={p_value:.6g} exceeds alpha={policy.confidence_alpha:.6g}; collect more discrimination"
        )
    else:
        verdict = "PASS"
        reasons.append("no regressions, retained probe held, and paired evidence passed the confidence threshold")

    statuses: dict[str, int] = {}
    for item in bank.items.values():
        statuses[item.status] = statuses.get(item.status, 0) + 1
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "reasons": reasons,
        "systems": {"baseline": baseline, "candidate": candidate},
        "policy": asdict(policy),
        "paired_evidence": {
            "items": len(rows),
            "gains": gains,
            "regressions": regressions,
            "ties": ties,
            "exact_mcnemar_two_sided_p": p_value,
            "resolution": resolution,
            "by_domain": dict(sorted(by_domain.items())),
            "items_detail": rows,
            "regression_reliability": regression_reliability,
        },
        "retained_probe": retained,
        "grade_runs": grade_runs or {},
        "bank": {
            "sha256": bank_hash(bank),
            "grade_events_sha256": file_sha256(bank.root / "grade_events.jsonl"),
            "status_counts": dict(sorted(statuses.items())),
        },
    }


def render_gate_html(report: dict[str, Any]) -> str:
    evidence = report["paired_evidence"]
    resolution = evidence["resolution"]
    retained = report.get("retained_probe") or {}
    grade_runs = html.escape(json.dumps(report.get("grade_runs", {}), indent=2, sort_keys=True))
    reasons = "<br>".join(html.escape(reason) for reason in report["reasons"])
    domain_rows = "".join(
        "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
            html.escape(domain), counts["items"], counts["gains"], counts["regressions"], counts["ties"]
        )
        for domain, counts in evidence["by_domain"].items()
    )
    color = {"PASS": "#176b3a", "HOLD": "#8a5a00", "BLOCK": "#9b1c1c"}[report["verdict"]]
    return f"""<!doctype html>
<html><head><meta charset=\"utf-8\"><title>Whetstone promotion gate</title>
<style>body{{font:16px system-ui,sans-serif;max-width:900px;margin:48px auto;color:#18212b}}h1{{margin-bottom:4px}}.verdict{{font-size:2.5rem;font-weight:800;color:{color}}}table{{border-collapse:collapse;width:100%;margin-top:18px}}th,td{{border:1px solid #ccd3db;padding:8px;text-align:left}}th{{background:#f3f5f7}}code{{font-size:.82rem;word-break:break-all}}</style></head>
<body><h1>Whetstone promotion gate</h1><div class=\"verdict\">{html.escape(report["verdict"])}</div>
<p>{reasons}</p><h2>Paired private-bank evidence</h2>
<p>{evidence["gains"]} gains, {evidence["regressions"]} regressions, {evidence["ties"]} ties across {evidence["items"]} items.<br>Exact two-sided McNemar p = <strong>{evidence["exact_mcnemar_two_sided_p"]:.6g}</strong>.</p>
<h2>Instrument resolution</h2><p>At alpha={resolution["alpha"]:.6g}, a clean paired result needs {resolution["minimum_clean_discordant_wins"]} discordant wins and zero regressions. This bank observed {resolution["observed_discordant_items"]} discordant items; even if all were clean wins, p would be {resolution["best_case_p_if_observed_discordants_were_all_clean"]:.6g}. It needs {resolution["additional_clean_discriminators_needed_if_current_outcomes_were_clean"]} more independent clean discriminator(s) to make that best case certifiable.</p>
<h2>Retained probe</h2><p>{retained.get("candidate_verified", "n/a")}/{retained.get("items", "n/a")} candidate verified vs {retained.get("base_verified", "n/a")}/{retained.get("items", "n/a")} baseline; delta {retained.get("delta", "n/a")}.</p>
<h2>Grading-run provenance</h2><pre>{grade_runs}</pre>
<h2>Domain evidence</h2><table><tr><th>domain</th><th>items</th><th>gains</th><th>regressions</th><th>ties</th></tr>{domain_rows}</table>
<h2>Audit commitment</h2><p>Private bank SHA-256: <code>{report["bank"]["sha256"]}</code></p></body></html>"""


def write_gate_report(report: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "promotion_gate.json"
    html_path = output_dir / "promotion_gate.html"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    html_path.write_text(render_gate_html(report), encoding="utf-8")
    return json_path, html_path
