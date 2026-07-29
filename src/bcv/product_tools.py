"""Stateless product surfaces over Whetstone's verifier mechanisms.

These functions deliberately operate on caller-supplied, disposable data.  They
never open the private examiner bank and never persist uploads.  The public web
tool is therefore useful for inspection and demonstrations; real private banks
belong in the local CLI or a customer-controlled deployment.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import math
import re
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

from bcv.gate import exact_mcnemar_p_value, power_statement
from bcv.markdown_editor import (
    MarkdownPatch,
    PatchError,
    PatchOperation,
    apply_markdown_patch,
    parse_sections,
    protected_tokens,
)
from bcv.memory_bench import Probe
from bcv.memstore import Memory as StoredMemory
from bcv.relevance import relevance_score, salience_prior


MAX_RECORDS = 5_000
MAX_MARKDOWN_CHARS = 200_000
MAX_EVENT_RECORDS = 5_000
CONTENT_IDENTITY_FIELDS = ("prompt", "content", "input", "task", "question", "expression")
STOPWORDS = {
    "about", "after", "again", "against", "also", "because", "before", "being",
    "between", "could", "current", "does", "from", "have", "into", "memory",
    "objective", "should", "that", "their", "there", "these", "this", "those",
    "through", "what", "when", "where", "which", "while", "with", "would",
}


class ProductInputError(ValueError):
    """A caller-visible schema or safety-boundary error."""


def _sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _records(value: Any, name: str, *, limit: int = MAX_RECORDS) -> list[dict[str, Any]]:
    """Accept a JSON array, JSONL string, or ``{"items": [...]}`` wrapper."""
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = []
            for line_number, line in enumerate(text.splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    parsed.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ProductInputError(f"{name} line {line_number} is not valid JSON") from error
        value = parsed
    if isinstance(value, dict) and isinstance(value.get("items"), list):
        value = value["items"]
    if not isinstance(value, list):
        raise ProductInputError(f"{name} must be a JSON array or JSONL records")
    if len(value) > limit:
        raise ProductInputError(f"{name} exceeds the {limit}-record public-demo limit")
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(value):
        if not isinstance(row, dict):
            raise ProductInputError(f"{name} row {index + 1} must be an object")
        rows.append(dict(row))
    return rows


def _item_id(row: dict[str, Any], name: str, index: int) -> str:
    value = row.get("item_id", row.get("id"))
    if not isinstance(value, (str, int)) or not str(value).strip():
        raise ProductInputError(f"{name} row {index + 1} needs a non-empty item_id")
    return str(value).strip()


def _as_bool(value: Any, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"pass", "passed", "true", "yes", "1"}:
            return True
        if normalized in {"fail", "failed", "false", "no", "0"}:
            return False
    raise ProductInputError(f"{label} must be pass/fail or boolean")


def parse_results(value: Any, name: str) -> dict[str, bool]:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            value = _records(text, name)
    if isinstance(value, dict) and "results" in value:
        value = value["results"]
    if isinstance(value, dict):
        if len(value) > MAX_RECORDS:
            raise ProductInputError(f"{name} exceeds the {MAX_RECORDS}-item public-demo limit")
        parsed: dict[str, bool] = {}
        for item_id, raw in value.items():
            answer = raw.get("passed") if isinstance(raw, dict) else raw
            parsed[str(item_id)] = _as_bool(answer, f"{name}[{item_id!r}]")
        return parsed
    rows = _records(value, name)
    parsed = {}
    for index, row in enumerate(rows):
        item_id = _item_id(row, name, index)
        raw = row.get("passed", row.get("outcome", row.get("result")))
        passed = _as_bool(raw, f"{name} row {index + 1} outcome")
        if item_id in parsed and parsed[item_id] != passed:
            raise ProductInputError(f"{name} has conflicting duplicate outcomes for {item_id}")
        parsed[item_id] = passed
    return parsed


def _identity_tokens(row: dict[str, Any]) -> dict[str, str]:
    tokens: dict[str, str] = {}
    for field in ("item_id", "id", "exposure_key", "content_hash"):
        value = row.get(field)
        if isinstance(value, (str, int)) and str(value).strip():
            normalized = "row_id" if field in {"item_id", "id"} else "content_value_sha256" if field == "content_hash" else field
            tokens[f"{normalized}:{str(value).strip()}"] = field
    for field in CONTENT_IDENTITY_FIELDS:
        if field in row and isinstance(row[field], (str, int, float, bool)):
            digest = hashlib.sha256(str(row[field]).encode("utf-8")).hexdigest()
            tokens[f"content_value_sha256:{digest}"] = f"exact_{field}"
    return tokens


def _content_value(row: dict[str, Any]) -> tuple[str, str] | None:
    for field in CONTENT_IDENTITY_FIELDS:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return field, value.strip()
    return None


def _similarity_terms(text: str) -> set[str]:
    terms = re.findall(r"[a-z0-9]+", text.lower().replace("'s", ""))
    return {term for term in terms if len(term) > 1 and term not in STOPWORDS}


def _text_similarity(left: set[str], right: set[str]) -> tuple[float, int]:
    if not left or not right:
        return 0.0, 0
    shared = len(left & right)
    if shared < 3:
        return 0.0, shared
    jaccard = shared / len(left | right)
    containment = shared / min(len(left), len(right))
    return round(0.6 * jaccard + 0.4 * containment, 6), shared


@lru_cache(maxsize=3)
def _dsl_observations(max_n: int) -> tuple[Any, ...]:
    from bcv.discovery import enumerate_graphs, observe_graph

    return tuple(observe_graph(graph) for graph in enumerate_graphs(max_n))


def audit_leakage(payload: dict[str, Any]) -> dict[str, Any]:
    exam = _records(payload.get("exam", []), "exam")
    exposure = _records(payload.get("exposure", []), "exposure")
    similarity_threshold = float(payload.get("similarity_threshold", 0.6))
    if not 0.5 <= similarity_threshold <= 1.0:
        raise ProductInputError("similarity_threshold must be in [0.5, 1.0]")
    fingerprint_max_n = int(payload.get("fingerprint_max_n", 4))
    if not 3 <= fingerprint_max_n <= 5:
        raise ProductInputError("fingerprint_max_n must be between 3 and 5")

    exposure_index: dict[str, list[dict[str, str]]] = defaultdict(list)
    for index, row in enumerate(exposure):
        source = str(row.get("source") or row.get("path") or f"exposure row {index + 1}")
        for token, reason in _identity_tokens(row).items():
            exposure_index[token].append({"source": source, "reason": reason})

    exam_rows: list[tuple[str, dict[str, Any]]] = []
    matches_by_item: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_ids: set[str] = set()
    for index, row in enumerate(exam):
        item_id = _item_id(row, "exam", index)
        if item_id in seen_ids:
            raise ProductInputError(f"exam has duplicate item_id {item_id}")
        seen_ids.add(item_id)
        exam_rows.append((item_id, row))
        for token, exam_reason in _identity_tokens(row).items():
            for match in exposure_index.get(token, ()):  # exact identity only
                matches_by_item[item_id].append({
                    "reason": "row_identity" if exam_reason in {"item_id", "id", "exposure_key"} else exam_reason,
                    "source": match["source"],
                    "tier": "exact",
                })

    # Graph-DSL behavioral equivalence is stronger than text similarity: it
    # compares the complete truth vector over a declared finite oracle corpus.
    # It can therefore quarantine, while still reporting the finite-horizon
    # false-merge boundary explicitly.
    dsl_exam = [(item_id, row["expression"]) for item_id, row in exam_rows if isinstance(row.get("expression"), str)]
    dsl_exposure = [
        (str(row.get("source") or row.get("path") or f"exposure row {index + 1}"), row["expression"])
        for index, row in enumerate(exposure)
        if isinstance(row.get("expression"), str)
    ]
    fingerprint_rows = 0
    fingerprint_warnings: list[str] = []
    if dsl_exam and dsl_exposure and bool(payload.get("enable_behavioral_fingerprint", True)):
        from bcv.leakage import behavioral_fingerprint

        observations = _dsl_observations(fingerprint_max_n)
        fingerprint_rows = len(observations)
        exposure_fingerprints: dict[str, list[str]] = defaultdict(list)
        for source, expression in dsl_exposure:
            try:
                exposure_fingerprints[behavioral_fingerprint(expression, observations)].append(source)
            except (SyntaxError, ValueError, TypeError, KeyError) as error:
                fingerprint_warnings.append(f"skipped unparseable exposure expression: {type(error).__name__}")
        for item_id, expression in dsl_exam:
            try:
                digest = behavioral_fingerprint(expression, observations)
            except (SyntaxError, ValueError, TypeError, KeyError) as error:
                fingerprint_warnings.append(f"skipped unparseable exam expression {item_id}: {type(error).__name__}")
                continue
            for source in exposure_fingerprints.get(digest, ()):
                matches_by_item[item_id].append({
                    "reason": "behavioral_fingerprint",
                    "source": source,
                    "tier": "behavioral",
                })

    # Text similarity is intentionally a human-review queue, never an automatic
    # quarantine. It uses no embeddings or external service and never reflects
    # the matched training text back to the caller.
    exposure_text: list[dict[str, Any]] = []
    inverted: dict[str, set[int]] = defaultdict(set)
    for index, row in enumerate(exposure):
        content = _content_value(row)
        if content is None or content[0] == "expression":
            continue
        field, value = content
        terms = _similarity_terms(value)
        record = {
            "source": str(row.get("source") or row.get("path") or f"exposure row {index + 1}"),
            "field": field,
            "terms": terms,
            "exact_digest": hashlib.sha256(value.encode("utf-8")).hexdigest(),
        }
        exposure_text.append(record)
        for term in terms:
            inverted[term].add(len(exposure_text) - 1)

    review_queue: list[dict[str, Any]] = []
    if bool(payload.get("enable_text_similarity", True)):
        for item_id, row in exam_rows:
            content = _content_value(row)
            if content is None or content[0] == "expression":
                continue
            field, value = content
            terms = _similarity_terms(value)
            candidates: set[int] = set()
            for term in terms:
                candidates.update(inverted.get(term, ()))
            scored = []
            digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
            for candidate_index in candidates:
                candidate = exposure_text[candidate_index]
                if candidate["exact_digest"] == digest:
                    continue
                score, shared = _text_similarity(terms, candidate["terms"])
                if score >= similarity_threshold:
                    scored.append({
                        "item_id": item_id,
                        "score": score,
                        "shared_terms": shared,
                        "exam_field": field,
                        "exposure_field": candidate["field"],
                        "source": candidate["source"],
                        "action": "human_review",
                    })
            review_queue.extend(sorted(scored, key=lambda row: (-row["score"], row["source"]))[:3])

    quarantined = []
    clean_exam = []
    tier_items: dict[str, set[str]] = defaultdict(set)
    for item_id, row in exam_rows:
        raw_matches = matches_by_item.get(item_id, [])
        unique = sorted({(m["tier"], m["reason"], m["source"]) for m in raw_matches})
        if unique:
            for tier, _, _ in unique:
                tier_items[tier].add(item_id)
            quarantined.append({
                "item_id": item_id,
                "matches": [
                    {"tier": tier, "reason": reason, "source": source}
                    for tier, reason, source in unique
                ],
            })
        else:
            clean_exam.append(row)

    summary = {
        "schema_version": 2,
        "exact_identity_only": not bool(tier_items.get("behavioral")),
        "exam_items": len(exam),
        "exposure_rows": len(exposure),
        "quarantined_items": len(quarantined),
        "clean_items": len(clean_exam),
        "exposure_rate": round(len(quarantined) / len(exam), 6) if exam else 0.0,
        "quarantined": quarantined,
        "clean_exam": clean_exam,
        "tier_counts": {tier: len(items) for tier, items in sorted(tier_items.items())},
        "review_queue": sorted(review_queue, key=lambda row: (-row["score"], row["item_id"])),
        "analysis_tiers": {
            "exact_identity": {"enabled": True, "action": "quarantine", "items": len(tier_items.get("exact", ()))},
            "behavioral_fingerprint": {
                "enabled": bool(dsl_exam and dsl_exposure),
                "action": "quarantine",
                "items": len(tier_items.get("behavioral", ())),
                "corpus_max_n": fingerprint_max_n,
                "observations": fingerprint_rows,
                "boundary": "Equivalent on this finite graph corpus; expressions may diverge beyond the declared horizon.",
            },
            "text_similarity": {
                "enabled": bool(payload.get("enable_text_similarity", True)),
                "action": "human_review_only",
                "threshold": similarity_threshold,
                "candidates": len(review_queue),
                "boundary": "Token-overlap triage, not semantic equivalence and never an automatic quarantine.",
            },
        },
        "analysis_warnings": sorted(set(fingerprint_warnings)),
        "input_sha256": _sha256({
            "exam": exam,
            "exposure": exposure,
            "fingerprint_max_n": fingerprint_max_n,
            "similarity_threshold": similarity_threshold,
        }),
        "claim_boundary": "Exact identity and finite-corpus DSL behavior can quarantine; text similarity is not semantic proof and only creates a human-review queue.",
    }
    summary["receipt_sha256"] = _sha256({key: value for key, value in summary.items() if key != "receipt_sha256"})
    return summary


def _policy(raw: Any) -> dict[str, Any]:
    raw = raw if isinstance(raw, dict) else {}
    policy = {
        "min_gains": int(raw.get("min_gains", 1)),
        "max_regressions": int(raw.get("max_regressions", 0)),
        "confidence_alpha": float(raw.get("confidence_alpha", raw.get("alpha", 0.05))),
        "require_retained_probe": bool(raw.get("require_retained_probe", False)),
    }
    if policy["min_gains"] < 0 or policy["max_regressions"] < 0:
        raise ProductInputError("min_gains and max_regressions must be non-negative")
    if not 0 < policy["confidence_alpha"] <= 1:
        raise ProductInputError("confidence_alpha must be in (0, 1]")
    return policy


def _retained_probe(raw: Any) -> dict[str, Any] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ProductInputError("retained_probe must be an object")
    base = raw.get("base_verified")
    candidate = raw.get("candidate_verified", raw.get("adapter_verified"))
    total = raw.get("items", raw.get("eval_examples"))
    if not all(isinstance(value, int) for value in (base, candidate, total)):
        raise ProductInputError("retained_probe needs integer base_verified, candidate_verified, and items")
    return {
        "base_verified": base,
        "candidate_verified": candidate,
        "items": total,
        "delta": candidate - base,
        "no_regression": candidate >= base,
    }


def _score_summary(results: dict[str, bool]) -> dict[str, int | float]:
    passed = sum(results.values())
    total = len(results)
    return {
        "passed": passed,
        "failed": total - passed,
        "total": total,
        "rate": round(passed / total, 6) if total else 0.0,
    }


def _additional_clean_gains_needed(
    gains: int,
    regressions: int,
    policy: dict[str, Any],
) -> int | None:
    if regressions > policy["max_regressions"]:
        return None
    for additional in range(0, 1_001):
        if gains + additional < policy["min_gains"]:
            continue
        if exact_mcnemar_p_value(gains + additional, regressions) <= policy["confidence_alpha"]:
            return additional
    return None


def gate_results(payload: dict[str, Any]) -> dict[str, Any]:
    baseline = parse_results(payload.get("baseline", {}), "baseline")
    candidate = parse_results(payload.get("candidate", {}), "candidate")
    if not baseline:
        raise ProductInputError("baseline and candidate results cannot be empty")
    if set(baseline) != set(candidate):
        raise ProductInputError("baseline and candidate must cover the identical item cohort")
    raw_domains = payload.get("domains", {})
    domains = {str(key): str(value) for key, value in raw_domains.items()} if isinstance(raw_domains, dict) else {}
    policy = _policy(payload.get("policy"))
    retained = _retained_probe(payload.get("retained_probe"))

    gains = regressions = ties = 0
    by_domain: dict[str, dict[str, int]] = {}
    item_rows = []
    for item_id in sorted(baseline):
        before, after = baseline[item_id], candidate[item_id]
        outcome = "gain" if after and not before else "regression" if before and not after else "tie"
        gains += outcome == "gain"
        regressions += outcome == "regression"
        ties += outcome == "tie"
        domain = domains.get(item_id, "unlabeled")
        bucket = by_domain.setdefault(domain, {
            "items": 0,
            "baseline_passes": 0,
            "candidate_passes": 0,
            "gains": 0,
            "regressions": 0,
            "ties": 0,
        })
        bucket["items"] += 1
        bucket["baseline_passes"] += int(before)
        bucket["candidate_passes"] += int(after)
        bucket[f"{outcome}s"] += 1
        item_rows.append({
            "item_id": item_id,
            "domain": domain,
            "baseline": before,
            "candidate": after,
            "outcome": outcome,
        })

    p_value = exact_mcnemar_p_value(gains, regressions)
    reasons: list[str] = []
    if regressions > policy["max_regressions"]:
        verdict = "BLOCK"
        reasons.append(f"{regressions} regression(s) exceed the policy limit of {policy['max_regressions']}")
    elif policy["require_retained_probe"] and (retained is None or not retained["no_regression"]):
        verdict = "BLOCK"
        reasons.append("retained probe is absent or regressed")
    elif gains < policy["min_gains"]:
        verdict = "HOLD"
        reasons.append(f"{gains} gain(s) are below the policy minimum of {policy['min_gains']}")
    elif p_value > policy["confidence_alpha"]:
        verdict = "HOLD"
        reasons.append(
            f"paired exact McNemar p={p_value:.6g} exceeds alpha={policy['confidence_alpha']:.6g}; collect more discrimination"
        )
    else:
        verdict = "PASS"
        reasons.append("regression policy held and the paired evidence passed the confidence threshold")

    for bucket in by_domain.values():
        bucket["delta"] = bucket["candidate_passes"] - bucket["baseline_passes"]

    additional_clean_gains = _additional_clean_gains_needed(gains, regressions, policy)
    if verdict == "PASS":
        next_action = "Promote the candidate and preserve this receipt with the exact item-set commitment."
    elif regressions > policy["max_regressions"]:
        regressed_items = [row["item_id"] for row in item_rows if row["outcome"] == "regression"]
        next_action = f"Diagnose or explicitly waive the regression set before rerunning: {', '.join(regressed_items)}."
    elif policy["require_retained_probe"] and (retained is None or not retained["no_regression"]):
        next_action = "Restore the retained capability probe before collecting more promotion evidence."
    elif additional_clean_gains is not None:
        next_action = f"Add {additional_clean_gains} clean paired gain(s) with zero new regressions to reach the current policy threshold."
    else:
        next_action = "Collect a larger paired cohort or revise the policy explicitly; this receipt cannot identify a finite clean-win path."

    decision_path = [
        {
            "check": "regression_budget",
            "state": "pass" if regressions <= policy["max_regressions"] else "fail",
            "observed": regressions,
            "requirement": f"<= {policy['max_regressions']}",
        },
        {
            "check": "retained_probe",
            "state": "not_required" if not policy["require_retained_probe"] else "pass" if retained and retained["no_regression"] else "fail",
            "observed": retained["delta"] if retained else None,
            "requirement": "no regression" if policy["require_retained_probe"] else "not required",
        },
        {
            "check": "minimum_gains",
            "state": "pass" if gains >= policy["min_gains"] else "hold",
            "observed": gains,
            "requirement": f">= {policy['min_gains']}",
        },
        {
            "check": "paired_exact_test",
            "state": "pass" if p_value <= policy["confidence_alpha"] else "hold",
            "observed": p_value,
            "requirement": f"<= {policy['confidence_alpha']}",
        },
    ]

    baseline_score = _score_summary(baseline)
    candidate_score = _score_summary(candidate)
    stable_input = {
        "baseline": baseline,
        "candidate": candidate,
        "domains": domains,
        "policy": policy,
        "retained_probe": retained,
    }
    report = {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "reasons": reasons,
        "systems": {
            "baseline": str(payload.get("baseline_name", "baseline")),
            "candidate": str(payload.get("candidate_name", "candidate")),
        },
        "policy": policy,
        "scorecard": {
            "baseline": baseline_score,
            "candidate": candidate_score,
            "pass_delta": candidate_score["passed"] - baseline_score["passed"],
            "rate_delta": round(candidate_score["rate"] - baseline_score["rate"], 6),
        },
        "decision_path": decision_path,
        "next_action": next_action,
        "paired_evidence": {
            "items": len(item_rows),
            "item_set_sha256": _sha256(sorted(baseline)),
            "gains": gains,
            "regressions": regressions,
            "ties": ties,
            "exact_mcnemar_two_sided_p": p_value,
            "resolution": power_statement(gains, regressions, policy["confidence_alpha"]),
            "additional_clean_gains_needed": additional_clean_gains,
            "by_domain": dict(sorted(by_domain.items())),
            "items_detail": item_rows,
        },
        "retained_probe": retained,
        "input_sha256": _sha256(stable_input),
    }
    report["receipt_sha256"] = _sha256({key: value for key, value in report.items() if key not in {"generated_at", "receipt_sha256"}})
    return report


def inspect_promotion(payload: dict[str, Any]) -> dict[str, Any]:
    audit = audit_leakage(payload)
    exam_rows = _records(payload.get("exam", []), "exam")
    exam_ids = {_item_id(row, "exam", index) for index, row in enumerate(exam_rows)}
    clean_ids = {_item_id(row, "clean exam", index) for index, row in enumerate(audit["clean_exam"])}
    quarantined_ids = exam_ids - clean_ids
    baseline = parse_results(payload.get("baseline", {}), "baseline")
    candidate = parse_results(payload.get("candidate", {}), "candidate")
    baseline_clean = {item: result for item, result in baseline.items() if item in clean_ids}
    candidate_clean = {item: result for item, result in candidate.items() if item in clean_ids}
    missing_baseline = sorted(clean_ids - set(baseline_clean))
    missing_candidate = sorted(clean_ids - set(candidate_clean))
    unknown_results = sorted((set(baseline) | set(candidate)) - exam_ids)
    complete = bool(clean_ids) and not missing_baseline and not missing_candidate and set(baseline_clean) == set(candidate_clean)
    raw_ids = exam_ids & set(baseline) & set(candidate)
    baseline_raw = {item: baseline[item] for item in raw_ids}
    candidate_raw = {item: candidate[item] for item in raw_ids}

    cohort = {
        "complete": complete,
        "eligible_items": len(clean_ids),
        "quarantined_result_items_removed": sorted(quarantined_ids & (set(baseline) | set(candidate))),
        "missing_from_baseline": missing_baseline,
        "missing_from_candidate": missing_candidate,
        "ignored_unknown_result_items": unknown_results,
    }
    if complete:
        domains = {
            _item_id(row, "exam", index): str(row.get("domain", "unlabeled"))
            for index, row in enumerate(exam_rows)
            if _item_id(row, "exam", index) in clean_ids
        }
        gate = gate_results({
            "baseline": baseline_clean,
            "candidate": candidate_clean,
            "domains": domains,
            "policy": payload.get("policy", {}),
            "retained_probe": payload.get("retained_probe"),
            "baseline_name": payload.get("baseline_name", "baseline"),
            "candidate_name": payload.get("candidate_name", "candidate"),
        })
    else:
        gate = {
            "schema_version": 2,
            "verdict": "HOLD",
            "reasons": ["no decision: both systems must cover the complete post-quarantine cohort"],
            "paired_evidence": {"items": 0, "gains": 0, "regressions": 0, "ties": 0},
            "next_action": "Grade both systems on every clean item; partial intersections are never treated as evidence.",
            "decision_path": [{
                "check": "complete_clean_cohort",
                "state": "hold",
                "observed": len(set(baseline_clean) & set(candidate_clean)),
                "requirement": len(clean_ids),
            }],
        }
    raw_baseline_score = _score_summary(baseline_raw)
    raw_candidate_score = _score_summary(candidate_raw)
    clean_baseline_score = _score_summary(baseline_clean)
    clean_candidate_score = _score_summary(candidate_clean)
    quarantined_with_results = quarantined_ids & raw_ids
    result = {
        "schema_version": 2,
        "audit": audit,
        "cohort": cohort,
        "gate": gate,
        "pipeline": [
            {"stage": "ingest", "state": "pass", "detail": f"{len(exam_rows)} exam items; {len(raw_ids)} paired results"},
            {"stage": "exposure_audit", "state": "pass" if not audit["quarantined_items"] else "quarantine", "detail": f"{audit['quarantined_items']} removed; {len(audit['review_queue'])} queued for review"},
            {"stage": "cohort_proof", "state": "pass" if complete else "hold", "detail": f"{len(clean_ids)} eligible items"},
            {"stage": "promotion_gate", "state": gate["verdict"].lower(), "detail": gate["reasons"][0]},
        ],
        "scorecard": {
            "raw": {
                "baseline": raw_baseline_score,
                "candidate": raw_candidate_score,
                "pass_delta": raw_candidate_score["passed"] - raw_baseline_score["passed"],
            },
            "clean": {
                "baseline": clean_baseline_score,
                "candidate": clean_candidate_score,
                "pass_delta": clean_candidate_score["passed"] - clean_baseline_score["passed"],
            },
            "quarantine_impact": {
                "paired_items_removed": len(quarantined_with_results),
                "baseline_passes_removed": sum(baseline[item] for item in quarantined_with_results),
                "candidate_passes_removed": sum(candidate[item] for item in quarantined_with_results),
            },
        },
    }
    # Hash only the stable sub-receipts and decision inputs.  gate_results carries
    # a human-readable generated_at timestamp, but repeating the same inspection
    # must still produce the same evidence identity.
    result["receipt_sha256"] = _sha256({
        "schema_version": result["schema_version"],
        "audit_receipt_sha256": audit["receipt_sha256"],
        "cohort": cohort,
        "gate_receipt_sha256": gate.get("receipt_sha256"),
        "gate_verdict": gate["verdict"],
        "gate_reasons": gate["reasons"],
        "paired_evidence": gate["paired_evidence"] if "receipt_sha256" not in gate else None,
    })
    return result


def bank_health(payload: dict[str, Any]) -> dict[str, Any]:
    definitions = _records(payload.get("items", []), "items") if payload.get("items") is not None else []
    domains: dict[str, str] = {}
    for index, row in enumerate(definitions):
        domains[_item_id(row, "items", index)] = str(row.get("domain", "unlabeled"))
    history = _records(payload.get("history", payload.get("results", [])), "history")
    if not history:
        raise ProductInputError("history needs at least one {item_id, system, passed} row")
    stats: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    system_totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for index, row in enumerate(history):
        item_id = _item_id(row, "history", index)
        system = row.get("system")
        if not isinstance(system, str) or not system.strip():
            raise ProductInputError(f"history row {index + 1} needs a system")
        passed = _as_bool(row.get("passed", row.get("outcome")), f"history row {index + 1} outcome")
        stats[item_id][system.strip()][0 if passed else 1] += 1
        system_totals[system.strip()][0] += int(passed)
        system_totals[system.strip()][1] += 1
        domains.setdefault(item_id, str(row.get("domain", "unlabeled")))

    rows = []
    for item_id in sorted(set(domains) | set(stats)):
        systems = stats.get(item_id, {})
        rates = {
            system: round(counts[0] / sum(counts), 6)
            for system, counts in sorted(systems.items())
            if sum(counts)
        }
        observations = sum(sum(counts) for counts in systems.values())
        discrimination = round(max(rates.values()) - min(rates.values()), 6) if len(rates) >= 2 else 0.0
        max_flip_rate = max(
            (min(counts) / sum(counts) for counts in systems.values() if sum(counts) >= 2),
            default=0.0,
        )
        if len(rates) < 2:
            classification = "under_observed"
        elif all(rate == 1.0 for rate in rates.values()):
            classification = "saturated"
        elif all(rate == 0.0 for rate in rates.values()):
            classification = "too_hard"
        elif max_flip_rate > 0:
            classification = "flaky"
        elif discrimination > 0:
            classification = "discriminating"
        else:
            classification = "flat"
        action = {
            "under_observed": "collect_more_grades",
            "saturated": "retire_or_mill_harder",
            "too_hard": "mill_easier_variants",
            "flaky": "review_or_retire",
            "discriminating": "retain",
            "flat": "refresh_or_retire",
        }[classification]
        mean_pass_rate = sum(rates.values()) / len(rates) if rates else 0.0
        rows.append({
            "item_id": item_id,
            "domain": domains.get(item_id, "unlabeled"),
            "systems": len(rates),
            "observations": observations,
            "pass_rates": rates,
            "discrimination": discrimination,
            "max_within_system_flip_rate": round(max_flip_rate, 6),
            "mean_pass_rate": round(mean_pass_rate, 6),
            "difficulty": round(1.0 - mean_pass_rate, 6),
            "reliability": round(1.0 - max_flip_rate, 6),
            "utility": round(discrimination * (1.0 - max_flip_rate), 6),
            "classification": classification,
            "recommended_action": action,
        })

    by_class: dict[str, int] = defaultdict(int)
    by_domain: dict[str, dict[str, int]] = defaultdict(lambda: {"items": 0, "discriminating": 0})
    for row in rows:
        by_class[row["classification"]] += 1
        by_domain[row["domain"]]["items"] += 1
        by_domain[row["domain"]]["discriminating"] += row["classification"] == "discriminating"
    domain_detail = {}
    for domain, counts in sorted(by_domain.items()):
        ratio = counts["discriminating"] / counts["items"] if counts["items"] else 0.0
        domain_detail[domain] = {
            **counts,
            "discriminator_share": round(ratio, 6),
            "state": "covered" if counts["discriminating"] else "frontier_gap",
        }
    item_count = len(rows)
    domain_count = len(domain_detail)
    readiness_components = {
        "discriminator_share": round(by_class.get("discriminating", 0) / item_count, 6) if item_count else 0.0,
        "stable_share": round((item_count - by_class.get("flaky", 0)) / item_count, 6) if item_count else 0.0,
        "observed_share": round(sum(row["systems"] >= 2 for row in rows) / item_count, 6) if item_count else 0.0,
        "domain_coverage": round(sum(value["state"] == "covered" for value in domain_detail.values()) / domain_count, 6) if domain_count else 0.0,
    }
    readiness_index = round(100 * sum(readiness_components.values()) / len(readiness_components), 1)
    action_priority = {
        "review_or_retire": 0,
        "collect_more_grades": 1,
        "mill_easier_variants": 2,
        "retire_or_mill_harder": 3,
        "refresh_or_retire": 4,
        "retain": 5,
    }
    action_queue = sorted(
        [
            {
                "item_id": row["item_id"],
                "domain": row["domain"],
                "classification": row["classification"],
                "action": row["recommended_action"],
                "utility": row["utility"],
            }
            for row in rows
            if row["recommended_action"] != "retain"
        ],
        key=lambda row: (action_priority[row["action"]], row["utility"], row["item_id"]),
    )
    system_ladder = sorted(
        [
            {
                "system": system,
                "passed": totals[0],
                "observations": totals[1],
                "pass_rate": round(totals[0] / totals[1], 6) if totals[1] else 0.0,
            }
            for system, totals in system_totals.items()
        ],
        key=lambda row: (row["pass_rate"], row["system"]),
    )
    result = {
        "schema_version": 2,
        "items": len(rows),
        "systems": sorted({system for item in stats.values() for system in item}),
        "system_ladder": system_ladder,
        "classification_counts": dict(sorted(by_class.items())),
        "by_domain": domain_detail,
        "frontier_gaps": sorted(domain for domain, counts in by_domain.items() if counts["discriminating"] == 0),
        "retirement_candidates": [row["item_id"] for row in rows if row["classification"] == "saturated"],
        "readiness": {
            "index": readiness_index,
            "components": readiness_components,
            "boundary": "Transparent four-component operability heuristic, not an IRT or psychometric validity claim.",
        },
        "action_queue": action_queue,
        "items_detail": rows,
        "claim_boundary": "Descriptive history diagnostics; sparse grades and system selection still bound every conclusion.",
        "input_sha256": _sha256({"items": definitions, "history": history}),
    }
    result["receipt_sha256"] = _sha256(result)
    return result


def safe_patch(payload: dict[str, Any]) -> dict[str, Any]:
    document = payload.get("document")
    if not isinstance(document, str) or not document:
        raise ProductInputError("document must be non-empty Markdown text")
    if len(document) > MAX_MARKDOWN_CHARS:
        raise ProductInputError(f"document exceeds the {MAX_MARKDOWN_CHARS}-character public-demo limit")
    raw_operations = payload.get("operations")
    if not isinstance(raw_operations, list) or not raw_operations or len(raw_operations) > 50:
        raise ProductInputError("operations must contain 1-50 patch operations")
    operations = []
    for index, raw in enumerate(raw_operations):
        if not isinstance(raw, dict):
            raise ProductInputError(f"operation {index + 1} must be an object")
        heading, find, replace = raw.get("target_heading"), raw.get("find"), raw.get("replace")
        if not all(isinstance(value, str) for value in (heading, find, replace)) or not heading or not find:
            raise ProductInputError(f"operation {index + 1} needs target_heading, non-empty find, and replace")
        allowed = raw.get("allow_token_changes", [])
        if not isinstance(allowed, list) or not all(isinstance(value, str) for value in allowed):
            raise ProductInputError(f"operation {index + 1} allow_token_changes must be a string array")
        operations.append(PatchOperation(heading, find, replace, tuple(allowed)))
    patch = MarkdownPatch(tuple(operations), str(payload.get("reason", "")))
    try:
        updated = apply_markdown_patch(document, patch)
    except PatchError as error:
        raise ProductInputError(str(error)) from error
    diff = "".join(difflib.unified_diff(
        document.splitlines(keepends=True),
        updated.splitlines(keepends=True),
        fromfile="original.md",
        tofile="patched.md",
    ))
    original_sections = {section.heading: section.text for section in parse_sections(document)}
    updated_sections = {section.heading: section.text for section in parse_sections(updated)}
    changed_headings = sorted({operation.target_heading for operation in operations})
    section_receipts = []
    for heading in changed_headings:
        before = original_sections[heading]
        after = updated_sections[heading]
        before_tokens = protected_tokens(before)
        after_tokens = protected_tokens(after)
        section_receipts.append({
            "heading": heading,
            "original_sha256": hashlib.sha256(before.encode("utf-8")).hexdigest(),
            "updated_sha256": hashlib.sha256(after.encode("utf-8")).hexdigest(),
            "character_delta": len(after) - len(before),
            "protected_tokens_before": len(before_tokens),
            "protected_tokens_after": len(after_tokens),
            "protected_tokens_removed": sorted(before_tokens - after_tokens),
        })
    diff_lines = diff.splitlines()
    added_lines = sum(line.startswith("+") and not line.startswith("+++") for line in diff_lines)
    removed_lines = sum(line.startswith("-") and not line.startswith("---") for line in diff_lines)
    untouched = sorted(set(original_sections) - set(changed_headings))
    result = {
        "schema_version": 2,
        "accepted": True,
        "updated_document": updated,
        "unified_diff": diff,
        "changed_sections": changed_headings,
        "untouched_sections": untouched,
        "section_receipts": section_receipts,
        "diff_stats": {
            "operations": len(operations),
            "added_lines": added_lines,
            "removed_lines": removed_lines,
            "character_delta": len(updated) - len(document),
            "sections_total": len(original_sections),
            "sections_changed": len(changed_headings),
            "sections_untouched": len(untouched),
        },
        "checks": {
            "patch_applied_cleanly": True,
            "untargeted_sections_byte_identical": True,
            "protected_tokens_not_removed_unless_allowed": True,
        },
        "original_sha256": hashlib.sha256(document.encode("utf-8")).hexdigest(),
        "updated_sha256": hashlib.sha256(updated.encode("utf-8")).hexdigest(),
        "claim_boundary": "Deterministic patch application and conservation checks; no model generated this edit.",
    }
    result["receipt_sha256"] = _sha256({key: value for key, value in result.items() if key != "receipt_sha256"})
    return result


def _entities(text: str) -> tuple[str, ...]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text.lower())
    return tuple(dict.fromkeys(token for token in tokens if token not in STOPWORDS))


def memory_relevance(payload: dict[str, Any]) -> dict[str, Any]:
    objective = payload.get("objective")
    if not isinstance(objective, str) or not objective.strip():
        raise ProductInputError("objective must be non-empty text")
    rows = _records(payload.get("memories", []), "memories", limit=1_000)
    if not rows:
        raise ProductInputError("memories cannot be empty")
    explicit = payload.get("objective_entities")
    if explicit is not None and (not isinstance(explicit, list) or not all(isinstance(value, str) for value in explicit)):
        raise ProductInputError("objective_entities must be a string array")
    objective_entities = tuple(dict.fromkeys(value.lower() for value in explicit)) if explicit else _entities(objective)
    if not objective_entities:
        raise ProductInputError("objective has no usable entities; supply objective_entities explicitly")
    context_raw = payload.get("context_entities", [])
    if not isinstance(context_raw, list) or not all(isinstance(value, str) for value in context_raw):
        raise ProductInputError("context_entities must be a string array")
    context = set(objective_entities) | {value.lower() for value in context_raw}
    current_step = int(payload.get("current_step", max(1, len(rows))))
    memories: list[StoredMemory] = []
    for index, row in enumerate(rows, 1):
        content = row.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ProductInputError(f"memories row {index} needs content")
        entity_values = row.get("entities")
        if entity_values is not None and (not isinstance(entity_values, list) or not all(isinstance(value, str) for value in entity_values)):
            raise ProductInputError(f"memories row {index} entities must be a string array")
        entities = tuple(value.lower() for value in entity_values) if entity_values else _entities(content)
        confidence = float(row.get("confidence", 0.8))
        if not 0 <= confidence <= 1:
            raise ProductInputError(f"memories row {index} confidence must be in [0, 1]")
        age = max(0, int(row.get("age", 0)))
        memories.append(StoredMemory(
            id=index,
            content=content,
            kind=str(row.get("kind", "episodic")),
            source=str(row.get("source", "uploaded")),
            confidence=confidence,
            entities=entities,
            created_step=max(0, current_step - age),
            last_used_step=max(0, current_step - age),
            use_count=int(row.get("use_count", 0)),
            parent_ids=(),
        ))
    counts: dict[str, int] = defaultdict(int)
    total = 0
    for memory in memories:
        for entity in memory.entities:
            counts[entity] += 1
            total += 1
    rarity = {entity: -math.log(count / total) for entity, count in counts.items()} if total else {}
    probe = Probe(question=objective, kind=str(payload.get("question_kind", "generic")), subject=objective_entities, answer="", oracle_text="")
    scored = []
    for memory in memories:
        scored.append({
            "id": memory.id,
            "content": memory.content,
            "entities": list(memory.entities),
            "token_cost": len(memory.content.split()),
            "salience": salience_prior(memory, context, rarity),
            "relevance": relevance_score(memory, probe, context, rarity),
        })
    salience_order = sorted(scored, key=lambda row: (-row["salience"], row["id"]))
    relevance_order = sorted(scored, key=lambda row: (-row["relevance"], row["id"]))
    salience_rank = {row["id"]: index + 1 for index, row in enumerate(salience_order)}
    relevance_rank = {row["id"]: index + 1 for index, row in enumerate(relevance_order)}
    top = max(1, math.ceil(len(scored) / 3))
    bottom_start = max(top + 1, math.floor(2 * len(scored) / 3) + 1)
    for row in scored:
        row["salience_rank"] = salience_rank[row["id"]]
        row["relevance_rank"] = relevance_rank[row["id"]]
        if row["salience_rank"] <= top and (row["relevance_rank"] >= bottom_start or row["relevance"] < 0):
            label = "shiny_trap"
        elif row["relevance_rank"] <= top and row["salience_rank"] >= bottom_start:
            label = "boring_but_decisive"
        else:
            label = "aligned_or_middle"
        row["classification"] = label
        row["salience"] = round(row["salience"], 6)
        row["relevance"] = round(row["relevance"], 6)
    token_budget = int(payload.get("token_budget", 90))
    if token_budget < 1 or token_budget > 10_000:
        raise ProductInputError("token_budget must be between 1 and 10000")
    def select(order: list[dict[str, Any]]) -> tuple[list[int], int]:
        selected_ids = []
        used_tokens = 0
        for item in order:
            cost = item["token_cost"]
            if used_tokens + cost <= token_budget:
                selected_ids.append(item["id"])
                used_tokens += cost
        return selected_ids, used_tokens

    selected, used = select(relevance_order)
    salience_selected, salience_used = select(salience_order)
    selected_set = set(selected)
    salience_set = set(salience_selected)
    for row in scored:
        row["selected_by_relevance"] = row["id"] in selected_set
        row["selected_by_salience"] = row["id"] in salience_set
    n = len(scored)
    rank_correlation = (
        round(1 - 6 * sum((row["salience_rank"] - row["relevance_rank"]) ** 2 for row in scored) / (n * (n * n - 1)), 6)
        if n > 1 else 1.0
    )
    salience_waste = sum(row["token_cost"] for row in scored if row["id"] in salience_set and row["relevance"] < 0)
    relevance_waste = sum(row["token_cost"] for row in scored if row["id"] in selected_set and row["relevance"] < 0)
    result = {
        "schema_version": 2,
        "objective": objective,
        "objective_entities": list(objective_entities),
        "token_budget": token_budget,
        "selected_by_relevance": selected,
        "selected_tokens": used,
        "selected_by_salience": salience_selected,
        "salience_selected_tokens": salience_used,
        "rank_correlation": rank_correlation,
        "budget_comparison": {
            "relevance": {
                "selected": selected,
                "tokens": used,
                "negative_relevance_tokens": relevance_waste,
            },
            "salience": {
                "selected": salience_selected,
                "tokens": salience_used,
                "negative_relevance_tokens": salience_waste,
            },
            "attention_waste_avoided_tokens": salience_waste - relevance_waste,
        },
        "shiny_traps": [row["id"] for row in scored if row["classification"] == "shiny_trap"],
        "boring_but_decisive": [row["id"] for row in scored if row["classification"] == "boring_but_decisive"],
        "ranking": sorted(scored, key=lambda row: row["relevance_rank"]),
        "claim_boundary": "Query-conditioned mechanism playground; the published 0.988 result is on the exact TinySeasons decision procedure, not arbitrary prose.",
    }
    result["receipt_sha256"] = _sha256(result)
    return result


def replay_trace(payload: dict[str, Any]) -> dict[str, Any]:
    raw_events = payload.get("events")
    if raw_events is None and isinstance(payload.get("result"), dict):
        raw_events = payload["result"].get("events")
    events = _records(raw_events or [], "events", limit=MAX_EVENT_RECORDS)
    checkpoints: dict[str, dict[str, int]] = {}
    controls: dict[str, int] = defaultdict(int)
    notes = [str(note) for note in payload.get("notes", [])] if isinstance(payload.get("notes", []), list) else []
    timeline = []
    branch = 0
    branches: dict[int, dict[str, Any]] = {
        0: {"id": 0, "parent": None, "started_at": None, "rewind_target": None, "events": 0, "verdicts": []}
    }
    rewinds = []
    verifier_accepts = verifier_rejects = 0
    for index, event in enumerate(events):
        step = int(event.get("step", index + 1))
        kind = str(event.get("kind", "event"))
        detail = str(event.get("detail", ""))
        control = None
        if kind == "control" and detail:
            control = detail.split(maxsplit=1)[0].upper()
            if control in {"SAVE", "LOAD", "CHECK", "SKETCH", "ANSWER"}:
                controls[control] += 1
                argument = detail[len(control):].strip()
                if control == "SAVE" and argument:
                    checkpoints[argument.split()[0]] = {"step": step, "branch": branch}
                elif control == "LOAD":
                    target, _, note = argument.partition("::")
                    parent = branch
                    branch += 1
                    checkpoint = checkpoints.get(target.strip())
                    branches[branch] = {
                        "id": branch,
                        "parent": parent,
                        "started_at": step,
                        "rewind_target": target.strip(),
                        "rewind_target_step": checkpoint["step"] if checkpoint else None,
                        "events": 0,
                        "verdicts": [],
                    }
                    rewinds.append({
                        "step": step,
                        "from_branch": parent,
                        "to_branch": branch,
                        "target": target.strip(),
                        "target_step": checkpoint["step"] if checkpoint else None,
                        "target_branch": checkpoint["branch"] if checkpoint else None,
                    })
                    if note.strip():
                        notes.append(note.strip())
        normalized_detail = detail.upper()
        if kind == "verifier" and "ACCEPT" in normalized_detail:
            verifier_accepts += 1
            branches[branch]["verdicts"].append("ACCEPT")
        elif kind == "verifier" and "REJECT" in normalized_detail:
            verifier_rejects += 1
            branches[branch]["verdicts"].append("REJECT")
        branches[branch]["events"] += 1
        timeline.append({
            "step": step,
            "kind": kind,
            "detail": detail,
            "control": control,
            "branch": branch,
            "source": str(event.get("source", "native")),
        })
    final_branch = branch
    critical_path = []
    cursor: int | None = final_branch
    while cursor is not None:
        critical_path.append(cursor)
        cursor = branches[cursor]["parent"]
    critical_path.reverse()
    result = {
        "schema_version": 2,
        "events": len(events),
        "controls": dict(sorted(controls.items())),
        "checkpoints": [{"name": name, "step": value["step"]} for name, value in sorted(checkpoints.items())],
        "checkpoint_detail": [
            {"name": name, "step": value["step"], "branch": value["branch"]}
            for name, value in sorted(checkpoints.items())
        ],
        "rewinds": rewinds,
        "branches": [branches[index] for index in sorted(branches)],
        "critical_path": critical_path,
        "final_branch": final_branch,
        "verifier_summary": {
            "accepts": verifier_accepts,
            "rejects": verifier_rejects,
            "accept_rate": round(verifier_accepts / (verifier_accepts + verifier_rejects), 6)
            if verifier_accepts + verifier_rejects else 0.0,
        },
        "notes": list(dict.fromkeys(notes)),
        "external_interventions": sum(1 for row in timeline if row["source"] != "native"),
        "timeline": timeline,
        "claim_boundary": "Transcript reconstruction only; this console does not claim hidden chain-of-thought access.",
    }
    result["receipt_sha256"] = _sha256(result)
    return result


def _greedy_coloring_certificate(graph: Any) -> tuple[list[int], dict[int, int]]:
    adjacency = graph.adjacency()
    degrees = graph.degrees()
    order = sorted(range(graph.n), key=lambda node: (-degrees[node], node))
    assignment: dict[int, int] = {}
    for node in order:
        used = {assignment[neighbor] for neighbor in adjacency[node] if neighbor in assignment}
        color = 0
        while color in used:
            color += 1
        assignment[node] = color
    return order, assignment


def _exact_coloring_certificate(graph: Any, color_count: int) -> dict[int, int]:
    adjacency = graph.adjacency()
    degrees = graph.degrees()
    order = sorted(range(graph.n), key=lambda node: (-degrees[node], node))
    assignment: dict[int, int] = {}

    def search(position: int) -> bool:
        if position == len(order):
            return True
        node = order[position]
        used = {assignment[neighbor] for neighbor in adjacency[node] if neighbor in assignment}
        for color in range(color_count):
            if color in used:
                continue
            assignment[node] = color
            if search(position + 1):
                return True
            del assignment[node]
        return False

    if not search(0):
        raise ProductInputError("internal exact-coloring certificate construction failed")
    return assignment


def _proper_coloring(edges: list[list[int]] | tuple[tuple[int, int], ...], assignment: dict[int, int]) -> bool:
    return all(assignment[int(left)] != assignment[int(right)] for left, right in edges)


def hunt_counterexample(payload: dict[str, Any]) -> dict[str, Any]:
    from bcv.graph_adversary import attack_expression

    expression = payload.get("expression")
    if not isinstance(expression, str) or not expression.strip() or len(expression) > 500:
        raise ProductInputError("expression must contain 1-500 characters")
    raw_ns = payload.get("ns", [8, 9, 10, 11])
    if not isinstance(raw_ns, list) or not raw_ns or len(raw_ns) > 5 or not all(isinstance(value, int) for value in raw_ns):
        raise ProductInputError("ns must be an array of 1-5 integers")
    ns = tuple(raw_ns)
    if min(ns) < 4 or max(ns) > 12:
        raise ProductInputError("public counterexample search is bounded to 4 <= n <= 12")
    restarts = int(payload.get("restarts", 4))
    steps = int(payload.get("steps", 800))
    seed = int(payload.get("seed", 0))
    if not 1 <= restarts <= 6 or not 50 <= steps <= 1_500:
        raise ProductInputError("public search requires 1-6 restarts and 50-1500 steps")
    started = time.perf_counter()
    try:
        attack = attack_expression(
            expression.strip(), ns=ns, restarts=restarts, steps=steps, seed=seed, library_path=None
        )
    except (SyntaxError, ValueError, TypeError, KeyError) as error:
        raise ProductInputError(str(error)) from error
    find = asdict(attack.find) if attack.find is not None else None
    result = {
        "schema_version": 2,
        "expression": expression.strip(),
        "status": "FALSIFIED" if attack.falsified else "NOT_FALSIFIED_WITHIN_BUDGET",
        "falsified": attack.falsified,
        "find": find,
        "budget": {
            "ns": list(ns),
            "restarts": restarts,
            "steps_per_restart": steps,
            "seed": seed,
            "exact_checks": attack.exact_checks,
            "best_greedy_seen": attack.best_greedy_seen,
        },
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "claim_boundary": "A found witness is exact. Failure to find one within this bounded search is not a proof.",
    }
    if find is not None:
        from bcv.discovery import Graph, graph_features

        graph = Graph(n=int(find["n"]), edges=tuple(tuple(edge) for edge in find["edges"]))
        greedy_order, greedy_assignment = _greedy_coloring_certificate(graph)
        optimal_assignment = _exact_coloring_certificate(graph, int(find["chromatic_number"]))
        degrees = graph.degrees()
        result["witness"] = {
            "gap": int(find["greedy_colors"]) - int(find["chromatic_number"]),
            "features": graph_features(graph),
            "greedy_order": greedy_order,
            "nodes": [
                {
                    "id": node,
                    "degree": degrees[node],
                    "greedy_color": greedy_assignment[node],
                    "optimal_color": optimal_assignment[node],
                    "greedy_order": greedy_order.index(node) + 1,
                }
                for node in range(graph.n)
            ],
            "checks": {
                "greedy_coloring_proper": _proper_coloring(find["edges"], greedy_assignment),
                "optimal_coloring_proper": _proper_coloring(find["edges"], optimal_assignment),
                "strict_gap_verified": len(set(greedy_assignment.values())) > len(set(optimal_assignment.values())),
            },
        }
        result["certificate_sha256"] = _sha256(find)
    result["receipt_sha256"] = _sha256(result)
    return result


def _record_array_schema(
    description: str,
    *,
    min_items: int = 0,
    max_items: int = MAX_RECORDS,
    item_properties: dict[str, Any] | None = None,
    item_required: tuple[str, ...] = (),
    item_additional_properties: bool = True,
) -> dict[str, Any]:
    item_schema: dict[str, Any] = {
        "type": "object",
        "properties": item_properties or {},
        "additionalProperties": item_additional_properties,
    }
    if item_required:
        item_schema["required"] = list(item_required)
    return {
        "type": "array",
        "description": description,
        "items": item_schema,
        "minItems": min_items,
        "maxItems": max_items,
    }


def _string_array_schema(description: str, *, max_items: int = 100) -> dict[str, Any]:
    return {
        "type": "array",
        "description": description,
        "items": {"type": "string"},
        "maxItems": max_items,
    }


def _result_map_schema(label: str) -> dict[str, Any]:
    return {
        "type": "object",
        "description": f"item_id -> boolean pass/fail result for the {label} system.",
        "minProperties": 1,
        "maxProperties": MAX_RECORDS,
        "additionalProperties": {"type": "boolean"},
    }


def _policy_input_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "description": "Explicit promotion policy. Omitted fields use the documented defaults.",
        "properties": {
            "min_gains": {"type": "integer", "minimum": 0, "default": 1},
            "max_regressions": {"type": "integer", "minimum": 0, "default": 0},
            "confidence_alpha": {
                "type": "number",
                "exclusiveMinimum": 0,
                "maximum": 1,
                "default": 0.05,
            },
            "require_retained_probe": {"type": "boolean", "default": False},
        },
        "additionalProperties": False,
    }


def _retained_probe_input_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "description": "Optional retained-capability result checked alongside the paired cohort.",
        "properties": {
            "base_verified": {"type": "integer", "minimum": 0},
            "candidate_verified": {"type": "integer", "minimum": 0},
            "items": {"type": "integer", "minimum": 0},
        },
        "required": ["base_verified", "candidate_verified", "items"],
        "additionalProperties": False,
    }


def input_schemas() -> dict[str, dict[str, Any]]:
    """Canonical machine contracts for the eight public Tier 0 tools.

    The runtime accepts a few convenience encodings such as JSONL strings, but
    MCP and OpenAPI advertise one narrow, typed JSON form that every client can
    construct without following an out-of-band example link.
    """
    exam_schema = _record_array_schema(
        "Exam rows. Each row needs item_id (or id) plus prompt/content/input/task/question/expression.",
        min_items=1,
    )
    exposure_schema = _record_array_schema(
        "Declared exposure rows carrying identity/content fields and an optional source or path."
    )
    leakage_properties: dict[str, Any] = {
        "exam": exam_schema,
        "exposure": exposure_schema,
        "fingerprint_max_n": {"type": "integer", "minimum": 3, "maximum": 5, "default": 4},
        "similarity_threshold": {
            "type": "number",
            "minimum": 0.5,
            "maximum": 1,
            "default": 0.6,
        },
        "enable_text_similarity": {"type": "boolean", "default": True},
        "enable_behavioral_fingerprint": {"type": "boolean", "default": True},
    }
    gate_properties: dict[str, Any] = {
        "baseline": _result_map_schema("baseline"),
        "candidate": _result_map_schema("candidate"),
        "domains": {
            "type": "object",
            "description": "Optional item_id -> domain label mapping.",
            "additionalProperties": {"type": "string"},
        },
        "baseline_name": {"type": "string", "default": "baseline"},
        "candidate_name": {"type": "string", "default": "candidate"},
        "policy": _policy_input_schema(),
        "retained_probe": _retained_probe_input_schema(),
    }
    operation_schema = {
        "target_heading": {
            "type": "string",
            "minLength": 1,
            "description": "Markdown heading text without the leading # characters.",
        },
        "find": {"type": "string", "minLength": 1},
        "replace": {"type": "string"},
        "allow_token_changes": _string_array_schema(
            "Protected literal tokens that this operation may intentionally change."
        ),
    }
    memory_properties = {
        "content": {"type": "string", "minLength": 1},
        "entities": _string_array_schema("Entities explicitly present in this memory."),
        "kind": {"type": "string", "default": "episodic"},
        "source": {"type": "string", "default": "uploaded"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.8},
        "age": {"type": "integer", "minimum": 0, "default": 0},
        "use_count": {"type": "integer", "minimum": 0, "default": 0},
    }
    event_properties = {
        "step": {"type": "integer", "minimum": 0},
        "kind": {
            "type": "string",
            "description": "Event class such as control, verifier, model, or observation.",
        },
        "detail": {"type": "string"},
        "source": {"type": "string", "default": "native"},
    }
    return {
        "inspect_promotion": {
            "type": "object",
            "description": "Audit exposure, prove a complete clean cohort, then gate baseline versus candidate.",
            "properties": {**leakage_properties, **gate_properties},
            "required": ["exam", "baseline", "candidate"],
            "additionalProperties": False,
        },
        "audit_leakage": {
            "type": "object",
            "description": "Audit declared exposure against an exam and export the clean remainder.",
            "properties": leakage_properties,
            "required": ["exam"],
            "additionalProperties": False,
        },
        "promotion_gate": {
            "type": "object",
            "description": "Compare identical baseline and candidate item cohorts under an explicit policy.",
            "properties": gate_properties,
            "required": ["baseline", "candidate"],
            "additionalProperties": False,
        },
        "bank_health": {
            "type": "object",
            "description": "Diagnose item lifecycle health from one or more grading-history rows.",
            "properties": {
                "items": _record_array_schema(
                    "Optional item definitions.",
                    item_properties={
                        "item_id": {"type": "string", "minLength": 1},
                        "domain": {"type": "string"},
                    },
                    item_required=("item_id",),
                ),
                "history": _record_array_schema(
                    "Observed item/system outcomes.",
                    min_items=1,
                    item_properties={
                        "item_id": {"type": "string", "minLength": 1},
                        "system": {"type": "string", "minLength": 1},
                        "passed": {"type": "boolean"},
                        "domain": {"type": "string"},
                    },
                    item_required=("item_id", "system", "passed"),
                ),
            },
            "required": ["history"],
            "additionalProperties": False,
        },
        "safe_patch": {
            "type": "object",
            "description": "Apply deterministic, section-scoped Markdown replacements under conservation checks.",
            "properties": {
                "document": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_MARKDOWN_CHARS,
                    "description": "Complete Markdown document to patch.",
                },
                "reason": {"type": "string"},
                "operations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": operation_schema,
                        "required": ["target_heading", "find", "replace"],
                        "additionalProperties": False,
                    },
                    "minItems": 1,
                    "maxItems": 50,
                },
            },
            "required": ["document", "operations"],
            "additionalProperties": False,
        },
        "counterexample_hunt": {
            "type": "object",
            "description": "Run a bounded graph search against one Whetstone predicate expression.",
            "properties": {
                "expression": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 500,
                    "description": (
                        "Graph predicate in the Whetstone DSL, for example: "
                        "is_connected and is_triangle_free and not is_bipartite"
                    ),
                },
                "ns": {
                    "type": "array",
                    "description": "Graph sizes searched.",
                    "items": {"type": "integer", "minimum": 4, "maximum": 12},
                    "minItems": 1,
                    "maxItems": 5,
                    "default": [8, 9, 10, 11],
                },
                "restarts": {"type": "integer", "minimum": 1, "maximum": 6, "default": 4},
                "steps": {"type": "integer", "minimum": 50, "maximum": 1500, "default": 800},
                "seed": {"type": "integer", "default": 0},
            },
            "required": ["expression"],
            "additionalProperties": False,
        },
        "memory_relevance": {
            "type": "object",
            "description": "Rank caller-supplied memories against a concrete objective under a token budget.",
            "properties": {
                "objective": {"type": "string", "minLength": 1},
                "objective_entities": _string_array_schema(
                    "Optional explicit entities when the objective text is not self-describing."
                ),
                "context_entities": _string_array_schema("Entities already active in context."),
                "token_budget": {"type": "integer", "minimum": 1, "maximum": 10_000, "default": 90},
                "current_step": {"type": "integer", "minimum": 0},
                "question_kind": {"type": "string", "default": "generic"},
                "memories": _record_array_schema(
                    "Memories to rank.",
                    min_items=1,
                    max_items=1_000,
                    item_properties=memory_properties,
                    item_required=("content",),
                    item_additional_properties=False,
                ),
            },
            "required": ["objective", "memories"],
            "additionalProperties": False,
        },
        "replay_trace": {
            "type": "object",
            "description": "Reconstruct checkpoints, rewinds, branches, and verifier outcomes from control events.",
            "properties": {
                "events": _record_array_schema(
                    "Ordered reasoning-emulator events.",
                    min_items=1,
                    max_items=MAX_EVENT_RECORDS,
                    item_properties=event_properties,
                    item_required=("kind", "detail"),
                    item_additional_properties=False,
                ),
                "notes": _string_array_schema("Optional analyst notes.", max_items=MAX_EVENT_RECORDS),
            },
            "required": ["events"],
            "additionalProperties": False,
        },
    }


def examples() -> dict[str, Any]:
    prompts = (
        "Repair a Python retry loop without changing its public return type.",
        "Reject a malformed JSON payload while preserving valid zero values.",
        "Identify the smallest safe patch for an off-by-one pagination bug.",
        "Keep an asynchronous worker idempotent after a duplicate delivery.",
        "Explain a delayed shipment without inventing a delivery date.",
        "Apply the refund policy while preserving the stated eligibility window.",
        "Reset a locked customer password after identity verification.",
        "Summarize a refund policy without changing eligibility dates.",
    )
    exam = [
        {"item_id": f"item-{index}", "domain": "code" if index <= 4 else "support", "prompt": prompt}
        for index, prompt in enumerate(prompts, 1)
    ]
    return {
        "inspector": {
            "exam": exam,
            "exposure": [
                {"item_id": "item-8", "source": "training.jsonl:42"},
                {"prompt": "After identity verification, safely reset the locked customer password.", "source": "support-sft.jsonl:91"},
            ],
            "baseline": {f"item-{index}": False for index in range(1, 9)},
            "candidate": {**{f"item-{index}": index <= 6 for index in range(1, 8)}, "item-8": True},
            "baseline_name": "v1",
            "candidate_name": "v2",
            "policy": {"min_gains": 1, "max_regressions": 0, "confidence_alpha": 0.05},
        },
        "leakage": {
            "exam": [
                {"item_id": "exact-row", "prompt": "Summarize the refund policy without changing eligibility dates."},
                {"item_id": "behavioral-row", "expression": "is_connected and not is_bipartite"},
                {"item_id": "review-row", "prompt": "Reset a locked customer password after identity verification."},
                {"item_id": "clean-row", "prompt": "Calculate shipment tax for a declared destination."},
            ],
            "exposure": [
                {"prompt": "Summarize the refund policy without changing eligibility dates.", "source": "declared-training.jsonl:7"},
                {"expression": "not is_bipartite and is_connected", "source": "graph-sft.jsonl:19"},
                {"prompt": "After identity verification, safely reset the locked customer password.", "source": "support-sft.jsonl:91"},
            ],
            "fingerprint_max_n": 4,
            "similarity_threshold": 0.78,
        },
        "gate": {
            "baseline": {f"item-{index}": False for index in range(1, 8)},
            "candidate": {f"item-{index}": index <= 6 for index in range(1, 8)},
            "domains": {f"item-{index}": "code" if index <= 4 else "support" for index in range(1, 8)},
            "baseline_name": "v1",
            "candidate_name": "v2",
            "policy": {"min_gains": 1, "max_regressions": 0, "confidence_alpha": 0.05},
        },
        "health": {
            "items": [
                {"item_id": "stable", "domain": "code"},
                {"item_id": "frontier", "domain": "code"},
                {"item_id": "hard", "domain": "support"},
                {"item_id": "noisy", "domain": "support"},
            ],
            "history": [
                {"item_id": "stable", "system": system, "passed": True, "domain": "code"}
                for system in ("small", "medium", "large")
            ] + [
                {"item_id": "frontier", "system": "small", "passed": False, "domain": "code"},
                {"item_id": "frontier", "system": "medium", "passed": False, "domain": "code"},
                {"item_id": "frontier", "system": "large", "passed": True, "domain": "code"},
                {"item_id": "hard", "system": "small", "passed": False, "domain": "support"},
                {"item_id": "hard", "system": "medium", "passed": False, "domain": "support"},
                {"item_id": "noisy", "system": "medium", "passed": True, "domain": "support"},
                {"item_id": "noisy", "system": "medium", "passed": False, "domain": "support"},
                {"item_id": "noisy", "system": "large", "passed": True, "domain": "support"},
            ],
        },
        "safepatch": {
            "document": "# Summary\nRelease 4 ships July 12, 2026.\n\n# Notes\nThe draft is wordy.\n",
            "reason": "Tighten the Notes section only.",
            "operations": [{
                "target_heading": "Notes",
                "find": "The draft is wordy.",
                "replace": "The draft is concise.",
                "allow_token_changes": [],
            }],
        },
        "counterexample": {
            "expression": "is_connected and is_triangle_free and not is_bipartite",
            "ns": [8, 9, 10, 11],
            "restarts": 4,
            "steps": 800,
            "seed": 0,
        },
        "memory": {
            "objective": "Who currently holds the obsidian key?",
            "objective_entities": ["obsidian_key"],
            "context_entities": ["comet", "harbor"],
            "token_budget": 18,
            "memories": [
                {"content": "URGENT: the crimson comet appeared over Harbor City.", "entities": ["comet", "harbor"], "confidence": 1.0},
                {"content": "Mara hands the obsidian key to Ivo at the archive.", "entities": ["obsidian_key", "ivo"], "confidence": 0.9},
                {"content": "The archive elevator was repainted blue.", "entities": ["archive", "elevator"], "confidence": 0.8},
                {"content": "STATE: Ivo currently holds the obsidian key.", "entities": ["obsidian_key", "ivo"], "kind": "semantic", "confidence": 0.95},
            ],
        },
        "replay": {
            "notes": [],
            "events": [
                {"step": 1, "kind": "control", "detail": "SAVE first_try"},
                {"step": 2, "kind": "control", "detail": "CHECK is_tree"},
                {"step": 3, "kind": "verifier", "detail": "REJECT counterexample n=6"},
                {"step": 4, "kind": "control", "detail": "LOAD first_try :: tree hypothesis failed"},
                {"step": 5, "kind": "control", "detail": "CHECK is_tree and max_degree <= 2"},
                {"step": 6, "kind": "verifier", "detail": "ACCEPT"},
                {"step": 7, "kind": "control", "detail": "ANSWER is_tree and max_degree <= 2"},
            ],
        },
    }


def catalog() -> list[dict[str, str]]:
    return [
        {"id": "inspector", "name": "Whetstone Inspector", "endpoint": "/api/inspect", "promise": "Quarantine exposure, compare paired outcomes, and issue a receipt."},
        {"id": "leakage", "name": "Eval Leak Auditor", "endpoint": "/api/leakage", "promise": "Exact declared-exposure audit with a clean exam export."},
        {"id": "gate", "name": "Promotion Gate", "endpoint": "/api/gate", "promise": "PASS, HOLD, or BLOCK from paired item-level results."},
        {"id": "health", "name": "Bank Health", "endpoint": "/api/health-report", "promise": "Find discriminators, saturation, flakiness, and frontier gaps."},
        {"id": "safepatch", "name": "SafePatch", "endpoint": "/api/safepatch", "promise": "Apply a section-scoped Markdown patch under conservation checks."},
        {"id": "counterexample", "name": "Counterexample Hunter", "endpoint": "/api/counterexample", "promise": "Search the supported graph DSL and return an exact witness when found."},
        {"id": "memory", "name": "Memory Relevance Debugger", "endpoint": "/api/memory", "promise": "Compare query-free salience against objective-conditioned relevance."},
        {"id": "replay", "name": "Agent Replay Console", "endpoint": "/api/replay", "promise": "Turn emulator events into checkpoints, rewinds, notes, and a timeline."},
    ]
