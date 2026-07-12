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
from typing import Any

from bcv.gate import exact_mcnemar_p_value, power_statement
from bcv.markdown_editor import MarkdownPatch, PatchError, PatchOperation, apply_markdown_patch
from bcv.memory_bench import Probe
from bcv.memstore import Memory as StoredMemory
from bcv.relevance import relevance_score, salience_prior


MAX_RECORDS = 5_000
MAX_MARKDOWN_CHARS = 200_000
MAX_EVENT_RECORDS = 5_000
CONTENT_IDENTITY_FIELDS = ("prompt", "content", "input", "task", "question")
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


def audit_leakage(payload: dict[str, Any]) -> dict[str, Any]:
    exam = _records(payload.get("exam", []), "exam")
    exposure = _records(payload.get("exposure", []), "exposure")
    exposure_index: dict[str, list[dict[str, str]]] = defaultdict(list)
    for index, row in enumerate(exposure):
        source = str(row.get("source") or row.get("path") or f"exposure row {index + 1}")
        for token, reason in _identity_tokens(row).items():
            exposure_index[token].append({"source": source, "reason": reason})

    clean_exam: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(exam):
        item_id = _item_id(row, "exam", index)
        if item_id in seen_ids:
            raise ProductInputError(f"exam has duplicate item_id {item_id}")
        seen_ids.add(item_id)
        matches: list[dict[str, str]] = []
        for token, exam_reason in _identity_tokens(row).items():
            for match in exposure_index.get(token, ()):  # exact identity only
                matches.append({
                    "reason": "row_identity" if exam_reason in {"item_id", "id", "exposure_key"} else exam_reason,
                    "source": match["source"],
                })
        if matches:
            unique = sorted({(m["reason"], m["source"]) for m in matches})
            quarantined.append({
                "item_id": item_id,
                "matches": [{"reason": reason, "source": source} for reason, source in unique],
            })
        else:
            clean_exam.append(row)

    summary = {
        "schema_version": 1,
        "exact_identity_only": True,
        "exam_items": len(exam),
        "exposure_rows": len(exposure),
        "quarantined_items": len(quarantined),
        "clean_items": len(clean_exam),
        "exposure_rate": round(len(quarantined) / len(exam), 6) if exam else 0.0,
        "quarantined": quarantined,
        "clean_exam": clean_exam,
        "input_sha256": _sha256({"exam": exam, "exposure": exposure}),
        "claim_boundary": "Declared/exact identity only; no semantic near-duplicate claim is made.",
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
        bucket = by_domain.setdefault(domain, {"items": 0, "gains": 0, "regressions": 0, "ties": 0})
        bucket["items"] += 1
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

    stable_input = {
        "baseline": baseline,
        "candidate": candidate,
        "domains": domains,
        "policy": policy,
        "retained_probe": retained,
    }
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "reasons": reasons,
        "systems": {
            "baseline": str(payload.get("baseline_name", "baseline")),
            "candidate": str(payload.get("candidate_name", "candidate")),
        },
        "policy": policy,
        "paired_evidence": {
            "items": len(item_rows),
            "item_set_sha256": _sha256(sorted(baseline)),
            "gains": gains,
            "regressions": regressions,
            "ties": ties,
            "exact_mcnemar_two_sided_p": p_value,
            "resolution": power_statement(gains, regressions, policy["confidence_alpha"]),
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
            "schema_version": 1,
            "verdict": "HOLD",
            "reasons": ["no decision: both systems must cover the complete post-quarantine cohort"],
            "paired_evidence": {"items": 0, "gains": 0, "regressions": 0, "ties": 0},
        }
    result = {
        "schema_version": 1,
        "audit": audit,
        "cohort": cohort,
        "gate": gate,
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
    for index, row in enumerate(history):
        item_id = _item_id(row, "history", index)
        system = row.get("system")
        if not isinstance(system, str) or not system.strip():
            raise ProductInputError(f"history row {index + 1} needs a system")
        passed = _as_bool(row.get("passed", row.get("outcome")), f"history row {index + 1} outcome")
        stats[item_id][system.strip()][0 if passed else 1] += 1
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
        rows.append({
            "item_id": item_id,
            "domain": domains.get(item_id, "unlabeled"),
            "systems": len(rates),
            "observations": observations,
            "pass_rates": rates,
            "discrimination": discrimination,
            "max_within_system_flip_rate": round(max_flip_rate, 6),
            "classification": classification,
        })

    by_class: dict[str, int] = defaultdict(int)
    by_domain: dict[str, dict[str, int]] = defaultdict(lambda: {"items": 0, "discriminating": 0})
    for row in rows:
        by_class[row["classification"]] += 1
        by_domain[row["domain"]]["items"] += 1
        by_domain[row["domain"]]["discriminating"] += row["classification"] == "discriminating"
    result = {
        "schema_version": 1,
        "items": len(rows),
        "systems": sorted({system for item in stats.values() for system in item}),
        "classification_counts": dict(sorted(by_class.items())),
        "by_domain": dict(sorted(by_domain.items())),
        "frontier_gaps": sorted(domain for domain, counts in by_domain.items() if counts["discriminating"] == 0),
        "retirement_candidates": [row["item_id"] for row in rows if row["classification"] == "saturated"],
        "items_detail": rows,
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
    result = {
        "schema_version": 1,
        "accepted": True,
        "updated_document": updated,
        "unified_diff": diff,
        "changed_sections": sorted({operation.target_heading for operation in operations}),
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
    selected = []
    used = 0
    for row in relevance_order:
        cost = len(row["content"].split())
        if used + cost <= token_budget:
            selected.append(row["id"])
            used += cost
    result = {
        "schema_version": 1,
        "objective": objective,
        "objective_entities": list(objective_entities),
        "token_budget": token_budget,
        "selected_by_relevance": selected,
        "selected_tokens": used,
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
    checkpoints: dict[str, int] = {}
    controls: dict[str, int] = defaultdict(int)
    notes = [str(note) for note in payload.get("notes", [])] if isinstance(payload.get("notes", []), list) else []
    timeline = []
    branch = 0
    rewinds = []
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
                    checkpoints[argument.split()[0]] = step
                elif control == "LOAD":
                    target, _, note = argument.partition("::")
                    branch += 1
                    rewinds.append({"step": step, "target": target.strip(), "target_step": checkpoints.get(target.strip())})
                    if note.strip():
                        notes.append(note.strip())
        timeline.append({
            "step": step,
            "kind": kind,
            "detail": detail,
            "control": control,
            "branch": branch,
            "source": str(event.get("source", "native")),
        })
    result = {
        "schema_version": 1,
        "events": len(events),
        "controls": dict(sorted(controls.items())),
        "checkpoints": [{"name": name, "step": step} for name, step in sorted(checkpoints.items())],
        "rewinds": rewinds,
        "notes": list(dict.fromkeys(notes)),
        "external_interventions": sum(1 for row in timeline if row["source"] != "native"),
        "timeline": timeline,
        "claim_boundary": "Transcript reconstruction only; this console does not claim hidden chain-of-thought access.",
    }
    result["receipt_sha256"] = _sha256(result)
    return result


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
        "schema_version": 1,
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
        result["certificate_sha256"] = _sha256(find)
    result["receipt_sha256"] = _sha256(result)
    return result


def examples() -> dict[str, Any]:
    exam = [
        {"item_id": f"item-{index}", "domain": "code" if index <= 4 else "support", "prompt": f"Disposable demo item {index}"}
        for index in range(1, 9)
    ]
    return {
        "inspector": {
            "exam": exam,
            "exposure": [{"item_id": "item-8", "source": "training.jsonl:42"}],
            "baseline": {f"item-{index}": False for index in range(1, 9)},
            "candidate": {**{f"item-{index}": index <= 6 for index in range(1, 8)}, "item-8": True},
            "baseline_name": "v1",
            "candidate_name": "v2",
            "policy": {"min_gains": 1, "max_regressions": 0, "confidence_alpha": 0.05},
        },
        "leakage": {
            "exam": exam[:3],
            "exposure": [{"prompt": "Disposable demo item 2", "source": "declared-training.jsonl:7"}],
        },
        "gate": {
            "baseline": {f"item-{index}": False for index in range(1, 8)},
            "candidate": {f"item-{index}": index <= 6 for index in range(1, 8)},
            "domains": {f"item-{index}": "code" for index in range(1, 8)},
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
            "token_budget": 36,
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
