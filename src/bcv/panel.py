"""Verifier panels: composed weak checks for domains without an exact oracle.

Where graph theory has exhaustive enumeration and chess has Stockfish, a
support ticket has no oracle. The panel is the honest fallback: several cheap,
independent checks, each allowed to abstain, aggregated by VETO — an answer
passes only if no check fails. Veto aggregation is a measured choice, not a
style: in the value-spectrum experiments (spectrum.py, Continuity arc), veto
was the only aggregation that survived reward hacking; averages and majorities
were gameable. An exam grader is exactly the place an optimizer will probe.

Panels must EARN trust, so calibration is first-class: run the panel over
labeled (case, answer, human_verdict) triples and get agreement, false-accept
and false-reject rates, and per-check attribution. A panel that has not been
calibrated reports itself as uncalibrated — the gate can then refuse to lean
on it. Whetstone grades what it can verify and declines what it cannot; the
calibration record is where that line is drawn in public.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from bcv.examiner import ExamItem

DEFAULT_SUPPORT_CALIBRATION = Path(__file__).resolve().parents[2] / "results" / "support_panel_calibration.json"

Verdict = str  # "pass" | "fail" | "abstain"


@dataclass(frozen=True)
class Check:
    name: str
    description: str
    fn: Callable[[dict, str], Verdict]

    def run(self, case: dict, answer: str) -> Verdict:
        verdict = self.fn(case, answer)
        if verdict not in ("pass", "fail", "abstain"):
            raise ValueError(f"check {self.name} returned invalid verdict {verdict!r}")
        return verdict


@dataclass
class PanelReport:
    passed: bool
    verdicts: dict[str, Verdict]
    failed_checks: list[str]
    abstentions: list[str]


@dataclass
class VerifierPanel:
    name: str
    checks: list[Check]
    min_effective_checks: int = 1  # fewer non-abstaining checks than this -> fail closed

    def grade(self, case: dict, answer: str) -> PanelReport:
        verdicts = {check.name: check.run(case, answer) for check in self.checks}
        failed = [name for name, verdict in verdicts.items() if verdict == "fail"]
        abstained = [name for name, verdict in verdicts.items() if verdict == "abstain"]
        effective = len(verdicts) - len(abstained)
        passed = not failed and effective >= self.min_effective_checks
        return PanelReport(passed=passed, verdicts=verdicts, failed_checks=failed, abstentions=abstained)


@dataclass
class PanelCalibration:
    panel: str
    cases: int
    agreement: float
    false_accepts: int  # panel passed, human failed — the dangerous direction
    false_rejects: int  # panel failed, human passed — the friction direction
    abstention_rate: float
    per_check_fail_counts: dict[str, int]

    def to_dict(self) -> dict:
        return {
            "panel": self.panel,
            "cases": self.cases,
            "agreement": round(self.agreement, 4),
            "false_accepts": self.false_accepts,
            "false_rejects": self.false_rejects,
            "abstention_rate": round(self.abstention_rate, 4),
            "per_check_fail_counts": dict(sorted(self.per_check_fail_counts.items())),
        }


def calibrate_panel(
    panel: VerifierPanel, labeled: Iterable[tuple[dict, str, bool]]
) -> PanelCalibration:
    """Measure panel agreement with human verdicts on labeled triples."""
    cases = agree = false_accepts = false_rejects = abstentions = total_verdicts = 0
    fail_counts: dict[str, int] = {check.name: 0 for check in panel.checks}
    for case, answer, human_pass in labeled:
        report = panel.grade(case, answer)
        cases += 1
        agree += report.passed == human_pass
        false_accepts += report.passed and not human_pass
        false_rejects += (not report.passed) and human_pass
        abstentions += len(report.abstentions)
        total_verdicts += len(report.verdicts)
        for name in report.failed_checks:
            fail_counts[name] += 1
    if cases == 0:
        raise ValueError("calibration requires at least one labeled case")
    return PanelCalibration(
        panel=panel.name,
        cases=cases,
        agreement=agree / cases,
        false_accepts=false_accepts,
        false_rejects=false_rejects,
        abstention_rate=abstentions / total_verdicts if total_verdicts else 0.0,
        per_check_fail_counts=fail_counts,
    )


def save_calibration(calibration: PanelCalibration, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(calibration.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_calibration(path: str | Path) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def calibration_admissible(
    calibration: dict | None,
    min_agreement: float = 0.9,
    max_false_accepts: int = 0,
) -> bool:
    """Whether a panel has earned use as a promotion checker.

    A missing calibration, or one with dangerous false accepts, is not merely a
    dashboard warning: it is insufficient authority to mint private exam items.
    """
    return bool(
        calibration
        and calibration.get("agreement", 0.0) >= min_agreement
        and calibration.get("false_accepts", max_false_accepts + 1) <= max_false_accepts
    )


# ----------------------------------------------------- support-agent panel
# The demo fuzzy domain: a support agent answers a ticket from a source
# document under a policy. Every check is cheap, mechanical, and explainable.


_SENTENCE = re.compile(r"[^.!?]+[.!?]")
_NUMBER = re.compile(r"\d[\d,.]*")


def _check_nonempty(case: dict, answer: str) -> Verdict:
    return "pass" if answer.strip() and len(answer.strip()) >= 20 else "fail"


def _check_grounded_quote(case: dict, answer: str) -> Verdict:
    """The reply must reuse substantive language from the source document."""
    source = case.get("source", "")
    if not source:
        return "abstain"
    answer_words = set(re.findall(r"[a-z]{5,}", answer.lower()))
    for sentence in _SENTENCE.findall(source):
        sentence_words = set(re.findall(r"[a-z]{5,}", sentence.lower()))
        if len(sentence_words) >= 3 and len(answer_words & sentence_words) >= max(2, len(sentence_words) // 2):
            return "pass"
    return "fail"


def _check_no_invented_numbers(case: dict, answer: str) -> Verdict:
    """Every number in the reply must appear in the source or the ticket."""
    numbers = _NUMBER.findall(answer)
    if not numbers:
        return "abstain"
    known = set(_NUMBER.findall(case.get("source", ""))) | set(_NUMBER.findall(case.get("question", "")))
    return "pass" if all(number in known for number in numbers) else "fail"


def _check_policy(case: dict, answer: str) -> Verdict:
    """No promises the policy forbids (refund guarantees, legal advice, ETAs...)."""
    forbidden = case.get("forbidden", ["guarantee a refund", "legal advice", "within 24 hours"])
    lowered = answer.lower()
    return "fail" if any(phrase.lower() in lowered for phrase in forbidden) else "pass"


SUPPORT_PANEL = VerifierPanel(
    name="support_agent_v1",
    checks=[
        Check("substantive_reply", "reply is non-trivial", _check_nonempty),
        Check("grounded_in_source", "reply reuses substantive source language", _check_grounded_quote),
        Check("no_invented_numbers", "every number traces to source or ticket", _check_no_invented_numbers),
        Check("policy_compliant", "no forbidden promises", _check_policy),
    ],
    min_effective_checks=2,
)


# ------------------------------------------------------------------ minting


def mint_support_items(
    tickets: list[dict],
    calibration_path: str | Path | None = DEFAULT_SUPPORT_CALIBRATION,
    max_items: int = 8,
    research_mode: bool = False,
) -> list[ExamItem]:
    """Support exam items, admitted only through a calibration gate.

    ``research_mode`` permits an explicitly non-production bank experiment, but
    callers must opt into it; absence of a trustworthy panel never defaults to
    a promotion-capable support bank.
    """
    calibration = load_calibration(calibration_path) if calibration_path else None
    if not research_mode and not calibration_admissible(calibration):
        raise ValueError(
            "support panel is not admission-calibrated; provide a calibration with "
            "agreement >= 0.9 and zero false accepts, or use research_mode=True"
        )
    items: list[ExamItem] = []
    for ticket in tickets[:max_items]:
        items.append(
            ExamItem(
                item_id=f"support_{uuid.uuid4().hex[:8]}",
                domain="support",
                kind="panel_case",
                payload={
                    "question": ticket["question"],
                    "source": ticket["source"],
                    "forbidden": ticket.get("forbidden", []),
                    "panel": SUPPORT_PANEL.name,
                    "calibration": calibration,
                    "research_mode": research_mode,
                },
                oracle=f"verifier_panel:{SUPPORT_PANEL.name}",
                source="ticket_corpus",
                horizon="panel_veto_v1",
                lineage=[SUPPORT_PANEL.name],
            )
        )
    return items


def support_item_prompt(item: ExamItem) -> str:
    return (
        "You are a customer-support agent. Answer the ticket using ONLY the source document. "
        "Do not promise anything the document does not state.\n\n"
        f"SOURCE DOCUMENT:\n{item.payload['source']}\n\n"
        f"TICKET:\n{item.payload['question']}\n\n"
        "Reply with the support answer only."
    )


def grade_support_answer(item: ExamItem, raw_answer: str | None) -> bool:
    if not raw_answer:
        return False
    return SUPPORT_PANEL.grade(item.payload, str(raw_answer)).passed
