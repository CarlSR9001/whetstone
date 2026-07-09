"""Local semantic-review panel for fuzzy support tasks.

This is deliberately a separate research panel, not a silent replacement for
the lexical v1 panel. Its model judgment is calibrated against labeled cases;
malformed or unavailable judgments abstain and the veto panel fails closed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from bcv.candidates import OpenAICompatibleCandidate
from bcv.panel import Check, Verdict, VerifierPanel, _check_no_invented_numbers, _check_nonempty, _check_policy


@dataclass
class LocalSemanticJudge:
    base_url: str
    model: str
    max_tokens: int = 1536
    timeout_seconds: float = 180.0

    def __post_init__(self) -> None:
        self.client = OpenAICompatibleCandidate(
            self.base_url, self.model, max_tokens=self.max_tokens, timeout_seconds=self.timeout_seconds
        )
        if self.client.is_external:
            raise ValueError("semantic panel only accepts an in-boundary local endpoint")

    def judge(self, case: dict, answer: str) -> Verdict:
        prompt = (
            "You are a strict support-quality verifier. Decide whether ANSWER is fully supported by SOURCE, "
            "answers QUESTION, preserves every material condition, and avoids promises or claims not in SOURCE. "
            "Do not reward copied words if their meaning is wrong. Output only JSON: "
            '{"verdict":"pass"|"fail"|"abstain"}. Use abstain only when the evidence is genuinely ambiguous.\n\n'
            f"SOURCE:\n{case.get('source', '')}\n\nQUESTION:\n{case.get('question', '')}\n\nANSWER:\n{answer}"
        )
        raw = self.client.generate_text(prompt, temperature=0.0)
        matches = re.findall(r"\{\s*\"verdict\"\s*:\s*\"(?:pass|fail|abstain)\"\s*\}", raw)
        if not matches:
            return "abstain"
        try:
            verdict = json.loads(matches[-1])["verdict"]
        except (json.JSONDecodeError, KeyError):
            return "abstain"
        return verdict if verdict in ("pass", "fail", "abstain") else "abstain"


def semantic_support_panel(judge: LocalSemanticJudge) -> VerifierPanel:
    return VerifierPanel(
        name=f"support_agent_semantic:{judge.model}",
        checks=[
            Check("substantive_reply", "reply is non-trivial", _check_nonempty),
            Check("no_invented_numbers", "every number traces to source or ticket", _check_no_invented_numbers),
            Check("policy_compliant", "no forbidden promises", _check_policy),
            Check("semantic_source_verdict", "local semantic judge verifies source faithfulness", lambda case, answer: judge.judge(case, answer)),
        ],
        min_effective_checks=3,
    )
