from __future__ import annotations

import json
from pathlib import Path

from bcv.panel import SUPPORT_PANEL, calibrate_panel


HARD_CORPUS = Path(__file__).resolve().parent.parent / "sample_docs" / "support_hard_calibration.jsonl"


def _triples():
    rows = [json.loads(line) for line in HARD_CORPUS.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows, [(row["case"], row["answer"], row["human_pass"]) for row in rows]


def test_hard_support_corpus_is_explicitly_expert_authored_and_mixed():
    rows, triples = _triples()
    assert len(rows) == 13
    assert {row["label_source"] for row in rows} == {"expert_authored_adversarial"}
    assert sum(human_pass for _, _, human_pass in triples) == 7
    assert sum(not human_pass for _, _, human_pass in triples) == 6


def test_hard_corpus_exposes_panel_disagreement_instead_of_claiming_clean_perfection():
    _, triples = _triples()
    result = calibrate_panel(SUPPORT_PANEL, triples)
    assert result.cases == 13
    assert result.agreement < 1.0
    assert result.false_accepts > 0 or result.false_rejects > 0
