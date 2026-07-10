from __future__ import annotations

import json

import bcv.engine_bakeoff as bakeoff
from bcv.examiner import ExamItem, ExaminerBank


def _chess_item(item_id: str, fen: str, acceptable: str) -> ExamItem:
    return ExamItem(
        item_id=item_id,
        domain="chess",
        kind="game_move",
        payload={"fen": fen, "acceptable": [[acceptable]], "rules": {"game": "chess"}},
        oracle="stockfish_d12",
        source="engine_frontier",
        horizon="d12_vs_d2",
        lineage=["stockfish"],
        status="promoted",
    )


def _go_item(item_id: str, acceptable: str) -> ExamItem:
    return ExamItem(
        item_id=item_id,
        domain="go",
        kind="game_move",
        payload={
            "moves": ["E5", "C5"],
            "to_move": "black",
            "acceptable": [[acceptable]],
            "rules": {"game": "go9"},
        },
        oracle="katago_v48",
        source="engine_frontier",
        horizon="v48_vs_v2",
        lineage=["katago_b6c96"],
        status="promoted",
    )


def _bank(tmp_path) -> ExaminerBank:
    bank = ExaminerBank(tmp_path / "bank")
    bank.add(_chess_item("chess_a", "startpos-fen-1", "e2e4"))
    bank.add(_chess_item("chess_b", "startpos-fen-2", "d2d4"))
    bank.add(_go_item("go_a", "G5"))
    bank.add(_go_item("go_b", "C4"))
    bank.save()
    return bank


def test_bakeoff_receipt_is_sanitized_and_gated(tmp_path, monkeypatch):
    _bank(tmp_path)

    canned = {
        "engine_shallow_d2_v2": {"chess_a": "a2a3", "chess_b": "a2a3", "go_a": "A1", "go_b": "A1"},
        "engine_mid_d8_v16": {"chess_a": "e2e4", "chess_b": "d2d4", "go_a": "G5", "go_b": "A1"},
    }

    def fake_tier_answers(items, tier):
        return {item.item_id: canned[tier.name][item.item_id] for item in items}

    monkeypatch.setattr(bakeoff, "tier_answers", fake_tier_answers)
    receipt = bakeoff.run_engine_bakeoff(
        root=tmp_path / "bank",
        receipt_path=tmp_path / "receipt.json",
        repeats=1,
    )

    assert receipt["systems"]["baseline"]["total"] == "0/4"
    assert receipt["systems"]["candidate"]["total"] == "3/4"
    assert receipt["systems"]["candidate"]["by_domain"] == {"chess": "2/2", "go": "1/2"}
    assert receipt["gate_strict"]["gains"] == 3
    assert receipt["gate_strict"]["regressions"] == 0
    assert receipt["gate_strict"]["verdict"] in ("PASS", "HOLD")  # 3-0 at alpha .05 -> HOLD
    assert "resolution" in receipt["gate_strict"]
    assert receipt["repeated_grading"]["repeats"] == 1

    # sanitization: no FENs, move histories, engine answers, or item ids anywhere
    serialized = json.dumps(receipt)
    for secret in ("startpos-fen", "e2e4", "G5", "chess_a", "go_b", '"moves"', '"fen"'):
        assert secret not in serialized, secret

    # the grades landed on the real bank with run manifests
    reloaded = ExaminerBank(tmp_path / "bank")
    graded = reloaded.items["chess_a"].graded
    assert set(graded) == {"engine_shallow_d2_v2", "engine_mid_d8_v16"}
    events = (tmp_path / "bank" / "grade_events.jsonl").read_text(encoding="utf-8")
    assert "engine_bakeoff" in events  # run manifest recorded


def test_repeated_grading_resolves_regression_reliability(tmp_path, monkeypatch):
    """A regression that flips across repeated grades is classified noisy, not
    unknown — the reliability-aware verdict can then act on measured evidence."""
    _bank(tmp_path)

    run_index = {"n": 0}

    def fake_tier_answers(items, tier):
        answers = {}
        for item in items:
            if tier.name == "engine_shallow_d2_v2":
                answers[item.item_id] = "C4" if item.item_id == "go_b" else "zz"
            else:
                # candidate solves three items always; go_b only on the first two runs
                if item.item_id == "go_b":
                    answers[item.item_id] = "C4" if run_index["n"] < 4 else "A1"
                else:
                    answers[item.item_id] = {"chess_a": "e2e4", "chess_b": "d2d4", "go_a": "G5"}[item.item_id]
        run_index["n"] += 1
        return answers

    monkeypatch.setattr(bakeoff, "tier_answers", fake_tier_answers)
    receipt = bakeoff.run_engine_bakeoff(
        root=tmp_path / "bank",
        receipt_path=None,
        repeats=3,
    )

    # latest pairing: 3 gains + 1 regression (go_b) -> strict BLOCKs
    assert receipt["gate_strict"]["verdict"] == "BLOCK"
    assert receipt["gate_strict"]["regressions"] == 1
    # flakiness is now measured (3 obs/system on go_b): the flip classifies noisy
    classifications = receipt["gate_reliability_aware"]["regression_classifications"]
    assert classifications[0]["classification"] == "noisy"
    assert classifications[0]["flip_rate"] is not None
    # one noisy regression is inside budget; verdict advances past the
    # regression check to the evidence checks (HOLD: 3 gains can't clear alpha)
    assert receipt["gate_reliability_aware"]["verdict"] == "HOLD"
    assert "item_id" not in json.dumps(receipt["gate_reliability_aware"])


def test_bakeoff_refuses_empty_bank(tmp_path):
    ExaminerBank(tmp_path / "empty").save()
    try:
        bakeoff.run_engine_bakeoff(root=tmp_path / "empty", receipt_path=None)
    except SystemExit as error:
        assert "no promoted items" in str(error)
    else:
        raise AssertionError("expected SystemExit on empty bank")
