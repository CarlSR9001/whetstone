from __future__ import annotations

import random

import bcv.exposure_audit as exposure_audit
from bcv.baduk import mint_go_exam_items, random_opening_vertices
from bcv.examiner import ExaminerBank


def test_random_opening_vertices_are_distinct_and_valid():
    rng = random.Random(7)
    vertices = random_opening_vertices(rng, size=9, plies=8)
    assert len(vertices) == 8
    assert len(set(vertices)) == 8
    for vertex in vertices:
        assert vertex[0] in "ABCDEFGHJ"  # GTP letters skip I
        assert 1 <= int(vertex[1:]) <= 9


def test_openings_differ_across_rng_states():
    a = random_opening_vertices(random.Random(1), 9, 6)
    b = random_opening_vertices(random.Random(2), 9, 6)
    assert a != b


def _row(moves, oracle_move="G5", shallow_move="A1"):
    return {
        "game": "go9",
        "moves": moves,
        "to_move": "white",
        "oracle_move": oracle_move,
        "shallow_move": shallow_move,
    }


def test_mint_gate_quarantines_published_positions(tmp_path, monkeypatch):
    monkeypatch.setattr(
        exposure_audit, "known_published_prefixes", lambda: {("E5", "C5")}
    )
    added = mint_go_exam_items(
        [_row(["E5", "C5"]), _row(["D4", "F6"])],
        per_bank=4,
        bank_root=tmp_path / "bank",
    )
    assert added == 1  # only the unpublished position promoted

    bank = ExaminerBank(tmp_path / "bank")
    statuses = {tuple(item.payload["moves"]): item.status for item in bank.items.values()}
    assert statuses[("E5", "C5")] == "quarantined"
    assert statuses[("D4", "F6")] == "promoted"
    quarantined = next(i for i in bank.items.values() if i.status == "quarantined")
    assert quarantined.leakage_match == "published_gtp_log"
    assert quarantined.leakage_risk == 1.0


def test_mint_gate_can_be_disabled_explicitly(tmp_path, monkeypatch):
    monkeypatch.setattr(
        exposure_audit, "known_published_prefixes", lambda: {("E5", "C5")}
    )
    added = mint_go_exam_items(
        [_row(["E5", "C5"])], per_bank=4, bank_root=tmp_path / "bank", check_published=False
    )
    assert added == 1  # research-mode override is explicit, never a default


def test_mint_gate_uses_committed_hash_when_raw_logs_are_absent(tmp_path, monkeypatch):
    moves = ["D4", "F6"]
    monkeypatch.setattr(exposure_audit, "known_published_prefixes", lambda: set())
    monkeypatch.setattr(
        exposure_audit, "KNOWN_PUBLISHED_PREFIX_HASHES", {exposure_audit.prefix_sha256(moves)}
    )
    added = mint_go_exam_items(
        [_row(moves)], per_bank=4, bank_root=tmp_path / "bank"
    )
    assert added == 0
    item = next(iter(ExaminerBank(tmp_path / "bank").items.values()))
    assert item.status == "quarantined"
