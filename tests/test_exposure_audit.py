from __future__ import annotations

from bcv.examiner import ExamItem, ExaminerBank
from bcv.exposure_audit import (
    KNOWN_PUBLISHED_PREFIX_HASHES,
    audit_bank,
    is_known_published_prefix,
    prefix_sha256,
    published_prefixes,
)

LOG = """
2026-07-02 13:52:21-0500: Controller: boardsize 9
2026-07-02 13:52:21-0500: Controller: komi 7
2026-07-02 13:52:21-0500: Controller: clear_board
2026-07-02 13:52:21-0500: Controller: play b E5
2026-07-02 13:52:21-0500: Controller: play w C5
2026-07-02 13:52:21-0500: Controller: genmove b
2026-07-02 13:52:21-0500: Controller: undo
2026-07-02 13:52:21-0500: Controller: play b D7
2026-07-02 13:52:21-0500: Controller: clear_board
2026-07-02 13:52:21-0500: Controller: play b G3
"""


def _go_item(item_id: str, moves: list[str]) -> ExamItem:
    return ExamItem(
        item_id=item_id,
        domain="go",
        kind="game_move",
        payload={"moves": moves, "to_move": "white", "acceptable": [["A1"]], "rules": {"game": "go9"}},
        oracle="katago_v48",
        source="engine_frontier",
        horizon="v48_vs_v2",
        lineage=["katago_b6c96"],
        status="promoted",
    )


def test_prefix_reconstruction_handles_undo_and_reset(tmp_path):
    log = tmp_path / "session.log"
    log.write_text(LOG, encoding="utf-8")
    prefixes = published_prefixes([log])
    assert ("E5",) in prefixes
    assert ("E5", "C5") in prefixes
    assert ("E5", "C5", "D7") in prefixes  # undo removed the genmove, then D7 played
    assert ("G3",) in prefixes  # clear_board reset the state
    assert ("E5", "C5", "G3") not in prefixes


def test_audit_burns_only_published_positions(tmp_path):
    log = tmp_path / "session.log"
    log.write_text(LOG, encoding="utf-8")
    prefixes = published_prefixes([log])

    bank = ExaminerBank(tmp_path / "bank")
    bank.add(_go_item("go_exposed", ["E5", "C5", "D7"]))
    bank.add(_go_item("go_private", ["C3", "F4", "E6"]))
    bank.save()

    dry = audit_bank(tmp_path / "bank", prefixes, burn=False, provider="test", reason="test")
    assert dry["exposed_items_found"] == 1
    assert dry["burned"] == 0
    assert ExaminerBank(tmp_path / "bank").items["go_exposed"].status == "promoted"

    live = audit_bank(tmp_path / "bank", prefixes, burn=True, provider="public repo", reason="log leak")
    assert live["burned"] == 1
    reloaded = ExaminerBank(tmp_path / "bank")
    assert reloaded.items["go_exposed"].status == "burned"
    assert reloaded.items["go_exposed"].exposures[0]["provider"] == "public repo"
    assert reloaded.items["go_private"].status == "promoted"
    assert live["go_items_still_reusable"] == 1


def test_committed_prefix_hashes_survive_without_raw_logs():
    assert len(KNOWN_PUBLISHED_PREFIX_HASHES) == 22
    assert len({prefix_sha256(prefix) for prefix in [("E5",), ("E5", "C5")]}) == 2
    # The concrete leaked positions are intentionally not repeated in this test;
    # the committed count and membership helper are the fresh-clone contract.
    assert is_known_published_prefix([]) is False
