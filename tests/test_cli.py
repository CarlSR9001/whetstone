from __future__ import annotations

import json
from pathlib import Path

from bcv import cli
from bcv.examiner import ExaminerBank

GOOD_SOLUTIONS = {
    "rle_roundtrip": """
def rle_encode(s):
    pairs = []
    for ch in s:
        if pairs and pairs[-1][0] == ch:
            pairs[-1] = (ch, pairs[-1][1] + 1)
        else:
            pairs.append((ch, 1))
    return pairs

def rle_decode(pairs):
    return "".join(ch * count for ch, count in pairs)
""",
    "merge_intervals": """
def merge_intervals(intervals):
    merged = []
    for a, b in sorted(intervals):
        if merged and a <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged
""",
    "balanced_brackets": """
def balanced(s):
    stack = []
    pairs = {')': '(', ']': '[', '}': '{'}
    for ch in s:
        if ch in '([{':
            stack.append(ch)
        elif ch in pairs:
            if not stack or stack.pop() != pairs[ch]:
                return False
    return not stack
""",
}


def _write_answers(path: Path, bank_root: Path, solutions: dict[str, str]) -> None:
    bank = ExaminerBank(bank_root)
    rows = []
    for item in bank.promoted_items():
        task = item.payload["task_id"]
        if task in solutions:
            rows.append({"item_id": item.item_id, "answer": f"```python\n{solutions[task]}\n```"})
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_full_cli_pipeline(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    bank_root = tmp_path / "bank"

    # init writes config + empty bank
    assert cli.main(["--root", str(bank_root), "init"]) == 0
    assert Path("whetstone.toml").exists()

    # mint the code domain
    assert cli.main(["--root", str(bank_root), "mint", "--domain", "code", "--max-items", "3"]) == 0

    # two systems: baseline answers nothing right, candidate solves all three
    baseline_answers = tmp_path / "baseline.jsonl"
    baseline_answers.write_text(json.dumps({"item_id": "none", "answer": "pass"}) + "\n", encoding="utf-8")
    candidate_answers = tmp_path / "candidate.jsonl"
    _write_answers(candidate_answers, bank_root, GOOD_SOLUTIONS)

    assert cli.main(["--root", str(bank_root), "grade", "--system", "base", "--answers", str(baseline_answers)]) == 0
    assert cli.main(["--root", str(bank_root), "grade", "--system", "cand", "--answers", str(candidate_answers)]) == 0

    # 3 gains, 0 regressions, p = 0.25: default alpha 0.05 -> HOLD (exit 2)
    assert cli.main(["--root", str(bank_root), "gate", "--baseline", "base", "--candidate", "cand"]) == 2
    # relaxed alpha accepted on the record -> PASS (exit 0)
    assert cli.main([
        "--root", str(bank_root), "gate", "--baseline", "base", "--candidate", "cand", "--alpha", "0.3",
    ]) == 0
    # swapped direction: 3 regressions -> BLOCK (exit 3)
    assert cli.main(["--root", str(bank_root), "gate", "--baseline", "cand", "--candidate", "base"]) == 3

    # the report artifacts exist and carry the verdict
    report_dir = bank_root / "gate_cand"
    report = json.loads((report_dir / "promotion_gate.json").read_text(encoding="utf-8"))
    assert report["paired_evidence"]["gains"] == 3
    assert (report_dir / "promotion_gate.html").exists()


def test_init_can_create_an_explicit_config_path(tmp_path):
    config = tmp_path / "ci" / "whetstone.toml"
    bank_root = tmp_path / "ci" / "bank"
    assert cli.main(["--config", str(config), "--root", str(bank_root), "init"]) == 0
    assert config.exists()
    assert (bank_root / "items.jsonl").exists()


def test_openai_compatible_candidate_forwards_completion_budget(monkeypatch):
    captured = {}

    class FakeCandidate:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("bcv.candidates.OpenAICompatibleCandidate", FakeCandidate)
    args = cli.build_parser().parse_args([
        "grade", "--system", "local", "--api-base", "http://127.0.0.1:11434/v1",
        "--model", "qwen3:8b", "--max-tokens", "2048",
    ])
    cli.build_candidate(args)
    assert captured["max_tokens"] == 2048


def test_gate_cli_exposes_reliability_aware_policy_controls():
    args = cli.build_parser().parse_args([
        "gate", "--baseline", "old", "--candidate", "new",
        "--regression-policy", "reliability_aware", "--max-noisy-regressions", "2",
        "--reliability-min-observations", "4", "--stable-flip-rate", "0.1",
    ])
    assert args.regression_policy == "reliability_aware"
    assert args.max_noisy_regressions == 2
    assert args.reliability_min_observations == 4
    assert args.stable_flip_rate == 0.1


def test_status_and_burn(tmp_path, capsys):
    bank_root = tmp_path / "bank"
    assert cli.main(["--root", str(bank_root), "mint", "--domain", "code", "--max-items", "2"]) == 0
    capsys.readouterr()

    assert cli.main(["--root", str(bank_root), "status"]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["buckets"]["promoted"] == 2
    assert status["metabolism"]["burned"] == 0

    bank = ExaminerBank(bank_root)
    item_id = bank.promoted_items()[0].item_id
    assert cli.main(["--root", str(bank_root), "burn", "--item", item_id, "--provider", "test", "--reason", "manual"]) == 0
    capsys.readouterr()

    assert cli.main(["--root", str(bank_root), "status"]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["metabolism"]["burned"] == 1
    assert status["buckets"]["promoted"] == 1


def test_calibrate_panel_command(tmp_path, monkeypatch, capsys):
    repo_corpus = Path(__file__).resolve().parent.parent / "sample_docs" / "support_calibration.jsonl"
    assert cli.main([
        "calibrate-panel", "--labeled", str(repo_corpus), "--out", str(tmp_path / "cal.json"),
    ]) == 0
    output = capsys.readouterr().out
    assert '"false_accepts": 0' in output
    assert (tmp_path / "cal.json").exists()
