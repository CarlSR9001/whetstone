from __future__ import annotations

from bcv.research_foundry import run_research_foundry


def test_research_foundry_feedback_condition_accumulates_repairs(tmp_path):
    result = run_research_foundry(
        root=tmp_path,
        max_n=6,
        rounds=2,
        max_rules=4,
        mode="scripted",
        train_adapter=False,
    )

    assert result.stateless.rounds[0].accepted == 0
    assert result.git_feedback.rounds[0].repairs > 0
    assert len(result.git_feedback.accepted_expressions) > len(result.stateless.accepted_expressions)
    assert result.git_feedback.sft_examples > 0
    assert (tmp_path / "comparison.json").exists()
    assert (tmp_path / "ledger").exists()


def test_research_foundry_stress_check_flags_scale_falsifications(tmp_path):
    result = run_research_foundry(
        root=tmp_path,
        max_n=6,
        rounds=2,
        max_rules=4,
        mode="scripted",
        train_adapter=False,
        stress_ns=(7, 8),
    )

    assert result.stress_ns == (7, 8)
    checked = result.stress_survived + result.stress_falsified
    assert checked > 0
    # The scripted feedback loop accepts repairs mined at n<=6; the stress pool
    # includes deterministic greedy adversaries, so some must die at scale.
    assert result.stress_falsified > 0
    assert (tmp_path / "generalization" / "generalization_report.json").exists()


def test_research_foundry_stress_feedback_metrics(tmp_path):
    result = run_research_foundry(
        root=tmp_path,
        max_n=6,
        rounds=2,
        max_rules=4,
        mode="scripted",
        train_adapter=False,
        stress_feedback_ns=(7,),
    )

    rounds = result.git_feedback.rounds
    # The scripted loop's n<=6-verified expressions must show horizon inheritance.
    assert any(round_row.scale_falsified > 0 for round_row in rounds)
    # Scripted single/two-feature proposals live inside the miner hull: zero novelty.
    assert all(round_row.novel_proposed == 0 for round_row in rounds)


def test_research_foundry_resumes_completed_rounds(tmp_path):
    first = run_research_foundry(root=tmp_path, max_n=6, rounds=2, max_rules=4, mode="scripted")
    second = run_research_foundry(root=tmp_path, max_n=6, rounds=2, max_rules=4, mode="scripted")

    assert not any(round_row.resumed for round_row in first.git_feedback.rounds)
    assert all(round_row.resumed for round_row in second.git_feedback.rounds)
    assert (
        second.git_feedback.accepted_expressions == first.git_feedback.accepted_expressions
    )
    assert second.git_feedback.rounds[-1].repairs == first.git_feedback.rounds[-1].repairs
