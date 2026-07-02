from __future__ import annotations

import json

from bcv.continual_proposer import ProposerLearner
from bcv.graph_agent import ProposedRule, evaluate_proposals


def _evaluation(tmp_path):
    # is_tree gets rejected and yields mined repairs; complete graphs get accepted.
    return evaluate_proposals(
        (
            ProposedRule("trees", "Trees should be exact.", "is_tree"),
            ProposedRule("complete", "Complete graphs should be exact.", "is_complete"),
        ),
        max_n=6,
        backend="test",
        model="test",
        root=tmp_path / "eval",
    )


def test_learner_gate_promotes_and_blocks(tmp_path):
    probe_scores = iter([0, 2, 1])  # baseline, round-0 candidate, round-1 candidate
    trained_dirs = []

    def fake_probe(adapter_path):
        return next(probe_scores)

    def fake_trainer(buffer_path, adapter_dir):
        trained_dirs.append(adapter_dir)
        return adapter_dir

    learner = ProposerLearner(
        model_name="fake",
        root=tmp_path / "learner",
        probe_fn=fake_probe,
        trainer_fn=fake_trainer,
    )
    evaluation = _evaluation(tmp_path)

    record0 = learner.observe_and_update(evaluation, round_index=0)
    assert record0.new_examples > 0
    assert record0.trained
    assert record0.probe_accepted == 2
    assert record0.promoted
    assert learner.adapter_path == trained_dirs[0]

    # Round 1 candidate regresses (probe 1 < best 2): gate must block promotion.
    record1 = learner.observe_and_update(evaluation, round_index=1)
    assert record1.trained
    assert not record1.promoted
    assert learner.adapter_path == trained_dirs[0]

    # Verifier-approved experience accumulated in the replay buffer.
    buffer_lines = [
        json.loads(line)
        for line in (tmp_path / "learner" / "experience_buffer.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(buffer_lines) == record0.new_examples + record1.new_examples
    for row in buffer_lines:
        assert row["messages"][2]["role"] == "assistant"
        assert "rules" in json.loads(row["messages"][2]["content"])


def test_learner_state_survives_restart(tmp_path):
    def fake_probe(adapter_path):
        return 1

    def fake_trainer(buffer_path, adapter_dir):
        return adapter_dir

    learner = ProposerLearner(
        model_name="fake", root=tmp_path / "learner", probe_fn=fake_probe, trainer_fn=fake_trainer
    )
    evaluation = _evaluation(tmp_path)
    record = learner.observe_and_update(evaluation, round_index=0)
    assert record.promoted

    # Simulate a crash + restart: fresh learner reloads state and replays the record.
    reborn = ProposerLearner(
        model_name="fake", root=tmp_path / "learner", probe_fn=fake_probe, trainer_fn=fake_trainer
    )
    assert reborn.adapter_path == learner.adapter_path
    assert reborn.best_probe == 1
    replayed = reborn.observe_and_update(evaluation, round_index=0)
    assert replayed == record
