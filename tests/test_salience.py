from __future__ import annotations

import math

from bcv.salience import (
    SalienceWeights,
    build_recap,
    recap_policies,
    salience_score,
    score_candidates,
    selection_f1,
)
from bcv.salience_eval import evaluate_selection
from bcv.tinyseasons import generate_season, load_corpus, generate_corpus


def test_corpus_ground_truth(tmp_path):
    path = generate_corpus(seasons=3, episodes=6, seed=1, root=tmp_path)
    episodes = load_corpus(path)
    assert len(episodes) == 18
    # Oracle dependencies exist and always point strictly backwards.
    with_deps = [e for e in episodes if e.required_prior_beats]
    assert with_deps
    for episode in with_deps:
        assert all(ep_idx < episode.index for ep_idx, _ in episode.required_prior_beats)
    assert all(1.0 <= e.rating <= 10.0 for e in episodes)


def test_salience_score_gibbs_identity():
    # S = rho * exp(-Phi): the Section-7 potential is exact, not an analogy.
    features = {
        "surprise": 1.2,
        "retention": 1.0,
        "momentum": 0.5,
        "continuity": 0.9,
        "age": 2,
        "distance": 0.4,
        "effort": 1.5,
        "fatigue": 0.0,
    }
    weights = SalienceWeights()
    score, rho, potential = salience_score(features, weights)
    assert math.isclose(score, rho * math.exp(-potential), rel_tol=1e-9)


def test_recap_respects_budget_and_policies_differ():
    season = generate_season(0, episodes=6, seed=3)
    prior, current = season.episodes[:5], season.episodes[5]
    policies = recap_policies(prior, current, token_budget=60, seed=3)
    assert set(policies) == {
        "salience",
        "novelty_only",
        "additive_ablation",
        "no_decay_ablation",
        "shuffled_surprise",
        "uniform",
        "recency",
    }
    for recap in policies.values():
        assert sum(len(b.text.split()) for b in recap) <= 60
    # Recency must differ from salience on a corpus with long-range threads.
    assert policies["recency"] != policies["salience"]


def test_selection_f1_and_fixed_budget_eval(tmp_path):
    path = generate_corpus(seasons=6, episodes=8, seed=5, root=tmp_path)
    episodes = load_corpus(path)
    report = evaluate_selection(episodes, token_budget=90, seed=5)
    f1 = report["selection_f1"]
    assert report["transitions"] > 10
    # The controller must beat the shiny-object and uniform baselines on oracle F1,
    # or the whole apparatus is dead on arrival (paper §12.4 counterfactual gating).
    assert f1["salience"] > f1["uniform"]
    assert f1["salience"] > f1["novelty_only"]


def test_selection_f1_empty_cases():
    assert selection_f1([], [(0, 1)]) == 0.0
    assert selection_f1([], []) == 0.0
