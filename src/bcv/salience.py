"""The salience controller of `Salience as an Internal Currency` (Garringer 2026),
implemented literally, then tightened.

Eq. 1 of the paper, for candidate beat i at context x_t:

    S_i = (wA*dA_i + wR*R_i + wM*M_i) * C_i * exp(-lambda*dt_i) * (1 - k*phi)
          / sqrt(d_i * tau_i + eps)

Tightening the Section-7 "GR analogy" into a statement: define the potential
Phi_i = -log(C_i * exp(-lambda*dt_i) * (1 - k*phi)) + 0.5*log(d_i*tau_i + eps)
and the information density rho_i = wA*dA_i + wR*R_i + wM*M_i. Then
S_i = rho_i * exp(-Phi_i), and the paper's softmax selection at temperature T is
exactly the Gibbs measure P(i) ∝ exp((log rho_i - Phi_i)/T) — an energy-based
retrieval model, no physics metaphor required. Building a recap under a token
budget B is budgeted maximization with tau_i as the knapsack weight; the greedy
S_i-descending fill implemented here is the standard density-greedy relaxation.

The paper's §12.4 falsification battery ships as first-class policies:
uniform, novelty-only (shiny-object), recency, additive-form ablation, no-decay
ablation, and shuffled-surprise. §12.5's fixed-budget comparison lives in
bcv/salience_eval.py.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from bcv.tinyseasons import Beat, Episode


@dataclass(frozen=True)
class SalienceWeights:
    wA: float = 1.0
    wR: float = 2.0
    wM: float = 1.5
    decay: float = 0.35  # lambda
    fatigue_k: float = 0.0
    epsilon: float = 0.05


@dataclass(frozen=True)
class ScoredBeat:
    beat: Beat
    score: float
    rho: float
    potential: float
    features: dict


def beat_features(
    beat: Beat,
    current_episode: Episode,
    open_threads: set[int],
    kind_frequency: dict[str, float],
) -> dict:
    """Operational proxies for the paper's telemetry terms (§3.1), computable
    from corpus statistics and derivable state — no oracle access."""
    current_entities = {entity for b in current_episode.beats for entity in b.entities}
    overlap = len(set(beat.entities) & current_entities)
    surprise = -math.log(kind_frequency.get(beat.kind, 0.05))  # rarity as dA
    retention = 1.0 if (beat.thread_id is not None and beat.thread_id in open_threads) else 0.15
    momentum = overlap / max(1, len(beat.entities))
    continuity = 1.0
    age = current_episode.index - beat.episode
    distance = 1.0 - overlap / max(1, len(set(beat.entities) | current_entities))
    effort = max(1.0, len(beat.text.split()) / 8.0)
    return {
        "surprise": surprise,
        "retention": retention,
        "momentum": momentum,
        "continuity": continuity,
        "age": age,
        "distance": distance,
        "effort": effort,
        "fatigue": 0.0,
    }


def salience_score(features: dict, weights: SalienceWeights, additive: bool = False, no_decay: bool = False) -> tuple[float, float, float]:
    rho = (
        weights.wA * features["surprise"]
        + weights.wR * features["retention"]
        + weights.wM * features["momentum"]
    )
    decay = 1.0 if no_decay else math.exp(-weights.decay * features["age"])
    fatigue = 1.0 - weights.fatigue_k * features["fatigue"]
    normalization = 1.0 / math.sqrt(features["distance"] * features["effort"] + weights.epsilon)
    if additive:
        # §12.4 ablation 2: break the multiplicative form.
        score = rho + features["continuity"] + decay + fatigue + normalization
    else:
        score = rho * features["continuity"] * decay * fatigue * normalization
    gate = features["continuity"] * decay * fatigue
    potential = -math.log(max(gate, 1e-9)) + 0.5 * math.log(
        features["distance"] * features["effort"] + weights.epsilon
    )
    return score, rho, potential


def open_threads_before(episodes: list[Episode], episode_index: int) -> set[int]:
    """Threads set up but not yet paid off before the given episode — derivable
    from the past alone (this is the estimable retention proxy, not the oracle)."""
    opened: set[int] = set()
    closed: set[int] = set()
    for episode in episodes:
        if episode.index >= episode_index:
            break
        for beat in episode.beats:
            if beat.thread_id is None:
                continue
            if beat.kind in ("give", "feud_start"):
                opened.add(beat.thread_id)
            elif beat.kind.startswith("payoff"):
                closed.add(beat.thread_id)
    return opened - closed


def kind_frequencies(episodes: list[Episode]) -> dict[str, float]:
    counts: dict[str, int] = {}
    total = 0
    for episode in episodes:
        for beat in episode.beats:
            counts[beat.kind] = counts.get(beat.kind, 0) + 1
            total += 1
    return {kind: count / total for kind, count in counts.items()}


def score_candidates(
    prior_episodes: list[Episode],
    current_episode: Episode,
    weights: SalienceWeights = SalienceWeights(),
    additive: bool = False,
    no_decay: bool = False,
    shuffle_surprise_seed: int | None = None,
) -> list[ScoredBeat]:
    frequencies = kind_frequencies(prior_episodes + [current_episode])
    open_threads = open_threads_before(prior_episodes + [current_episode], current_episode.index)
    feature_rows = [
        (beat, beat_features(beat, current_episode, open_threads, frequencies))
        for episode in prior_episodes
        for beat in episode.beats
    ]
    if shuffle_surprise_seed is not None:
        # §12.4 ablation 3: permute surprise across candidates.
        rng = random.Random(shuffle_surprise_seed)
        surprises = [features["surprise"] for _, features in feature_rows]
        rng.shuffle(surprises)
        for (_, features), shuffled in zip(feature_rows, surprises):
            features["surprise"] = shuffled
    scored = []
    for beat, features in feature_rows:
        score, rho, potential = salience_score(features, weights, additive=additive, no_decay=no_decay)
        scored.append(ScoredBeat(beat=beat, score=score, rho=rho, potential=potential, features=features))
    return scored


def build_recap(scored: list[ScoredBeat], token_budget: int) -> list[Beat]:
    """Budgeted greedy fill in descending salience (density-greedy relaxation)."""
    chosen: list[Beat] = []
    used = 0
    for item in sorted(scored, key=lambda s: -s.score):
        cost = len(item.beat.text.split())
        if used + cost > token_budget:
            continue
        chosen.append(item.beat)
        used += cost
    return sorted(chosen, key=lambda b: (b.episode, b.index))


def recap_policies(
    prior_episodes: list[Episode],
    current_episode: Episode,
    token_budget: int,
    seed: int = 0,
    weights: SalienceWeights = SalienceWeights(),
) -> dict[str, list[Beat]]:
    """Every policy in the paper's §12.4/§12.5 battery, same budget for all."""
    scored = score_candidates(prior_episodes, current_episode, weights)
    novelty_weights = SalienceWeights(wA=weights.wA, wR=0.0, wM=0.0, decay=weights.decay)
    policies: dict[str, list[Beat]] = {
        "salience": build_recap(scored, token_budget),
        "novelty_only": build_recap(
            score_candidates(prior_episodes, current_episode, novelty_weights), token_budget
        ),
        "additive_ablation": build_recap(
            score_candidates(prior_episodes, current_episode, weights, additive=True), token_budget
        ),
        "no_decay_ablation": build_recap(
            score_candidates(prior_episodes, current_episode, weights, no_decay=True), token_budget
        ),
        "shuffled_surprise": build_recap(
            score_candidates(prior_episodes, current_episode, weights, shuffle_surprise_seed=seed),
            token_budget,
        ),
    }
    rng = random.Random(seed)
    all_beats = [beat for episode in prior_episodes for beat in episode.beats]
    shuffled = all_beats[:]
    rng.shuffle(shuffled)
    uniform: list[Beat] = []
    used = 0
    for beat in shuffled:
        cost = len(beat.text.split())
        if used + cost <= token_budget:
            uniform.append(beat)
            used += cost
    policies["uniform"] = sorted(uniform, key=lambda b: (b.episode, b.index))

    recency: list[Beat] = []
    used = 0
    for beat in reversed(all_beats):
        cost = len(beat.text.split())
        if used + cost > token_budget:
            break
        recency.append(beat)
        used += cost
    policies["recency"] = sorted(recency, key=lambda b: (b.episode, b.index))
    return policies


def selection_f1(recap: list[Beat], required: list[tuple[int, int]]) -> float:
    """Precision/recall of the recap against the oracle load-bearing beats."""
    if not required:
        return 0.0
    chosen = {(beat.episode, beat.index) for beat in recap}
    oracle = set(required)
    true_positive = len(chosen & oracle)
    if true_positive == 0:
        return 0.0
    precision = true_positive / len(chosen)
    recall = true_positive / len(oracle)
    return 2 * precision * recall / (precision + recall)
