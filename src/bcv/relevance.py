"""Relevance as its own math, separate from salience.

Salience is an attention-prior: what grabs the system, computable without knowing
the question (novelty, intensity, recency, goal-free structure). Relevance is
objective-conditioned counterfactual usefulness:

    R(x | Q) = E[ Delta DecisionQuality | x ]

— would the answer be worse without x? On TinySeasons this is EXACTLY computable:
the extractive answerer is the decision procedure, so counterfactual relevance is
correctness-with-x minus correctness-without-x. Ground truth, not vibes. That lets
us do something the salience paper couldn't: validate the *estimator* against the
true quantity, and measure how badly salience proxies for relevance (the
"high-salience / low-relevance trap" and the "boring footnote" quadrant).

The estimator implements the Forte-style form with operational proxies:

    Rel*(x|Q) = w_voi*VoI + w_causal*CausalReach + w_repair*ModelRepair
                - w_cost*AttentionCost - rho*Salience(x)*(1 - VoI)

with the last term the adversarial salience-hijack penalty: the shinier something
is while not touching the question, the more dangerous it is.

Architecturally: salience proposes, relevance disposes — the attention-level
instance of the propose-cheap / verify-exact pattern (spec decoding, the refinery,
the emulator's CHECK). The two-stage pager makes that literal: a cheap salience
prefilter nominates candidates, the query-conditioned relevance score spends the
budget.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from bcv.memstore import Memory, MemoryStore, PagerWeights
from bcv.memory_bench import Probe, extractive_answer, probes_for
from bcv.tinyseasons import Episode, load_corpus


STATE_VERBS = ("hands", "travels", "STATE:")


@dataclass(frozen=True)
class RelevanceWeights:
    w_voi: float = 3.0
    w_causal: float = 2.0
    w_repair: float = 2.0
    w_cost: float = 0.3
    rho: float = 1.0  # salience-hijack penalty


def salience_prior(memory: Memory, context_entities: set[str], rarity: dict[str, float]) -> float:
    """Query-free attention prior: rarity + context overlap (what 'grabs')."""
    overlap = len(set(memory.entities) & context_entities) / max(1, len(memory.entities))
    surprise = sum(rarity.get(entity, 1.0) for entity in memory.entities) / max(1, len(memory.entities))
    return surprise + 2.0 * overlap


def relevance_score(
    memory: Memory,
    probe: Probe,
    context_entities: set[str],
    rarity: dict[str, float],
    weights: RelevanceWeights = RelevanceWeights(),
) -> float:
    query_entities = set(probe.subject)
    memory_entities = set(memory.entities)
    voi = len(memory_entities & query_entities) / max(1, len(query_entities))
    # CausalReach: does x touch the state machinery of THIS question kind?
    if probe.kind == "holder":
        causal = 1.0 if ("hands" in memory.content or "STATE:" in memory.content) and voi > 0 else 0.0
    elif probe.kind == "location":
        causal = 1.0 if "travels" in memory.content and voi > 0 else 0.0
    else:
        causal = 1.0 if any(verb in memory.content for verb in STATE_VERBS) and voi > 0 else 0.0
    # ModelRepair: consolidated state facts patch the world model directly.
    repair = 1.0 if memory.kind in ("semantic", "derived") and voi > 0 else 0.0
    cost = len(memory.content.split()) / 8.0
    hijack = salience_prior(memory, context_entities, rarity) * (1.0 - voi)
    return (
        weights.w_voi * voi
        + weights.w_causal * causal
        + weights.w_repair * repair
        - weights.w_cost * cost
        - weights.rho * hijack
    )


def counterfactual_relevance(probe: Probe, memory: Memory, base_visible: list[str]) -> int:
    """The real quantity: Delta decision quality from adding x. Exact, by construction."""
    without = extractive_answer(probe, base_visible)
    with_x = extractive_answer(probe, base_visible + [memory.content])
    return int(with_x == probe.answer) - int(without == probe.answer)


def page_by_relevance(
    store: MemoryStore,
    probe: Probe,
    context_entities: tuple[str, ...],
    step: int,
    token_budget: int,
    two_stage: bool = False,
    prefilter_factor: int = 4,
) -> list[Memory]:
    rarity = store._entity_rarity()
    context = set(context_entities)
    candidates = store.live_memories()
    if two_stage:
        # Salience proposes: cheap query-free prefilter down to a shortlist...
        candidates.sort(key=lambda m: -salience_prior(m, context, rarity))
        shortlist_budget = token_budget * prefilter_factor
        shortlist: list[Memory] = []
        used = 0
        for memory in candidates:
            cost = len(memory.content.split())
            if used + cost > shortlist_budget:
                continue
            shortlist.append(memory)
            used += cost
        candidates = shortlist
    # ...relevance disposes: query-conditioned ranking spends the real budget.
    scored = sorted(
        candidates,
        key=lambda m: -relevance_score(m, probe, context, rarity),
    )
    chosen: list[Memory] = []
    used = 0
    for memory in scored:
        cost = len(memory.content.split())
        if used + cost > token_budget:
            continue
        chosen.append(memory)
        used += cost
    return sorted(chosen, key=lambda m: (m.created_step, m.id))


def evaluate(
    corpus_path: str | Path,
    token_budget: int = 90,
    max_seasons: int | None = None,
    root: str | Path = ".bcv_runs/relevance_eval",
) -> dict:
    """Relevance arms vs the salience pager, plus estimator validation:
    rank agreement of salience and relevance-estimate against TRUE counterfactual
    relevance, and the two danger quadrants counted."""
    episodes = load_corpus(corpus_path)
    by_season: dict[int, list[Episode]] = defaultdict(list)
    for episode in episodes:
        by_season[episode.season].append(episode)
    seasons = sorted(by_season)
    if max_seasons is not None:
        seasons = seasons[:max_seasons]

    arms = ("salience", "relevance", "two_stage")
    correct = {arm: 0 for arm in arms}
    total_probes = 0
    # Estimator validation accumulators.
    top1_hits = {"salience": 0, "relevance_est": 0}
    validation_queries = 0
    quadrants = {"high_sal_low_rel": 0, "low_sal_high_rel": 0, "aligned": 0}

    for season_index in seasons:
        season = sorted(by_season[season_index], key=lambda e: e.index)
        store = MemoryStore()
        step = 0
        for position, episode in enumerate(season):
            for beat in episode.beats:
                step += 1
                if beat.kind == "color":
                    continue
                store.remember(beat.text, beat.entities, step)
            store.consolidate_state_facts(step)
            if position == 0:
                continue
            current_text = [beat.text for beat in episode.beats]
            current_entities = tuple({e for beat in episode.beats for e in beat.entities})
            rarity = store._entity_rarity()
            context = set(current_entities)
            for probe in probes_for(season[: position + 1]):
                total_probes += 1
                paged_sal = store.page_in(
                    current_entities + probe.subject, step, token_budget, PagerWeights(), reinforce=False
                )
                if extractive_answer(probe, [m.content for m in paged_sal] + current_text) == probe.answer:
                    correct["salience"] += 1
                for arm, two_stage in (("relevance", False), ("two_stage", True)):
                    paged = page_by_relevance(
                        store, probe, current_entities, step, token_budget, two_stage=two_stage
                    )
                    if extractive_answer(probe, [m.content for m in paged] + current_text) == probe.answer:
                        correct[arm] += 1
                # Estimator validation on a subsample (exact counterfactuals are O(memories)).
                if total_probes % 7 == 0:
                    memories = store.live_memories()
                    if len(memories) < 5:
                        continue
                    true_rel = {
                        m.id: counterfactual_relevance(probe, m, current_text) for m in memories
                    }
                    if not any(value > 0 for value in true_rel.values()):
                        continue
                    validation_queries += 1
                    best_sal = max(memories, key=lambda m: salience_prior(m, context, rarity))
                    best_rel = max(memories, key=lambda m: relevance_score(m, probe, context, rarity))
                    top1_hits["salience"] += int(true_rel[best_sal.id] > 0)
                    top1_hits["relevance_est"] += int(true_rel[best_rel.id] > 0)
                    sal_values = sorted(memories, key=lambda m: -salience_prior(m, context, rarity))
                    median_rank = len(sal_values) // 2
                    for rank, memory in enumerate(sal_values):
                        genuinely_relevant = true_rel[memory.id] > 0
                        shiny = rank < median_rank
                        if shiny and not genuinely_relevant:
                            quadrants["high_sal_low_rel"] += 1
                        elif not shiny and genuinely_relevant:
                            quadrants["low_sal_high_rel"] += 1
                        elif shiny and genuinely_relevant:
                            quadrants["aligned"] += 1

    report = {
        "probes": total_probes,
        "token_budget": token_budget,
        "accuracy": {arm: round(correct[arm] / total_probes, 4) for arm in arms},
        "estimator_validation": {
            "queries": validation_queries,
            "top1_finds_truly_relevant": {
                name: round(hits / max(1, validation_queries), 4) for name, hits in top1_hits.items()
            },
        },
        "quadrants": quadrants,
    }
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Relevance-vs-salience evaluation.")
    parser.add_argument("--corpus", default=".bcv_runs/tinyseasons/corpus.jsonl")
    parser.add_argument("--token-budget", type=int, default=90)
    parser.add_argument("--max-seasons", type=int, default=None)
    parser.add_argument("--root", default=".bcv_runs/relevance_eval")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.corpus, args.token_budget, args.max_seasons, args.root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
