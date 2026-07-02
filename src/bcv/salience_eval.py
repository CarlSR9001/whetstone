"""Fixed-budget evaluation of the salience controller — the paper's §12.5, runnable.

Every policy gets the SAME recap token budget per episode transition. Two metrics:

1. Selection F1 (CPU, exact): does the policy pick the beats the oracle says the
   next episode actually depends on? This is available only because TinySeasons has
   ground truth by construction; it is the sharp version of the paper's claim.
2. Delta-L (Eq. 16, GPU, prefill-only): mean next-episode NLL under a frozen LM
   conditioned on each policy's recap. Salience earns its keep iff its NLL beats
   uniform/novelty/recency at equal budget. Short bursts only — no decode.

Falsifiers, per the paper: salience <= uniform on F1 or Delta-L kills the
controller; salience ~= shuffled_surprise kills the surprise term; salience ~=
additive_ablation kills the multiplicative form; salience ~= no_decay kills decay.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from bcv.salience import recap_policies, selection_f1
from bcv.tinyseasons import Episode, episode_text, load_corpus


def evaluate_selection(
    episodes: list[Episode],
    token_budget: int = 90,
    seed: int = 0,
) -> dict:
    """CPU metric: per-policy F1 against oracle load-bearing beats."""
    by_season: dict[int, list[Episode]] = defaultdict(list)
    for episode in episodes:
        by_season[episode.season].append(episode)
    f1_sums: dict[str, float] = defaultdict(float)
    transitions = 0
    for season_episodes in by_season.values():
        season_episodes.sort(key=lambda e: e.index)
        for position in range(1, len(season_episodes)):
            current = season_episodes[position]
            if not current.required_prior_beats:
                continue
            prior = season_episodes[:position]
            policies = recap_policies(prior, current, token_budget, seed=seed)
            for name, recap in policies.items():
                f1_sums[name] += selection_f1(recap, current.required_prior_beats)
            transitions += 1
    return {
        "transitions": transitions,
        "token_budget": token_budget,
        "selection_f1": {name: round(total / transitions, 4) for name, total in sorted(f1_sums.items())},
    }


def evaluate_nll(
    episodes: list[Episode],
    token_budget: int = 90,
    max_transitions: int = 24,
    model_name: str = "Qwen/Qwen3-1.7B",
    seed: int = 0,
) -> dict:
    """Eq. 16: next-episode NLL conditioned on each policy's recap, fixed budget."""
    from bcv.transformers_client import TransformersLocalClient

    by_season: dict[int, list[Episode]] = defaultdict(list)
    for episode in episodes:
        by_season[episode.season].append(episode)
    client = TransformersLocalClient(model_name=model_name)
    nll_sums: dict[str, float] = defaultdict(float)
    scored = 0
    try:
        for season_index in sorted(by_season):
            season_episodes = sorted(by_season[season_index], key=lambda e: e.index)
            for position in range(2, len(season_episodes)):
                if scored >= max_transitions:
                    break
                current = season_episodes[position]
                if not current.required_prior_beats:
                    continue
                prior = season_episodes[:position]
                target = episode_text(current)
                policies = recap_policies(prior, current, token_budget, seed=seed)
                for name, recap in policies.items():
                    recap_text = " ".join(beat.text for beat in recap)
                    prefix = f"Previously: {recap_text}\nNext episode: "
                    nll_sums[name] += client.score_nll(prefix, target)
                scored += 1
            if scored >= max_transitions:
                break
    finally:
        client.unload()
    return {
        "transitions_scored": scored,
        "token_budget": token_budget,
        "model": model_name,
        "mean_nll": {name: round(total / scored, 4) for name, total in sorted(nll_sums.items())} if scored else {},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed-budget salience evaluation (paper §12.5).")
    parser.add_argument("--corpus", default=".bcv_runs/tinyseasons/corpus.jsonl")
    parser.add_argument("--token-budget", type=int, default=90)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nll", action="store_true", help="Also run the GPU Eq. 16 NLL comparison.")
    parser.add_argument("--max-transitions", type=int, default=24)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--root", default=".bcv_runs/salience_eval")
    args = parser.parse_args()

    episodes = load_corpus(args.corpus)
    report: dict = {"selection": evaluate_selection(episodes, args.token_budget, args.seed)}
    if args.nll:
        report["nll"] = evaluate_nll(
            episodes,
            token_budget=args.token_budget,
            max_transitions=args.max_transitions,
            model_name=args.model,
            seed=args.seed,
        )
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
