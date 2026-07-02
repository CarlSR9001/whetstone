"""TinySeasons: a serialized story corpus with ground truth by construction.

Episodes are sequences of BEATS drawn from a template grammar over recurring
characters, items, and feuds. Long-horizon THREADS follow the heat -> comeback ->
payoff arc (the booking policy of the salience paper's case study): a setup beat
plants a grievance or an item, reinforcement beats accumulate "heat", and a payoff
beat cashes the thread out episodes later. Because the generator tracks world state
exactly, every episode transition comes with ORACLE labels: which prior beats are
load-bearing for the next episode (their facts are referenced or required by it).

That oracle is what makes the corpus a falsification instrument for the salience
controller (Eq. 1 of the paper): a selection policy claims certain beats deserve the
recap budget; the oracle says which beats actually mattered; a frozen language model
says (via next-episode NLL) whether the selected working set predicts the future
better than baselines at the same budget — the paper's Eq. 16, made runnable.

Per-episode RATINGS are computed, not vibes: payoffs of long-held threads score
high (retention cashed out), contradictions score low (continuity violated), plus
seeded noise — an ecological reward signal in miniature.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path


CHARACTERS = ("Mara", "Okonkwo", "Liv", "Petra", "Dain", "Sylvie", "Ash", "Rook")
ITEMS = ("the brass key", "the ledger", "the compass", "the sealed letter", "the lantern")
PLACES = ("the mill", "the harbor", "the archive", "the foundry", "the chapel")


@dataclass(frozen=True)
class Beat:
    episode: int
    index: int
    kind: str  # give | travel | feud_start | feud_heat | payoff_item | payoff_feud | color
    text: str
    entities: tuple[str, ...]
    thread_id: int | None


@dataclass
class Episode:
    season: int
    index: int
    beats: list[Beat]
    rating: float
    # Oracle: global beat ids (episode, index) whose facts this episode depends on.
    required_prior_beats: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class Season:
    index: int
    episodes: list[Episode]


def generate_season(season_index: int, episodes: int = 8, beats_per_episode: int = 12, seed: int = 0) -> Season:
    rng = random.Random(seed * 7919 + season_index)
    holder: dict[str, str] = {item: rng.choice(CHARACTERS) for item in ITEMS}
    location: dict[str, str] = {who: rng.choice(PLACES) for who in CHARACTERS}
    feuds: dict[int, tuple[str, str]] = {}
    open_item_threads: dict[int, str] = {}  # thread -> item planted for a later payoff
    thread_setup_beat: dict[int, tuple[int, int]] = {}
    thread_heat: dict[int, int] = {}
    next_thread = 0

    season = Season(index=season_index, episodes=[])
    for episode_index in range(episodes):
        beats: list[Beat] = []
        required: set[tuple[int, int]] = set()
        payoff_scores: list[float] = []

        def add(kind: str, text: str, entities: tuple[str, ...], thread: int | None) -> None:
            beats.append(
                Beat(
                    episode=episode_index,
                    index=len(beats),
                    kind=kind,
                    text=text,
                    entities=entities,
                    thread_id=thread,
                )
            )

        for _ in range(beats_per_episode):
            roll = rng.random()
            if roll < 0.18:
                item = rng.choice(ITEMS)
                giver = holder[item]
                receiver = rng.choice([c for c in CHARACTERS if c != giver])
                thread = next_thread
                next_thread += 1
                open_item_threads[thread] = item
                thread_setup_beat[thread] = (episode_index, len(beats))
                thread_heat[thread] = 0
                holder[item] = receiver
                add("give", f"{giver} hands {item} to {receiver} at {location[giver]}.", (giver, receiver, item), thread)
            elif roll < 0.30:
                who = rng.choice(CHARACTERS)
                place = rng.choice([p for p in PLACES if p != location[who]])
                location[who] = place
                add("travel", f"{who} travels to {place}.", (who, place), None)
            elif roll < 0.42:
                a, b = rng.sample(CHARACTERS, 2)
                thread = next_thread
                next_thread += 1
                feuds[thread] = (a, b)
                thread_setup_beat[thread] = (episode_index, len(beats))
                thread_heat[thread] = 0
                add("feud_start", f"{a} accuses {b} of betrayal; a grudge takes root.", (a, b), thread)
            elif roll < 0.58 and feuds:
                thread = rng.choice(sorted(feuds))
                a, b = feuds[thread]
                thread_heat[thread] += 1
                required.add(thread_setup_beat[thread])
                add("feud_heat", f"{a} and {b} clash again; the old accusation resurfaces.", (a, b), thread)
            elif roll < 0.70 and open_item_threads:
                thread = rng.choice(sorted(open_item_threads))
                item = open_item_threads.pop(thread)
                who = holder[item]
                heat = episode_index - thread_setup_beat[thread][0]
                required.add(thread_setup_beat[thread])
                payoff_scores.append(1.0 + 0.5 * heat)
                add(
                    "payoff_item",
                    f"{who} finally uses {item} to open the vault beneath {location[who]}.",
                    (who, item),
                    thread,
                )
            elif roll < 0.80 and feuds:
                thread = rng.choice(sorted(feuds))
                a, b = feuds.pop(thread)
                heat = thread_heat.pop(thread, 0)
                required.add(thread_setup_beat[thread])
                payoff_scores.append(1.0 + 0.4 * heat)
                add(
                    "payoff_feud",
                    f"{a} confronts {b} at {location[a]}; the grudge is settled at last.",
                    (a, b),
                    thread,
                )
            else:
                who = rng.choice(CHARACTERS)
                add("color", f"{who} shares a quiet moment at {location[who]}.", (who,), None)

        rating = 5.0 + 1.5 * sum(payoff_scores) - 0.3 * beats_per_episode / 12 + rng.gauss(0, 0.4)
        episode = Episode(
            season=season_index,
            index=episode_index,
            beats=beats,
            rating=round(min(10.0, max(1.0, rating)), 2),
            required_prior_beats=sorted(b for b in required if b[0] < episode_index),
        )
        season.episodes.append(episode)
    return season


def episode_text(episode: Episode) -> str:
    return " ".join(beat.text for beat in episode.beats)


def generate_corpus(
    seasons: int = 20,
    episodes: int = 8,
    beats_per_episode: int = 12,
    seed: int = 0,
    root: str | Path = ".bcv_runs/tinyseasons",
) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "corpus.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for season_index in range(seasons):
            season = generate_season(season_index, episodes, beats_per_episode, seed)
            for episode in season.episodes:
                handle.write(json.dumps(asdict(episode), sort_keys=True) + "\n")
    return path


def load_corpus(path: str | Path) -> list[Episode]:
    episodes: list[Episode] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        raw["beats"] = [Beat(**{**b, "entities": tuple(b["entities"])}) for b in raw["beats"]]
        raw["required_prior_beats"] = [tuple(b) for b in raw["required_prior_beats"]]
        episodes.append(Episode(**raw))
    return episodes


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the TinySeasons corpus.")
    parser.add_argument("--seasons", type=int, default=20)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--beats", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--root", default=".bcv_runs/tinyseasons")
    args = parser.parse_args()
    path = generate_corpus(args.seasons, args.episodes, args.beats, args.seed, args.root)
    episodes = load_corpus(path)
    ratings = [e.rating for e in episodes]
    with_oracle = sum(1 for e in episodes if e.required_prior_beats)
    print(
        json.dumps(
            {
                "path": str(path),
                "episodes": len(episodes),
                "mean_rating": round(sum(ratings) / len(ratings), 2),
                "episodes_with_cross_episode_dependencies": with_oracle,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
