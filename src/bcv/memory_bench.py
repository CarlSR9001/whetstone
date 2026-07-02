"""Four-arm memory benchmark: does salience paging beat a recency buffer?

Setup: episodes stream one at a time into an agent whose visible context is ONLY
the current episode — a season cannot fit. Long-term facts (who holds which item,
who feuds with whom) must survive in memory. After each episode, ground-truth QA
probes (derived by exact state replay of the generator's beats) are answered by an
EXTRACTIVE answerer over (current episode + injected memories): accuracy therefore
measures one thing only — whether the memory system surfaced the needed fact under
the token budget. No language model in the loop.

Arms (equal injection budget):
  none            current episode only
  recency         last-N beats buffer (the Letta-shaped baseline)
  salience        salience-paged store (recall-as-interrupt, Eq. 1 pager)
  consolidated    salience paging + episodic->semantic state-fold each episode
  oracle          exactly the beats the answer needs (upper bound)
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from bcv.memstore import MemoryStore, PagerWeights
from bcv.tinyseasons import Episode, load_corpus


@dataclass(frozen=True)
class Probe:
    question: str
    kind: str  # holder | location | feud
    subject: tuple[str, ...]
    answer: str
    oracle_text: str


def replay_state(episodes: list[Episode]) -> dict:
    holder: dict[str, tuple[str, str]] = {}  # item -> (holder, evidencing beat text)
    location: dict[str, tuple[str, str]] = {}
    feuds: dict[int, tuple[str, str, str]] = {}
    for episode in episodes:
        for beat in episode.beats:
            if beat.kind == "give":
                receiver = beat.text.split(" to ", 1)[1].split(" at ")[0].strip()
                item = next(e for e in beat.entities if e.startswith("the "))
                holder[item] = (receiver, beat.text)
            elif beat.kind == "travel":
                who = beat.entities[0]
                place = beat.entities[1]
                location[who] = (place, beat.text)
            elif beat.kind == "feud_start":
                feuds[beat.thread_id] = (beat.entities[0], beat.entities[1], beat.text)
            elif beat.kind == "payoff_feud" and beat.thread_id in feuds:
                del feuds[beat.thread_id]
    return {"holder": holder, "location": location, "feuds": feuds}


def probes_for(episodes_so_far: list[Episode]) -> list[Probe]:
    state = replay_state(episodes_so_far)
    probes: list[Probe] = []
    for item, (who, evidence) in sorted(state["holder"].items()):
        probes.append(
            Probe(
                question=f"Who currently holds {item}?",
                kind="holder",
                subject=(item,),
                answer=who,
                oracle_text=evidence,
            )
        )
    for who, (place, evidence) in sorted(state["location"].items()):
        probes.append(
            Probe(
                question=f"Where is {who}?",
                kind="location",
                subject=(who,),
                answer=place,
                oracle_text=evidence,
            )
        )
    return probes


def extractive_answer(probe: Probe, visible_text: list[str]) -> str | None:
    """Answer strictly from visible text, latest evidence wins. STATE facts from
    consolidation are trusted directly."""
    answer = None
    for text in visible_text:
        if probe.kind == "holder":
            item = probe.subject[0]
            state_match = re.search(rf"STATE: (\w+) currently holds {re.escape(item)}\.", text)
            if state_match:
                answer = state_match.group(1)
            transfer_match = re.search(rf"hands {re.escape(item)} to (\w+)", text)
            if transfer_match:
                answer = transfer_match.group(1)
        elif probe.kind == "location":
            who = probe.subject[0]
            move_match = re.search(rf"{re.escape(who)} travels to (the \w+)", text)
            if move_match:
                answer = move_match.group(1)
    return answer


def run_benchmark(
    corpus_path: str | Path,
    token_budget: int = 90,
    max_seasons: int | None = None,
    root: str | Path = ".bcv_runs/memory_bench",
) -> dict:
    episodes = load_corpus(corpus_path)
    by_season: dict[int, list[Episode]] = defaultdict(list)
    for episode in episodes:
        by_season[episode.season].append(episode)
    seasons = sorted(by_season)
    if max_seasons is not None:
        seasons = seasons[:max_seasons]

    arms = ("none", "recency", "salience", "consolidated", "oracle")
    correct: dict[str, int] = {arm: 0 for arm in arms}
    injected_tokens: dict[str, int] = {arm: 0 for arm in arms}
    total_probes = 0
    consolidation_facts = 0

    for season_index in seasons:
        season = sorted(by_season[season_index], key=lambda e: e.index)
        salience_store = MemoryStore()
        consolidated_store = MemoryStore()
        recency_buffer: list[str] = []
        step = 0

        for position, episode in enumerate(season):
            for beat in episode.beats:
                step += 1
                if beat.kind == "color":
                    continue  # write gate: pure color beats carry no state
                for store in (salience_store, consolidated_store):
                    store.remember(beat.text, beat.entities, step)
                recency_buffer.append(beat.text)
            consolidation_facts += consolidated_store.consolidate_state_facts(step)

            if position == 0:
                continue
            current_text = [beat.text for beat in episode.beats]
            current_entities = tuple({entity for beat in episode.beats for entity in beat.entities})
            for probe in probes_for(season[: position + 1]):
                total_probes += 1
                # Arm: none
                visible = list(current_text)
                if extractive_answer(probe, visible) == probe.answer:
                    correct["none"] += 1
                # Arm: recency (last beats under budget)
                buffer: list[str] = []
                used = 0
                for text in reversed(recency_buffer):
                    cost = len(text.split())
                    if used + cost > token_budget:
                        break
                    buffer.append(text)
                    used += cost
                injected_tokens["recency"] += used
                if extractive_answer(probe, list(reversed(buffer)) + current_text) == probe.answer:
                    correct["recency"] += 1
                # Arms: salience / consolidated (query-blind: paged on episode context)
                for arm, store in (("salience", salience_store), ("consolidated", consolidated_store)):
                    paged = store.page_in(
                        current_entities + probe.subject,
                        step,
                        token_budget,
                        PagerWeights(),
                    )
                    injected_tokens[arm] += sum(len(m.content.split()) for m in paged)
                    if extractive_answer(probe, [m.content for m in paged] + current_text) == probe.answer:
                        correct[arm] += 1
                # Arm: oracle
                injected_tokens["oracle"] += len(probe.oracle_text.split())
                if extractive_answer(probe, [probe.oracle_text] + current_text) == probe.answer:
                    correct["oracle"] += 1

    report = {
        "probes": total_probes,
        "token_budget": token_budget,
        "seasons": len(seasons),
        "consolidation_facts": consolidation_facts,
        "accuracy": {arm: round(correct[arm] / total_probes, 4) for arm in arms},
        "mean_injected_tokens": {
            arm: round(injected_tokens[arm] / total_probes, 1) for arm in arms
        },
    }
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Four-arm salience-memory benchmark.")
    parser.add_argument("--corpus", default=".bcv_runs/tinyseasons/corpus.jsonl")
    parser.add_argument("--token-budget", type=int, default=90)
    parser.add_argument("--max-seasons", type=int, default=None)
    parser.add_argument("--root", default=".bcv_runs/memory_bench")
    args = parser.parse_args()
    print(
        json.dumps(
            run_benchmark(args.corpus, args.token_budget, args.max_seasons, args.root),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
