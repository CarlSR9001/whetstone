"""A salience-paged long-term memory: SQLite below, the Eq. 1 pager above.

The design goal is recall-as-INTERRUPT, not recall-as-query. Letta-style memory is
a tool the model must remember to call; here the harness runs the pager every step:
all live memories are candidates, the salience controller scores them against the
current context, and the top set under a token budget is PUSHED into the working
context. Memory influences decisions by showing up, the way human memories do.

Schema follows the blueprint's memory-hygiene MVP: every row carries kind
(episodic | semantic | derived), source, confidence, lineage (parent_ids), and
reinforcement telemetry (last_used_step, use_count). Nothing is silently mutated:
consolidation and pondering write NEW rows with parents and retire the old ones.

Pager λ defaults to the F1 side of the retention-vs-recency Pareto front measured
in the TinySeasons experiment: memory is a facts-consumer, and aging out old
open-thread facts is exactly the failure mode. Age here is time since last
REINFORCEMENT (the paper's Δt), so frequently re-used old memories stay warm.

Consolidation is an episodic→semantic state-fold: many transfer events for the same
item collapse into one current-state fact that no single episodic memory contains —
the honest, checkable version of "pondering an old memory yields a new idea."
"""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Memory:
    id: int
    content: str
    kind: str
    source: str
    confidence: float
    entities: tuple[str, ...]
    created_step: int
    last_used_step: int
    use_count: int
    parent_ids: tuple[int, ...]


@dataclass(frozen=True)
class PagerWeights:
    wA: float = 0.5
    wR: float = 2.0
    wM: float = 2.0
    decay: float = 0.02  # facts-consumer: retention side of the Pareto front
    epsilon: float = 0.05
    theta_store: float = 0.0  # write gate; 0 stores everything except duplicates


class MemoryStore:
    def __init__(self, path: str | Path = ":memory:") -> None:
        self.connection = sqlite3.connect(str(path))
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY,
                content TEXT NOT NULL,
                kind TEXT NOT NULL,
                source TEXT NOT NULL,
                confidence REAL NOT NULL,
                entities TEXT NOT NULL,
                created_step INTEGER NOT NULL,
                last_used_step INTEGER NOT NULL,
                use_count INTEGER NOT NULL DEFAULT 0,
                parent_ids TEXT NOT NULL DEFAULT '[]',
                retired INTEGER NOT NULL DEFAULT 0
            )"""
        )
        self.connection.commit()

    # ------------------------------------------------------------------ write

    def remember(
        self,
        content: str,
        entities: tuple[str, ...],
        step: int,
        kind: str = "episodic",
        source: str = "observation",
        confidence: float = 0.8,
        parent_ids: tuple[int, ...] = (),
    ) -> int | None:
        """Gated write; exact-duplicate contents are reinforced, not re-stored."""
        row = self.connection.execute(
            "SELECT id, use_count FROM memories WHERE content = ? AND retired = 0", (content,)
        ).fetchone()
        if row is not None:
            self.connection.execute(
                "UPDATE memories SET last_used_step = ?, use_count = use_count + 1 WHERE id = ?",
                (step, row[0]),
            )
            self.connection.commit()
            return row[0]
        cursor = self.connection.execute(
            "INSERT INTO memories (content, kind, source, confidence, entities, created_step,"
            " last_used_step, use_count, parent_ids) VALUES (?,?,?,?,?,?,?,0,?)",
            (content, kind, source, confidence, json.dumps(sorted(entities)), step, step, json.dumps(list(parent_ids))),
        )
        self.connection.commit()
        return cursor.lastrowid

    def retire(self, memory_id: int) -> None:
        self.connection.execute("UPDATE memories SET retired = 1 WHERE id = ?", (memory_id,))
        self.connection.commit()

    # ------------------------------------------------------------------ read

    def live_memories(self) -> list[Memory]:
        rows = self.connection.execute(
            "SELECT id, content, kind, source, confidence, entities, created_step,"
            " last_used_step, use_count, parent_ids FROM memories WHERE retired = 0"
        ).fetchall()
        return [
            Memory(
                id=r[0],
                content=r[1],
                kind=r[2],
                source=r[3],
                confidence=r[4],
                entities=tuple(json.loads(r[5])),
                created_step=r[6],
                last_used_step=r[7],
                use_count=r[8],
                parent_ids=tuple(json.loads(r[9])),
            )
            for r in rows
        ]

    def counts(self) -> dict[str, int]:
        rows = self.connection.execute(
            "SELECT kind, COUNT(*) FROM memories WHERE retired = 0 GROUP BY kind"
        ).fetchall()
        return {kind: count for kind, count in rows}

    # ------------------------------------------------------------------ pager

    def page_in(
        self,
        context_entities: tuple[str, ...],
        step: int,
        token_budget: int,
        weights: PagerWeights = PagerWeights(),
        reinforce: bool = True,
    ) -> list[Memory]:
        """Recall-as-interrupt: score every live memory against the context and
        push the top set under budget. Paged-in memories are reinforced (their
        Δt resets), so useful old facts stay warm instead of aging out."""
        context = set(context_entities)
        entity_rarity = self._entity_rarity()
        scored: list[tuple[float, Memory]] = []
        for memory in self.live_memories():
            entity_set = set(memory.entities)
            overlap = len(entity_set & context)
            momentum = overlap / max(1, len(entity_set))
            surprise = sum(entity_rarity.get(entity, 1.0) for entity in memory.entities) / max(
                1, len(memory.entities)
            )
            retention = memory.confidence * (1.5 if memory.kind in ("semantic", "derived") else 1.0)
            rho = weights.wA * surprise + weights.wR * retention + weights.wM * momentum
            age = max(0, step - memory.last_used_step)
            distance = 1.0 - overlap / max(1, len(entity_set | context))
            effort = max(1.0, len(memory.content.split()) / 8.0)
            score = rho * math.exp(-weights.decay * age) / math.sqrt(distance * effort + weights.epsilon)
            scored.append((score, memory))
        chosen: list[Memory] = []
        used = 0
        for score, memory in sorted(scored, key=lambda pair: -pair[0]):
            cost = len(memory.content.split())
            if used + cost > token_budget:
                continue
            chosen.append(memory)
            used += cost
        if reinforce and chosen:
            self.connection.executemany(
                "UPDATE memories SET last_used_step = ?, use_count = use_count + 1 WHERE id = ?",
                [(step, memory.id) for memory in chosen],
            )
            self.connection.commit()
        return sorted(chosen, key=lambda m: (m.created_step, m.id))

    def _entity_rarity(self) -> dict[str, float]:
        counts: dict[str, int] = {}
        total = 0
        for memory in self.live_memories():
            for entity in memory.entities:
                counts[entity] = counts.get(entity, 0) + 1
                total += 1
        if not total:
            return {}
        return {entity: -math.log(count / total) for entity, count in counts.items()}

    # --------------------------------------------------------- consolidation

    def consolidate_state_facts(self, step: int, patterns: tuple[str, ...] = ("hands",)) -> int:
        """Episodic -> semantic state-fold: for each item transferred in episodic
        'X hands ITEM to Y' memories, derive ONE current-holder fact from the
        latest transfer, with lineage to every folded episodic row. The derived
        fact exists in no single memory — it requires ordering the chain."""
        transfers: dict[str, list[Memory]] = {}
        for memory in self.live_memories():
            if memory.kind != "episodic":
                continue
            if not any(pattern in memory.content for pattern in patterns):
                continue
            item = next((e for e in memory.entities if e.startswith("the ")), None)
            if item is None:
                continue
            transfers.setdefault(item, []).append(memory)
        derived = 0
        for item, chain in transfers.items():
            if len(chain) < 2:
                continue
            chain.sort(key=lambda m: (m.created_step, m.id))
            latest = chain[-1]
            holder = _receiver_from_transfer(latest.content)
            if holder is None:
                continue
            content = f"STATE: {holder} currently holds {item}."
            existing = self.connection.execute(
                "SELECT id, content FROM memories WHERE kind = 'semantic' AND retired = 0 AND content LIKE ?",
                (f"STATE: %holds {item}.%",),
            ).fetchall()
            for old_id, old_content in existing:
                if old_content != content:
                    self.retire(old_id)
            self.remember(
                content,
                entities=(holder, item),
                step=step,
                kind="semantic",
                source="consolidation",
                confidence=0.9,
                parent_ids=tuple(m.id for m in chain),
            )
            derived += 1
        return derived

    # ----------------------------------------------------------------- goals

    def set_goal(self, content: str, entities: tuple[str, ...], step: int) -> int | None:
        """A standing Q: the thing relevance is computed AGAINST. Without an active
        goal, relevance does not exist and attention degrades to salience."""
        return self.remember(content, entities, step, kind="goal", source="agent", confidence=1.0)

    def active_goals(self) -> list[Memory]:
        return [memory for memory in self.live_memories() if memory.kind == "goal"]

    def goal_entities(self) -> tuple[str, ...]:
        entities: list[str] = []
        for goal in self.active_goals():
            entities.extend(goal.entities)
        return tuple(dict.fromkeys(entities))

    # ---------------------------------------------------------------- ponder

    def ponder(self, step: int, client=None, max_pairs: int = 3) -> list[int]:
        """Offline recombination: sample old, high-value memories that share an
        entity, and derive a connection. With a model the connection is generated;
        without, a rule-based template keeps the loop testable. Derived memories
        enter at LOW confidence with full lineage — they must earn promotion by
        being useful (reinforcement), per the drift lesson: ungated self-generated
        memories are how a store fills with slop."""
        memories = [m for m in self.live_memories() if m.kind == "episodic"]
        by_entity: dict[str, list[Memory]] = {}
        for memory in memories:
            for entity in memory.entities:
                by_entity.setdefault(entity, []).append(memory)
        created: list[int] = []
        for entity, group in sorted(by_entity.items(), key=lambda kv: -len(kv[1])):
            if len(created) >= max_pairs:
                break
            if len(group) < 3:
                continue
            group.sort(key=lambda m: m.created_step)
            oldest, newest = group[0], group[-1]
            if client is not None:
                prompt = (
                    f"Old memory: {oldest.content}\nRecent memory: {newest.content}\n"
                    f"In one short sentence, state a non-obvious connection or hypothesis about {entity}."
                )
                try:
                    insight = client.generate_text(prompt, temperature=0.7).strip().split("\n")[0][:200]
                except Exception:
                    continue
            else:
                insight = (
                    f"PATTERN: {entity} recurs across {len(group)} events "
                    f"from step {oldest.created_step} to {newest.created_step}."
                )
            memory_id = self.remember(
                insight,
                entities=(entity,),
                step=step,
                kind="derived",
                source="ponder",
                confidence=0.4,
                parent_ids=(oldest.id, newest.id),
            )
            if memory_id is not None:
                created.append(memory_id)
        return created


def _receiver_from_transfer(content: str) -> str | None:
    # "GIVER hands ITEM to RECEIVER at PLACE."
    marker = " to "
    if marker not in content:
        return None
    tail = content.split(marker, 1)[1]
    receiver = tail.split(" at ")[0].strip().rstrip(".")
    return receiver or None
