from __future__ import annotations

from bcv.memstore import MemoryStore, PagerWeights
from bcv.memory_bench import extractive_answer, run_benchmark, Probe
from bcv.tinyseasons import generate_corpus


def test_write_gate_dedupes_and_reinforces():
    store = MemoryStore()
    first = store.remember("Mara hands the brass key to Liv at the mill.", ("Mara", "Liv", "the brass key"), step=1)
    second = store.remember("Mara hands the brass key to Liv at the mill.", ("Mara", "Liv", "the brass key"), step=5)
    assert first == second
    memory = store.live_memories()[0]
    assert memory.use_count == 1
    assert memory.last_used_step == 5


def test_pager_pushes_relevant_memories_and_reinforces():
    store = MemoryStore()
    store.remember("Mara hands the brass key to Liv at the mill.", ("Mara", "Liv", "the brass key"), step=1)
    store.remember("Rook travels to the chapel.", ("Rook", "the chapel"), step=2)
    for filler in range(20):
        store.remember(f"Ash shares a quiet moment number {filler}.", ("Ash",), step=3)
    paged = store.page_in(("Liv", "the brass key"), step=50, token_budget=20)
    contents = [m.content for m in paged]
    assert any("brass key" in c for c in contents)
    assert all("quiet moment" not in c for c in contents[:1])
    # Reinforcement: the paged-in memory's age reset, so it stays warm at step 90.
    again = store.page_in(("Liv", "the brass key"), step=90, token_budget=20)
    assert any("brass key" in m.content for m in again)


def test_consolidation_folds_transfer_chain_to_current_holder():
    store = MemoryStore()
    store.remember("Mara hands the ledger to Liv at the mill.", ("Mara", "Liv", "the ledger"), step=1)
    store.remember("Liv hands the ledger to Rook at the harbor.", ("Liv", "Rook", "the ledger"), step=2)
    derived = store.consolidate_state_facts(step=3)
    assert derived == 1
    semantic = [m for m in store.live_memories() if m.kind == "semantic"]
    assert len(semantic) == 1
    assert semantic[0].content == "STATE: Rook currently holds the ledger."
    assert len(semantic[0].parent_ids) == 2
    # A later transfer supersedes: old STATE fact retires, new one appears.
    store.remember("Rook hands the ledger to Ash at the chapel.", ("Rook", "Ash", "the ledger"), step=4)
    store.consolidate_state_facts(step=5)
    states = [m for m in store.live_memories() if m.kind == "semantic"]
    assert len(states) == 1
    assert "Ash currently holds the ledger" in states[0].content


def test_ponder_creates_low_confidence_derived_memories_with_lineage():
    store = MemoryStore()
    for step in range(1, 5):
        store.remember(f"Mara clashes with Rook, round {step}.", ("Mara", "Rook"), step=step)
    created = store.ponder(step=10)
    assert created
    derived = [m for m in store.live_memories() if m.kind == "derived"]
    assert derived[0].confidence < 0.5
    assert len(derived[0].parent_ids) == 2


def test_extractive_answer_latest_evidence_wins():
    probe = Probe(
        question="Who currently holds the compass?",
        kind="holder",
        subject=("the compass",),
        answer="Petra",
        oracle_text="Dain hands the compass to Petra at the foundry.",
    )
    visible = [
        "Sylvie hands the compass to Dain at the archive.",
        "Dain hands the compass to Petra at the foundry.",
    ]
    assert extractive_answer(probe, visible) == "Petra"
    assert extractive_answer(probe, ["STATE: Petra currently holds the compass."]) == "Petra"


def test_benchmark_arms_order(tmp_path):
    corpus = generate_corpus(seasons=5, episodes=8, seed=21, root=tmp_path)
    report = run_benchmark(corpus, token_budget=90, root=tmp_path / "bench")
    accuracy = report["accuracy"]
    assert report["probes"] > 50
    assert accuracy["oracle"] > 0.95
    # The core claims: paging beats the recency buffer, consolidation beats raw paging.
    assert accuracy["salience"] > accuracy["recency"]
    assert accuracy["consolidated"] >= accuracy["salience"]
    assert accuracy["none"] < accuracy["salience"]
