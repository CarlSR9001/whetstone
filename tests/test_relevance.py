from __future__ import annotations

from bcv.memstore import MemoryStore
from bcv.memory_bench import Probe
from bcv.relevance import (
    counterfactual_relevance,
    evaluate,
    page_by_relevance,
    relevance_score,
    salience_prior,
)
from bcv.tinyseasons import generate_corpus


def _probe():
    return Probe(
        question="Who currently holds the ledger?",
        kind="holder",
        subject=("the ledger",),
        answer="Rook",
        oracle_text="Liv hands the ledger to Rook at the harbor.",
    )


def test_relevance_separates_boring_footnote_from_shiny_trap():
    store = MemoryStore()
    boring = store.remember("Liv hands the ledger to Rook at the harbor.", ("Liv", "Rook", "the ledger"), step=1)
    store.remember("A comet blazes over the foundry as Mara duels Ash.", ("Mara", "Ash", "the foundry"), step=2)
    memories = {m.id: m for m in store.live_memories()}
    rarity = store._entity_rarity()
    context = {"Mara", "Ash", "the foundry"}  # the shiny event dominates context
    probe = _probe()

    footnote, trap = memories[boring], [m for m in memories.values() if m.id != boring][0]
    # Salience prefers the shiny in-context event; relevance inverts that.
    assert salience_prior(trap, context, rarity) > salience_prior(footnote, context, rarity)
    assert relevance_score(footnote, probe, context, rarity) > relevance_score(trap, probe, context, rarity)
    # Ground truth agrees: only the footnote changes decision quality.
    assert counterfactual_relevance(probe, footnote, []) == 1
    assert counterfactual_relevance(probe, trap, []) == 0


def test_page_by_relevance_budget_and_two_stage():
    store = MemoryStore()
    store.remember("Liv hands the ledger to Rook at the harbor.", ("Liv", "Rook", "the ledger"), step=1)
    for filler in range(30):
        store.remember(f"Ash shares a quiet moment number {filler} at the chapel.", ("Ash", "the chapel"), step=2)
    probe = _probe()
    for two_stage in (False, True):
        paged = page_by_relevance(store, probe, ("Ash", "the chapel"), step=40, token_budget=25, two_stage=two_stage)
        assert sum(len(m.content.split()) for m in paged) <= 25
        assert any("ledger" in m.content for m in paged)


def test_relevance_beats_salience_on_corpus(tmp_path):
    corpus = generate_corpus(seasons=4, episodes=8, seed=31, root=tmp_path)
    report = evaluate(corpus, token_budget=90, root=tmp_path / "eval")
    accuracy = report["accuracy"]
    assert accuracy["relevance"] > accuracy["salience"]
    validation = report["estimator_validation"]["top1_finds_truly_relevant"]
    assert validation["relevance_est"] > validation["salience"]
