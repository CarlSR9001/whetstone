from __future__ import annotations

from bcv.novelty import NoveltyJudge


def test_novelty_judge_semantics():
    judge = NoveltyJudge(max_n=5)

    # A mined-style repair is enumerated.
    mined = judge.judge("(is_tree) and (max_degree == 3)", base_expression="is_tree")
    assert mined.parseable
    assert mined.enumerated_one_atom
    assert not mined.semantically_novel

    # A string paraphrase with the same extension is still enumerated:
    # novelty is judged on match-sets, not syntax.
    paraphrase = judge.judge("is_tree and not max_degree_le_2", base_expression="is_tree")
    assert paraphrase.enumerated_two_atom
    assert not paraphrase.semantically_novel

    # A disjunction of equalities has no <=2-atom conjunctive equivalent.
    novel = judge.judge("is_tree and (n == 3 or n == 5)", base_expression="is_tree")
    assert novel.parseable
    assert novel.support > 0
    assert novel.semantically_novel

    # Garbage is flagged unparseable, never novel.
    garbage = judge.judge("is_tree.__class__", base_expression="is_tree")
    assert not garbage.parseable
    assert not garbage.semantically_novel


def test_free_hull_judgement():
    judge = NoveltyJudge(max_n=5)
    free_atom = judge.judge("is_complete")
    assert free_atom.enumerated_one_atom
    conj = judge.judge("is_connected and is_bipartite")
    assert conj.enumerated_two_atom
