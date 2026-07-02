from __future__ import annotations

from bcv.discovery import Graph, exact_chromatic_number, greedy_degree_desc_coloring
from bcv.graph_adversary import attack_expression, load_library, record_find, AdversaryFind
from bcv.graph_agent import compile_feature_expression
from bcv.graph_generalize import build_observation_pool


def test_annealer_falsifies_known_false_class(tmp_path):
    # Bipartite connected max_degree>=3 dies to crown-style graphs; the annealer
    # should find some bipartite graph greedy miscolors without being told how.
    result = attack_expression(
        "is_bipartite and is_connected and max_degree >= 3",
        ns=(8,),
        restarts=8,
        steps=1500,
        seed=1,
        library_path=tmp_path / "library.jsonl",
    )
    assert result.falsified
    find = result.find
    graph = Graph(find.n, find.edges)
    predicate = compile_feature_expression(result.expression)
    from bcv.graph_adversary import observe_graph_cheap

    assert predicate(observe_graph_cheap(graph))
    assert greedy_degree_desc_coloring(graph) == find.greedy_colors
    assert exact_chromatic_number(graph) == find.chromatic_number
    assert find.greedy_colors > find.chromatic_number
    # The find must be persisted and deduplicated.
    assert len(load_library(tmp_path / "library.jsonl")) == 1
    record_find(find, tmp_path / "library.jsonl")
    assert len(load_library(tmp_path / "library.jsonl")) == 1


def test_annealer_respects_provably_true_class(tmp_path):
    # Stars: greedy is provably optimal, so the annealer must come up empty.
    result = attack_expression(
        "is_tree and has_universal_vertex",
        ns=(8,),
        restarts=3,
        steps=500,
        seed=2,
        library_path=tmp_path / "library.jsonl",
    )
    assert not result.falsified
    assert load_library(tmp_path / "library.jsonl") == ()


def test_graph_from_payload_validates():
    from bcv.graph_adversary import _graph_from_payload

    assert _graph_from_payload({"n": 6, "edges": [[0, 1], [1, 2]]}) is not None
    assert _graph_from_payload({"n": 6, "edges": [[0, 0]]}) is None
    assert _graph_from_payload({"n": 6, "edges": [[0, 6]]}) is None
    assert _graph_from_payload({"n": 2, "edges": []}) is None
    assert _graph_from_payload(["not", "a", "dict"]) is None


class _FakeClient:
    """Deterministic stand-in for a model: garbage, wrong class, then a real kill."""

    backend = "fake"
    model = "fake"

    def __init__(self):
        from bcv.graph_generalize import crown_graph_interleaved

        crown = crown_graph_interleaved(4)
        self._responses = [
            {"n": 8, "edges": [[0, 0]]},
            {"n": 8, "edges": [[0, 1], [1, 2], [2, 3]]},
            {"n": crown.n, "edges": [list(edge) for edge in crown.edges]},
        ]
        self.calls = 0

    def generate_json(self, prompt, temperature=0.0):
        response = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        return response


def test_attack_with_model_verifies_and_gives_feedback(tmp_path):
    from bcv.graph_adversary import attack_with_model

    result = attack_with_model(
        _FakeClient(),
        "is_bipartite and is_connected and max_degree >= 3",
        tries=5,
        library_path=tmp_path / "library.jsonl",
    )
    assert result.falsified
    assert result.find.method == "model"
    assert result.restarts_used == 3  # malformed, wrong class, then the kill
    assert result.find.greedy_colors > result.find.chromatic_number


def test_pool_loads_adversary_library(tmp_path):
    library = tmp_path / "library.jsonl"
    find = AdversaryFind(
        expression="is_bipartite",
        graph_id="test",
        n=8,
        edges=((0, 1), (2, 3)),
        chromatic_number=2,
        greedy_colors=2,
        method="anneal",
        found_on="2026-07-01",
    )
    record_find(find, library)
    with_library, _, _ = build_observation_pool(ns=(7,), samples_per_np=5, relabels=1, seed=3, library_path=library)
    without_library, _, _ = build_observation_pool(ns=(7,), samples_per_np=5, relabels=1, seed=3)
    assert len(with_library) == len(without_library) + 1
