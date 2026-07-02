from __future__ import annotations

from bcv.taste_adversary import run_taste_adversary_benchmark


def test_taste_adversary_prefers_grounded_strong_outputs(tmp_path):
    result = run_taste_adversary_benchmark(tmp_path)

    assert result.strong_beats_generic == result.cases
    assert result.strong_beats_weird == result.cases

