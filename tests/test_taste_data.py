from __future__ import annotations

from bcv.taste_data import build_scaled_taste_dataset


def test_scaled_taste_dataset_prefers_strong_variants(tmp_path):
    result = build_scaled_taste_dataset(tmp_path, variants_per_prompt=4)

    assert result["preference_examples"] == result["prompt_specs"] * 4
    assert result["sft_examples"] == result["preference_examples"]
    assert result["chosen_strong"] / result["preference_examples"] >= 0.9
    assert result["avg_strong_minus_weak_margin"] > 0
