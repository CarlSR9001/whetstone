from __future__ import annotations

from bcv.taste import TasteContext, export_seed_taste_datasets, prefer_pair, score_taste


def test_taste_prefers_specific_memorable_answer_over_generic(tmp_path):
    context = TasteContext(
        prompt="Why do people like superhero movies?",
        domain="explanation",
        audience="technical generalist",
        mode="explanation",
    )
    weak = "People like superhero movies because they are fun and exciting."
    strong = "People like superhero movies because they are emotionally low-risk machines: the movie tells you who to root for, when to feel awe, and what victory means."
    slop = "People like superhero movies because they are fun, exciting, and have action."

    preference = prefer_pair(context, weak, strong, slop)

    assert preference.chosen == strong
    assert preference.chosen_score > preference.rejected_score


def test_taste_penalizes_generic_ai_smell():
    context = TasteContext(prompt="Explain generic AI prose.", domain="explanation")
    generic = "It is important to note that generic AI prose is nuanced and multifaceted overall."
    slop = "Generic AI prose is generic and not specific."

    scores = score_taste(context, generic, slop)

    assert scores.ai_smell_penalty >= 3
    assert scores.main_failure_mode.startswith("high_")


def test_taste_dataset_export_writes_sft_and_preferences(tmp_path):
    result = export_seed_taste_datasets(tmp_path)

    assert result["preference_examples"] >= 3
    assert result["sft_examples"] >= 3
    assert result["avg_margin"] > 0

