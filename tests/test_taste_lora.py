from __future__ import annotations

from bcv.taste_lora import heldout_taste_prompts


def test_heldout_taste_prompts_are_not_empty():
    prompts = heldout_taste_prompts()

    assert len(prompts) >= 3
    assert all("slop_references" in item for item in prompts)
    assert all(len(item["slop_references"]) >= 2 for item in prompts)
