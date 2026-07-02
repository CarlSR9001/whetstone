from __future__ import annotations

from bcv.tripwire import synthetic_document_examples, train_document_tripwire


def test_synthetic_tripwire_examples_have_both_classes():
    examples = synthetic_document_examples()
    labels = {item.accepted for item in examples}

    assert labels == {True, False}


def test_document_tripwire_trains_to_high_accuracy(tmp_path):
    result = train_document_tripwire(tmp_path, epochs=80)

    assert result.examples == 320
    assert result.accuracy >= 0.98
    assert result.final_loss < 0.2

