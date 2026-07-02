from __future__ import annotations

from bcv.lora_smoke import LoraSmokeResult


def test_lora_smoke_result_shape():
    result = LoraSmokeResult(
        accepted=False,
        base_model="base",
        adapter_path="adapter",
        loss=None,
        trainable_parameters=None,
        total_parameters=None,
        device="cpu",
        failure="not run",
    )

    assert result.accepted is False
    assert result.failure == "not run"

