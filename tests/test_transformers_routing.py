from __future__ import annotations

from bcv.transformers_client import RoutedAdapterCandidate


def test_routed_adapter_policy_is_explicit_and_domain_specific():
    assert RoutedAdapterCandidate.uses_adapter(
        "Repair a rejected conjecture. Claim: example"
    )
    assert not RoutedAdapterCandidate.uses_adapter(
        "Write a Python function that returns the requested result."
    )
    assert RoutedAdapterCandidate.routing_policy == "repair_prompt_adapter_else_base"
