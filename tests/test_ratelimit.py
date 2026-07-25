from __future__ import annotations

import bcv.ratelimit as ratelimit


def test_sliding_window_limit_is_strictly_key_bounded(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(ratelimit.time, "monotonic", lambda: now[0])
    limit = ratelimit.SlidingWindowLimit(requests=1, seconds=10, max_keys=3)

    assert limit.allow("a")
    assert limit.allow("b")
    assert limit.allow("c")
    assert not limit.allow("d")
    assert len(limit.events) == 3

    # Once the old windows expire, a full prune admits the new identity and
    # drops every stale queue instead of retaining them forever.
    now[0] = 11.0
    assert limit.allow("d")
    assert set(limit.events) == {"d"}


def test_capacity_refusal_does_not_reset_an_active_quota(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(ratelimit.time, "monotonic", lambda: now[0])
    limit = ratelimit.SlidingWindowLimit(requests=2, seconds=60, max_keys=2)

    assert limit.allow("a")
    assert limit.allow("a")
    assert not limit.allow("a")
    assert limit.allow("b")
    assert not limit.allow("c")
    assert set(limit.events) == {"a", "b"}
