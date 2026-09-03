# The MIT License (MIT)
# Copyright © 2026 Swarm

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from __future__ import annotations

from swarm.utils import misc


def test_ttl_hash_gen_advances_when_window_changes(monkeypatch):
    now = {"t": 100.0}
    monkeypatch.setattr(misc.time, "time", lambda: now["t"])
    gen = misc._ttl_hash_gen(2)

    assert next(gen) == 0
    now["t"] = 101.9
    assert next(gen) == 0
    now["t"] = 102.0
    assert next(gen) == 1


def test_ttl_cache_returns_cached_value_before_expiry(monkeypatch):
    now = {"t": 50.0}
    monkeypatch.setattr(misc.time, "time", lambda: now["t"])
    calls = {"count": 0}

    @misc.ttl_cache(ttl=5)
    def fn(x):
        calls["count"] += 1
        return x + calls["count"]

    assert fn(10) == 11
    assert fn(10) == 11
    assert calls["count"] == 1

    now["t"] = 56.0
    assert fn(10) == 12
    assert calls["count"] == 2


def test_ttl_cache_typed_distinguishes_argument_types(monkeypatch):
    now = {"t": 10.0}
    monkeypatch.setattr(misc.time, "time", lambda: now["t"])
    calls = {"count": 0}

    @misc.ttl_cache(ttl=100, typed=True)
    def fn(x):
        calls["count"] += 1
        return calls["count"]

    assert fn(1) == 1
    assert fn(1.0) == 2
    assert calls["count"] == 2


def test_ttl_get_block_uses_cache_for_repeated_calls():
    class _Subtensor:
        def __init__(self):
            self.calls = 0

        def get_current_block(self):
            self.calls += 1
            return self.calls

    class _Obj:
        def __init__(self):
            self.subtensor = _Subtensor()

    obj = _Obj()
    first = misc.ttl_get_block(obj)
    second = misc.ttl_get_block(obj)

    assert first == 1
    assert second == 1
    assert obj.subtensor.calls == 1
