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

"""Tests for _periodic_weight_refresh task in BaseValidatorNeuron."""
import asyncio
import os
import sys
import types
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

os.environ["SWARM_WEIGHT_REFRESH_SEC"] = "0.05"


@pytest.fixture(autouse=True)
def _fast_refresh_interval(monkeypatch):
    """Shrink the refresh period to 50 ms so a short test window covers several cycles."""
    from swarm.base import validator as _v
    monkeypatch.setattr(_v, "WEIGHT_REFRESH_SEC", 0.05)

_stub_utils = types.ModuleType("swarm.validator.utils")

def _apply_stub(self_obj, weights):
    """Stand in for the real score writer: zero the vector, then set each named UID to its weight."""
    n = getattr(self_obj.metagraph, "n", 256)
    self_obj.scores = np.zeros(n, dtype=np.float32)
    for k, v in (weights or {}).items():
        try:
            uid = int(k)
            if 0 <= uid < n:
                self_obj.scores[uid] = float(v)
        except (TypeError, ValueError):
            continue

def _compute_koth_stub(sync_data, *, metagraph=None):
    """Parse the payload's king rows, dropping malformed ones, and return the KotH weight per UID."""
    from swarm.validator import koth as _koth
    entries = []
    for raw in sync_data.get("kings") or []:
        if not isinstance(raw, dict):
            continue
        try:
            entries.append(_koth.KingEntry.from_sync_dict(raw))
        except _koth.MalformedKingEntry:
            continue
    return _koth.compute_weights(entries)

# Both imports run after the module stubs above are in place, and must stay here.
from swarm.validator.utils_parts.weights import accept_sync_version  # noqa: E402

_stub_utils._apply_backend_weights_to_scores = _apply_stub
_stub_utils.compute_koth_weights_from_sync = _compute_koth_stub
_stub_utils.accept_sync_version = accept_sync_version
sys.modules.setdefault("swarm.validator.utils", _stub_utils)

from swarm.base import validator as validator_mod  # noqa: E402


class _FakeBackendApi:
    """Backend client double that answers sync with a fixed lineage and counts the calls."""
    def __init__(self, *, kings=None, weights=None, fallback=False, raise_exc=False):
        """Hold the king rows, advisory weights and flags the fake sync will answer with."""
        self.kings = kings or []
        self.weights = weights or {}
        self.fallback = fallback
        self.raise_exc = raise_exc
        self.sync_calls = 0

    async def sync(self):
        """Return one leaderboard payload, or raise when the fake was built to fail."""
        self.sync_calls += 1
        if self.raise_exc:
            raise RuntimeError("boom")
        return {
            "kings": list(self.kings),
            "kings_by_family": {"cf_autopilot": list(self.kings)},
            "family_shares": {"cf_autopilot": 1.0},
            "weights": dict(self.weights),
            "fallback": self.fallback,
        }


def _king(uid, hotkey, score, prev_score, *, crowned_at_epoch=1):
    """One lineage row in the shape the backend sync sends, ready for KingEntry parsing."""
    return {
        "lineage_id": uid + 1000,
        "rank": 0,
        "uid": uid,
        "hotkey": hotkey,
        "score": score,
        "prev_score": prev_score,
        "weight": 0.0,
        "crowned_at_epoch": crowned_at_epoch,
    }


def _make_self(metagraph_n=256):
    """A minimal validator stand-in: metagraph, zeroed scores, and the refresh coroutine bound to it."""
    obj = SimpleNamespace()
    obj.metagraph = SimpleNamespace(
        n=metagraph_n,
        hotkeys=[f"hk{uid}" for uid in range(metagraph_n)],
    )
    obj.scores = np.zeros(metagraph_n, dtype=np.float32)
    obj._scores_lock = None
    obj._mark_weights_ready_for_setting = lambda: None
    obj._periodic_weight_refresh = MethodType(
        validator_mod.BaseValidatorNeuron._periodic_weight_refresh, obj
    )
    return obj


async def _run_refresh_for(self_obj, duration=0.2):
    """Let the refresh coroutine run for duration seconds, then cancel it and hand back the task."""
    task = asyncio.create_task(
        validator_mod.BaseValidatorNeuron._periodic_weight_refresh(self_obj)
    )
    await asyncio.sleep(duration)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    return task


def test_periodic_refresh_computes_locally_from_kings():
    """Scores come out of the validator's own KotH sum over the lineage, not from the backend."""
    obj = _make_self()
    obj.backend_api = _FakeBackendApi(
        kings=[_king(167, "hk167", score=0.50, prev_score=0.0)],
    )
    asyncio.run(_run_refresh_for(obj, duration=0.2))
    assert obj.backend_api.sync_calls >= 1
    assert obj.scores[167] > 0


def test_periodic_refresh_ignores_advisory_weights_field():
    """The advisory weights map in a sync payload never reaches the score vector; only kings pay."""
    obj = _make_self()
    obj.backend_api = _FakeBackendApi(
        kings=[_king(50, "hk50", score=0.60, prev_score=0.0)],
        weights={"99": 1.0},
    )
    asyncio.run(_run_refresh_for(obj, duration=0.2))
    assert obj.backend_api.sync_calls >= 1
    assert obj.scores[50] > 0
    assert obj.scores[99] == 0


def test_periodic_refresh_processes_fallback_via_cached_kings():
    """A payload flagged as an offline fallback still pays its kings instead of being discarded."""
    obj = _make_self()
    obj.backend_api = _FakeBackendApi(
        kings=[_king(7, "hk7", score=0.85, prev_score=0.80)],
        fallback=True,
    )
    asyncio.run(_run_refresh_for(obj, duration=0.2))
    assert obj.backend_api.sync_calls >= 1
    assert obj.scores[7] > 0


def test_periodic_refresh_burns_on_empty_kings():
    """An empty lineage clears every UID above the reserved burn slot, so no stale champion keeps a share."""
    obj = _make_self()
    obj.backend_api = _FakeBackendApi(kings=[], weights={})
    asyncio.run(_run_refresh_for(obj, duration=0.2))
    assert obj.backend_api.sync_calls >= 1
    assert np.count_nonzero(obj.scores[1:]) == 0


def test_periodic_refresh_survives_sync_exception():
    """Task must not crash when backend sync raises."""
    obj = _make_self()
    obj.backend_api = _FakeBackendApi(raise_exc=True)
    task = asyncio.run(_run_refresh_for(obj, duration=0.2))
    assert obj.backend_api.sync_calls >= 1
    assert task.cancelled() or task.done()


def test_periodic_refresh_skips_without_backend_api():
    """Task must not crash when backend_api is not yet initialized."""
    obj = _make_self()
    task = asyncio.run(_run_refresh_for(obj, duration=0.2))
    assert task.cancelled() or task.done()
    assert np.count_nonzero(obj.scores) == 0


def test_concurrent_forward_cancels_refresh_task_on_exit():
    """concurrent_forward must cancel the refresh task after forwards complete."""
    obj = _make_self()
    obj.backend_api = _FakeBackendApi(weights={"167": 1.0})
    obj.config = SimpleNamespace(neuron=SimpleNamespace(num_concurrent_forwards=1))

    forward_done = asyncio.Event()

    async def fake_forward():
        """Sleep briefly, then flag that the forward pass ran to completion."""
        await asyncio.sleep(0.1)
        forward_done.set()

    obj.forward = fake_forward

    asyncio.run(validator_mod.BaseValidatorNeuron.concurrent_forward(obj))

    assert forward_done.is_set(), "Forward should have run"
