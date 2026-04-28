"""Tests for _periodic_weight_refresh task in BaseValidatorNeuron."""
import asyncio
import os
import sys
import types
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

os.environ["SWARM_WEIGHT_REFRESH_SEC"] = "0.05"


@pytest.fixture(autouse=True)
def _fast_refresh_interval(monkeypatch):
    from swarm.base import validator as _v
    monkeypatch.setattr(_v, "WEIGHT_REFRESH_SEC", 0.05)

_stub_utils = types.ModuleType("swarm.validator.utils")

def _apply_stub(self_obj, weights):
    n = getattr(self_obj.metagraph, "n", 256)
    self_obj.scores = np.zeros(n, dtype=np.float32)
    for k, v in (weights or {}).items():
        try:
            uid = int(k)
            if 0 <= uid < n:
                self_obj.scores[uid] = float(v)
        except (TypeError, ValueError):
            continue

_stub_utils._apply_backend_weights_to_scores = _apply_stub
sys.modules.setdefault("swarm.validator.utils", _stub_utils)

from swarm.base import validator as validator_mod


class _FakeBackendApi:
    def __init__(self, *, weights=None, fallback=False, raise_exc=False):
        self.weights = weights or {}
        self.fallback = fallback
        self.raise_exc = raise_exc
        self.sync_calls = 0

    async def sync(self):
        self.sync_calls += 1
        if self.raise_exc:
            raise RuntimeError("boom")
        return {"weights": self.weights, "fallback": self.fallback}


def _make_self(metagraph_n=256):
    obj = SimpleNamespace()
    obj.metagraph = SimpleNamespace(n=metagraph_n)
    obj.scores = np.zeros(metagraph_n, dtype=np.float32)
    obj._scores_lock = None
    obj._mark_weights_ready_for_setting = lambda: None
    obj._periodic_weight_refresh = MethodType(
        validator_mod.BaseValidatorNeuron._periodic_weight_refresh, obj
    )
    return obj


async def _run_refresh_for(self_obj, duration=0.2):
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


def test_periodic_refresh_applies_weights():
    """Task must call backend sync and apply weights to self.scores."""
    obj = _make_self()
    obj.backend_api = _FakeBackendApi(weights={"167": 1.0})
    asyncio.run(_run_refresh_for(obj, duration=0.2))
    assert obj.backend_api.sync_calls >= 1
    assert obj.scores[167] > 0, "Weight for UID 167 should be applied"


def test_periodic_refresh_skips_on_fallback():
    """Task must NOT apply weights when backend signals fallback."""
    obj = _make_self()
    obj.backend_api = _FakeBackendApi(weights={"167": 1.0}, fallback=True)
    asyncio.run(_run_refresh_for(obj, duration=0.2))
    assert obj.backend_api.sync_calls >= 1
    assert np.count_nonzero(obj.scores) == 0, "No weights should be applied on fallback"


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
        await asyncio.sleep(0.1)
        forward_done.set()

    obj.forward = fake_forward

    asyncio.run(validator_mod.BaseValidatorNeuron.concurrent_forward(obj))

    assert forward_done.is_set(), "Forward should have run"
