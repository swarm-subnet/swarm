"""Validator-side absolute-allocation burn: scores get the raw share directly,
UID0 takes the remainder, and an over-allocation is clamped to the full pool."""
import numpy as np
import pytest
from types import SimpleNamespace

from swarm.constants import UID_ZERO
from swarm.validator.utils_parts.weights import (
    _apply_backend_weights_to_scores_unlocked as apply_weights,
)


def _self(n=16):
    return SimpleNamespace(metagraph=SimpleNamespace(n=n), scores=np.zeros(n, dtype=np.float32))


def _sum(self):
    return float(self.scores.sum())


def test_absolute_weights_applied_without_keep_scaling():
    s = _self()
    apply_weights(s, {"5": 0.10, "7": 0.10})
    assert s.scores[5] == pytest.approx(0.10)
    assert s.scores[7] == pytest.approx(0.10)
    assert s.scores[UID_ZERO] == pytest.approx(0.80)
    assert _sum(s) == pytest.approx(1.0)


def test_empty_map_full_burn():
    s = _self()
    apply_weights(s, {})
    assert s.scores[UID_ZERO] == pytest.approx(1.0)
    assert np.count_nonzero(s.scores[1:]) == 0


def test_uid0_entry_is_ignored_and_burns():
    s = _self()
    apply_weights(s, {"0": 0.5, "5": 0.10})
    assert s.scores[5] == pytest.approx(0.10)
    assert s.scores[UID_ZERO] == pytest.approx(0.90)
    assert _sum(s) == pytest.approx(1.0)


def test_out_of_range_and_nonfinite_burn():
    s = _self(n=10)
    apply_weights(s, {"999": 0.4, "-3": 0.2, "5": 0.10, "6": float("nan")})
    assert s.scores[5] == pytest.approx(0.10)
    assert s.scores[UID_ZERO] == pytest.approx(0.90)
    assert _sum(s) == pytest.approx(1.0)


def test_overallocation_is_clamped_to_full_pool():
    s = _self()
    apply_weights(s, {"5": 0.7, "7": 0.7})
    assert s.scores[5] == pytest.approx(0.5)
    assert s.scores[7] == pytest.approx(0.5)
    assert s.scores[UID_ZERO] == pytest.approx(0.0)
    assert _sum(s) == pytest.approx(1.0)
