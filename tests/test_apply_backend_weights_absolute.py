"""Validator-side weight application. With burning off (the default) miners keep
their raw share and UID0 stays at zero; set_weights later renormalizes the raw
shares to 1.0. The parked BURN_FRACTION lever, when enabled, scales miners down
to KEEP_FRACTION and routes BURN_FRACTION to UID0."""
import numpy as np
import pytest
from types import SimpleNamespace

from bittensor.utils.weight_utils import process_weights

from swarm.constants import UID_ZERO
from swarm.validator.utils_parts import weights as wmod
from swarm.validator.utils_parts.weights import (
    _apply_backend_weights_to_scores_unlocked as apply_weights,
)


def _self(n=16):
    return SimpleNamespace(metagraph=SimpleNamespace(n=n), scores=np.zeros(n, dtype=np.float32))


def _sum(self):
    return float(self.scores.sum())


def test_no_burn_leaves_miner_scores_and_zero_uid0():
    s = _self()
    apply_weights(s, {"5": 0.10, "7": 0.10})
    assert s.scores[5] == pytest.approx(0.10)
    assert s.scores[7] == pytest.approx(0.10)
    # Nothing on UID0; set_weights renormalizes the raw shares to 1.0 downstream.
    assert s.scores[UID_ZERO] == pytest.approx(0.0)
    assert _sum(s) == pytest.approx(0.20)


def test_empty_map_leaves_all_zero():
    s = _self()
    apply_weights(s, {})
    # All-zero -> set_weights refuses and holds the last good weights (no burn).
    assert np.count_nonzero(s.scores) == 0


def test_uid0_entry_is_ignored():
    s = _self()
    apply_weights(s, {"0": 0.5, "5": 0.10})
    assert s.scores[5] == pytest.approx(0.10)
    assert s.scores[UID_ZERO] == pytest.approx(0.0)


def test_out_of_range_and_nonfinite_skipped():
    s = _self(n=10)
    apply_weights(s, {"999": 0.4, "-3": 0.2, "5": 0.10, "6": float("nan")})
    assert s.scores[5] == pytest.approx(0.10)
    assert s.scores[UID_ZERO] == pytest.approx(0.0)


def test_overallocation_is_clamped_to_full_pool():
    s = _self()
    apply_weights(s, {"5": 0.7, "7": 0.7})
    assert s.scores[5] == pytest.approx(0.5)
    assert s.scores[7] == pytest.approx(0.5)
    assert s.scores[UID_ZERO] == pytest.approx(0.0)
    assert _sum(s) == pytest.approx(1.0)


def test_parked_lever_routes_fraction_to_uid0(monkeypatch):
    monkeypatch.setattr(wmod, "BURN_EMISSIONS", True)
    monkeypatch.setattr(wmod, "BURN_FRACTION", 0.75)
    monkeypatch.setattr(wmod, "KEEP_FRACTION", 0.25)
    s = _self()
    apply_weights(s, {"5": 0.10, "7": 0.10})
    # Miners scaled to KEEP_FRACTION (0.25) by their raw share; UID0 = BURN_FRACTION.
    assert s.scores[5] == pytest.approx(0.125)
    assert s.scores[7] == pytest.approx(0.125)
    assert s.scores[UID_ZERO] == pytest.approx(0.75)
    assert _sum(s) == pytest.approx(1.0)


def test_sparse_one_champion_survives_chain_processing_no_uid0_burn():
    # Single-champion vector through Bittensor's processor (min=1, max=1.0): champion 1.0, UID0 0.
    n = 256
    uids = np.arange(n)
    weights = np.zeros(n, dtype=np.float32)
    weights[50] = 1.0
    out_uids, out_weights = process_weights(
        uids=uids, weights=weights, num_neurons=n,
        min_allowed_weights=1, max_weight_limit=1.0,
    )
    out = {int(u): float(w) for u, w in zip(out_uids, out_weights)}
    assert out.get(50, 0.0) == pytest.approx(1.0)
    assert out.get(UID_ZERO, 0.0) == pytest.approx(0.0)
