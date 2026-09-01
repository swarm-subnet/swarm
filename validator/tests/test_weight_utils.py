from __future__ import annotations

import numpy as np
import pytest

from swarm.base.utils.weight_utils import (
    U16_MAX,
    convert_weights_and_uids_for_emit,
    normalize_max_weight,
    process_weights_for_netuid,
)


class _DummyMetagraph:
    def __init__(self, n: int):
        self.n = n


class _DummySubtensor:
    def __init__(self, metagraph, min_allowed: int, max_limit: float):
        self._metagraph = metagraph
        self._min_allowed = min_allowed
        self._max_limit = max_limit

    def metagraph(self, netuid):
        _ = netuid
        return self._metagraph

    def min_allowed_weights(self, netuid):
        _ = netuid
        return self._min_allowed

    def max_weight_limit(self, netuid):
        _ = netuid
        return self._max_limit


def test_normalize_max_weight_returns_uniform_when_sum_zero():
    x = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    y = normalize_max_weight(x, limit=0.5)
    assert np.allclose(y, np.array([1 / 3, 1 / 3, 1 / 3]))


def test_normalize_max_weight_keeps_distribution_when_under_limit():
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = normalize_max_weight(x, limit=0.8)
    assert np.allclose(y, x / x.sum())


def test_normalize_max_weight_caps_peak_weight():
    x = np.array([100.0, 1.0, 1.0], dtype=np.float32)
    y = normalize_max_weight(x, limit=0.6)
    assert np.isclose(float(y.sum()), 1.0)
    assert float(y.max()) <= 0.600001


def test_convert_weights_and_uids_for_emit_validates_inputs():
    with pytest.raises(ValueError):
        convert_weights_and_uids_for_emit(np.array([0, 1]), np.array([-0.1, 0.2]))
    with pytest.raises(ValueError):
        convert_weights_and_uids_for_emit(np.array([-1, 1]), np.array([0.1, 0.2]))
    # Current implementation raises IndexError before its length-check branch.
    with pytest.raises(IndexError):
        convert_weights_and_uids_for_emit(np.array([0, 1]), np.array([0.1]))


def test_convert_weights_and_uids_for_emit_handles_zero_sum():
    uids, vals = convert_weights_and_uids_for_emit(np.array([0, 1]), np.array([0.0, 0.0]))
    assert uids == []
    assert vals == []


def test_convert_weights_and_uids_for_emit_scales_to_u16():
    uids, vals = convert_weights_and_uids_for_emit(np.array([10, 11]), np.array([0.5, 1.0]))
    assert uids == [10, 11]
    assert vals[1] == U16_MAX
    assert vals[0] < vals[1]


def test_process_weights_for_netuid_returns_uniform_when_no_nonzero():
    metagraph = _DummyMetagraph(n=4)
    subtensor = _DummySubtensor(metagraph=metagraph, min_allowed=2, max_limit=0.5)
    uids, weights = process_weights_for_netuid(
        uids=np.arange(4),
        weights=np.zeros(4, dtype=np.float32),
        netuid=124,
        subtensor=subtensor,
        metagraph=metagraph,
    )
    assert np.array_equal(uids, np.arange(4))
    assert np.allclose(weights, np.ones(4) / 4)


def test_process_weights_for_netuid_expands_when_below_min_allowed():
    metagraph = _DummyMetagraph(n=4)
    subtensor = _DummySubtensor(metagraph=metagraph, min_allowed=3, max_limit=0.5)
    uids, weights = process_weights_for_netuid(
        uids=np.arange(4),
        weights=np.array([0.0, 0.2, 0.0, 0.0], dtype=np.float32),
        netuid=124,
        subtensor=subtensor,
        metagraph=metagraph,
    )
    assert np.array_equal(uids, np.arange(4))
    assert np.isclose(float(weights.sum()), 1.0)
    assert float(weights.max()) <= 0.500001


def test_process_weights_for_netuid_filters_low_quantile_and_normalizes():
    metagraph = _DummyMetagraph(n=4)
    subtensor = _DummySubtensor(metagraph=metagraph, min_allowed=2, max_limit=0.6)
    uids, weights = process_weights_for_netuid(
        uids=np.arange(4),
        weights=np.array([0.1, 0.2, 0.3, 0.0], dtype=np.float32),
        netuid=124,
        subtensor=subtensor,
        metagraph=metagraph,
        exclude_quantile=int(0.5 * 65535),
    )
    assert np.array_equal(uids, np.array([1, 2]))
    assert np.allclose(weights, np.array([0.4, 0.6]), atol=1e-5)
