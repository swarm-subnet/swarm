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

import numpy as np
import pytest

from swarm.core.action import canonicalize_action

LOW = np.array([-1.0, -1.0, -1.0, 0.0, -1.0], dtype=np.float32)
HIGH = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)


def test_action_is_clipped_to_the_family_bounds():
    action = canonicalize_action([9.0, -9.0, 0.0, 5.0, -5.0], LOW, HIGH, act_dim=5)
    assert np.array_equal(action, np.array([1.0, -1.0, 0.0, 1.0, -1.0], dtype=np.float32))


def test_non_finite_values_become_zero():
    action = canonicalize_action(
        [np.nan, np.inf, -np.inf, 0.5, 0.0], LOW, HIGH, act_dim=5
    )
    assert np.all(np.isfinite(action))
    assert action[0] == 0.0 and action[1] == 0.0 and action[2] == 0.0


def test_wrong_length_falls_back_to_a_zero_action():
    assert np.array_equal(
        canonicalize_action([1.0, 2.0], LOW, HIGH, act_dim=5),
        np.zeros(5, dtype=np.float32),
    )


@pytest.mark.parametrize("dtype", ["S8", "U2", "V8"])
def test_non_numeric_dtype_never_raises(dtype):
    """A miner controls the action dtype; a raise here would escape the per-act
    guard and be charged as ENV_FAILURE, which is excluded from every score mean."""
    raw = np.frombuffer(b"\x00" * 40, dtype=dtype)
    action = canonicalize_action(raw, LOW, HIGH, act_dim=5)
    assert np.array_equal(action, np.zeros(5, dtype=np.float32))


def test_object_dtype_never_raises():
    action = canonicalize_action(np.array([None] * 5, dtype=object), LOW, HIGH, act_dim=5)
    assert np.array_equal(action, np.zeros(5, dtype=np.float32))


def test_swarm_action_keeps_the_per_drone_shape():
    low = np.tile(LOW, (3, 1))
    high = np.tile(HIGH, (3, 1))
    action = canonicalize_action(
        np.arange(15, dtype=np.float32) / 15.0, low, high, n_drones=3, act_dim=5
    )
    assert action.shape == (3, 5)
    assert action.dtype == np.float32


def test_swarm_wrong_length_falls_back_to_zeros():
    low = np.tile(LOW, (3, 1))
    high = np.tile(HIGH, (3, 1))
    action = canonicalize_action([1.0, 2.0, 3.0], low, high, n_drones=3, act_dim=5)
    assert action.shape == (3, 5)
    assert not action.any()


def test_quantisation_collapses_last_bit_differences():
    """Two validators stepping the same seed must agree bit-for-bit."""
    a = canonicalize_action([0.1234567890123, 0, 0, 0, 0], LOW, HIGH, act_dim=5)
    b = canonicalize_action([0.1234567890123 + 1e-9, 0, 0, 0, 0], LOW, HIGH, act_dim=5)
    assert np.array_equal(a, b)
