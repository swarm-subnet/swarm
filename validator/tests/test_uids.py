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

"""Which UIDs a validator is allowed to query, and what random selection does to a thin pool."""

from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np

from swarm.utils.uids import check_uid_availability, get_random_uids


def _make_metagraph():
    """Return a four-UID stub metagraph where uid 1 is not serving and uids 2 and 3 hold permits."""
    return SimpleNamespace(
        axons=[
            SimpleNamespace(is_serving=True),
            SimpleNamespace(is_serving=False),
            SimpleNamespace(is_serving=True),
            SimpleNamespace(is_serving=True),
        ],
        validator_permit=np.array([False, False, True, True]),
        S=np.array([0, 0, 50, 200]),
        n=np.array(4),
    )


def test_check_uid_availability_filters_non_serving_uid():
    """An axon that is not serving is unavailable, whatever its stake or permit."""
    metagraph = _make_metagraph()
    assert check_uid_availability(metagraph, uid=1, vpermit_tao_limit=100) is False


def test_check_uid_availability_filters_validator_with_too_much_stake():
    """A permit plus stake above the tao limit rules the UID out even while its axon serves."""
    metagraph = _make_metagraph()
    assert check_uid_availability(metagraph, uid=3, vpermit_tao_limit=100) is False


def test_check_uid_availability_allows_serving_validator_below_limit():
    """A permitted validator whose stake sits under the tao limit stays in the pool."""
    metagraph = _make_metagraph()
    assert check_uid_availability(metagraph, uid=2, vpermit_tao_limit=100) is True


def test_get_random_uids_applies_exclusions_and_caps_k():
    """Asking for more than the pool holds gives back what is available, minus the excluded UID."""
    metagraph = _make_metagraph()
    self_obj = SimpleNamespace(
        metagraph=metagraph,
        config=SimpleNamespace(neuron=SimpleNamespace(vpermit_tao_limit=100)),
    )
    random.seed(0)

    uids = get_random_uids(self_obj, k=10, exclude=[0])
    assert isinstance(uids, np.ndarray)
    assert set(uids.tolist()) == {2}
