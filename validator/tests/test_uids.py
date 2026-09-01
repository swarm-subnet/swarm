from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np

from swarm.utils.uids import check_uid_availability, get_random_uids


def _make_metagraph():
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
    metagraph = _make_metagraph()
    assert check_uid_availability(metagraph, uid=1, vpermit_tao_limit=100) is False


def test_check_uid_availability_filters_validator_with_too_much_stake():
    metagraph = _make_metagraph()
    assert check_uid_availability(metagraph, uid=3, vpermit_tao_limit=100) is False


def test_check_uid_availability_allows_serving_validator_below_limit():
    metagraph = _make_metagraph()
    assert check_uid_availability(metagraph, uid=2, vpermit_tao_limit=100) is True


def test_get_random_uids_applies_exclusions_and_caps_k():
    metagraph = _make_metagraph()
    self_obj = SimpleNamespace(
        metagraph=metagraph,
        config=SimpleNamespace(neuron=SimpleNamespace(vpermit_tao_limit=100)),
    )
    random.seed(0)

    uids = get_random_uids(self_obj, k=10, exclude=[0])
    assert isinstance(uids, np.ndarray)
    assert set(uids.tolist()) == {2}
