"""Exact comparison with the controller before the scalar storage changes."""

import numpy as np

from swarm.challenge_families.solar_patrol import airframe
from validator.tests.sentinel_original_controller import OriginalControl
from validator.tests.test_solar_airframe import _CONTROL_ENV, _random_flight


def test_controller_bits():
    """Ten thousand successive random states preserve outputs and all loop memory bits."""
    old, new = OriginalControl(_CONTROL_ENV), airframe.M4TDControl(_CONTROL_ENV)
    for args in _random_flight(10_000):
        for before, after in zip(old.computeControl(**args), new.computeControl(**args)):
            assert np.asarray(before).tobytes() == np.asarray(after).tobytes()
        for name in ('integral_pos_e', 'integral_vel_e', 'integral_rpy_e', 'last_rpy'):
            assert np.asarray(getattr(old, name)).tobytes() == np.asarray(getattr(new, name)).tobytes()
