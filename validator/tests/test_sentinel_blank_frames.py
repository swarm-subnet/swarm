"""Blank camera frames are shared and immutable without changing bytes."""

import numpy as np
import pytest

from swarm.challenge_families.solar_patrol.contract import Frames


def test_blank_frames_shared():
    """Separate episodes share only immutable zero image storage."""
    first, second = Frames(), Frames()
    for key in ('rgb', 'thermal', 'zoom'):
        frame = getattr(first, key)
        assert frame is getattr(second, key)
        assert frame.tobytes() == np.zeros(frame.shape, dtype=np.float32).tobytes()
        with pytest.raises(ValueError):
            frame.flat[0] = 1
