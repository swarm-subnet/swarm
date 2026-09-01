from __future__ import annotations

from swarm import constants as C


def test_values():
    assert C.SAR_CONFIRM_HORIZ_RADIUS == 2.0
    assert C.SAR_HOVER_BAND == (2.0, 4.0)
    assert C.SAR_CONFIRM_SPEED_MAX == 1.0
    assert C.SAR_HYSTERESIS_GRACE == 0.1
    assert C.SAR_NO_TOUCH_RADIUS == 0.8
    assert C.SAR_DWELL_SEC == 2.0
    assert C.SAR_SEARCH_RADIUS == 30.0
    assert C.SAR_SWEEP_WIDTH == 24.0
    assert C.SAR_TIME_TERM_BUFFER == 1.03
