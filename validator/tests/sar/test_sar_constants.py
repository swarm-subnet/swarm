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
