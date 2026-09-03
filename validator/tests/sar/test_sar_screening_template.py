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

from swarm.constants import SAR_SCREENING_TEMPLATE


def test_50_slots():
    assert len(SAR_SCREENING_TEMPLATE) == 50
    for slot in SAR_SCREENING_TEMPLATE:
        assert "moving_platform" not in slot
        assert "goal_height_range" not in slot
        assert "challenge_type" in slot
        assert "distance_range" in slot


def test_map_distribution():
    seen = {slot["challenge_type"] for slot in SAR_SCREENING_TEMPLATE}
    assert seen == {1, 2, 3, 4, 5, 6}


def test_legacy_template_removed():
    """D.4: legacy SCREENING_TEMPLATE deleted post-cutover."""
    from swarm import constants
    assert not hasattr(constants, "SCREENING_TEMPLATE")
