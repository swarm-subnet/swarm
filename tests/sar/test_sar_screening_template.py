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
