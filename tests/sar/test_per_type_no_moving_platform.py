"""D.4 guardrail: per-challenge-type telemetry / templates carry no
moving_platform keys after cutover.
"""
from __future__ import annotations

import json

from swarm.constants import SAR_SCREENING_TEMPLATE
from swarm.validator.utils_parts.evaluation import _EMPTY_PER_TYPE, _empty_per_type


def test_screening_template_slots_have_no_platform_keys():
    for slot in SAR_SCREENING_TEMPLATE:
        keys = set(slot.keys())
        assert "moving_platform" not in keys
        assert "platform_speed" not in keys
        assert "platform_radius" not in keys


def test_empty_per_type_has_no_moving_platform_field():
    assert "moving_platform" not in _EMPTY_PER_TYPE
    bucket = _empty_per_type()
    serialised = json.dumps(bucket, default=str)
    assert "moving_platform" not in serialised
    assert "MOVING_PLATFORM" not in serialised
