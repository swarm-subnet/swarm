from __future__ import annotations

from swarm.protocol import FailureReason, ValidationResult
from swarm.benchmark.engine_parts.workers import _pack_validation_result


def test_pack_serialises_enum_via_value():
    vr = ValidationResult(
        uid=9, success=False, time_sec=2.0, score=0.01,
        failure_reason=FailureReason.TIMEOUT,
    )
    packed = _pack_validation_result(vr)
    assert packed[4] == "TIMEOUT"


def test_pack_passes_through_plain_string():
    vr = ValidationResult(
        uid=9, success=False, time_sec=2.0, score=0.01,
        failure_reason="OBSTACLE_COLLISION",
    )
    packed = _pack_validation_result(vr)
    assert packed[4] == "OBSTACLE_COLLISION"
