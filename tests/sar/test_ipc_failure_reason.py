"""Regression for Codex round-A HIGH-2: failure_reason must survive the
process IPC round-trip via _pack_validation_result / _unpack_validation_result."""
from __future__ import annotations

from swarm.protocol import FailureReason, ValidationResult
from swarm.benchmark.engine_parts.workers import (
    _pack_validation_result,
    _unpack_validation_result,
)


def test_pack_widens_to_five_tuple():
    vr = ValidationResult(
        uid=3, success=False, time_sec=11.2, score=0.01,
        failure_reason=FailureReason.SPAWN_FAILURE.value,
    )
    packed = _pack_validation_result(vr)
    assert len(packed) == 5
    assert packed[4] == "SPAWN_FAILURE"


def test_unpack_roundtrip_preserves_reason():
    vr = ValidationResult(
        uid=4, success=False, time_sec=8.7, score=0.01,
        failure_reason=FailureReason.INFEASIBLE.value,
    )
    back = _unpack_validation_result(_pack_validation_result(vr))
    assert back.uid == 4
    assert back.failure_reason == "INFEASIBLE"


def test_unpack_backward_compatible_4_tuple():
    legacy = (5, True, 12.5, 0.85)
    back = _unpack_validation_result(legacy)
    assert back.uid == 5
    assert back.success is True
    assert back.score == 0.85
    assert back.failure_reason == "NONE"
