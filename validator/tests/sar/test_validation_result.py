from __future__ import annotations

import pytest
from dataclasses import asdict

from swarm.protocol import ValidationResult


def test_keyword_only_failure_reason():
    vr = ValidationResult(1, True, 5.0, 0.9)
    assert vr.failure_reason == "NONE"
    with pytest.raises(TypeError):
        ValidationResult(1, True, 5.0, 0.9, "TIMEOUT")


def test_round_trip_with_reason():
    vr = ValidationResult(
        uid=7, success=False, time_sec=12.3, score=0.01, failure_reason="TIMEOUT"
    )
    blob = asdict(vr)
    back = ValidationResult(**blob)
    assert back == vr
    assert back.failure_reason == "TIMEOUT"


def test_parallel_py_unpack():
    packed = (5, True, 9.0, 1.0)
    vr = ValidationResult(*packed)
    assert vr.uid == 5 and vr.success is True
    assert vr.failure_reason == "NONE"


def test_workers_py_unpack():
    packets = [(1, False, 0.0, 0.0), (2, True, 4.5, 0.8)]
    results = [ValidationResult(*packed) for packed in packets]
    assert len(results) == 2
    assert results[1].score == 0.8
    assert all(r.failure_reason == "NONE" for r in results)
