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
