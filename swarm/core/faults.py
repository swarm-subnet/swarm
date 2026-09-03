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

"""Fault vocabulary for submission evaluation.

Separates faults the miner owns (their agent failed to start or answer) from
faults the validator's own infrastructure owns. Only miner faults may reach a
miner's score; infrastructure faults release the work for another attempt.
"""

from __future__ import annotations

from enum import Enum


class ReasonCode(str, Enum):
    LOAD_FAILED = "LOAD_FAILED"

    INFRA_DOCKER = "INFRA_DOCKER"
    INFRA_RUNNER_RESET = "INFRA_RUNNER_RESET"
    INFRA_IMAGE_MISMATCH = "INFRA_IMAGE_MISMATCH"
    INFRA_CALIBRATION = "INFRA_CALIBRATION"


INFRA_FAULT_CODES = frozenset(
    {
        ReasonCode.INFRA_DOCKER,
        ReasonCode.INFRA_RUNNER_RESET,
        ReasonCode.INFRA_IMAGE_MISMATCH,
        ReasonCode.INFRA_CALIBRATION,
    }
)


class EvaluationFault(Exception):
    """Raised when evaluation cannot proceed; carries the code that classifies it."""

    def __init__(self, reason: ReasonCode, detail: str = "") -> None:
        super().__init__(detail or reason.value)
        self.reason = reason
        self.detail = detail
