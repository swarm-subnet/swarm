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
