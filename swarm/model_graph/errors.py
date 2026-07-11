"""Closed reason-code vocabulary for model-graph admission and runtime faults.

Every accepted or rejected intake and every evaluation failure carries exactly
one of these codes. Backend and validator must emit identical codes for
identical inputs; tests enforce parity against shared fixtures.
"""

from __future__ import annotations

from enum import Enum


class ReasonCode(str, Enum):
    OK = "MG_OK"

    ARCHIVE_REJECTED = "MG_ARCHIVE_REJECTED"
    ARCHIVE_TOO_LARGE = "MG_ARCHIVE_TOO_LARGE"
    ARCHIVE_BAD_LAYOUT = "MG_ARCHIVE_BAD_LAYOUT"

    MANIFEST_MISSING = "MG_MANIFEST_MISSING"
    MANIFEST_INVALID = "MG_MANIFEST_INVALID"
    MANIFEST_UNSUPPORTED_VERSION = "MG_MANIFEST_UNSUPPORTED_VERSION"

    GRAPH_INVALID = "MG_GRAPH_INVALID"
    GRAPH_CYCLE = "MG_GRAPH_CYCLE"
    GRAPH_UNBOUND_INPUT = "MG_GRAPH_UNBOUND_INPUT"
    GRAPH_DANGLING_OUTPUT = "MG_GRAPH_DANGLING_OUTPUT"
    GRAPH_MEMORY_INVALID = "MG_GRAPH_MEMORY_INVALID"

    MODEL_FILE_INVALID = "MG_MODEL_FILE_INVALID"
    MODEL_HASH_MISMATCH = "MG_MODEL_HASH_MISMATCH"
    PROFILE_DENIED_OP = "MG_PROFILE_DENIED_OP"
    PROFILE_DENIED_DOMAIN = "MG_PROFILE_DENIED_DOMAIN"
    PROFILE_DENIED_DTYPE = "MG_PROFILE_DENIED_DTYPE"
    PROFILE_DENIED_FEATURE = "MG_PROFILE_DENIED_FEATURE"
    PROFILE_BAD_OPSET = "MG_PROFILE_BAD_OPSET"
    SHAPE_INVALID = "MG_SHAPE_INVALID"
    SHAPE_CONTRACT_MISMATCH = "MG_SHAPE_CONTRACT_MISMATCH"

    RESOURCE_CAP_EXCEEDED = "MG_RESOURCE_CAP_EXCEEDED"

    LOAD_FAILED = "MG_LOAD_FAILED"
    NODE_ERROR = "MG_NODE_ERROR"
    OUTPUT_CONTRACT = "MG_OUTPUT_CONTRACT"
    OUTPUT_NONFINITE = "MG_OUTPUT_NONFINITE"
    STEP_SOFT_TIMEOUT = "MG_STEP_SOFT_TIMEOUT"
    STEP_HARD_TIMEOUT = "MG_STEP_HARD_TIMEOUT"
    RUNTIME_CIRCUIT_OPEN = "MG_RUNTIME_CIRCUIT_OPEN"

    INFRA_DOCKER = "INFRA_DOCKER"
    INFRA_RUNNER_RESET = "INFRA_RUNNER_RESET"
    INFRA_IMAGE_MISMATCH = "INFRA_IMAGE_MISMATCH"
    INFRA_CALIBRATION = "INFRA_CALIBRATION"


ARTIFACT_FAULT_CODES = frozenset(
    {
        ReasonCode.LOAD_FAILED,
        ReasonCode.NODE_ERROR,
        ReasonCode.OUTPUT_CONTRACT,
        ReasonCode.OUTPUT_NONFINITE,
        ReasonCode.STEP_SOFT_TIMEOUT,
        ReasonCode.STEP_HARD_TIMEOUT,
    }
)

INFRA_FAULT_CODES = frozenset(
    {
        ReasonCode.INFRA_DOCKER,
        ReasonCode.INFRA_RUNNER_RESET,
        ReasonCode.INFRA_IMAGE_MISMATCH,
        ReasonCode.INFRA_CALIBRATION,
    }
)

_DETAIL_MAX_CHARS = 500


class ModelGraphError(Exception):
    """Structured failure carrying a stable reason code and bounded detail."""

    def __init__(
        self,
        reason: ReasonCode,
        detail: str = "",
        *,
        node_id: str | None = None,
        model_id: str | None = None,
    ) -> None:
        self.reason = reason
        self.detail = detail[:_DETAIL_MAX_CHARS]
        self.node_id = node_id
        self.model_id = model_id
        location = ""
        if node_id:
            location += f" node={node_id}"
        if model_id:
            location += f" model={model_id}"
        super().__init__(f"{reason.value}{location}: {self.detail}")

    def to_record(self) -> dict:
        return {
            "reason_code": self.reason.value,
            "detail": self.detail,
            "node_id": self.node_id,
            "model_id": self.model_id,
        }
