"""Model-graph submissions: the only policy format accepted by Swarm.

A submission is an archive containing manifest.json plus the ONNX models it
references. The manifest declares a pure tensor-wiring DAG; the subnet owns
the runner that executes it. Importing this package never creates an ONNX
Runtime session.
"""

from .admission import AdmissionResult, admit_artifact, admit_artifact_subprocess
from .constants import (
    EXECUTION_PROFILE_ID,
    FAMILY_GRAPH_CONTRACTS,
    MODEL_GRAPH_VERSION,
    RUNNER_ABI,
    SUBMISSION_INTERFACE_VERSION,
    VALIDATOR_CONTRACT,
)
from .errors import ARTIFACT_FAULT_CODES, INFRA_FAULT_CODES, ModelGraphError, ReasonCode
from .manifest import GraphManifest, parse_manifest
from .onnx_profile import profile_digest

__all__ = [
    "ARTIFACT_FAULT_CODES",
    "AdmissionResult",
    "EXECUTION_PROFILE_ID",
    "FAMILY_GRAPH_CONTRACTS",
    "GraphManifest",
    "INFRA_FAULT_CODES",
    "MODEL_GRAPH_VERSION",
    "ModelGraphError",
    "RUNNER_ABI",
    "ReasonCode",
    "SUBMISSION_INTERFACE_VERSION",
    "VALIDATOR_CONTRACT",
    "admit_artifact",
    "admit_artifact_subprocess",
    "parse_manifest",
    "profile_digest",
]
