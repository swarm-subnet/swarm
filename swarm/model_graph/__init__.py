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
