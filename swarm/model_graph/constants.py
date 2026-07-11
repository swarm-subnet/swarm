"""Pinned identifiers, caps, and family contracts for model-graph submissions.

Single source of truth for the model_graph.v1 ABI. The backend mirrors the
schema and execution profile byte-for-byte; these constants must only change
together with a new profile/ABI release.
"""

from __future__ import annotations

import re

MODEL_GRAPH_VERSION = "model_graph.v1"
SUBMISSION_INTERFACE_VERSION = "model_graph.v1"
EXECUTION_PROFILE_ID = "swarm.onnx-neural.cpu.v1"
RUNNER_ABI = "graph_runner.v1"
VALIDATOR_CONTRACT = "model_graph_rpc.v1"

GRAPH_MANIFEST_FILENAME = "manifest.json"
MODEL_DIR_PREFIX = "models/"
MODEL_FILE_SUFFIX = ".onnx"

ONNX_IR_VERSION = 10
ONNX_OPSET_VERSION = 18
ONNX_PIN = "1.22.0"
ONNXRUNTIME_PIN = "1.23.2"
NUMPY_PIN = "2.2.6"

ORT_INTRA_OP_THREADS = 2
ORT_INTER_OP_THREADS = 1

MAX_COMPRESSED_ARCHIVE_BYTES = 50 * 1024 * 1024
MAX_UNCOMPRESSED_TOTAL_BYTES = 50 * 1024 * 1024
MAX_PER_FILE_COMPRESSION_RATIO = 20
MAX_GRAPH_MANIFEST_BYTES = 256 * 1024
MAX_ONNX_TOTAL_BYTES = 48 * 1024 * 1024
MAX_INITIALIZER_BYTES = 40 * 1024 * 1024
MAX_ARCHIVE_FILES = 17
MAX_MODELS = 16
MAX_GRAPH_NODES = 16
MAX_GRAPH_EDGES = 64
MAX_ONNX_NODES_PER_MODEL = 4096
MAX_ONNX_NODES_TOTAL = 8192
MAX_ONNX_INITIALIZERS_TOTAL = 4096
MAX_TENSOR_RANK = 6
MAX_TENSOR_ELEMENTS = 16_777_216
MAX_INTERMEDIATE_TOTAL_BYTES = 1024 * 1024 * 1024
MAX_MEMORY_FLOATS = 262_144
MAX_TICK_FLOPS = 20_000_000_000
MAX_EVERY_N_STEPS = 256

STATIC_ADMISSION_WALL_SEC = 20.0
STATIC_ADMISSION_ADDRESS_BYTES = 1024 * 1024 * 1024
SESSION_PROBE_WALL_SEC = 20.0
SESSION_PROBE_ADDRESS_BYTES = 3 * 1024 * 1024 * 1024
RUNNER_STARTUP_WALL_SEC = 30.0
ARTIFACT_FAULT_CIRCUIT_LIMIT = 3

ACTION_QUANT_STEP = 2.0 ** -20
RGB_REQUEST_THRESHOLD = 0.5

IDENTIFIER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")

OBS_PREFIX = "obs."
MEMORY_PREFIX = "memory."

SWARM_AXIS_TOKEN = "N"
SWARM_AXIS_DIM = -1
SWARM_NUM_DRONES_RANGE = (2, 8)

FAMILY_GRAPH_CONTRACTS: dict[str, dict] = {
    "cf_autopilot": {
        "observations": {
            "depth": [128, 128, 1],
            "state": [141],
        },
        "action_shape": [5],
        "action_lower": [-1.0, -1.0, -1.0, 0.0, -1.0],
        "action_upper": [1.0, 1.0, 1.0, 1.0, 1.0],
        "rgb_request_index": None,
    },
    "cf_search_and_rescue": {
        "observations": {
            "depth": [256, 256, 1],
            "rgb": [256, 256, 3],
            "state": [165],
        },
        "action_shape": [6],
        "action_lower": [-1.0, -1.0, -1.0, 0.0, -1.0, 0.0],
        "action_upper": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "rgb_request_index": 5,
    },
    "cf_swarm_autopilot": {
        "observations": {
            "depth": ["N", 128, 128, 1],
            "state": ["N", 190],
        },
        "action_shape": ["N", 5],
        "action_lower": [-1.0, -1.0, -1.0, 0.0, -1.0],
        "action_upper": [1.0, 1.0, 1.0, 1.0, 1.0],
        "rgb_request_index": None,
    },
    "cf_swarm_sar": {
        "observations": {
            "depth": ["N", 256, 256, 1],
            "rgb": ["N", 256, 256, 3],
            "state": ["N", 214],
        },
        "action_shape": ["N", 6],
        "action_lower": [-1.0, -1.0, -1.0, 0.0, -1.0, 0.0],
        "action_upper": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "rgb_request_index": 5,
    },
    "cf_interceptor": {
        "observations": {
            "depth": [1024, 1024, 1],
            "state": [140],
        },
        "action_shape": [5],
        "action_lower": [-1.0, -1.0, -1.0, 0.0, -1.0],
        "action_upper": [1.0, 1.0, 1.0, 1.0, 1.0],
        "rgb_request_index": None,
    },
}


def family_has_swarm_axis(family_id: str) -> bool:
    contract = FAMILY_GRAPH_CONTRACTS[family_id]
    return any(
        shape and shape[0] == SWARM_AXIS_TOKEN
        for shape in contract["observations"].values()
    )


def resolve_shape(shape: list, num_drones: int | None = None) -> tuple[int, ...]:
    """Family contract shape as concrete dims, or SWARM_AXIS_DIM when unresolved."""
    resolved = []
    for d in shape:
        if d == SWARM_AXIS_TOKEN:
            resolved.append(SWARM_AXIS_DIM if num_drones is None else int(num_drones))
        else:
            resolved.append(int(d))
    return tuple(resolved)
