"""Generate the deterministic graph-only template artifact source tree."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from swarm.model_graph.constants import (
    EXECUTION_PROFILE_ID,
    MODEL_GRAPH_VERSION,
    ONNX_IR_VERSION,
    ONNX_OPSET_VERSION,
    RUNNER_ABI,
)


def build_autopilot_model() -> bytes:
    inputs = [
        helper.make_tensor_value_info("depth", TensorProto.FLOAT, [128, 128, 1]),
        helper.make_tensor_value_info("state", TensorProto.FLOAT, [141]),
    ]
    output = helper.make_tensor_value_info("action", TensorProto.FLOAT, [5])
    initializers = [
        numpy_helper.from_array(np.array([1, -1], np.int64), "flat_shape"),
        numpy_helper.from_array(np.array([5], np.int64), "action_shape"),
        numpy_helper.from_array(np.zeros((128 * 128 + 141, 5), np.float32), "weight"),
        numpy_helper.from_array(np.zeros(5, np.float32), "bias"),
    ]
    nodes = [
        helper.make_node("Reshape", ["depth", "flat_shape"], ["depth_flat"]),
        helper.make_node("Reshape", ["state", "flat_shape"], ["state_flat"]),
        helper.make_node("Concat", ["depth_flat", "state_flat"], ["features"], axis=1),
        helper.make_node("Gemm", ["features", "weight", "bias"], ["raw_action"]),
        helper.make_node("Tanh", ["raw_action"], ["bounded"]),
        helper.make_node("Reshape", ["bounded", "action_shape"], ["action"]),
    ]
    model = helper.make_model(
        helper.make_graph(nodes, "swarm_template", inputs, [output], initializers),
        opset_imports=[helper.make_operatorsetid("", ONNX_OPSET_VERSION)],
    )
    model.ir_version = ONNX_IR_VERSION
    onnx.checker.check_model(model)
    return model.SerializeToString()


def generate(output_dir: Path) -> None:
    model = build_autopilot_model()
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "policy.onnx").write_bytes(model)
    manifest = {
        "model_graph_version": MODEL_GRAPH_VERSION,
        "family_id": "cf_autopilot",
        "execution_profile": EXECUTION_PROFILE_ID,
        "runner_abi": RUNNER_ABI,
        "models": [{
            "id": "policy",
            "file": "models/policy.onnx",
            "sha256": hashlib.sha256(model).hexdigest(),
        }],
        "memory": [],
        "nodes": [{
            "id": "policy",
            "model": "policy",
            "inputs": {"depth": "obs.depth", "state": "obs.state"},
            "every_n_steps": 1,
        }],
        "action": "policy.action",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args(argv)
    generate(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
