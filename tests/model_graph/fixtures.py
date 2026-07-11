"""Deterministic ONNX and artifact builders shared by the model-graph tests."""

from __future__ import annotations

import hashlib
import json
import zipfile
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

_RNG = np.random.default_rng(20260710)


def _finalize(graph) -> bytes:
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
    )
    model.ir_version = ONNX_IR_VERSION
    onnx.checker.check_model(model)
    return model.SerializeToString()


def _tensor(name: str, shape) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, list(shape))


def _elements(shape) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n


def dense_model(inputs: dict[str, tuple], out_name: str, out_shape: tuple) -> bytes:
    """Flatten every input, concatenate, one Gemm + Tanh, reshape to out_shape.

    Single-drone families only: all shapes are literal. Output is exactly
    out_shape, matching the family action or a memory slot.
    """
    value_infos = [_tensor(name, shape) for name, shape in inputs.items()]
    nodes = []
    flat_names = []
    total_in = 0
    for name, shape in inputs.items():
        flat = f"{name}_flat"
        nodes.append(helper.make_node("Reshape", [name, "shape_row"], [flat]))
        flat_names.append(flat)
        total_in += _elements(shape)

    initializers = [
        numpy_helper.from_array(np.array([1, -1], dtype=np.int64), "shape_row"),
        numpy_helper.from_array(
            np.array(list(out_shape), dtype=np.int64), "shape_out"
        ),
    ]

    if len(flat_names) > 1:
        nodes.append(helper.make_node("Concat", flat_names, ["joined"], axis=1))
        gemm_in = "joined"
    else:
        gemm_in = flat_names[0]

    out_features = _elements(out_shape)
    weight = (_RNG.standard_normal((total_in, out_features)) * 0.01).astype(np.float32)
    bias = np.zeros(out_features, dtype=np.float32)
    initializers += [numpy_helper.from_array(weight, "W"), numpy_helper.from_array(bias, "B")]
    nodes.append(helper.make_node("Gemm", [gemm_in, "W", "B"], ["gemm_out"]))
    nodes.append(helper.make_node("Tanh", ["gemm_out"], ["activated"]))
    nodes.append(helper.make_node("Reshape", ["activated", "shape_out"], [out_name]))

    graph = helper.make_graph(
        nodes, "dense", value_infos, [_tensor(out_name, out_shape)], initializers
    )
    return _finalize(graph)


def swarm_dense_model(inputs: dict[str, tuple], out_name: str, out_features: int) -> bytes:
    """Same as dense_model but the leading drone axis stays symbolic ('N')."""
    value_infos = [_tensor(name, ["N", *shape[1:]]) for name, shape in inputs.items()]
    nodes = []
    flat_names = []
    total_in = 0
    for name, shape in inputs.items():
        flat = f"{name}_flat"
        nodes.append(helper.make_node("Flatten", [name], [flat], axis=1))
        flat_names.append(flat)
        total_in += _elements(shape[1:])

    if len(flat_names) > 1:
        nodes.append(helper.make_node("Concat", flat_names, ["joined"], axis=1))
        gemm_in = "joined"
    else:
        gemm_in = flat_names[0]

    weight = (_RNG.standard_normal((total_in, out_features)) * 0.01).astype(np.float32)
    bias = np.zeros(out_features, dtype=np.float32)
    initializers = [numpy_helper.from_array(weight, "W"), numpy_helper.from_array(bias, "B")]
    nodes.append(helper.make_node("Gemm", [gemm_in, "W", "B"], ["gemm_out"]))
    nodes.append(helper.make_node("Tanh", ["gemm_out"], [out_name]))

    graph = helper.make_graph(
        nodes, "swarm_dense", value_infos, [_tensor(out_name, ["N", out_features])], initializers
    )
    return _finalize(graph)


def op_model(op_type: str, **node_kwargs) -> bytes:
    """A minimal single-op model used to probe the execution profile."""
    inp = _tensor("x", [1, 4])
    out = _tensor("y", [1, 4])
    node = helper.make_node(op_type, ["x"], ["y"], **node_kwargs)
    graph = helper.make_graph([node], f"op_{op_type}", [inp], [out])
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
    )
    model.ir_version = ONNX_IR_VERSION
    return model.SerializeToString()


def build_artifact(tmp_path: Path, manifest: dict, models: dict[str, bytes]) -> Path:
    """Write models/*.onnx + manifest.json (hashes filled in) into a zip."""
    manifest = json.loads(json.dumps(manifest))
    for entry in manifest["models"]:
        entry["sha256"] = hashlib.sha256(models[entry["file"]]).hexdigest()
    zip_path = tmp_path / "artifact.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        for name, data in models.items():
            zf.writestr(name, data)
    return zip_path


def base_manifest(family_id: str, **overrides) -> dict:
    manifest = {
        "model_graph_version": MODEL_GRAPH_VERSION,
        "family_id": family_id,
        "execution_profile": EXECUTION_PROFILE_ID,
        "runner_abi": RUNNER_ABI,
        "models": [],
        "nodes": [],
        "action": "",
    }
    manifest.update(overrides)
    return manifest


def autopilot_models() -> dict[str, bytes]:
    return {
        "models/pilot.onnx": dense_model(
            {"depth": (128, 128, 1), "state": (141,)}, "action", (5,)
        )
    }


def autopilot_manifest() -> dict:
    return base_manifest(
        "cf_autopilot",
        models=[{"id": "pilot", "file": "models/pilot.onnx", "sha256": "0" * 64}],
        nodes=[
            {
                "id": "pilot",
                "model": "pilot",
                "inputs": {"depth": "obs.depth", "state": "obs.state"},
            }
        ],
        action="pilot.action",
    )


def autopilot_artifact(tmp_path: Path) -> Path:
    return build_artifact(tmp_path, autopilot_manifest(), autopilot_models())
