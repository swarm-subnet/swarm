"""Admission: schema, archive safety, execution profile, shapes, caps."""

from __future__ import annotations

import json
import multiprocessing
import zipfile
from pathlib import Path

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from swarm.model_graph import (
    ReasonCode,
    admit_artifact,
    admit_artifact_subprocess,
    parse_manifest,
)
from swarm.model_graph.constants import ONNX_IR_VERSION, ONNX_OPSET_VERSION
from swarm.model_graph.errors import ModelGraphError
from swarm.model_graph.onnx_profile import inspect_onnx_bytes

from .fixtures import (
    autopilot_artifact,
    autopilot_manifest,
    base_manifest,
    build_artifact,
    dense_model,
    swarm_dense_model,
)


def _raw_model(nodes, inputs, outputs, initializers=()):
    graph = helper.make_graph(nodes, "g", inputs, outputs, list(initializers))
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
    )
    model.ir_version = ONNX_IR_VERSION
    return model.SerializeToString()


def _t(name, shape):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, list(shape))


def _single_node_artifact(tmp_path: Path, model_bytes: bytes, family="cf_autopilot", inputs=None):
    manifest = base_manifest(
        family,
        models=[{"id": "p", "file": "models/p.onnx", "sha256": "0" * 64}],
        nodes=[{"id": "p", "model": "p", "inputs": inputs or {"state": "obs.state"}}],
        action="p.action",
    )
    return build_artifact(tmp_path, manifest, {"models/p.onnx": model_bytes})


def _daemonic_admission(zip_path, queue):
    from swarm.model_graph import admit_artifact_subprocess

    result = admit_artifact_subprocess(Path(zip_path))
    queue.put((result.accepted, result.reason_code))


class TestHappyPath:
    def test_autopilot_artifact_is_admitted(self, tmp_path):
        result = admit_artifact(autopilot_artifact(tmp_path))
        assert result.accepted, result.detail
        assert result.reason_code == ReasonCode.OK.value
        assert result.family_id == "cf_autopilot"
        assert result.worst_tick_flops > 0
        assert len(result.artifact_sha256) == 64
        assert len(result.profile_digest) == 64

    def test_swarm_family_uses_symbolic_drone_axis(self, tmp_path):
        pilot = swarm_dense_model({"depth": (8, 128, 128, 1), "state": (8, 190)}, "action", 5)
        manifest = base_manifest(
            "cf_swarm_autopilot",
            models=[{"id": "p", "file": "models/p.onnx", "sha256": "0" * 64}],
            nodes=[{"id": "p", "model": "p", "inputs": {"depth": "obs.depth", "state": "obs.state"}}],
            action="p.action",
        )
        result = admit_artifact(build_artifact(tmp_path, manifest, {"models/p.onnx": pilot}))
        assert result.accepted, result.detail

    def test_memory_and_cadence_graph(self, tmp_path):
        planner = dense_model({"state": (141,)}, "context", (32,))
        pilot = dense_model({"ctx": (32,), "state": (141,)}, "action", (5,))
        manifest = base_manifest(
            "cf_autopilot",
            models=[
                {"id": "planner", "file": "models/planner.onnx", "sha256": "0" * 64},
                {"id": "pilot", "file": "models/pilot.onnx", "sha256": "0" * 64},
            ],
            memory=[{"name": "ctx", "shape": [32], "writer": "planner.context"}],
            nodes=[
                {"id": "planner", "model": "planner", "inputs": {"state": "obs.state"},
                 "every_n_steps": 10},
                {"id": "pilot", "model": "pilot",
                 "inputs": {"ctx": "memory.ctx", "state": "obs.state"}},
            ],
            action="pilot.action",
        )
        models = {"models/planner.onnx": planner, "models/pilot.onnx": pilot}
        result = admit_artifact(build_artifact(tmp_path, manifest, models))
        assert result.accepted, result.detail
        assert result.memory_floats == 32

    def test_all_five_families_have_a_valid_minimal_graph(self, tmp_path):
        from swarm.model_graph.constants import FAMILY_GRAPH_CONTRACTS, family_has_swarm_axis

        for family, contract in FAMILY_GRAPH_CONTRACTS.items():
            obs = contract["observations"]
            if family_has_swarm_axis(family):
                shapes = {k: tuple(8 if d == "N" else d for d in v) for k, v in obs.items()}
                model = swarm_dense_model(shapes, "action", contract["action_shape"][1])
            else:
                shapes = {k: tuple(v) for k, v in obs.items()}
                model = dense_model(shapes, "action", tuple(contract["action_shape"]))
            manifest = base_manifest(
                family,
                models=[{"id": "p", "file": "models/p.onnx", "sha256": "0" * 64}],
                nodes=[{"id": "p", "model": "p", "inputs": {k: f"obs.{k}" for k in obs}}],
                action="p.action",
            )
            family_dir = tmp_path / family
            family_dir.mkdir()
            result = admit_artifact(build_artifact(family_dir, manifest, {"models/p.onnx": model}))
            assert result.accepted, f"{family}: {result.reason_code} {result.detail}"

    def test_constant_node_shape_matches_initializer_shape(self, tmp_path):
        weight = np.zeros((141, 5), dtype=np.float32)
        bias = np.zeros(5, dtype=np.float32)
        shape_row = np.array([1, -1], dtype=np.int64)
        shape_out = np.array([5], dtype=np.int64)

        def model(shapes_as_constants: bool) -> bytes:
            nodes = []
            if shapes_as_constants:
                nodes.append(helper.make_node(
                    "Constant", [], ["shape_row"],
                    value=numpy_helper.from_array(shape_row, "shape_row_v"),
                ))
                nodes.append(helper.make_node(
                    "Constant", [], ["shape_out"],
                    value=numpy_helper.from_array(shape_out, "shape_out_v"),
                ))
                initializers = []
            else:
                initializers = [
                    numpy_helper.from_array(shape_row, "shape_row"),
                    numpy_helper.from_array(shape_out, "shape_out"),
                ]
            initializers += [
                numpy_helper.from_array(weight, "W"),
                numpy_helper.from_array(bias, "B"),
            ]
            nodes += [
                helper.make_node("Reshape", ["state", "shape_row"], ["flat"]),
                helper.make_node("Gemm", ["flat", "W", "B"], ["gemm_out"]),
                helper.make_node("Tanh", ["gemm_out"], ["activated"]),
                helper.make_node("Reshape", ["activated", "shape_out"], ["action"]),
            ]
            return _raw_model(nodes, [_t("state", [141])], [_t("action", [5])], initializers)

        results = {}
        for label, as_constants in (("initializer", False), ("constant", True)):
            case_dir = tmp_path / label
            case_dir.mkdir()
            artifact = _single_node_artifact(case_dir, model(as_constants))
            results[label] = admit_artifact(artifact)
        assert results["initializer"].accepted, results["initializer"].detail
        assert results["constant"].accepted, results["constant"].detail
        assert results["initializer"].reason_code == results["constant"].reason_code
        assert results["initializer"].worst_tick_flops == results["constant"].worst_tick_flops

        inspections = {}
        for label, as_constants in (("initializer", False), ("constant", True)):
            inspections[label] = inspect_onnx_bytes(model(as_constants))
        assert inspections["initializer"].initializer_count == inspections["constant"].initializer_count
        assert inspections["initializer"].initializer_bytes == inspections["constant"].initializer_bytes


class TestMalformedModels:
    def test_constant_with_empty_payload_is_a_model_file_fault(self):
        from onnx import TensorProto as TP

        empty = TP()
        empty.name = "v"
        empty.data_type = TP.FLOAT
        empty.dims.extend([5])
        node = helper.make_node("Constant", [], ["c"], value=empty)
        out = helper.make_node("Identity", ["c"], ["y"])
        model = _raw_model([node, out], [], [_t("y", [5])])
        with pytest.raises(ModelGraphError) as exc:
            inspect_onnx_bytes(model)
        assert exc.value.reason == ReasonCode.MODEL_FILE_INVALID

    def test_constant_without_output_is_a_model_file_fault(self):
        value = numpy_helper.from_array(np.zeros(2, np.float32), "v")
        node = helper.make_node("Constant", [], [], value=value)
        model = _raw_model([node], [_t("x", [2])], [_t("x", [2])])
        with pytest.raises(ModelGraphError) as exc:
            inspect_onnx_bytes(model)
        assert exc.value.reason == ReasonCode.MODEL_FILE_INVALID

    def test_initializer_with_empty_payload_is_a_model_file_fault(self):
        from onnx import TensorProto as TP

        empty = TP()
        empty.name = "w"
        empty.data_type = TP.FLOAT
        empty.dims.extend([4])
        node = helper.make_node("Identity", ["w"], ["y"])
        model = _raw_model([node], [], [_t("y", [4])], [empty])
        with pytest.raises(ModelGraphError) as exc:
            inspect_onnx_bytes(model)
        assert exc.value.reason == ReasonCode.MODEL_FILE_INVALID

    def test_shape_graph_is_rejected_as_denied_op(self, tmp_path):
        nodes = [
            helper.make_node("Shape", ["state"], ["s"]),
            helper.make_node("Reshape", ["state", "s"], ["action"]),
        ]
        model = _raw_model(nodes, [_t("state", (141,))], [_t("action", (141,))])
        result = admit_artifact(_single_node_artifact(tmp_path, model))
        assert not result.accepted
        assert result.reason_code == ReasonCode.PROFILE_DENIED_OP.value
        assert "Shape" in result.detail


class TestExecutionProfileBlocksHardDecisions:
    """The core anti-hardcoding wall: no operator may express an if/else."""

    @pytest.mark.parametrize(
        "op", ["Greater", "Less", "Equal", "Where", "ArgMax", "TopK", "OneHot",
               "Hardmax", "NonZero", "Sign", "Floor", "Ceil", "Round",
               "ConstantOfShape", "Tile", "Range", "ScatterND", "GatherElements",
               "RandomNormal", "Dropout", "Einsum", "If", "Loop", "Shape", "Cast"],
    )
    def test_denied_operator_is_rejected(self, tmp_path, op):
        from swarm.model_graph.onnx_profile import load_profile

        profile = load_profile()
        assert op in profile["denied_ops"], f"{op} must be in the published denied list"
        assert op not in profile["unconditional_ops"]
        assert op not in profile["restricted_ops"]

    def test_greater_where_fsm_is_rejected(self, tmp_path):
        inits = [
            numpy_helper.from_array(np.full((1, 5), 0.6, np.float32), "th"),
            numpy_helper.from_array(np.ones((1, 5), np.float32), "hi"),
            numpy_helper.from_array(-np.ones((1, 5), np.float32), "lo"),
            numpy_helper.from_array(np.array([1, -1], np.int64), "sr"),
            numpy_helper.from_array(np.array([0], np.int64), "st"),
            numpy_helper.from_array(np.array([5], np.int64), "en"),
            numpy_helper.from_array(np.array([1], np.int64), "ax"),
        ]
        nodes = [
            helper.make_node("Reshape", ["state", "sr"], ["s2"]),
            helper.make_node("Slice", ["s2", "st", "en", "ax"], ["sl"]),
            helper.make_node("Greater", ["sl", "th"], ["cond"]),
            helper.make_node("Where", ["cond", "hi", "lo"], ["action"]),
        ]
        model = _raw_model(nodes, [_t("state", (141,))], [_t("action", (1, 5))], inits)
        result = admit_artifact(_single_node_artifact(tmp_path, model))
        assert not result.accepted
        assert result.reason_code == ReasonCode.PROFILE_DENIED_OP.value
        assert "Greater" in result.detail

    def test_runtime_indexed_gather_rom_is_rejected(self, tmp_path):
        table = numpy_helper.from_array(
            (np.arange(100 * 5).reshape(100, 5) / 100).astype(np.float32), "table"
        )
        nodes = [
            helper.make_node("Cast", ["state"], ["idx"], to=TensorProto.INT64),
            helper.make_node("Gather", ["table", "idx"], ["action"], axis=0),
        ]
        model = _raw_model(nodes, [_t("state", (1,))], [_t("action", (1, 5))], [table])
        result = admit_artifact(_single_node_artifact(tmp_path, model))
        assert not result.accepted
        assert result.reason_code == ReasonCode.PROFILE_DENIED_OP.value
        assert "Cast" in result.detail

    def test_custom_domain_is_rejected(self, tmp_path):
        node = helper.make_node("MyOp", ["state"], ["action"], domain="com.evil")
        graph = helper.make_graph([node], "g", [_t("state", (141,))], [_t("action", (5,))])
        model = helper.make_model(
            graph,
            opset_imports=[
                helper.make_operatorsetid("", ONNX_OPSET_VERSION),
                helper.make_operatorsetid("com.evil", 1),
            ],
        )
        model.ir_version = ONNX_IR_VERSION
        result = admit_artifact(_single_node_artifact(tmp_path, model.SerializeToString()))
        assert not result.accepted
        assert result.reason_code == ReasonCode.PROFILE_DENIED_DOMAIN.value

    def test_tree_ensemble_domain_is_rejected(self, tmp_path):
        node = helper.make_node("TreeEnsembleRegressor", ["state"], ["action"], domain="ai.onnx.ml")
        graph = helper.make_graph([node], "g", [_t("state", (141,))], [_t("action", (5,))])
        model = helper.make_model(
            graph,
            opset_imports=[
                helper.make_operatorsetid("", ONNX_OPSET_VERSION),
                helper.make_operatorsetid("ai.onnx.ml", 3),
            ],
        )
        model.ir_version = ONNX_IR_VERSION
        result = admit_artifact(_single_node_artifact(tmp_path, model.SerializeToString()))
        assert not result.accepted
        assert result.reason_code == ReasonCode.PROFILE_DENIED_DOMAIN.value

    def test_wrong_opset_is_rejected(self, tmp_path):
        graph = helper.make_graph(
            [helper.make_node("Identity", ["state"], ["action"])],
            "g",
            [_t("state", (5,))],
            [_t("action", (5,))],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 17)])
        model.ir_version = ONNX_IR_VERSION
        result = admit_artifact(_single_node_artifact(tmp_path, model.SerializeToString()))
        assert not result.accepted
        assert result.reason_code == ReasonCode.PROFILE_BAD_OPSET.value

    def test_non_float_input_is_rejected(self, tmp_path):
        inp = helper.make_tensor_value_info("state", TensorProto.INT64, [141])
        nodes = [helper.make_node("Identity", ["state"], ["action"])]
        graph = helper.make_graph(
            nodes, "g", [inp], [helper.make_tensor_value_info("action", TensorProto.INT64, [141])]
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
        )
        model.ir_version = ONNX_IR_VERSION
        result = admit_artifact(_single_node_artifact(tmp_path, model.SerializeToString()))
        assert not result.accepted
        assert result.reason_code == ReasonCode.PROFILE_DENIED_DTYPE.value


class TestArchiveSafety:
    def test_undeclared_file_is_rejected(self, tmp_path):
        path = autopilot_artifact(tmp_path)
        with zipfile.ZipFile(path, "a") as zf:
            zf.writestr("evil.py", "import os")
        result = admit_artifact(path)
        assert not result.accepted
        assert result.reason_code == ReasonCode.ARCHIVE_BAD_LAYOUT.value

    def test_parent_path_entry_is_rejected(self, tmp_path):
        path = tmp_path / "bad.zip"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("manifest.json", "{}")
            zf.writestr("../outside.onnx", b"x")
        result = admit_artifact(path)
        assert not result.accepted
        assert result.reason_code == ReasonCode.ARCHIVE_REJECTED.value

    def test_missing_manifest_is_rejected(self, tmp_path):
        path = tmp_path / "nomanifest.zip"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("models/p.onnx", b"x")
        result = admit_artifact(path)
        assert not result.accepted
        assert result.reason_code == ReasonCode.MANIFEST_MISSING.value

    def test_compression_ratio_limit(self, tmp_path):
        path = tmp_path / "high_ratio.zip"
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("manifest.json", "{}")
            zf.writestr("models/p.onnx", b"\0" * (2 * 1024 * 1024))
        result = admit_artifact(path)
        assert not result.accepted
        assert result.reason_code == ReasonCode.ARCHIVE_REJECTED.value

    def test_corrupt_archive_is_rejected(self, tmp_path):
        path = tmp_path / "corrupt.zip"
        path.write_bytes(b"not a zip file at all")
        result = admit_artifact(path)
        assert not result.accepted
        assert result.reason_code == ReasonCode.ARCHIVE_REJECTED.value

    def test_tampered_model_hash_is_rejected(self, tmp_path):
        source = autopilot_artifact(tmp_path)
        with zipfile.ZipFile(source) as zf:
            data = {n: zf.read(n) for n in zf.namelist()}
        manifest = json.loads(data["manifest.json"])
        manifest["models"][0]["sha256"] = "a" * 64
        path = tmp_path / "tampered.zip"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("manifest.json", json.dumps(manifest))
            zf.writestr("models/pilot.onnx", data["models/pilot.onnx"])
        result = admit_artifact(path)
        assert not result.accepted
        assert result.reason_code == ReasonCode.MODEL_HASH_MISMATCH.value


class TestManifestSemantics:
    def test_unknown_key_is_rejected(self):
        doc = autopilot_manifest()
        doc["surprise"] = 1
        with pytest.raises(ModelGraphError) as exc:
            parse_manifest(json.dumps(doc).encode())
        assert exc.value.reason == ReasonCode.MANIFEST_INVALID

    def test_wrong_version_is_rejected(self):
        doc = autopilot_manifest()
        doc["model_graph_version"] = "model_graph.v2"
        with pytest.raises(ModelGraphError) as exc:
            parse_manifest(json.dumps(doc).encode())
        assert exc.value.reason == ReasonCode.MANIFEST_UNSUPPORTED_VERSION

    def test_cycle_is_rejected(self):
        doc = base_manifest(
            "cf_autopilot",
            models=[{"id": "m", "file": "models/m.onnx", "sha256": "0" * 64}],
            nodes=[
                {"id": "a", "model": "m", "inputs": {"x": "b.out"}},
                {"id": "b", "model": "m", "inputs": {"x": "a.out"}},
            ],
            action="a.out",
        )
        with pytest.raises(ModelGraphError) as exc:
            parse_manifest(json.dumps(doc).encode())
        assert exc.value.reason == ReasonCode.GRAPH_CYCLE

    def test_self_reference_is_rejected(self):
        doc = base_manifest(
            "cf_autopilot",
            models=[{"id": "m", "file": "models/m.onnx", "sha256": "0" * 64}],
            nodes=[{"id": "a", "model": "m", "inputs": {"x": "a.out"}}],
            action="a.out",
        )
        with pytest.raises(ModelGraphError) as exc:
            parse_manifest(json.dumps(doc).encode())
        assert exc.value.reason == ReasonCode.GRAPH_CYCLE

    def test_two_writers_for_one_memory_slot_is_rejected(self):
        doc = base_manifest(
            "cf_autopilot",
            models=[{"id": "m", "file": "models/m.onnx", "sha256": "0" * 64}],
            memory=[
                {"name": "h", "shape": [4], "writer": "a.out"},
                {"name": "h", "shape": [4], "writer": "b.out"},
            ],
            nodes=[
                {"id": "a", "model": "m", "inputs": {"x": "obs.state"}},
                {"id": "b", "model": "m", "inputs": {"x": "obs.state"}},
            ],
            action="a.out",
        )
        with pytest.raises(ModelGraphError) as exc:
            parse_manifest(json.dumps(doc).encode())
        assert exc.value.reason == ReasonCode.GRAPH_MEMORY_INVALID

    def test_unknown_observation_channel_is_rejected(self):
        doc = autopilot_manifest()
        doc["nodes"][0]["inputs"]["depth"] = "obs.rgb"
        with pytest.raises(ModelGraphError) as exc:
            parse_manifest(json.dumps(doc).encode())
        assert exc.value.reason == ReasonCode.GRAPH_UNBOUND_INPUT

    def test_reading_undeclared_memory_is_rejected(self):
        doc = autopilot_manifest()
        doc["nodes"][0]["inputs"]["depth"] = "memory.ghost"
        with pytest.raises(ModelGraphError) as exc:
            parse_manifest(json.dumps(doc).encode())
        assert exc.value.reason == ReasonCode.GRAPH_MEMORY_INVALID

    def test_every_n_steps_out_of_range_is_rejected(self):
        doc = autopilot_manifest()
        doc["nodes"][0]["every_n_steps"] = 257
        with pytest.raises(ModelGraphError):
            parse_manifest(json.dumps(doc).encode())

    def test_topological_order_is_independent_of_key_order(self):
        models = [{"id": "m", "file": "models/m.onnx", "sha256": "0" * 64}]
        nodes_forward = [
            {"id": "a", "model": "m", "inputs": {"x": "obs.state"}},
            {"id": "b", "model": "m", "inputs": {"x": "a.out"}},
        ]
        doc1 = base_manifest("cf_autopilot", models=models, nodes=nodes_forward, action="b.out")
        doc2 = base_manifest(
            "cf_autopilot", models=models, nodes=list(reversed(nodes_forward)), action="b.out"
        )
        first = parse_manifest(json.dumps(doc1).encode()).topo_order
        second = parse_manifest(json.dumps(doc2).encode()).topo_order
        assert first == second == ("a", "b")


class TestShapeAndResourceContracts:
    def test_wrong_action_shape_is_rejected(self, tmp_path):
        model = dense_model({"state": (141,)}, "action", (7,))
        result = admit_artifact(_single_node_artifact(tmp_path, model))
        assert not result.accepted
        assert result.reason_code == ReasonCode.SHAPE_CONTRACT_MISMATCH.value

    def test_wrong_observation_shape_is_rejected(self, tmp_path):
        model = dense_model({"state": (99,)}, "action", (5,))
        result = admit_artifact(_single_node_artifact(tmp_path, model))
        assert not result.accepted
        assert result.reason_code == ReasonCode.SHAPE_CONTRACT_MISMATCH.value

    def test_unbound_model_input_is_rejected(self, tmp_path):
        model = dense_model({"depth": (128, 128, 1), "state": (141,)}, "action", (5,))
        artifact = _single_node_artifact(tmp_path, model, inputs={"state": "obs.state"})
        result = admit_artifact(artifact)
        assert not result.accepted
        assert result.reason_code == ReasonCode.GRAPH_UNBOUND_INPUT.value

    def test_dangling_output_is_rejected(self, tmp_path):
        planner = dense_model({"state": (141,)}, "context", (32,))
        pilot = dense_model({"state": (141,)}, "action", (5,))
        manifest = base_manifest(
            "cf_autopilot",
            models=[
                {"id": "planner", "file": "models/planner.onnx", "sha256": "0" * 64},
                {"id": "pilot", "file": "models/pilot.onnx", "sha256": "0" * 64},
            ],
            nodes=[
                {"id": "planner", "model": "planner", "inputs": {"state": "obs.state"}},
                {"id": "pilot", "model": "pilot", "inputs": {"state": "obs.state"}},
            ],
            action="pilot.action",
        )
        models = {"models/planner.onnx": planner, "models/pilot.onnx": pilot}
        result = admit_artifact(build_artifact(tmp_path, manifest, models))
        assert not result.accepted
        assert result.reason_code == ReasonCode.GRAPH_DANGLING_OUTPUT.value

    def test_memory_shape_must_match_its_writer(self, tmp_path):
        planner = dense_model({"state": (141,)}, "context", (32,))
        pilot = dense_model({"ctx": (16,), "state": (141,)}, "action", (5,))
        manifest = base_manifest(
            "cf_autopilot",
            models=[
                {"id": "planner", "file": "models/planner.onnx", "sha256": "0" * 64},
                {"id": "pilot", "file": "models/pilot.onnx", "sha256": "0" * 64},
            ],
            memory=[{"name": "ctx", "shape": [16], "writer": "planner.context"}],
            nodes=[
                {"id": "planner", "model": "planner", "inputs": {"state": "obs.state"}},
                {"id": "pilot", "model": "pilot",
                 "inputs": {"ctx": "memory.ctx", "state": "obs.state"}},
            ],
            action="pilot.action",
        )
        models = {"models/planner.onnx": planner, "models/pilot.onnx": pilot}
        result = admit_artifact(build_artifact(tmp_path, manifest, models))
        assert not result.accepted
        assert result.reason_code == ReasonCode.GRAPH_MEMORY_INVALID.value

    def test_oversized_initializers_are_rejected(self, tmp_path):
        from swarm.model_graph import constants as C
        from swarm.model_graph.onnx_profile import inspect_onnx_bytes

        big = (C.MAX_INITIALIZER_BYTES // 4) + 1024
        weight = numpy_helper.from_array(np.zeros((1, big), np.float32), "W")
        bias = numpy_helper.from_array(np.zeros(big, np.float32), "B")
        nodes = [
            helper.make_node("Gemm", ["state", "W", "B"], ["h"]),
            helper.make_node("Identity", ["h"], ["action"]),
        ]
        model = _raw_model(
            nodes, [_t("state", (1, 1))], [_t("action", (1, big))], [weight, bias]
        )
        with pytest.raises(ModelGraphError) as exc:
            inspect_onnx_bytes(model)
        assert exc.value.reason == ReasonCode.RESOURCE_CAP_EXCEEDED

    def test_incompressible_oversized_archive_is_rejected(self, tmp_path):
        from swarm.model_graph import constants as C

        filler = np.random.default_rng(7).bytes(C.MAX_UNCOMPRESSED_TOTAL_BYTES + 4096)
        path = tmp_path / "big.zip"
        with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as zf:
            zf.writestr("manifest.json", "{}")
            zf.writestr("models/p.onnx", filler)
        result = admit_artifact(path)
        assert not result.accepted
        assert result.reason_code in {
            ReasonCode.ARCHIVE_TOO_LARGE.value,
            ReasonCode.RESOURCE_CAP_EXCEEDED.value,
        }


class TestDeterminism:
    def test_admission_is_reproducible(self, tmp_path):
        path = autopilot_artifact(tmp_path)
        records = [admit_artifact(path).to_record() for _ in range(5)]
        assert all(r == records[0] for r in records)

    def test_profile_digest_is_stable(self):
        from swarm.model_graph.onnx_profile import profile_digest

        assert profile_digest() == profile_digest()
        assert len(profile_digest()) == 64


class TestSubprocessBoundary:
    def test_subprocess_matches_inline_admission(self, tmp_path):
        path = autopilot_artifact(tmp_path)
        boxed = admit_artifact_subprocess(path)
        assert boxed.to_record() == admit_artifact(path).to_record()

    def test_subprocess_rejects_on_wall_timeout(self, tmp_path):
        path = autopilot_artifact(tmp_path)
        result = admit_artifact_subprocess(path, timeout=0.001)
        assert not result.accepted
        assert result.reason_code == ReasonCode.RESOURCE_CAP_EXCEEDED.value

    def test_subprocess_rejects_bad_archive(self, tmp_path):
        path = tmp_path / "bad.zip"
        path.write_bytes(b"not a zip")
        result = admit_artifact_subprocess(path)
        assert not result.accepted
        assert result.reason_code == ReasonCode.ARCHIVE_REJECTED.value

    def test_subprocess_falls_back_inline_in_daemonic_workers(self, tmp_path):
        path = autopilot_artifact(tmp_path)
        context = multiprocessing.get_context("spawn")
        queue = context.Queue()
        worker = context.Process(
            target=_daemonic_admission, args=(str(path), queue), daemon=True
        )
        worker.start()
        accepted, reason_code = queue.get(timeout=60)
        worker.join(10)
        assert accepted, reason_code
        assert reason_code == ReasonCode.OK.value
