"""Graph runner: DAG execution, memory, cadence, reset, failures, determinism."""

from __future__ import annotations

import numpy as np
import pytest
from onnx import helper, numpy_helper

from swarm.model_graph import ReasonCode, admit_artifact
from swarm.model_graph.action import action_digest, canonicalize_action
from swarm.model_graph.constants import ACTION_QUANT_STEP
from swarm.model_graph.errors import ModelGraphError
from swarm.model_graph.runner import GraphRunner
from swarm.model_graph.server import ObservationDecoder
from swarm.model_graph.session_probe import probe_artifact_subprocess

from .fixtures import (
    autopilot_artifact,
    base_manifest,
    build_artifact,
    dense_model,
    swarm_dense_model,
)


def _obs(depth_shape=(128, 128, 1), state_len=141, fill=0.25):
    return {
        "depth": np.full(depth_shape, fill, dtype=np.float32),
        "state": np.full((state_len,), 0.1, dtype=np.float32),
    }


class TestBasicExecution:
    def test_act_returns_a_canonical_action(self, tmp_path):
        runner = GraphRunner.from_artifact(autopilot_artifact(tmp_path))
        action = runner.act(_obs())
        assert action.shape == (5,)
        assert action.dtype == np.float32
        assert np.isfinite(action).all()
        assert (action >= np.array([-1, -1, -1, 0, -1], np.float32)).all()
        assert (action <= 1.0).all()
        assert runner.tick == 1

    def test_repeated_ticks_are_deterministic(self, tmp_path):
        runner = GraphRunner.from_artifact(autopilot_artifact(tmp_path))
        first = [action_digest(runner.act(_obs())) for _ in range(5)]
        runner.reset()
        second = [action_digest(runner.act(_obs())) for _ in range(5)]
        assert first == second

    def test_fresh_runners_agree_bit_for_bit(self, tmp_path):
        artifact = autopilot_artifact(tmp_path)
        digests = []
        for _ in range(3):
            runner = GraphRunner.from_artifact(artifact)
            digests.append(action_digest(runner.act(_obs())))
        assert len(set(digests)) == 1

    def test_determinism_harness_repeats_reset_and_act_100_times(self, tmp_path):
        runner = GraphRunner.from_artifact(autopilot_artifact(tmp_path))
        digests = []
        for _ in range(100):
            runner.reset()
            digests.append(action_digest(runner.act(_obs())))
        assert len(set(digests)) == 1

    def test_unique_model_sha_uses_one_session(self, tmp_path):
        model = dense_model({"state": (141,)}, "action", (5,))
        manifest = base_manifest(
            "cf_autopilot",
            models=[
                {"id": "first", "file": "models/first.onnx", "sha256": "0" * 64},
                {"id": "second", "file": "models/second.onnx", "sha256": "0" * 64},
            ],
            nodes=[{"id": "pilot", "model": "second", "inputs": {"state": "obs.state"}}],
            action="pilot.action",
        )
        artifact = build_artifact(
            tmp_path,
            manifest,
            {"models/first.onnx": model, "models/second.onnx": model},
        )
        runner = GraphRunner.from_artifact(artifact)
        assert len(runner._sessions_by_sha) == 1

    def test_observation_dtype_is_not_silently_cast(self, tmp_path):
        runner = GraphRunner.from_artifact(autopilot_artifact(tmp_path))
        observation = _obs()
        observation["state"] = observation["state"].astype(np.float64)
        with pytest.raises(ModelGraphError) as exc:
            runner.act(observation)
        assert exc.value.reason == ReasonCode.OUTPUT_CONTRACT

    def test_session_probe_runs_in_bounded_subprocess(self, tmp_path):
        probe_artifact_subprocess(autopilot_artifact(tmp_path), timeout=10)

    def test_session_probe_child_crash_is_a_load_failure(self, tmp_path, monkeypatch):
        import multiprocessing
        import os
        import signal

        from swarm.model_graph import session_probe

        def crash(artifact, connection):
            os.kill(os.getpid(), signal.SIGKILL)

        fork_context = multiprocessing.get_context("fork")
        monkeypatch.setattr(session_probe, "_probe_child", crash)
        monkeypatch.setattr(
            session_probe.multiprocessing, "get_context", lambda _method: fork_context
        )
        with pytest.raises(ModelGraphError) as exc:
            probe_artifact_subprocess(autopilot_artifact(tmp_path), timeout=10)
        assert exc.value.reason == ReasonCode.LOAD_FAILED
        assert "without a result" in exc.value.detail
        assert not multiprocessing.active_children()

    def test_swarm_family_runs_at_both_drone_counts(self, tmp_path):
        pilot = swarm_dense_model({"depth": (8, 128, 128, 1), "state": (8, 190)}, "action", 5)
        manifest = base_manifest(
            "cf_swarm_autopilot",
            models=[{"id": "p", "file": "models/p.onnx", "sha256": "0" * 64}],
            nodes=[{"id": "p", "model": "p", "inputs": {"depth": "obs.depth", "state": "obs.state"}}],
            action="p.action",
        )
        artifact = build_artifact(tmp_path, manifest, {"models/p.onnx": pilot})
        assert admit_artifact(artifact).accepted

        runner = GraphRunner.from_artifact(artifact)
        for n in (2, 8):
            runner.reset(num_drones=n)
            observation = {
                "depth": np.full((n, 128, 128, 1), 0.2, np.float32),
                "state": np.full((n, 190), 0.1, np.float32),
            }
            action = runner.act(observation, num_drones=n)
            assert action.shape == (n, 5)

    def test_golden_observation_shapes_cover_every_family(self, tmp_path):
        from swarm.model_graph.constants import FAMILY_GRAPH_CONTRACTS, family_has_swarm_axis

        for family_id, contract in FAMILY_GRAPH_CONTRACTS.items():
            family_dir = tmp_path / family_id
            family_dir.mkdir()
            if family_has_swarm_axis(family_id):
                state_shape = tuple(contract["observations"]["state"])
                model = swarm_dense_model(
                    {"state": (8, *state_shape[1:])}, "action", contract["action_shape"][1]
                )
            else:
                state_shape = tuple(contract["observations"]["state"])
                model = dense_model(
                    {"state": state_shape}, "action", tuple(contract["action_shape"])
                )
            manifest = base_manifest(
                family_id,
                models=[{"id": "p", "file": "models/p.onnx", "sha256": "0" * 64}],
                nodes=[{"id": "p", "model": "p", "inputs": {"state": "obs.state"}}],
                action="p.action",
            )
            artifact = build_artifact(
                family_dir, manifest, {"models/p.onnx": model}
            )
            runner = GraphRunner.from_artifact(artifact)
            counts = (2, 8) if family_has_swarm_axis(family_id) else (None,)
            for n in counts:
                runner.reset(num_drones=n)
                observation = {}
                for name, raw_shape in contract["observations"].items():
                    shape = tuple(n if dim == "N" else dim for dim in raw_shape)
                    observation[name] = np.zeros(shape, np.float32)
                action = runner.act(observation, num_drones=n)
                expected = tuple(n if dim == "N" else dim for dim in contract["action_shape"])
                assert action.shape == expected
                if "rgb" in observation:
                    observation["rgb"].fill(0.75)
                    runner.reset(num_drones=n)
                    assert runner.act(observation, num_drones=n).shape == expected


class TestMemoryAndCadence:
    def _memory_artifact(self, tmp_path, every_n_steps=1):
        accumulator = dense_model({"prev": (4,), "state": (141,)}, "next", (4,))
        pilot = dense_model({"ctx": (4,), "state": (141,)}, "action", (5,))
        manifest = base_manifest(
            "cf_autopilot",
            models=[
                {"id": "acc", "file": "models/acc.onnx", "sha256": "0" * 64},
                {"id": "pilot", "file": "models/pilot.onnx", "sha256": "0" * 64},
            ],
            memory=[{"name": "h", "shape": [4], "writer": "acc.next"}],
            nodes=[
                {"id": "acc", "model": "acc",
                 "inputs": {"prev": "memory.h", "state": "obs.state"},
                 "every_n_steps": every_n_steps},
                {"id": "pilot", "model": "pilot",
                 "inputs": {"ctx": "memory.h", "state": "obs.state"}},
            ],
            action="pilot.action",
        )
        models = {"models/acc.onnx": accumulator, "models/pilot.onnx": pilot}
        return build_artifact(tmp_path, manifest, models)

    def test_memory_starts_zeroed_and_carries_forward(self, tmp_path):
        artifact = self._memory_artifact(tmp_path)
        assert admit_artifact(artifact).accepted
        runner = GraphRunner.from_artifact(artifact)
        assert np.array_equal(runner._memory["h"], np.zeros(4, np.float32))
        runner.act(_obs())
        assert not np.array_equal(runner._memory["h"], np.zeros(4, np.float32))

    def test_reset_zeroes_memory_and_tick(self, tmp_path):
        runner = GraphRunner.from_artifact(self._memory_artifact(tmp_path))
        for _ in range(3):
            runner.act(_obs())
        assert runner.tick == 3
        runner.reset()
        assert runner.tick == 0
        assert np.array_equal(runner._memory["h"], np.zeros(4, np.float32))

    def test_memory_reads_previous_tick_not_current(self, tmp_path):
        runner = GraphRunner.from_artifact(self._memory_artifact(tmp_path))
        before = runner._memory["h"].copy()
        runner.act(_obs())
        after_first = runner._memory["h"].copy()
        assert np.array_equal(before, np.zeros(4, np.float32))
        assert not np.array_equal(after_first, before)

    def test_every_n_steps_reuses_the_last_output(self, tmp_path):
        runner = GraphRunner.from_artifact(self._memory_artifact(tmp_path, every_n_steps=3))
        runner.act(_obs())
        memory_after_tick0 = runner._memory["h"].copy()
        runner.act(_obs())
        runner.act(_obs())
        assert np.array_equal(runner._memory["h"], memory_after_tick0)
        runner.act(_obs())
        assert not np.array_equal(runner._memory["h"], memory_after_tick0)

    def test_every_node_runs_on_tick_zero(self, tmp_path):
        runner = GraphRunner.from_artifact(self._memory_artifact(tmp_path, every_n_steps=256))
        action = runner.act(_obs())
        assert action.shape == (5,)
        assert ("acc", "next") in runner._cached_outputs


class TestFailureSemantics:
    def test_nonfinite_action_is_rejected_and_memory_is_not_committed(self, tmp_path):
        nan_weight = numpy_helper.from_array(
            np.full((141, 5), np.nan, np.float32), "W"
        )
        bias = numpy_helper.from_array(np.zeros(5, np.float32), "B")
        shape_row = numpy_helper.from_array(np.array([1, -1], np.int64), "sr")
        shape_out = numpy_helper.from_array(np.array([5], np.int64), "so")
        nodes = [
            helper.make_node("Reshape", ["state", "sr"], ["flat"]),
            helper.make_node("Gemm", ["flat", "W", "B"], ["gemm"]),
            helper.make_node("Reshape", ["gemm", "so"], ["action"]),
        ]
        from .test_admission import _raw_model, _t

        model = _raw_model(
            nodes, [_t("state", (141,))], [_t("action", (5,))], [nan_weight, bias, shape_row, shape_out]
        )
        manifest = base_manifest(
            "cf_autopilot",
            models=[{"id": "p", "file": "models/p.onnx", "sha256": "0" * 64}],
            nodes=[{"id": "p", "model": "p", "inputs": {"state": "obs.state"}}],
            action="p.action",
        )
        artifact = build_artifact(tmp_path, manifest, {"models/p.onnx": model})
        assert admit_artifact(artifact).accepted

        runner = GraphRunner.from_artifact(artifact)
        with pytest.raises(ModelGraphError) as exc:
            runner.act(_obs())
        assert exc.value.reason == ReasonCode.OUTPUT_NONFINITE
        assert runner.tick == 0

    def test_failed_tick_leaves_memory_untouched(self, tmp_path):
        runner = GraphRunner.from_artifact(autopilot_artifact(tmp_path))
        with pytest.raises(ModelGraphError):
            runner.act({"depth": np.zeros((4, 4, 1), np.float32), "state": np.zeros(141, np.float32)})
        assert runner.tick == 0


class TestActionAdapter:
    def test_clipping_and_quantization(self):
        raw = np.array([2.0, -2.0, 0.5, -3.0, 0.123456789], dtype=np.float32)
        action = canonicalize_action(raw, "cf_autopilot")
        assert action[0] == 1.0
        assert action[1] == -1.0
        assert action[3] == 0.0
        assert np.isclose(action[4] % ACTION_QUANT_STEP, 0.0, atol=1e-9)

    def test_sar_rgb_request_is_binarized(self):
        raw = np.array([0, 0, 0, 0.5, 0, 0.51], dtype=np.float32)
        action = canonicalize_action(raw, "cf_search_and_rescue")
        assert action[5] == 1.0
        raw[5] = 0.5
        assert canonicalize_action(raw, "cf_search_and_rescue")[5] == 0.0

    def test_wrong_shape_is_a_contract_fault(self):
        with pytest.raises(ModelGraphError) as exc:
            canonicalize_action(np.zeros(4, np.float32), "cf_autopilot")
        assert exc.value.reason == ReasonCode.OUTPUT_CONTRACT

    def test_wrong_dtype_is_a_contract_fault(self):
        with pytest.raises(ModelGraphError) as exc:
            canonicalize_action(np.zeros(5, np.float64), "cf_autopilot")
        assert exc.value.reason == ReasonCode.OUTPUT_CONTRACT

    def test_nan_is_a_nonfinite_fault(self):
        raw = np.zeros(5, np.float32)
        raw[2] = np.nan
        with pytest.raises(ModelGraphError) as exc:
            canonicalize_action(raw, "cf_autopilot")
        assert exc.value.reason == ReasonCode.OUTPUT_NONFINITE

    def test_swarm_action_shape_follows_drone_count(self):
        raw = np.zeros((3, 5), np.float32)
        action = canonicalize_action(raw, "cf_swarm_autopilot", num_drones=3)
        assert action.shape == (3, 5)
        with pytest.raises(ModelGraphError):
            canonicalize_action(raw, "cf_swarm_autopilot", num_drones=4)


class TestObservationDecoder:
    class _Tensor:
        def __init__(self, data, shape, dtype="float32"):
            self.data = data
            self.shape = shape
            self.dtype = dtype

    class _Entry:
        def __init__(self, key, tensor):
            self.key = key
            self.tensor = tensor

    def test_inline_and_compact_zero_tensors(self):
        decoder = ObservationDecoder()
        entries = [
            self._Entry("state", self._Tensor(np.ones(3, np.float32).tobytes(), [3])),
            self._Entry("rgb", self._Tensor(b"", [2, 2, 3])),
        ]
        result = decoder.decode(entries)
        assert np.array_equal(result["state"], np.ones(3, np.float32))
        assert np.array_equal(result["rgb"], np.zeros((2, 2, 3), np.float32))
        assert not result["state"].flags.writeable

    def test_rejects_non_float32_wire_dtype(self):
        decoder = ObservationDecoder()
        with pytest.raises(ValueError, match="dtype"):
            decoder.decode([self._Entry("state", self._Tensor(b"", [3], "float64"))])
