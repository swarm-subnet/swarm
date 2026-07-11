"""Executes an admitted model graph.

One ORT session per unique model file, shared across nodes. Memory slots are
double buffered: reads see the previous tick, writes commit only after the
whole tick succeeds, so a failed tick never leaves partial state behind.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

from .action import canonicalize_action
from .archive import read_archive
from .constants import (
    FAMILY_GRAPH_CONTRACTS,
    ORT_INTER_OP_THREADS,
    ORT_INTRA_OP_THREADS,
    SWARM_AXIS_DIM,
    family_has_swarm_axis,
    resolve_shape,
)
from .errors import ModelGraphError, ReasonCode
from .manifest import (
    Binding,
    GraphManifest,
    MemoryBinding,
    NodeBinding,
    ObsBinding,
    parse_manifest,
)


def _session_options() -> ort.SessionOptions:
    options = ort.SessionOptions()
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    options.intra_op_num_threads = ORT_INTRA_OP_THREADS
    options.inter_op_num_threads = ORT_INTER_OP_THREADS
    options.enable_mem_pattern = False
    options.add_session_config_entry("session.use_deterministic_compute", "1")
    return options


def _concrete_shape(shape: tuple[int, ...], num_drones: int | None) -> tuple[int, ...]:
    return tuple(num_drones if d == SWARM_AXIS_DIM else d for d in shape)


class GraphRunner:
    """Runs one admitted artifact for one episode at a time."""

    def __init__(self, manifest: GraphManifest, models: dict[str, bytes]) -> None:
        self.manifest = manifest
        self._sessions_by_sha: dict[str, ort.InferenceSession] = {}
        self._sessions: dict[str, ort.InferenceSession] = {}
        options = _session_options()
        for model in manifest.models:
            session = self._sessions_by_sha.get(model.sha256)
            if session is None:
                try:
                    session = ort.InferenceSession(
                        models[model.file],
                        sess_options=options,
                        providers=["CPUExecutionProvider"],
                    )
                except Exception as exc:
                    raise ModelGraphError(
                        ReasonCode.LOAD_FAILED, str(exc), model_id=model.model_id
                    ) from exc
                self._sessions_by_sha[model.sha256] = session
            self._sessions[model.model_id] = session

        self._num_drones: int | None = None
        self._observation: dict[str, np.ndarray] = {}
        self._memory: dict[str, np.ndarray] = {}
        self._cached_outputs: dict[tuple[str, str], np.ndarray] = {}
        self._tick = 0
        self.reset()

    @classmethod
    def from_artifact(cls, zip_path: Path) -> "GraphRunner":
        from .admission import admit_artifact

        result = admit_artifact(Path(zip_path))
        if not result.accepted:
            raise ModelGraphError(ReasonCode(result.reason_code), result.detail)
        contents = read_archive(Path(zip_path))
        return cls(parse_manifest(contents.manifest_bytes), contents.files)

    @property
    def tick(self) -> int:
        return self._tick

    def reset(self, num_drones: int | None = None) -> None:
        """Zero all memory and caches; sessions are preserved."""
        self._num_drones = num_drones
        self._tick = 0
        self._cached_outputs.clear()
        if family_has_swarm_axis(self.manifest.family_id) and num_drones is None:
            self._memory = {}
            return
        self._memory = {
            slot.name: np.zeros(_concrete_shape(slot.shape, num_drones), dtype=np.float32)
            for slot in self.manifest.memory
        }

    def _resolve(self, binding: Binding, tick_outputs: dict[tuple[str, str], np.ndarray]) -> np.ndarray:
        if isinstance(binding, ObsBinding):
            return self._observation[binding.key]
        if isinstance(binding, MemoryBinding):
            return self._memory[binding.name]
        if isinstance(binding, NodeBinding):
            key = (binding.node_id, binding.output)
            if key in tick_outputs:
                return tick_outputs[key]
            if key in self._cached_outputs:
                return self._cached_outputs[key]
            raise ModelGraphError(
                ReasonCode.NODE_ERROR,
                f"output '{binding.output}' is not available this tick",
                node_id=binding.node_id,
            )
        raise ModelGraphError(ReasonCode.GRAPH_INVALID, "unknown binding type")

    def _validate_observation(
        self, observation: dict[str, np.ndarray], num_drones: int | None
    ) -> tuple[dict[str, np.ndarray], int | None]:
        contract = FAMILY_GRAPH_CONTRACTS[self.manifest.family_id]
        if not isinstance(observation, dict) or set(observation) != set(contract["observations"]):
            got = sorted(observation) if isinstance(observation, dict) else type(observation).__name__
            raise ModelGraphError(
                ReasonCode.OUTPUT_CONTRACT,
                f"observation keys {got} do not match {sorted(contract['observations'])}",
            )
        if family_has_swarm_axis(self.manifest.family_id):
            inferred = {np.asarray(value).shape[0] for value in observation.values() if np.asarray(value).ndim}
            if len(inferred) != 1:
                raise ModelGraphError(ReasonCode.OUTPUT_CONTRACT, "swarm observation batch axes disagree")
            observed_n = inferred.pop()
            if num_drones is not None and observed_n != num_drones:
                raise ModelGraphError(ReasonCode.OUTPUT_CONTRACT, "num_drones disagrees with observation")
            num_drones = int(observed_n)
            if not 2 <= num_drones <= 8:
                raise ModelGraphError(ReasonCode.OUTPUT_CONTRACT, "swarm size must be 2..8")
        elif num_drones is not None:
            raise ModelGraphError(ReasonCode.OUTPUT_CONTRACT, "num_drones is invalid for this family")

        validated: dict[str, np.ndarray] = {}
        for key, shape in contract["observations"].items():
            value = np.asarray(observation[key])
            expected = resolve_shape(shape, num_drones)
            if value.dtype != np.float32 or tuple(value.shape) != expected:
                raise ModelGraphError(
                    ReasonCode.OUTPUT_CONTRACT,
                    f"observation '{key}' must be float32 {expected}, got {value.dtype} {tuple(value.shape)}",
                )
            validated[key] = np.ascontiguousarray(value)
        return validated, num_drones

    def act(self, observation: dict[str, np.ndarray], num_drones: int | None = None) -> np.ndarray:
        """Run one tick and return the canonical action.

        Any failure aborts the tick: no memory is committed and no cached
        output is replaced.
        """
        self._observation, num_drones = self._validate_observation(observation, num_drones)
        if num_drones != self._num_drones:
            if self._tick != 0:
                raise ModelGraphError(ReasonCode.OUTPUT_CONTRACT, "swarm size changed without reset")
            self.reset(num_drones)

        tick_outputs: dict[tuple[str, str], np.ndarray] = {}
        for node_id in self.manifest.topo_order:
            node = self.manifest.node(node_id)
            due = self._tick % node.every_n_steps == 0
            if not due:
                continue
            feeds = {
                name: self._resolve(binding, tick_outputs)
                for name, binding in node.inputs.items()
            }
            session = self._sessions[node.model_id]
            try:
                outputs = session.run(None, feeds)
            except Exception as exc:
                raise ModelGraphError(
                    ReasonCode.NODE_ERROR, str(exc), node_id=node_id, model_id=node.model_id
                ) from exc
            for meta, value in zip(session.get_outputs(), outputs):
                array = np.asarray(value)
                if array.dtype != np.float32:
                    raise ModelGraphError(
                        ReasonCode.OUTPUT_CONTRACT,
                        f"output '{meta.name}' dtype {array.dtype} is not float32",
                        node_id=node_id,
                    )
                if not np.isfinite(array).all():
                    raise ModelGraphError(
                        ReasonCode.OUTPUT_NONFINITE,
                        f"output '{meta.name}' contains NaN or infinity",
                        node_id=node_id,
                    )
                tick_outputs[(node_id, meta.name)] = array

        action_key = (self.manifest.action.node_id, self.manifest.action.output)
        raw_action = (
            tick_outputs.get(action_key)
            if action_key in tick_outputs
            else self._cached_outputs.get(action_key)
        )
        if raw_action is None:
            raise ModelGraphError(
                ReasonCode.OUTPUT_CONTRACT,
                "action output was not produced this tick",
                node_id=self.manifest.action.node_id,
            )
        action = canonicalize_action(raw_action, self.manifest.family_id, self._num_drones)

        pending_memory = {}
        for slot in self.manifest.memory:
            key = (slot.writer.node_id, slot.writer.output)
            if key in tick_outputs:
                value = tick_outputs[key]
                expected = _concrete_shape(slot.shape, self._num_drones)
                if tuple(value.shape) != expected:
                    raise ModelGraphError(
                        ReasonCode.OUTPUT_CONTRACT,
                        f"memory '{slot.name}' expects {expected}, got {tuple(value.shape)}",
                        node_id=slot.writer.node_id,
                    )
                pending_memory[slot.name] = np.array(value, dtype=np.float32, copy=True)

        self._memory.update(pending_memory)
        self._cached_outputs.update(tick_outputs)
        self._tick += 1
        return action
