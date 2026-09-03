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

"""Parser and canonicalizer for the model-graph artifact manifest.

Validation here is structural and self-contained (no third-party schema
library) so the backend mirror behaves byte-for-byte identically. The JSON
schema file next to this module is the published structural contract; this
parser enforces it plus the graph semantics the schema cannot express
(family gating of the drone axis, reference resolution, cycle detection).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .constants import (
    EXECUTION_PROFILE_ID,
    FAMILY_GRAPH_CONTRACTS,
    IDENTIFIER_RE,
    MAX_EVERY_N_STEPS,
    MAX_GRAPH_EDGES,
    MAX_GRAPH_MANIFEST_BYTES,
    MAX_GRAPH_NODES,
    MAX_MODELS,
    MAX_TENSOR_RANK,
    MEMORY_PREFIX,
    MODEL_GRAPH_VERSION,
    OBS_PREFIX,
    RUNNER_ABI,
    SWARM_AXIS_DIM,
    SWARM_AXIS_TOKEN,
    SWARM_NUM_DRONES_RANGE,
    family_has_swarm_axis,
)
from .errors import ModelGraphError, ReasonCode

_MODEL_FILE_RE = re.compile(r"^models/[A-Za-z0-9_.-]{1,80}\.onnx$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TENSOR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:/-]{0,63}$")
_MAX_MEMORY_SLOTS = 32
_MAX_NODE_INPUTS = 16


@dataclass(frozen=True)
class ObsBinding:
    key: str


@dataclass(frozen=True)
class MemoryBinding:
    name: str


@dataclass(frozen=True)
class NodeBinding:
    node_id: str
    output: str


Binding = ObsBinding | MemoryBinding | NodeBinding


@dataclass(frozen=True)
class ModelDecl:
    model_id: str
    file: str
    sha256: str


@dataclass(frozen=True)
class MemoryDecl:
    name: str
    shape: tuple[int, ...]
    writer: NodeBinding


@dataclass(frozen=True)
class NodeDecl:
    node_id: str
    model_id: str
    inputs: dict[str, Binding]
    every_n_steps: int = 1


@dataclass(frozen=True)
class GraphManifest:
    family_id: str
    models: tuple[ModelDecl, ...]
    memory: tuple[MemoryDecl, ...]
    nodes: tuple[NodeDecl, ...]
    action: NodeBinding
    topo_order: tuple[str, ...] = field(default=())

    @property
    def memory_floats(self) -> int:
        """Worst-case float count, drone axis at its maximum."""
        total = 0
        for slot in self.memory:
            n = 1
            for d in slot.shape:
                n *= SWARM_NUM_DRONES_RANGE[1] if d == SWARM_AXIS_DIM else d
            total += n
        return total

    def node(self, node_id: str) -> NodeDecl:
        for n in self.nodes:
            if n.node_id == node_id:
                return n
        raise KeyError(node_id)


def _invalid(detail: str) -> ModelGraphError:
    return ModelGraphError(ReasonCode.MANIFEST_INVALID, detail)


def _require(obj: dict, key: str, kind: type, detail: str):
    if key not in obj:
        raise _invalid(f"missing key '{key}' in {detail}")
    value = obj[key]
    if kind is int and isinstance(value, bool):
        raise _invalid(f"'{key}' must be an integer in {detail}")
    if not isinstance(value, kind):
        raise _invalid(f"'{key}' has wrong type in {detail}")
    return value


def _check_no_extra_keys(obj: dict, allowed: set[str], detail: str) -> None:
    extra = set(obj) - allowed
    if extra:
        raise _invalid(f"unknown keys {sorted(extra)} in {detail}")


def _identifier(value, detail: str) -> str:
    if not isinstance(value, str) or not IDENTIFIER_RE.match(value):
        raise _invalid(f"invalid identifier in {detail}")
    return value


def _parse_binding(raw, detail: str) -> Binding:
    if not isinstance(raw, str):
        raise _invalid(f"binding must be a string in {detail}")
    if raw.startswith(OBS_PREFIX):
        key = raw[len(OBS_PREFIX):]
        _identifier(key, detail)
        return ObsBinding(key)
    if raw.startswith(MEMORY_PREFIX):
        name = raw[len(MEMORY_PREFIX):]
        _identifier(name, detail)
        return MemoryBinding(name)
    if "." not in raw:
        raise _invalid(f"binding '{raw}' is not obs.*, memory.* or node.output in {detail}")
    node_id, output = raw.split(".", 1)
    _identifier(node_id, detail)
    if not _TENSOR_NAME_RE.match(output):
        raise _invalid(f"invalid output tensor name in binding '{raw}' ({detail})")
    return NodeBinding(node_id, output)


def parse_manifest(raw_bytes: bytes) -> GraphManifest:
    if len(raw_bytes) > MAX_GRAPH_MANIFEST_BYTES:
        raise ModelGraphError(
            ReasonCode.RESOURCE_CAP_EXCEEDED,
            f"manifest.json is {len(raw_bytes)} bytes (cap {MAX_GRAPH_MANIFEST_BYTES})",
        )
    try:
        doc = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _invalid(f"manifest.json is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(doc, dict):
        raise _invalid("manifest root must be a JSON object")

    _check_no_extra_keys(
        doc,
        {
            "model_graph_version",
            "family_id",
            "execution_profile",
            "runner_abi",
            "models",
            "memory",
            "nodes",
            "action",
        },
        "manifest root",
    )

    version = _require(doc, "model_graph_version", str, "manifest root")
    if version != MODEL_GRAPH_VERSION:
        raise ModelGraphError(
            ReasonCode.MANIFEST_UNSUPPORTED_VERSION,
            f"model_graph_version '{version}' is not supported",
        )
    profile = _require(doc, "execution_profile", str, "manifest root")
    if profile != EXECUTION_PROFILE_ID:
        raise ModelGraphError(
            ReasonCode.MANIFEST_UNSUPPORTED_VERSION,
            f"execution_profile '{profile}' is not supported",
        )
    abi = _require(doc, "runner_abi", str, "manifest root")
    if abi != RUNNER_ABI:
        raise ModelGraphError(
            ReasonCode.MANIFEST_UNSUPPORTED_VERSION,
            f"runner_abi '{abi}' is not supported",
        )

    family_id = _require(doc, "family_id", str, "manifest root")
    if family_id not in FAMILY_GRAPH_CONTRACTS:
        raise _invalid(f"unknown family_id '{family_id}'")
    contract = FAMILY_GRAPH_CONTRACTS[family_id]

    raw_models = _require(doc, "models", list, "manifest root")
    if not 1 <= len(raw_models) <= MAX_MODELS:
        raise _invalid(f"models must contain 1..{MAX_MODELS} entries")
    models: list[ModelDecl] = []
    seen_model_ids: set[str] = set()
    seen_files: set[str] = set()
    for i, entry in enumerate(raw_models):
        if not isinstance(entry, dict):
            raise _invalid(f"models[{i}] must be an object")
        _check_no_extra_keys(entry, {"id", "file", "sha256"}, f"models[{i}]")
        model_id = _identifier(_require(entry, "id", str, f"models[{i}]"), f"models[{i}].id")
        file = _require(entry, "file", str, f"models[{i}]")
        sha256 = _require(entry, "sha256", str, f"models[{i}]")
        if not _MODEL_FILE_RE.match(file):
            raise _invalid(f"models[{i}].file '{file}' must match models/<name>.onnx")
        if not _SHA256_RE.match(sha256):
            raise _invalid(f"models[{i}].sha256 is not a lowercase hex sha256")
        if model_id in seen_model_ids:
            raise _invalid(f"duplicate model id '{model_id}'")
        if file in seen_files:
            raise _invalid(f"duplicate model file '{file}'")
        seen_model_ids.add(model_id)
        seen_files.add(file)
        models.append(ModelDecl(model_id, file, sha256))

    raw_nodes = _require(doc, "nodes", list, "manifest root")
    if not 1 <= len(raw_nodes) <= MAX_GRAPH_NODES:
        raise _invalid(f"nodes must contain 1..{MAX_GRAPH_NODES} entries")
    nodes: list[NodeDecl] = []
    seen_node_ids: set[str] = set()
    edge_count = 0
    for i, entry in enumerate(raw_nodes):
        if not isinstance(entry, dict):
            raise _invalid(f"nodes[{i}] must be an object")
        _check_no_extra_keys(entry, {"id", "model", "inputs", "every_n_steps"}, f"nodes[{i}]")
        node_id = _identifier(_require(entry, "id", str, f"nodes[{i}]"), f"nodes[{i}].id")
        model_id = _identifier(_require(entry, "model", str, f"nodes[{i}]"), f"nodes[{i}].model")
        if node_id in seen_node_ids:
            raise _invalid(f"duplicate node id '{node_id}'")
        seen_node_ids.add(node_id)
        if model_id not in seen_model_ids:
            raise _invalid(f"nodes[{i}] references unknown model '{model_id}'")
        raw_inputs = _require(entry, "inputs", dict, f"nodes[{i}]")
        if not 1 <= len(raw_inputs) <= _MAX_NODE_INPUTS:
            raise _invalid(f"nodes[{i}].inputs must contain 1..{_MAX_NODE_INPUTS} bindings")
        inputs: dict[str, Binding] = {}
        for input_name, raw_binding in raw_inputs.items():
            if not _TENSOR_NAME_RE.match(input_name):
                raise _invalid(f"invalid input tensor name '{input_name}' in nodes[{i}]")
            binding = _parse_binding(raw_binding, f"nodes[{i}].inputs['{input_name}']")
            if isinstance(binding, ObsBinding) and binding.key not in contract["observations"]:
                raise ModelGraphError(
                    ReasonCode.GRAPH_UNBOUND_INPUT,
                    f"obs.{binding.key} does not exist for family {family_id}",
                    node_id=node_id,
                )
            inputs[input_name] = binding
            edge_count += 1
        every_n = entry.get("every_n_steps", 1)
        if isinstance(every_n, bool) or not isinstance(every_n, int):
            raise _invalid(f"nodes[{i}].every_n_steps must be an integer")
        if not 1 <= every_n <= MAX_EVERY_N_STEPS:
            raise _invalid(f"nodes[{i}].every_n_steps out of range 1..{MAX_EVERY_N_STEPS}")
        nodes.append(NodeDecl(node_id, model_id, inputs, every_n))

    raw_memory = doc.get("memory", [])
    if not isinstance(raw_memory, list) or len(raw_memory) > _MAX_MEMORY_SLOTS:
        raise _invalid(f"memory must be a list of at most {_MAX_MEMORY_SLOTS} slots")
    memory: list[MemoryDecl] = []
    seen_memory: set[str] = set()
    for i, entry in enumerate(raw_memory):
        if not isinstance(entry, dict):
            raise _invalid(f"memory[{i}] must be an object")
        _check_no_extra_keys(entry, {"name", "shape", "writer"}, f"memory[{i}]")
        name = _identifier(_require(entry, "name", str, f"memory[{i}]"), f"memory[{i}].name")
        if name in seen_memory:
            raise ModelGraphError(
                ReasonCode.GRAPH_MEMORY_INVALID, f"memory slot '{name}' has multiple writers"
            )
        seen_memory.add(name)
        raw_shape = _require(entry, "shape", list, f"memory[{i}]")
        if not 1 <= len(raw_shape) <= MAX_TENSOR_RANK:
            raise _invalid(f"memory[{i}].shape rank must be 1..{MAX_TENSOR_RANK}")
        shape: list[int] = []
        for axis, d in enumerate(raw_shape):
            if d == SWARM_AXIS_TOKEN and axis == 0 and family_has_swarm_axis(family_id):
                shape.append(SWARM_AXIS_DIM)
                continue
            if isinstance(d, bool) or not isinstance(d, int) or d < 1:
                raise _invalid(f"memory[{i}].shape entries must be positive integers")
            shape.append(d)
        writer = _parse_binding(_require(entry, "writer", str, f"memory[{i}]"), f"memory[{i}].writer")
        if not isinstance(writer, NodeBinding):
            raise ModelGraphError(
                ReasonCode.GRAPH_MEMORY_INVALID,
                f"memory[{i}].writer must reference a node output",
            )
        if writer.node_id not in seen_node_ids:
            raise ModelGraphError(
                ReasonCode.GRAPH_MEMORY_INVALID,
                f"memory[{i}].writer references unknown node '{writer.node_id}'",
            )
        memory.append(MemoryDecl(name, tuple(shape), writer))
        edge_count += 1

    memory_names = {slot.name for slot in memory}
    for node in nodes:
        for input_name, binding in node.inputs.items():
            if isinstance(binding, MemoryBinding) and binding.name not in memory_names:
                raise ModelGraphError(
                    ReasonCode.GRAPH_MEMORY_INVALID,
                    f"input '{input_name}' reads undeclared memory '{binding.name}'",
                    node_id=node.node_id,
                )
            if isinstance(binding, NodeBinding) and binding.node_id not in seen_node_ids:
                raise ModelGraphError(
                    ReasonCode.GRAPH_UNBOUND_INPUT,
                    f"input '{input_name}' references unknown node '{binding.node_id}'",
                    node_id=node.node_id,
                )

    action = _parse_binding(_require(doc, "action", str, "manifest root"), "action")
    if not isinstance(action, NodeBinding):
        raise _invalid("action must reference a node output")
    if action.node_id not in seen_node_ids:
        raise _invalid(f"action references unknown node '{action.node_id}'")

    if edge_count > MAX_GRAPH_EDGES:
        raise ModelGraphError(
            ReasonCode.RESOURCE_CAP_EXCEEDED,
            f"graph has {edge_count} edges (cap {MAX_GRAPH_EDGES})",
        )

    topo_order = _topological_order(nodes)
    return GraphManifest(
        family_id=family_id,
        models=tuple(models),
        memory=tuple(memory),
        nodes=tuple(nodes),
        action=action,
        topo_order=topo_order,
    )


def _topological_order(nodes: list[NodeDecl]) -> tuple[str, ...]:
    """Kahn's algorithm over same-tick edges; ascending node_id breaks ties.

    Memory bindings read the previous tick and are not edges here.
    """
    dependencies: dict[str, set[str]] = {n.node_id: set() for n in nodes}
    dependents: dict[str, set[str]] = {n.node_id: set() for n in nodes}
    for node in nodes:
        for binding in node.inputs.values():
            if isinstance(binding, NodeBinding):
                if binding.node_id == node.node_id:
                    raise ModelGraphError(
                        ReasonCode.GRAPH_CYCLE,
                        "node reads its own same-tick output; use a memory slot",
                        node_id=node.node_id,
                    )
                dependencies[node.node_id].add(binding.node_id)
                dependents[binding.node_id].add(node.node_id)

    ready = sorted(nid for nid, deps in dependencies.items() if not deps)
    order: list[str] = []
    remaining = {nid: set(deps) for nid, deps in dependencies.items()}
    while ready:
        nid = ready.pop(0)
        order.append(nid)
        newly_ready = []
        for dep in dependents[nid]:
            remaining[dep].discard(nid)
            if not remaining[dep]:
                newly_ready.append(dep)
        if newly_ready:
            ready = sorted(set(ready) | set(newly_ready))
    if len(order) != len(nodes):
        stuck = sorted(set(dependencies) - set(order))
        raise ModelGraphError(
            ReasonCode.GRAPH_CYCLE, f"cycle involving nodes {stuck}"
        )
    return tuple(order)
