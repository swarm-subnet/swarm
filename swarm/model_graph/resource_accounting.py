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

"""Deterministic aggregate resource accounting for admitted ONNX graphs."""

from __future__ import annotations

from dataclasses import dataclass

from .constants import (
    MAX_INITIALIZER_BYTES,
    MAX_INTERMEDIATE_TOTAL_BYTES,
    MAX_MEMORY_FLOATS,
    MAX_ONNX_INITIALIZERS_TOTAL,
    MAX_ONNX_NODES_TOTAL,
    MAX_ONNX_TOTAL_BYTES,
    MAX_TICK_FLOPS,
)
from .errors import ModelGraphError, ReasonCode
from .manifest import GraphManifest
from .onnx_profile import ModelInspection


@dataclass(frozen=True)
class ResourceTotals:
    onnx_bytes: int
    initializer_bytes: int
    onnx_nodes: int
    initializers: int
    intermediate_bytes: int
    memory_floats: int
    worst_tick_flops: int


def account_resources(
    manifest: GraphManifest,
    inspections: dict[str, ModelInspection],
    model_sizes: dict[str, int],
) -> ResourceTotals:
    """Count per-file storage once and per-node execution work once.

    Multiple graph nodes may call the same model declaration. Initializers and
    intermediate allocations are counted once per declared model, while FLOPs
    are counted once per graph node because tick zero executes every node.
    """
    totals = ResourceTotals(
        onnx_bytes=sum(model_sizes[m.model_id] for m in manifest.models),
        initializer_bytes=sum(inspections[m.model_id].initializer_bytes for m in manifest.models),
        onnx_nodes=sum(inspections[m.model_id].node_count for m in manifest.models),
        initializers=sum(inspections[m.model_id].initializer_count for m in manifest.models),
        intermediate_bytes=sum(inspections[m.model_id].intermediate_bytes for m in manifest.models),
        memory_floats=manifest.memory_floats,
        worst_tick_flops=sum(inspections[n.model_id].flops for n in manifest.nodes),
    )
    caps = (
        ("total ONNX bytes", totals.onnx_bytes, MAX_ONNX_TOTAL_BYTES),
        ("total initializer bytes", totals.initializer_bytes, MAX_INITIALIZER_BYTES),
        ("total ONNX nodes", totals.onnx_nodes, MAX_ONNX_NODES_TOTAL),
        ("total initializers", totals.initializers, MAX_ONNX_INITIALIZERS_TOTAL),
        ("intermediate bytes", totals.intermediate_bytes, MAX_INTERMEDIATE_TOTAL_BYTES),
        ("memory floats", totals.memory_floats, MAX_MEMORY_FLOATS),
        ("worst-tick FLOPs", totals.worst_tick_flops, MAX_TICK_FLOPS),
    )
    for label, actual, cap in caps:
        if actual > cap:
            raise ModelGraphError(
                ReasonCode.RESOURCE_CAP_EXCEEDED,
                f"{label} {actual} exceed {cap}",
            )
    return totals
