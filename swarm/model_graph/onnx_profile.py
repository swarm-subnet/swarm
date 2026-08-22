"""Static ONNX inspection against the published execution profile.

Reads the protobuf and accepts only the operations listed in the profile,
returning the model's IO signature plus static costs. Never creates an ORT
session.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import onnx
from onnx import TensorProto, numpy_helper

from .constants import (
    MAX_INITIALIZER_BYTES,
    MAX_ONNX_INITIALIZERS_TOTAL,
    MAX_ONNX_NODES_PER_MODEL,
    MAX_TENSOR_ELEMENTS,
    MAX_TENSOR_RANK,
    ONNX_IR_VERSION,
    ONNX_OPSET_VERSION,
    SWARM_AXIS_DIM,
    SWARM_AXIS_TOKEN,
    SWARM_NUM_DRONES_RANGE,
)
from .errors import ModelGraphError, ReasonCode

_PROFILE_PATH = Path(__file__).with_name("execution_profile.v1.json")

_CONSTANT_INPUT_INDICES = {
    "Reshape": (1,),
    "Squeeze": (1,),
    "Unsqueeze": (1,),
    "Split": (1,),
    "Slice": (1, 2, 3, 4),
    "Pad": (1, 2, 3),
    "Expand": (1,),
    "Resize": (1, 2, 3),
    "Gather": (1,),
}
_ALLOWED_RESIZE_MODES = frozenset({"nearest", "linear"})


@lru_cache(maxsize=1)
def load_profile() -> dict:
    return json.loads(_PROFILE_PATH.read_bytes().decode("utf-8"))


@lru_cache(maxsize=1)
def profile_digest() -> str:
    return hashlib.sha256(_PROFILE_PATH.read_bytes()).hexdigest()


@dataclass(frozen=True)
class TensorSignature:
    name: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class ModelInspection:
    inputs: tuple[TensorSignature, ...]
    outputs: tuple[TensorSignature, ...]
    node_count: int
    initializer_count: int
    initializer_bytes: int
    intermediate_bytes: int
    flops: int


def _denied(reason: ReasonCode, detail: str, model_id: str | None) -> ModelGraphError:
    return ModelGraphError(reason, detail, model_id=model_id)


def _worst_case_elements(dims) -> int:
    """Element count with the symbolic drone axis at its maximum."""
    total = 1
    for d in dims:
        total *= SWARM_NUM_DRONES_RANGE[1] if d == SWARM_AXIS_DIM else d
    return total


def _static_shape(value_info, model_id: str | None) -> tuple[int, ...]:
    """Return the tensor shape; the swarm drone axis is encoded as SWARM_AXIS_DIM.

    Only the leading dimension may be symbolic, and only when named exactly
    ``N``. Every other dimension must be a positive literal so allocation is
    bounded before a session exists.
    """
    ttype = value_info.type.tensor_type
    if ttype.elem_type != TensorProto.FLOAT:
        raise _denied(
            ReasonCode.PROFILE_DENIED_DTYPE,
            f"tensor '{value_info.name}' is not float32",
            model_id,
        )
    dims: list[int] = []
    for axis, dim in enumerate(ttype.shape.dim):
        kind = dim.WhichOneof("value")
        if kind == "dim_param" and axis == 0 and dim.dim_param == SWARM_AXIS_TOKEN:
            dims.append(SWARM_AXIS_DIM)
            continue
        if kind != "dim_value" or dim.dim_value < 1:
            raise _denied(
                ReasonCode.SHAPE_INVALID,
                f"tensor '{value_info.name}' has a symbolic or non-positive dimension",
                model_id,
            )
        dims.append(int(dim.dim_value))
    if len(dims) > MAX_TENSOR_RANK:
        raise _denied(
            ReasonCode.SHAPE_INVALID,
            f"tensor '{value_info.name}' rank {len(dims)} exceeds {MAX_TENSOR_RANK}",
            model_id,
        )
    elements = _worst_case_elements(dims)
    if elements > MAX_TENSOR_ELEMENTS:
        raise _denied(
            ReasonCode.RESOURCE_CAP_EXCEEDED,
            f"tensor '{value_info.name}' has {elements} elements (cap {MAX_TENSOR_ELEMENTS})",
            model_id,
        )
    return tuple(dims)


def _node_flops(node, shapes: dict[str, tuple[int, ...]]) -> int:
    if node.op_type == "Constant":
        return 0

    def elements(name: str) -> int:
        shape = shapes.get(name)
        if not shape:
            return 0
        return _worst_case_elements(shape)

    if node.op_type in {"MatMul", "Gemm"}:
        out = elements(node.output[0])
        a_shape = shapes.get(node.input[0], ())
        inner = a_shape[-1] if a_shape else 1
        return 2 * out * inner
    if node.op_type in {"Conv", "ConvTranspose"}:
        out = elements(node.output[0])
        w_shape = shapes.get(node.input[1], ())
        per_output = 1
        for d in w_shape[1:]:
            per_output *= d
        return 2 * out * max(per_output, 1)
    if node.op_type in {"LSTM", "GRU", "RNN"}:
        total = 0
        for name in node.input[1:3]:
            w_shape = shapes.get(name, ())
            n = 1
            for d in w_shape:
                n *= d
            total += 2 * n
        return total
    return elements(node.output[0]) if node.output else 0


def inspect_onnx_bytes(data: bytes, *, model_id: str | None = None) -> ModelInspection:
    profile = load_profile()
    unconditional = frozenset(profile["unconditional_ops"])
    restricted = frozenset(profile["restricted_ops"])

    try:
        model = onnx.load_model_from_string(data)
    except Exception as exc:
        raise _denied(ReasonCode.MODEL_FILE_INVALID, f"unparsable ONNX: {exc}", model_id) from exc

    if model.ir_version != ONNX_IR_VERSION:
        raise _denied(
            ReasonCode.PROFILE_BAD_OPSET,
            f"ir_version {model.ir_version} != {ONNX_IR_VERSION}",
            model_id,
        )
    if len(model.opset_import) != 1:
        raise _denied(
            ReasonCode.PROFILE_DENIED_DOMAIN,
            "exactly one opset import (default domain) is allowed",
            model_id,
        )
    opset = model.opset_import[0]
    if opset.domain not in ("", "ai.onnx"):
        raise _denied(
            ReasonCode.PROFILE_DENIED_DOMAIN, f"domain '{opset.domain}' is not allowed", model_id
        )
    if opset.version != ONNX_OPSET_VERSION:
        raise _denied(
            ReasonCode.PROFILE_BAD_OPSET,
            f"opset {opset.version} != {ONNX_OPSET_VERSION}",
            model_id,
        )
    if model.functions:
        raise _denied(
            ReasonCode.PROFILE_DENIED_FEATURE, "local functions are not allowed", model_id
        )

    graph = model.graph
    if graph.sparse_initializer:
        raise _denied(
            ReasonCode.PROFILE_DENIED_FEATURE, "sparse initializers are not allowed", model_id
        )

    initializer_bytes = 0
    constant_names: set[str] = set()
    for init in graph.initializer:
        if init.data_location == TensorProto.EXTERNAL:
            raise _denied(
                ReasonCode.PROFILE_DENIED_FEATURE,
                f"initializer '{init.name}' uses external data",
                model_id,
            )
        if init.data_type != TensorProto.FLOAT and init.data_type != TensorProto.INT64:
            raise _denied(
                ReasonCode.PROFILE_DENIED_DTYPE,
                f"initializer '{init.name}' is neither float32 nor int64 metadata",
                model_id,
            )
        try:
            array = numpy_helper.to_array(init)
        except Exception as exc:
            raise _denied(
                ReasonCode.MODEL_FILE_INVALID,
                f"initializer '{init.name}' payload is malformed: {exc}",
                model_id,
            ) from exc
        initializer_bytes += array.nbytes
        constant_names.add(init.name)
    if len(graph.initializer) > MAX_ONNX_INITIALIZERS_TOTAL:
        raise _denied(
            ReasonCode.RESOURCE_CAP_EXCEEDED,
            f"{len(graph.initializer)} initializers (cap {MAX_ONNX_INITIALIZERS_TOTAL})",
            model_id,
        )
    if initializer_bytes > MAX_INITIALIZER_BYTES:
        raise _denied(
            ReasonCode.RESOURCE_CAP_EXCEEDED,
            f"initializer bytes {initializer_bytes} exceed {MAX_INITIALIZER_BYTES}",
            model_id,
        )

    if len(graph.node) > MAX_ONNX_NODES_PER_MODEL:
        raise _denied(
            ReasonCode.RESOURCE_CAP_EXCEEDED,
            f"{len(graph.node)} nodes (cap {MAX_ONNX_NODES_PER_MODEL})",
            model_id,
        )

    initializer_count = len(graph.initializer)
    constant_shapes: dict[str, tuple[int, ...]] = {}
    for node in graph.node:
        if node.domain not in ("", "ai.onnx"):
            raise _denied(
                ReasonCode.PROFILE_DENIED_DOMAIN,
                f"op '{node.op_type}' from domain '{node.domain}' is not allowed",
                model_id,
            )
        for attr in node.attribute:
            if attr.g.ByteSize() or attr.graphs:
                raise _denied(
                    ReasonCode.PROFILE_DENIED_FEATURE,
                    f"op '{node.op_type}' carries a subgraph",
                    model_id,
                )
        if node.op_type == "Constant":
            if len(node.output) != 1:
                raise _denied(
                    ReasonCode.MODEL_FILE_INVALID,
                    "Constant must declare exactly one output",
                    model_id,
                )
            output_name = node.output[0]
            tensor = None
            for attr in node.attribute:
                if attr.name != "value":
                    raise _denied(
                        ReasonCode.PROFILE_DENIED_FEATURE,
                        f"Constant attribute '{attr.name}' is not allowed",
                        model_id,
                    )
                tensor = attr.t
            if tensor is None:
                raise _denied(
                    ReasonCode.PROFILE_DENIED_FEATURE,
                    "Constant must carry a dense 'value' tensor",
                    model_id,
                )
            if tensor.data_location == TensorProto.EXTERNAL:
                raise _denied(
                    ReasonCode.PROFILE_DENIED_FEATURE,
                    f"Constant '{output_name}' uses external data",
                    model_id,
                )
            if tensor.data_type != TensorProto.FLOAT and tensor.data_type != TensorProto.INT64:
                raise _denied(
                    ReasonCode.PROFILE_DENIED_DTYPE,
                    f"Constant '{output_name}' is neither float32 nor int64 metadata",
                    model_id,
                )
            try:
                array = numpy_helper.to_array(tensor)
            except Exception as exc:
                raise _denied(
                    ReasonCode.MODEL_FILE_INVALID,
                    f"Constant '{output_name}' payload is malformed: {exc}",
                    model_id,
                ) from exc
            initializer_bytes += array.nbytes
            initializer_count += 1
            constant_shapes[output_name] = tuple(int(d) for d in tensor.dims)
            constant_names.add(output_name)
            continue
        if node.op_type in unconditional:
            pass
        elif node.op_type in restricted:
            for idx in _CONSTANT_INPUT_INDICES.get(node.op_type, ()):
                if idx < len(node.input) and node.input[idx]:
                    if node.input[idx] not in constant_names:
                        raise _denied(
                            ReasonCode.PROFILE_DENIED_FEATURE,
                            f"op '{node.op_type}' requires a constant input at index {idx}",
                            model_id,
                        )
            if node.op_type == "Resize":
                mode = next(
                    (a.s.decode() for a in node.attribute if a.name == "mode"), "nearest"
                )
                if mode not in _ALLOWED_RESIZE_MODES:
                    raise _denied(
                        ReasonCode.PROFILE_DENIED_FEATURE,
                        f"Resize mode '{mode}' is not allowed",
                        model_id,
                    )
            if node.op_type == "Pad":
                mode = next(
                    (a.s.decode() for a in node.attribute if a.name == "mode"), "constant"
                )
                if mode != "constant":
                    raise _denied(
                        ReasonCode.PROFILE_DENIED_FEATURE,
                        f"Pad mode '{mode}' is not allowed",
                        model_id,
                    )
        else:
            raise _denied(
                ReasonCode.PROFILE_DENIED_OP,
                f"operator '{node.op_type}' is not in the execution profile",
                model_id,
            )

    if initializer_count > MAX_ONNX_INITIALIZERS_TOTAL:
        raise _denied(
            ReasonCode.RESOURCE_CAP_EXCEEDED,
            f"{initializer_count} initializers (cap {MAX_ONNX_INITIALIZERS_TOTAL})",
            model_id,
        )
    if initializer_bytes > MAX_INITIALIZER_BYTES:
        raise _denied(
            ReasonCode.RESOURCE_CAP_EXCEEDED,
            f"initializer bytes {initializer_bytes} exceed {MAX_INITIALIZER_BYTES}",
            model_id,
        )

    try:
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    except Exception as exc:
        raise _denied(
            ReasonCode.SHAPE_INVALID, f"shape inference failed: {exc}", model_id
        ) from exc

    shapes: dict[str, tuple[int, ...]] = {}
    for value_info in list(inferred.graph.input) + list(inferred.graph.output):
        shapes[value_info.name] = _static_shape(value_info, model_id)
    intermediate_bytes = 0
    for value_info in inferred.graph.value_info:
        if value_info.name in constant_names:
            continue
        shape = _static_shape(value_info, model_id)
        shapes[value_info.name] = shape
        intermediate_bytes += _worst_case_elements(shape) * 4
    shapes.update(constant_shapes)
    for init in graph.initializer:
        shapes[init.name] = tuple(int(d) for d in init.dims)

    flops = sum(_node_flops(node, shapes) for node in graph.node)

    inputs = tuple(
        TensorSignature(v.name, shapes[v.name]) for v in inferred.graph.input
    )
    outputs = tuple(
        TensorSignature(v.name, shapes[v.name]) for v in inferred.graph.output
    )
    if not outputs:
        raise _denied(ReasonCode.MODEL_FILE_INVALID, "model declares no outputs", model_id)

    return ModelInspection(
        inputs=inputs,
        outputs=outputs,
        node_count=len(graph.node),
        initializer_count=initializer_count,
        initializer_bytes=initializer_bytes,
        intermediate_bytes=intermediate_bytes,
        flops=flops,
    )
