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

"""Static admission for model-graph artifacts.

Runs archive, manifest, ONNX-profile, family-shape, and resource checks and
returns either an accepted result or one stable rejection record. Identical
logic runs in the backend mirror; both must emit the same reason code.
"""

from __future__ import annotations

import hashlib
import multiprocessing
import os
import resource
from dataclasses import dataclass
from pathlib import Path

from .archive import read_archive
from .constants import (
    EXECUTION_PROFILE_ID,
    FAMILY_GRAPH_CONTRACTS,
    RUNNER_ABI,
    STATIC_ADMISSION_ADDRESS_BYTES,
    STATIC_ADMISSION_WALL_SEC,
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
from .onnx_profile import ModelInspection, inspect_onnx_bytes, profile_digest
from .resource_accounting import account_resources


@dataclass(frozen=True)
class AdmissionResult:
    accepted: bool
    reason_code: str
    detail: str = ""
    family_id: str | None = None
    artifact_sha256: str | None = None
    profile_id: str = EXECUTION_PROFILE_ID
    profile_digest: str = ""
    runner_abi: str = RUNNER_ABI
    node_id: str | None = None
    model_id: str | None = None
    worst_tick_flops: int = 0
    memory_floats: int = 0

    def to_record(self) -> dict:
        return {
            "accepted": self.accepted,
            "reason_code": self.reason_code,
            "detail": self.detail,
            "family_id": self.family_id,
            "artifact_sha256": self.artifact_sha256,
            "profile_id": self.profile_id,
            "profile_digest": self.profile_digest,
            "runner_abi": self.runner_abi,
            "node_id": self.node_id,
            "model_id": self.model_id,
            "worst_tick_flops": self.worst_tick_flops,
            "memory_floats": self.memory_floats,
        }


def _expected_shape(
    binding: Binding,
    manifest: GraphManifest,
    inspections: dict[str, ModelInspection],
    num_drones: int | None = None,
) -> tuple[int, ...] | None:
    contract = FAMILY_GRAPH_CONTRACTS[manifest.family_id]
    if isinstance(binding, ObsBinding):
        return resolve_shape(contract["observations"][binding.key], num_drones)
    if isinstance(binding, MemoryBinding):
        for slot in manifest.memory:
            if slot.name == binding.name:
                return slot.shape
        return None
    if isinstance(binding, NodeBinding):
        node = manifest.node(binding.node_id)
        inspection = inspections[node.model_id]
        for out in inspection.outputs:
            if out.name == binding.output:
                return out.shape
        raise ModelGraphError(
            ReasonCode.GRAPH_DANGLING_OUTPUT,
            f"node '{binding.node_id}' has no output named '{binding.output}'",
            node_id=binding.node_id,
        )
    return None


def _check_graph_against_models(
    manifest: GraphManifest,
    inspections: dict[str, ModelInspection],
    num_drones: int | None = None,
) -> int:
    total_flops = 0
    for node in manifest.nodes:
        inspection = inspections[node.model_id]
        declared = {sig.name: sig.shape for sig in inspection.inputs}
        if set(declared) != set(node.inputs):
            missing = sorted(set(declared) - set(node.inputs))
            extra = sorted(set(node.inputs) - set(declared))
            raise ModelGraphError(
                ReasonCode.GRAPH_UNBOUND_INPUT,
                f"binding mismatch (unbound={missing}, unknown={extra})",
                node_id=node.node_id,
                model_id=node.model_id,
            )
        for input_name, binding in node.inputs.items():
            expected = _expected_shape(binding, manifest, inspections, num_drones)
            if expected is None:
                raise ModelGraphError(
                    ReasonCode.GRAPH_MEMORY_INVALID,
                    f"input '{input_name}' binds to an unresolvable source",
                    node_id=node.node_id,
                )
            if declared[input_name] != expected:
                raise ModelGraphError(
                    ReasonCode.SHAPE_CONTRACT_MISMATCH,
                    f"input '{input_name}' expects {expected}, model declares {declared[input_name]}",
                    node_id=node.node_id,
                    model_id=node.model_id,
                )
        total_flops += inspection.flops

    for slot in manifest.memory:
        writer_shape = _expected_shape(slot.writer, manifest, inspections, num_drones)
        if writer_shape != slot.shape:
            raise ModelGraphError(
                ReasonCode.GRAPH_MEMORY_INVALID,
                f"memory '{slot.name}' declares {slot.shape} but its writer produces {writer_shape}",
                node_id=slot.writer.node_id,
            )

    contract = FAMILY_GRAPH_CONTRACTS[manifest.family_id]
    action_shape = _expected_shape(manifest.action, manifest, inspections, num_drones)
    expected_action = resolve_shape(contract["action_shape"], num_drones)
    if action_shape != expected_action:
        raise ModelGraphError(
            ReasonCode.SHAPE_CONTRACT_MISMATCH,
            f"action must be {expected_action}, graph produces {action_shape}",
            node_id=manifest.action.node_id,
        )

    return total_flops


def _consumed_outputs(manifest: GraphManifest) -> set[tuple[str, str]]:
    consumed: set[tuple[str, str]] = {(manifest.action.node_id, manifest.action.output)}
    for node in manifest.nodes:
        for binding in node.inputs.values():
            if isinstance(binding, NodeBinding):
                consumed.add((binding.node_id, binding.output))
    for slot in manifest.memory:
        consumed.add((slot.writer.node_id, slot.writer.output))
    return consumed


def admit_artifact(zip_path: Path) -> AdmissionResult:
    artifact_sha = hashlib.sha256(Path(zip_path).read_bytes()).hexdigest()
    digest = profile_digest()
    try:
        contents = read_archive(Path(zip_path))
        manifest = parse_manifest(contents.manifest_bytes)

        declared_files = {model.file for model in manifest.models}
        if declared_files != set(contents.files):
            undeclared = sorted(set(contents.files) - declared_files)
            missing = sorted(declared_files - set(contents.files))
            raise ModelGraphError(
                ReasonCode.ARCHIVE_BAD_LAYOUT,
                f"archive/model mismatch (undeclared={undeclared}, missing={missing})",
            )

        model_sizes: dict[str, int] = {}
        inspections: dict[str, ModelInspection] = {}
        for model in manifest.models:
            data = contents.files[model.file]
            actual = hashlib.sha256(data).hexdigest()
            if actual != model.sha256:
                raise ModelGraphError(
                    ReasonCode.MODEL_HASH_MISMATCH,
                    f"'{model.file}' hash mismatch",
                    model_id=model.model_id,
                )
            model_sizes[model.model_id] = len(data)
            inspection = inspect_onnx_bytes(data, model_id=model.model_id)
            inspections[model.model_id] = inspection

        consumed = _consumed_outputs(manifest)
        for node in manifest.nodes:
            for out in inspections[node.model_id].outputs:
                if (node.node_id, out.name) not in consumed:
                    raise ModelGraphError(
                        ReasonCode.GRAPH_DANGLING_OUTPUT,
                        f"output '{out.name}' is never consumed",
                        node_id=node.node_id,
                    )

        _check_graph_against_models(manifest, inspections)
        totals = account_resources(manifest, inspections, model_sizes)

    except ModelGraphError as exc:
        return AdmissionResult(
            accepted=False,
            reason_code=exc.reason.value,
            detail=exc.detail,
            artifact_sha256=artifact_sha,
            profile_digest=digest,
            node_id=exc.node_id,
            model_id=exc.model_id,
        )

    return AdmissionResult(
        accepted=True,
        reason_code=ReasonCode.OK.value,
        family_id=manifest.family_id,
        artifact_sha256=artifact_sha,
        profile_digest=digest,
        worst_tick_flops=totals.worst_tick_flops,
        memory_floats=manifest.memory_floats,
    )


def _admission_child(zip_path: str, connection) -> None:
    try:
        if os.name == "posix":
            resource.setrlimit(
                resource.RLIMIT_AS,
                (STATIC_ADMISSION_ADDRESS_BYTES, STATIC_ADMISSION_ADDRESS_BYTES),
            )
        record = admit_artifact(Path(zip_path)).to_record()
    except MemoryError:
        record = AdmissionResult(
            accepted=False,
            reason_code=ReasonCode.RESOURCE_CAP_EXCEEDED.value,
            detail="static admission exceeded the address-space budget",
        ).to_record()
    except BaseException as exc:
        record = {"crash": f"{type(exc).__name__}: {exc}"}
    try:
        connection.send(record)
    except OSError:
        pass
    finally:
        connection.close()


def admit_artifact_subprocess(
    zip_path: Path, timeout: float = STATIC_ADMISSION_WALL_SEC
) -> AdmissionResult:
    """Run static admission under an address-space and wall-time boundary."""
    if multiprocessing.current_process().daemon:
        # Daemonic workers cannot spawn children; their artifacts already passed boxed intake.
        return admit_artifact(Path(zip_path))
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_admission_child, args=(str(Path(zip_path)), child))
    process.start()
    child.close()
    try:
        if not parent.poll(timeout):
            return AdmissionResult(
                accepted=False,
                reason_code=ReasonCode.RESOURCE_CAP_EXCEEDED.value,
                detail=f"static admission exceeded {timeout:.1f}s",
            )
        try:
            record = parent.recv()
        except (EOFError, OSError):
            return AdmissionResult(
                accepted=False,
                reason_code=ReasonCode.RESOURCE_CAP_EXCEEDED.value,
                detail="static admission terminated before producing a result",
            )
    finally:
        parent.close()
        process.join(2)
        if process.is_alive():
            process.kill()
            process.join()
    if "crash" in record:
        raise RuntimeError(f"static admission failed: {record['crash']}")
    return AdmissionResult(**record)
