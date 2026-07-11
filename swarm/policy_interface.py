"""Graph-only policy contract helpers used by CLI and validator intake."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import numpy as np

from swarm.core.observation import smoke_observation
from swarm.domain_model import (
    CHALLENGE_FAMILY_IDS,
    get_policy_interface_contract,
    get_supported_interface_versions,
)
from swarm.model_graph import SUBMISSION_INTERFACE_VERSION, admit_artifact
from swarm.model_graph.action import canonicalize_action
from swarm.model_graph.constants import FAMILY_GRAPH_CONTRACTS
from swarm.model_graph.runner import GraphRunner
from swarm.model_graph.session_probe import probe_runner

POLICY_CONTRACT_FILENAME = "manifest.json"
POLICY_CONTRACT_VERSION = "model_graph.v1"


class PolicyInterfaceError(ValueError):
    pass


def resolve_policy_interface_version(
    family_id: str, requested_version: str | None = None
) -> str:
    if family_id not in CHALLENGE_FAMILY_IDS:
        raise PolicyInterfaceError(f"unknown_challenge_family:{family_id}")
    supported = get_supported_interface_versions(family_id)
    selected = requested_version or SUBMISSION_INTERFACE_VERSION
    if selected != SUBMISSION_INTERFACE_VERSION or selected not in supported:
        raise PolicyInterfaceError(f"unsupported_interface_version:{family_id}:{selected}")
    return selected


def build_artifact_policy_contract(
    family_id: str, interface_version: str = SUBMISSION_INTERFACE_VERSION
) -> dict[str, Any]:
    resolve_policy_interface_version(family_id, interface_version)
    return get_policy_interface_contract(family_id, interface_version)


def render_artifact_policy_contract(
    family_id: str, interface_version: str = SUBMISSION_INTERFACE_VERSION
) -> str:
    """Render the registry contract for tooling output, never for artifact execution."""
    return json.dumps(build_artifact_policy_contract(family_id, interface_version), indent=2) + "\n"


def verify_policy_package_contract(
    zip_path: Path,
) -> tuple[bool, str, dict[str, Any] | None]:
    result = admit_artifact(Path(zip_path))
    if not result.accepted or result.family_id is None:
        return False, f"{result.reason_code}:{result.detail}", None
    return True, "ok", build_artifact_policy_contract(result.family_id)


def smoke_test_policy_package(zip_path: Path) -> tuple[bool, str]:
    result = admit_artifact(Path(zip_path))
    if not result.accepted:
        return False, f"{result.reason_code}:{result.detail}"
    try:
        runner = GraphRunner.from_artifact(Path(zip_path))
        probe_runner(runner)
    except Exception as exc:
        return False, str(exc)
    return True, "ok"


def build_smoke_test_observation(
    family_id: str,
    interface_version: str = SUBMISSION_INTERFACE_VERSION,
    num_drones: int | None = None,
) -> dict[str, np.ndarray]:
    resolve_policy_interface_version(family_id, interface_version)
    canonical = get_policy_interface_contract(family_id, interface_version)
    action_dim = int(FAMILY_GRAPH_CONTRACTS[family_id]["action_shape"][-1])
    fills = {
        key: float(spec.get("fill", 0.0))
        for key, spec in canonical.get("smoke_test_observation", {}).items()
    }
    observation = smoke_observation(
        canonical["observation_assembly"], ctrl_freq=50, action_dim=action_dim, fills=fills
    )
    if num_drones is not None:
        observation = {
            key: np.stack([value] * int(num_drones), axis=0)
            for key, value in observation.items()
        }
    return {key: np.ascontiguousarray(value, dtype=np.float32) for key, value in observation.items()}


def validate_action_output(
    action: Any, action_space: dict[str, Any], num_drones: int | None = None
) -> np.ndarray:
    """Compatibility helper for family tests; graph execution uses action.py directly."""
    family_id = next(
        (
            fid for fid, contract in FAMILY_GRAPH_CONTRACTS.items()
            if list(contract["action_shape"]) == list(action_space["shape"])
            or list(action_space["shape"]) == ["dynamic", contract["action_shape"][-1]]
        ),
        None,
    )
    if family_id is None:
        raise PolicyInterfaceError("unknown_action_contract")
    try:
        return canonicalize_action(np.asarray(action), family_id, num_drones)
    except Exception as exc:
        raise PolicyInterfaceError(str(exc)) from exc


__all__ = [
    "POLICY_CONTRACT_FILENAME", "POLICY_CONTRACT_VERSION", "PolicyInterfaceError",
    "build_artifact_policy_contract", "build_smoke_test_observation",
    "render_artifact_policy_contract",
    "resolve_policy_interface_version", "smoke_test_policy_package",
    "validate_action_output", "verify_policy_package_contract",
]
