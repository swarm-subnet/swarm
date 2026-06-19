from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from swarm.constants import SIM_DT
from swarm.core.observation import smoke_observation
from swarm.core.submission_policy import validate_submission_zip
from swarm.domain_model import (
    CHALLENGE_FAMILY_IDS,
    UnknownChallengeFamilyError,
    UnknownPolicyInterfaceError,
    get_policy_interface_contract,
    get_supported_interface_versions,
)

POLICY_CONTRACT_FILENAME = "swarm_policy_contract.json"
POLICY_CONTRACT_VERSION = "policy_contract.v1"


class PolicyInterfaceError(ValueError):
    """Raised when an artifact policy contract is invalid."""


def resolve_policy_interface_version(
    family_id: str,
    requested_version: str | None = None,
) -> str:
    if family_id not in CHALLENGE_FAMILY_IDS:
        raise PolicyInterfaceError(f"unknown_challenge_family:{family_id}")
    supported = get_supported_interface_versions(family_id)
    if requested_version is None:
        return supported[0]
    if requested_version not in supported:
        raise PolicyInterfaceError(
            f"unsupported_interface_version:{family_id}:{requested_version}"
        )
    return requested_version


def build_artifact_policy_contract(
    family_id: str,
    interface_version: str,
) -> dict[str, Any]:
    canonical = get_policy_interface_contract(family_id, interface_version)
    return {
        "contract_version": canonical["contract_version"],
        "family_id": canonical["family_id"],
        "interface_version": canonical["interface_version"],
        "entry_point": canonical["entry_point"],
        "observation_space": canonical["observation_space"],
        "action_space": canonical["action_space"],
    }


def render_artifact_policy_contract(
    family_id: str,
    interface_version: str,
) -> str:
    payload = build_artifact_policy_contract(family_id, interface_version)
    return json.dumps(payload, indent=2) + "\n"


def read_policy_contract_from_zip(zip_path: Path) -> dict[str, Any]:
    try:
        with zipfile.ZipFile(zip_path) as zf:
            try:
                raw = zf.read(POLICY_CONTRACT_FILENAME)
            except KeyError as exc:
                raise PolicyInterfaceError(
                    f"missing_policy_contract:{POLICY_CONTRACT_FILENAME}"
                ) from exc
    except zipfile.BadZipFile as exc:
        raise PolicyInterfaceError("invalid_policy_package:corrupt_zip") from exc

    try:
        payload = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise PolicyInterfaceError("invalid_policy_contract_json") from exc

    if not isinstance(payload, dict):
        raise PolicyInterfaceError("invalid_policy_contract:expected_object")
    return payload


def validate_policy_contract_payload(payload: dict[str, Any]) -> dict[str, Any]:
    family_id = payload.get("family_id")
    if not isinstance(family_id, str) or not family_id:
        raise PolicyInterfaceError("invalid_policy_contract:family_id")

    interface_version = payload.get("interface_version")
    if not isinstance(interface_version, str) or not interface_version:
        raise PolicyInterfaceError("invalid_policy_contract:interface_version")

    try:
        expected = build_artifact_policy_contract(family_id, interface_version)
    except UnknownChallengeFamilyError as exc:
        raise PolicyInterfaceError(str(exc).strip("'")) from exc
    except UnknownPolicyInterfaceError:
        raise PolicyInterfaceError(
            f"unsupported_interface_version:{family_id}:{interface_version}"
        ) from None

    if payload.get("contract_version") != POLICY_CONTRACT_VERSION:
        raise PolicyInterfaceError(
            f"unsupported_policy_contract_version:{payload.get('contract_version')}"
        )

    expected_keys = set(expected)
    actual_keys = set(payload)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        suffix_parts: list[str] = []
        if missing:
            suffix_parts.append(f"missing={','.join(missing)}")
        if extra:
            suffix_parts.append(f"extra={','.join(extra)}")
        raise PolicyInterfaceError(
            "policy_contract_fields_mismatch:" + ";".join(suffix_parts)
        )

    if payload["entry_point"] != expected["entry_point"]:
        raise PolicyInterfaceError(
            f"entry_point_mismatch:{family_id}:{interface_version}"
        )
    if payload["observation_space"] != expected["observation_space"]:
        raise PolicyInterfaceError(
            f"observation_space_mismatch:{family_id}:{interface_version}"
        )
    if payload["action_space"] != expected["action_space"]:
        raise PolicyInterfaceError(f"action_space_mismatch:{family_id}:{interface_version}")

    return expected


def verify_policy_package_contract(
    zip_path: Path,
) -> tuple[bool, str, dict[str, Any] | None]:
    ok, reason = validate_submission_zip(zip_path)
    if not ok:
        return False, reason, None

    try:
        payload = read_policy_contract_from_zip(zip_path)
        contract = validate_policy_contract_payload(payload)
    except PolicyInterfaceError as exc:
        return False, str(exc), None
    return True, "ok", contract


def build_smoke_test_observation(
    family_id: str,
    interface_version: str,
    num_drones: int | None = None,
) -> dict[str, np.ndarray]:
    contract = get_policy_interface_contract(family_id, interface_version)
    action_dim = int(contract["action_space"]["shape"][-1])
    ctrl_freq = int(round(1.0 / SIM_DT))
    fills = {
        key: float(spec.get("fill", 0.0))
        for key, spec in contract.get("smoke_test_observation", {}).items()
    }
    obs = smoke_observation(
        contract["observation_assembly"],
        ctrl_freq=ctrl_freq,
        action_dim=action_dim,
        fills=fills,
    )
    if num_drones is not None:
        obs = {key: np.stack([value] * int(num_drones), axis=0) for key, value in obs.items()}
    return obs


def validate_action_output(
    action: Any,
    action_space: dict[str, Any],
    num_drones: int | None = None,
) -> np.ndarray:
    action_array = np.asarray(action, dtype=np.float32)
    expected_shape = tuple(action_space["shape"])
    if num_drones is not None:
        expected_shape = tuple(
            int(num_drones) if dim == "dynamic" else dim for dim in expected_shape
        )
    shape_ok = len(action_array.shape) == len(expected_shape) and all(
        dim == "dynamic" or int(dim) == actual
        for dim, actual in zip(expected_shape, action_array.shape)
    )
    if not shape_ok:
        raise PolicyInterfaceError(
            f"invalid_action_shape:{list(action_array.shape)}!=:{list(expected_shape)}"
        )
    if not np.isfinite(action_array).all():
        raise PolicyInterfaceError("invalid_action_values:not_finite")

    lower_bound = np.asarray(action_space["lower_bound"], dtype=np.float32)
    upper_bound = np.asarray(action_space["upper_bound"], dtype=np.float32)
    if np.any(action_array < (lower_bound - 1e-6)) or np.any(
        action_array > (upper_bound + 1e-6)
    ):
        raise PolicyInterfaceError("invalid_action_values:out_of_bounds")
    return action_array


def smoke_test_policy_package(
    zip_path: Path,
) -> tuple[bool, str]:
    ok, reason, contract = verify_policy_package_contract(zip_path)
    if not ok or contract is None:
        return False, reason

    family_id = str(contract["family_id"])
    interface_version = str(contract["interface_version"])
    canonical = get_policy_interface_contract(family_id, interface_version)
    num_range = canonical.get("num_drones_range")
    smoke_counts = sorted({int(num_range[0]), int(num_range[-1])}) if num_range else [None]

    with tempfile.TemporaryDirectory(prefix="swarm_policy_smoke_") as tmpdir:
        extract_dir = Path(tmpdir)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)

        entry_point = contract["entry_point"]
        module_name = str(entry_point["module"])
        class_name = str(entry_point["class_name"])
        act_method_name = str(entry_point["act_method"])
        reset_method_name = str(entry_point["reset_method"])

        module_path = extract_dir / f"{module_name}.py"
        if not module_path.is_file():
            return False, f"entry_point_missing_module:{module_name}"

        synthetic_module_name = f"_swarm_smoke_{uuid.uuid4().hex}"
        sys.path.insert(0, str(extract_dir))
        try:
            spec = importlib.util.spec_from_file_location(
                synthetic_module_name,
                module_path,
            )
            if spec is None or spec.loader is None:
                return False, f"entry_point_import_failed:{module_name}"
            module = importlib.util.module_from_spec(spec)
            sys.modules[synthetic_module_name] = module
            spec.loader.exec_module(module)
        except Exception as exc:
            return False, f"entry_point_import_failed:{module_name}:{exc}"
        finally:
            sys.path.pop(0)
            sys.modules.pop(synthetic_module_name, None)

        controller_class = getattr(module, class_name, None)
        if controller_class is None:
            return False, f"entry_point_missing_class:{class_name}"

        try:
            controller = controller_class()
        except Exception as exc:
            return False, f"controller_init_failed:{exc}"

        reset_method = getattr(controller, reset_method_name, None)
        if not callable(reset_method):
            return False, f"missing_reset_method:{reset_method_name}"
        act_method = getattr(controller, act_method_name, None)
        if not callable(act_method):
            return False, f"missing_act_method:{act_method_name}"

        try:
            for n_drones in smoke_counts:
                reset_method()
                observation = build_smoke_test_observation(
                    family_id, interface_version, num_drones=n_drones,
                )
                action = act_method(observation)
                validate_action_output(action, contract["action_space"], num_drones=n_drones)
        except PolicyInterfaceError as exc:
            return False, str(exc)
        except Exception as exc:
            return False, f"runtime_smoke_failed:{exc}"

    return True, "ok"


__all__ = [
    "POLICY_CONTRACT_FILENAME",
    "POLICY_CONTRACT_VERSION",
    "PolicyInterfaceError",
    "build_artifact_policy_contract",
    "build_smoke_test_observation",
    "read_policy_contract_from_zip",
    "render_artifact_policy_contract",
    "resolve_policy_interface_version",
    "smoke_test_policy_package",
    "validate_action_output",
    "validate_policy_contract_payload",
    "verify_policy_package_contract",
]
