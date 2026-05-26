from __future__ import annotations

import json
import zipfile
from pathlib import Path

from swarm.policy_interface import (
    POLICY_CONTRACT_FILENAME,
    build_artifact_policy_contract,
    render_artifact_policy_contract,
    smoke_test_policy_package,
    verify_policy_package_contract,
)


def _write_zip(path: Path, files: dict[str, str]) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, content in files.items():
            zf.writestr(name, content)


def _base_controller_source(action_body: str) -> str:
    return "\n".join(
        [
            "import numpy as np",
            "",
            "class DroneFlightController:",
            "    def reset(self):",
            "        return None",
            "",
            "    def act(self, observation):",
            f"        {action_body}",
            "",
        ]
    )


def test_verify_policy_package_contract_accepts_generated_contract(tmp_path):
    zip_path = tmp_path / "submission.zip"
    _write_zip(
        zip_path,
        {
            "drone_agent.py": _base_controller_source(
                "return np.array([0.0, 0.0, 0.0, 0.5, 0.0], dtype=np.float32)"
            ),
            POLICY_CONTRACT_FILENAME: render_artifact_policy_contract(
                "cf_search_and_rescue",
                "submission_zip.v1",
            ),
        },
    )

    ok, reason, contract = verify_policy_package_contract(zip_path)

    assert ok is True
    assert reason == "ok"
    assert contract == build_artifact_policy_contract(
        "cf_search_and_rescue",
        "submission_zip.v1",
    )


def test_verify_policy_package_contract_rejects_missing_contract(tmp_path):
    zip_path = tmp_path / "submission.zip"
    _write_zip(
        zip_path,
        {
            "drone_agent.py": _base_controller_source(
                "return np.array([0.0, 0.0, 0.0, 0.5, 0.0], dtype=np.float32)"
            ),
        },
    )

    ok, reason, contract = verify_policy_package_contract(zip_path)

    assert ok is False
    assert reason == f"missing_policy_contract:{POLICY_CONTRACT_FILENAME}"
    assert contract is None


def test_verify_policy_package_contract_rejects_unsupported_interface_version(tmp_path):
    zip_path = tmp_path / "submission.zip"
    payload = build_artifact_policy_contract(
        "cf_search_and_rescue",
        "submission_zip.v1",
    )
    payload["interface_version"] = "submission_zip.v999"
    _write_zip(
        zip_path,
        {
            "drone_agent.py": _base_controller_source(
                "return np.array([0.0, 0.0, 0.0, 0.5, 0.0], dtype=np.float32)"
            ),
            POLICY_CONTRACT_FILENAME: json.dumps(payload, indent=2),
        },
    )

    ok, reason, contract = verify_policy_package_contract(zip_path)

    assert ok is False
    assert reason == "unsupported_interface_version:cf_search_and_rescue:submission_zip.v999"
    assert contract is None


def test_verify_policy_package_contract_rejects_observation_space_mismatch(tmp_path):
    zip_path = tmp_path / "submission.zip"
    payload = build_artifact_policy_contract(
        "cf_search_and_rescue",
        "submission_zip.v1",
    )
    payload["observation_space"]["fields"]["depth"]["shape"] = [64, 64, 1]
    _write_zip(
        zip_path,
        {
            "drone_agent.py": _base_controller_source(
                "return np.array([0.0, 0.0, 0.0, 0.5, 0.0], dtype=np.float32)"
            ),
            POLICY_CONTRACT_FILENAME: json.dumps(payload, indent=2),
        },
    )

    ok, reason, contract = verify_policy_package_contract(zip_path)

    assert ok is False
    assert reason == "observation_space_mismatch:cf_search_and_rescue:submission_zip.v1"
    assert contract is None


def test_smoke_test_policy_package_accepts_valid_controller(tmp_path):
    zip_path = tmp_path / "submission.zip"
    _write_zip(
        zip_path,
        {
            "drone_agent.py": _base_controller_source(
                "return np.array([0.0, 0.0, 0.0, 0.5, 0.0], dtype=np.float32)"
            ),
            POLICY_CONTRACT_FILENAME: render_artifact_policy_contract(
                "cf_search_and_rescue",
                "submission_zip.v1",
            ),
        },
    )

    assert smoke_test_policy_package(zip_path) == (True, "ok")


def test_smoke_test_policy_package_rejects_invalid_action_shape(tmp_path):
    zip_path = tmp_path / "submission.zip"
    _write_zip(
        zip_path,
        {
            "drone_agent.py": _base_controller_source(
                "return np.array([0.0, 0.0, 0.0, 0.5], dtype=np.float32)"
            ),
            POLICY_CONTRACT_FILENAME: render_artifact_policy_contract(
                "cf_search_and_rescue",
                "submission_zip.v1",
            ),
        },
    )

    ok, reason = smoke_test_policy_package(zip_path)

    assert ok is False
    assert reason == "invalid_action_shape:[4]!=:[5]"
