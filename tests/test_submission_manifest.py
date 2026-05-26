import json
import zipfile
from pathlib import Path

from swarm.submission_manifest import (
    REPO_LAYOUT_RULES,
    SUBMISSION_MANIFEST_FILENAME,
    SUBMISSION_MANIFEST_VERSION,
    SubmissionArtifact,
    build_submission_manifest_payload,
    parse_submission_manifest_payload,
    validate_submission_repo,
)


def _write_zip(path: Path, files: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, content in files.items():
            zf.writestr(name, content)


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _valid_manifest_payload(artifact_path: str, sha256: str) -> dict:
    return {
        "manifest_version": SUBMISSION_MANIFEST_VERSION,
        "repo_layout_rules": dict(REPO_LAYOUT_RULES),
        "artifacts": [
            {
                "family_id": "cf_search_and_rescue",
                "interface_version": "submission_zip.v1",
                "artifact_path": artifact_path,
                "sha256": sha256,
                "metadata": {
                    "notes": "baseline SAR agent",
                },
            }
        ],
    }


def test_parse_submission_manifest_accepts_valid_payload(tmp_path):
    artifact_path = "artifacts/cf_search_and_rescue/submission.zip"
    artifact_file = tmp_path / artifact_path
    _write_zip(
        artifact_file,
        {"drone_agent.py": "class DroneFlightController:\n    pass\n"},
    )

    manifest = parse_submission_manifest_payload(
        _valid_manifest_payload(artifact_path, _sha256(artifact_file))
    )

    assert manifest.manifest_version == SUBMISSION_MANIFEST_VERSION
    assert len(manifest.artifacts) == 1
    assert manifest.artifacts[0].family_id == "cf_search_and_rescue"
    assert manifest.artifacts[0].artifact_path == artifact_path


def test_parse_submission_manifest_rejects_invalid_manifest_version():
    payload = _valid_manifest_payload(
        "artifacts/cf_search_and_rescue/submission.zip",
        "0" * 64,
    )
    payload["manifest_version"] = "submission_manifest.v999"

    try:
        parse_submission_manifest_payload(payload)
    except ValueError as exc:
        assert str(exc) == "unsupported_manifest_version"
    else:  # pragma: no cover
        raise AssertionError("Expected invalid manifest version to be rejected")


def test_parse_submission_manifest_rejects_duplicate_family_id():
    payload = {
        "manifest_version": SUBMISSION_MANIFEST_VERSION,
        "repo_layout_rules": dict(REPO_LAYOUT_RULES),
        "artifacts": [
            {
                "family_id": "cf_search_and_rescue",
                "interface_version": "submission_zip.v1",
                "artifact_path": "artifacts/cf_search_and_rescue/submission.zip",
                "sha256": "1" * 64,
                "metadata": {},
            },
            {
                "family_id": "cf_search_and_rescue",
                "interface_version": "submission_zip.v1",
                "artifact_path": "artifacts/cf_search_and_rescue/alt.zip",
                "sha256": "2" * 64,
                "metadata": {},
            },
        ],
    }

    try:
        parse_submission_manifest_payload(payload)
    except ValueError as exc:
        assert str(exc) == "duplicate_family_id:cf_search_and_rescue"
    else:  # pragma: no cover
        raise AssertionError("Expected duplicate family_id rejection")


def test_parse_submission_manifest_rejects_unsupported_family():
    payload = {
        "manifest_version": SUBMISSION_MANIFEST_VERSION,
        "repo_layout_rules": dict(REPO_LAYOUT_RULES),
        "artifacts": [
            {
                "family_id": "cf_unknown",
                "interface_version": "submission_zip.v1",
                "artifact_path": "artifacts/cf_unknown/submission.zip",
                "sha256": "1" * 64,
                "metadata": {},
            }
        ],
    }

    try:
        parse_submission_manifest_payload(payload)
    except ValueError as exc:
        assert str(exc) == "unsupported_family_id:cf_unknown"
    else:  # pragma: no cover
        raise AssertionError("Expected unsupported family rejection")


def test_parse_submission_manifest_rejects_unsupported_interface_version():
    payload = {
        "manifest_version": SUBMISSION_MANIFEST_VERSION,
        "repo_layout_rules": dict(REPO_LAYOUT_RULES),
        "artifacts": [
            {
                "family_id": "cf_search_and_rescue",
                "interface_version": "submission_zip.v999",
                "artifact_path": "artifacts/cf_search_and_rescue/submission.zip",
                "sha256": "1" * 64,
                "metadata": {},
            }
        ],
    }

    try:
        parse_submission_manifest_payload(payload)
    except ValueError as exc:
        assert (
            str(exc)
            == "unsupported_interface_version:cf_search_and_rescue:submission_zip.v999"
        )
    else:  # pragma: no cover
        raise AssertionError("Expected unsupported interface version rejection")


def test_validate_submission_repo_rejects_missing_artifact(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    payload = _valid_manifest_payload(
        "artifacts/cf_search_and_rescue/submission.zip",
        "f" * 64,
    )
    (repo_root / SUBMISSION_MANIFEST_FILENAME).write_text(json.dumps(payload), encoding="utf-8")

    ok, reason, manifest = validate_submission_repo(repo_root)

    assert ok is False
    assert reason == "missing_artifact:cf_search_and_rescue:artifacts/cf_search_and_rescue/submission.zip"
    assert manifest is not None


def test_build_submission_manifest_payload_sorts_artifacts_by_family_id():
    payload = build_submission_manifest_payload(
        [
            SubmissionArtifact(
                family_id="cf_search_and_rescue",
                interface_version="submission_zip.v1",
                artifact_path="artifacts/cf_search_and_rescue/submission.zip",
                sha256="b" * 64,
                metadata={"notes": "sar"},
            ),
            SubmissionArtifact(
                family_id="cf_autopilot",
                interface_version="submission_zip.v1",
                artifact_path="artifacts/cf_autopilot/submission.zip",
                sha256="a" * 64,
                metadata={"notes": "autopilot"},
            ),
        ]
    )

    assert payload["manifest_version"] == SUBMISSION_MANIFEST_VERSION
    assert payload["repo_layout_rules"] == dict(REPO_LAYOUT_RULES)
    assert [artifact["family_id"] for artifact in payload["artifacts"]] == [
        "cf_autopilot",
        "cf_search_and_rescue",
    ]
