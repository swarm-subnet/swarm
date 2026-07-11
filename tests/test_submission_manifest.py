from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from swarm.cli import _package_model_artifact
from swarm.model_graph_template.generate_template import generate
from swarm.submission_manifest import (
    REPO_LAYOUT_RULES,
    SUBMISSION_MANIFEST_FILENAME,
    SUBMISSION_MANIFEST_VERSION,
    SubmissionArtifact,
    SubmissionManifestError,
    build_submission_manifest_payload,
    parse_submission_manifest_payload,
    validate_submission_repo,
    write_submission_manifest,
)


def _artifact(tmp_path: Path, family_id="cf_autopilot"):
    source = tmp_path / "source"
    generate(source)
    output = tmp_path / "repo" / "artifacts" / family_id / "model_graph.zip"
    return _package_model_artifact(
        source_dir=source,
        output_zip=output,
        family_id=family_id,
        interface_version="model_graph.v1",
        overwrite=True,
    )


def _entry(packaged) -> SubmissionArtifact:
    return SubmissionArtifact(
        family_id=packaged.family_id,
        interface_version=packaged.interface_version,
        artifact_path=f"artifacts/{packaged.family_id}/model_graph.zip",
        sha256=packaged.sha256,
        size_bytes=packaged.output_zip.stat().st_size,
        metadata={},
    )


def test_parse_accepts_exact_single_graph_artifact(tmp_path):
    packaged = _artifact(tmp_path)
    parsed = parse_submission_manifest_payload(build_submission_manifest_payload([_entry(packaged)]))
    assert parsed.artifacts[0].interface_version == "model_graph.v1"


@pytest.mark.parametrize("artifacts", [[], [{}, {}]])
def test_parse_requires_exactly_one_artifact(artifacts):
    payload = {
        "manifest_version": SUBMISSION_MANIFEST_VERSION,
        "repo_layout_rules": dict(REPO_LAYOUT_RULES),
        "artifacts": artifacts,
    }
    with pytest.raises(SubmissionManifestError, match="exactly_one_artifact_required"):
        parse_submission_manifest_payload(payload)


def test_parse_rejects_old_interface():
    payload = {
        "manifest_version": SUBMISSION_MANIFEST_VERSION,
        "repo_layout_rules": dict(REPO_LAYOUT_RULES),
        "artifacts": [{
            "family_id": "cf_autopilot", "interface_version": "model_graph.v0",
            "artifact_path": "artifacts/cf_autopilot/model_graph.zip", "sha256": "0" * 64,
            "size_bytes": 1, "metadata": {},
        }],
    }
    with pytest.raises(SubmissionManifestError, match="unsupported_interface_version"):
        parse_submission_manifest_payload(payload)


def test_validate_repo_checks_size_hash_and_inner_family(tmp_path):
    packaged = _artifact(tmp_path)
    repo = packaged.output_zip.parents[2]
    write_submission_manifest(repo, [_entry(packaged)])
    assert validate_submission_repo(repo)[:2] == (True, "ok")
    payload = json.loads((repo / SUBMISSION_MANIFEST_FILENAME).read_text())
    payload["artifacts"][0]["size_bytes"] += 1
    (repo / SUBMISSION_MANIFEST_FILENAME).write_text(json.dumps(payload))
    assert validate_submission_repo(repo)[1].startswith("artifact_size_mismatch")


def test_manifest_is_mandatory_even_when_root_zip_exists(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "submission.zip").write_bytes(b"not used")
    assert validate_submission_repo(repo)[1] == f"missing_manifest:{SUBMISSION_MANIFEST_FILENAME}"
