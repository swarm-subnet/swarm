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

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from swarm.cli import _package_model_artifact
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


_AGENT_SOURCE = """import numpy as np


class DroneFlightController:
    def act(self, observation):
        state = np.asarray(observation["state"], dtype=np.float32)
        dim = {dim}
        if state.ndim == 2:
            return np.zeros((state.shape[0], dim), dtype=np.float32)
        return np.zeros(dim, dtype=np.float32)

    def reset(self):
        pass
"""


def _artifact(tmp_path: Path, family_id="cf_autopilot"):
    source = tmp_path / "source"
    source.mkdir(parents=True, exist_ok=True)
    action_dim = 6 if "search_and_rescue" in family_id or "sar" in family_id else 5
    (source / "drone_agent.py").write_text(_AGENT_SOURCE.format(dim=action_dim))
    output = tmp_path / "repo" / "artifacts" / family_id / "submission.zip"
    return _package_model_artifact(
        source_dir=source,
        output_zip=output,
        family_id=family_id,
        interface_version="submission_zip.v1",
        overwrite=True,
    )


def _entry(packaged) -> SubmissionArtifact:
    return SubmissionArtifact(
        family_id=packaged.family_id,
        interface_version=packaged.interface_version,
        artifact_path=f"artifacts/{packaged.family_id}/submission.zip",
        sha256=packaged.sha256,
        size_bytes=packaged.output_zip.stat().st_size,
        metadata={},
    )


def test_parse_accepts_exact_single_artifact(tmp_path):
    packaged = _artifact(tmp_path)
    parsed = parse_submission_manifest_payload(build_submission_manifest_payload([_entry(packaged)]))
    assert parsed.artifacts[0].interface_version == "submission_zip.v1"


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
            "family_id": "cf_autopilot", "interface_version": "submission_zip.v0",
            "artifact_path": "artifacts/cf_autopilot/submission.zip", "sha256": "0" * 64,
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


def _normalize_imports(source: str) -> str:
    """Map both repos' import roots onto one form; the logic must be identical."""
    return (
        source.replace("from swarm.core.submission_policy", "from POLICY")
        .replace("from app.submission_policy", "from POLICY")
        .replace("from swarm.", "from app.")
    )


def test_backend_mirror_matches_apart_from_imports():
    swarm_pkg = Path(__file__).resolve().parents[2] / "swarm" / "submission_manifest"
    backend = Path(__file__).resolve().parents[3] / "swarm-backend" / "app"
    if not backend.is_dir():
        pytest.skip("swarm-backend repository is not checked out next to swarm")
    ours = (swarm_pkg / "__init__.py").read_text()
    theirs = (backend / "submission_manifest.py").read_text()
    assert _normalize_imports(ours) == _normalize_imports(theirs)
    ours_schema = (swarm_pkg / "submission_manifest.schema.json").read_bytes()
    assert ours_schema == (backend / "submission_manifest.schema.json").read_bytes()
