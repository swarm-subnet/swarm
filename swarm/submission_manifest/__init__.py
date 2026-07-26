"""Mandatory repository manifest for the graph-only submission contract."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

from swarm.domain_model import CHALLENGE_FAMILY_IDS, get_supported_interface_versions
from swarm.core.submission_policy import (
    SUBMISSION_INTERFACE_VERSION,
    validate_submission_zip,
)


class SubmissionManifestError(ValueError):
    pass


_SCHEMA_RESOURCE = "submission_manifest.schema.json"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class SubmissionArtifact:
    family_id: str
    interface_version: str
    artifact_path: str
    sha256: str
    size_bytes: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SubmissionManifest:
    manifest_version: str
    repo_layout_rules: dict[str, str]
    artifacts: tuple[SubmissionArtifact, ...]


def load_submission_manifest_schema() -> dict[str, Any]:
    return json.loads(files(__package__).joinpath(_SCHEMA_RESOURCE).read_text(encoding="utf-8"))


_SCHEMA = load_submission_manifest_schema()
SUBMISSION_MANIFEST_VERSION = str(_SCHEMA["manifest_version"])
SUBMISSION_MANIFEST_FILENAME = str(_SCHEMA["manifest_filename"])
REPO_LAYOUT_RULES = dict(_SCHEMA["repo_layout_rules"])
REQUIRED_ARTIFACT_FIELDS = tuple(_SCHEMA["required_artifact_fields"])
SUPPORTED_INTERFACE_VERSIONS = {
    family_id: get_supported_interface_versions(family_id)
    for family_id in CHALLENGE_FAMILY_IDS
}


def _sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_path(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise SubmissionManifestError("invalid_artifact_path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        raise SubmissionManifestError("invalid_artifact_path")
    normalized = str(path)
    if not normalized.startswith(f"{REPO_LAYOUT_RULES['artifacts_dir']}/"):
        raise SubmissionManifestError("invalid_artifact_path")
    if path.suffix.lower() != REPO_LAYOUT_RULES["artifact_extension"]:
        raise SubmissionManifestError("invalid_artifact_path")
    return normalized


def parse_submission_manifest_payload(payload: Any) -> SubmissionManifest:
    if not isinstance(payload, dict):
        raise SubmissionManifestError("invalid_manifest:expected_object")
    expected_root = {"manifest_version", "repo_layout_rules", "artifacts"}
    if set(payload) != expected_root:
        raise SubmissionManifestError("invalid_manifest_fields")
    if payload["manifest_version"] != SUBMISSION_MANIFEST_VERSION:
        raise SubmissionManifestError("unsupported_manifest_version")
    if payload["repo_layout_rules"] != REPO_LAYOUT_RULES:
        raise SubmissionManifestError("invalid_repo_layout_rules")
    entries = payload["artifacts"]
    if not isinstance(entries, list) or len(entries) != 1:
        raise SubmissionManifestError("exactly_one_artifact_required")
    raw = entries[0]
    if not isinstance(raw, dict) or set(raw) != set(REQUIRED_ARTIFACT_FIELDS):
        raise SubmissionManifestError("invalid_artifact_fields")
    family_id = raw["family_id"]
    if family_id not in CHALLENGE_FAMILY_IDS:
        raise SubmissionManifestError(f"unsupported_family_id:{family_id}")
    interface = raw["interface_version"]
    if interface != SUBMISSION_INTERFACE_VERSION or interface not in SUPPORTED_INTERFACE_VERSIONS[family_id]:
        raise SubmissionManifestError(f"unsupported_interface_version:{family_id}:{interface}")
    sha256 = raw["sha256"]
    if not isinstance(sha256, str) or _SHA256_RE.fullmatch(sha256) is None:
        raise SubmissionManifestError("invalid_sha256")
    size_bytes = raw["size_bytes"]
    if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 1:
        raise SubmissionManifestError("invalid_size_bytes")
    metadata = raw["metadata"]
    if not isinstance(metadata, dict):
        raise SubmissionManifestError("invalid_artifact_metadata")
    artifact = SubmissionArtifact(
        family_id=family_id,
        interface_version=interface,
        artifact_path=_artifact_path(raw["artifact_path"]),
        sha256=sha256,
        size_bytes=size_bytes,
        metadata=dict(metadata),
    )
    return SubmissionManifest(SUBMISSION_MANIFEST_VERSION, dict(REPO_LAYOUT_RULES), (artifact,))


def load_submission_manifest(manifest_path: Path) -> SubmissionManifest:
    try:
        return parse_submission_manifest_payload(json.loads(manifest_path.read_text(encoding="utf-8")))
    except FileNotFoundError as exc:
        raise SubmissionManifestError(f"missing_manifest:{SUBMISSION_MANIFEST_FILENAME}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SubmissionManifestError("invalid_manifest_json") from exc


def resolve_submission_manifest(repo_root: Path) -> SubmissionManifest:
    return load_submission_manifest(Path(repo_root) / SUBMISSION_MANIFEST_FILENAME)


def build_submission_manifest_payload(artifacts: Sequence[SubmissionArtifact]) -> dict[str, Any]:
    if len(artifacts) != 1:
        raise SubmissionManifestError("exactly_one_artifact_required")
    artifact = artifacts[0]
    return {
        "manifest_version": SUBMISSION_MANIFEST_VERSION,
        "repo_layout_rules": dict(REPO_LAYOUT_RULES),
        "artifacts": [{
            "family_id": artifact.family_id,
            "interface_version": artifact.interface_version,
            "artifact_path": artifact.artifact_path,
            "sha256": artifact.sha256,
            "size_bytes": artifact.size_bytes,
            "metadata": dict(artifact.metadata),
        }],
    }


def write_submission_manifest(repo_root: Path, artifacts: Sequence[SubmissionArtifact]) -> Path:
    path = Path(repo_root) / SUBMISSION_MANIFEST_FILENAME
    path.write_text(json.dumps(build_submission_manifest_payload(artifacts), indent=2) + "\n", encoding="utf-8")
    return path


def validate_submission_repo(
    repo_root: Path,
    *,
    run_admission: bool = True,
) -> tuple[bool, str, SubmissionManifest | None]:
    """Validate a submission repo layout, artifact bytes, and (optionally) admission.

    ``run_admission=False`` skips the zip content checks for callers that run
    them separately; the manifest, presence, size, and hash checks always run.
    """
    try:
        manifest = resolve_submission_manifest(repo_root)
    except SubmissionManifestError as exc:
        return False, str(exc), None
    artifact = manifest.artifacts[0]
    path = Path(repo_root) / artifact.artifact_path
    if not path.is_file():
        return False, f"missing_artifact:{artifact.family_id}:{artifact.artifact_path}", manifest
    if path.stat().st_size != artifact.size_bytes:
        return False, f"artifact_size_mismatch:{artifact.family_id}:{artifact.artifact_path}", manifest
    if _sha256sum(path) != artifact.sha256:
        return False, f"artifact_hash_mismatch:{artifact.family_id}:{artifact.artifact_path}", manifest
    if run_admission:
        accepted, detail = validate_submission_zip(path)
        if not accepted:
            return False, f"invalid_artifact:{artifact.family_id}:{detail}", manifest
    return True, "ok", manifest


__all__ = [
    "REPO_LAYOUT_RULES", "REQUIRED_ARTIFACT_FIELDS", "SUBMISSION_MANIFEST_FILENAME",
    "SUBMISSION_MANIFEST_VERSION", "SUPPORTED_INTERFACE_VERSIONS", "SubmissionArtifact",
    "SubmissionManifest", "SubmissionManifestError", "build_submission_manifest_payload",
    "load_submission_manifest", "load_submission_manifest_schema", "parse_submission_manifest_payload",
    "resolve_submission_manifest", "validate_submission_repo", "write_submission_manifest",
]
