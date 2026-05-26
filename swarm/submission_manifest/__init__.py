from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

from swarm.core.submission_policy import validate_submission_zip
from swarm.domain_model import CHALLENGE_FAMILY_IDS, get_supported_interface_versions


class SubmissionManifestError(ValueError):
    """Raised when a submission manifest is malformed or violates repo rules."""


_SCHEMA_RESOURCE = "submission_manifest.schema.json"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class SubmissionArtifact:
    family_id: str
    interface_version: str
    artifact_path: str
    sha256: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SubmissionManifest:
    manifest_version: str
    repo_layout_rules: dict[str, str]
    artifacts: tuple[SubmissionArtifact, ...]
    legacy_fallback_used: bool = False


def load_submission_manifest_schema() -> dict[str, Any]:
    schema_text = files(__package__).joinpath(_SCHEMA_RESOURCE).read_text(
        encoding="utf-8",
    )
    return json.loads(schema_text)


_SCHEMA = load_submission_manifest_schema()

SUBMISSION_MANIFEST_VERSION = str(_SCHEMA["manifest_version"])
SUBMISSION_MANIFEST_FILENAME = str(_SCHEMA["manifest_filename"])
LEGACY_FALLBACK = dict(_SCHEMA["legacy_fallback"])
REPO_LAYOUT_RULES = dict(_SCHEMA["repo_layout_rules"])
REQUIRED_ARTIFACT_FIELDS = tuple(_SCHEMA["required_artifact_fields"])
SUPPORTED_INTERFACE_VERSIONS = {
    family_id: get_supported_interface_versions(family_id)
    for family_id in CHALLENGE_FAMILY_IDS
}


def _sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _require_manifest_dict(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise SubmissionManifestError("invalid_manifest:expected_object")
    return payload


def _validate_repo_layout_rules(payload: dict[str, Any]) -> dict[str, str]:
    repo_layout_rules = payload.get("repo_layout_rules")
    if repo_layout_rules != REPO_LAYOUT_RULES:
        raise SubmissionManifestError("invalid_repo_layout_rules")
    return dict(repo_layout_rules)


def _validate_artifact_path(path_value: Any) -> str:
    if not isinstance(path_value, str) or not path_value.strip():
        raise SubmissionManifestError("invalid_artifact_path")
    pure_path = PurePosixPath(path_value)
    if pure_path.is_absolute() or ".." in pure_path.parts:
        raise SubmissionManifestError("invalid_artifact_path")
    normalized = str(pure_path)
    if not normalized.startswith(f"{REPO_LAYOUT_RULES['artifacts_dir']}/"):
        raise SubmissionManifestError("invalid_artifact_path")
    if pure_path.suffix.lower() != REPO_LAYOUT_RULES["artifact_extension"]:
        raise SubmissionManifestError("invalid_artifact_path")
    return normalized


def _validate_sha256_hex(value: Any) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise SubmissionManifestError("invalid_sha256")
    return value


def parse_submission_manifest_payload(payload: Any) -> SubmissionManifest:
    manifest = _require_manifest_dict(payload)
    if manifest.get("manifest_version") != SUBMISSION_MANIFEST_VERSION:
        raise SubmissionManifestError("unsupported_manifest_version")

    repo_layout_rules = _validate_repo_layout_rules(manifest)
    artifacts_payload = manifest.get("artifacts")
    if not isinstance(artifacts_payload, list) or not artifacts_payload:
        raise SubmissionManifestError("invalid_artifacts")

    seen_family_ids: set[str] = set()
    seen_paths: set[str] = set()
    parsed_artifacts: list[SubmissionArtifact] = []

    for artifact_payload in artifacts_payload:
        if not isinstance(artifact_payload, dict):
            raise SubmissionManifestError("invalid_artifact_entry")

        missing_fields = [
            field_name
            for field_name in REQUIRED_ARTIFACT_FIELDS
            if field_name not in artifact_payload
        ]
        if missing_fields:
            raise SubmissionManifestError(
                f"missing_artifact_field:{','.join(sorted(missing_fields))}"
            )

        family_id = artifact_payload["family_id"]
        if family_id not in CHALLENGE_FAMILY_IDS:
            raise SubmissionManifestError(f"unsupported_family_id:{family_id}")
        if family_id in seen_family_ids:
            raise SubmissionManifestError(f"duplicate_family_id:{family_id}")

        interface_version = artifact_payload["interface_version"]
        if interface_version not in SUPPORTED_INTERFACE_VERSIONS.get(family_id, ()):
            raise SubmissionManifestError(
                f"unsupported_interface_version:{family_id}:{interface_version}"
            )

        artifact_path = _validate_artifact_path(artifact_payload["artifact_path"])
        if artifact_path in seen_paths:
            raise SubmissionManifestError(f"duplicate_artifact_path:{artifact_path}")

        metadata = artifact_payload["metadata"]
        if not isinstance(metadata, dict):
            raise SubmissionManifestError("invalid_artifact_metadata")

        parsed_artifacts.append(
            SubmissionArtifact(
                family_id=family_id,
                interface_version=interface_version,
                artifact_path=artifact_path,
                sha256=_validate_sha256_hex(artifact_payload["sha256"]),
                metadata=dict(metadata),
            )
        )
        seen_family_ids.add(family_id)
        seen_paths.add(artifact_path)

    return SubmissionManifest(
        manifest_version=SUBMISSION_MANIFEST_VERSION,
        repo_layout_rules=repo_layout_rules,
        artifacts=tuple(parsed_artifacts),
        legacy_fallback_used=False,
    )


def load_submission_manifest(manifest_path: Path) -> SubmissionManifest:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SubmissionManifestError(
            f"missing_manifest:{SUBMISSION_MANIFEST_FILENAME}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise SubmissionManifestError("invalid_manifest_json") from exc
    return parse_submission_manifest_payload(payload)


def build_legacy_submission_manifest() -> SubmissionManifest:
    family_id = str(LEGACY_FALLBACK["default_family_id"])
    artifact_path = str(LEGACY_FALLBACK["artifact_path"])
    interface_version = SUPPORTED_INTERFACE_VERSIONS[family_id][0]
    return SubmissionManifest(
        manifest_version=SUBMISSION_MANIFEST_VERSION,
        repo_layout_rules=dict(REPO_LAYOUT_RULES),
        artifacts=(
            SubmissionArtifact(
                family_id=family_id,
                interface_version=interface_version,
                artifact_path=artifact_path,
                sha256="0" * 64,
                metadata={"legacy_submission_zip": True},
            ),
        ),
        legacy_fallback_used=True,
    )


def build_submission_manifest_payload(
    artifacts: Sequence[SubmissionArtifact],
) -> dict[str, Any]:
    ordered_artifacts = sorted(
        artifacts,
        key=lambda item: (item.family_id, item.artifact_path),
    )
    return {
        "manifest_version": SUBMISSION_MANIFEST_VERSION,
        "repo_layout_rules": dict(REPO_LAYOUT_RULES),
        "artifacts": [
            {
                "family_id": artifact.family_id,
                "interface_version": artifact.interface_version,
                "artifact_path": artifact.artifact_path,
                "sha256": artifact.sha256,
                "metadata": dict(artifact.metadata),
            }
            for artifact in ordered_artifacts
        ],
    }


def write_submission_manifest(
    repo_root: Path,
    artifacts: Sequence[SubmissionArtifact],
) -> Path:
    manifest_path = repo_root / SUBMISSION_MANIFEST_FILENAME
    payload = build_submission_manifest_payload(artifacts)
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def resolve_submission_manifest(
    repo_root: Path,
    *,
    allow_legacy_fallback: bool = True,
) -> SubmissionManifest:
    manifest_path = repo_root / SUBMISSION_MANIFEST_FILENAME
    if manifest_path.exists():
        return load_submission_manifest(manifest_path)

    legacy_path = repo_root / str(LEGACY_FALLBACK["artifact_path"])
    if allow_legacy_fallback and legacy_path.exists():
        return build_legacy_submission_manifest()

    raise SubmissionManifestError(f"missing_manifest:{SUBMISSION_MANIFEST_FILENAME}")


def validate_submission_repo(
    repo_root: Path,
    *,
    allow_legacy_fallback: bool = True,
) -> tuple[bool, str, SubmissionManifest | None]:
    try:
        manifest = resolve_submission_manifest(
            repo_root,
            allow_legacy_fallback=allow_legacy_fallback,
        )
    except SubmissionManifestError as exc:
        return False, str(exc), None

    for artifact in manifest.artifacts:
        artifact_file = repo_root / artifact.artifact_path
        if not artifact_file.is_file():
            return (
                False,
                f"missing_artifact:{artifact.family_id}:{artifact.artifact_path}",
                manifest,
            )
        if not manifest.legacy_fallback_used:
            if _sha256sum(artifact_file) != artifact.sha256:
                return (
                    False,
                    f"artifact_hash_mismatch:{artifact.family_id}:{artifact.artifact_path}",
                    manifest,
                )
        ok, reason = validate_submission_zip(artifact_file)
        if not ok:
            return False, f"invalid_artifact:{artifact.family_id}:{reason}", manifest

    return True, "ok", manifest


__all__ = [
    "LEGACY_FALLBACK",
    "REPO_LAYOUT_RULES",
    "REQUIRED_ARTIFACT_FIELDS",
    "SUBMISSION_MANIFEST_FILENAME",
    "SUBMISSION_MANIFEST_VERSION",
    "SUPPORTED_INTERFACE_VERSIONS",
    "SubmissionArtifact",
    "SubmissionManifest",
    "SubmissionManifestError",
    "build_submission_manifest_payload",
    "build_legacy_submission_manifest",
    "load_submission_manifest",
    "load_submission_manifest_schema",
    "parse_submission_manifest_payload",
    "resolve_submission_manifest",
    "validate_submission_repo",
    "write_submission_manifest",
]
