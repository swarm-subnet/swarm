"""Graph artifact archive policy shared by local packaging and validators."""

from __future__ import annotations

from pathlib import Path

from swarm.model_graph.admission import admit_artifact
from swarm.model_graph.archive import read_archive
from swarm.model_graph.constants import GRAPH_MANIFEST_FILENAME, MAX_UNCOMPRESSED_TOTAL_BYTES
from swarm.model_graph.errors import ModelGraphError
from swarm.model_graph.manifest import parse_manifest

REQUIRED_ROOT_FILES: tuple[str, ...] = (GRAPH_MANIFEST_FILENAME,)
FORBIDDEN_SUFFIXES: tuple[str, ...] = ()
MAX_UNCOMPRESSED_BYTES: int = MAX_UNCOMPRESSED_TOTAL_BYTES
STRUCTURE_FAILURE_REASON_PREFIXES: tuple[str, ...] = ("MG_",)


def check_safety(
    zip_path: Path, *, max_uncompressed: int = MAX_UNCOMPRESSED_BYTES
) -> tuple[bool, str]:
    if max_uncompressed != MAX_UNCOMPRESSED_BYTES:
        return False, "custom_archive_cap_not_supported"
    try:
        read_archive(Path(zip_path))
    except (ModelGraphError, OSError) as exc:
        return False, str(exc)
    return True, "ok"


def check_structure(zip_path: Path) -> tuple[bool, str]:
    try:
        contents = read_archive(Path(zip_path))
        manifest = parse_manifest(contents.manifest_bytes)
        if {m.file for m in manifest.models} != set(contents.files):
            return False, "MG_ARCHIVE_BAD_LAYOUT: declared model files do not match archive"
    except (ModelGraphError, OSError) as exc:
        return False, str(exc)
    return True, "ok"


def validate_submission_zip(zip_path: Path) -> tuple[bool, str]:
    result = admit_artifact(Path(zip_path))
    return result.accepted, "ok" if result.accepted else f"{result.reason_code}:{result.detail}"
