"""Reads a model-graph artifact package.

Each entry is checked against the expected layout and size limits, then read
individually under byte caps. Layout is exact — one manifest.json plus the
declared models/*.onnx files.
"""

from __future__ import annotations

import hashlib
import zipfile
from dataclasses import dataclass
from pathlib import Path

from .constants import (
    GRAPH_MANIFEST_FILENAME,
    MAX_ARCHIVE_FILES,
    MAX_COMPRESSED_ARCHIVE_BYTES,
    MAX_PER_FILE_COMPRESSION_RATIO,
    MAX_UNCOMPRESSED_TOTAL_BYTES,
    MODEL_DIR_PREFIX,
    MODEL_FILE_SUFFIX,
)
from .errors import ModelGraphError, ReasonCode

_UNIX_LINK_MODE = 0xA000


@dataclass(frozen=True)
class ArchiveContents:
    manifest_bytes: bytes
    files: dict[str, bytes]


def _rejected(detail: str) -> ModelGraphError:
    return ModelGraphError(ReasonCode.ARCHIVE_REJECTED, detail)


def _validate_entry_name(name: str) -> None:
    if name.endswith("/"):
        raise ModelGraphError(ReasonCode.ARCHIVE_BAD_LAYOUT, f"directory entry '{name}' not allowed")
    if "\\" in name or name.startswith("/") or ".." in Path(name).parts:
        raise _rejected(f"invalid entry path '{name}'")
    if name != GRAPH_MANIFEST_FILENAME and not (
        name.startswith(MODEL_DIR_PREFIX) and name.endswith(MODEL_FILE_SUFFIX)
    ):
        raise ModelGraphError(
            ReasonCode.ARCHIVE_BAD_LAYOUT,
            f"'{name}' is neither {GRAPH_MANIFEST_FILENAME} nor models/*.onnx",
        )


def read_archive(zip_path: Path) -> ArchiveContents:
    size = zip_path.stat().st_size
    if size > MAX_COMPRESSED_ARCHIVE_BYTES:
        raise ModelGraphError(
            ReasonCode.ARCHIVE_TOO_LARGE,
            f"archive is {size} bytes (cap {MAX_COMPRESSED_ARCHIVE_BYTES})",
        )

    try:
        zf = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile as exc:
        raise ModelGraphError(ReasonCode.ARCHIVE_REJECTED, f"unreadable archive: {exc}") from exc

    with zf:
        infos = zf.infolist()
        if len(infos) > MAX_ARCHIVE_FILES:
            raise ModelGraphError(
                ReasonCode.RESOURCE_CAP_EXCEEDED,
                f"{len(infos)} entries (cap {MAX_ARCHIVE_FILES})",
            )

        seen_lower: set[str] = set()
        total_uncompressed = 0
        for info in infos:
            if (info.external_attr >> 16) & _UNIX_LINK_MODE == _UNIX_LINK_MODE:
                raise _rejected(f"link entry not allowed: '{info.filename}'")
            _validate_entry_name(info.filename)
            lower = info.filename.lower()
            if lower in seen_lower:
                raise ModelGraphError(
                    ReasonCode.ARCHIVE_BAD_LAYOUT,
                    f"duplicate or case-colliding entry '{info.filename}'",
                )
            seen_lower.add(lower)
            if info.compress_size > 0:
                ratio = info.file_size / info.compress_size
                if ratio > MAX_PER_FILE_COMPRESSION_RATIO:
                    raise _rejected(
                        f"'{info.filename}' expansion ratio {ratio:.1f} exceeds "
                        f"{MAX_PER_FILE_COMPRESSION_RATIO}"
                    )
            total_uncompressed += info.file_size
            if total_uncompressed > MAX_UNCOMPRESSED_TOTAL_BYTES:
                raise ModelGraphError(
                    ReasonCode.RESOURCE_CAP_EXCEEDED,
                    f"uncompressed size exceeds {MAX_UNCOMPRESSED_TOTAL_BYTES} bytes",
                )

        names = {info.filename for info in infos}
        if GRAPH_MANIFEST_FILENAME not in names:
            raise ModelGraphError(
                ReasonCode.MANIFEST_MISSING,
                f"archive has no root {GRAPH_MANIFEST_FILENAME}",
            )

        files: dict[str, bytes] = {}
        for info in infos:
            with zf.open(info, "r") as handle:
                data = handle.read(info.file_size + 1)
            if len(data) != info.file_size:
                raise _rejected(f"'{info.filename}' size does not match its header")
            files[info.filename] = data

    manifest_bytes = files.pop(GRAPH_MANIFEST_FILENAME)
    return ArchiveContents(manifest_bytes=manifest_bytes, files=files)
